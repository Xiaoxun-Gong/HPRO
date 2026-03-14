import sys
import time
import traceback
from math import ceil, log2
from contextlib import contextmanager

import numpy as np
from scipy.sparse import csr_matrix

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
except ModuleNotFoundError:
    MPI = None
    comm = None

def is_master(comm=comm):
    return comm is None or comm.rank == 0

class one_by_one:
    # todo: get rid of this
    '''Do sth one by one within pool'''
    def __init__(self, pool_size=8):
        self.pool_size = pool_size
        
    def __enter__(self):
        if comm is not None and (comm.rank % self.pool_size != 0):
            comm.recv(source=comm.rank-1, tag=11)
    
    def __exit__(self, *args):
        if comm is not None and comm.rank < comm.size-1 and ((comm.rank+1) % self.pool_size != 0):
            comm.send(1, dest=comm.rank+1, tag=11)
        if comm is not None:
            comm.barrier()
          
class one_by_one_old:
    '''Only begin to do something when the last pool has finished'''
    def __init__(self, pool_size=1):
        self.pool_size = pool_size
        
    def __enter__(self):
        if comm is not None and comm.rank >= self.pool_size:
            comm.recv(source=comm.rank-self.pool_size, tag=11)
    
    def __exit__(self, *args):
        if comm is not None and comm.rank < comm.size-self.pool_size:
            comm.send(1, dest=comm.rank+self.pool_size, tag=11)
        if comm is not None:
            comm.barrier()

def mpi_watch(f):
    """Decorator. Terminate all mpi process if an exception is raised."""
    def g(*args, **kwargs):
        with mpi_abort_if_exception():
            return f(*args, **kwargs)
    return g

@contextmanager
def mpi_abort_if_exception():
    """Terminate all mpi process if an exception is raised."""
    try:
        yield
    except:
        sys.stdout.flush()
        sys.stderr.flush()
        time.sleep(1) # wait for other nodes if they have sth to print
        if not is_master():
            time.sleep(5) # wait for root to get here first
            print(f'Error from proc {comm.rank}:')
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        if comm is None:
            exit(1)
        else:
            comm.Abort(1)

def distrib_grps(length, ngrps, displ_last_elem=False):
    '''distribute a vector with `length` into `ngrps` groups'''
    avg, rem = divmod(length, ngrps)
    count = [avg + 1 if p < rem else avg for p in range(ngrps)]
    if displ_last_elem:
        displ = np.cumsum([0] + count)
    else:
        displ = np.cumsum([0] + count[:-1])
    count = np.array(count)
    return count, displ

def distrib_vec(length, displ_last_elem=False, comm=comm):
    '''length of displ will be comm.size if displ_last_elem=False, or comm.size+1 if displ_last_elem=True'''
    if comm is not None:
        rank = comm.rank
        size = comm.size
    else:
        rank = 0
        size = 1
    count, displ = distrib_grps(length, size, displ_last_elem=displ_last_elem)
    return rank, count, displ

# mpi send/receive/sum csr matrix

def mpi_send_csr(comm, matcsr, dest, tag=0):
    assert matcsr.indptr.dtype == np.int32
    assert matcsr.indices.dtype == np.int32
    assert matcsr.data.dtype == np.float64 # todo: complex
    auxdata = np.array([matcsr.shape[0], matcsr.shape[1], len(matcsr.indptr), len(matcsr.indices)], dtype='i8')
    comm.Send([auxdata, MPI.INTEGER8], dest=dest, tag=tag)
    comm.Send([matcsr.indptr, MPI.INTEGER4], dest=dest, tag=tag+1)
    comm.Send([matcsr.indices, MPI.INTEGER4], dest=dest, tag=tag+2)
    comm.Send([matcsr.data, MPI.REAL8], dest=dest, tag=tag+3)

def mpi_recv_csr(comm, source, tag=0):
    auxdata = np.empty(4, dtype='i8')
    comm.Recv([auxdata, MPI.INTEGER8], source=source, tag=tag)
    shape1, shape2, len_indptr, len_data = auxdata
    indptr = np.empty(len_indptr, dtype='i4')
    indices = np.empty(len_data, dtype='i4')
    data = np.empty(len_data, dtype='f8')
    comm.Recv([indptr, MPI.INTEGER4], source=source, tag=tag+1)
    comm.Recv([indices, MPI.INTEGER4], source=source, tag=tag+2)
    comm.Recv([data, MPI.REAL8], source=source, tag=tag+3)
    return csr_matrix((data, indices, indptr), shape=(shape1, shape2))

def mpi_sum_csr(matcsr, comm=comm):
    '''
    Summation pattern:
                     power      fac
    0 1 2 3 4 5 6
    ├-┘ ├-┘ ├-┘ |        1        2
    0   2   4   6
    ├---┘   ├---┘        2        4
    0       4
    ├-------┘            3        8
    0
    '''
    for power in range(1, ceil(log2(comm.size))+1):
        fac = 2**power
        lastfac = 2**(power-1)
        if (comm.rank%fac==0) and (comm.rank+lastfac<comm.size):
            matcsr = matcsr + mpi_recv_csr(comm, comm.rank+lastfac)
        elif (comm.rank%fac==lastfac):
            mpi_send_csr(comm, matcsr, comm.rank-lastfac)
            matcsr = None
    
    if not is_master(comm=comm):
        assert matcsr is None
        
    return matcsr