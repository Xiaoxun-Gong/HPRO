import sys
import time, datetime
import traceback
import importlib_resources
import json
from collections import namedtuple
from math import ceil
from tqdm import tqdm
from tqdm.utils import disp_len, _unicode
import numpy as np

comm = None
MPI = None

def is_master(comm=comm):
    return comm is None or comm.rank == 0

class tqdm_mpi_tofile(tqdm):
    def __init__(self, iterable=None, **kwargs):
        if 'total' in kwargs:
            total = kwargs['total']
        else:
            total = len(iterable)
        miniters = ceil(total / 10)
        kwargs['miniters'] = miniters
        kwargs['file'] = sys.stdout
        kwargs['delay'] = 1e-5
        kwargs['leave'] = True # total%miniters!=0
        kwargs['bar_format'] = '{l_bar}{bar:40}{r_bar}{bar:-10b}'
        disable = kwargs.pop('disable', False)
        kwargs['disable'] = not is_master() or disable
        # prevent tqdm from changing miniters by itself,
        # see /home1/09019/xiaoxun/.local/lib/python3.9/site-packages/tqdm/_monitor.py", line 82
        kwargs['mininterval'] = 1
        kwargs['maxinterval'] = 1e10 
        super().__init__(iterable=iterable, **kwargs)
        # self.dynamic_miniters = False
        # self.miniters = miniters
        self.start_time = time.time()
    
    # @property
    # def miniters(self):
    #     return self._miniters
    
    # @miniters.setter
    # def miniters(self, v):
    #     self._miniters = v
    #     traceback.print_stack()
    
    
    @staticmethod
    def status_printer(file):
        """
        Manage the printing and in-place updating of a line of characters.
        Note that if the string is longer than a line, then in-place
        updating may not work (it will print a new line at each refresh).
        """
        fp = file
        fp_flush = getattr(fp, 'flush', lambda: None)  # pragma: no cover
        if fp in (sys.stderr, sys.stdout):
            getattr(sys.stderr, 'flush', lambda: None)()
            getattr(sys.stdout, 'flush', lambda: None)()

        def fp_write(s):
            fp.write(_unicode(s))
            fp_flush()

        last_len = [0]

        def print_status(s):
            len_s = disp_len(s)
            fp_write(s + (' ' * max(last_len[0] - len_s, 0)) + '\n') # always write new bar in new line
            last_len[0] = len_s

        return print_status

    def close(self):
        """Cleanup and (if leave=False) close the progressbar."""
        if self.disable:
            return

        # Prevent multiple closures
        self.disable = True

        # decrement instance pos and remove from internal set
        pos = abs(self.pos)
        self._decr_instances(self)

        if self.last_print_t < self.start_t + self.delay:
            # haven't ever displayed; nothing to clear
            return

        # GUI mode
        if getattr(self, 'sp', None) is None:
            return

        # annoyingly, _supports_unicode isn't good enough
        def fp_write(s):
            self.fp.write(_unicode(s))

        try:
            fp_write('')
        except ValueError as e:
            if 'closed' in str(e):
                return
            raise  # pragma: no cover

        leave = pos == 0 if self.leave is None else self.leave

        with self._lock:
            if leave:
                # stats for overall rate (no weighted average)
                self._ema_dt = lambda: None
                self.display(pos=0)
                # fp_write('\n')
            # else:
                # clear previous display
                # if self.display(msg='', pos=pos) and not pos:
                    # fp_write('\r')
        
        elapsed = time.time() - self.start_time
        fp_write(f'Done, elapsed time: {elapsed:4.1f}s.\n\n') # write timing message

    # def set_postfix(self, ordered_dict=None, refresh=False, **kwargs):
    #     return super().set_postfix(self, ordered_dict=ordered_dict, refresh=refresh, **kwargs)
    
    def update(self, n=1):
        if not self.disable:
            if self.n + n >= self.total:
                self.n += n
                self.last_print_n = self.n
                self.last_print_t = time.time()
                return
        return super().update(n=n)
    
    def __iter__(self):
        """Backward-compatibility to use: for x in tqdm(iterable)"""

        # Inlining instance variables as locals (speed optimisation)
        iterable = self.iterable

        # If the bar is disabled, then just walk the iterable
        # (note: keep this check outside the loop for performance)
        if self.disable:
            for obj in iterable:
                yield obj
            return

        try:
            for obj in iterable:
                yield obj
                self.update()
        finally:
            self.close()

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
        try:
            return f(*args, **kwargs)
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
    return g

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

with importlib_resources.files('HPRO').joinpath('periodic_table.json').open('r') as f:
    periodic_table = json.load(f)
    
def atom_name2number(element_names):
    return [periodic_table[spc]["Atomic no"] for spc in element_names]

def atom_number2name(atomic_numbers):
    element_names = []
    for n in atomic_numbers:
        for k, v in periodic_table.items():
            if v["Atomic no"] == n:
                element_names.append(k)
                break
    return element_names


KGData = namedtuple('KGData', ['ng', 'nbnd', 'miller_idc', 'unkg', 'kgcart'])
# miller_idc(ng, 3): int
# unkg(nbnd, ng): complex
# kgcart(ng, 3): float, already scaled by 2pi
# These are all distributed among different processors
VKBGData = namedtuple('VKBGData', ['ng', 'miller_idc', 'vkbg', 'kgcart'])
# miller_idc(ng, 3)
# vkbg: int->(nh, ng), \beta_{ah'sk}
# kgcart(ng, 3)


def set_range(xrange, xmin, xmax):
    if xrange is None:
        return (xmin, xmax)
    else:
        return (xmin if xrange[0] is None else xrange[0],
                xmax if xrange[1] is None else xrange[1])

def slice_same(array):
    '''
    array must be 1D
    Example: [1,1,2,2,2,3,3] -> [0, 2, 5, 7]
    '''
    return np.nonzero(np.r_[1, np.diff(array), 1])[0]

def simple_timer(formatstr):
    def timer_decorator(f):
        def g(*args, **kwargs):
            start_time = time.time()
            ret = f(*args, **kwargs)
            if is_master():
                total_time = datetime.timedelta(seconds=int(time.time()-start_time))
                # print(f'total wall time: {str(total_time)}\n')
                print(formatstr.format(t=str(total_time)))
            return ret
        return g
    return timer_decorator
