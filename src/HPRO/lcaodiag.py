import numpy as np
import os, sys
import time, datetime

from .deephio import load_deeph_HS
from .utils import is_master, MPI, comm, mpi_watch, simple_timer, distrib_grps, tqdm_mpi_tofile
from .constants import hartree2ev

import sys
use_slepc4py = True
try:
    import slepc4py
    slepc4py.init(sys.argv)
    from petsc4py import PETSc
    from slepc4py import SLEPc
except ModuleNotFoundError:
    use_slepc4py = False
    from scipy.sparse.linalg import eigsh

'''
This module has several functions to diagonalize atomic orbital Hamiltonians.
'''

class LCAODiagKernel:

    @mpi_watch
    def __init__(self, matH=None, matS=None):
        self.matH = matH
        self.matS = matS

        self.nk = None
        self.kpts = None
        self.hskpos = None
        self.hsksymbol = None

        self.nao = None
        self.eigs = None
        self.wfnao = None

        self.nkpools = None
        self.igrp = None
        self.comm_pool = None
        self.count_proc = None
    
    @mpi_watch
    @simple_timer('\nLoad matrices done, total wall time = {t}')
    def load_deeph_mats(self, folder, hmatfname='hamiltonians.h5', smatfname='overlaps.h5'):

        assert self.kpts is not None, 'Must call setk() before loading matrices'

        if is_master(comm=self.comm_pool):
            if is_master(): print('Loading Hamiltonian and overlap matrices')
            start_time = time.time()
            matH = load_deeph_HS(folder, hmatfname, energy_unit=True)
            matS = load_deeph_HS(folder, smatfname, energy_unit=False)
            if is_master(): print('Done, elapsed time:', datetime.timedelta(seconds=int(time.time()-start_time)))
            self.matH = matH
            self.matS = matS

            errorh = matH.hermitianize()
            if is_master(): print('Hamiltonian non-Hermiticity error (Ha):', errorh)
            errors = matS.hermitianize()
            if is_master(): print('Overlap non-Hermiticity error:', errors)
            matH.to_csr()
            matS.to_csr()

            nao = matH.norb_total
        
        else:  
            nao = 0
        
        if comm is not None:
            nao = self.comm_pool.bcast(nao, root=0)
        self.nao = nao


    @mpi_watch
    def setk(self, kpts, kptwts, kptsymbol, type='path', nkpools=None):
        '''
        Example: G-K-M-G
        kpts = 
          [[0.000000000000,  0.000000000000,  0.000000000000],
           [0.333333333333,  0.333333333333,  0.000000000000],  
           [0.500000000000,  0.000000000000,  0.000000000000],  
           [0.000000000000,  0.000000000000,  0.000000000000]]
        kptwts = 
          [20, 10, 17, 1]
        kptsymbol = 
          ['G', 'K', 'M', 'G']
        '''

        kpts = np.array(kpts)
        kptwts = np.array(kptwts)

        assert type=='path'
        assert len(kpts) == len(kptwts)
        assert kptwts[-1] == 1
        nk = np.sum(kptwts)
        kpts_full = np.empty((nk, 3), dtype='f8')
        pos = 0
        ikhs = -1
        for ikhs in range(len(kpts)-1):
            wk = kptwts[ikhs]
            arange = np.arange(0., 1., 1/wk)
            kstart = kpts[ikhs]
            kend = kpts[ikhs+1]
            kpts_full[pos:pos+wk, :] = kstart[None, :] + (kend - kstart)[None, :] * arange[:, None]
            pos += wk
        kpts_full[pos, :] = kpts[ikhs+1]
        pos += 1
        assert pos == nk

        self.nk = len(kpts_full)
        self.kpts = kpts_full
        self.hskpos = list(np.concatenate(([0], np.cumsum(kptwts)[:-1])))
        self.hsksymbol = kptsymbol

        # print(self.kpts)
        # print(self.hskpos)
        # print(self.kpts[self.hskpos])

        if comm is not None:
            # distribute kpoints into groups

            if nkpools is None:
                nkpools = min(self.nk, comm.size)
            assert nkpools <= comm.size

            count_proc, displ_proc = distrib_grps(comm.size, nkpools, displ_last_elem=True)
            igrp = np.searchsorted(displ_proc, comm.rank, side='right') - 1
            if nkpools == min(self.nk, comm.size):
                comm_pool = MPI.COMM_SELF
            else:
                msg = 'Number of k point pools must be equal to min(# k points, # processors) when slepc4py is not installed'
                assert use_slepc4py, msg
                comm_pool = comm.Split(color=igrp, key=comm.rank)
            # comm_pool = comm.Split(color=igrp, key=comm.rank)


        else:
            igrp = 0
            count_proc = [1]
            comm_pool = None
            nkpools = 1

        self.igrp = igrp
        self.count_proc = count_proc
        self.comm_pool = comm_pool
        self.nkpools = nkpools
    
    @mpi_watch
    @simple_timer('\nJob done, total wall time = {t}\n')
    def diag(self, nbnd, efermi=None, tole=1e-8, max_it=100, sort=None):
        '''
        efermi is provided in eV, not Hartree
        '''

        if efermi is not None:
            efermi /= hartree2ev
        
        if sort is None:
            # sort is by default True if efermi not provided
            sort = efermi is None

        # distribute kpoints into groups
        comm_pool = self.comm_pool
        count_k, displ_k = distrib_grps(self.nk, self.nkpools, displ_last_elem=True)
        nkloc = count_k[self.igrp]
        if is_master():
            print('\nParallization report:')
            print('Total number of k points:', self.nk)
            if comm is not None:
                print('Total number of MPI tasks:', comm.size)
            else:
                print('Module mpi4py not found, executing serially')
            print(f'K points are distributed into {self.nkpools} pools')
            print(f'Each pool has {np.min(self.count_proc)} to {np.max(self.count_proc)} MPI tasks')
            print('   Note: if the above number is larger than 1, please make sure MKL cluster Pardiso is linked to PETSc')
            print(f'Each pool is dealing with {np.min(count_k)} to {np.max(count_k)} k-points')
            print(f'Each MPI task has {os.environ.get("OMP_NUM_THREADS", 1)} OMP threads\n')

            if use_slepc4py:
                print('Using SLEPc diagonalization with MKL Pardiso')
            else:
                print('Using scipy diagonalization with ARPACK')

        eigs = np.empty((nkloc, nbnd), dtype='f8')
        if is_master(comm=comm_pool):
            wfnao = np.empty((nkloc, nbnd, self.nao), dtype='c16')
        else:
            wfnao = None

        if is_master(): print('Begin diagonalization')
        for ikpt in tqdm_mpi_tofile(range(count_k[self.igrp])):
            kpt = self.kpts[ikpt + displ_k[self.igrp]]

            if is_master(comm=comm_pool):
                Hk = self.matH.r2k(kpt)
                Sk = self.matS.r2k(kpt)
                if use_slepc4py:
                    Hpetsc = mat_scipy2petsc(Hk, comm=comm_pool)
                    Spetsc = mat_scipy2petsc(Sk, comm=comm_pool)
            else:
                if use_slepc4py:
                    Hpetsc = mat_scipy2petsc(None, comm=comm_pool)
                    Spetsc = mat_scipy2petsc(None, comm=comm_pool)
            
            if use_slepc4py:
                if efermi is None:
                    eigmax, _ = diag_slepc(Hpetsc, Spetsc, 1, 'LM', tole=0.1, comm=comm_pool)
                    eigmax = eigmax.item()
                    if eigmax > 0:
                        eigmin, _ = diag_slepc(Hpetsc - eigmax * Spetsc, Spetsc, 1, 'LM', tole=0.1, comm=comm_pool)
                        eigmin = eigmin.item()
                        sigma = eigmin + eigmax
                    else:
                        sigma = eigmax
                else:
                    sigma = efermi
                eigs_k, vecs_k = diag_slepc(Hpetsc, Spetsc, nbnd, 'TR', sigma=sigma, comm=comm_pool, tole=tole, max_it=max_it)

                Hpetsc.destroy()
                Spetsc.destroy()

            else:
                assert is_master(comm_pool)
                if efermi is None:
                    eigs_k, vecs_k = eigsh(Hk, k=nbnd, M=Sk, which='SR', tol=tole, maxiter=max_it)
                else:
                    eigs_k, vecs_k = eigsh(Hk, k=nbnd, M=Sk, sigma=efermi, tol=tole, maxiter=max_it)
                vecs_k = vecs_k.T

            if sort:
                argsort = np.argsort(eigs_k)
                eigs_k = eigs_k[argsort]
                if is_master(comm_pool): vecs_k = vecs_k[argsort, :]
            
            eigs[ikpt, :] = eigs_k
            if is_master(comm=comm_pool):
                wfnao[ikpt, :, :] = vecs_k
            
        # Collect eigs and wfnao from groups
        if comm is not None:
            comm_m = comm.Split(color=0 if is_master(comm=comm_pool) else 1, key=comm.rank)
            if is_master(comm=comm_pool):
                assert comm_m.rank == self.igrp
                if is_master(comm=comm_m):
                    eigs_recv = np.empty((self.nk, nbnd), dtype='f8')
                    wfnao_recv = np.empty((self.nk, nbnd, self.nao), dtype='c16')
                else:
                    eigs_recv = wfnao_recv = None
                comm_m.Gatherv([eigs, nkloc * nbnd, MPI.REAL8],
                               [eigs_recv, count_k*nbnd, displ_k[:-1]*nbnd, MPI.REAL8], root=0)
                comm_m.Gatherv([wfnao, nkloc*nbnd*self.nao, MPI.COMPLEX16],
                               [wfnao_recv, count_k*nbnd*self.nao, displ_k[:-1]*nbnd*self.nao, MPI.COMPLEX16], root=0)
                eigs = eigs_recv
                wfnao = wfnao_recv
    
        if is_master(comm=comm):
            for ikpt in range(self.nk):
                kpt = self.kpts[ikpt]
                print(f'k ={kpt[0]:9.5f}{kpt[1]:9.5f}{kpt[2]:9.5f}   nbnd ={nbnd:4d}', end='')
                if ikpt in self.hskpos:
                    print('  ', self.hsksymbol[self.hskpos.index(ikpt)])
                else:
                    print()
                for ibnd in range(nbnd):
                    print(f'{eigs[ikpt, ibnd]*hartree2ev:9.4f}', end='')
                    if ibnd % 8 == 7:
                        print()
                if ibnd % 8 != 7:
                    print()

                # debug: check wfn
                # wfn = wfnao[ikpt, 0]
                # Hk = self.matH.r2k(kpt)
                # Sk = self.matS.r2k(kpt)
                # print(np.sum(Sk.dot(wfn) * wfn.conj()))
                # print(np.sum(Hk.dot(wfn) * wfn.conj() * hartree2ev))

        self.eigs = eigs
        self.wfnao = wfnao
    
    @mpi_watch
    def write(self, path='./', eigfname='eig.dat'):
        if not is_master():
            return
        nbnd = self.eigs.shape[1]
        f = open(f'{path}/{eigfname}', 'w')
        f.write('Band energies in eV\n')
        f.write('      nk    nbnd\n')
        f.write(f'{self.nk:8d}{nbnd:8d}\n')
        for ikpt in range(self.nk):
            kpt = self.kpts[ikpt]
            f.write(f'{kpt[0]:13.9f}{kpt[1]:13.9f}{kpt[2]:13.9f}{nbnd:8d}')
            if ikpt in self.hskpos:
                f.write('  ' + self.hsksymbol[self.hskpos.index(ikpt)] + '\n')
            else:
                f.write('\n')
            for ibnd in range(nbnd):
                f.write(f'{1:8d}{ibnd+1:8d}{self.eigs[ikpt, ibnd]*hartree2ev:15.9f}\n')
        f.close()
        # todo: write wavefunctions
    

def mat_scipy2petsc(matcsr, comm=comm):
    if is_master(comm=comm):
        auxdata = np.array([matcsr.shape[0], matcsr.shape[1], len(matcsr.indptr), len(matcsr.indices)], dtype='i8')
    else:
        auxdata = np.empty(4, dtype='i8')
    if comm is not None:
        comm.Bcast([auxdata, 4, MPI.INTEGER8], root=0)
    shape1, shape2, len_indptr, len_data = auxdata

    # if is_master():
    #     print('Matrix size:', shape1)

    A = PETSc.Mat().create(comm=comm)
    A.setSizes([shape1, shape2])
    A.setFromOptions()
    A.setUp()

    ranges = A.getOwnershipRanges()
    rstart, rend = A.getOwnershipRange()

    if is_master(comm=comm):
        indptr = matcsr.indptr
        assert indptr.dtype == np.int32
    else:
        indptr = np.empty(len_indptr, dtype='i4')
    if comm is not None:
        comm.Bcast([indptr, len_indptr, MPI.INTEGER4], root=0)
    
    # indices
    ranges_dat = indptr[ranges]
    counts = np.diff(ranges_dat)
    disps = ranges_dat[:-1]
    if is_master(comm=comm):
        send = matcsr.indices
        assert send.dtype == np.int32
    else:
        send = None
    if comm is not None:
        count_loc = counts[comm.rank]
        indices = np.empty(count_loc, dtype='i4')
        comm.Scatterv([send, counts, disps, MPI.INTEGER4],
                      [indices, count_loc, MPI.INTEGER4], root=0)
    else:
        indices = send
    
    # data
    if is_master(comm=comm):
        send = matcsr.data
        assert send.dtype == np.complex128
    else:
        send = None
    if comm is not None:
        count_loc = counts[comm.rank]
        data = np.empty(count_loc, dtype='c16')
        comm.Scatterv([send, counts, disps, MPI.COMPLEX16],
                      [data, count_loc, MPI.COMPLEX16], root=0)
    else:
        data = send
    
    indptr = indptr[rstart:rend+1] - indptr[rstart]
    A.setValuesCSR(indptr, indices, data)
    A.assemble()

    return A
    

def diag_slepc(matA, matB, nev, which, sigma=None, tole=1e-8, max_it=100, PC=None, comm=comm):
    '''
    matA, matB: PETSc matrices
    nev: number of eigenvalues
    which: LM = largest magnitude; TR = target real
    sigma: must provide sigma when which=TR
    '''

    size, _ = matA.getSize()
    # get the ownership of petsc vectors within pool
    V = PETSc.Vec().create(comm=comm)
    V.setSizes(size)
    V.setUp()
    vranges = V.getOwnershipRanges()
    vrange_loc = V.getOwnershipRange()
    V.destroy()
    # print(vranges)
    # print(vrange_loc)
    vec_loc = np.empty(vrange_loc[1] - vrange_loc[0], dtype='c16')
    counts_vec = np.diff(vranges)
    disps_vec = vranges[:-1]
    
    eigs = np.empty(nev)
    if is_master(comm):
        vecs = np.empty((nev, size), dtype='c16')
    else:
        vecs = None

    Eps = SLEPc.EPS()
    Eps.create(comm=comm)
    Eps.setOperators(matA, matB)
    Eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
    Eps.setDimensions(nev=nev)
    Eps.setType(Eps.Type.KRYLOVSCHUR)
    if which == 'TR':
        assert sigma is not None
        Eps.setWhichEigenpairs(Eps.Which.TARGET_REAL)
        Eps.setTarget(sigma)
    elif which == 'LM':
        Eps.setWhichEigenpairs(Eps.Which.LARGEST_MAGNITUDE)
    else:
        raise NotImplementedError(which)
    Eps.setTolerances(tol=tole, max_it=max_it)
    Eps.setFromOptions()

    # set numerical tolerence
    BV = Eps.getBV()
    BV.setDefiniteTolerance(1e-8)

    ST = Eps.getST()
    KSP = ST.getKSP()
    if which == 'TR':
        # set spectral transformation type
        ST.setType(ST.Type.SINVERT) # shift-and-invert is the most stable
    elif which == 'LM':
        pass

    if PC is None:
        PC = KSP.getPC()
        # set factorization solver
        PC.setType(PC.Type.LU)
        if comm.size == 1:
            PC.setFactorSolverType(PETSc.Mat().SolverType.MKL_PARDISO)
        else:
            PC.setFactorSolverType(PETSc.Mat().SolverType.MKL_CPARDISO)
    else:
        KSP.setPC(PC)


    Eps.solve()

    nconv = Eps.getConverged()
    msg = 'Some bands not converged'
    assert nconv >= nev, msg

    vr, wr = matA.getVecs()
    vi, wi = matA.getVecs()

    for iev in range(nev):
        e = Eps.getEigenpair(iev, vr, vi)
        eigs[iev] = e.real
        # if ibnd == 0: print(e.real * hartree2ev)
        vr_np = vr.getArray(readonly=True)
        vi_np = vi.getArray(readonly=True)
        vec_loc[:] = vr_np + 1j * vi_np
        if comm is not None and comm.size > 1:
            recv = vecs[iev] if is_master(comm=comm) else None
            comm.Gatherv([vec_loc, len(vec_loc), MPI.COMPLEX16],
                         [recv, counts_vec, disps_vec, MPI.COMPLEX16], root=0)
        else:
            vecs[iev, :] = vec_loc
    
    # Eps.view()
    Eps.destroy()
    
    return eigs, vecs