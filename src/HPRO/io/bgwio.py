import numpy as np
from scipy.io import FortranFile

from ..utils.misc import KGData, VKBGData, mytqdm, set_range
from ..utils.mpi import is_master, one_by_one

'''
This module contains classes for reading BerkeleyGW related files, including WFN, VSC, VKB
'''

class bgw_wfn:
    def __init__(self, wfnfilename):
        self.file = FortranFile(wfnfilename)
        self._header_read = False
    
    def close(self):
        self.file.close()

    def read_header(self):
        assert not self._header_read
        
        rec = self.file.read_record('S1').tobytes().decode()
        self.stitle = rec[0:32].rstrip(' ')
        self.sdate = rec[32:64].rstrip(' ')
        self.stime = rec[64:96].rstrip(' ')
        # print(stitle, sdate, stime)
        
        rec = self.file.read_record('i4', 'i4', 'i4', 'i4', 'i4', 'f8', 'i4', 'i4', 'i4', 'f8')
        rec = map(lambda x: x.item(), rec)
        self.nsf, self.ng_g, self.ntran, self.cell_symmetry, self.nat, self.ecutrho, self.nk, self.nb, self.npwx_g, self.ecutwfc = rec # nk = nk_g / ns
        # print(nsf, ng_g, ntran, cell_symmetry, nat, ecutrho, nk, nb, npwx_g, ecutwfc)
        
        # should have selected wfng=.true. when generating WFN
        rec = self.file.read_record(*(['i4']*6 + ['f8']*3))
        rec = map(lambda x: x.item(), rec)
        self.nr1, self.nr2, self.nr3, self.wfng_nk1, self.wfng_nk2, self.wfng_nk3, self.wfng_dk1, self.wfng_dk2, self.wfng_dk3 = rec
        # print(nr1, nr2, nr3, wfng_nk1, wfng_nk2, wfng_nk3, wfng_dk1, wfng_dk2, wfng_dk3)
        
        rec = self.file.read_record('f8')
        self.omega = rec[0] # cell volume, in bohr^3
        self.alat = rec[1] # in bohr
        self.at = rec[2:11].reshape(3, 3)
        self.adot = rec[11:20].reshape(3, 3)
        # print(omega, alat)
        # print(at)
        # print(adot) # ai dot aj, in bohr^2
        
        rec = self.file.read_record('f8')
        self.recvol = rec[0]
        self.tpiba = rec[1] # in bohr-1
        self.bg = rec[2:11].reshape(3, 3) # reciprocal lattice vecs
        self.bdot = rec[11:20].reshape(3, 3)
        # print(recvol, tpiba)
        # print(bg)
        # print(adot)
        
        rec = self.file.read_record('i4')
        self.rotmat = rec.reshape(self.ntran, 3, 3) # rotation matrices
        # print(rotmat)
        
        rec = self.file.read_record('f8')
        self.frac_tran = rec.reshape(self.ntran, 3) # fractional translations
        # print(frac_tran)
        
        rec = self.file.read_record(*(['f8']*3 + ['i4'])*self.nat)
        # print(rec)
        self.tau = np.empty((self.nat, 3), dtype=float) # (nat, 3) atomic positions (alat)
        self.atomic_number = np.empty(self.nat, dtype=int) # (nat) atomic numbers
        for iat in range(self.nat):
            for d in range(3):
                self.tau[iat, d] = rec[iat*4+d][0]
            self.atomic_number[iat] = rec[iat*4+3][0]
        # print(tau) # alat
        # print(atomic_number)
        
        self.ngk_g = self.file.read_record('i4') # (nk) number of g-vecs at ks
        # print(ngk_g)
        
        self.wk = self.file.read_record('f8') # (nk) weight of ks
        # print(wk)
        
        rec = self.file.read_record('f8')
        self.xk = rec.reshape(self.nk, 3) # (nk, 3) k coordinates (crystal)
        # print(xk)
        
        self.ifmin = self.file.read_record('i4') # (nk*ns) filled bands
        self.ifmax = self.file.read_record('i4')
        # print(ifmin)
        # print(ifmax)
        
        rec = self.file.read_record('f8')
        self.et_g = rec.reshape(self.nk*self.nsf, self.nb) # (nk*ns, nb) energy (Ry)
        # print(et_g)
        
        rec = self.file.read_record('f8')
        self.wg_g = rec.reshape(self.nk*self.nsf, self.nb) # (nk*ns, nb) occupations
        # print(wg_g) 
        
        nrecord = self.file.read_record('i4').item()
        self.ng_g = self.file.read_record('i4').item() # number of charge_density g-vecs
        # print(nrecord, ng_g)
        
        rec = self.file.read_record('i4')
        self.g_g = rec.reshape(self.ng_g, 3) # charge_density g-vecs miller indices
        # np.savetxt('g_g.dat', g_g, fmt='%3i')
        
        self._header_read = True
        
    def read_data(self, bandrange=(None,None), gvecrange=None):
        # bandrange: 2-element tuple, [start, stop)
        # gvecrange: (nk, 2) [start, stop)
        assert self._header_read
        bandrange1 = set_range(bandrange, 0, self.nb)
        
        self.kgdatas = []
        for ik in mytqdm(range(self.nk)):
            
            # pw2bgw.f90 line 1062: if (is.eq.0)
            # not sure about this
            
            nrecord = self.file.read_record('i4').item()
            ngk_g = self.file.read_record('i4').item()
            rec = self.file.read_record('i4')
            gk_g = rec.reshape(ngk_g, 3) # (ngk_g, 3), miller indices at k
            
            gvecrange1 = (0 if gvecrange is None else gvecrange[ik, 0],
                          ngk_g if gvecrange is None else gvecrange[ik, 1])
            assert 0<=gvecrange1[0] and gvecrange1[0]<=gvecrange1[1] and gvecrange1[1]<=ngk_g
            
            gk_g = gk_g[gvecrange1[0]:gvecrange1[1], :]
            
            kgcart = (self.xk[ik][None, :] + gk_g) @ self.bg * self.tpiba
            unkg = np.empty((bandrange1[1]-bandrange1[0], gvecrange1[1]-gvecrange1[0]), dtype=np.complex128)
            for ispin in range(self.nsf): # now only works for nsf=1
                with one_by_one():
                    for ib in range(self.nb):
                        if ib<bandrange1[0] or ib>=bandrange1[1]:
                            self.file._fp.seek((4+8)*2+ngk_g*16+8, 1) 
                            # At the start and end of each record, there is an integer (4 bytes) indicating record length.
                            continue
                        nrecord = self.file.read_record('i4').item()
                        ngk_g = self.file.read_record('i4').item()
                        # only supports no spin
                        self.file._fp.seek(4+16*gvecrange1[0], 1)
                        unkg[ib-bandrange1[0], :] = np.fromfile(self.file._fp, dtype='c16', count=gvecrange1[1]-gvecrange1[0])
                        self.file._fp.seek(4+16*(ngk_g-gvecrange1[1]), 1)
            
            kgdata = KGData(gvecrange[ik,1]-gvecrange[ik,0], self.nb, gk_g, unkg, kgcart)
            self.kgdatas.append(kgdata)
        
        # print(f.read_record('S1'))


class bgw_vsc:
    def __init__(self, vscfile):
        self.file = FortranFile(vscfile, 'r')
        self._header_read = False
    
    def close(self):
        self.file.close()

    def read_header(self):        
        assert not self._header_read
        
        rec = self.file.read_record('S1').tobytes().decode()
        self.stitle = rec[0:32].rstrip(' ')
        self.sdate = rec[32:64].rstrip(' ')
        self.stime = rec[64:96].rstrip(' ')
        # print(self.stitle, self.sdate, self.stime)
        
        rec = self.file.read_record('i4', 'i4', 'i4', 'i4', 'i4', 'f8')
        rec = map(lambda x: x.item(), rec)
        self.nsf, self.ng_g, self.ntran, self.cell_symmetry, self.nat, self.ecutrho = rec # nk = nk_g / ns
        # print(self.nsf, self.ng_g, self.ntran, self.cell_symmetry, self.nat, self.ecutrho)
        if self.nsf == 4:
            self.ns = 1
            self.nspinor = 2
        else:
            self.ns = self.nsf
            self.nspinor = 1
        
        rec = self.file.read_record(*(['i4']*3))
        rec = map(lambda x: x.item(), rec)
        self.nr1, self.nr2, self.nr3 = rec

        rec = self.file.read_record('f8')
        self.omega = rec[0] # cell volume, in bohr^3
        self.alat = rec[1] # in bohr
        self.at = rec[2:11].reshape(3, 3)
        self.adot = rec[11:20].reshape(3, 3)
        # print(omega, alat)
        # print(at)
        # print(adot) # ai dot aj, in bohr^2
        
        rec = self.file.read_record('f8')
        self.recvol = rec[0]
        self.tpiba = rec[1] # in bohr-1
        self.bg = rec[2:11].reshape(3, 3) # reciprocal lattice vecs
        self.bdot = rec[11:20].reshape(3, 3)
        # print(recvol, tpiba)
        # print(bg)
        # print(adot)
        
        rec = self.file.read_record('i4')
        self.rotmat = rec.reshape(self.ntran, 3, 3) # rotation matrices
        # print(rotmat)
        
        rec = self.file.read_record('f8')
        self.frac_tran = rec.reshape(self.ntran, 3) # fractional translations
        # print(frac_tran)
        
        rec = self.file.read_record(*(['f8']*3 + ['i4'])*self.nat)
        self.tau = np.empty((self.nat, 3), dtype=float) # (nat, 3) atomic positions (alat)
        self.atomic_number = np.empty(self.nat, dtype=int) # (nat) atomic numbers
        for iat in range(self.nat):
            for d in range(3):
                self.tau[iat, d] = rec[iat*4+d][0]
            self.atomic_number[iat] = rec[iat*4+3][0]
        # print(tau) # alat
        # print(atomic_number)
        
        self._header_read = True
        
    def read_data(self):
        
        assert self._header_read
        
        nrecord = self.file.read_record('i4').item()
        self.ng_g = self.file.read_record('i4').item() # number of charge_density g-vecs
        self.g_g = self.file.read_record('i4').reshape(self.ng_g, 3)
        # print(nrecord, ng_g)
        
        nrecord = self.file.read_record('i4').item()
        self.ng_g = self.file.read_record('i4').item() # number of charge_density g-vecs
        self.vscg = self.file.read_record('c16').reshape(self.ns, self.ng_g) * 0.5 # Ry->Ha


class bgw_vkb:
    def __init__(self, vkbfilename):
        self.file = FortranFile(vkbfilename)
        self._header_read = False
    
    def close(self):
        self.file.close()
        
    def read_header(self):
        assert not self._header_read
        
        rec = self.file.read_record('S1').tobytes().decode()
        self.stitle = rec[0:32].rstrip(' ')
        self.sdate = rec[32:64].rstrip(' ')
        self.stime = rec[64:96].rstrip(' ')
        # print(stitle, sdate, stime)
        
        rec = self.file.read_record('i4', 'i4', 'i4', 'i4', 'i4', 'f8', 'i4', 'i4', 'i4', 'i4', 'i4', 'f8')
        rec = map(lambda x: x.item(), rec)
        self.nsf, self.ng_g, self.ntran, self.cell_symmetry, self.nat, self.ecutrho, self.nk, self.nsp, self.nkb, self.nhm, self.npwx_g, self.ecutwfc = rec # nk = nk_g / ns
        # print(nsf, ng_g, ntran, cell_symmetry, nat, ecutrho, nk, nsp, nkb, nhm, npwx_g, ecutwfc)
        
        # should have selected wfng=.true. when generating WFN
        rec = self.file.read_record(*(['i4']*6 + ['f8']*3))
        rec = map(lambda x: x.item(), rec)
        self.nr1, self.nr2, self.nr3, self.wfng_nk1, self.wfng_nk2, self.wfng_nk3, self.wfng_dk1, self.wfng_dk2, self.wfng_dk3 = rec
        # print(nr1, nr2, nr3, wfng_nk1, wfng_nk2, wfng_nk3, wfng_dk1, wfng_dk2, wfng_dk3)
        
        rec = self.file.read_record('f8')
        self.omega = rec[0] # cell volume, in bohr^3
        self.alat = rec[1] # in bohr
        self.at = rec[2:11].reshape(3, 3)
        self.adot = rec[11:20].reshape(3, 3)
        # print(omega, alat)
        # print(at)
        # print(adot) # ai dot aj, in bohr^2
        
        rec = self.file.read_record('f8')
        self.recvol = rec[0]
        self.tpiba = rec[1] # in bohr-1
        self.bg = rec[2:11].reshape(3, 3) # reciprocal lattice vecs
        self.bdot = rec[11:20].reshape(3, 3)
        # print(recvol, tpiba)
        # print(bg)
        # print(adot)
        
        rec = self.file.read_record('i4')
        self.rotmat = rec.reshape(self.ntran, 3, 3) # rotation matrices
        # print(rotmat)
        
        rec = self.file.read_record('f8')
        self.frac_tran = rec.reshape(self.ntran, 3) # fractional translations
        # print(frac_tran)
        
        rec = self.file.read_record(*(['f8']*3 + ['i4'])*self.nat)
        self.tau = np.empty((self.nat, 3), dtype=float) # (nat, 3) atomic positions (alat)
        self.atomic_number = np.empty(self.nat, dtype=int) # (nat) atomic numbers
        for iat in range(self.nat):
            for d in range(3):
                self.tau[iat, d] = rec[iat*4+d][0]
            self.atomic_number[iat] = rec[iat*4+3][0]
        # print(tau) # alat
        # print(self.atomic_number)
        
        self.ngk_g = self.file.read_record('i4') # (nk) number of g-vecs at ks
        # print(ngk_g)
        
        self.wk = self.file.read_record('f8') # (nk) weight of ks
        # print(wk)
        
        rec = self.file.read_record('f8')
        self.xk = rec.reshape(self.nk, 3) # (nk, 3) k coordinates (crystal)
        # print(xk)
        
        self.ityp = self.file.read_record('i4') # (nat)
        # print(self.ityp)

        self.nh = self.file.read_record('i4') # (nsp)
        # print(self.nh)
        
        # definition of deeq can be found in PW/src/add_vupsi.f90: module add_vupsi: subroutine add_vupsi_nc]
        # order of spin indices: (up, up), (up, down), (down, up), (down, down)

        deeq = {}
        dtype = 'c16' if self.nsf == 4 else 'f8'
        tmp = self.file.read_record(dtype).reshape(self.nsf, self.nat, self.nhm, self.nhm)
        tmp = tmp.transpose(0, 1, 3, 2) # fortran to C order
        for iat in range(self.nat): 
            spc = self.atomic_number[iat]
            if spc not in deeq:
                nh_sp = self.nh[self.ityp[iat]-1]
                deeq[spc] = np.ascontiguousarray(tmp[:, iat, :nh_sp, :nh_sp]) * 0.5 # Ry->Ha
        self.deeq = deeq
        
        nrecord = self.file.read_record('i4').item()
        self.ng_g = self.file.read_record('i4').item() # number of charge_density g-vecs
        # print(nrecord, ng_g)
        
        rec = self.file.read_record('i4')
        self.g_g = rec.reshape(self.ng_g, 3) # charge_density g-vecs miller indices
        # np.savetxt('g_g.dat', g_g, fmt='%3i')
        
        self._header_read = True
        
    def read_data(self):
        
        assert self._header_read
        
        self.vkbgdatas = []
        for ik in range(self.nk):
            
            nrecord = self.file.read_record('i4').item()
            ngk_g = self.file.read_record('i4').item()
            rec = self.file.read_record('i4')
            gk_g = rec.reshape(ngk_g, 3) # (ngk_g, 3), miller indices at k
            
            kgcart = (self.xk[ik][None, :] + gk_g) @ self.bg * self.tpiba
            vkbg_k = {}
            # vkbg_k = np.empty((self.nkb, ngk_g), dtype='c16')
            for iat in range(self.nat):
                spc = self.atomic_number[iat]
                nh_sp = self.nh[self.ityp[iat]-1]
                if spc not in vkbg_k:
                    tmp = np.empty((nh_sp, ngk_g), dtype='c16')
                    rcart = self.tau[iat] * self.alat
                    phase = np.exp(1j * np.dot(kgcart, rcart))
                for ih in range(nh_sp):
                    if spc in vkbg_k:
                        self.file._fp.seek((4+8)*2+ngk_g*16+8, 1)
                    else:
                        nrecord_ = self.file.read_record('i4').item()
                        ngk_g_ = self.file.read_record('i4').item()
                        tmp[ih, :] = self.file.read_record('c16')
                if spc not in vkbg_k:
                    vkbg_k[spc] = tmp * phase[None, :]
            # for ikb in range(self.nkb):
            #     nrecord = self.file.read_record('i4').item()
            #     ngk_g = self.file.read_record('i4').item()
            #     vkbg_k[ikb, :] = self.file.read_record('c16')
                
            vkbgdata = VKBGData(ngk_g, gk_g, vkbg_k, kgcart)
            self.vkbgdatas.append(vkbgdata)
        
        # for vkbgdata in self.vkbgdatas:
        #     print(vkbgdata.ng, vkbgdata.nkb, vkbgdata.vkbg.shape)


def bgw_filetype(filepath):
    file = FortranFile(filepath)
    rec = file.read_record('S1').tobytes().decode()
    file.close()
    stitle = rec[0:32].rstrip(' ')
    sdate = rec[32:64].rstrip(' ')
    stime = rec[64:96].rstrip(' ')
    return stitle

