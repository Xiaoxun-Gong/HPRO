import numpy as np
from scipy.io import FortranFile

from .utils import KGData, VKBGData, tqdm_mpi_tofile, one_by_one, set_range, is_master

'''
This module contains functions for reading BerkeleyGW related files
'''

class bgw_vsc:
    '''
    VSC file: stores the FT of total effective local potential
    '''
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
                self.tau[iat, d] = rec[iat*4+d]
            self.atomic_number[iat] = rec[iat*4+3]
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
        self.vscg = self.file.read_record('c16').reshape(self.nsf, self.ng_g) * 0.5 # Ry->Ha

def bgw_filetype(filepath):
    file = FortranFile(filepath)
    rec = file.read_record('S1').tobytes().decode()
    stitle = rec[0:32].rstrip(' ')
    sdate = rec[32:64].rstrip(' ')
    stime = rec[64:96].rstrip(' ')
    return stitle
