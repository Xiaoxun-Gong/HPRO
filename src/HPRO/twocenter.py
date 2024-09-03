from math import pi

import numpy as np
import scipy.special as sp

from .utils import slice_same
from .orbutils import LinearRGD, GridFunc
from .from_gpaw.gaunt import gaunt
# from .lcaodata import pwc
from .matlcao import MatLCAO, pwc
from .constants import TWOCENTER_RGRID_DEN

class TwoCenterIntgSplines:
    '''
    For two orbitals, we first calculate the radial part of the overlap matrix elements (S_l(R) in Eq. 28). This is computed
    on a radial grid, and then splines are computed to interpolate S_l(R). This is done in the `__init__` method. 

    In the `calc` method, the relative position vectors between two orbitals are provided, and we calculate the corresponding
    overlap matrix. We first interpolate S_l(R) to the distance between orbitals. Then product of S_l(R), Gaunt coeffcients 
    and spherical harmonics are calculated according to Eq. 26.
    '''
    def __init__(self, gridfuncQ1, gridfuncQ2, rcut, kind=1, G1=None):
        # kind: 1 - overlap; 2 - kinetic matrix element
        
        grid_nR = int(rcut * TWOCENTER_RGRID_DEN)
        gridQ = gridfuncQ1.rgd
        assert gridfuncQ1.rgd == gridfuncQ2.rgd
        l1 = gridfuncQ1.l
        l2 = gridfuncQ2.l
        
        # gridfuncQ1 = grid_R2G(gridQ, gridfunc1, l1)
        # gridfuncQ2 = grid_R2G(gridQ, gridfunc2, l2)
        gridR = LinearRGD(0, rcut, grid_nR)
        G1 = gaunt(1)
        
        Slm_R = np.empty((3, gridR.npoints)) # (l, R)
        for iR in range(gridR.npoints): # todo: parallize
            R = gridR.rfunc[iR]
            for l in range(0, 3):
                kr = gridQ.rfunc * R
                j_l = sp.spherical_jn(l, kr)
                prefac = (-1)**((l1-l2-l)//2) / (2*pi**2)
                tmp = gridQ.sips(j_l*gridfuncQ1.func*gridfuncQ2.func, n=2) * prefac
                if kind == 2: tmp /= 2
                Slm_R[l, iR] = tmp
                
        Slm_R_func = []
        for l in range(0, 3):
            func = GridFunc(gridR, Slm_R[l], l=l)
            func.calc_generator()
            Slm_R_func.append(func)
        
        self.l1 = l1
        self.l2 = l2
        self.GLLL = G1[:, l1**2:(l1+1)**2, l2**2:(l2+1)**2]
        self.Slm_R_func = Slm_R_func
    
    def calc(self, Rvec):
        '''
        Rvec(..., 3): relative position of two orbitals
        '''
        
        l=1
        rshape0 = Rvec.shape[:-1]
        Rvec = Rvec.reshape(-1, 3)
        SR_lm = np.empty((Rvec.shape[0], 9))
        pos = 0
        SR_lm[:, pos:pos+2*l+1] = self.Slm_R_func[l].generate3D(Rvec)
        SR_lm_lm = np.sum(self.GLLL[:, :, :] * 
                          SR_lm[:, :, None], axis=1)
        
        return SR_lm_lm.reshape(rshape0 + (2*self.l1+1, 2*self.l2+1))


def calc_overlap(lcaodata1, dictatuple, lcaodata2=None, Ecut=50):
    '''
    Calculate the overlap matrices.

    Parameters:
        Ecut: cutoff energy of radial grid in reciprocal space, in Hartree
    '''

    is_selfolp = lcaodata2 is None
    if is_selfolp:
        lcaodata2 = lcaodata1
    else:
        assert lcaodata1.structure == lcaodata2.structure
    
    stru = lcaodata1.structure

    lcaodata1.calc_phiQ(Ecut)
    if not is_selfolp:
        lcaodata2.calc_phiQ(Ecut)
        
    pairs_ij = pwc(stru, lcaodata1.cutoffs, cutoffs2=lcaodata2.cutoffs)
    overlaps = MatLCAO.setc(pairs_ij, lcaodata1, lcaodata2=lcaodata2, filling_value=None)
    
    translations = overlaps.translations
    atom_pairs = overlaps.atom_pairs
    spc_pairs = stru.atomic_numbers[atom_pairs]
    
    slices_ij = slice_same(spc_pairs[:, 0] * 200 + spc_pairs[:, 1])
    
    for ix_ij in range(len(slices_ij) - 1):
        start_ij = slices_ij[ix_ij]
        end_ij = slices_ij[ix_ij + 1]
        nthisij = end_ij - start_ij
        thisij = slice(start_ij, end_ij)
        spc1, spc2 = spc_pairs[start_ij]
        
        size1 = lcaodata1.orbslices_spc[spc1][-1]
        size2 = lcaodata2.orbslices_spc[spc2][-1]
        S_thisij = np.empty((nthisij, size1, size2))
        
        pos_ij = stru.atomic_positions_cart[atom_pairs[thisij, :]]
        Rs_thisij = translations[thisij, :] @ stru.rprim + pos_ij[:, 1] - pos_ij[:, 0]
        orbpairs_thisij = dictatuple[(spc1, spc2)]
        ix_orbpair = 0
        for jorb in range(lcaodata2.norb_spc[spc2]):
            for iorb in range(lcaodata1.norb_spc[spc1]):
                slice1 = slice(lcaodata1.orbslices_spc[spc1][iorb],
                               lcaodata1.orbslices_spc[spc1][iorb+1])
                slice2 = slice(lcaodata2.orbslices_spc[spc2][jorb],
                               lcaodata2.orbslices_spc[spc2][jorb+1])
                orbpair = orbpairs_thisij[ix_orbpair]
                olp = orbpair.calc(Rs_thisij)
                S_thisij[:, slice1, slice2] = olp

                ix_orbpair += 1
        
        for ii in range(nthisij):
            overlaps.mats[start_ij + ii] = S_thisij[ii]
    
    if is_selfolp:
        overlaps.duplicate()
        
    return overlaps