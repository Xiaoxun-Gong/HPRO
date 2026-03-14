from math import pi

import numpy as np
import scipy.special as sp

from .. import config as CFG
from ..utils.misc import slice_same
from ..utils.orbutils import LinearRGD, GridFunc
from ..from_gpaw.gaunt import gaunt
from ..matao.matao import MatAO
from ..matao.findpairs import pairs_within_cutoff

class TwoCenterIntgSplines:
    '''
    For two orbitals, we first calculate the radial part of the overlap matrix elements (S_l(R) in Eq. 28). This is computed
    on a radial grid, and then splines are computed to interpolate S_l(R). This is done in the `__init__` method. 

    In the `calc` method, the relative position vectors between two orbitals are provided, and we calculate the corresponding
    overlap matrix. We first interpolate S_l(R) to the distance between orbitals. Then product of S_l(R), Gaunt coeffcients 
    and spherical harmonics are calculated according to Eq. 26.
    '''
    def __init__(self, gridfuncQ1, gridfuncQ2, rcut, kind=1, GLLL=None):
        # kind: 1 - overlap; 2 - kinetic matrix element
        
        grid_nR = int(rcut * CFG.TWOCENTER_RGRID_DEN)
        gridQ = gridfuncQ1.rgd
        assert gridfuncQ1.rgd == gridfuncQ2.rgd
        l1 = gridfuncQ1.l
        l2 = gridfuncQ2.l
        
        # gridfuncQ1 = grid_R2G(gridQ, gridfunc1, l1)
        # gridfuncQ2 = grid_R2G(gridQ, gridfunc2, l2)
        gridR = LinearRGD(0, rcut, grid_nR)
        lmax = max(l1, l2)
        
        Slm_R = np.empty((2*lmax+1, gridR.npoints)) # (l, R)
        for iR in range(gridR.npoints): # todo: parallize
            R = gridR.rfunc[iR]
            for l in range(0, 2*lmax+1):
                kr = gridQ.rfunc * R
                j_l = sp.spherical_jn(l, kr)
                prefac = (-1)**((l1-l2-l)//2) / (2*pi**2)
                if kind == 1:
                    Slm_R[l, iR] = gridQ.integrate(j_l*gridfuncQ1.func*gridfuncQ2.func, n=2) * prefac
                elif kind == 2:
                    Slm_R[l, iR] = gridQ.integrate(j_l*gridfuncQ1.func*gridfuncQ2.func, n=4) * prefac/2
                else:
                    raise NotImplementedError
                
        Slm_R_func = []
        for l in range(0, 2*lmax+1):
            func = GridFunc(gridR, Slm_R[l], l=l)
            func.calc_spline()
            Slm_R_func.append(func)
        
        if GLLL is None:
            GLLL = gaunt(lmax)
        
        self.lmax = lmax
        self.l1 = l1
        self.l2 = l2
        self.GLLL = GLLL[0:(2*lmax+1)**2, l1**2:(l1+1)**2, l2**2:(l2+1)**2]
        self.Slm_R_func = Slm_R_func
    
    def calc(self, Rvec):
        '''
        Rvec(..., 3): relative position of two orbitals
        '''
            
        rshape0 = Rvec.shape[:-1]
        Rvec = Rvec.reshape(-1, 3)
        SR_lm = np.empty((Rvec.shape[0], (2*self.lmax+1)**2))
        pos = 0
        for l in range(0, 2*self.lmax+1):
            SR_lm[:, pos:pos+2*l+1] = self.Slm_R_func[l].getval3D(Rvec)
            pos += 2 * l + 1
        SR_lm_lm = np.sum(self.GLLL[None, :, :, :] * 
                          SR_lm[:, :, None, None], axis=1)
        
        return SR_lm_lm.reshape(rshape0 + (2*self.l1+1, 2*self.l2+1))

def calc_overlap(aodata1, aodata2=None, Ecut=50, kind=1):
    '''
    Calculate the overlap matrices.

    Parameters:
        Ecut: cutoff energy of radial grid in reciprocal space, in Hartree
        kind: 1 - overlap; 2 - kinetic matrix element
    '''

    is_selfolp = aodata2 is None
    if is_selfolp:
        aodata2 = aodata1
    else:
        assert aodata1.structure == aodata2.structure
    
    stru = aodata1.structure

    aodata1.calc_phiQ(Ecut)
    if not is_selfolp:
        aodata2.calc_phiQ(Ecut)
    
    
    # find lmax and GLLL
    lmax = 0
    for spc in stru.atomic_species:
        l1max = max(aodata1.ls_spc[spc])
        l2max = 0 if is_selfolp else max(aodata2.ls_spc[spc])
        lmax = max(l1max, l2max, lmax)
    GLLL = gaunt(lmax)
    
    # initialize splines of two-center integral
    orbpairs = {} # Dict[(int, int) -> List]
    for ispc in range(stru.nspc):
        range2 = range(ispc, stru.nspc) if is_selfolp else range(stru.nspc)
        for jspc in range2:
            spc1 = stru.atomic_species[ispc]
            spc2 = stru.atomic_species[jspc]
            orbpairs_thisij = []
            for iorb in range(aodata1.nradial_spc[spc1]):
                # for Z1==Z2: only needs to calculate half of the splines
                istartj = iorb if (is_selfolp and (spc1==spc2)) else 0
                for jorb in range(istartj, aodata2.nradial_spc[spc2]):
                    r1 = aodata1.phirgrids_spc[spc1][iorb].rcut
                    r2 = aodata2.phirgrids_spc[spc2][jorb].rcut
                    rcut = r1 + r2
                    tic_splines = TwoCenterIntgSplines(aodata1.phiQlist_spc[spc1][iorb],
                                                       aodata2.phiQlist_spc[spc2][jorb],
                                                       rcut, kind=kind,
                                                       GLLL=GLLL)
                    orbpairs_thisij.append(tic_splines)
            orbpairs[(spc1, spc2)] = orbpairs_thisij
    
    pairs_ij = pairs_within_cutoff(stru, aodata1.cutoffs, cutoffs2=aodata2.cutoffs)
    if is_selfolp:
        pairs_ij.sort()
        pairs_ij.remove_ji()
    overlaps = MatAO.init_mats(pairs_ij, aodata1, aodata2=aodata2, filling_value=None)
    
    translations = overlaps.translations
    atom_pairs = overlaps.atom_pairs
    spc_pairs = stru.atomic_numbers[atom_pairs]
    
    # If self-olp:
    # Pairs are sorted first according to atomic numbers, then according to atomic species.
    # Therefore, pairs with the same atomic species are close to each other.
    # Furthermore, for each pair of atomic species (Z2, Z1) where Z2>Z1, 
    # it must appear after the pair (Z1, Z2).
    
    slices_ij = slice_same(spc_pairs[:, 0] * 200 + spc_pairs[:, 1])
    
    for ix_ij in range(len(slices_ij) - 1):
        start_ij = slices_ij[ix_ij]
        end_ij = slices_ij[ix_ij + 1]
        nthisij = end_ij - start_ij
        thisij = slice(start_ij, end_ij)
        spc1, spc2 = spc_pairs[start_ij]
        
        # allocate S array
        size1 = aodata1.norbfull_spc[spc1]
        size2 = aodata2.norbfull_spc[spc2]
        S_thisij = np.empty((nthisij, size1, size2))
        
        # calculate overlap using splines
        pos_ij = stru.atomic_positions_cart[atom_pairs[thisij, :]]
        Rs_thisij = translations[thisij, :] @ stru.rprim + pos_ij[:, 1] - pos_ij[:, 0]
        orbpairs_thisij = orbpairs[(spc1, spc2)]
        ix_orbpair = 0
        for iorb in range(aodata1.nradial_spc[spc1]):
            istartj = iorb if (is_selfolp and (spc1==spc2)) else 0
            for jorb in range(istartj, aodata2.nradial_spc[spc2]):
                # print(iorb, jorb)
                # print(spc1, spc2, iorb, jorb)
                slice1 = slice(aodata1.orbslices_spc[spc1][iorb],
                               aodata1.orbslices_spc[spc1][iorb+1])
                slice2 = slice(aodata2.orbslices_spc[spc2][jorb],
                               aodata2.orbslices_spc[spc2][jorb+1])
                orbpair_splines = orbpairs_thisij[ix_orbpair]
                olp = orbpair_splines.calc(Rs_thisij)
                S_thisij[:, slice1, slice2] = olp
                if is_selfolp and (spc1==spc2) and (jorb>iorb):
                    olp = orbpair_splines.calc(-Rs_thisij)
                    S_thisij[:, slice2, slice1] = olp.transpose(0, 2, 1)
                ix_orbpair += 1
        
        # send overlaps to their correct positions
        for ii in range(nthisij):
            # ipair = argsort_ijji[start_ij + ii]
            overlaps.mats[start_ij + ii] = S_thisij[ii]
    
    if is_selfolp:
        overlaps.unfold_with_hermiticity()
        
    return overlaps
