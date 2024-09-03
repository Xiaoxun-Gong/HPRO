import numpy as np
import scipy.special as sp

from .from_gpaw.spherical_harmonics import Y

'''
Real spherical harmonics follow the definition:
    https://en.wikipedia.org/wiki/Table_of_spherical_harmonics#Real_spherical_harmonics
Work functions for generating spherical harmonics are adapted from GPAW
'''
    
def spharm(r, l):
    rshape = r.shape
    r = r.reshape((-1, 3))
    _, x, y, z = r_to_xyz(r)
    spharm = spharm_xyz(l, x, y, z)
    return spharm.reshape(rshape[:-1]+(2*l+1,))
    
def r_to_xyz(r):
    '''
    Returns:
        rnorm, x, y, z: x, y, z are normalized
    '''
    eps = 1e-8
    rnorm = np.linalg.norm(r, axis=-1)
    x = r[..., 0] / (rnorm + eps/10)
    y = r[..., 1] / (rnorm + eps/10)
    z = r[..., 2] / (rnorm + eps/10)
    
    # r=0
    r_is_zero = rnorm < eps
    x[r_is_zero] = 0.0
    y[r_is_zero] = 0.0
    z[r_is_zero] = 1.0
    
    return rnorm, x, y, z

def spharm_xyz(l, x, y, z):
    # x, y, z must be normalized
    assert x.shape[0] == y.shape[0] == z.shape[0]
    assert len(x.shape) == len(y.shape) == len(z.shape) == 1
    spharm = np.zeros((x.shape[0], 2*l+1), dtype='f8')
    for m in range(-l, l+1):
        spharm[:, m+l] = Y(l**2+m+l, x, y, z)
    return spharm

def spharm_old(r, l):
    raise DeprecationWarning()
    '''
    Real spherical harmonics:
    https://en.wikipedia.org/wiki/Table_of_spherical_harmonics#Real_spherical_harmonics
    
    Parameters:
    ---------
    r(..., xyz): unnormalized real vectors
    l: integer
    
    Returns:
    ---------
    spharm(..., 2*l+1): spherical harmonics Y_lm(r)
    '''
    
    rshape = r.shape
    r = r.reshape((-1, 3))
    
    eps = 1e-8
    rnorm = np.linalg.norm(r, axis=-1)
    theta = np.arccos(r[..., 2] / (rnorm + eps/10))
    phi = np.arctan2(r[..., 1], r[..., 0])
    # r=0
    r_is_zero = rnorm < eps
    theta[r_is_zero] = 0.0
    phi[r_is_zero] = 0.0
    
    spharm = np.zeros((r.shape[0], 2*l+1), dtype=np.complex128)
    for m in range(-l, l+1):
        if m < 0:
            # note that scipy uses different names for theta, phi
            out = 1.j/np.sqrt(2.0)*(         sp.sph_harm( m, l, phi, theta)
                                    -(-1)**m*sp.sph_harm(-m, l, phi, theta))
        elif m == 0:
            out = sp.sph_harm(0, l, phi, theta)
        else:  # m>0
            out = 1.0/np.sqrt(2.0)*(         sp.sph_harm(-m, l, phi, theta)
                                    +(-1)**m*sp.sph_harm( m, l, phi, theta))
        spharm[..., m+l] = out
    
    assert np.max(np.abs(spharm.imag)) < 1e-8
    spharm = spharm.real

    return spharm.reshape(rshape[:-1]+(2*l+1,))


'''
Utility functions related to FT of atomic orbitals
'''

def spbessel_transfrorm(l, k, rgd, R_r, norm='forward'):
    r'''
    Calculate spherical bessel transformation.
    If norm='forward', then calculate
    F_l(k) = 4 \pi (-i)^l \int dr r^2 R_l(r) j_l(kr)
    If norm='backward', then calculate
    R_l(r) = \frac{1}{(2\pi)^3} 4 \pi i^l \int dk k^2 F_l(k) j_l(kr)
    
    Parameters:
    ---------
    l: integer
    k: float
    R_r(ngrid): function R_l(r) on a radial grid
    rgrid: RadialGrid object
    norm: 'forward' or 'backward'
    
    Returns:
    ---------
    sbt_real: float, F_l(k)
    cplx_phase: phase part of spherical bessel transform
    '''
    
    r = rgd.rfunc
    kr = k * r
    j_l = sp.spherical_jn(l, kr)
    sbt_real = rgd.sips(j_l*R_r)
    if norm=='forward':
        sbt_real *= 4.0*np.pi
        cplx_phase = (-1j)**l
    elif norm=='backward':
        sbt_real *= 1. / (2*np.pi**2)
        cplx_phase = (1j)**l
    else:
        raise ValueError(f'Norm must be "forward" or "backward", not {norm}')
    return sbt_real, cplx_phase


'''
Utility functions related to k-points and g-vecs
'''

class kGsphere:
    '''
    Find all g-vectors in the sphere of given cutoff energy
    '''
    def __init__(self, rprim, ecut):
        maxgnorm = np.sqrt(2 * ecut)
        maxgidx = np.floor(maxgnorm * np.linalg.norm(rprim, axis=-1) / (2*np.pi) + 1).astype(int)
        gk_g_all = np.stack(np.meshgrid(np.linspace(-maxgidx[0], maxgidx[0], 2*maxgidx[0]+1, dtype=int),
                                        np.linspace(-maxgidx[1], maxgidx[1], 2*maxgidx[1]+1, dtype=int),
                                        np.linspace(-maxgidx[2], maxgidx[2], 2*maxgidx[2]+1, dtype=int)),
                            axis=-1).reshape(-1, 3)
        
        self.rprim = rprim
        self.gprim = np.linalg.inv(rprim.T)
        self.maxgnorm = maxgnorm
        self.maxgidx = maxgidx
        self.FFTgrid = 2*(1+maxgidx)
        self.gk_g_all = gk_g_all
        
    def get_gk_g(self, kpt):
        '''
        Only works for -1<=kpt<=1
        kpt(abc): k point vector in reduced coordinate
        '''
        kgcart_all = 2 * np.pi * (kpt[None, :] + self.gk_g_all) @ self.gprim
        within_cutoff = np.linalg.norm(kgcart_all, axis=-1) <= self.maxgnorm
        gk_g = self.gk_g_all[within_cutoff]
        ngk_g = gk_g.shape[0]
        kgcart = kgcart_all[within_cutoff]
        
        return ngk_g, gk_g, kgcart


def same_kpt(kpt1, kpt2):
    eps = 1.0e-5
    return np.all(np.abs(kpt1 - kpt2) < eps, axis=-1)


def diff_by_G(kpt1, kpt2):
    '''
    Check if two kpts are only different by reciprocal lattice vector G

    Parameters:
    ---------
      kpt1: [(extra_dimensions), 3]
      kpt2: [(extra_dimensions), 3]
    
    Returns:
    ---------
      same_kpt [(extra_dimensions)] containing boolean values
    '''
    eps = 1.0e-5
    kdiff = kpt1 - kpt2
    kdiff1BZ = firstBZ(kdiff)
    samekpt = np.all(np.abs(kdiff1BZ) < eps, axis=-1)
    return samekpt


# def same_kpt_sym(kpt1, kpt2, symopt=0):
#     '''
#     Parameters:
#     ---------
#       kpt1: [(extra_dimensions), 3]
#       kpt2: [(extra_dimensions), 3]
#       symopt: 0 - no symmetry; 1 - only time-reversal symmetry
    
#     Returns:
#     ---------
#       same_kpt [(extra_dimensions)] containing boolean values
#     '''
#     assert symopt in [0,  1]
#     if symopt == 0:
#         samekpt = same_kpt(kpt1, kpt2)
#     elif symopt == 1:
#         samekpt = same_kpt(kpt1, kpt2)
#         samekpt += same_kpt(-kpt1, kpt2)
#     return samekpt


def firstBZ(kpt):
    '''
    find kpt in 1BZ (-0.5, 0.5]
    '''
    return -np.divmod(-kpt+0.5, 1)[1] + 0.5


def find_kidx(kpt, kqpt, allow_multiple=True):
    '''
    Find the index of kqpt (one point) in the kpt (all points).
    If multiple match exists, only return the index of the first match.
    If there's no match, then return -1
    '''

    kidx = np.where(same_kpt(kpt, kqpt))[0]
    if not ((kidx.ndim==1) and (kidx.shape[0]>=1)):
        msg = f'\nCannot find kpt {kqpt} among k points:\n'
        msg += str(kpt)
        raise ValueError(msg)
    if (not allow_multiple) and (kidx.shape[0]>1):
        msg = f'Multiple kpt {kqpt} found among k points:\n'
        msg += str(kpt)
        raise ValueError(msg)
    return kidx[0].item()


def make_kkmap(kpt1, kpt2):
    '''
    maps every point in kpt1 to kpt2
    
    Returns:
    ---------
    kkmap[nk2]: kkmap[i] is the index of the k-point in kpt1 corresponding to the ith k-point in kpt2
    '''
    # nk = kpt1.shape[0]
    # assert kpt2.shape[0] == nk
    nk2 = kpt2.shape[0]
    
    kkmap = np.zeros(nk2, dtype=int)
    for ik, kqpt in enumerate(kpt2):
        kkmap[ik] = find_kidx(kpt1, kqpt, allow_multiple=True)
      
    return kkmap


def kgrid_with_tr(gridsize):
    '''
    Generate k grid with time-reversal symmetry
    '''
    if len(gridsize) == 3:
        kgrid = np.array(gridsize)
        shift = np.zeros(3)
    elif len(gridsize) == 6:
        kgrid = np.array(gridsize[:3])
        shift = np.array(gridsize[3:])
    else:
        raise ValueError()
    kpts, kptwts = [], []
    grid_interval = 1 / kgrid
    for n1 in range(kgrid[0]):
        for n2 in range(kgrid[1]):
            for n3 in range(kgrid[2]):
                ns = np.array([n1, n2, n3])
                kpt = (ns + shift) * grid_interval
                found = False
                for ikold in range(len(kpts)):
                    if diff_by_G(-kpt, kpts[ikold]):
                        kptwts[ikold] += 1
                        found = True
                if not found:
                    kpts.append(firstBZ(kpt))
                    kptwts.append(1)
    kptwts = np.array(kptwts)
    kpts = np.stack(kpts)
    total = np.sum(kptwts)
    assert total == np.prod(kgrid)
    kptwts = kptwts / total
    return kpts, kptwts