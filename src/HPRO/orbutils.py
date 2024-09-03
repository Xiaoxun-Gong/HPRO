import xml.etree.ElementTree as ET
from math import pi
import numpy as np
from scipy.integrate import simpson
from scipy.interpolate import CubicSpline
import scipy.special as sp

from .structure import Structure
from .from_gpaw.gaunt import gaunt
from .mathutils import r_to_xyz, spbessel_transfrorm, spharm_xyz

# == radial grids ==

class GridFunc:
    '''
    This is the class for functions that can be separated into a radial part and angular part.
    '''
    def __init__(self, rgd, func, l=0, rcut=None):
        """
        Initializes a GridFunc object.

        Parameters:
            rgd (RadialGrid): The radial grid object.
            func (array): The function values on the radial grid.
            l (int, optional): The angular momentum quantum number. Defaults to 0.
            rcut (float, optional): The radial cutoff. If None, it will be automatically detected. Defaults to None.
        """
        self.rgd = rgd
        self.func = func
        self.generator = None
        self.l = l
        
        # get rcut
        if rcut is None and func is not None:
            # automatically detect rcut
            # deeph case, there is no rgd and func
            assert len(func) == rgd.npoints
            norm = rgd.sips(func**2)
            nonzero, = np.where(np.abs(func)[1:] * np.diff(rgd.rfunc) * rgd.rfunc[1:]**2 >= norm * 1e-4)
            if len(nonzero) == 0 or nonzero[-1]>=len(func)-1:
                last_nonzero = len(func) - 1
            else:
                last_nonzero = nonzero[-1] + 1
            rcut = self.rgd.rfunc[last_nonzero]
        self.rcut = rcut
    
    def calc_generator(self):
        '''
        Calculate and return a generator for the radial function.
        '''
        self.generator = self.rgd.generator(self.func)
        return self.generator

    def generate(self, r):
        '''
        Generate the atomic orbital function at a given set of radial coordinates.
        '''
        if self.generator is None:
            self.calc_generator()
        assert np.max(r) <= self.rgd.rend
        assert np.min(r) >= self.rgd.rstart
        return self.rgd.generate(self.generator, r)

    def generatexyz(self, Rnorm, x, y, z):
        Ylm = spharm_xyz(self.l, x, y, z)
        Rlm = self.generate(Rnorm)
        return Rlm[:, None] * Ylm[:, :]

    def generate3D(self, Rvec):
        '''
        Generate the atomic orbital function at a given set of cartesian coordinates.
        '''
        rshape0 = Rvec.shape[:-1]
        Rvec = Rvec.reshape(-1, 3)
        Rnorm , x, y, z = r_to_xyz(Rvec)
        # deal with points within the range of the radial grid (outsiders are set to zero)
        Rwithin = Rnorm <= self.rgd.rend
        nwithin = np.sum(Rwithin)
        R_lm = np.zeros((Rvec.shape[0], 2*self.l+1))
        if nwithin > 0:
            Rnorm, x, y, z = Rnorm[Rwithin], x[Rwithin], y[Rwithin], z[Rwithin]
            R_lm[Rwithin, :] = self.generatexyz(Rnorm, x, y, z)
        # deal with points smaller than self.rgd.rstart
        Rsmall = Rnorm < self.rgd.rstart
        nsmall = np.sum(Rsmall)
        if nsmall > 0:
            R_lm[Rsmall, :] = 0. if self.l > 0 else self.func[0]
        return R_lm.reshape(rshape0 + (2*self.l+1,))

    def generate3D_noselect(self, Rvec):
        '''
        Same as `generate3D` but assume all points are already within the range of generated orbitals. 
        This can be more efficient than `generate3D` but use with care.
        '''
        rshape0 = Rvec.shape[:-1]
        Rvec = Rvec.reshape(-1, 3)
        Rnorm, x, y, z = r_to_xyz(Rvec)
        R_lm = self.generatexyz(Rnorm, x, y, z)
        return R_lm.reshape(rshape0 + (2*self.l+1,))

class RadialGrid:
    '''
    This is the base class for radial grids.
    '''
    def __init__(self, rfunc):
        self.rfunc = rfunc
        self.rstart = rfunc[0]
        self.rend = rfunc[-1]
        self.npoints = len(rfunc)
    
    def sips(self, func_on_grid, n=2):
        assert len(func_on_grid) == self.npoints
        return simpson(func_on_grid * self.rfunc**n, self.rfunc)

    def generator(self, func_on_grid):
        return CubicSpline(self.rfunc, func_on_grid)
    
    def generate(self, generator, r):
        return generator(r)
    
    def r2i_ceil(self, r):
        return np.searchsorted(self.rfunc, r)
    
    def __eq__(self, other):
        if self is other:
            return True
        else:
            if not self.__class__ is other.__class__: return False
            if not np.all(np.abs(self.rfunc-other.rfunc)) < 1e-6: return False
            return True
    
class LinearRGD(RadialGrid):
    def __init__(self, rstart, rend, npoints):
        # includes both ends
        assert rend > rstart >= 0
        assert npoints > 1
        rfunc = np.linspace(rstart, rend, npoints)
        super().__init__(rfunc)
        self.dx = (self.rend - self.rstart) / (self.npoints - 1)
    
    def sips(self, func_on_grid, n=2):
        assert len(func_on_grid) <= self.npoints
        return simpson(func_on_grid * self.rfunc**n, dx=self.dx)

    @classmethod
    def from_explicit_grid(cls, rfunc):
        rstart = rfunc[0]
        rend = rfunc[-1]
        npoints = len(rfunc)
        obj = cls(rstart, rend, npoints)
        assert np.all(np.abs(obj.rfunc - rfunc)) < 1e-6
        return obj

class FracPolyRGD(RadialGrid):
    # r=a*i/(n-i)
    def __init__(self, a, n):
        ilist = np.array(list(range(n)))
        rfunc = a * ilist / (n - ilist)
        super().__init__(rfunc)
        self.dr_dx = a * n / (n - ilist) ** 2
        self.a = a
    
    def sips(self, func_on_grid, n=2):
        assert len(func_on_grid) == self.npoints
        # return simpson(func_on_grid * self.dr_dx)
        return simpson(func_on_grid * self.rfunc**n, self.rfunc)

class ExpRGD(RadialGrid):
    # r=a*exp(d*i) or r=a*(exp(d*i)-1)
    def __init__(self, npoints, a, d, minus1=False):
        ilist = np.arange(0, npoints, dtype=float) # istart=0
        rfunc = a * np.exp(d * ilist)
        if minus1:
            rfunc -= a
        super().__init__(rfunc)
        self.a = a
        self.d = d
        self.minus1 = minus1
    
    def sips(self, func_on_grid, n=2):
        assert len(func_on_grid) == self.npoints
        return simpson(func_on_grid * self.rfunc**(n+1), dx=self.d)
    
    @classmethod
    def from_explicit_grid(cls, rfunc):
        a = rfunc[0]
        npoints = len(rfunc) - 1
        d = np.log(rfunc[npoints-1] / rfunc[0]) / (npoints-1)
        npoints = len(rfunc)
        obj = cls(npoints, a, d)
        assert np.all(np.abs(obj.rfunc - rfunc)) < 1e-6
        return obj
    
    def generator(self, func_on_grid):
        return CubicSpline(np.arange(0, self.npoints, dtype=float), func_on_grid)
    
    def generate(self, generator, r):
        assert np.all(r>self.rstart)
        i_fromr = np.log(r/self.a) / self.d 
        return generator(i_fromr)

# grid FTs

def grid_overlap(gridfunc1, gridfunc2):
    '''
    Overlap of two functions centered at the same point
    '''
    assert gridfunc1.rgd == gridfunc2.rgd
    rgd = gridfunc1.rgd
    return rgd.sips(gridfunc1.func * gridfunc2.func)

def grid_G2R(rgrid, gridfuncG, return_real=True):
    '''
    Perform radial Fourier transformation from reciprocal space to real space.
    '''
    # todo: fast Bessel transform
    dtype = 'f8' if return_real else 'c16'
    funcR = np.empty(rgrid.npoints, dtype=dtype)
    for ir in range(rgrid.npoints):
        phi, phase = spbessel_transfrorm(gridfuncG.l, rgrid.rfunc[ir], gridfuncG.rgd, gridfuncG.func, norm='backward')
        funcR[ir] = phi if return_real else phi * phase
    return GridFunc(rgrid, funcR, l=gridfuncG.l)

def grid_R2G(Ggrid, gridfuncR, return_real=True):
    '''
    Perform radial Fourier transformation from real space to reciprocal space.
    '''
    # todo: fast Bessel transform
    dtype = 'f8' if return_real else 'c16'
    funcR = np.empty(Ggrid.npoints, dtype=dtype)
    for ir in range(Ggrid.npoints):
        phi, phase = spbessel_transfrorm(gridfuncR.l, Ggrid.rfunc[ir], gridfuncR.rgd, gridfuncR.func, norm='forward')
        funcR[ir] = phi if return_real else phi * phase
    return GridFunc(Ggrid, funcR, l=gridfuncR.l)
    
# == load orbitals ==

def parse_siesta_ion(filename):
    
    phirgrids = []
    norb = 0
    
    ionfile = open(filename, 'r')
    line = ionfile.readline()
    while line:
        if line.find('#orbital l, n, z, is_polarized, population') > 0:
            sp = line.split()
            l = int(sp[0])
            norb += 1
            
            line = ionfile.readline()
            assert line.split()[0] == '500'
            
            phirgrid = np.zeros((2, 500)) # r, R(r)
            for ipt in range(500):
                phirgrid[:, ipt] = list(map(float, ionfile.readline().split()))
            # found this from sisl/io/siesta/siesta_nc.py: ncSileSiesta.read_basis(self): 
            # sorb = SphericalOrbital(l, (r * Bohr2Ang, psi), orb_q0[io])
            phirgrid[1, :] *= np.power(phirgrid[0, :], l) 
            rgd = LinearRGD.from_explicit_grid(phirgrid[0])
            phirgrids.append(GridFunc(rgd, phirgrid[1], l=l))
                
        line = ionfile.readline()
    ionfile.close()
    
    return norb, phirgrids


def parse_gpaw_basis(filename):
    root = ET.parse(filename).getroot()
    gridfuncs = {}
    for gridfunc in root.findall('radial_grid'):
        if gridfunc.attrib['eq'] == 'r=d*i':
            istart = int(gridfunc.attrib['istart'])
            iend = int(gridfunc.attrib['iend'])
            d = float(gridfunc.attrib['d'])
            rgd = LinearRGD(istart*d, iend*d, iend-istart+1)
            gridid = gridfunc.attrib['id']
            gridfuncs[gridid] = rgd
        else:
            raise NotImplementedError

    phirgrids = []
    for basisfunc in root.findall('basis_function'):
        l = int(basisfunc.attrib['l'])
        gridid = basisfunc.attrib['grid']
        phi = np.array(list(map(float, basisfunc.text.split())))
        gridlen = len(phi)
        rgd = LinearRGD.from_explicit_grid(gridfuncs[gridid].rfunc[:gridlen])
        phirgrids.append(GridFunc(rgd, phi, l=l))

    norb = len(phirgrids)
    
    return norb, phirgrids

def parse_deeph_orbtyps(deephsave):
    stru = Structure.from_deeph(deephsave)
    orbital_types = []
    with open(f'{deephsave}/orbital_types.dat') as f:
        line = f.readline()
        while line:
            orbital_types.append(list(map(int, line.split())))
            line = f.readline()
    orbital_types_spc = {}
    for atom_nbr, orbitals in zip(stru.atomic_numbers, orbital_types):
        if atom_nbr in orbital_types_spc:
            assert orbitals == orbital_types_spc[atom_nbr]
        else:
            orbital_types_spc[atom_nbr] = orbitals
    return orbital_types_spc, stru


class OrbPair:
    def __init__(self, rgrid1, rgrid2, rcut, index=1):
        
        grid_nR = int(rcut * 8.33)
        gridR = LinearRGD(0, rcut, grid_nR)
        gridQ = rgrid1.rgd
        assert rgrid1.rgd == rgrid2.rgd
        l1 = rgrid1.l
        l2 = rgrid2.l
        l3 = max(l1, l2)
        self.lmax = l3
        
        Aalpha = np.empty((2*l3+1, gridR.npoints))
        for iR in range(gridR.npoints):
            R = gridR.rfunc[iR]
            for yy in range(0, 2*l3+1):
                kr = gridQ.rfunc * R
                j_l = sp.spherical_jn(yy, kr)
                xx = (-1)**((l1-l2-yy)//2) / (2*pi**2)
                if index == 1 or index == 3:
                    Aalpha[yy, iR] = gridQ.sips(j_l*rgrid1.func*rgrid2.func, n=2) * xx
                elif index == 2:
                    Aalpha[yy, iR] = gridQ.sips(j_l*rgrid1.func*rgrid2.func, n=4) * xx/2
                
        Aalpha_list = []
        for yy in range(0, 2*l3+1):
            func = GridFunc(gridR, Aalpha[yy], l=yy)
            func.calc_generator()
            Aalpha_list.append(func)
        
        self.l1 = l1
        self.l2 = l2
        self.gamma = gaunt(l3)[0:(2*l3+1)**2, l1**2:(l1+1)**2, l2**2:(l2+1)**2]
        self.Alpha_list = Aalpha_list
    
    def calc(self, Rvec):
        rshape0 = Rvec.shape[:-1]
        Rvec = Rvec.reshape(-1, 3)
        Alpha = np.empty((Rvec.shape[0], (2*self.lmax+1)**2))
        pos = 0
        for l in range(0, 2*self.lmax+1):
            Alpha[:, pos:pos+2*l+1] = self.Alpha_list[l].generate3D(Rvec)
            pos += 2 * l + 1
        Alpha2 = np.sum(self.gamma[None, :, :, :] * 
                          Alpha[:, :, None, None], axis=1)
        
        return Alpha2.reshape(rshape0 + (2*self.l1+1, 2*self.l2+1))

def read_upf(filename):
    """
    Read a QE pseudopotential file in the upf format.

    Let nproj be the number of projector functions, and nproj_full be the sum of 2*l+1 of each projector:

    Returns:
        funch_full array(nproj_full, nproj_full): A 2D array representing the funch matrix.
        projR_list (list): A list of GridFunc objects representing the projector functions.
    """
    
    root = ET.parse(filename).getroot()

    header_elem = root.find('PP_HEADER')
    nproj = int(header_elem.attrib['number_of_proj'])

    r_elem = root.find('PP_MESH').find('PP_R')
    gridsize = int(r_elem.attrib['size'])
    rgridfunc = np.fromiter(map(float, r_elem.text.split()), float, count=gridsize)
    rgrid = LinearRGD.from_explicit_grid(rgridfunc)

    nloc_elem = root.find('PP_NONLOCAL')

    funch_elem = nloc_elem.find('PP_DIJ')
    funch = np.fromiter(map(float, funch_elem.text.split()), float, count=nproj**2).reshape((nproj, nproj)) / 2. # Ry to Har
    
    projR_list = []
    l_list = []
    for iproj in range(nproj):
        projelem = nloc_elem.find(f'PP_BETA.{iproj+1}')
        l = int(projelem.attrib['angular_momentum'])
        rcut = float(projelem.attrib['cutoff_radius'])
        assert int(projelem.attrib['size']) == gridsize
        projfunc = np.fromiter(map(float, projelem.text.split()), float, count=gridsize)
        projfunc[1:] /= rgrid.rfunc[1:] # function in upf is stored as R(r)*r
        projfunc[0] = projfunc[1] if l==0 else 0.
        l_list.append(l)
        projR_list.append(GridFunc(rgrid, projfunc, l=l, rcut=rcut))
    
    funch_full = []
    for iorb in range(len(l_list)):
        funch_row = []
        l1 = l_list[iorb]
        for jorb in range(len(l_list)):
            l2 = l_list[jorb]
            if l1 == l2:
                funch_row.append(np.identity(2*l1+1) * funch[iorb, jorb])
            else:
                assert np.abs(funch[iorb, jorb]) < 1e-8
                funch_row.append(np.zeros((2*l1+1, 2*l2+1)))
        funch_full.append(funch_row)
    funch_full = np.block(funch_full)
    
    return funch_full, projR_list
