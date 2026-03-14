import numpy as np
from scipy.integrate import simpson
from scipy.interpolate import CubicSpline

from .math import r_to_xyz, spbessel_transfrorm, spharm_xyz

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
        self.spline = None
        self.l = l
        
        # get rcut
        if rcut is None and func is not None:
            # automatically detect rcut
            # deeph case, there is no rgd and func
            assert len(func) == rgd.npoints
            norm = rgd.integrate(func**2)
            nonzero, = np.where(np.abs(func)[1:] * np.diff(rgd.rfunc) * rgd.rfunc[1:]**2 >= norm * 1e-4)
            if len(nonzero) == 0 or nonzero[-1]>=len(func)-1:
                last_nonzero = len(func) - 1
            else:
                last_nonzero = nonzero[-1] + 1
            rcut = self.rgd.rfunc[last_nonzero]
        self.rcut = rcut
    
    def calc_spline(self):
        '''
        Calculate and return a spline for the radial function.
        '''
        self.spline = self.rgd.spline(self.func)
        return self.spline

    def getval(self, r):
        '''
        Calculate the value of the radial function at a given radius through spline interpolation.
        '''
        if self.spline is None:
            self.calc_spline()
        assert np.max(r) <= self.rgd.rend
        assert np.min(r) >= self.rgd.rstart
        return self.rgd.getval(self.spline, r)

    def getvalxyz(self, Rnorm, x, y, z):
        Ylm = spharm_xyz(self.l, x, y, z)
        Rlm = self.getval(Rnorm)
        return Rlm[:, None] * Ylm[:, :]

    def getval3D(self, Rvec):
        '''
        Calculate the value of the function at a given set of cartesian coordinates.

        Parameters:
            Rvec array(npoints, 3): A set of Cartesian coordinates.

        Returns:
            array(npoints, 2l+1): The value of the function at the given coordinates.
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
            R_lm[Rwithin, :] = self.getvalxyz(Rnorm, x, y, z)
        # deal with points smaller than self.rgd.rstart
        Rsmall = Rnorm < self.rgd.rstart
        nsmall = np.sum(Rsmall)
        if nsmall > 0:
            R_lm[Rsmall, :] = 0. if self.l > 0 else self.func[0]
        return R_lm.reshape(rshape0 + (2*self.l+1,))

    def getval3D_noselect(self, Rvec):
        '''
        Same as `getval3D` but assume all points are already within the range of the radial grid. 
        This can be more efficient than `getval3D` but use with care.
        '''
        rshape0 = Rvec.shape[:-1]
        Rvec = Rvec.reshape(-1, 3)
        Rnorm, x, y, z = r_to_xyz(Rvec)
        R_lm = self.getvalxyz(Rnorm, x, y, z)
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
    
    def integrate(self, func_on_grid, n=2):
        assert len(func_on_grid) == self.npoints
        return simpson(func_on_grid * self.rfunc**n, self.rfunc)

    def spline(self, func_on_grid):
        return CubicSpline(self.rfunc, func_on_grid)
    
    def getval(self, spline, r):
        return spline(r)
    
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
    
    def integrate(self, func_on_grid, n=2):
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
    
    def integrate(self, func_on_grid, n=2):
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
    
    def integrate(self, func_on_grid, n=2):
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
    
    def spline(self, func_on_grid):
        return CubicSpline(np.arange(0, self.npoints, dtype=float), func_on_grid)
    
    def getval(self, spline, r):
        assert np.all(r>self.rstart)
        i_fromr = np.log(r/self.a) / self.d 
        return spline(i_fromr)

# grid FTs

def grid_overlap(gridfunc1, gridfunc2):
    '''
    Overlap of two functions centered at the same point
    '''
    assert gridfunc1.rgd == gridfunc2.rgd
    rgd = gridfunc1.rgd
    return rgd.integrate(gridfunc1.func * gridfunc2.func)

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
    
