from itertools import chain

import numpy as np

from ..utils.mpi import distrib_vec, MPI, comm
from ..utils.misc import mytqdm
from ..matao.matao import MatAO

# from time import time
# class timer:
#     def __init__(self):
#         self.start_time = 0.
#         self.total_time = 0.
#     def start(self):
#         self.start_time = time()
#     def stop(self):
#         self.total_time += time() - self.start_time
# t = timer()

# Usage:
# FFTgrid = np.array(vlocr.shape)
# Hloc = gridintg_two_center(self.structure, basis, FFTgrid, olp_basis.get_pairs_ij(), vlocr)

def gridintg_two_center(structure, aodata, FFTgrid, pairs_ij, func):
    raise DeprecationWarning()
    '''
    Integrate the function (func) between two atomic orbitals on a real space grid:
    \int d^3r \phi_i^*(r) f(r) \phi_j(r)
    For each orbital pair, first find the grid points that are simultaneously within the cutoff radius of two orbitals, 
    then carry out direct integration on the points.
    '''
    assert np.isrealobj(func), 'Complex array not implemented'
    rprimFFT = structure.rprim / FFTgrid[:, None]
    dvol = structure.cell_volume / np.prod(FFTgrid)
    
    # get the grids for all orbitals at all sites
    grids_site_orb = []
    for iatom in range(structure.natom):
        spc = structure.atomic_numbers[iatom]
        poscart = structure.atomic_positions_cart[iatom]
        grids_site = []
        for iorb in range(aodata.nradial_spc[spc]):
            rcutorb = aodata.phirgrids_spc[spc][iorb].rcut
            grids_site.append(GridPoints.within_sphere(rprimFFT, rcutorb, poscart))
        grids_site_orb.append(grids_site)
    
    vloc = MatAO.init_mats(pairs_ij, aodata, filling_value=0., dtype='f8')
    vloc.shuffle()

    rank, count, displ = distrib_vec(vloc.npairs, displ_last_elem=True)

    # intialize grids and phi product
    # pairs = aodata.pairs_within_cutoff()
    if comm is not None: comm.Barrier()
    if rank == 0:
        print(f'\nConstructing local part of Hamiltonian operator with {count[rank]} blocks')
    for ipair in mytqdm(range(displ[rank], displ[rank+1])):
        atm1, atm2 = vloc.atom_pairs[ipair]
        spc1, spc2 = structure.atomic_numbers[atm1], structure.atomic_numbers[atm2]
        for iorb in range(aodata.nradial_spc[spc1]):
            phirgrid1 = aodata.phirgrids_spc[spc1][iorb]
            grid1 = grids_site_orb[atm1][iorb]
            for jorb in range(aodata.nradial_spc[spc2]):
                phirgrid2 = aodata.phirgrids_spc[spc2][jorb]
                slice1 = slice(aodata.orbslices_spc[spc1][iorb],
                               aodata.orbslices_spc[spc1][iorb+1])
                slice2 = slice(aodata.orbslices_spc[spc2][jorb],
                               aodata.orbslices_spc[spc2][jorb+1])
                grid2 = grids_site_orb[atm2][jorb].translate(vloc.translations[ipair]*FFTgrid)
                olpgrid = GridPoints.combine(grid1, grid2)
                if olpgrid.is_empty():
                    vloc.mats[ipair][slice1, slice2] = 0.
                    continue
                olpcoords = olpgrid.list_coords()
                assert olpcoords.shape[0]>0
                assert len(olpcoords.shape)==2
                olpcoords_cart = olpcoords @ rprimFFT
                # t.start()
                phi1 = phirgrid1.getval3D_noselect(olpcoords_cart - structure.atomic_positions_cart[atm1])
                phi2 = phirgrid2.getval3D_noselect(olpcoords_cart - structure.atomic_positions_cart[atm2] -
                                                   vloc.translations[ipair] @ structure.rprim)
                # t.stop()
                olpcoords_uc = np.divmod(olpcoords, FFTgrid[None, :])[1]
                x_uc, y_uc, z_uc = olpcoords_uc[:, 0], olpcoords_uc[:, 1], olpcoords_uc[:, 2]
                func_on_grid = func[x_uc, y_uc, z_uc]
                mat = np.sum(func_on_grid[:, None, None] * phi1[:, :, None] * phi2[:, None, :], axis=0)
                vloc.mats[ipair][slice1, slice2] = mat * dvol

    if comm is not None:
        vloc.mpi_gather(displ, dtype=MPI.REAL8, root=0)
    # if comm is not None and comm.rank != 0:
    #     vloc.delete_mats()
    # ! Notice that only root is correct afterwards
    
    vloc.unfold_with_hermiticity()
    
    return vloc

def gridintg_two_center_old(structure, aodata, FFTgrid, pairs_ij, func):
    raise DeprecationWarning()
    assert np.isrealobj(func), 'Complex array not implemented'
    rprimFFT = structure.rprim / FFTgrid[:, None]
    dvol = structure.cell_volume / np.prod(FFTgrid)
    
    # get the grids for all orbitals at all sites
    grids_site_orb = []
    for iatom in range(structure.natom):
        spc = structure.atomic_numbers[iatom]
        poscart = structure.atomic_positions_cart[iatom]
        grids_site = []
        for iorb in range(aodata.nradial_spc[spc]):
            rcutorb = aodata.phirgrids_spc[spc][iorb].rcut
            grids_site.append(GridPoints.within_sphere(rprimFFT, rcutorb, poscart))
        grids_site_orb.append(grids_site)
    
    vloc = MatAO.init_mats(pairs_ij, aodata, filling_value=0., dtype='f8')
    vloc.shuffle()

    rank, count, displ = distrib_vec(vloc.npairs, displ_last_elem=True)

    # intialize grids and phi product
    # pairs = aodata.pairs_within_cutoff()
    olpgrids_atmp_orbp = [] # (atom_pairs, orbital_pairs)
    if comm is not None: comm.Barrier()
    if rank == 0:
        print(f'\nConstructing local part of Hamiltonian operator with {count[rank]} blocks')
    for ipair in mytqdm(range(displ[rank], displ[rank+1])):
        atm1, atm2 = vloc.atom_pairs[ipair]
        spc1, spc2 = structure.atomic_numbers[atm1], structure.atomic_numbers[atm2]
        olpgrids_orbp = []
        for iorb in range(aodata.nradial_spc[spc1]):
            phirgrid1 = aodata.phirgrids_spc[spc1][iorb]
            grid1 = grids_site_orb[atm1][iorb]
            for jorb in range(aodata.nradial_spc[spc2]):
                phirgrid2 = aodata.phirgrids_spc[spc2][jorb]
                grid2 = grids_site_orb[atm2][jorb].translate(vloc.translations[ipair]*FFTgrid)
                olpgrid = GridPoints.combine(grid1, grid2)
                if olpgrid.is_empty():
                    olpgrids_orbp.append(None)
                    continue
                olpcoords = olpgrid.list_coords()
                assert olpcoords.shape[0]>0
                assert len(olpcoords.shape)==2
                olpcoords_cart = olpcoords @ rprimFFT
                # t.start()
                phi1 = phirgrid1.getval3D_noselect(olpcoords_cart - structure.atomic_positions_cart[atm1])
                phi2 = phirgrid2.getval3D_noselect(olpcoords_cart - structure.atomic_positions_cart[atm2] -
                                                   vloc.translations[ipair] @ structure.rprim)
                # t.stop()
                phiprod = phi1[:, :, None] * phi2[:, None, :] # (ngrid, 2l1+1, 2l2+1)
                olpcoords_uc = np.divmod(olpcoords, FFTgrid[None, :])[1]
                olpgrids_orbp.append(OlpGrids(olpcoords_uc, phiprod))
        olpgrids_atmp_orbp.append(olpgrids_orbp)
    if comm is not None: comm.Barrier()
    # print(t.total_time)
    
    # calculate integrals
    for ipair in range(displ[rank], displ[rank+1]):
        atm1, atm2 = vloc.atom_pairs[ipair]
        spc1, spc2 = structure.atomic_numbers[atm1], structure.atomic_numbers[atm2]
        iorbp = 0
        for iorb in range(aodata.nradial_spc[spc1]):
            for jorb in range(aodata.nradial_spc[spc2]):
                slice1 = slice(aodata.orbslices_spc[spc1][iorb],
                               aodata.orbslices_spc[spc1][iorb+1])
                slice2 = slice(aodata.orbslices_spc[spc2][jorb],
                               aodata.orbslices_spc[spc2][jorb+1])
                olpgrids = olpgrids_atmp_orbp[ipair-displ[rank]][iorbp]
                if olpgrids is None:
                    vloc.mats[ipair][slice1, slice2] = 0.
                else:
                    vloc.mats[ipair][slice1, slice2] = olpgrids.integrate(func) * dvol
                iorbp += 1

    if comm is not None:
        vloc.mpi_gather(displ, dtype=MPI.REAL8, root=0)
    # if comm is not None and comm.rank != 0:
    #     vloc.delete_mats()
    # ! Notice that only root is correct afterwards
    
    vloc.unfold_with_hermiticity()
    
    return vloc
    
                
class OlpGrids:
    def __init__(self, olpcoords_uc, phiprod):
        self.olpcords_uc = olpcoords_uc
        self.phiprod = phiprod
        
    def integrate(self, realspacefunc):
        x_uc, y_uc, z_uc = self.olpcords_uc[:, 0], self.olpcords_uc[:, 1], self.olpcords_uc[:, 2]
        func_on_grid = realspacefunc[x_uc, y_uc, z_uc]
        return np.sum(func_on_grid[:, None, None] * self.phiprod[:, :, :], axis=0)
        

class GridPoints:
    '''
    This class deals with grid points in 3D space. 
    There is an x limit, and for each x, there is a y limit and for each y, there is a z limit.
    Note that here x, y and z are lattice directions of the grid, not cartesian directions.
    '''
    def __init__(self, xlim, ylim_x, zlim_x_y):
        '''
        Attributes:
        xlim:      tuple(int, int)
                   The (x_min, x_max+1) of x coordinates
        ylim_x:    tuple(int(lenx), int(lenx)), lenx=x_max-x_min+1
                   The (y_min, y_max+1) of y coordinates at each x coordinate
        zlim_x_y:  tuple(list(int(leny)), list(int(leny)))
                   leny=y_max-y_min+1, length of list = lenx
                   The (z_min, z_max+1) of z coordinates at each x and y coordinate
        nx:        int
        ny_x:      int(lenx)
        nz_x_y:    list(int(leny)), length of list = lenx
        '''
        
        nx = xlim[1] - xlim[0]
        ny_x = np.empty(nx, dtype=int)
        nz_x_y = []
        for ix in range(nx):
            nz_y = zlim_x_y[1][ix] - zlim_x_y[0][ix]
            ny_x[ix] = np.sum(nz_y)
            nz_x_y.append(nz_y)
        
        self.xlim = xlim
        self.ylim_x = ylim_x
        self.zlim_x_y = zlim_x_y
        
        self.nx = nx
        self.ny_x = ny_x
        self.nz_x_y = nz_x_y
    
    def is_empty(self):
        xlim, ylim_x, zlim_x_y = self.xlim, self.ylim_x, self.zlim_x_y
        if xlim[0] >= xlim[1]: return True
        if np.all(ylim_x[0] >= ylim_x[1]): return True
        for ix in range(self.nx):
            if np.any(zlim_x_y[0][ix] < zlim_x_y[1][ix]): return False
        return True    
    
    def list_coords(self):
        '''
        Get the list of grid coordinates (integers) within the range specified by this object.
        '''
        xlim, ylim_x, zlim_x_y = self.xlim, self.ylim_x, self.zlim_x_y
        gx = np.arange(xlim[0], xlim[1], 1)
        gy, gz = [], []
        for ix in range(self.nx):
            gy.append(np.arange(ylim_x[0][ix], ylim_x[1][ix], 1))
            zmin_y, zmax_y = zlim_x_y[0][ix], zlim_x_y[1][ix]
            ny = len(zmin_y)
            for iy in range(ny):
                zmin, zmax1 = zmin_y[iy], zmax_y[iy]
                gz.append(range(zmin, zmax1))
        xrep = self.ny_x
        gx = np.repeat(gx, xrep)
        if len(gy) > 0:
            gy = np.concatenate(gy)
            yrep = np.concatenate(self.nz_x_y)
            gy = np.repeat(gy, yrep)
        else:
            gy = np.array([], dtype=int)
        gz = np.fromiter(chain.from_iterable(gz), dtype=int, count=np.sum(xrep))
        return np.stack((gx, gy, gz), axis=1)
    
    def translate(self, dxyz):
        '''
        Translate the range of grid points by dxyz
        '''
        dx, dy, dz = dxyz
        xlim = (self.xlim[0] + dx, self.xlim[1] + dx)
        ylim_x = (self.ylim_x[0] + dy, self.ylim_x[1] + dy)
        zlim_x_y = ([], [])
        for ix in range(self.nx):
            zlim_x_y[0].append(self.zlim_x_y[0][ix] + dz)
            zlim_x_y[1].append(self.zlim_x_y[1][ix] + dz)
        return self.__class__(xlim, ylim_x, zlim_x_y)

    @classmethod
    def within_sphere(cls, rprim, rcut, offsetcart=(0., 0., 0.)):
        '''
        Find grid points within sphere of radius rcut.
        The center of the sphere is at offset.
        
        Parameters:
        ---------
        rprim(abc, xyz): primary lattice vectors, arbitrary unit
        rcut: cutoff radius
        offsetcart: offset of the origin, in cartesian coordinate
        '''
        
        gprim = np.linalg.inv(rprim.T)
        thickx = 1. / np.linalg.norm(gprim[0])
        offsetx = np.dot(offsetcart, gprim[0]) # offset[0] 
        
        # project to yz plane and decompose
        rprimyz = np.stack([gprim[0], rprim[1], rprim[2]])
        gprimyz = np.linalg.inv(rprimyz.T)
        # offsetcart = offset @ rprim
        thicky = 1. / np.linalg.norm(gprimyz[1])
        offsety = np.dot(offsetcart, gprimyz[1])
        doffsety = np.dot(rprim[0], gprimyz[1])
        
        # project to z vector
        thickz = np.linalg.norm(rprim[2])
        offsetz = np.dot(offsetcart, rprim[2])  / thickz**2
        doffsetz = np.dot(rprim[0], rprim[2]) / thickz**2
        ddoffsetz = np.dot(rprim[1], rprim[2]) / thickz**2 # y dot z / |z|^2
        
        # x range
        nx = rcut / thickx
        gxmax = np.floor(offsetx + nx).astype(int) + 1 # note: is float
        gxmin = np.ceil(offsetx - nx).astype(int)
        xlim = (gxmin, gxmax)
        
        # y range
        g_x = np.arange(gxmin, gxmax, 1)
        ryz_x = np.sqrt(rcut**2 - ((g_x-offsetx) * thickx)**2)
        offsety_x = offsety - doffsety * g_x
        ny_x = ryz_x / thicky
        gymax_x = np.floor(offsety_x + ny_x).astype(int) + 1
        gymin_x = np.ceil(offsety_x - ny_x).astype(int)
        ylim_x = (gymin_x, gymax_x)
        
        # z range
        gzmax_x_y, gzmin_x_y = [], []
        for ix in range(len(g_x)):
            g_y = np.arange(gymin_x[ix], gymax_x[ix], 1)
            rz_y = np.sqrt(ryz_x[ix]**2 - ((g_y-offsety_x[ix]) * thicky)**2)
            nz_y = rz_y / thickz
            offsetz_y = offsetz - doffsetz * g_x[ix] - ddoffsetz * g_y
            gzmax_y = np.floor(offsetz_y + nz_y).astype(int) + 1
            gzmin_y = np.ceil(offsetz_y - nz_y).astype(int)
            gzmax_x_y.append(gzmax_y)
            gzmin_x_y.append(gzmin_y)
        zlim_x_y = (gzmin_x_y, gzmax_x_y)
        
        return cls(xlim, ylim_x, zlim_x_y)

    @classmethod
    def combine(cls, gdpts1, gdpts2):
        '''
        Find the overlap of two GridPoints objects into a new set of grid points.
        '''
        xlim = (max(gdpts1.xlim[0], gdpts2.xlim[0]),
                min(gdpts1.xlim[1], gdpts2.xlim[1]))
        if xlim[1] <= xlim[0]:
            return GridPoints((0,0), (np.array([], dtype=int), np.array([], dtype=int)), ([], []))
        
        slicex1 = slice(xlim[0] - gdpts1.xlim[0], xlim[1] - gdpts1.xlim[0])
        ylim_x1 = (gdpts1.ylim_x[0][slicex1], gdpts1.ylim_x[1][slicex1])
        zlim_x_y1 = (gdpts1.zlim_x_y[0][slicex1], gdpts1.zlim_x_y[1][slicex1])
        
        slicex2 = slice(xlim[0] - gdpts2.xlim[0], xlim[1] - gdpts2.xlim[0])
        ylim_x2 = (gdpts2.ylim_x[0][slicex2], gdpts2.ylim_x[1][slicex2])
        zlim_x_y2 = (gdpts2.zlim_x_y[0][slicex2], gdpts2.zlim_x_y[1][slicex2])
        
        ylim_x = (np.max((ylim_x1[0], ylim_x2[0]), axis=0),
                np.min((ylim_x1[1], ylim_x2[1]), axis=0))
        
        yinvalid_x = ylim_x[0] > ylim_x[1]
        ylim_x[1][yinvalid_x] = ylim_x[0][yinvalid_x]
        zlim_x_y = ([], [])
        
        for ix in range(xlim[1]-xlim[0]):
            ymin, ymax = ylim_x[0][ix], ylim_x[1][ix]
            ymin1, ymin2 = ylim_x1[0][ix], ylim_x2[0][ix]
            
            slicey1 = slice(ymin - ymin1, ymax - ymin1)
            zlim_y1 = (zlim_x_y1[0][ix][slicey1], zlim_x_y1[1][ix][slicey1])
            slicey2 = slice(ymin - ymin2, ymax - ymin2)
            zlim_y2 = (zlim_x_y2[0][ix][slicey2], zlim_x_y2[1][ix][slicey2])
            
            zmin_y = np.max((zlim_y1[0], zlim_y2[0]), axis=0)
            zmax_y = np.min((zlim_y1[1], zlim_y2[1]), axis=0)
            zinvalid_y = zmin_y > zmax_y
            zmax_y[zinvalid_y] = zmin_y[zinvalid_y]
            zlim_x_y[0].append(zmin_y)
            zlim_x_y[1].append(zmax_y)
        
        return cls(xlim, ylim_x, zlim_x_y)