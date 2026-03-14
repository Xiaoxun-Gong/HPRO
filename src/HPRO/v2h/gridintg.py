import numpy as np
from scipy.spatial import KDTree
from scipy.sparse import csr_matrix

from .. import config as CFG
from ..utils.misc import index_traverse, mytqdm, Timer
from ..utils.mpi import is_master, comm, distrib_vec, mpi_sum_csr
from ..matao.mataocsr import MatAOCSR

t1 = Timer('OrbProd')
t2 = Timer('OrbPairToUC')
t3 = Timer('CSRAdd')

class GridIntgWorker:
    def __init__(self, structure, gridsize, aodata, rmax=None):
        '''
        Prepare AO functions on grid points.

        Coarse grid point loop:
            Atomic species loop:
                Radial function loop: irad
                    1. Find all functions within r+dr of the center of the coarse grid using KDTree (iatsc)
                       dr is defined below
                    2. Calculate orbital function value at all fine grid points within this coarse grid volume
                
        Returns:
            iosc_ongrid(ndata), fosc_ongrid(ncoords, ndata), igdptr(ndata):
            (ncoords is the number of fine grid points inside a coarse grid volume)
            iosc_ongrid[igdptr[igdco]:igdptr[igdco+1]] are indices of atomic orbitals in supercell that have 
                                                       been computed on coarse grid point igdco
            fosc_ongrid[:, igdptr[igdco]:igdptr[igdco+1]] are values of atomic orbitals. The first dimension 
                                                          corresponds to different fine grid points within 
                                                          this coarse grid volume.
        '''

        gridsize = np.array(gridsize)
        assert structure == aodata.structure

        # Determine nsubdiv (ratio of coarse grid size to fine grid size):
        # find the smallest number of fine grid points enclosed by fine grid when i is between 13 to 19
        npoints = []
        itrials = list(range(*CFG.GRIDINTG_NSUBDIV_RANGE))
        for i in itrials:
            gridsizeco = (gridsize - 1) // i + 1
            ngridco = np.prod(gridsizeco)
            npoints.append(ngridco * i**3)
        self.nsubdiv = itrials[np.argmin(npoints)]
        # print('nsubdiv:', self.nsubdiv)

        self.structure = structure
        self.gridsizefi = gridsize
        self.gridsizeco = (gridsize - 1) // self.nsubdiv + 1
        assert np.min(self.gridsizeco) > 0
        self.aodata = aodata

        # Preparations of the coarse integration grid
        rprimgridfi = structure.rprim / self.gridsizefi[:, None]
        rprimgridco = rprimgridfi * self.nsubdiv
        offsets = index_traverse(np.arange(0, self.nsubdiv, 1),
                                 np.arange(0, self.nsubdiv, 1),
                                 np.arange(0, self.nsubdiv, 1)) @ rprimgridfi
        ncoords = self.nsubdiv ** 3
        # dr is 1/2 of the maximum of the distances between any pair of vertices of the parallelipiped
        rvertices = index_traverse([0, 1], [0, 1], [0, 1]) @ (rprimgridco - rprimgridfi)
        dr = np.max(np.linalg.norm(rvertices[:, None, :] - rvertices[None, :, :], axis=2)) / 2
        dxyz = np.full(3, (self.nsubdiv-1) / 2) @ rprimgridfi 
        if rmax is None: rmax = max(aodata.cutoffs.values())
        rmax = 2 * (rmax + dr)
        self.mataocsr = MatAOCSR(aodata, maxr=rmax)

        # Create KDTree for each atomic species
        trees_spc, mapiat_spc = [], []
        supercell = self.mataocsr.orbinfo2.structure
        for ispc in range(supercell.nspc):
            spc = structure.atomic_species[ispc]
            is_thisspc = (supercell.atomic_numbers == spc)
            trees_spc.append(KDTree(supercell.atomic_positions_cart[is_thisspc]))
            mapiat_spc.append(np.where(is_thisspc)[0])
        
        # MPI distribution of coarse grid
        ntotcogrid = np.prod(self.gridsizeco)
        rank, count, displ = distrib_vec(ntotcogrid, displ_last_elem=True)
        if is_master():
            print(f"Each MPI process has {np.min(count)} to {np.max(count)} coarse grid points\n")
            if ntotcogrid < comm.size:
                msg = f'Note: number of coarse grid points ({ntotcogrid}) is smaller than the number of MPI processes ({comm.size}), some processors will be idle.\n'
                print(msg)
        
        igd, igdloc = 0, 0
        igdptr = [0]
        iosc_ongrid = []
        fosc_ongrid = []
        if is_master() and CFG.GRIDINTG_ENABLE_TQDM: print('Preparing AO basis functions on grid')
        t = mytqdm(total=count[rank], disable=not CFG.GRIDINTG_ENABLE_TQDM)
        for ia_co in range(self.gridsizeco[0]):
            for ib_co in range(self.gridsizeco[1]):
                for ic_co in range(self.gridsizeco[2]):
                    if (displ[rank]<=igd) and (igd<displ[rank+1]): # this point is on this rank
                        # find the center of coarse grid volume
                        corner = np.array([ia_co, ib_co, ic_co]) @ rprimgridco
                        center = corner + dxyz
                        # find the points inside the coarse grid volume
                        ptcoords = corner[None, :] + offsets # (ncoords, 3)

                        nphi_thispoint = 0
                        for ispc in range(supercell.nspc):
                            spc = structure.atomic_species[ispc]
                            for irad in range(aodata.nradial_spc[spc]):
                                phirgrid = aodata.phirgrids_spc[spc][irad]

                                # Treat AOs within r+dr of the center of the coarse grid to be "nonzero"
                                # And compute their values in the coarse grid
                                r = phirgrid.rcut + dr
                                iat_tree = np.array(trees_spc[ispc].query_ball_point(center, r))
                                if len(iat_tree) == 0: continue
                                iatsc = mapiat_spc[ispc][iat_tree]

                                # Compute orbital values
                                rdiff = ptcoords[:, None, :] - supercell.atomic_positions_cart[None, iatsc, :] # (ncoords, nat, 3)
                                phi = phirgrid.getval3D(rdiff).reshape(ncoords, len(iatsc)*(2*phirgrid.l+1)) # (ncoords, nat*(2l+1))

                                # Prepare array of (iat_full, irad_full, m_full) that have the same length as phi
                                # Find the indices of the "nonzero" orbitals in the supercell
                                tmp = index_traverse(iatsc, np.arange(-phirgrid.l, phirgrid.l+1, 1))
                                iat_full, m_full = tmp[:, 0], tmp[:, 1]
                                assert len(iat_full) == phi.shape[1]
                                irad_full = np.full(phi.shape[1], irad)
                                iorbsc = self.mataocsr.orbinfo2.find_orbindx3(iat_full, irad_full, m_full)

                                iosc_ongrid.append(iorbsc)
                                fosc_ongrid.append(phi)
                                nphi_thispoint += len(iorbsc)

                        igdptr.append(igdptr[-1] + nphi_thispoint)
                        igdloc += 1
                        t.update()

                    igd += 1
        t.close()
        
        self.igdptr = igdptr
        self.iosc_ongrid = np.concatenate(iosc_ongrid) if iosc_ongrid else np.empty(0)
        self.fosc_ongrid = np.concatenate(fosc_ongrid, axis=1) if iosc_ongrid else np.empty((ncoords, 0)) # todo: ascontiguousarray
        # print(self.fosc_ongrid.shape)

    def gridintg(self, func):
        '''
        Carry out integration on real space grid.

        Coarse grid point loop: igd
            1. Calculate phi1*func*phi2 for all fine grid points and all AO pairs in this coarse grid volume
            2. For all orbital pairs (iorb1, iorb2) created in the last step, find the equivalent orbital pair 
               (iorb1p, iorb2p) such that iorb1p is in the unit cell
            3. Add all values to CSR matrix
        '''

        assert func.shape == tuple(self.gridsizefi)
        self.mataocsr.reset_mat()

        dvol = self.structure.cell_volume / np.prod(self.gridsizefi)

        # MPI distribution of coarse grid
        rank, count, displ = distrib_vec(np.prod(self.gridsizeco), displ_last_elem=True)

        igd, igdloc = 0, 0
        if is_master() and CFG.GRIDINTG_ENABLE_TQDM: print('Integrating on real space grid')
        t = mytqdm(total=count[rank], disable=not CFG.GRIDINTG_ENABLE_TQDM)
        for ia_co in range(self.gridsizeco[0]):
            for ib_co in range(self.gridsizeco[1]):
                for ic_co in range(self.gridsizeco[2]):

                    if (displ[rank]<=igd) and (igd<displ[rank+1]): # this point is on this rank

                        sliceorb_thispoint = slice(self.igdptr[igdloc], self.igdptr[igdloc+1])
                        if sliceorb_thispoint.stop > sliceorb_thispoint.start: 
                            iosc_thispoint = self.iosc_ongrid[sliceorb_thispoint]
                            fosc_thispoint = self.fosc_ongrid[:, sliceorb_thispoint]

                            # get func values on this point
                            # need to deal with the case where a coarse volume lies on the edge of the unit cell
                            slicea = slice(ia_co*self.nsubdiv, min((ia_co+1)*self.nsubdiv, self.gridsizefi[0]))
                            sliceb = slice(ib_co*self.nsubdiv, min((ib_co+1)*self.nsubdiv, self.gridsizefi[1]))
                            slicec = slice(ic_co*self.nsubdiv, min((ic_co+1)*self.nsubdiv, self.gridsizefi[2]))
                            lena = slicea.stop - slicea.start; assert lena > 0
                            lenb = sliceb.stop - sliceb.start; assert lenb > 0
                            lenc = slicec.stop - slicec.start; assert lenc > 0
                            f = np.zeros((self.nsubdiv, self.nsubdiv, self.nsubdiv), dtype='f8')
                            f[0:lena, 0:lenb, 0:lenc] = func[slicea, sliceb, slicec]
                            f = f.reshape(-1)

                            # Integration on this point: intg[i,j] = sum_k phi1.T[i, k] * f[k] * phi2[k, j] * dvol
                            # Note: no conjugate on phi1 because phi is real
                            t1.start()
                            if CFG.GRIDINTG_USE_HERMITICITY:
                                io, iop = np.tril_indices(len(iosc_thispoint)) # lower triangle indices
                                intg = np.dot(f*dvol, fosc_thispoint[:, io] * fosc_thispoint[:, iop])
                            else:
                                intg = np.dot(fosc_thispoint.T, (f*dvol)[:, None]*fosc_thispoint)
                                intg = intg.reshape(-1)

                            # Get iorb1 and iorb2: pairs of connected atomic orbital basis by this point
                            if CFG.GRIDINTG_USE_HERMITICITY:
                                iorb1 = iosc_thispoint[io]; iorb2 = iosc_thispoint[iop]
                            else:
                                pairs = index_traverse(iosc_thispoint, iosc_thispoint)
                                iorb1, iorb2 = pairs[:, 0], pairs[:, 1]
                            t1.stop()
                            t2.start()
                            # Translate (iorb1, iorb2) to its equivalent (iorb1p, iorb2p)
                            # where both iorb1 and iorb2 are in the supercell
                            # but iorb1p is in the unit cell, iorb2p is in the unit cell
                            iorb1p, iorb2p = self.mataocsr.orbinfo2.orbpair_translate_to_uc(iorb1, iorb2)
                            t2.stop()

                            t3.start()
                            self.mataocsr.mat += csr_matrix((intg, (iorb1p, iorb2p)), shape=self.mataocsr.mat.shape)
                            t3.stop()

                        igdloc += 1
                        t.update()
                    igd += 1
        t.close()

        self.mataocsr.mat.eliminate_zeros()
        t3.start()
        self.mataocsr.mat = mpi_sum_csr(self.mataocsr.mat)
        t3.stop()
        if is_master() and CFG.GRIDINTG_USE_HERMITICITY:
            self.mataocsr.unfold_with_hermiticity()
        if not is_master():
            self.mataocsr.reset_mat()

        return self.mataocsr
        
