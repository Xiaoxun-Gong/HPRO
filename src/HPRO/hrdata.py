import numpy as np

from .bgwio import bgw_vsc
from .lcaodata import LCAOData
from .utils import slice_same, tqdm_mpi_tofile
from .matlcao import pairs_to_indices, indices_to_pairs, MatLCAO
from .gridintg import GridPoints

'''
This module implements several functions needed by real-space construction of AO Hamiltonian.
This includes constructing Hamiltonian in real space, and constructing VKB under AO basis.
'''

def read_vloc(filename, interface):
    if interface == 'bgw':
        vscread = bgw_vsc(filename)
        vscread.read_header()
        # if is_master:
        #     print('Reading self-consistent potential from VSC')
        vscread.read_data()
        vscread.close()
        FFTgrid = np.array([vscread.nr1, vscread.nr2, vscread.nr3])
        vscg_full = np.zeros(FFTgrid, dtype='c16')
        _, g_g_full = np.divmod(vscread.g_g, FFTgrid)
        vscg_full[g_g_full[:, 0], g_g_full[:, 1], g_g_full[:, 2]] = vscread.vscg
        vlocr = np.fft.ifftn(vscg_full, s=FFTgrid, norm='forward')
        assert np.max(np.abs(vlocr.imag)) < 1e-6
        vlocr = vlocr.real
    else:
        raise NotImplementedError(f'Unknown vloc interface: {interface}')
    
    return vlocr

def read_hrr(structure, pspdir, funchfile=None, interface='qe'):
    if interface == 'qe':
        assert funchfile is None
        projR = LCAOData(structure, None, basis_path_root=pspdir, aocode='qe-projR')
        funch = []
        for zatm in structure.atomic_numbers:
            funch.append(projR.funch_spc[zatm])
        funcg = None
    else:
        raise NotImplementedError(f'Unknown vnloc interface: {interface}')    

    return funch, funcg, projR

def constructH(item, vlocr, basis, FFTgrid, rprimFFT, votk, grids_site_orb, Hmain):
    '''
    Build Hamiltonian operator in atomic orbital basis according to the formula:
    H_{i\alpha,j\beta} = \langle \phi_{i\alpha} | -\frac{1}{2}\nabla^2 | \phi_{j\beta} \rangle + \int \mathrm{d}^3r\, \phi_{i\alpha}^*(\boldsymbol{r}) V_\text{eff}(\boldsymbol{r}) \phi_{j\beta}(\boldsymbol{r}) + \sum_{a\gamma\delta} \langle \phi_{i\alpha} | p_{a\gamma} \rangle D_{a\gamma\delta} \langle p_{a\delta} | \phi_{j\beta} \rangle
    '''
    print(f'\nConstructing Hamiltonian operator with {Hmain.npairs} blocks')
    for ipair in tqdm_mpi_tofile(range(Hmain.npairs)):
        atm1, atm2 = Hmain.atom_pairs[ipair]
        spc1, spc2 = item.structure.atomic_numbers[atm1], item.structure.atomic_numbers[atm2]
        for iorb in range(basis.norb_spc[spc1]):
            phirgrid1 = basis.phirgrids_spc[spc1][iorb]
            grid1 = grids_site_orb[atm1][iorb]
            for jorb in range(basis.norb_spc[spc2]):
                phirgrid2 = basis.phirgrids_spc[spc2][jorb]
                slice1 = slice(basis.orbslices_spc[spc1][iorb],
                            basis.orbslices_spc[spc1][iorb+1])
                slice2 = slice(basis.orbslices_spc[spc2][jorb],
                            basis.orbslices_spc[spc2][jorb+1])
                grid2 = grids_site_orb[atm2][jorb].translate(Hmain.translations[ipair]*FFTgrid)
                plsgrid = GridPoints.pls(grid1, grid2)
                if plsgrid.null():
                    Hmain.mats[ipair][slice1, slice2] = 0.
                    continue
                plslcd = plsgrid.lcd()
                assert plslcd.shape[0]>0
                assert len(plslcd.shape)==2
                plscrt = plslcd @ rprimFFT
                # t.start()
                phi1 = phirgrid1.generate3D_noselect(plscrt - item.structure.atomic_positions_cart[atm1])
                phi2 = phirgrid2.generate3D_noselect(plscrt - item.structure.atomic_positions_cart[atm2] -
                                                Hmain.translations[ipair] @ item.structure.rprim)
                # t.stop()
                plslcd_uc = np.divmod(plslcd, FFTgrid[None, :])[1]
                x_uc, y_uc, z_uc = plslcd_uc[:, 0], plslcd_uc[:, 1], plslcd_uc[:, 2]
                f2 = vlocr[x_uc, y_uc, z_uc]
                mat = np.sum(f2[:, None, None] * phi1[:, :, None] * phi2[:, None, :], axis=0)
                Hmain.mats[ipair][slice1, slice2] = mat * votk

def calc_vkb(olp_proj_ao, Dij=None):
    '''
    Construct VKB in atomic orbital basis according to the formula:
    <ia| Vkb |i'a'> = \sum_{jbb'} <ia|jb> D_{jbb'} <jb'|i'a'>
    where i, i', j are atom indices, a, a', b, b' are orbital indices

    Matrix D is optional. If not given, output will be all zero.
    '''
    if Dij is not None:
        for iD in range(len(Dij)):
            D = Dij[iD]
            if not np.isrealobj(D):
                # Future: D is complex
                assert np.max(np.abs(D.imag)) < 1e-8
                Dij[iD] = D.real
    olp_proj_ao.sort_atom1()
    translations = olp_proj_ao.translations
    atom_pairs = olp_proj_ao.atom_pairs
    trans1, atms2, mats3 = [], [], []
    
    # do matrix multiplications
    slice_jatm = slice_same(atom_pairs[:, 0])
    njatm = len(slice_jatm) - 1
    for ix_atm in range(njatm):
        startj = slice_jatm[ix_atm]
        endj = slice_jatm[ix_atm + 1]
        atomj = atom_pairs[startj, 0]
        ix_js, ix_jps = np.tril_indices(endj - startj) # lower-triangle indices
        ix_js += startj; ix_jps += startj
        # ix_js, ix_jps = np.meshgrid(np.arange(startj, endj, 1), np.arange(startj, endj, 1), indexing='ij')
        # ix_js = ix_js.reshape(-1); ix_jps = ix_jps.reshape(-1)
        trans1.append(translations[ix_jps] - translations[ix_js])
        atms2.append(np.stack((atom_pairs[ix_js, 1], atom_pairs[ix_jps, 1]), axis=1))
        for ix_j, ix_jp in zip(ix_js, ix_jps):
            mat = olp_proj_ao.mats[ix_j]
            matp = olp_proj_ao.mats[ix_jp]
            if Dij is not None:
                D = Dij[atomj]
                mats3.append(mat @ D @ matp.T)
            else:
                # mats_new.append(mat.T @ matp)
                mats3.append(np.zeros((mat.shape[1], matp.shape[1])))
    trans1 = np.concatenate(trans1, axis=0)
    atms2 = np.concatenate(atms2, axis=0)
    
    # collect terms with the same translations and atom pairs and sum them up
    xds1 = pairs_to_indices(olp_proj_ao.structure, trans1, atms2)
    slice3 = slice_same(xds1)
    y = len(slice3) - 1
    mats2 = []
    for ipair in range(y):
        slice2 = slice(slice3[ipair], slice3[ipair+1])
        mats2.append(np.sum([mats3[i] for i in slice2], axis=0))
    indx1 = np.unique(xds1)
    trans2, atm1 = indices_to_pairs(olp_proj_ao.structure.natom, indx1)
    
    vkb = MatLCAO(olp_proj_ao.structure, trans2, atm1, mats2, olp_proj_ao.lcaodata2)
    
    return vkb