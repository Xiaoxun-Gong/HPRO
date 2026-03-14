import numpy as np

from ..constants import hartree2ev
from ..utils.math import kgrid_with_tr, make_kkmap, kGsphere
from ..utils.misc import set_range, mytqdm, VKBGData, atom_number2name
from ..utils.mpi import is_master
from ..io.aodata import AOData, calc_FT_kg_orb_spcs
from ..io.bgwio import bgw_vsc, bgw_vkb
from ..io.gpawio import gpaw_psp
from ..io.struio import from_bgw, from_poscar, from_gpaw

'''
This module implements classes used to build and store the full Hamiltonian in PW basis.
'''

class HGData:
    '''
    Attributes:
    ---------
    ng_g: int, number of charge-density g-vectors
    g_g(ng_g, 3): miller indices of charge-density g-vectors
    FFTgrid(3): dimensions of FFT grid
    vsc(ng_g): complex, local part of Hamiltonian, in Hartree
    nh_spcs: int->int, number of KB projectors of each atomic species
    deeq: int->(ns, nh, nh): d_{hh'as}
    nk: int
    vkbgdatas: List[VKBGdata], len=nk
    '''
        
    def __init__(self, vscfile=None, vkbfile=None, structure=None, kgrid=None, interface='bgw'):
        
        if interface == 'bgw':
            assert vscfile is not None
            assert vkbfile is not None
            
            # check structures are consistent
            stru_vsc = from_bgw(vscfile)
            stru_vkb = from_bgw(vkbfile)
            eps = 1e-6
            issame = np.all(np.abs(stru_vsc.rprim-stru_vkb.rprim) < eps)
            issame *= np.all(stru_vsc.atomic_species==stru_vkb.atomic_species)
            assert issame
            if structure is None:
                structure = stru_vsc
            else:
                assert structure == stru_vsc
            
            vscread = bgw_vsc(vscfile)
            vkbread = bgw_vkb(vkbfile)
            
            vscread.read_header()
            vkbread.read_header()
            if is_master():
                print('Reading self-consistent potential from VSC')
            vscread.read_data()
            if is_master():
                print('Reading KB projectors from VKB\n')
            vkbread.read_data()
            
            FFTgrid = np.array([vscread.nr1, vscread.nr2, vscread.nr3])
            index_vec = gvec_index(vscread.ng_g, vscread.g_g, FFTgrid)
            
            nh_spcs = {}
            for iat in range(vkbread.nat):
                spc = vkbread.atomic_number[iat]
                if spc not in nh_spcs:
                    nh_spcs[spc] = vkbread.nh[vkbread.ityp[iat]-1]
            for spc in nh_spcs.keys():
                assert spc in structure.atomic_species
            for spc in structure.atomic_species:
                assert spc in nh_spcs
                
            nk = vkbread.nk
            kpts = vkbread.xk
            kptwts = vkbread.wk / np.sum(vkbread.wk)
            
            self.ng_g = vscread.ng_g
            self.g_g = vscread.g_g
            self.FFTgrid = FFTgrid
            self.index_vec = index_vec
            self.vscg = vscread.vscg
            
            self.nh_spcs = nh_spcs
            self.deeq = vkbread.deeq
            self.vkbgdatas = vkbread.vkbgdatas
            
            self.structure = structure
            
            self.nk = nk
            self.kpts = kpts
            self.kptwts = kptwts
            self.kpts_cart = 2 * np.pi * self.kpts @ self.structure.gprim
            self.ecutwfn = vkbread.ecutwfc / 2.0
            
            if kgrid is not None:
                self.reduce_kgrid(kgrid)
            
            vscread.close()
            vkbread.close()
        
        else:
            raise NotImplementedError(interface)
        
        self.interface = interface
        self.ispaw = False
        
    def build_hamblock(self, ik, gvecrange1=(None,None), gvecrange2=(None,None)):
        '''Build plane-wave Hamiltonian block in Hartree'''
        '''This method requires deeq to be same for all atoms of the same species,
           thus does not work for PAW PP.'''
        assert self.interface == 'bgw'
        
        gvecrange1 = set_range(gvecrange1, 0, self.vkbgdatas[ik].ng)
        gvecrange2 = set_range(gvecrange2, 0, self.vkbgdatas[ik].ng)
        
        hamblock = np.zeros((gvecrange1[1]-gvecrange1[0], gvecrange2[1]-gvecrange2[0]), dtype='c16')
        
        # kinetic part
        idiagmin = max(gvecrange1[0], gvecrange2[0])
        idiagmax = min(gvecrange1[1], gvecrange2[1])
        kgcart_diag = self.vkbgdatas[ik].kgcart[idiagmin:idiagmax]
        if idiagmin < idiagmax:
            ekin = 0.5 * np.sum(np.power(kgcart_diag, 2), axis=1).astype('c16')
            for idiag in range(idiagmin, idiagmax):
                hamblock[idiag-gvecrange1[0], idiag-gvecrange2[0]] += ekin[idiag-idiagmin]
        
        # nonlocal part
        # V_{NL}^a = \sum_{h,h'}^{N_h(a)}\ket{\beta_{a h s k}} d_{h h' a s} \bra{\beta_{a h' s k}}
        nat = self.structure.natom
        atomic_species = self.structure.atomic_species
        atomic_numbers = self.structure.atomic_numbers
        deeq = self.deeq
        betapsi = self.vkbgdatas[ik].vkbg
        atomic_pos = self.structure.atomic_positions_cart
        
        # Preparation of KB projector parts
        ggcart = 2 * np.pi * self.g_g @ self.structure.gprim
        phase_gg_spcs = {}
        hnloc_right_spcs = {}
        for spc in atomic_species:
            phase_gg_spcs[spc] = np.zeros(self.ng_g, dtype='c16')
            betapsi2 = betapsi[spc][:, slice(*gvecrange2)]
            hnloc_right_spcs[spc] = deeq[spc][0, :, :] @ betapsi2.conj() #  (nh, nh) @ (nh, ng)
        # use for loop to save memory
        for iat in range(nat):
            spc = atomic_numbers[iat]
            phase_gg_spcs[spc] += np.exp(-1j * np.dot(ggcart, atomic_pos[iat]))
        
        # use a for loop along gvecrange1 to save memory
        if is_master():
            print(f'Building {gvecrange1[1]-gvecrange1[0]} plane-wave Hamiltonain blocks')
        for ig1 in mytqdm(range(*gvecrange1)):
            gk_g1 = self.vkbgdatas[ik].miller_idc[ig1]
            gk_g2 = self.vkbgdatas[ik].miller_idc[slice(*gvecrange2)]
            gdiff = gk_g1[None, :] - gk_g2[:, :]
            ixgdiff = findvector(gdiff, self.FFTgrid, self.index_vec, self.g_g)
            # local part
            hamblock[ig1-gvecrange1[0], :] += self.vscg[0, ixgdiff] # ! only no spin
            # nonlocal part
            for spc in atomic_species:
                betapsi1 = betapsi[spc][:, ig1]
                phase_kg = phase_gg_spcs[spc][ixgdiff]
                hamblock[ig1-gvecrange1[0], :] += betapsi1 @ hnloc_right_spcs[spc] * phase_kg # (ng, nh) @ (nh, ng)
        
        return hamblock
    
    def reduce_kgrid(self, kgrid):
        '''
        Reduce the k-grid and k-weights using time-reversal symmetry.
        '''
        
        kpts_new, kptwts_new = kgrid_with_tr(kgrid)
        
        nk_new = len(kpts_new)
        map_old_new = make_kkmap(self.kpts, kpts_new)
            
        kpts_cart_new = self.kpts_cart[map_old_new, :]
        
        vkbgdatas_new = []
        for ik_new in range(nk_new):
            vkbgdatas_new.append(self.vkbgdatas[map_old_new[ik_new]])
        
        self.nk = nk_new
        self.kpts = kpts_new
        self.kptwts = kptwts_new
        self.kpts_cart = kpts_cart_new
        self.vkbgdatas = vkbgdatas_new


class HGDataPAW:
    '''
    Attributes:
    ---------
    FFTngf(3): dimensions of fine FFT grid
    vscg_full(FFTngf): complex, local part of Hamiltonian in reciprocal space, in Hartree
    nh_spcs: int->int, number of KB projectors of each atomic species
    cdij: List[(lmmax[spc], lmmax[spc])], length=number of atoms
    cqij: dimension same as cdij
    nk: int
    vkbgdatas: List[VKBGdata], len=nk
    '''
        
    def __init__(self, gpawsave=None, gpaw_datadir=None, structure=None, kgrid=None, ecutwfn=None, 
                 interface='gpaw'):
        
        kpts, kptwts = kgrid_with_tr(kgrid)
        nk = len(kpts)
        
        if interface == 'gpaw':
            stru_save = from_gpaw(gpawsave)
            if structure is None:
                structure = stru_save
            else:
                assert structure == stru_save
            assert ecutwfn is not None
            
            from ase.io import ulm
            with ulm.open(gpawsave) as f:
                vscr = f.hamiltonian.potential[0] / hartree2ev # ! no spin
                cdij_raw = f.hamiltonian.atomic_hamiltonian_matrices[0] / hartree2ev # ! no spin
            
            FFTngf = np.array(vscr.shape)
            vscg = np.fft.fftn(vscr, norm='forward')
            # vscg = vscr
            
            gpawprojR = AOData(structure, None, gpaw_datadir, aocode='gpaw-projR')
            
            kgsphere = kGsphere(structure.rprim, ecutwfn)
            vkbgdatas = []
            
            if is_master():
                print(f'Calculating FT of PAW projectors at {nk} k-points')
            for ikpt in mytqdm(range(nk)):
                kpt = kpts[ikpt, :]
                ngk_g, gk_g, kgcart = kgsphere.get_gk_g(kpt)
                FT_kg_orb_spcs = calc_FT_kg_orb_spcs(ngk_g, kgcart, gpawprojR, ecutwfn)
                vkbg_spc = {}
                for k, v in FT_kg_orb_spcs.items():
                    vkbg_spc[k] = v.T / np.sqrt(structure.cell_volume)
                vkbgdata = VKBGData(ngk_g, gk_g, vkbg_spc, kgcart)
                vkbgdatas.append(vkbgdata)
            
            cdij = []
            pos = 0
            for iat in range(structure.natom):
                spc = structure.atomic_numbers[iat]
                nsize = gpawprojR.norbfull_spc[spc]
                offset = nsize * (nsize+1) // 2
                cdij.append(unpack(cdij_raw[pos:pos+offset], nsize))
                pos += offset
            assert pos == len(cdij_raw)
            
            nh_spcs = {}
            for spc in structure.atomic_species:
                nh_spcs[spc] = gpawprojR.norbfull_spc[spc]
            
            cqij_dict = {}
            spc_numbers = structure.atomic_species
            spc_names = atom_number2name(spc_numbers)
            for spc_nu, spc_na in zip(spc_numbers, spc_names):
                gpawpsp = gpaw_psp(f'{gpaw_datadir}/{spc_na}.PBE.gz') # todo: not using PBE
                cqij_dict[spc_nu] = gpawpsp.get_cqij()
            cqij = []
            for spc in structure.atomic_numbers:
                cqij.append(cqij_dict[spc])
            
            self.FFTngf = FFTngf
            self.vscg_full = vscg
            self.ecutwfn = ecutwfn
            
            self.nh_spcs = nh_spcs
            self.cdij = cdij # List[(lmmax[spc], lmmax[spc])]
            self.cqij = cqij
            self.vkbgdatas = vkbgdatas

        else:
            raise NotImplementedError(interface)
        
        self.structure = structure
            
        self.nk = nk
        self.kpts = kpts
        self.kptwts = kptwts
        self.kpts_cart = 2 * np.pi * self.kpts @ self.structure.gprim
        
        self.interface = interface
        self.ispaw = True
        
    def build_hs_paw(self, ik, kind='h', gvecrange1=(None,None), gvecrange2=(None,None)):
        '''Build plane-wave Hamiltonian block in Hartree (or overlap block when kind='s')'''
        assert kind in ['h', 's']
        
        gvecrange1 = set_range(gvecrange1, 0, self.vkbgdatas[ik].ng)
        gvecrange2 = set_range(gvecrange2, 0, self.vkbgdatas[ik].ng)
        
        hamblock = np.zeros((gvecrange1[1]-gvecrange1[0], gvecrange2[1]-gvecrange2[0]), dtype='c16')
        
        # kinetic part
        idiagmin = max(gvecrange1[0], gvecrange2[0])
        idiagmax = min(gvecrange1[1], gvecrange2[1])
        kgcart_diag = self.vkbgdatas[ik].kgcart[idiagmin:idiagmax]
        if idiagmin < idiagmax:
            if kind == 'h':
                ekin = 0.5 * np.sum(np.power(kgcart_diag, 2), axis=1).astype('c16')
            elif kind == 's':
                ekin = np.ones(idiagmax-idiagmin, dtype='c16')
            for idiag in range(idiagmin, idiagmax):
                hamblock[idiag-gvecrange1[0], idiag-gvecrange2[0]] += ekin[idiag-idiagmin]
        
        # nonlocal part
        # V_{NL}^a = \sum_{h,h'}^{N_h(a)}\ket{\beta_{a h s k}} d_{h h' a s} \bra{\beta_{a h' s k}}
        nat = self.structure.natom
        atomic_species = self.structure.atomic_species
        atomic_numbers = self.structure.atomic_numbers
        betapsi = self.vkbgdatas[ik].vkbg
        atomic_pos = self.structure.atomic_positions_cart
        kgcart1 = self.vkbgdatas[ik].kgcart[slice(*gvecrange1)]
        kgcart2 = self.vkbgdatas[ik].kgcart[slice(*gvecrange2)]
        FFTngf = self.FFTngf
        if kind == 'h':
            deeq = self.cdij
        else:
            deeq = self.cqij
        
        # use a for loop along gvecrange1 to save memory
        if kind == 'h':
            if is_master():
                print(f'Building local part of Hamiltonian with {gvecrange1[1]-gvecrange1[0]} blocks')
            for ig1 in mytqdm(range(*gvecrange1)):
                gk_g1 = self.vkbgdatas[ik].miller_idc[ig1]
                gk_g2 = self.vkbgdatas[ik].miller_idc[slice(*gvecrange2)]
                gdiff = gk_g1[None, :] - gk_g2[:, :]
                idx1 = np.divmod(gdiff[..., 0], FFTngf[0])[1]
                idx2 = np.divmod(gdiff[..., 1], FFTngf[1])[1]
                idx3 = np.divmod(gdiff[..., 2], FFTngf[2])[1]
                # local part
                hamblock[ig1-gvecrange1[0], :] += self.vscg_full[idx1, idx2, idx3] # ! only no spin
        
        if is_master():
            print(f'Building nonlocal projectors of {nat} atoms')
        for iat in mytqdm(range(nat)):
            spc = atomic_numbers[iat]
            phase1 = np.exp(-1j * np.dot(kgcart1, atomic_pos[iat]))
            phase2 = np.exp(-1j * np.dot(kgcart2, atomic_pos[iat]))
            betapsi1 = betapsi[spc][:, slice(*gvecrange1)] * phase1[None, :]
            betapsi2 = betapsi[spc][:, slice(*gvecrange2)] * phase2[None, :]
            tmp = deeq[iat][:, :] @ betapsi2.conj() #  (nh, nh) @ (nh, ng) # ! only no spin
            # use a for loop along gvecrange1 to save memory
            for ig in range(gvecrange1[1] - gvecrange1[0]):
                hamblock[ig, :] += betapsi1[:, ig] @ tmp # (ng, nh) @ (nh, ng)
        
        return hamblock


def gvec_index(ng, g_g, FFTgrid):
    '''Map every g-vector to a unique integer.'''
    # BerkeleyGW/Common/input_utils.f90
    index_vec = np.zeros(np.prod(FFTgrid), dtype=int)
    if not ( np.all(2*g_g<FFTgrid[None,:]) and np.all(2*g_g>=-FFTgrid[None,:]) ):
        raise AssertionError("gvectors must be in the interval [-FFTgrid/2, FFTgrid/2)")
    iadd = ( (g_g[:,0] + FFTgrid[0]//2) * FFTgrid[1] + g_g[:,1] + FFTgrid[1]//2 ) * FFTgrid[2] \
            + g_g[:,2] + FFTgrid[2]//2
    for ig in range(ng):
        index_vec[iadd[ig]] = ig
    return index_vec

def findvector(kk, FFTgrid, index_vec, g_g):
    '''Find the index of every vector kk in gg.'''
    # BerkeleyGW/Common/misc.f90
    # kk: shape(..., 3)
    iout = ( (kk[...,0] + FFTgrid[0]//2) * FFTgrid[1] + kk[...,1] + FFTgrid[1]//2 ) * FFTgrid[2] \
            + kk[...,2] + FFTgrid[2]//2
    iout = index_vec[iout]
    assert np.all(kk==g_g[iout, :])
    return iout

def unpack(vec, nsize):
    # gpaw/utilities/__init__.py pack2
    assert len(vec) == nsize * (nsize+1) // 2
    mat = np.empty((nsize, nsize))
    ipos = 0
    for i in range(nsize):
        mat[i, i] = vec[ipos]
        ipos += 1
        for j in range(i+1, nsize):
            mat[i, j] = vec[ipos]
            mat[j, i] = vec[ipos]
            ipos += 1
    assert ipos == len(vec)
    return mat