import numpy as np
from scipy.sparse import csr_matrix

from .. import config as CFG
from ..utils.supercell import minimum_supercell
from .matao import PairsInfo, MatAO

class OrbInfo:
    def __init__(self, aodata, structure=None):
        if structure is None:
            self.structure = stru = aodata.structure
        else:
            self.structure = stru = structure

        self.aodata = aodata
        
        # get total number of orbitals
        no = 0
        for spc in stru.atomic_species:
            nat_spc = np.sum(stru.atomic_numbers == spc)
            no += nat_spc * aodata.norbfull_spc[spc]
        
        self.norb = no
        
        self._maxnrad = max(aodata.nradial_spc.values())
        self._maxl = max(max(l) for l in aodata.ls_spc.values())
        self._maxatorb = max(aodata.norbfull_spc.values())
        
        # (io, [iat, iatorb, irad, m])
        # iat: index of atom
        # iatorb: index of orbital in atom
        # irad: index of radial function
        # m: angular momentum
        orbinfo_arr = np.empty((no, 4), dtype='i8')

        orbinfo_spc = {}
        for spc in stru.atomic_species:
            orbinfo_this = np.empty((aodata.norbfull_spc[spc], 3), dtype='i8') # (io, [iatorb, irad, m])
            pos = 0
            orbinfo_this[:, 0] = np.arange(0, aodata.norbfull_spc[spc], 1)
            for irad in range(aodata.nradial_spc[spc]):
                gridfunc = aodata.phirgrids_spc[spc][irad]
                l = gridfunc.l
                radslice = slice(pos, pos+(2*l+1))
                orbinfo_this[radslice, 1] = irad
                orbinfo_this[radslice, 2] = np.arange(-l, l+1, 1)
                pos += 2*l+1
            orbinfo_spc[spc] = orbinfo_this
        
        pos = 0
        for iat in range(stru.natom):
            spc = stru.atomic_numbers[iat]
            slice_atom = slice(pos, pos+aodata.norbfull_spc[spc])
            orbinfo_arr[slice_atom, 0] = iat
            orbinfo_arr[slice_atom, 1:4] = orbinfo_spc[spc]
            pos += aodata.norbfull_spc[spc]
        
        self.orbinfo_arr = orbinfo_arr
        self.iat_orbs = orbinfo_arr[:, 0]
        self.iatorb_orbs = orbinfo_arr[:, 1]
        self.irad_orbs = orbinfo_arr[:, 2]
        self.m_orbs = orbinfo_arr[:, 3]

        # vhash2: hash value of (iat, iatorb) (hash mapping defined below)
        vhash2 = hash2(self.iat_orbs, self.iatorb_orbs, self._maxatorb)
        assert np.all(vhash2[:-1] <= vhash2[1:]) # already sorted
        self.vhash2 = vhash2
        
        # vhash3: hash value of (iat, irad, m) (hash mapping defined below)
        vhash3 = hash3(self.iat_orbs, self.irad_orbs, self.m_orbs, self._maxnrad, self._maxl)
        assert np.all(vhash3[:-1] <= vhash3[1:]) # already sorted
        self.vhash3 = vhash3
    
    def find_orbindx2(self, iat, iatorb):
        '''
        Given (iat, iatorb), find orbital index.
        '''
        vhash = hash2(iat, iatorb, self._maxatorb)
        assert np.all(np.isin(vhash, self.vhash2))
        return np.searchsorted(self.vhash2, vhash, side='left')
        
    def find_orbindx3(self, iat, irad, m):
        '''
        Given (iat, irad, m), find orbital index.
        '''
        vhash = hash3(iat, irad, m, self._maxnrad, self._maxl)
        assert np.all(np.isin(vhash, self.vhash3))
        return np.searchsorted(self.vhash3, vhash, side='left')


class OrbInfoSuperCell(OrbInfo):
    def __init__(self, aodata, structure_sc, orbinfo_uc=None):
        super().__init__(aodata, structure_sc)
        self.orbinfo_uc = OrbInfo(aodata) if orbinfo_uc is None else orbinfo_uc
        self.structure_uc = aodata.structure

        # cache some data for iorb_sc2uc
        self.trans_cuc_orbs, self.iouc_orbs = self._iorb_sc2uc(np.arange(0, self.norb, 1), return_trans_cuc=True)
        self.trans_orbs, _ = self._iorb_sc2uc(np.arange(0, self.norb, 1), return_trans_cuc=False)
        self.trans_cuc_orbs = self.trans_cuc_orbs.astype('i4')
        self.trans_orbs = self.trans_orbs.astype('i4')
        self.iouc_orbs = self.iouc_orbs.astype('i4')
        
        # cache for iorb_uc2sc
        da = structure_sc.limits_abc[0, 1] - structure_sc.limits_abc[0, 0]
        db = structure_sc.limits_abc[1, 1] - structure_sc.limits_abc[1, 0]
        dc = structure_sc.limits_abc[2, 1] - structure_sc.limits_abc[2, 0]
        starta = structure_sc.limits_abc[0, 0]
        startb = structure_sc.limits_abc[1, 0]
        startc = structure_sc.limits_abc[2, 0]
        nouc = self.orbinfo_uc.norb
        iosc_uc = np.empty(structure_sc.nsc*nouc, dtype='i4')
        for isc in range(structure_sc.nsc):
            iosc_uc[isc*nouc:(isc+1)*nouc] = \
                self._iorb_uc2sc(np.repeat(structure_sc.translations_cuc[isc][None, :], nouc, axis=0),
                                 np.arange(0, nouc, 1), True)
        iosc_uc = iosc_uc.reshape((da, db, dc, nouc))
        self.cache_uc2sc = (starta, startb, startc, iosc_uc)
    
    def _iorb_sc2uc(self, iorbsc, return_trans_cuc):
        iatsc = self.iat_orbs[iorbsc]
        translations, iatuc = self.structure.iat_sc2uc(iatsc, return_trans_cuc)
        iorbuc = self.orbinfo_uc.find_orbindx2(iatuc, self.iatorb_orbs[iorbsc])
        return translations, iorbuc
    
    def iorb_sc2uc(self, iorbsc, return_trans_cuc):
        # This is the faster version using cached data
        # return self._iorb_sc2uc(iorbsc, return_trans_cuc)
        if return_trans_cuc:
            return self.trans_cuc_orbs[iorbsc], self.iouc_orbs[iorbsc]
        else:
            return self.trans_orbs[iorbsc], self.iouc_orbs[iorbsc]

    def _iorb_uc2sc(self, translations, iorbuc, input_trans_cuc):
        iatuc = self.orbinfo_uc.iat_orbs[iorbuc]
        iatsc = self.structure.iat_uc2sc(translations, iatuc, input_trans_cuc)
        iatorb = self.orbinfo_uc.iatorb_orbs[iorbuc]
        return self.find_orbindx2(iatsc, iatorb)
    
    def iorb_uc2sc(self, translations, iorbuc, input_trans_cuc):
        # This is the faster version using cached data
        # return self._iorb_uc2sc(translations, iorbuc, input_trans_cuc)
        assert input_trans_cuc
        starta, startb, startc, iosc_uc = self.cache_uc2sc
        return iosc_uc[translations[:, 0] - starta,
                       translations[:, 1] - startb,
                       translations[:, 2] - startc, iorbuc]

    def orbpair_translate_to_uc(self, iorb1, iorb2):
        '''
        For a hopping between orbital pair (iorb1, iorb2) in the supercell,
        find the equivalent orbital pair (iorb1p, iorb2p) such that iorb1p is in the unit cell
        '''
        translations1_cuc, iorbuc1 = self.iorb_sc2uc(iorb1, True)
        translations2_cuc, iorbuc2 = self.iorb_sc2uc(iorb2, True)
        translations_cuc = translations2_cuc - translations1_cuc
        iorb1p = iorbuc1
        iorb2p = self.iorb_uc2sc(translations_cuc, iorbuc2, True)
        return iorb1p, iorb2p

    def orbpair_invert(self, iorb1, iorb2):
        '''
        For a hopping between orbital pair (iorb1, iorb2) where iorb1 is in the unit cell and
        iorb2 in the supercell, find the orbital pair (iorb1p, iorb2p) which is equivalent to
        (iorb2, iorb1) and iorb1p is in the unit cell.
        This is used in unfold_with_hermiticity.
        '''
        translations_cuc, iorbuc2 = self.iorb_sc2uc(iorb2, True)
        iorb1p = iorbuc2
        iorb2p = self.iorb_uc2sc(-translations_cuc, iorb1, True)
        return iorb1p, iorb2p

def hash2(iat, iatorb, maxatorb):
    '''
    Map a group of (iat, iatorb) to unique indices
    iat, iatorb must be integer arrays of the same dimensions
    maxatorb is an integer
    '''
    return iat * maxatorb + iatorb

def hash3(iat, irad, m, maxnrad, maxl):
    '''
    Map a group of (iat, irad, m) to unique indices
    iat, irad, m must be integer arrays of the same dimensions
    maxnrad, maxl are integers
    '''
    return (iat * maxnrad + irad) * (2*maxl+1) + maxl + m


class MatAOCSR:
    def __init__(self, aodata1, aodata2=None, maxr=None):
        '''
        maxr: cutoff radius to make supercell. If None, will be determined by basis cutoff: max(r1+r2)
        '''
        if aodata2 is None:
            aodata2 = aodata1
        else:
            assert aodata2.structure == aodata1.structure

        self.structure = aodata1.structure
        self.orbinfo1 = OrbInfo(aodata1)

        if maxr is None:
            maxr = max(aodata1.cutoffs.values()) + max(aodata2.cutoffs.values())
        self.supercell = minimum_supercell(self.structure, maxr)
        self.orbinfo2 = OrbInfoSuperCell(aodata2, self.supercell, 
                                         orbinfo_uc=self.orbinfo1 if aodata2 is aodata1 else None)
        self.mat = csr_matrix((self.orbinfo1.norb, self.orbinfo2.norb), dtype='f8')
    
    def reset_mat(self):
        self.mat = csr_matrix((self.orbinfo1.norb, self.orbinfo2.norb), dtype='f8')
    
    def to_matao(self, verbose=False):
        # sizes: total number of orbitals per atom
        atom_nbrs = self.structure.atomic_numbers
        sizes1 = np.empty(self.structure.natom, 'i4')
        sizes2 = np.empty(self.structure.natom, 'i4')
        for iat in range(self.structure.natom):
            sizes1[iat] = self.orbinfo1.aodata.norbfull_spc[atom_nbrs[iat]]
            sizes2[iat] = self.orbinfo2.aodata.norbfull_spc[atom_nbrs[iat]]

        # Create a PairsInfo object for every orbital in unit cell
        pairs_iouc, slices_iouc = [], []
        for iorbuc in range(self.orbinfo1.norb):
            indslice = slice(self.mat.indptr[iorbuc], self.mat.indptr[iorbuc+1])
            iorbsc2 = self.mat.indices[indslice]

            # find atom pairs
            trans_cuc, iorbuc2 = self.orbinfo2.iorb_sc2uc(iorbsc2, True)
            iatuc1 = self.orbinfo1.iat_orbs[iorbuc]
            iatuc1 = np.full(len(iorbsc2), iatuc1, dtype='i8')
            iatsc2 = self.orbinfo2.orbinfo_uc.iat_orbs[iorbuc2]
            atom_pairs = np.stack((iatuc1, iatsc2), axis=1)

            # get translations
            translations = self.structure.trans_cuc_to_original(trans_cuc, iatuc1, iatsc2)

            pairs_iouc.append(PairsInfo(self.structure, translations, atom_pairs))
            if verbose and (pairs_iouc[-1].npairs == 0): # for debugging
                print('empty pair', iorbuc, self.orbinfo1.iat_orbs[iorbuc], self.orbinfo1.iatorb_orbs[iorbuc])
            slices_iouc.append(indslice)
        
        # Find all pairs
        indices_uniq = np.unique(np.concatenate([pair.get_indices() for pair in pairs_iouc]))
        pairs_all = PairsInfo.from_indices(self.structure, indices_uniq)
        pairs_all.sort()
        matao = MatAO.init_mats(pairs_all, self.orbinfo1.aodata, aodata2=self.orbinfo2.aodata, filling_value=None)
        if CFG.DEBUG_CSR_CNVRT: 
            checkmat = MatAO.init_mats(pairs_all, self.orbinfo1.aodata, aodata2=self.orbinfo2.aodata, filling_value=None)

        # allocate the concatenated matrix: each matrix in matao is a slice of it
        matblockptr = np.empty(pairs_all.npairs+1, dtype='i8')
        indtmp = 0
        for ipair in range(pairs_all.npairs):
            matblockptr[ipair] = indtmp
            size1 = sizes1[pairs_all.atom_pairs[ipair, 0]]
            size2 = sizes2[pairs_all.atom_pairs[ipair, 1]]
            indtmp += size1 * size2
        matblockptr[ipair+1] = indtmp
        hmatcat = np.zeros(indtmp, dtype='f8')
        if CFG.DEBUG_CSR_CNVRT: icheck = np.zeros(indtmp, dtype='i4')

        for ipair in range(pairs_all.npairs):
            size1 = sizes1[pairs_all.atom_pairs[ipair, 0]]
            size2 = sizes2[pairs_all.atom_pairs[ipair, 1]]
            matao.mats[ipair] = hmatcat[matblockptr[ipair]:matblockptr[ipair+1]].reshape(size1, size2)
            if CFG.DEBUG_CSR_CNVRT: 
                checkmat.mats[ipair] = icheck[matblockptr[ipair]:matblockptr[ipair+1]].reshape(size1, size2)

        # get positions of matrix elements in the concatenated matrix
        for iorbuc in range(self.orbinfo1.norb):
            iorbsc2 = self.mat.indices[slices_iouc[iorbuc]]
            pairs = pairs_iouc[iorbuc]
            pairindc = pairs_all.findipairs(pairs)
            iatorb1 = self.orbinfo1.iatorb_orbs[iorbuc]
            iatorb2 = self.orbinfo2.iatorb_orbs[iorbsc2]
            size2 = sizes2[pairs_iouc[iorbuc].atom_pairs[:, 1]]
            elempos = matblockptr[pairindc] + iatorb1*size2 + iatorb2
            hmatcat[elempos] = self.mat.data[slices_iouc[iorbuc]]
            if CFG.DEBUG_CSR_CNVRT: icheck[elempos] += 1
        # print(np.sum(icheck==1), len(icheck))
        if CFG.DEBUG_CSR_CNVRT: assert np.all((icheck==1) + (icheck==0))

        return matao

    def unfold_with_hermiticity(self):
        assert self.orbinfo2.aodata is self.orbinfo1.aodata
        coomat = self.mat.tocoo(copy=True)
        iorb1p, iorb2p = self.orbinfo2.orbpair_invert(coomat.row, coomat.col)
        coomat.data[coomat.col==iorb2p] = 0. # remove diagonal
        matT = csr_matrix((coomat.data, (iorb1p, iorb2p)), shape=coomat.shape)
        self.mat = self.mat + matT