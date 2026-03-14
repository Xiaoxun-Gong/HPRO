import numpy as np

from .misc import index_traverse, sort_translations_siesta
from .structure import Structure

class SuperCell(Structure):
    def __init__(self, structure_uc, limits_abc, ordering='normal'):
        '''
        limits_abc array(3, 2): integers [[a_min, a_max], [b_min, b_max], [c_min, c_max]]
                                For example, to make a 3x3 supercell, this is [[0, 3], [0, 3], [0, 3]]
                                left bound is closed and right bound is open.
        
        Attributes:
            stru_uc: corresponding unit cell structure
            limits_abc: limits_abc
            translations_cuc: (nsc, 3) For each unit cell forming the supercell, this is its translation from the original unit cell.
            translations: (natsc, 3) translations from original atom position to supercell position
            iuc: index of original unit cell among all translated unit cells
        '''
        limits_abc = np.array(limits_abc)
        assert np.all(limits_abc[:, 0] <= 0)
        assert np.all(limits_abc[:, 1] >= 0)
        sizes_abc = limits_abc[:, 1] - limits_abc[:, 0]
        nsc = np.prod(sizes_abc)

        self.stru_uc = structure_uc
        self.limits_abc = limits_abc
        self.nsc = nsc

        translations_cuc = index_traverse(np.arange(limits_abc[0, 0], limits_abc[0, 1], 1),
                                          np.arange(limits_abc[1, 0], limits_abc[1, 1], 1),
                                          np.arange(limits_abc[2, 0], limits_abc[2, 1], 1))
        assert ordering in ['normal', 'siesta']
        if ordering == 'siesta':
            translations_cuc = sort_translations_siesta(translations_cuc)
        
        self.translations_cuc = translations_cuc
        self.iuc = np.where(np.all(translations_cuc == 0, axis=1))[0]

        tmp = translations_cuc[:, None, :] - structure_uc.trans_from_cuc[None, :, :]
        self.translations = tmp.reshape(-1, 3)

        self._max_translations = np.max(np.abs(translations_cuc))
        self._vhash = hash_translations(translations_cuc, self._max_translations)
        
        rprim = structure_uc.rprim * sizes_abc[:, None]

        # supercell translations is the slowest changing index, the unit cell atomic index is fastest
        atomic_numbers = np.repeat(structure_uc.atomic_numbers[None, :], nsc, axis=0).reshape(-1)

        pos_red = structure_uc.atomic_positions_red_uc
        tmp = pos_red[None, :, :] + translations_cuc[:, None, :]
        atomic_positions = tmp.reshape(nsc*structure_uc.natom, 3) / sizes_abc[None, :]

        super(SuperCell, self).__init__(rprim, atomic_numbers, atomic_positions, 
                                        structure_uc.efermi, atomic_positions_is_cart=False)
    
    @classmethod
    def make_supercell(cls, structure_uc, sizes):
        limits_abc = np.array([[0, sizes[0]], [0, sizes[1]], [0, sizes[2]]])
        return cls(structure_uc, limits_abc)
    
    def iat_sc2uc(self, iatsc, return_trans_cuc):
        '''
        Given the index of an atoms in the supercell, return the primitive translation from unit 
        cell to this atom, and atom index in the unit cell.

        Inputs:
            atm_idx: array(...), integer
        Returns:
            translations: array(..., 3), integer
            atmidx_uc: array(...), integer
        '''
        assert np.all(iatsc < self.natom)
        assert np.all(iatsc >= 0)
        natuc = self.stru_uc.natom
        if return_trans_cuc:
            translations = self.translations_cuc[iatsc//natuc, :]
        else:
            translations = self.translations[iatsc, :]
        iatuc = iatsc % natuc
        return translations, iatuc
    
    def iat_uc2sc(self, translations, iatuc, input_trans_cuc):
        if input_trans_cuc:
            translations_cuc = translations
        else:
            translations_cuc = translations + self.stru_uc.trans_from_cuc[iatuc, :]
        vhash = hash_translations(translations_cuc, self._max_translations)
        assert np.all(np.isin(vhash, self._vhash))
        itrans = np.searchsorted(self._vhash, vhash)
        return itrans * self.stru_uc.natom + iatuc

def hash_translations(translations, max_translations):
    tmp = max_translations * 2 + 1
    return (translations + max_translations) @ np.array([tmp**2, tmp, 1])

def minimum_supercell(structure, r):
    '''
    Find the minimum supercell that contains all the volume "reachable" from the unitcell within distance r
    '''
    # unit cell "thickness" along ith lattice vector direction is 1/|G_i|
    max_n_extend = np.floor(r * np.linalg.norm(structure.gprim, axis=-1) + 1).astype(int)
    return SuperCell(structure, np.stack([-max_n_extend, max_n_extend+1], axis=1))


'''
Note about conventional unit cell (CUC):

CUC refers to the volume enclosed by the parallelepiped formed by three lattice vectors. If an atom 
is in the CUC, its reduced position should be within [0, 1). Converting all atoms to be within the CUC will
make it easier for the code to find atom pairs within cutoff.

However, not all atoms in the structure are in CUC. Therefore, we have structure.trans_from_cuc to deal with this.
It gives information on how to translate an atom from its position in CUC to its original position. For example, 
the following will hold for an atom not in the CUC:
    atomic_position_red = [-0.1, 0., 0.]
    atomic_position_red_uc = [0.9, 0., 0.]
    trans_from_cuc = [-1, 0, 0]

This is also tricky when we build a supercell. In this code, a supercell is made up of several translated CUCs.
The number of translated CUC is nsc.
    SuperCell.translations_cuc (nsc, 3) means the primitive translation vector between the original CUC and the 
                                        translated CUC. This is also the translation vector to translate an atom
                                        from its position in the original CUC to its position in the translated CUC.
    SuperCell.translations (natsc, 3) means the primitive translation vector to translate an atom from its original
                                      position to its new position in the translated CUC.
    These are related by: 
                   translations = translations_cuc - trans_from_cuc.                                  Eq. (1)
Please note that the dimensions of translations_cuc and translations are different. Eq. (1) is the simplified
version of the full formula.

When we use pairs_within_cutoff, what we found are actually pairs of atoms where the first atom is in the CUC,
and the second atom is in the translated CUC. The relative translations are between CUCs, not the relative
translations between atoms in their original positions. Therefore, we need to convert that using 
Structure.trans_cuc_to_original.
'''