import numpy as np
from scipy.sparse import csr_matrix

from .constants import hpro_rng
from .utils import atom_number2name

BASE = 20

class PairsInfo:
    '''
    This class is used to deal with atom pairs. The first atom in the pair is always assumed to be in the unit cell at (0, 0, 0),
    and the second atom is in the unit cell at (R1, R2, R3) where R1, R2, R3 are translations along lattice vectors.

    Each atom pair can be mapped to a unique integer index according to the function `pairs_to_indices` defined below. This is 
    beneficial for purposes such as sorting and searching of atom pairs.

    Attributes:
        npairs: int
        translations array(npairs, 3): translations along lattice vectors
        atom_pairs array(npairs, 2): first column is the atom in unit cell, second column is the atom in translated unit cell
    '''
    
    def __init__(self, structure, translations, atom_pairs):
        self.structure = structure
        self.npairs = translations.shape[0]
        assert atom_pairs.shape[0] == self.npairs
        self.translations = translations
        self.atom_pairs = atom_pairs
        self._indices = None
    
    @classmethod
    def from_indices(cls, structure, indices):
        translations, atom_pairs = indices_to_pairs(structure.natom, indices)
        pairs = PairsInfo(structure, translations, atom_pairs)
        pairs._indices = indices
        return pairs
    
    def get_indices(self):
        if self._indices is None:
            self._indices = pairs_to_indices(self.structure, self.translations, self.atom_pairs)
        return self._indices

    def is_sorted(self):
        indices = self.get_indices()
        return np.all(indices[:-1] <= indices[1:])
    
    def sort(self):
        if self.is_sorted():
            return
        indices = self.get_indices()
        argsort = np.argsort(indices)
        self.slice(argsort, new_indices=indices[argsort])
    
    def sort_atom1(self):
        argsort = np.argsort(self.atom_pairs[:, 0], kind='stable')
        self.slice(argsort)
    
    def slice(self, val, new_indices=None):
        self.translations = self.translations[val]
        self.atom_pairs = self.atom_pairs[val]
        self.npairs = self.translations.shape[0]
        self._indices = new_indices

    def remove_ji(self):
        self.sort()
        translations_inv = -self.translations
        atom_pairs_inv = self.atom_pairs[:, np.array([1, 0])]
        indices_inv = pairs_to_indices(self.structure, translations_inv, atom_pairs_inv)
        argsort = np.argsort(indices_inv)
        arange = np.arange(0, self.npairs, 1)
        is_ij = arange <= argsort
        self.slice(is_ij)
    
    def samepairs(self, other):
        self.sort()
        other.sort()
        if self.npairs != other.npairs: return False
        if not np.all(self.translations == other.translations): return False
        if not np.all(self.atom_pairs == other.atom_pairs): return False
        return True
    
    def __eq__(self, other):
        return self.samepairs(other)

    def get_keystr(self, ipair):
        Rijab = self.translations[ipair].tolist() + (self.atom_pairs[ipair] + 1).tolist()
        return str(Rijab)

    def shuffle(self):
        argsort = np.arange(self.npairs)
        hpro_rng.shuffle(argsort)
        self.slice(argsort)
    
    def get_distances(self):
        rprim = self.structure.rprim
        atom_pos_cart = self.structure.atomic_positions_cart
        cart1 = self.translations @ rprim + atom_pos_cart[self.atom_pairs[:, 1], :]
        cart2 = atom_pos_cart[self.atom_pairs[:, 0]]
        pair_distance = np.linalg.norm(cart1 - cart2, axis=-1)
        return pair_distance
    
    def findipairs(self, pairs):
        indices = self.get_indices()
        assert np.all(np.isin(pairs.get_indices(), indices))
        listipair = np.searchsorted(indices, pairs.get_indices())
        return listipair

    def __add__(self, other):
        assert self.__class__ is other.__class__
        assert self.structure == other.structure
        self.sort()
        other.sort()
        indices1 = self.get_indices()
        indices2 = other.get_indices()
        indices_all = np.union1d(indices1, indices2)
        return self.__class__.from_indices(self.structure, indices_all)
    

def pairs_to_indices(structure, translations, atom_pairs):
    '''
    Convert pairs to its unique integer index.
    '''
    assert np.max(translations) < BASE // 2
    assert np.min(translations) >= -BASE // 2
    natom = structure.natom
    atom_numbers = structure.atomic_numbers
    atm1 = atom_pairs[:, 0]
    atm2 = atom_pairs[:, 1]
    spc1 = atom_numbers[atm1]
    spc2 = atom_numbers[atm2]
    rem = translations + BASE//2 # from [-BASE//2, BASE//2) to [0, BASE)
    return ((spc1 * 200 + spc2) * BASE**3 * natom**2 +
            (atm1 * natom + atm2) * BASE**3 +
            rem @ np.array([BASE**2, BASE, 1]))

def indices_to_pairs(natom, indices):
    '''
    This is the inverse of the function `paris_to_indices`.
    '''
    npairs = len(indices)
    translations = np.empty((npairs, 3), dtype=int)
    atom_pairs = np.empty((npairs, 2), dtype=int)
    quo, translations[:, 2] = np.divmod(indices, BASE)
    quo, translations[:, 1] = np.divmod(quo, BASE)
    quo, translations[:, 0] = np.divmod(quo, BASE)
    translations -= BASE//2 # from [0, BASE) to [-BASE//2, BASE//2)
    quo, atom_pairs[:, 1] = np.divmod(quo, natom)
    quo, atom_pairs[:, 0] = np.divmod(quo, natom)
    return translations, atom_pairs


class MatLCAO(PairsInfo):
    '''
    This is the class to store matrices in the LCAO basis. The matrices are separated by blocks, and each block corresponds
    to the matrices of one atom pair. The matrix blocks are stored as a list of arrays in the `mats` attribute. 

    Attributes:
        mats list[array]: The matrix blocks. Length of this list is the same as `self.npairs`.
        lcaodata1 (LCAOData): Information for basis functions on the left of the matrix.
        lcaodata2 (LCAOData, optional): Information for basis functions on the right of the matrix. Defaults to lcaodata1 if not provided.
    '''
    def __init__(self, structure, translations, atom_pairs, mats, lcaodata1, lcaodata2=None):
        super(MatLCAO, self).__init__(structure, translations, atom_pairs)
        self.mats = mats
        self.lcaodata1 = lcaodata1
        self.lcaodata2 = lcaodata2 if lcaodata2 is not None else lcaodata1
        assert self.lcaodata1.structure == self.lcaodata2.structure == self.structure
        self.matscsr_R = None
        self.norb_total = None

    @classmethod
    def from_pairs(cls, pairs, mats, lcaodata1, lcaodata2=None):
        '''
        Create MatLCAO object using pairs.
        '''
        return cls(pairs.structure, pairs.translations, pairs.atom_pairs, mats, lcaodata1, lcaodata2=lcaodata2)
    
    def slice(self, val, new_indices=None):
        '''
        Take a slice from the full matrix.
        '''
        super(MatLCAO, self).slice(val, new_indices=new_indices)
        self.mats = [self.mats[i] for i in np.r_[val]]
    
    def get_pairs(self):
        return PairsInfo(self.structure, self.translations, self.atom_pairs)
    
    def get_pairs_ij(self):
        pairs = self.get_pairs()
        pairs.remove_ji()
        return pairs
    
    def duplicate(self):
        '''
        Duplicate this object so that it becomes a full matrix.
        '''
        # todo: check lcaodata1==lcaodata2
        translations_inv = -self.translations
        atom_pairs_inv = self.atom_pairs[:, np.array([1, 0])]
        indices = self.get_indices()
        indices_inv = pairs_to_indices(self.structure, translations_inv, atom_pairs_inv)
        not_redundant = ~np.isin(indices_inv, indices)
        for ipair in range(self.npairs):
            if not_redundant[ipair]:
                if self.mats[ipair] is None:
                    self.mats.append(None)
                elif np.isrealobj(self.mats[ipair]):
                    self.mats.append(self.mats[ipair].T.copy())
                else:
                    self.mats.append(self.mats[ipair].T.conj())
        translations_new = np.concatenate((self.translations, 
                                           translations_inv[not_redundant]), axis=0)
        atom_pairs_new = np.concatenate((self.atom_pairs,
                                         atom_pairs_inv[not_redundant]), axis=0)
        indices_new = np.concatenate((indices, indices_inv[not_redundant]))
        self.translations = translations_new
        self.atom_pairs = atom_pairs_new
        self.npairs = translations_new.shape[0]
        self._indices = indices_new
    
    @classmethod
    def setc(cls, pairs, lcaodata1, lcaodata2=None, filling_value=0, dtype=np.complex128):
        """
        Generate the MatLCAO object with proper error handling.
        """
        if lcaodata2 is None:
            lcaodata2 = lcaodata1
        atom_nbrs = pairs.structure.atomic_numbers
        mats = []
        for ipair in range(pairs.npairs):
            if filling_value is not None:
                size1 = lcaodata1.orbslices_spc[atom_nbrs[pairs.atom_pairs[ipair, 0]]][-1]
                size2 = lcaodata2.orbslices_spc[atom_nbrs[pairs.atom_pairs[ipair, 1]]][-1]
                mats.append(np.full((size1, size2), filling_value, dtype=dtype))
            else:
                mats.append(None)
        return cls.from_pairs(pairs, mats, lcaodata1, lcaodata2=lcaodata2)
    
    def delete_mats(self):
        self.mats = [None for _ in range(self.npairs)]
    
    def __add__(self, other):
        return self._add_sub(other, isadd=True)
    
    def __sub__(self, other):
        return self._add_sub(other, isadd=False)

    def _add_sub(self, other, isadd=True):
        '''
        Addition or subtraction of two matrices. Returns a new MatLCAO object.
        '''
        assert self.__class__ is other.__class__
        assert self.structure == other.structure
        self.sort()
        other.sort()
        indices1 = self.get_indices()
        indices2 = other.get_indices()
        indices_all = np.union1d(indices1, indices2)
        translations_all, atom_pairs_all = indices_to_pairs(self.structure.natom, indices_all)
        npairs_all = translations_all.shape[0]
        all_in1 = np.isin(indices_all, indices1)
        all_in2 = np.isin(indices_all, indices2)
        mats_all = []
        ix1, ix2 = 0, 0
        for ipair in range(npairs_all):
            in1 = all_in1[ipair]
            in2 = all_in2[ipair]
            if not in1: # in2 not in1
                assert in2
                mats_all.append(other.mats[ix2] if isadd else -other.mats[ix2])
                ix2 += 1
            elif not in2: # in1 not in2
                mats_all.append(self.mats[ix1])
                ix1 += 1
            else: # in1 and in2
                mats_all.append(self.mats[ix1] + (other.mats[ix2] if isadd else -other.mats[ix2]))
                ix1 += 1; ix2 += 1
        assert ix1 == self.npairs
        assert ix2 == other.npairs
        return self.__class__(self.structure, translations_all, atom_pairs_all, mats_all, self.lcaodata1, lcaodata2=self.lcaodata2)

    def convert_to(self, other):
        '''
        Convert this MatLCAO object to other object.
        '''
        self.sort()
        other.sort()
        indices_self = self.get_indices()
        indices_other = other.get_indices()
        pos_other_in_self, = np.where(np.isin(indices_other, indices_self))
        self_in_other = np.isin(indices_self, indices_other)
        assert np.sum(self_in_other) == len(pos_other_in_self)
        ix_other = 0
        for ix_self in range(self.npairs):
            if self_in_other[ix_self]:
                self.mats[ix_self][...] = other.mats[pos_other_in_self[ix_other]]
                ix_other += 1

    def reduce(self, op='mean'):
        '''
        Find the mean (or mean absolute, or mean square) value of all the matrix blocks.

        Parameters:
            op (str, optional): 'mean', 'mean_abs', or 'mean_square'. Defaults to 'mean'.
        '''
        assert op in ['mean', 'mean_abs', 'mean_square']
        totaln = 0
        total = 0.
        for ipair in range(self.npairs):
            mat = self.mats[ipair]
            totaln += np.prod(mat.shape)
            if op == 'mean':
                total += np.sum(mat)
            elif op == 'mean_abs':
                total += np.sum(np.abs(mat))
            elif op == 'mean_square':
                total += np.sum(np.abs(mat) ** 2)
        return total / totaln
    
    def extremum(self, op='max'):
        '''
        Find the extremum value of all the matrix blocks.

        Parameters:
            op (str, optional): 'min', 'max', 'min_abs', or 'max_abs'. Defaults to 'max'.
        '''
        assert op in ['min', 'max', 'min_abs', 'max_abs']
        last_m = m = None
        pos = None
        for ipair in range(self.npairs):
            mat = self.mats[ipair]
            if op == 'max':
                val = np.max(mat)
            elif op == 'min':
                val = np.min(mat)
            elif op == 'max_abs':
                val = np.max(np.abs(mat))
            elif op == 'min_abs':
                val = np.min(np.abs(mat))
            
            last_m = m
            if m is None:
                m = val
            elif op in ['min', 'min_abs']:
                m = min(m, val)
            elif op in ['max', 'max_abs']:
                m = max(m, val)

            if m != last_m:
                pos = np.concatenate((self.translations[ipair], self.atom_pairs[ipair]))
        
        return m, pos

    def r2k(self, k):
        '''
        Convert the real-space matrix to k space according to the rule:
        H_{ij}(k) = \sum_R H_{ij}(R) e^{ikR},
        where R is the lattice translation, ij is the atom pair.
        
        Parameters:
        ---------
        k(3): desired k-point in reduced coordinate
        
        Returns:
        ---------
        H(k) matrix :
            scipy.sparse.coo_matrix (if sparse is True)
            np.ndarray (if sparse is False)
        '''
        
        matscsr_R = self.to_csr()
        
        norb_total = self.norb_total
        matk = csr_matrix((norb_total, norb_total), dtype='c16')

        for R in matscsr_R.keys():
            trans = np.array(R)
            matk += matscsr_R[R] * np.exp(2.*np.pi*1j * np.dot(k, trans))
        
        return matk
    
    
    def to_csr(self):
        '''
        Convert the matrices to sparse CSR format.

        Returns:
        ---------
        matscsr_R: Dict[Tuple(int, int, int) -> scipy.sparse.csr.csr_matrix]
                   where R is the lattice translation vector.

        Use with care! If mats have been changed, use matlcao.matscsr_R=None to reset matscsr_R.
        '''

        if (self.matscsr_R is not None) and (self.norb_total is not None):
            return self.matscsr_R

        site_norbits = np.zeros(self.structure.natom, dtype=int)
        for iatm in range(self.structure.natom):
            atm_nbr = self.structure.atomic_numbers[iatm]
            site_norbits[iatm] = self.lcaodata1.orbslices_spc[atm_nbr][-1]
        norb_cumsum = np.cumsum(site_norbits)
        norb_total = norb_cumsum[-1]

        self.norb_total = norb_total

        # find the number of nonzero entries in sparse matrix
        i_j_set_R = {}
        ndata_R = {}
        for ipair in range(self.npairs):
            R = tuple(self.translations[ipair])
            i_j = tuple(self.atom_pairs[ipair])
            if R not in i_j_set_R:
                i_j_set_R[R] = set()
            if R not in ndata_R:
                ndata_R[R] = 0
            if i_j not in i_j_set_R[R]:
                n1, n2 = site_norbits[np.array(i_j)]
                i_j_set_R[R].add(i_j)
                ndata_R[R] += n1 * n2
        del i_j_set_R

        # allocate vectors
        mats_R = {}
        for R in ndata_R.keys():
            ndata = ndata_R[R]
            row = np.empty(ndata, dtype=int)
            col = np.empty(ndata, dtype=int)
            data = np.empty(ndata, dtype='f8') # we are using real numbers here
            pos = 0
            mats_R[R] = (row, col, data, pos)

        # fill in values
        for ipair in range(self.npairs):
            R = tuple(self.translations[ipair])
            row, col, data, pos = mats_R[R]
            i_j = self.atom_pairs[ipair]
            iatm, jatm = i_j
            n1, n2 = site_norbits[i_j]
            row_start = norb_cumsum[iatm]
            row_range = np.linspace(row_start - n1, row_start - 1, n1, dtype=int)
            col_start = norb_cumsum[jatm]
            col_range = np.linspace(col_start - n2, col_start - 1, n2, dtype=int)
            row_ind, col_ind = np.meshgrid(row_range, col_range, indexing='ij')
            pos_slice = slice(pos, pos+n1*n2)
            row[pos_slice] = row_ind.reshape(-1)
            col[pos_slice] = col_ind.reshape(-1)
            data[pos_slice] = self.mats[ipair].reshape(-1)
            pos += n1 * n2
            mats_R[R] = (row, col, data, pos)
        
        for R in mats_R.keys():
            row, col, data, pos = mats_R[R]
            assert pos == ndata_R[R]
            mats_R[R] = csr_matrix((data, (row, col)), shape=(norb_total, norb_total))
        
        self.matscsr_R = mats_R
        return self.matscsr_R
    
    def hermitianize(self):
        """
        Hermitianizes the matrix according to (H+H^\dagger)/2

        Returns:
            float: The total absolute difference between H and H^\dagger divided by the total number of matrix elements.
        """
        # todo: check lcaodata1==lcaodata2
        self.sort()
        translations_inv = -self.translations
        atom_pairs_inv = self.atom_pairs[:, np.array([1, 0])]
        # indices = self.get_indices()
        indices_inv = pairs_to_indices(self.structure, translations_inv, atom_pairs_inv)
        arg_inv_org = np.argsort(indices_inv)

        totale = 0.
        totaln = 0
        for ipair in range(self.npairs):
            this = self.mats[ipair]
            iother = arg_inv_org[ipair]
            if np.isrealobj(self.mats[ipair]):
                other = self.mats[iother].T # removed .copy()
            else:
                other = self.mats[iother].T.conj()
            self.mats[ipair] = (this + other) / 2.
            if ipair != iother:
                totale += np.sum(np.abs(this - other))
                totaln += np.prod(this.shape)
            else:
                # upper half only
                totale += np.sum(np.abs(this - other)) / 2.
                size = this.shape[0]
                totaln += (size - 1) * size // 2
        
        return totale / totaln


def pwc(structure, cutoffs, cutoffs2=None):
    '''
    find all valid translations and atom pairs using cutoffs.
    '''
    # This still needs to be optimized, since this is the bottleneck in overlap calculation.

    rprim = structure.rprim
    gprim = structure.gprim
    natom = structure.natom
    atom_numbers = structure.atomic_numbers
    atom_pos_cart = structure.atomic_positions_cart_uc
    if cutoffs2 is None:
        cutoffs2 = cutoffs

    maxr = max(max(cutoffs.values()), max(cutoffs2.values()))
    # unit cell "thickness" is 1/|G_i|
    max_n_extend = np.floor(2 * maxr * np.linalg.norm(gprim, axis=-1) + 1).astype(int) # [3]

    all_translations_2 = np.stack(np.meshgrid(np.arange(-max_n_extend[0], max_n_extend[0]+1, 1),
                                            np.arange(-max_n_extend[1], max_n_extend[1]+1, 1),
                                            np.arange(-max_n_extend[2], max_n_extend[2]+1, 1), indexing='ij'),
                                  axis=-1).reshape(-1, 3)
    all_trans_cart_2 = all_translations_2 @ rprim
    all_atoms_cart_2 = all_trans_cart_2[:, None, :] + atom_pos_cart[None, :, :] # [ntranslations_2, natom, 3]

    atom_names = atom_number2name(atom_numbers)
    distance_tol1 = np.array([cutoffs[atom_names[i]] for i in range(natom)]) # [natom]
    distance_tol2 = np.array([cutoffs2[atom_names[i]] for i in range(natom)]) 

    translations = []
    atom_pairs = []
    # from tqdm import tqdm
    # for iatom in tqdm(range(natom)):
    for iatom in range(natom):
        distance_tol = distance_tol1[iatom] + distance_tol2
        distance_tol = distance_tol[None, :, None]
        diff_cart = all_atoms_cart_2 - atom_pos_cart[iatom]
        # first select a square box
        within_box = np.where(np.all(np.abs(diff_cart) <= distance_tol, axis=2))
        translations_inbox = all_translations_2[within_box[0], :]
        atom_pairs_inbox = np.empty((translations_inbox.shape[0], 2), dtype='i8')
        atom_pairs_inbox[:, 0] = iatom
        atom_pairs_inbox[:, 1] = within_box[1]
        diff_cart = diff_cart[within_box[0], within_box[1], :]
        distance_tol = distance_tol[0, within_box[1], 0]
        # then select a ball within the box
        pair_distance = np.linalg.norm(diff_cart, axis=1)
        within_cutoff = pair_distance <= distance_tol
        translations.append(translations_inbox[within_cutoff, :])
        atom_pairs.append(atom_pairs_inbox[within_cutoff, :])
    
    translations = np.concatenate(translations, axis=0)
    atom_pairs = np.concatenate(atom_pairs, axis=0)
 
    translations = structure.trans_uc_to_original(translations, atom_pairs[:, 0], atom_pairs[:, 1])
    
    return PairsInfo(structure, translations, atom_pairs)
