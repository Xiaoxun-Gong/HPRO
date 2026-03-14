import numpy as np
from scipy.sparse import csr_matrix

from ..constants import hpro_rng
from .. import config as CFG
from ..utils.mpi import MPI, comm


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
        '''
        Make a slice of the current PairsInfo object. Please be careful that this is in-place action!
        '''
        self.translations = self.translations[val]
        self.atom_pairs = self.atom_pairs[val]
        self.npairs = self.translations.shape[0]
        self._indices = new_indices

    def remove_ji(self):
        """
        Removes atom pairs that are redundant upon change (i, j) -> (j, i). The one with smaller index is kept.
        """
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
        '''
        Randomly shuffle the atom pairs. This is for MPI distribution of pairs.
        '''
        argsort = np.arange(self.npairs)
        hpro_rng.shuffle(argsort)
        self.slice(argsort)
    
    def get_distances(self):
        '''
        Get the cartesian distances between all atom pairs.
        '''
        rprim = self.structure.rprim
        atom_pos_cart = self.structure.atomic_positions_cart
        cart1 = self.translations @ rprim + atom_pos_cart[self.atom_pairs[:, 1], :]
        cart2 = atom_pos_cart[self.atom_pairs[:, 0]]
        pair_distance = np.linalg.norm(cart1 - cart2, axis=-1)
        return pair_distance
    
    def findipairs(self, pairs):
        """
        Finds the indices of pairs in the current object's indices, provided all the pairs are in the current object.

        Parameters:
            pairs (PairsInfo): The PairsInfo object containing the pairs to find.

        Returns:
            listipair: The indices of the pairs in the current object's indices.
        """
        self.sort()
        indices = self.get_indices()
        assert np.all(np.isin(pairs.get_indices(), indices))
        listipair = np.searchsorted(indices, pairs.get_indices())
        return listipair

    def __add__(self, other):
        '''
        Combines two PairsInfo objects.
        '''
        assert self.__class__ is other.__class__
        assert self.structure == other.structure
        self.sort()
        other.sort()
        indices1 = self.get_indices()
        indices2 = other.get_indices()
        indices_all = np.union1d(indices1, indices2)
        return self.__class__.from_indices(self.structure, indices_all)

    def unfold_with_hermiticity(self):
        '''
        Unfold the pairs: generate pair (ji) from pair (ij).
        This is usually called within MatAO.unfold_with_hermiticity().
        '''
        translations_inv = -self.translations
        atom_pairs_inv = self.atom_pairs[:, np.array([1, 0])]
        indices = self.get_indices()
        indices_inv = pairs_to_indices(self.structure, translations_inv, atom_pairs_inv)
        not_redundant = ~np.isin(indices_inv, indices)

        translations_new = np.concatenate((self.translations, 
                                           translations_inv[not_redundant]), axis=0)
        atom_pairs_new = np.concatenate((self.atom_pairs,
                                         atom_pairs_inv[not_redundant]), axis=0)
        indices_new = np.concatenate((indices, indices_inv[not_redundant]))
        self.translations = translations_new
        self.atom_pairs = atom_pairs_new
        self.npairs = translations_new.shape[0]
        self._indices = indices_new

        return not_redundant
    

def pairs_to_indices(structure, translations, atom_pairs):
    '''
    This function is defined in such a way that sorting of pairs according to indices will always be performed first 
    according to atomic numbers, then iatom(1), then iatom(2), then translations(1), then translations(2), finally 
    according to translations(3).
    '''
    if len(translations) == 0: 
        return np.empty(0, dtype=translations.dtype)
    assert np.max(translations) < CFG.BASE_TRANSLATIONS // 2
    assert np.min(translations) >= -CFG.BASE_TRANSLATIONS // 2
    natom = structure.natom
    atom_numbers = structure.atomic_numbers
    atm1 = atom_pairs[:, 0]
    atm2 = atom_pairs[:, 1]
    spc1 = atom_numbers[atm1]
    spc2 = atom_numbers[atm2]
    rem = translations + CFG.BASE_TRANSLATIONS//2 # from [-BASE//2, BASE//2) to [0, BASE)
    return ((spc1 * 200 + spc2) * CFG.BASE_TRANSLATIONS**3 * natom**2 +
            (atm1 * natom + atm2) * CFG.BASE_TRANSLATIONS**3 +
            rem @ np.array([CFG.BASE_TRANSLATIONS**2, CFG.BASE_TRANSLATIONS, 1]))

def indices_to_pairs(natom, indices):
    '''
    This is the inverse of the function `paris_to_indices`.
    '''
    npairs = len(indices)
    translations = np.empty((npairs, 3), dtype=int)
    atom_pairs = np.empty((npairs, 2), dtype=int)
    quo, translations[:, 2] = np.divmod(indices, CFG.BASE_TRANSLATIONS)
    quo, translations[:, 1] = np.divmod(quo, CFG.BASE_TRANSLATIONS)
    quo, translations[:, 0] = np.divmod(quo, CFG.BASE_TRANSLATIONS)
    translations -= CFG.BASE_TRANSLATIONS//2 # from [0, BASE) to [-BASE//2, BASE//2)
    quo, atom_pairs[:, 1] = np.divmod(quo, natom)
    quo, atom_pairs[:, 0] = np.divmod(quo, natom)
    return translations, atom_pairs


class MatAO(PairsInfo):
    '''
    This is the class to store matrices in the LCAO basis. The matrices are separated by blocks, and each block corresponds
    to the matrices of one atom pair. The matrix blocks are stored as a list of arrays in the `mats` attribute. 

    Attributes:
        mats list[array]: The matrix blocks. Length of this list is the same as `self.npairs`.
        aodata1 (AOData): Information for basis functions on the left of the matrix.
        aodata2 (AOData, optional): Information for basis functions on the right of the matrix. Defaults to aodata1 if not provided.
    '''
    def __init__(self, structure, translations, atom_pairs, mats, aodata1, aodata2=None, spinful=False):
        super(MatAO, self).__init__(structure, translations, atom_pairs)
        self.mats = mats
        assert len(mats) == self.npairs
        self.aodata1 = aodata1
        self.aodata2 = aodata2 if aodata2 is not None else aodata1
        self.spinful = spinful
        assert self.aodata1.structure == self.aodata2.structure == self.structure
        self.matscsr_R = None
        self.norb_total = None

    def spinless_to_spinful(self):
        '''
        Convert self to a spinful matrix, if it is not already
        Fills the spin-diagonal part with original spinless matrices
        '''
        if self.spinful:
            return
        for ix_self in range(self.npairs):
            mat = self.mats[ix_self]
            self.mats[ix_self] = np.kron(np.eye(2), mat)
        self.spinful = True
        if self.norb_total is not None:
            self.norb_total *= 2

    @classmethod
    def from_pairs(cls, pairs, mats, aodata1, aodata2=None, spinful=False):
        '''
        This is useful when we want to initialize a MatAO object when we already have the PairsInfo object and the matrix blocks.
        '''
        return cls(pairs.structure, pairs.translations, pairs.atom_pairs, mats, aodata1, aodata2=aodata2, spinful=spinful)

    def slice(self, val, new_indices=None):
        super(MatAO, self).slice(val, new_indices=new_indices)
        self.mats = [self.mats[i] for i in np.r_[val]]
    
    def get_pairs(self):
        return PairsInfo(self.structure, self.translations, self.atom_pairs)
    
    def get_pairs_ij(self):
        pairs = self.get_pairs()
        pairs.remove_ji()
        return pairs
    
    def unfold_with_hermiticity(self):
        '''
        Unfold the matrix blocks, assuming that the whole LCAO matrix is Hermitian. Notice that diagonal blocks (i.e., i==j) are 
        not changed in this process.
        '''
        # todo: check aodata1==aodata2
        not_redundant = super(MatAO, self).unfold_with_hermiticity()
        assert len(not_redundant) == len(self.mats)
        for ipair in range(len(not_redundant)):
            if not_redundant[ipair]:
                if self.mats[ipair] is None:
                    self.mats.append(None)
                elif np.isrealobj(self.mats[ipair]):
                    self.mats.append(self.mats[ipair].T.copy())
                else:
                    self.mats.append(self.mats[ipair].T.conj())
    
    @classmethod
    def init_mats(cls, pairs, aodata1, aodata2=None, filling_value=0, dtype=np.complex128, spinful=False):
        """
        Initialize the matrix blocks for a MatAO object, with pairs and aodata already known.

        Parameters:
            pairs (PairsInfo): The pairs information.
            aodata1 (AOData): Information for basis functions on the left of the matrix.
            aodata2 (AOData, optional): Information for basis functions on the right of the matrix. Defaults to aodata1 if not provided.
            filling_value: The value to fill the matrix blocks with. Defaults to 0.
            dtype (numpy dtype, optional): The data type of the matrix blocks. Defaults to np.complex128.

        Returns:
            MatAO: A new MatAO object with the initialized matrix blocks.
        """
        if aodata2 is None:
            aodata2 = aodata1
        atom_nbrs = pairs.structure.atomic_numbers
        mats = []
        for ipair in range(pairs.npairs):
            if filling_value is not None:
                size1 = aodata1.norbfull_spc[atom_nbrs[pairs.atom_pairs[ipair, 0]]]
                size2 = aodata2.norbfull_spc[atom_nbrs[pairs.atom_pairs[ipair, 1]]]
                if spinful:
                    size1 *= 2
                    size2 *= 2
                mats.append(np.full((size1, size2), filling_value, dtype=dtype))
            else:
                mats.append(None)
        return cls.from_pairs(pairs, mats, aodata1, aodata2=aodata2, spinful=spinful)

    def mpi_gather(self, displ, dtype, root=0):
        '''
        Gather matrix blocks to the root processer. All processors should have the same pairs, except that on some processors 
        several matrix blocks can be empty or all zeros.

        Parameters:
            displ (list[int]): The displacements for each processor, i.e. the ith processor holds displ[i] to displ[i+1]-1 matrix blocks.
            dtype (MPI dtype): The data type of the matrix blocks.
        '''
        if comm is None:
            return
        assert comm.size < self.npairs
        curr_rank = 0
        for ipair in range(self.npairs):
            if ipair >= displ[curr_rank+1]: # does not work if size>npairs
                curr_rank += 1
            if comm.rank == root and curr_rank == root:
                # hoppings[ipair] = hoppings_thisproc[ipair]
                pass
            elif comm.rank == curr_rank:
                comm.Send([self.mats[ipair], dtype], dest=root, tag=22)
            elif comm.rank == 0:
                comm.Recv([self.mats[ipair], dtype], source=curr_rank, tag=22)

    def mpi_reduce(self, dtype=None, op=None, root=0):
        """
        Perform a MPI reduce operation on the matrix blocks across all processes.

        Parameters:
            dtype (MPI dtype, optional): The data type of the matrix blocks. Defaults to MPI.COMPLEX16.
            op (MPI operation, optional): The reduction operation to perform. Defaults to MPI.SUM.
        """
        assert comm is not None
        if dtype is None: dtype=MPI.COMPLEX16
        if op is None: op=MPI.SUM
        if comm is None:
            return
        for ipair in range(self.npairs):
            if comm.rank == 0:
                h_red = np.empty_like(self.mats[ipair])
            else:
                h_red = None
            comm.Reduce([self.mats[ipair], dtype],
                        [h_red, dtype], op=op, root=root)
            self.mats[ipair] = h_red
    
    def delete_mats(self):
        self.mats = [None for _ in range(self.npairs)]
    
    def __add__(self, other):
        return self._add_sub(other, isadd=True)
    
    def __sub__(self, other):
        return self._add_sub(other, isadd=False)

    def _add_sub(self, other, isadd=True):
        '''
        Addition or subtraction of two matrices. Returns a new MatAO object.
        '''
        assert self.__class__ is other.__class__
        assert self.structure == other.structure
        assert self.spinful == other.spinful
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
        return self.__class__(self.structure, translations_all, atom_pairs_all, mats_all, self.aodata1, aodata2=self.aodata2, spinful=self.spinful)

    def fillvalue(self, other):
        '''
        Fill the matrix blocks of self with the values of the matrix blocks of other.
        '''
        # todo: this can be optimized using super().findipairs()
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
        Notice that this does not create a MatAOCSR object.

        Returns:
        ---------
        matscsr_R: Dict[Tuple(int, int, int) -> scipy.sparse.csr.csr_matrix]
                   where R is the lattice translation vector.

        Use with care! If mats have been changed, use matao.matscsr_R=None to reset matscsr_R.
        '''
        # todo: create mataocsr object

        if (self.matscsr_R is not None) and (self.norb_total is not None):
            return self.matscsr_R

        site_norbits = np.zeros(self.structure.natom, dtype=int)
        for iatm in range(self.structure.natom):
            atm_nbr = self.structure.atomic_numbers[iatm]
            site_norbits[iatm] = self.aodata1.norbfull_spc[atm_nbr]
            if self.spinful:
                site_norbits[iatm] *= 2
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
            if self.spinful:
                data = np.empty(ndata, dtype='c16')
            else:
                data = np.empty(ndata, dtype='f8')
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
        # todo: check aodata1==aodata2
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
