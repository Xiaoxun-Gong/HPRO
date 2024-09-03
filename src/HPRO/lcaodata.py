import os
import re
import numpy as np
from scipy.sparse import coo_matrix
from .structure import Structure
from .constants import AOFT_QGRID_DEN
from .utils import atom_number2name
from .orbutils import parse_siesta_ion, parse_deeph_orbtyps, parse_gpaw_basis, read_upf, grid_R2G, GridFunc, LinearRGD


def argsort(seq):
    # https://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python
    return sorted(range(len(seq)), key=seq.__getitem__)

def findfile(folder, pattern):
    files = []
    for name in os.listdir(folder):
        if os.path.isfile(f'{folder}/{name}'):
            if re.match(pattern, name) is not None:
                files.append(name)
    if len(files) == 0:
        raise AssertionError(f'No valid file found under {folder} according to pattern {pattern}')
    if len(files) > 1:
        raise AssertionError(f'Multiple valid files found under {folder} according to pattern {pattern}')
    return f'{folder}/{files[0]}'


class LCAOData:
    '''
    Stores information about atomic orbital functions for several elements needed by the calculation.

    Attributes:
    ---------
    structure: structure.Structure, storing the structure information
    aocode: str
    ls_spc: Dict[int -> List[int]], list of angular momentum quantum numbers of orbitals
    phirgrids_spc: Dict[int -> List[GridFunc]], functions of each orbital
    orbslices_spc: Dict[int -> List[int]], the slices of each orbital if we concatenate all orbitals of different l's together
    norb_spc: Dict[int -> int], number of radial functions, not number of AOs. Number of aos is orbslices_spc[spc][norb_spc[spc]]
    cutoffs: Dict[int -> float], cutoff radius for each atomic species
    phiQlist_spc: Dict[int -> List[GridFunc]], list of Fourier transformed orbital functions (calculated by `calc_phiQ`)
    phiQEcut: float, cutoff energy for Fourier transform of orbitals
    '''
    
    def __init__(self, structure: Structure, cutoffs=None, basis_path_root='./', aocode='siesta'):
        '''
        Parameters:
        ---------
        cutoffs: Dict[str -> float], cutoff radius for each atomic species, in bohr.
        basis_path_root: str, folder containing x.ion
        '''
        
        self.structure = structure
        self.aocode = aocode
        
        self.ls_spc = {}
        self.phirgrids_spc = {}
        self.norb_spc = {}
        self.funch_spc = {}
        self.Qij_spc = {}
        
        spc_numbers = structure.atomic_species
        spc_names = atom_number2name(spc_numbers)
        for spc_nu, spc_na in zip(spc_numbers, spc_names):
            funch, Qij = None, None
            # if spc_nu not in self.orbitals_types:
            if aocode == 'siesta':
                norb, phirgrids = parse_siesta_ion(f'{basis_path_root}/{spc_na}.ion')
            elif aocode == 'gpaw':
                norb, phirgrids = parse_gpaw_basis(findfile(basis_path_root, f'^{spc_na}\..*\.basis$'))
            elif aocode == 'qe-projR':
                funch, phirgrids = read_upf(findfile(basis_path_root, f'^{spc_na}\.(upf|UPF)$'))
                norb = len(phirgrids)
            elif aocode == 'deeph':
                break # handle it later
            else:
                raise NotImplementedError(f'Interface to {aocode} not implemented')
            
            # In case the orbital angular momentum does not appear in order, 
            # we sort them to follow the convention of deeph
            if aocode != 'gpaw-projR':
                orbitals_argsort = argsort([orb.l for orb in phirgrids])
                # print(orbitals_argsort)
            else:
                orbitals_argsort = list(range(len(phirgrids)))
            
            self.phirgrids_spc[spc_nu] = [phirgrids[i] for i in orbitals_argsort]
            self.ls_spc[spc_nu] = [phirgrids[i].l for i in orbitals_argsort]
            self.norb_spc[spc_nu] = norb

            if funch is not None:
                self.funch_spc[spc_nu] = funch
            if Qij is not None:
                self.Qij_spc[spc_nu] = Qij
        
        if aocode == 'deeph':
            orbitals_types, stru_read = parse_deeph_orbtyps(basis_path_root)
            assert stru_read == structure
            orbitals_num = {}
            phirgrids_dummy_spc = {}
            for spc, orbital_types in orbitals_types.items():
                orbitals_num[spc] = len(orbital_types)
                phirgrids_dummy_spc[spc] = [GridFunc(None, None, l=l) for l in orbital_types]
            self.phirgrids_spc = phirgrids_dummy_spc
            self.ls_spc = orbitals_types
            self.norb_spc = orbitals_num

        orbslices_spc = {}
        norbfull_spc = {}
        for spc, orbital_types in self.ls_spc.items():
            orbital_slices = [0]
            for l in orbital_types:
                orbital_slices.append(orbital_slices[-1] + 2*l+1)
            orbslices_spc[spc] = orbital_slices
            norbfull_spc[spc] = orbital_slices[-1]
        self.orbslices_spc = orbslices_spc
        self.norbfull_spc = norbfull_spc
    
        if cutoffs is None and aocode!='deeph':
            cutoffs = {}
            for spc_nu, spc_na in zip(spc_numbers, spc_names):
                max_cutoff = max(phirgrid.rcut for phirgrid in self.phirgrids_spc[spc_nu])
                cutoffs[spc_na] = max_cutoff
    
        self.cutoffs = cutoffs
        self.phiQlist_spc = None
        self.phiQEcut = None
        
    def show_basis_cutoff(self):
        for species, phirgrids in self.phirgrids_spc.items():
            max_cutoff = max(phirgrid.rcut for phirgrid in phirgrids)
            print(species, max_cutoff)

    def norm_check(self):
        print('DEBUG: Check if the integral is 1:')
        for phirgrids in self.phirgrids_spc.values():
            for phirgrid in phirgrids:
                phi_r = phirgrid.func
                norm = phirgrid.rgd.sips(phi_r*phi_r)
                print(phirgrid.l, norm)
    
    def init_hamiltonians(self, use_hermiticity=False):
        raise DeprecationWarning()
        '''
        Parameters:
        ---------
        cutoffs: see pwc
        
        Returns:
        pairs: see pwc
        hoppings: List[array], the ith item is the hopping corresponding to ith atom pair
        '''
        
        atom_nbrs = self.structure.atomic_numbers
        orbslices_spc = self.orbslices_spc
        
        pairs = self.pwc()
                    
        hoppings = []
        for ipair in range(pairs.npairs):
            m = pairs.map_ijji[ipair]
            if use_hermiticity and m>=0: 
                hoppings.append(None)
            else:
                size1 = orbslices_spc[atom_nbrs[pairs.atom_pairs[ipair, 0]]][-1]
                size2 = orbslices_spc[atom_nbrs[pairs.atom_pairs[ipair, 1]]][-1]
                hoppings.append(np.zeros((size1, size2), dtype=np.complex128))
            
        return pairs, hoppings
    
    def calc_phiQ(self, Ecut):
        if self.phiQEcut is not None and self.phiQlist_spc is not None:
            if self.phiQEcut >= Ecut: return
        grid_nq = int(np.sqrt(Ecut) * AOFT_QGRID_DEN)
        gridQ = LinearRGD(0, np.sqrt(2*Ecut), grid_nq)
        phiQlist_spc = {}
        for spc in self.structure.atomic_species:
            phiQlist = []
            for iorb in range(self.norb_spc[spc]):
                phirgrid = self.phirgrids_spc[spc][iorb]
                phiQlist.append(grid_R2G(gridQ, phirgrid))
            phiQlist_spc[spc] = phiQlist
        self.phiQlist_spc = phiQlist_spc
        self.phiQEcut = Ecut
    
    def check_rstart(self):
        msg = 'All the radial grids for atomic orbitals must start with r=0'
        for spc, phirgrids in self.phirgrids_spc.items():
            for phirgrid in phirgrids:
                assert phirgrid.rgd.rstart == 0, msg
    
    def H_r2k(self, k, pairs, hoppings, sparse=True):
        raise DeprecationWarning()
        '''
        Add up all hopping with same atom pairs but different periodic translations
        according to the rule:
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
        
        site_norbits = np.zeros(self.structure.natom, dtype=int)
        for iatm in range(self.structure.natom):
            atm_nbr = self.structure.atomic_numbers[iatm]
            site_norbits[iatm] = self.orbslices_spc[atm_nbr][-1]
        norb_cumsum = np.cumsum(site_norbits)
        norb_total = norb_cumsum[-1]
        
        H_ij_k = {}
        
        ndata = 0
        for ipair in range(pairs.npairs):
            i_j = tuple(pairs.atom_pairs[ipair])
            if i_j not in H_ij_k:
                n1, n2 = site_norbits[np.array(i_j)]
                H_ij_k[i_j] = np.zeros((n1, n2), dtype='c16')
                ndata += n1 * n2
            trans = pairs.translations[ipair]
            H_ij_k[i_j] += hoppings[ipair] * np.exp(2.*np.pi*1j * np.dot(k, trans))
        
        if sparse:
            row = np.empty(ndata, dtype=int)
            col = np.empty(ndata, dtype=int)
            data = np.empty(ndata, dtype='c16')
            pos = 0
            for i_j, hblock in H_ij_k.items():
                iatm, jatm = i_j
                n1, n2 = site_norbits[np.array(i_j)]
                row_start = norb_cumsum[iatm]
                row_range = np.linspace(row_start - n1, row_start - 1, n1, dtype=int)
                col_start = norb_cumsum[jatm]
                col_range = np.linspace(col_start - n2, col_start - 1, n2, dtype=int)
                row_ind, col_ind = np.meshgrid(row_range, col_range, indexing='ij')
                pos_slice = slice(pos, pos+n1*n2)
                row[pos_slice] = row_ind.reshape(-1)
                col[pos_slice] = col_ind.reshape(-1)
                data[pos_slice] = hblock.reshape(-1)
                pos += n1 * n2
            
            hmat = coo_matrix((data, (row, col)), shape=(norb_total, norb_total))
        
        else:
            hmat = np.zeros((norb_total, norb_total), dtype='c16')
            for i_j, hblock in H_ij_k.items():
                iatm, jatm = i_j
                n1, n2 = site_norbits[np.array(i_j)]
                row_start = norb_cumsum[iatm]
                row_slice = slice(row_start - n1, row_start)
                col_start = norb_cumsum[jatm]
                col_slice = slice(col_start - n2, col_start)
                hmat[row_slice, col_slice] = hblock
            
        return hmat

    def echo_info(self):
        print(f'Format: {self.aocode.split("-")[0]}')
        for spc in self.structure.atomic_species:
            name, = atom_number2name([spc])
            print(f'Element {name}:')
            for iorb in range(self.norb_spc[spc]):
                l = self.ls_spc[spc][iorb]
                phirgrid = self.phirgrids_spc[spc][iorb]
                rcut = phirgrid.rcut
                phi_r = phirgrid.func
                norm = phirgrid.rgd.sips(phi_r*phi_r)
                print(f'Orbital {iorb+1}: l = {l}, cutoff = {rcut:6.3f} a.u., norm = {norm:6.3f}')


def calc_FT_kg_orb_spcs(ng, kgcart, lcaodata, Ecut):
    stru = lcaodata.structure
    lcaodata.calc_phiQ(Ecut)
    FT_kg_orb_spcs = {}
    for spc in stru.atomic_species:
        nradial = lcaodata.norb_spc[spc]
        orbslices = lcaodata.orbslices_spc[spc]
        nao = orbslices[nradial]
        FT_kg_orb = np.empty((ng, nao), dtype=np.complex128)
        for iorb in range(nradial):
            slice_orb = slice(orbslices[iorb], orbslices[iorb+1])
            orbQ = lcaodata.phiQlist_spc[spc][iorb]
            FT_kg_orb[:, slice_orb] = orbQ.generate3D(kgcart) * (-1j)**orbQ.l # todo: make it real
        FT_kg_orb_spcs[spc] = FT_kg_orb
    return FT_kg_orb_spcs
