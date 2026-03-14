import numpy as np

from ..constants import bohr2ang
from .misc import atom_number2name


class Structure:
    '''
    Attributes:
    ---------
    rprim(3, 3): R_ij is the jth cartesian component of ith lattice vector, in bohr
    gprim(3, 3): G_ij is the jth cartesian component of ith primitive reciprocal lattice vector (NOT SCALED BY 2pi)
    atomic_species: (nspc), unique atomic numbers
    nspc: int, number of species
    natom: int, number of atoms
    
    atomic_positions_red(natom, 3): fractional coordinates
    atomic_positions_cart(natom, 3): in bohr
    cell_volume: in bohr^3
    
    trans_from_cuc(natom, 3): how to translate the atoms from conventional unit cell (i.e. [0, 1)) 
                              to its original position
    '''
    
    def __init__(self, rprim, atomic_numbers, atomic_positions, efermi=0.0, atomic_positions_is_cart=False):
        '''
        Parameters
        ---------
        rprim(3, 3): R_ij is the jth cartesian component of ith lattice vector, in bohr
        atomic_numbers: (natom) atomic numbers
        atomic_positions_red(natom, 3)
        efermi: in hartree
        '''
        
        rprim = np.array(rprim)
        self.rprim = rprim
        self.gprim = np.linalg.inv(rprim.T)
        
        atomic_numbers = np.array(atomic_numbers, dtype=int)
        self.atomic_numbers = atomic_numbers
        self.atomic_species = np.sort(np.unique(atomic_numbers))
        self.natom = len(atomic_numbers)
        self.nspc = len(self.atomic_species)
        
        assert len(atomic_positions) == self.natom
        if not atomic_positions_is_cart:
            self.atomic_positions_red = np.array(atomic_positions)
            self.atomic_positions_cart = self.atomic_positions_red @ self.rprim
        else:
            self.atomic_positions_cart = np.array(atomic_positions)
            self.atomic_positions_red = self.atomic_positions_cart @ self.gprim.T
        
        trans_to_uc, self.atomic_positions_red_uc = np.divmod(self.atomic_positions_red, 1)
        self.trans_from_cuc = trans_to_uc.astype(int)
        self.atomic_positions_cart_uc = self.atomic_positions_red_uc @ self.rprim
        
        self.cell_volume = np.abs(np.linalg.det(rprim))
        
        self.efermi = efermi  
    
    def trans_cuc_to_original(self, translation, iatom1, iatom2):
        '''
        This function takes translation between atoms in conventional unit cells, and converts the translations
        to between the original positions of the atoms (see notes under supercell.py)
        '''
        return translation + self.trans_from_cuc[iatom1, :] - self.trans_from_cuc[iatom2, :]
        
    def __eq__(self, other):
        return self.issame(other)
        
    def issame(self, other, eps=1e-6):
        assert other.__class__ == self.__class__
        if not np.all(np.abs(self.rprim-other.rprim) < eps): return False
        if not np.all(self.atomic_numbers==other.atomic_numbers): return False
        if not np.all(np.abs(self.atomic_positions_cart_uc-other.atomic_positions_cart_uc) < eps): return False
        return True
    
    def echo_info(self):
        print('Structure information:')
        print('Primitive lattice vectors (angstrom):')
        print('a = (' + ' '.join(f'{self.rprim[0, i]*bohr2ang:12.7f}' for i in range(3)) + ')')
        print('b = (' + ' '.join(f'{self.rprim[1, i]*bohr2ang:12.7f}' for i in range(3)) + ')')
        print('c = (' + ' '.join(f'{self.rprim[2, i]*bohr2ang:12.7f}' for i in range(3)) + ')')
        print('Atomic species and numbers in unit cell:', end='')
        atom_names = atom_number2name(self.atomic_species)
        for ispc in range(self.nspc):
            name = atom_names[ispc]
            number = self.atomic_species[ispc]
            print(f' {name}: {np.sum(number==self.atomic_numbers)}', end='')
            print('.' if ispc==self.nspc-1 else ';', end='')
        print()
