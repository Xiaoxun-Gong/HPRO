import json
import os
import numpy as np
import xml.etree.ElementTree as ET

from .constants import bohr2ang, hartree2ev
from .utils import atom_name2number, atom_number2name
from .bgwio import bgw_vsc, bgw_filetype

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
    
    _trans_to_uc(natom, 3): how to translate the atoms to [0, 1) unit cell
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
        self._trans_to_uc = trans_to_uc.astype(int)
        self.atomic_positions_cart_uc = self.atomic_positions_red_uc @ self.rprim
        
        self.cell_volume = np.abs(np.linalg.det(rprim))
        
        self.efermi = efermi  
    
    def trans_uc_to_original(self, translation, iatom1, iatom2):
        return translation + self._trans_to_uc[iatom1, :] - self._trans_to_uc[iatom2, :]
    
    @classmethod
    def from_qexml(cls, qexml_path):
        qeroot = ET.parse(qexml_path).getroot()
        cell_elem = qeroot.find('output').find('atomic_structure').find('cell')
        rprim = np.array([list(map(float, cell_elem[i].text.split())) for i in range(3)])
        atomic_position_elem = qeroot.find('output').find('atomic_structure').find('atomic_positions')
        atomic_positions_cart = np.array([list(map(float, atom_elem.text.split())) for atom_elem in atomic_position_elem.findall('atom')])
        atomic_names = [atom_elem.attrib['name'] for atom_elem in atomic_position_elem.findall('atom')]
        atomic_numbers = np.array(atom_name2number(atomic_names))
        band_structure_elem = qeroot.find('output').find('band_structure')
        efermi = float(band_structure_elem.find('fermi_energy').text)
        return cls(rprim, atomic_numbers, atomic_positions_cart, efermi, atomic_positions_is_cart=True)
    
    @classmethod
    def from_bgw(cls, filepath):
        filetype = bgw_filetype(filepath).split('-')[0]
        # filetype = filetype.upper()
        if filetype not in ['VSC']:
            raise NotImplementedError(f'Interface to {filetype} not implemented')
        elif filetype == 'VSC':
            obj = bgw_vsc(filepath)
        obj.read_header()
        rprim = obj.alat * obj.at
        atomic_numbers = obj.atomic_number
        atomic_positions = obj.tau * obj.alat
        efermi = 0 # todo
        obj.close()
        return cls(rprim, atomic_numbers, atomic_positions, efermi, atomic_positions_is_cart=True)
    
    @classmethod
    def from_deeph(cls, deephsave):
        rprim = np.loadtxt(f'{deephsave}/lat.dat').T / bohr2ang
        atom_pos_cart = np.loadtxt(f'{deephsave}/site_positions.dat').T / bohr2ang
        elements = np.loadtxt(f'{deephsave}/element.dat')
        with open(f'{deephsave}/info.json') as f:
            info = json.load(f)
            if 'fermi_level' in info:
                efermi = info['fermi_level'] / hartree2ev
            else:
                efermi = 0.
        if len(elements.shape) == 0:
            elements = elements[None]
            atom_pos_cart = atom_pos_cart[None, :]
        return cls(rprim, elements, atom_pos_cart, efermi, atomic_positions_is_cart=True)
        
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

def load_structure(path, interface):
    interface = interface.lower()
    if interface == 'qe':
        if path.endswith('.xml'):
            stru = Structure.from_qexml(path)
        elif os.path.isdir(path):
            stru = Structure.from_qexml(f'{path}/data-file-schema.xml')
        else:
            raise ValueError(f'Invalid QE structure path: {path}')
    elif interface == 'bgw':
        stru = Structure.from_bgw(path)
    elif interface == 'vasp':
        with open(path) as f:
            stru = Structure.from_poscar(f)
    elif interface == 'gpaw':
        stru = Structure.from_gpaw(path)
    elif interface == 'deeph':
        stru = Structure.from_deeph(path)
    else:
        raise NotImplementedError(f'Unknown structure interface: {interface}')
    return stru
