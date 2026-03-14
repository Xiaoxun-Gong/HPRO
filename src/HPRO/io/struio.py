import json
import os
import xml.etree.ElementTree as ET

import numpy as np

from ..constants import bohr2ang, hartree2ev
from .. import config as CFG
from ..utils.structure import Structure
from ..utils.misc import atom_name2number, atom_number2name, unique_nosort
from .bgwio import bgw_wfn, bgw_vsc, bgw_vkb, bgw_filetype

def load_structure(path, interface):
    interface = interface.lower()
    if interface == 'qe':
        if path.endswith('.xml'):
            stru = from_qexml(path)
        elif os.path.isdir(path):
            stru = from_qexml(f'{path}/data-file-schema.xml')
        else:
            raise ValueError(f'Invalid QE structure path: {path}')
    elif interface == 'bgw':
        stru = from_bgw(path)
    elif interface == 'vasp':
        with open(path) as f:
            stru = from_poscar(f)
    elif interface == 'gpaw':
        stru = from_gpaw(path)
    elif interface == 'deeph':
        stru = from_deeph(path)
    else:
        raise NotImplementedError(f'Unknown structure interface: {interface}')
    return stru

def from_qexml(qexml_path):
    qeroot = ET.parse(qexml_path).getroot()
    cell_elem = qeroot.find('output').find('atomic_structure').find('cell')
    rprim = np.array([list(map(float, cell_elem[i].text.split())) for i in range(3)])
    atomic_position_elem = qeroot.find('output').find('atomic_structure').find('atomic_positions')
    atomic_positions_cart = np.array([list(map(float, atom_elem.text.split())) for atom_elem in atomic_position_elem.findall('atom')])
    atomic_names = [atom_elem.attrib['name'] for atom_elem in atomic_position_elem.findall('atom')]
    atomic_numbers = np.array(atom_name2number(atomic_names))
    band_structure_elem = qeroot.find('output').find('band_structure')
    efermi = float(band_structure_elem.find('fermi_energy').text)
    return Structure(rprim, atomic_numbers, atomic_positions_cart, efermi, atomic_positions_is_cart=True)

def from_bgw(filepath):
    filetype = bgw_filetype(filepath).split('-')[0]
    # filetype = filetype.upper()
    if filetype not in ['WFN', 'VSC', 'VKB']:
        raise NotImplementedError(f'Interface to {filetype} not implemented')
    if filetype == 'WFN':
        obj = bgw_wfn(filepath)
    elif filetype == 'VSC':
        obj = bgw_vsc(filepath)
    elif filetype == 'VKB':
        obj = bgw_vkb(filepath)
    obj.read_header()
    rprim = obj.alat * obj.at
    atomic_numbers = obj.atomic_number
    atomic_positions = obj.tau * obj.alat
    efermi = 0 # todo
    obj.close()
    return Structure(rprim, atomic_numbers, atomic_positions, efermi, atomic_positions_is_cart=True)

def from_poscar(fp, return_spc_in_order=False):
    fp.readline()
    alat = float(fp.readline().strip()) / bohr2ang
    rprim = np.empty((3, 3))
    for i in range(3):
        rprim[i] = list(map(float, fp.readline().split()))
    rprim *= alat
    spc_names = fp.readline().split()
    spc_names = [name.split("/")[0].split("_")[0] for name in spc_names] # handle names like K_sv or K/xxxx in some versions of vasp
    spc_nbrs = atom_name2number(spc_names)
    num_spcs = list(map(int, fp.readline().split()))
    atomic_numbers = []
    for spc, num in zip(spc_nbrs, num_spcs):
        atomic_numbers.extend(spc for _ in range(num))
    kind = fp.readline().strip().lower()
    if kind[0] == 'c':
        is_cart = True
    elif kind[0] == 'd':
        is_cart = False
    atm_pos = np.empty((len(atomic_numbers), 3))
    for iatm in range(len(atomic_numbers)):
        # avoid case like '0 0 0 C'  --ZXL
        atm_pos[iatm] = list(map(float, fp.readline().split()[0:3]))
    if is_cart:
        atm_pos /= bohr2ang
    newobj = Structure(rprim, atomic_numbers, atm_pos, 0, atomic_positions_is_cart=is_cart)
    if return_spc_in_order:
        return newobj, spc_nbrs
    else:
        return newobj

def from_gpaw(gpawsave):
    from ase.io import ulm
    with ulm.open(gpawsave) as f:
        rprim = np.array(f.atoms.cell) / bohr2ang
        atm_nbrs = np.array(f.atoms.numbers)
        pos_cart = np.array(f.atoms.positions) / bohr2ang
        efermi = f.wave_functions.fermi_levels[0] / hartree2ev # todo: why is this an array?
    return Structure(rprim, atm_nbrs, pos_cart, efermi, atomic_positions_is_cart=True)

def from_deeph(deephsave):
    if CFG.DEEPH_USE_NEW_INTERFACE:
        with open(f'{deephsave}/POSCAR', 'r') as f:
            stru = from_poscar(f)
        if os.path.isfile(f'{deephsave}/info.json'):
            with open(f'{deephsave}/info.json') as f:
                info = json.load(f)
                efermi = info.get('fermi_energy_eV', 0.) / hartree2ev
            stru.efermi = efermi
        return stru
    else:
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
        return Structure(rprim, elements, atom_pos_cart, efermi, atomic_positions_is_cart=True)

def from_siesta(siestasave, sysname='siesta'):
    with open(f'{siestasave}/{sysname}.STRUCT_OUT', 'r') as fp:
        rprim = np.empty((3, 3))
        for i in range(3):
            rprim[i] = list(map(float, fp.readline().split()))
        rprim /= bohr2ang
        natm = int(fp.readline())
        atm_nbrs = np.empty(natm, dtype='i8')
        atm_pos_frac = np.empty((natm, 3))
        for iatm in range(natm):
            line = fp.readline()
            line_sp = line.split()
            atm_nbrs[iatm] = int(line_sp[1])
            atm_pos_frac[iatm, :] = list(map(float, line_sp[2:5]))
    return Structure(rprim, atm_nbrs, atm_pos_frac, 0.0, atomic_positions_is_cart=False)

def sort_atoms(stru):
    # sort atoms by their occurances in atomic_numbers
    species_uniq = unique_nosort(stru.atomic_numbers)
    iatm_argsort = np.empty(stru.natom, dtype=np.int64)
    for spc in species_uniq:
        eq = stru.atomic_numbers==spc
        iatm_argsort[eq] = np.where(eq)[0]
    return species_uniq, iatm_argsort

_fmt = "22.16f" # "14.8f"
def to_poscar(stru, file):
    '''
    In order to comply with VASP standard, we sort atoms by their occurances in atomic numbers list.
    In this way, ordering of atoms in the output POSCAR can be same as the input POSCAR.
    However, if not using POSCAR as structure input, the atom ordering in the generated POSCAR will 
    in general be different to that stored in the code.
    '''
    with open(file, 'w') as f:
        f.write('POSCAR generated by HPRO\n')
        f.write('1.0\n')
        rprimang = stru.rprim * bohr2ang
        f.write(f'{rprimang[0, 0]:{_fmt}} {rprimang[0, 1]:{_fmt}} {rprimang[0, 2]:{_fmt}}\n')
        f.write(f'{rprimang[1, 0]:{_fmt}} {rprimang[1, 1]:{_fmt}} {rprimang[1, 2]:{_fmt}}\n')
        f.write(f'{rprimang[2, 0]:{_fmt}} {rprimang[2, 1]:{_fmt}} {rprimang[2, 2]:{_fmt}}\n')
        species_uniq, iatm_argsort = sort_atoms(stru)
        f.write(' '.join(atom_number2name(species_uniq)) + '\n')
        f.write(' '.join(str(np.sum(stru.atomic_numbers==spc)) for spc in species_uniq) + '\n')
        f.write('Direct\n')
        redpos = stru.atomic_positions_red[iatm_argsort]
        for iatm in range(stru.natom):
            f.write(f'{redpos[iatm, 0]:{_fmt}} {redpos[iatm, 1]:{_fmt}} {redpos[iatm, 2]:{_fmt}}\n')