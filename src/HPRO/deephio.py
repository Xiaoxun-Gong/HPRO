import os
import json
import numpy as np
import h5py
from scipy.linalg import block_diag

from .constants import bohr2ang, hartree2ev
from .structure import Structure
from .matlcao import MatLCAO
from .lcaodata import LCAOData # this might be unsafe?
from .utils import slice_same

'''
This module implemements several functions for reading and writing files in deeph format
'''

def save_structure_deeph(structure, savedir):

    os.makedirs(savedir, exist_ok=True)

    rprim = structure.rprim
    gprim = structure.gprim
    atom_nbrs = structure.atomic_numbers
    atm_pos_cart = structure.atomic_positions_cart
    efermi = structure.efermi
    
    np.savetxt(f'{savedir}/lat.dat', rprim.T * bohr2ang)
    np.savetxt(f'{savedir}/element.dat', atom_nbrs, fmt='%-3i')
    np.savetxt(f'{savedir}/site_positions.dat', atm_pos_cart.T * bohr2ang)
    np.savetxt(f'{savedir}/rlat.dat', 2*np.pi*gprim.T / bohr2ang)

    info = {"isspinful": False, "fermi_level": efermi * hartree2ev}
    with open(f'{savedir}/info.json', 'w') as f:
        json.dump(info, f)

Us_openmx2wiki = {
    0: np.eye(1),
    1: np.eye(3)[[1, 2, 0]],
    2: np.eye(5)[[2, 4, 0, 3, 1]],
    3: np.eye(7)[[6, 4, 2, 0, 1, 3, 5]]
}

def get_Us_openmx2wiki(ls_spc):
    '''
    DeepH follows the OpenMX definition of spherical harmonics, but this software follows Wikipedia's convention.
    So we need to convert them.
    '''
    orbitals_Us_openmx2wiki = {}
    for spc, orbital_types in ls_spc.items():
        U2deeph = [Us_openmx2wiki[l] for l in orbital_types]
        orbitals_Us_openmx2wiki[spc] = block_diag(*U2deeph)
    return orbitals_Us_openmx2wiki

def save_mat_deeph(savedir, matlcao, filename='hamiltonians.h5', energy_unit=True):

    lcaodata = matlcao.lcaodata1
    # todo: check lcaodata1 == lcaodata2

    os.makedirs(savedir, exist_ok=True)

    atom_nbrs = lcaodata.structure.atomic_numbers
    ls_spc = lcaodata.ls_spc

    with open(f'{savedir}/orbital_types.dat', 'w') as f:
        for nspc in atom_nbrs:
            f.write(' '.join(map(str, ls_spc[nspc])))
            f.write('\n')
    
    # here real spherical harmonics follow wikipedia convention, need to convert to openmx convension
    
    orbitals_Us_openmx2wiki = get_Us_openmx2wiki(ls_spc)

    h5file = h5py.File(f'{savedir}/{filename}', 'w', libver='latest')

    # from tqdm import tqdm
    # for ipair in tqdm(range(matlcao.npairs)):
    for ipair in range(matlcao.npairs):
        spc1 = atom_nbrs[matlcao.atom_pairs[ipair, 0]]
        spc2 = atom_nbrs[matlcao.atom_pairs[ipair, 1]]
        key = matlcao.get_keystr(ipair)
        mat = orbitals_Us_openmx2wiki[spc1].T @ matlcao.mats[ipair] @ orbitals_Us_openmx2wiki[spc2]
        if energy_unit:
            mat *= hartree2ev
        h5file[key] = mat
    
    h5file.close()

def get_mat0(ao_data, funch=None):
    for ih in range(len(funch)):
        h = funch[ih]
        if not np.isrealobj(h):
            # Future: D is complex
            assert np.max(np.abs(h.imag)) < 1e-8
            funch[ih] = h.real
    ao_data.sort_atom1()
    translations = ao_data.translations
    atom_pairs = ao_data.atom_pairs
    trans, atoms, mats = [], [], []
    
    slice_jatm = slice_same(atom_pairs[:, 0])
    njatm = len(slice_jatm) - 1
    for ix_atm in range(njatm):
        startj = slice_jatm[ix_atm]
        endj = slice_jatm[ix_atm + 1]
        atomj = atom_pairs[startj, 0]
        ix_js, ix_jps = np.tril_indices(endj - startj)
        ix_js += startj; ix_jps += startj
        trans.append(translations[ix_jps] - translations[ix_js])
        atoms.append(np.stack((atom_pairs[ix_js, 1], atom_pairs[ix_jps, 1]), axis=1))
        for ix_j, ix_jp in zip(ix_js, ix_jps):
            mat = ao_data.mats[ix_j]
            matp = ao_data.mats[ix_jp]
            h = funch[atomj]
            mats.append(mat.T @ h @ matp)
    trans = np.concatenate(trans, axis=0)
    atoms = np.concatenate(atoms, axis=0)

    return trans, atoms, mats


def load_deeph_HS(folder, filename, energy_unit=True):
    stru = Structure.from_deeph(folder)
    lcaodata = LCAOData(stru, None, basis_path_root=folder, aocode='deeph')

    orbitals_Us_openmx2wiki = get_Us_openmx2wiki(lcaodata.ls_spc)
    
    hoppings = []
    translations = []
    atom_pairs = []
    npairs = 0
    with h5py.File(f'{folder}/{filename}') as f:
        for k, v in f.items():
            npairs += 1
            Rijab = eval(k)
            translations.append(Rijab[:3])
            atom_pairs.append(np.array(Rijab[3:5]) - 1)

            spc1 = stru.atomic_numbers[atom_pairs[-1][0]]
            spc2 = stru.atomic_numbers[atom_pairs[-1][1]]
            hmat = np.array(v)
            hmat = orbitals_Us_openmx2wiki[spc1] @ hmat @ orbitals_Us_openmx2wiki[spc2].T
            if energy_unit: hmat /= hartree2ev
            hoppings.append(hmat)
    translations = np.array(translations)
    atom_pairs = np.array(atom_pairs)
    npairs = len(atom_pairs)
    return MatLCAO(stru, translations, atom_pairs, hoppings, lcaodata)

def analyze_hdecay_deeph():
    raise NotImplementedError()