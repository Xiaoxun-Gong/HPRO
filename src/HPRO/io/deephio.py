import os
import json
import numpy as np
import h5py
from scipy.linalg import block_diag

from ..constants import bohr2ang, hartree2ev
from .. import config as CFG
from ..utils.misc import atom_number2name
from ..matao.matao import MatAO
from .aodata import AOData
from .struio import from_deeph, sort_atoms, to_poscar

'''
This module implemements several functions for reading and writing files in deeph format
'''

def save_structure_deeph(structure, savedir, spinful=False):
    os.makedirs(savedir, exist_ok=True)

    if CFG.DEEPH_USE_NEW_INTERFACE:
        to_poscar(structure, f'{savedir}/POSCAR')
    else:
        rprim = structure.rprim
        gprim = structure.gprim
        atom_nbrs = structure.atomic_numbers
        atm_pos_cart = structure.atomic_positions_cart
        efermi = structure.efermi
        
        np.savetxt(f'{savedir}/lat.dat', rprim.T * bohr2ang)
        np.savetxt(f'{savedir}/element.dat', atom_nbrs, fmt='%-3i')
        np.savetxt(f'{savedir}/site_positions.dat', atm_pos_cart.T * bohr2ang)
        np.savetxt(f'{savedir}/rlat.dat', 2*np.pi*gprim.T / bohr2ang)

        info = {"isspinful": spinful, "fermi_level": efermi * hartree2ev}
        with open(f'{savedir}/info.json', 'w') as f:
            json.dump(info, f)

Us_openmx2wiki = {
    0: np.eye(1),
    1: np.eye(3)[[1, 2, 0]],
    2: np.eye(5)[[2, 4, 0, 3, 1]],
    3: np.eye(7)[[6, 4, 2, 0, 1, 3, 5]]
}

def parse_hs_filetype(filetype, fname_override=None):
    filetype = filetype.lower()
    if filetype[0] == 'h':
        if CFG.DEEPH_USE_NEW_INTERFACE:
            filename = 'hamiltonian.h5'
        else:
            filename = 'hamiltonians.h5'
        energy_unit = True
    elif filetype[0] == 'o':
        if CFG.DEEPH_USE_NEW_INTERFACE:
            filename = 'overlap.h5'
        else:
            filename = 'overlaps.h5'
        energy_unit = False
    else:
        raise NotImplementedError(filetype)
        
    if fname_override is not None:
        filename = fname_override

    return filename, energy_unit

def get_Us_openmx2wiki(ls_spc):
    orbitals_Us_openmx2wiki = {}
    for spc, orbital_types in ls_spc.items():
        U2deeph = [Us_openmx2wiki[l] for l in orbital_types]
        orbitals_Us_openmx2wiki[spc] = block_diag(*U2deeph)
    return orbitals_Us_openmx2wiki

def save_mat_deeph(savedir, matao, filetype, fname_override=None):
    filename, energy_unit = parse_hs_filetype(filetype, fname_override=fname_override)
    os.makedirs(savedir, exist_ok=True)

    if CFG.DEEPH_USE_NEW_INTERFACE:
        stru = matao.structure
        aodata = matao.aodata1

        # save info dict
        info_dict = {}
        info_dict['atoms_quantity'] = stru.natom
        info_dict['orbits_quantity'] = sum(aodata.norbfull_spc[spc] for spc in stru.atomic_numbers)
        info_dict['orthogonal_basis'] = False
        info_dict['spinful'] = matao.spinful
        info_dict['fermi_energy_eV'] = stru.efermi * hartree2ev
        info_dict['elements_orbital_map'] = {atom_number2name([number])[0]: ls
                                            for number, ls in aodata.ls_spc.items()}

        with open(f'{savedir}/info.json', 'w') as f:
            json.dump(info_dict, f)

        # POSCAR requires all atoms with same kind are next to each other
        # This might not be satisfied in the original structure
        # so we need to map old atom index to new atom index
        _, iatm_argsort = sort_atoms(stru)
        mapatm = np.argsort(iatm_argsort)

        # save matrix
        atom_pairs = np.concatenate((matao.translations, mapatm[matao.atom_pairs]),
                                     axis=1, dtype='i8')
        displ = np.empty(matao.npairs+1, dtype='i8')
        shapes = np.empty((matao.npairs, 2), dtype='i8')
        displ[0] = 0
        flatmat = []
        for ipair in range(matao.npairs):
            mat = matao.mats[ipair]
            if energy_unit: mat = mat * hartree2ev
            displ[ipair+1] = displ[ipair] + mat.size
            shapes[ipair, :] = mat.shape
            flatmat.append(mat.reshape(-1))
        flatmat = np.concatenate(flatmat)

        with h5py.File(f'{savedir}/{filename}', 'w') as f:
            f.create_dataset('atom_pairs', data=atom_pairs)
            f.create_dataset('chunk_boundaries', data=displ)
            f.create_dataset('chunk_shapes', data=shapes)
            f.create_dataset('entries', data=flatmat)

    else:
        if matao.spinful:
            raise NotImplementedError("Saving spinful matrices in legacy deeph format is not supported yet")

        aodata = matao.aodata1
        # todo: check aodata1 == aodata2

        atom_nbrs = aodata.structure.atomic_numbers
        ls_spc = aodata.ls_spc

        with open(f'{savedir}/orbital_types.dat', 'w') as f:
            for nspc in atom_nbrs:
                f.write(' '.join(map(str, ls_spc[nspc])))
                f.write('\n')
        
        # here real spherical harmonics follow wikipedia convention, need to convert to openmx convension
        
        orbitals_Us_openmx2wiki = get_Us_openmx2wiki(ls_spc)

        h5file = h5py.File(f'{savedir}/{filename}', 'w', libver='latest')

        # from tqdm import tqdm
        # for ipair in tqdm(range(matao.npairs)):
        for ipair in range(matao.npairs):
            spc1 = atom_nbrs[matao.atom_pairs[ipair, 0]]
            spc2 = atom_nbrs[matao.atom_pairs[ipair, 1]]
            key = matao.get_keystr(ipair)
            mat = orbitals_Us_openmx2wiki[spc1].T @ matao.mats[ipair] @ orbitals_Us_openmx2wiki[spc2]
            if energy_unit:
                mat *= hartree2ev
            h5file[key] = mat
        
        h5file.close()

def load_mat_deeph(folder, filetype, fname_override=None):
    filename, energy_unit = parse_hs_filetype(filetype, fname_override=fname_override)

    if CFG.DEEPH_USE_NEW_INTERFACE:
        stru = from_deeph(folder)
        aodata = AOData(stru, None, basis_path_root=folder, aocode='deeph')

        with open(f'{folder}/info.json', 'r') as f:
            spinful = json.load(f)['spinful']

        with h5py.File(f'{folder}/{filename}') as f:
            atom_pairs = f['atom_pairs'][()]
            displ = f['chunk_boundaries'][()]
            shapes = f['chunk_shapes'][()]
            flatmat = f['entries'][()]
        flatmat = flatmat / hartree2ev if energy_unit else flatmat
        npairs = atom_pairs.shape[0]
        mats = []
        for ipair in range(npairs):
            mats.append(flatmat[displ[ipair]:displ[ipair+1]].reshape(shapes[ipair, :]))
        
        return MatAO(stru, atom_pairs[:, 0:3], atom_pairs[:, 3:5], mats, aodata, spinful=spinful)
    else:
        with open(f'{folder}/info.json', 'r') as f:
            isspinful = json.load(f)['isspinful']
        if isspinful:
            raise NotImplementedError("Loading spinful matrices in legacy deeph format is not supported yet")

        stru = from_deeph(folder)
        aodata = AOData(stru, None, basis_path_root=folder, aocode='deeph')

        orbitals_Us_openmx2wiki = get_Us_openmx2wiki(aodata.ls_spc)
        
        hoppings = []
        translations = []
        atom_pairs = []
        npairs = 0
        with h5py.File(f'{folder}/{filename}', libver='latest') as f:
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
        return MatAO(stru, translations, atom_pairs, hoppings, aodata)

def analyze_hdecay_deeph(savedir, filetype='h', mode='mean'):
    import matplotlib.pyplot as plt
    hmats = load_mat_deeph(savedir, filetype)
    distance = hmats.get_distances()
    hop_magnitude = np.zeros(hmats.npairs)
    for ipair in range(hmats.npairs):
        if mode == 'mean':
            hop_magnitude[ipair] = np.mean(np.abs(hmats.mats[ipair]))
        elif mode == 'max':
            hop_magnitude[ipair] = np.max(np.abs(hmats.mats[ipair]))
        else:
            raise NotImplementedError(mode)
    fig, ax = plt.subplots()
    ax.scatter(distance, hop_magnitude)
    ax.set_ylabel('$H(r)$ (eV)')
    ax.set_xlabel('$r$ (Bohr)')
    ax.set_yscale('log')
    fig.savefig(f'{savedir}/Hdecay.png')
    plt.close(fig)