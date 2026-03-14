from itertools import chain
import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import KDTree

from ..utils.misc import atom_number2name
from ..utils.supercell import minimum_supercell
from .matao import PairsInfo


def neighbors_csr(iat1: np.ndarray, iat2: np.ndarray, iorb2, n_nodes_uc, n_nodes_sc,
                  dedup: bool = True):
    if dedup:
        data = np.ones_like(iat1, dtype=bool)
    else:
        data = np.ones_like(iat1, dtype=np.int32)

    A = csr_matrix((data, (iat1, iat2)), shape=(n_nodes_uc, n_nodes_sc))
    indptr, indices = A.indptr, A.indices
    neighbors = []
    for i in range(n_nodes_uc):
        neighbors_i = indices[indptr[i]:indptr[i+1]]
        iorb2_i = np.array([iorb2]*len(neighbors_i))
        neighbors.append(np.stack((neighbors_i, iorb2_i), axis=1))
    outdeg = np.diff(indptr)

    return neighbors, outdeg, A

def merge_neighbors(groups):
    n_nodes = max(len(g) for g in groups)
    merged = [[] for _ in range(n_nodes)]
    for g in groups:
        for i, nb in enumerate(g):
            merged[i].extend(nb)
    return [np.array(nb) for nb in merged]

def pairs_within_cutoff(structure1, cutoffs1, structure2=None, cutoffs2=None):
    '''
    find all atom pairs (i, j) satisfying |r_i-r_j| <= (cutoff_i+cutoff_j),
    where r_i and r_j are atomic positions in structure1 and structure2 respectively.
    (the two structures must have the same lattice vectors and atomic species)
    This is the new implementation with scipy.spatial.KDTree which can be 300x faster on MATBG

    Returns:
        PairsInfo: The PairsInfo object containing the atom pairs within cutoff.
    '''
    if structure2 is None:
        structure2 = structure1
    if cutoffs2 is None:
        cutoffs2 = cutoffs1

    maxr = max(cutoffs1.values()) + max(cutoffs2.values())
    supercell2 = minimum_supercell(structure2, 2*maxr)

    atom_names_1 = atom_number2name(structure1.atomic_species)
    atom_names_2 = atom_number2name(structure2.atomic_species)
    assert atom_names_1 == atom_names_2, "The two structures must have the same atomic species"
    atom_names = atom_names_1
    cutoffs_1 = [cutoffs1[name] for name in atom_names]
    cutoffs_2 = [cutoffs2[name] for name in atom_names]

    # 1 corresponds to first atoms in atom pairs, and 2 corresponds to second atoms.
    # 1 is always in unit cell, 2 is always in supercell.
    trees1_spc, trees2_spc = [], []
    # mapiat maps index of atom of specifc atomic number (i.e. atom index in the tree) 
    # to index of it in original cell
    mapiat1_spc, mapiat2_spc = [], []
    for ispc in range(structure1.nspc):
        spc = structure1.atomic_species[ispc]

        is_thisspc = (structure1.atomic_numbers == spc)
        trees1_spc.append(KDTree(structure1.atomic_positions_cart_uc[is_thisspc]))
        mapiat1_spc.append(np.where(is_thisspc)[0])

        is_thisspc = (supercell2.atomic_numbers == spc)
        trees2_spc.append(KDTree(supercell2.atomic_positions_cart[is_thisspc]))
        mapiat2_spc.append(np.where(is_thisspc)[0])

    allpairs = []
    for ispc1 in range(structure1.nspc):
        for ispc2 in range(structure2.nspc):
            tree1 = trees1_spc[ispc1]
            tree2 = trees2_spc[ispc2]
            map1 = mapiat1_spc[ispc1]
            map2 = mapiat2_spc[ispc2]

            res = tree1.query_ball_tree(tree2, r=cutoffs_1[ispc1]+cutoffs_2[ispc2])
            iatspc1 = []
            for i1 in range(len(res)):
                n = len(res[i1])
                if n > 0: iatspc1.append(np.full(n, i1, dtype=int))

            if len(iatspc1) > 0: 
                iatspc1 = np.concatenate(iatspc1)
                iatspc2 = np.fromiter(chain.from_iterable(res), dtype=int)
                allpairs.append(np.stack((map1[iatspc1], map2[iatspc2]), axis=1))

    allpairs = np.concatenate(allpairs)

    translations_cuc, iat2_uc = supercell2.iat_sc2uc(allpairs[:, 1], True)

    allpairs[:, 1] = iat2_uc

    translations = structure2.trans_cuc_to_original(translations_cuc, allpairs[:, 0], allpairs[:, 1])

    return PairsInfo(structure2, translations, allpairs)


def find_orb_pairs_direct(structure1, cutoffs1_orb, structure2=None, cutoffs2_orb=None):
    """
    Find orbital pairs between two structures.
    cutoffs1_orb: cutoff of atomic orbitals for each species
    cutoffs2_orb: cutoff of atomic orbitals for each species
    Returns:
        translations_spc_orb: dict, key is (ispc1, iorb1, ispc2, iorb2), value is (N_pairs, 3) array of translations
        allpairs_spc_orb: dict, key is (ispc1, iorb1, ispc2, iorb2), value is (N_pairs, 2) array of atom pairs (iat1_uc, iat2_uc)
        pairs_key: (N_pairs, 9) array, each row is (tx, ty, tz, iat1_uc, iat2_uc, iorb1, iorb2, l1, l2)
    """
    if structure2 is None:
        structure2 = structure1
    if cutoffs2_orb is None:
        cutoffs2_orb = cutoffs1_orb

    rmax_1 = max([max(cutoffs) for cutoffs in cutoffs1_orb.values()])
    rmax_2 = max([max(cutoffs) for cutoffs in cutoffs2_orb.values()])
    maxr = rmax_1 + rmax_2
    supercell2 = minimum_supercell(structure2, 2*maxr)

    atom_names_1 = atom_number2name(structure1.atomic_species)
    atom_names_2 = atom_number2name(structure2.atomic_species)
    assert atom_names_1 == atom_names_2, "The two structures must have the same atomic species"
    atom_names = atom_names_1
    atom_spcs = structure1.atomic_species
    cutoffs_1 = [cutoffs1_orb[name] for name in atom_names]
    cutoffs_2 = [cutoffs2_orb[name] for name in atom_names]

    # 1 corresponds to first atoms in atom pairs, and 2 corresponds to second atoms.
    # 1 is always in unit cell, 2 is always in supercell.
    trees1_spc, trees2_spc = [], []
    # mapiat maps index of atom of specifc atomic number (i.e. atom index in the tree) 
    # to index of it in original cell
    mapiat1_spc, mapiat2_spc = [], []
    for ispc in range(structure1.nspc):
        spc = structure1.atomic_species[ispc]

        is_thisspc = (structure1.atomic_numbers == spc)
        trees1_spc.append(KDTree(structure1.atomic_positions_cart_uc[is_thisspc]))
        mapiat1_spc.append(np.where(is_thisspc)[0])

        is_thisspc = (supercell2.atomic_numbers == spc)
        trees2_spc.append(KDTree(supercell2.atomic_positions_cart[is_thisspc]))
        mapiat2_spc.append(np.where(is_thisspc)[0])

    allpairs_spc_orb = {}
    translations_spc_orb = {}
    pairs_key_list = []
    for ispc1 in range(structure1.nspc):
        for ispc2 in range(structure2.nspc):
            tree1 = trees1_spc[ispc1]
            tree2 = trees2_spc[ispc2]
            map1 = mapiat1_spc[ispc1]
            map2 = mapiat2_spc[ispc2]

            for iorb1 in range(len(cutoffs_1[ispc1])):
                for iorb2 in range(len(cutoffs_2[ispc2])):
                    res = tree1.query_ball_tree(tree2, r=cutoffs_1[ispc1][iorb1]+cutoffs_2[ispc2][iorb2]+1.0e-6)
                    iatspc1 = []
                    for i1 in range(len(res)):
                        n = len(res[i1])
                        if n > 0: iatspc1.append(np.full(n, i1, dtype=int))

                    if len(iatspc1) > 0: 
                        iatspc1 = np.concatenate(iatspc1)
                        iatspc2 = np.fromiter(chain.from_iterable(res), dtype=int)
                        pairs_orb1_orb2 = np.stack((map1[iatspc1], map2[iatspc2]), axis=1)
                        allpairs_spc_orb[(ispc1, iorb1, ispc2, iorb2)] = pairs_orb1_orb2

                        iat1 = pairs_orb1_orb2[:, 0]
                        translations_cuc, iat2_uc = supercell2.iat_sc2uc(pairs_orb1_orb2[:, 1], True)
                        allpairs_spc_orb[(ispc1, iorb1, ispc2, iorb2)][:, 1] = iat2_uc
                        translations = structure2.trans_cuc_to_original(translations_cuc, pairs_orb1_orb2[:, 0], pairs_orb1_orb2[:, 1])
                        translations_spc_orb[(ispc1, iorb1, ispc2, iorb2)] = translations

                        pairs_key_this = np.concatenate([translations, iat1[:, None], iat2_uc[:, None], 
                                          np.full(len(iat1), iorb1, dtype=int)[:, None], 
                                          np.full(len(iat1), iorb2, dtype=int)[:, None]],
                                          axis=1)
                        pairs_key_list.append(pairs_key_this)
                        if np.any(np.all(pairs_key_this[:,:5] == np.array([1,1,1,215,215]), axis=1)):
                            breakpoint()
    
    pairs_key = np.concatenate(pairs_key_list, axis=0)
    _, idx = np.unique(pairs_key, axis=0, return_index=True)
    pairs_key = pairs_key[np.sort(idx)] # each row is (tx, ty, tz, iat1_uc, iat2_uc, iorb1, iorb2)

    return translations_spc_orb, allpairs_spc_orb, pairs_key


def find_orb_pairs_proj(structure1, cutoffs1, cutoffs2_orb):
    """
    cutoff1: max cutoff of KB projectors for each species
    cutoffs2_orb: cutoff of atomic orbitals
    Returns:
        pairs_key: (N_pairs, 7) array, each row is (tx, ty, tz, iat1_uc, iat2_uc, iorb1, iorb2)
    """
    structure2 = structure1

    rmax_1 = max(cutoffs1.values())
    rmax_2 = max([max(cutoffs) for cutoffs in cutoffs2_orb.values()])
    maxr = rmax_1 + rmax_2
    supercell2 = minimum_supercell(structure2, 2*maxr)

    atom_names_1 = atom_number2name(structure1.atomic_species)
    atom_names_2 = atom_number2name(structure2.atomic_species)
    assert atom_names_1 == atom_names_2, "The two structures must have the same atomic species"
    atom_names = atom_names_1
    atom_spcs = structure1.atomic_species
    cutoffs_1 = [cutoffs1[name] for name in atom_names]
    cutoffs_2 = [cutoffs2_orb[name] for name in atom_names]

    # 1 corresponds to first atoms in atom pairs, and 2 corresponds to second atoms.
    # 1 is always in unit cell, 2 is always in supercell.
    trees1_spc, trees2_spc = [], []
    # mapiat maps index of atom of specifc atomic number (i.e. atom index in the tree) 
    # to index of it in original cell
    mapiat1_spc, mapiat2_spc = [], []
    for ispc in range(structure1.nspc):
        spc = structure1.atomic_species[ispc]

        is_thisspc = (structure1.atomic_numbers == spc)
        trees1_spc.append(KDTree(structure1.atomic_positions_cart_uc[is_thisspc]))
        mapiat1_spc.append(np.where(is_thisspc)[0])

        is_thisspc = (supercell2.atomic_numbers == spc)
        trees2_spc.append(KDTree(supercell2.atomic_positions_cart[is_thisspc]))
        mapiat2_spc.append(np.where(is_thisspc)[0])

    neighbors = []
    for ispc2 in range(structure2.nspc):
        tree2 = trees2_spc[ispc2]
        map2 = mapiat2_spc[ispc2]
        for iorb2 in range(len(cutoffs_2[ispc2])):
            for ispc1 in range(structure1.nspc):
                tree1 = trees1_spc[ispc1]
                map1 = mapiat1_spc[ispc1]

                res = tree1.query_ball_tree(tree2, r=cutoffs_1[ispc1]+cutoffs_2[ispc2][iorb2]+1.0e-6)
                iatspc1 = []
                for i1 in range(len(res)):
                    n = len(res[i1])
                    if n > 0: iatspc1.append(np.full(n, i1, dtype=int))

                if len(iatspc1) > 0: 
                    iatspc1 = np.concatenate(iatspc1)
                    iatspc2 = np.fromiter(chain.from_iterable(res), dtype=int)
                    pairs_spc1_orb2 = np.stack((map1[iatspc1], map2[iatspc2]), axis=1)

                    iat1_uc = pairs_spc1_orb2[:, 0]
                    iat2 = pairs_spc1_orb2[:, 1]
                    neighbors_this, _, _ = neighbors_csr(iat1_uc, iat2, iorb2, structure1.natom, supercell2.natom)
                    neighbors.append(neighbors_this)
    neighbors = merge_neighbors(neighbors)

    pairs_key_list = []
    order = np.argsort(atom_spcs)
    ispcs = order[np.searchsorted(atom_spcs, structure1.atomic_numbers, sorter=order)]
    for iatom in range(structure1.natom):
        neighbors_i = neighbors[iatom]
        iat_nbr = neighbors_i[:, 0]
        iorb_nbr = neighbors_i[:, 1]
        translations_cuc, iat_nbr_uc = supercell2.iat_sc2uc(iat_nbr, True)
        translations_cuc += supercell2.translations[iat_nbr_uc]

        N_nbr_this = len(iat_nbr)
        translations_pairs = (- translations_cuc[:, None, :] + translations_cuc[None, :, :]).reshape(N_nbr_this**2, 3)
        i_i, i_j = np.meshgrid(iat_nbr_uc, iat_nbr_uc, indexing='ij')
        iorb_i, iorb_j = np.meshgrid(iorb_nbr, iorb_nbr, indexing='ij')
        iat_pairs = np.stack([i_i.ravel(), i_j.ravel()], axis=1)  
        iorb_pairs = np.stack([iorb_i.ravel(), iorb_j.ravel()], axis=1)

        pairs_key_this = np.concatenate([translations_pairs, iat_pairs, iorb_pairs], axis=1)  # (N_pairs, 3+2+2)
        pairs_key_list.append(pairs_key_this)

    pairs_key = np.concatenate(pairs_key_list, axis=0)
    _, idx = np.unique(pairs_key[:,:7], axis=0, return_index=True)
    pairs_key = pairs_key[np.sort(idx)] # each row is (tx, ty, tz, iat1_uc, iat2_uc, iorb1, iorb2)

    return pairs_key