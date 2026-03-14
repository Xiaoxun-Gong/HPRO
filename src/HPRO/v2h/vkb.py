import numpy as np

from ..utils.misc import slice_same
from ..matao.matao import PairsInfo, pairs_to_indices, indices_to_pairs, MatAO
from ..matao.findpairs import pairs_within_cutoff

def calc_vkb(olp_proj_ao, Dij, spinful=False):
    '''
    Construct VKB in atomic orbital basis according to the formula:
    <ia| Vkb |i'a'> = \sum_{jbb'} <ia|jb> D_{jbb'} <jb'|i'a'>
    where i, i', j are atom indices, a, a', b, b' are orbital indices

    Matrix D is optional. If not given, matrices will all be None.
    '''

    if Dij is not None:
        for iD in range(len(Dij)):
            D = Dij[iD]
            if not spinful:
                if not np.isrealobj(D):
                    assert np.max(np.abs(D.imag)) < 1e-8
                    Dij[iD] = D.real
                
    olp_proj_ao.sort_atom1()
    translations = olp_proj_ao.translations
    atom_pairs = olp_proj_ao.atom_pairs
    translations_new, atom_pairs_new, mats_new = [], [], []
    
    # do matrix multiplications
    slice_jatm = slice_same(atom_pairs[:, 0])
    njatm = len(slice_jatm) - 1
    for ix_atm in range(njatm):
        startj = slice_jatm[ix_atm]
        endj = slice_jatm[ix_atm + 1]
        ix_js, ix_jps = np.tril_indices(endj - startj) # lower-triangle indices
        ix_js += startj; ix_jps += startj
        # ix_js, ix_jps = np.meshgrid(np.arange(startj, endj, 1), np.arange(startj, endj, 1), indexing='ij')
        # ix_js = ix_js.reshape(-1); ix_jps = ix_jps.reshape(-1)
        translations_new.append(translations[ix_jps] - translations[ix_js])
        atom_pairs_new.append(np.stack((atom_pairs[ix_js, 1], atom_pairs[ix_jps, 1]), axis=1))
        if Dij is not None:
            atomj = atom_pairs[startj, 0]
            for ix_j, ix_jp in zip(ix_js, ix_jps):
                mat = olp_proj_ao.mats[ix_j]
                matp = olp_proj_ao.mats[ix_jp]
                # dimensions of D are (orb1, orb2) in spinless case and (spin1, spin2, orb1, orb2) in spinful case
                D = Dij[atomj]
                if not spinful:
                    mats_new.append(mat.T @ D @ matp) # ! no conjugate here because all mats are real
                else:
                    mat_j = mat.conj().T @ D @ matp
                    A = np.block([[mat_j[0,0], mat_j[0,1]],[mat_j[1,0], mat_j[1,1]]])
                    mats_new.append(A)
    translations_new = np.concatenate(translations_new, axis=0)
    atom_pairs_new = np.concatenate(atom_pairs_new, axis=0)
    
    # collect terms with the same translations and atom pairs and sum them up
    indices_new = pairs_to_indices(olp_proj_ao.structure, translations_new, atom_pairs_new)
    argsort = np.argsort(indices_new, kind='stable')
    indices_new = indices_new[argsort]
    slice_samepair = slice_same(indices_new)
    npairs_final = len(slice_samepair) - 1
    if Dij is not None:
        mats_final = []
        for ipair in range(npairs_final):
            slice_thispair = slice(slice_samepair[ipair], slice_samepair[ipair+1])
            mats_final.append(np.sum([mats_new[i] for i in argsort[slice_thispair]], axis=0))
    indices_final = np.unique(indices_new)
    translations_final, atom_pairs_final = indices_to_pairs(olp_proj_ao.structure.natom, indices_final)
    
    if Dij is not None:
        vkb = MatAO(olp_proj_ao.structure, translations_final, atom_pairs_final, mats_final, olp_proj_ao.aodata2, spinful=spinful)
    else:
        vkb = PairsInfo(olp_proj_ao.structure, translations_final, atom_pairs_final)
    vkb.unfold_with_hermiticity()
    
    return vkb

def get_nloc_pairs(structure, cutoffs_proj, cutoffs_basis):
    '''
    Get the output atom pairs of the function `calc_vkb` without calculating Vkb.
    '''
    pairs_proj_basis = pairs_within_cutoff(structure, cutoffs_proj, cutoffs2=cutoffs_basis)
    vkb_none = calc_vkb(pairs_proj_basis, None)
    return vkb_none