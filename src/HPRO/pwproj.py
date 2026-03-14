import time
import numpy as np

from .utils.mpi import distrib_vec, is_master, MPI, comm
from .utils.misc import mytqdm
from .io.aodata import calc_FT_kg_orb_spcs
from .matao.matao import MatAO


def Hpw_to_Hao_wfn(structure, wfndata, aodata, pairs, wfn_proj_atoms, nbndmin, nbndmax, 
                   band_shift=0):
    '''
    Project PW wavefunctions to AO basis
    '''

    # pairs must be full instead of hermiticity-reduced
    hoppings = MatAO.init_mats(pairs, aodata, filling_value=0., dtype='c16')
        
    rank, count, displ = distrib_vec(pairs.npairs, displ_last_elem=True)
    
    rprim = structure.rprim
    nk = wfndata.nk
    nbnd = nbndmax - nbndmin + 1
    kpts_cart = wfndata.kpts_cart
    norbfull_spc = aodata.norbfull_spc
    atom_nbrs = structure.atomic_numbers
    kptwts = wfndata.kptwts
    
    Hbnd_k = wfndata.get_H_band_basis(nbndmin, nbndmax, 
                                      band_shift=band_shift)
    
    if comm is not None:
        wfn_proj_atoms_recv = []
        for iatm, nspc in enumerate(atom_nbrs):
            if comm.rank == 0:
                buf = wfn_proj_atoms[iatm]
            else:
                buf = np.empty((nk, nbnd, norbfull_spc[nspc]), dtype=np.complex128)
            comm.Bcast([buf, MPI.COMPLEX16], root=0)
            wfn_proj_atoms_recv.append(buf) # List[array(nk, nbnd, norbitals)], len=natom
        wfn_proj_atoms = wfn_proj_atoms_recv
        
    # if comm.rank == 0:
    #     print(wfn_proj_atoms[0].nbytes)
    
    if is_master():
        print('\n=======================================\n')
        print(f'Constructing {count[0]} hopping blocks')
    for ipair in mytqdm(range(displ[rank], displ[rank+1])):
        trans_cart = pairs.translations[ipair] @ rprim
        phase = np.exp(-1j*np.dot(kpts_cart, trans_cart))
        i1, i2 = pairs.atom_pairs[ipair]
        proj1 = wfn_proj_atoms[i1]
        proj2 = wfn_proj_atoms[i2].conj()
        for ik in range(nk):
            hoppings.mats[ipair][...] += phase[ik] * kptwts[ik] * (proj1[ik, ...].T @ (Hbnd_k[ik] @ proj2[ik, ...]))

    if comm is not None:
        hoppings.mpi_gather(displ, dtype=MPI.COMPLEX16, root=0)
    if comm is not None and comm.rank != 0:
        hoppings.delete_mats()
    
    if is_master():
        for ipair in range(pairs.npairs):
            hoppings.mats[ipair] = hoppings.mats[ipair].real # enforce TR symmetry
        
    return hoppings


def Hpw_to_Hao_HG(structure, hgdata, aodata, pairs_ij, h_or_s='h'):
    '''
    Project full PW Hamiltonian to AO basis
    '''
    
    rprim = structure.rprim
    atom_pos = structure.atomic_positions_cart
    atom_nbrs = structure.atomic_numbers
    vol = structure.cell_volume
    nat = structure.natom
    nk = hgdata.nk
    kptwts = hgdata.kptwts
    kpts = hgdata.kpts
    kpts_cart = hgdata.kpts_cart
    
    # pairs, hoppings = aodata.init_hamiltonians(use_hermiticity=True)
    assert pairs_ij.is_sorted()
    hoppings = MatAO.init_mats(pairs_ij, aodata, filling_value=0., dtype='c16')
        
    for ik in range(nk):
        tstartk = time.time()
        
        if is_master():
            krel = kpts[ik]
            print('\n====================================================================')
            print(f'Dealing with k = {krel[0]:12.8f} {krel[1]:12.8f} {krel[2]:12.8f}  [{ik+1:3d} / {nk:3d}]')
            print('====================================================================\n')
            
        kpt_cart = kpts_cart[ik]
        wk = kptwts[ik]
        ng = hgdata.vkbgdatas[ik].ng
        kgcart = hgdata.vkbgdatas[ik].kgcart
        
        rank, count, displ = distrib_vec(ng, displ_last_elem=True)
        
        FT_kg_orb_spcs = calc_FT_kg_orb_spcs(ng, kgcart, aodata, hgdata.ecutwfn)
        
        # if is_master():
        #     print('Building plane-wave Hamiltonian')
        slice1 = slice(displ[rank], displ[rank+1])
        if not hgdata.ispaw:
            hamblock = hgdata.build_hamblock(ik, gvecrange1=(displ[rank],displ[rank+1]))
        else:
            hamblock = hgdata.build_hs_paw(ik, kind=h_or_s, gvecrange1=(displ[rank],displ[rank+1]))
        
        atom_pairs_ij = hoppings.atom_pairs
        nuniq = len(np.unique(atom_pairs_ij[:,0]*nat + atom_pairs_ij[:,1]))
        
        if is_master():
            print(f'Constructing {nuniq} hopping blocks')
        t = mytqdm(total=nuniq)
        last_pair = None
        for ipair in range(hoppings.npairs):
            this_pair = hoppings.atom_pairs[ipair]
            trans_cart = hoppings.translations[ipair] @ rprim
            phase_trans = np.exp(-1j * np.dot(kpt_cart, trans_cart))
            if last_pair is None or not np.all(last_pair==this_pair):
                # Some hoppings have same atom pairs but different lattice translations.
                # In this case, they are different from each other only by a phase.
                # Notice that same atom pairs are next to each other in atom_pairs, see:
                # aodata.py: pairs_within_cutoff()
                iatm1, iatm2 = hoppings.atom_pairs[ipair, :]
                pos1, pos2 = atom_pos[hoppings.atom_pairs[ipair, :]]
                phase1 = np.exp(-1j * np.dot(kgcart[slice1, :], pos1))
                phase2 = np.exp(-1j * np.dot(kgcart, pos2))
                spc1 = atom_nbrs[iatm1]
                spc2 = atom_nbrs[iatm2]
                tmp = hamblock @ (phase2[:, None] * FT_kg_orb_spcs[spc2])
                tmph = (phase1[:, None] * FT_kg_orb_spcs[spc1][slice1, :]).T.conj() @ tmp
                if comm is not None: comm.Barrier()
                t.update()
            hoppings.mats[ipair] += tmph * (phase_trans * wk / vol)
            last_pair = this_pair
        t.close()
        
        hamblock = None
        
        if comm is not None:
            comm.Barrier()
        tendk = time.time()
        if is_master():
            print(f'Time spent on this k-point: {tendk-tstartk:.1f}s\n')
            
    if comm is not None:
        hoppings.mpi_reduce(dtype=MPI.COMPLEX16, op=MPI.SUM, root=0)
    if comm is not None and comm.rank != 0:
        hoppings.delete_mats()
    
    if is_master():
        for ipair in range(hoppings.npairs):
            hoppings.mats[ipair] = hoppings.mats[ipair].real # enforce TR symmetry
    
    hoppings.unfold_with_hermiticity()
    return hoppings


def wfn_proj_to_atoms(structure, aodata, wfndata, nbndmin, nbndmax, verbose=True):
    r'''
    Parameters:
    ---------
    nbndmin and nbndmax are 1-based indices
    
    Returns:
    ---------
    wfn_proj_atoms: List[array(nk, nbnd, norbitals)]], len=natom, <\phi_{i\alpha}|u_{nk}>
    '''
            
    atom_nbrs = structure.atomic_numbers
    atm_pos_cart = structure.atomic_positions_cart
    vol = structure.cell_volume
    nat = structure.natom
    norbfull_spc = aodata.norbfull_spc
    
    nbnd = nbndmax - nbndmin + 1
    nk = wfndata.nk
    kpts = wfndata.kpts
    
    # wfn_proj_atoms_eachk = [] # List[List[array(nbnd, norbitals)]], len1=nk, len2=natom
    if is_master():
        wfn_proj_atoms = []
        for nspc in atom_nbrs:
            wfn_proj_atoms.append(np.zeros((nk, nbnd, norbfull_spc[nspc]), dtype=np.complex128)) # List[array(nk, nbnd, norbitals)], len=natom
    
    if is_master():
        if not verbose: print(f'Calculating projections of wavefunctions on {nk} k points')
    for ik in mytqdm(range(nk), disable=verbose):
        
        kgdata = wfndata.kgdatas[ik]
        
        ng = kgdata.ng
        kgcart = kgdata.kgcart
        unkg = kgdata.unkg[nbndmin-1:nbndmax]
        
        if is_master():
            if verbose:
                krel = kpts[ik]
                print('\n====================================================================')
                print(f'Dealing with k = {krel[0]:12.8f} {krel[1]:12.8f} {krel[2]:12.8f}  [{ik+1:3d} / {nk:3d}]')
                print('====================================================================\n')
            
        FT_kg_orb_spcs = calc_FT_kg_orb_spcs(ng, kgcart, aodata, wfndata.ecutwfn)
        
        if is_master():
            if verbose: print(f'Projecting to {nat} atoms')
        # wfn_proj_atoms = [] # List[array(nbnd, norbitals)], len=natom
        for iatom, spc in mytqdm(enumerate(atom_nbrs), total=nat, disable=not verbose):
            
            orbital_center = atm_pos_cart[iatom]
            FT_kg_orb = FT_kg_orb_spcs[spc]
            phase = np.exp(1j * np.dot(kgcart, orbital_center))
            wfn_proj_atom = np.sum(phase[None, :, None] * unkg[:, :, None] * 
                                    FT_kg_orb[None, :, :].conj(), axis=1) / np.sqrt(vol)
            
            if comm is not None:
                if comm.rank == 0:
                    wfn_proj_atom_combined = np.zeros_like(wfn_proj_atom)
                else:
                    wfn_proj_atom_combined = None
                comm.Reduce([wfn_proj_atom, MPI.COMPLEX16],
                            [wfn_proj_atom_combined, MPI.COMPLEX16], op=MPI.SUM, root=0)
                if comm.rank == 0:
                    wfn_proj_atoms[iatom][ik, ...] = wfn_proj_atom_combined
            else:
                wfn_proj_atoms[iatom][ik, ...] = wfn_proj_atom
            
    if is_master():
        return wfn_proj_atoms
    else:
        return None
