import numpy as np
import h5py

from ..constants import hartree2ev
from ..io.bgwio import bgw_vsc
from .aodata import AOData

'''
This module implements several functions needed by real-space construction of AO Hamiltonian.
This includes reading local potentials, reading nonlocal part of pseudopotentials, and constructing VKB 
under AO basis.
'''

def read_vloc(filename, interface):
    if interface == 'bgw':
        vscread = bgw_vsc(filename)
        vscread.read_header()
        # if is_master:
        #     print('Reading self-consistent potential from VSC')
        vscread.read_data()
        vscread.close()
        FFTgrid = np.array([vscread.nr1, vscread.nr2, vscread.nr3])
        vscg_full = np.zeros(FFTgrid, dtype='c16')
        _, g_g_full = np.divmod(vscread.g_g, FFTgrid)
        vscg_full[g_g_full[:, 0], g_g_full[:, 1], g_g_full[:, 2]] = vscread.vscg
        vlocr = np.fft.ifftn(vscg_full, norm='forward')
        assert np.max(np.abs(vlocr.imag)) < 1e-6
        vlocr = vlocr.real
    elif interface == 'deephr':
        with h5py.File(filename, 'r') as f:
            vlocr = np.array(f['Vtot']) / 2. # Ry->Ha
    elif interface == 'gpaw':
        from ase.io import ulm
        with ulm.open(filename) as f:
            vlocr = f.hamiltonian.potential[0] / hartree2ev # ! no spin
        # FFTgrid = np.array(vlocr.shape)
    else:
        raise NotImplementedError(f'Unknown vloc interface: {interface}')
    
    return vlocr

def unpack(vec, nsize):
    # gpaw/utilities/__init__.py pack2
    assert len(vec) == nsize * (nsize+1) // 2
    mat = np.empty((nsize, nsize))
    ipos = 0
    for i in range(nsize):
        mat[i, i] = vec[ipos]
        ipos += 1
        for j in range(i+1, nsize):
            mat[i, j] = vec[ipos]
            mat[j, i] = vec[ipos]
            ipos += 1
    assert ipos == len(vec)
    return mat

def read_vnloc(structure, pspdir, dijfile=None, interface='qe'):
    if interface == 'qe':
        assert dijfile is None
        projR = AOData(structure, None, basis_path_root=pspdir, aocode='qe-projR')
        Dij = []
        for zatm in structure.atomic_numbers:
            Dij.append(projR.Dij_spc[zatm])
        Qij = None
    elif interface == 'siesta':
        assert dijfile is None
        projR = AOData(structure, None, basis_path_root=pspdir, aocode='siesta-projR')
        Dij = []
        for zatm in structure.atomic_numbers:
            Dij.append(projR.Dij_spc[zatm])
        Qij = None
    elif interface == 'gpaw':
        projR = AOData(structure, None, basis_path_root=pspdir, aocode='gpaw-projR')
        if dijfile is not None:
            from ase.io import ulm
            with ulm.open(dijfile) as f:
                cdij_raw = f.hamiltonian.atomic_hamiltonian_matrices[0] / hartree2ev # ! no spin
            # Dij = []
            # pos = 0
            # for iat in range(structure.natom):
            #     spc = structure.atomic_numbers[iat]
            #     nsize = projR.norbfull_spc[spc]
            #     offset = nsize * (nsize+1) // 2
            #     cdijmat = np.empty((nsize, nsize))
            #     cdijmat[np.tril_indices(nsize)] = cdij_raw[pos:pos+offset]
            #     cdijmat[np.triu_indices(nsize)] = cdij_raw[pos:pos+offset]
            #     Dij.append(cdijmat)
            #     pos += offset
            # assert pos == len(cdij_raw)
            Dij = []
            pos = 0
            for iat in range(structure.natom):
                spc = structure.atomic_numbers[iat]
                nsize = projR.norbfull_spc[spc]
                offset = nsize * (nsize+1) // 2
                Dij.append(unpack(cdij_raw[pos:pos+offset], nsize))
                pos += offset
            assert pos == len(cdij_raw)
        else:
            Dij = None
        Qij = []
        for zatm in structure.atomic_numbers:
            Qij.append(projR.Qij_spc[zatm])
    else:
        raise NotImplementedError(f'Unknown vnloc interface: {interface}')    

    return Dij, Qij, projR
