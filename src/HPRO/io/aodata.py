import os
import re
import json
import xml.etree.ElementTree as ET
import numpy as np

from .. import config as CFG
from ..utils.structure import Structure
from ..utils.misc import atom_number2name, atom_name2number
from ..utils.orbutils import grid_R2G, GridFunc, LinearRGD
from ..io.gpawio import gpaw_psp
from ..io.struio import from_deeph
from ..utils.math import get_dmat_coeffs

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


class AOData:
    '''
    Stores information about atomic orbital functions for several elements needed by the calculation.

    Attributes:
    ---------
    structure: structure.Structure, storing the structure information
    aocode: str
    ls_spc: Dict[int -> List[int]], list of angular momentum quantum numbers of orbitals
    phirgrids_spc: Dict[int -> List[GridFunc]], functions of each orbital
    orbslices_spc: Dict[int -> List[int]], the slices of each orbital if we concatenate all orbitals of different l's together
    nradial_spc: Dict[int -> int], number of radial functions, not number of AOs. Number of AOs is norbfull_spc[spc]
    norbfull_spc: Dict[int -> int], number of AOs
    Dij_spc: Dict[int -> array], pseudopotential Dij matrix
    Qij_spc: Dict[int -> array], pseudopotential Qij matrix (PAW only)
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
        self.spinful = False
        self.magnetic = False
        assert not self.magnetic, "Magnetic calculation is not supported yet."
        
        self.ls_spc = {}
        self.phirgrids_spc = {}
        self.nradial_spc = {}
        self.Dij_spc = {}
        self.Qij_spc = {}
        
        spc_numbers = structure.atomic_species
        spc_names = atom_number2name(spc_numbers)
        for spc_nu, spc_na in zip(spc_numbers, spc_names):
            Dij, Qij = None, None
            valence_charge = None
            # if spc_nu not in self.orbitals_types:
            if aocode == 'siesta':
                # .ion file is the same for spinful and spinless calculations.
                nradial, phirgrids, _, _, valence_charge, _ = parse_siesta_ion(f'{basis_path_root}/{spc_na}.ion')
            elif aocode == 'gpaw':
                nradial, phirgrids = parse_gpaw_basis(findfile(basis_path_root, rf'^{spc_na}\..*\.basis$'))
            elif aocode == 'gpaw-projR':
                # todo: not using PBE?
                gpawpsp = gpaw_psp(findfile(basis_path_root, rf'^{spc_na}(\..*|)\.PBE(.gz|)$'))
                nradial, phirgrids = len(gpawpsp.l_list), gpawpsp.projR_list
                Qij = gpawpsp.get_cqij()
            elif aocode == 'qe-projR':
                Dij, phirgrids = read_upf(findfile(basis_path_root, rf'^{spc_na}\.(upf|UPF)$'))
                nradial = len(phirgrids)
            elif aocode == 'siesta-projR':
                _, _, nradial, phirgrids, _, Dij = parse_siesta_ion(f'{basis_path_root}/{spc_na}.ion')
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
            self.nradial_spc[spc_nu] = nradial

            if Dij is not None:
                self.Dij_spc[spc_nu] = Dij
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
            self.nradial_spc = orbitals_num

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
            cutoffs_orb = {}
            for spc_nu, spc_na in zip(spc_numbers, spc_names):
                cutoffs_orb[spc_na] = [phirgrid.rcut for phirgrid in self.phirgrids_spc[spc_nu]]
                cutoffs[spc_na] = max(cutoffs_orb[spc_na])
            self.cutoffs_orb = cutoffs_orb

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
                norm = phirgrid.rgd.integrate(phi_r*phi_r)
                print(phirgrid.l, norm)
    
    def calc_phiQ(self, Ecut):
        if self.phiQEcut is not None and self.phiQlist_spc is not None:
            if self.phiQEcut >= Ecut: return
        grid_nq = int(np.sqrt(Ecut) * CFG.AOFT_QGRID_DEN)
        gridQ = LinearRGD(0, np.sqrt(2*Ecut), grid_nq)
        phiQlist_spc = {}
        for spc in self.structure.atomic_species:
            phiQlist = []
            for iorb in range(self.nradial_spc[spc]):
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

    def echo_info(self):
        print(f'Format: {self.aocode.split("-")[0]}')
        for spc in self.structure.atomic_species:
            name, = atom_number2name([spc])
            print(f'Element {name}:')
            for iorb in range(self.nradial_spc[spc]):
                l = self.ls_spc[spc][iorb]
                phirgrid = self.phirgrids_spc[spc][iorb]
                rcut = phirgrid.rcut
                phi_r = phirgrid.func
                norm = phirgrid.rgd.integrate(phi_r*phi_r)
                print(f'Orbital {iorb+1}: l = {l}, cutoff = {rcut:6.3f} a.u., norm = {norm:6.3f}')

def calc_FT_kg_orb_spcs(ng, kgcart, aodata, Ecut):
    stru = aodata.structure
    aodata.calc_phiQ(Ecut)
    FT_kg_orb_spcs = {}
    for spc in stru.atomic_species:
        nradial = aodata.nradial_spc[spc]
        orbslices = aodata.orbslices_spc[spc]
        FT_kg_orb = np.empty((ng, aodata.norbfull_spc[spc]), dtype=np.complex128)
        for iorb in range(nradial):
            slice_orb = slice(orbslices[iorb], orbslices[iorb+1])
            orbQ = aodata.phiQlist_spc[spc][iorb]
            FT_kg_orb[:, slice_orb] = orbQ.getval3D(kgcart) * (-1j)**orbQ.l # todo: make it real
        FT_kg_orb_spcs[spc] = FT_kg_orb
    return FT_kg_orb_spcs


# = Load orbitals =

def parse_deeph_orbtyps(deephsave):
    stru = from_deeph(deephsave)
    if CFG.DEEPH_USE_NEW_INTERFACE:
        with open(f'{deephsave}/info.json') as f:
            info = json.load(f)
            orbtypes = info['elements_orbital_map']
            orbital_types_spc = {atom_name2number([spcname])[0]: thisorbtype 
                                 for spcname, thisorbtype in orbtypes.items()}
    else:
        orbital_types = []
        with open(f'{deephsave}/orbital_types.dat') as f:
            line = f.readline()
            while line:
                orbital_types.append(list(map(int, line.split())))
                line = f.readline()
        orbital_types_spc = {}
        for atom_nbr, orbitals in zip(stru.atomic_numbers, orbital_types):
            if atom_nbr in orbital_types_spc:
                assert orbitals == orbital_types_spc[atom_nbr]
            else:
                orbital_types_spc[atom_nbr] = orbitals
    return orbital_types_spc, stru

def parse_gpaw_basis(filename):
    root = ET.parse(filename).getroot()
    gridfuncs = {}
    for gridfunc in root.findall('radial_grid'):
        if gridfunc.attrib['eq'] == 'r=d*i':
            istart = int(gridfunc.attrib['istart'])
            iend = int(gridfunc.attrib['iend'])
            d = float(gridfunc.attrib['d'])
            rgd = LinearRGD(istart*d, iend*d, iend-istart+1)
            gridid = gridfunc.attrib['id']
            gridfuncs[gridid] = rgd
        else:
            raise NotImplementedError

    phirgrids = []
    for basisfunc in root.findall('basis_function'):
        l = int(basisfunc.attrib['l'])
        gridid = basisfunc.attrib['grid']
        phi = np.array(list(map(float, basisfunc.text.split())))
        gridlen = len(phi)
        rgd = LinearRGD.from_explicit_grid(gridfuncs[gridid].rfunc[:gridlen])
        phirgrids.append(GridFunc(rgd, phi, l=l))

    norb = len(phirgrids)
    
    return norb, phirgrids

def parse_siesta_ion(filename):
    phirgrids_basis = []
    phirgrids_proj = []
    l_list_proj = []
    j_list_proj = []
    Dij_list = []
    norb_basis = 0
    norb_proj = 0
    rel = None

    ionfile = open(filename, 'r')
    line = ionfile.readline()
    while line:
        if line.find('rel') > 0 and rel is None:
            rel = True
        elif line.find('nrl') > 0 and rel is None:
            rel = False

        if line.find('# Valence charge') > 0:
            sp = line.split()
            valence_charge = float(sp[0])

        # basis orbitals
        if line.find('#orbital l, n, z, is_polarized, population') > 0:
            sp = line.split()
            l = int(sp[0])
            n = int(sp[1])
            z = int(sp[2])
            is_polarized = bool(int(sp[3]))
            population = float(sp[4])
            norb_basis += 1
            
            line_sp = ionfile.readline().split()
            assert line_sp[0] == '500'
            rcut = float(line_sp[2])
            
            phirgrid = np.zeros((2, 500)) # r, R(r)
            for ipt in range(500):
                phirgrid[:, ipt] = list(map(float, ionfile.readline().split()))
            # found this from sisl/io/siesta/siesta_nc.py: ncSileSiesta.read_basis(self): 
            # sorb = SphericalOrbital(l, (r * Bohr2Ang, psi), orb_q0[io])
            phirgrid[1, :] *= np.power(phirgrid[0, :], l) 
            rgd = LinearRGD.from_explicit_grid(phirgrid[0])
            assert np.abs(rgd.rend - rcut) < 1e-6
            phirgrids_basis.append(GridFunc(rgd, phirgrid[1], l=l, rcut=rcut))

        # projector orbitals
        projector_header = None
        if rel:
            projector_header = '#kb l, j, n (sequence number), Reference energy'
        elif not rel:
            projector_header = '#kb l, n (sequence number), Reference energy'
        if projector_header is not None and line.find(projector_header) > 0:
            sp = line.split()
            l = int(sp[0])
            if rel:
                j = float(sp[1])
                n = int(sp[2])
                ekb = float(sp[3])
                j_list_proj.append(j)
            else:
                n = int(sp[1])
                ekb = float(sp[2])

            norb_proj += 1
            Dij_list.append(ekb / 2.) # Ry to Har

            line_sp = ionfile.readline().split()
            assert line_sp[0] == '500'
            rcut = float(line_sp[2])

            phirgrid = np.zeros((2, 500)) # r, R(r)
            for ipt in range(500):
                phirgrid[:, ipt] = list(map(float, ionfile.readline().split()))
            phirgrid[1, :] *= np.power(phirgrid[0, :], l) 
            rgd = LinearRGD.from_explicit_grid(phirgrid[0])
            assert np.abs(rgd.rend - rcut) < 1e-6
            l_list_proj.append(l)
            phirgrids_proj.append(GridFunc(rgd, phirgrid[1], l=l, rcut=rcut))

        line = ionfile.readline()
    ionfile.close()

    Dij = np.diag(np.array(Dij_list))
    orbital_slices = np.cumsum(2 * np.array(l_list_proj) + 1)
    orbital_slices = np.insert(orbital_slices, 0, 0)
    norbfull = orbital_slices[-1]
    if rel:
        Dij_full = np.zeros((2, 2, norbfull, norbfull), dtype=np.complex128)
    else:
        Dij_full = np.zeros((norbfull, norbfull), dtype=np.float64)
    for iorb in range(len(l_list_proj)):
        l1 = l_list_proj[iorb]
        if rel: j1 = j_list_proj[iorb]
        for jorb in range(len(l_list_proj)):
            l2 = l_list_proj[jorb]
            if rel: j2 = j_list_proj[jorb]
            if (l1 == l2) and ((not rel) or (j1 == j2)):
                if rel:
                    Dij_full[:, :, orbital_slices[iorb]:orbital_slices[iorb+1], 
                                   orbital_slices[jorb]:orbital_slices[jorb+1]] = \
                        Dij[iorb, jorb] * get_dmat_coeffs(l1, j1)
                else:
                    np.fill_diagonal(Dij_full[orbital_slices[iorb]:orbital_slices[iorb+1],
                                              orbital_slices[jorb]:orbital_slices[jorb+1]], Dij[iorb, jorb])
            else:
                assert np.abs(Dij[iorb, jorb]) < 1e-8

    return norb_basis, phirgrids_basis, norb_proj, phirgrids_proj, valence_charge, Dij_full

def read_upf(filename):
    """
    Read a QE pseudopotential file in the upf format.

    Let nproj be the number of projector functions, and nproj_full be the sum of 2*l+1 of each projector:

    Returns:
        Dij_full array(nproj_full, nproj_full): A 2D array representing the Dij matrix.
        projR_list (list): A list of GridFunc objects representing the projector functions.
    """
    
    root = ET.parse(filename).getroot()

    header_elem = root.find('PP_HEADER')
    nproj = int(header_elem.attrib['number_of_proj'])

    rel = header_elem.attrib['relativistic']
    if rel == 'full':
        rel = True
        j_list = []
        socelem = root.find('PP_SPIN_ORB')
        for iproj in range(nproj):
            relelem = socelem.find(f'PP_RELBETA.{iproj+1}')
            j_list.append(float(relelem.attrib['jjj']))
    elif rel == 'scalar':
        rel = False
    else:
        raise ValueError(rel)

    r_elem = root.find('PP_MESH').find('PP_R')
    gridsize = int(r_elem.attrib['size'])
    rgridfunc = np.fromiter(map(float, r_elem.text.split()), float, count=gridsize)
    rgrid = LinearRGD.from_explicit_grid(rgridfunc)

    nloc_elem = root.find('PP_NONLOCAL')

    Dij_elem = nloc_elem.find('PP_DIJ')
    Dij = np.fromiter(map(float, Dij_elem.text.split()), float, count=nproj**2).reshape((nproj, nproj)) / 2. # Ry to Har
    
    projR_list = []
    l_list = []
    for iproj in range(nproj):
        projelem = nloc_elem.find(f'PP_BETA.{iproj+1}')
        l = int(projelem.attrib['angular_momentum'])
        rcut = float(projelem.attrib['cutoff_radius'])
        assert int(projelem.attrib['size']) == gridsize
        projfunc = np.fromiter(map(float, projelem.text.split()), float, count=gridsize)
        projfunc[1:] /= rgrid.rfunc[1:] # function in upf is stored as R(r)*r
        projfunc[0] = projfunc[1] if l==0 else 0.
        l_list.append(l)
        projR_list.append(GridFunc(rgrid, projfunc, l=l, rcut=rcut))
    
    orbital_slices = np.cumsum(2 * np.array(l_list) + 1)
    orbital_slices = np.insert(orbital_slices, 0, 0)
    norbfull = orbital_slices[-1]
    if rel:
        Dij_full = np.zeros((2, 2, norbfull, norbfull), dtype=np.complex128)
    else:
        Dij_full = np.zeros((norbfull, norbfull), dtype=np.float64)
    for iorb in range(len(l_list)):
        l1 = l_list[iorb]
        if rel: j1 = j_list[iorb]
        for jorb in range(len(l_list)):
            l2 = l_list[jorb]
            if rel: j2 = j_list[jorb]
            if (l1 == l2) and ((not rel) or (j1 == j2)):
                if rel:
                    Dij_full[:, :, orbital_slices[iorb]:orbital_slices[iorb+1], 
                                   orbital_slices[jorb]:orbital_slices[jorb+1]] = \
                        Dij[iorb, jorb] * get_dmat_coeffs(l1, j1)
                else:
                    np.fill_diagonal(Dij_full[orbital_slices[iorb]:orbital_slices[iorb+1],
                                              orbital_slices[jorb]:orbital_slices[jorb+1]], Dij[iorb, jorb])
            else:
                assert np.abs(Dij[iorb, jorb]) < 1e-8
    return Dij_full, projR_list