# io/ Module Design - Input/Output Interfaces

## Responsibility

Handle reading/writing of various DFT software formats, including crystal structures, atomic orbitals, and potential data.

## Core Modules

### aodata.py - Atomic Orbital Data

```python
class AOData:
    """
    Store atomic orbital basis set information
    
    Attributes:
        structure: Associated crystal structure
        ls_spc: Dict[int -> List[int]], angular momentum list for each species
        phirgrids_spc: Dict[int -> List[GridFunc]], radial wavefunctions
        norbfull_spc: Dict[int -> int], total orbitals per species
        cutoffs: Cutoff radii
        
    Supported formats:
        - 'siesta': .ion files
        - 'gpaw': .basis files
        - 'deeph': Read orbital type info only
    """
```

**GridFunc Class** (orbutils.py):
```python
class GridFunc:
    """Radial grid function + angular momentum info"""
    def getval3D(rvec):
        """Evaluate at 3D coordinates (includes spherical harmonics)"""
    def getval1D(r):
        """Radial part only"""
```

**Fourier Transform**:
```python
def calc_phiQ(Ecut):
    """
    Calculate Fourier transform of orbital functions
    Used for wavefunction projection method
    """
```

### struio.py - Crystal Structure

**Unified Loading Interface**:
```python
def load_structure(path, interface):
    """
    Supported interfaces:
    - 'qe': Quantum ESPRESSO XML
    - 'bgw': BerkeleyGW (WFN/VSC/VKB)
    - 'vasp': POSCAR
    - 'gpaw': .gpw file
    - 'deeph': POSCAR + info.json
    """
```

**Structure Class** (utils/structure.py):
```python
class Structure:
    rprim: (3,3)        # Lattice vectors (Bohr)
    atomic_numbers: (natom,)  # Atomic numbers
    atomic_positions_cart: (natom, 3)  # Cartesian coords (Bohr)
    efermi: float       # Fermi level (Hartree)
```

### deephio.py - DeepH Format

**Saving**:
```python
def save_mat_deeph(savedir, matao, filetype):
    """
    filetype: 'h' (hamiltonian) or 'o' (overlap)
    
    Output files:
    - hamiltonian.h5 / overlap.h5
    - POSCAR
    - info.json
    """
```

**HDF5 Structure**:
```
hamiltonian.h5:
├── atom_pairs: (npairs, 5) [R1,R2,R3,i,j]
├── chunk_boundaries: (npairs+1,)
├── chunk_shapes: (npairs, 2)
└── entries: Flattened matrix elements
```

**Loading**:
```python
def load_mat_deeph(folder, filetype):
    """Returns MatAO object"""
```

### bgwio.py - BerkeleyGW Format

Read binary format WFN/VSC/VKB files:
- WFN: Wavefunctions
- VSC: Self-consistent potential
- VKB: KB projectors

### hrloader.py - Real-space Potential

```python
def read_vloc(vscdir, interface):
    """Read local potential V_loc(r)"""
    
def read_vnloc(structure, upfdir, interface):
    """
    Read nonlocal potential parameters
    Returns: (Dij, Qij, projR)
    - Dij: Nonlocal potential matrix elements
    - Qij: PAW overlap correction
    - projR: Projector functions (AOData format)
    """
```

### siestaio.py - SIESTA Format

```python
def get_hs_siesta(indir, sysname):
    """
    Read SIESTA calculated H/S matrices
    Used for siesta2deeph conversion
    """
```

## Supported File Formats

| Data Type | Format | Read | Write |
|-----------|--------|------|-------|
| Crystal Structure | QE XML | Yes | - |
| Crystal Structure | POSCAR | Yes | Yes |
| Crystal Structure | BGW Binary | Yes | - |
| Atomic Orbitals | SIESTA .ion | Yes | - |
| Atomic Orbitals | GPAW .basis | Yes | - |
| Hamiltonian | DeepH HDF5 | Yes | Yes |
| Potential | BGW VSC | Yes | - |
| Pseudopotential | UPF | Yes | - |

## Design Decisions

1. **Unified Interface**: All formats unified through `load_xxx()` functions
2. **Lazy Loading**: Large files read on demand
3. **Format Auto-detection**: Parser auto-selected by file extension
