# HPRO Overall Architecture Design

## Project Purpose

HPRO (Hamiltonian Projection and Reconstruction to atomic Orbitals) converts plane-wave DFT calculation results to Hamiltonian matrices in atomic orbital basis.

## Core Workflow

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  DFT Input Data │ ──▶ │  PW2AOkernel    │ ──▶ │  DeepH Output   │
│  (QE/BGW/GPAW)  │     │  (Core Engine)  │     │  (H/S Matrices) │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                               │
                               ▼
                        ┌─────────────────┐
                        │  AODiagKernel   │
                        │  (Verification) │
                        └─────────────────┘
```

## Two Conversion Methods

### Method 1: Wavefunction Projection (`run_pw2ao`)
- Uses complete plane-wave Hamiltonian H(G,G')
- Or uses Bloch wavefunctions for projection
- Applicable when: complete PW data is available

### Method 2: Real-space Reconstruction (`run_pw2ao_rs`)
- Integrates on real-space grid
- Decomposed into: kinetic + local potential + nonlocal potential
- Applicable when: VSC potential files are available (more common)

## Module Dependency Graph

```
kernel.py (Top-level API)
    ├── io/
    │   ├── struio.py    (Crystal structure)
    │   ├── aodata.py    (Atomic orbital basis)
    │   ├── deephio.py   (DeepH format)
    │   ├── bgwio.py     (BerkeleyGW)
    │   └── hrloader.py  (Real-space potential)
    │
    ├── matao/
    │   ├── matao.py     (Matrix storage core)
    │   ├── mataocsr.py  (Sparse format)
    │   └── findpairs.py (Atom pair search)
    │
    ├── pw/
    │   ├── wfndata.py   (Wavefunction data)
    │   └── hgdata.py    (H(G) data)
    │
    ├── v2h/
    │   ├── gridintg.py  (Real-space integration)
    │   ├── twocenter.py (Two-center integrals)
    │   └── vkb.py       (Nonlocal potential)
    │
    └── utils/
        ├── mpi.py       (Parallel)
        ├── structure.py (Structure class)
        └── math.py      (Math utilities)
```

## Key Data Flow

1. **Structure Info**: Read from DFT files → `Structure` object
2. **Atomic Orbitals**: Read from SIESTA/GPAW → `AOData` object
3. **Potential Data**: Read from VSC → Real-space grid data
4. **Matrix Calculation**: Real-space integration → `MatAO` object
5. **Output**: `MatAO` → HDF5 files

## Unit Conventions

| Quantity | Internal Unit | Input/Output |
|----------|---------------|--------------|
| Energy   | Hartree       | eV (convert on output) |
| Length   | Bohr          | Angstrom (VASP) |
| k-points | Reduced coord | Reduced coord |

## DeepH Output Format

New version (HDF5):
- `hamiltonian.h5`: H matrix blocks
- `overlap.h5`: S matrix blocks
- `POSCAR`: Crystal structure
- `info.json`: Metadata
