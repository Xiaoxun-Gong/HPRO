# HPRO: Hamiltonian Projection and Reconstruction to atomic Orbitals

[![PyPI version](https://badge.fury.io/py/hpro.svg)](https://badge.fury.io/py/hpro)
[![Python](https://img.shields.io/pypi/pyversions/hpro.svg)](https://pypi.org/project/hpro/)
[![License](https://img.shields.io/badge/License-GPL%203.0-blue.svg)](https://opensource.org/licenses/GPL-3.0)
[![GitHub](https://img.shields.io/badge/GitHub-Xiaoxun--Gong/HPRO-blue.svg)](https://github.com/Xiaoxun-Gong/HPRO)

**HPRO** is a Python package for converting plane-wave DFT calculation results to Hamiltonian matrices in atomic orbital basis. It serves as a bridge between first-principles calculations and tight-binding models or machine learning applications like [DeepH](https://github.com/m3gOfHeads/DeepH-pack).

## Features

- **Two conversion methods**: Wavefunction projection and real-space reconstruction
- **Multiple DFT interfaces**: Quantum ESPRESSO, BerkeleyGW, GPAW, SIESTA
- **MPI parallelization**: Scalable calculations on HPC clusters
- **DeepH format output**: Direct compatibility with DeepH-pack

## Installation

### Basic Installation

```bash
pip install hpro
```

Or install from source:

```bash
pip install git+https://github.com/Xiaoxun-Gong/HPRO.git
```

### With MPI Support

```bash
pip install hpro[mpi]
```

### With Full Parallel Support (MPI + SLEPc)

```bash
pip install hpro[slepc]
```

> **Note**: SLEPc requires PETSc to be installed with MKL Pardiso. See the [SLEPc documentation](https://slepc.upv.es/documentation/) for details.

## Quick Start

```python
from HPRO import PW2AOkernel

# Create kernel for real-space reconstruction
kernel = PW2AOkernel(
    aodata_interface='siesta',      # SIESTA orbital files
    aodata_root='./aobasis',
    hrdata_interface='qe-bgw',      # QE + BerkeleyGW
    vscdir='./scf/VSC',
    upfdir='./pseudos',
    ecutwfn=30
)

# Run reconstruction
kernel.run_pw2ao_rs('./output')
```

For diagonalization and band structure calculation:

```python
from HPRO.aodiag import AODiagKernel

diag = AODiagKernel()
diag.setk(kpts, weights, symbols)
diag.load_deeph_mats('./output')
diag.diag(nbnd=36)
diag.write('./output')
```

## Supported DFT Codes

| Code | Input | Output |
|------|-------|--------|
| Quantum ESPRESSO | Wavefunctions, potential | ✓ |
| BerkeleyGW | WFN, VSC, VKB | ✓ |
| GPAW | .gpw files | ✓ |
| SIESTA | .ion files, H/S matrices | ✓ |
| VASP | POSCAR structure | - |

## Documentation

- [AGENTS.md](AGENTS.md) - Development guidelines
- [design/](design/) - Module design documentation

## Requirements

- Python >= 3.9
- numpy >= 1.7
- scipy
- h5py
- tqdm
- matplotlib

## Citation

If you use HPRO in your research, please cite:

```bibtex
@software{hpro2024,
  author = {Gong, Xiaoxun},
  title = {HPRO: Hamiltonian Projection and Reconstruction to atomic Orbitals},
  year = {2024},
  url = {https://github.com/Xiaoxun-Gong/HPRO}
}
```

## License

GNU General Public License v3.0 or later (GPL-3.0-or-later)

## Author

Xiaoxun Gong (xiaoxun.gong@berkeley.edu)
