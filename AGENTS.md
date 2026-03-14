# HPRO Project Guidelines

HPRO (Hamiltonian Projection and Reconstruction to atomic Orbitals) is a Python package for converting plane-wave DFT results to atomic orbital basis.

## Build Commands

```bash
# Install from PyPI
pip install hpro

# Install from source
pip install git+https://github.com/Xiaoxun-Gong/HPRO.git

# Install with optional MPI support
pip install hpro[mpi]

# Install with full parallel support (MPI + SLEPc)
pip install hpro[slepc]

# Development install (from repository root)
pip install -e .
```

## Testing

Currently no automated test suite exists. Manual testing workflow:

```bash
# Run the MoS2 demo to verify functionality
cd examples/MoS2_qe_demo/reconstruction
python calc.py        # Hamiltonian reconstruction
python diag.py        # Diagonalization (in aohamiltonian/)
```

## Code Style Guidelines

### Imports

```python
# Standard library imports first
import os
import sys
import json
import time
from collections import namedtuple
from contextlib import contextmanager

# Third-party imports second
import numpy as np
from scipy.sparse import csr_matrix
from scipy.linalg import eigh

# Local imports last
from .utils.mpi import is_master, comm
from .io.deephio import save_mat_deeph
```

### Naming Conventions

- **Classes**: PascalCase (e.g., `PW2AOkernel`, `MatAO`, `AOData`, `GridIntgWorker`)
- **Functions/methods**: snake_case (e.g., `run_pw2ao_rs`, `calc_overlap`, `get_distances`)
- **Private methods**: Prefix with underscore (e.g., `_add_sub`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `HARTREE2EV`, `BOHR2ANG`)
- **Module variables**: lowercase (e.g., `comm`, `periodic_table`)

### Type Hints

Type hints are not currently used in the codebase. When adding them:

```python
def pairs_to_indices(structure: Structure, 
                     translations: np.ndarray, 
                     atom_pairs: np.ndarray) -> np.ndarray:
    ...

def r2k(self, k: np.ndarray) -> csr_matrix:
    ...
```

### Docstrings

Use descriptive docstrings for public APIs:

```python
def run_pw2ao_rs(self, savedir: str, cutoffs=None, analyze_hdecay=True):
    '''
    Convert plane-wave Hamiltonian to atomic orbital basis by integration in real space.

    Parameters:
    ---------
    savedir:         place where result will be saved
    cutoffs:         Dict[str -> float], cutoff radius for each atomic species, in bohr.
    analyze_hdecay:  whether to analyze the dependence of the Hamiltonian matrix elements 
                     on the hopping distance
    '''
```

### Error Handling

- Use `assert` for internal invariants and debugging
- Use `raise ValueError` / `NotImplementedError` for user-facing errors
- MPI-safe error handling with `@mpi_watch` decorator:

```python
from .utils.mpi import mpi_watch

@mpi_watch
def critical_function():
    # If exception raised, all MPI processes abort
    ...
```

### Code Organization

1. **Module structure**:
   - Constants at top
   - Public classes/functions
   - Private helper functions at bottom

2. **Class structure**:
   - `__init__` first
   - `@classmethod` factory methods
   - Public methods
   - Private methods (`_method_name`)
   - `@property` decorators last

3. **Keep functions focused**: Each function should do one thing well

### Numerical Conventions

- **Energies**: Hartree internally, convert to eV for output (use `hartree2ev` constant)
- **Distances**: Bohr internally, Angstrom for VASP/POSCAR I/O
- **Complex arrays**: Use `dtype='c16'` (complex128) for k-space matrices
- **Real arrays**: Use `dtype='f8'` (float64) for real-space matrices
- **Sparse matrices**: Use scipy CSR format for large matrices

### MPI Parallelism

```python
# Check if master process
from .utils.mpi import is_master, comm

if is_master():
    print("Only master prints")

# Distribute work across processes
from .utils.mpi import distrib_vec
rank, count, displ = distrib_vec(total_items)

# Use decorator for MPI-safe methods
@mpi_watch
def parallel_method(self):
    ...
```

### File I/O Patterns

- Use `h5py` for large matrix storage (DeepH format)
- Use `numpy` for simple arrays
- Use `json` for metadata
- Always use context managers for file operations:

```python
with h5py.File(f'{savedir}/hamiltonian.h5', 'w') as f:
    f.create_dataset('entries', data=flatmat)
```

### Adding New Interfaces

When adding support for new DFT codes:

1. Create I/O module in `src/HPRO/io/` (e.g., `newcodeio.py`)
2. Add structure reader to `struio.py`'s `load_structure()`
3. Add orbital reader to `aodata.py` if needed
4. Update `PW2AOkernel.__init__()` to accept new interface types

## Key Files Reference

| File | Purpose |
|------|---------|
| `kernel.py` | Main API: `PW2AOkernel` class |
| `aodiag.py` | Diagonalization: `AODiagKernel` |
| `matao/matao.py` | Matrix storage: `MatAO`, `PairsInfo` |
| `io/deephio.py` | DeepH format read/write |
| `io/aodata.py` | Atomic orbital data handling |
| `utils/mpi.py` | MPI utilities |
