# utils/ Module Design - Utility Modules

## Responsibility

Provide basic data structures, MPI parallel utilities, math functions and other low-level support.

## Core Modules

### mpi.py - MPI Parallel Support

**Key Variables**:
```python
comm = MPI.COMM_WORLD  # Global communicator
MPI = None             # Module reference (None without mpi4py)
```

**Key Functions**:
```python
def is_master(comm=comm):
    """Check if master process"""
    return comm is None or comm.rank == 0

def distrib_vec(length, displ_last_elem=False):
    """
    Distribute vector elements across processes
    Returns: (rank, count, displ)
    """

def distrib_grps(length, ngrps):
    """
    Divide length elements into ngrps groups
    Returns: (count, displ)
    """
```

**Decorators**:
```python
@mpi_watch
def critical_function():
    """
    MPI-safe decorator
    Any process error → all processes abort
    """
```

**CSR Matrix Communication**:
```python
def mpi_send_csr(comm, matcsr, dest)
def mpi_recv_csr(comm, source)
def mpi_sum_csr(matcsr, comm)  # Logarithmic complexity sum
```

### structure.py - Crystal Structure

```python
class Structure:
    """
    Crystal structure information
    
    Attributes:
        rprim: (3,3) Lattice vectors (Bohr)
        gprim: (3,3) Reciprocal lattice vectors
        atomic_numbers: (natom,) Atomic numbers
        atomic_positions_cart: (natom, 3) Cartesian coords (Bohr)
        atomic_positions_red: (natom, 3) Reduced coords
        efermi: Fermi level (Hartree)
        natom: Number of atoms
        nspc: Number of species
        cell_volume: Unit cell volume
    """
    
    def echo_info():
        """Print structure information"""
```

### math.py - Math Utilities

**k-point Processing**:
```python
def kgrid_with_tr(kgrid):
    """
    Reduce k-points using time-reversal symmetry
    k → -k keep only one
    """

def make_kkmap(kpts_old, kpts_new):
    """Build k-point index mapping"""
```

**G-vectors**:
```python
def kGsphere(rprim, ecut):
    """
    Generate all G-vectors within cutoff energy
    
    Returns:
        ng, g_g, kgcart
    """
```

**Spherical Harmonics Related**:
```python
def get_dmat_coeffs(l, j):
    """
    Get spin-orbit coupling coefficient matrix
    Used for relativistic pseudopotentials
    """
```

### orbutils.py - Orbital Utilities

**Radial Grid**:
```python
class LinearRGD:
    """Linear radial grid r = r0 + i*dr"""
    
    def integrate(self, f):
        """Integral ∫ f(r) r² dr"""
```

**Grid Function**:
```python
class GridFunc:
    """
    Function on radial grid + angular momentum info
    
    Attributes:
        rgd: Radial grid
        func: Function values f(r)
        l: Angular momentum
        rcut: Cutoff radius
    """
    
    def getval3D(self, rvec):
        """
        Evaluate at 3D coordinates (includes spherical harmonics)
        f(r) * Y_lm(θ,φ)
        """
    
    def getval1D(self, r):
        """Radial part only"""
```

**Fourier Transform**:
```python
def grid_R2G(gridQ, phirgrid):
    """
    Real space → Reciprocal space
    
    φ(G) = ∫ φ(r) j_l(Gr) r² dr
    """
```

### misc.py - Miscellaneous Utilities

**Progress Bar**:
```python
class mytqdm(tqdm):
    """MPI-safe progress bar, only shows on master process"""
```

**Timer**:
```python
@simple_timer('Job done, time = {t}')
def long_function():
    """Auto-print elapsed time"""

class Timer:
    """Manual timer"""
```

**Atomic Symbol Conversion**:
```python
def atom_name2number(element_names)  # ['Mo', 'S'] -> [42, 16]
def atom_number2name(atomic_numbers) # [42, 16] -> ['Mo', 'S']
```

**Namedtuple**:
```python
KGData = namedtuple('KGData', ['ng', 'nbnd', 'miller_idc', 'unkg', 'kgcart'])
VKBGData = namedtuple('VKBGData', ['ng', 'miller_idc', 'vkbg', 'kgcart'])
```

### supercell.py - Supercell Processing

```python
class OrbInfo:
    """
    Orbital information in supercell
    Used for real-space integration
    """
```

## Design Decisions

1. **Optional MPI**: Automatically falls back to serial without mpi4py
2. **Global Communicator**: Use `comm` module variable, avoid passing everywhere
3. **Lazy Import**: GPAW/ASE related code imported only when needed
4. **Namedtuple**: Lightweight data structures for simple data passing
