# v2h/ Module Design - Potential to Hamiltonian Conversion

## Responsibility

Convert real-space potential to Hamiltonian matrix elements in atomic orbital basis.

## Core Algorithm: Real-space Grid Integration

### GridIntgWorker - Grid Integration Worker

```python
class GridIntgWorker:
    """
    Integrate on real-space grid: H_αβ = ∫ φ_α(r) V(r) φ_β(r) dr
    
    Optimization strategies:
    1. Coarse grid: Group fine grid into coarse grid blocks
    2. KDTree: Fast search for orbitals near grid points
    3. MPI distribution: Distribute coarse grid points across processes
    """
    
    def __init__(self, structure, gridsize, aodata):
        """
        Preparation phase:
        1. Determine coarse grid size (auto-select nsubdiv)
        2. Build supercell orbital index
        3. Build KDTree for each atomic species
        4. Precompute orbital values at each coarse grid point
        """
    
    def gridintg(self, func):
        """
        Integration phase:
        
        for each coarse grid point:
            1. Get orbital values at this point: fosc_thispoint
            2. Get potential value f (from func array)
            3. Calculate intg = φ^T @ (f * dvol * φ)
            4. Map orbital pairs back to unit cell
            5. Accumulate to CSR matrix
        
        Returns: MatAOCSR object
        """
```

### Two-center Integrals - twocenter.py

```python
def calc_overlap(aodata1, aodata2=None, Ecut, kind):
    """
    Calculate matrix elements via two-center integrals
    
    kind=1: Overlap matrix S_αβ = ∫ φ_α(r) φ_β(r) dr
    kind=2: Kinetic matrix T_αβ = ∫ φ_α(r) (-∇²/2) φ_β(r) dr
    
    Algorithm:
    1. Fourier transform to reciprocal space
    2. Use Parseval's theorem
    3. Numerical integration
    """
```

### Nonlocal Potential - vkb.py

```python
def calc_vkb(olp_proj_ao, Dij, spinful=False):
    """
    Calculate nonlocal potential contribution
    
    V_nloc = Σ_a |β_a⟩ D_ij ⟨β_j|
    
    Parameters:
        olp_proj_ao: Overlap between projector and orbital ⟨β|φ⟩
        Dij: Nonlocal potential coefficient matrix
    
    Returns: MatAO object
    """

def get_nloc_pairs(structure, projR_cutoffs, ao_cutoffs):
    """
    Find all atom pairs requiring nonlocal potential calculation
    
    Atom pair (i,j) needs calculation when:
    - i is within projector cutoff OR
    - j is within projector cutoff
    """
```

## Hamiltonian Decomposition

```
H = H_kin + H_loc + H_nloc

H_kin (kinetic):     Two-center integral
H_loc (local):       Grid integration
H_nloc (nonlocal):   KB projector integral
```

## Overlap Matrix Decomposition

```
S = S_basis + S_nloc (PAW only)

S_basis:  Two-center integral
S_nloc:   Q_ij projector correction (PAW only)
```

## Numerical Precision Control

| Parameter | Description | Location |
|-----------|-------------|----------|
| GRIDINTG_NSUBDIV_RANGE | Coarse grid subdivision range | config.py |
| TWOCENTER_RGRID_DEN | Radial grid density | config.py |
| AOFT_QGRID_DEN | Fourier transform grid density | config.py |

## Design Decisions

1. **Two-phase Calculation**: Precompute orbitals → integrate, reduce redundant computation
2. **KDTree Acceleration**: O(log n) lookup for nearby orbitals
3. **Sparse Storage**: Results stored directly in CSR matrix
4. **MPI Load Balancing**: Coarse grid points evenly distributed
