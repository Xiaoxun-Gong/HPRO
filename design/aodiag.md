# aodiag.py Module Design - Atomic Orbital Hamiltonian Diagonalization

## Responsibility

Diagonalize Hamiltonian in atomic orbital basis to calculate band structures for verification of reconstruction results.

## Core Class: AODiagKernel

```python
class AODiagKernel:
    """
    Atomic orbital Hamiltonian diagonalization
    
    Attributes:
        matH, matS: Hamiltonian and Overlap matrices (MatAO objects)
        nk: Number of k-points
        kpts: k-point coordinates
        eigs: Eigenvalues (Hartree)
        wfnao: Eigenvectors
    """
```

## Workflow

```python
# 1. Set k-point path
kernel.setk(kpts, kptwts, kptsymbol, type='path')

# 2. Load matrices
kernel.load_deeph_mats(folder)

# 3. Diagonalize
kernel.diag(nbnd, efermi=None)

# 4. Output
kernel.write(path)
```

## k-point Path Setup

```python
def setk(kpts, kptwts, kptsymbol, type='path'):
    """
    type='path': Standard band path
        kpts = [G, K, M, G]  # High-symmetry points
        kptwts = [20, 10, 17, 1]  # Points per segment
        kptsymbol = ['G', 'K', 'M', 'G']
    
    type='grid': Uniform k-point grid
        kpts = Uniformly distributed k-points
    
    type='path_siesta': SIESTA format path
    """
```

**Path Interpolation**: Linear interpolation between high-symmetry points

## Diagonalization Methods

### Method 1: SLEPc (Recommended, MPI Support)

```python
# Use SLEPc's Krylov-Schur method
# Supports shift-and-invert to find states near specific energy

if use_slepc4py:
    eigs_k, vecs_k = diag_slepc(
        Hpetsc, Spetsc, nbnd, 'TR', 
        sigma=efermi,  # Target energy
        comm=comm_pool
    )
```

**Parallelization Strategy**:
- k-points distributed to different process pools (nkpools)
- Multiple processes within each pool share matrices

### Method 2: SciPy ARPACK (Single Process)

```python
# Use scipy.sparse.linalg.eigsh
if efermi is None:
    eigs_k, vecs_k = eigsh(Hk, k=nbnd, M=Sk, which='SR')
else:
    eigs_k, vecs_k = eigsh(Hk, k=nbnd, M=Sk, sigma=efermi)
```

## Ill-conditioned Overlap Matrix Handling

```python
# ill_project option: Project out ill-conditioned subspace of S
if ill_project:
    # 1. Orthogonalize eigenvectors
    Q, _ = qr(vecs_k)
    # 2. Check condition number of S in eigenspace
    Sk_sub = Q.T.conj() @ Sk @ Q
    eigs_Sk, vecs_Sk = eigh(Sk_sub)
    # 3. Project out subspace with small eigenvalues
    project_index = np.where(np.abs(eigs_Sk) > ill_threshold)[0]
    # 4. Re-diagonalize
```

## Matrix Transformation: R → k

```python
def r2k(k):
    """
    Real-space matrix → k-space matrix
    
    H(k) = Σ_R H(R) * exp(2πi * k·R)
    
    Implemented using sparse matrix addition
    """
```

## Output Format (eig.dat)

```
Band energies in eV
      nk    nbnd
      48       36
 0.000000000  0.000000000  0.000000000      36  G
 1  1   -6.123456789
 1  2   -4.567890123
 ...
```

## Parallelization Strategy

```
              ┌─────────────┐
              │ Total Proc N│
              └─────────────┘
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
    ┌───────┐   ┌───────┐   ┌───────┐
    │Pool 0 │   │Pool 1 │   │Pool 2 │ ...  (nkpools)
    │k:0-15 │   │k:16-31│   │k:32-47│
    └───────┘   └───────┘   └───────┘
        │
    ┌───┴───┐
    ▼       ▼
  ┌───┐   ┌───┐
  │P0 │   │P1 │  (Multiple processes per pool share matrices)
  └───┘   └───┘
```

## Design Decisions

1. **SLEPc Priority**: Preferred for large-scale problems, supports MPI parallel
2. **k-point Parallelism**: k-points are independent, naturally parallel
3. **Shift-and-invert**: Fast finding of states near Fermi level
4. **Ill-conditioning Handling**: Optional subspace projection
