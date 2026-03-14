# pwproj.py Module Design - Plane-wave Projection Algorithm

## Responsibility

Project Hamiltonian or wavefunctions from plane-wave basis to atomic orbital basis.

## Core Algorithms

### Method 1: Complete PW Hamiltonian Projection

```python
def Hpw_to_Hao_HG(structure, hgdata, aodata, pairs_ij, h_or_s='h'):
    """
    Project using complete H(G,G')
    
    H_αβ = (1/Ω) Σ_k w_k Σ_{G,G'} φ_α*(k+G) H_{G,G'}(k) φ_β(k+G')
    
    Steps:
    1. For each k-point:
       a. Build PW Hamiltonian block H_{G1,G2}(k)
       b. Calculate Fourier transform of orbitals φ(k+G)
       c. Matrix multiplication for contribution
    2. Sum over all k-points
    3. Unfold using Hermiticity
    """
```

**Optimization**:
- Hopping for same atom pairs calculated once (different translations differ only by phase)
- MPI distributes by G-vectors

### Method 2: Wavefunction Projection

```python
def Hpw_to_Hao_wfn(structure, wfndata, aodata, pairs, 
                   wfn_proj_atoms, nbndmin, nbndmax):
    """
    Project using Bloch wavefunctions
    
    H_αβ = Σ_k w_k Σ_n ⟨φ_α|ψ_nk⟩ ε_nk ⟨ψ_nk|φ_β⟩
         = Σ_k w_k φ_α*(k) H_band(k) φ_β(k)
    
    Where H_band(k) = diag(ε_1, ε_2, ..., ε_n) is diagonal
    """
```

### Wavefunction Projection Precomputation

```python
def wfn_proj_to_atoms(structure, aodata, wfndata, nbndmin, nbndmax):
    """
    Precompute ⟨φ_α|ψ_nk⟩ for all atoms and orbitals
    
    Returns: List[array(nk, nbnd, norb)] length=natom
    
    Calculation:
    ⟨φ_α|ψ_nk⟩ = (1/√Ω) Σ_G φ_α*(k+G) ψ_nk(G) exp(-ik·r_α)
    """
```

## Key Data Flow

```
              PW Data
                 │
     ┌───────────┼───────────┐
     ▼           ▼           ▼
  HGData     WFNData      Both
     │           │           │
     ▼           ▼           │
Hpw_to_Hao_HG  Hpw_to_Hao_wfn
     │           │           │
     └───────────┼───────────┘
                 ▼
              MatAO
```

## Hermiticity Handling

```python
# Store only half: (i,j) not (j,i)
pairs.remove_ji()

# Unfold after calculation
hoppings.unfold_with_hermiticity()
# Auto-generate (j,i) = (i,j)^T*
```

## Time-reversal Symmetry

```python
# H(-k) = H*(k)
# Therefore only need half the k-points
# Final result takes real part
hoppings.mats[ipair] = hoppings.mats[ipair].real
```

## MPI Parallel Strategy

### G-vector Distribution
```python
rank, count, displ = distrib_vec(ng)
# Each process calculates G[displ[rank]:displ[rank+1]]
```

### Result Collection
```python
# Method 1: Gather
hoppings.mpi_gather(displ, dtype=MPI.COMPLEX16, root=0)

# Method 2: Reduce (for k-point summation)
hoppings.mpi_reduce(dtype=MPI.COMPLEX16, op=MPI.SUM, root=0)
```

## Design Decisions

1. **Two Methods Available**: HGData (more accurate) or WFNData (more flexible)
2. **Phase Optimization**: Core part calculated once for same atom pairs
3. **Hermiticity**: Calculate half, store half, unfold on output
4. **MPI-friendly**: Both G-vectors and k-points easily parallelized
