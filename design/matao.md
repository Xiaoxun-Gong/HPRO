# matao/ Module Design - Matrix Storage and Atom Pair Management

## Responsibility

Manage matrix storage in atomic orbital basis and handle atom pairs with translation vectors.

## Core Classes

### PairsInfo - Atom Pair Information

```python
class PairsInfo:
    """
    Store atom pairs and their translation vectors
    
    Convention:
    - First atom in unit cell at (0,0,0)
    - Second atom in translated unit cell at (R1,R2,R3)
    
    Attributes:
        npairs: Number of atom pairs
        translations: (npairs, 3) translation vectors
        atom_pairs: (npairs, 2) atom index pairs
    """
```

**Key Methods**:
- `sort()`: Sort by unique index
- `remove_ji()`: Remove redundant (j,i) pairs, keep only (i,j)
- `unfold_with_hermiticity()`: Generate (j,i) pairs from (i,j)
- `get_distances()`: Calculate real-space distances for all pairs

**Index Calculation** (`pairs_to_indices`):
```python
index = (spc1*200 + spc2) * BASE^3 * natom^2
      + (atm1*natom + atm2) * BASE^3
      + (R1+BASE/2)*BASE^2 + (R2+BASE/2)*BASE + (R3+BASE/2)
```
Ensures sorting order: atomic number → iatom → jatom → R1 → R2 → R3

### MatAO - Atomic Orbital Matrix

```python
class MatAO(PairsInfo):
    """
    Store matrix in LCAO basis
    
    Stored in blocks by atom pairs, each block is a matrix
    
    Attributes:
        mats: List[array] matrix block list
        aodata1: Left basis information
        aodata2: Right basis information (usually equals aodata1)
        spinful: Whether spin-polarized
    """
```

**Key Methods**:
- `r2k(k)`: Real-space matrix → k-space matrix
  ```
  H(k) = Σ_R H(R) * exp(2πi * k·R)
  ```
- `to_csr()`: Convert to sparse CSR format
- `hermitianize()`: Hermitianize H = (H + H†)/2
- `spinless_to_spinful()`: Spinless → spinful (Kronecker product)

**Matrix Operations**:
- `__add__`, `__sub__`: Matrix addition (supports different pair sets)
- `fillvalue()`: Fill current matrix with values from another

### MatAOCSR - Sparse Format

Used for efficient storage during real-space integration.

## MPI Parallel Support

```python
# Gather matrix blocks to master process
hoppings.mpi_gather(displ, dtype=MPI.COMPLEX16, root=0)

# Reduce matrices from all processes
hoppings.mpi_reduce(dtype=MPI.COMPLEX16, op=MPI.SUM, root=0)
```

## findpairs.py

### pairs_within_cutoff()
```python
def pairs_within_cutoff(structure, cutoffs):
    """
    Find all atom pairs within cutoff radius
    
    Parameters:
        structure: Crystal structure
        cutoffs: Dict[int -> float], cutoff radius for each species (Bohr)
    
    Returns:
        PairsInfo object
    """
```

## Design Decisions

1. **Block Storage**: Stored by atom pairs for parallel processing
2. **Lazy Sparsification**: Convert to CSR format only when needed
3. **Hermiticity Optimization**: Store only half, unfold on output
4. **Sorting Consistency**: Sort before all operations for index matching
