# pw/ Module Design - Plane-wave Data Processing

## Responsibility

Process Hamiltonian and wavefunction data in plane-wave basis.

## Core Classes

### HGData - H(G) Data (Norm-Conserving PP)

```python
class HGData:
    """
    Complete Hamiltonian in plane-wave basis H(G,G')
    
    Attributes:
        vscg: Fourier components of local potential V(G)
        deeq: KB projector coefficients d_{hh'}
        vkbgdatas: List[VKBGData], data for each k-point
        ecutwfn: Cutoff energy
    """
    
    def build_hamblock(ik, gvecrange1, gvecrange2):
        """
        Build PW Hamiltonian block
        
        H_{G1,G2}(k) = T_{G1,G2} + V_loc(G1-G2) + V_nloc(k)
        
        Returns: (ng1, ng2) complex matrix
        """
```

**VKBGData** (namedtuple):
```python
VKBGData(
    ng,           # Number of G-vectors
    miller_idc,   # (ng, 3) Miller indices
    vkbg,         # Dict[int -> (nh, ng)] KB projector
    kgcart        # (ng, 3) k+G vectors (Cartesian)
)
```

### HGDataPAW - PAW Data

```python
class HGDataPAW(HGData):
    """
    Hamiltonian and overlap matrix for PAW method
    
    Additional attributes:
        FFTngf: Fine FFT grid
        vscg_full: Complete potential Fourier components
        cdij: D_ij for each atom
        cqij: Q_ij for each atom (overlap correction)
    """
    
    def build_hs_paw(ik, kind='h'):
        """
        Build H or S matrix
        kind='h': Hamiltonian
        kind='s': Overlap matrix
        """
```

### WFNData - Wavefunction Data

```python
class WFNData:
    """
    Wavefunctions in plane-wave basis
    
    Attributes:
        kgdatas: List[KGData], data for each k-point
        nbnd: Number of bands
        kpts, kptwts: k-points and weights
    """
    
    def get_H_band_basis(nbndmin, nbndmax):
        """
        Get Hamiltonian in band basis
        
        H_{nm}(k) = ε_n(k) δ_{nm}
        
        Returns: (nk, nbnd, nbnd) diagonal matrix
        """
```

**KGData** (namedtuple):
```python
KGData(
    ng,           # Number of G-vectors
    nbnd,         # Number of bands
    miller_idc,   # (ng, 3)
    unkg,         # (nbnd, ng) Wavefunction coefficients
    kgcart        # (ng, 3)
)
```

## Supported Interfaces

| Data Type | Interface | Description |
|-----------|-----------|-------------|
| H(G) | 'bgw' | BerkeleyGW VSC+VKB |
| H(G) | 'gpaw' | GPAW .gpw file |
| Wavefunction | 'qe' | QE wfc*.hdf5 |
| Wavefunction | 'bgw' | BerkeleyGW WFN |

## Design Decisions

1. **G-vector Distribution**: MPI parallel by distributing G-vectors
2. **Memory Optimization**: Process large matrices in blocks, avoid loading all at once
3. **k-point Reduction**: Use time-reversal symmetry to reduce k-points
