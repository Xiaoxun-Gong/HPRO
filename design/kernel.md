# kernel.py Module Design

## Responsibility

Top-level API that coordinates all submodules to perform PW→AO conversion.

## Core Class: PW2AOkernel

### Initialization Parameters

```python
PW2AOkernel(
    # Atomic orbital data source
    aodata_interface='siesta',  # 'siesta', 'gpaw', 'deeph'
    aodata_root='./aobasis',
    
    # Plane-wave data source (Method 1)
    wfn_interface='qe',         # 'qe', 'bgw'
    wfndata_root='./scf',
    
    # H(G) data source (Method 1)
    hgdata_interface='bgw',     # 'bgw', 'gpaw'
    vscdir='./VSC',
    vkbdir='./VKB',
    
    # Real-space potential source (Method 2)
    hrdata_interface='qe-bgw',  # 'qe-bgw', 'qe-deephr', 'gpaw'
    upfdir='./pseudos',
    ecutwfn=30,                 # Cutoff energy (Hartree)
    
    # Structure info (usually auto-detected)
    structure_interface=None,
    structure_path=None,
)
```

### Two Core Methods

#### run_pw2ao() - Wavefunction Projection
```python
def run_pw2ao(savedir, cutoffs=None, nbndmin=None, nbndmax=None):
    """
    Project using H(G) or wavefunctions
    
    Flow:
    1. If hgdata exists → Hpw_to_Hao_HG() (full PW Hamiltonian)
    2. If wfndata exists → Hpw_to_Hao_wfn() (wavefunction projection)
    """
```

#### run_pw2ao_rs() - Real-space Reconstruction
```python
def run_pw2ao_rs(savedir, cutoffs=None):
    """
    Reconstruct via real-space integration
    
    Flow:
    1. Calculate overlap: S = S_basis + S_nloc(PAW)
    2. Calculate Hamiltonian: H = H_kin + H_loc + H_nloc
       - H_kin: Two-center integral (kinetic)
       - H_loc: Grid integration (local potential)
       - H_nloc: KB projector integral (nonlocal potential)
    """
```

## Helper Functions

### siesta2deeph()
Convert SIESTA H/S matrices directly to DeepH format, bypassing PW steps.

## Design Decisions

1. **Lazy Initialization**: Data loaded only when needed
2. **Unified Interface**: All DFT software through unified parameter interface
3. **Auto Structure Detection**: Crystal structure inferred from existing data
4. **MPI Support**: Key methods decorated with `@mpi_watch`

## Typical Usage

```python
# Real-space reconstruction (common)
kernel = PW2AOkernel(
    aodata_interface='siesta',
    aodata_root='../aobasis',
    hrdata_interface='qe-bgw',
    vscdir='../scf/VSC',
    upfdir='../pseudos',
    ecutwfn=30
)
kernel.run_pw2ao_rs('./output')

# Wavefunction projection
kernel = PW2AOkernel(
    aodata_interface='siesta',
    aodata_root='../aobasis',
    wfn_interface='qe',
    wfndata_root='../scf',
    hgdata_interface='bgw',
    vscdir='../VSC',
    vkbdir='../VKB',
)
kernel.run_pw2ao('./output')
```
