from math import sqrt

# Spherical harmonics
SPHERICAL_HARMONICS_LMAX = 10  # Maximum angular momentum for spherical harmonics

# pairs info
BASE_TRANSLATIONS = 20 # Assume all primitive translations are within [-BASE//2, BASE//2)

# two-center integrals
TWOCENTER_RGRID_DEN = 100. / 12. # 100 points at 12 bohr
AOFT_QGRID_DEN = 1 / sqrt(500) * 2048 # 2048 points at 500 Hartree

# mataocsr to matao
DEBUG_CSR_CNVRT = False

# real-space integrals
GRIDINTG_NSUBDIV_RANGE = (13, 20)
GRIDINTG_USE_HERMITICITY = False
GRIDINTG_ENABLE_TQDM = True

# use new deeph data interface
DEEPH_USE_NEW_INTERFACE = True