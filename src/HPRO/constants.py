import numpy as np

hartree2ev = 27.211386024367243
bohr2ang = 0.5291772105638411

TWOCENTER_RGRID_DEN = 100. / 12. # 100 points at 12 bohr
AOFT_QGRID_DEN = 1 / np.sqrt(500) * 2048 # 2048 points at 500 Hartree

hpro_rng = np.random.default_rng(seed=42)
