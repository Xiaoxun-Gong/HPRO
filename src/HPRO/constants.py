import os
import numpy as np

hartree2ev = 27.211386024367243
bohr2ang = 0.5291772105638411

hpro_rng = np.random.default_rng(seed=42)

OPENMX_VERSION = '3.9'
OPENMX_BASIS_TABLE = os.path.join(os.path.dirname(__file__), 'from_openmx', 'opmx_basis.txt')
OPENMX_LINEAR_RGRID_DEN = 100.0


def _load_openmx_basis_specs_2019(filename=OPENMX_BASIS_TABLE):
    """
    Load OpenMX 3.9 DFT_DATA19 PAO Quick/Standard/Precise recommendations.
    """
    basis_specs = {'quick': {}, 'standard': {}, 'precise': {}}
    with open(filename, 'r') as basis_file:
        for lineno, line in enumerate(basis_file, start=1):
            fields = line.split()
            if not fields or fields[0].startswith('#'):
                continue
            if len(fields) < 7:
                raise ValueError(f'Invalid OpenMX basis table line {lineno}: {line.rstrip()}')

            element, atomic_number = fields[0], int(fields[1])
            if atomic_number <= 0:
                continue

            basis_specs['quick'][element] = fields[4]
            basis_specs['standard'][element] = fields[5]
            basis_specs['precise'][element] = fields[6]
    return basis_specs


OPENMX_BASIS_SPECS_2019 = _load_openmx_basis_specs_2019()

OPENMX_L_LABELS = {'s': 0, 'p': 1, 'd': 2, 'f': 3, 'g': 4, 'h': 5}
