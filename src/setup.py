import os
import re
from setuptools import Extension, find_packages, setup

with open('HPRO/__init__.py', 'r') as f:
    txt = f.read()
version = re.search("__version__ = '(.*)'", txt)[1]

setup(name='HPRO',
      version=version,
      description='Hamiltonian Projection and Reconstruction to atomic Orbitals',
    #   long_description='long_description',
      maintainer='Xiaoxun Gong',
      maintainer_email='xiaoxun.gong@berkeley.edu',
    #   url='url',
    #   license='license',
    #   platforms=['unix'],
      packages=find_packages(
          where='./',
          include=['HPRO', 'HPRO.from_gpaw']
      ),
      package_data={"HPRO": ["periodic_table.json"]},
    #   entry_points={
    #       'console_scripts': ['hello-world = timmins:hello_world'],
    #       },
    #   setup_requires=['numpy>=1.7',
    #                   'cython>=3.0'],
      install_requires=['numpy>=1.7',
                        'scipy',
                        'tqdm',
                        'h5py',
                        'matplotlib'],
      extras_require={'mpi': ['mpi4py'],
                      'slepc': ['mpi4py', 'petsc4py', 'slepc4py']},
    #   ext_modules=extensions,
    #   scripts=scripts,
    #   cmdclass=cmdclass
      )