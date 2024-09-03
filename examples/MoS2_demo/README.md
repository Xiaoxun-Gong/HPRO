# MoS2 demo

Author: Xiaoxun Gong (xiaoxun.gong@berkeley.edu) 

In this demo, you will be guided to perform a PW DFT calculation using Quantum ESPRESSO (QE), use the output to reconstruct the corresponding DFT Hamiltonian under AO basis, diagonalize it, and then compare to the QE band structures.

## System environment

1. Quantum ESPRESSO package version 7+, with compiled `pw.x` and `pw2bgw.x` (this `pw2bgw.x` executable can be compiled alongside `pp.x` with command `make pp`.) Although recommended, QE is not mandatory to run this demo.
2. Install the package `HPRO` following the `README.md` in the top directory.

## 1 SCF calculation

If you don't have QE installation, you can skip this step. All the reference output are contained in the `scf_ref` folder. Remove `scf` folder and rename `scf_ref` folder to `scf` in order to proceed to the next step. If you have QE installation:

1. Go to `scf` folder. Read the contents of `pw.in`. Then run command `pw.x -in pw.in > pw.out`. You can also use `mpirun` if QE is compiled with MPI.
2. After the scf calculation finishes, run command `pw2bgw.x -in pw2bgw.in > pw2bgw.out`. This step gets `VSC` from QE calculation output, which is the total effective local potential required to build the Hamiltonian.

## 2 Band calculation

If you don't have QE installation, you can skip this step. All the required output are contained in the `bands_ref` folder. Remove `bands` folder, and rename `bands_ref` folder to `bands` in order to proceed to the next step. If you have QE installation:

1. Go to `bands/MoS2.save` folder. Copy `data-file-schema.xml` file from `scf/MoS2.save` folder here. 
2. Stay in the same folder. The `charge-density.hdf5` file in this folder is a soft link from the scf calculated charge density file. However, if your QE is not compiled with HDF5 library, you will need to link by yourself by executing `ln -s ../../scf/MoS2.save/charge-density.dat . ` because the charge density file now has a different name.
3. Go back to `bands` folder. Read the contents of `pw.in`, and run `pw.x -in pw.in > pw.out`.
4. After calculation finishes, go to `bands/MoS2.save` folder and run `python qe_getband.py`. You should have `band.json` in this folder right now.

## 3 Reconstruction

1. Go to `reconstruction` folder. Read the contents of `calc.py`. Then run `python calc.py`. You can also use `mpirun -np xx python calc.py` if you have installed `mpi4py`.
2. You should find all the results in `aohamiltonian` folder.

## 4 Compare bands

1. Go to `reconstruction/aohamiltonian` folder. Run `python diag.py` to diagonalize the Hamiltonian you have obtained. You can also use `mpirun` if `mpi4py` is installed. After the run finishes, a file `eig.dat` will be generated which will store all the eigenvalues at different k points.
2. Run `python plotband.py` in `aohamiltonian` folder. This script will read `band.json` from `bands/MoS2.save/band.json` to plot the blue dots, and read `reconstruction/aohamiltonian/eig.dat` to plot the red lines. They are compared against each other in `band.png` generated in the same folder.
3. You should now get the same plot as Fig. 3b in the manuscript.
