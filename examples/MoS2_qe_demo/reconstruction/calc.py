from HPRO import PW2AOkernel

kernel = PW2AOkernel(
    aodata_interface='siesta',
    aodata_root='../aobasis', 
    hrdata_interface='qe-bgw',
    vscdir='../scf/VSC',
    upfdir='../pseudos',
    ecutwfn=30
)
kernel.run_pw2ao_rs('./aohamiltonian')
