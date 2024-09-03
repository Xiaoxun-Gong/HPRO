from HPRO.lcaodiag import LCAODiagKernel

kernel = LCAODiagKernel()
kernel.setk([[0.000000000000,  0.000000000000,  0.000000000000],
             [0.333333333333,  0.333333333333,  0.000000000000],  
             [0.500000000000,  0.000000000000,  0.000000000000],  
             [0.000000000000,  0.000000000000,  0.000000000000]],
             [20, 10, 17, 1],
             ['\u0413', 'K', 'M', '\u0413'])
kernel.load_deeph_mats('./')
kernel.diag(nbnd=36, efermi=None)
kernel.write('./')
