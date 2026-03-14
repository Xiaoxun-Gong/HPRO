import numpy as np

from scipy.io import FortranFile
from scipy.sparse import csr_matrix

from ..matao.mataocsr import MatAOCSR
from .aodata import AOData
from .struio import from_siesta
from .hrloader import read_vnloc

def read_orbindx(filename):
    with open(filename) as fp:
        line = fp.readline()
        line_sp = line.split()
        nouc = int(line_sp[0])
        nosc = int(line_sp[1])
        orb_indx = np.empty((nosc, 7), dtype='i8') #(io, [ia, iao, isc1, isc2, isc3, iuo, l])
        # meanings of these variables can be found in the end of .ORB_INDX

        for _ in range(2): fp.readline()
        for io in range(nosc):
            line = fp.readline()
            line_sp = line.split()
            orb_indx[io, :] = list(map(int, (line_sp[1], line_sp[4], line_sp[12], line_sp[13], line_sp[14], line_sp[15], line_sp[6])))
    
    orb_indx[:, 0:2] -= 1; orb_indx[:, 5] -= 1 # 0-based
    
    # sort iao according to l (openmx standard)
    argsort = np.argsort(orb_indx[:, 0]*10 + orb_indx[:, 6], kind='stable')
    tmp = orb_indx[:, 1].copy()
    orb_indx[argsort, 1] = tmp

    assert np.all(orb_indx[:nouc, 5] == np.arange(0, nouc, 1, dtype='i4'))
    tmp = orb_indx[orb_indx[:, 5], 0].copy()
    orb_indx[:, 0] = tmp # make ia in unit cell

    return nouc, nosc, orb_indx

class siesta_hsx2:
    def __init__(self, savedir, sysname='siesta'):
        f = FortranFile(f'{savedir}/{sysname}.HSX', 'r')

        self.version = f.read_record('i4')
        assert self.version[0] == 1

        self.is_dp = f.read_record('i4')

        rec = f.read_record('i4')
        self.na_u = rec[0]; self.no_u = rec[1]; self.nspin = rec[2]; self.nspc = rec[3]; self.nsc = rec[4:7]
        # no_u: orbitals in unit cell
        assert self.nspin == 1

        rec = f.read_record('f8')
        ucell = rec[0:9].reshape(3, 3) # bohr
        self.Ef = rec[9]; self.qtot = rec[10]; self.temp = rec[11] # electronic temperature

        # print(np.max(np.abs(ucell-stru.rprim)))

        rec = f.read_record('i4')
        rec = f.read_record('i4')
        for _ in range(self.nspc): f.read_record('i4')

        numh = f.read_record('i4') # (no_u), number of hoppings for each unit-cell orbital
        self.numh = numh

        # get all connected atom pairs for each orbital in unit cell
        # listh[i] contains all orbitals connected to unit-cell orbital i
        listh = []
        for io in range(self.no_u):
            rec = f.read_record('i4') - 1
            listh.append(rec)
        self.listh = listh

        hmat = []
        for ispin in range(self.nspin):
            for io in range(self.no_u):
                hmat.append(f.read_record('f8') / 2.0) # Ry -> Ha
        self.hmat = hmat
        
        smat = []
        for io in range(self.no_u):
            smat.append(f.read_record('f8'))
        self.smat = smat
        
        f.close()

def get_hs_siesta(savedir, upfdir, sysname='siesta'):
    stru = from_siesta(savedir, sysname=sysname)
    nouc, nosc, orb_indx = read_orbindx(f'{savedir}/{sysname}.ORB_INDX')
    aodata = AOData(stru, basis_path_root=savedir, aocode='siesta')
    Dij, Qij, projR = read_vnloc(stru, upfdir, interface='qe')
    hsx = siesta_hsx2(savedir, sysname=sysname)

    maxr = 2 * (max(aodata.cutoffs.values()) + max(projR.cutoffs.values()))
    hmat = MatAOCSR(aodata, maxr=maxr)
    smat = MatAOCSR(aodata, maxr=maxr)

    # map iorbsc and iorbuc from siesta to MatAOCSR
    map_iorbuc = hmat.orbinfo1.find_orbindx2(orb_indx[:nouc, 0], orb_indx[:nouc, 1])
    invmap_iouc = np.argsort(map_iorbuc)
    iouc = hmat.orbinfo2.orbinfo_uc.find_orbindx2(orb_indx[:, 0], orb_indx[:, 1])
    map_iorbsc = hmat.orbinfo2.iorb_uc2sc(orb_indx[:, 2:5], iouc, False)

    indptr = [0]
    indices = []
    datah, datas = [], []
    for iouc in range(nouc):
        iouc_siesta = invmap_iouc[iouc]
        indptr.append(indptr[-1] + hsx.numh[iouc_siesta])
        indices.append(map_iorbsc[hsx.listh[iouc_siesta]])
        datah.append(hsx.hmat[iouc_siesta])
        datas.append(hsx.smat[iouc_siesta])
    
    indptr = np.array(indptr)
    indices = np.concatenate(indices)
    datah = np.concatenate(datah)
    datas = np.concatenate(datas)

    hmat_tmp = csr_matrix((datah, indices, indptr), shape=hmat.mat.shape)
    smat_tmp = csr_matrix((datas, indices, indptr), shape=smat.mat.shape)

    # fix (-1)^m coefficient difference of spharms
    coeff_left = (-1) ** (hmat.orbinfo1.m_orbs % 2)
    coeff_right = (-1) ** (hmat.orbinfo2.m_orbs % 2)
    hmat_tmp = hmat_tmp.multiply(coeff_left[:, None]).multiply(coeff_right[None, :])
    smat_tmp = smat_tmp.multiply(coeff_left[:, None]).multiply(coeff_right[None, :])

    hmat.mat = hmat_tmp.tocsr()
    smat.mat = smat_tmp.tocsr()

    return stru, aodata, hmat, smat
