import xml.etree.ElementTree as ET
import h5py
import numpy as np
from scipy.sparse import coo_array, diags_array

from ..constants import hartree2ev
from ..utils.structure import Structure
from ..utils.misc import mytqdm, KGData
from ..utils.mpi import one_by_one, distrib_vec, is_master
from ..utils.math import make_kkmap, kgrid_with_tr
from ..io.bgwio import bgw_wfn
from ..io.struio import from_qexml


class WFNData:
    '''
    Stores PW wavefunctions and energies
    
    Attributes:
    ---------
    structure: structure.Structure object
    nk: int
    nbnd: int
    kpts(nk, 3): kpts in reduced coordinate
    kptwts(nk): kpt weights
    kpts_cart(nk, 3): kpts in cartesian coordinate, scaled by 2pi
    eig(nspin, nk, nband): in hartree
    kgdatas: List[KGData], len=nk
    '''
    
    def __init__(self, structure: Structure, wfn_path_root='./', pwcode='qe', 
                 kgrid=None, maxband=None, 
                 ):
        # maxband is 1-based index
        
        self.structure = structure
        
        if pwcode not in ['qe', 'bgw']:
            raise NotImplementedError(f'{pwcode}')
        
        if pwcode == 'qe':
            
            # check structure consistency
            stru = from_qexml(f'{wfn_path_root}/data-file-schema.xml')
            assert stru == self.structure, 'Structure mismatch'
            
            qexmlroot = ET.parse(f'{wfn_path_root}/data-file-schema.xml').getroot()
            
            alat = float(qexmlroot.find('output').find('atomic_structure').attrib['alat'])
            band_structure_elem = qexmlroot.find('output').find('band_structure')
            nk = int(band_structure_elem.find('nks').text)
            nbnd = int(band_structure_elem.find('nbnd').text)
            kpts = np.zeros((nk, 3))
            eig = np.zeros((1, nk, nbnd)) # only supports no spin
            kptwts = np.zeros((nk))
            for ik, ks_eig_elem in enumerate(band_structure_elem.findall('ks_energies')):
                kpts[ik] = list(map(float, ks_eig_elem.find('k_point').text.split()))
                eig[0, ik] = list(map(float, ks_eig_elem.find('eigenvalues').text.split()))
                kptwts[ik] = float(ks_eig_elem.find('k_point').attrib['weight'])
            kptwts = kptwts / np.sum(kptwts)
            kpts = kpts @ structure.rprim.T / alat # cartesian to reduced. kpt in qe is given in unit 2pi/alat.
            ecutwfn = float(qexmlroot.find('input').find('basis').find('ecutwfc').text)
            
            self.nk = nk
            self.nbnd = nbnd
            self.kpts = kpts
            self.kptwts = kptwts
            assert np.abs(np.sum(kptwts)-1) < 1e-5, f'{np.sum(kptwts)}'
            self.kpts_cart = 2 * np.pi * kpts @ structure.gprim
            self.eig = eig
            self.ecutwfn = ecutwfn
        
        elif pwcode == 'bgw':
            
            wfnread = bgw_wfn(f'{wfn_path_root}')
            wfnread.read_header()
            
            nk = wfnread.nk
            nbnd = wfnread.nb
            kpts = wfnread.xk
            kptwts = wfnread.wk / np.sum(wfnread.wk)
            eig = wfnread.et_g.reshape(1, nk, nbnd) / 2.0 # ! no spin
            ecutwfn = wfnread.ecutwfc / 2.
            
            self.nk = nk
            self.nbnd = nbnd
            self.kpts = kpts
            self.kptwts = kptwts
            self.kpts_cart = 2 * np.pi * self.kpts @ structure.gprim
            self.eig = eig
            self.ecutwfn = ecutwfn
        
        if maxband is not None:
            assert maxband <= self.nbnd
            
            nbnd = maxband
            self.nbnd = maxband
            self.eig = self.eig[:, :, :maxband]
            
        self.kgdatas = []
            
        if is_master():
            print(f'Reading wavefunctions on {nk} k points')
        
        if pwcode == 'qe':
            for ik in mytqdm(range(nk)):
                
                wfcfile = h5py.File(f'{wfn_path_root}/wfc{ik+1}.hdf5')
                ng_total = wfcfile.attrs['igwx'] # ! need to check this
                
                rank, count, displ = distrib_vec(ng_total, displ_last_elem=True)
                
                miller_idc = np.array(wfcfile['MillerIndices'][displ[rank]:displ[rank+1], :])
                
                ng = len(miller_idc)
                assert displ[rank+1] - displ[rank] == ng
                assert displ[-1] == ng_total
                
                kgcart = 2 * np.pi * (kpts[ik][None, :] + miller_idc) @ structure.gprim
                
                unkg = np.zeros((nbnd, ng), dtype=np.complex128)
                # unkg_read = np.array(wfcfile['evc'])
                with one_by_one():
                    unkg_read = wfcfile['evc'][:nbnd, 2*displ[rank]:2*displ[rank+1]]
                unkg.real = np.array(unkg_read[:, 0::2])
                unkg.imag = np.array(unkg_read[:, 1::2])
                
                wfcfile.close()
                
                kgdata = KGData(ng=ng, nbnd=nbnd, miller_idc=miller_idc, unkg=unkg, kgcart=kgcart)
                # force the number of bands to be the same throughout k-pionts
                self.kgdatas.append(kgdata)
                
            # kgcart_list = []
            # for ik in range(nk):
            #     kgcart_list.append(kgcart)
        
        elif pwcode == 'bgw':
            
            gvecrange = np.empty((nk, 2), dtype=int)
            for ik in range(nk):
                ng_total = wfnread.ngk_g[ik]
                rank, count, displ = distrib_vec(ng_total, displ_last_elem=True)
                gvecrange[ik, :] = displ[rank], displ[rank+1]
            
            wfnread.read_data(bandrange=(0, self.nbnd), gvecrange=gvecrange)
            self.kgdatas = wfnread.kgdatas
            
            wfnread.close()
            
        if kgrid is not None:
            self.reduce_kgrid(kgrid)
            
    
    def reduce_kgrid(self, kgrid):
        '''
        Reduce the k-grid and k-weights using time-reversal symmetry.
        '''
        
        kpts_new, kptwts_new = kgrid_with_tr(kgrid)
        
        nk_new = len(kpts_new)
        map_old_new = make_kkmap(self.kpts, kpts_new) # make kkmap with TR symmetry
            
        kpts_cart_new = self.kpts_cart[map_old_new, :]
        eig_new = self.eig[:, map_old_new, :]
        
        kgdatas_new = []
        for ik_new in range(nk_new):
            kgdatas_new.append(self.kgdatas[map_old_new[ik_new]])
        # Note: in principle, G vectors should be changed when k points are mapped
        # This is because k=0.5 G=(1,0,0) is different from k=1.5 G=(1,0,0) although these two k points are equivalent
        # This also happens when TR symmetry is used to map k points
        # However this does not matter in the projection process so we are lazy here
        
        self.nk = nk_new
        self.kpts = kpts_new
        self.kptwts = kptwts_new
        self.kpts_cart = kpts_cart_new
        self.eig = eig_new
        self.kgdatas = kgdatas_new

    def check_wfn_sum(self):
        for ik, kgdata in enumerate(self.kgdatas):
            unkg_sum = np.sum(np.power(np.abs(kgdata.unkg), 2), axis=1)
            maxdiv = np.max(np.abs(unkg_sum - 1))
            print(ik, maxdiv)
    
    def get_H_band_basis(self, nbndmin, nbndmax, 
                         band_shift=0):
        # get a list of sparse matrix for each k-point, which are the Hamiltonian matrices in band basis
        
        assert nbndmin <= nbndmax <= self.nbnd


        mats_k = [] 
        for ik in range(self.nk):
            mats_k.append(diags_array(self.eig[0, ik, nbndmin-1:nbndmax] + band_shift))


        return mats_k

