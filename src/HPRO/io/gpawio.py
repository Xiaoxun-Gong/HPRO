import gzip
import xml.etree.ElementTree as ET

import numpy as np

from ..utils.orbutils import FracPolyRGD, ExpRGD, GridFunc, grid_overlap

class gpaw_psp:
    def __init__(self, filename):
        if filename.endswith('.gz'):
            with gzip.open(filename) as f:
                source = f.read()
        else:
            with open(filename) as f:
                source = f.read()
        
        root = ET.fromstring(source)
    
        l_list = []
        ids = []
        rcs = []
        states_elem = root.find('valence_states')
        for state in states_elem.findall('state'):
            l_list.append(int(state.attrib['l']))
            ids.append(state.attrib['id'])
            rcs.append(float(state.attrib['rc']))
        # print(l_list)
        
        gridfuncs = {}
        for gridfunc in root.findall('radial_grid'):
            if gridfunc.attrib['eq'] == 'r=a*i/(n-i)':
                istart = int(gridfunc.attrib['istart'])
                iend = int(gridfunc.attrib['iend'])
                a = float(gridfunc.attrib['a'])
                n = int(gridfunc.attrib['n'])
                assert istart==0 and iend==n-1
                rgrid = FracPolyRGD(a, n)
            elif gridfunc.attrib['eq'] == 'r=a*(exp(d*i)-1)':
                istart = int(gridfunc.attrib['istart'])
                assert istart==0
                iend = int(gridfunc.attrib['iend'])
                a = float(gridfunc.attrib['a'])
                d = float(gridfunc.attrib['d'])
                rgrid = ExpRGD(iend+1, a, d, minus1=True)
            else:
                raise NotImplementedError
            gridid = gridfunc.attrib['id']
            gridfuncs[gridid] = rgrid
        
        def getfunc(name, findall=True):
            elems = root.findall(name)
            if findall: 
                assert len(elems) == len(l_list)
            else:
                assert len(elems) == 1
            func_list = []
            for ielem in range(len(elems)):
                elem = elems[ielem]
                if findall: assert elem.attrib['state'] == ids[ielem]
                l = l_list[ielem] if findall else 0
                func = list(map(float, elem.text.split()))
                gridid = elem.attrib['grid']
                gridlen = len(func)
                phirgrid = np.array(func)
                if name=='projector_function':
                    rcut = rcs[ielem]
                else:
                    rcut = None
                func_list.append(GridFunc(gridfuncs[gridid], phirgrid, l=l, rcut=rcut))
            return func_list
        
        projR_list = getfunc('projector_function')
        phiPS_list = getfunc('pseudo_partial_wave')
        phiAE_list = getfunc('ae_partial_wave')
        
        self.l_list = l_list
        self.projR_list = projR_list
        self.phiPS_list = phiPS_list
        self.phiAE_list = phiAE_list
        self.cqij = None
    
    def check_ortho(self):
        l_list = self.l_list
        projR_list = self.projR_list
        phiPS_list = self.phiPS_list
        
        print(l_list)
        norb = len(l_list)
        for iorb in range(norb):
            for jorb in range(norb):
                if l_list[iorb] == l_list[jorb]:
                    projR = projR_list[iorb]
                    phiPS = phiPS_list[jorb]
                    olp = grid_overlap(projR, phiPS)
                    print(f'<proj{iorb}|phiPS{jorb}>={olp}')
    
    def get_cqij(self):
        if self.cqij is not None:
            return self.cqij 

        l_list = self.l_list
        phiAE_list = self.phiAE_list
        phiPS_list = self.phiPS_list
        
        orbital_slices = [0]
        for l in l_list:
            orbital_slices.append(orbital_slices[-1] + 2*l+1)
        size = orbital_slices[-1]
        cqij = np.zeros((size, size))
        norb = len(l_list)
        for iorb in range(norb):
            for jorb in range(iorb, norb):
                if l_list[iorb] == l_list[jorb]:
                    olp_ae = grid_overlap(phiAE_list[iorb], phiAE_list[jorb])
                    olp_ps = grid_overlap(phiPS_list[iorb], phiPS_list[jorb])
                    q = olp_ae - olp_ps
                    l = l_list[iorb]
                    starti = orbital_slices[iorb]
                    startj = orbital_slices[jorb]
                    for ii in range(2*l+1):
                        cqij[starti+ii, startj+ii] = q
                        if iorb != jorb:
                            cqij[startj+ii, starti+ii] = q
        
        self.cqij = cqij
        return cqij
