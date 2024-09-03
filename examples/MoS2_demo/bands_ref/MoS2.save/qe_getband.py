import numpy as np
import xml.etree.ElementTree as ET
import json

bohr2ang = 0.5291772109
hartree2eV = 27.21138602

def save_band_json(lat, kpts, eig, efermi, plot_hsk_symbols):
  '''
  only supports non-spinful case
  
  Arguments
  ---------
  lat(3, 3): R_ij is the jth cartesian component of ith lattice vector
  kpts(nk, 3): in reduced coordinate
  eig(nspin, nk, nband): in hartree
  efermi: in hartree
  plot_hsk_symbols: a list, length must be same with number-of-kpaths + 1
  
  '''

  rlat = np.linalg.inv(lat.T)
  kpts_cartesian = np.matmul(kpts, rlat)

  hsk_idcs = [0]
  nkpt = kpts.shape[0]
  for i in range(nkpt-3):
    x1, x2, x3 = kpts_cartesian[i], kpts_cartesian[i+1], kpts_cartesian[i+2]
    is_sameline = np.sum(np.power(np.cross(x1-x2, x2-x3), 2)) > 1e-15
    if is_sameline:
      hsk_idcs.append(i+1)
  hsk_idcs.append(nkpt-1)
  print('Found high-symmetry ks:')
  print(kpts[hsk_idcs])

  kpoints_coords = [0.0]
  for i in range(1, nkpt):
    dis = np.sqrt(np.sum(np.power(kpts_cartesian[i] - kpts_cartesian[i-1], 2)))
    kpoints_coords.append(kpoints_coords[-1] + dis)
  hsk_coords = []
  for idc in hsk_idcs:
    hsk_coords.append(kpoints_coords[idc])

  eig -= efermi # shift E_F to zero
  eig *= hartree2eV

  nk = len(eig[0])
  nb = len(eig[0,0])

  bands_data = dict()
  bands_data["hsk_coords"] = hsk_coords
  bands_data["plot_hsk_symbols"] = plot_hsk_symbols
  bands_data["kpath_num"] = len(hsk_coords) - 1
  bands_data["spin_num"] = 1

  bands_data["band_num_each_spin"] = eig.shape[2]
  bands_data["kpoints_coords"] = kpoints_coords
  bands_data["spin_up_energys"] = eig[0].T.tolist()
  bands_data["spin_dn_energys"] = []

  with open('band.json', 'w') as f:
    json.dump(bands_data, f)


if __name__ == '__main__':
  plot_hsk_symbols = ['\u0413', 'K', 'M', '\u0413']
  
  qeroot = ET.parse('data-file-schema.xml').getroot()
  
  cell = qeroot.find('input').find('atomic_structure').find('cell')
  lat = np.array([list(map(float, cell[i].text.split())) for i in range(3)]) * bohr2ang
  
  band_structure_elem = qeroot.find('output').find('band_structure')
  
  nk = int(band_structure_elem.find('nks').text)
  nbnd = int(band_structure_elem.find('nbnd').text)
  kpts = np.zeros((nk, 3))
  eig = np.zeros((1, nk, nbnd)) # only supports no spin
  for ik, ks_eig_elem in enumerate(band_structure_elem.findall('ks_energies')):
    kpts[ik] = list(map(float, ks_eig_elem.find('k_point').text.split()))
    eig[0, ik] = list(map(float, ks_eig_elem.find('eigenvalues').text.split()))
#   print(eig[0,0])
  kpts = kpts @ lat.T # cartesian to reduced
  
  efermi = float(band_structure_elem.find('fermi_energy').text)
  print(efermi * hartree2eV)
  
  save_band_json(lat, kpts, eig, efermi, plot_hsk_symbols)
