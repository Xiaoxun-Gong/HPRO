#!/usr/bin/env python
# requires band_deeph.json and band_dft.json created by plot_bands.py

import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

min_plot_energy = -2.5
max_plot_energy = 4

trim_gap = True
trim_min = 0.1
trim_max = 1.0

fontsize = 22

legend = False
legend_kw = {'loc': 'center left', 'bbox_to_anchor': (0.5, 0.53)}

file_tag = 'band'
plot_format = 'png'
plot_dpi = 400

basic_folder = './'
basic_filename = 'eig.dat'

plot_args = [(('./', 'eig.dat', 'scatter', 4.785239202,), dict(color='blue', s=5, zorder=4)),
             (('../../bands/MoS2.save', 'band.json', 'plot', 0.0), dict(color='red', linestyle='-', linewidth=1.5, zorder=3))]
# plot_args = [(('./', 'scatter', 4.8133), dict(color='blue', s=5, zorder=4))]

def loaddata(folder, filename):
        # load dft
    if filename.endswith('.json'):
        with open(f'{folder}/{filename}', 'r') as f:
            data = json.load(f)
        for key, val in data.items():
            if type(val) is list:
                data[key] = np.array(val)
    
    else:
        bohr2ang = 0.5291772105638411
        rprim = np.loadtxt(f'{folder}/lat.dat').T / bohr2ang
        gprim = np.linalg.inv(rprim.T)
        f = open(f'{folder}/{filename}')
        f.readline(); f.readline()
        line_sp = f.readline().split()
        nk, nbnd = map(int, line_sp)
        eigs = np.empty((nk, nbnd))
        kpts = np.empty((nk, 3))
        hsk_idcs = []
        hsk_symbols = []
        line = f.readline()
        ik = 0
        while line:
            line_sp = line.split()
            if len(line_sp)>0 and '.' in line_sp[0]:
                kpts[ik] = list(map(float, line_sp[:3]))
                assert nbnd == int(line_sp[3])
                if len(line_sp) == 5:
                    hsk_symbols.append(line_sp[4])
                    hsk_idcs.append(ik)
                for _ in range(nbnd):
                    line = f.readline()
                    line_sp = line.split()
                    ibnd = int(line_sp[1]) - 1
                    eigs[ik, ibnd] = float(line_sp[2])
                ik += 1
            line = f.readline()
        assert ik == nk
        
        kcart = kpts @ gprim
        dis = np.linalg.norm(np.diff(kcart, axis=0), axis=1)
        kpoints_coords = np.cumsum(dis)
        kpoints_coords = np.concatenate(([0.], kpoints_coords))
        hsk_coords = []
        for ik in hsk_idcs:
            hsk_coords.append(kpoints_coords[ik])

        data = {}
        data['band_num_each_spin'] = nbnd
        data["kpoints_coords"] = kpoints_coords
        data["spin_up_energys"] = eigs.T
        data["hsk_coords"] = hsk_coords
        data["plot_hsk_symbols"] = hsk_symbols
        data["spin_num"] = 1

    return data
        
        
data_basic = loaddata(basic_folder, basic_filename)
hsk_coords = data_basic["hsk_coords"]
plot_hsk_symbols = data_basic["plot_hsk_symbols"]
spin_num = data_basic["spin_num"]


## Design the Figure
# For GUI less server
plt.switch_backend('agg') 
# Set the Fonts
# plt.rcParams.update({'font.size': 14,
#                      'font.family': 'STIXGeneral',
#                      'mathtext.fontset': 'stix'})
plt.rcParams.update({'font.size': fontsize,
                    'font.family': 'Arial',
                    'mathtext.fontset': 'cm'})
# Set the spacing between the axis and labels
plt.rcParams['xtick.major.pad'] = '6'
plt.rcParams['ytick.major.pad'] = '6'
# Set the ticks 'inside' the axis
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

x_min = 0.0
x_max = hsk_coords[-1]

# Create the figure and axis object
if trim_gap:
    fig, (ax_cond, ax_val) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': (max_plot_energy-trim_max, trim_min-min_plot_energy)}, figsize=(5.5, 5.5))
    
    # fig.tight_layout(pad=0.3)
else:
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 5.5))
# Set the range of plot
if trim_gap:
    ax_val.set_xlim(x_min, x_max)
else:
    ax.set_xlim(x_min, x_max)
if trim_gap:
    ax_val.set_ylim(min_plot_energy, trim_min)
    ax_cond.set_ylim(trim_max, max_plot_energy)
else:
    ax.set_ylim(min_plot_energy, max_plot_energy)
# Set the label of x and y axis
if trim_gap:
    ax_val.tick_params('y', labelsize=0.8*fontsize)
    ax_cond.tick_params('y', labelsize=0.8*fontsize)
    ax_val.set_xlabel('')
    ax_val.set_ylabel('Energy (eV)')
    ax_val.yaxis.set_label_coords(0.055, 0.5, transform=fig.transFigure)
else:
    ax.set_xlabel('')
    ax.set_ylabel('Energy (eV)')
    ax.tick_params('y', labelsize=0.8*fontsize)
# Set the Ticks of x and y axis
if trim_gap:
    ax_val.set_xticks(hsk_coords)
    ax_val.set_xticklabels(plot_hsk_symbols)
    ax_val.tick_params('x', labelsize=fontsize)
else:
    ax.set_xticks(hsk_coords)
    ax.set_xticklabels(plot_hsk_symbols)
    ax.tick_params('x', labelsize=fontsize)

# Plot the solid lines for High symmetic k-points
for kpath_i in range(len(hsk_coords)):
    if trim_gap:
        ax_val.vlines(hsk_coords[kpath_i], min_plot_energy, trim_min, colors="black", linewidth=0.7)
        ax_cond.vlines(hsk_coords[kpath_i], trim_max, max_plot_energy, colors="black", linewidth=0.7)
    else:
        ax.vlines(hsk_coords[kpath_i], min_plot_energy, max_plot_energy, colors="black", linewidth=0.7)
# Plot the fermi energy surface with a dashed line
if not trim_gap:
    ax.hlines(0.0, x_min, x_max, colors="black", 
            linestyles="dashed", linewidth=0.7)
else:
    ax_val.hlines(0.0, x_min, x_max, colors="black", 
            linestyles="dashed", linewidth=0.7)

def plot(folder, filename, func, efermi, **kwargs):

    data = loaddata(folder, filename)
    nbnd = data["band_num_each_spin"]
    kpoints_coords = data["kpoints_coords"]
    spin_up_energys = data["spin_up_energys"]

    # Plot the DFT Band Structure
    for band_i in range(nbnd):
        x = kpoints_coords
        x *= x_max / x[-1]
        y = spin_up_energys[band_i] - efermi
        if trim_gap:
            getattr(ax_val, func)(x[y<2], y[y<2], **kwargs)
            getattr(ax_cond, func)(x[y>-2], y[y>-2], **kwargs)
        else:
            getattr(ax, func)(x, y, **kwargs)
    if spin_num == 2:
        raise NotImplementedError
        # for band_i in range(band_num_each_spin_dft):
        #     x = kpoints_coords_dft
        #     y = spin_dn_energys_dft[band_i]
        #     ax.plot(x, y, '-', color='#0564c3', linewidth=1)

# =====
# plot('band_dft.json', 'plot', 0.0, color='red', linestyle='-', linewidth=1.5, zorder=3)
# plot('./', 'scatter', 4.8133, color='blue', s=5, zorder=4)
for args, kwargs in plot_args:
    plot(*args, **kwargs)
# =====

# create legends
if legend:
    band_dft_proxy = mlines.Line2D([], [], linestyle='-', color='r', linewidth=1.5)
    band_deeph_proxy = mlines.Line2D([], [], linestyle=':', color='b', linewidth=1.5)
    fig.legend((band_dft_proxy, band_deeph_proxy), ('DFT', 'DeepH-E3'), prop={'size': 0.8*fontsize}, **legend_kw)

if trim_gap:
    ax_cond.spines.bottom.set_visible(False)
    ax_val.spines.top.set_visible(False)
    ax_cond.xaxis.tick_top()
    ax_cond.tick_params(labeltop=False)  # don't put tick labels at the top
    ax_val.xaxis.tick_bottom()
    
    d = .5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax_cond.plot([0, 1], [0, 0], transform=ax_cond.transAxes, **kwargs)
    ax_val.plot([0, 1], [1, 1], transform=ax_val.transAxes, **kwargs)
            
# Save the figure
plot_filename = "%s.%s" %(file_tag, plot_format)
plt.tight_layout()
fig.subplots_adjust(hspace=0.05)
plt.savefig(plot_filename, format=plot_format, dpi=plot_dpi, transparent=False)
plt.savefig('band.svg', transparent=True)



