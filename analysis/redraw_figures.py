"""Redraw existing figures without recomputing experiments.

Usage: python analysis/redraw_figures.py /path/to/extracted/v3
Reads frozen 2021 maps/results; writes the two manuscript PNGs and a trace JSON.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('evidence_root', type=Path)
args = parser.parse_args()
out = Path(__file__).resolve().parents[1]
source = args.evidence_root / 'evidence' / 'final_2021'
maps_path = source / 'spatial_final_2021.npz'
results_path = source / 'cp_results_final_2021.json'
maps = np.load(maps_path)
results = json.loads(results_path.read_text())['results']['convlstm']['levels']['0.9']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 18,
                     'axes.linewidth': 0.7, 'savefig.facecolor': 'white'})

fig = plt.figure(figsize=(9, 2.7))
trace = {'description': 'Display-only redraw of frozen results; no new experiments.',
         'map_scale': [0.35, 1.0], 'map_minima': {}, 'autumn_coverage': {}}
for i, (method, label) in enumerate([('split', 'Split CP'), ('sacp', 'SACP'), ('aci', 'Delayed ACI')]):
    ax = fig.add_axes([0.02 + i * 0.33, 0.34, 0.30, 0.50])
    values = maps[f'convlstm_{method}_L90'][11, 3]
    assert np.isfinite(values).all() and values.min() >= 0.35 and values.max() <= 1.0
    im = ax.imshow(values, origin='upper', interpolation='nearest',
                   cmap='RdBu_r', vmin=0.35, vmax=1.0, aspect='equal')
    ax.set_title(label, fontsize=19, pad=7)
    ax.set_xticks([])
    ax.set_yticks([])
    trace['map_minima'][method] = float(values.min())
cax = fig.add_axes([0.18, 0.22, 0.64, 0.075])
cb = fig.colorbar(im, cax=cax, orientation='horizontal', ticks=[0.35, 0.5, 0.7, 0.9, 1.0])
cb.ax.tick_params(labelsize=17, length=3, pad=2)
cb.set_label('Empirical coverage', fontsize=18, labelpad=1)
fig.savefig(out / 'paper' / 'fig3_spatial_final_2021_ieee.png', dpi=400)
plt.close(fig)

plt.rcParams.update({'font.size': 14})
fig, ax = plt.subplots(figsize=(7.4, 3.3))
seasons = ['DJF', 'MAM', 'JJA', 'SON']
methods = [('split', 'Split CP', '#1f77b4'), ('normalized', 'Normalized', '#2ca02c'),
           ('gaussian', 'Gaussian', '#7f7f7f'), ('online', 'Online refresh', '#9467bd'),
           ('aci', 'Delayed ACI', '#d62728'), ('weighted', 'Seasonal', '#ff7f0e'),
           ('sacp', 'SACP', '#17becf')]
for i, (method, label, color) in enumerate(methods):
    values = [results[method]['by_season'][season]['coverage'] for season in seasons]
    ax.bar(np.arange(4) + (i-3)*0.11, values, width=0.11, label=label, color=color)
    trace['autumn_coverage'][method] = values[-1]
ax.axhline(0.9, color='black', linestyle='--', linewidth=1.0)
ax.set_ylim(0.70, 1.00)
ax.set_yticks([0.70, 0.80, 0.90, 1.00])
ax.set_xticks(np.arange(4), seasons)
ax.set_ylabel('Coverage')
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.025), ncol=3,
          fontsize=10.5, columnspacing=0.85, handlelength=1.1,
          handletextpad=0.4, borderpad=0.2, frameon=False)
fig.subplots_adjust(left=0.125, right=0.99, top=0.74, bottom=0.14)
fig.savefig(out / 'paper' / 'fig4_seasonal_final_2021_ieee.png', dpi=400)
plt.close(fig)
trace['source_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (maps_path, results_path)}
(out/'analysis'/'figure_redraw_trace.json').write_text(json.dumps(trace, indent=2)+'\n')
print(json.dumps(trace, indent=2))
