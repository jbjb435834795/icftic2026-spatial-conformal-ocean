"""Recompute global / cellwise / Gaussian-bandwidth comparisons from frozen results.

Usage: python reviewer_analysis.py /path/to/extracted/928
No retraining, new forecast generation, or external validation is performed.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1]).resolve()
out = Path(__file__).resolve().parent
dev = json.loads((root / 'evidence/cp_results_dev_2020.json').read_text())
variables = dev['variables']
families = ['convlstm', 'fno2d', 'direct_resnet']
rows = []
for family in families:
    sweep = json.loads((root / f'evidence/sacp_sweep_{family}_dev_2020.json').read_text())
    for level in ['0.8', '0.9', '0.95']:
        base = dev['results'][family]['levels'][level]['split']
        for sigma in ['global', '0', '1', '2', '4']:
            item = base if sigma == 'global' else sweep['sigma_results'][sigma][level]
            ratios = [item['by_variable'][v]['winkler'] / base['by_variable'][v]['winkler'] for v in variables]
            widths = [item['by_variable'][v]['width'] / base['by_variable'][v]['width'] for v in variables]
            rows.append(dict(family=family, level=level, sigma=sigma,
                             coverage=item['overall']['coverage'],
                             macro_winkler_skill_pct=100 * (1 - np.mean(ratios)),
                             macro_width_ratio=np.mean(widths)))
with (out / 'spatial_comparison.csv').open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

spatial = np.load(root / 'evidence/final_2021/spatial_final_2021.npz')
spatial_rows = []
for method in ['split', 'sacp', 'aci']:
    a = spatial[f'convlstm_{method}_L90']
    for aggregation, x in [('day12_SST', a[11, 3]), ('all_leads_variables', a.mean(axis=(0, 1)))]:
        spatial_rows.append(dict(method=method, aggregation=aggregation,
                                 minimum=float(x.min()), mean=float(x.mean()),
                                 fraction_below_085=float((x < .85).mean())))
with (out / 'spatial_aggregation_check.csv').open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(spatial_rows[0])); w.writeheader(); w.writerows(spatial_rows)

stats = json.loads((root / 'evidence/final_2021/stats_final_2021.json').read_text())
p = np.array([r['p_raw'] for r in stats['records']])
order = np.argsort(p)
adjusted = np.minimum(1, np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order]))
saved = np.array([r['p_holm'] for r in stats['records']])[order]
assert np.allclose(adjusted, saved, rtol=0, atol=1e-14)
assert len(p) == 2100

summary = {'provenance': 'Frozen 2020 development summaries; no new experiments.',
           'holm_scope': 'All 2100 stored tests jointly; 60 entries per displayed series.',
           'holm_max_absolute_difference': float(np.max(np.abs(adjusted - saved))),
           'external_validation_performed': False}
for sigma in ['global', '0', '1', '2', '4']:
    r = [x for x in rows if x['sigma'] == sigma and x['level'] == '0.9']
    summary[sigma] = {k: [float(min(x[k] for x in r)), float(max(x[k] for x in r))]
                      for k in ['coverage', 'macro_winkler_skill_pct', 'macro_width_ratio']}
(out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
