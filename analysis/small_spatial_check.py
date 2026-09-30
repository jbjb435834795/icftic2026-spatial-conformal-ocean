"""Small, fixed within-domain spatial robustness check.

Uses existing empirical coverage maps only; no new forecast or calibration fit.
Before calculating results, fix: nominal .90, sigma=1, all 3 learned forecasters,
all 5 variables and 12 leads, and four equal 16x32 index-space tiles.
No tile, model, variable, or lead is selected using performance.
No independent-region/generalization or statistical-significance claim is made.

Usage: python small_spatial_check.py /path/to/extracted/v3
"""
import csv
import json
import sys
from pathlib import Path
import numpy as np

root = Path(sys.argv[1]).resolve()
out = Path(__file__).resolve().parent
families = ['convlstm', 'fno2d', 'direct_resnet']
methods = ['split', 'sacp', 'aci']
tiles = [
    ('T1', slice(0,16), slice(0,32)),
    ('T2', slice(0,16), slice(32,64)),
    ('T3', slice(16,32), slice(0,32)),
    ('T4', slice(16,32), slice(32,64)),
]
plan = dict(nominal=.9, sacp_sigma=1, families=families, methods=methods,
            variable_order=['SSH','u','v','SST','SSS'], leads=list(range(1,13)),
            tiles={name:[rs.start,rs.stop,cs.start,cs.stop] for name,rs,cs in tiles},
            metric='mean absolute deviation from .90 of each cell-variable-lead empirical coverage; absolute value before averaging',
            interpretation='Post hoc descriptive within-domain stratification of frozen 2021 maps; not external validation.',
            inference='No iid-cell standard errors, p-values, or confidence intervals are calculated.')
(out/'small_spatial_check_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
z = np.load(root/'evidence/final_2021/spatial_final_2021.npz')
rows=[]; details=[]
for f in families:
    for name,rs,cs in tiles:
        for m in methods:
            a=z[f'{f}_{m}_L90'][:,:,rs,cs]
            assert a.shape==(12,5,16,32)
            rows.append(dict(family=f,tile=name,method=m,
                mean_coverage=float(a.mean()),
                mean_abs_coverage_gap=float(np.abs(a-.9).mean()),
                fraction_cell_variable_lead_below_085=float((a<.85).mean())))
            for h in range(12):
                for v in range(5):
                    x=a[h,v]
                    details.append(dict(family=f,tile=name,method=m,lead=h+1,
                        variable=plan['variable_order'][v],mean_coverage=float(x.mean()),
                        mean_abs_coverage_gap=float(np.abs(x-.9).mean())))
for filename, data in [('small_spatial_tiles.csv',rows),('small_spatial_all_cases.csv',details)]:
    with (out/filename).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
comparisons=[]
for f in families:
    for name,_,_ in tiles:
        subset={x['method']:x for x in rows if x['family']==f and x['tile']==name}
        base=subset['split']['mean_abs_coverage_gap']
        value=subset['sacp']['mean_abs_coverage_gap']
        comparisons.append(dict(family=f,tile=name,split_gap=base,sacp_gap=value,
                                absolute_reduction_pp=100*(base-value),
                                relative_reduction_pct=100*(1-value/base)))
detail_idx={(r['family'],r['tile'],r['lead'],r['variable'],r['method']):r for r in details}
case_changes=[]
for key,base in detail_idx.items():
    if key[-1]!='split':continue
    sacp=detail_idx[(*key[:-1],'sacp')]
    case_changes.append(base['mean_abs_coverage_gap']-sacp['mean_abs_coverage_gap'])
result=dict(plan=plan,tile_comparisons=comparisons,
    tile_aggregates_improved=sum(x['absolute_reduction_pp']>0 for x in comparisons),
    tile_aggregates_total=len(comparisons),
    detailed_cases_improved=int((np.array(case_changes)>1e-12).sum()),
    detailed_cases_worse=int((np.array(case_changes)<-1e-12).sum()),
    detailed_cases_tied=int((np.abs(case_changes)<=1e-12).sum()),
    detailed_cases_total=len(case_changes),
    worst_detailed_increase_pp=float(-100*min(case_changes)),
    new_external_validation=False)
(out/'small_spatial_summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='plan'},indent=2))
