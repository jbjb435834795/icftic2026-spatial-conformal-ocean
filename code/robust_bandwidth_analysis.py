#!/usr/bin/env python3
"""Robustness analysis for the SACP bandwidth sweep.

Uses only frozen 2020 development-year aggregate sweep JSONs. It does NOT
reconstruct 2021 intervals and does NOT claim independent temporal validation.

Outputs:
  bandwidth_margin.csv
  loo_combo_transfer.csv
  loo_combo_summary.csv
  leave_one_model_out.csv
  leave_one_level_out.csv
  ROBUSTNESS_SUMMARY.md
"""
from __future__ import annotations
import csv, json, pathlib, statistics
from collections import Counter, defaultdict

BASE = pathlib.Path('/mnt/data/run_new_results/original/ICFTIC2026_paper_package/evidence')
OUT = pathlib.Path('/mnt/data/run_new_results/next_experiment/results')
OUT.mkdir(parents=True, exist_ok=True)
MODELS = ['convlstm','fno2d','direct_resnet']
MODEL_NAMES = {'convlstm':'ConvLSTM','fno2d':'FNO','direct_resnet':'ResNet'}
VARS = ['zos','uo','vo','thetao','so']
VAR_NAMES = {'zos':'SSH','uo':'u','vo':'v','thetao':'SST','so':'SSS'}
LEVELS = ['0.8','0.9','0.95']
SIGMAS = ['0','1','2','4']

def load(name):
    with open(BASE/name, encoding='utf-8') as f:
        return json.load(f)

def write_csv(name, rows):
    if not rows:
        return
    with open(OUT/name, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

cpdev = load('cp_results_dev_2020.json')
sweeps = {m: load(f'sacp_sweep_{m}_dev_2020.json')['sigma_results'] for m in MODELS}

# normalized Winkler ratio to split CP for a single architecture x level x variable x sigma.
def ratio(m, level, v, sigma):
    b = cpdev['results'][m]['levels'][level]['split']['by_variable'][v]['winkler']
    x = sweeps[m][sigma][level]['by_variable'][v]['winkler']
    return float(x / b)

def coverage(m, level, v, sigma):
    return float(sweeps[m][sigma][level]['by_variable'][v]['coverage'])

def choose_sigma(training_combos, v):
    # training_combos are (model, level). Minimize mean dimensionless ratio.
    scored=[]
    for s in SIGMAS:
        vals=[ratio(m,l,v,s) for m,l in training_combos]
        scored.append((statistics.fmean(vals), float(s), s))
    scored.sort(key=lambda z:(z[0],z[1]))
    return scored[0][2], scored

# 1) Best vs runner-up margin for all 45 architecture-level-variable combinations.
margin_rows=[]
for m in MODELS:
    for level in LEVELS:
        for v in VARS:
            vals=[]
            for s in SIGMAS:
                r=ratio(m,level,v,s)
                vals.append((r,float(s),s))
            vals.sort(key=lambda z:(z[0],z[1]))
            best, second = vals[0], vals[1]
            margin_rows.append({
                'model':MODEL_NAMES[m], 'level':float(level), 'variable':VAR_NAMES[v],
                'best_sigma':best[1], 'runnerup_sigma':second[1],
                'best_skill_vs_split_pct':100*(1-best[0]),
                'runnerup_skill_vs_split_pct':100*(1-second[0]),
                'best_minus_runnerup_skill_pp':100*(second[0]-best[0]),
                'runnerup_winkler_excess_over_best_pct':100*(second[0]/best[0]-1),
                'best_coverage':coverage(m,level,v,best[2]),
                'runnerup_coverage':coverage(m,level,v,second[2]),
            })
write_csv('bandwidth_margin.csv', margin_rows)

# 2) Leave-one-combination-out across all 3x3 model-level combinations per variable.
combos=[(m,l) for m in MODELS for l in LEVELS]
loo_rows=[]
for v in VARS:
    for hm,hl in combos:
        train=[c for c in combos if c != (hm,hl)]
        sel,_=choose_sigma(train,v)
        selected=ratio(hm,hl,v,sel)
        fixed1=ratio(hm,hl,v,'1')
        oracle_s=min(SIGMAS, key=lambda s:(ratio(hm,hl,v,s),float(s)))
        oracle=ratio(hm,hl,v,oracle_s)
        loo_rows.append({
            'heldout_model':MODEL_NAMES[hm], 'heldout_level':float(hl), 'variable':VAR_NAMES[v],
            'selected_sigma_from_other_8':float(sel), 'oracle_sigma_heldout':float(oracle_s),
            'selected_skill_vs_split_pct':100*(1-selected),
            'fixed_sigma1_skill_vs_split_pct':100*(1-fixed1),
            'oracle_skill_vs_split_pct':100*(1-oracle),
            'regret_to_oracle_pp':100*(selected-oracle),
            'gain_vs_fixed_sigma1_pp':100*(fixed1-selected),
            'selected_matches_oracle': int(sel==oracle_s),
        })
write_csv('loo_combo_transfer.csv', loo_rows)

summary_rows=[]
for vn in [VAR_NAMES[v] for v in VARS] + ['ALL']:
    rr=loo_rows if vn=='ALL' else [r for r in loo_rows if r['variable']==vn]
    summary_rows.append({
        'variable':vn,
        'n_holdouts':len(rr),
        'mean_selected_skill_pct':statistics.fmean(r['selected_skill_vs_split_pct'] for r in rr),
        'mean_fixed_sigma1_skill_pct':statistics.fmean(r['fixed_sigma1_skill_vs_split_pct'] for r in rr),
        'mean_oracle_skill_pct':statistics.fmean(r['oracle_skill_vs_split_pct'] for r in rr),
        'mean_regret_to_oracle_pp':statistics.fmean(r['regret_to_oracle_pp'] for r in rr),
        'mean_gain_vs_fixed_sigma1_pp':statistics.fmean(r['gain_vs_fixed_sigma1_pp'] for r in rr),
        'oracle_match_fraction':statistics.fmean(r['selected_matches_oracle'] for r in rr),
        'beats_or_ties_fixed1_fraction':statistics.fmean(int(r['gain_vs_fixed_sigma1_pp']>=-1e-12) for r in rr),
    })
write_csv('loo_combo_summary.csv', summary_rows)

# 3) Leave one architecture out: choose from other two architectures at same level.
lomo=[]
for hm in MODELS:
    train_models=[m for m in MODELS if m!=hm]
    for level in LEVELS:
        for v in VARS:
            sel,_=choose_sigma([(m,level) for m in train_models],v)
            sr=ratio(hm,level,v,sel); fr=ratio(hm,level,v,'1')
            oracle_s=min(SIGMAS,key=lambda s:(ratio(hm,level,v,s),float(s)))
            oracle=ratio(hm,level,v,oracle_s)
            lomo.append({
                'heldout_model':MODEL_NAMES[hm], 'level':float(level), 'variable':VAR_NAMES[v],
                'selected_sigma_from_other_models':float(sel), 'oracle_sigma':float(oracle_s),
                'selected_skill_pct':100*(1-sr), 'fixed1_skill_pct':100*(1-fr),
                'oracle_skill_pct':100*(1-oracle), 'regret_to_oracle_pp':100*(sr-oracle),
                'gain_vs_fixed1_pp':100*(fr-sr), 'matches_oracle':int(sel==oracle_s)
            })
write_csv('leave_one_model_out.csv', lomo)

# 4) Leave one nominal level out: choose from other two levels for same model.
lolo=[]
for m in MODELS:
    for hl in LEVELS:
        train_levels=[l for l in LEVELS if l!=hl]
        for v in VARS:
            sel,_=choose_sigma([(m,l) for l in train_levels],v)
            sr=ratio(m,hl,v,sel); fr=ratio(m,hl,v,'1')
            oracle_s=min(SIGMAS,key=lambda s:(ratio(m,hl,v,s),float(s)))
            oracle=ratio(m,hl,v,oracle_s)
            lolo.append({
                'model':MODEL_NAMES[m], 'heldout_level':float(hl), 'variable':VAR_NAMES[v],
                'selected_sigma_from_other_levels':float(sel), 'oracle_sigma':float(oracle_s),
                'selected_skill_pct':100*(1-sr), 'fixed1_skill_pct':100*(1-fr),
                'oracle_skill_pct':100*(1-oracle), 'regret_to_oracle_pp':100*(sr-oracle),
                'gain_vs_fixed1_pp':100*(fr-sr), 'matches_oracle':int(sel==oracle_s)
            })
write_csv('leave_one_level_out.csv', lolo)

# Summaries.
gaps=[r['best_minus_runnerup_skill_pp'] for r in margin_rows]
small05=sum(g<0.05 for g in gaps)/len(gaps)
small10=sum(g<0.10 for g in gaps)/len(gaps)
allrow=next(r for r in summary_rows if r['variable']=='ALL')

# Per-variable mode selection under leave-one-combo-out.
sel_modes={}
for v in [VAR_NAMES[x] for x in VARS]:
    c=Counter(r['selected_sigma_from_other_8'] for r in loo_rows if r['variable']==v)
    sel_modes[v]=(c.most_common(1)[0][0], c.most_common(1)[0][1]/sum(c.values()))

lines=[]
lines += ['# SACP bandwidth robustness analysis', '']
lines += ['All results use only the frozen 2020 development-year sweep. They are robustness diagnostics, not an independent final-year validation.', '']
lines += ['## Best-vs-runner-up separation', '']
lines += [f'- 45 architecture × level × variable selections were examined.',
          f'- Median extra Winkler skill of the best bandwidth over the runner-up: **{statistics.median(gaps):.3f} percentage points**.',
          f'- Mean extra skill: **{statistics.fmean(gaps):.3f} pp**; maximum: **{max(gaps):.3f} pp**.',
          f'- **{100*small05:.0f}%** of selections have a best-vs-runner-up gap below 0.05 pp; **{100*small10:.0f}%** are below 0.10 pp.',
          '- Interpretation: the preferred bandwidth rank is often stable, but the objective surface is shallow. This supports cautious "scale preference" language rather than a strong physical-scale claim.', '']
lines += ['## Leave-one-combination-out transfer', '']
lines += ['For each variable, one of the nine architecture × nominal-level combinations is held out. The bandwidth is chosen from the other eight using mean dimensionless Winkler ratio, then evaluated on the held-out combination.', '']
lines += ['| Variable | CV-selected skill | fixed σ=1 skill | oracle skill | regret to oracle | oracle match | >= fixed σ=1 |',
          '|---|---:|---:|---:|---:|---:|---:|']
for r in summary_rows:
    if r['variable']=='ALL': continue
    lines.append(f"| {r['variable']} | {r['mean_selected_skill_pct']:.3f}% | {r['mean_fixed_sigma1_skill_pct']:.3f}% | {r['mean_oracle_skill_pct']:.3f}% | {r['mean_regret_to_oracle_pp']:.3f} pp | {100*r['oracle_match_fraction']:.0f}% | {100*r['beats_or_ties_fixed1_fraction']:.0f}% |")
lines += ['', f"Across all 45 held-out cases, the transferred selector achieves **{allrow['mean_selected_skill_pct']:.3f}%** mean skill versus **{allrow['mean_fixed_sigma1_skill_pct']:.3f}%** for fixed σ=1, with **{allrow['mean_regret_to_oracle_pp']:.3f} pp** mean regret to the within-case oracle. It matches the oracle bandwidth in **{100*allrow['oracle_match_fraction']:.0f}%** of cases.", '']
lines += ['## Leave-one-combination-out modal selections', '', '| Variable | modal selected σ | frequency |', '|---|---:|---:|']
for v,(s,f) in sel_modes.items():
    lines.append(f'| {v} | {s:g} | {100*f:.0f}% |')
lines += ['', 'These transfer tests reuse the same development year, so they demonstrate cross-architecture / cross-level robustness only. They do not replace a later unseen-year evaluation or a time-block bootstrap based on raw per-window scores.', '']
(OUT/'ROBUSTNESS_SUMMARY.md').write_text('\n'.join(lines), encoding='utf-8')
print('\n'.join(lines))
