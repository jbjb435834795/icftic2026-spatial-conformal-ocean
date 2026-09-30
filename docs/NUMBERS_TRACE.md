# NUMBERS_TRACE.md — 论文表格数字到来源文件的映射

This file maps every table entry of the paper to its source file and generating script.
本文件把 `the manuscript` 中每个表格条目
映射到证据包内的冻结结果文件。所有路径相对本包根目录。

## 表 I（2021 评估，90%，逐预报器）

| 数字 | 来源 |
|---|---|
| Coverage 列（split / SACP / normalized / weighted） | `evidence/final_2021/cp_results_final_2021.json` → `results.{family}.levels["0.9"].{method}.overall.coverage` |
| Coverage 列（online refresh / ACI，延迟反馈版） | 同上文件，`online` / `aci` 条目（本包内为延迟反馈重算值；瞬时反馈旧值存于 `online_instantfb` / `aci_instantfb`） |
| Skill (%) 列 | `evidence/final_2021/skill_ci_final_2021.json` → `results.{family}.{method}.skill_pct`（计算脚本 `code/skill_ci.py`，输入为 `delayed_partial_{family}_final_2021_arrays.npz` 的逐窗口逐变量 Winkler） |
| Skill 95% CI | 同上文件 `ci95` 字段 |
| Reject./60 列 | `evidence/final_2021/stats_final_2021.json` → `by_series["{family}_{method}_L90"].reject`（移动块 bootstrap + Holm，脚本 `code/stats_tests.py`，输入 `perwindow_final_2021.npz`） |

## 表 II（开发年带宽鲁棒性，2020）

| 数字 | 来源 |
|---|---|
| Modal σ / Stability | `results/best_sigma_by_model_level_variable.csv`（由 `code/robust_bandwidth_analysis.py` 从 `evidence/sacp_sweep_{model}_dev_2020.json` 计算） |
| Gap (pp) | `results/bandwidth_margin.csv` |
| LOO oracle match | `results/loo_combo_transfer.csv`、`results/loo_combo_summary.csv` |

## 表 III（名义水平间 SACP skill，2021）

| 数字 | 来源 |
|---|---|
| 80/90/95% skill | `evidence/final_2021/cp_results_final_2021.json` → `results.{family}.levels["{lv}"].{sacp,split}.overall.winkler` 按论文式(2)聚合；90% 的直接值另见 `skill_ci_final_2021.json` |

## 正文关键数字

| 数字 | 来源 |
|---|---|
| 冷舌区 split 最低格点覆盖 0.64 → SACP 0.81；<0.85 格点 20% → 1.4% | `evidence/final_2021/spatial_final_2021.npz` 键 `convlstm_{split,sacp,aci}_L90`（脚本 `code/fill_numbers.py` 的 cellfrac 宏） |
| SON 季节覆盖（图2） | `cp_results_final_2021.json` → `results.convlstm.levels["0.9"].{method}.by_season.SON.coverage` |
| 第12天 RMSE（如正文引用） | `cp_results_final_2021.json` → `results.{family}.point_rmse_by_lead_var[11]` |
| 窗口数 351（平年）/352（闰年 2020） | `evidence/final_2021/perwindow_final_2021.npz` 任一 `*_cover` 数组形状 (351,12,5)；dev_2020 为 (352,12,5) |

## 生成链条（脚本 → 输入 → 输出）

```
predictions/*.npz（冻结逐窗口预测，2019 校准 + 2020 开发 + 2021 终测）
  ├── code/evaluate_cp.py      → eval/partial_{family}_{split}.json        （split/normalized/cqr/weighted/gaussian + 瞬时反馈 online/aci）
  ├── code/sacp_eval.py        → eval/partial_{family}_sacp_{split}.json   （SACP σ 扫描与 σ=1 终测）
  ├── code/delayed_online_eval.py → eval/delayed_partial_{family}_{split}.json（延迟反馈 online/aci + 逐窗口 Winkler）
  ├── code/delayed_integrate.py    → 覆盖合并 npz/json（瞬时值备份于 instantfb_backup_{split}.npz）
  ├── code/stats_tests.py     → eval/stats_{split}.json（块 bootstrap + Holm）
  ├── code/skill_ci.py        → eval/skill_ci_final_2021.json（配对块 bootstrap CI）
  └── code/figures.py         → 图 1–3（fig3/fig4/fig5 PDF）
```

注：`predictions/*.npz`（逐窗口预测数组，约 1.4 GB）保留在实验室归档中，
未随包分发；包内 JSON/npz 足以复现论文全部表格数字与统计检验。
