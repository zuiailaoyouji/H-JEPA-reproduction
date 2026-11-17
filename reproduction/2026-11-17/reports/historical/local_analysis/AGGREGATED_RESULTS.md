# H-JEPA unified result aggregation

Collected at 2026-10-27T13:41:57+08:00 from `/home/songxw/hjepa-data/ckpts`.

Missing results remain missing; they are never converted to zero. Local training, fresh protocol-2 training, and external/unknown weights are separate provenance classes. Planning summaries first average planner seeds within each training seed, then compute mean and SE across training seeds.

## Coverage

| Env | Model | Run | Source | Protocol | Training | Planning | Posthoc | Sessions | Resume warning |
|---|---|---|---|---|---|---|---|---:|---|
| ant | ant_hjepa_l2 | seed42 | local_training | legacy_or_unspecified_b32_accum4 | incomplete_stale | blocked_on_incomplete_training | blocked_on_incomplete_training | 1 | False |
| ant | ant_hjepa_l2 | seed42_paper_v2_20261027 | local_fresh_training | v2_b128_accum1 | in_progress | pending_training | pending_training | 1 | False |
| cube | cube_hjepa_l2 | seed42 | local_training | legacy_or_unspecified_b32_accum4 | complete | complete | complete | 1 | False |
| cube | cube_hjepa_l2 | seed42_paper_v2_20261027 | local_fresh_training | v2_b128_accum1 | in_progress | pending_training | pending_training | 2 | True |
| cube | cube_hjepa_l3 | seed42 | local_training | legacy_or_unspecified_b32_accum4 | incomplete_stale | blocked_on_incomplete_training | blocked_on_incomplete_training | 1 | False |
| fourroom | fourroom_hjepa_l2 | seed42 | local_training | legacy_or_unspecified_b32_accum4 | complete | complete | complete | 1 | False |
| fourroom | fourroom_hjepa_l2 | seed42_paper_v2_20261027 | local_fresh_training | v2_b128_accum1 | in_progress | pending_training | pending_training | 1 | False |
| fourroom | fourroom_hjepa_l3 | seed42 | local_training | legacy_or_unspecified_b32_accum4 | complete | complete | missing | 1 | False |
| pusht | pusht_hjepa_l2 | seed42 | local_training | legacy_or_unspecified_b32_accum4 | complete | complete | complete | 5 | False |
| pusht | pusht_hjepa_l2 | seed42_paper_v2_20261027 | local_fresh_training | v2_b128_accum1 | in_progress | pending_training | pending_training | 1 | False |
| pusht | pusht_hjepa_l3 | seed42 | local_training | legacy_or_unspecified_b32_accum4 | incomplete_stale | blocked_on_incomplete_training | blocked_on_incomplete_training | 1 | False |

## Complete Planning Results

| Env | Model | Train seed | Planner seed | Evaluation | Metric | Value | Tasks | Protocol | Source |
|---|---|---:|---:|---|---|---:|---:|---|---|
| cube | cube_hjepa_l2 | 42 | 42 | automatic_cube_l2 | success_rate_pct | 30 | 50 | legacy_or_unspecified_b32_accum4 | local_training |
| fourroom | fourroom_hjepa_l2 | 42 | 42 | automatic_fourroom_l2 | success_rate_pct | 100 | 50 | legacy_or_unspecified_b32_accum4 | local_training |
| fourroom | fourroom_hjepa_l3 | 42 | 42 | automatic_fourroom_l3 | success_rate_pct | 92 | 50 | legacy_or_unspecified_b32_accum4 | local_training |
| pusht | pusht_hjepa_l2 | 42 | 42 | automatic_pusht_l2 | success_rate_pct | 30 | 50 | legacy_or_unspecified_b32_accum4 | local_training |

## Summary Groups

| Env | Model | Evaluation | Metric | Seeds | Mean | SE | Protocol | Source |
|---|---|---|---|---:|---:|---:|---|---|
| cube | cube_hjepa_l2 | automatic_cube_l2 | success_rate_pct | 1 | 30 | NA | legacy_or_unspecified_b32_accum4 | local_training |
| fourroom | fourroom_hjepa_l2 | automatic_fourroom_l2 | success_rate_pct | 1 | 100 | NA | legacy_or_unspecified_b32_accum4 | local_training |
| fourroom | fourroom_hjepa_l3 | automatic_fourroom_l3 | success_rate_pct | 1 | 92 | NA | legacy_or_unspecified_b32_accum4 | local_training |
| pusht | pusht_hjepa_l2 | automatic_pusht_l2 | success_rate_pct | 1 | 30 | NA | legacy_or_unspecified_b32_accum4 | local_training |

Machine-readable files: `aggregated_results.json`, `aggregated_runs.csv`, `aggregated_coverage.csv`, `aggregated_planning_results.csv`, `aggregated_probe_results.csv`, and `aggregated_planning_summary.csv`.
