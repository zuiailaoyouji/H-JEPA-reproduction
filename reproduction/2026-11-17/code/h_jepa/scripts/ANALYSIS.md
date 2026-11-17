# Paper analysis scripts

All commands below are run from `h_jepa/`. By convention, use
`$HJEPA_HOME/analyse` as the output root. Figure and table artifacts use the
paper number in their filename. Directory-scanning analyzers accept repaired local runs with
`training_protocol_version: 2` and runs listed in a pinned `OFFICIAL_WEIGHTS_MANIFEST.json`, so
obsolete local runs cannot be mixed into reproduced results.

The exact public-release matrix is launched by `scripts/run_official_reproduction.py` and validated
by `scripts/analyze_official_release.py`. See
`local_env/official_reproduction_20261027/REPRODUCTION_SCOPE.md` for the boundary between exact,
qualified, and unavailable paper experiments.

For the pinned release, the dynamic runner assigns the next pending cell whenever a GPU becomes
free and skips only outputs that pass strict completeness checks:

```bash
python scripts/run_official_reproduction.py \
  --experiments droid probe depth cost --gpus 0,1,2,3,4,6
```

`scripts/progress_dashboard.py` serves the runner state, GPU allocation, per-task progress, and ETA.
`scripts/finalize_official_reproduction.py` may run alongside the queue: it waits for complete
coverage with no failures, then executes the strict aggregators, full Figure 7 geometry pass, release
hash verification, and tests. Its state is written to `finalizer_status.json`; it never treats a
partial queue as ready for aggregation.

## Released evaluation entry points

Do not replace these with the post-processing scripts in this directory:

| Paper result | Released entry point | Inputs |
|---|---|---|
| Figure 6 (bottom) | `scripts/eval_depth.sh` | completed simulation checkpoints |
| Tables 1 and 9 | `scripts/eval_cost_ladder.sh` | completed H-JEPA checkpoints |
| Figure 9(b) | `scripts/eval_droid.sh` | completed DROID checkpoints and evaluation clips |

These commands run the evaluations and write the paper metrics beside each
checkpoint. Their seed requirements and output layouts are documented in the
repository README.

## Added analysis entry points

### Tables 3 and 11: protocol arithmetic and distributed SIGReg

```bash
python scripts/analyze_protocol_tables.py \
  --output-dir "$HJEPA_HOME/analyse/protocol_tables"
```

Table 3 is recomputed from the public HDF5 episode lengths and the loader's recursive clip-span
formula. Table 11 reruns the float64 sharding experiment for a recorded random seed and asserts that
the global statistic and gradient norm are identical under one, two, and four equal shards. The
paper does not publish the random seed for its fixed Gaussian samples and directions, so the manifest
labels Table 11 as an exact property reproduction on a new random instance rather than claiming the
same absolute values.

### Figures 4, 5, and 12: probes and temporal frequencies

```bash
python scripts/analyze_probes.py "$HJEPA_HOME/ckpts" \
  --model hjepa_l2 --model hjepa_l4 \
  --require-seeds 3 --output-dir "$HJEPA_HOME/analyse"

python scripts/analyze_frequency.py \
  --centroids-csv "$HJEPA_HOME/analyse/spectral_centroids.csv" \
  --probe-csv "$HJEPA_HOME/analyse/probe_summary.csv" \
  --output-dir "$HJEPA_HOME/analyse"
```

`analyze_frequency.py` does not load a checkpoint. Its built-in specification
covers the four released simulation datasets and automatically checks
`$HJEPA_HOME`, `~/hjepa-data`, then `~/.stable_worldmodel`. Omit
`--centroids-csv` for the initial HDF5 scan; supplying it reuses those statistics
when probe results become available. Spectral centroids use amplitude-normalized
trajectories and periodogram weights (`abs(rFFT) ** 2`). Figure 4 is drawn from the
four-level models, while Figures 5 and 12(b) strictly pair levels 1 and 2 from the
two-level models. Figure 4 is saved as `figure_4_left` for probe curves and
`figure_4_right` for seed-42 Ant/FourRoom post-hoc decoder examples. In Figure 5,
environments below the paper's 2x frequency-gap threshold contribute one open,
entity-averaged "all retained" point; Figure 12(b) retains every entity. Figure 12 in the paper also has
Humanoid and DROID entries; supply a JSON `--spec` with those datasets and
entity columns when those unreleased inputs are available. An optional
top-level `roles` mapping can label entities as `retained` or `discarded` for
Figure 5. The script never fills missing environments with paper numbers.
`probe_summary.csv` is restricted to the entities listed in paper Table 6; any configured auxiliary
targets remain auditable in `probe_all_targets_summary.csv` and the per-seed file.

### Figure 7: AntMaze cost geometry

```bash
python scripts/analyze_cost_geometry.py CHECKPOINT \
  --anchor-xy X Y --output-dir "$HJEPA_HOME/analyse"
```

The script encodes every validation-dataset state by default and reports Euclidean distance to one
fixed anchor at each level. The paper does not publish the anchor coordinates or its 0-1 color
normalization; the figure divides each level by its maximum while `cost_geometry.csv` retains both
raw and normalized distance, and `cost_geometry_manifest.json` records the qualification. Use
`--num-points` only for an explicitly labeled subsampled diagnostic.

### Figure 8 and Table 10: prediction-cost monotonicity

```bash
python scripts/analyze_cost_monotonicity.py HWM_SEED42_CKPT HJEPA_SEED42_CKPT ... \
  --num-episodes 64 --output-dir "$HJEPA_HOME/analyse"
```

Each level-1 rollout is re-anchored at the true state at the start of every
level-2 segment. The script uses held-out expert actions, projects predictions
into level 2, and reports negative Spearman correlation with time.

### Figure 9(c): DROID compute sweep

```bash
python scripts/analyze_droid_compute_sweep.py SWEEP_ROOT \
  --compute-axis optimizer_work --require-seeds 3 \
  --output-dir "$HJEPA_HOME/analyse"
```

Existing planning records contain sample counts, horizons, optimizer
iterations, and wall-clock time, but no measured FLOPs. Therefore the default
x-axis is the explicitly named work proxy `samples x horizon x iterations`.
Use `--compute-axis tflops` only with evaluation records augmented with a
measured `flops` or `tflops` field; the script fails rather than relabeling a
proxy as TFLOPs.

### Figure 10: training dynamics

```bash
python scripts/analyze_training_dynamics.py "$HJEPA_HOME/ckpts" \
  --model hjepa_l4 --output-dir "$HJEPA_HOME/analyse"
```

Resumed CSV fragments are merged by global step. Curves are averaged within
epoch, then across seeds; bands are the seed minimum and maximum as in the
paper.

Exact Figure 10 reproduction requires four-level simulation models trained for the Table 4 budgets
(FourRoom 7, AntMaze 5, Push-T 9, Cube 8 epochs) and three seeds. Its separate DROID row requires
100-epoch LeWM+IDM and two-level H-JEPA logs. When only the retained local two-level seed-42 runs
are available, write them to a separately labeled directory and treat them only as partial trend
evidence:

```bash
python scripts/analyze_training_dynamics.py "$HJEPA_HOME/ckpts" \
  --model hjepa_l2 --output-dir "$HJEPA_HOME/analyse/local_partial_training"
```

The generated manifest records levels, completed epochs, the corresponding paper schedule, schedule
completion, and seed count. A single seed is plotted without a min/max band, and filenames use
`training_dynamics_partial_` rather than claiming an exact paper figure.
`training_dynamics_partial_simulation_overview` arranges the three simulation metrics by environment
in the same row/column orientation as the top portion of Figure 10.

### Figures 11 and 16: qualitative strips

```bash
python scripts/assemble_qualitative_figures.py 11 executions.csv \
  --output-dir "$HJEPA_HOME/analyse"
python scripts/assemble_qualitative_figures.py 16 droid_decodings.csv \
  --output-dir "$HJEPA_HOME/analyse"
```

Figure 11 manifest columns (`environment_steps` is optional but recommended;
it is a semicolon-separated list for the displayed frames):

```text
environment,frames_dir,goal_image,environment_steps
```

Figure 16 manifest columns:

```text
clip,expert_dir,plan_dir,fidelity,level2_indices
```

`level2_indices` is an optional semicolon-separated list of displayed plan
columns. The assembler does not generate missing simulator recordings or
DROID decoder rollouts.

### Figures 13--15, 17--18, and Table 12: external experiment grids

The public repository does not include the stagewise checkpoints, RoboCasa
outputs, anti-collapse sweep checkpoints, or collapse-ladder logs. Once those
runs have produced CSV manifests, render them with:

```bash
python scripts/analyze_ablations.py stagewise stagewise.csv --output-dir "$HJEPA_HOME/analyse"
python scripts/analyze_ablations.py calibration cube.csv --figure 14 --output-dir "$HJEPA_HOME/analyse"
python scripts/analyze_ablations.py calibration robocasa.csv --figure 15 --output-dir "$HJEPA_HOME/analyse"
python scripts/analyze_ablations.py anticollapse anticollapse.csv --output-dir "$HJEPA_HOME/analyse"
python scripts/analyze_ablations.py coefficients coefficients.csv --output-dir "$HJEPA_HOME/analyse"
python scripts/analyze_ablations.py collapse_table collapse.csv --output-dir "$HJEPA_HOME/analyse"
```

Required schemas:

| Kind | Required columns |
|---|---|
| `stagewise` | `procedure,level1_data,level2_data,seed,success_rate` |
| `calibration` | `configuration,episode,fidelity,outcome` |
| `anticollapse` | `sigreg,idm,detach,planner_seed,effective_rank,fidelity` |
| `coefficients` | `level,sigreg,idm,seed,fidelity` (optional `selected` marks the paper cell) |
| `collapse_table` | grouping columns such as `label,sigreg,time_repulsion`, plus the available Table 12 metrics |

The scripts compute statistics only from supplied runs. They intentionally do
not embed values reported by the paper.
