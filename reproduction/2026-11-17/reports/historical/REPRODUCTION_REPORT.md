# H-JEPA official-release reproduction report

Date: 2026-10-28

This report records the inputs, protocol decisions, generated artifacts, and verification for the
official-checkpoint reproduction. Numerical coverage and final metrics are filled only from completed
runs; paper values are never inserted as substitutes for unavailable experiments.

## Pinned inputs

- Paper: arXiv:2610.06805.
- Model release: `jepa-world-models/h-jepa`, revision
  `08fc8a21208671a87741f2960aeba9b5db4420e0`.
- Official weights: 93 checkpoints and 93 normalizers, verified against
  `ckpts/OFFICIAL_WEIGHTS_MANIFEST.json` (12,433,147,717 bytes across all release files).
- DROID evaluation assets: 16 curated episodes / 48 files, verified against
  `droid/DROID_EVAL_ASSET_MANIFEST.json`.
- Simulation evaluation: 50 fixed tasks per environment and paired model seeds 42, 43, and 44.
- DROID evaluation: training seeds 1, 1000, and 10000 crossed with planner seeds 1, 2, and 3.

## Result layout

- Official release: `analyse/official_release_08fc8a21/`.
- Qualified Figure 7: `analyse/qualified_figure7/`.
- Qualified Figure 12(a): root `analyse/figure_12a.*`, with the inferred method in
  `analyse/frequency_method.json`.
- Tables 3 and 11: `analyse/protocol_tables/`.
- Retained local partial training trends: `analyse/local_partial_training/`.
- Live runner state: `analyse/official_release_08fc8a21/runner_status_droid_probe_depth_cost.json`.

## Paper protocol implemented

- Figure 4 uses frozen probes from four-level Ant/FourRoom models over three seeds. Its right panel
  uses post-hoc level 1-3 decoders and is explicitly separated from the probe metrics.
- Figures 5 and 12(b) pair levels 1 and 2 of the released two-level models. Figure 5 aggregates all
  entities into one open point when the frequency gap is below the paper's 2x threshold; Figure
  12(b) retains entity-level points.
- Figure 6 (bottom) evaluates LeWM, H-JEPA, and HWM by hierarchy depth over the fixed 50-task suites.
- Tables 1 and 9 use native/projected cost levels from the same H-JEPA checkpoints and tasks.
- Figure 9(b) averages the 16 DROID clip fidelities first within training seed over planner seeds,
  then reports mean and seed spread for the three released models.
- Figure 10-style local curves merge resumed CSV fragments by global step, average within epoch, and
  then across seeds. The retained runs contain only L2 seed 42, so they are labeled as partial trends
  and contain no artificial multi-seed uncertainty band.
- Figure 12(a) standardizes each episode/entity dimension, excludes the FFT DC term, and pools
  periodogram-weighted frequencies. This is a qualified inference because the paper does not publish
  its exact FFT weighting or cross-episode pooling rule.

The host places this user's sessions under a one-core CPU quota (`cpu.max=100000/100000`). An
in-process (`num_workers=0`) loader was tested after an interruption, but six concurrent cells failed
to finish a first epoch in about 12 minutes versus about 6.5 minutes for the original six-worker
FourRoom configuration. The paper/repository setting `num_workers=6` was therefore restored. This
worker-count check did not change the dataset, seeded shuffle, batch size, epochs, optimizer, frozen
model, or metric computation, and produced no accepted result. The interrupted post-hoc heads had no
final metrics or per-epoch checkpoint, so the runner's completeness check rejected them and restarted
the cells from scratch. No official model checkpoint was retrained or modified.

The final queue runs in the persistent `hjepa_repro` tmux session so client messages cannot terminate
child evaluations. GPU 5 is excluded because it is occupied by another user. At the final restart,
GPU 0 and GPU 2 also had about 9.1 GiB and 15.7 GiB of external allocations, respectively, so the
same six-card pool was ordered as `1,3,4,6,0,2`: decoder-heavy cells go to otherwise empty cards and
the two shared cards receive the lightest available cells. All cards are 40 GiB A100s; the maximum
observed H-JEPA task allocation was about 7.3 GiB, leaving ample memory headroom.

## Completed results

- DROID released subset (mean fidelity +/- standard deviation over three training seeds):
  LeWM+IDM 33.5679 +/- 1.5242; HWM-L2 34.7389 +/- 3.5580; H-JEPA-L2 35.8636 +/- 1.5531.
- Spectral frequency gaps from the released simulation datasets: Ant 6.266x, FourRoom 2.294x,
  Cube 2.729x, Push-T 2.671x.
- Table 3 was recomputed from dataset lengths and recursive clip spans. Table 11 reproduced exact
  invariance of the global statistic and gradient norm under 1, 2, and 4 shards on a recorded new
  random instance; the paper's absolute random instance is unpublished.
- The remaining simulation/probe metrics are populated after the dynamic runner reaches complete
  coverage.

## Local partial training curves

The preserved `seed42_paper_v2_20261027` checkpoints are used only for separately labeled trend
plots. Available/paper epochs are Ant 1/5, Cube 5/8, FourRoom 7/7, and Push-T 5/9. The overview shows
prediction, SIGReg, and total loss by environment; the official release has no training logs, so it
cannot produce the paper's exact L4 three-seed Figure 10.

## Storage cleanup

About 144 GiB of redundant sources, interrupted downloads, and obsolete local runs were removed.
The validated split HDF5 files needed by probing, frequency analysis, and evaluation were retained.
The four user-requested `seed42_paper_v2_20261027` local runs were retained. Full byte-level details
and the intentionally replaced paths are in `ASSET_CLEANUP.md`.

## Public-release limits

The release does not contain every artifact needed for all paper figures. Figure 6 (top), Figure 8 /
Table 10, Figure 9(a,c) and the no-IDM Figure 9(b) bar, exact Figure 10, Figures 11 and 13-18, and
Table 12 cannot be claimed as exact release-weight reproductions. The specific missing checkpoint,
log, task, or protocol item for each result is listed in `REPRODUCTION_SCOPE.md`.

## Verification

Final strict coverage, tests, and rendering checks are recorded here after the active evaluation
queue completes.
