# Official-checkpoint reproduction asset audit

Date: 2026-10-27

Data root: `/home/songxw/hjepa-data` -> `/mnt/disk2/songxw/hjepa-data`

## Preserved local runs

The following repaired, protocol-v2 training runs are user-requested artifacts and must not be
overwritten or deleted:

- `ckpts/ant/ant_hjepa_l2/seed42_paper_v2_20261027` (about 272 MiB)
- `ckpts/cube/cube_hjepa_l2/seed42_paper_v2_20261027` (about 624 MiB)
- `ckpts/fourroom/fourroom_hjepa_l2/seed42_paper_v2_20261027` (about 697 MiB)
- `ckpts/pusht/pusht_hjepa_l2/seed42_paper_v2_20261027` (about 624 MiB)

The existing `analyse/` directory is retained except for the three invalid frequency-analysis
artifacts documented below. New official-checkpoint results must use a separate output directory so
local-training measurements are not mixed with release-weight results.

## Deleted redundant data

- `cube_single_expert.h5` (101,942,558,720 bytes): source file used only by
  `scripts/data/split_cube.py`. The structurally validated 9,800-episode training split and
  200-episode validation split are retained and are the only Cube HDF5 files referenced by the
  training, probing, frequency, and evaluation configs.
- `.downloads/cube_single_expert.tar.zst` (46,184,624,478 bytes) and its range sidecar: compressed
  source for the already completed and validated Cube split.
- `.cache/huggingface/download/*.incomplete` and stale lock files: interrupted dataset-download
  cache, not experiment input.
- `droid/droid_raw_eval16.tar` (252,037,120 bytes): checksum-verified release archive removed after
  all 16 curated episodes (48 files, 252,280,708 bytes) were extracted and validated against the
  frozen clip manifest. The retained extraction and dataset revision are recorded in
  `droid/DROID_EVAL_ASSET_MANIFEST.json`.

## Replaced local runs

The following incomplete or obsolete non-protocol-v2 `seed42` run directories (about 5.1 GiB in
total) are removed before downloading the official release into the same standard paths:

- `ckpts/ant/ant_hjepa_l2/seed42`
- `ckpts/cube/cube_hjepa_l2/seed42`
- `ckpts/cube/cube_hjepa_l3/seed42`
- `ckpts/fourroom/fourroom_hjepa_l2/seed42`
- `ckpts/fourroom/fourroom_hjepa_l3/seed42`
- `ckpts/pusht/pusht_hjepa_l2/seed42`
- `ckpts/pusht/pusht_hjepa_l3/seed42`

These paths will contain release revision `08fc8a21208671a87741f2960aeba9b5db4420e0`
after download. Release checksums and byte sizes are recorded separately in
`OFFICIAL_WEIGHTS_MANIFEST.json`.

## Recomputed analysis artifacts

The original `analyse/spectral_centroids.csv`, `analyse/figure_12a.png`, and
`analyse/figure_12a.pdf` used FFT magnitude weights. On the checksum-verified official Ant dataset,
that implementation gives a 2.15x per-entity frequency gap and cannot reproduce the paper's stated
per-episode median of 6.4x. The three files were deleted and regenerated with periodogram weights
(`abs(rFFT) ** 2`), which give a 6.27x dataset-level Ant gap. The paper does not publish its exact
FFT weighting and cross-episode pooling formula, so `analyse/frequency_method.json` records this
inference and the remaining difference from the reported 8.3x value.

The root-level `figure_10_*` PNG/PDF files and root-level
`training_dynamics_per_seed.csv` / `training_dynamics_summary.csv` were also removed. They were
generated from incomplete local two-level seed-42 runs but were named like the paper's four-level,
three-seed Figure 10. Recomputed plots and CSVs are isolated under
`analyse/local_partial_training/`; filenames, titles, and `training_dynamics_manifest.json` identify
them as local partial trends and record the actual epoch coverage.

The initial `official_release_08fc8a21/figure_9b.png` and PDF were deleted after confirming that the
release lacks the paper's no-IDM collapsed-LeWM checkpoint. The same 27 valid released-model
evaluations are retained and rendered as `figure_9b_released_subset`, covering LeWM+IDM, HWM,
and H-JEPA without implying that the missing fourth bar was reproduced.
