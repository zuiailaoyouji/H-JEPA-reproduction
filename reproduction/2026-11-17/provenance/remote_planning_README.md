# H-JEPA Results-Only Bundle

Run: 2026-10-28. Packaged: 2026-10-29.

## Contents

- hjepa-data/ckpts/: newly created or modified evaluation outputs only.
  All 9,600 simulation task chunks and 192 planning-cell outputs are included.
  This directory deliberately contains no model weights.
- hjepa-data/analyse/official_release_08fc8a21/: summary CSVs and plots.
  A few unchanged summary files are retained as standalone references.
  DROID evaluations were already present in the source archive; only their
  small summary tables and plot are included here, not old detailed outputs.
- hjepa-data/analyse/qualified_figure7/: Figure 7 CSV, PNG, PDF and manifest.
- logs/: control and node run logs; staging copies of results are excluded.
- provenance/: final workflow status, execution settings and cancellation record.
- benchmarks/: small performance reports, without benchmark input tensors.
- manifest.json: byte size, SHA256 and provenance for every payload file.

## Scope And Qualifications

219/219 selected planning cells completed. No main world-model training.
Probe/Decoder was cancelled; its changed logs/configuration are partial output,
not completed evaluations. No new world-model or Decoder checkpoint is included.
World-model parameter gradients were frozen for accelerated planning, while
action gradients stayed enabled; results are not claimed bitwise identical.
Figure 7 uses all 502,500 validation states, an anchor near (8, 8), and per-level
maximum-distance normalization. The paper did not publish the anchor or color
normalization; see cost_geometry_manifest.json for the qualification.
Official weights passed verification; all 25 project tests passed.

Dataset downloads, runtime environments, deployment packages, source model
weights, unchanged old detailed results and repeated result copies are excluded.
Original source files are not deleted or modified by this packaging operation.
