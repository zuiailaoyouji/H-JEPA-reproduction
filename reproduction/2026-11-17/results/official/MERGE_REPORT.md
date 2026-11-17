# H-JEPA Result Merge Report

Date: 2026-10-29

## Source

- Archive: `/home/songxw/hjepa-data/analyse/H-JEPA_results_only_20261029.zip`
- Archive SHA256: `9ed4a78ab2cd209770e40725084f12eadf4a5440f55c97a65d453907c9d0342d`
- Extracted audit copy: `/home/songxw/H-JEPA_results_only_20261029_extracted`
- Archive integrity: all 11,346 manifest payload files passed SHA256 verification.

## Merge Policy

- Imported only planning outputs under checkpoint `eval_*` directories.
- Excluded all remote `final_probing_decoding_eval_v2` files because the remote
  Probe/Decoder run was cancelled and incomplete.
- Kept the locally completed Probe and DROID outputs as authoritative.
- Regenerated all aggregate CSVs and plots from the merged checkpoint tree.
- Copied the qualified Figure 7 data and plots into `qualified_figure7/`.

## Coverage

- DROID: 27/27, local completed evaluation.
- Probe/Decoder: 18/18, local completed evaluation.
- Depth: 84/84, imported planning evaluation.
- Cost ladder: 108/108, imported planning evaluation.

The local Probe and DROID aggregate CSVs are byte-identical to the previously
verified `probe_droid_final` outputs.

## Recovery

- The 15 incomplete files backed up before the planning-result merge were
  removed after the merged outputs and all experiment counts were verified.
- Duplicate files accidentally extracted into `/home/songxw` were also removed
  after byte-for-byte comparison with the retained audit extraction.
- Both cleanup targets were moved to the system trash and remain recoverable
  until that trash is emptied.

## Qualification

The remote planning run froze world-model parameter gradients while retaining
action gradients and used the original float32 math-SDPA path. Its provenance
reports a maximum frozen-parameter benchmark action difference of
`0.00017464160919189453`, so the planning results are protocol-aligned but not
claimed to be bitwise identical.

Figure 7 uses all 502,500 validation states, an anchor near `(8, 8)`, and
per-level maximum-distance normalization. The paper did not publish the anchor
or color normalization, so Figure 7 remains explicitly qualified.
