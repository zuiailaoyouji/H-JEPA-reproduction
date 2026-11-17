# H-JEPA official-release reproduction scope

Paper: arXiv:2610.06805. Model release: `jepa-world-models/h-jepa`, pinned at
`08fc8a21208671a87741f2960aeba9b5db4420e0`.

## Exact checkpoint-supported experiments

| Paper result | Public inputs | Reproduction |
|---|---|---|
| Figure 4, left | four-level Ant/FourRoom weights and probing data | frozen MLP probes, 3 model seeds |
| Figure 4, right | same weights/data | post-hoc decoders at levels 1-3 |
| Figure 5 / Figure 12(b), released subset | two-level simulation weights and probing data | FourRoom, Ant, Cube, Push-T; Humanoid and DROID inputs are not released |
| Figure 6, bottom | 84 simulation weights and 50 fixed tasks/environment | depth comparison, paired seeds 42/43/44 |
| Tables 1 and 9 | H-JEPA simulation weights and the same fixed tasks | native/projected cost ladder, paired seeds 42/43/44 |
| Figure 9(b), released subset | 9 DROID weights and 16 evaluation clips | exact protocol for LeWM+IDM, HWM, and H-JEPA; planner seeds 1/2/3 for training seeds 1/1000/10000 |

The runner enumerates 84 depth cells, 108 cost-ladder cells, 27 DROID cells, and 18
probe/decoder cells. It validates exact episode counts and refuses checkpoints not listed in the
pinned, hash-verified release manifest.

## Qualified analyses

| Paper result | Limitation |
|---|---|
| Figure 7 | The release supports latent cost maps, but the exact anchor and sampled state set are not specified. |
| Figure 12(a) | Data and sampling rate are public; the FFT weighting and exact cross-episode pooling formula are not. `frequency_method.json` records the inferred implementation. |
| Tables 3 and 11 | Reproducible directly from dataset/config arithmetic and the distributed-SIGReg implementation; they do not require weights. |
| Figure 9(b) | The slow-feature-collapsed LeWM without IDM is not released, so only the other three bars can be reproduced. |

## Not exactly reproducible from the release

| Paper result | Missing public artifact or protocol mismatch |
|---|---|
| Figure 6, top | Compute-sweep task set, full candidate grid, and measured planner FLOPs |
| Figure 8 / Table 10 | Paper uses a separately trained two-level model with `s2=3`; released simulation weights use `s2=2` |
| Figure 9(a), 9(c) | no-IDM/anti-collapse checkpoints and exact candidate/TFLOP sweep; the same missing no-IDM model also prevents the first bar of Figure 9(b) |
| Figure 10 | Per-step/epoch training logs are not included with the released weights. Retained local L2 seed-42 runs provide separately labeled 1-7 epoch trend plots, not the paper's L4, 3-seed simulation result (Table 4 uses 5-9 epochs per environment) or its separate 100-epoch DROID row. |
| Figure 11 | Selected task identities and execution recordings are not included |
| Figures 13, 16 | Stagewise Ant/DROID checkpoints are not included |
| Figures 14, 15 | Cube/RoboCasa outcome manifests and RoboCasa assets are not included |
| Figures 17, 18 / Table 12 | Anti-collapse, coefficient-grid, and collapse-ladder checkpoints/logs are not included |

Rendering scripts for unavailable experiment grids accept explicit result manifests, but they never
substitute paper numbers for missing runs. Existing local protocol-v2 training artifacts remain
separate from results produced by the official release.
