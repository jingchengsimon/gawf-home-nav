# Lightweight RoboTHOR behavior analysis

This analysis compares the existing five GRU1 and five GaWF1 5,001,216-step
checkpoints; it does not train or alter any policy. The 1,783 common evaluated
Val episode IDs and existing SR/SPL remain unchanged.

## Definitions

- **Planar curvature (lower is better):** magnitude of
  `(dx * ddz - dz * ddx) / (dx² + dz²)^(3/2)` on logged x-z positions.
  Derivatives use `numpy.gradient` at action-time samples. Average first within
  each episode, then across the same 1,783 episodes. Speed≤1e-4 m/action
  contributes zero, including stationary paths. No smoothing or resampling.
  Units are inverse meters. Position curvature cannot measure in-place turns.
- **Expert-normalized SEL (higher is better):** mean `S*w/max(w,e)`.
  `e` is the actual policy action count including End; `w` is a shared successful
  reference from AI2-THOR's ObjectNavExpertAction, with action noise disabled,
  the same starts, object categories, 30° rotations, 0.25m movement and 1m
  visibility. End counts for both. The reference is not guaranteed globally
  optimal. This is our documented adaptation, not Codebook-equivalent scoring.

Codebook's paper describes an expert for its reference action counts; its current
public repository (commit `ed4feaf4944a01ad1d8b12a3dfc46f2c9cb4c992`, 32 Python
files inspected) does not include its Curvature/SEL calculations or that expert.
The user explicitly accepted a reference method for GRU versus GaWF without
matching Codebook's numbers. The project's sentence “Lower SEL” conflicts with
its own formula and upward table arrow; higher SEL is used here.

## Coverage and results

The expert attempted all 1,783 shared episodes. It produced 1,516 valid reference
lengths; 267 failed, mainly because the built-in expert's navigation queries fail
or return no action. Failures are retained and never imputed as zero reference
length. All ten model runs use exactly the same 1,516 episode IDs for SEL.
SR/SPL and curvature use all 1,783 common episodes. Uncertainty is sample SD
across five training seeds (ddof=1), not SEM.

| Model | Curvature (m⁻¹) | Expert-normalized SEL (%) |
| --- | ---: | ---: |
| GRU1 | 0.098 ± 0.030 | 7.43 ± 1.22 |
| GaWF1 | 0.131 ± 0.031 | 7.95 ± 1.39 |

GaWF has higher mean SR, SPL and SEL but does not have lower mean curvature.
The selected videos illustrate particular shorter/smoother successful routes;
they must not be described as representative of all validation episodes.
GRU's mean episode length is 40.63 actions versus GaWF's 167.14, partly reflecting
premature stops and different success rates. Neither short episodes nor planar
curvature alone demonstrate effective exploration or few in-place rotations.

## Receipts

The registered parent is
`nav-robothor-embclip-val-expert-reference-sjc-v2-20261024` with eight independent
shards on GPU1. It reused 332 preserved serial records; the serial run is marked
superseded. All 1,783 episode records are unique. Expert data never enter an
actor, critic, recurrent state or feedback gate.

The parent result leaf contains `episodes.jsonl`, `summary.json`, `exit.json` and
`behaviour_metrics.json`; verified small summaries are also local. Public source
snapshots are `project-page/dist/assets/sources/behaviour_metrics.json` and
`performance_table.json`. The deployment audit records source and metric hashes.

Tests cover straight/stationary/short paths, a radius-two circle, rigid-transform
invariance, independent SEL recomputation, action/pose alignment, dataset and
result identity, unique episodes, checkpoint/metrics hashes and clean exits.

## Baseball Bat Grad-CAM

The selected GRU1 view uses `FloorPlan_Val1_5_BaseballBat_5`; GaWF1 uses
`FloorPlan_Val1_2_BaseballBat_3`. At selection time the chosen best seeds had no
shared successful BaseballBat episode. This exception was authorized by the
user. The other eleven goal columns share an episode, though terminal images
and recurrent histories differ. These maps are qualitative stopping-sensitivity
examples, not a controlled causal test of top-down feedback. No linear probing
or nearest-neighbor analysis is claimed.

Sources: [Codebook method and metrics](https://arxiv.org/html/2311.04193#S4.SS1.SSS1),
[official code](https://github.com/allenai/procthor-rl/tree/ed4feaf4944a01ad1d8b12a3dfc46f2c9cb4c992).
