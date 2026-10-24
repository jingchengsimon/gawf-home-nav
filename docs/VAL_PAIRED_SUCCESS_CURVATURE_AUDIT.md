# Paired-success curvature audit

This is read-only postprocessing of the ten existing lightweight RoboTHOR Val
runs (GRU1 and GaWF1, seeds 0–4, 5,001,216 frames). No policy was rerun. No project
page file was changed. The audit verifies the parent behavior snapshot, original
curvature source, raw metrics, checkpoint identities, seeds and clean exits.
All 17,830 original episode curvature scores reproduce within 1e-12.

## What the original curvature measures

The original score uses finite differences of logged x-z positions, sampled once
per action, then averages curvature over those samples. Stationary-speed samples
(at most 1e-4 m/action) and paths with fewer than three positions contribute zero.
This measures position-path bending; it does not measure rotation actions.

Consequently, in-place turns can score zero, repeated stationary positions can
dilute an episode's average, and premature stopping can truncate difficult paths.
In the full common Val set, GRU's fraction of episodes with less than 1 cm of
total travel is 28.09%, versus GaWF's 11.03%. Failed End actions within the first
10 actions occur in 24.40% versus 7.26% of episodes, respectively (five-seed means).
These are descriptive diagnostics, not causal attribution of the score gap.

One actual GRU example, seed4 `FloorPlan_Val3_5_BasketBall_6`, has 500 actions,
300 rotation commands, zero travel and zero curvature. The limitation also affects
GaWF: seed3 `FloorPlan_Val2_2_Television_3` has 500 actions, 296 rotations,
zero travel and zero curvature.

## Compare successful navigation on the same episodes

For each training seed index, retain exactly those episode IDs where both models
succeeded. Counts for seeds 0–4 are 148, 153, 162, 153 and 104. These are 720
episode-seed pairs, not 720 unique episodes. Average within each seed, then report
the five equally weighted seed means and sample SD (ddof=1).

| Metric | GRU1 | GaWF1 |
| --- | ---: | ---: |
| Original curvature (m⁻¹) | 0.130 ± 0.023 | 0.151 ± 0.032 |
| Rotation commands | 15.06 ± 4.45 | 24.92 ± 3.98 |
| All actions | 24.55 ± 6.64 | 40.93 ± 9.80 |
| SPL (%) | 63.56 ± 4.38 | 56.24 ± 4.36 |
| Expert-normalized SEL (%) | 53.96 ± 5.34 | 43.64 ± 3.13 |

SEL uses only episodes with a successful shared expert reference: 140, 144, 154,
144 and 97 per seed. Failed expert references are excluded symmetrically.
Selecting successes makes SR identically 100%; it is not a new benchmark SR.

For an even stricter check, 30 fixed episode IDs succeeded in all ten runs.
GRU/GaWF curvature is 0.062 ± 0.026 / 0.094 ± 0.035; rotation commands are
3.81 ± 1.26 / 8.56 ± 3.63. These are especially easy survivors and cannot represent
the entire Val distribution.

## Sensitivity to stationary samples

Simply dropping consecutive duplicate positions and retaining finite-difference
curvature is numerically fragile: central derivatives can approach zero at
reversals and produce very large scores. Do not use that diagnostic as a corrected
headline metric.

As a separate geometric diagnostic, sum the absolute angle changes between
successive displacement vectors and divide by traveled meters. Remove consecutive
displacements at most 1 cm, and require at least two segments and 0.5 m of travel
in both paired models. This is positional bending density (rad/m), not the
original curvature or a measurement of in-place yaw.

There are 97, 90, 102, 91 and 50 qualifying paired-success episodes per seed.
GRU is 0.880 ± 0.223 rad/m and GaWF 1.124 ± 0.163 rad/m. The ordering does not
reverse after excluding stationary and very short traces under this diagnostic.
Tests check stationary paths, idle-padding sensitivity, geometric diagnostic
invariance to idle padding, and a 180-degree positional reversal.

## Interpretation

Early stopping and in-place turns can depress the original aggregate curvature,
but they are not a sufficient explanation for GaWF's higher mean: it also uses
more turns/actions on the paired-success subsets. Full-Val SR/SPL and
success-conditioned efficiency answer different questions. The data support
higher overall task completion for GaWF under this training budget; they do not
support a general claim of fewer turns or smoother routes.

The machine-readable audit is saved locally at
`results/embclip-paired-success-curvature-audit-20261024/audit-with-bending.json`.
Source/output hashes and cohort counts are in
`experiments/deployment/sjc-embclip-paired-success-curvature-audit-20261024.json`.
