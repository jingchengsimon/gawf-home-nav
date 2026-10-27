# RoboTHOR Challenge 2021 Test adaptation

> Historical audit: the unscored Challenge Test experiment is outside the current Val comparison. Its retired scripts are recoverable from Git history; current executable entry points are listed in [CODE_LAYOUT](CODE_LAYOUT.md).

The user authorized an independent official Test adaptation on 2026-10-24.
The existing AI2-THOR 5.0.0 training and validation environment is retained.
No training weights, model parameters or project-page scores are changed.

## Independent runtime

- Environment: `.envs/embclip-robothor-challenge2021`, AI2-THOR 2.7.2.
- Assets: `assets/embclip-robothor-challenge2021`.
- Official simulator commit: `bad5bc2b250615cb766ffb45d455c211329af17e`.
- Simulator archive: 579,427,830 bytes; SHA256
  `fba7b5f2fbb7a48f37d39cbf17d0d9301092945aaa0dd7215b20a85a421c287c`.
- Executable SHA256:
  `19d9b4735933c6ff013a05831a69acd3a10269b2934b38e51ff641b0202e01b1`.

The dedicated virtualenv owns the 2.7.2 client. It reads existing heavy dependencies
from the original environment after its own site-packages on the import path.
The setup compares the original environment's package list before and after;
they are identical. The simulator uses the existing user X display. No sudo,
display access changes or system package installation is used.

Official Challenge settings are 640×480 RGB, a 500-action limit, locobot,
stochastic actions, 0.25m movement, 30-degree rotations, 1m visibility distance,
vertical FOV 63.453048374758716 and no depth. The simulator build and raw rendering
resolution differ from our 5.0.0/400×300 training environment; the trained 224×224
RGB sensor processing is retained. This is an evaluation of existing checkpoints
on the official old build, not a claim to reproduce full-budget training.

## Verified compatibility

All 10 Challenge Test scenes load and execute an action. All 2,040 unique Test
episode records and all 10 final GRU1/GaWF1 checkpoints are hash checked; each
checkpoint has the correct training seed and 5,001,216 steps.

Current AllenAct configuration imports require AI2-THOR≥3.2.0. The adapter instead
constructs only the policy and its original RGB/category sensors and CLIP
preprocessor. It loads the GaWF class AST unchanged from its hash-checked training
source; no vendor code or reported simulator version is patched.

On SJC, comparison against the original 5M constructors verifies identical state
dictionaries, all 12 category mappings, normalization at both raw resolutions,
action probabilities, critic outputs and recurrent/feedback states. Four-step
sequences include reset and continuation. Maximum numerical difference is 0 for
both architectures. Both policies then execute a bounded Test-scene smoke with
the independent 2.7.2 controller. These smokes are not full Test evaluations.

## Official scoring is unavailable

The official repository submits trajectories to
`https://leaderboard.allenai.org/robothor_objectnav/submissions/public`.
That hostname currently returns NXDOMAIN, including a public DNS response whose
authority is the allenai.org DNS zone. SJC also cannot resolve it. No submission
can reach the official scoring service. Public Test episodes hide shortest paths,
and official inference intentionally omits Test success/SPL computation.

A working official scoring endpoint or organizer-provided official offline scorer
is the remaining prerequisite. No full five-seed Test runs have been launched or
reported as scored; no evaluation scores have been added to HTML.

### Offline-scoring audit (2026-10-24)

The public Challenge repository, its branches/issues, the official AI2-THOR metrics
utilities, and the newer ProcTHOR/Codebook evaluation implementation were checked.
No public official Challenge Test scorer or replacement submission endpoint was
found. A fresh DNS query still returns NXDOMAIN for the documented submission host.

The newer `allenai/object-nav-eval` loader explicitly skips RoboTHOR when loading
the Test split; its RoboTHOR files contain only Val/minival. It cannot supply the
missing Challenge Test scoring inputs.

A registered bounded evaluator-only probe ran on the unchanged official build,
with one Val positive control and one episode from each of the ten Test scenes.
Val returned 48 object records and a valid shortest path. All ten Test scenes
returned an empty object list and `GetShortestPath` returned `null`, even though
the API's `lastActionSuccess` was true. `Stop` also returned true with no scoring
payload; this flag is not navigation success. No policy was loaded for this probe.

This is intentional in the official build: `AgentManager.stripMetadataForTest`
clears object metadata and non-allowlisted action returns for `FloorPlan_test*`.
The general `compute_single_spl` utility exists, but still requires the missing
success label and shortest path. The audit does not modify or bypass this filter.

Probe ID: `nav-robothor-embclip-challenge2021-offline-scoring-audit-sjc-v1-20261024`.
Verified receipts and pinned source links:
`experiments/deployment/sjc-embclip-challenge2021-offline-scoring-audit-20261024.json`.
The requested ten full Test evaluations remain unlaunched because they could
produce trajectories but not the requested official SR/SPL.

Sources: [official Test metadata filter](https://github.com/allenai/ai2thor/blob/bad5bc2b250615cb766ffb45d455c211329af17e/unity/Assets/Scripts/AgentManager.cs#L74-L104),
[official evaluation dataset loader](https://github.com/allenai/object-nav-eval/blob/main/main.py),
[official Stop-status clarification](https://github.com/allenai/robothor-challenge/issues/30#issuecomment-1249883099).

## Sources and receipts

- `scripts/embclip/challenge2021/prepare_runtime.py`: pinned download and setup.
- `scripts/embclip/challenge2021/policy_adapter.py`: architecture-only loader.
- `scripts/embclip/challenge2021/check_policy_equivalence.py`: numerical gate.
- `scripts/embclip/challenge2021/preflight.py`: scene and inference preflight.

Execution uses frozen source copies in independent registered SJC result leaves.
Successful preflight ID:
`nav-robothor-embclip-challenge2021-preflight-sjc-v3-20261024`.
Failed compatibility attempts remain in their original leaves.

The source hashes, verified receipts and remaining prerequisite are summarized in
`experiments/deployment/sjc-embclip-challenge2021-readiness-20261024.json`.

References: [official Challenge](https://github.com/allenai/robothor-challenge),
[official build download](https://github.com/allenai/robothor-challenge/blob/main/robothor_challenge/scripts/download_thor_build.py).
