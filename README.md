# GaWF Visual Navigation

[Project website](https://jingchengsimon.github.io/gawf-home-nav/)

A lightweight comparison of recurrent navigation policies with GaWF, which
modulates recurrent weights using feedback from the preceding step.

## Experiments

- **RoboTHOR ObjectNav:** EmbCLIP GRU512 versus parameter-matched GaWF1160;
  RGB and a target category; five training seeds per model, 5M environment steps
  per seed, 60 training scenes and 16 parallel samplers. AI2-THOR 5.0.0, 500 actions
  per episode. Pretrained CLIP ResNet-50 stays frozen; the remaining navigation
  modules start from random initialization. The released 130M EmbCLIP checkpoint
  is an evaluation reference, not an initialization for our 5M models.
- **Gibson PointNav v2:** DD-PPO LSTM512×2 versus parameter-matched GaWF×2;
  RGB plus a fixed initial goal cue `g0`; one seed per model, 1M environment frames,
  eight training scenes and four validation scenes. No depth, online GPS/compass,
  updated goal vector or true pose enters the policy. [PointNav protocol and history](docs/POINTNAV.md).

Top-down maps, object annotations and true poses are evaluator-only. Distance-based
rewards are training supervision. These are lightweight experiments, not full-budget
reproductions of the original papers.

## Results

### RoboTHOR ObjectNav validation

| Model | SR (%) ↑ | SPL (%) ↑ | SEL (%) ↑ |
| --- | ---: | ---: | ---: |
| EmbCLIP (5M) | 13.38 ± 2.73 | 8.25 ± 1.40 | 7.43 ± 1.22 |
| EmbCLIP-GaWF (5M) | **19.55 ± 3.09** | **10.10 ± 0.93** | **7.95 ± 1.39** |
| EmbCLIP (130M, released checkpoint) | 45.20 | 21.54 | 24.53 |

The 5M results are mean ± sample SD across five training seeds. SR/SPL use the same
1,783 Val episode IDs across all eleven runs; SEL uses their shared 1,516 valid
expert references. The single official checkpoint has no across-seed SD and contains
130,091,717 actual training steps. SEL follows the metric suggested by
[EmbCLIP-Codebook](https://embodied-codebook.github.io/), using our documented
expert-reference adaptation. [Unrounded results and provenance](project-page/sources/performance_table.json).

### RGB-only Gibson PointNav v2 validation

| Model (1M, seed0) | SR (%) ↑ | SPL (%) ↑ |
| --- | ---: | ---: |
| LSTM2 | 0.00 | 0.000 |
| GaWF2 | 0.35 | 0.026 |

Each model was evaluated on the same 284 episodes. This is 0 versus 1 successful
episode, insufficient for a robust model-ranking claim. These RGB-only scores are
separate from historical RGB-D or GPS/compass results.
[Source snapshot](project-page/sources/pointnav_validation.json).

## Code entry points

| Purpose | Entry points |
| --- | --- |
| RoboTHOR models | [GRU config](scripts/training/embclip/config.py), [GaWF adapter](scripts/training/embclip/models.py), [canonical GaWF core](scripts/training/canonical_core/gawf.py) |
| PointNav core configuration | [RGB-only protocol](configs/pointnav2020/light_rgb/protocol.json), [task](configs/pointnav2020/light_rgb/task.yaml), [DD-PPO](configs/pointnav2020/light_rgb/ddppo.yaml) |
| Grad-CAM | [analysis and selection](scripts/analysis/embclip/gradcam.py), [figures](scripts/analysis/embclip/plots.py) |
| Trajectory visualization | [recording and replay](scripts/analysis/embclip/media.py) |

Setup and execution details: [RoboTHOR](docs/ROBOTHOR_EMBCLIP.md) and
[PointNav](docs/POINTNAV.md). Data, checkpoints and simulator environments are
prepared separately.

## Project page

The website presents selected paired navigation examples, Grad-CAM views and the
RoboTHOR results table. Selected demos illustrate particular advantages; they do
not establish that GaWF uses fewer turns/actions on average over all Val episodes.
[Page source and asset notes](project-page/README.md).

For a local preview, including video seeking:

```sh
python3 scripts/analysis/project/preview.py
```

Open http://127.0.0.1:8766/. GitHub Pages publishes `project-page/dist` when those
files are pushed to `sjc`. The static assets can be viewed without installing a simulator;
regenerating figures or recordings additionally requires their source artifacts.

## Code organization

[Current file tree, entry points and file counts](docs/CODE_LAYOUT.md).
Only the final training/evaluation implementations remain in `scripts/`.
Historical protocols, manifests, result snapshots and simulator environments are retained;
retired launchers and duplicate stage implementations remain recoverable from Git history.

Built on [EmbCLIP / AllenAct](https://github.com/allenai/embodied-clip) and
[Habitat DD-PPO](https://github.com/facebookresearch/habitat-lab).
Visualization and SEL references: [EmbCLIP-Codebook](https://embodied-codebook.github.io/).
