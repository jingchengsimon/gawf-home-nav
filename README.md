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
| RoboTHOR models | [GRU config](scripts/embclip/light5m_gru1_config.py), [GaWF adapter](scripts/embclip/light_gawf1_config.py), [canonical GaWF core](scripts/embclip/canonical_core/gawf.py) |
| RoboTHOR training | [seed0/1 driver](scripts/embclip/run_light_5m_sjc.py), [seed2 driver](scripts/embclip/run_light_5m_sjc_seed2.py), [seed3/4 driver](scripts/embclip/run_light_5m_sjc_seed34.py) |
| RoboTHOR evaluation | [seed0/1 evaluator](scripts/embclip/eval_light5m_sjc.py), [seed2–4 evaluator](scripts/embclip/eval_light5m_sjc_seed234.py), [GaWF seed2/4 completion evaluator](scripts/embclip/eval_light5m_remaining_sjc.py), [official checkpoint evaluator](scripts/embclip/eval_official_val_sjc.py) |
| Shared-episode metrics | [SR/SPL/SEL comparison](scripts/embclip/compare_official_light_val.py), [table builder](scripts/embclip/build_performance_table.py) |
| RoboTHOR videos and Grad-CAM | [autonomous recording](scripts/embclip/record_minimal_suite_sjc.py), [logged-trajectory replay](scripts/embclip/render_logged_navigation_sjc.py), [terminal Grad-CAM](scripts/embclip/analyze_terminal_views_sjc.py) |
| Figure and page assembly | [teaser](scripts/embclip/plot_project_teaser.py), [RGB/Grad-CAM grids](scripts/embclip/plot_twelve_goal_gradcam.py), [video asset assembly](scripts/build_minimal_project_page.py), [page behavior](project-page/dist/assets/page.js) |
| PointNav RGB-only training | [configuration](configs/pointnav2020/light_rgb/protocol.json), [driver](scripts/run_pointnav2020_light_rgb.py), [trainer](scripts/train_pointnav2020_light_rgb_longrun.py), [GaWF adapter](scripts/pointnav2020_gawf.py) |
| PointNav RGB-only evaluation | [evaluator](scripts/eval_pointnav2020_light_rgb.py), [validation plot](scripts/plot_pointnav2020_rgb_eval.py) |

Experiment drivers use registered manifests and frozen source hashes, with separate
output directories for each run. Some recording/evaluation drivers execute frozen
copies from result directories; they are not generic one-command launchers.
Read the [RoboTHOR protocol](docs/ROBOTHOR_EMBCLIP.md) or
[PointNav guide](docs/POINTNAV.md) before preparing a new run. Existing completed
IDs and output directories should not be reused. Licensed data, checkpoints,
simulator environments and raw analysis arrays are prepared separately; pinned
upstream sources are recorded in [vendor/sources.json](vendor/sources.json).

## Project page

The website presents selected paired navigation examples, Grad-CAM views and the
RoboTHOR results table. Selected demos illustrate particular advantages; they do
not establish that GaWF uses fewer turns/actions on average over all Val episodes.
[Page source and asset notes](project-page/README.md).

For a local preview, including video seeking:

```sh
python3 scripts/serve_project_page.py
```

Open http://127.0.0.1:8766/. GitHub Pages publishes `project-page/dist` when those
files are pushed to `sjc`. The static assets can be viewed without installing a simulator;
regenerating figures or recordings additionally requires their source artifacts.

## Preserved history

Gibson PointNav v2, legacy RGB-D/GPS protocols, training scripts and experiment
records are retained. Current comparisons use the RGB-only configuration above;
older configurations and partial-seed plotting scripts are historical references.
No large directory migration or deletion is part of this documentation update.

Built on [EmbCLIP / AllenAct](https://github.com/allenai/embodied-clip) and
[Habitat DD-PPO](https://github.com/facebookresearch/habitat-lab).
Visualization and SEL references: [EmbCLIP-Codebook](https://embodied-codebook.github.io/).
