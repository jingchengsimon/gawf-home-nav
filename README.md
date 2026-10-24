# GaWF Visual Navigation

A lightweight embodied navigation project comparing recurrent memory models with
GaWF, which gates recurrent weights using feedback from the preceding step.

- **RoboTHOR ObjectNav:** EmbCLIP GRU512 vs parameter-matched GaWF1160, RGB and
  target category, 5M training frames per seed, five seeds per model.
- **Gibson PointNav v2:** DD-PPO LSTM vs GaWF, RGB and a fixed initial goal cue,
  1M training frames. No online GPS, compass or depth is supplied to the policy.

The project page presents paired navigation videos and twelve goals with paired RGB/gradient views.
Validation SR/SPL and training figures remain available in the project assets.
Top-down maps and true poses are used only by
visualization/evaluation; distance-based rewards are training supervision.
These lightweight results are not a reproduction of the original papers' scores.

## Project page

[Open the project website](https://jingchengsimon.github.io/gawf-home-nav/).
GitHub Pages publishes `project-page/dist` automatically when its files are
pushed to `sjc`.

For a local preview:

```sh
python3 scripts/serve_project_page.py
```

Open http://127.0.0.1:8766/. The page also opens directly from
`project-page/dist/index.html` with its assets beside it.

## Code

Model adapters and GaWF are in `scripts/embclip/`; RGB-only PointNav configuration
is in `configs/pointnav2020/light_rgb/`. Simulator dependencies, licensed datasets
and checkpoints must be prepared separately. See `vendor/sources.json` for the
pinned upstream repositories; model and simulator execution requires their
compatible environments.

Built on [EmbCLIP / AllenAct](https://github.com/allenai/embodied-clip) and
[Habitat DD-PPO](https://github.com/facebookresearch/habitat-lab).
Visualization layout references [Codebook](https://embodied-codebook.github.io/).
