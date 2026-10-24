# GaWF navigation project page

Published at https://jingchengsimon.github.io/gawf-home-nav/ through
`.github/workflows/project-page.yml`. Push changes to `project-page/dist` on
`sjc` to update the site. GitHub Actions uploads the committed static directory;
no simulator, checkpoint or training dependency is needed for deployment.

Open `dist/index.html`, keeping its `assets/` directory beside it. The page uses
local assets and buildless JavaScript; it performs no runtime data fetches.

For an HTTP preview with video seeking, run from the repository root:

```sh
python3 scripts/serve_project_page.py
```

Open http://127.0.0.1:8766/. Keep the server running while watching videos;
cached page text and posters can remain visible after it stops, but video loading
will fail. The code is published at https://github.com/jingchengsimon/gawf-home-nav.

## Presentation

The page uses locally bundled Noto Sans with a clear heading hierarchy.
Descriptions remain 16px; Results is 30px, part headings 22px, case headings
18px and centered gray captions 14px. Demos are ordered by object
name. Each model has a full trajectory, a live trajectory and its RGB view. One
image-mounted play control and an icon beside each object heading start both
videos; either control pauses the pair. Each method has a gray timeline with a
red elapsed portion. The shorter video holds its last frame until the longer one
finishes. Four object cases are shown: BaseballBat, GarbageCan, HousePlant and SprayBottle.
The new BaseballBat and SprayBottle cases have successful GaWF terminals whose
RGB targets have been visually checked. All source candidates are retained.

Video overlays, scene identifiers, seed selectors, score cards and developer
annotations are omitted from the visible page. Detailed provenance remains in
`dist/assets/sources/protocol.json`, the experiment manifests and result receipts.
All source recordings, including unselected and unsuccessful cases, are retained
in their SJC result leaves. Selected videos and small source snapshots are mirrored
locally. A two-row teaser combines the enlarged HousePlant full trajectories with AlarmClock,
BaseballBat and SprayBottle Grad-CAM overlays. Results contains I. Performance (SR/SPL mean ± sample SD),
II. four paired navigation demos, and III. two RGB/gradient grids.
A dashed line separates Trajectory and Grad-CAM groups in the teaser.
Each visual section has a short introduction and one shared caption. Text-to-image
gaps are halved while the 48px margins between result parts are retained.
Existing quantitative and training figures are retained as assets but are not displayed.
The title links to the public GitHub repository; there is no footer link list.
The earlier page source and figures are in `archive/20261023-before-minimal/`;
the original media remain in `dist/assets/videos/`.

## Scientific boundary

Paired videos share an episode, actual initial pose, simulator build, sensors,
noise configuration, step limit and seed values. Top-down views, true poses and
object markers are evaluator-only. RoboTHOR policy inputs are RGB, target category
and permitted self-generated recurrent/action history. PointNav uses RGB and a
fixed initial goal cue. Distance-based rewards remain training supervision.

Qualitative examples are selected to illustrate observed advantages. They are not
an estimate of aggregate performance; the separately retained SR/SPL figure uses common validation
episode IDs across the evaluated models. Coverage and missing identities are
recorded in the source snapshots.

The twelve goals are split across two four-by-six image grids. Each row has
three adjacent RGB/Grad-CAM pairs, with one goal heading shared by each pair.
Model labels use normal weight and upright text. The grids use each model's own
successful validation terminal view. Most views are re-rendered from logged poses,
with recurrent state reconstructed from that model's re-rendered RGB/action
history. Eleven categories share an episode and initial pose; BaseballBat uses different successful episodes, marked
on the page. Visual inputs and histories can differ, and re-rendering is not
claimed pixel-identical to the original evaluation. The Vase and HousePlant GaWF panels use actual saved successful End decisions
from fixed-seed autonomous validation rollouts; their baselines retain verified
pose-replayed successful terminal views. Maps show local gradient
sensitivity, not intrinsic attention or a causal comparison holding observations
fixed. No model weights are updated. Earlier diagnostics remain in local history.

The displayed AlarmClock, HousePlant, Laptop and Vase examples are selected
from successful terminal-view candidates after checking the visible RGB target;
HousePlant, Laptop and Vase selection additionally considers overlap between the target and
the unchanged End-action Grad-CAM. This is an illustrative selection, not a
random sample or evidence of typical localization accuracy. Candidate histories,
selection identities and source hashes are retained in the attribution receipt.

The maps use gradients of the End-action log probability with respect to the
frozen CLIP RN50 final 2048-channel 7-by-7 feature tensor. Spatially averaged
gradients weight each channel, the weighted channels are summed, and negative
values are removed. The 7-by-7 result is enlarged and overlaid on the original
RGB using the same within-map maximum normalization rule for both models. It is
not an individual input-channel activation, an object detector, or native policy
attention. Red can reflect other context supporting stopping, so successful
navigation does not guarantee a target-centered map.

The exact backbone, fusion and recurrent dimensions are recorded in
`dist/assets/sources/embclip_feature_flow.json`. The current RoboTHOR policy uses
only the CLIP visual branch plus a learned category embedding; the paper's
Figure 5 is a separate zero-shot architecture with a CLIP text branch.

## Source entry points

```text
scripts/embclip/eval_light5m_remaining_sjc.py
scripts/embclip/record_minimal_suite_sjc.py
scripts/embclip/analyze_terminal_views_sjc.py
scripts/embclip/analyze_twelve_goals_sjc.py
scripts/embclip/analyze_visible_gradcam_candidates_sjc.py
scripts/embclip/sample_terminal_gradcam_sjc.py
scripts/embclip/assemble_terminal_gradcam_selection.py
scripts/embclip/plot_twelve_goal_gradcam.py
scripts/embclip/collect_project_training.py
scripts/embclip/plot_gru1_gawf1_val.py
scripts/build_project_assets.py
scripts/build_minimal_project_page.py
```

Simulator recording and model analysis run only on SJC. New execution sources are
frozen copies in independent result leaves. Matplotlib figures are regenerated
locally from verified source snapshots. Media asset names include a source and
transform fingerprint to avoid reusing videos from different recordings.

## Layout and wording reference

Font assignments were checked against the [Codebook project page](https://embodied-codebook.github.io/)
and its `static/css/index.css`: Noto Sans for body text; Google Sans for publication
title, authors and teaser. This page uses only Noto Sans to retain a single typeface.
The bundled fonts include their SIL Open Font License. New headings, descriptions
and captions adapt the reference page’s analysis and behavior sections to our
lightweight RoboTHOR comparison; they do not claim the Codebook mechanism,
curvature or SEL measurements. Reference and font records are in `sources/`.

Rebuild the teaser from the existing verified arrays and full trajectories with:

```sh
MPLCONFIGDIR=/tmp/gawf-project-mpl python scripts/embclip/plot_project_teaser.py
```

The teaser uses the same overlay normalization as the full Grad-CAM grids.
Its source hashes and panel identities are in `dist/assets/sources/teaser-summary.json`.

## Performance table

The table uses all five normally exited evaluation runs per model, restricted
to the same intersection of 1,783 validation episode IDs across all ten runs.
Individual runs cover 1,783 or 1,785 of the expected 1,800 episodes; this table
does not claim full coverage. SR and SPL are percentages. The mean gives each
seed equal weight; SD is the sample standard deviation across seeds (ddof=1),
not SEM. The 5M label describes the 5,001,216-frame training checkpoints.

Rebuild with `python scripts/embclip/build_performance_table.py`. The fresh
read-only SJC collector snapshot, actual exit receipts and unrounded statistics
are in `sources/performance_validation_snapshot.json`,
`sources/performance_validation_exits.json` and `sources/performance_table.json`,
with corresponding copies under `dist/assets/sources/`.

The Performance introduction states that the 5M-step comparison uses about 2.5%
of the paper’s 200M RoboTHOR budget (97.5% fewer environment steps per seed),
with a link to Section 4.1. This excludes CLIP pretraining and does not imply
matching the paper’s full-training or test protocol. Checkpoint/configuration,
optimizer and recorded-demo checks are in `sources/training_budget_audit.json`.

## Method figure

The Method section follows the teaser and precedes Results. Its three panels
reuse the supplied PDF’s left GaWF cell and Figure 2 of EmbCLIP. The cell’s
internal artwork is imported unchanged. The two architecture panels retain the
original Figure 2 paths, with Model renamed GRU/GaWF and the input photograph
replaced by the same verified RoboTHOR Laptop RGB frame. The GaWF panel adds
a delayed feedback arrow. The description identifies this signal as previous
detached raw policy logits; no model parameters or training configuration change.
The figure and source hashes are recorded in `sources/method_figure.json`.

`scripts/build_method_figure.py` uses pypdf, reportlab, numpy, Pillow and Poppler
to compose existing PDF artwork and render the static web image. Its source PDFs
and checked RGB frame are explicit arguments; no generative drawing is used.
