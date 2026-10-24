# GaWF navigation project page

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

The page uses one font, one text size and black text. Demos are ordered by object
name. Each model has a full trajectory, a live trajectory and its RGB view. One
image-mounted play control and an icon beside each object heading start both
videos; either control pauses the pair. Each method has a gray timeline with a
red elapsed portion. The shorter video holds its last frame until the longer one
finishes. Four object cases are shown: Apple, GarbageCan, HousePlant and Vase. The Vase
case uses FloorPlan_Val1_1_Vase_6; GaWF succeeds and the recorded terminal RGB
visibly contains the blue vase.

Video overlays, scene identifiers, seed selectors, score cards and developer
annotations are omitted from the visible page. Detailed provenance remains in
`dist/assets/sources/protocol.json`, the experiment manifests and result receipts.
All source recordings, including unselected and unsuccessful cases, are retained
locally. The page currently shows four paired demos followed by two RGB/gradient grids.
Quantitative and training figures are retained as assets but are not displayed.
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
claimed pixel-identical to the original evaluation. The Vase GaWF panel uses an
actual saved successful End decision from a new
fixed-seed autonomous validation rollout; its baseline retains the original
pose-replayed successful terminal view. Maps show local gradient
sensitivity, not intrinsic attention or a causal comparison holding observations
fixed. No model weights are updated. Earlier diagnostics remain in local history.

The displayed AlarmClock, HousePlant, Laptop and Vase examples are selected
from successful terminal-view candidates after checking the visible RGB target;
Laptop and Vase selection additionally considers overlap between the target and
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
