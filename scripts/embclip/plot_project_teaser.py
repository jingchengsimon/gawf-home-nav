"""Compose existing trajectory and Grad-CAM results into the project teaser."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIST = ROOT / 'project-page/dist'
SOURCE = ROOT / 'results/nav-robothor-embclip-visible-gradcam-selection-local-v2-20261023'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.lines import Line2D

    summary_path = SOURCE / 'summary.json'
    summary = json.loads(summary_path.read_text())
    assert summary['status'] == 'complete'
    assert json.loads((SOURCE / 'exit.json').read_text())['return_code'] == 0
    project = json.loads((DIST / 'assets/project-data.js').read_text().split(' = ', 1)[1].rstrip(';\n'))
    plant = next(row for row in project['demos'] if row['target'] == 'HousePlant')
    records = {row['target']: row for row in summary['records']}
    font = FontProperties(fname=DIST / 'assets/fonts/noto-sans-400.ttf', size=16)
    emphasis = FontProperties(fname=DIST / 'assets/fonts/noto-sans-700.ttf', size=22)
    fig = plt.figure(figsize=(14.8, 7.2), facecolor='white')
    grid = fig.add_gridspec(2, 5, width_ratios=[.60, 1.72, 1, 1, 1],
                           left=.003, right=.997, top=.84, bottom=.015,
                           wspace=.065, hspace=.08)
    checks = []
    first_row = []
    for row, model in enumerate(('gru1', 'gawf1')):
        label = fig.add_subplot(grid[row, 0]); label.axis('off')
        label.text(1, .5, 'EmbCLIP' if row == 0 else 'EmbCLIP-\nGaWF',
                   ha='right', va='center', fontproperties=emphasis, color='black')
        path = DIST / next(r for r in plant['models'] if r['model'] == model)['final_path']
        ax = fig.add_subplot(grid[row, 1]); ax.imshow(plt.imread(path)); ax.axis('off')
        ax.set_anchor('W')
        if row == 0:
            first_row.append(ax)
        checks.append(dict(model=model, kind='full_trajectory', episode_id=plant['episode_id'],
                           source=str(path.relative_to(ROOT)), sha256=digest(path)))
        for column, (target, title) in enumerate((('AlarmClock', 'Alarm Clock'),
                                                ('Laptop', 'Laptop'),
                                                ('SprayBottle', 'Spray Bottle')), start=2):
            record = records[target]['models'][model]
            cam_path = SOURCE / f'{target}_{model}_cam.npy'
            rgb_path = SOURCE / f'{target}_{model}_rgb.npy'
            assert digest(cam_path) == record['cam_sha256']
            assert digest(rgb_path) == record['rgb_sha256']
            cam, rgb = np.load(cam_path), np.load(rgb_path)
            assert cam.shape == (7, 7) and rgb.shape == (224, 224, 3)
            assert np.isfinite(cam).all() and np.isfinite(rgb).all()
            ax = fig.add_subplot(grid[row, column]); ax.imshow(rgb)
            ax.imshow(cam, cmap='jet', vmin=0, vmax=max(float(cam.max()), 1e-12),
                      alpha=.48, extent=(-.5, 223.5, 223.5, -.5), interpolation='bilinear')
            ax.axis('off')
            if row == 0:
                first_row.append(ax)
            checks.append(dict(model=model, kind='gradcam_overlay', target=target,
                               episode_id=record['episode_id'], cam_sha256=digest(cam_path),
                               rgb_sha256=digest(rgb_path)))
    # Align headings above panels despite their different aspect ratios.
    for ax, title in zip(first_row, ('House Plant', 'Alarm Clock', 'Laptop', 'Spray Bottle')):
        bounds = ax.get_position()
        fig.text((bounds.x0+bounds.x1)/2, .86, title, ha='center', va='bottom', fontproperties=font)
    trajectory, first_cam, _, last_cam = [ax.get_position() for ax in first_row]
    fig.text((trajectory.x0+trajectory.x1)/2, .945, 'Trajectory',
             ha='center', va='bottom', fontproperties=emphasis)
    fig.text((first_cam.x0+last_cam.x1)/2, .945, 'Grad-CAM',
             ha='center', va='bottom', fontproperties=emphasis)
    divider = (trajectory.x1+first_cam.x0)/2
    fig.add_artist(Line2D([divider, divider], [.015, .985], transform=fig.transFigure,
                          color='#666', linewidth=1, linestyle=(0, (3, 3))))
    output = DIST / 'assets/teaser-summary.png'
    fig.savefig(output, dpi=180, facecolor='white'); plt.close(fig)
    receipt = dict(rows=2, visual_columns=4, label_columns=1, source_summary_sha256=digest(summary_path),
                   raw_rgb_columns=0, image_sha256=digest(output), panels=checks,
                   layout=dict(width_ratios=[.60, 1.72, 1, 1, 1],
                               group_titles=['Trajectory', 'Grad-CAM'], dashed_separator=True),
                   emphasis='Group headings and model labels: Noto Sans Bold, 22pt.',
                   normalization='Same as full figures: per-map positive maximum, jet overlay alpha 0.48.',
                   comparison='Paired HousePlant trajectory episode; Grad-CAM uses separate successful episodes and each model\'s own view.')
    (DIST / 'assets/sources/teaser-summary.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(output)


if __name__ == '__main__':
    main()
