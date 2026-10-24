"""Fill the project table from a verified common-episode validation snapshot."""
import hashlib
import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / 'project-page'


def main():
    snapshot_path = PAGE / 'sources/performance_validation_snapshot.json'
    snapshot = json.loads(snapshot_path.read_text())
    exits = json.loads((PAGE / 'sources/performance_validation_exits.json').read_text())
    runs = snapshot['runs']
    assert len(runs) == len(exits) == 10
    assert {r['experiment_id'] for r in runs} == {r['experiment_id'] for r in exits}
    assert all(r['exit']['return_code'] == 0 for r in exits)
    assert 0 < snapshot['common_episode_count'] <= snapshot['expected_episode_count'] == 1800
    stats = {}
    rows = []
    for model, name in [('gru1', 'EmbCLIP'), ('gawf1', 'EmbCLIP-GaWF')]:
        selected = sorted((r for r in runs if r['model'] == model), key=lambda r: r['seed'])
        assert [r['seed'] for r in selected] == snapshot['selected_seeds'][model] == list(range(5))
        stats[model] = {}
        cells = []
        for metric in ('sr', 'spl'):
            values = [r['common_' + metric] * 100 for r in selected]
            mean, std = statistics.mean(values), statistics.stdev(values)
            stats[model][metric] = dict(mean_percent=mean, std_percentage_points=std,
                                       per_seed_percent=values)
            value = f'{mean:.2f} ± {std:.2f}'
            cells.append(f'<td><strong>{value}</strong></td>' if model == 'gawf1' else f'<td>{value}</td>')
        benchmark = '<th scope="rowgroup" rowspan="2">RoboTHOR-Val</th>' if model == 'gru1' else ''
        label = f'<strong>{name}</strong>' if model == 'gawf1' else name
        rows.append(f'<tr>{benchmark}<th scope="row">{label}</th>{"".join(cells)}</tr>')
    assert all(stats['gawf1'][m]['mean_percent'] > stats['gru1'][m]['mean_percent'] for m in ('sr', 'spl'))
    caption = (f'5M-frame training. Mean ± SD across five runs per model on '
               f'{snapshot["common_episode_count"]:,} shared validation episodes.')
    table = ('<section id="performance">\n<h3>I. Performance</h3>\n'
             '<p>With lightweight 5M-step training (about 2.5% of EmbCLIP’s original '
             '<a href="https://arxiv.org/html/2111.09888v2">200M-step budget</a>, '
             'or 97.5% fewer environment steps), EmbCLIP-GaWF achieves higher validation '
             'SR and SPL than EmbCLIP.</p>\n'
             '<div class="table-scroll"><table class="quantitative">\n'
             f'<caption>{caption}</caption>\n'
             '<thead><tr><th scope="col" rowspan="2">Benchmark</th>'
             '<th scope="col" rowspan="2">Model</th>'
             '<th scope="colgroup" colspan="2">Object navigation</th></tr>\n'
             '<tr><th scope="col">SR (%) ↑</th><th scope="col">SPL (%) ↑</th></tr></thead>\n'
             f'<tbody>{"".join(rows)}</tbody>\n</table></div>\n</section>')
    index = PAGE / 'dist/index.html'
    html, count = re.subn(r'<section id="performance">.*?</section>', table, index.read_text(), count=1, flags=re.S)
    assert count == 1
    index.write_text(html)
    receipt = dict(dataset='RoboTHOR ObjectNav Val', training_frames=5001216,
                   reference_training_budget=dict(environment_steps=200000000,
                       source='https://arxiv.org/html/2111.09888v2', section='4.1',
                       actual_budget_percent=5001216/200000000*100,
                       fewer_environment_steps_percent=(1-5001216/200000000)*100,
                       excludes='CLIP pretraining and GPU-hours; comparison is per seed.'),
                   common_episode_count=snapshot['common_episode_count'],
                   expected_episode_count=snapshot['expected_episode_count'],
                   common_episode_ids_sha256=snapshot['common_episode_ids_sha256'],
                   seeds_per_model=5, seeds=list(range(5)),
                   aggregation='Per-seed SR/SPL on the intersection across all ten runs; equal-weight mean across seeds.',
                   standard_deviation='Sample standard deviation across seed scores (ddof=1), not SEM.',
                   units='Both means in percent; standard deviations in percentage points.',
                   coverage='All processes exited normally. Individual coverage is partial; only shared evaluated IDs enter the comparison.',
                   source_snapshot_sha256=hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
                   statistics=stats)
    for folder in (PAGE / 'sources', PAGE / 'dist/assets/sources'):
        (folder / 'performance_table.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
