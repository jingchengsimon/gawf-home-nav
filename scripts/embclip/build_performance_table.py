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
    behaviour_path = PAGE / 'sources/behaviour_metrics.json'
    behaviour = json.loads(behaviour_path.read_text())
    assert behaviour['common_episode_count'] == snapshot['common_episode_count']
    assert behaviour['common_episode_ids_sha256'] == snapshot['common_episode_ids_sha256']
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
        for metric, precision, best in [('curvature', 3, 'gru1'), ('sel', 2, 'gawf1')]:
            record = behaviour['statistics'][model][metric]
            stats[model][metric] = dict(**record, units='1/m' if metric == 'curvature' else 'percent')
            if metric == 'sel':
                stats[model][metric]['episode_count'] = behaviour['sel_episode_count']
            value = f'{record["mean"]:.{precision}f} ± {record["std"]:.{precision}f}'
            if metric == 'sel':
                cells.append(f'<td><strong>{value}</strong></td>' if model == best else f'<td>{value}</td>')
        benchmark = '<th scope="rowgroup" rowspan="2">RoboTHOR-Val</th>' if model == 'gru1' else ''
        label = f'<strong>{name}</strong>'
        rows.append(f'<tr>{benchmark}<th scope="row">{label}</th>{"".join(cells)}</tr>')
    rows.append('<tr class="paper-reference"><th scope="rowgroup">RoboTHOR-Test</th>'
                '<th scope="row"><strong>EmbCLIP</strong> (200M, paper)</th>'
                '<td>47.00</td><td>20.00</td><td>-</td></tr>')
    assert all(stats['gawf1'][m]['mean_percent'] > stats['gru1'][m]['mean_percent'] for m in ('sr', 'spl'))
    caption = ('We compare success rate (SR), success weighted by path length (SPL), and '
               'success weighted by episode length (SEL). SEL follows the metric introduced by '
               '<a href="https://embodied-codebook.github.io/">EmbCLIP-Codebook</a>.')
    table = ('<section id="performance">\n<h3>I. Performance</h3>\n'
             '<p>With lightweight 5M-step training (about 2.5% of <strong>EmbCLIP</strong>’s original '
             '<a href="https://arxiv.org/html/2111.09888v2">200M-step budget</a>), '
             '<strong>EmbCLIP-GaWF</strong> achieves higher overall '
             'success rate (SR) and success weighted by path length (SPL) than <strong>EmbCLIP</strong>.</p>\n'
             '<div class="table-scroll"><table class="quantitative">\n'
             f'<caption>{caption}</caption>\n'
             '<thead><tr><th scope="col" rowspan="2">Benchmark</th>'
             '<th scope="col" rowspan="2">Model</th>'
             '<th scope="colgroup" colspan="3">Object navigation</th></tr>\n'
             '<tr><th scope="col">SR (%) ↑</th><th scope="col">SPL (%) ↑</th>'
             '<th scope="col">SEL (%) ↑</th></tr></thead>\n'
             f'<tbody>{"".join(rows)}</tbody>\n</table></div>\n</section>')
    index = PAGE / 'dist/index.html'
    html, count = re.subn(r'<section id="performance">.*?</section>', table, index.read_text(), count=1, flags=re.S)
    assert count == 1
    index.write_text(html)
    receipt = dict(dataset='RoboTHOR ObjectNav Val', training_frames=5001216,
                   paper_reference=dict(model='EmbCLIP',split='RoboTHOR-Test',
                       training_environment_steps=200000000,sr_percent=47.0,spl_percent=20.0,
                       source='https://arxiv.org/html/2111.09888v2',table=1,
                       standard_deviation='Not reported; no uncertainty is invented.',
                       role='Literature reference, not a run on our common validation episode set.'),
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
                   units='SR/SPL/SEL are percentages; curvature is 1/m. SD uses the same unit as each metric.',
                   coverage='All processes exited normally. Individual coverage is partial; only shared evaluated IDs enter the comparison.',
                   source_snapshot_sha256=hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
                   statistics=stats)
    receipt['extra_metric_protocol'] = {k: behaviour[k] for k in
        ('curvature_definition', 'stationary_limit', 'sel_definition', 'sel_episode_count', 'expert_summary_sha256')}
    receipt['extra_metric_source_sha256'] = behaviour['source_sha256']
    for folder in (PAGE / 'sources', PAGE / 'dist/assets/sources'):
        (folder / 'performance_table.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
