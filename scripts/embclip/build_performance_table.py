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
    official_path = PAGE / 'sources/official_light_validation.json'
    comparison = json.loads(official_path.read_text())
    assert behaviour['common_episode_count'] == snapshot['common_episode_count']
    assert behaviour['common_episode_ids_sha256'] == snapshot['common_episode_ids_sha256']
    assert comparison['common_episode_ids_sha256'] == snapshot['common_episode_ids_sha256']
    assert comparison['common_episode_count'] == snapshot['common_episode_count']
    assert comparison['sel_episode_count'] == behaviour['sel_episode_count']
    assert comparison['expert_summary_sha256'] == behaviour['expert_summary_sha256']
    assert comparison['behaviour_source_sha256'] == behaviour['source_sha256']
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
        for metric in ('sr', 'spl', 'sel'):
            verified = comparison['aggregates'][model]['common_'+metric]
            displayed = stats[model][metric]
            mean = displayed['mean'] if metric == 'sel' else displayed['mean_percent']
            std = displayed['std'] if metric == 'sel' else displayed['std_percentage_points']
            assert abs(mean-verified['mean_percent']) < 1e-12
            assert abs(std-verified['sample_std_percentage_points']) < 1e-12
        benchmark = '<th scope="rowgroup" rowspan="3">RoboTHOR</th>' if model == 'gru1' else ''
        label = f'<strong>{name}</strong>'
        rows.append(f'<tr>{benchmark}<th scope="row">{label}</th>{"".join(cells)}</tr>')
    official = comparison['aggregates']['official']
    official_runs = [r for r in comparison['runs'] if r['model'] == 'official']
    assert official['n'] == len(official_runs) == 1
    assert official_runs[0]['training_steps'] == 130091717
    assert official_runs[0]['checkpoint_sha256'] == 'b7867c833412212d1921ef3b66baae64d0eec18c6a8825ba4a2c5808407d50a2'
    stats['official'] = {metric: official['common_'+metric] for metric in ('sr', 'spl', 'sel')}
    assert all(record['sample_std_percentage_points'] is None for record in stats['official'].values())
    cells = ''.join(f'<td>{stats["official"][metric]["mean_percent"]:.2f}</td>' for metric in ('sr', 'spl', 'sel'))
    rows.append('<tr class="budget-reference"><th scope="row"><strong>EmbCLIP</strong> (130M)</th>'+cells+'</tr>')
    assert all(stats['gawf1'][m]['mean_percent'] > stats['gru1'][m]['mean_percent'] for m in ('sr', 'spl'))
    caption = ('We compare success rate (SR), success weighted by path length (SPL), and '
               'success weighted by episode length (SEL). SEL follows the metric introduced by '
               '<a href="https://embodied-codebook.github.io/">EmbCLIP-Codebook</a>.')
    table = ('<section id="performance">\n<h3>I. Performance</h3>\n'
             '<p>With lightweight 5M-step training (about 2.5% of <strong>EmbCLIP</strong>’s original '
             '<a href="https://arxiv.org/html/2111.09888v2">200M-step budget</a>), '
             '<strong>EmbCLIP-GaWF</strong> achieves higher overall '
             'success rate (SR) and success weighted by path length (SPL) than equally trained <strong>EmbCLIP</strong>.</p>\n'
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
                   official_reference=dict(model='EmbCLIP',split='RoboTHOR-Val',
                       training_environment_steps=130091717,
                       standard_deviation='One released checkpoint; no across-seed standard deviation.',
                       role='Measured on the same common validation episodes and shared SEL references.',
                       run=official_runs[0],statistics=stats['official'],
                       source_snapshot_sha256=hashlib.sha256(official_path.read_bytes()).hexdigest()),
                   reference_training_budget=dict(environment_steps=200000000,
                       source='https://arxiv.org/html/2111.09888v2', section='4.1',
                       actual_budget_percent=5001216/200000000*100,
                       fewer_environment_steps_percent=(1-5001216/200000000)*100,
                       excludes='CLIP pretraining and GPU-hours; comparison is per seed.'),
                   common_episode_count=snapshot['common_episode_count'],
                   expected_episode_count=snapshot['expected_episode_count'],
                   common_episode_ids_sha256=snapshot['common_episode_ids_sha256'],
                   seeds_per_model=5, seeds=list(range(5)),
                   aggregation='SR/SPL on the intersection across all eleven runs; five equal-weight seeds per lightweight model, one official checkpoint. Shared expert-reference subset for SEL.',
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
