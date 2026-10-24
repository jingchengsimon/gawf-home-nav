"""Assemble twelve verified target examples without modifying their RGB or CAM arrays."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--candidates', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError('Output must be a new leaf')
    selected = json.loads(args.selection.read_text())
    summaries = {}
    leaves = {'original': args.original, str(args.candidates): args.candidates}
    for replacement in selected['replacements'].values():
        leaf = Path(replacement.get('analysis', str(args.candidates)))
        leaves[str(leaf)] = leaf
    for name, leaf in leaves.items():
        summary = json.loads((leaf / 'summary.json').read_text())
        if summary['status'] != 'complete' or json.loads((leaf / 'exit.json').read_text())['return_code'] != 0:
            raise RuntimeError('Source analysis has no verified successful exit')
        summaries[name] = summary
    records = []
    plan = []
    for old in summaries['original']['records']:
        target = old['target']
        if target in selected['replacements']:
            identity = selected['replacements'][target]['candidate_id']
            chosen_leaf = Path(selected['replacements'][target].get('analysis', str(args.candidates)))
            matches = [r for r in summaries[str(chosen_leaf)]['records'] if r['candidate_id'] == identity and r['target'] == target]
            if len(matches) != 1:
                raise RuntimeError('Candidate identity is not unique')
            record, leaf, prefix = copy.deepcopy(matches[0]), chosen_leaf, identity
        else:
            record, leaf, prefix = copy.deepcopy(old), args.original, target
        record['source_analysis'] = leaf.name
        record['source_summary_sha256'] = digest(leaf / 'summary.json')
        for model in ('gru1', 'gawf1'):
            for kind in ('cam', 'rgb'):
                source = leaf / (prefix + '_' + model + '_' + kind + '.npy')
                if digest(source) != record['models'][model][kind + '_sha256']:
                    raise RuntimeError('Selected source array changed')
                plan.append((source, target + '_' + model + '_' + kind + '.npy'))
        records.append(record)
    if len(records) != 12 or len({r['target'] for r in records}) != 12:
        raise RuntimeError('Expected twelve unique categories')
    args.output.mkdir(parents=True)
    for source, name in plan:
        shutil.copyfile(source, args.output / name)
    summary = dict(status='complete', records=records, comparison_mode='individual_terminal',
                   assembler_sha256=digest(Path(__file__)),
                   method=summaries['original']['method'],
                   rendering='Each model uses its own successful terminal RGB and reconstructed or actually saved recurrent state, as specified by the source analysis and per-record provenance. Original validation pose replays and fixed-seed autonomous validation samples are both explicitly identified.',
                   normalization=summaries['original']['normalization'], weights_updated=False,
                   selection=selected, selection_sha256=digest(args.selection),
                   limits='Illustrative examples selected partly for target visibility and target-aligned genuine End-action Grad-CAM; not a localization benchmark or a causal proof of navigation advantage.')
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (args.output / 'selection.json').write_bytes(args.selection.read_bytes())
    (args.output / 'exit.json').write_text('{"return_code": 0}\n')
    print('Assembled 48 unchanged source arrays for twelve categories:', args.output)


if __name__ == '__main__':
    main()
