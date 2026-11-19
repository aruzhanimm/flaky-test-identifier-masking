"""Read-only, standard-library checks of the released corpus and predictions.

Scores are reconstructed from integer confusion counts. Bootstrap intervals
and fitted models are not retrained by this validator.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS_SHA256 = '4fbee74d76717535e80f7c40841852784a23c1907954e8227ad2ae31f87c5127'
FULL_LABELS = ['ID', 'NIO', 'OD', 'TD']
RUNS = ['main_v3', 'main_v3_repeat', 'main_v3_three_labels', 'extension_v4',
        'extension_v5_models_2026-11-19_batched', 'extension_v5_mechanism_2026-11-19']


def require(value, message):
    if not value:
        raise ValueError(message)


def close(actual, expected, context):
    require(math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12),
            f'{context}: reconstructed {actual}, recorded {expected}')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def lines(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(records, condition, labels):
    truth = [row['label'] for row in records]
    predicted = [row['predictions'][condition] for row in records]
    counts = Counter(zip(truth, predicted))
    actual = Counter(truth)
    guesses = Counter(predicted)
    classes = {}
    for label in labels:
        tp = counts[label, label]
        fp, fn = guesses[label] - tp, actual[label] - tp
        classes[label] = {
            'tp': tp, 'fp': fp, 'fn': fn, 'support': actual[label],
            'precision': tp / (tp + fp) if tp + fp else 0,
            'recall': tp / (tp + fn) if tp + fn else 0,
            'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0,
        }
    return {
        'pooled_macro_f1': sum(v['f1'] for v in classes.values()) / len(labels),
        'accuracy': sum(t == p for t, p in zip(truth, predicted)) / len(records),
        'per_class': classes,
        'confusion_matrix': [[counts[a, b] for b in labels] for a in labels],
    }


def validate_scores(records, conditions, labels, expected, context):
    for condition in conditions:
        got, saved = score(records, condition, labels), expected[condition]
        for key in ('pooled_macro_f1', 'accuracy'):
            if key in saved:
                close(got[key], saved[key], f'{context}/{condition}/{key}')
        if 'confusion_matrix' in saved:
            require(got['confusion_matrix'] == saved['confusion_matrix'], f'{context}/{condition}/matrix')
        per_class = saved.get('per_class', saved.get('classification_report', {}))
        for label in labels:
            for key, value in per_class.get(label, {}).items():
                name = 'f1' if key == 'f1-score' else key
                if name in got['per_class'][label]:
                    close(got['per_class'][label][name], value, f'{context}/{condition}/{label}/{key}')


def validate_splits(records, splits, labels, context):
    by_id = {r['sample_id']: r for r in records}
    require(len(by_id) == len(records), context + ': duplicate sample IDs')
    all_ids, evaluated, held_groups = set(by_id), [], []
    canonical = {}
    for split in splits:
        train, test = split['train_sample_ids'], split['test_sample_ids']
        require(len(train) == len(set(train)) and len(test) == len(set(test)), context + ': repeated split ID')
        require(not set(train) & set(test) and set(train) | set(test) == all_ids, context + ': invalid partition')
        group = split['held_out_group']
        require({by_id[i]['project_group'] for i in test} == {group}, context + ': mixed held-out groups')
        require(group not in {by_id[i]['project_group'] for i in train}, context + ': project leakage')
        require({by_id[i]['label'] for i in train} == set(labels), context + ': missing training category')
        evaluated.extend(test)
        held_groups.append(group)
        canonical[group] = (frozenset(train), frozenset(test))
    require(len(held_groups) == len(set(held_groups)), context + ': repeated held-out group')
    require(Counter(evaluated) == Counter(by_id.keys()), context + ': not exactly one prediction per test')
    return canonical


def validate_run(root, name, corpus):
    directory = root / 'results' / name
    metrics = read(directory / 'metrics.json')
    records = lines(directory / 'predictions.jsonl')
    labels, conditions = metrics['labels'], list(metrics['metrics'])
    by_id = {r['sample_id']: r for r in records}
    expected_ids = {sid for sid, row in corpus.items() if row['label'] in labels}
    require(set(by_id) == expected_ids and len(records) == len(by_id) == metrics['tests'], name + ': coverage')
    require(all(set(r['predictions']) == set(conditions) for r in records), name + ': condition coverage')
    groups = defaultdict(list)
    for row in records:
        original = corpus[row['sample_id']]
        require(row['label'] == original['label'] and row['project_group'] == original['project_group'], name + ': source identity')
        require(set(row['predictions'].values()) <= set(labels), name + ': unknown prediction label')
        groups[row['project_group']].append(row)
    require(len(groups) == metrics.get('groups', metrics.get('project_groups')), name + ': group count')
    splits = validate_splits(records, read(directory / 'splits.json'), labels, name)
    validate_scores(records, conditions, labels, metrics['metrics'], name)
    project_rows = read(directory / 'project_scores.json')
    require(len(project_rows) == len(groups), name + ': project-score count')
    require({p['project_group'] for p in project_rows} == set(groups), name + ': project-score identities')
    means, fixed_means = defaultdict(list), defaultdict(list)
    for project in project_rows:
        group_records = groups[project['project_group']]
        require(project['support'] == len(group_records), name + ': project support')
        supported = sorted({r['label'] for r in group_records})
        for condition in conditions:
            present = score(group_records, condition, supported)['pooled_macro_f1']
            close(present, project['scores'][condition], name + ': present-label project F1')
            means[condition].append(present)
            if 'fixed_four_label_scores' in project:
                fixed = score(group_records, condition, labels)['pooled_macro_f1']
                close(fixed, project['fixed_four_label_scores'][condition], name + ': fixed-label project F1')
                fixed_means[condition].append(fixed)
    for condition in conditions:
        close(sum(means[condition]) / len(groups), metrics['metrics'][condition]['mean_project_macro_f1'], name + ': mean project F1')
        if fixed_means[condition]:
            close(sum(fixed_means[condition]) / len(groups), metrics['metrics'][condition]['mean_project_fixed_four_label_f1'], name + ': fixed-label mean')
    if 'paired_delta_mean' in metrics:
        close(sum(m-o for m, o in zip(means['masked'], means['original'])) / len(groups), metrics['paired_delta_mean'], name + ': paired mean')
    for comparison, saved in metrics.get('comparisons', {}).items():
        original, masked = comparison + '_original', comparison + '_masked'
        before, after = (metrics['metrics'][c]['pooled_macro_f1'] for c in (original, masked))
        close(before, saved['pooled_original'], name + ': comparison original')
        close(after, saved['pooled_masked'], name + ': comparison masked')
        close(after-before, saved['pooled_delta'], name + ': pooled delta')
        deltas = [m-o for m, o in zip(means[masked], means[original])]
        close(sum(deltas)/len(deltas), saved['project_delta_mean'], name + ': comparison project delta')
        changed = [r for r in records if r['predictions'][original] != r['predictions'][masked]]
        require(len(changed) == saved['changed_predictions'], name + ': changed predictions')
        require(sum(r['predictions'][original] == r['label'] for r in changed) == saved['changed_right_to_wrong'], name + ': harmful changes')
        require(sum(r['predictions'][masked] == r['label'] for r in changed) == saved['changed_wrong_to_right'], name + ': helpful changes')
    if 'predictions_sha256' in metrics:
        require(sha(directory/'predictions.jsonl') == metrics['predictions_sha256'], name + ': prediction hash')
    return {'tests': len(records), 'groups': len(groups), 'conditions': len(conditions)}, by_id, splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT, help='Repository root (defaults to this script location)')
    parser.add_argument('--output', type=Path, help='Optional new JSON report; existing files are never overwritten')
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise ValueError('Choose a new output file; existing reports are preserved')
    root = args.root.resolve()
    corpus_path = root/'data/processed/corpus.jsonl'
    require(sha(corpus_path) == CORPUS_SHA256, 'Canonical corpus checksum mismatch')
    corpus_rows = lines(corpus_path)
    corpus = {r['sample_id']: r for r in corpus_rows}
    require(len(corpus_rows) == len(corpus) == 1051, 'Canonical test count')
    require(len({r['canonical_test_id'] for r in corpus_rows}) == 1051, 'Canonical test identity uniqueness')
    require(len({r['project_group'] for r in corpus_rows}) == 127, 'Canonical project count')
    require(Counter(r['label'] for r in corpus_rows) == {'ID':794, 'NIO':121, 'OD':126, 'TD':10}, 'Canonical category counts')
    for row in corpus_rows:
        for condition in ('original', 'masked'):
            digest = hashlib.sha256(row[condition+'_text'].encode('utf-8')).hexdigest()
            require(digest == row[condition+'_sha256'], 'Corpus diagnostic text checksum')
    summary, records, partitions = {}, {}, {}
    for name in RUNS:
        summary[name], records[name], partitions[name] = validate_run(root, name, corpus)
    require((root/'results/main_v3/predictions.jsonl').read_bytes() == (root/'results/main_v3_repeat/predictions.jsonl').read_bytes(), 'Exact repeat predictions')
    require(partitions['main_v3'] == partitions['main_v3_repeat'], 'Exact repeat partitions')
    fit_hashes = lambda name: [(r['fold'], r['condition'], r['vocabulary_sha256'], r['coef_sha256'])
                             for r in read(root/'results'/name/'fits.json')]
    require(fit_hashes('main_v3') == fit_hashes('main_v3_repeat'), 'Exact repeat fitted hashes')
    surefire = read(root/'results/extension_v4/surefire_analysis.json')
    subset = surefire['predictions']
    require(len(subset) == surefire['tests'] == 136 and surefire['groups'] == 14, 'Surefire population')
    require({r['sample_id'] for r in subset} == {sid for sid, r in corpus.items()
            if r['format'].startswith('surefire_') and r['label'] in ('OD', 'TD')}, 'Surefire subset identity')
    validate_splits(subset, surefire['splits'], ['OD', 'TD'], 'Surefire')
    validate_scores(subset, list(surefire['metrics']), ['OD', 'TD'], surefire['metrics'], 'Surefire')
    summary['surefire_OD_TD'] = {'tests':136, 'groups':14, 'conditions':len(surefire['metrics'])}
    base = records['main_v3']
    extension = records['extension_v4']
    model_name, mechanism_name = RUNS[-2:]
    require(summary[model_name]['conditions'] == 16 and summary[mechanism_name]['conditions'] == 8, 'Complete extension condition counts')
    for name in ('extension_v4', model_name, mechanism_name):
        require(partitions[name] == partitions['main_v3'], name + ': primary split memberships')
    for sid in corpus:
        require(extension[sid]['predictions']['original'] == base[sid]['predictions']['original'], 'Extension original reproduction')
        require(extension[sid]['predictions']['full'] == base[sid]['predictions']['masked'], 'Extension masked reproduction')
        grid, stage = records[model_name][sid]['predictions'], records[mechanism_name][sid]['predictions']
        for new, old in [('word_lr_original','original'), ('word_lr_masked','masked')]:
            require(grid[new] == base[sid]['predictions'][old], 'Grid LR reproduction')
        for new, old in [('word_svc_original','svc_original'), ('word_svc_masked','svc_full')]:
            require(grid[new] == extension[sid]['predictions'][old], 'Grid SVC reproduction')
        for new, old in [('OOO','original'), ('MMM','masked')]:
            require(stage[new] == base[sid]['predictions'][old], 'Mechanism endpoint reproduction')
        for new, old in [('OOM','original_model_masked_text'), ('MMO','masked_model_original_text')]:
            require(stage[new] == extension[sid]['predictions'][old], 'Mechanism input-swap reproduction')
    recovered_name = 'extension_v5_recovered_project_2026-11-19'
    directory = root/'results'/recovered_name
    recovered = lines(directory/'predictions.jsonl')
    source = {r['sample_id']:r for r in lines(root/'data/extension_candidates_2026-11-19/eligible_candidates.jsonl')}
    metrics, split = read(directory/'metrics.json'), read(directory/'split.json')
    require(len(source) == len(recovered) == 5 and {r['sample_id'] for r in recovered} == set(source), 'Recovered coverage')
    require(not set(source) & set(corpus), 'Recovered sample overlap')
    for key in ('canonical_test_id','project_group','original_sha256'):
        require({r[key] for r in source.values()}.isdisjoint({r[key] for r in corpus.values()}), 'Recovered overlap: '+key)
    require({r['label'] for r in recovered} == {'ID'} and len({r['project_group'] for r in recovered}) == 1, 'Recovered labels/groups')
    require(set(split['train_sample_ids']) == set(corpus) and set(split['evaluation_sample_ids']) == set(source), 'Recovered split')
    for row in recovered:
        require(all(row[k] == source[row['sample_id']][k] for k in ('label','project_group','canonical_test_id')), 'Recovered source identity')
        require(set(row['predictions']) == set(metrics['metrics']), 'Recovered condition coverage')
    for condition, saved in metrics['metrics'].items():
        correct = sum(r['label'] == r['predictions'][condition] for r in recovered)
        require(correct == saved['correct'] and saved['total'] == 5, 'Recovered correct count')
        close(correct/5, saved['accuracy'], 'Recovered accuracy')
    summary[recovered_name] = {'tests':5, 'groups':1, 'conditions':len(metrics['metrics'])}
    report = {'status':'passed', 'corpus_sha256':CORPUS_SHA256, 'runs':summary,
              'checks':['source identities', 'integer confusion-count scores', 'project score denominators',
                        'project-disjoint partitions', 'cross-run prediction reproduction', 'recovered-project separation'],
              'bootstrap_intervals':'not recomputed', 'model_training':'not performed'}
    text = json.dumps(report, indent=2)
    if args.output:
        with args.output.open('x', encoding='utf-8') as stream:
            stream.write(text+'\n')
    print(text)


if __name__ == '__main__':
    main()
