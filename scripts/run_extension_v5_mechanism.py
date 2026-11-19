"""Train-only 2 x 2 x 2 diagnostic of TF-IDF fitting, LR fitting and input text."""
import argparse
import collections
import datetime
import hashlib
import itertools
import json
import platform
import time
import warnings
from pathlib import Path

import numpy as np
import scipy
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import LeaveOneGroupOut
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = tuple(''.join(v) for v in itertools.product('OM', repeat=3))
TEXT_KEYS = {'O': 'original_text', 'M': 'masked_text'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def count_metrics(truth, predictions, labels):
    """Independent confusion-count implementation, without sklearn metrics."""
    order = {c: i for i, c in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for actual, predicted in zip(truth, predictions, strict=True):
        matrix[order[actual]][order[predicted]] += 1
    class_scores = {}
    for c, i in order.items():
        tp = matrix[i][i]
        fp = sum(row[i] for row in matrix) - tp
        fn = sum(matrix[i]) - tp
        class_scores[c] = {
            'precision': tp / (tp + fp) if tp + fp else 0.,
            'recall': tp / (tp + fn) if tp + fn else 0.,
            'f1-score': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
            'support': sum(matrix[i]),
        }
    return {
        'confusion_matrix': matrix,
        'pooled_macro_f1': sum(v['f1-score'] for v in class_scores.values()) / len(labels),
        'accuracy': sum(matrix[i][i] for i in range(len(labels))) / len(truth),
        'classification_report': class_scores,
    }


def vocabulary_hash(vectorizer):
    return digest(json.dumps({k: int(v) for k, v in vectorizer.vocabulary_.items()}, sort_keys=True).encode())


def path_differences(scores):
    """All six telescoping paths; no unique causal allocation is implied."""
    stage_names = ['vectorizer_fit', 'classifier_fit', 'evaluation_input']
    results = []
    for order in itertools.permutations(range(3)):
        condition = list('OOO')
        previous = 'OOO'
        steps = []
        for position in order:
            condition[position] = 'M'
            current = ''.join(condition)
            steps.append({'stage': stage_names[position], 'from': previous, 'to': current,
                          'delta_macro_f1': scores[current] - scores[previous]})
            previous = current
        results.append({'order': [stage_names[i] for i in order], 'steps': steps,
                        'total_delta': sum(s['delta_macro_f1'] for s in steps)})
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='extension_v5_mechanism_2026-11-19')
    args = parser.parse_args()
    dest = (ROOT / 'results' / args.name).resolve()
    if not dest.is_relative_to(ROOT / 'results'):
        raise ValueError('Output must remain inside results')
    if dest.exists():
        raise ValueError('A new run name is required; existing results will not be overwritten')
    corpus_path = ROOT / 'data/processed/corpus.jsonl'
    config_path = ROOT / 'config_model_v3.json'
    protocol_path = ROOT / 'protocols/fitting_stages.md'
    config = load(config_path)
    audit = load(ROOT / 'data/processed/audit.json')
    assert audit['extraction_complete'] and audit['corpus_sha256'] == sha(corpus_path)
    rows = sorted((json.loads(line) for line in corpus_path.read_text(encoding='utf-8').splitlines()),
                  key=lambda r: r['sample_id'])
    labels = config['primary_labels']
    assert len(rows) == 1051
    assert len({r['sample_id'] for r in rows}) == len(rows)
    assert len({r['canonical_test_id'] for r in rows}) == len(rows)
    truth = np.array([r['label'] for r in rows])
    groups = np.array([r['project_group'] for r in rows])
    assert len(set(groups)) == 127
    prior = {r['sample_id']: r for r in (json.loads(line) for line in
             (ROOT / 'results/main_v3/predictions.jsonl').read_text(encoding='utf-8').splitlines())}
    prior_extension = {r['sample_id']: r for r in (json.loads(line) for line in
                       (ROOT / 'results/extension_v4/predictions.jsonl').read_text(encoding='utf-8').splitlines())}
    prior_fits = {(r['fold'], r['condition']): r for r in load(ROOT / 'results/main_v3/fits.json')}
    prior_splits = {r['fold']: r for r in load(ROOT / 'results/main_v3/splits.json')}
    settings = dict(config['vectorizer'])
    settings['ngram_range'] = tuple(settings['ngram_range'])
    dest.mkdir(parents=True)
    started_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    provenance = {'started_utc': started_utc, 'protocol_sha256': sha(protocol_path),
                  'config_sha256': sha(config_path), 'corpus_sha256': sha(corpus_path),
                  'code_sha256': sha(Path(__file__)),
                  'environment': {'python': platform.python_version(), 'platform': platform.platform(),
                                  'numpy': np.__version__, 'scipy': scipy.__version__,
                                  'sklearn': sklearn.__version__, 'computation_threads': 1},
                  'conditions': {c: dict(zip(['vectorizer_fit_text', 'classifier_fit_text', 'test_text'],
                                ['original' if x == 'O' else 'masked' for x in c])) for c in CONDITIONS},
                  'exploratory': True}
    dump(dest / 'provenance.json', provenance)
    dump(dest / 'config.json', config)
    fits, vectorizer_records, splits, projects, predictions = [], [], [], [], []
    started = time.perf_counter()
    endpoints_fits_ok = True
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        for fold, (train, test) in enumerate(LeaveOneGroupOut().split(np.zeros(len(rows)), truth, groups)):
            train_groups, test_groups = set(groups[train]), set(groups[test])
            assert train_groups.isdisjoint(test_groups) and len(test_groups) == 1
            assert set(truth[train]) == set(labels)
            train_ids = [rows[i]['sample_id'] for i in train]
            test_ids = [rows[i]['sample_id'] for i in test]
            assert train_ids == prior_splits[fold]['train_sample_ids']
            assert test_ids == prior_splits[fold]['test_sample_ids']
            splits.append({'fold': fold, 'held_out_group': str(groups[test[0]]),
                           'train_sample_ids': train_ids, 'test_sample_ids': test_ids})
            training = {c: [rows[i][TEXT_KEYS[c]] for i in train] for c in 'OM'}
            testing = {c: [rows[i][TEXT_KEYS[c]] for i in test] for c in 'OM'}
            vectorizers, caches, outputs = {}, {}, {}
            for vector_source in 'OM':
                vectorizer = TfidfVectorizer(**settings)
                fitted_train = vectorizer.fit_transform(training[vector_source])
                vectorizers[vector_source] = vectorizer
                train_matrices = {c: fitted_train if c == vector_source else vectorizer.transform(training[c])
                                  for c in 'OM'}
                test_matrices = {c: vectorizer.transform(testing[c]) for c in 'OM'}
                caches[vector_source] = (train_matrices, test_matrices)
                vhash = vocabulary_hash(vectorizer)
                base_condition = 'original' if vector_source == 'O' else 'masked'
                assert vhash == prior_fits[(fold, base_condition)]['vocabulary_sha256']
                vectorizer_records.append({'fold': fold, 'fit_text': vector_source,
                    'vocabulary_size': len(vectorizer.vocabulary_), 'vocabulary_sha256': vhash,
                    'idf_sha256': digest(vectorizer.idf_.tobytes()),
                    'training_nnz': {c: int(m.nnz) for c, m in train_matrices.items()},
                    'test_nnz': {c: int(m.nnz) for c, m in test_matrices.items()},
                    'zero_training_documents': {c: int(np.sum(m.getnnz(axis=1) == 0)) for c, m in train_matrices.items()},
                    'zero_test_documents': {c: int(np.sum(m.getnnz(axis=1) == 0)) for c, m in test_matrices.items()}})
                for classifier_source in 'OM':
                    model = LogisticRegression(**config['classifier'])
                    fit_start = time.perf_counter()
                    model.fit(train_matrices[classifier_source], truth[train])
                    coefficient_hash = digest(model.coef_.tobytes())
                    if vector_source == classifier_source:
                        same = coefficient_hash == prior_fits[(fold, base_condition)]['coef_sha256']
                        endpoints_fits_ok = endpoints_fits_ok and same
                        assert same, ('Endpoint coefficient mismatch', fold, vector_source)
                    fits.append({'fold': fold, 'vectorizer_fit_text': vector_source,
                                 'classifier_fit_text': classifier_source, 'coef_sha256': coefficient_hash,
                                 'classes': [str(c) for c in model.classes_], 'iterations': model.n_iter_.tolist(),
                                 'seconds': time.perf_counter() - fit_start})
                    for input_source in 'OM':
                        outputs[vector_source + classifier_source + input_source] = model.predict(test_matrices[input_source])
            ovec, mvec = vectorizers['O'], vectorizers['M']
            common = sorted(set(ovec.vocabulary_) & set(mvec.vocabulary_))
            union = set(ovec.vocabulary_) | set(mvec.vocabulary_)
            changes = np.array([mvec.idf_[mvec.vocabulary_[t]] - ovec.idf_[ovec.vocabulary_[t]] for t in common])
            scores = {c: float(f1_score(truth[test], pred, labels=sorted(set(truth[test])), average='macro', zero_division=0))
                      for c, pred in outputs.items()}
            projects.append({'fold': fold, 'project_group': str(groups[test[0]]),
                'repository': rows[test[0]]['canonical_repository'], 'support': len(test),
                'supported_labels': sorted(set(str(c) for c in truth[test])), 'scores': scores,
                'vocabulary_overlap': {'original': len(ovec.vocabulary_), 'masked': len(mvec.vocabulary_),
                    'common': len(common), 'union': len(union), 'jaccard': len(common) / len(union),
                    'common_idf_changed': int(np.sum(np.abs(changes) > 1e-12)),
                    'common_idf_mean_absolute_change': float(np.mean(np.abs(changes))),
                    'common_idf_max_absolute_change': float(np.max(np.abs(changes)))}})
            for local, index in enumerate(test):
                row = rows[index]
                result = {'sample_id': row['sample_id'], 'project_group': row['project_group'],
                          'repository': row['canonical_repository'], 'label': row['label'],
                          'format': row['format'], 'fold': fold,
                          'predictions': {c: str(outputs[c][local]) for c in CONDITIONS}}
                original = prior[row['sample_id']]['predictions']
                extension = prior_extension[row['sample_id']]['predictions']
                assert result['predictions']['OOO'] == original['original']
                assert result['predictions']['MMM'] == original['masked']
                assert result['predictions']['OOM'] == extension['original_model_masked_text']
                assert result['predictions']['MMO'] == extension['masked_model_original_text']
                predictions.append(result)
            if fold % 10 == 0:
                print(f'Mechanism fold {fold + 1}/127 complete; {time.perf_counter() - started:.1f}s', flush=True)
    predictions.sort(key=lambda r: r['sample_id'])
    pred_path = dest / 'predictions.jsonl'
    pred_path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in predictions), encoding='utf-8')
    # Reopen the saved artifact, then reconstruct its scores independently.
    saved = [json.loads(s) for s in pred_path.read_text(encoding='utf-8').splitlines()]
    y = [r['label'] for r in saved]
    metrics, independent = {}, {}
    for condition in CONDITIONS:
        p = [r['predictions'][condition] for r in saved]
        result = {'pooled_macro_f1': float(f1_score(y, p, labels=labels, average='macro', zero_division=0)),
                  'accuracy': float(accuracy_score(y, p)),
                  'mean_project_macro_f1': float(np.mean([g['scores'][condition] for g in projects])),
                  'classification_report': classification_report(y, p, labels=labels, output_dict=True, zero_division=0),
                  'confusion_matrix': confusion_matrix(y, p, labels=labels).tolist()}
        alternative = count_metrics(y, p, labels)
        assert abs(result['pooled_macro_f1'] - alternative['pooled_macro_f1']) < 1e-12
        assert abs(result['accuracy'] - alternative['accuracy']) < 1e-12
        assert result['confusion_matrix'] == alternative['confusion_matrix']
        for label in labels:
            for metric in ['precision', 'recall', 'f1-score', 'support']:
                assert abs(result['classification_report'][label][metric] - alternative['classification_report'][label][metric]) < 1e-12
        metrics[condition], independent[condition] = result, alternative
    paths = path_differences({c: m['pooled_macro_f1'] for c, m in metrics.items()})
    summary = {'run': args.name, 'tests': len(saved), 'groups': len(projects), 'labels': labels,
               'conditions': provenance['conditions'], 'metrics': metrics, 'paths': paths,
               'endpoint_pooled_delta': metrics['MMM']['pooled_macro_f1'] - metrics['OOO']['pooled_macro_f1'],
               'vectorizer_fits': len(vectorizer_records), 'classifier_fits': len(fits),
               'seconds': time.perf_counter() - started, 'predictions_sha256': sha(pred_path),
               'mean_vocabulary_jaccard': float(np.mean([p['vocabulary_overlap']['jaccard'] for p in projects])),
               'checks': {'project_splits_match_main': True, 'train_only_vectorizer_fits': True,
                          'all_training_folds_contain_all_labels': True, 'convergence_warnings': 0,
                          'endpoints_match_main_predictions_vocab_coefficients': endpoints_fits_ok,
                          'cross_input_predictions_match_extension_v4': True, 'independent_count_metrics_match': True},
               'interpretation': 'Stage interventions are conditional diagnostics. Vocabulary, IDF and normalisation change jointly; attribution depends on intervention order. No population-level causal or significance claim.',
               'completed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    dump(dest / 'metrics.json', summary)
    dump(dest / 'project_scores.json', projects)
    dump(dest / 'splits.json', splits)
    dump(dest / 'fits.json', fits)
    dump(dest / 'vectorizers.json', vectorizer_records)
    dump(dest / 'independent_count_metrics.json', independent)
    print(json.dumps({'scores': {c: m['pooled_macro_f1'] for c, m in metrics.items()},
                      'seconds': summary['seconds'], 'checks': summary['checks']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
