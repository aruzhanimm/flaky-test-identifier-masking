"""Fixed, post-hoc representation/model comparison; preserve all prior runs."""
import argparse
from collections import Counter
import datetime
import hashlib
import json
from pathlib import Path
import platform
import shutil
import time
import traceback
import warnings

import numpy as np
import scipy
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
LABELS = ['ID', 'NIO', 'OD', 'TD']
PROTOCOL = 'protocols/models.md'
CORPUS_HASH = '4fbee74d76717535e80f7c40841852784a23c1907954e8227ad2ae31f87c5127'


def scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'Unsupported JSON value: {type(value).__name__}')


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=scalar), encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def metrics_from_counts(truth, prediction, labels):
    """Independent of sklearn; use explicit integer confusion counts."""
    if len(truth) != len(prediction) or not truth:
        raise ValueError('Expected nonempty aligned truth and predictions')
    per_class = {}
    for label in labels:
        tp = sum(y == label and p == label for y, p in zip(truth, prediction))
        fp = sum(y != label and p == label for y, p in zip(truth, prediction))
        fn = sum(y == label and p != label for y, p in zip(truth, prediction))
        per_class[label] = dict(tp=tp, fp=fp, fn=fn, support=tp+fn,
            precision=tp/(tp+fp) if tp+fp else 0.0,
            recall=tp/(tp+fn) if tp+fn else 0.0,
            f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0)
    return dict(macro_f1=sum(r['f1'] for r in per_class.values())/len(labels),
                accuracy=sum(y == p for y, p in zip(truth, prediction))/len(truth),
                per_class=per_class,
                confusion_matrix=[[sum(y == a and p == b for y, p in zip(truth, prediction))
                                   for b in labels] for a in labels])


def pairs():
    return [(rep, learner) for rep in ('word', 'char') for learner in ('lr', 'svc', 'cnb')] + [
        ('word', 'lr_c0p1'), ('word', 'lr_c10')]


def model(learner, config):
    if learner.startswith('lr'):
        settings = dict(config['classifier'])
        settings['C'] = {'lr':1.0, 'lr_c0p1':0.1, 'lr_c10':10.0}[learner]
        return LogisticRegression(**settings)
    if learner == 'svc':
        return LinearSVC(C=1, class_weight='balanced', loss='squared_hinge', dual='auto',
                         tol=0.0001, max_iter=10000, random_state=config['seed'])
    if learner == 'cnb':
        return ComplementNB(alpha=1.0, force_alpha=True, fit_prior=True, norm=False)
    raise ValueError(learner)


def settings(config, representation):
    if representation == 'word':
        result = dict(config['vectorizer'])
        result['ngram_range'] = tuple(result['ngram_range'])
        return result
    if representation == 'char':
        return dict(analyzer='char', ngram_range=(3, 5), lowercase=True, min_df=1,
                    max_features=30000, sublinear_tf=True, norm='l2')
    raise ValueError(representation)


def checked_score(truth, pred, labels):
    independent = metrics_from_counts(truth, pred, labels)
    expected = float(f1_score(truth, pred, labels=labels, average='macro', zero_division=0))
    assert abs(expected-independent['macro_f1']) < 1e-12
    assert abs(float(accuracy_score(truth, pred))-independent['accuracy']) < 1e-12
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',message='A single label was found',category=UserWarning)
        assert confusion_matrix(truth, pred, labels=labels).tolist() == independent['confusion_matrix']
    return independent


def run(args):
    dest = ROOT/'results'/args.name
    if dest.exists() and not args.resume:
        raise ValueError('Existing result directory; choose a new run name')
    if not dest.resolve().is_relative_to((ROOT/'results').resolve()):
        raise ValueError('Result directory must be under results')
    corpus_path = ROOT/'data/processed/corpus.jsonl'
    assert sha(corpus_path) == CORPUS_HASH
    config = load(ROOT/'config_model_v3.json')
    assert config['primary_labels'] == LABELS
    assert (ROOT/PROTOCOL).is_file()
    rows = sorted([json.loads(s) for s in corpus_path.read_text(encoding='utf-8').splitlines()], key=lambda r:r['sample_id'])
    assert len(rows) == len({r['sample_id'] for r in rows}) == len({r['canonical_test_id'] for r in rows}) == 1051
    y = np.array([r['label'] for r in rows]); groups = np.array([r['project_group'] for r in rows])
    assert len(set(groups)) == 127
    saved_splits = {str(s['held_out_group']):s for s in load(ROOT/'results/main_v3/splits.json')}
    old_main = {r['sample_id']:r for r in (json.loads(s) for s in (ROOT/'results/main_v3/predictions.jsonl').read_text().splitlines())}
    old_extension = {r['sample_id']:r for r in (json.loads(s) for s in (ROOT/'results/extension_v4/predictions.jsonl').read_text().splitlines())}
    old_main_fits = {(f['fold'],f['condition']):f for f in load(ROOT/'results/main_v3/fits.json')}
    old_extension_fits = {(f['fold'],f['condition']):f for f in load(ROOT/'results/extension_v4/fits.json')}
    conditions = [f'{rep}_{learner}_{version}' for rep,learner in pairs() for version in ('original','masked')]
    started = time.perf_counter()
    provenance = dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        post_hoc=True, protocol_sha256=sha(ROOT/PROTOCOL),corpus_sha256=sha(corpus_path),
        original_config_sha256=sha(ROOT/'config_model_v3.json'),splits_sha256=sha(ROOT/'results/main_v3/splits.json'),
        code_sha256={f:sha(ROOT/'scripts'/f) for f in ['run_extension_v5_models.py','test_extension_v5_models.py']},
        environment=dict(python=platform.python_version(),platform=platform.platform(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,computation_threads=1))
    run_config=dict(original=config,word=settings(config,'word'),char=settings(config,'char'),
         learners={name:model(name,config).get_params() for name in ('lr','svc','cnb','lr_c0p1','lr_c10')},
         conditions=conditions,bootstrap_resamples=5000,bootstrap_seed=20261013)
    canonical_config=json.loads(json.dumps(run_config,default=scalar))
    if dest.exists():
        previous=load(dest/'provenance.json')
        for key in ('protocol_sha256','corpus_sha256','original_config_sha256','splits_sha256','code_sha256','environment'):
            assert previous[key]==provenance[key],f'Resume mismatch: {key}'
        assert load(dest/'config.json')==canonical_config
        assert not (dest/'metrics.json').exists(),'Completed result cannot be overwritten'
        provenance=previous
    else:
        dest.mkdir(parents=True);(dest/'folds').mkdir()
        if args.import_checkpoints:
            source=(ROOT/'results'/args.import_checkpoints).resolve()
            assert source.is_relative_to((ROOT/'results').resolve()) and source!=dest.resolve()
            assert load(source/'config.json')==canonical_config
            previous=load(source/'provenance.json')
            for key in ('corpus_sha256','original_config_sha256','splits_sha256','environment'):
                assert previous[key]==provenance[key],f'Checkpoint import mismatch: {key}'
            imported={}
            for checkpoint in sorted((source/'folds').glob('*.json')):
                shutil.copy2(checkpoint,dest/'folds'/checkpoint.name);imported[checkpoint.name]=sha(checkpoint)
            provenance['imported_checkpoints']=dict(source=args.import_checkpoints,source_provenance_sha256=sha(source/'provenance.json'),files=imported,
                reason='Operational recovery after shell-process interruption; identical scientific configuration; source checkpoints retained byte-for-byte.')
        dump(dest/'provenance.json',provenance);dump(dest/'config.json',run_config)
    prediction_rows=[];project_rows=[];fits=[];vectors=[];splits=[]
    reproduction_mismatches=[];fitted_state_mismatches=[]
    computed_this_batch=0;fold_seconds=0.0
    try:
        with threadpool_limits(limits=1), warnings.catch_warnings():
            warnings.simplefilter('error',ConvergenceWarning)
            for fold,(train,test) in enumerate(LeaveOneGroupOut().split(np.zeros(len(rows)),y,groups)):
                fold_started=time.perf_counter();group=str(groups[test[0]])
                assert set(groups[train]).isdisjoint(set(groups[test])) and len(set(groups[test])) == 1
                assert set(y[train]) == set(LABELS)
                split=dict(fold=fold,held_out_group=group,
                    train_sample_ids=[rows[i]['sample_id'] for i in train],test_sample_ids=[rows[i]['sample_id'] for i in test])
                assert split['train_sample_ids'] == saved_splits[group]['train_sample_ids']
                assert split['test_sample_ids'] == saved_splits[group]['test_sample_ids']
                splits.append(split)
                checkpoint_path=dest/'folds'/f'{fold:03d}.json'
                if checkpoint_path.exists():
                    checkpoint=load(checkpoint_path)
                    assert checkpoint['project']['fold']==fold and str(checkpoint['project']['project_group'])==group
                    assert [p['sample_id'] for p in checkpoint['predictions']]==split['test_sample_ids']
                    assert len(checkpoint['fits'])==16 and len(checkpoint['vectorizers'])==4
                    assert all(set(p['predictions'])==set(conditions) for p in checkpoint['predictions'])
                    project_rows.append(checkpoint['project']);prediction_rows.extend(checkpoint['predictions'])
                    fits.extend(checkpoint['fits']);vectors.extend(checkpoint['vectorizers']);fold_seconds+=checkpoint['seconds']
                    continue
                if args.batch_size and computed_this_batch>=args.batch_size:
                    print(f'BATCH_COMPLETE {len(project_rows)}/127 folds; checkpoint hashes preserved',flush=True)
                    return
                predictions={};fold_fits=[];fold_vectors=[]
                for representation in ('word','char'):
                    learners=('lr','svc','cnb','lr_c0p1','lr_c10') if representation=='word' else ('lr','svc','cnb')
                    for version in ('original','masked'):
                        vs=time.perf_counter();vectorizer=TfidfVectorizer(**settings(config,representation))
                        key=version+'_text'
                        xtrain=vectorizer.fit_transform([rows[i][key] for i in train])
                        xtest=vectorizer.transform([rows[i][key] for i in test])
                        vh=hashlib.sha256(json.dumps({k:int(v) for k,v in vectorizer.vocabulary_.items()},sort_keys=True).encode()).hexdigest()
                        vr=dict(fold=fold,representation=representation,version=version,training_documents=len(train),evaluation_documents=len(test),
                            features=len(vectorizer.vocabulary_),vocabulary_sha256=vh,idf_sha256=hashlib.sha256(vectorizer.idf_.tobytes()).hexdigest(),
                            seconds=time.perf_counter()-vs)
                        vectors.append(vr);fold_vectors.append(vr)
                        for learner in learners:
                            fs=time.perf_counter();classifier=model(learner,config);classifier.fit(xtrain,y[train]);pred=classifier.predict(xtest)
                            name=f'{representation}_{learner}_{version}';predictions[name]=pred
                            fitted=classifier.feature_log_prob_ if learner=='cnb' else classifier.coef_
                            fh=hashlib.sha256(fitted.tobytes()).hexdigest()
                            fr=dict(fold=fold,condition=name,features=xtrain.shape[1],model_sha256=fh,
                                model_array='feature_log_prob_' if learner=='cnb' else 'coef_',
                                vocabulary_sha256=vh,classes=classifier.classes_.tolist(),seconds=time.perf_counter()-fs)
                            if hasattr(classifier,'n_iter_'):fr['iterations']=np.atleast_1d(classifier.n_iter_).tolist()
                            fits.append(fr);fold_fits.append(fr)
                            reference=None
                            if representation=='word' and learner=='lr':reference=old_main_fits[(fold,version)]
                            if representation=='word' and learner=='svc':reference=old_extension_fits[(fold,'svc_original' if version=='original' else 'svc_full')]
                            if reference is not None:
                                if reference['vocabulary_sha256']!=vh or reference['coef_sha256']!=fh:
                                    fitted_state_mismatches.append(dict(fold=fold,condition=name))
                supported=sorted(set(y[test]));scores={};fixed_scores={}
                for name,pred in predictions.items():
                    scores[name]=checked_score(y[test].tolist(),pred.tolist(),supported)['macro_f1']
                    fixed_scores[name]=checked_score(y[test].tolist(),pred.tolist(),LABELS)['macro_f1']
                project=dict(fold=fold,project_group=group,repository=rows[test[0]]['canonical_repository'],support=len(test),
                    category_counts=dict(Counter(y[test])),supported_labels=supported,scores=scores,fixed_four_label_scores=fixed_scores,
                    delta={f'{rep}_{learner}':scores[f'{rep}_{learner}_masked']-scores[f'{rep}_{learner}_original'] for rep,learner in pairs()})
                project_rows.append(project);fold_predictions=[]
                for j,i in enumerate(test):
                    r=rows[i];entry=dict(sample_id=r['sample_id'],canonical_test_id=r['canonical_test_id'],project_group=group,
                        repository=r['canonical_repository'],label=r['label'],format=r['format'],fold=fold,
                        predictions={name:str(pred[j]) for name,pred in predictions.items()})
                    for actual,reference in [('word_lr_original',old_main[r['sample_id']]['predictions']['original']),
                        ('word_lr_masked',old_main[r['sample_id']]['predictions']['masked']),
                        ('word_svc_original',old_extension[r['sample_id']]['predictions']['svc_original']),
                        ('word_svc_masked',old_extension[r['sample_id']]['predictions']['svc_full'])]:
                        if entry['predictions'][actual] != reference:reproduction_mismatches.append(dict(sample_id=r['sample_id'],condition=actual))
                    prediction_rows.append(entry);fold_predictions.append(entry)
                dump(dest/'folds'/f'{fold:03d}.json',dict(project=project,predictions=fold_predictions,fits=fold_fits,vectorizers=fold_vectors,seconds=time.perf_counter()-fold_started))
                fold_seconds+=time.perf_counter()-fold_started;computed_this_batch+=1
                print(f'Completed fold {fold+1}/127; {rows[test[0]]["canonical_repository"]}; elapsed {time.perf_counter()-started:.1f}s',flush=True)
        assert len(prediction_rows)==len({r['sample_id'] for r in prediction_rows})==1051
        assert len(fits)==2032 and len(vectors)==508
        # Recheck reproduction over every checkpoint, including imported/resumed folds.
        reproduction_mismatches=[];fitted_state_mismatches=[]
        for entry in prediction_rows:
            sid=entry['sample_id']
            for actual,reference in [('word_lr_original',old_main[sid]['predictions']['original']),('word_lr_masked',old_main[sid]['predictions']['masked']),
                ('word_svc_original',old_extension[sid]['predictions']['svc_original']),('word_svc_masked',old_extension[sid]['predictions']['svc_full'])]:
                if entry['predictions'][actual]!=reference:reproduction_mismatches.append(dict(sample_id=sid,condition=actual))
        for fitted in fits:
            name=fitted['condition'];reference=None
            if name in ('word_lr_original','word_lr_masked'):reference=old_main_fits[(fitted['fold'],name.rsplit('_',1)[1])]
            if name in ('word_svc_original','word_svc_masked'):reference=old_extension_fits[(fitted['fold'],'svc_original' if name.endswith('_original') else 'svc_full')]
            if reference is not None and (reference['vocabulary_sha256']!=fitted['vocabulary_sha256'] or reference['coef_sha256']!=fitted['model_sha256']):
                fitted_state_mismatches.append(dict(fold=fitted['fold'],condition=name))
        assert not reproduction_mismatches, reproduction_mismatches[:10]
        assert not fitted_state_mismatches, fitted_state_mismatches[:10]
        prediction_rows.sort(key=lambda r:r['sample_id'])
        predfile=dest/'predictions.jsonl';predfile.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in prediction_rows),encoding='utf-8')
        truth=[r['label'] for r in prediction_rows];metrics={}
        for condition in conditions:
            pure=checked_score(truth,[r['predictions'][condition] for r in prediction_rows],LABELS)
            metrics[condition]=dict(pooled_macro_f1=pure['macro_f1'],accuracy=pure['accuracy'],per_class=pure['per_class'],
                confusion_matrix=pure['confusion_matrix'],
                mean_project_macro_f1=float(np.mean([r['scores'][condition] for r in project_rows])),
                mean_project_fixed_four_label_f1=float(np.mean([r['fixed_four_label_scores'][condition] for r in project_rows])))
        indices=np.random.default_rng(20261013).integers(0,127,size=(5000,127));comparison={}
        for rep,learner in pairs():
            name=f'{rep}_{learner}';a=name+'_original';b=name+'_masked';delta=np.array([p['delta'][name] for p in project_rows])
            changed=[r for r in prediction_rows if r['predictions'][a]!=r['predictions'][b]]
            comparison[name]=dict(pooled_original=metrics[a]['pooled_macro_f1'],pooled_masked=metrics[b]['pooled_macro_f1'],
                pooled_delta=metrics[b]['pooled_macro_f1']-metrics[a]['pooled_macro_f1'],project_delta_mean=float(delta.mean()),
                project_delta_median=float(np.median(delta)),conditional_project_bootstrap_95=np.quantile(delta[indices].mean(axis=1),[.025,.975]).tolist(),
                projects_improved=int(np.sum(delta>0)),projects_worsened=int(np.sum(delta<0)),projects_unchanged=int(np.sum(delta==0)),
                changed_predictions=len(changed),changed_wrong_to_right=sum(r['predictions'][a]!=r['label'] and r['predictions'][b]==r['label'] for r in changed),
                changed_right_to_wrong=sum(r['predictions'][a]==r['label'] and r['predictions'][b]!=r['label'] for r in changed))
        summary=dict(tests=1051,groups=127,labels=LABELS,conditions=conditions,metrics=metrics,comparisons=comparison,
            seconds=fold_seconds,seconds_scope='Sum of successful fold computation times, excluding process startup/recovery and final aggregation',final_batch_wall_seconds=time.perf_counter()-started,vectorizer_fits=len(vectors),classifier_fits=len(fits),
            convergence_warnings=0,reproduction_prediction_mismatches=reproduction_mismatches,reproduction_fitted_state_mismatches=fitted_state_mismatches,
            independent_count_metrics_match_sklearn=True,saved_split_memberships_identical=True,protocol_sha256=sha(ROOT/PROTOCOL),
            corpus_sha256=sha(corpus_path),predictions_sha256=sha(predfile),completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            inference_caution='Complete fixed post-hoc grid; no test-based tuning. Conditional bootstrap intervals do not make population or multiplicity-adjusted confirmatory claims.')
        dump(dest/'metrics.json',summary);dump(dest/'project_scores.json',project_rows);dump(dest/'fits.json',fits);dump(dest/'vectorizers.json',vectors);dump(dest/'splits.json',splits)
        print(json.dumps(dict(seconds=summary['seconds'],comparisons=comparison),indent=2),flush=True)
        print('FULL_RUN_COMPLETE',flush=True)
    except Exception:
        dump(dest/'failure.json',dict(elapsed_seconds=time.perf_counter()-started,traceback=traceback.format_exc(),
            completed_folds=len(project_rows),reproduction_prediction_mismatches=reproduction_mismatches,
            reproduction_fitted_state_mismatches=fitted_state_mismatches))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--name',default='extension_v5_models_2026-11-19')
    parser.add_argument('--batch-size',type=int,default=0);parser.add_argument('--resume',action='store_true');parser.add_argument('--import-checkpoints')
    run(parser.parse_args())
