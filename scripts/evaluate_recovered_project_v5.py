"""Frozen full-training-set evaluation on qualified same-source Elide holdout."""
from pathlib import Path
import argparse,json,hashlib,sys,datetime,time,platform,collections,warnings
import numpy as np,scipy,sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
R=Path(__file__).resolve().parents[1]
D=R/'data/extension_candidates_2026-11-19'
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def rows(p):return [json.loads(x)for x in p.read_text(encoding='utf-8').splitlines()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(name,x):(O/name).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
    global O
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',required=True,help='New directory name under results; existing runs are never replaced')
    args=parser.parse_args()
    O=(R/'results'/args.name).resolve()
    if not O.is_relative_to(R/'results'):raise ValueError('Output must remain under results')
    if O.exists():raise ValueError('Choose a new run name; recorded runs are immutable')
    trainpath=R/'data/processed/corpus.jsonl';testpath=D/'eligible_candidates.jsonl'
    train=sorted(rows(trainpath),key=lambda r:r['sample_id']);test=sorted(rows(testpath),key=lambda r:r['sample_id'])
    config=load(R/'config_model_v3.json'); svcconfig=load(R/'results/extension_v4/config.json')['svc']
    assert sha(trainpath)=='4fbee74d76717535e80f7c40841852784a23c1907954e8227ad2ae31f87c5127'
    assert len(train)==1051 and len(test)==5 and len({r['sample_id']for r in test})==5
    for key in ['canonical_test_id','project_group','original_sha256']:
        assert {r[key]for r in train}.isdisjoint({r[key]for r in test}),key
    assert {r['label']for r in train}=={'ID','NIO','OD','TD'} and {r['label']for r in test}=={'ID'}
    assert len({r['project_group']for r in test})==1
    O.mkdir(parents=True)
    protocol=R/'protocols/recovered_project.md'
    provenance={'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'train_sha256':sha(trainpath),'evaluation_sha256':sha(testpath),'config_sha256':sha(R/'config_model_v3.json'),'svc_config_sha256':sha(R/'results/extension_v4/config.json'),'protocol_sha256':sha(protocol),'code_sha256':sha(Path(__file__)),'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'fit_threads':1,'evaluation_population':'five source-rescued ID tests from one previously excluded ReproFlake repository; not independent-source external validation','java_reruns':0}
    dump('provenance.json',provenance);dump('config.json',{'tfidf':config['vectorizer'],'logistic_regression':config['classifier'],'linear_svc':svcconfig})
    dump('split.json',{'train_sample_ids':[r['sample_id']for r in train],'evaluation_sample_ids':[r['sample_id']for r in test],'train_groups':sorted({r['project_group']for r in train}),'evaluation_groups':sorted({r['project_group']for r in test})})
    predictions=[{k:r[k]for k in ['sample_id','canonical_test_id','canonical_repository','project_group','configured_test','revision','label','original_sha256','masked_sha256','format']}|{'predictions':{}}for r in test]
    fits=[];start=time.perf_counter();settings=dict(config['vectorizer']);settings['ngram_range']=tuple(settings['ngram_range'])
    with threadpool_limits(limits=1),warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        for learner in ['lr','svc']:
            for condition in ['original','masked']:
                name=learner+'_'+condition;key=condition+'_text';t=time.perf_counter()
                model=LogisticRegression(**config['classifier']) if learner=='lr'else LinearSVC(**svcconfig)
                pipe=Pipeline([('tfidf',TfidfVectorizer(**settings)),('model',model)])
                pipe.fit([r[key]for r in train],[r['label']for r in train])
                pp=pipe.predict([r[key]for r in test]);vec=pipe.named_steps['tfidf'];mod=pipe.named_steps['model']
                for row,pred in zip(predictions,pp):row['predictions'][name]=str(pred)
                fits.append({'condition':name,'training_documents':len(train),'evaluation_documents':len(test),'vocabulary_size':len(vec.vocabulary_),'vocabulary_sha256':hashlib.sha256(json.dumps({k:int(v)for k,v in vec.vocabulary_.items()},sort_keys=True).encode()).hexdigest(),'coef_sha256':hashlib.sha256(mod.coef_.tobytes()).hexdigest(),'seconds':time.perf_counter()-t})
                print(name,'correct',sum(p==r['label']for p,r in zip(pp,test)),'/',len(test),flush=True)
    majority=min(collections.Counter(r['label']for r in train).items(),key=lambda x:(-x[1],x[0]))[0]
    formats=collections.defaultdict(collections.Counter)
    for r in train:formats[r['format']][r['label']]+=1
    rules={k:min(v.items(),key=lambda x:(-x[1],x[0]))[0]for k,v in formats.items()}
    for row in predictions:row['predictions'].update(majority=majority,source_format=rules.get(row['format'],majority))
    metrics={name:{'correct':sum(r['predictions'][name]==r['label']for r in predictions),'total':5,'accuracy':sum(r['predictions'][name]==r['label']for r in predictions)/5,'predicted_label_counts':dict(collections.Counter(r['predictions'][name]for r in predictions))}for name in predictions[0]['predictions']}
    changes={}
    for l in ['lr','svc']:
        changes[l]={'changed':sum(r['predictions'][l+'_original']!=r['predictions'][l+'_masked']for r in predictions),'harmful':sum(r['predictions'][l+'_original']==r['label'] and r['predictions'][l+'_masked']!=r['label']for r in predictions),'helpful':sum(r['predictions'][l+'_original']!=r['label'] and r['predictions'][l+'_masked']==r['label']for r in predictions)}
    dump('metrics.json',{'training_tests':1051,'evaluation_tests':5,'evaluation_projects':1,'evaluation_labels':{'ID':5},'metrics':metrics,'paired_changes':changes,'fit_seconds':time.perf_counter()-start,'interpretation':'One-project all-ID holdout. Accuracy cannot establish four-category discrimination; majority and format-only rules also predict ID.'})
    dump('fits.json',fits)
    (O/'predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n'for r in predictions),encoding='utf-8')
    dump('checks.json',{'source_corpus_unchanged':sha(trainpath)==provenance['train_sha256'],'candidate_file_unchanged':sha(testpath)==provenance['evaluation_sha256'],'canonical_tests_disjoint':True,'repository_groups_disjoint':True,'exact_diagnostic_hashes_disjoint':True,'fresh_training_only_pipeline_each_condition':True,'convergence_warnings':0,'fitted_models':4})
    print(json.dumps(metrics,indent=2))

if __name__=='__main__':main()
