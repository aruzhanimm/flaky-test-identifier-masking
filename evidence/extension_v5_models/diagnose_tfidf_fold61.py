"""Diagnose the recorded nonfinite TF-IDF failure without altering the experiment."""
from pathlib import Path
import sys,json,hashlib,warnings
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer,TfidfVectorizer
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from run_extension_v5_models import settings,model,sha

def digest(x):
    return hashlib.sha256(x.data.tobytes()+x.indices.tobytes()+x.indptr.tobytes()).hexdigest()

def detail(x):
    return dict(shape=list(x.shape),dtype=str(x.dtype),nnz=x.nnz,minimum=float(x.data.min()),maximum=float(x.data.max()),
                finite=bool(np.isfinite(x.data).all()),strictly_positive=bool((x.data>0).all()),sha256=digest(x))

config=json.loads((ROOT/'config_model_v3.json').read_text())
rows=sorted([json.loads(s)for s in (ROOT/'data/processed/corpus.jsonl').read_text(encoding='utf-8').splitlines()],key=lambda r:r['sample_id'])
y=np.array([r['label']for r in rows]);groups=np.array([r['project_group']for r in rows])
train,test=list(LeaveOneGroupOut().split(np.zeros(len(rows)),y,groups))[61]
checkpoint=json.loads((ROOT/'results/extension_v5_models_2026-11-19_batched/folds/061.json').read_text())
expected_fits={r['condition']:r for r in checkpoint['fits']}
expected_vectors={(r['representation'],r['version']):r for r in checkpoint['vectorizers']}
result=dict(fold=61,repository=rows[test[0]]['canonical_repository'],corpus_sha256=sha(ROOT/'data/processed/corpus.jsonl'),conditions=[])
with threadpool_limits(limits=1),warnings.catch_warnings():
    warnings.simplefilter('error',ConvergenceWarning)
    for representation in ('word','char'):
        for version in ('original','masked'):
            cfg=settings(config,representation)
            texts=[rows[i][version+'_text']for i in train];testtexts=[rows[i][version+'_text']for i in test]
            countcfg={k:v for k,v in cfg.items()if k not in ('sublinear_tf','norm')};countcfg['dtype']=np.float64
            countvec=CountVectorizer(**countcfg);counts=countvec.fit_transform(texts)
            counts_before=detail(counts)
            assert counts_before['strictly_positive'] and counts_before['finite']
            tfidf=TfidfTransformer(sublinear_tf=True,norm='l2');separate=tfidf.fit_transform(counts.copy())
            vec=TfidfVectorizer(**cfg);x=vec.fit_transform(texts);xtest=vec.transform(testtexts)
            vh=hashlib.sha256(json.dumps({k:int(v)for k,v in vec.vocabulary_.items()},sort_keys=True).encode()).hexdigest()
            ih=hashlib.sha256(vec.idf_.tobytes()).hexdigest()
            saved=expected_vectors[(representation,version)]
            record=dict(representation=representation,version=version,counts=counts_before,
                idf=dict(minimum=float(tfidf.idf_.min()),maximum=float(tfidf.idf_.max()),finite=bool(np.isfinite(tfidf.idf_).all())),
                transformed=detail(x),separate_count_then_tfidf_identical=bool((x!=separate).nnz==0),
                vocabulary_matches_checkpoint=vh==saved['vocabulary_sha256'],idf_matches_checkpoint=ih==saved['idf_sha256'],classifiers=[])
            difference=x-separate
            record['separate_transform_max_abs_difference']=float(abs(difference.data).max()) if difference.nnz else 0.0
            print(json.dumps({k:v for k,v in record.items()if k!='classifiers'}),flush=True)
            assert record['separate_transform_max_abs_difference']<1e-12 and record['vocabulary_matches_checkpoint'] and record['idf_matches_checkpoint']
            learners=('lr','svc','cnb','lr_c0p1','lr_c10')if representation=='word'else('lr','svc','cnb')
            for learner in learners:
                name=f'{representation}_{learner}_{version}';before=digest(x);testbefore=digest(xtest)
                clf=model(learner,config);clf.fit(x,y[train]);pred=clf.predict(xtest)
                fitted=clf.feature_log_prob_ if learner=='cnb'else clf.coef_
                fh=hashlib.sha256(fitted.tobytes()).hexdigest()
                entry=dict(condition=name,train_csr_unchanged=before==digest(x),test_csr_unchanged=testbefore==digest(xtest),
                    fitted_hash_matches_checkpoint=fh==expected_fits[name]['model_sha256'],
                    predictions_match_checkpoint=all(str(p)==r['predictions'][name]for p,r in zip(pred,checkpoint['predictions'])))
                assert all(v for k,v in entry.items()if k!='condition'),entry
                record['classifiers'].append(entry)
            result['conditions'].append(record)
result['all_checks_passed']=True
print(json.dumps(result,indent=2))
