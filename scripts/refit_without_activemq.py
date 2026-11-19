"""Refit the frozen primary word-LR LOPO design after removing ActiveMQ.

Post-hoc sensitivity: remove ActiveMQ from every training and evaluation
partition; preserve the original representation and classifier parameters.
"""
import argparse
from collections import Counter
import datetime
import hashlib
import json
from pathlib import Path
import platform
import statistics
import time
import traceback
import warnings

import numpy as np
import scipy
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import LeaveOneGroupOut
from threadpoolctl import threadpool_limits

from analyze_project_sensitivity import bootstrap, changes, require, score, sha

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["ID", "NIO", "OD", "TD"]
CONDITIONS = ["original", "masked"]
CORPUS_HASH = "4fbee74d76717535e80f7c40841852784a23c1907954e8227ad2ae31f87c5127"


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/project_sensitivity_refit")
    args = parser.parse_args()
    destination = (ROOT / args.output).resolve()
    require(destination.is_relative_to(ROOT / "results"), "Output must be inside results")
    require(not destination.exists(), "Choose a new output directory")
    corpus_path, config_path = ROOT / "data/processed/corpus.jsonl", ROOT / "config_model_v3.json"
    require(sha(corpus_path) == CORPUS_HASH, "Canonical corpus checksum mismatch")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    require(config["primary_labels"] == LABELS, "Frozen label order changed")
    all_rows = sorted([json.loads(line) for line in corpus_path.read_text(encoding="utf-8").splitlines()], key=lambda r: r["sample_id"])
    excluded = [r for r in all_rows if r["project_group"] == "206387"]
    require(len(excluded) == 46 and {r["canonical_repository"] for r in excluded} == {"https://github.com/apache/activemq"}, "ActiveMQ identity mismatch")
    rows = [r for r in all_rows if r["project_group"] != "206387"]
    require(len(rows) == len({r["sample_id"] for r in rows}) == 1005, "Retained sample coverage mismatch")
    y = np.array([r["label"] for r in rows])
    groups = np.array([r["project_group"] for r in rows])
    require(len(set(groups)) == 126, "Retained project coverage mismatch")
    saved_splits = {r["held_out_group"]: r for r in json.loads((ROOT / "results/main_v3/splits.json").read_text())}
    previous = {r["sample_id"]: r for r in (json.loads(line) for line in (ROOT / "results/main_v3/predictions.jsonl").read_text().splitlines())}
    excluded_ids = {r["sample_id"] for r in excluded}
    settings = dict(config["vectorizer"])
    settings["ngram_range"] = tuple(settings["ngram_range"])
    started = time.perf_counter()
    provenance = dict(post_hoc=True, started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      purpose="ActiveMQ removed from all training and evaluation folds; no parameter tuning",
                      source_corpus_sha256=sha(corpus_path), source_config_sha256=sha(config_path),
                      source_predictions_sha256=sha(ROOT / "results/main_v3/predictions.jsonl"),
                      source_splits_sha256=sha(ROOT / "results/main_v3/splits.json"),
                      code_sha256={name: sha(ROOT / "scripts" / name) for name in ["refit_without_activemq.py", "analyze_project_sensitivity.py"]},
                      environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                                       sklearn=sklearn.__version__, computation_threads=1),
                      excluded_project="206387", excluded_tests=46)
    destination.mkdir(parents=True)
    dump(destination / "config.json", config)
    dump(destination / "provenance.json", provenance)
    predictions, projects, splits, fits = [], [], [], []
    try:
        with threadpool_limits(limits=1), warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            warnings.simplefilter("error", RuntimeWarning)
            for fold, (train, test) in enumerate(LeaveOneGroupOut().split(np.zeros(len(rows)), y, groups)):
                group = str(groups[test[0]])
                require(set(groups[train]).isdisjoint(set(groups[test])), "Project leakage")
                require(set(y[train]) == set(LABELS), "Missing training label")
                train_ids = [rows[i]["sample_id"] for i in train]
                test_ids = [rows[i]["sample_id"] for i in test]
                require(set(train_ids) == set(saved_splits[group]["train_sample_ids"]) - excluded_ids, "Training split differs beyond ActiveMQ exclusion")
                require(test_ids == saved_splits[group]["test_sample_ids"], "Evaluation split differs")
                splits.append(dict(fold=fold, held_out_group=group, train_sample_ids=train_ids, test_sample_ids=test_ids))
                fold_predictions = {}
                for condition in CONDITIONS:
                    fit_started = time.perf_counter()
                    vectorizer = TfidfVectorizer(**settings)
                    xtrain = vectorizer.fit_transform([rows[i][condition + "_text"] for i in train])
                    xtest = vectorizer.transform([rows[i][condition + "_text"] for i in test])
                    require(np.isfinite(xtrain.data).all() and np.isfinite(xtest.data).all() and np.isfinite(vectorizer.idf_).all(), "Nonfinite TF-IDF values")
                    classifier = LogisticRegression(**config["classifier"])
                    classifier.fit(xtrain, y[train])
                    require(np.isfinite(classifier.coef_).all() and np.isfinite(classifier.intercept_).all(), "Nonfinite fitted coefficients")
                    predicted = classifier.predict(xtest)
                    fold_predictions[condition] = predicted
                    fits.append(dict(fold=fold, project_group=group, condition=condition, training_documents=len(train),
                                     evaluation_documents=len(test), features=xtrain.shape[1],
                                     vocabulary_sha256=hashlib.sha256(json.dumps({k: int(v) for k, v in vectorizer.vocabulary_.items()}, sort_keys=True).encode()).hexdigest(),
                                     idf_sha256=hashlib.sha256(vectorizer.idf_.tobytes()).hexdigest(),
                                     coef_sha256=hashlib.sha256(classifier.coef_.tobytes()).hexdigest(),
                                     intercept_sha256=hashlib.sha256(classifier.intercept_.tobytes()).hexdigest(),
                                     iterations=classifier.n_iter_.tolist(), seconds=time.perf_counter() - fit_started))
                fold_rows = []
                for j, i in enumerate(test):
                    r = rows[i]
                    record = dict(sample_id=r["sample_id"], canonical_test_id=r["canonical_test_id"],
                                  project_group=group, repository=r["canonical_repository"], label=r["label"], fold=fold,
                                  original_sha256=r["original_sha256"], masked_sha256=r["masked_sha256"],
                                  predictions={c: str(fold_predictions[c][j]) for c in CONDITIONS})
                    fold_rows.append(record)
                    predictions.append(record)
                supported = sorted({r["label"] for r in fold_rows})
                metrics = {c: score(fold_rows, c) for c in CONDITIONS}
                present = {c: score(fold_rows, c, supported)["pooled_macro_f1"] for c in CONDITIONS}
                projects.append(dict(fold=fold, project_group=group, repository=rows[test[0]]["canonical_repository"],
                                     tests=len(test), metrics=metrics, mean_present_label_f1=present,
                                     delta_mean_present_label_f1=present["masked"] - present["original"], changes=changes(fold_rows)))
                if fold % 10 == 0 or fold == 125:
                    print(f"Completed {fold + 1}/126 folds; elapsed {time.perf_counter() - started:.1f}s", flush=True)
        require(len(predictions) == len({r["sample_id"] for r in predictions}) == 1005 and len(fits) == 252, "Incomplete run")
        predictions.sort(key=lambda r: r["sample_id"])
        metrics = {c: score(predictions, c) for c in CONDITIONS}
        for c in CONDITIONS:
            metrics[c]["mean_present_label_project_f1"] = statistics.fmean(p["mean_present_label_f1"][c] for p in projects)
            direct = float(f1_score([r["label"] for r in predictions], [r["predictions"][c] for r in predictions], labels=LABELS, average="macro", zero_division=0))
            require(abs(direct - metrics[c]["pooled_macro_f1"]) < 1e-12, "Independent count metric disagrees with sklearn")
        delta = dict(pooled_macro_f1=metrics["masked"]["pooled_macro_f1"] - metrics["original"]["pooled_macro_f1"],
                     mean_present_label_project_f1=metrics["masked"]["mean_present_label_project_f1"] - metrics["original"]["mean_present_label_project_f1"],
                     deteriorated_projects=sum(p["delta_mean_present_label_f1"] < 0 for p in projects),
                     improved_projects=sum(p["delta_mean_present_label_f1"] > 0 for p in projects),
                     unchanged_projects=sum(p["delta_mean_present_label_f1"] == 0 for p in projects))
        prediction_path = destination / "predictions.jsonl"
        prediction_path.write_text("".join(json.dumps(r) + "\n" for r in predictions), encoding="utf-8")
        summary = dict(status="completed", post_hoc=True, tests=1005, projects=126, labels=LABELS,
                       class_support=dict(Counter(r["label"] for r in rows)),
                       category_project_groups={label: len({r["project_group"] for r in rows if r["label"] == label}) for label in LABELS},
                       exclusion_scope="ActiveMQ removed from every training and evaluation partition before refitting",
                       metrics=metrics, paired_delta=delta, changes=changes(predictions),
                       decisions_different_from_evaluation_only_exclusion={c: sum(r["predictions"][c] != previous[r["sample_id"]]["predictions"][c] for r in predictions) for c in CONDITIONS},
                       conditional_project_bootstrap=bootstrap(projects, config["bootstrap_resamples"], config["seed"]),
                       classifier_fits=len(fits), convergence_warnings=0, nonfinite_values=0,
                       independent_count_metrics_match_sklearn=True,
                       source_parameters_unchanged=True, retained_test_memberships_identical=True,
                       predictions_sha256=sha(prediction_path), seconds=time.perf_counter() - started,
                       completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        dump(destination / "metrics.json", summary)
        dump(destination / "project_scores.json", projects)
        dump(destination / "splits.json", splits)
        dump(destination / "fits.json", fits)
        (destination / "README.md").write_text("# ActiveMQ-excluded LOPO refit\n\nThis post-hoc sensitivity removes 46 ActiveMQ tests before fitting and evaluates 1005 tests from the remaining 126 projects. Original and masked word-TF-IDF logistic-regression pipelines use the unchanged primary configuration. Each vocabulary, IDF and model is fitted only on the other 125 retained projects. No model or parameter selection is performed.\n\nReproduce with the recorded model environment and a new output directory:\n\n```sh\npython scripts/refit_without_activemq.py --output results/project_sensitivity_refit_replication\n```\n\n`metrics.json` records scores, integer class counts, paired changes and descriptive conditional bootstrap intervals. `splits.json` contains all memberships; `fits.json` contains 252 fit records and fitted-state hashes. Nonfinite matrices/coefficients and convergence/runtime warnings stop the run rather than being repaired or ignored. An interrupted run records `failure.json` and must not be treated as completed.\n", encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
    except Exception:
        dump(destination / "failure.json", dict(status="failed", completed_folds=len(projects), seconds=time.perf_counter() - started, traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
