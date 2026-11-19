"""Rescore frozen paired LOPO predictions under project exclusions (stdlib only).

This post-hoc sensitivity analysis changes the evaluation population only.
No vocabulary, IDF or classifier is fitted. In particular, ActiveMQ remains
in the training sets that produced predictions for other original LOPO folds.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import statistics

ROOT = Path(__file__).resolve().parents[1]
LABELS = ("ID", "NIO", "OD", "TD")
CONDITIONS = ("original", "masked")
ACTIVEMQ = "https://github.com/apache/activemq"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def score(rows, condition, labels=LABELS):
    counts = Counter((row["label"], row["predictions"][condition]) for row in rows)
    support = Counter(row["label"] for row in rows)
    predicted = Counter(row["predictions"][condition] for row in rows)
    per_class = {}
    for label in labels:
        tp = counts[label, label]
        fp, fn = predicted[label] - tp, support[label] - tp
        per_class[label] = dict(support=support[label], predicted=predicted[label],
                                tp=tp, fp=fp, fn=fn,
                                precision=tp / predicted[label] if predicted[label] else 0.0,
                                recall=tp / support[label] if support[label] else 0.0,
                                f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    return dict(pooled_macro_f1=statistics.fmean(c["f1"] for c in per_class.values()),
                accuracy=sum(counts[label, label] for label in LABELS) / len(rows),
                per_class=per_class,
                confusion_matrix=[[counts[a, b] for b in LABELS] for a in LABELS])


def changes(rows):
    changed = [r for r in rows if r["predictions"]["original"] != r["predictions"]["masked"]]
    harms = sum(r["predictions"]["original"] == r["label"] for r in changed)
    gains = sum(r["predictions"]["masked"] == r["label"] for r in changed)
    return dict(changed_predictions=len(changed), right_to_wrong=harms,
                wrong_to_right=gains, wrong_to_different_wrong=len(changed) - harms - gains)


def quantile(values, probability):
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def matrix_f1(matrix):
    return statistics.fmean(
        2 * matrix[i * 4 + i] / (sum(matrix[i * 4:i * 4 + 4]) + sum(matrix[i::4]))
        if sum(matrix[i * 4:i * 4 + 4]) + sum(matrix[i::4]) else 0.0
        for i in range(4))


def bootstrap(projects, repetitions, seed):
    """Paired project resampling; both conditions share each sampled cluster."""
    rng = random.Random(seed)
    n = len(projects)
    mean_deltas, pooled_deltas = [], []
    absent = Counter()
    matrices = [[sum(p["metrics"][c]["confusion_matrix"], []) for c in CONDITIONS] for p in projects]
    deltas = [p["delta_mean_present_label_f1"] for p in projects]
    for _ in range(repetitions):
        weights = Counter(rng.choices(range(n), k=n))
        mean_deltas.append(sum(deltas[i] * weight for i, weight in weights.items()) / n)
        totals = [[0] * 16, [0] * 16]
        for i, weight in weights.items():
            for condition in range(2):
                for cell, value in enumerate(matrices[i][condition]):
                    totals[condition][cell] += value * weight
        pooled_deltas.append(matrix_f1(totals[1]) - matrix_f1(totals[0]))
        for i, label in enumerate(LABELS):
            absent[label] += not sum(totals[0][i * 4:i * 4 + 4])
    return dict(repetitions=repetitions, seed=seed, rng="Python random.Random",
                sampling_unit="project_group; paired, with replacement; original group count draws",
                interval="95% percentile; linearly interpolated quantiles",
                delta_mean_present_label_project_f1_95=[quantile(mean_deltas, p) for p in (0.025, 0.975)],
                delta_pooled_fixed_four_label_macro_f1_95=[quantile(pooled_deltas, p) for p in (0.025, 0.975)],
                resamples_without_label={label: absent[label] for label in LABELS},
                caution="Descriptive conditional intervals from frozen overlapping LOPO fits, not independent retraining or population inference. The four pooled labels remain fixed even when a bootstrap sample lacks a label. No hypothesis test or causal claim.")


def write_csv(path, records):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", default="results/project_sensitivity")
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261119)
    args = parser.parse_args()
    root = args.root.resolve()
    destination = (root / args.output).resolve()
    require(destination.is_relative_to(root), "Output must remain inside the repository")
    require(not destination.exists(), "Choose a fresh output directory; previous results are preserved")
    require(args.bootstrap_resamples >= 1000, "Use at least 1000 bootstrap resamples")
    source = root / "results/main_v3"
    prediction_path = source / "predictions.jsonl"
    rows = lines(prediction_path)
    published = read(source / "metrics.json")
    require(sha(prediction_path) == published["predictions_sha256"], "Saved prediction hash mismatch")
    require(len(rows) == len({r["sample_id"] for r in rows}) == 1051, "Canonical unique sample coverage")
    require(all(r["label"] in LABELS and all(r["predictions"][c] in LABELS for c in CONDITIONS) for r in rows), "Unknown label")
    groups = defaultdict(list)
    for row in rows:
        groups[row["project_group"]].append(row)
    require(len(groups) == 127, "Canonical project coverage")
    require(Counter(r["label"] for r in rows) == {"ID": 794, "NIO": 121, "OD": 126, "TD": 10}, "Canonical class coverage")
    project_results = []
    for group, members in sorted(groups.items()):
        repositories = {r["repository"].rstrip("/").lower() for r in members}
        require(len(repositories) == 1, "Inconsistent project repository")
        present_labels = sorted({r["label"] for r in members})
        metrics = {c: score(members, c) for c in CONDITIONS}
        present = {c: score(members, c, present_labels)["pooled_macro_f1"] for c in CONDITIONS}
        project_results.append(dict(project_group=group, repository=next(iter(repositories)),
                                    tests=len(members), class_support=dict(Counter(r["label"] for r in members)),
                                    metrics=metrics, mean_present_label_f1=present,
                                    delta_mean_present_label_f1=present["masked"] - present["original"],
                                    changes=changes(members)))
    activemq = {p["project_group"] for p in project_results if p["repository"] == ACTIVEMQ}
    singletons = {p["project_group"] for p in project_results if p["tests"] == 1}
    require(activemq == {"206387"}, "Unexpected ActiveMQ project identity")
    scenarios = [("all_projects", set()), ("exclude_activemq_evaluation", activemq),
                 ("exclude_singleton_evaluation", singletons),
                 ("exclude_activemq_and_singletons_evaluation", activemq | singletons)]
    results, summary_rows, class_rows = {}, [], []
    for name, excluded in scenarios:
        selected = [r for r in rows if r["project_group"] not in excluded]
        selected_projects = [p for p in project_results if p["project_group"] not in excluded]
        metrics = {c: score(selected, c) for c in CONDITIONS}
        for c in CONDITIONS:
            metrics[c]["mean_present_label_project_f1"] = statistics.fmean(p["mean_present_label_f1"][c] for p in selected_projects)
        change = changes(selected)
        paired = dict(pooled_fixed_four_label_macro_f1=metrics["masked"]["pooled_macro_f1"] - metrics["original"]["pooled_macro_f1"],
                      mean_present_label_project_f1=metrics["masked"]["mean_present_label_project_f1"] - metrics["original"]["mean_present_label_project_f1"],
                      deteriorated_projects=sum(p["delta_mean_present_label_f1"] < 0 for p in selected_projects),
                      improved_projects=sum(p["delta_mean_present_label_f1"] > 0 for p in selected_projects),
                      unchanged_projects=sum(p["delta_mean_present_label_f1"] == 0 for p in selected_projects))
        if name == "all_projects":
            for c in CONDITIONS:
                for key in ("pooled_macro_f1", "accuracy"):
                    require(math.isclose(metrics[c][key], published["metrics"][c][key], abs_tol=1e-12), "Baseline score mismatch")
                require(math.isclose(metrics[c]["mean_present_label_project_f1"], published["metrics"][c]["mean_project_macro_f1"], abs_tol=1e-12), "Baseline project score mismatch")
                require(metrics[c]["confusion_matrix"] == published["metrics"][c]["confusion_matrix"], "Baseline confusion matrix mismatch")
            require(change == dict(changed_predictions=56, right_to_wrong=56, wrong_to_right=0, wrong_to_different_wrong=0), "Baseline changed-decision mismatch")
        results[name] = dict(tests=len(selected), projects=len(selected_projects),
                             excluded_projects=sorted(excluded), excluded_tests=len(rows) - len(selected),
                             class_support={label: sum(r["label"] == label for r in selected) for label in LABELS},
                             category_project_groups={label: len({r["project_group"] for r in selected if r["label"] == label}) for label in LABELS},
                             metrics=metrics, paired_delta=paired, changes=change,
                             conditional_project_bootstrap=bootstrap(selected_projects, args.bootstrap_resamples, args.seed))
        summary_rows.append(dict(scenario=name, tests=len(selected), projects=len(selected_projects),
                                 pooled_original=metrics["original"]["pooled_macro_f1"], pooled_masked=metrics["masked"]["pooled_macro_f1"],
                                 pooled_delta=paired["pooled_fixed_four_label_macro_f1"],
                                 mean_project_original=metrics["original"]["mean_present_label_project_f1"],
                                 mean_project_masked=metrics["masked"]["mean_present_label_project_f1"],
                                 mean_project_delta=paired["mean_present_label_project_f1"], **change))
        for label in LABELS:
            for c in CONDITIONS:
                class_rows.append(dict(scenario=name, label=label, condition=c, **metrics[c]["per_class"][label]))
    concentration, concentration_rows = {}, []
    for label in LABELS:
        counts = Counter(r["project_group"] for r in rows if r["label"] == label)
        ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        total = sum(counts.values())
        class_projects = []
        for group, support in ordered:
            members = [r for r in groups[group] if r["label"] == label]
            record = dict(label=label, project_group=group, repository=members[0]["repository"],
                          support=support, share_of_class=support / total,
                          correct_original=sum(r["predictions"]["original"] == label for r in members),
                          correct_masked=sum(r["predictions"]["masked"] == label for r in members))
            class_projects.append(record)
            concentration_rows.append(record)
        concentration[label] = dict(tests=total, projects=len(counts), largest_project=class_projects[0],
                                    largest_project_share=ordered[0][1] / total,
                                    top_two_projects_share=sum(n for _, n in ordered[:2]) / total,
                                    project_distribution=class_projects)
    report = dict(analysis="Post-hoc project sensitivity of frozen primary word-LR LOPO predictions",
                  scope="Evaluation-only exclusions. No model refitting. ActiveMQ remains in original training folds for other projects.",
                  score_definitions={"pooled_macro_f1": "Macro-F1 over fixed ID/NIO/OD/TD labels; zero for undefined class F1.",
                                     "mean_present_label_project_f1": "Equal-weight mean over retained projects; each project's macro-F1 includes labels present in its ground truth only. This differs from fixed-four-label pooled macro-F1."},
                  label_order=list(LABELS), source_predictions=str(prediction_path.relative_to(root)),
                  source_predictions_sha256=sha(prediction_path), source_metrics_sha256=sha(source / "metrics.json"),
                  script_sha256=sha(Path(__file__)), python_version=platform.python_version(),
                  baseline_reproduced=True, scenarios=results, class_project_concentration=concentration,
                  projects=project_results,
                  refitting_interpretation="These scores test the concentration of evaluation outcomes. To claim robustness to removing ActiveMQ from the available training corpus, remove it before fitting and rerun all remaining LOPO folds; that is a separate estimand and was not done here.")
    destination.mkdir(parents=True)
    (destination / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_csv(destination / "summary.csv", summary_rows)
    write_csv(destination / "class_metrics.csv", class_rows)
    write_csv(destination / "class_project_concentration.csv", concentration_rows)
    write_csv(destination / "project_scores.csv", [dict(project_group=p["project_group"], repository=p["repository"],
               tests=p["tests"], singleton=p["tests"] == 1, activemq=p["project_group"] in activemq,
               original=p["mean_present_label_f1"]["original"], masked=p["mean_present_label_f1"]["masked"],
               delta=p["delta_mean_present_label_f1"], **p["changes"]) for p in project_results])
    description = """# Project sensitivity of frozen LOPO predictions

This post-hoc analysis rescored the primary word-TF-IDF logistic-regression predictions after excluding ActiveMQ, one-test project groups, or both from evaluation. No model was fitted and no canonical result was changed. Exclusions concern evaluation only: ActiveMQ and the singleton projects remain in the original training folds used to predict other projects.

`summary.csv` contains the four population comparisons. `class_metrics.csv` gives integer TP/FP/FN/support and class metrics. `project_scores.csv` gives each project's paired scores and decision changes. `class_project_concentration.csv` gives every class-bearing project and its share. `metrics.json` includes source hashes, exclusions, matrices, exact denominators and paired bootstrap intervals.

Pooled macro-F1 always includes all four fixed categories. Mean project F1 gives projects equal weight, averaging only the labels present in each project's ground truth. Singleton projects can consequently contribute a score of one for a single correct decision. The two summaries answer different questions.

Bootstrap intervals are descriptive and conditional on frozen fits. Original and masked conditions share each resampled project set; the training folds overlap. Rare categories may be absent from a bootstrap sample; the JSON records these counts and retains the four-label pooled denominator. These intervals do not establish population representativeness, independent replications or causation.

Reproduce with standard-library Python (choose a fresh output directory):

```sh
python scripts/analyze_project_sensitivity.py --output results/project_sensitivity_replication
```

Removing ActiveMQ from all training folds would require a separate refitted LOPO experiment. The present analysis establishes the sensitivity of the reported evaluation aggregate to its inclusion, not its influence on fitted models.
"""
    (destination / "README.md").write_text(description, encoding="utf-8")
    print(json.dumps({"baseline_reproduced": True, "output": str(destination.relative_to(root)), "summary": summary_rows}, indent=2))


if __name__ == "__main__":
    main()
