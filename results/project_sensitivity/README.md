# Project sensitivity of frozen LOPO predictions

This post-hoc analysis rescored the primary word-TF-IDF logistic-regression predictions after excluding ActiveMQ, one-test project groups, or both from evaluation. No model was fitted and no canonical result was changed. Exclusions concern evaluation only: ActiveMQ and the singleton projects remain in the original training folds used to predict other projects.

`summary.csv` contains the four population comparisons. `class_metrics.csv` gives integer TP/FP/FN/support and class metrics. `project_scores.csv` gives each project's paired scores and decision changes. `class_project_concentration.csv` gives every class-bearing project and its share. `metrics.json` includes source hashes, exclusions, matrices, exact denominators and paired bootstrap intervals.

Pooled macro-F1 always includes all four fixed categories. Mean project F1 gives projects equal weight, averaging only the labels present in each project's ground truth. Singleton projects can consequently contribute a score of one for a single correct decision. The two summaries answer different questions.

Bootstrap intervals are descriptive and conditional on frozen fits. Original and masked conditions share each resampled project set; the training folds overlap. Rare categories may be absent from a bootstrap sample; the JSON records these counts and retains the four-label pooled denominator. These intervals do not establish population representativeness, independent replications or causation.

Reproduce with standard-library Python (choose a fresh output directory):

```sh
python scripts/analyze_project_sensitivity.py --output results/project_sensitivity_replication
```

Removing ActiveMQ from all training folds would require a separate refitted LOPO experiment. The present analysis establishes the sensitivity of the reported evaluation aggregate to its inclusion, not its influence on fitted models.
