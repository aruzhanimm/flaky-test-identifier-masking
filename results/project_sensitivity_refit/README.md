# ActiveMQ-excluded LOPO refit

This post-hoc sensitivity removes 46 ActiveMQ tests before fitting and evaluates 1005 tests from the remaining 126 projects. Original and masked word-TF-IDF logistic-regression pipelines use the unchanged primary configuration. Each vocabulary, IDF and model is fitted only on the other 125 retained projects. No model or parameter selection is performed.

Reproduce with the recorded model environment and a new output directory:

```sh
python scripts/refit_without_activemq.py --output results/project_sensitivity_refit_replication
```

`metrics.json` records scores, integer class counts, paired changes and descriptive conditional bootstrap intervals. `splits.json` contains all memberships; `fits.json` contains 252 fit records and fitted-state hashes. Nonfinite matrices/coefficients and convergence/runtime warnings stop the run rather than being repaired or ignored. An interrupted run records `failure.json` and must not be treated as completed.
