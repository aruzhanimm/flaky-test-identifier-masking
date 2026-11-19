RECOVERED UNSEEN-PROJECT EVALUATION — 19 NOVEMBER 2026

Written before any predictions on the five qualified Elide observations.

Training data are the unchanged 1,051-row canonical corpus, SHA-256 4fbee74d76717535e80f7c40841852784a23c1907954e8227ad2ae31f87c5127. Fit exactly four pipelines: original/masked TF-IDF + logistic regression and original/masked TF-IDF + LinearSVC. Use the existing config_model_v3.json vectorizer and logistic-regression settings and the existing extension_v4/config.json SVC settings, without tuning.

Each pipeline fits its vectorizer, IDF and classifier only on the full original 1,051 training observations. Predict the five separately qualified Elide observations from eligible_candidates.jsonl once. The entire Elide repository group is absent from training. No cross-validation training fold is recycled and no new test is appended to the canonical training corpus. All hyperparameters were chosen in earlier protocols; observed candidate scores cannot change inclusion, preprocessing or model settings.

Primary reported quantities here: correct/total and accuracy for each representation and learner, paired changed decisions and their direction, and individual predictions with source/test/commit identity. Report simple training-majority and training-source-format rules as context. This is a five-test, one-project, all-ID holdout rescued from the same ReproFlake collection. Do not calculate or describe its accuracy as four-category macro-F1, independent-source external validation, or population-wide evidence. No significance test is proposed for five correlated tests from one project.

The original 127-fold study remains the primary experiment. The new analysis is a bounded follow-up showing frozen-model behavior on a previously excluded repository whose target-failure evidence and source provenance are now resolved. No Java tests are rerun in this evaluation.

Checks: exact train/evaluation canonical-test, repository-group and raw-diagnostic hash disjointness; 1,051 training rows and five unique evaluation rows; all evaluation labels traceable to source ID entries; inherited exception strings retained under masking; no fit on evaluation text; no convergence warning; hashes of corpus, candidate file, protocol, configuration and implementation recorded. Fit on one CPU thread. Archive predictions and evidence before interpreting outcomes.
