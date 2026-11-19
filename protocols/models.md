# Extension v5: model and representation robustness

Protocol recorded on 19 November 2026 before fitting or inspecting outcomes of these new conditions. This is an explicitly post-hoc extension of the established exploratory study, not a preregistration of the original research. Existing results, corpus, labels and protocols remain unchanged.

## Question and fixed design

Assess whether the paired effect of identifier masking is reproduced across text representations, classifier families and fixed logistic-regression regularisation strengths. The research question remains category classification of already-known flaky tests in previously unseen Java projects.

Use the canonical 1,051-test corpus (SHA-256 `4fbee74d76717535e80f7c40841852784a23c1907954e8227ad2ae31f87c5127`), the same ID/NIO/OD/TD labels, the same 127 repository groups and the exact saved `main_v3` train/test memberships. A fresh vectorizer is fitted only to training texts for each fold, representation and original/masked condition. Its fitted sparse matrices may be shared by the different classifiers evaluated on that identical representation; no held-out text contributes vocabulary or IDF statistics. Each classifier is new for each fold and condition. Check that all four labels occur in every training fold.

Factorial comparison: word versus character TF-IDF; LogisticRegression, LinearSVC and ComplementNB; original versus the existing masking-v2 text. These 12 conditions include reruns of the four established word LR/SVC conditions, which must be checked against saved predictions and fitted-state hashes where available.

Word TF-IDF retains every setting from `config_model_v3.json`: lowercase, word unigrams/bigrams, token pattern `(?u)\b\w\w+\b`, min_df=1, max_features=30000, sublinear_tf=True, L2 norm. Character TF-IDF uses analyzer=`char`, ngram_range=(3,5), lowercase=True, min_df=1, max_features=30000, sublinear_tf=True, norm=`l2`. The character analyzer includes whitespace/punctuation and can preserve identifier fragments; it is an alternative representation, not anonymisation or removal of collection-format information.

LogisticRegression retains C=1, balanced class weights, lbfgs, max_iter=2000, random_state=20261013. LinearSVC retains C=1, balanced class weights, squared_hinge, dual=`auto`, tol=0.0001, max_iter=10000, random_state=20261013. ComplementNB uses alpha=1, force_alpha=True, fit_prior=True, norm=False; nonnegative TF-IDF input, no extra resampling/class weighting. In addition, repeat word LogisticRegression for C=0.1 and C=10 under both input conditions. C=1 is shared with the factorial comparison. Thus 16 unique conditions, 508 vectorizer fits and 2,032 classifier fits are planned. No test-informed tuning or selection of the best C.

## Evaluation and interpretation

Preserve one out-of-project prediction per test and condition. Report pooled four-category macro-F1, per-category precision/recall/F1 and confusion matrices, accuracy, present-label project-average F1, fixed-four-label project-average F1, the paired project differences, and counts of improved/worsened/unchanged projects and decisions. Project scores retain the original definition over true labels present in that project; fixed-four-label scores are a separate sensitivity measure.

For each model/representation pair and each additional C, report a descriptive paired project bootstrap interval using 5,000 whole-project resamples and seed 20261013, retaining pairing. These are conditional exploratory intervals; overlapping training folds and the availability sample limit population inference. Report the complete fixed experiment grid, with no confirmatory multiplicity claim. A higher score does not by itself establish improved generalisation: category-format confounding and sparse TD/NIO projects persist.

## Verification and stopping rules

Use a single CPU computation thread. Treat convergence warnings as failures, save the failure/checkpoint and stop rather than silently altering hyperparameters. Any convergence repair requires a separately documented run/protocol deviation. No Java tests, Docker executions or new data collection are claimed.

Recompute metrics independently from integer TP/FP/FN counts and check against scikit-learn. Verify exact saved train/test memberships, paired IDs, complete/unique held-out coverage and the four reproduction conditions. Record per-fold checkpoints, vocabulary/IDF/model hashes, input/protocol/config/code hashes, versions, timing and any failure. Never overwrite a pre-existing result directory. Intended result directory: `results/extension_v5_models_2026-11-19/`; evidence: `evidence/extension_v5_models/`.
