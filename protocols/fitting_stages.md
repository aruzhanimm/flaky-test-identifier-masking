# Representation and classifier mechanism diagnostic

Protocol date: 19 November 2026. This extension is specified before computing its new conditions, after inspection of main_v3 and extension_v4. It is an exploratory follow-up, not an independent confirmatory experiment.

## Research purpose

The original paired comparison changes the training text, fitted TF-IDF representation and fitted classifier simultaneously. Existing cross-input checks keep the whole fitted pipeline fixed. This diagnostic separately varies the corpus used to fit TF-IDF, the text used to fit logistic regression under that fixed feature mapping, and the held-out input representation.

## Frozen population and settings

Use the unchanged 1,051-test corpus from 127 canonical repository groups in data/processed/corpus.jsonl (SHA-256 4fbee74d76717535e80f7c40841852784a23c1907954e8227ad2ae31f87c5127). Original/masked texts, labels, seed 20261013, vectorizer settings and logistic-regression settings are inherited unchanged from config_model_v3.json. Hyperparameters are not selected by these results. Each repository group is held out once; all transformations and classifiers are fitted exclusively on the remaining groups. Each training fold must contain all four labels.

## Eight conditions

Each three-letter condition denotes (TF-IDF fitting text, classifier fitting text, evaluation text); O is original and M is masked. Evaluate OOO, OOM, OMO, OMM, MOO, MOM, MMO and MMM. For every fold:

1. Fit separate TF-IDF vectorizers on original or masked training text.
2. Under each fitted vectorizer, transform both original and masked training text, then fit a separate logistic-regression classifier on each transformation.
3. Apply each fitted vectorizer/classifier pair to both held-out original and masked text.

This requires 254 vectorizer fits, 508 classifier fits and 1,016 held-out prediction batches. OOO and MMM must reproduce the main_v3 original and masked predictions and recorded fitted-state hashes. OOM and MMO are compared with the existing cross-input predictions in extension_v4.

Changing the vectorizer source changes vocabulary selection, IDF and the resulting normalisation jointly; these are not separately identified. A placeholder absent from the original-fitted vocabulary is legitimately out of vocabulary in that intervention. Such feature support differences are recorded and are part of the diagnostic, not silently corrected using held-out data.

## Outputs and interpretation

Save sample-level predictions, split manifests, per-project scores, pooled four-label macro-F1, accuracy, confusion matrices and per-category precision/recall/F1/support. Project scores retain the main study's present-label denominator. Save vocabulary/coefficient/IDF hashes, vocabulary overlap, common-feature IDF change, matrix sparsity and fit iterations. Report all eight outcomes.

For explanation, enumerate all six orders for moving from OOO to MMM and report the changes at each stage. The three stage changes telescope to the observed endpoint difference within an order, but their allocation depends on the path. A small main effect or large interaction is not interpreted as a causal diagnosis of a flaky test or a universally removable source of error. No new population-level significance claim is specified for these dependent diagnostic conditions.

An independent implementation using contingency counts must reconstruct macro-F1, accuracy and confusion matrices from saved predictions. The original and masked endpoint reproduction and matching split membership are required. Convergence warnings are treated as errors. Record start/end times, environment and data/config/protocol/code hashes. Do not overwrite any existing run directory, source corpus, historical result or prior artifacts.
