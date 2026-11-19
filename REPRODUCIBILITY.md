# Reproducibility and release scope

This release contains the computational study: paired corpus, source diagnostics, scientific protocols, analysis scripts, complete fixed comparison results and validation evidence. New runs must use new output names. The recorded corpus, predictions, configurations, fits and train/test membership files are preserved rather than regenerated for publication.

## Provenance of the public package

The main and operation protocols are English scientific-content projections of the dated local protocols. The v5 protocols retain their scientific design. Editorial instructions and unrelated workspace inventories are omitted; machine-specific execution paths in a small number of failure metadata records are replaced with neutral project/environment prefixes. These projections do not alter observed diagnostics, model settings, predictions or metric values. `evidence/publication/projections.json` records original protocol and projected-metadata hashes, and `evidence/publication/manifest.json` identifies the released files and their checksums.

Stored `protocol_sha256` and code-hash fields describe the historical execution versions. They are not claims that the public projections or portability adapters are byte-identical to those historical files. Current released sources have separate manifest hashes. Portability changes add a fresh-run name and main guard to the Elide runner, use scientific protocol paths, use the canonical QuickCheck fixture in the masking tests, and make verification read-only by default. Scientific hyperparameters and transformation logic are unchanged.

## Execution deviations

The operation extension initially encountered a NumPy-integer serialization error. Only output serialization was corrected; the complete experiment was repeated with unchanged scientific settings.

The model grid's first process was interrupted after twelve completed folds. Execution resumed in verified bounded batches, preserving complete checkpoint bytes and checking unchanged corpus, configuration, splits and environment. Later, one fresh TF-IDF fit reported an invalid logarithm/NaN. The same fold completed in a fresh process with unchanged code, data and settings. Independent diagnostics verified positive finite counts, finite IDF, equality of separate CountVectorizer/TF-IDF matrices, no sparse-matrix mutation by classifiers, and exact reproduction of all four vectorizer states and sixteen model hashes/predictions. No NaNs were replaced and no observations were dropped; the original transient cause remains unestablished. Details and the successful fold checkpoint are retained under `evidence/extension_v5_models` and the model result directory.

The successful complete model-grid status is `completion_checks.json`; its retained `failure.json` is historical failure evidence, not the final completion status. Aggregate predictions, splits, fitted hashes and metrics cover all 127 folds. Redundant checkpoints other than the independently diagnosed fold are unnecessary to reconstruct or validate those aggregate outputs.

## Project-composition and masking-coverage checks

The later post-hoc project analysis distinguishes rescoring frozen predictions after evaluation exclusions from refitting every fold after removing ActiveMQ from training and evaluation. The latter uses 1,005 tests, 126 folds and 252 fresh word-LR fits with unchanged settings. Predictions, memberships, fitted-state hashes and independent count-based verification are retained in `results/project_sensitivity_refit`; the frozen-prediction comparisons are in `results/project_sensitivity`.

The masking audit traces each replacement to original character positions and exactly reproduces all saved masked diagnostics. It reports lexical coverage rather than AST-resolved identifier recall. Its retained-name and path-fragment examples, including the observed `dubbo://` path-rule edge case, are documented in `results/masking_coverage`. The historical masker and canonical corpus remain unchanged.

The publication manifest describes the initial release. These later analyses record their own input, code and output hashes alongside their results.

## Validation

`python scripts/validate_saved_results.py` uses independent integer TP/FP/FN to reconstruct metrics, validates sample/group coverage and checks repeated predictions and fitted hashes. It includes the Surefire subtask and separate Elide population. It writes nothing unless an explicit fresh `--output` path is supplied.

`python scripts/verify_corpus.py` checks retained diagnostic CRC/size/SHA-256, reparses exact targets and validates paired text hashes and protected exception names. `python -m unittest discover -s scripts -p "test_*.py"` checks parser formats, exclusions, masking edge cases, serialization and metric arithmetic. Synthetic unit-test fixtures are not research observations.

The public package does not claim that Java tests were rerun, that the extensions were preregistered, or that observed source labels are independently confirmed causes. It supports reproduction of the saved Python-model experiments and inspection of their diagnostic provenance.
