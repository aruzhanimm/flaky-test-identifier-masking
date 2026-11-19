# Flaky-Test Identifier Masking

Reproducible experiments on **identifier masking in cross-project classification of flaky-test categories**.

**Aruzhan Kartam · Saniya Yerzhan · Vitaliy Golubenko**  
Astana IT University, Kazakhstan

The study asks how replacing project and test identifiers in failure diagnostics changes category prediction on Java repositories excluded from training. It concerns **already-known flaky tests**, not binary flaky/non-flaky detection. The four inherited categories are implementation-dependent (ID), non-idempotent-outcome (NIO), order-dependent (OD) and timing-dependent (TD).

## Main findings

The canonical corpus contains **1,051 tests from 127 repository groups**. Original and masked conditions use identical observations and held-out groups. Vocabulary, IDF and classifiers are fitted using training projects only.

| TF-IDF | Classifier | Original macro-F1 | Masked macro-F1 | Change, percentage points |
|---|---|---:|---:|---:|
| Word | Logistic regression | 0.5991 | 0.5660 | −3.30 |
| Word | LinearSVC | 0.5782 | 0.5778 | −0.04 |
| Word | ComplementNB | 0.2475 | 0.2472 | −0.03 |
| Character | Logistic regression | 0.5647 | 0.5694 | +0.47 |
| Character | LinearSVC | 0.5688 | 0.5835 | +1.47 |
| Character | ComplementNB | 0.2475 | 0.2473 | −0.02 |
| Word | Logistic regression, C=0.1 | 0.4931 | 0.4839 | −0.93 |
| Word | Logistic regression, C=10 | 0.6028 | 0.6005 | −0.23 |

Each row is a paired comparison; the complete grid comprises 16 fitted conditions. LR and LinearSVC use C=1 unless indicated. The score is pooled macro-F1 over the four fixed categories; it is not the mean of project-level scores.

- In the primary word-LR comparison, all 56 changed decisions become incorrect. Forty-five losses occur in ActiveMQ; ten projects deteriorate and 117 are unchanged.
- Removing ActiveMQ from both training and evaluation and refitting 126 project-held-out folds reduces the word-LR penalty to 1.35 percentage points (0.6201 to 0.6066; 1,005 tests). Eight correct predictions become incorrect across five projects. Separate frozen-prediction exclusions quantify the effect of singleton projects.
- A transformation audit finds that 1,005 diagnostics change and 15,557 of 292,860 original identifier-shaped lexical tokens are affected (5.31%). All 2,846 configured-class string occurrences are replaced. These lexical counts characterize partial masking, not complete Java-identifier anonymization.
- Character LinearSVC corrects six errors in four projects without new errors. Character LR instead gains three correct OD decisions and loses 19 ID decisions: its pooled F1 rises while accuracy and mean project F1 fall.
- An eight-condition fitting-stage experiment separates TF-IDF fitting, LR fitting and evaluation input. With the other two stages fixed, training LR on masked representations reduces pooled F1 by 2.74–3.53 percentage points across four comparisons. These are context-dependent contrasts, not independent causal contributions.
- Five recovered ID failures from **Elide**, a previously excluded repository, form a separate same-source project check. All four frozen word pipelines and both diagnostic baselines predict all five correctly. They are not added to the canonical corpus or presented as an independent four-category benchmark.

## Quick start: verify saved results

Python 3.12 or later is sufficient for the read-only arithmetic and provenance checks; no ML packages or network access are needed:

```sh
python scripts/validate_saved_results.py
python scripts/verify_corpus.py
```

The first command independently reconstructs metrics and checks project separation, unique coverage, exact repeats and the recovered-project split. The second reparses retained source diagnostics and verifies CRC, SHA-256, target matching and exception preservation. Neither command retrains the models.

## Reproduce the experiments

Recorded environment: Python 3.12.14, scikit-learn 1.9.1, NumPy 2.5.4 and SciPy 1.18.1. Computation uses one CPU thread. Exact dependencies are in [requirements-model.txt](requirements-model.txt).

```sh
python -m venv .venv
# Activate the environment for your operating system, then:
python -m pip install -r requirements-model.txt
python -m unittest discover -s scripts -p "test_*.py"

python scripts/evaluate.py --name replication_main
python scripts/evaluate.py --name replication_three_labels --three-label
python scripts/run_extension_v4.py --name replication_operations
python scripts/run_extension_v5_models.py --name replication_models
python scripts/run_extension_v5_mechanism.py --name replication_stages
python scripts/evaluate_recovered_project_v5.py --name replication_elide
python scripts/refit_without_activemq.py --output results/replication_without_activemq
python scripts/analyze_project_sensitivity.py --output results/replication_project_sensitivity
python scripts/analyze_masking_coverage.py --output results/replication_masking_coverage
```

Choose fresh run names: existing result directories are never silently replaced. The model grid supports bounded checkpoint batches, for example `--batch-size 8`, followed by the same command with `--resume`. A complete saved reference run is already included; there is no need to retrain merely to inspect the results. CPU time depends on the machine; the recorded successful-fold compute time for the 16-condition grid was about 618 seconds, excluding process recovery.

## Repository contents

| Location | Contents |
|---|---|
| [scripts/](scripts) | Parsers, masking rules, experiments, independent checks and 28 focused tests |
| [protocols/](protocols) | Scientific protocols, pre-final amendments and declared post-hoc extensions |
| [data/processed/](data/processed) | Canonical paired corpus, category coverage, exclusions and duplicate audit |
| [data/extracted/](data/extracted), [data/first_failures/](data/first_failures) | Retained diagnostics and selection ledger supporting target-level verification |
| [data/extension_candidates_2026-11-19/](data/extension_candidates_2026-11-19) | Candidate eligibility ledger and the separate Elide observations |
| [results/main_v3/](results/main_v3) | Primary paired comparison |
| [results/main_v3_repeat/](results/main_v3_repeat), [results/main_v3_three_labels/](results/main_v3_three_labels) | Exact repeat and preplanned no-NIO sensitivity |
| [results/extension_v4/](results/extension_v4) | Masking operations, fixed-model input swaps, SVC and Surefire OD/TD controls |
| [results/extension_v5_models_2026-11-19_batched/](results/extension_v5_models_2026-11-19_batched) | Complete model/representation/regularisation grid |
| [results/extension_v5_mechanism_2026-11-19/](results/extension_v5_mechanism_2026-11-19) | Eight controlled fitting-stage conditions |
| [results/extension_v5_recovered_project_2026-11-19/](results/extension_v5_recovered_project_2026-11-19) | Frozen-pipeline Elide evaluation |
| [results/project_sensitivity/](results/project_sensitivity), [results/project_sensitivity_refit/](results/project_sensitivity_refit) | Frozen-prediction population checks and a complete ActiveMQ-excluded LOPO refit |
| [results/masking_coverage/](results/masking_coverage) | Exact transformation coverage, retained-name examples and denominator definitions |
| [evidence/](evidence) | Source/revision checks, numerical-failure diagnosis and publication integrity records |

See [DATA.md](DATA.md) for fields and label definitions, [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for scope and release details, and [ATTRIBUTION.md](ATTRIBUTION.md) for source attribution and reuse terms.

## Interpretation limits

The effect of masking depends on the learner, representation, regularisation and evaluated population. Source format is associated with category subsets: NonDex contains ID, NIO JSON contains NIO, and Surefire contains OD/TD. A training-only format rule reaches pooled macro-F1 0.7405; project exclusion alone does not establish collection-independent learning. TD has only ten retained tests and NIO only three project groups. The project bootstrap is conditional and exploratory because training folds overlap and the corpus is availability-based. Source labels are inherited; masking is partial.

The model, operation, fitting-stage and recovered-data extensions followed inspection of the primary results. Complete fixed comparisons and execution deviations are retained. No new Java or Docker executions are claimed. See the protocols for the distinction between the initial study, preplanned sensitivity and later diagnostics.

## Citation

Use [CITATION.cff](CITATION.cff) to cite this software/data-analysis release, and also cite the original [ReproFlake Results dataset](https://doi.org/10.5281/zenodo.18451345). Source diagnostics and third-party code retain their original attribution and terms.
