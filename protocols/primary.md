# Primary paired evaluation protocol

This is an English scientific-content projection of the dated local protocol of 13 October 2026. The historical protocol hash is recorded in `evidence/publication/projections.json`. It was not a public preregistration. Diagnostic formats and pilot scores had been inspected; the amendments below were made before freezing the final corpus and fitting the final main models.

## Question and population

How does masking project and test identifiers in failure logs affect macro-F1 for category classification on previously unseen Java projects? The task classifies categories of already-known flaky tests, not whether arbitrary tests are flaky. Source labels are inherited and are not independently established causal diagnoses.

One canonical configured project/test, including its recorded parameterised case, is one observation. Select the earliest available confirmed target failure by recorded iteration and source-table row. NIO requires the exact embedded target result; Surefire prefers XML, otherwise the same class's text report. Repeated executions do not increase sample size. Across eligible variants, prefer iDoFT before JIRA and then lexical container order; this is deterministic selection, not proof of the earliest historical revision.

Require a confirmed exact-target failure, nonempty diagnostics, a known label, an attributable full revision and a verified repository group. Exclude passing/skipped targets, missing or disputed revisions and unclassified records. Distinguish acquisition failure from observed absence of a failure. Class-initialisation and missing-class flags require inspection rather than automatic exclusion, because they can arise from order dependence. Java tests and Docker environments were not rerun for this analysis.

Canonical groups use GitHub repository-network-root identity to join aliases and known forks. This does not rule out common libraries or independent mirrors. Exact cross-project diagnostic duplicates are resolved by lexical canonical-test identity; different tests within a project remain grouped together. ID, NIO, OD and TD each require at least three project groups. The preplanned sensitivity excludes NIO because its tests are concentrated in one repository. Eligibility changes must precede model outcomes and retain reasons.

## Representations and fitting

Input contains target exception, message and stack trace, not labels, archive names, neighbouring tests, XML properties, durations or build summaries. Both conditions normalise line endings and remove ANSI escapes, trailing whitespace and explicit NonDex/testrunner/Surefire/JaCoCo collection frames. Missing fields are not inferred from labels.

Masking uses explicit project/test metadata without learning from outcome scores. Preserve exception class names, including chained causes. Replace target class/method names with shared placeholders; standalone method replacement requires at least six characters, while short names require qualified context. Replace non-generic repository identifiers and qualifying paths. Generic framework names and numerical values remain. These rules are partial masking, not complete anonymisation; residual production identifiers may remain.

Hold out each repository group once and fit fresh TF-IDF and LR on training projects only for every original/masked condition. Keep paired sample IDs and labels identical. Settings are in `config_model_v3.json`: word unigrams/bigrams, lowercase, two-or-more-character word tokens, min_df=1, max_features=30000, sublinear TF and L2 norm; LR C=1, balanced weights, lbfgs, max_iter=2000 and seed 20261013. Each training fold must contain all eligible labels. Use one CPU computation thread; convergence warnings stop the run. No held-out-score tuning is performed.

Training-only baselines predict the majority label or the modal label for a source format. Lexical ordering breaks ties, and unseen formats fall back to the training majority. Baseline format metadata is not supplied to the text classifiers.

## Outcomes and uncertainty

The primary project score averages F1 over the true labels present in that held-out project, explicitly identical for both conditions; undefined scores are zero. The estimand is the unweighted mean masked-minus-original project difference. H0 states that the expected paired project difference is zero; H1 is two-sided. A nominal 95% interval excluding zero supplies exploratory evidence against H0, not a universal causal claim.

Also report pooled held-out macro-F1 over all four fixed labels, per-category precision/recall/F1/support, confusion matrices, accuracy, project scores and all changed decisions. A high present-label project score is not equivalent to strong four-category prediction. Resample 5,000 whole project-score pairs using seed 20261013 and use percentile endpoints. The interval is descriptive and conditional on the saved predictions: training folds overlap, rare categories have few projects and availability sampling limits population inference. Do not treat log rows as independent for a t-test or interpret a null result as equivalence.

Preserve predictions, splits, configurations, vocabulary/coefficient hashes, counts and timings. Independently recalculate metrics and repeat the primary evaluation. Publish every changed case; changes after score inspection must be versioned and explained. No observation is removed to improve a score.

## Recorded pre-final amendments

The initial 435-test pilot produced pooled F1 0.5822096365 in both conditions, with two opposite prediction changes. Inspection then showed that generic framework components could be masked when a repository name contained them. Masking v2 protects junit, testng, hamcrest, mockito and jupiter; the same pilot IDs, labels and original texts were retained. This followed access to pilot scores and was a rule-correction, not selection of the highest score.

Parser v3 was introduced before final fitting after finding actual JUnit 5 XML names such as `testScanOverEmpty(ScanTimestampProvider)[1]` under the exact configured class. It accepts explicit Java signatures and parameter suffixes, rejects arbitrary prefixes/display-name mapping and flags ambiguous overloads. Class-level errors are not assigned to an arbitrary target method. Final corpus construction uses this parser and masking v2. Earlier pilot outputs are not substituted for the final evaluation.
