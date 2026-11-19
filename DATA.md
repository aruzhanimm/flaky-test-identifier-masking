# Data definitions

## Canonical population

`data/processed/corpus.jsonl` contains one JSON object per canonical configured test, not one row per repeated execution. Its SHA-256 is:

```text
4fbee74d76717535e80f7c40841852784a23c1907954e8227ad2ae31f87c5127
```

| Label | Meaning in the source collection | Tests | Repository groups |
|---|---|---:|---:|
| ID | Implementation-dependent: assumptions about underspecified API behaviour | 794 | 118 |
| NIO | Non-idempotent-outcome: a repeated same-JVM execution fails after an initial pass | 121 | 3 |
| OD | Order-dependent: outcome depends on preceding tests | 126 | 8 |
| TD | Timing-dependent: concurrency/asynchronous timing | 10 | 6 |
| Total | Groups overlap across categories | 1,051 | 127 |

The initial inventory contains 1,115 configured records. Eligibility, exclusions and duplicate resolution are preserved alongside the corpus. Labels are inherited from source evidence rather than independently re-established causal diagnoses.

## Main fields

| Field | Meaning |
|---|---|
| `sample_id` | Stable identifier used to align paired predictions |
| `canonical_test_id` | Canonical test identity used for duplicate checks |
| `project_group` | GitHub network-root grouping used for held-out evaluation |
| `canonical_repository` | Attributed repository URL |
| `configured_test` | Exact Java class and method requested by the source configuration |
| `container`, `source`, `revision` | Source provenance and attributable revision |
| `label`, `format` | Inherited category and diagnostic-collection format |
| `original_text`, `masked_text` | Paired classifier inputs; only these diagnostic texts enter the models |
| `original_sha256`, `masked_sha256` | Integrity checks for the paired text |
| `diagnostic_member` | Archived member path, local retained source, size, CRC and SHA-256 |
| `exception_type` | Observed target exception, if present; absent values are not invented |

Selection ledgers can refer to acquisition attempts outside the released subset. The release includes sources required to verify every canonical retained target and the parser regression fixtures; it does not redistribute the multi-gigabyte archive or its complete file index.

## Separate recovered project

`data/extension_candidates_2026-11-19/eligible_candidates.jsonl` contains five ID observations from one Elide group absent from the canonical corpus. Target source files match the attributed full revision byte for byte. These records form a separate evaluation population; they are not silently merged into the 1,051-row benchmark. The candidate ledger includes rejected/conditional cases and their reasons.

## Saved predictions and metrics

Predictions use sample IDs, true source labels and one predicted label per condition. `splits.json` records training and held-out memberships; `fits.json` and `vectorizers.json` record fitted-state hashes and settings where applicable. Pooled macro-F1 averages four fixed category scores. Project F1 averages the categories present in that project's true labels. The two scores have different denominators and must not be interchanged.
