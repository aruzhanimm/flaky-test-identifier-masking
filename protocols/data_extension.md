DATA EXTENSION PROTOCOL — 19 NOVEMBER 2026

Scope: a new, separately tracked extension of the existing paired failure-log study. The original 1,051-row corpus, existing results and protocols remain unchanged. This protocol is written before fitting any model on newly qualified observations. Candidate inspection is exploratory; no preregistration claim is made.

Candidate sources, in priority order:
1. The seven previously excluded ReproFlake records with missing/invalid/disputed revision attribution, using original diagnostic artifacts plus independently checkable public build/source metadata.
2. The 25 ReproFlake TD records whose archived outcome table has no failure; seek exact existing public target-failure evidence or genuinely rerun only if a suitable execution environment exists.
3. Public Java flaky-test research catalogs and archived failure logs, especially iDoFT notes and FlakeFlagger/FlakeRake. A catalog row alone is insufficient.

Inclusion requires: (a) exact configured Java target and a confirmed failure with nonempty diagnostics, (b) repository identity and full 40-character source revision tied to the logged execution, (c) a category supported by the source evidence and definition compatible with the existing ID/OD/NIO/TD taxonomy, (d) traceable source URL/member and SHA-256 of retained evidence, (e) original and masked text produced by the existing cleanup/masking rules, and (f) successful overlap checks.

Revision resolution must establish the actual execution revision. Resolving an abbreviated hash or finding a modern test file alone does not prove that an archived failure ran that revision. A label inferred only from a token such as Timeout is ineligible. iDoFT TD and ReproFlake TD require definition-level compatibility, not an abbreviation match. Mutation-induced true failures are not flaky-category observations.

Overlap: map aliases/forks to canonical GitHub network-root repository groups; compare canonical test identity, revision and diagnostic hashes against the frozen corpus. Multiple failures of one test are not independent new tests. Additional records in existing projects constitute a same-source extension, not external project validation. ReproFlake-derived iDoFT records are not independent merely because obtained from another URL.

Exclusions remain explicit: no target failure, missing/ambiguous revision, incompatible or absent label, duplicate test, unresolved project identity, unavailable source, or unavailable execution requirements. Do not replace absent logs with generated examples; do not count reruns as new projects.

Preserve every candidate in candidate_ledger.json with status, evidence and remaining prerequisite. Save any newly eligible rows to a separate eligible_candidates.jsonl before fitting. If none qualifies, report zero and preserve the ledger. Models must not determine which candidate is included.

External evaluation, if feasible: freeze training corpus/model choices before examining external scores; exclude all overlapping project groups; report category/project support and collection-format differences. A targeted subset without all four categories is a bounded subtask, not a four-category external replication.

Acquisition bounds: read public metadata and small diagnostics; no blind multi-gigabyte downloads, mass Java builds, or system-level Docker/Java installation in this task. The retained evidence supports candidate eligibility and does not establish independent external validation.
