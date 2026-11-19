# Masking coverage audit

Run `python scripts/analyze_masking_coverage.py --output results/masking_coverage_NEW`
from the repository. Existing result directories are never overwritten. Standard
Python is sufficient; no fitted model or external data is required.

The audit validates every saved text hash and independently traces each production
masking substitution back to its original character positions. All 1,051
traced outputs equal both the production function and the saved masked text.

## Results and denominator

- Diagnostics changed: 1,005/1,051
  (95.62%).
- Identifier-shaped original lexical tokens affected: 15,557/292,860
  (5.31%). A token is a non-overlapping
  `[A-Za-z_$][\w$]*` match in the original diagnostic. Affected means at least
  one original character was consumed by a masking substitution. This includes
  prose, strings, packages and exceptions; it is not Java identifier resolution
  and is not the fitted TF-IDF vocabulary. New placeholder tokens are excluded.
- Configured full/simple-class string occurrences fully replaced:
  2,846/2,846.
- Repository-name string occurrences fully replaced:
  4,697/4,739.
  40 remain inside preserved exception names;
  2 are partially replaced.

`summary.json` defines all quantities and records file hashes. `per_document.csv`
and `by_label.csv` give per-diagnostic and per-label counts. Every scoped original
occurrence, source offset, status and masking operation is recorded in
`scoped_identifier_occurrences.csv`. Kind-specific denominators overlap and must
not be summed. Matching is lexical: a method-name spelling in a package or quoted
assertion is not necessarily a method invocation.

## Scope and observed residuals

This is a metadata-scoped intervention, not full identifier anonymization. The
configured class, sufficiently long exact method name, repository basename and
selected repository-name components are targeted. Other classes, package tokens,
methods and names can remain. The standalone method rule requires the complete
configured method string to contain at least six characters. A parameterized
base name is replaced following the configured class, but its other standalone
occurrences need not be replaced. All retained base-method spelling matches in
this corpus are `test`; some are homonyms in strings or package names.

Exception names are preserved by design. In one diagnostic (sample
`bf4842389bcb8e7a10a4cdf5`) the existing path rule consumes the final `o://...`
of two `dubbo://...` URLs as a drive-like path, producing `dubbPROJECT_PATH`.
These are counted as partial replacement, not successful complete replacement.
No masking implementation, corpus or experimental result was changed by this
audit. A stricter path parser would be a separate sensitivity analysis.

`residual_examples.json` shows retained `StompTest` frames where that name differs
from the configured class. The mask does not resolve inherited declaring classes.
The coverage fraction therefore describes the scale of this specific
intervention, and cannot measure removal of all project-identifying information.
