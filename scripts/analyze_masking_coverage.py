"""Audit the scope of saved identifier masking without changing corpus or models.

Run from any directory: python scripts/analyze_masking_coverage.py
Only the standard library is required. Every traced result must equal both the
production mask() output and the saved masked_text before any result is written.
"""
import argparse
import collections
import csv
import hashlib
import json
import platform
import re
from pathlib import Path

from representations import EXCEPTION, GENERIC, mask

ROOT = Path(__file__).resolve().parents[1]
LEXICAL = re.compile(r'[A-Za-z_$][\w$]*')
PATH = re.compile(r'(?:[A-Za-z]:[\\/]|(?<![:\w])/)[^\s<>"\x27)\],;]+')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def names_for(meta, repository):
    names = {meta['project'].rstrip('/').split('/')[-1]}
    if repository:
        names.add(repository.rstrip('/').split('/')[-1])
    for name in list(names):
        names.update(p for p in re.split(r'[-_]', name)
                     if len(p) >= 4 and p.lower() not in GENERIC)
    return {n for n in names if n.lower() not in GENERIC}


def bounded(values, ignorecase=False):
    values = sorted(set(filter(None, values)), key=lambda x: (-len(x), x))
    if not values:
        return re.compile(r'(?!)')
    return re.compile(r'(?<![\w$])(?:' + '|'.join(map(re.escape, values)) + r')(?![\w])',
                      re.I if ignorecase else 0)


class Trace:
    """Map surviving characters to their original positions, not a text diff.

    Generated placeholders have no original positions. Temporary exception
    protection is excluded from replacement counts. Final exception restoration
    does not need a position map because no operation follows it.
    """
    def __init__(self, text):
        self.text = text
        self.origins = list(range(len(text)))
        self.positions = collections.defaultdict(set)
        self.calls = collections.Counter()

    def sub(self, pattern, replacement, operation=None):
        parts, origins, end = [], [], 0
        for match in pattern.finditer(self.text):
            value = replacement(match) if callable(replacement) else replacement
            start, stop = match.span()
            parts.append(self.text[end:start])
            origins.extend(self.origins[end:start])
            parts.append(value)
            if value == match.group():
                origins.extend(self.origins[start:stop])
            else:
                if operation:
                    self.calls[operation] += 1
                    self.positions[operation].update(i for i in self.origins[start:stop]
                                                     if i is not None)
                origins.extend([None] * len(value))
            end = stop
        parts.append(self.text[end:])
        origins.extend(self.origins[end:])
        self.text = ''.join(parts)
        self.origins = origins
        assert len(self.text) == len(self.origins)

    def literal(self, value, replacement, operation, ignorecase=False):
        if value:
            self.sub(bounded([value], ignorecase), replacement, operation)


def traced_mask(text, meta, repository, exception_type):
    trace = Trace(text)
    klass, method = meta['configured_test'].split('#', 1)
    simple, base = klass.split('.')[-1], method.split('[', 1)[0]
    protected = sorted(set(EXCEPTION.findall(text)) |
                       ({exception_type} if exception_type else set()),
                       key=len, reverse=True)
    kept = {}
    protected_positions = set()
    for i, value in enumerate(protected):
        key = f'ZZPROTECTEDEXCEPTION{i}ZZ'
        for found in re.finditer(re.escape(value), text):
            protected_positions.update(range(*found.span()))
        kept[key] = value
        trace.sub(re.compile(re.escape(value)), key)
    names = names_for(meta, repository)
    for url in (meta['project'], repository):
        if url:
            trace.sub(re.compile(re.escape(url)), 'PROJECT', 'repository_url')
    path_names = [n.lower() for n in names | {simple, klass.replace('.', '/')}
                  if len(n) >= 4]
    trace.sub(PATH, lambda m: 'PROJECT_PATH' if any(n in m.group().lower()
                                                  for n in path_names) else m.group(),
              'project_path')
    for value in (klass + '.' + method, klass + '#' + method, klass + '.' + base):
        trace.literal(value, 'TEST_CLASS.TEST_METHOD', 'configured_class_and_method')
    trace.literal(klass, 'TEST_CLASS', 'configured_class')
    trace.literal(simple, 'TEST_CLASS', 'configured_class')
    trace.sub(re.compile(r'(?<=TEST_CLASS\.)' + re.escape(base) + r'(?!\w)'),
              'TEST_METHOD', 'configured_method_after_class')
    if len(method) >= 6:
        trace.literal(method, 'TEST_METHOD', 'configured_method_standalone')
    for name in sorted(names, key=len, reverse=True):
        trace.literal(name, 'PROJECT', 'repository_name', ignorecase=True)
    for key, value in kept.items():
        trace.text = trace.text.replace(key, value)
    return trace, protected_positions


def write_csv(path, rows):
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', default='data/processed/corpus.jsonl')
    parser.add_argument('--output', default='results/masking_coverage')
    args = parser.parse_args()
    corpus_path = (ROOT / args.corpus).resolve()
    destination = (ROOT / args.output).resolve()
    if not destination.is_relative_to(ROOT / 'results'):
        raise ValueError('Output must remain inside results')
    if destination.exists():
        raise ValueError('Use a new output directory; existing results are not overwritten')
    rows = sorted([json.loads(x) for x in corpus_path.read_text(encoding='utf-8').splitlines()],
                  key=lambda r: r['sample_id'])
    assert len({r['sample_id'] for r in rows}) == len(rows)
    ledger_files = sorted((ROOT / 'data/first_failures').glob('*.json'))
    ledger = {r['container']: r for p in ledger_files
              for r in [json.loads(p.read_text(encoding='utf-8'))]}
    document_rows, occurrences, examples = [], [], []
    stage_counts = collections.Counter()
    by_kind = collections.defaultdict(collections.Counter)
    metadata_sha = hashlib.sha256()
    for row in rows:
        meta = ledger[row['container']]['metadata']
        assert row['configured_test'] == meta['configured_test']
        metadata_sha.update(json.dumps({'sample_id': row['sample_id'], 'metadata': meta},
                                       sort_keys=True).encode())
        original, masked = row['original_text'], row['masked_text']
        assert hashlib.sha256(original.encode()).hexdigest() == row['original_sha256']
        assert hashlib.sha256(masked.encode()).hexdigest() == row['masked_sha256']
        trace, protected = traced_mask(original, meta, row['canonical_repository'],
                                       row['exception_type'])
        production = mask(original, meta, row['canonical_repository'], row['exception_type'])
        assert trace.text == production == masked, row['sample_id']
        changed_positions = set().union(*trace.positions.values()) if trace.positions else set()
        assert not (changed_positions & protected), row['sample_id']
        for exception in set(EXCEPTION.findall(original)) | (
                {row['exception_type']} if row['exception_type'] else set()):
            assert exception in masked
        tokens = list(LEXICAL.finditer(original))
        affected = sum(bool(changed_positions.intersection(range(*m.span()))) for m in tokens)
        fully = sum(set(range(*m.span())).issubset(changed_positions) for m in tokens)
        klass, method = row['configured_test'].split('#', 1)
        simple, base = klass.split('.')[-1], method.split('[', 1)[0]
        patterns = {
            'configured_class': bounded([klass, simple]),
            'configured_method_exact': bounded([method]),
            'configured_base_method': bounded([base]),
            'repository_name': bounded(names_for(meta, row['canonical_repository']), True),
        }
        residuals = collections.Counter()
        for kind, pattern in patterns.items():
            for match in pattern.finditer(original):
                positions = set(range(*match.span()))
                n_changed = len(positions & changed_positions)
                status = ('fully_replaced' if n_changed == len(positions) else
                          'partially_replaced' if n_changed else 'retained')
                protected_count = len(positions & protected)
                operations = '|'.join(sorted(k for k, value in trace.positions.items()
                                             if positions & value))
                by_kind[kind]['original_occurrences'] += 1
                by_kind[kind][status] += 1
                if protected_count:
                    by_kind[kind]['overlapping_protected_exception'] += 1
                reason = ''
                if status != 'fully_replaced':
                    if protected_count:
                        reason = 'preserved_exception'
                    elif kind.startswith('configured_') and 'method' in kind:
                        reason = ('parameterized_base_spelling_not_standalone_target' if method != base
                                  else 'short_standalone_method_spelling' if len(method) < 6
                                  else 'other_scoped_match')
                    elif status == 'partially_replaced':
                        reason = 'partial_overlap_with_' + operations
                occurrences.append({'sample_id': row['sample_id'], 'label': row['label'],
                    'kind': kind, 'value': match.group(), 'start': match.start(), 'end': match.end(),
                    'status': status, 'protected_characters': protected_count,
                    'replacement_operations': operations, 'retention_reason': reason})
                if status != 'fully_replaced':
                    residuals[kind] += 1
        stage_counts.update(trace.calls)
        document_rows.append({'sample_id': row['sample_id'], 'label': row['label'],
            'project_group': row['project_group'], 'text_changed': int(original != masked),
            'original_lexical_tokens': len(tokens), 'affected_lexical_tokens': affected,
            'fully_replaced_lexical_tokens': fully, 'retained_lexical_tokens': len(tokens) - affected,
            'affected_lexical_fraction': affected / len(tokens) if tokens else None,
            'configured_method_characters': len(method),
            'configured_base_method_characters': len(base),
            'parameterized_configured_method': int(method != base),
            'configured_method_standalone_rule_enabled': int(len(method) >= 6),
            'residual_configured_class_occurrences': residuals['configured_class'],
            'residual_configured_base_method_occurrences': residuals['configured_base_method'],
            'residual_repository_name_occurrences': residuals['repository_name']})
        if 'StompTest' in original and 'StompTest' in masked:
            examples.append({'sample_id': row['sample_id'], 'configured_test': row['configured_test'],
                'canonical_repository': row['canonical_repository'], 'revision': row['revision'],
                'original_frames': [s for s in original.splitlines() if 'StompTest' in s],
                'masked_frames': [s for s in masked.splitlines() if 'StompTest' in s],
                'explanation': 'StompTest is not the configured class. Other class names are outside '
                               'this metadata-scoped intervention; its presence is not evidence '
                               'of full anonymity. Inheritance requires separate source inspection.'})
    total = sum(r['original_lexical_tokens'] for r in document_rows)
    affected = sum(r['affected_lexical_tokens'] for r in document_rows)
    by_label = []
    for label in sorted({r['label'] for r in document_rows}):
        subset = [r for r in document_rows if r['label'] == label]
        original_count = sum(r['original_lexical_tokens'] for r in subset)
        affected_count = sum(r['affected_lexical_tokens'] for r in subset)
        by_label.append({'label': label, 'documents': len(subset),
            'changed_documents': sum(r['text_changed'] for r in subset),
            'original_lexical_tokens': original_count, 'affected_lexical_tokens': affected_count,
            'affected_lexical_fraction': affected_count / original_count})
    for counts in by_kind.values():
        for key in ('fully_replaced', 'partially_replaced', 'retained', 'overlapping_protected_exception'):
            counts.setdefault(key, 0)
        counts['fully_replaced_fraction'] = counts['fully_replaced'] / counts['original_occurrences']
    summary = {
        'schema_version': 1,
        'provenance': {'corpus': str(corpus_path.relative_to(ROOT)).replace('\\', '/'),
            'corpus_sha256': sha(corpus_path), 'script_sha256': sha(Path(__file__)),
            'masking_code_sha256': sha(ROOT / 'scripts/representations.py'),
            'used_metadata_sha256': metadata_sha.hexdigest(), 'python_version': platform.python_version()},
        'validation': {'saved_text_hashes_verified': True, 'all_traced_outputs_equal_production_and_saved': True,
            'protected_exception_characters_not_counted_as_replaced': True,
            'all_original_exception_names_preserved': True},
        'definitions': {
            'document_denominator': 'One saved diagnostic per retained canonical test in the given corpus.',
            'lexical_token': 'A non-overlapping regex match of [A-Za-z_$][\\w$]* in original_text '
                             '(case-sensitive, Python Unicode \\w). Includes code, prose and exception '
                             'names; not a Java parser or the fitted TF-IDF vocabulary.',
            'affected_lexical_token': 'An original lexical token with at least one original character '
                                      'consumed by a non-protection masking replacement. Counted once '
                                      'even if multiple operations overlap; generated placeholders are excluded.',
            'scoped_occurrences': 'Per document, longest-first non-overlapping bounded matches of '
                'the configured full or simple class; the exact configured method including parameters; '
                'the base method before [; or repository '
                'basenames plus non-generic hyphen/underscore parts of length >=4. Repository matching '
                'is case-insensitive. These are spelling matches, potentially including homonyms in '
                'prose, strings or packages, not resolved Java identifiers. Kinds can overlap; '
                'do not sum their denominators.',
            'fully_replaced': 'All original characters in that occurrence were consumed by masking; '
                              'this measures original occurrences, not uniqueness of spelling in output.',
            'retained': 'No original character in that occurrence was consumed by masking.',
            'scope_limit': 'Metadata-scoped intervention, not universal identifier anonymization. '
                'Other classes, packages, methods, file names, project-related words and protected '
                'exceptions may remain. Scoped counts cannot estimate recall over all project identifiers.'},
        'documents': len(rows), 'changed_documents': sum(r['text_changed'] for r in document_rows),
        'original_lexical_tokens': total, 'affected_lexical_tokens': affected,
        'fully_replaced_lexical_tokens': sum(r['fully_replaced_lexical_tokens'] for r in document_rows),
        'retained_lexical_tokens': total - affected, 'affected_lexical_fraction': affected / total,
        'scoped_occurrences_by_kind': dict(by_kind), 'replacement_operations': dict(stage_counts),
        'by_label': by_label,
        'documents_with_short_configured_method': sum(r['configured_method_characters'] < 6 for r in document_rows),
        'documents_with_parameterized_configured_method': sum(r['parameterized_configured_method'] for r in document_rows),
        'retained_or_partial_occurrence_reasons': dict(collections.Counter(
            r['retention_reason'] for r in occurrences if r['status'] != 'fully_replaced')),
        'documents_with_retained_scoped_occurrences': {
            k: sum(r[k] > 0 for r in document_rows) for k in (
                'residual_configured_class_occurrences', 'residual_configured_base_method_occurrences',
                'residual_repository_name_occurrences')},
        'stomp_residual_example_documents': len(examples),
    }
    destination.mkdir(parents=True)
    for filename, value in (('summary.json', summary), ('residual_examples.json', examples)):
        (destination / filename).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    write_csv(destination / 'per_document.csv', document_rows)
    write_csv(destination / 'scoped_identifier_occurrences.csv', occurrences)
    write_csv(destination / 'by_label.csv', by_label)
    report = f'''# Masking coverage audit

Run `python scripts/analyze_masking_coverage.py --output results/masking_coverage_NEW`
from the repository. Existing result directories are never overwritten. Standard
Python is sufficient; no fitted model or external data is required.

The audit validates every saved text hash and independently traces each production
masking substitution back to its original character positions. All {len(rows):,}
traced outputs equal both the production function and the saved masked text.

## Results and denominator

- Diagnostics changed: {summary['changed_documents']:,}/{len(rows):,}
  ({100 * summary['changed_documents'] / len(rows):.2f}%).
- Identifier-shaped original lexical tokens affected: {affected:,}/{total:,}
  ({100 * affected / total:.2f}%). A token is a non-overlapping
  `[A-Za-z_$][\\w$]*` match in the original diagnostic. Affected means at least
  one original character was consumed by a masking substitution. This includes
  prose, strings, packages and exceptions; it is not Java identifier resolution
  and is not the fitted TF-IDF vocabulary. New placeholder tokens are excluded.
- Configured full/simple-class string occurrences fully replaced:
  {by_kind['configured_class']['fully_replaced']:,}/{by_kind['configured_class']['original_occurrences']:,}.
- Repository-name string occurrences fully replaced:
  {by_kind['repository_name']['fully_replaced']:,}/{by_kind['repository_name']['original_occurrences']:,}.
  {by_kind['repository_name']['retained']} remain inside preserved exception names;
  {by_kind['repository_name']['partially_replaced']} are partially replaced.

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
'''
    (destination / 'README.md').write_text(report, encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
