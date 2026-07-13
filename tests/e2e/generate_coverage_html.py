#!/usr/bin/env python3
"""Generate a self-contained interactive HTML coverage report by scanning
test files for ``@pytest.mark.e2e_model`` and ``@pytest.mark.e2e_coverage``
decorators via AST parsing.

Usage::

    python tests/e2e/generate_coverage_html.py              # write coverage.html
    python tests/e2e/generate_coverage_html.py --check-missing  # list unmarked tests
    python tests/e2e/generate_coverage_html.py -o /tmp/cov.html
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
E2E_PR_ROOT = REPO_ROOT / "tests" / "e2e" / "pull_request"
OUTPUT_DEFAULT = Path(__file__).resolve().parent / "coverage.html"

# Allow ``from tests.e2e.coverage_taxonomy import ...`` to resolve both when
# this script is run directly (``python tests/e2e/generate_coverage_html.py``)
# and when imported by CI (``from tests.e2e.generate_coverage_html import ...``).
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Taxonomy — single source of truth lives in coverage_taxonomy.py.
# Import here so the generator, the runtime conftest check, and the CI gate
# all share the exact same allowed values, labels, and dimension order.
# ---------------------------------------------------------------------------
from tests.e2e.coverage_taxonomy import (  # noqa: E402
    ALLOWED_VALUES,
    DIM_LABELS,
    DIM_ORDER,
    INVALID_COMBOS,
    validate_coverage_marker,
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class TestRecord:
    filepath: str  # relative path from E2E_PR_ROOT
    test_name: str  # function name
    card_count: int  # 1, 2, or 4
    models: list[str] = field(default_factory=list)
    coverage: dict[str, list[str]] = field(default_factory=dict)

    def has_coverage(self) -> bool:
        """Return True if at least one e2e_coverage marker was found."""
        return any(v for v in self.coverage.values())


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------
def _source_to_str(node: Any) -> str | None:
    """Extract a Python string constant from an AST node."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _extract_marker_args(call_node: ast.Call) -> list[str]:
    """Extract positional string args from a marker call like
    ``@pytest.mark.e2e_model("Qwen/Qwen3-0.6B")``.
    """
    values: list[str] = []
    for arg in call_node.args:
        s = _source_to_str(arg)
        if s is not None:
            values.append(s)
    return values


def _extract_marker_kwargs(call_node: ast.Call) -> dict[str, list[str]]:
    """Extract keyword args from a marker call like
    ``@pytest.mark.e2e_coverage(type="dense", feature="lora,mtp")``.

    Each keyword value is split on comma to support multi-value fields.
    """
    result: dict[str, list[str]] = {}
    for kw in call_node.keywords:
        if kw.arg is None:
            continue
        raw = _source_to_str(kw.value)
        if raw is None:
            continue
        # Split on comma, strip whitespace, filter empties
        vals = [v.strip() for v in raw.split(",") if v.strip()]
        result[kw.arg] = vals
    return result


def _detect_card_count(rel_path: str) -> int:
    """Infer NPU card count from directory path."""
    parts = rel_path.replace("\\", "/").split("/")
    for p in parts:
        if p == "one_card":
            return 1
        if p == "two_card":
            return 2
        if p == "four_card":
            return 4
    return 1


def _is_marker_call(dec: ast.expr, marker_name: str) -> ast.Call | None:
    """Check if *dec* is ``@pytest.mark.<marker_name>(...)`` or
    ``@pytest.mark.<marker_name>`` (no-arg form).
    """
    # Handle: @pytest.mark.e2e_model(...)
    if isinstance(dec, ast.Call):
        if isinstance(dec.func, ast.Attribute) and dec.func.attr == marker_name:
            # Check it's pytest.mark.xxx not some_other.mark.xxx
            if isinstance(dec.func.value, ast.Attribute) and dec.func.value.attr == "mark":
                if isinstance(dec.func.value.value, ast.Name) and dec.func.value.value.id == "pytest":
                    return dec
    return None


def _process_test_file(filepath: Path, root: Path | None = None) -> list[TestRecord]:
    """Parse one test file and return metadata for each test function."""
    if root is None:
        root = E2E_PR_ROOT
    try:
        source = filepath.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source)
    except SyntaxError:
        return []

    rel_path = str(filepath.relative_to(root))
    card_count = _detect_card_count(rel_path)

    results: list[TestRecord] = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.name.startswith("test_"):
            continue

        models: list[str] = []
        coverage: dict[str, list[str]] = {}

        for dec in node.decorator_list:
            # e2e_model marker
            if call := _is_marker_call(dec, "e2e_model"):
                models = _extract_marker_args(call)
                continue

            # e2e_coverage marker
            if call := _is_marker_call(dec, "e2e_coverage"):
                kw = _extract_marker_kwargs(call)
                for dim, vals in kw.items():
                    if dim in coverage:
                        coverage[dim].extend(vals)
                    else:
                        coverage[dim] = vals
                continue

        results.append(
            TestRecord(
                filepath=rel_path,
                test_name=node.name,
                card_count=card_count,
                models=models,
                coverage=coverage,
            )
        )

    return results


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _validate(records: list[TestRecord]) -> list[str]:
    """Check marker values against the taxonomy, returning warnings.

    Thin wrapper over :func:`coverage_taxonomy.validate_coverage_marker` so
    the CI gate (``.github/workflows/scripts/coverage.py``) can keep importing
    this name. The generator's ``TestRecord.coverage`` stores already-split
    token lists; here we re-join them into the raw comma-string form that
    ``validate_coverage_marker`` expects (the same form ``marker.kwargs``
    yields at pytest runtime).
    """
    warnings: list[str] = []
    for rec in records:
        if not rec.has_coverage():
            continue
        raw_coverage = {dim: ",".join(vals) for dim, vals in rec.coverage.items()}
        for w in validate_coverage_marker(raw_coverage):
            warnings.append(f"[{rec.filepath}::{rec.test_name}] {w}")
    return warnings


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
@dataclass
class Summary:
    total_tests: int = 0
    marked_tests: int = 0
    unmarked_tests: int = 0
    by_card: dict[int, int] = field(default_factory=dict)
    by_dim: dict[str, dict[str, int]] = field(default_factory=dict)
    # Number of tests that provide each dimension (denominator for coverage %)
    tests_per_dim: dict[str, int] = field(default_factory=dict)
    total_dim_unique: dict[str, int] = field(default_factory=dict)


def _compute_summary(records: list[TestRecord]) -> Summary:
    s = Summary()
    s.total_tests = len(records)
    s.marked_tests = sum(1 for r in records if r.has_coverage())
    s.unmarked_tests = s.total_tests - s.marked_tests

    for r in records:
        s.by_card[r.card_count] = s.by_card.get(r.card_count, 0) + 1

    for r in records:
        for dim, vals in r.coverage.items():
            if dim not in s.by_dim:
                s.by_dim[dim] = {}
            for v in vals:
                s.by_dim[dim][v] = s.by_dim[dim].get(v, 0) + 1

    # Count how many tests provide each dimension (denominator for coverage %).
    # A dimension is "provided" only if the test declared that key with at least
    # one non-empty value — empty feature="" counts as not provided.
    for r in records:
        provided = {dim for dim, vals in r.coverage.items() if vals}
        for dim in provided:
            s.tests_per_dim[dim] = s.tests_per_dim.get(dim, 0) + 1

    # Count unique values per dimension from taxonomy (for coverage %)
    for dim in ALLOWED_VALUES:
        s.total_dim_unique[dim] = len(ALLOWED_VALUES[dim])

    return s


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------
def _render_css() -> str:
    return """
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
           background: #f5f7fa; color: #1a1a2e; line-height: 1.6; }
    .container { max-width: 1600px; margin: 0 auto; padding: 20px; }
    h1 { font-size: 1.8rem; margin-bottom: 8px; color: #0f0f23; }
    h2 { font-size: 1.25rem; margin: 24px 0 12px; color: #0f0f23;
         border-bottom: 2px solid #e2e8f0; padding-bottom: 6px; }
    .subtitle { color: #666; margin-bottom: 24px; }

    /* Summary cards */
    .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                    gap: 12px; margin-bottom: 24px; }
    .summary-card { background: #fff; border-radius: 10px; padding: 16px 20px;
                    box-shadow: 0 1px 4px rgba(0,0,0,.08); text-align: center; }
    .summary-card .num { font-size: 2rem; font-weight: 700; color: #2563eb; }
    .summary-card .label { font-size: .8rem; color: #666; margin-top: 4px; }
    /* Progress card spans two columns */
    .summary-card.progress-card { grid-column: span 2; text-align: left; }
    .summary-card.progress-card .num { font-size: 1.6rem; }
    .summary-card.progress-card .label { margin-top: 2px; }
    .summary-card.progress-card .num span { font-size: 1rem; color: #888; }
    .progress-track { height: 14px; background: #e5e7eb; border-radius: 7px;
                      overflow: hidden; margin: 10px 0 4px; }
    .progress-fill { height: 100%; border-radius: 7px; background: #2563eb;
                     transition: width .4s; }
    .progress-fill.low { background: #ef4444; }
    .progress-fill.medium { background: #f59e0b; }
    .progress-sub { font-size: .78rem; color: #888; }

    /* N-dimensional cross-coverage explorer */
    .explorer-section { background: #fff; border-radius: 10px; padding: 16px 20px;
                        margin-bottom: 16px; box-shadow: 0 1px 4px rgba(0,0,0,.08); }
    .explorer-section h3 { font-size: 1rem; margin-bottom: 4px; color: #333; }
    .explorer-section .section-hint { font-size: .8rem; color: #888; margin-bottom: 12px; }
    .dim-checkboxes { display: flex; flex-wrap: wrap; gap: 8px 16px; margin-bottom: 12px; }
    .dim-checkboxes label { font-size: .85rem; display: flex; align-items: center;
                            gap: 4px; cursor: pointer; padding: 4px 10px;
                            border: 1px solid #d1d5db; border-radius: 6px; background: #fff; }
    .dim-checkboxes label:hover { background: #f1f5f9; }
    .dim-checkboxes input { margin: 0; }
    .dim-checkboxes label.checked { background: #dbeafe; border-color: #2563eb; color: #1e40af; font-weight: 600; }
    .explorer-controls { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin-bottom: 12px; }
    .explorer-controls button { padding: 5px 12px; border: 1px solid #2563eb; background: #2563eb;
            color: #fff; border-radius: 6px; cursor: pointer; font-size: .82rem; }
    .explorer-controls button:hover { background: #1d4ed8; }
    .explorer-controls button.ghost { background: #fff; color: #2563eb; }
    .explorer-controls button.ghost:hover { background: #eff6ff; }
    .explorer-controls label.zero-cov-toggle {
        font-size: .82rem; display: flex; align-items: center; gap: 4px;
        cursor: pointer; border: 1px solid #d1d5db; border-radius: 6px;
        padding: 4px 10px; background: #fff;
    }
    .explorer-summary { font-size: .82rem; color: #666; margin-left: auto; }
    .combo-table { width: 100%; border-collapse: collapse; font-size: .82rem; }
    .combo-table th { background: #f1f5f9; text-align: left; padding: 6px 10px;
                      font-weight: 600; white-space: nowrap; border-bottom: 1px solid #e2e8f0; }
    .combo-table td { padding: 4px 10px; border-bottom: 1px solid #f1f5f9; vertical-align: top; }
    .combo-table tr.zero td { color: #cbd5e1; background: #fafbfc; }
    .combo-table tr.zero td .combo-val { color: #cbd5e1; }
    .combo-table tr.zero td .combo-count { background: #fee2e2; color: #991b1b; }
    .combo-table tr:hover td { background: #f8fafc; }
    .combo-table tr.zero:hover td { background: #f5f5f5; }
    .combo-count { display: inline-block; min-width: 24px; text-align: center;
                   padding: 1px 7px; border-radius: 10px; font-weight: 600;
                   font-size: .78rem; background: #dbeafe; color: #1e40af; cursor: pointer; }
    .combo-val { font-family: 'SF Mono', Menlo, Consolas, monospace; font-size: .8rem; }
    .combo-tests { font-size: .72rem; color: #888; margin-top: 2px; }
    .combo-table-wrapper { max-height: 520px; overflow: auto; border: 1px solid #e5e7eb;
                           border-radius: 6px; }
    #explorer-pager { padding: 8px 10px; }
    #explorer-pager button { padding: 5px 12px; border: 1px solid #2563eb;
            background: #2563eb; color: #fff; border-radius: 6px; cursor: pointer; font-size: .82rem; }
    #explorer-pager button:hover { background: #1d4ed8; }

    /* Toolbar */
    .toolbar { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 16px; align-items: center; }
    .toolbar select, .toolbar input { padding: 6px 10px; border: 1px solid #d1d5db;
            border-radius: 6px; font-size: .85rem; background: #fff; }
    .toolbar input[type="text"] { flex: 1; min-width: 200px; }
    .toolbar button { padding: 6px 14px; border: 1px solid #2563eb; background: #2563eb;
            color: #fff; border-radius: 6px; cursor: pointer; font-size: .85rem; }
    .toolbar button:hover { background: #1d4ed8; }

    /* Table */
    .card-group { margin-bottom: 20px; }
    .card-group summary { font-size: 1.1rem; font-weight: 600; cursor: pointer;
            padding: 8px 12px; background: #e8ecf1; border-radius: 8px; }
    .card-group summary:hover { background: #dde2e9; }
    .table-wrapper { overflow-x: auto; margin-top: 8px; }
    table { width: 100%; border-collapse: collapse; font-size: .82rem; }
    th { background: #f1f5f9; text-align: left; padding: 8px 10px; font-weight: 600;
         white-space: nowrap; cursor: pointer; user-select: none; position: sticky; top: 0; }
    th:hover { background: #e2e8f0; }
    td { padding: 6px 10px; border-bottom: 1px solid #f1f5f9; vertical-align: top; }
    tr:hover td { background: #f8fafc; }
    tr.unmarked-row td { color: #9ca3af; font-style: italic; }
    tr.unmarked-row:hover td { background: #f9fafb; }
    .tag { display: inline-block; padding: 1px 6px; margin: 1px 2px; border-radius: 4px;
           font-size: .75rem; font-weight: 500; }
    .tag-arch { background: #dbeafe; color: #1e40af; }
    .tag-feature { background: #fce7f3; color: #9d174d; }
    .tag-parallel { background: #d1fae5; color: #065f46; }
    .tag-deploy { background: #fef3c7; color: #92400e; }
    .tag-hardware { background: #ede9fe; color: #5b21b6; }
    .tag-quant { background: #fee2e2; color: #991b1b; }
    .tag-graph { background: #e0f2fe; color: #075985; }
    .tag-model { background: #f0fdf4; color: #166534; }
    .badge-card { display: inline-block; padding: 2px 8px; border-radius: 12px;
                  font-size: .7rem; font-weight: 700; }
    .badge-1 { background: #dbeafe; color: #1e40af; }
    .badge-2 { background: #fef3c7; color: #92400e; }
    .badge-4 { background: #fce7f3; color: #9d174d; }

    .hidden { display: none; }
    .no-results { text-align: center; padding: 40px; color: #999; }
    .count-info { font-size: .85rem; color: #666; padding: 8px 0; }
    .missing-warn { background: #fef2f2; border: 1px solid #fecaca; color: #991b1b;
                    border-radius: 8px; padding: 12px 16px; margin-bottom: 16px; font-size: .85rem; }
    .missing-warn summary { cursor: pointer; font-weight: 600; background: none; padding: 0; }
    .missing-warn ul { margin: 8px 0 0 20px; }
    """


# ---------------------------------------------------------------------------
# JavaScript snippets for the interactive coverage report.
#
# Split into module-level constants (plain strings, no f-string brace
# escaping) and assembled by ``_render_javascript``. The sentinel placeholders
# ``__DATA__`` / ``__DIM_ORDER__`` / ``__ALLOWED__`` / ``__DIM_LABELS__`` are
# substituted via ``str.replace`` (not ``str.format``) so that the many literal
# JS braces in object/function bodies need no escaping.
# ---------------------------------------------------------------------------
_JS_HEAD = """
    let DATA = __DATA__;
    const CARD_MAP = {1: '1-Card', 2: '2-Card', 4: '4-Card'};
    const TAG_CLASS = {
        arch: 'tag-arch', feature: 'tag-feature', parallel: 'tag-parallel',
        deploy: 'tag-deploy', hardware: 'tag-hardware',
        quantization: 'tag-quant', graph_mode: 'tag-graph'
    };
    const DIM_ORDER = __DIM_ORDER__;

    let currentSort = {col: null, asc: true};

    function buildTags(vals, cls) {
        if (!vals || !vals.length) return '';
        return vals.map(v => `<span class="tag ${cls}">${v}</span>`).join('');
    }

    function cardBadge(n) {
        return `<span class="badge-card badge-${n}">${CARD_MAP[n]}</span>`;
    }
"""

_JS_TABLE = """
    function renderTable(records) {
        const groups = {1: [], 2: [], 4: []};
        for (const r of records) {
            const c = r.card_count || 1;
            (groups[c] = groups[c] || []).push(r);
        }

        const container = document.getElementById('table-container');
        const countInfo = document.getElementById('count-info');
        countInfo.textContent = `Showing ${records.length} of ${DATA.length} tests`;

        if (records.length === 0) {
            container.innerHTML = '<div class="no-results">No matching tests</div>';
            return;
        }

        let html = '';
        for (const card of [1, 2, 4]) {
            const items = groups[card];
            if (!items || !items.length) continue;
            html += `<div class="card-group">
                <details open>
                    <summary>${CARD_MAP[card]} Tests (${items.length})</summary>
                    <div class="table-wrapper">
                        <table>
                            <thead><tr>
                                <th onclick="sortTable('filepath')">File</th>
                                <th onclick="sortTable('test_name')" style="min-width:220px">Test</th>
                                <th>Models</th>
                                <th>Arch</th>
                                <th>Features</th>
                                <th>Parallel</th>
                                <th>Deploy</th>
                                <th>HW</th>
                                <th>Quant</th>
                                <th>Graph</th>
                            </tr></thead>
                            <tbody>`;
            for (const r of items) {
                const cov = r.coverage || {};
                const isMarked = Object.values(cov).some(arr => arr && arr.length > 0);
                const rowCls = isMarked ? '' : ' class="unmarked-row"';
                html += `<tr${rowCls}>
                    <td style="white-space:nowrap;font-size:.78rem">${r.filepath}</td>
                    <td style="font-size:.8rem">${r.test_name}</td>
                    <td>${buildTags(r.models, 'tag-model')}</td>
                    <td>${buildTags(cov.arch, 'tag-arch')}</td>
                    <td>${buildTags(cov.feature, 'tag-feature')}</td>
                    <td>${buildTags(cov.parallel, 'tag-parallel')}</td>
                    <td>${buildTags(cov.deploy, 'tag-deploy')}</td>
                    <td>${buildTags(cov.hardware, 'tag-hardware')}</td>
                    <td>${buildTags(cov.quantization, 'tag-quant')}</td>
                    <td>${buildTags(cov.graph_mode, 'tag-graph')}</td>
                </tr>`;
            }
            html += '</tbody></table></div></details></div>';
        }
        container.innerHTML = html;
    }
"""

_JS_FILTER = """
    function filter() {
        const search = (document.getElementById('search-box').value || '').toLowerCase();
        const filterCard = document.getElementById('filter-card').value;
        const filterType = document.getElementById('filter-type').value;
        const filterGraph = document.getElementById('filter-graph').value;
        const showUnmarked = document.getElementById('show-unmarked').checked;

        let filtered = DATA;
        if (search) {
            filtered = filtered.filter(r =>
                r.filepath.toLowerCase().includes(search) ||
                r.test_name.toLowerCase().includes(search) ||
                (r.models || []).some(m => m.toLowerCase().includes(search)) ||
                Object.values(r.coverage || {}).flat().some(v => v.toLowerCase().includes(search))
            );
        }
        if (filterCard) {
            const c = parseInt(filterCard);
            filtered = filtered.filter(r => r.card_count === c);
        }
        if (filterType) {
            filtered = filtered.filter(r =>
                (r.coverage || {}).arch && r.coverage.arch.includes(filterType)
            );
        }
        if (filterGraph) {
            filtered = filtered.filter(r =>
                (r.coverage || {}).graph_mode && r.coverage.graph_mode.includes(filterGraph)
            );
        }
        if (!showUnmarked) {
            filtered = filtered.filter(r => {
                const cov = r.coverage || {};
                return Object.values(cov).some(arr => arr && arr.length > 0);
            });
        }
        // Combination filter (set by clicking an explorer combo count)
        if (window._hmFilter) {
            filtered = filtered.filter(r => {
                const cov = r.coverage || {};
                for (const [dim, val] of Object.entries(window._hmFilter)) {
                    const arr = cov[dim];
                    if (!arr || !arr.includes(val)) return false;
                }
                return true;
            });
        }
        renderTable(filtered);
        // Show the active combination filter with a clear link, or fall back
        // to the explorer's coverage summary.
        const statusEl = document.getElementById('explorer-summary');
        if (window._hmFilter) {
            const parts = Object.entries(window._hmFilter).map(([d,v]) => `${d}=${v}`);
            statusEl.innerHTML = `Filtering: ${parts.join(' AND ')} ` +
                `<a href="#" onclick="window._hmFilter=null;filter();return false;" style="color:#2563eb">✕ clear</a>`;
        } else {
            renderExplorerSummary();
        }
    }
"""

_JS_CORE = """
    function sortTable(col) {
        if (currentSort.col === col) {
            currentSort.asc = !currentSort.asc;
        } else {
            currentSort.col = col;
            currentSort.asc = true;
        }
        const dir = currentSort.asc ? 1 : -1;
        DATA = [...DATA].sort((a, b) => {
            const va = a[col] || '';
            const vb = b[col] || '';
            return va.localeCompare(vb) * dir;
        });
        filter();
    }

    function exportCSV() {
        let csv = 'File,Test,Card,Models,Arch,Feature,Parallel,Deploy,Hardware,Quantization,GraphMode\\n';
        for (const r of DATA) {
            const cov = r.coverage || {};
            csv += [
                r.filepath, r.test_name, r.card_count,
                (r.models || []).join(';'),
                (cov.arch || []).join(';'),
                (cov.feature || []).join(';'),
                (cov.parallel || []).join(';'),
                (cov.deploy || []).join(';'),
                (cov.hardware || []).join(';'),
                (cov.quantization || []).join(';'),
                (cov.graph_mode || []).join(';'),
            ].join(',') + '\\n';
        }
        const blob = new Blob([csv], {type: 'text/csv'});
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'e2e_coverage.csv';
        a.click();
        URL.revokeObjectURL(url);
    }

    // Shared taxonomy (injected by _render_javascript) + coverage-set helper
    // used by the N-dim explorer.
    const ALLOWED = __ALLOWED__;
    const DIM_LABELS_JS = __DIM_LABELS__;
    const INVALID = __INVALID__;

    function getCoverageSet(r, dim) {
        const cov = r.coverage || {};
        const vals = cov[dim];
        return (vals && vals.length) ? new Set(vals) : new Set();
    }
"""

_JS_EXPLORER = """
    // Pagination state for the cross-coverage table. Covered combos can be
    // numerous for wide selections, so rows are rendered a page at a time.
    let explorerPage = 0;
    const EXPLORER_PAGE_SIZE = 50;
    // When true the explorer enumerates NOT-covered (zero-coverage) combos
    // instead of covered ones. Zero-coverage combos are produced on demand by
    // scanning the taxonomy Cartesian product (mixed-radix) and skipping
    // covered + invalid combos — never materialized in full.
    let explorerShowZero = false;
    const ZERO_SCAN_CAP = 200000;

    // Build the full value list for a dim. Every dimension enumerates from
    // the taxonomy (ALLOWED) — multi-value dims like feature are enumerated
    // by their single constituent values (lora, mtp, ...), and a test that
    // declares feature="lora,mtp" matches BOTH the "lora" and "mtp" cells.
    // This keeps the explorer consistent with the per-dimension bars and the
    // 2-D heatmap, which all split multi-value markers into single tokens.
    function explorerValuesFor(dim) {
        return ALLOWED[dim] || [];
    }

    function selectedExplorerDims() {
        return [...document.querySelectorAll('.dim-checkboxes input:checked')]
            .map(cb => cb.value);
    }

    // Cartesian product helper
    function cartesian(arrs) {
        let out = [[]];
        for (const arr of arrs) {
            const next = [];
            for (const combo of out) {
                for (const v of arr) next.push([...combo, v]);
            }
            out = next;
        }
        return out;
    }

    // True iff the combo (array aligned with dims) is excluded by an
    // INVALID_COMBOS rule. A rule is only evaluated when every dim it names is
    // currently in dims — otherwise the rule is skipped (it can't be decided
    // for a partial view and would over-prune).
    function isInvalidCombo(combo, dims) {
        const map = {};
        for (let j = 0; j < dims.length; j++) map[dims[j]] = combo[j];
        return INVALID.some(rule => {
            for (const d in rule) if (!(d in map)) return false;
            for (const d in rule) if (map[d] !== rule[d]) return false;
            return true;
        });
    }

    // Mixed-radix decode: index i -> combo. dims[0] is the least-significant
    // digit. Lets the zero-coverage scan jump to any offset without
    // materializing prior combos.
    function decodeCombo(idx, dims) {
        const combo = [];
        let rem = idx;
        for (let j = 0; j < dims.length; j++) {
            const list = ALLOWED[dims[j]] || [];
            combo[j] = list[rem % list.length];
            rem = Math.floor(rem / list.length);
        }
        return combo;
    }

    function totalCombos(dims) {
        let t = 1;
        for (const d of dims) t *= (ALLOWED[d] || []).length;
        return t;
    }

    // Scan the Cartesian product starting at startScanIdx, collecting combos
    // that are neither covered nor invalid, up to pageSize rows. Returns the
    // rows, the next scan offset, and exhaustion/scan-cap flags. Bounded by
    // ZERO_SCAN_CAP so a page never scans unboundedly even when invalid rules
    // prune most of the space.
    function collectZeroPage(dims, coveredSet, startScanIdx, pageSize) {
        const total = totalCombos(dims);
        const rows = [];
        let i = startScanIdx, scanned = 0;
        while (i < total && rows.length < pageSize && scanned < ZERO_SCAN_CAP) {
            const combo = decodeCombo(i, dims);
            const key = combo.join(String.fromCharCode(1));
            if (!coveredSet.has(key) && !isInvalidCombo(combo, dims)) rows.push(combo);
            i++; scanned++;
        }
        return {
            rows, nextScanIdx: i, exhausted: i >= total,
            scanCapped: scanned >= ZERO_SCAN_CAP && rows.length < pageSize,
        };
    }

    // Compute only the combinations that are actually covered by at least
    // one test. This iterates DATA once and, per test, takes the Cartesian
    // product of THAT test's declared value-sets (intersected with the
    // taxonomy) — bounded by the tests' small per-dim value counts, never by
    // the taxonomy's full product (which is 846,720 for all 7 dims). Invalid
    // combos are skipped so a test that only covers an invalid combo does not
    // inflate coveredCount. Zero-coverage combos are not materialized here.
    function computeCoveredCombos(dims) {
        const map = new Map();   // key -> { combo, count, tests }
        const coveredSet = new Set();
        let totalCount = 1;
        for (const d of dims) totalCount *= (ALLOWED[d] || []).length;
        for (const r of DATA) {
            const sets = dims.map(d => {
                const vals = (r.coverage || {})[d];
                const allowed = ALLOWED[d] || [];
                return (vals && vals.length) ? vals.filter(v => allowed.includes(v)) : [];
            });
            if (sets.some(s => s.length === 0)) continue;
            let combos = [[]];
            for (const s of sets) {
                const next = [];
                for (const c of combos) for (const v of s) next.push([...c, v]);
                combos = next;
            }
            for (const combo of combos) {
                if (isInvalidCombo(combo, dims)) continue;
                // U+0001 delimiter — taxonomy tokens never contain it, so no
                // collision between e.g. ("a|b","c") and ("a","b|c").
                const key = combo.join(String.fromCharCode(1));
                let entry = map.get(key);
                if (!entry) { entry = { combo, count: 0, tests: [] }; map.set(key, entry); }
                entry.count++;
                entry.tests.push(r);
                coveredSet.add(key);
            }
        }
        const covered = [...map.values()];
        const coveredCount = covered.length;
        return { covered, coveredCount, coveredSet, totalCount };
    }

    function renderExplorerSummary() {
        // Show the covered-combo count for the currently selected dims.
        // Deliberately does NOT report a total/zero-coverage count: computing
        // the valid total requires enumerating the taxonomy Cartesian product
        // (846,720 for all 7 dims) minus invalid combos — too expensive. The
        // concrete zero-coverage list is produced on demand via the
        // "Show zero-coverage" toggle, which paginates that scan.
        const dims = selectedExplorerDims();
        const summary = document.getElementById('explorer-summary');
        if (dims.length < 2) { summary.textContent = ''; return; }
        const { coveredCount } = computeCoveredCombos(dims);
        summary.innerHTML = `<b>${coveredCount}</b> covered combinations` +
            (coveredCount === 0 ? ' — none for this selection' : '');
    }

    function renderExplorer() {
        const dims = selectedExplorerDims();
        const container = document.getElementById('explorer-table');
        const summary = document.getElementById('explorer-summary');

        // Reset pagination on every re-render (toggle / preset / clear /
        // zero-coverage switch).
        explorerPage = 0;

        if (dims.length < 2) {
            container.innerHTML = '<div style="color:#999;padding:12px">'
                + 'Select at least 2 dimensions to compute cross-coverage.</div>';
            summary.textContent = '';
            return;
        }

        const { covered, coveredCount, coveredSet, totalCount } = computeCoveredCombos(dims);

        // Only overwrite the summary line when no combination filter is active
        // — otherwise filter() owns it to show the "✕ clear" link.
        if (!window._hmFilter) {
            summary.innerHTML = `<b>${coveredCount}</b> covered combinations` +
                (coveredCount === 0 ? ' — none for this selection' : '');
        }

        // Sort covered combos: count desc, then by combo values (asc). Only
        // covered combos are sorted/rows — zero-coverage is enumerated on demand.
        covered.sort((a, b) => {
            if (b.count !== a.count) return b.count - a.count;
            return a.combo.join('|').localeCompare(b.combo.join('|'));
        });

        // Shared state for the pagination renderers. zeroScanIdx is the
        // cursor for the zero-coverage scan (Load more resumes from it).
        window._explorerState = { covered, coveredSet, dims, totalCount, coveredCount, zeroScanIdx: 0 };
        container.innerHTML = '';
        renderExplorerPage();
    }

    // Render one page of the cross-coverage table. On page 0 it builds the
    // table skeleton (a single '#' column — see bug fix below); subsequent
    // pages just append rows via renderExplorerRows. In zero-coverage mode it
    // scans the taxonomy product on demand for not-covered combos.
    function renderExplorerPage() {
        const state = window._explorerState;
        if (!state) return;
        const { dims } = state;
        const container = document.getElementById('explorer-table');

        if (explorerPage === 0) {
            // Header: ONE '#' column total (previously one per dim, all
            // showing the identical per-combo count — pure redundancy).
            let html = '<table class="combo-table"><thead><tr>';
            for (const d of dims) html += `<th>${DIM_LABELS_JS[d]}</th>`;
            html += '<th>#</th>';
            html += '<th style="min-width:200px">Tests</th></tr></thead><tbody></tbody></table>';
            html += '<div id="explorer-pager"></div>';
            container.innerHTML = html;
        }

        if (explorerShowZero) {
            const { coveredSet, dims: d2 } = state;
            const scanFrom = explorerPage === 0 ? 0 : state.zeroScanIdx;
            const { rows, nextScanIdx, exhausted, scanCapped } =
                collectZeroPage(d2, coveredSet, scanFrom, EXPLORER_PAGE_SIZE);
            state.zeroScanIdx = nextScanIdx;
            renderZeroRows(rows, d2);
            renderZeroPager(rows.length, exhausted, scanCapped, nextScanIdx, state.totalCount);
        } else {
            const start = explorerPage * EXPLORER_PAGE_SIZE;
            const end = Math.min(start + EXPLORER_PAGE_SIZE, state.covered.length);
            renderExplorerRows(start, end);
            renderExplorerPager(end);
        }
    }

    // Append rows [start, end) of the covered-combos table.
    function renderExplorerRows(start, end) {
        const state = window._explorerState;
        if (!state) return;
        const { covered, dims } = state;
        const tbody = document.querySelector('#explorer-table tbody');
        if (!tbody) return;
        let html = '';
        for (let i = start; i < end; i++) {
            const { combo, count, tests } = covered[i];
            const comboJson = JSON.stringify(combo).replace(/'/g, "&#39;");
            const dimsJson = JSON.stringify(dims);
            html += `<tr data-combo='${comboJson}' data-dims='${dimsJson}'>`;
            for (let j = 0; j < dims.length; j++) {
                const val = combo[j] || '<span style="color:#cbd5e1">(empty)</span>';
                html += `<td><span class="combo-val">${val}</span></td>`;
            }
            // ONE count cell per row (after all dim columns).
            html += `<td><span class="combo-count">${count}</span></td>`;
            const testStr = tests.slice(0, 5).map(t => t.test_name).join(', ') +
                (tests.length > 5 ? ` … (+${tests.length - 5})` : '');
            html += `<td><div class="combo-tests">${testStr || '—'}</div></td></tr>`;
        }
        if (start === 0) tbody.innerHTML = html;
        else tbody.insertAdjacentHTML('beforeend', html);
    }

    // Append zero-coverage rows. These are display-only (no data-combo, so the
    // delegated click handler ignores them) and rendered greyed via tr.zero.
    function renderZeroRows(rows, dims) {
        const tbody = document.querySelector('#explorer-table tbody');
        if (!tbody) return;
        let html = '';
        for (const combo of rows) {
            html += `<tr class="zero">`;
            for (let j = 0; j < dims.length; j++) {
                const val = combo[j] || '<span style="color:#cbd5e1">(empty)</span>';
                html += `<td><span class="combo-val">${val}</span></td>`;
            }
            html += `<td><span class="combo-count">0</span></td>`;
            html += `<td><div class="combo-tests">—</div></td></tr>`;
        }
        if (window._explorerState && explorerPage === 0) tbody.innerHTML = html;
        else tbody.insertAdjacentHTML('beforeend', html);
    }

    // Render/refresh the "Load more" pager below the table (covered mode).
    function renderExplorerPager(renderedCount) {
        const state = window._explorerState;
        if (!state) return;
        const total = state.covered.length;
        const pager = document.getElementById('explorer-pager');
        if (!pager) return;
        if (renderedCount >= total) {
            pager.innerHTML = `<span style="color:#999;font-size:.78rem;padding:6px 10px">` +
                `Showing all ${total} covered combinations</span>`;
            return;
        }
        pager.innerHTML = `<button class="ghost" id="explorer-load-more" style="margin:8px 0">` +
            `Load more (showing ${renderedCount} of ${total})</button>`;
        document.getElementById('explorer-load-more').addEventListener('click', () => {
            explorerPage++;
            const start = explorerPage * EXPLORER_PAGE_SIZE;
            const end = Math.min(start + EXPLORER_PAGE_SIZE, state.covered.length);
            renderExplorerRows(start, end);
            renderExplorerPager(end);
        });
    }

    // Pager for zero-coverage mode. Shows progress across the taxonomy
    // product (scanned / total) and a Load-more that resumes the scan.
    function renderZeroPager(rowsLen, exhausted, scanCapped, scannedIdx, totalCombos) {
        const pager = document.getElementById('explorer-pager');
        if (!pager) return;
        if (exhausted) {
            pager.innerHTML = `<span style="color:#999;font-size:.78rem;padding:6px 10px">` +
                `End of zero-coverage combinations (scanned all ${totalCombos})</span>`;
            return;
        }
        if (scanCapped) {
            pager.innerHTML = `<span style="color:#b45309;font-size:.78rem;padding:6px 10px">` +
                `Scan limit (${ZERO_SCAN_CAP}) reached this page — narrow the selection ` +
                `or add INVALID_COMBOS rules to prune more.</span>`;
            return;
        }
        pager.innerHTML = `<button class="ghost" id="explorer-load-more" style="margin:8px 0">` +
            `Load more zero-coverage (scanned ${scannedIdx} / ${totalCombos})</button>`;
        document.getElementById('explorer-load-more').addEventListener('click', () => {
            explorerPage++;
            renderExplorerPage();
        });
    }

    function toggleExplorerDim(cb) {
        cb.closest('label').classList.toggle('checked', cb.checked);
        renderExplorer();
    }

    function selectExplorerPreset(name) {
        const presets = {
            arch_quant_graph: ['arch', 'quantization', 'graph_mode'],
            arch_feature_hw: ['arch', 'feature', 'hardware'],
            feature_par_graph: ['feature', 'parallel', 'graph_mode'],
            all: DIM_ORDER,
        };
        const target = presets[name] || [];
        document.querySelectorAll('.dim-checkboxes input').forEach(cb => {
            cb.checked = target.includes(cb.value);
            cb.closest('label').classList.toggle('checked', cb.checked);
        });
        renderExplorer();
    }

    // Toggle between covered and zero-coverage enumeration. renderExplorer
    // resets the page/scan cursor, so flipping is always a clean restart.
    function toggleZeroCoverage() {
        explorerShowZero = document.getElementById('show-zero-coverage').checked;
        renderExplorer();
    }
"""

_JS_INIT = """
    document.addEventListener('DOMContentLoaded', () => {
        filter();
        document.getElementById('search-box').addEventListener('input', () => {
            window._hmFilter = null;
            filter();
        });
        document.getElementById('filter-card').addEventListener('change', filter);
        document.getElementById('filter-type').addEventListener('change', filter);
        document.getElementById('filter-graph').addEventListener('change', filter);
        document.getElementById('show-unmarked').addEventListener('change', filter);
        // N-dim explorer: default to arch x quantization x graph_mode
        ['arch', 'quantization', 'graph_mode'].forEach(d => {
            const cb = document.querySelector(`.dim-checkboxes input[value="${d}"]`);
            if (cb) { cb.checked = true; cb.closest('label').classList.add('checked'); }
        });
        renderExplorer();
        document.querySelectorAll('.dim-checkboxes input').forEach(cb =>
            cb.addEventListener('change', () => toggleExplorerDim(cb)));
        // Delegated click on any covered combo's count -> filter the test
        // matrix to that exact combo. Registered once here (not per render)
        // so it survives pagination row replacement. Zero-coverage rows carry
        // no data-combo (display-only) and are skipped.
        document.getElementById('explorer-table').addEventListener('click', (e) => {
            const el = e.target.closest('.combo-count');
            if (!el) return;
            const tr = el.closest('tr');
            if (!tr || !tr.dataset.combo) return;
            const combo = JSON.parse(tr.dataset.combo);
            const dimsList = JSON.parse(tr.dataset.dims);
            document.getElementById('search-box').value = '';
            window._hmFilter = {};
            for (let k = 0; k < dimsList.length; k++) {
                const v = combo[k];
                if (v) window._hmFilter[dimsList[k]] = v;
            }
            document.getElementById('show-unmarked').checked = false;
            filter();
            document.getElementById('table-container').scrollIntoView({behavior: 'smooth', block: 'start'});
        });
    });
"""


def _render_javascript(records_json: str) -> str:
    """Assemble the interactive coverage report's client-side JavaScript.

    The JS body is split across ``_JS_HEAD``/``_JS_TABLE``/``_JS_FILTER``/
    ``_JS_CORE``/``_JS_EXPLORER``/``_JS_INIT`` constants; the ``__*__``
    placeholders (data, dim order, allowed values, dim labels) are substituted
    here.
    """
    head = _JS_HEAD.replace("__DATA__", records_json).replace("__DIM_ORDER__", json.dumps(DIM_ORDER))
    core = (
        _JS_CORE.replace(
            "__ALLOWED__",
            json.dumps({k: sorted(v) for k, v in ALLOWED_VALUES.items()}),
        )
        .replace("__DIM_LABELS__", json.dumps(DIM_LABELS))
        .replace("__INVALID__", json.dumps(INVALID_COMBOS))
    )
    return head + _JS_TABLE + _JS_FILTER + core + _JS_EXPLORER + _JS_INIT


def _render_html(records: list[TestRecord], summary: Summary, warnings: list[str]) -> str:
    records_data: list[dict] = []
    for r in records:
        records_data.append(
            {
                "filepath": r.filepath,
                "test_name": r.test_name,
                "card_count": r.card_count,
                "models": r.models,
                "coverage": {k: v for k, v in r.coverage.items()},
            }
        )

    # Build filter options
    type_options = "".join(f'<option value="{v}">{v}</option>' for v in sorted(ALLOWED_VALUES.get("arch", [])))
    graph_options = "".join(f'<option value="{v}">{v}</option>' for v in sorted(ALLOWED_VALUES.get("graph_mode", [])))

    # Warnings section
    warning_html = ""
    if warnings:
        warning_html = '<div class="missing-warn"><details><summary>'
        warning_html += f"⚠ {len(warnings)} marker value warning(s)</summary><ul>"
        for w in warnings[:50]:
            warning_html += f"<li>{w}</li>"
        if len(warnings) > 50:
            warning_html += f"<li>... and {len(warnings) - 50} more</li>"
        warning_html += "</ul></details></div>"

    mark_pct = round(summary.marked_tests / max(summary.total_tests, 1) * 100)
    pct_class = "low" if mark_pct < 20 else ("medium" if mark_pct < 60 else "")

    # Build N-dim cross-coverage explorer section
    dim_cb_items = "".join(
        f'<label><input type="checkbox" value="{d}">{DIM_LABELS.get(d, d)}</label>' for d in DIM_ORDER
    )
    preset_specs = [
        ("arch_quant_graph", "arch × quant × graph"),
        ("arch_feature_hw", "arch × feature × hw"),
        ("feature_par_graph", "feature × parallel × graph"),
        ("all", "All dimensions"),
    ]
    preset_buttons = "".join(
        f'<button class="ghost" onclick="selectExplorerPreset(\'{name}\')">{label}</button>'
        for name, label in preset_specs
    )
    explorer_html = f"""<div class="explorer-section">
        <h3>🧩 Cross-Coverage Explorer</h3>
        <p class="section-hint">Pick any N dimensions to enumerate the full cross product and
        see which combinations are covered. Click a count to filter the test matrix.
        Multi-value dimensions (feature, parallel) count a test once per distinct
        value-set it declares. Toggle <b>Show zero-coverage</b> to page through the
        not-covered combinations (semantically invalid combos, declared in
        <code>coverage_taxonomy.py</code>, are excluded from both views).</p>
        <div class="dim-checkboxes">{dim_cb_items}</div>
        <div class="explorer-controls">
            {preset_buttons}
            <label class="zero-cov-toggle">
                <input type="checkbox" id="show-zero-coverage" onchange="toggleZeroCoverage()"> Show zero-coverage
            </label>
            <span class="explorer-summary" id="explorer-summary"></span>
        </div>
        <div class="combo-table-wrapper"><div id="explorer-table"></div></div>
    </div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>E2E Test Coverage Report</title>
<style>{_render_css()}</style>
</head>
<body>
<div class="container">

<h1>🔬 E2E Test Coverage Report</h1>
<p class="subtitle">Generated from <code>tests/e2e/pull_request/</code> —
{summary.total_tests} test functions across 1-card, 2-card, and 4-card configurations.</p>

<div class="summary-grid">
    <div class="summary-card"><div class="num">{summary.total_tests}</div><div class="label">Total Tests</div></div>
    <div class="summary-card progress-card">
        <div class="num">{summary.marked_tests} <span>/ {summary.total_tests}</span></div>
        <div class="label">Marked Tests ({mark_pct}%)</div>
        <div class="progress-track"><div class="progress-fill {pct_class}" style="width:{mark_pct}%"></div></div>
        <div class="progress-sub">{summary.unmarked_tests} tests still unmarked · target 100%</div>
    </div>
    <div class="summary-card"><div class="num">{summary.by_card.get(1, 0)}</div>
    <div class="label">1-Card Tests</div></div>
    <div class="summary-card"><div class="num">{summary.by_card.get(2, 0)}</div>
    <div class="label">2-Card Tests</div></div>
    <div class="summary-card"><div class="num">{summary.by_card.get(4, 0)}</div>
    <div class="label">4-Card Tests</div></div>
</div>

{warning_html}

<h2>🧩 Cross-Coverage Explorer</h2>
{explorer_html}

<h2>Test Matrix</h2>

<div class="toolbar">
    <input type="text" id="search-box" placeholder="🔍 Search test name, model, or any tag...">
    <select id="filter-card">
        <option value="">All Cards</option>
        <option value="1">1-Card</option>
        <option value="2">2-Card</option>
        <option value="4">4-Card</option>
    </select>
    <select id="filter-type">
        <option value="">All Architectures</option>
        {type_options}
    </select>
    <select id="filter-graph">
        <option value="">All Graph Modes</option>
        {graph_options}
    </select>
    <label style="font-size:.85rem;display:flex;align-items:center;gap:4px;cursor:pointer">
        <input type="checkbox" id="show-unmarked"> Show unmarked
    </label>
    <button onclick="exportCSV()">📥 Export CSV</button>
</div>

<div id="count-info" class="count-info"></div>
<div id="table-container"></div>

</div>
<script>{_render_javascript(json.dumps(records_data))}</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate E2E coverage HTML report")
    parser.add_argument("-o", "--output", default=str(OUTPUT_DEFAULT), help="Output HTML file path")
    parser.add_argument("--check-missing", action="store_true", help="List tests without e2e markers")
    args = parser.parse_args()

    # Collect all test files
    test_files = sorted(E2E_PR_ROOT.rglob("test_*.py"))
    if not test_files:
        print("No test files found.", file=sys.stderr)
        sys.exit(1)

    # Parse
    all_records: list[TestRecord] = []
    for fp in test_files:
        all_records.extend(_process_test_file(fp))

    all_records.sort(key=lambda r: (r.filepath, r.test_name))

    # Validate
    warnings = _validate(all_records)

    # Summary
    summary = _compute_summary(all_records)

    # Check-missing mode
    if args.check_missing:
        unmarked = [r for r in all_records if not r.has_coverage()]
        if unmarked:
            print(f"{len(unmarked)} unmarked test(s) (out of {len(all_records)} total):\n")
            for r in unmarked:
                print(f"  {r.filepath}::{r.test_name}")
        else:
            print(f"All {len(all_records)} test(s) have e2e coverage markers. ✅")

        if warnings:
            print(f"\n{len(warnings)} marker value warning(s):\n")
            for w in warnings:
                print(f"  {w}")

        return

    # Generate HTML
    html = _render_html(all_records, summary, warnings)
    out_path = Path(args.output)
    out_path.write_text(html, encoding="utf-8")
    print(f"✅ Wrote {len(all_records)} test records to {out_path}")
    marked = summary.marked_tests
    total = summary.total_tests
    print(f"   Marked: {marked}/{total} ({round(marked / max(total, 1) * 100)}%)")
    if warnings:
        print(f"   ⚠ {len(warnings)} marker value warning(s)")
    if summary.unmarked_tests:
        print(f"   💡 {summary.unmarked_tests} unmarked tests hidden (check 'Show unmarked' in HTML)")


if __name__ == "__main__":
    main()
