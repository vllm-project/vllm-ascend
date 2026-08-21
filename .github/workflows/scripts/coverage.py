# How to use this script: in vllm-ascend directory
# python .github/workflows/scripts/coverage.py
#
# Validates test_config.yaml against the full directory-scanned test universe
# (matching select_tests.py --run-all-modules):
#   - estimated_times / skip_tests must not reference missing paths
#   - every E2E and NPU-convention UT file must have estimated_times
#   - no CPU UT file may leak an estimated_times entry (CPU UTs run as a batch)
#   - runner_mapping and partition keys must be valid and used
#   - E2E coverage markers must use the defined taxonomy
import contextlib
import sys
from pathlib import Path

import regex as re
import yaml

with open(".github/workflows/scripts/test_config.yaml") as f:
    meta = yaml.safe_load(f) or {}

# The full-suite test universe is derived by directory scan (matching
# select_tests.py --run-all-modules), not from a module list.
FULL_UT_ROOT = Path("tests/ut")
FULL_E2E_ROOTS = (
    Path("tests/e2e/pull_request/one_card"),
    Path("tests/e2e/pull_request/two_card"),
    Path("tests/e2e/pull_request/four_card"),
    Path("tests/e2e/pull_request/eight_card"),
)


def pytest_node_file_path(path: str) -> str:
    return path.split("::", 1)[0]


def _collect_test_files() -> set[str]:
    """Directory-scan UT + E2E roots to get the full set of test_*.py files."""
    files: set[str] = set()
    for root in (FULL_UT_ROOT, *FULL_E2E_ROOTS):
        if root.is_dir():
            for f in root.rglob("test_*.py"):
                if "__pycache__" not in f.parts:
                    files.add(str(f))
    return files


all_expanded_files = _collect_test_files()

# ============================================================
# 1. BROKEN PATHS — estimated_times / skip_tests references a file that
#    does not exist
# ============================================================
_et = dict(meta.get("estimated_times", {}) or {})
_rm = dict(meta.get("runner_mapping", {}) or {})
_part = dict(meta.get("partition", {}) or {})
_skip = meta.get("skip_tests", []) or []

broken = sorted(p for origins in (_et.keys(), _skip) for p in origins if not Path(pytest_node_file_path(p)).exists())

# ============================================================
# 2. Test universe breakdown (UT vs E2E, CPU vs NPU)
# ============================================================
e2e_files = {p for p in all_expanded_files if "tests/e2e/" in p}
ut_files = {p for p in all_expanded_files if "tests/ut/" in p}

npu_ut_patterns = []
for pattern_str in _rm:
    with contextlib.suppress(re.error):
        npu_ut_patterns.append(re.compile(pattern_str))

npu_ut_files: set[str] = set()
cpu_ut_files: set[str] = set()
for p in ut_files:
    if any(pat.search(p) for pat in npu_ut_patterns):
        npu_ut_files.add(p)
    else:
        cpu_ut_files.add(p)

# ============================================================
# 3. estimated_times coverage
# ============================================================
need_et_files = e2e_files | npu_ut_files
existing_et_keys = set(_et.keys())
missing_et = sorted(need_et_files - existing_et_keys)
# CPU UT should NOT have estimated_times
cpu_ut_leaked = sorted(cpu_ut_files & existing_et_keys)

# ============================================================
# 4. Correctness of runner_mapping
# ============================================================
rm_errors: list[str] = []
for pattern_str, runner_config in sorted(_rm.items()):
    try:
        pat = re.compile(pattern_str)
    except re.error as e:
        rm_errors.append(f"Pattern {pattern_str!r}: invalid regex — {e}")
        continue
    if "default" not in runner_config:
        rm_errors.append(f"Pattern {pattern_str!r}: missing 'default' key")
        continue
    matched = [p for p in all_expanded_files if pat.search(p)]
    if not matched:
        rm_errors.append(f"Pattern {pattern_str!r}: matches 0 tests (unused)")

# ============================================================
# 5. partition validity
# ============================================================
actual_runner_keys: set[str] = set()
for p in all_expanded_files:
    for pat_str, rc in _rm.items():
        if re.compile(pat_str).search(p):
            actual_runner_keys.update(rc.values())
            break

part_errors: list[str] = []
for key in sorted(_part):
    if "_x" not in key:
        part_errors.append(f"Key {key!r}: missing '_x' separator")
        continue
    parts = key.rsplit("_x", 1)
    if not parts[1].isdigit():
        part_errors.append(f"Key {key!r}: num_npus '{parts[1]}' is not a number")
        continue
    if key == "cpu_x0":
        # CPU is the default fallback runner, always valid
        continue
    if key not in actual_runner_keys:
        part_errors.append(f"Key {key!r}: no tests route to this runner (unused)")

# ============================================================
# 6. E2E marker coverage (values enforced; unmarked still transitional)
# ============================================================
_marker_unmarked: list[str] = []
_marker_unknown_values: list[str] = []
try:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from tests.e2e.generate_coverage_html import (  # type: ignore[import-not-found]
        E2E_PR_ROOT as _E2E_PR_ROOT,
    )
    from tests.e2e.generate_coverage_html import (
        _process_test_file,
        _validate,
    )

    for fp in sorted(_E2E_PR_ROOT.rglob("test_*.py")):
        records = _process_test_file(fp, root=_E2E_PR_ROOT)
        for r in records:
            if not r.has_coverage():
                _marker_unmarked.append(f"{r.filepath}::{r.test_name}")
        # Accumulate across all files (do NOT reassign — that would discard
        # every file's warnings except the last one).
        _marker_unknown_values.extend(_validate(records))
except Exception:
    _marker_unmarked = []
    _marker_unknown_values = []

# ============================================================
# REPORT
# ============================================================
print("=" * 70)
print("REVIEW RESULT")
print("=" * 70)

print(f"\n[1] BROKEN PATHS in yaml (referenced but don't exist): {len(broken)}")
if broken:
    for p in broken:
        print(f"    ✗ {p}")
else:
    print("    ✓ None — all referenced paths exist")

print(
    f"\n[2] Test universe: {len(all_expanded_files)} files "
    f"(E2E: {len(e2e_files)}, UT-CPU: {len(cpu_ut_files)}, UT-NPU: {len(npu_ut_files)})"
)

print("\n[3] estimated_times coverage (file-level):")
print(f"    E2E: {len([p for p in e2e_files if p in existing_et_keys])}/{len(e2e_files)} covered")
print(f"    NPU UT: {len([p for p in npu_ut_files if p in existing_et_keys])}/{len(npu_ut_files)} covered")
print(f"    CPU UT (should be 0): {len(cpu_ut_leaked)} leaked")
if missing_et:
    for p in missing_et:
        print(f"    ✗ MISSING: {p}")
else:
    print("    ✓ All E2E + NPU UT tests have estimated_times")
if cpu_ut_leaked:
    for p in cpu_ut_leaked:
        print(f"    ✗ LEAKED (CPU UT should not have et): {p}")
else:
    print("    ✓ No CPU UT entries in estimated_times")

print("\n[4] runner_mapping validation:")
if rm_errors:
    for err in rm_errors:
        print(f"    ✗ {err}")
else:
    print("    ✓ All patterns valid and match at least one test")

print("\n[5] partition validation:")
if part_errors:
    for err in part_errors:
        print(f"    ✗ {err}")
else:
    print("    ✓ All partition keys valid and map to active runners")

print("\n[6] E2E marker coverage (values enforced; unmarked still transitional):")
if _marker_unmarked:
    print(f"    ⚠ {len(_marker_unmarked)} test(s) without e2e_coverage marker:")
    for p in _marker_unmarked[:20]:
        print(f"      - {p}")
    if len(_marker_unmarked) > 20:
        print(f"      ... and {len(_marker_unmarked) - 20} more")
else:
    print("    ✓ All tests have e2e_coverage markers")
if _marker_unknown_values:
    print(f"    ✗ {len(_marker_unknown_values)} unknown marker value(s) — failing:")
    for w in _marker_unknown_values[:10]:
        print(f"      - {w}")
    if len(_marker_unknown_values) > 10:
        print(f"      ... and {len(_marker_unknown_values) - 10} more")
else:
    print("    ✓ All marker values are within the taxonomy")

print("\n" + "=" * 70)

has_errors = bool(broken or missing_et or cpu_ut_leaked or rm_errors or part_errors or _marker_unknown_values)
if has_errors:
    sys.exit(1)
