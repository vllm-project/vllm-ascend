# How to use this script: in vllm-ascend directory
# python .github/workflows/scripts/coverage.py
import contextlib
import sys
from pathlib import Path

import regex as re
import yaml

with open(".github/workflows/scripts/test_config.yaml") as f:
    meta = yaml.safe_load(f) or {}


def pytest_node_file_path(path: str) -> str:
    return path.split("::", 1)[0]


_configured_paths = set()
for tests in (meta.get("curated_tests", {}) or {}).values():
    if not isinstance(tests, list):
        continue
    _configured_paths.update(t.rstrip("/") for t in tests if isinstance(t, str))
_configured_paths.update(t.rstrip("/") for t in (meta.get("skip_tests", []) or []) if isinstance(t, str))

_pins = [t.rstrip("/") for t in (meta.get("accuracy_tests", []) or []) if isinstance(t, str)]
_configured_paths.update(_pins)

# ============================================================
# 1. BROKEN PATHS — A non-existent path is referenced in yaml
# ============================================================
broken = sorted(p for p in _configured_paths if not Path(pytest_node_file_path(p)).exists())

# ============================================================
# 2. estimated_times coverage
# ============================================================
_et = dict(meta.get("estimated_times", {}) or {})
_rm = dict(meta.get("runner_mapping", {}) or {})
_part = dict(meta.get("partition", {}) or {})

# Build NPU UT regex patterns from runner_mapping
npu_ut_patterns = []
for pattern_str in _rm:
    with contextlib.suppress(re.error):
        npu_ut_patterns.append(re.compile(pattern_str))

# Expand all E2E / NPU UT test files in the repo (file-level)
e2e_files = {str(f) for f in Path("tests/e2e/pull_request").rglob("test_*.py")}
ut_files = {str(f) for f in Path("tests/ut").rglob("test_*.py")}
npu_ut_files = set()
cpu_ut_files = set()
for p in ut_files:
    if any(pat.search(p) for pat in npu_ut_patterns):
        npu_ut_files.add(p)
    else:
        cpu_ut_files.add(p)

# Need estimated_times: E2E + NPU UT (file-level)
need_et_files = e2e_files | npu_ut_files
existing_et_keys = set(_et.keys())
missing_et = sorted(need_et_files - existing_et_keys)
# CPU UT should NOT have estimated_times
cpu_ut_leaked = sorted(cpu_ut_files & existing_et_keys)

# ============================================================
# 3. Correctness of runner_mapping
# ============================================================
all_expanded_files = e2e_files | ut_files
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

rm_broken = len(rm_errors) > 0

# ============================================================
# 4. partition validity
# ============================================================
part_errors: list[str] = []
# Collect actual runner keys used in routing
actual_runner_keys: set[str] = set()
for p in all_expanded_files:
    for pat_str, rc in _rm.items():
        if re.compile(pat_str).search(p):
            for rk in rc.values():
                actual_runner_keys.add(rk)
            break

for key in sorted(actual_runner_keys - _part.keys()):
    part_errors.append(f"Referenced partition {key!r}: missing configuration")

for key, val in sorted(_part.items()):
    if not isinstance(val, dict):
        part_errors.append(f"Key {key!r}: configuration must be a mapping")
        continue
    if "-" not in key:
        part_errors.append(f"Key {key!r}: missing '-' separator")
        continue
    parts = key.rsplit("-", 1)
    if not parts[1].isdigit():
        part_errors.append(f"Key {key!r}: num_npus '{parts[1]}' is not a number")
        continue
    if key == "cpu-0":
        # CPU is the default fallback runner, always valid
        continue
    if key not in actual_runner_keys:
        part_errors.append(f"Key {key!r}: no tests route to this runner (unused)")

part_broken = len(part_errors) > 0

# ============================================================
# 5. E2E marker coverage (values enforced; unmarked still transitional)
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

print("\n[2] estimated_times coverage (file-level):")
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

print("\n[3] runner_mapping validation:")
if rm_errors:
    for err in rm_errors:
        print(f"    ✗ {err}")
else:
    print("    ✓ All patterns valid and match at least one test")

print("\n[4] partition and pinned route validation:")
if part_errors:
    for err in part_errors:
        print(f"    ✗ {err}")
else:
    print("    ✓ All partition keys valid and map to active runners")

print("\n[5] E2E marker coverage (values enforced; unmarked still transitional):")
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

has_errors = bool(
    broken
    or missing_et
    or cpu_ut_leaked
    or rm_errors
    or part_errors
    # Out-of-taxonomy marker values are a hard CI failure — values must come
    # from tests/e2e/coverage_taxonomy.py. Unmarked tests remain WARNING-only
    # during the migration period.
    or _marker_unknown_values
)
if has_errors:
    sys.exit(1)
