# How to use this script: in vllm-ascend directory
# python .github/workflows/scripts/coverage.py
import ast
import contextlib
import sys
from pathlib import Path

import regex as re
import yaml

with open(".github/workflows/scripts/test_config.yaml") as f:
    docs = list(yaml.safe_load_all(f))
    config = docs[0]
    meta = docs[1] if len(docs) >= 2 else {}


def pytest_node_file_path(path: str) -> str:
    return path.split("::", 1)[0]


all_yaml_paths = set()
all_yaml_e2e_paths = set()
all_yaml_ut_paths = set()

for module in config:
    for t in module.get("tests", []):
        t = t.rstrip("/")
        all_yaml_paths.add(t)
        if "tests/e2e/" in t:
            all_yaml_e2e_paths.add(t)
        if "tests/ut/" in t:
            all_yaml_ut_paths.add(t)

# ============================================================
# 1. BROKEN PATHS — A non-existent path is referenced in yaml
# ============================================================
broken = sorted(p for p in all_yaml_paths if not Path(pytest_node_file_path(p)).exists())

# ============================================================
# 2. E2E pull_request coverage
# ============================================================
yaml_e2e_pr = {p for p in all_yaml_e2e_paths if "tests/e2e/pull_request/" in p}
resolved_e2e = set()
for p in yaml_e2e_pr:
    path = Path(pytest_node_file_path(p))
    if path.is_file():
        resolved_e2e.add(str(path))
    elif path.is_dir():
        for f in path.rglob("test_*.py"):
            resolved_e2e.add(str(f))
actual_e2e = {str(f) for f in Path("tests/e2e/pull_request").rglob("test_*.py")}
uncovered_e2e = sorted(actual_e2e - resolved_e2e)

# ============================================================
# 3. UT coverage
# ============================================================
resolved_ut = set()
for p in all_yaml_ut_paths:
    path = Path(pytest_node_file_path(p))
    if path.is_file():
        resolved_ut.add(str(path))
    elif path.is_dir():
        for f in path.rglob("test_*.py"):
            resolved_ut.add(str(f))
actual_ut = {str(f) for f in Path("tests/ut").rglob("test_*.py")}
uncovered_ut = sorted(actual_ut - resolved_ut)

# ============================================================
# 4. source code coverage
# ============================================================
source_deps = {d.rstrip("/") for module in config for d in module.get("source_file_dependencies", [])}
covered_source = set()
for f in Path("vllm_ascend").rglob("*.py"):
    sf = str(f)
    for dep in source_deps:
        if sf == dep or sf.startswith(dep + "/"):
            covered_source.add(sf)
            break
    if f.name == "__init__.py":
        parent = str(f.parent)
        for dep in source_deps:
            if parent == dep or parent.startswith(dep + "/"):
                covered_source.add(sf)
                break
uncovered_source = sorted({str(f) for f in Path("vllm_ascend").rglob("*.py") if str(f) not in covered_source})

# ============================================================
# 5. estimated_times coverage
# ============================================================
_et = dict(meta.get("estimated_times", {}) or {})
_rm = dict(meta.get("runner_mapping", {}) or {})
_part = dict(meta.get("partition", {}) or {})

# Build NPU UT regex patterns from runner_mapping
npu_ut_patterns = []
for pattern_str in _rm:
    with contextlib.suppress(re.error):
        npu_ut_patterns.append(re.compile(pattern_str))

# Expand all test paths
all_expanded = set()
for p in all_yaml_paths:
    pp = Path(pytest_node_file_path(p))
    if pp.exists() and pp.is_dir():
        for f in sorted(pp.rglob("test_*.py")):
            all_expanded.add(str(f))
    else:
        all_expanded.add(p)

# Strip ::nodeid suffix so counting is at file level (same as step 2)
all_expanded_files = {pytest_node_file_path(p) for p in all_expanded}

# Separate E2E / NPU UT / CPU UT
e2e_files = {p for p in all_expanded_files if "tests/e2e/" in p}
ut_files = {p for p in all_expanded_files if "tests/ut/" in p}
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
# 6. Correctness of runner_mapping
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
    matched = [p for p in all_expanded if pat.search(p)]
    if not matched:
        rm_errors.append(f"Pattern {pattern_str!r}: matches 0 tests (unused)")

rm_broken = len(rm_errors) > 0

# ============================================================
# 7. partition validity
# ============================================================
part_errors: list[str] = []
# Collect actual runner keys used in routing
actual_runner_keys: set[str] = set()
for p in all_expanded:
    for pat_str, rc in _rm.items():
        if re.compile(pat_str).search(p):
            for rk in rc.values():
                actual_runner_keys.add(rk)
            break

for key, val in sorted(_part.items()):
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

part_broken = len(part_errors) > 0

# ============================================================
# 8. E2E decorator coverage — every test_* fn must have
#    @pytest.mark.e2e_features(+@pytest.mark.e2e_model)
#    and features/models must be in ALLOWED_FEATURES/ALLOWED_MODELS
# ============================================================

e2e_root = Path("tests/e2e")
conftest_path = e2e_root / "conftest.py"


def _parse_frozenset_from_conftest(varname: str) -> frozenset[str]:
    """Extract a frozenset[str] variable from tests/e2e/conftest.py via AST."""
    if not conftest_path.exists():
        return frozenset()
    try:
        source = conftest_path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source)
    except Exception:
        return frozenset()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == varname:
                if isinstance(node.value, ast.Call):
                    if len(node.value.args) == 1:
                        arg = node.value.args[0]
                        if isinstance(arg, ast.List):
                            return frozenset(
                                elt.value
                                for elt in arg.elts
                                if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                            )
    return frozenset()


def _parse_e2e_marks(source_code: str, method_name: str) -> tuple[list[str], list[str]]:
    """Extract (features, models) from @pytest.mark.e2e_features / e2e_model."""
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return [], []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name != method_name:
            continue
        features, models = [], []
        for dec in node.decorator_list:
            if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute):
                if dec.func.attr == "e2e_features":
                    for arg in dec.args:
                        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                            features.append(arg.value)
                if dec.func.attr == "e2e_model":
                    for arg in dec.args:
                        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                            models.append(arg.value)
        return features, models
    return [], []


allowed_features = _parse_frozenset_from_conftest("ALLOWED_FEATURES")
allowed_models = _parse_frozenset_from_conftest("ALLOWED_MODELS")
missing_marks: list[str] = []
unknown_features: list[str] = []
unknown_models: list[str] = []

e2e_pr = Path("tests/e2e/pull_request")
if e2e_pr.exists():
    for f in sorted(e2e_pr.rglob("test_*.py")):
        rel = str(f.relative_to(e2e_root))
        try:
            source = f.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not node.name.startswith("test_"):
                continue
            method = node.name
            features, models = _parse_e2e_marks(source, method)
            if not features:
                missing_marks.append(f"{rel}::{method}: missing @pytest.mark.e2e_features")
            if not models:
                missing_marks.append(f"{rel}::{method}: missing @pytest.mark.e2e_model")
            for feat in features:
                if allowed_features and feat not in allowed_features:
                    unknown_features.append(f"{rel}::{method}: '{feat}'")
            for model in models:
                if allowed_models and model not in allowed_models:
                    unknown_models.append(f"{rel}::{method}: '{model}'")

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

print(f"\n[2] E2E pull_request coverage: {len(actual_e2e)} total files, {len(uncovered_e2e)} uncovered")
if uncovered_e2e:
    for p in uncovered_e2e:
        print(f"    ✗ {p}")
else:
    print(f"    ✓ ALL {len(actual_e2e)} E2E test files covered")

print(
    f"\n[3] UT coverage: {len(actual_ut)} total files, {len(uncovered_ut)} uncovered"
    f"  (CPU: {len(cpu_ut_files)}, NPU: {len(npu_ut_files)})"
)
if uncovered_ut:
    for p in uncovered_ut:
        print(f"    ✗ {p}")
else:
    print(f"    ✓ ALL {len(actual_ut)} UT test files covered")

total_py = len(list(Path("vllm_ascend").rglob("*.py")))
print(f"\n[4] Source code coverage: {total_py} total .py files, {len(uncovered_source)} uncovered")
if uncovered_source:
    for p in uncovered_source:
        print(f"    ✗ {p}")
else:
    print("    ✓ ALL source .py files covered (including __init__.py)")

init_uncovered = sorted({str(f) for f in Path("vllm_ascend").rglob("__init__.py") if str(f) not in covered_source})
if init_uncovered:
    print(f"\n    Note: __init__.py files NOT in source_file_dependencies ({len(init_uncovered)}):")
    for p in init_uncovered:
        print(f"      - {p}")
    print("    (These are trivial files; their parent dirs are covered by prefix match)")

print("\n[5] estimated_times coverage (file-level):")
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

print("\n[6] runner_mapping validation:")
if rm_errors:
    for err in rm_errors:
        print(f"    ✗ {err}")
else:
    print("    ✓ All patterns valid and match at least one test")

print("\n[7] partition validation:")
if part_errors:
    for err in part_errors:
        print(f"    ✗ {err}")
else:
    print("    ✓ All partition keys valid and map to active runners")

print("\n[8] E2E decorator coverage (@pytest.mark.e2e_features / e2e_model):")
e2e_pr_count = (
    len(list(Path("tests/e2e/pull_request").rglob("test_*.py"))) if Path("tests/e2e/pull_request").exists() else 0
)
print(f"    Files: {e2e_pr_count} total")
print(f"    Missing marks: {len(missing_marks)}")
print(f"    Unknown features: {len(unknown_features)}")
print(f"    Unknown models: {len(unknown_models)}")
if missing_marks:
    for p in missing_marks:
        print(f"    ✗ MISSING MARKS: {p}")
else:
    print("    ✓ All test methods have @pytest.mark.e2e_features and @pytest.mark.e2e_model")
if unknown_features:
    for p in unknown_features:
        print(f"    ✗ UNKNOWN FEATURE: {p}")
else:
    print("    ✓ All features are in ALLOWED_FEATURES")
if unknown_models:
    for p in unknown_models:
        print(f"    ✗ UNKNOWN MODEL: {p}")
else:
    print("    ✓ All models are in ALLOWED_MODELS")

# ============================================================
# FIX INSTRUCTIONS
# ============================================================
has_errors = bool(
    broken
    or uncovered_e2e
    or uncovered_ut
    or uncovered_source
    or missing_et
    or cpu_ut_leaked
    or rm_errors
    or part_errors
    or missing_marks
    or unknown_features
    or unknown_models
)

if has_errors:
    print("\n" + "=" * 70)
    print("HOW TO FIX")
    print("=" * 70)

    if broken:
        print(f"\n  [1] BROKEN PATHS ({len(broken)}):")
        print("      These paths are referenced in .github/workflows/scripts/test_config.yaml")
        print("      but do not exist on disk.")
        print("      → Fix: update the paths in .github/workflows/scripts/test_config.yaml,")
        print("        or restore the missing files.")

    if uncovered_e2e:
        print(f"\n  [2] UNCOVERED E2E TESTS ({len(uncovered_e2e)}):")
        print("      These test files exist under tests/e2e/pull_request/ but are not")
        print("      referenced in .github/workflows/scripts/test_config.yaml.")
        print("      → Fix: add the missing test files to")
        print("        .github/workflows/scripts/test_config.yaml under the")
        print("        appropriate module's 'tests' list.")

    if uncovered_ut:
        print(f"\n  [3] UNCOVERED UT TESTS ({len(uncovered_ut)}):")
        print("      These test files exist under tests/ut/ but are not referenced in")
        print("      .github/workflows/scripts/test_config.yaml.")
        print("      → Fix: add the missing test files to")
        print("        .github/workflows/scripts/test_config.yaml under the")
        print("        appropriate module's 'tests' list.")

    if uncovered_source:
        print(f"\n  [4] UNCOVERED SOURCE ({len(uncovered_source)}):")
        print("      These .py files under vllm_ascend/ are not covered by any module's")
        print("      'source_file_dependencies' in .github/workflows/scripts/test_config.yaml.")
        print("      → Fix: add the missing files or their parent directories to")
        print("        'source_file_dependencies' in")
        print("        .github/workflows/scripts/test_config.yaml.")

    if missing_et:
        print(f"\n  [5] MISSING estimated_times ({len(missing_et)}):")
        print("      These E2E/NPU UT test files have no entry in the 'estimated_times'")
        print("      section of .github/workflows/scripts/test_config.yaml.")
        print("      → Fix: add an estimated time (in seconds) for each missing file in")
        print("        the 'estimated_times' map in")
        print("        .github/workflows/scripts/test_config.yaml.")

    if cpu_ut_leaked:
        print(f"\n  [6] CPU UT LEAKED in estimated_times ({len(cpu_ut_leaked)}):")
        print("      These CPU UT files should NOT have entries in 'estimated_times'.")
        print("      → Fix: remove these entries from the 'estimated_times' map in")
        print("        .github/workflows/scripts/test_config.yaml.")

    if rm_errors:
        print(f"\n  [7] runner_mapping ERRORS ({len(rm_errors)}):")
        print("      Issues with runner_mapping patterns in")
        print("      .github/workflows/scripts/test_config.yaml.")
        print("      → Fix: correct the regex patterns or add missing 'default' keys.")

    if part_errors:
        print(f"\n  [8] partition ERRORS ({len(part_errors)}):")
        print("      Issues with partition keys in")
        print("      .github/workflows/scripts/test_config.yaml.")
        print("      → Fix: ensure keys follow the '<runner>_x<num_npus>' format and")
        print("        map to active runners.")

    if missing_marks or unknown_features or unknown_models:
        print("\n  [9] E2E DECORATOR ISSUES:")
        if missing_marks:
            print(f"      {len(missing_marks)} test method(s) missing decorators (BLOCKING).")
            print("      → Fix: add @pytest.mark.e2e_features(...) and @pytest.mark.e2e_model(...)")
            print("        to each test function. Features must be from ALLOWED_FEATURES,")
            print("        models from ALLOWED_MODELS in tests/e2e/conftest.py.")
            print("        Example: @pytest.mark.e2e_features('dense', 'gqa', 'eager_mode')")
            print("                 @pytest.mark.e2e_model('Qwen/Qwen3-8B')")
        if unknown_features:
            print(f"      {len(unknown_features)} feature(s) not in ALLOWED_FEATURES (BLOCKING).")
            print("      → Fix: either correct the feature name or add it to ALLOWED_FEATURES")
            print("        in tests/e2e/conftest.py.")
        if unknown_models:
            print(f"      {len(unknown_models)} model(s) not in ALLOWED_MODELS (BLOCKING).")
            print("      → Fix: either correct the model name or add it to ALLOWED_MODELS")
            print("        in tests/e2e/conftest.py.")

    print()
    sys.exit(1)
