# How to use this script: in vllm-ascend directory
# python .github/workflows/scripts/coverage.py
from pathlib import Path

import yaml

with open(".github/workflows/scripts/test_config.yaml") as f:
    config = yaml.safe_load(f)

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
# 1. BROKEN PATHS — yaml 中引用了不存在的路径
# ============================================================
broken = sorted(p for p in all_yaml_paths if not Path(p).exists())

# ============================================================
# 2. E2E pull_request 覆盖
# ============================================================
yaml_e2e_pr = {p for p in all_yaml_e2e_paths if "tests/e2e/pull_request/" in p}
resolved_e2e = set()
for p in yaml_e2e_pr:
    path = Path(p)
    if path.is_file():
        resolved_e2e.add(str(path))
    elif path.is_dir():
        for f in path.rglob("test_*.py"):
            resolved_e2e.add(str(f))
actual_e2e = {str(f) for f in Path("tests/e2e/pull_request").rglob("test_*.py")}
uncovered_e2e = sorted(actual_e2e - resolved_e2e)

# ============================================================
# 3. UT 覆盖
# ============================================================
resolved_ut = set()
for p in all_yaml_ut_paths:
    path = Path(p)
    if path.is_file():
        resolved_ut.add(str(path))
    elif path.is_dir():
        for f in path.rglob("test_*.py"):
            resolved_ut.add(str(f))
actual_ut = {str(f) for f in Path("tests/ut").rglob("test_*.py")}
uncovered_ut = sorted(actual_ut - resolved_ut)

# ============================================================
# 4. 源码覆盖
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

print(f"\n[3] UT coverage: {len(actual_ut)} total files, {len(uncovered_ut)} uncovered")
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

# Also check: which __init__.py files are NOT covered
init_uncovered = sorted({str(f) for f in Path("vllm_ascend").rglob("__init__.py") if str(f) not in covered_source})
if init_uncovered:
    print(f"\n    Note: __init__.py files NOT in source_file_dependencies ({len(init_uncovered)}):")
    for p in init_uncovered:
        print(f"      - {p}")
    print("    (These are trivial files; their parent dirs are covered by prefix match)")

print("\n" + "=" * 70)
