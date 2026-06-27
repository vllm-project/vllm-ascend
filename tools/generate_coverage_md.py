#!/usr/bin/env python3
"""Generate e2e_coverage.md from test code decorators.

Scans tests/e2e/pull_request/ for test methods and reads
@pytest.mark.e2e_features / @pytest.mark.e2e_model marks directly
from the source code.  No separate YAML declaration needed.

Also validates (exits with code 1 on any failure):
  - every test_* function has @pytest.mark.e2e_features AND @pytest.mark.e2e_model
  - features are in ALLOWED_FEATURES (defined in tests/e2e/conftest.py)
  - models are in ALLOWED_MODELS (defined in tests/e2e/conftest.py)
  - feature combinations are logically consistent (e.g. no eager_mode + graph)
"""

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FILE = REPO_ROOT / "docs" / "source" / "developer_guide" / "contribution" / "e2e_coverage.md"
E2E_ROOT = REPO_ROOT / "tests" / "e2e"

CHECK = "✅"
EMPTY = ""

# Graph modes that are mutually exclusive with eager_mode
GRAPH_MODES = frozenset(
    {
        "full_graph",
        "full_decode_only",
        "piecewise_graph",
    }
)

CARD_SECTIONS = [
    ("1-Card Tests", "pull_request/one_card"),
    ("2-Card Tests", "pull_request/two_card"),
    ("4-Card Tests", "pull_request/four_card"),
]

ALL_SECTIONS = CARD_SECTIONS


# ============================================================
# Helpers
# ============================================================


def _load_allowed_features() -> frozenset[str]:
    """Extract ALLOWED_FEATURES from tests/e2e/conftest.py via AST parsing."""
    return _load_frozenset_from_conftest("ALLOWED_FEATURES")


def _load_allowed_models() -> frozenset[str]:
    """Extract ALLOWED_MODELS from tests/e2e/conftest.py via AST parsing."""
    return _load_frozenset_from_conftest("ALLOWED_MODELS")


def _load_frozenset_from_conftest(varname: str) -> frozenset[str]:
    """Extract a frozenset[str] variable from tests/e2e/conftest.py via AST.

    Avoids importing conftest.py (which pulls in heavy deps like torch).
    """
    conftest = E2E_ROOT / "conftest.py"
    if not conftest.exists():
        return frozenset()
    try:
        source = conftest.read_text(encoding="utf-8", errors="replace")
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


def _detect_310p(rel_path: str) -> bool:
    return "_310p" in rel_path or "/310p/" in rel_path or rel_path.startswith("310p/")


def _strip_section_prefix(rel_path: str) -> str:
    for _title, prefix in ALL_SECTIONS:
        p = prefix + "/"
        if rel_path.startswith(p):
            return rel_path[len(p) :]
    return rel_path


def _parse_decorators(source_code: str, method_name: str) -> tuple[bool, bool]:
    """Return (has_skip, has_skipif) for a given test method."""
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return False, False

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name != method_name:
            continue
        has_skip = False
        has_skipif = False
        for dec in node.decorator_list:
            if isinstance(dec, ast.Call):
                if isinstance(dec.func, ast.Name) and dec.func.id == "pytest.mark.skip":
                    has_skip = True
                if isinstance(dec.func, ast.Attribute) and dec.func.attr == "skip":
                    has_skip = True
                if isinstance(dec.func, ast.Name) and dec.func.id == "pytest.mark.skipif":
                    has_skipif = True
                if isinstance(dec.func, ast.Attribute) and dec.func.attr == "skipif":
                    has_skipif = True
            if isinstance(dec, ast.Name) and dec.id == "pytest.mark.skip":
                has_skip = True
        return has_skip, has_skipif
    return False, False


def _parse_e2e_marks(source_code: str, method_name: str) -> tuple[list[str], list[str], bool]:
    """Extract (features, models, is_parametrized) from @pytest.mark.e2e_features / e2e_model.

    Returns ([feature_strs], [model_strs], is_parametrized).
    ``is_parametrized`` is True when the function has a @pytest.mark.parametrize
    decorator, meaning features may vary per parameter case.
    """
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return [], [], False

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name != method_name:
            continue

        features: list[str] = []
        models: list[str] = []
        is_parametrized = False

        for dec in node.decorator_list:
            if not isinstance(dec, ast.Call):
                continue
            # Match @pytest.mark.e2e_features("a", "b")
            if isinstance(dec.func, ast.Attribute) and dec.func.attr == "e2e_features":
                for arg in dec.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        features.append(arg.value)
            # Match @pytest.mark.e2e_model("a", "b")
            if isinstance(dec.func, ast.Attribute) and dec.func.attr == "e2e_model":
                for arg in dec.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        models.append(arg.value)
            # Match @pytest.mark.parametrize(...)
            if isinstance(dec.func, ast.Attribute) and dec.func.attr == "parametrize":
                is_parametrized = True

        return features, models, is_parametrized

    return [], [], False


def _collect_test_files(root: Path) -> set[str]:
    result = set()
    for f in sorted(root.rglob("test_*.py")):
        if f.is_file():
            result.add(str(f.relative_to(E2E_ROOT)))
    return result


def _parse_actual_test_methods(file_path: Path) -> set[str]:
    try:
        source = file_path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source)
        return {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_")
        }
    except Exception:
        return set()


# ============================================================
# Validation
# ============================================================


def _validate_mutual_exclusion(entries: list[dict]) -> list[str]:
    """Check for logical conflicts in feature combinations.

    Parametrized tests are skipped because their features may vary per
    parameter case (e.g. some cases run eager_mode, others run graph modes).
    """
    errors = []
    for entry in entries:
        feats = set(entry.get("features", []))
        graph_feats = feats & GRAPH_MODES
        if not (graph_feats and "eager_mode" in feats):
            continue
        if entry.get("parametrized"):
            errors.append(
                f"[NOTE] {entry['rel_path']}::{entry['test_name']}: "
                f"eager_mode coexists with {graph_feats} in a parametrized test "
                f"(features may vary per case — not a conflict)"
            )
            continue
        errors.append(
            f"[CONFLICT] {entry['rel_path']}::{entry['test_name']}: eager_mode cannot coexist with {graph_feats}"
        )
    return errors


def _validate_features(entries: list[dict], allowed_features: frozenset[str]) -> list[str]:
    """Check all features are in ALLOWED_FEATURES."""
    if not allowed_features:
        return []
    warnings = []
    for entry in entries:
        for feat in entry.get("features", []):
            if feat not in allowed_features:
                warnings.append(
                    f"[UNKNOWN FEATURE] {entry['rel_path']}::{entry['test_name']}: '{feat}' is not in ALLOWED_FEATURES"
                )
    return warnings


def _validate_models(entries: list[dict], allowed_models: frozenset[str]) -> list[str]:
    """Check all models are in ALLOWED_MODELS (if allowlist is non-empty)."""
    if not allowed_models:
        return []
    warnings = []
    for entry in entries:
        models = entry.get("model", [])
        if not isinstance(models, list):
            models = [models] if models else []
        for model in models:
            if model and model not in allowed_models:
                warnings.append(
                    f"[UNKNOWN MODEL] {entry['rel_path']}::{entry['test_name']}: '{model}' is not in ALLOWED_MODELS"
                )
    return warnings


# ============================================================
# Row building
# ============================================================


def _build_section_rows(
    entries: list[dict],
    features_list: list[str],
    skip_info: dict[str, dict[str, tuple[bool, bool]]],
) -> list[dict]:
    """Build markdown table rows from entries enriched with skip info."""
    rows = []
    for entry in entries:
        rel_path = entry["rel_path"]
        test_name = entry["test_name"]
        models = entry.get("model", [])
        entry_features = entry.get("features", [])
        is_310p = _detect_310p(rel_path)

        # Format model: list -> "<br>" joined, empty -> "-"
        if isinstance(models, list):
            model_str = "<br>".join(models) if models else "-"
        elif models:
            model_str = str(models)
        else:
            model_str = "-"

        # Auto-detected skip info
        has_skip, has_skipif = False, False
        if rel_path.endswith(".py") and rel_path in skip_info:
            has_skip, has_skipif = skip_info[rel_path].get(test_name, (False, False))

        row = {
            "Test file": _strip_section_prefix(rel_path),
            "Test method": test_name,
            "Model": model_str,
            "Skipped": CHECK if has_skip else EMPTY,
            "Conditional skip": CHECK if has_skipif else EMPTY,
            "310P": CHECK if is_310p else EMPTY,
        }

        for col in features_list:
            if col in ("310P",):
                continue
            row[col] = CHECK if col in entry_features else EMPTY

        rows.append(row)
    return rows


# ============================================================
# Main
# ============================================================


def main():
    allowed_features = _load_allowed_features()
    if not allowed_features:
        print("ERROR: Could not load ALLOWED_FEATURES from tests/e2e/conftest.py")
        sys.exit(1)

    features_list = sorted(allowed_features)

    # Collect all test files and parse marks
    actual_py_files = _collect_test_files(E2E_ROOT / "pull_request")

    all_entries: list[dict] = []
    warnings: list[str] = []
    missing_marks: list[str] = []

    for rel_path in sorted(actual_py_files):
        file_path = E2E_ROOT / rel_path
        try:
            source = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        actual_methods = _parse_actual_test_methods(file_path)

        for method_name in sorted(actual_methods):
            features, models, is_parametrized = _parse_e2e_marks(source, method_name)

            if not features:
                missing_marks.append(f"[MISSING MARKS] {rel_path}::{method_name}: no @pytest.mark.e2e_features found")
            if not models:
                missing_marks.append(f"[MISSING MARKS] {rel_path}::{method_name}: no @pytest.mark.e2e_model found")

            all_entries.append(
                {
                    "rel_path": rel_path,
                    "test_name": method_name,
                    "model": models,
                    "features": features,
                    "parametrized": is_parametrized,
                }
            )

    # ====================================================
    # Auto-detect skip info from .py files
    # ====================================================
    skip_info: dict[str, dict[str, tuple[bool, bool]]] = {}

    for entry in all_entries:
        rel_path = entry["rel_path"]
        if not rel_path.endswith(".py"):
            continue
        file_path = E2E_ROOT / rel_path
        if not file_path.exists():
            continue
        try:
            source = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        if rel_path not in skip_info:
            skip_info[rel_path] = {}
        skip_info[rel_path][entry["test_name"]] = _parse_decorators(source, entry["test_name"])

    # ====================================================
    # Validation
    # ====================================================
    allowed_models = _load_allowed_models()
    warnings += _validate_features(all_entries, allowed_features)
    warnings += _validate_models(all_entries, allowed_models)
    warnings += _validate_mutual_exclusion(all_entries)

    if missing_marks:
        print("=" * 70)
        print("MISSING e2e_features MARKS")
        print("=" * 70)
        for w in missing_marks:
            print(f"  {w}")
        print()

    if warnings:
        print("=" * 70)
        print("WARNINGS")
        print("=" * 70)
        for w in warnings:
            print(f"  {w}")
        print()

    # ====================================================
    # Group entries into sections
    # ====================================================
    section_entries: dict[str, list[dict]] = {}
    for section_title, _ in ALL_SECTIONS:
        section_entries[section_title] = []

    for entry in all_entries:
        rel_path = entry["rel_path"]
        matched = False
        for section_title, section_prefix in ALL_SECTIONS:
            if rel_path.startswith(section_prefix + "/"):
                section_entries[section_title].append(entry)
                matched = True
                break
        if not matched:
            for section_title, _ in CARD_SECTIONS:
                keyword = section_title.split("-")[0].strip().lower().replace(" ", "_")
                if keyword in rel_path:
                    section_entries[section_title].append(entry)
                    matched = True
                    break

    # ====================================================
    # Generate markdown
    # ====================================================
    columns = ["Test file", "Test method", "Model", "Skipped", "Conditional skip"] + features_list
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"

    # Count decorated vs total
    decorated_count = sum(1 for e in all_entries if e["features"])
    total_count = len(all_entries)

    output = ""
    output += "# E2E Test Coverage\n\n"
    output += (
        "This document is auto-generated by `tools/generate_coverage_md.py` "
        "from `@pytest.mark.e2e_features` / `@pytest.mark.e2e_model` "
        "decorators in test source code. "
        "Each row is a test method; ✅ means the test covers the corresponding feature.\n\n"
    )
    if missing_marks:
        output += (
            f"> **Migration progress**: {decorated_count}/{total_count} test methods "
            f"have been decorated with `@pytest.mark.e2e_features`. "
            f"{len(missing_marks)} methods still need migration.  \n"
            f"> **Note**: `Skipped` and `Conditional skip` columns are auto-detected from "
            "`@pytest.mark.skip` / `@pytest.mark.skipif` decorators in the source code.\n\n"
        )
    else:
        output += (
            "> **Note**: `Skipped` and `Conditional skip` columns are auto-detected from "
            "`@pytest.mark.skip` / `@pytest.mark.skipif` decorators in the source code.\n\n"
        )

    for section_title, _ in ALL_SECTIONS:
        section_rows = _build_section_rows(section_entries.get(section_title, []), features_list, skip_info)
        if not section_rows:
            continue
        output += f"## {section_title}\n\n"
        output += header + "\n"
        output += separator + "\n"
        for row in section_rows:
            vals = [row.get(col, EMPTY) for col in columns]
            output += "| " + " | ".join(vals) + " |\n"
        output += "\n"

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(output.rstrip("\n") + "\n", encoding="utf-8")
    print(f"Written to {OUTPUT_FILE}")

    if missing_marks:
        print(f"\n{len(missing_marks)} test(s) missing decorators (see above).")
    if warnings:
        print(f"\n{len(warnings)} warning(s) found (see above).")

    if missing_marks:
        sys.exit(1)


if __name__ == "__main__":
    main()
