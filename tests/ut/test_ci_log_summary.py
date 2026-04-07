import importlib.util
import regex as re
import sys
from pathlib import Path


def _load_ci_log_summary_module():
    module_path = Path(__file__).resolve().parents[2] / ".github/workflows/scripts/ci_log_summary.py"
    spec = importlib.util.spec_from_file_location("ci_log_summary_for_test", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.modules.setdefault("regex", re)
    spec.loader.exec_module(module)
    return module


ci_log_summary = _load_ci_log_summary_module()


def test_format_error_block_uses_clear_markdown_sections():
    error = {
        "error_type": "RuntimeError",
        "error_message": "boom happened",
        "category": "Code Bug",
        "failed_test_files": ["tests/e2e/singlecard/test_example.py"],
        "failed_test_cases": ["tests/e2e/singlecard/test_example.py::test_case"],
        "context": ["Traceback (most recent call last):", "RuntimeError: boom happened"],
    }

    lines = ci_log_summary._format_error_block(1, error)

    assert lines[0] == "#### 1. `RuntimeError`"
    assert "**Message**" in lines
    assert "```text" in lines
    assert "boom happened" in lines
    assert "**Category:** `Code Bug`" in lines
    assert "**Failed test files**" in lines
    assert "- `tests/e2e/singlecard/test_example.py`" in lines
    assert "**Failed test cases**" in lines
    assert "- `tests/e2e/singlecard/test_example.py::test_case`" in lines
    assert "**Context**" in lines
    assert "Traceback (most recent call last):" in lines
    assert "RuntimeError: boom happened" in lines


def test_render_summary_includes_formatted_distinct_issues():
    result = {
        "run_id": None,
        "run_url": None,
        "failed_test_files": ["tests/e2e/singlecard/test_example.py"],
        "failed_test_cases": ["tests/e2e/singlecard/test_example.py::test_case"],
        "distinct_errors": [
            {
                "error_type": "AssertionError",
                "error_message": "expected foo, got bar",
                "category": "Code Bug",
                "failed_test_files": ["tests/e2e/singlecard/test_example.py"],
                "failed_test_cases": ["tests/e2e/singlecard/test_example.py::test_case"],
                "context": ["E   AssertionError: expected foo, got bar"],
            }
        ],
        "code_bugs": [
            {
                "error_type": "AssertionError",
                "error_message": "expected foo, got bar",
                "category": "Code Bug",
            }
        ],
        "env_flakes": [],
    }

    rendered = ci_log_summary.render_summary(result, step_name="Run singlecard-full test", mode="e2e")

    assert "### Distinct Issues" in rendered
    assert "#### 1. `AssertionError`" in rendered
    assert "**Message**" in rendered
    assert "```text" in rendered
    assert "**Context**" in rendered
