import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


def _load_ci_utils_module():
    module_path = Path(__file__).resolve().parents[2] / ".github/workflows/scripts/ci_utils.py"
    spec = importlib.util.spec_from_file_location("ci_utils_for_test", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ci_utils = _load_ci_utils_module()


def test_run_tests_emits_github_actions_groups_and_annotations(monkeypatch, capsys):
    calls = []
    return_codes = iter([0, 1])

    def fake_run(cmd):
        calls.append(cmd)
        return SimpleNamespace(returncode=next(return_codes))

    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setattr(ci_utils.subprocess, "run", fake_run)

    files = [
        ci_utils.TestFile(name="tests/e2e/singlecard/test_a.py"),
        ci_utils.TestFile(name="tests/e2e/singlecard/test_b.py::test_case"),
    ]

    exit_code, records = ci_utils.run_tests(files, continue_on_error=True)
    output = capsys.readouterr().out

    assert exit_code == -1
    assert len(records) == 2
    assert calls == [
        ["pytest", "-sv", "--durations=0", "--color=yes", "tests/e2e/singlecard/test_a.py"],
        ["pytest", "-sv", "--durations=0", "--color=yes", "tests/e2e/singlecard/test_b.py::test_case"],
    ]
    assert "::group::[1/2] tests/e2e/singlecard/test_a.py" in output
    assert "::group::[2/2] tests/e2e/singlecard/test_b.py" in output
    assert output.count("::endgroup::") == 2
    assert "[1/2] START  tests/e2e/singlecard/test_a.py" in output
    assert "[1/2] PASSED  tests/e2e/singlecard/test_a.py" in output
    assert "[2/2] FAILED (exit code 1)  tests/e2e/singlecard/test_b.py::test_case" in output
    assert "::notice::[1/2] PASSED  tests/e2e/singlecard/test_a.py" in output
    assert "::error::[2/2] FAILED tests/e2e/singlecard/test_b.py." in output
    assert "Please go to the Summary section to quickly review the error overview" in output
    assert output.index("::group::[1/2] tests/e2e/singlecard/test_a.py") < output.index(
        "[1/2] START  tests/e2e/singlecard/test_a.py"
    )
    assert output.index("::endgroup::") < output.index("::notice::[1/2] PASSED  tests/e2e/singlecard/test_a.py")


def test_run_tests_skips_groups_and_annotations_outside_github_actions(monkeypatch, capsys):
    def fake_run(_cmd):
        return SimpleNamespace(returncode=0)

    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.setattr(ci_utils.subprocess, "run", fake_run)

    exit_code, records = ci_utils.run_tests([ci_utils.TestFile(name="tests/e2e/singlecard/test_local.py")])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert len(records) == 1
    assert "::group::" not in output
    assert "::endgroup::" not in output
    assert "::notice::" not in output
    assert "::error::" not in output
    assert "[1/1] START  tests/e2e/singlecard/test_local.py" in output
