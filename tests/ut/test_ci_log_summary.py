from __future__ import annotations

import importlib.util
from pathlib import Path


def load_ci_log_summary_module():
    module_path = (
        Path(__file__).resolve().parents[2]
        / ".github"
        / "workflows"
        / "scripts"
        / "ci_log_summary.py"
    )
    spec = importlib.util.spec_from_file_location("ci_log_summary", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_summary_for_pytest_failure():
    module = load_ci_log_summary_module()
    log_text = """
2026-03-03T15:41:37Z pytest -sv tests/ut/sample/test_feature.py
2026-03-03 15:42:30,706 - 49378 - vllmProfiler - INFO - vLLM 0.1.dev1+g6d4f9d3ad.empty
_______________________ test_feature _______________________

    def test_feature():
>       raise TypeError("bad shape")
E       TypeError: bad shape

tests/ut/sample/test_feature.py:10: TypeError
FAILED tests/ut/sample/test_feature.py::test_feature - TypeError: bad shape
""".strip()

    module.gh_api_json = lambda endpoint, **params: {"sha": "6d4f9d3ad5aa3750697edcf013ad080619ae25e9"}

    result = module.analyze_log(log_text)
    summary = module.render_summary(
        result,
        step_name="Run unit test",
        mode="ut",
    )

    assert result["failed_test_files"] == ["tests/ut/sample/test_feature.py"]
    assert result["failed_test_cases"] == ["tests/ut/sample/test_feature.py::test_feature"]
    assert result["run_id"] is None
    assert result["run_url"] is None
    assert "good_commit" in result
    assert result["bad_commit"] == "6d4f9d3ad5aa3750697edcf013ad080619ae25e9"
    assert result["failed_jobs_count"] == 1
    assert result["total_jobs"] == 1
    assert result["job_summary"] == [{"name": "local-log", "conclusion": "failure"}]
    assert result["job_results"][0]["job_name"] == "local-log"
    assert result["job_results"][0]["failed_test_files"] == ["tests/ut/sample/test_feature.py"]
    assert result["job_results"][0]["failed_test_cases"] == ["tests/ut/sample/test_feature.py::test_feature"]
    assert len(result["code_bugs"]) == 1
    assert result["code_bugs"][0]["failed_test_files"] == ["tests/ut/sample/test_feature.py"]
    assert result["code_bugs"][0]["failed_test_cases"] == ["tests/ut/sample/test_feature.py::test_feature"]
    assert "Run unit test" in summary
    assert "tests/ut/sample/test_feature.py::test_feature" in summary
    assert "TypeError" in summary
    assert "bad shape" in summary


def test_build_summary_for_run_suite_failure():
    module = load_ci_log_summary_module()
    log_text = """
2026-03-03T15:41:37Z pytest -sv /workspace/tests/e2e/test_case.py
[rank0]: test start
✗ FAILED: tests/e2e/test_case.py returned exit code 1
(EngineCore_DP0 pid=30130) ERROR 03-03 15:41:37 [core.py:1100] Worker failed with error
AttributeError: backend missing
""".strip()

    result = module.analyze_log(log_text)
    summary = module.render_summary(
        result,
        step_name="Run e2e test",
        mode="e2e",
    )

    assert result["failed_test_files"] == ["tests/e2e/test_case.py"]
    assert result["failed_test_cases"] == []
    assert result["job_results"][0]["errors"][0]["failed_test_files"] == ["tests/e2e/test_case.py"]
    assert result["job_results"][0]["errors"][0]["failed_test_cases"] == []
    assert len(result["code_bugs"]) == 1
    assert "backend missing" in summary
    assert "tests/e2e/test_case.py" in summary


def test_extract_failed_tests_handles_ansi_wrapped_failed_lines():
    module = load_ci_log_summary_module()
    log_text = (
        "2026-03-03T15:41:37Z pytest -sv tests/e2e/test_case.py\n"
        "\x1b[31mFAILED\x1b[0m tests/e2e/test_case.py::test_case - TypeError: boom\n"
    )

    result = module.analyze_log(log_text)

    assert result["failed_test_files"] == ["tests/e2e/test_case.py"]
    assert result["failed_test_cases"] == ["tests/e2e/test_case.py::test_case"]


def test_extract_failed_files_and_cases_are_separated():
    module = load_ci_log_summary_module()
    log_text = """
2026-03-03T15:41:37Z pytest -sv /workspace/tests/e2e/test_case.py::test_case
FAILED tests/e2e/test_case.py::test_case - TypeError: boom
✗ FAILED: tests/e2e/test_case.py::test_case returned exit code 1
""".strip()

    result = module.analyze_log(log_text)

    assert result["failed_test_files"] == ["tests/e2e/test_case.py"]
    assert result["failed_test_cases"] == ["tests/e2e/test_case.py::test_case"]


def test_context_prefers_traceback_start():
    module = load_ci_log_summary_module()
    log_text = """
2026-03-03T15:41:37Z pytest -sv tests/e2e/test_case.py
Traceback (most recent call last):
  File "/tmp/sample.py", line 1, in <module>
    raise TypeError("boom")
TypeError: boom
FAILED tests/e2e/test_case.py::test_case - TypeError: boom
""".strip()

    result = module.analyze_log(log_text)

    assert result["code_bugs"][0]["context"][0] == "Traceback (most recent call last):"


def test_context_prefers_pytest_failure_block_when_no_traceback():
    module = load_ci_log_summary_module()
    log_text = """
2026-03-03T15:41:37Z pytest -sv tests/e2e/test_auto_fit.py
=================================== FAILURES ===================================
_________________________ test_auto_fit_max_model_len __________________________

    def test_auto_fit_max_model_len():
>       vllm_config = VllmConfig(model_config=model_config)
E       AttributeError: 'CompilationConfig' object has no attribute 'compile_ranges_split_points'

FAILED tests/e2e/test_auto_fit.py::test_auto_fit_max_model_len - AttributeError: 'CompilationConfig' object has no attribute 'compile_ranges_split_points'
""".strip()

    result = module.analyze_log(log_text)

    assert result["code_bugs"][0]["context"][0].startswith("_________________________ test_auto_fit_max_model_len")
    assert result["code_bugs"][0]["context"][-1].startswith("E       AttributeError:")
    summary = module.render_summary(result, step_name="Run e2e test", mode="e2e")
    assert "_________________________ test_auto_fit_max_model_len" in summary


def test_context_strips_github_actions_prefixes():
    module = load_ci_log_summary_module()
    log_text = """
e2e-test / singlecard-full (0)\tUNKNOWN STEP\t2026-03-12T08:56:45.0947989Z _________________________ test_auto_fit_max_model_len __________________________
e2e-test / singlecard-full (0)\tUNKNOWN STEP\t2026-03-12T08:56:45.0976375Z E           AttributeError: 'CompilationConfig' object has no attribute 'compile_ranges_split_points'
FAILED tests/e2e/test_auto_fit.py::test_auto_fit_max_model_len - AttributeError: 'CompilationConfig' object has no attribute 'compile_ranges_split_points'
""".strip()

    result = module.analyze_log(log_text)

    assert result["code_bugs"][0]["context"][0].startswith("_________________________ test_auto_fit_max_model_len")


def test_process_run_matches_extract_style_shape():
    module = load_ci_log_summary_module()

    run_info = {
        "html_url": "https://example.test/runs/123",
        "created_at": "2026-03-12T01:02:03Z",
    }
    jobs = {
        "jobs": [
            {"id": 11, "name": "job-pass", "conclusion": "success"},
            {"id": 12, "name": "job-fail", "conclusion": "failure"},
        ]
    }
    log_text = """
2026-03-03T15:41:37Z pytest -sv tests/e2e/test_case.py
______________________________ test_case ______________________________

    def test_case():
>       raise AttributeError("backend missing")
E       AttributeError: backend missing

FAILED tests/e2e/test_case.py::test_case - AttributeError: backend missing
""".strip()

    def fake_json(endpoint: str, **params):
        if endpoint.endswith("/actions/runs/123"):
            return run_info
        if endpoint.endswith("/actions/runs/123/jobs"):
            return jobs
        raise AssertionError((endpoint, params))

    def fake_raw(endpoint: str) -> str:
        assert endpoint.endswith("/actions/jobs/12/logs")
        return log_text

    module.gh_api_json = fake_json
    module.gh_api_raw = fake_raw
    module.get_good_commit = lambda: "abc1234"

    result = module.process_run(123)

    assert result["run_id"] == 123
    assert result["run_url"] == "https://example.test/runs/123"
    assert result["run_created_at"] == "2026-03-12T01:02:03Z"
    assert result["good_commit"] == "abc1234"
    assert result["total_jobs"] == 2
    assert result["failed_jobs_count"] == 1
    assert result["job_summary"] == [
        {"name": "job-pass", "conclusion": "success"},
        {"name": "job-fail", "conclusion": "failure"},
    ]
    assert result["job_results"][0]["job_name"] == "job-fail"
    assert result["failed_test_files"] == ["tests/e2e/test_case.py"]
    assert result["failed_test_cases"] == ["tests/e2e/test_case.py::test_case"]
    assert result["code_bugs"][0]["error_type"] == "AttributeError"
