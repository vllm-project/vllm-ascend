from __future__ import annotations

import importlib.util
import json
import io
from contextlib import redirect_stdout
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
    assert result["bad_commit"] == "6d4f9d3ad"
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
    assert "### Overview" in summary
    assert "### Failed Tests" in summary
    assert "### Distinct Issues" in summary
    assert "1. `TypeError`: bad shape" in summary


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


def test_run_suite_start_line_maps_root_cause_to_failed_case():
    module = load_ci_log_summary_module()
    log_text = """
2026-03-03T15:41:37Z [1/2] START  tests/e2e/test_case.py::test_case
2026-03-03T15:41:40Z Traceback (most recent call last):
2026-03-03T15:41:41Z TypeError: backend missing
2026-03-03T15:41:45Z =========================== short test summary info ============================
2026-03-03T15:41:46Z FAILED tests/e2e/test_case.py::test_case - RuntimeError: Engine core initialization failed. See root cause above.
2026-03-03T15:41:47Z [1/2] FAILED (exit code 1)  tests/e2e/test_case.py::test_case  (12s)
""".strip()

    result = module.analyze_log(log_text)

    assert result["failed_test_cases"] == ["tests/e2e/test_case.py::test_case"]
    assert result["code_bugs"][0]["error_type"] == "TypeError"
    assert result["code_bugs"][0]["failed_test_files"] == ["tests/e2e/test_case.py"]
    assert result["code_bugs"][0]["failed_test_cases"] == ["tests/e2e/test_case.py::test_case"]


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


def test_context_prefers_first_root_cause_before_during_handling_wrapper():
    module = load_ci_log_summary_module()
    log_text = """
2026-02-27T21:28:23.4537706Z (EngineCore_DP0 pid=58926) ERROR 02-27 21:28:23 [core.py:1078] EngineCore failed to start.
2026-02-27T21:28:23.4538596Z (EngineCore_DP0 pid=58926) ERROR 02-27 21:28:23 [core.py:1078] Traceback (most recent call last):
2026-02-27T21:28:23.4558739Z (EngineCore_DP0 pid=58926) ERROR 02-27 21:28:23 [core.py:1078]   File "/tmp/sample.py", line 10, in first_path
2026-02-27T21:28:23.4565695Z (EngineCore_DP0 pid=58926) ERROR 02-27 21:28:23 [core.py:1078] TypeError: CudagraphDispatcher.dispatch() got an unexpected keyword argument 'disable_full'
2026-02-27T21:28:23.4567442Z (EngineCore_DP0 pid=58926) ERROR 02-27 21:28:23 [core.py:1078] During handling of the above exception, another exception occurred:
2026-02-27T21:28:23.4568899Z (EngineCore_DP0 pid=58926) ERROR 02-27 21:28:23 [core.py:1078] Traceback (most recent call last):
2026-02-27T21:28:23.4571117Z (EngineCore_DP0 pid=58926) ERROR 02-27 21:28:23 [core.py:1078]   File "/tmp/sample.py", line 20, in wrapped_path
2026-02-27T21:28:23.4623380Z (EngineCore_DP0 pid=58926) TypeError: CudagraphDispatcher.dispatch() got an unexpected keyword argument 'disable_full'
""".strip()

    result = module.analyze_log(log_text)

    context = result["code_bugs"][0]["context"]
    assert context[0] == "Traceback (most recent call last):"
    assert any('File "/tmp/sample.py", line 10, in first_path' in line for line in context)
    assert not any('File "/tmp/sample.py", line 20, in wrapped_path' in line for line in context)
    assert not any("During handling of the above exception" in line for line in context)


def test_context_uses_nearest_coherent_traceback_block_for_same_pid():
    module = load_ci_log_summary_module()
    log_text = """
2026-02-28T01:46:56.8692206Z (EngineCore_DP0 pid=22376) ERROR 02-28 01:46:56 [core.py:1078] Traceback (most recent call last):
2026-02-28T01:46:56.8703123Z (EngineCore_DP0 pid=22376) ERROR 02-28 01:46:56 [core.py:1078]     available_gpu_memory = self.model_executor.determine_available_memory()
2026-02-28T01:46:56.8712880Z (EngineCore_DP0 pid=22376) Traceback (most recent call last):
2026-02-28T01:46:56.8713411Z (EngineCore_DP0 pid=22376)   File "/usr/local/python3.11.14/lib/python3.11/multiprocessing/process.py", line 314, in _bootstrap
2026-02-28T01:46:56.8727906Z (EngineCore_DP0 pid=22376)     result = run_method(self.driver_worker, method, args, kwargs)
2026-02-28T01:46:56.8730934Z (EngineCore_DP0 pid=22376) ERROR 02-28 01:46:56 [core.py:1078]   File "/__w/vllm-ascend/vllm-ascend/vllm_ascend/worker/worker.py", line 335, in determine_available_memory
2026-02-28T01:46:56.8764454Z (EngineCore_DP0 pid=22376) ERROR 02-28 01:46:56 [core.py:1078] TypeError: AscendMMEncoderAttention.forward_oot() got an unexpected keyword argument 'sequence_lengths'
""".strip()

    result = module.analyze_log(log_text)

    context = result["code_bugs"][0]["context"]
    assert context[0] == "(EngineCore_DP0 pid=22376) Traceback (most recent call last):"
    assert any("multiprocessing/process.py" in line for line in context)
    assert any("worker.py\", line 335" in line for line in context)
    assert context[-1].endswith(
        "TypeError: AscendMMEncoderAttention.forward_oot() got an unexpected keyword argument 'sequence_lengths'"
    )


def test_pytest_failure_block_prefers_outer_assertion_error():
    module = load_ci_log_summary_module()
    log_text = '''
2026-03-13T03:25:54Z =================================== FAILURES ===================================
2026-03-13T03:25:54Z ____________________ test_build_summary_for_pytest_failure _____________________
2026-03-13T03:25:54Z     def test_build_summary_for_pytest_failure():
2026-03-13T03:25:54Z         log_text = """
2026-03-13T03:25:54Z     2026-03-03T15:41:37Z pytest -sv tests/ut/sample/test_feature.py
2026-03-13T03:25:54Z     >       raise TypeError("bad shape")
2026-03-13T03:25:54Z     E       TypeError: bad shape
2026-03-13T03:25:54Z         assert result["bad_commit"] == "6d4f9d3ad5aa3750697edcf013ad080619ae25e9"
2026-03-13T03:25:54Z E       AssertionError: assert '6d4f9d3ad' == '6d4f9d3ad5aa3750697edcf013ad080619ae25e9'
2026-03-13T03:25:54Z tests/ut/test_ci_log_summary.py:52: AssertionError
2026-03-13T03:25:54Z =========================== short test summary info ============================
2026-03-13T03:25:54Z FAILED tests/ut/test_ci_log_summary.py::test_build_summary_for_pytest_failure - AssertionError: assert '6d4f9d3ad' == '6d4f9d3ad5aa3750697edcf013ad080619ae25e9'
'''.strip()

    result = module.analyze_log(log_text)

    assert [e["error_type"] for e in result["code_bugs"]] == ["AssertionError"]
    assert "bad shape" not in result["code_bugs"][0]["error_message"]
    assert result["code_bugs"][0]["failed_test_files"] == ["tests/ut/test_ci_log_summary.py"]
    assert result["code_bugs"][0]["failed_test_cases"] == [
        "tests/ut/test_ci_log_summary.py::test_build_summary_for_pytest_failure"
    ]
    summary = module.render_summary(result, step_name="Run unit test", mode="ut")
    assert "### Distinct Issues" in summary
    assert "1. `AssertionError`:" in summary


def test_wrapper_assertion_is_suppressed_when_specific_root_cause_exists():
    module = load_ci_log_summary_module()
    log_text = """
2026-03-03T15:41:37Z pytest -sv tests/e2e/test_camem.py
Traceback (most recent call last):
  File "/tmp/sample.py", line 1, in <module>
    raise TypeError("bad shape")
TypeError: bad shape
=================================== FAILURES ===================================
_______________________________ test_end_to_end ________________________________

    def wrapper():
>       assert _exitcode == 0, (f"function {f} failed when called with" f" args {args} and kwargs {kwargs}")
E       AssertionError: function <function test_end_to_end at 0x1234> failed when called with args () and kwargs {}

tests/e2e/test_camem.py:10: AssertionError
=========================== short test summary info ============================
FAILED tests/e2e/test_camem.py::test_end_to_end - AssertionError: function <function test_end_to_end at 0x1234> failed when called with args () and kwargs {}
""".strip()

    result = module.analyze_log(log_text)

    assert [e["error_type"] for e in result["code_bugs"]] == ["TypeError"]
    assert result["code_bugs"][0]["failed_test_cases"] == ["tests/e2e/test_camem.py::test_end_to_end"]


def test_wrapper_assertion_is_kept_when_no_specific_root_cause_exists():
    module = load_ci_log_summary_module()
    log_text = """
2026-03-03T15:41:37Z pytest -sv tests/e2e/test_camem.py
=================================== FAILURES ===================================
_______________________________ test_end_to_end ________________________________

    def wrapper():
>       assert _exitcode == 0, (f"function {f} failed when called with" f" args {args} and kwargs {kwargs}")
E       AssertionError: function <function test_end_to_end at 0x1234> failed when called with args () and kwargs {}

tests/e2e/test_camem.py:10: AssertionError
=========================== short test summary info ============================
FAILED tests/e2e/test_camem.py::test_end_to_end - AssertionError: function <function test_end_to_end at 0x1234> failed when called with args () and kwargs {}
""".strip()

    result = module.analyze_log(log_text)

    assert [e["error_type"] for e in result["code_bugs"]] == ["AssertionError"]
    assert result["code_bugs"][0]["failed_test_cases"] == ["tests/e2e/test_camem.py::test_end_to_end"]


def test_render_json_contains_structured_fields():
    module = load_ci_log_summary_module()
    result = {
        "run_id": 123,
        "failed_test_files": ["tests/e2e/test_case.py"],
        "failed_test_cases": ["tests/e2e/test_case.py::test_case"],
        "distinct_errors": [],
        "code_bugs": [],
        "env_flakes": [],
    }

    rendered = module.render_json(result)
    parsed = json.loads(rendered)

    assert parsed["run_id"] == 123
    assert parsed["failed_test_files"] == ["tests/e2e/test_case.py"]
    assert parsed["failed_test_cases"] == ["tests/e2e/test_case.py::test_case"]


def test_render_llm_json_contains_reduced_fields_only():
    module = load_ci_log_summary_module()
    result = {
        "run_id": 123,
        "run_url": "https://example.test/runs/123",
        "good_commit": "good123",
        "bad_commit": "bad123",
        "failed_test_files": ["tests/e2e/test_case.py"],
        "failed_test_cases": ["tests/e2e/test_case.py::test_case"],
        "code_bugs": [{"error_type": "TypeError", "error_message": "boom"}],
        "env_flakes": [{"error_type": "OSError", "error_message": "[Errno 116] Stale file handle"}],
        "distinct_errors": [{"ignored": True}],
        "job_results": [{"ignored": True}],
    }

    rendered = module.render_llm_json(result)
    parsed = json.loads(rendered)

    assert parsed == {
        "run_id": 123,
        "run_url": "https://example.test/runs/123",
        "good_commit": "good123",
        "bad_commit": "bad123",
        "failed_test_files_count": 1,
        "failed_test_cases_count": 1,
        "failed_test_files": ["tests/e2e/test_case.py"],
        "failed_test_cases": ["tests/e2e/test_case.py::test_case"],
        "code_bugs": [{"error_type": "TypeError", "error_message": "boom"}],
        "env_flakes": [{"error_type": "OSError", "error_message": "[Errno 116] Stale file handle"}],
    }


def test_main_supports_output_json(tmp_path):
    module = load_ci_log_summary_module()
    log_file = tmp_path / "sample.log"
    log_file.write_text(
        "\n".join(
            [
                "2026-03-03T15:41:37Z pytest -sv tests/ut/sample/test_feature.py",
                "_______________________ test_feature _______________________",
                "",
                "    def test_feature():",
                ">       raise TypeError(\"bad shape\")",
                "E       TypeError: bad shape",
                "",
                "FAILED tests/ut/sample/test_feature.py::test_feature - TypeError: bad shape",
            ]
        ),
        encoding="utf-8",
    )
    argv = [
        "ci_log_summary.py",
        "--log-file",
        str(log_file),
        "--mode",
        "ut",
        "--step-name",
        "Run unit test",
        "--format",
        "json",
    ]

    stdout = io.StringIO()
    old_argv = module.sys.argv
    try:
        module.sys.argv = argv
        with redirect_stdout(stdout):
            module.main()
    finally:
        module.sys.argv = old_argv

    parsed = json.loads(stdout.getvalue())
    assert parsed["failed_test_files"] == ["tests/ut/sample/test_feature.py"]
    assert parsed["code_bugs"][0]["error_type"] == "TypeError"


def test_main_supports_llm_json_format(tmp_path):
    module = load_ci_log_summary_module()
    log_file = tmp_path / "sample.log"
    log_file.write_text(
        "\n".join(
            [
                "2026-03-03T15:41:37Z pytest -sv tests/e2e/test_case.py",
                "Traceback (most recent call last):",
                "  File \"/tmp/sample.py\", line 1, in <module>",
                "    raise AttributeError(\"backend missing\")",
                "AttributeError: backend missing",
                "FAILED tests/e2e/test_case.py::test_case - AttributeError: backend missing",
            ]
        ),
        encoding="utf-8",
    )

    argv = [
        "ci_log_summary.py",
        "--log-file",
        str(log_file),
        "--mode",
        "e2e",
        "--step-name",
        "Run e2e test",
        "--format",
        "llm-json",
    ]

    stdout = io.StringIO()
    old_argv = module.sys.argv
    try:
        module.sys.argv = argv
        with redirect_stdout(stdout):
            module.main()
    finally:
        module.sys.argv = old_argv

    parsed = json.loads(stdout.getvalue())
    assert sorted(parsed.keys()) == [
        "bad_commit",
        "code_bugs",
        "env_flakes",
        "failed_test_cases",
        "failed_test_cases_count",
        "failed_test_files",
        "failed_test_files_count",
        "good_commit",
        "run_id",
        "run_url",
    ]
    assert parsed["failed_test_cases"] == ["tests/e2e/test_case.py::test_case"]
    assert parsed["failed_test_cases_count"] == 1
    assert parsed["failed_test_files_count"] == 1
    assert parsed["code_bugs"][0]["error_type"] == "AttributeError"


def test_main_defaults_mode_and_step_name_for_summary(tmp_path):
    module = load_ci_log_summary_module()
    log_file = tmp_path / "sample.log"
    log_file.write_text(
        "\n".join(
            [
                "2026-03-03T15:41:37Z pytest -sv tests/e2e/test_case.py",
                "Traceback (most recent call last):",
                "  File \"/tmp/sample.py\", line 1, in <module>",
                "    raise AttributeError(\"backend missing\")",
                "AttributeError: backend missing",
                "FAILED tests/e2e/test_case.py::test_case - AttributeError: backend missing",
            ]
        ),
        encoding="utf-8",
    )

    argv = [
        "ci_log_summary.py",
        "--log-file",
        str(log_file),
    ]

    stdout = io.StringIO()
    old_argv = module.sys.argv
    try:
        module.sys.argv = argv
        with redirect_stdout(stdout):
            module.main()
    finally:
        module.sys.argv = old_argv

    output = stdout.getvalue()
    assert "## Test Failure Summary: Run test" in output
    assert "- Mode: `e2e`" in output
    assert "1. `AttributeError`: backend missing" in output


def test_main_supports_llm_json_with_default_mode_and_step_name(tmp_path):
    module = load_ci_log_summary_module()
    log_file = tmp_path / "sample.log"
    log_file.write_text(
        "\n".join(
            [
                "2026-03-03T15:41:37Z pytest -sv tests/e2e/test_case.py",
                "Traceback (most recent call last):",
                "  File \"/tmp/sample.py\", line 1, in <module>",
                "    raise AttributeError(\"backend missing\")",
                "AttributeError: backend missing",
                "FAILED tests/e2e/test_case.py::test_case - AttributeError: backend missing",
            ]
        ),
        encoding="utf-8",
    )

    argv = [
        "ci_log_summary.py",
        "--log-file",
        str(log_file),
        "--format",
        "llm-json",
    ]

    stdout = io.StringIO()
    old_argv = module.sys.argv
    try:
        module.sys.argv = argv
        with redirect_stdout(stdout):
            module.main()
    finally:
        module.sys.argv = old_argv

    parsed = json.loads(stdout.getvalue())
    assert parsed["failed_test_cases"] == ["tests/e2e/test_case.py::test_case"]
    assert parsed["code_bugs"][0]["error_type"] == "AttributeError"


def test_main_supports_summary_format_with_output_file(tmp_path):
    module = load_ci_log_summary_module()
    log_file = tmp_path / "sample.log"
    summary_file = tmp_path / "summary.md"
    log_file.write_text(
        "\n".join(
            [
                "2026-03-03T15:41:37Z pytest -sv tests/e2e/test_case.py",
                "Traceback (most recent call last):",
                "  File \"/tmp/sample.py\", line 1, in <module>",
                "    raise AttributeError(\"backend missing\")",
                "AttributeError: backend missing",
                "FAILED tests/e2e/test_case.py::test_case - AttributeError: backend missing",
            ]
        ),
        encoding="utf-8",
    )

    argv = [
        "ci_log_summary.py",
        "--log-file",
        str(log_file),
        "--mode",
        "e2e",
        "--step-name",
        "Run e2e test",
        "--output",
        str(summary_file),
    ]

    stdout = io.StringIO()
    old_argv = module.sys.argv
    try:
        module.sys.argv = argv
        with redirect_stdout(stdout):
            module.main()
    finally:
        module.sys.argv = old_argv

    summary = summary_file.read_text(encoding="utf-8")

    assert stdout.getvalue() == ""
    assert "## Test Failure Summary: Run e2e test" in summary
    assert "### Overview" in summary
    assert "### Distinct Issues" in summary
    assert "1. `AttributeError`: backend missing" in summary


def test_main_supports_json_format_with_output_file(tmp_path):
    module = load_ci_log_summary_module()
    log_file = tmp_path / "sample.log"
    output_file = tmp_path / "summary.json"
    log_file.write_text(
        "\n".join(
            [
                "2026-03-03T15:41:37Z pytest -sv tests/e2e/test_case.py",
                "Traceback (most recent call last):",
                "  File \"/tmp/sample.py\", line 1, in <module>",
                "    raise AttributeError(\"backend missing\")",
                "AttributeError: backend missing",
                "FAILED tests/e2e/test_case.py::test_case - AttributeError: backend missing",
            ]
        ),
        encoding="utf-8",
    )

    argv = [
        "ci_log_summary.py",
        "--log-file",
        str(log_file),
        "--format",
        "json",
        "--output",
        str(output_file),
    ]

    stdout = io.StringIO()
    old_argv = module.sys.argv
    try:
        module.sys.argv = argv
        with redirect_stdout(stdout):
            module.main()
    finally:
        module.sys.argv = old_argv

    parsed = json.loads(output_file.read_text(encoding="utf-8"))
    assert stdout.getvalue() == ""
    assert parsed["failed_test_cases"] == ["tests/e2e/test_case.py::test_case"]
    assert parsed["code_bugs"][0]["error_type"] == "AttributeError"


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


def test_process_run_keeps_job_local_error_mappings_isolated():
    module = load_ci_log_summary_module()

    run_info = {
        "html_url": "https://example.test/runs/456",
        "created_at": "2026-03-12T01:02:03Z",
    }
    jobs = {
        "jobs": [
            {"id": 21, "name": "job-a", "conclusion": "failure"},
            {"id": 22, "name": "job-b", "conclusion": "failure"},
        ]
    }
    log_a = """
2026-03-03T15:41:37Z pytest -sv tests/e2e/test_a.py
Traceback (most recent call last):
  File "/tmp/sample.py", line 1, in <module>
    raise TypeError("same root cause")
TypeError: same root cause
FAILED tests/e2e/test_a.py::test_a - TypeError: same root cause
""".strip()
    log_b = """
2026-03-03T15:41:37Z pytest -sv tests/e2e/test_b.py
Traceback (most recent call last):
  File "/tmp/sample.py", line 1, in <module>
    raise TypeError("same root cause")
TypeError: same root cause
FAILED tests/e2e/test_b.py::test_b - TypeError: same root cause
""".strip()

    def fake_json(endpoint: str, **params):
        if endpoint.endswith("/actions/runs/456"):
            return run_info
        if endpoint.endswith("/actions/runs/456/jobs"):
            return jobs
        raise AssertionError((endpoint, params))

    def fake_raw(endpoint: str) -> str:
        if endpoint.endswith("/actions/jobs/21/logs"):
            return log_a
        if endpoint.endswith("/actions/jobs/22/logs"):
            return log_b
        raise AssertionError(endpoint)

    module.gh_api_json = fake_json
    module.gh_api_raw = fake_raw
    module.get_good_commit = lambda: "abc1234"

    result = module.process_run(456)

    assert len(result["code_bugs"]) == 1
    assert result["code_bugs"][0]["failed_test_files"] == [
        "tests/e2e/test_a.py",
        "tests/e2e/test_b.py",
    ]
    assert result["job_results"][0]["errors"][0]["failed_test_files"] == ["tests/e2e/test_a.py"]
    assert result["job_results"][0]["errors"][0]["failed_test_cases"] == ["tests/e2e/test_a.py::test_a"]
    assert result["job_results"][1]["errors"][0]["failed_test_files"] == ["tests/e2e/test_b.py"]
    assert result["job_results"][1]["errors"][0]["failed_test_cases"] == ["tests/e2e/test_b.py::test_b"]


def test_same_signature_in_two_tests_keeps_two_job_occurrences_but_one_distinct_issue():
    module = load_ci_log_summary_module()
    log_text = """
2026-03-03T15:41:37Z pytest -sv tests/e2e/test_a.py::test_a
Traceback (most recent call last):
  File "/tmp/a.py", line 1, in <module>
    raise TypeError("same root cause")
TypeError: same root cause
2026-03-03T15:41:38Z pytest -sv tests/e2e/test_b.py::test_b
Traceback (most recent call last):
  File "/tmp/b.py", line 1, in <module>
    raise TypeError("same root cause")
TypeError: same root cause
=========================== short test summary info ============================
FAILED tests/e2e/test_a.py::test_a - TypeError: same root cause
FAILED tests/e2e/test_b.py::test_b - TypeError: same root cause
""".strip()

    result = module.analyze_log(log_text)

    assert len(result["job_results"][0]["errors"]) == 2
    assert result["job_results"][0]["errors"][0]["failed_test_cases"] == ["tests/e2e/test_a.py::test_a"]
    assert result["job_results"][0]["errors"][1]["failed_test_cases"] == ["tests/e2e/test_b.py::test_b"]
    assert len(result["code_bugs"]) == 1
    assert result["code_bugs"][0]["failed_test_cases"] == [
        "tests/e2e/test_a.py::test_a",
        "tests/e2e/test_b.py::test_b",
    ]


def test_file_level_fallback_binds_nearest_failed_case_when_case_mapping_is_missing():
    module = load_ci_log_summary_module()
    log_text = """
2026-03-03T15:41:37Z pytest -sv /workspace/tests/e2e/test_vlm.py
2026-03-03T15:41:40Z tests/e2e/test_vlm.py::test_first
2026-03-03T15:41:41Z tests/e2e/test_vlm.py::test_second
2026-03-03T15:41:42Z Traceback (most recent call last):
2026-03-03T15:41:43Z   File "/tmp/sample.py", line 1, in <module>
2026-03-03T15:41:44Z     raise TypeError("file-scoped root cause")
2026-03-03T15:41:45Z TypeError: file-scoped root cause
2026-03-03T15:41:46Z =========================== short test summary info ============================
2026-03-03T15:41:47Z FAILED tests/e2e/test_vlm.py::test_second - RuntimeError: Engine core initialization failed. See root cause above.
""".strip()

    result = module.analyze_log(log_text)

    assert result["failed_test_files"] == ["tests/e2e/test_vlm.py"]
    assert result["failed_test_cases"] == ["tests/e2e/test_vlm.py::test_second"]
    assert result["code_bugs"][0]["failed_test_files"] == ["tests/e2e/test_vlm.py"]
    assert result["code_bugs"][0]["failed_test_cases"] == ["tests/e2e/test_vlm.py::test_second"]


def test_extract_failed_case_mentions_scans_line_tokens_once():
    module = load_ci_log_summary_module()
    log_text = """
2026-03-03T15:41:37Z tests/e2e/test_a.py::test_a [Gloo] Rank 0
2026-03-03T15:41:38Z tests/e2e/test_b.py::test_b [Gloo] Rank 0
""".strip()

    mentions = module.extract_failed_case_mentions(
        log_text,
        ["tests/e2e/test_a.py::test_a", "tests/e2e/test_b.py::test_b"],
    )

    assert mentions["tests/e2e/test_a.py::test_a"] == [0]
    assert mentions["tests/e2e/test_b.py::test_b"] == [1]


def test_context_is_compressed_for_very_long_traceback():
    module = load_ci_log_summary_module()
    long_tail = "\n".join(f'  File "/tmp/sample.py", line {i}, in f{i}' for i in range(60))
    log_text = f"""
2026-03-03T15:41:37Z pytest -sv tests/e2e/test_case.py::test_case
2026-03-03T15:41:38Z Traceback (most recent call last):
{long_tail}
2026-03-03T15:41:39Z TypeError: backend missing
2026-03-03T15:41:40Z =========================== short test summary info ============================
2026-03-03T15:41:41Z FAILED tests/e2e/test_case.py::test_case - TypeError: backend missing
""".strip()

    result = module.analyze_log(log_text)
    context = result["code_bugs"][0]["context"]

    assert len(context) <= 37
    assert context[0] == "Traceback (most recent call last):"
    assert context[-1] == "TypeError: backend missing"
    assert "..." in context
