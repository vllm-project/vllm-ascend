from __future__ import annotations

import inspect
import importlib.util
from pathlib import Path


def load_ci_log_summary_v2_module():
    module_path = (
        Path(__file__).resolve().parents[2]
        / ".github"
        / "workflows"
        / "scripts"
        / "ci_log_summary_v2.py"
    )
    spec = importlib.util.spec_from_file_location("ci_log_summary_v2", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v2_prefers_traceback_for_failed_case():
    module = load_ci_log_summary_v2_module()
    log_text = """
2026-03-03T15:41:37Z [1/2] START  tests/e2e/test_case.py::test_case
2026-03-03T15:41:40Z Traceback (most recent call last):
2026-03-03T15:41:41Z TypeError: backend missing
2026-03-03T15:41:45Z =========================== short test summary info ============================
2026-03-03T15:41:46Z FAILED tests/e2e/test_case.py::test_case - RuntimeError: Engine core initialization failed. See root cause above.
""".strip()

    result = module.analyze_log(log_text)

    assert result["failed_test_cases"] == ["tests/e2e/test_case.py::test_case"]
    assert result["code_bugs"][0]["error_type"] == "TypeError"
    assert result["code_bugs"][0]["failed_test_cases"] == ["tests/e2e/test_case.py::test_case"]
    assert result["code_bugs"][0]["context"][0] == "Traceback (most recent call last):"


def test_v2_is_independent_from_ci_log_summary_module():
    module = load_ci_log_summary_v2_module()
    source = inspect.getsource(module)

    assert "_load_base_module" not in source
    assert "_base =" not in source
    assert "ci_log_summary_base" not in source


def test_v2_uses_summary_error_with_failure_block_context():
    module = load_ci_log_summary_v2_module()
    log_text = """
2026-03-03T15:41:37Z pytest -sv tests/e2e/test_auto_fit.py
=================================== FAILURES ===================================
_________________________ test_auto_fit_max_model_len __________________________

    def test_auto_fit_max_model_len():
>       vllm_config = VllmConfig(model_config=model_config)
E       AttributeError: 'CompilationConfig' object has no attribute 'compile_ranges_split_points'

=================================== short test summary info ===================================
FAILED tests/e2e/test_auto_fit.py::test_auto_fit_max_model_len - AttributeError: 'CompilationConfig' object has no attribute 'compile_ranges_split_points'
""".strip()

    result = module.analyze_log(log_text)

    assert result["failed_test_cases"] == ["tests/e2e/test_auto_fit.py::test_auto_fit_max_model_len"]
    assert result["code_bugs"][0]["error_type"] == "AttributeError"
    assert result["code_bugs"][0]["source"] == "case_summary_payload"
    assert result["code_bugs"][0]["context"][0].startswith("_________________________ test_auto_fit_max_model_len")


def test_v2_falls_back_to_summary_payload_for_failed_case():
    module = load_ci_log_summary_v2_module()
    log_text = """
2026-03-03T15:41:37Z pytest -sv /workspace/tests/e2e/test_case.py::test_case
2026-03-03T15:41:45Z =========================== short test summary info ============================
2026-03-03T15:41:46Z FAILED tests/e2e/test_case.py::test_case - assert "worker crashed with TypeError: backend missing"
""".strip()

    result = module.analyze_log(log_text)

    assert result["failed_test_cases"] == ["tests/e2e/test_case.py::test_case"]
    assert result["code_bugs"][0]["error_type"] == "TypeError"
    assert result["code_bugs"][0]["error_message"] == "backend missing"
    assert result["code_bugs"][0]["context"] == [
        'FAILED tests/e2e/test_case.py::test_case - assert "worker crashed with TypeError: backend missing"'
    ]


def test_v2_extracts_timeout_expired_from_embedded_summary_payload():
    module = load_ci_log_summary_v2_module()
    log_text = """
2026-03-03T15:41:37Z pytest -sv /workspace/tests/e2e/test_case.py::test_case
2026-03-03T15:41:45Z =========================== short test summary info ============================
2026-03-03T15:41:46Z FAILED tests/e2e/test_case.py::test_case - assert "worker crashed with subprocess.TimeoutExpired: command timed out after 300 seconds"
""".strip()

    result = module.analyze_log(log_text)

    assert result["failed_test_cases"] == ["tests/e2e/test_case.py::test_case"]
    assert result["code_bugs"][0]["error_type"] == "subprocess.TimeoutExpired"
    assert result["code_bugs"][0]["error_message"] == "command timed out after 300 seconds"
    assert result["code_bugs"][0]["source"] == "case_summary_payload"


def test_v2_matches_parameterized_failure_block_header_for_summary_context():
    module = load_ci_log_summary_v2_module()
    log_text = """
2026-03-03T15:41:37Z [1/1] START  tests/e2e/compile/test_case.py
2026-03-03T15:41:45Z =================================== FAILURES ===================================
2026-03-03T15:41:46Z _____________ test_case[True-1e-05-257-64-dtype0] ______________
2026-03-03T15:41:47Z >       build_config()
2026-03-03T15:41:48Z E       AttributeError: 'CompilationConfig' object has no attribute 'compile_ranges_split_points'
2026-03-03T15:41:49Z =================================== short test summary info ===================================
2026-03-03T15:41:50Z FAILED tests/e2e/compile/test_case.py::test_case[True-1e-05-257-64-dtype0] - AttributeError: 'CompilationConfig' object has no attribute 'compile_ranges_split_points'
2026-03-03T15:41:51Z [1/1] FAILED (exit code 1)  tests/e2e/compile/test_case.py  (30s)
""".strip()

    result = module.analyze_log(log_text)

    assert result["code_bugs"][0]["source"] == "case_summary_payload"
    assert result["code_bugs"][0]["context"][0].startswith(
        "_____________ test_case[True-1e-05-257-64-dtype0] ______________"
    )
    assert any("AttributeError:" in line for line in result["code_bugs"][0]["context"])


def test_v2_does_not_treat_type_ignore_as_traceback_error():
    module = load_ci_log_summary_v2_module()
    log_text = """
2026-03-03T15:41:37Z [1/1] START  tests/e2e/test_case.py::test_case
2026-03-03T15:41:40Z Traceback (most recent call last):
2026-03-03T15:41:41Z   File "/tmp/sample.py", line 1, in <module>
2026-03-03T15:41:42Z     self.worker.init_device()  # type: ignore
2026-03-03T15:41:43Z TypeError: backend missing
2026-03-03T15:41:45Z =========================== short test summary info ============================
2026-03-03T15:41:46Z FAILED tests/e2e/test_case.py::test_case - RuntimeError: Engine core initialization failed. See root cause above.
""".strip()

    result = module.analyze_log(log_text)

    assert result["code_bugs"][0]["error_type"] == "TypeError"
    assert result["code_bugs"][0]["error_message"] == "backend missing"
    assert result["code_bugs"][0]["context"][-1] == "TypeError: backend missing"


def test_v2_drops_errors_when_no_short_summary_exists():
    module = load_ci_log_summary_v2_module()
    log_text = """
2026-03-03T15:41:37Z pytest -sv /workspace/tests/e2e/test_case.py
[rank0]: test start
✗ FAILED: tests/e2e/test_case.py returned exit code 1
(EngineCore_DP0 pid=30130) ERROR 03-03 15:41:37 [core.py:1100] Worker failed with error
AttributeError: backend missing
""".strip()

    result = module.analyze_log(log_text)

    assert result["failed_test_cases"] == []
    assert result["distinct_errors"] == []
    assert result["code_bugs"] == []


def test_v2_extracts_runtime_error_from_traceback():
    module = load_ci_log_summary_v2_module()
    log_text = """
2026-03-03T15:41:37Z [1/1] START  tests/e2e/test_case.py::test_case
2026-03-03T15:41:40Z Traceback (most recent call last):
2026-03-03T15:41:41Z   File "/tmp/sample.py", line 1, in <module>
2026-03-03T15:41:42Z RuntimeError: NPU out of memory. Tried to allocate 614.00 MiB
2026-03-03T15:41:45Z =========================== short test summary info ============================
2026-03-03T15:41:46Z FAILED tests/e2e/test_case.py::test_case - vllm.v1.engine.exceptions.EngineDeadError: EngineCore encountered an issue. See stack trace above.
""".strip()

    result = module.analyze_log(log_text)

    assert result["failed_test_cases"] == ["tests/e2e/test_case.py::test_case"]
    assert result["code_bugs"][0]["error_type"] == "RuntimeError"
    assert "NPU out of memory" in result["code_bugs"][0]["error_message"]
    assert result["code_bugs"][0]["source"] == "case_traceback"


def test_v2_falls_back_to_summary_exception_when_no_core_error_exists():
    module = load_ci_log_summary_v2_module()
    log_text = """
2026-03-03T15:41:37Z [1/1] START  tests/e2e/test_case.py::test_case
2026-03-03T15:41:45Z =========================== short test summary info ============================
2026-03-03T15:41:46Z FAILED tests/e2e/test_case.py::test_case - vllm.v1.engine.exceptions.EngineDeadError: EngineCore encountered an issue. See stack trace above.
""".strip()

    result = module.analyze_log(log_text)

    assert result["failed_test_cases"] == ["tests/e2e/test_case.py::test_case"]
    assert result["code_bugs"][0]["error_type"] == "vllm.v1.engine.exceptions.EngineDeadError"
    assert result["code_bugs"][0]["source"] == "case_summary_payload"


def test_v2_uses_failure_block_context_for_summary_timeout_expired():
    module = load_ci_log_summary_v2_module()
    log_text = """
2026-03-03T15:41:37Z [1/1] START  tests/e2e/test_external_launcher.py
2026-03-03T15:41:45Z =================================== FAILURES ===================================
2026-03-03T15:41:46Z ______________ test_qwen3_external_launcher_with_sleepmode_level2 ______________
2026-03-03T15:41:47Z def test_qwen3_external_launcher_with_sleepmode_level2():
2026-03-03T15:41:48Z >           raise TimeoutExpired(cmd, timeout)
2026-03-03T15:41:49Z E           subprocess.TimeoutExpired: Command '['python', 'offline_external_launcher.py']' timed out after 300 seconds
2026-03-03T15:41:50Z /usr/local/python3.11.14/lib/python3.11/subprocess.py:1253: TimeoutExpired
2026-03-03T15:41:51Z =========================== short test summary info ============================
2026-03-03T15:41:52Z FAILED tests/e2e/test_external_launcher.py::test_qwen3_external_launcher_with_sleepmode_level2 - subprocess.TimeoutExpired: Command '['python', 'offline_external_launcher.py']' timed out after 300 seconds
2026-03-03T15:41:53Z [1/1] FAILED (exit code 1)  tests/e2e/test_external_launcher.py  (300s)
""".strip()

    result = module.analyze_log(log_text)

    assert result["failed_test_cases"] == [
        "tests/e2e/test_external_launcher.py::test_qwen3_external_launcher_with_sleepmode_level2"
    ]
    assert result["code_bugs"][0]["error_type"] == "subprocess.TimeoutExpired"
    assert result["code_bugs"][0]["source"] == "case_summary_payload"
    assert result["code_bugs"][0]["context"][0].startswith(
        "______________ test_qwen3_external_launcher_with_sleepmode_level2 ______________"
    )
    assert any("E           subprocess.TimeoutExpired:" in line for line in result["code_bugs"][0]["context"])


def test_v2_uses_summary_payload_in_final_fallback_instead_of_unknown_failure():
    module = load_ci_log_summary_v2_module()
    log_text = """
2026-03-03T15:41:37Z [1/1] START  tests/e2e/test_case.py::test_case
2026-03-03T15:41:45Z =========================== short test summary info ============================
2026-03-03T15:41:46Z FAILED tests/e2e/test_case.py::test_case - worker crashed before traceback was captured
""".strip()

    result = module.analyze_log(log_text)

    assert result["failed_test_cases"] == ["tests/e2e/test_case.py::test_case"]
    assert result["code_bugs"][0]["source"] == "case_summary_fallback"
    assert result["code_bugs"][0]["error_type"] == "SummaryFailure"
    assert result["code_bugs"][0]["error_message"] == "worker crashed before traceback was captured"
    assert result["code_bugs"][0]["context"] == [
        "FAILED tests/e2e/test_case.py::test_case - worker crashed before traceback was captured"
    ]


def test_v2_uses_test_session_starts_as_invocation_boundary():
    module = load_ci_log_summary_v2_module()
    log_text = """
2026-03-03T15:41:37Z [1/2] START  tests/e2e/test_file.py
2026-03-03T15:41:38Z ============================= test session starts ==============================
2026-03-03T15:41:39Z tests/e2e/test_file.py::test_first PASSED
2026-03-03T15:41:40Z ============================= test session starts ==============================
2026-03-03T15:41:41Z tests/e2e/test_file.py::test_second
2026-03-03T15:41:42Z Traceback (most recent call last):
2026-03-03T15:41:43Z TypeError: second failure
2026-03-03T15:41:45Z =========================== short test summary info ============================
2026-03-03T15:41:46Z FAILED tests/e2e/test_file.py::test_second - TypeError: second failure
""".strip()

    result = module.analyze_log(log_text)

    assert result["failed_test_cases"] == ["tests/e2e/test_file.py::test_second"]
    assert result["code_bugs"][0]["error_type"] == "TypeError"
    assert result["code_bugs"][0]["error_message"] == "second failure"


def test_v2_prefers_first_traceback_over_later_wrapper_traceback():
    module = load_ci_log_summary_v2_module()
    log_text = """
2026-03-03T15:41:37Z [1/1] START  tests/e2e/test_case.py::test_case
2026-03-03T15:41:38Z ============================= test session starts ==============================
2026-03-03T15:41:40Z Traceback (most recent call last):
2026-03-03T15:41:41Z TypeError: first root cause
2026-03-03T15:41:50Z Traceback (most recent call last):
2026-03-03T15:41:51Z RuntimeError: Server at 0.0.0.0 exited unexpectedly.
2026-03-03T15:41:55Z =========================== short test summary info ============================
2026-03-03T15:41:56Z FAILED tests/e2e/test_case.py::test_case - RuntimeError: Server at 0.0.0.0 exited unexpectedly.
""".strip()

    result = module.analyze_log(log_text)

    assert result["failed_test_cases"] == ["tests/e2e/test_case.py::test_case"]
    assert result["code_bugs"][0]["error_type"] == "TypeError"
    assert result["code_bugs"][0]["error_message"] == "first root cause"
    assert result["code_bugs"][0]["context"][-1] == "TypeError: first root cause"


def test_v2_prefers_first_root_cause_in_pytest_failure_block():
    module = load_ci_log_summary_v2_module()
    log_text = """
2026-03-03T15:41:37Z pytest -sv tests/e2e/test_case.py::test_case
=================================== FAILURES ===================================
_______________________________ test_case _____________________________________

Traceback (most recent call last):
TypeError: first root cause
RuntimeError: Engine core initialization failed. See root cause above.

=================================== short test summary info ===================================
FAILED tests/e2e/test_case.py::test_case - RuntimeError: Engine core initialization failed. See root cause above.
""".strip()

    result = module.analyze_log(log_text)

    assert result["failed_test_cases"] == ["tests/e2e/test_case.py::test_case"]
    assert result["code_bugs"][0]["error_type"] == "TypeError"
    assert result["code_bugs"][0]["error_message"] == "first root cause"
    assert result["code_bugs"][0]["source"] == "case_traceback"


def test_v2_strips_worker_prefixes_from_context():
    module = load_ci_log_summary_v2_module()
    log_text = """
2026-03-03T15:41:37Z [1/1] START  tests/e2e/test_case.py::test_case
2026-03-03T15:41:38Z ============================= test session starts ==============================
2026-03-03T15:41:40Z (EngineCore_DP0 pid=56546) Traceback (most recent call last):
2026-03-03T15:41:41Z (Worker pid=56547) TypeError: backend missing
2026-03-03T15:41:45Z =========================== short test summary info ============================
2026-03-03T15:41:46Z FAILED tests/e2e/test_case.py::test_case - TypeError: backend missing
""".strip()

    result = module.analyze_log(log_text)

    assert result["failed_test_cases"] == ["tests/e2e/test_case.py::test_case"]
    assert result["code_bugs"][0]["context"] == [
        "Traceback (most recent call last):",
        "TypeError: backend missing",
    ]


def test_v2_filters_wrapper_runtime_error_when_earlier_root_cause_exists():
    module = load_ci_log_summary_v2_module()
    log_text = """
2026-03-03T15:41:37Z [1/1] START  tests/e2e/test_case.py::test_case
2026-03-03T15:41:38Z ============================= test session starts ==============================
2026-03-03T15:41:40Z Traceback (most recent call last):
2026-03-03T15:41:41Z RuntimeError: NPU out of memory. Tried to allocate 614.00 MiB
2026-03-03T15:41:50Z Traceback (most recent call last):
2026-03-03T15:41:51Z vllm.v1.engine.exceptions.EngineDeadError: EngineCore encountered an issue. See stack trace above.
2026-03-03T15:41:55Z =========================== short test summary info ============================
2026-03-03T15:41:56Z FAILED tests/e2e/test_case.py::test_case - vllm.v1.engine.exceptions.EngineDeadError: EngineCore encountered an issue. See stack trace above.
""".strip()

    result = module.analyze_log(log_text)

    assert result["failed_test_cases"] == ["tests/e2e/test_case.py::test_case"]
    assert result["code_bugs"][0]["error_type"] == "RuntimeError"
    assert "NPU out of memory" in result["code_bugs"][0]["error_message"]
