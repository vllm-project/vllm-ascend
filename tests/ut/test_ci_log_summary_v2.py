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


def test_v2_falls_back_to_pytest_failure_block():
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
