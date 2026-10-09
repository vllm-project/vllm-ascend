"""CPU dispatch regressions and direct NPU Qrita equivalence tests.

Compilation failures may fall back in production. The direct kernel tests
intentionally do not use that fallback, so compiler/accuracy bugs stay visible.
"""

import logging
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.logger import init_logger
from vllm.triton_utils import HAS_TRITON

import vllm_ascend.sample.sampler as sampler_module
import vllm_ascend.sample.topk_topp as dispatch_module
import vllm_ascend.worker.v2.sample.apply_top_k_top_p as mrv2
from vllm_ascend.sample.sampler import (
    AscendSampler,
    _apply_top_k_top_p_pytorch,
    _apply_top_k_top_p_torch_npu,
)


class _FakeProfile:
    def __init__(self, supports_cann):
        self._supports_cann = supports_cann

    def supports(self, capability):
        return capability.name == "NPU_TOP_K_TOP_P" and self._supports_cann


class _FakeCompilationError(Exception):
    pass


@pytest.fixture(autouse=True)
def _reset_dispatchers():
    dispatch_module._get_dispatcher.cache_clear()
    dispatch_module._compilation_error_types.cache_clear()
    yield
    dispatch_module._get_dispatcher.cache_clear()
    dispatch_module._compilation_error_types.cache_clear()


@pytest.mark.parametrize("has_mlir_error", [True, False])
def test_compiler_exception_types_support_ascend_and_standard_triton(has_mlir_error, monkeypatch):
    class CompilationError(Exception):
        pass

    class MLIRCompilationError(Exception):
        pass

    errors = SimpleNamespace(CompilationError=CompilationError)
    if has_mlir_error:
        errors.MLIRCompilationError = MLIRCompilationError
    monkeypatch.setitem(sys.modules, "triton.compiler.errors", errors)
    expected = (CompilationError, MLIRCompilationError) if has_mlir_error else (CompilationError,)
    assert dispatch_module._compilation_error_types() == expected


@pytest.fixture
def triton_path():
    with (
        patch.object(dispatch_module, "HAS_TRITON", True),
        patch.object(dispatch_module.envs, "VLLM_BATCH_INVARIANT", False),
        patch.object(dispatch_module, "is_950", return_value=False),
        patch.object(dispatch_module, "is_310p", return_value=False),
        patch.object(dispatch_module, "_compilation_error_types", return_value=(_FakeCompilationError,)),
    ):
        yield


@pytest.mark.parametrize(
    "has_triton,batch_invariant,supports_cann",
    [(True, False, False), (True, True, True), (True, True, False), (False, False, True), (False, False, False)],
)
def test_dispatch_keeps_legacy_choice_when_required(has_triton, batch_invariant, supports_cann):
    with (
        patch.object(sampler_module, "HAS_TRITON", has_triton),
        patch.object(sampler_module.envs, "VLLM_BATCH_INVARIANT", batch_invariant),
        patch.object(sampler_module, "get_current_hardware_profile", return_value=_FakeProfile(supports_cann)),
    ):
        chosen = sampler_module._apply_top_k_top_p_dispatch()
    expected = (
        sampler_module._apply_top_k_top_p_ascend
        if has_triton and not batch_invariant
        else _apply_top_k_top_p_torch_npu
        if supports_cann
        else _apply_top_k_top_p_pytorch
    )
    assert chosen is expected


@pytest.mark.parametrize("entry", [sampler_module._apply_top_k_top_p_ascend, mrv2.apply_top_k_top_p_npu])
def test_both_entries_use_triton_without_removed_config(entry, triton_path):
    # No AscendConfig mock: enable_reduce_sample no longer exists upstream.
    logits = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    k = torch.tensor([1], dtype=torch.int32)
    p = torch.tensor([0.9])
    with patch.object(dispatch_module, "apply_top_k_top_p_triton", return_value=logits) as kernel:
        assert entry(logits, k, p) is logits
    kernel.assert_called_once_with(logits, k, p)


@pytest.mark.parametrize("entry", [sampler_module._apply_top_k_top_p_ascend, mrv2.apply_top_k_top_p_npu])
def test_no_filters_are_a_true_noop(entry, triton_path):
    logits = torch.randn(4, 64)
    with patch.object(dispatch_module, "apply_top_k_top_p_triton") as kernel:
        assert entry(logits, None, None) is logits
    kernel.assert_not_called()
    assert dispatch_module._get_dispatcher.cache_info().currsize == 0


@pytest.mark.parametrize("entry", [sampler_module._apply_top_k_top_p_ascend, mrv2.apply_top_k_top_p_npu])
@pytest.mark.parametrize(
    "guard,value", [("is_950", True), ("is_310p", True), ("HAS_TRITON", False), ("VLLM_BATCH_INVARIANT", True)]
)
def test_hardware_and_mode_guards_keep_sort_fallback(entry, guard, value, triton_path):
    logits = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    k = torch.tensor([2], dtype=torch.int32)
    target = dispatch_module.envs if guard == "VLLM_BATCH_INVARIANT" else dispatch_module
    kwargs = {"return_value": value} if guard.startswith("is_") else {"new": value}
    with (
        patch.object(target, guard, **kwargs),
        patch.object(dispatch_module, "apply_top_k_top_p_triton") as kernel,
    ):
        masked = entry(logits.clone(), k, None)
    kernel.assert_not_called()
    assert torch.equal(torch.isfinite(masked), torch.tensor([[True, True, False, False]]))
    assert torch.equal(masked[:, :2], logits[:, :2])


@pytest.mark.parametrize("entry", [sampler_module._apply_top_k_top_p_ascend, mrv2.apply_top_k_top_p_npu])
@pytest.mark.parametrize("filters", ["k", "p", "both"])
def test_compile_failure_falls_back_and_is_not_retried(entry, filters, triton_path):
    logits = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    k = torch.tensor([2], dtype=torch.int32) if filters != "p" else None
    p = torch.tensor([0.7]) if filters != "k" else None
    # Compare with the corresponding pre-PR production fallback.
    fallback = (
        _apply_top_k_top_p_torch_npu
        if entry is sampler_module._apply_top_k_top_p_ascend
        else mrv2.apply_top_k_top_p_pytorch
    )
    expected = fallback(logits.clone(), k, p)
    with (
        patch.object(
            dispatch_module, "apply_top_k_top_p_triton", side_effect=_FakeCompilationError("compiler abort")
        ) as kernel,
        patch.object(dispatch_module.logger, "warning", wraps=dispatch_module.logger.warning) as warning,
    ):
        for _ in range(2):
            actual = entry(logits.clone(), k, p)
            assert torch.equal(actual, expected)
    kernel.assert_called_once()
    warning.assert_called_once()


@pytest.mark.parametrize("entry", [sampler_module._apply_top_k_top_p_ascend, mrv2.apply_top_k_top_p_npu])
def test_compile_failure_logs_traceback_with_real_vllm_logger(entry, triton_path):
    # warning_once does not accept exc_info in either supported vLLM version.
    # Exercise the real logger API: an unrestricted MagicMock hid this failure.
    real_logger = init_logger("vllm_ascend.sample.topk_topp_regression")
    records: list[logging.LogRecord] = []

    class RecordHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = RecordHandler()
    previous_level = real_logger.level
    real_logger.setLevel(logging.WARNING)
    real_logger.addHandler(handler)
    logits = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    k = torch.tensor([2], dtype=torch.int32)
    error = _FakeCompilationError("compiler abort regression")
    try:
        with (
            patch.object(dispatch_module, "logger", real_logger),
            patch.object(dispatch_module, "apply_top_k_top_p_triton", side_effect=error) as kernel,
        ):
            for _ in range(2):
                masked = entry(logits.clone(), k, None)
                assert torch.equal(torch.isfinite(masked), torch.tensor([[True, True, False, False]]))
    finally:
        real_logger.removeHandler(handler)
        real_logger.setLevel(previous_level)
    kernel.assert_called_once()
    assert len(records) == 1
    record = records[0]
    assert record.levelno == logging.WARNING
    assert "Using sort-based masking for this specialization" in record.getMessage()
    assert record.exc_info is not None
    assert record.exc_info[1] is error
    assert record.exc_info[2] is not None
    formatted = logging.Formatter().format(record)
    assert "Traceback (most recent call last)" in formatted
    assert "compiler abort regression" in formatted


@pytest.mark.parametrize(
    "error", [RuntimeError("device error"), AssertionError("bad input"), ValueError("invalid shape")]
)
def test_runtime_and_input_errors_are_not_swallowed(error, triton_path):
    logits = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    k = torch.tensor([2], dtype=torch.int32)
    fallback = MagicMock()
    with patch.object(dispatch_module, "apply_top_k_top_p_triton", side_effect=error) as kernel:
        for _ in range(2):
            with pytest.raises(type(error), match=str(error)):
                dispatch_module.apply_top_k_top_p_with_fallback(logits, k, None, fallback)
    assert kernel.call_count == 2
    fallback.assert_not_called()


@pytest.mark.parametrize("variant", ["batch", "vocab", "dtype", "stride", "filters", "filter_dtype"])
def test_compile_failure_does_not_disable_other_specializations(variant, triton_path):
    logits = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    k = torch.tensor([2], dtype=torch.int32)
    fallback = MagicMock(return_value=logits)
    sentinel = object()
    with patch.object(
        dispatch_module, "apply_top_k_top_p_triton", side_effect=[_FakeCompilationError(), sentinel]
    ) as kernel:
        dispatch_module.apply_top_k_top_p_with_fallback(logits, k, None, fallback)
        other = logits
        other_k = k
        other_p = None
        if variant == "batch":
            other = logits.repeat(2, 1)
            other_k = k.repeat(2)
        elif variant == "vocab":
            other = torch.cat((logits, logits), dim=1)
        elif variant == "dtype":
            other = logits.to(torch.float64)
        elif variant == "stride":
            other = torch.cat((logits, logits), dim=1)[:, ::2]
        elif variant == "filters":
            other_p = torch.tensor([0.9])
        elif variant == "filter_dtype":
            other_k = k.to(torch.int64)
        assert dispatch_module.apply_top_k_top_p_with_fallback(other, other_k, other_p, fallback) is sentinel
    assert kernel.call_count == 2
    fallback.assert_called_once()


def _npu_runtime_available() -> bool:
    """True only on a real NPU host with Triton importable."""
    if not HAS_TRITON:
        return False
    try:
        return bool(torch.npu.is_available())
    except Exception:
        return False


@pytest.mark.skipif(
    not _npu_runtime_available(),
    reason="Kernel equivalence needs a real NPU with Triton",
)
class TestTopkToppKernelEquivalence:
    """Small-shape NPU comparison against the sort+mask reference."""

    @staticmethod
    def _reference(logits, k, p):
        return _apply_top_k_top_p_pytorch(logits, k, p)

    def test_small_shapes_match_reference(self):
        from vllm_ascend.ops.triton.v2.sample.topk_topp import apply_top_k_top_p_triton

        torch.manual_seed(0)
        device = "npu:0"
        for batch, vocab in ((1, 256), (4, 1024), (8, 4096)):
            logits = torch.randn(batch, vocab, device=device, dtype=torch.float32)
            k = torch.randint(1, vocab, (batch,), device=device, dtype=torch.int32)
            k[0] = vocab  # one no-op row
            p = torch.rand(batch, device=device, dtype=torch.float32) * 0.8 + 0.1
            p[-1] = 1.0  # one no-op row

            expected = self._reference(logits.clone(), k, p)
            actual = apply_top_k_top_p_triton(logits.clone(), k, p)

            assert torch.isfinite(expected).sum() == torch.isfinite(actual).sum(), (
                f"kept-count mismatch at batch={batch}, vocab={vocab}"
            )
            assert torch.allclose(
                torch.nan_to_num(expected, neginf=0.0),
                torch.nan_to_num(actual, neginf=0.0),
                atol=1e-6,
                rtol=1e-6,
            )


def test_sampler_still_constructs():
    """Sanity: AscendSampler init is unaffected by the new dispatch."""
    sampler = AscendSampler(logprobs_mode="raw_logprobs")
    assert isinstance(sampler.topk_topp_sampler.apply_top_k_top_p, object)
