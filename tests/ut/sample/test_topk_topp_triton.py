"""Unit tests for the Qrita top-k/top-p Triton kernel dispatch and warmup.

The dispatch-guard tests run on CPU with the torch_npu mocks installed by
``tests/ut/conftest.py``. The kernel-equivalence tests need a real NPU and
Triton runtime and are skipped on CPU hosts (the full single-operator
accuracy suite lives under
``tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_topk_topp.py``).
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.triton_utils import HAS_TRITON

import vllm_ascend.sample.sampler as sampler_module
from vllm_ascend.sample.sampler import (
    AscendSampler,
    _apply_top_k_top_p_pytorch,
    _apply_top_k_top_p_torch_npu,
)


class _FakeProfile:
    def __init__(self, supports_cann):
        self._supports_cann = supports_cann

    def supports(self, capability):
        if capability.__class__.__name__ == "HardwareCapability" and capability.name == "NPU_TOP_K_TOP_P":
            return self._supports_cann
        return False


def test_dispatch_prefers_triton_when_available():
    """HAS_TRITON + non-batch-invariant selects the Ascend Triton wrapper."""
    with (
        patch.object(sampler_module, "HAS_TRITON", True),
        patch.object(sampler_module.envs, "VLLM_BATCH_INVARIANT", False),
        patch("vllm_ascend.sample.sampler.get_current_hardware_profile", return_value=_FakeProfile(False)),
    ):
        chosen = sampler_module._apply_top_k_top_p_dispatch()
        assert chosen is sampler_module._apply_top_k_top_p_ascend


def test_dispatch_bypasses_triton_for_batch_invariant():
    """batch_invariant mode keeps the original (non-Triton) selection."""
    with (
        patch.object(sampler_module, "HAS_TRITON", True),
        patch.object(sampler_module.envs, "VLLM_BATCH_INVARIANT", True),
        patch("vllm_ascend.sample.sampler.get_current_hardware_profile", return_value=_FakeProfile(True)),
    ):
        chosen = sampler_module._apply_top_k_top_p_dispatch()
        assert chosen is _apply_top_k_top_p_torch_npu

    with (
        patch.object(sampler_module, "HAS_TRITON", True),
        patch.object(sampler_module.envs, "VLLM_BATCH_INVARIANT", True),
        patch("vllm_ascend.sample.sampler.get_current_hardware_profile", return_value=_FakeProfile(False)),
    ):
        chosen = sampler_module._apply_top_k_top_p_dispatch()
        assert chosen is _apply_top_k_top_p_pytorch


def test_dispatch_without_triton_keeps_legacy_choice():
    with (
        patch.object(sampler_module, "HAS_TRITON", False),
        patch("vllm_ascend.sample.sampler.get_current_hardware_profile", return_value=_FakeProfile(True)),
    ):
        assert sampler_module._apply_top_k_top_p_dispatch() is _apply_top_k_top_p_torch_npu

    with (
        patch.object(sampler_module, "HAS_TRITON", False),
        patch("vllm_ascend.sample.sampler.get_current_hardware_profile", return_value=_FakeProfile(False)),
    ):
        assert sampler_module._apply_top_k_top_p_dispatch() is _apply_top_k_top_p_pytorch


def test_ascend_wrapper_routes_a5_to_cann_op():
    """On A5 the pre-existing wrapper is kept even with Triton available.

    The A5 hardware profile does not advertise NPU_TOP_K_TOP_P, so this
    must not depend on that capability at all: is_950() alone routes
    back to the CANN-op wrapper (which itself keeps the reduce-sample
    behaviour and otherwise sorts).
    """
    logits = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    k = torch.tensor([1], dtype=torch.int32)
    sentinel = object()
    cfg = MagicMock()
    cfg.enable_reduce_sample = False

    for a5_profile_supports_cann in (True, False):
        with (
            patch(
                "vllm_ascend.sample.sampler.get_current_hardware_profile",
                return_value=_FakeProfile(a5_profile_supports_cann),
            ),
            patch("vllm_ascend.sample.sampler.is_950", return_value=True),
            patch("vllm_ascend.sample.sampler.get_ascend_config", return_value=cfg),
            patch.object(sampler_module, "_apply_top_k_top_p_torch_npu", return_value=sentinel) as cann,
        ):
            out = sampler_module._apply_top_k_top_p_ascend(logits, k, None)
        assert out is sentinel
        cann.assert_called_once()


def test_ascend_wrapper_routes_reduce_sample_to_cann_op():
    """reduce-sample mode needs the gathered-tuple path on every device.

    enable_reduce_sample=True must reach the CANN-op wrapper for every
    is_950 x capability combination (A2/A3, A5, and 310P class devices
    alike); the non-reduce A2/A3 -> Triton case is covered separately
    by test_ascend_wrapper_uses_triton_off_a5.
    """
    logits = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    k = torch.tensor([1], dtype=torch.int32)
    sentinel = object()
    cfg = MagicMock()
    cfg.enable_reduce_sample = True

    for a5 in (True, False):
        for a2a3_profile_supports_cann in (True, False):
            with (
                patch(
                    "vllm_ascend.sample.sampler.get_current_hardware_profile",
                    return_value=_FakeProfile(a2a3_profile_supports_cann),
                ),
                patch("vllm_ascend.sample.sampler.is_950", return_value=a5),
                patch("vllm_ascend.sample.sampler.get_ascend_config", return_value=cfg),
                patch.object(sampler_module, "_apply_top_k_top_p_torch_npu", return_value=sentinel) as cann,
                patch.object(sampler_module, "apply_top_k_top_p_triton") as triton,
            ):
                out = sampler_module._apply_top_k_top_p_ascend(logits, k, None, top_k=2)
            assert out is sentinel
            cann.assert_called_once()
            triton.assert_not_called()


def test_ascend_wrapper_uses_triton_off_a5():
    logits = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    k = torch.tensor([1], dtype=torch.int32)
    p = torch.tensor([0.9])
    sentinel = object()
    cfg = MagicMock()
    cfg.enable_reduce_sample = False

    # A2/A3 advertise NPU_TOP_K_TOP_P but are not 950; with reduce-sample
    # disabled the Triton kernel is the path there.
    with (
        patch("vllm_ascend.sample.sampler.get_current_hardware_profile", return_value=_FakeProfile(True)),
        patch("vllm_ascend.sample.sampler.is_950", return_value=False),
        patch("vllm_ascend.sample.sampler.get_ascend_config", return_value=cfg),
        patch.object(sampler_module, "apply_top_k_top_p_triton", return_value=sentinel) as triton,
    ):
        out = sampler_module._apply_top_k_top_p_ascend(logits, k, p)
    assert out is sentinel
    triton.assert_called_once_with(logits, k, p)

    # k=p=None short-circuits before touching the kernel.
    with (
        patch("vllm_ascend.sample.sampler.get_current_hardware_profile", return_value=_FakeProfile(True)),
        patch("vllm_ascend.sample.sampler.get_ascend_config", return_value=cfg),
    ):
        out = sampler_module._apply_top_k_top_p_ascend(logits, None, None)
    assert out is logits


def test_mrv2_entry_runs_kernel_on_warmup():
    """k=p=None reaching the MRV2 hook still exercises the kernel.

    Current upstream callers short-circuit before this hook when both
    filters are disabled, so this is a defensive fallback; if reached,
    it runs the kernel with effective no-op values (k=V, p=1.0).
    """
    import vllm_ascend.worker.v2.sample.apply_top_k_top_p as mrv2
    from vllm_ascend.worker.v2.sample.apply_top_k_top_p import apply_top_k_top_p_npu

    logits = torch.randn(4, 64, dtype=torch.float32)
    # On a CPU runner HAS_TRITON is False at import time, so the guarded
    # import in apply_top_k_top_p.py never bound apply_top_k_top_p_triton
    # into the module; create the attribute so the patch works either way.
    with (
        patch.object(sampler_module, "HAS_TRITON", True),
        patch.object(mrv2, "HAS_TRITON", True),
        patch.object(mrv2, "apply_top_k_top_p_triton", create=True) as kern,
    ):
        kern.return_value = logits
        apply_top_k_top_p_npu(logits, None, None)
        kern.assert_called_once()
        warmup_k = kern.call_args[0][1]
        warmup_p = kern.call_args[0][2]
        # k=V and p=1.0 are both effective no-ops for the kernel, but
        # still allocate the buffer / tables.
        assert torch.equal(warmup_k, torch.full((4,), 64, dtype=torch.int32))
        assert torch.equal(warmup_p, torch.ones(4, dtype=torch.float32))


def test_mrv2_entry_falls_back_without_triton():
    from vllm_ascend.worker.v2.sample.apply_top_k_top_p import apply_top_k_top_p_npu

    logits = torch.tensor([[4.0, 3.0, 2.0, 1.0], [1.0, 2.0, 3.0, 4.0]])
    k = torch.tensor([1, 4], dtype=torch.int32)

    with patch("vllm_ascend.worker.v2.sample.apply_top_k_top_p.HAS_TRITON", False):
        out = apply_top_k_top_p_npu(logits.clone(), None, None)
        assert out.shape == logits.shape

        cfg = MagicMock()
        cfg.enable_reduce_sample = False
        with patch("vllm_ascend.sample.sampler.get_ascend_config", return_value=cfg):
            masked = apply_top_k_top_p_npu(logits.clone(), k, None)
        assert masked.shape == logits.shape
        assert torch.isfinite(masked).any()


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
        cfg = SimpleNamespace(enable_reduce_sample=False)
        with patch("vllm_ascend.sample.sampler.get_ascend_config", return_value=cfg):
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
