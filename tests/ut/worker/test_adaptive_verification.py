import importlib
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

from vllm_ascend.ascend_forward_context import get_mrv2_in_profile_run
from vllm_ascend.attention.dsa_v1 import AscendDSAMetadataBuilder
from vllm_ascend.attention.indexer import AscendSFAIndexerMetadataBuilder
from vllm_ascend.attention.sfa_v1 import AscendSFAMetadataBuilder
from vllm_ascend.device.device_op import BaseDeviceAdaptor
from vllm_ascend.worker.v2.aclgraph_utils import ModelWithContext
from vllm_ascend.worker.v2.model_runner import NPUModelRunner

_NATIVE_TORCH_SUM = torch.sum


@pytest.mark.parametrize(
    "builder_cls",
    [AscendDSAMetadataBuilder, AscendSFAIndexerMetadataBuilder, AscendSFAMetadataBuilder],
)
@pytest.mark.parametrize(
    "speculative_config,expected",
    [
        (None, AttentionCGSupport.UNIFORM_BATCH),
        (SimpleNamespace(method="dspark", enable_adaptive_verification=False), AttentionCGSupport.UNIFORM_BATCH),
        (SimpleNamespace(method="eagle", enable_adaptive_verification=True), AttentionCGSupport.UNIFORM_BATCH),
        (SimpleNamespace(method="dspark", enable_adaptive_verification=True), AttentionCGSupport.ALWAYS),
    ],
)
def test_adaptive_verification_cudagraph_support(builder_cls, speculative_config, expected):
    config = SimpleNamespace(speculative_config=speculative_config)
    assert builder_cls.get_cudagraph_support(config, Mock()) is expected


def test_aclgraph_model_forwards_confidence_computation():
    model = Mock()
    expected = torch.tensor([0.25, 0.75])
    model.compute_confidence.return_value = expected
    wrapped = ModelWithContext(model, is_draft_model=True, is_draft_model_prefill=False)
    hidden = torch.randn(2, 4)
    markov = torch.randn(2, 4)

    assert wrapped.compute_confidence(hidden, markov) is expected
    model.compute_confidence.assert_called_once_with(hidden, markov)


@pytest.mark.parametrize(
    ("adaptive_verification", "context_len", "expected"),
    [
        (object(), 8192, True),
        (object(), 0, False),
        (None, 8192, False),
    ],
)
def test_adaptive_tail_dummy_run_balances_moe_routing(adaptive_verification, context_len, expected):
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.max_num_reqs = 32
    runner.max_num_tokens = 256
    runner.speculator = None
    runner.adaptive_verification = adaptive_verification
    runner.ascend_config = SimpleNamespace(xlite_graph_config=SimpleNamespace(enabled=False))
    runner.eplb = Mock()
    observed = []

    def fake_dummy_run(*args, **kwargs):
        observed.append(get_mrv2_in_profile_run())
        return None, None

    with (
        patch.object(GPUModelRunner, "_dummy_run", side_effect=fake_dummy_run),
        patch("vllm_ascend.worker.v2.model_runner.lmhead_tp_enable", return_value=False),
    ):
        runner._dummy_run(256, context_len=context_len)

    assert observed == [expected]
    runner.eplb.step.assert_called_once_with(is_dummy=True, is_profile=False)
    assert get_mrv2_in_profile_run() is False


def test_adaptive_verification_patch_uses_uncompiled_budget_assignment(monkeypatch):
    import vllm.v1.worker.gpu.spec_decode.adaptive_verification as adaptive

    monkeypatch.setattr(adaptive, "_assign_draft_token_budget_compiled", object())
    module = importlib.import_module("vllm_ascend.patch.worker.patch_v2.patch_adaptive_verification")
    importlib.reload(module)

    assert adaptive._assign_draft_token_budget_compiled is module._assign_draft_token_budget_ascend
    assert module._original_assign_draft_token_budget is adaptive._assign_draft_token_budget


def test_adaptive_verification_patch_preserves_budget_assignment(monkeypatch):
    import vllm.v1.worker.gpu.spec_decode.adaptive_verification as adaptive

    # Isolate this test from batch-invariant tests that patch torch.sum globally.
    monkeypatch.setattr(torch, "sum", _NATIVE_TORCH_SUM)
    monkeypatch.setattr(adaptive, "_assign_draft_token_budget_compiled", object())
    module = importlib.import_module("vllm_ascend.patch.worker.patch_v2.patch_adaptive_verification")
    importlib.reload(module)
    confidence_probs = torch.tensor([[0.95, 0.8, 0.5], [0.9, 0.85, 0.7], [0.7, 0.6, 0.5]])
    idx_mapping = torch.tensor([0, 1])
    capacities = torch.tensor([3, 2], dtype=torch.int32)
    expected = capacities.clone()
    adaptive._assign_draft_token_budget(confidence_probs, idx_mapping, expected, draft_budget=3, num_steps=3)

    with patch(
        "vllm_ascend.device.device_op.DeviceOperator.index_fill",
        side_effect=lambda tensor, dim, indices, value: tensor.scatter_(dim, indices, value),
    ) as mock_index_fill:
        module._assign_draft_token_budget_ascend(confidence_probs, idx_mapping, capacities, draft_budget=3, num_steps=3)

    mock_index_fill.assert_called_once()
    torch.testing.assert_close(capacities, expected)


def test_index_fill_mode_can_reenter_native_device_adaptor(monkeypatch):
    import vllm.v1.worker.gpu.spec_decode.adaptive_verification as adaptive

    monkeypatch.setattr(adaptive, "_assign_draft_token_budget_compiled", object())
    module = importlib.import_module("vllm_ascend.patch.worker.patch_v2.patch_adaptive_verification")
    importlib.reload(module)
    tensor = torch.zeros(5)
    indices = torch.tensor([1, 3])

    with patch("vllm_ascend.device.device_op.DeviceOperator", BaseDeviceAdaptor), module._IndexFillMode():
        result = tensor.index_fill_(0, indices, 2)

    assert result is tensor
    torch.testing.assert_close(result, torch.tensor([0, 2, 0, 2, 0], dtype=result.dtype))
