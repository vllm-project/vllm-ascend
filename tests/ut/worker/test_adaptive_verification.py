import importlib
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from vllm.v1.attention.backend import AttentionCGSupport

from vllm_ascend.attention.dsa_v1 import AscendDSAMetadataBuilder
from vllm_ascend.attention.indexer import AscendSFAIndexerMetadataBuilder
from vllm_ascend.attention.sfa_v1 import AscendSFAMetadataBuilder
from vllm_ascend.worker.v2.aclgraph_utils import ModelWithContext


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
    "enable_adaptive_verification,positions_len,expected_tokens",
    [(False, 8, 8), (True, 6, 6)],
)
def test_sfa_metadata_uses_reallocated_adaptive_token_shape(
    enable_adaptive_verification, positions_len, expected_tokens
):
    builder = AscendSFAMetadataBuilder.__new__(AscendSFAMetadataBuilder)
    builder.speculative_config = SimpleNamespace(
        method="dspark", enable_adaptive_verification=enable_adaptive_verification
    )
    builder.kernel_block_size = 128
    builder.nope = False
    builder._prepare_parallel_metadata = Mock(side_effect=lambda _, cos, sin, slot, *args: (cos, sin, slot, {}))
    builder.metadata_cls = Mock(return_value=Mock())
    builder.model_config = Mock(get_head_size=Mock(return_value=128))
    builder.attn_mask_builder = Mock()
    common = SimpleNamespace(
        num_reqs=2,
        num_actual_tokens=6,
        num_input_tokens=8,
        positions=torch.arange(positions_len),
        slot_mapping=torch.arange(8),
        block_table_tensor=torch.zeros((2, 1), dtype=torch.int32),
        query_start_loc=torch.tensor([0, 3, 6], dtype=torch.int32),
        seq_lens=torch.tensor([8, 9], dtype=torch.int32),
        _seq_lens_cpu=torch.tensor([8, 9], dtype=torch.int32),
        seq_lens_cpu=None,
        causal=True,
        attn_state=Mock(),
        max_query_len=2,
        max_seq_len=9,
        group_len=None,
        group_key_idx=None,
        group_key_cache_idx=None,
    )

    with (
        patch(
            "vllm_ascend.attention.sfa_v1.get_cos_and_sin_mla",
            return_value=(torch.ones(positions_len), torch.zeros(positions_len)),
        ),
        patch("vllm_ascend.attention.sfa_v1.get_ascend_config") as get_config,
    ):
        get_config.return_value.c8_reshape_optim_enabled = False
        builder._build(common)

    kwargs = builder.metadata_cls.call_args.kwargs
    assert kwargs["num_input_tokens"] == expected_tokens
    assert kwargs["positions"].shape[0] == expected_tokens
    assert kwargs["slot_mapping"].shape[0] == expected_tokens


def test_adaptive_verification_patch_uses_uncompiled_budget_assignment(monkeypatch):
    import vllm.v1.worker.gpu.spec_decode.adaptive_verification as adaptive

    monkeypatch.setattr(adaptive, "_assign_draft_token_budget_compiled", object())
    module = importlib.import_module("vllm_ascend.patch.worker.patch_v2.patch_adaptive_verification")
    importlib.reload(module)

    assert adaptive._assign_draft_token_budget_compiled is adaptive._assign_draft_token_budget


def test_ascend_acceptance_estimator_predicts_and_records_features():
    pytest.importorskip("vllm.v1.worker.gpu.spec_decode.acceptance_estimator")
    from vllm_ascend.worker.v2.spec_decode.acceptance_estimator import (
        AscendOnlineAcceptanceEstimator,
    )

    estimator = AscendOnlineAcceptanceEstimator(
        max_num_reqs=3,
        num_speculative_steps=2,
        device=torch.device("cpu"),
    )
    estimator.slope.fill_(1.0)
    estimator.intercepts.zero_()

    logits = torch.tensor([[2.0, 0.0], [0.0, 0.0]])
    idx_mapping = torch.tensor([1, 2])
    confidence = torch.zeros(2, 2)
    estimator.predict(
        logits,
        idx_mapping,
        torch.tensor(0),
        confidence,
        torch.ones(3),
    )

    expected = torch.softmax(logits, dim=-1).amax(dim=-1)
    torch.testing.assert_close(confidence[:, 0], expected)
    torch.testing.assert_close(estimator.predictions[idx_mapping, 0], expected)


def test_adaptive_verification_patch_installs_ascend_estimator():
    pytest.importorskip("vllm.v1.worker.gpu.spec_decode.acceptance_estimator")
    import vllm.v1.worker.gpu.spec_decode.speculator as speculator_module

    from vllm_ascend.patch.worker.patch_v2 import patch_adaptive_verification
    from vllm_ascend.worker.v2.spec_decode.acceptance_estimator import (
        AscendOnlineAcceptanceEstimator,
    )

    importlib.reload(patch_adaptive_verification)

    assert speculator_module.OnlineAcceptanceEstimator is AscendOnlineAcceptanceEstimator


def test_ascend_acceptance_estimator_accumulates_verdicts():
    pytest.importorskip("vllm.v1.worker.gpu.spec_decode.acceptance_estimator")
    from vllm_ascend.worker.v2.spec_decode.acceptance_estimator import (
        AscendOnlineAcceptanceEstimator,
    )

    estimator = AscendOnlineAcceptanceEstimator(
        max_num_reqs=2,
        num_speculative_steps=2,
        device=torch.device("cpu"),
    )
    estimator.features[1] = torch.tensor([0.5, -0.5])
    estimator.predictions[1] = torch.tensor([0.75, 0.25])

    # One accepted draft, followed by one rejected draft. num_sampled also
    # includes the bonus token, matching the model-runner contract.
    estimator.step(
        idx_mapping=torch.tensor([1]),
        num_sampled=torch.tensor([2]),
        num_rejected=torch.tensor([1]),
    )

    torch.testing.assert_close(estimator.counts, torch.ones(2))
    torch.testing.assert_close(estimator.grad[:, 1], torch.tensor([0.25, -0.25]))
