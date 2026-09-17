import importlib
from types import SimpleNamespace
from unittest.mock import Mock

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


def test_adaptive_verification_patch_uses_uncompiled_budget_assignment(monkeypatch):
    import vllm.v1.worker.gpu.spec_decode.adaptive_verification as adaptive

    monkeypatch.setattr(adaptive, "_assign_draft_token_budget_compiled", object())
    module = importlib.import_module("vllm_ascend.patch.worker.patch_v2.patch_adaptive_verification")
    importlib.reload(module)

    assert adaptive._assign_draft_token_budget_compiled is adaptive._assign_draft_token_budget
