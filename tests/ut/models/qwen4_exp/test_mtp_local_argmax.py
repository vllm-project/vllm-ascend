# SPDX-License-Identifier: Apache-2.0
"""Greedy Qwen4Exp draft selection keeps the full-logit argmax semantics."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from vllm.model_executor.layers import logits_processor as logits_module
from vllm.model_executor.models.interfaces import LocalArgmaxMixin

from vllm_ascend.models.qwen4_exp.mtp import AscendQwen4ExpMTP
from vllm_ascend.ops.vocab_parallel_embedding import AscendLogitsProcessor, VocabParallelMode


def _make_model(weight, *, tp_size=1, rank=0, vocab_size=None, scale=1.0, soft_cap=None):
    vocab_size = weight.shape[0] * tp_size if vocab_size is None else vocab_size
    shard_start = rank * weight.shape[0]
    head = nn.Module()
    head.weight = nn.Parameter(weight, requires_grad=False)
    head.tp_size = tp_size
    head.parallel_mode = VocabParallelMode.STANDARD if tp_size > 1 else VocabParallelMode.REPLICATED
    head.shard_indices = SimpleNamespace(
        org_vocab_start_index=shard_start,
        num_org_vocab_padding=max(0, shard_start + weight.shape[0] - vocab_size),
    )
    # A quantized head also supplies apply(): selection must retain this
    # dispatch rather than reading a potentially packed weight directly.
    head.quant_method = SimpleNamespace(
        apply=Mock(side_effect=lambda layer, x, bias=None: F.linear(x, layer.weight, bias))
    )
    processor = AscendLogitsProcessor.__new__(AscendLogitsProcessor)
    nn.Module.__init__(processor)
    processor.head_dtype = None
    processor.scale = scale
    processor.soft_cap = soft_cap
    processor.org_vocab_size = vocab_size
    processor.logits_as_input = False
    processor.use_all_gather = True
    model = AscendQwen4ExpMTP.__new__(AscendQwen4ExpMTP)
    nn.Module.__init__(model)
    model.lm_head = head
    model.logits_processor = processor
    return model


@pytest.mark.parametrize("tp_size", [1, 2, 4])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("scale,soft_cap", [(1.0, None), (0.5, None), (1.5, 4.0)])
def test_local_argmax_matches_full_logits(monkeypatch, tp_size, dtype, scale, soft_cap):
    padded_vocab_size, vocab_size, batch_size = 32, 27, 3
    hidden = torch.eye(batch_size, dtype=dtype)
    weight = torch.zeros(padded_vocab_size, batch_size, dtype=dtype)
    # Ties within and across shards must choose the smallest global token ID.
    weight[[1, 9, 17], 0] = 3
    weight[23, 1] = 4
    weight[26, 2] = 2
    weight[vocab_size:] = 100  # Padded entries must never be selected.
    shard_weights = weight.chunk(tp_size)
    models = [
        _make_model(shard, tp_size=tp_size, rank=rank, vocab_size=vocab_size, scale=scale, soft_cap=soft_cap)
        for rank, shard in enumerate(shard_weights)
    ]
    full_logits = F.linear(hidden, weight)
    processed = full_logits.clone()
    if soft_cap is not None:
        processed = torch.tanh(processed / soft_cap) * soft_cap
    processed *= scale
    processed[:, vocab_size:] = -float("inf")
    candidates = []
    for rank, shard_logits in enumerate(processed.chunk(tp_size, dim=-1)):
        values, indices = shard_logits.max(dim=-1)
        candidates.append(torch.stack((values.float(), (indices + rank * shard_logits.shape[-1]).float()), dim=-1))

    calls = []

    def all_gather(tensor, dim=-1):
        calls.append((tensor.shape, tensor.dtype))
        return torch.cat(candidates, dim=-1) if tensor.shape[-1] == 2 else full_logits.clone()

    monkeypatch.setattr(logits_module, "tensor_model_parallel_all_gather", all_gather)
    for model in models:
        calls.clear()
        expected = model.compute_logits(hidden).argmax(dim=-1)
        assert calls == ([] if tp_size == 1 else [(torch.Size([batch_size, padded_vocab_size // tp_size]), dtype)])
        calls.clear()
        actual = model.get_top_tokens(hidden)
        torch.testing.assert_close(actual, expected)
        assert actual.tolist() == [1, 23, 26]
        # Two fp32 candidates per request replace the local vocabulary shard.
        assert calls == ([] if tp_size == 1 else [(torch.Size([batch_size, 2]), torch.float32)])
        assert model.lm_head.quant_method.apply.call_count == 2


def test_local_argmax_keeps_upstream_head_and_d2t_dispatch(monkeypatch):
    model = _make_model(torch.zeros(8, 2))
    hidden = torch.ones(3, 2)
    processor = Mock(return_value=torch.tensor([2, 4, 1]))
    monkeypatch.setattr(model.logits_processor, "get_top_tokens", processor)
    # Reuse the upstream mixin, including its optional draft-to-target map.
    assert issubclass(AscendQwen4ExpMTP, LocalArgmaxMixin)
    model.draft_id_to_target_id = torch.tensor([0, 10, 20, 30, 40, 50, 60, 70])
    torch.testing.assert_close(model.get_top_tokens(hidden), torch.tensor([22, 44, 11]))
    processor.assert_called_once_with(model.lm_head, hidden)


@pytest.mark.parametrize("mode", [VocabParallelMode.FINE_GRAINED, VocabParallelMode.PCP_X_TP])
def test_local_argmax_rejects_nonstandard_head_groups(monkeypatch, mode):
    model = _make_model(torch.zeros(8, 2))
    model.lm_head.parallel_mode = mode
    processor = Mock()
    monkeypatch.setattr(model.logits_processor, "get_top_tokens", processor)
    with pytest.raises(NotImplementedError, match="standard TP or replicated LM head"):
        model.get_top_tokens(torch.ones(3, 2))
    processor.assert_not_called()


def test_local_argmax_rejects_nonpositive_logit_scale():
    model = _make_model(torch.zeros(8, 2), scale=-1.0)
    with pytest.raises(ValueError, match="non-positive logit scaling"):
        model.get_top_tokens(torch.ones(3, 2))


def test_upstream_argmax_preserves_embedding_bias():
    model = _make_model(torch.zeros(8, 2))
    hidden = torch.ones(3, 2)
    bias = torch.tensor([0.0, 1.0, 0.0, 4.0, 0.0, 4.0, 0.0, 0.0])
    expected = model.logits_processor(model.lm_head, hidden, bias).argmax(dim=-1)
    actual = model.logits_processor.get_top_tokens(model.lm_head, hidden, bias)
    torch.testing.assert_close(actual, expected)
    assert actual.tolist() == [3, 3, 3]
    for call in model.lm_head.quant_method.apply.call_args_list:
        assert call.kwargs["bias"] is bias
