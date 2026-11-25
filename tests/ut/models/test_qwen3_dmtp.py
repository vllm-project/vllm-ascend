# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import pytest
import torch
from torch import nn

from vllm_ascend.models.qwen3_dmtp import MTPT1Residual, Qwen3DMTPForCausalLM, Qwen3DMTPModel


def _tiny_model() -> Qwen3DMTPForCausalLM:
    # Exercise the real loader and embedding entry point without creating TP
    # groups or allocating a target model.
    wrapper = Qwen3DMTPForCausalLM.__new__(Qwen3DMTPForCausalLM)
    nn.Module.__init__(wrapper)
    model = Qwen3DMTPModel.__new__(Qwen3DMTPModel)
    nn.Module.__init__(model)
    model._context_kv_proj = None
    model.embed_tokens = nn.Embedding(32, 4, dtype=torch.bfloat16)
    model.input_embedding_scale = 1.0
    model.has_separate_mask_embedding = False
    model.mtp_prefix_conditioner = MTPT1Residual(4, 2, 1e-6, torch.bfloat16)
    wrapper.model = model
    return wrapper


def test_t1_embedding_matches_checkpoint_reference_for_each_request():
    torch.manual_seed(47)
    wrapper = _tiny_model()
    model = wrapper.model
    down = torch.randn(2, 4, dtype=torch.bfloat16)
    up = torch.randn(4, 2, dtype=torch.bfloat16)
    model.load_weights(iter([("mtp_prefix_conditioner.down.weight", down), ("mtp_prefix_conditioner.up.weight", up)]))
    tokens = torch.tensor([[2, 3, 7, 7, 7, 7, 7, 7], [3, 2, 7, 7, 7, 7, 7, 7]])
    predictor = torch.tensor([[5.0, 1.0, -1.0, -5.0], [3.0, -5.0, 0.0, 1.0]], dtype=torch.bfloat16)
    result = wrapper.embed_dmtp_inputs(tokens, predictor)
    reference = model.embed_tokens(tokens).clone()
    for row, state in enumerate(predictor):
        normalized = torch.nn.functional.rms_norm(state.float(), (4,), eps=1e-6).bfloat16()
        reference[row, 1] += up @ torch.nn.functional.silu(down @ normalized)
    torch.testing.assert_close(result, reference, rtol=0, atol=0)


def test_loader_rejects_checkpoint_with_missing_t1_residual():
    model = _tiny_model().model
    with pytest.raises(ValueError, match=r"mtp_prefix_conditioner\.up\.weight"):
        model.load_weights(iter([("mtp_prefix_conditioner.down.weight", torch.zeros(2, 4))]))


def test_loader_rejects_qkv_with_k_and_v_but_no_q():
    model = _tiny_model().model
    attn = nn.Module()
    attn.qkv_proj = nn.Linear(4, 12, bias=False)
    writes = []

    def load_shard(param, tensor):
        writes.append(tensor.shard_id)
        param.data.chunk(3)["qkv".index(tensor.shard_id)].copy_(tensor)

    attn.qkv_proj.weight.weight_loader = load_shard
    layer = nn.Module()
    layer.self_attn = attn
    model.layers = nn.ModuleList([layer])
    weights = [
        (name, parameter.detach().clone())
        for name, parameter in model.named_parameters()
        if name.startswith("mtp_prefix_conditioner.")
    ]
    for shard in ("k", "v"):
        weights.append((f"layers.0.self_attn.{shard}_proj.weight", torch.randn(4, 4)))
    with pytest.raises(ValueError, match=r"layers\.0\.self_attn\.q_proj\.weight"):
        model.load_weights(iter(weights))
    assert writes == ["k", "v"]
