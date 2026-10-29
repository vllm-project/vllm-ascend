# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU contracts of the Muse Spark Ascend adapter.

The NPU kernels are replaced by PyTorch references here; the end-to-end run
on NPU lives in ``tests/e2e/pull_request/one_card/test_muse_spark.py``.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from torch import nn

pytest.importorskip("vllm.models.muse_spark.amd.model", reason="vLLM without Muse Spark")

import vllm.models.muse_spark.common.layers as upstream_layers  # noqa: E402
from vllm.models.muse_spark.amd.model import MuseSpark1xAttention  # noqa: E402
from vllm.models.muse_spark.common.layers import (  # noqa: E402
    MuseSpark1xRMSNorm,
    MuseSpark1xRMSNormWithInputScale,
)

import vllm_ascend.models.muse_spark as muse_spark  # noqa: E402
from vllm_ascend import models  # noqa: E402

NUM_TOKENS = 5
NUM_HEADS = 2
HEAD_DIM = 8
EPS = 1e-5


def _registered_architectures() -> dict[str, str]:
    registered: dict[str, str] = {}
    with patch.object(
        models.ModelRegistry,
        "register_model",
        side_effect=lambda name, target: registered.setdefault(name, target),
    ):
        models.register_model()
    return registered


def test_both_entry_points_are_registered():
    registered = _registered_architectures()

    assert registered["MuseSpark1xForCausalLM"] == "vllm_ascend.models.muse_spark:AscendMuseSpark1xForCausalLM"
    assert (
        registered["MuseSparkForConditionalGeneration"]
        == "vllm_ascend.models.muse_spark:AscendMuseSparkForConditionalGeneration"
    )


def test_registration_is_skipped_without_upstream_package():
    with patch.object(models, "_upstream_has_module", return_value=False):
        registered = _registered_architectures()

    assert "MuseSpark1xForCausalLM" not in registered
    assert "MuseSparkForConditionalGeneration" not in registered


def _post_expert_norm(hidden_size: int) -> nn.Module:
    norm = muse_spark.AscendMuseSpark1xRMSNormWithInputScale(
        hidden_size=hidden_size,
        eps=EPS,
        has_weight=True,
        is_post_norm=True,
        zero_centered_gamma=True,
        post_norm_gain_center_type="one",
    )
    with torch.no_grad():
        norm.weight.normal_(0, 0.1)
    return norm


def test_reference_experts_normalize_each_route_before_the_weighted_sum():
    torch.manual_seed(0)
    num_experts, hidden_size, intermediate_size, top_k = 4, 6, 5, 2
    experts = muse_spark.MuseSparkReferenceExperts(num_experts, hidden_size, intermediate_size, torch.float32)
    with torch.no_grad():
        experts.w13_weight.normal_()
        experts.w2_weight.normal_()
    norm = _post_expert_norm(hidden_size)
    hidden_states = torch.randn(NUM_TOKENS, hidden_size)
    topk_ids = torch.stack([torch.randperm(num_experts)[:top_k] for _ in range(NUM_TOKENS)]).to(torch.int32)
    topk_weights = torch.rand(NUM_TOKENS, top_k)

    output = experts(hidden_states, topk_weights, topk_ids, norm)

    expected = torch.zeros(NUM_TOKENS, hidden_size)
    unnormalized = torch.zeros(NUM_TOKENS, hidden_size)
    for token in range(NUM_TOKENS):
        for route in range(top_k):
            expert = int(topk_ids[token, route])
            gate, up = (experts.w13_weight[expert] @ hidden_states[token]).chunk(2)
            row = experts.w2_weight[expert] @ (F.silu(gate) * up)
            expected[token] += topk_weights[token, route] * norm.forward_native(row, out_dtype=torch.float32)
            unnormalized[token] += topk_weights[token, route] * row
    torch.testing.assert_close(output, expected)

    # A standard MoE combine followed by one norm is a different function.
    norm_after_sum = norm.forward_native(unnormalized, out_dtype=torch.float32)
    assert not torch.allclose(output, norm_after_sum)


class _Container(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = MuseSpark1xRMSNorm(HEAD_DIM, eps=EPS, has_weight=True, zero_centered_gamma=True)
        self.scaled_norm = MuseSpark1xRMSNormWithInputScale(HEAD_DIM, eps=EPS, has_weight=True)


def test_adopt_ascend_modules_moves_norms_to_the_torch_reference():
    container = _Container()
    with torch.no_grad():
        container.norm.weight.normal_()
        container.scaled_norm.weight.normal_()
    weights = (container.norm.weight, container.scaled_norm.weight)

    muse_spark.adopt_ascend_modules(container)

    assert type(container.norm) is muse_spark.AscendMuseSpark1xRMSNorm
    assert type(container.scaled_norm) is muse_spark.AscendMuseSpark1xRMSNormWithInputScale
    assert (container.norm.weight, container.scaled_norm.weight) == weights
    x = torch.randn(NUM_TOKENS, HEAD_DIM)
    with patch.object(upstream_layers, "fused_rmsnorm_fwd", side_effect=AssertionError("fused kernel called")):
        for norm in (container.norm, container.scaled_norm):
            assert not norm.supports_fused_kernel(x)
            torch.testing.assert_close(norm(x), norm.forward_native(x))


def _attention(rotary_emb=None) -> muse_spark.AscendMuseSpark1xAttention:
    attention = object.__new__(MuseSpark1xAttention)
    nn.Module.__init__(attention)
    attention.num_heads = NUM_HEADS
    attention.num_kv_heads = NUM_HEADS
    attention.head_dim = HEAD_DIM
    attention.q_size = NUM_HEADS * HEAD_DIM
    attention.kv_size = NUM_HEADS * HEAD_DIM
    attention.rms_norm_eps = EPS
    attention.use_o_gate = True
    attention.fuse_o_norm_gate = True
    attention.use_attn_temperature_tuning = False
    attention.qk_norm = MuseSpark1xRMSNorm(HEAD_DIM, eps=EPS, has_weight=False)
    attention.rotary_emb = rotary_emb
    attention.fused_qk_norm_rope_kv_insert = True
    attention.attn = SimpleNamespace(kv_cache_update_bypass=True)
    muse_spark.adopt_ascend_modules(attention)
    return attention


def test_attention_hands_the_kv_cache_write_back_to_the_backend():
    attention = _attention()

    assert type(attention) is muse_spark.AscendMuseSpark1xAttention
    assert type(attention.qk_norm) is muse_spark.AscendMuseSpark1xRMSNorm
    assert attention.fused_qk_norm_rope_kv_insert is False
    assert attention.attn.kv_cache_update_bypass is False


def test_qk_norm_and_rope_run_in_fp32_and_round_once():
    rope_inputs: list[torch.Tensor] = []

    def forward_native(positions, query, key):
        rope_inputs.extend((query, key))
        return query + 1, key + 1

    attention = _attention(rotary_emb=SimpleNamespace(forward_native=forward_native))
    qkv = torch.randn(NUM_TOKENS, 4 * NUM_HEADS * HEAD_DIM, dtype=torch.bfloat16)

    q, k, v, og = attention._qk_norm_rope_kv_insert(qkv, torch.arange(NUM_TOKENS))

    # The upstream kernel rounds once at the store, so RoPE must see the fp32
    # norm output itself, not a bf16 copy of it.
    q_in, k_in, _, _ = qkv.split(NUM_HEADS * HEAD_DIM, dim=-1)
    q_norm, k_norm = (
        attention.qk_norm.forward_native(x.reshape(-1, HEAD_DIM), out_dtype=torch.float32).reshape(NUM_TOKENS, -1)
        for x in (q_in, k_in)
    )
    assert len(rope_inputs) == 2
    assert torch.equal(rope_inputs[0], q_norm)
    assert torch.equal(rope_inputs[1], k_norm)
    assert torch.equal(q, (q_norm + 1).to(torch.bfloat16))
    assert torch.equal(k, (k_norm + 1).to(torch.bfloat16))
    assert q.dtype == k.dtype == v.dtype == torch.bfloat16
    assert og is not None and og.shape == (NUM_TOKENS, NUM_HEADS * HEAD_DIM)


def _torch_rms_norm_gated(x, g, weight, bias, activation, eps):
    x32 = x.float()
    normed = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return (normed * torch.sigmoid(g.float())).to(x.dtype)


def test_output_gate_uses_the_weightless_ascend_gated_norm():
    attention = _attention()
    attention.qkv_proj = lambda hidden: (hidden, None)
    attention.attn = lambda q, k, v: q
    attention.o_proj = lambda x: (x, None)
    hidden = torch.randn(NUM_TOKENS, 4 * NUM_HEADS * HEAD_DIM)
    calls = []

    def recording_gated_norm(x, g, weight, bias, activation, eps):
        calls.append((weight, bias, activation, g.is_contiguous()))
        return _torch_rms_norm_gated(x, g, weight, bias, activation, eps)

    with patch.object(muse_spark, "rms_norm_gated", side_effect=recording_gated_norm):
        output = attention(torch.arange(NUM_TOKENS), hidden)

    assert calls == [(None, None, "sigmoid", True)]
    q, _, _, og = attention._qk_norm_rope_kv_insert(hidden, torch.arange(NUM_TOKENS))
    assert og is not None
    shape = (NUM_TOKENS, NUM_HEADS, HEAD_DIM)
    expected = _torch_rms_norm_gated(q.reshape(shape), og.reshape(shape), None, None, "sigmoid", EPS)
    torch.testing.assert_close(output, expected.reshape(NUM_TOKENS, -1))


def test_quantized_checkpoints_are_rejected_before_construction():
    model = object.__new__(muse_spark.AscendMuseSpark1xForCausalLM)
    vllm_config = SimpleNamespace(quant_config=SimpleNamespace(get_name=lambda: "ascend"))

    with pytest.raises(NotImplementedError, match="unquantized"):
        muse_spark.AscendMuseSpark1xForCausalLM.__init__(model, vllm_config=vllm_config)


def test_reference_moe_rejects_expert_parallelism():
    vllm_config = SimpleNamespace(model_config=SimpleNamespace(hf_text_config=SimpleNamespace()))
    ep_group = SimpleNamespace(device_group=SimpleNamespace(size=lambda: 2))

    with (
        patch.object(muse_spark, "get_ep_group", return_value=ep_group),
        pytest.raises(NotImplementedError, match="expert parallelism"),
    ):
        muse_spark.AscendMuseSpark1xMoE(vllm_config, layer_idx=1)
