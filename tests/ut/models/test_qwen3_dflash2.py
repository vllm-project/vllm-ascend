# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E402
"""Check the Ascend DFlash2 MLA draft's attention settings and context KV path."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

upstream_dflash = pytest.importorskip("vllm.model_executor.models.qwen3_dflash")

from vllm.models.kimi_k3.amd import linear as upstream_kimi
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator

from vllm_ascend.models import qwen3_dflash2 as dflash2
from vllm_ascend.models.kimi_k3 import kimi_k3
from vllm_ascend.ops import mla


def test_grouped_conv_keeps_nonfinite_taps_inside_each_request():
    hidden_states = torch.tensor([[1.0, 1.0], [torch.nan, torch.inf], [2.0, 3.0], [5.0, 7.0]])
    output = dflash2._grouped_conv(
        hidden_states, torch.zeros(4, 2, 1), torch.ones(2, 2), block_size=2, num_groups=1, group_size=2, taps=2
    )
    torch.testing.assert_close(output[2:], torch.tensor([[2.0, 3.0], [7.0, 10.0]]))


@pytest.fixture
def build_mla_layer(monkeypatch):
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(dtype=torch.bfloat16, hf_text_config=SimpleNamespace(num_hidden_layers=2)),
        compilation_config=SimpleNamespace(static_forward_context={}),
        speculative_config=SimpleNamespace(num_speculative_tokens=7),
    )
    monkeypatch.setattr(upstream_dflash, "get_current_vllm_config", lambda: vllm_config)
    monkeypatch.setattr(mla, "get_current_vllm_config", lambda: vllm_config)
    monkeypatch.setattr(mla, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(upstream_kimi, "get_tensor_model_parallel_world_size", lambda: 1)
    for module in ("MergedColumnParallelLinear", "ColumnParallelLinear", "RowParallelLinear", "RMSNorm"):
        monkeypatch.setattr(upstream_kimi, module, nn.Identity)
    monkeypatch.setattr(kimi_k3, "get_rope", lambda *args, **kwargs: nn.Identity())

    # Capture what the registered Kimi wrapper supplies at backend construction.
    def attention_layer(*, sliding_window, dflash_draft_window, **kwargs):
        return SimpleNamespace(
            sliding_window=sliding_window,
            impl=SimpleNamespace(dflash_draft_window=dflash_draft_window),
            process_weights_after_loading=lambda: None,
        )

    monkeypatch.setattr(mla, "AscendMLAAttention", attention_layer)
    for module in ("Qwen2MLP", "RMSNorm", "DFlashGroupedConv"):
        monkeypatch.setattr(dflash2, module, nn.Identity)

    def build(layer_idx=0, dflash_config=None, **fields):
        config = SimpleNamespace(
            rope_parameters={"rope_type": "default"},
            max_position_embeddings=4096,
            hidden_size=8,
            intermediate_size=16,
            hidden_act="silu",
            rms_norm_eps=1e-6,
            q_lora_rank=4,
            kv_lora_rank=4,
            qk_nope_head_dim=2,
            qk_rope_head_dim=2,
            v_head_dim=2,
            num_attention_heads=2,
            num_hidden_layers=2,
            dflash_config={"conv_kernel_size": 2, "conv_group_size": 2, **(dflash_config or {})},
            **fields,
        )
        return dflash2.DFlash2Qwen3MLADecoderLayer(vllm_config, config=config, layer_idx=layer_idx)

    return build


_SLIDING = "sliding_attention"
_FULL = "full_attention"


@pytest.mark.parametrize(
    ("fields", "layer_idx", "window", "causal", "needs_non_causal", "uniform_causal"),
    [
        ({"sliding_window": 1024}, 0, None, False, True, False),
        ({"layer_types": [_SLIDING] * 2, "sliding_window": 1024, "max_window_layers": 1}, 1, 1024, True, False, True),
        ({"layer_types": [_SLIDING, _FULL], "sliding_window": 1024}, 0, 1024, True, True, False),
        ({"layer_types": [_SLIDING, _FULL], "sliding_window": 1024}, 1, None, False, True, False),
        (
            {"layer_types": [_FULL] * 2, "dflash_config": {"use_swa": True, "swa_window_size": 512}},
            0,
            512,
            False,
            True,
            False,
        ),
        ({"layer_types": [_SLIDING] * 2, "sliding_window": 0}, 0, 0, True, False, False),
    ],
)
def test_mla_layer_attention_settings(
    build_mla_layer, fields, layer_idx, window, causal, needs_non_causal, uniform_causal
):
    layer = build_mla_layer(layer_idx, **fields)
    attention = layer.self_attn._attention_layer
    assert attention.sliding_window == window
    assert attention.impl.dflash_draft_window is uniform_causal
    assert attention.non_causal_multi_token_decode is needs_non_causal
    draft = SimpleNamespace(model=SimpleNamespace(layers=[layer]))
    assert dflash2.DFlash2Qwen3ForCausalLM.get_draft_attn_causal(draft) == [causal]


@pytest.fixture
def mla_context_model(monkeypatch):
    class TupleLinear(nn.Linear):
        def forward(self, x):
            return super().forward(x), None

    model = dflash2.DFlash2Qwen3MLAModel.__new__(dflash2.DFlash2Qwen3MLAModel)
    nn.Module.__init__(model)
    model.hidden_norm = nn.LayerNorm(3)
    model.use_aux_hidden_state = False
    model.layers = nn.ModuleList([nn.Module()])
    attn = model.layers[0].self_attn = nn.Module()
    attn._attention_layer = nn.Module()
    attn.fused_qkv_a_proj = TupleLinear(3, 4, bias=False)
    attn.kv_a_layernorm = nn.LayerNorm(2)
    attn.q_lora_rank = 1
    attn.impl = SimpleNamespace(kv_lora_rank=2)
    monkeypatch.setattr(dflash2, "get_cos_and_sin_mla", lambda _: (None, None))
    return model


@pytest.mark.skipif(
    not hasattr(DFlashSpeculator, "free_unused_weights"), reason="Core has no prefill-only weight pruning"
)
@pytest.mark.parametrize("supports_prolog", [False, True])
def test_mla_context_write_after_weight_pruning(mla_context_model, supports_prolog):
    model = mla_context_model
    attn = model.layers[0].self_attn
    attn.kv_cache = torch.zeros(4, 3)
    attn.q_a_layernorm = nn.LayerNorm(1)
    attn.q_b_proj = nn.Linear(1, 2, bias=False)
    attn.o_proj = nn.Linear(2, 3, bias=False)
    attn.impl.supports_prolog_v3_writer = lambda: supports_prolog
    attn.impl._cache_requires_prolog_v3_writer = lambda cache: False

    def write(kv, cos, sin, cache, slots, *, return_bf16_kv=True):
        cache[0][slots] = attn.kv_a_layernorm(kv[:, :2])
        cache[1][slots] = kv[:, 2:]

    attn.impl.exec_kv_prefill = write
    states = torch.tensor([[0.0, 1.0, 3.0], [2.0, 0.0, 1.0]])
    positions, slots = torch.arange(2), torch.tensor([3, 1])
    model.precompute_and_store_context_kv(states, positions, slots)
    expected = attn.kv_cache.clone()
    attn.kv_cache[slots] = torch.nan
    spec = SimpleNamespace(model=model, context_kv_only=True, _speculator_name="dflash")
    DFlashSpeculator.free_unused_weights(spec, nn.Module())
    model.precompute_and_store_context_kv(states, positions, slots)
    torch.testing.assert_close(attn.kv_cache, expected)
    assert attn.o_proj.weight.numel() == 0
    # Cache layout is chosen after pruning. PROLOG_V3 also reads the Q weights.
    assert (attn.q_a_layernorm.weight.numel(), attn.q_b_proj.weight.numel()) == ((1, 2) if supports_prolog else (0, 0))


def test_mla_profiles_context_without_cache(mla_context_model):
    model = mla_context_model
    proj = model.layers[0].self_attn.fused_qkv_a_proj
    states = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    with patch.object(proj, "forward", wraps=proj.forward) as project:
        model.precompute_and_store_context_kv(states, torch.arange(4))
    project.assert_called_once()
    torch.testing.assert_close(project.call_args.args[0], model.hidden_norm(states))
