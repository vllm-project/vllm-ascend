# SPDX-License-Identifier: Apache-2.0
# UTs for vllm_ascend/models/step5/model.py -- CPU only.
#
# Every NPU entry point (DeviceOperator fused ops, the DSA impl, the CSA
# ring / sparse-summary cache layers, the platform Attention backend and
# the parallel linear projections) is patched in the module namespace, so
# the tests exercise construction, validation and dispatch logic only.

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
from vllm.model_executor.models.utils import PPMissingLayer
from vllm.v1.kv_cache_interface import FullAttentionSpec

import vllm_ascend.models.step5.model as step5_model
from vllm_ascend.models.step5.model import (
    AscendStep5Attention,
    AscendStep5DSAAttention,
    AscendStep5ForCausalLM,
    AscendStep5MoE,
    AscendStep5RMSNorm,
    _kv_cache_torch_dtype,
    _reshape_step5_fused_expert_weights,
)

# ---------------------------------------------------------------------------
# _kv_cache_torch_dtype
# ---------------------------------------------------------------------------


def _dtype_vc(dtype=torch.bfloat16):
    return SimpleNamespace(model_config=SimpleNamespace(dtype=dtype))


@pytest.mark.parametrize("cache_dtype", ["auto", None, ""])
def test_kv_cache_dtype_auto_falls_back_to_model_dtype(cache_dtype):
    assert _kv_cache_torch_dtype(SimpleNamespace(cache_dtype=cache_dtype), _dtype_vc()) is torch.bfloat16


def test_kv_cache_dtype_without_cache_config():
    assert _kv_cache_torch_dtype(None, _dtype_vc(torch.float16)) is torch.float16


@pytest.mark.parametrize(
    "cache_dtype,expected",
    [
        ("bfloat16", torch.bfloat16),
        ("bf16", torch.bfloat16),
        ("float16", torch.float16),
        ("fp16", torch.float16),
        ("half", torch.float16),
    ],
)
def test_kv_cache_dtype_explicit_names(cache_dtype, expected):
    assert _kv_cache_torch_dtype(SimpleNamespace(cache_dtype=cache_dtype), _dtype_vc()) is expected


def test_kv_cache_dtype_rejects_unsupported_dtype():
    with pytest.raises(ValueError, match="Unsupported step5-ascend KV cache dtype: fp8"):
        _kv_cache_torch_dtype(SimpleNamespace(cache_dtype="fp8"), _dtype_vc())


# ---------------------------------------------------------------------------
# _reshape_step5_fused_expert_weights (ModelSlim MXFP8 2D -> 3D layout)
# ---------------------------------------------------------------------------


def test_reshape_converts_2d_fused_expert_weights_to_stacked_3d():
    tensor = torch.arange(64 * 4, dtype=torch.float32).reshape(64, 4)

    [(name, reshaped)] = list(
        _reshape_step5_fused_expert_weights([("model.layers.2.moe.gate_proj.weight", tensor)], num_experts=16)
    )

    assert name == "model.layers.2.moe.gate_proj.weight"
    assert reshaped.shape == (16, 4, 4)
    assert torch.equal(reshaped.reshape(64, 4), tensor)


@pytest.mark.parametrize("projection", ["gate_proj", "up_proj", "down_proj"])
@pytest.mark.parametrize("suffix", ["weight", "weight_scale", "input_scale"])
def test_reshape_covers_all_expert_projection_suffixes(projection, suffix):
    tensor = torch.arange(32 * 2, dtype=torch.float32).reshape(32, 2)

    [(name, reshaped)] = list(
        _reshape_step5_fused_expert_weights([(f"model.layers.0.moe.{projection}.{suffix}", tensor)], num_experts=16)
    )

    assert reshaped.shape == (16, 2, 2)


@pytest.mark.parametrize(
    "name",
    [
        "model.layers.1.mlp.gate_proj.weight",
        "model.layers.2.moe.share_expert.gate_proj.weight",
        "model.layers.3.moe.experts.0.gate_proj.weight",
    ],
)
def test_reshape_passes_through_non_fused_names(name):
    tensor = torch.arange(64 * 4, dtype=torch.float32).reshape(64, 4)

    [(out_name, out_tensor)] = list(_reshape_step5_fused_expert_weights([(name, tensor)], num_experts=16))

    assert out_name == name
    assert out_tensor.shape == tensor.shape
    assert out_tensor is tensor


def test_reshape_passes_through_when_leading_dim_equals_num_experts():
    tensor = torch.zeros(16, 4)

    [(_, out_tensor)] = list(
        _reshape_step5_fused_expert_weights([("model.layers.0.moe.gate_proj.weight", tensor)], num_experts=16)
    )

    assert out_tensor.shape == (16, 4)
    assert out_tensor is tensor


def test_reshape_passes_through_when_leading_dim_not_divisible():
    tensor = torch.zeros(17, 4)

    [(_, out_tensor)] = list(
        _reshape_step5_fused_expert_weights([("model.layers.0.moe.gate_proj.weight", tensor)], num_experts=16)
    )

    assert out_tensor.shape == (17, 4)


def test_reshape_passes_through_non_2d_tensors():
    stacked = torch.zeros(16, 4, 4)
    vector = torch.zeros(64)

    out = list(
        _reshape_step5_fused_expert_weights(
            [
                ("model.layers.0.moe.gate_proj.weight", stacked),
                ("model.layers.0.moe.gate_proj.weight_scale", vector),
            ],
            num_experts=16,
        )
    )

    assert out[0][1] is stacked
    assert out[1][1] is vector


# ---------------------------------------------------------------------------
# AscendStep5DSAAttention construction / validation
# ---------------------------------------------------------------------------


def _vllm_config(
    dtype=torch.bfloat16,
    cache_dtype="auto",
    spec_config=None,
    max_num_seqs=16,
    max_model_len=4096,
    block_size=128,
):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            dtype=dtype,
            max_model_len=max_model_len,
            hf_config=SimpleNamespace(norm_dtype="fp32"),
        ),
        speculative_config=spec_config,
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        cache_config=SimpleNamespace(cache_dtype=cache_dtype, block_size=block_size),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )


def _sparse_config(**overrides):
    kwargs = dict(
        sparse_indexer_num_heads=8,
        sparse_indexer_num_k_heads=1,
        index_tp_size=1,
        proxy_dim=128,
        sparse_indexer_rope_dim=128,
        topk=16,
        region_block_size=64,
    )
    kwargs.update(overrides)
    return SimpleNamespace(**kwargs)


def _make_dsa(vllm_config, sparse_config, prefix="model.layers.0.self_attn", **overrides):
    kwargs = dict(
        hidden_size=1024,
        num_heads=8,
        num_kv_heads=1,
        head_dim=128,
        max_position=2048,
        rms_norm_eps=1e-6,
        prefix=prefix,
        vllm_config=vllm_config,
        sparse_config=sparse_config,
        norm_dtype=torch.float32,
        # the decoder layer hands the attention a copy of cache_config
        cache_config=vllm_config.cache_config,
    )
    kwargs.update(overrides)
    return AscendStep5DSAAttention(**kwargs)


@pytest.fixture
def dsa_env(monkeypatch):
    built: dict[str, Any] = {}

    def fake_qkv_indexer_linear(*args, **kwargs):
        built["qkv_indexer"] = dict(args=args, kwargs=kwargs)
        return SimpleNamespace(params_dtype=torch.float32)

    def fake_indexer_w(*args, **kwargs):
        built["indexer_w"] = dict(args=args, kwargs=kwargs)
        return SimpleNamespace()

    def fake_row_parallel(*args, **kwargs):
        return SimpleNamespace(_args=(args, kwargs))

    def fake_get_rope(head_size, max_position, rope_parameters, dtype):
        return SimpleNamespace(cos_sin_cache=torch.zeros(8, 2 * head_size), _rope_params=rope_parameters)

    def fake_impl(*args, **kwargs):
        built["impl"] = dict(args=args, kwargs=kwargs)
        impl = SimpleNamespace()
        built["impl_obj"] = impl
        return impl

    def fake_summary(prefix, target_impl, static_forward_context):
        built["summary"] = SimpleNamespace(
            prefix=prefix, target_impl=target_impl, static_forward_context=static_forward_context
        )
        return SimpleNamespace(_prefix=prefix)

    def fake_ring(prefix, main_layer_name, num_speculative_tokens, static_forward_context):
        built["ring"] = SimpleNamespace(
            prefix=prefix,
            main_layer_name=main_layer_name,
            num_speculative_tokens=num_speculative_tokens,
            static_forward_context=static_forward_context,
        )
        return SimpleNamespace(_prefix=prefix)

    def fake_validate(**kwargs):
        built["geometry"] = kwargs

    monkeypatch.setattr(step5_model, "Step5FusedQKVIndexerLinear", fake_qkv_indexer_linear)
    monkeypatch.setattr(step5_model, "Step5SparseIndexerIndexTPLinear", fake_indexer_w)
    monkeypatch.setattr(step5_model, "RowParallelLinear", fake_row_parallel)
    monkeypatch.setattr(step5_model, "get_rope", fake_get_rope)
    monkeypatch.setattr(step5_model, "AscendStep5DSAImpl", fake_impl)
    monkeypatch.setattr(step5_model, "AscendStep5SparseSummaryCacheLayer", fake_summary)
    monkeypatch.setattr(step5_model, "AscendStep5CSATailRing", fake_ring)
    monkeypatch.setattr(step5_model, "_validate_step5_dsa_parallel_geometry", fake_validate)
    monkeypatch.setattr(step5_model, "get_tensor_model_parallel_world_size", lambda: 1)
    return built


def test_dsa_attention_registers_layer_and_builds_components(dsa_env):
    vllm_config = _vllm_config()
    attn = _make_dsa(vllm_config, _sparse_config())

    # the layer registers itself under {prefix}.attn for the KV-cache planner
    assert attn.layer_name == "model.layers.0.self_attn.attn"
    assert vllm_config.compilation_config.static_forward_context[attn.layer_name] is attn
    assert attn.kv_cache_torch_dtype is torch.bfloat16
    assert attn.kv_cache_dtype == "auto"
    assert attn.tp_size == 1

    # fused qkv+indexer GEMM is shaped from the sparse config
    qkv_kwargs = dsa_env["qkv_indexer"]["kwargs"]
    assert qkv_kwargs["indexer_q_output_size"] == 8 * 128
    assert qkv_kwargs["indexer_kv_output_size"] == 1 * 128
    assert qkv_kwargs["gate_output_size"] is None
    assert qkv_kwargs["proxy_dim"] == 128
    assert qkv_kwargs["index_tp_size"] == 1

    # parallel geometry is validated before anything is built
    assert dsa_env["geometry"] == dict(
        total_num_heads=8,
        total_num_kv_heads=1,
        indexer_num_heads=8,
        index_tp_size=1,
        tp_size=1,
    )

    # the DSA impl gets the runtime shapes from the vllm config
    assert dsa_env["impl"]["args"] == (8, 128, 128**-0.5, 1)
    assert dsa_env["impl"]["kwargs"] == dict(
        kv_cache_dtype="auto",
        topk_regions=16,
        region_size=64,
        proxy_dim=128,
        max_num_seqs=16,
        max_model_len=4096,
        block_size=128,
        decode_out_dtype=torch.bfloat16,
        num_speculative_tokens=0,
    )

    # ring/summary layers share the static forward context and the impl
    assert dsa_env["summary"].static_forward_context is vllm_config.compilation_config.static_forward_context
    assert dsa_env["summary"].target_impl is dsa_env["impl_obj"]
    assert dsa_env["ring"].main_layer_name == attn.layer_name
    assert dsa_env["ring"].static_forward_context is vllm_config.compilation_config.static_forward_context
    assert attn.impl is dsa_env["impl_obj"]
    assert attn.impl.csa_ring is attn.csa_tail_ring

    # head_dim 128 does not take the fused QKNorm+RoPE path
    assert attn.use_fused_qknorm_rope is False
    assert attn.rope_cos is None and attn.rope_sin is None
    assert attn._fused_rotary_pairs == 0


def test_dsa_attention_forwards_speculative_token_count(dsa_env):
    vllm_config = _vllm_config(spec_config=SimpleNamespace(method="mtp", num_speculative_tokens=3))
    _make_dsa(vllm_config, _sparse_config())

    assert dsa_env["impl"]["kwargs"]["num_speculative_tokens"] == 3
    assert dsa_env["ring"].num_speculative_tokens == 3

    # a non-MTP spec config contributes nothing
    vllm_config = _vllm_config(spec_config=SimpleNamespace(method="eagle", num_speculative_tokens=3))
    _make_dsa(vllm_config, _sparse_config(), prefix="model.layers.1.self_attn")
    assert dsa_env["impl"]["kwargs"]["num_speculative_tokens"] == 0
    assert dsa_env["ring"].num_speculative_tokens == 0


def test_dsa_attention_fused_qknorm_rope_enabled_at_head_dim_192(dsa_env):
    attn = _make_dsa(_vllm_config(), _sparse_config(), head_dim=192, hidden_size=1536)

    assert attn.use_fused_qknorm_rope is True
    assert attn._fused_rotary_pairs == 192 // 2
    assert attn.rope_cos.shape == (8, 192)
    assert attn.rope_sin.shape == (8, 192)


def test_dsa_attention_rejects_qkv_bias(dsa_env):
    with pytest.raises(ValueError, match="bias-free GEMM"):
        _make_dsa(_vllm_config(), _sparse_config(), qkv_bias=True)


def test_dsa_attention_requires_exactly_one_local_kv_head(dsa_env):
    with pytest.raises(ValueError, match="exactly one local KV head"):
        _make_dsa(_vllm_config(), _sparse_config(), num_kv_heads=2)


def test_dsa_attention_rejects_unsupported_head_dim(dsa_env):
    with pytest.raises(ValueError, match="head_dim must be 128/192"):
        _make_dsa(_vllm_config(), _sparse_config(), head_dim=96)


def test_dsa_attention_rejects_duplicate_layer_name(dsa_env):
    vllm_config = _vllm_config()
    _make_dsa(vllm_config, _sparse_config())
    with pytest.raises(ValueError, match="Duplicate layer name"):
        _make_dsa(vllm_config, _sparse_config())


def test_dsa_attention_get_kv_cache_spec(dsa_env):
    vllm_config = _vllm_config(cache_dtype="fp16")
    attn = _make_dsa(vllm_config, _sparse_config())

    spec = attn.get_kv_cache_spec(vllm_config)

    assert spec == FullAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=128,
        head_size_v=128,
        dtype=torch.float16,
    )


# ---------------------------------------------------------------------------
# AscendStep5Attention (dense sliding layers)
# ---------------------------------------------------------------------------


@pytest.fixture
def dense_attn_env(monkeypatch):
    built: dict[str, Any] = {}

    def fake_linear(*args, **kwargs):
        return SimpleNamespace(params_dtype=torch.float32)

    def fake_attention(*args, **kwargs):
        built["attention"] = dict(args=args, kwargs=kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(step5_model, "QKVParallelLinear", fake_linear)
    monkeypatch.setattr(step5_model, "RowParallelLinear", fake_linear)
    monkeypatch.setattr(step5_model, "ColumnParallelLinear", fake_linear)
    monkeypatch.setattr(
        step5_model,
        "get_rope",
        lambda head_size, max_position, rope_parameters, dtype: SimpleNamespace(
            cos_sin_cache=torch.zeros(8, 2 * head_size)
        ),
    )
    monkeypatch.setattr(step5_model, "Attention", fake_attention)
    monkeypatch.setattr(step5_model, "get_tensor_model_parallel_world_size", lambda: 1)
    return built


def _make_dense(**overrides):
    kwargs = dict(
        hidden_size=1536,
        num_heads=8,
        num_kv_heads=4,
        head_dim=192,
        max_position=2048,
        rms_norm_eps=1e-6,
        prefix="model.layers.1.self_attn",
        norm_dtype=torch.float32,
        sliding_window=512,
        vllm_config=_vllm_config(),
    )
    kwargs.update(overrides)
    return AscendStep5Attention(**kwargs)


def test_dense_attention_enables_fused_qknorm_rope_at_192(dense_attn_env):
    attn = _make_dense()

    assert attn.use_fused_qknorm_rope is True
    assert attn._fused_rotary_pairs == 96
    assert attn.rope_cos.shape == (8, 192)
    assert attn.rope_sin.shape == (8, 192)
    # q/k norms are the Ascend Gemma-backed RMSNorm
    assert isinstance(attn.q_norm, AscendStep5RMSNorm)
    assert isinstance(attn.k_norm, AscendStep5RMSNorm)
    assert attn.q_norm.weight.shape == (192,)

    # the platform Attention layer carries the sliding window and kv heads
    attn_kwargs = dense_attn_env["attention"]["kwargs"]
    assert attn_kwargs["num_kv_heads"] == 4
    assert attn_kwargs["per_layer_sliding_window"] == 512
    assert attn_kwargs["prefix"] == "model.layers.1.self_attn.attn"
    assert dense_attn_env["attention"]["args"][:3] == (8, 192, 192**-0.5)


def test_dense_attention_disables_fused_path_at_other_head_dims(dense_attn_env):
    attn = _make_dense(head_dim=128)

    assert attn.use_fused_qknorm_rope is False
    assert attn._fused_rotary_pairs == 0
    assert attn.rope_cos is None and attn.rope_sin is None


def test_dense_attention_honors_per_layer_rope_disable(dense_attn_env):
    attn = _make_dense(head_dim=192, use_rope_layers=[False, True], prefix="model.layers.0.self_attn")

    assert attn.use_rope is False
    assert attn.use_fused_qknorm_rope is False
    assert attn.rope_cos is None


# ---------------------------------------------------------------------------
# AscendStep5ForCausalLM.update_physical_experts_metadata (EPLB)
# ---------------------------------------------------------------------------


def _fake_moe():
    moe = object.__new__(AscendStep5MoE)
    moe.n_logical_experts = 16
    moe.n_routed_experts = 16
    moe.n_physical_experts = 16
    moe.n_local_physical_experts = 16
    moe.n_redundant_experts = 0
    moe.experts = MagicMock()
    return moe


def _causal_lm_shell(moes):
    model = object.__new__(AscendStep5ForCausalLM)
    model.num_logical_experts = 16
    model.num_routed_experts = 16
    model.num_physical_experts = 16
    model.num_local_physical_experts = 16
    model.num_redundant_experts = 0
    layers = [SimpleNamespace(moe=moe) for moe in moes]
    layers.append(object.__new__(PPMissingLayer))
    layers.append(SimpleNamespace())  # dense layer without a .moe attribute
    model.model = SimpleNamespace(layers=layers)
    return model


def test_update_physical_experts_metadata_rejects_local_change():
    model = _causal_lm_shell([])
    with pytest.raises(ValueError, match="cannot change the number of local physical experts"):
        model.update_physical_experts_metadata(num_physical_experts=24, num_local_physical_experts=8)


def test_update_physical_experts_metadata_updates_moe_layers():
    moe_a, moe_b = _fake_moe(), _fake_moe()
    model = _causal_lm_shell([moe_a, moe_b])

    model.update_physical_experts_metadata(num_physical_experts=20, num_local_physical_experts=16)

    assert model.num_physical_experts == 20
    assert model.num_redundant_experts == 4
    assert model.num_logical_experts == 16
    assert model.num_local_physical_experts == 16
    for moe in (moe_a, moe_b):
        assert moe.n_physical_experts == 20
        assert moe.n_local_physical_experts == 16
        assert moe.n_redundant_experts == 4
        moe.experts.update_expert_map.assert_called_once_with()
