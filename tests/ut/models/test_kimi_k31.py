# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn
from transformers import LlamaConfig
from vllm.config import ModelConfig, VllmConfig, get_current_vllm_config, set_current_vllm_config
from vllm.model_executor.model_loader import utils as loader_utils
from vllm.model_executor.models.utils import init_vllm_registered_model
from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig

from vllm_ascend.attention.indexer import AscendSFAIndexerBackend
from vllm_ascend.attention.sfa_v1 import AscendSFAImpl, PreprocessType
from vllm_ascend.models import register_model
from vllm_ascend.models.kimi_k3.kimi_k3 import AscendKimiMLAAttention
from vllm_ascend.ops import rotary_embedding
from vllm_ascend.patch.platform.patch_kimi_k31_config import KimiK31Config, KimiK31TextConfig
from vllm_ascend.quantization.methods import AscendW8A8MXFP8DynamicLinearMethod


def text_config(**changes):
    config = dict(
        num_hidden_layers=5,
        linear_attn_config={"kda_layers": [1, 2, 3], "full_attn_layers": [4, 5]},
        dsa_config={
            "dsa_layers": [4, 5],
            "index_topk": 2048,
            "index_num_heads": 32,
            "index_head_dim": 128,
            "index_rope_dim": 0,
            "index_k_norm_type": "layernorm",
        },
    )
    config.update(changes)
    return config


def test_nested_dsa_config_roundtrip_and_layer_numbering():
    config = KimiK31Config(text_config=text_config())
    text = config.text_config
    assert isinstance(text, KimiK31TextConfig)
    assert text.index_n_heads == 32
    assert text.index_topk == 2048
    assert text.index_rope_dim == 0
    assert text.is_kda_layer(0)
    assert not text.is_dsa_layer(0)
    assert text.is_dsa_layer(3)
    assert text.is_dsa_layer(4)
    restored = KimiK31Config(**config.to_dict())
    assert restored.text_config.dsa_config == text.dsa_config
    assert restored.text_config.is_dsa_layer(4)


@pytest.mark.parametrize("inner_kind", ["k31_nope", "k31_rope", "kimi_linear", "llama"])
def test_sfa_k31_identity_in_registered_inner_model_context(tmp_path, monkeypatch, inner_kind):
    # Exercise the real multimodal -> text handoff. Only the allocated model
    # class and unrelated device/distributed services are replaced.
    register_model()
    # Keep inspection in the CPU test process, where device dependencies are mocked.
    monkeypatch.setattr("vllm.model_executor.models.registry._run_in_subprocess", lambda fn: fn())
    config = KimiK31Config(
        architectures=["KimiK31ForConditionalGeneration"],
        text_config=text_config(mla_use_nope=inner_kind != "k31_rope"),
    )
    config.save_pretrained(tmp_path)
    outer = VllmConfig(
        model_config=ModelConfig(
            model=str(tmp_path),
            tokenizer_mode="kimi_k3",
            dtype="bfloat16",
            max_model_len=2048,
            limit_mm_per_prompt={"image": 0, "video": 0},
        )
    )
    ascend_config = SimpleNamespace(enable_sparse_sfa_c8=False, enable_sparse_sfa_turboquant=False, enable_mlapo=False)
    monkeypatch.setattr("vllm_ascend.attention.sfa_v1.get_ascend_config", lambda: ascend_config)
    monkeypatch.setattr("vllm_ascend.attention.sfa_v1.get_tensor_model_parallel_world_size", lambda: 8)
    monkeypatch.setattr("vllm_ascend.attention.sfa_v1.is_pcp_decode_sharding_enabled", lambda _: False)
    monkeypatch.setattr("vllm_ascend.attention.sfa_v1.is_rl_weight_update_enabled", lambda _: False)
    monkeypatch.setattr("vllm_ascend.attention.sfa_v1.enable_sp", lambda: False)

    def make_impl():
        scheme = AscendW8A8MXFP8DynamicLinearMethod.__new__(AscendW8A8MXFP8DynamicLinearMethod)
        impl = AscendSFAImpl(
            num_heads=12,
            head_size=576,
            scale=0.1,
            num_kv_heads=1,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
            logits_soft_cap=None,
            attn_type=None,
            kv_sharing_target_layer_name=None,
            q_lora_rank=1536,
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            qk_head_dim=192,
            v_head_dim=128,
            q_b_proj=SimpleNamespace(_chunk_size=0),
            fused_qkv_a_proj=SimpleNamespace(quant_method=SimpleNamespace(quant_method=scheme)),
            kv_b_proj=object(),
            o_proj=object(),
            kv_a_layernorm=object(),
            q_a_layernorm=object(),
            indexer=SimpleNamespace(head_dim=128, wq_b=SimpleNamespace(weight_scale=torch.ones(1))),
            layer_name="language_model.model.layers.3.self_attn",
        )
        # DCP supplies this group; no distributed process is needed here.
        impl.dcp_group = object()
        impl.enable_sparse_sfa_c8 = True
        return impl

    class CaptureInnerModel(nn.Module):
        def __init__(self, vllm_config, prefix=""):
            super().__init__()
            assert get_current_vllm_config() is vllm_config
            self.impl = make_impl()

    monkeypatch.setattr(loader_utils, "get_model_architecture", lambda _: (CaptureInnerModel, "CaptureInnerModel"))
    inner_hf = outer.model_config.hf_config.text_config
    architectures = ["KimiLinearForCausalLM"]
    if inner_kind == "kimi_linear":
        inner_hf = KimiLinearConfig(**inner_hf.to_dict())
        # An instance field must not impersonate the dedicated config class.
        inner_hf.is_kimi_k31 = True
    elif inner_kind == "llama":
        inner_hf = LlamaConfig(architectures=["LlamaForCausalLM"])
        architectures = ["LlamaForCausalLM"]

    with set_current_vllm_config(outer):
        outer_impl = make_impl()
        assert outer_impl.is_kimi_k31
        assert outer_impl.use_rope is (inner_kind == "k31_rope")
        inner = init_vllm_registered_model(
            vllm_config=outer,
            hf_config=inner_hf,
            architectures=architectures,
            prefix="language_model",
        )
        assert get_current_vllm_config() is outer
    assert outer.model_config.hf_config.model_type == "kimi_k31"
    expected_k31 = inner_kind.startswith("k31_")
    assert inner.impl.is_kimi_k31 is expected_k31
    assert inner.impl.use_rope is (inner_kind != "k31_nope")
    with patch.object(inner.impl, "_try_enable_type", return_value=True):
        # Quantized SFA selects PROLOG_V3 independently of the model family.
        assert inner.impl._resolve_preprocess_type(torch.bfloat16) == PreprocessType.PROLOG_V3
    if inner_kind == "k31_nope":
        tail = torch.randn(2, 12, 64, dtype=torch.bfloat16)
        with patch("torch_npu.npu_interleave_rope", create=True) as rope:
            assert inner.impl.rope_single(tail, None, None) is tail
        rope.assert_not_called()


@pytest.mark.parametrize("layers", [[0, 4], [4, 6], [3, 4]])
def test_invalid_dsa_layer_layout_is_rejected(layers):
    values = text_config()
    values["dsa_config"]["dsa_layers"] = layers
    with pytest.raises(ValueError):
        KimiK31TextConfig(**values)


class TupleLinear(nn.Linear):
    def forward(self, x):
        return super().forward(x), None


@pytest.mark.parametrize("has_triton", [True, False])
def test_no_rope_indexer_preserves_layernorm_keys_and_head_weights(has_triton):
    backend = AscendSFAIndexerBackend.__new__(AscendSFAIndexerBackend)
    nn.Module.__init__(backend)
    backend.head_dim = 8
    backend.qk_rope_head_dim = 0
    backend.wk_weights_proj = TupleLinear(6, 11, bias=False)
    backend.k_norm = nn.LayerNorm(8)
    backend.enable_sparse_li_c8 = False
    backend.enable_sparse_li_c4 = False
    x = torch.randn(4, 6)
    projected = backend.wk_weights_proj(x)[0]
    with patch("vllm_ascend.attention.indexer.HAS_TRITON", has_triton):
        keys, scales, weights = backend.forward_k(x, None, None)
    torch.testing.assert_close(keys[:, 0], backend.k_norm(projected[:, :8]))
    torch.testing.assert_close(weights, projected[:, 8:])
    assert scales is None


@pytest.mark.parametrize("use_rope", [True, False])
def test_rope_metadata_lookup_does_not_require_current_config(use_rope):
    cos_buffer = torch.ones(4, 1, 1, 64)
    sin_buffer = torch.zeros_like(cos_buffer)
    cos_cache = torch.randn(4, 64)
    sin_cache = torch.randn(4, 64)
    positions = torch.tensor([2, 1])
    with (
        patch.object(rotary_embedding, "_cos_mla", cos_buffer),
        patch.object(rotary_embedding, "_sin_mla", sin_buffer),
        patch.object(rotary_embedding, "_cos_cache", cos_cache),
        patch.object(rotary_embedding, "_sin_cache", sin_cache),
        patch.object(rotary_embedding, "get_current_vllm_config", side_effect=AssertionError("No forward config")),
    ):
        cos, sin = rotary_embedding.get_cos_and_sin_mla(positions, use_cache=False, use_rope=use_rope)
    expected_cos = cos_cache[positions, None, None] if use_rope else cos_buffer[:2]
    expected_sin = sin_cache[positions, None, None] if use_rope else sin_buffer[:2]
    torch.testing.assert_close(cos, expected_cos)
    torch.testing.assert_close(sin, expected_sin)
    if not use_rope:
        assert cos.data_ptr() == cos_buffer.data_ptr()
        assert sin.data_ptr() == sin_buffer.data_ptr()


def test_kimi_attention_cache_proxy_supports_v2_cleanup_assignment():
    attention = AscendKimiMLAAttention.__new__(AscendKimiMLAAttention)
    nn.Module.__init__(attention)
    inner = SimpleNamespace(kv_cache=[torch.zeros(1)])
    attention.mla_attn = SimpleNamespace(mla_attn=inner)
    replacement: list[torch.Tensor] = []
    attention.kv_cache = replacement
    assert inner.kv_cache is replacement
    assert attention.kv_cache is replacement
