from types import SimpleNamespace as Ns

import pytest
import torch
from torch import nn
from vllm.config import CacheConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.model_executor.layers.linear import UnquantizedLinearMethod

from vllm_ascend.attention import mla_v1
from vllm_ascend.device.hardware_profile import HardwareCapability
from vllm_ascend.models.kimi_k3 import kimi_k3, kimi_k3_dspark
from vllm_ascend.ops import mla as mla_ops
from vllm_ascend.quantization import utils as quant_utils
from vllm_ascend.quantization.methods.kv_cache import kv_c8
from vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8 import AscendW8A8MXFP8DynamicLinearMethod as MXFP8


class Linear(nn.Module):
    def __init__(self, input_size, output_size, *, quant_config=None, prefix="", **kwargs):
        super().__init__()
        if isinstance(output_size, list):
            output_size = sum(output_size)
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        if prefix.startswith("target."):
            self.quant_method = Ns(quant_method=object.__new__(MXFP8))
        else:
            self.quant_method = UnquantizedLinearMethod()
        self.quant_config = quant_config


def norm(hidden_size, eps):
    return Ns(weight=torch.ones(hidden_size), variance_epsilon=eps)


@pytest.fixture(params=[1, 8])
def construction(monkeypatch, request):
    import vllm.model_executor.layers.attention.mla_attention as attention
    import vllm.models.kimi_k3.amd.linear as upstream

    hf = Ns(model_type="kimi_linear", num_hidden_layers=1, kv_lora_rank=512, qk_rope_head_dim=64)
    config = Ns(
        additional_config={"mla_cache_mode": "C8"},
        model_config=Ns(runner_type="generate", dtype=torch.bfloat16, hf_config=hf, hf_text_config=hf, use_mla=True),
        parallel_config=Ns(
            prefill_context_parallel_size=1,
            decode_context_parallel_size=request.param,
            cp_kv_cache_interleave_size=1664,
        ),
        cache_config=CacheConfig(cache_dtype="auto", block_size=1664),
        compilation_config=Ns(static_forward_context={}, cudagraph_mode=CUDAGraphMode.NONE),
        speculative_config=Ns(method="dspark"),
        kv_transfer_config=None,
        quant_config=Ns(enable_fa_quant=True, enabling_fa_quant=lambda c, prefix, **kw: prefix.startswith("target.")),
        attention_config=Ns(mla_prefill_backend=None),
        kernel_config=Ns(enable_jit_warmup=False),
        use_v2_model_runner=True,
    )
    options = Ns(mla_cache_mode="C8", enable_kv_nz=False)
    hardware = Ns(
        supports=lambda capability: capability in (HardwareCapability.FP8_ATTENTION, HardwareCapability.MLA_FLASH)
    )
    for module in (mla_v1, mla_ops, attention, kv_c8):
        monkeypatch.setattr(module, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(mla_v1, "get_ascend_config", lambda: options)
    for module in (mla_v1, kv_c8):
        monkeypatch.setattr(module, "get_current_hardware_profile", lambda: hardware)
    monkeypatch.setattr(quant_utils, "get_current_hardware_profile", lambda: hardware)
    monkeypatch.setattr(mla_v1, "enabling_mlapo", lambda _: False)
    monkeypatch.setattr(mla_v1, "enable_dcp", lambda: False)
    for module in (mla_v1, attention):
        monkeypatch.setattr(module, "get_dcp_group", lambda: Ns(world_size=request.param, rank_in_group=0))
    monkeypatch.setattr(mla_v1, "get_flashmla_ops", lambda: (object(), object()))
    for module in (mla_ops, upstream):
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 1)
    for name in ("MergedColumnParallelLinear", "ColumnParallelLinear", "ReplicatedLinear", "RowParallelLinear"):
        monkeypatch.setattr(upstream, name, Linear)
    for module in (upstream, kimi_k3_dspark):
        monkeypatch.setattr(module, "RMSNorm", norm)
    monkeypatch.setattr(upstream, "KimiK3MultiHeadLatentAttentionWrapper", mla_ops.AscendMultiHeadLatentAttention)
    monkeypatch.setattr(attention, "get_attn_backend", lambda *a, **kw: mla_v1.AscendMLABackend)
    monkeypatch.setattr(attention, "get_mla_prefill_backend", lambda _: lambda **kw: nn.Identity())
    for name in ("_DecodeConcatQuantFP8", "QuantFP8"):
        monkeypatch.setattr(attention, name, lambda **kw: nn.Identity())
    monkeypatch.setattr(kimi_k3, "get_rope", lambda *a, **kw: nn.Identity())
    monkeypatch.setattr(kimi_k3_dspark, "KimiMLP", lambda **kw: nn.Identity())
    monkeypatch.setattr(kimi_k3_dspark, "get_draft_quant_config", lambda _: None)
    return config, options


@pytest.mark.parametrize("role", ["target", "draft_causal", "draft_noncausal"])
def test_real_constructors_isolate_target_c8_from_draft(construction, role):
    config, options = construction
    dimensions = dict(
        hidden_size=16, qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=16, q_lora_rank=16, kv_lora_rank=512
    )
    if role.startswith("draft"):
        hf = Ns(
            **dimensions,
            num_attention_heads=8,
            intermediate_size=32,
            hidden_act="silu",
            rms_norm_eps=1e-5,
            max_position_embeddings=128,
            rope_parameters={"rope_type": "default"},
            full_attention_causal=role == "draft_causal",
        )
        decoder = kimi_k3_dspark.AscendK3DSparkDecoderLayer(
            vllm_config=config, config=hf, layer_idx=0, start_layer_id=1, prefix="draft"
        )
        attention = decoder.self_attn._attention_layer
        assert attention.impl.use_mla_rope and not attention.impl.kimi_c8_mode
        assert attention.impl.use_flashmla is (
            config.parallel_config.decode_context_parallel_size == 1 or role == "draft_noncausal"
        )
        assert attention.non_causal_multi_token_decode == (role == "draft_noncausal")
        assert decoder.self_attn.fused_qkv_a_proj.quant_config is None and attention.quant_config is None
    else:

        def create():
            return kimi_k3.AscendKimiMLAAttention(
                config=Ns(rms_norm_eps=1e-5),
                num_heads=2,
                **dimensions,
                use_output_gate=False,
                use_rope=False,
                cache_config=config.cache_config,
                quant_config=None,
                prefix="target.self_attn",
            )

        attention = create()._attention_layer
        assert attention.impl.kimi_c8_mode and not attention.impl.use_mla_rope
        assert not hasattr(attention, "fa_k")
    assert isinstance(attention.impl, mla_v1.AscendMLAImpl) and not attention.impl.fa_quant_layer
    assert config.cache_config.cache_dtype == "auto" and config.quant_config is not None
