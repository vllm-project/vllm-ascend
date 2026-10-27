# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from torch import nn

from vllm_ascend.models import kimi_k3
from vllm_ascend.models.glm5next import attention as glm5next
from vllm_ascend.patch.worker import patch_deepseek_v2 as deepseek


@pytest.mark.parametrize("module", [deepseek, glm5next, kimi_k3], ids=["deepseek", "glm5next", "kimi_k3"])
@pytest.mark.parametrize("factor", [1.0, 2.0, 36.0])
@pytest.mark.parametrize("mscale_all_dim", [0.0, 1.0])
@pytest.mark.parametrize(
    "overrides,expected_rope",
    [
        ({}, "deepseek_yarn"),
        ({"attention_factor": None}, "deepseek_yarn"),
        ({"attention_factor": 1.0}, "deepseek_llama_scaling"),
        ({"attention_factor": 1.0, "apply_yarn_scaling": True}, "deepseek_llama_scaling"),
        ({"attention_factor": 1.2, "apply_yarn_scaling": False}, "deepseek_yarn"),
        ({"rope_type": "default", "attention_factor": 1.0}, "default"),
    ],
)
def test_mla_yarn_attention_scale(monkeypatch, module, factor, mscale_all_dim, overrides, expected_rope):
    # Mock projections and device/distributed setup, but run each real constructor.
    # Check the scale reaching the attention backend, not only the selected RoPE.
    base_scale = 256**-0.5
    rope_parameters = {
        "rope_type": "yarn",
        "factor": factor,
        "mscale_all_dim": mscale_all_dim,
        **overrides,
    }
    config = SimpleNamespace(rms_norm_eps=1e-5, rope_parameters=rope_parameters, max_position_embeddings=8192)
    get_rope = Mock(return_value=nn.Identity())
    monkeypatch.setattr(module, "get_rope", get_rope)
    kwargs = dict(
        config=config,
        hidden_size=32,
        num_heads=4,
        qk_nope_head_dim=192,
        qk_rope_head_dim=64,
        v_head_dim=8,
        q_lora_rank=16,
        kv_lora_rank=8,
        prefix="model.layers.0.self_attn",
    )
    if module is kimi_k3:
        backend = SimpleNamespace()

        def init_upstream(self, **kwargs):
            nn.Module.__init__(self)
            self.scaling = base_scale
            self.mla_attn = nn.Module()
            self.mla_attn.mla_attn = SimpleNamespace(impl=backend)

        monkeypatch.setattr(kimi_k3.UpstreamKimiMLAAttention, "__init__", init_upstream)
        attention = kimi_k3.AscendKimiMLAAttention(**kwargs, use_output_gate=False, use_rope=True)
        backend_scale = backend.scale
    else:
        for name in ("DeepSeekV2FusedQkvAProjLinear", "ColumnParallelLinear", "RowParallelLinear", "RMSNorm"):
            monkeypatch.setattr(module, name, lambda *args, **kwargs: nn.Identity())
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 1)
        wrapper = Mock(return_value=nn.Module())
        wrapper.return_value.mla_attn = SimpleNamespace()
        monkeypatch.setattr(module, "MultiHeadLatentAttentionWrapper", wrapper)
        if module is deepseek:
            attention = nn.Module()
            deepseek._deepseek_v2_mla_attention_init(attention, vllm_config=SimpleNamespace(), **kwargs)
        else:
            config.index_topk = None
            attention = glm5next.Glm5NextMLAAttention(vllm_config=SimpleNamespace(), **kwargs)
        backend_scale = wrapper.call_args.args[2]

    expected_scale = base_scale
    if expected_rope == "deepseek_yarn" and factor > 1:
        expected_scale *= (1.0 + 0.1 * mscale_all_dim * math.log(factor)) ** 2
    assert get_rope.call_args.kwargs["rope_parameters"]["rope_type"] == expected_rope
    assert attention.scaling == pytest.approx(expected_scale)
    assert backend_scale == pytest.approx(expected_scale)
