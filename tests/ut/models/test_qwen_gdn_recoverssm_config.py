# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration tests can run without a vLLM or NPU installation."""

import runpy
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

helpers = runpy.run_path(str(Path(__file__).parents[3] / "vllm_ascend/models/qwen_gdn_recoverssm.py"))


def make_config():
    return NS(
        model_config=NS(
            dtype=torch.bfloat16,
            hf_text_config=NS(
                model_type="qwen3_5_moe_text",
                linear_num_key_heads=16,
                linear_num_value_heads=32,
                linear_key_head_dim=128,
                linear_value_head_dim=128,
                linear_conv_kernel_dim=4,
            ),
        ),
        speculative_config=NS(method="mtp", num_speculative_tokens=4),
        use_v2_model_runner=True,
        cache_config=NS(
            use_replayssm=True,
            mamba_cache_mode="none",
            enable_prefix_caching=False,
            mamba_ssm_cache_dtype="auto",
            mamba_cache_dtype="auto",
        ),
        mamba_config=NS(enable_stochastic_rounding=False),
        parallel_config=NS(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        kv_transfer_config=None,
    )


@pytest.mark.parametrize("tp", [1, 2, 4, 8, 16])
def test_target_geometry_and_model_layer_cache_agree(tp):
    config = make_config()
    config.parallel_config.tensor_parallel_size = tp
    helpers["validate_qwen_gdn_recoverssm"](config)
    original = ((8192 // tp, 7), (32 // tp, 128, 128))
    expected = (*original, (32 // tp, 5, 128), (16 // tp, 5, 128), (32 // tp, 5))

    class BaseModel:
        @classmethod
        def get_mamba_state_shape_from_config(cls, config):
            return original

        @classmethod
        def get_mamba_state_dtype_from_config(cls, config):
            return (torch.bfloat16, torch.float32)

    class Model(helpers["QwenGDNRecoverSSMModelMixin"], BaseModel):
        pass

    assert Model.get_mamba_state_shape_from_config(config) == expected
    assert helpers["append_gdn_record_shapes"](original, 16 // tp, 5) == expected
    assert Model.get_mamba_state_dtype_from_config(config) == (torch.bfloat16,) + (torch.float32,) * 4
    config.cache_config.use_replayssm = False
    assert Model.get_mamba_state_shape_from_config(config) == original
    assert Model.get_mamba_state_dtype_from_config(config) == (torch.bfloat16, torch.float32)


@pytest.mark.parametrize(
    "section,field,value,message",
    [
        ("speculative_config", "method", "eagle3", "requires MTP"),
        ("speculative_config", "num_speculative_tokens", 0, "requires MTP"),
        (None, "speculative_config", None, "requires MTP"),
        (None, "use_v2_model_runner", False, "Runner V2"),
        ("cache_config", "enable_prefix_caching", True, "prefix caching disabled"),
        ("cache_config", "mamba_cache_mode", "align", "cache mode none"),
        ("cache_config", "mamba_ssm_cache_dtype", "bfloat16", "FP32 SSM"),
        ("cache_config", "mamba_cache_dtype", "float32", "BF16 convolution"),
        ("model_config", "dtype", torch.float16, "BF16 activations"),
        ("mamba_config", "enable_stochastic_rounding", True, "stochastic rounding"),
        ("parallel_config", "tensor_parallel_size", 3, "divide 16"),
        ("parallel_config", "pipeline_parallel_size", 2, "PP=PCP=DCP=1"),
        ("parallel_config", "prefill_context_parallel_size", 2, "PP=PCP=DCP=1"),
        ("parallel_config", "decode_context_parallel_size", 2, "PP=PCP=DCP=1"),
        (None, "kv_transfer_config", NS(), "KV transfer"),
    ],
)
def test_rejects_unsupported_execution_before_allocation(section, field, value, message):
    config = make_config()
    setattr(getattr(config, section) if section else config, field, value)
    with pytest.raises(ValueError, match=message):
        helpers["validate_qwen_gdn_recoverssm"](config)


def test_target_selector_does_not_enable_other_architectures():
    config = make_config()
    for model_type in ("qwen3_next", "qwen3_5_text", "glm5_next_text", "kimi_k3"):
        config.model_config.hf_text_config.model_type = model_type
        assert not helpers["use_qwen_gdn_recoverssm"](config)
    config.model_config.hf_text_config.model_type = "qwen3_5_moe_text"
    config.model_config.hf_text_config.linear_num_key_heads = 8
    with pytest.raises(ValueError, match="geometry"):
        helpers["validate_qwen_gdn_recoverssm"](config)
