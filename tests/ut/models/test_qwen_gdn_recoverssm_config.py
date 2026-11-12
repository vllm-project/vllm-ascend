# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GDN configuration and model/layer descriptor contract tests."""

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


def base_shapes(config):
    from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator

    hf = config.model_config.hf_text_config
    spec = config.speculative_config
    return MambaStateShapeCalculator.gated_delta_net_state_shape(
        config.parallel_config.tensor_parallel_size,
        hf.linear_num_key_heads,
        hf.linear_num_value_heads,
        hf.linear_key_head_dim,
        hf.linear_value_head_dim,
        hf.linear_conv_kernel_dim,
        (spec.num_speculative_tokens or 0) if spec is not None else 0,
    )


@pytest.mark.parametrize("tp", [1, 2, 4, 8, 16])
def test_target_geometry_and_recoverssm_record_layout_agree(tp):
    config = make_config()
    config.parallel_config.tensor_parallel_size = tp
    helpers["validate_qwen_gdn_recoverssm"](config)
    original = base_shapes(config)
    expected = (*original, (32 // tp, 5, 257))

    assert helpers["gdn_mamba_state_shape_from_config"](config) == expected
    expected_dtypes = (torch.bfloat16, torch.bfloat16, torch.float32)
    assert helpers["gdn_mamba_state_dtype_from_config"](config) == expected_dtypes
    config.cache_config.use_replayssm = False
    assert not helpers["_uses_gdn_recoverssm"](config)


@pytest.mark.parametrize(
    "section,field,value,message",
    [
        ("speculative_config", "num_speculative_tokens", 0, "requires speculative decoding"),
        (None, "speculative_config", None, "requires speculative decoding"),
        (None, "use_v2_model_runner", False, "Runner V2"),
        ("cache_config", "mamba_ssm_cache_dtype", "float16", "BF16 or FP32 SSM"),
        ("cache_config", "mamba_cache_dtype", "float32", "BF16 convolution"),
        ("model_config", "dtype", torch.float16, "BF16 activations"),
        ("mamba_config", "enable_stochastic_rounding", True, "stochastic rounding"),
        ("parallel_config", "tensor_parallel_size", 3, "divide key and value heads"),
        ("parallel_config", "pipeline_parallel_size", 2, "PP=PCP=DCP=1"),
        ("parallel_config", "prefill_context_parallel_size", 2, "PP=PCP=DCP=1"),
        ("parallel_config", "decode_context_parallel_size", 2, "PP=PCP=DCP=1"),
        (None, "kv_transfer_config", NS(kv_connector="AscendStoreConnector"), "requires Mooncake"),
    ],
)
def test_rejects_unsupported_execution_before_allocation(section, field, value, message):
    config = make_config()
    setattr(getattr(config, section) if section else config, field, value)
    with pytest.raises(ValueError, match=message):
        helpers["validate_qwen_gdn_recoverssm"](config)


def test_speculative_methods_share_recoverssm_validation():
    for method in ("mtp", "eagle3", "dspark"):
        config = make_config()
        config.speculative_config.method = method
        helpers["validate_qwen_gdn_recoverssm"](config)


@pytest.mark.parametrize("connector", ["MooncakeConnectorV2", "MooncakePullConnector"])
@pytest.mark.parametrize("role", ["kv_producer", "kv_consumer"])
@pytest.mark.parametrize("method", ["mtp", "dspark", "eagle3"])
def test_pd_configuration_preserves_the_same_state_layout(connector, role, method):
    config = make_config()
    config.speculative_config.method = method
    config.parallel_config.tensor_parallel_size = 4
    config.kv_transfer_config = NS(kv_connector=connector, kv_role=role)
    helpers["validate_qwen_gdn_recoverssm"](config)

    original = base_shapes(config)
    assert helpers["gdn_mamba_state_shape_from_config"](config) == (
        *original,
        (8, 5, 257),
    )
    assert helpers["gdn_mamba_state_dtype_from_config"](config) == (
        torch.bfloat16,
        torch.bfloat16,
        torch.float32,
    )


@pytest.mark.parametrize("connector", ["AscendStoreConnector", "MooncakeLayerwiseConnector", "MultiConnector", None])
def test_pd_rejects_connectors_without_the_supported_state_transfer_contract(connector):
    config = make_config()
    config.kv_transfer_config = NS(kv_connector=connector, kv_role="kv_consumer")
    with pytest.raises(ValueError, match="requires MooncakeConnectorV2 or MooncakePullConnector"):
        helpers["validate_qwen_gdn_recoverssm"](config)


@pytest.mark.parametrize("connector", ["MooncakeConnectorV2", "MooncakePullConnector"])
@pytest.mark.parametrize("role", ["kv_producer", "kv_consumer"])
@pytest.mark.parametrize("cache_mode,prefix_caching", [("none", True), ("align", False), ("align", True)])
def test_pd_requires_none_without_prefix_caching(connector, role, cache_mode, prefix_caching):
    config = make_config()
    config.cache_config.mamba_cache_mode = cache_mode
    config.cache_config.enable_prefix_caching = prefix_caching
    config.kv_transfer_config = NS(kv_connector=connector, kv_role=role)
    with pytest.raises(ValueError, match="PD requires Mamba cache mode none and prefix caching disabled"):
        helpers["validate_qwen_gdn_recoverssm"](config)


@pytest.mark.parametrize("model_type", ["qwen3_next", "qwen3_5_text", "qwen3_5_moe_text", "qwen4_exp_text"])
def test_target_selector_enables_qwen_gdn_architectures(model_type):
    config = make_config()
    config.model_config.hf_text_config.model_type = model_type
    assert helpers["_uses_gdn_recoverssm"](config)
    config.cache_config.use_replayssm = False
    assert not helpers["_uses_gdn_recoverssm"](config)


def test_qwen4_exp_release_config_is_registered_without_widening_selector():
    config = make_config()
    config.model_config.hf_text_config.model_type = "qwen4_exp_text"
    config.model_config.hf_config = NS(architectures=["Qwen4ExpForConditionalGeneration"])

    assert helpers["is_gdn_model"](config)
    assert helpers["_uses_gdn_recoverssm"](config)

    config.model_config.hf_config.architectures = ["CustomQwen4ExpForConditionalGeneration"]
    assert not helpers["is_gdn_model"](config)
    config.model_config.hf_config.architectures = ["Qwen4ExpForConditionalGeneration"]
    config.model_config.hf_text_config.model_type = "custom_qwen4_exp_text"
    assert not helpers["is_gdn_model"](config)


def test_qwen4_exp_mtp_draft_config_is_registered_exactly():
    config = make_config()
    config.model_config.hf_text_config.model_type = "qwen4_exp_text"
    config.model_config.hf_config = NS(architectures=["Qwen4ExpMTP"])

    assert helpers["is_gdn_model"](config)
    assert helpers["_uses_gdn_recoverssm"](config)

    for architecture in (
        "CustomQwen4ExpMTP",
        "Qwen4ExpMTPForConditionalGeneration",
        "Qwen4ExpMTPDerivative",
    ):
        config.model_config.hf_config.architectures = [architecture]
        assert not helpers["is_gdn_model"](config)
        assert not helpers["_uses_gdn_recoverssm"](config)


@pytest.mark.parametrize("model_type", ["glm5_next_text", "kimi_k3", "qwen3"])
def test_target_selector_does_not_enable_models_without_gdn_fields(model_type):
    config = make_config()
    config.model_config.hf_text_config = NS(model_type=model_type)
    assert not helpers["is_gdn_model"](config)
    assert not helpers["_uses_gdn_recoverssm"](config)


def test_target_selector_rejects_unregistered_derivatives():
    config = make_config()
    config.model_config.hf_text_config.model_type = "custom_qwen_derivative"
    assert not helpers["_uses_gdn_recoverssm"](config)
    del config.model_config.hf_text_config.linear_conv_kernel_dim
    assert not helpers["is_gdn_model"](config)
    assert not helpers["_uses_gdn_recoverssm"](config)


def test_align_with_prefix_caching_is_supported():
    config = make_config()
    config.cache_config.mamba_cache_mode = "align"
    config.cache_config.enable_prefix_caching = True
    helpers["validate_qwen_gdn_recoverssm"](config)


@pytest.mark.parametrize("cache_mode,prefix_caching", [("none", True), ("align", False)])
def test_non_pd_rejects_cache_mode_prefix_caching_mismatch(cache_mode, prefix_caching):
    config = make_config()
    config.cache_config.mamba_cache_mode = cache_mode
    config.cache_config.enable_prefix_caching = prefix_caching
    with pytest.raises(ValueError, match="requires Mamba cache mode"):
        helpers["validate_qwen_gdn_recoverssm"](config)


@pytest.mark.parametrize("tp", [1, 4])
def test_state_helpers_append_records_only_when_active(tp):
    config = make_config()
    config.parallel_config.tensor_parallel_size = tp
    original = base_shapes(config)
    expected = (*original, (32 // tp, 5, 257))
    assert helpers["gdn_mamba_state_shape_from_config"](config) == expected
    assert helpers["gdn_mamba_state_dtype_from_config"](config) == (
        torch.bfloat16,
        torch.bfloat16,
        torch.float32,
    )
    config.cache_config.use_replayssm = False
    assert helpers["gdn_mamba_state_shape_from_config"](config) == original
    assert helpers["gdn_mamba_state_dtype_from_config"](config) == (
        torch.bfloat16,
        torch.bfloat16,
    )


@pytest.mark.parametrize(
    "ssm_dtype,expected", [("auto", torch.bfloat16), ("bfloat16", torch.bfloat16), ("float32", torch.float32)]
)
def test_checkpoint_dtype_is_preserved(ssm_dtype, expected):
    config = make_config()
    config.cache_config.mamba_ssm_cache_dtype = ssm_dtype
    helpers["validate_qwen_gdn_recoverssm"](config)
    assert helpers["gdn_mamba_state_dtype_from_config"](config) == (
        torch.bfloat16,
        expected,
        torch.float32,
    )


@pytest.mark.parametrize("architecture", sorted(helpers["_GDN_ARCHITECTURES"]))
def test_selector_accepts_only_registered_architectures(architecture):
    config = make_config()
    config.model_config.hf_config = NS(architectures=[architecture])
    assert helpers["_uses_gdn_recoverssm"](config)
    config.model_config.hf_config.architectures = ["CustomQwenForCausalLM"]
    assert not helpers["_uses_gdn_recoverssm"](config)


@pytest.mark.parametrize("tp", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("num_spec", [None, 0, 1, 4])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("ssm_dtype", ["auto", "bfloat16", "float32"])
def test_registered_model_and_layer_descriptors_agree(tp, num_spec, enabled, ssm_dtype):
    # Load the same worker patch that supplies descriptors to the V2 GDN layer.
    from vllm.model_executor.layers.mamba.mamba_utils import MambaStateDtypeCalculator
    from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5MoeForCausalLM
    from vllm.model_executor.models.qwen3_next import Qwen3NextForCausalLM

    import vllm_ascend.patch.worker.patch_qwen3_5  # noqa: F401
    from vllm_ascend.attention.gdn import AscendGatedDeltaNetAttention
    from vllm_ascend.models.qwen3_5_vl import (
        AscendQwen3_5ForConditionalGeneration,
        AscendQwen3_5MoeForConditionalGeneration,
    )

    config = make_config()
    config.parallel_config.tensor_parallel_size = tp
    if num_spec is None:
        config.speculative_config = None
    else:
        config.speculative_config.num_speculative_tokens = num_spec
    config.cache_config.use_replayssm = enabled
    config.cache_config.mamba_ssm_cache_dtype = ssm_dtype
    layer = NS(
        model_config=config.model_config,
        cache_config=config.cache_config,
        tp_size=tp,
        num_k_heads=16,
        num_v_heads=32,
        head_k_dim=128,
        head_v_dim=128,
        conv_kernel_size=4,
        num_spec=num_spec or 0,
        use_gdn_recoverssm=helpers["_uses_gdn_recoverssm"](config),
    )
    base_dtypes = MambaStateDtypeCalculator.gated_delta_net_state_dtype(torch.bfloat16, "auto", ssm_dtype)
    active = enabled and (num_spec or 0) > 0
    expected_dtypes = (*base_dtypes, torch.float32) if active else base_dtypes
    expected_shapes = AscendGatedDeltaNetAttention.get_state_shape(layer)
    assert AscendGatedDeltaNetAttention.get_state_dtype(layer) == expected_dtypes
    assert len(expected_shapes) == len(expected_dtypes) == (3 if active else 2)
    if active:
        assert expected_shapes[2] == (32 // tp, num_spec + 1, 257)
    for model_cls, architecture, model_type in (
        (Qwen3_5ForCausalLM, "Qwen3_5ForCausalLM", "qwen3_5_text"),
        (Qwen3_5MoeForCausalLM, "Qwen3_5MoeForCausalLM", "qwen3_5_moe_text"),
        (Qwen3NextForCausalLM, "Qwen3NextForCausalLM", "qwen3_next"),
        (AscendQwen3_5ForConditionalGeneration, "Qwen3_5ForConditionalGeneration", "qwen3_5_text"),
        (AscendQwen3_5MoeForConditionalGeneration, "Qwen3_5MoeForConditionalGeneration", "qwen3_5_moe_text"),
    ):
        config.model_config.hf_config = NS(architectures=[architecture])
        config.model_config.hf_text_config.model_type = model_type
        assert helpers["_uses_gdn_recoverssm"](config) == active
        assert helpers["gdn_mamba_state_shape_from_config"](config) == expected_shapes
        assert helpers["gdn_mamba_state_dtype_from_config"](config) == expected_dtypes
        assert model_cls.get_mamba_state_shape_from_config(config) == expected_shapes
        # Repeated calls must not append another replay view.
        assert model_cls.get_mamba_state_shape_from_config(config) == expected_shapes
        assert model_cls.get_mamba_state_dtype_from_config(config) == expected_dtypes


@pytest.mark.parametrize("num_spec", [None, 0])
def test_no_speculation_keeps_base_descriptors_but_explicit_enable_is_invalid(num_spec):
    config = make_config()
    config.speculative_config = None if num_spec is None else NS(num_speculative_tokens=num_spec)
    assert not helpers["_uses_gdn_recoverssm"](config)
    assert helpers["gdn_mamba_state_shape_from_config"](config) == base_shapes(config)
    assert helpers["gdn_mamba_state_dtype_from_config"](config) == (torch.bfloat16, torch.bfloat16)
    with pytest.raises(ValueError, match="requires speculative decoding"):
        helpers["validate_qwen_gdn_recoverssm"](config)


def test_unsupported_architecture_keeps_base_descriptors():
    config = make_config()
    config.model_config.hf_config = NS(architectures=["CustomQwenForCausalLM"])
    assert helpers["gdn_mamba_state_shape_from_config"](config) == base_shapes(config)
    assert helpers["gdn_mamba_state_dtype_from_config"](config) == (torch.bfloat16, torch.bfloat16)
