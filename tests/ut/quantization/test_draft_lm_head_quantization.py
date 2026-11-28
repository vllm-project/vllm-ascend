# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Tests for the Ascend draft LM head quantization formats.

The Ascend configuration declares the formats for the shared-head copy.
Its format selection and factories can be tested on older vLLM pairings.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.device.hardware_profile import DeviceAdaptorFamily
from vllm_ascend.ops.vocab_parallel_embedding import AscendParallelLMHead, VocabParallelMode, VocabParallelPlan
from vllm_ascend.quantization.draft_lm_head import (
    AscendDraftLMHeadQuantizationConfig,
    create_int8_lm_head_method,
    create_mxfp8_lm_head_method,
)
from vllm_ascend.quantization.method_adapters import AscendEmbeddingMethod
from vllm_ascend.quantization.methods.registry import get_scheme_class
from vllm_ascend.quantization.methods.w8a8.w8a8_dynamic import AscendW8A8DynamicLmHeadMethod
from vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8 import (
    AscendW8A8MXFP8DynamicLinearMethod,
    AscendW8A8MXFP8DynamicLmHeadMethod,
)


def spec_runtime():
    return SimpleNamespace(use_v2_model_runner=True, quant_config=None, additional_config={})


def _profile(family):
    return SimpleNamespace(device_adaptor_family=family)


@pytest.mark.parametrize(
    "family,expected_factories",
    [
        (DeviceAdaptorFamily.FP8_OPTIMIZED, {"mxfp8": create_mxfp8_lm_head_method}),
        (DeviceAdaptorFamily.STANDARD, {"int8": create_int8_lm_head_method}),
        (DeviceAdaptorFamily.COMPATIBILITY, {}),
    ],
)
def test_config_formats_follow_hardware_family(family, expected_factories):
    with patch("vllm_ascend.quantization.draft_lm_head.get_current_hardware_profile", return_value=_profile(family)):
        methods = AscendDraftLMHeadQuantizationConfig._draft_lm_head_methods()
    assert methods == {"fp8": None, "nvfp4": None} | {
        name: f"{factory.__module__}.{factory.__name__}" for name, factory in expected_factories.items()
    }


def test_mxfp8_factory_builds_lm_head_method():
    with (
        patch(
            "vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8.get_current_vllm_config",
            return_value=spec_runtime(),
        ),
        patch(
            "vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8.get_dynamic_mx_quant_scale_alg",
            return_value=1,
        ),
    ):
        method = create_mxfp8_lm_head_method()
    assert type(method) is AscendEmbeddingMethod
    assert type(method.quant_method) is AscendW8A8MXFP8DynamicLmHeadMethod
    # Online copies carry no quant description; the MX group size is fixed.
    assert method.quant_method.group_size == 32


def test_int8_factory_builds_lm_head_method():
    method = create_int8_lm_head_method()
    assert type(method) is AscendEmbeddingMethod
    assert type(method.quant_method) is AscendW8A8DynamicLmHeadMethod


@pytest.mark.parametrize(
    "quant_type,layer_type",
    [("W8A8_DYNAMIC", "lm_head"), ("W8A8_MXFP8", "lm_head")],
)
def test_lm_head_schemes_are_not_registered(quant_type, layer_type):
    """The online LmHead schemes are factory-owned, not checkpoint formats:
    registering them would change ModelSlim dispatch for W8A8 checkpoints."""
    assert get_scheme_class(quant_type, layer_type) is None


def test_online_mxfp8_copy_rebuilds_scale_next_to_weight():
    head = torch.nn.Module()
    head.weight = torch.nn.Parameter(torch.arange(24, dtype=torch.float16).view(8, 3), requires_grad=False)
    # The copy's scale buffer is created after the model is already on the
    # device, so nothing moves it; the method must rebuild it next to the
    # quantized weight instead of filling this buffer in place. The sentinel
    # value detects in-place reuse: the pre-fix code shared this storage.
    original_scale = torch.full((8, 1), 7.0)
    head.weight_scale = torch.nn.Parameter(original_scale, requires_grad=False)

    def fake_mx_quant(weight, dst_type, block_size, scale_alg):
        assert dst_type == torch.float8_e4m3fn
        assert block_size == 32
        return weight.to(dst_type), torch.ones(weight.shape[0], 1, dtype=torch.float32)

    with (
        patch("torch_npu.npu_dynamic_mx_quant", side_effect=fake_mx_quant),
        patch.object(AscendW8A8MXFP8DynamicLinearMethod, "process_weights_after_loading", lambda self, layer: None),
        patch(
            "vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8.get_current_vllm_config",
            return_value=spec_runtime(),
        ),
        patch(
            "vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8.get_dynamic_mx_quant_scale_alg",
            return_value=1,
        ),
    ):
        method = AscendW8A8MXFP8DynamicLmHeadMethod()
        method.process_weights_after_loading(head)
    # The scales land in a fresh buffer next to the quantized weight...
    assert head.weight_scale.data.device == head.weight.data.device
    assert head.weight_scale.shape == (8, 1)
    assert head.weight.dtype == torch.float8_e4m3fn
    assert torch.equal(head.weight_scale.data, torch.ones(8, 1))
    # ...and the original (e.g. host-side) buffer is left untouched.
    assert torch.equal(original_scale, torch.full((8, 1), 7.0))


def test_online_mxfp8_copy_is_idempotent():
    head = torch.nn.Module()
    head.weight = torch.nn.Parameter(torch.arange(24, dtype=torch.float16).view(8, 3), requires_grad=False)
    head.weight_scale = torch.nn.Parameter(torch.empty(8, 1), requires_grad=False)
    head._mxfp8_transformed = True
    with (
        patch("torch_npu.npu_dynamic_mx_quant") as quant,
        patch(
            "vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8.get_current_vllm_config",
            return_value=spec_runtime(),
        ),
        patch(
            "vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8.get_dynamic_mx_quant_scale_alg",
            return_value=1,
        ),
    ):
        method = AscendW8A8MXFP8DynamicLmHeadMethod()
        method.process_weights_after_loading(head)
    quant.assert_not_called()
    assert head.weight.dtype == torch.float16


def test_explicit_quant_method_survives_construction():
    # Single-rank standard sharding: the full vocabulary lives on this rank.
    plan = VocabParallelPlan(VocabParallelMode.STANDARD, 0, 1)
    explicit = AscendEmbeddingMethod(AscendW8A8DynamicLmHeadMethod())
    with (
        patch("vllm_ascend.ops.vocab_parallel_embedding._resolve_vocab_parallel_plan", return_value=plan),
        patch("vllm_ascend.ops.vocab_parallel_embedding.get_current_vllm_config", return_value=spec_runtime()),
    ):
        head = AscendParallelLMHead(
            32,
            8,
            params_dtype=torch.float16,
            org_num_embeddings=32,
            quant_config=None,
            prefix="lm_head",
            quant_method=explicit,
        )
    # An online-quantized copy arrives with a preselected method; generic
    # config dispatch must not override it.
    assert head.quant_method is explicit
    assert head.prefix == "lm_head"
    assert head.input_size == 8
    assert head.input_size_per_partition == 8
    assert head.output_size_per_partition == head.num_embeddings_per_partition
    assert head.output_partition_sizes == [head.output_size_per_partition]
