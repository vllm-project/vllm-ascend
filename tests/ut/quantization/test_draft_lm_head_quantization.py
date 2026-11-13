# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Tests for the draft LM head quantization bridge and its LM head methods."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.model_executor.layers.quantization.online import lm_head as online_lm_head_module

from vllm_ascend.device.hardware_profile import DeviceAdaptorFamily
from vllm_ascend.ops.vocab_parallel_embedding import AscendParallelLMHead, VocabParallelMode, VocabParallelPlan
from vllm_ascend.quantization.draft_lm_head import (
    DRAFT_LM_HEAD_QUANTIZATION,
    _register_lm_head_methods,
    _resolve_format,
    draft_lm_head_quantization_scope,
)
from vllm_ascend.quantization.method_adapters import AscendEmbeddingMethod
from vllm_ascend.quantization.methods.w8a8.w8a8_dynamic import AscendW8A8DynamicLmHeadMethod
from vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8 import (
    AscendW8A8MXFP8DynamicLinearMethod,
    AscendW8A8MXFP8DynamicLmHeadMethod,
)


def config(
    draft=None,
    target=None,
    upstream_field=None,
    v2=True,
):
    entries = (("draft_lm_head_quantization", draft), ("lm_head_quantization", target))
    return SimpleNamespace(
        use_v2_model_runner=v2,
        additional_config={key: value for key, value in entries if value},
        speculative_config=SimpleNamespace(draft_lm_head_quantization=upstream_field),
    )


def spec_runtime():
    return SimpleNamespace(use_v2_model_runner=True, quant_config=None, additional_config={})


@pytest.fixture(autouse=True)
def standard_device():
    with patch(
        "vllm_ascend.quantization.draft_lm_head.get_current_hardware_profile",
        return_value=SimpleNamespace(device_adaptor_family=DeviceAdaptorFamily.STANDARD),
    ):
        yield


@pytest.mark.parametrize(
    "family,expected",
    [(DeviceAdaptorFamily.STANDARD, "int8"), (DeviceAdaptorFamily.FP8_OPTIMIZED, "mxfp8")],
)
def test_resolve_format_by_hardware_family(family, expected):
    with patch(
        "vllm_ascend.quantization.draft_lm_head.get_current_hardware_profile",
        return_value=SimpleNamespace(device_adaptor_family=family),
    ):
        assert _resolve_format() == expected


def test_resolve_format_rejects_310p():
    with patch(
        "vllm_ascend.quantization.draft_lm_head.get_current_hardware_profile",
        return_value=SimpleNamespace(device_adaptor_family=DeviceAdaptorFamily.COMPATIBILITY),
    ), pytest.raises(ValueError, match="not yet adapted"):
        _resolve_format()


def test_scope_is_noop_when_not_configured():
    cfg = config()
    with draft_lm_head_quantization_scope(cfg) as entered:
        assert entered is None
    assert cfg.speculative_config.draft_lm_head_quantization is None


def test_scope_requires_speculative_config():
    cfg = config(draft=DRAFT_LM_HEAD_QUANTIZATION)
    cfg.speculative_config = None
    with pytest.raises(ValueError, match="conflicts"), draft_lm_head_quantization_scope(cfg):
        pass


def test_scope_rejects_upstream_field_conflict():
    cfg = config(draft=DRAFT_LM_HEAD_QUANTIZATION, upstream_field="fp8")
    with pytest.raises(ValueError, match="conflicts"), draft_lm_head_quantization_scope(cfg):
        pass


def test_scope_rejects_unsupported_value():
    cfg = config(draft="int4")
    with pytest.raises(ValueError, match="Unsupported"), draft_lm_head_quantization_scope(cfg):
        pass


def test_scope_absorbs_when_target_lm_head_is_quantized():
    cfg = config(draft=DRAFT_LM_HEAD_QUANTIZATION, target="w8a8_dynamic")
    # Absorption must leave the upstream field untouched: the share branch
    # hands the drafter the quantized target head without a copy.
    with draft_lm_head_quantization_scope(cfg):
        assert cfg.speculative_config.draft_lm_head_quantization is None
    assert cfg.speculative_config.draft_lm_head_quantization is None


def test_scope_requires_v2_model_runner():
    cfg = config(draft=DRAFT_LM_HEAD_QUANTIZATION, v2=False)
    with pytest.raises(ValueError, match="Model Runner V2"), draft_lm_head_quantization_scope(cfg):
        pass


@pytest.mark.parametrize(
    "family,expected",
    [(DeviceAdaptorFamily.STANDARD, "int8"), (DeviceAdaptorFamily.FP8_OPTIMIZED, "mxfp8")],
)
def test_scope_bridges_format_and_restores(family, expected):
    cfg = config(draft=DRAFT_LM_HEAD_QUANTIZATION)
    with (
        patch(
            "vllm_ascend.quantization.draft_lm_head.get_current_hardware_profile",
            return_value=SimpleNamespace(device_adaptor_family=family),
        ),
        patch("vllm_ascend.quantization.draft_lm_head._register_lm_head_methods"),
    ):
        with draft_lm_head_quantization_scope(cfg):
            assert cfg.speculative_config.draft_lm_head_quantization == expected
        assert cfg.speculative_config.draft_lm_head_quantization is None


def test_scope_restores_field_after_error():
    cfg = config(draft=DRAFT_LM_HEAD_QUANTIZATION)
    with (
        patch("vllm_ascend.quantization.draft_lm_head._register_lm_head_methods"),
        pytest.raises(RuntimeError, match="boom"),
        draft_lm_head_quantization_scope(cfg),
    ):
        raise RuntimeError("boom")
    assert cfg.speculative_config.draft_lm_head_quantization is None


def test_register_lm_head_methods_registers_lazy_factories():
    methods = online_lm_head_module._LM_HEAD_METHODS
    added = [key for key in ("int8", "mxfp8") if key not in methods]
    try:
        _register_lm_head_methods()
        assert callable(methods["int8"])
        assert callable(methods["mxfp8"])
        int8_method = methods["int8"]()
        assert type(int8_method) is AscendEmbeddingMethod
        assert type(int8_method.quant_method) is AscendW8A8DynamicLmHeadMethod
        # The MXFP8 scheme reads the current vllm config at construction.
        with (
            patch(
                "vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8.get_current_vllm_config",
                return_value=spec_runtime(),
            ),
            patch("vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8.get_dynamic_mx_quant_scale_alg", return_value=1),
        ):
            mxfp8_method = methods["mxfp8"]()
        assert type(mxfp8_method) is AscendEmbeddingMethod
        assert type(mxfp8_method.quant_method) is AscendW8A8MXFP8DynamicLmHeadMethod
    finally:
        for key in added:
            del methods[key]


def test_explicit_quant_method_survives_construction():
    # Single-rank standard sharding: the full vocabulary lives on this rank.
    plan = VocabParallelPlan(VocabParallelMode.STANDARD, 0, 1)
    explicit = AscendEmbeddingMethod(AscendW8A8DynamicLmHeadMethod())
    with (
        patch("vllm_ascend.ops.vocab_parallel_embedding._resolve_vocab_parallel_plan", return_value=plan),
        patch("vllm_ascend.ops.vocab_parallel_embedding.get_current_vllm_config_or_none", return_value=spec_runtime()),
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
        patch("vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8.get_current_vllm_config", return_value=spec_runtime()),
        patch("vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8.get_dynamic_mx_quant_scale_alg", return_value=1),
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
        patch("vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8.get_current_vllm_config", return_value=spec_runtime()),
        patch("vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8.get_dynamic_mx_quant_scale_alg", return_value=1),
    ):
        method = AscendW8A8MXFP8DynamicLmHeadMethod()
        method.process_weights_after_loading(head)
    quant.assert_not_called()
    assert head.weight.dtype == torch.float16
