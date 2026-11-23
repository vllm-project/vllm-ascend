# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from vllm_ascend.quantization.methods.w8a8 import w8a8_mxfp8

INPUT_SIZE = 256
OUTPUT_SIZE = 16
GROUP_SIZE = 32
SCALES_PER_PAIR = 2


@pytest.fixture
def scheme():
    config = SimpleNamespace(quant_config=SimpleNamespace(quant_description={"group_size": GROUP_SIZE}))
    with (
        patch.object(w8a8_mxfp8, "require_mx_weight_format"),
        patch.object(w8a8_mxfp8, "get_current_vllm_config", return_value=config),
        patch.object(w8a8_mxfp8, "get_dynamic_mx_quant_scale_alg", return_value=0),
    ):
        return w8a8_mxfp8.AscendW8A8MXFP8DynamicLinearMethod()


def _make_layer(*, keep_nd: bool | None = True, single_group_nz: bool = False) -> nn.Module:
    layer = nn.Module()
    # Distinct, exactly representable values expose transpose or scale-lane
    # mistakes without depending on a device's MXFP8 quantizer.
    weight = (torch.arange(OUTPUT_SIZE * INPUT_SIZE).reshape(OUTPUT_SIZE, INPUT_SIZE) % 17 - 8).to(torch.float8_e4m3fn)
    scale = (
        torch.arange(OUTPUT_SIZE * INPUT_SIZE // GROUP_SIZE).reshape(OUTPUT_SIZE, INPUT_SIZE // GROUP_SIZE) % 7 + 120
    ).to(torch.uint8)
    layer.weight = nn.Parameter(weight, requires_grad=False)
    layer.weight_scale = nn.Parameter(scale, requires_grad=False)
    if keep_nd is not None:
        layer._mxfp8_keep_nd = keep_nd
    layer._single_group_gmm_nz = single_group_nz
    return layer


def _expected_scale(original: torch.Tensor) -> torch.Tensor:
    return original.reshape(OUTPUT_SIZE, INPUT_SIZE // GROUP_SIZE // SCALES_PER_PAIR, SCALES_PER_PAIR).transpose(0, 1)


def _assert_nd_layout(layer: nn.Module, original_weight: torch.Tensor, original_scale: torch.Tensor) -> None:
    assert not layer._mxfp8_gmm_nz_layout
    assert layer.weight.shape == (INPUT_SIZE, OUTPUT_SIZE)
    assert layer.weight_scale.shape == (INPUT_SIZE // GROUP_SIZE // SCALES_PER_PAIR, OUTPUT_SIZE, SCALES_PER_PAIR)
    assert layer.weight.is_contiguous()
    assert layer.weight_scale.is_contiguous()
    assert layer.weight.dtype == torch.float8_e4m3fn
    assert layer.weight_scale.dtype == torch.uint8
    torch.testing.assert_close(layer.weight.float(), original_weight.T.float(), rtol=0, atol=0)
    torch.testing.assert_close(layer.weight_scale, _expected_scale(original_scale), rtol=0, atol=0)


@pytest.mark.parametrize("single_group_nz", [False, True])
def test_keep_nd_preserves_mxfp8_layout_despite_nz_configuration(scheme, single_group_nz):
    layer = _make_layer(single_group_nz=single_group_nz)
    original_weight = layer.weight.detach().clone()
    original_scale = layer.weight_scale.detach().clone()
    with (
        patch.object(w8a8_mxfp8, "_should_trans_nz", return_value=True),
        patch.object(w8a8_mxfp8, "maybe_trans_nz", side_effect=lambda value, **kwargs: value.clone()) as cast,
    ):
        scheme.process_weights_after_loading(layer)
        _assert_nd_layout(layer, original_weight, original_scale)
        cast.assert_not_called()
    assert layer._mxfp8_transformed
    assert layer._mxfp8_original_shapes == {
        "weight": original_weight.shape,
        "weight_scale": original_scale.shape,
    }


def test_keep_nd_reload_updates_captured_buffers_without_changing_addresses(scheme):
    layer = _make_layer(single_group_nz=True)
    expected_weight = layer.weight.detach().clone()
    expected_scale = layer.weight_scale.detach().clone()
    with (
        patch.object(w8a8_mxfp8, "_should_trans_nz", return_value=True),
        patch.object(w8a8_mxfp8, "maybe_trans_nz", side_effect=lambda value, **kwargs: value.clone()) as cast,
    ):
        scheme.process_weights_after_loading(layer)
        weight_ptr = layer._mxfp8_weight_buf.data_ptr()
        scale_ptr = layer._mxfp8_scale_buf.data_ptr()
        for update in (1, 2):
            scheme.restore_weights_for_rl_loading(layer)
            assert not layer._mxfp8_transformed
            torch.testing.assert_close(layer.weight.float(), expected_weight.float(), rtol=0, atol=0)
            torch.testing.assert_close(layer.weight_scale, expected_scale, rtol=0, atol=0)
            expected_weight = (expected_weight.float() + update / 4).to(torch.float8_e4m3fn)
            expected_scale = (expected_scale.to(torch.int16) + update).to(torch.uint8)
            layer.weight.data.copy_(expected_weight)
            layer.weight_scale.data.copy_(expected_scale)

            scheme.process_weights_after_loading(layer)

            _assert_nd_layout(layer, expected_weight, expected_scale)
            assert layer._mxfp8_weight_buf.data_ptr() == weight_ptr
            assert layer._mxfp8_scale_buf.data_ptr() == scale_ptr
            assert layer.weight.data_ptr() == weight_ptr
            assert layer.weight_scale.data_ptr() == scale_ptr
        cast.assert_not_called()


@pytest.mark.parametrize("keep_nd", [None, False], ids=["flag_absent", "flag_false"])
@pytest.mark.parametrize("single_group_nz", [False, True])
def test_default_keeps_existing_nz_conversion(scheme, keep_nd, single_group_nz):
    layer = _make_layer(keep_nd=keep_nd, single_group_nz=single_group_nz)
    original_weight = layer.weight.detach().clone()
    original_scale = layer.weight_scale.detach().clone()
    with (
        patch.object(w8a8_mxfp8, "_should_trans_nz", return_value=True),
        patch.object(w8a8_mxfp8, "maybe_trans_nz", side_effect=lambda value, **kwargs: value.clone()) as cast,
    ):
        scheme.process_weights_after_loading(layer)
        cast.assert_called_once()
        assert cast.call_args.kwargs["customize_dtype"] == torch.float8_e4m3fn
    expected_weight = original_weight.T
    expected_scale = _expected_scale(original_scale)
    if single_group_nz:
        expected_weight = expected_weight.unsqueeze(0)
        expected_scale = expected_scale.unsqueeze(0)
    assert layer._mxfp8_gmm_nz_layout is single_group_nz
    assert cast.call_args.args[0].shape == expected_weight.shape
    torch.testing.assert_close(layer.weight.float(), expected_weight.float(), rtol=0, atol=0)
    torch.testing.assert_close(layer.weight_scale, expected_scale, rtol=0, atol=0)
