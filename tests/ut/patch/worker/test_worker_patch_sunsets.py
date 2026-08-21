# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.model_loader import base_loader, utils

pytest.importorskip("torch_npu")

from vllm_ascend.models.layer.attention.layer import DSAAttention


def test_upstream_post_load_processing_handles_dsa_attention() -> None:
    assert issubclass(DSAAttention, AttentionLayerBase)

    layer = DSAAttention.__new__(DSAAttention)
    nn.Module.__init__(layer)
    layer.impl = SimpleNamespace(process_weights_after_loading=MagicMock())
    model = nn.Sequential(layer)
    model_config = SimpleNamespace(dtype=torch.bfloat16, quantization=None)

    utils.process_weights_after_loading(model, model_config, torch.device("cpu"))

    layer.impl.process_weights_after_loading.assert_called_once_with(torch.bfloat16)
    assert base_loader.process_weights_after_loading is utils.process_weights_after_loading


@pytest.mark.parametrize(
    "loader_module",
    [
        "vllm_ascend.model_loader.netloader.netloader",
        "vllm_ascend.model_loader.rfork.rfork_loader",
    ],
)
def test_ascend_loaders_use_upstream_post_load_processing(loader_module: str) -> None:
    module = importlib.import_module(loader_module)

    assert module.process_weights_after_loading is utils.process_weights_after_loading


def test_npugraph_ex_handles_triton_value_pack_arguments() -> None:
    pytest.importorskip("torch_npu")
    pytest.importorskip("npugraph_ex")
    from npugraph_ex.core._concrete_graph import ValuePack
    from npugraph_ex.npu_fx_compiler import (
        _NpuGraphConverter,
        _unpack_meta,
    )

    value_pack = ValuePack(
        meta={"value": "meta-value"},
        npu_meta={"value": "npu-value"},
    )

    assert value_pack["value"] == "meta-value"
    meta_args, meta_kwargs = _unpack_meta(
        ({"input": value_pack},),
        {"keyword": value_pack},
    )
    npu_args, npu_kwargs = _NpuGraphConverter._unpack_npu(
        MagicMock(),
        ({"input": value_pack},),
        {"keyword": value_pack},
    )

    assert meta_args == [{"input": value_pack.meta}]
    assert meta_kwargs == {"keyword": value_pack.meta}
    assert npu_args == [{"input": value_pack.npu}]
    assert npu_kwargs == {"keyword": value_pack.npu}
