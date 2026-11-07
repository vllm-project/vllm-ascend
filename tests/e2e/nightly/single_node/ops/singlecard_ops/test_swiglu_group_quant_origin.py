# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the Native and Meta contracts for an omitted MX origin output."""

import importlib

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.utils import bootstrap_custom_op_env

pytestmark = pytest.mark.skipif(
    get_ascend_device_type() != AscendDeviceType.A5,
    reason="SwigluGroupQuant MX requires Ascend A5",
)


@pytest.fixture(scope="module", autouse=True)
def native_binding():
    # A5 loads this binding directly; the generic custom-op gate can be off.
    bootstrap_custom_op_env()
    if not hasattr(torch.ops._C_ascend, "npu_swiglu_group_quant"):
        importlib.import_module("vllm_ascend.vllm_ascend_C")


def _bytes(x):
    return x.contiguous().view(torch.uint8).cpu()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("dst_type", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("rows,width,valid_rows", [(1, 128, 1), (128, 256, 128), (1024, 4608, 517)])
@pytest.mark.parametrize("clamp_value", [0.0, 10.0])
@torch.inference_mode()
def test_mx_omitted_origin(dtype, dst_type, rows, width, valid_rows, clamp_value):
    x = torch.randn(rows, width, device="npu", dtype=dtype)
    group = torch.tensor([valid_rows], device="npu", dtype=torch.int64)
    kwargs = dict(topk_weight=None, group_index=group, dst_type=dst_type, quant_mode=2, clamp_value=clamp_value)
    requested = torch.ops._C_ascend.npu_swiglu_group_quant(x, output_origin=True, **kwargs)
    omitted = torch.ops._C_ascend.npu_swiglu_group_quant(x, output_origin=False, **kwargs)
    assert requested[2].shape == (rows, width // 2)
    assert omitted[2].shape == (0,)
    assert omitted[2].dtype == dtype
    assert omitted[2].device == x.device
    # The capacity tail after group_index is unspecified, not valid output.
    for actual, expected in zip(omitted[:2], requested[:2]):
        assert torch.equal(_bytes(actual[:valid_rows]), _bytes(expected[:valid_rows]))
    for output_origin, native in ((True, requested), (False, omitted)):
        meta = torch.ops._C_ascend.npu_swiglu_group_quant(
            torch.empty_like(x, device="meta"),
            output_origin=output_origin,
            **{**kwargs, "group_index": torch.empty_like(group, device="meta")},
        )
        assert [(t.shape, t.dtype) for t in meta] == [(t.shape, t.dtype) for t in native]


@pytest.mark.parametrize("mode", [1, 2, 3])
@pytest.mark.parametrize("output_origin", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_origin_meta_contract(mode, output_origin, dtype):
    x = torch.empty((128, 256), dtype=dtype, device="meta")
    outputs = torch.ops._C_ascend.npu_swiglu_group_quant(
        x,
        topk_weight=None,
        group_index=None,
        dst_type=torch.float8_e4m3fn,
        quant_mode=mode,
        output_origin=output_origin,
    )
    expected = (0,) if mode == 2 and not output_origin else (128, 128)
    assert outputs[2].shape == expected
    assert outputs[2].dtype == dtype


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_mx_origin_graph_replay(dtype):
    x = torch.randn(128, 256, device="npu", dtype=dtype)
    kwargs = dict(topk_weight=None, group_index=None, dst_type=torch.float8_e4m3fn, quant_mode=2, output_origin=False)
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            torch.ops._C_ascend.npu_swiglu_group_quant(x, **kwargs)
    torch.npu.current_stream().wait_stream(stream)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        captured = torch.ops._C_ascend.npu_swiglu_group_quant(x, **kwargs)
    assert captured[2].shape == (0,)
    addresses = [t.data_ptr() for t in captured]
    for seed in (41, 42, 43):
        torch.manual_seed(seed)
        x.copy_(torch.randn_like(x))
        reference = torch.ops._C_ascend.npu_swiglu_group_quant(x, **kwargs)
        graph.replay()
        torch.npu.synchronize()
        assert [t.data_ptr() for t in captured] == addresses
        for actual, expected in zip(captured[:2], reference[:2]):
            assert torch.equal(_bytes(actual), _bytes(expected))
