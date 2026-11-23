# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Selected-row PLE lookup and graph replay on real NPU hardware."""

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.models.deepseek_v41.engram.npu import CHUNK_ROWS, HostUvaBuffer
from vllm_ascend.models.qwen4_exp.ple_quant import AscendPLEQuantizedEmbeddingMethod
from vllm_ascend.ops.triton.qwen4_exp.ple_lookup_quantized import _lookup_quantized, lookup_quantized
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type


@pytest.mark.parametrize("mxfp8", [False, True])
def test_resident_selected_rows_and_graph(mxfp8):
    if mxfp8 and get_ascend_device_type() != AscendDeviceType.A5:
        pytest.skip("MXFP8 PLE requires 950DT; A3 uses INT8 PLE")
    method = AscendPLEQuantizedEmbeddingMethod(mxfp8=mxfp8)
    codes = torch.arange(4 * 160).reshape(4, 160).remainder(15).sub(7).to(method.weight_dtype)
    scales = torch.tensor([125, 126, 127, 128, 129], dtype=torch.uint8).expand(4, -1).contiguous()
    decoded_scales = torch.tensor([0.25, 0.5, 1, 2, 4]).expand(4, -1).contiguous()
    layer = SimpleNamespace(
        weight=codes.npu(), weight_scale=(scales if mxfp8 else decoded_scales).npu(), params_dtype=torch.bfloat16
    )
    ids = torch.tensor([[3, 0], [1, 3]], device="npu")
    expected = (codes.float().unflatten(-1, (5, 32)) * decoded_scales.unsqueeze(-1)).flatten(-2)
    expected = expected[ids.cpu()].bfloat16()
    torch.testing.assert_close(method.embedding(layer, ids).cpu(), expected, rtol=0, atol=0)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = method.embedding(layer, ids)
    graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("width", [160, 256])
@pytest.mark.parametrize("mxfp8", [False, True])
def test_host_lookup_chunk_boundary_and_graph(width, mxfp8):
    if mxfp8 and get_ascend_device_type() != AscendDeviceType.A5:
        pytest.skip("MXFP8 PLE host lookup requires 950DT")
    method = AscendPLEQuantizedEmbeddingMethod(mxfp8=mxfp8)
    # Use the real allocation chunk size: this also exercises 64-bit global
    # row indices without making the kernel address a single enormous tile.
    rows = CHUNK_ROWS + 2
    device = torch.device("npu", torch.npu.current_device())
    owner = torch.nn.Module()
    codes = HostUvaBuffer((rows, width), method.weight_dtype, device, "codes", owner)
    try:
        scales = torch.empty((rows, width // 32), dtype=method.scale_dtype, device=device)
        selected = [0, 1, CHUNK_ROWS, CHUNK_ROWS + 1]
        values = torch.arange(4 * width).reshape(4, width).remainder(15).sub(7).to(method.weight_dtype)
        exponents = torch.arange(width // 32, dtype=torch.uint8).add(124).expand(4, -1).clone()
        # Include E8M0 zero and NaN encodings in addition to normal scales.
        exponents[0, 0] = 0
        exponents[1, 0] = 255
        decoded = torch.exp2(exponents.double() - 127).float().masked_fill(exponents == 255, float("nan"))
        if not mxfp8:
            values[0, :2] = torch.tensor([-128, 127], dtype=torch.int8)
            decoded = torch.linspace(0.125, 1.875, width // 32).expand(4, -1).clone()
            decoded[0, 0] = 0
        for index, row in enumerate(selected):
            codes.tensor[row].copy_(values[index])
            scales[row].copy_(exponents[index] if mxfp8 else decoded[index])
        start = 37
        local_ids = torch.tensor([*selected, -1, rows])
        ids = (local_ids + start).to(device)
        output = torch.empty((6, width), dtype=torch.bfloat16, device=device)
        expected = (values.float().unflatten(-1, (-1, 32)) * decoded.unsqueeze(-1)).flatten(-2).bfloat16()
        expected = torch.cat((expected, torch.zeros((2, width), dtype=torch.bfloat16)))

        def lookup():
            lookup_quantized(codes.ptrs, scales, ids, output, start, start + rows, CHUNK_ROWS, is_mxfp8=mxfp8)

        lookup()
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0, equal_nan=True)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            lookup()
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0, equal_nan=True)
        # Graphs retain device pointers into the host buffers.
        del graph
        empty = torch.empty((0, width), dtype=torch.bfloat16, device=device)
        lookup_quantized(codes.ptrs, scales, ids[:0], empty, start, start + rows, CHUNK_ROWS, is_mxfp8=mxfp8)
    finally:
        torch.npu.synchronize()
        codes.close()


@pytest.mark.parametrize("mxfp8", [False, True])
@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
def test_host_lookup_dynamic_shapes_without_recompilation(mxfp8, id_dtype, monkeypatch):
    """After one eager specialization, token count and shard offsets stay dynamic.

    The 160-wide rows are the model shape. Counts cover decode, batched decode,
    MTP-shaped inputs and prefill; sliced IDs exercise unaligned/noncontiguous
    views. This is eager JIT priming, not registry compile-only warmup.
    """
    if mxfp8 and get_ascend_device_type() != AscendDeviceType.A5:
        pytest.skip("MXFP8 PLE host lookup requires 950DT")
    rows, width = 513, 160
    method = AscendPLEQuantizedEmbeddingMethod(mxfp8=mxfp8)
    device = torch.device("npu", torch.npu.current_device())
    owner = torch.nn.Module()
    codes = HostUvaBuffer((rows, width), method.weight_dtype, device, "codes", owner)
    graph = None
    try:
        scales = torch.empty((rows, width // 32), dtype=method.scale_dtype, device=device)
        values = torch.arange(rows * width).reshape(rows, width).remainder(15).sub(7).to(method.weight_dtype)
        codes.tensor.copy_(values)
        scales.fill_(127 if mxfp8 else 1)
        ids = torch.tensor([0], dtype=id_dtype, device=device)
        output = torch.empty((1, width), dtype=torch.bfloat16, device=device)
        lookup_quantized(codes.ptrs, scales, ids, output, 0, rows, CHUNK_ROWS, is_mxfp8=mxfp8)
        torch.npu.synchronize()

        def forbid_compile(*args, **kwargs):
            raise AssertionError("PLE lookup recompiled after eager priming")

        # Triton binds compile on each JITFunction instance. Patching only the
        # module entry points after priming would miss this bound reference.
        monkeypatch.setattr(_lookup_quantized, "compile", forbid_compile)
        for shape, start in (((16,), 1), ((17,), 16), ((8, 4), 37), ((256,), 0), ((4096,), 37), ((4096, 16), 37)):
            host_ids = torch.arange(torch.tensor(shape).prod().item(), dtype=id_dtype).reshape(shape) % (rows + 2)
            host_ids += start - 1
            backing = torch.empty((*shape, 2), dtype=id_dtype, device=device)
            ids = backing[..., 1]
            ids.copy_(host_ids)
            output = torch.empty((*shape, width), dtype=torch.bfloat16, device=device)
            lookup_quantized(codes.ptrs, scales, ids, output, start, start + rows, CHUNK_ROWS, is_mxfp8=mxfp8)
            owned = (host_ids >= start) & (host_ids < start + rows)
            expected = values.float()[(host_ids.long() - start).clamp(0, rows - 1)].bfloat16()
            expected = torch.where(owned.unsqueeze(-1), expected, 0)
            torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            lookup_quantized(codes.ptrs, scales, ids, output, start, start + rows, CHUNK_ROWS, is_mxfp8=mxfp8)
        ids.fill_(start)
        graph.replay()
        torch.testing.assert_close(output.cpu(), values[0].float().bfloat16().expand_as(output.cpu()), rtol=0, atol=0)
    finally:
        torch.npu.synchronize()
        if graph is not None:
            graph.reset()
        codes.close()


@pytest.mark.parametrize(
    "invalid",
    [
        "output_layout",
        "output_dtype",
        "output_shape",
        "width",
        "ids_dtype",
        "pointer_dtype",
        "scale_pointer_table",
        "scale_dtype",
        "pointer_size",
        "pointer_layout",
        "device",
        "bounds",
        "chunk",
        "scalar_type",
        "format",
    ],
)
def test_host_lookup_rejects_unsupported_inputs(invalid):
    # Pointer values are deliberately not valid addresses: every case must be
    # rejected from metadata before a kernel can dereference the code pointer table.
    device = torch.device("npu", torch.npu.current_device())
    pointers = torch.zeros(1, dtype=torch.int64, device=device)
    args = dict(
        code_chunk_ptrs=pointers,
        scales=torch.empty((32, 5), dtype=torch.float32, device=device),
        embedding_ids=torch.zeros(2, dtype=torch.int64, device=device),
        output=torch.empty((2, 160), dtype=torch.bfloat16, device=device),
        shard_start=0,
        shard_end=32,
        chunk_rows=CHUNK_ROWS,
        is_mxfp8=False,
    )
    if invalid == "output_layout":
        args["output"] = torch.empty((2, 320), dtype=torch.bfloat16, device=device)[:, ::2]
    elif invalid == "output_dtype":
        args["output"] = args["output"].float()
    elif invalid == "output_shape":
        args["output"] = args["output"][:1]
    elif invalid == "width":
        args["output"] = torch.empty((2, 159), dtype=torch.bfloat16, device=device)
    elif invalid == "ids_dtype":
        args["embedding_ids"] = args["embedding_ids"].float()
    elif invalid == "pointer_dtype":
        args["code_chunk_ptrs"] = pointers.int()
    elif invalid == "scale_pointer_table":
        args["scales"] = pointers
    elif invalid == "scale_dtype":
        args["scales"] = args["scales"].byte()
    elif invalid == "pointer_size":
        args["code_chunk_ptrs"] = pointers[:0]
    elif invalid == "pointer_layout":
        args["code_chunk_ptrs"] = pointers.reshape(1, 1)
    elif invalid == "device":
        args["embedding_ids"] = args["embedding_ids"].cpu()
    elif invalid == "bounds":
        args["shard_end"] = 0
    elif invalid == "chunk":
        args["chunk_rows"] = 0
    elif invalid == "scalar_type":
        args["shard_start"] = 0.0
    else:
        args["is_mxfp8"] = 1
    with pytest.raises(ValueError):
        lookup_quantized(**args)
