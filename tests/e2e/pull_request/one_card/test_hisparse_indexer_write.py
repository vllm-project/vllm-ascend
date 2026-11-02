# SPDX-License-Identifier: Apache-2.0
"""Actual Indexer writes must preserve K, scale bits and shared-slab gaps."""

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.attention.indexer import AscendSFAIndexerBackend
from vllm_ascend.core.kv_cache_interface import AscendSFAIndexerCacheSpec
from vllm_ascend.distributed.kv_transfer.hisparse.binding_optimized import _quantized_indexer_views


@pytest.mark.parametrize("capture", [False, True])
def test_page_strided_indexer_write_preserves_bits_and_padding(capture):
    if not torch.npu.is_available() or not torch.npu.get_device_name(0).startswith("Ascend950"):
        pytest.skip("The native E4M3FN HiSparse format currently targets A5")
    torch.npu.set_device(0)
    pages, block, width, page_stride = 3, 128, 128, 419840
    size = (pages - 1) * page_stride + block * width + block * 4
    raw = torch.full((size,), 90, dtype=torch.uint8, device="npu")
    packed = torch.as_strided(
        raw.view(torch.float8_e4m3fn), (pages, 1, block, width), (page_stride, block * width, width, 1)
    )
    spec = AscendSFAIndexerCacheSpec(
        block_size=block,
        num_kv_heads=1,
        head_size=width,
        dtype=torch.float8_e4m3fn,
        cache_dtype_str="fp8",
        scale_dim=1,
        scale_dtype=torch.float32,
        cache_sparse_li_c8=True,
        cache_sparse_li_c4=False,
        sfa_dcp_replicated_indexer_size=1,
    )
    key, scale = _quantized_indexer_views(packed, spec)
    with pytest.raises(RuntimeError, match="view size"):
        key.view(-1, width)
    writer = AscendSFAIndexerBackend.__new__(AscendSFAIndexerBackend)
    torch.nn.Module.__init__(writer)
    writer.k_cache = SimpleNamespace(
        kv_cache=(key, scale), hisparse_indexer_payload_format="ascend_indexer_e4m3fn_key_fp32_scale_v1"
    )
    writer.enable_sparse_li_c8 = True
    writer.enable_sparse_li_c4 = False
    values = torch.zeros((5, width), dtype=torch.uint8, device="npu").view(torch.float8_e4m3fn)
    scales = torch.zeros(5, dtype=torch.float32, device="npu")
    slots = torch.full((5,), -1, dtype=torch.int32, device="npu")

    def write():
        writer.write_cache(values, scales, slots)

    write()
    torch.npu.synchronize()
    graph = None
    if capture:
        stream = torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            write()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            write()
    try:
        for iteration, logical in enumerate([[-1, 0, 127, 128, 383], [0, 129, -1, 382, 255], [128, 256, 1, -1, 127]]):
            payload = (torch.arange(5 * width, dtype=torch.int64) + iteration * 17).to(torch.uint8).view(5, width)
            scale_bits = torch.tensor(
                [0x3F800000, 0x80000000, 0x7FC00123, 0x7F800000, iteration], dtype=torch.int64
            ).to(torch.int32)
            values.view(torch.uint8).copy_(payload)
            scales.view(torch.int32).copy_(scale_bits)
            slots.copy_(torch.tensor(logical, dtype=torch.int32))
            raw.fill_(90)
            torch.npu.synchronize()
            graph.replay() if graph is not None else write()
            torch.npu.synchronize()
            expected = torch.full((size,), 90, dtype=torch.uint8)
            scale_bytes = scale_bits.view(torch.uint8).view(5, 4)
            for row, slot in enumerate(logical):
                if slot < 0:
                    continue
                page, offset = divmod(slot, block)
                start = page * page_stride + offset * width
                expected[start : start + width] = payload[row]
                start = page * page_stride + block * width + offset * 4
                expected[start : start + 4] = scale_bytes[row]
            torch.testing.assert_close(raw.cpu(), expected, rtol=0, atol=0)
    finally:
        if graph is not None:
            graph.reset()
