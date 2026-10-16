# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Packed K-scale writes must preserve bytes owned by other requests."""

import ctypes

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.attention.attention_c8_mxfp import (
    mxfp_k_scale_slot_index,
    mxfp_packet_size_bytes,
    mxfp_paged_cache_views,
    scatter_mxfp_k_scale_cache,
)


def _write_reference(raw, slots, values, blocks, heads, pitch, offset):
    cache = mxfp_paged_cache_views(raw[offset:], blocks, heads, 256, 256, 512, page_stride_bytes=pitch)[2]
    cache[slots // 512, :, slots % 512 // 16, :, slots % 16, :] = values


@pytest.mark.parametrize("heads", [1, 2])
@pytest.mark.parametrize("padding", [0, 256])
@pytest.mark.parametrize("offset", [0, 4096])
@pytest.mark.parametrize("noncontiguous_source", [False, True])
def test_scatter_matches_reference_and_graph_replay(heads, padding, offset, noncontiguous_source):
    blocks, tokens, dim = 32, 64, 256
    pitch = mxfp_packet_size_bytes(heads, 512, dim, dim) + padding
    raw = torch.full((offset + blocks * pitch,), 13, dtype=torch.int8, device="npu")
    cache = mxfp_paged_cache_views(raw[offset:], blocks, heads, dim, dim, 512, page_stride_bytes=pitch)[2]
    expected = raw.cpu()
    slots = torch.arange(tokens, dtype=torch.int64) * 73 + 512
    indices = mxfp_k_scale_slot_index(slots.npu(), 512)
    values = torch.arange(tokens * heads * 8, dtype=torch.int32).remainder(251).to(torch.uint8)
    values = values.reshape(tokens, heads, dim // 64, 2)
    if noncontiguous_source:
        backing = torch.empty((*values.shape[:-1], 4), dtype=torch.uint8, device="npu")
        source_bytes = backing[..., ::2]
        source_bytes.copy_(values.npu())
    else:
        source_bytes = values.npu()
    source = source_bytes.view(torch.float8_e8m0fnu) if offset else source_bytes
    _write_reference(expected, slots, values, blocks, heads, pitch, offset)
    scatter_mxfp_k_scale_cache(source, cache, indices)
    torch.npu.synchronize()
    # Compare the complete allocation, including other cache components and padding.
    torch.testing.assert_close(raw.cpu(), expected, rtol=0, atol=0)

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        scatter_mxfp_k_scale_cache(source, cache, indices)
    new_slots = slots + 19
    indices[0].copy_((new_slots // 512).npu())
    indices[1].copy_((new_slots % 512 // 16).npu())
    indices[2].copy_((new_slots % 16).npu())
    new_values = values + 1
    source_bytes.copy_(new_values.npu())
    _write_reference(expected, new_slots, new_values, blocks, heads, pitch, offset)
    graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(raw.cpu(), expected, rtol=0, atol=0)


def test_empty_scatter_preserves_cache():
    raw = torch.full((2 * mxfp_packet_size_bytes(1, 512, 256, 256),), 13, dtype=torch.int8, device="npu")
    cache = mxfp_paged_cache_views(raw, 2, 1, 256, 256, 512)[2]
    before = raw.cpu()
    indices = mxfp_k_scale_slot_index(torch.empty(0, dtype=torch.int64, device="npu"), 512)
    scatter_mxfp_k_scale_cache(torch.empty((0, 1, 4, 2), dtype=torch.uint8, device="npu"), cache, indices)
    torch.testing.assert_close(raw.cpu(), before, rtol=0, atol=0)


def test_graph_scatter_preserves_concurrent_transfer_to_unselected_page():
    acl = pytest.importorskip("acl")
    blocks, heads, dim, tokens = 2048, 2, 256, 32
    pitch = mxfp_packet_size_bytes(heads, 512, dim, dim)
    raw = torch.zeros(blocks * pitch, dtype=torch.int8, device="npu")
    cache = mxfp_paged_cache_views(raw, blocks, heads, dim, dim, 512)[2]
    indices = mxfp_k_scale_slot_index(torch.arange(tokens, dtype=torch.int64, device="npu") * 11 + 512, 512)
    source = torch.full((tokens, heads, dim // 64, 2), 120, dtype=torch.uint8, device="npu")
    marker_bytes = 4096
    # The scatter addresses block 1; SDMA writes directly into block 50.
    marker_address = cache.data_ptr() + 50 * cache.stride(0)
    byte_offset = marker_address - raw.data_ptr()
    assert byte_offset + marker_bytes <= raw.numel()
    max_delay_bytes = 1 << 24
    resources = []

    def allocate(allocate_fn, release_fn, *args):
        resource, status = allocate_fn(*args)
        assert status == 0
        resources.append((release_fn, resource))
        return resource

    try:
        host_marker = allocate(acl.rt.malloc_host, acl.rt.free_host, marker_bytes)
        ctypes.memset(host_marker, 69, marker_bytes)
        host_delay = allocate(acl.rt.malloc_host, acl.rt.free_host, max_delay_bytes)
        ctypes.memset(host_delay, 0, max_delay_bytes)
        device_delay = allocate(acl.rt.malloc, acl.rt.free, max_delay_bytes, 0)
        stream = allocate(acl.rt.create_stream, acl.rt.destroy_stream)
        read_buffer = ctypes.create_string_buffer(marker_bytes)
        for _ in range(2):
            scatter_mxfp_k_scale_cache(source, cache, indices)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            scatter_mxfp_k_scale_cache(source, cache, indices)
        for writer_first in (False, True):
            for delay_bytes in (0, 1 << 18, 1 << 20, 1 << 22, 1 << 24):
                for _ in range(4):
                    raw[byte_offset : byte_offset + marker_bytes].zero_()
                    torch.npu.synchronize()

                    def write_marker(delay_bytes=delay_bytes):
                        if delay_bytes:
                            assert (
                                acl.rt.memcpy_async(device_delay, max_delay_bytes, host_delay, delay_bytes, 1, stream)
                                == 0
                            )
                        assert (
                            acl.rt.memcpy_async(marker_address, marker_bytes, host_marker, marker_bytes, 1, stream) == 0
                        )

                    if writer_first:
                        write_marker()
                        graph.replay()
                    else:
                        graph.replay()
                        write_marker()
                    assert acl.rt.synchronize_stream(stream) == 0
                    torch.npu.synchronize()
                    assert (
                        acl.rt.memcpy(ctypes.addressof(read_buffer), marker_bytes, marker_address, marker_bytes, 2) == 0
                    )
                    assert read_buffer.raw == bytes([69]) * marker_bytes
    finally:
        for release, resource in reversed(resources):
            release(resource)
