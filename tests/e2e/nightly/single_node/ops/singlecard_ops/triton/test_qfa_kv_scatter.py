# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""int8_qfa K/V payload and K-descale writes must preserve bytes owned by other requests."""

from unittest.mock import MagicMock

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.attention.attention_a3_qfa import (
    qfa_int8_packet_section_sizes,
    qfa_int8_paged_cache_views,
)
from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.ops.triton.qfa_kv_scatter import (
    _scatter_k_descale_kernel,
    _scatter_kv_payload_kernel,
    scatter_k_descale,
    scatter_kv_payload,
)

BLOCK_SIZE = 128
DIM = 256
FRAGS = DIM // 32
ASCEND_ONLY = pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3, AscendDeviceType.A5),
    reason="Requires Ascend A2/A3/A5",
)


def _pitch(heads, padding=0):
    k_elems, kd_elems, _ = qfa_int8_packet_section_sizes(heads, BLOCK_SIZE, DIM)
    return 2 * k_elems + 4 * kd_elems + padding


def _views_on(raw, heads, blocks, padding=0):
    return qfa_int8_paged_cache_views(raw, blocks, heads, DIM, BLOCK_SIZE, page_stride_bytes=_pitch(heads, padding))


def _payloads(tokens, heads, seed):
    g = torch.Generator().manual_seed(seed)
    k = torch.randint(-127, 128, (tokens, heads, DIM), generator=g).to(torch.int8)
    v = torch.randint(-127, 128, (tokens, heads, DIM), generator=g).to(torch.int8)
    kd = (torch.rand((tokens, heads), generator=g) * 3.0).to(torch.float32)
    return k, v, kd


def _write_reference(views, slots, k, v, kd):
    """The strided index_put_ golden over the packet views."""
    k_nz, kd_nz, v_nz = views
    tokens, heads = k.shape[0], k.shape[1]
    pages, pos = slots // BLOCK_SIZE, slots % BLOCK_SIZE
    frag_index = (
        pages.view(tokens, 1, 1).expand(tokens, heads, FRAGS),
        torch.arange(heads).view(1, heads, 1).expand(tokens, heads, FRAGS),
        torch.arange(FRAGS).view(1, 1, FRAGS).expand(tokens, heads, FRAGS),
        pos.view(tokens, 1, 1).expand(tokens, heads, FRAGS),
    )
    k_nz[frag_index] = k.reshape(tokens, heads, FRAGS, 32)
    v_nz[frag_index] = v.reshape(tokens, heads, FRAGS, 32)
    kd_nz[
        (pages.view(tokens, 1).expand(tokens, heads),
         torch.arange(heads).view(1, heads).expand(tokens, heads),
         pos.view(tokens, 1).expand(tokens, heads))
    ] = kd


def _scatter(views, k, v, kd, slot_index):
    scatter_kv_payload(k, views[0], slot_index)
    scatter_kv_payload(v, views[2], slot_index)
    scatter_k_descale(kd, views[1], slot_index)


@ASCEND_ONLY
@pytest.mark.parametrize("heads", [1, 2, 4])
@pytest.mark.parametrize("padding", [0, 256])
@pytest.mark.parametrize("offset", [0, 4096])
@pytest.mark.parametrize("noncontiguous_source", [False, True])
@pytest.mark.parametrize("tokens", [1, 64, 65, 257])
def test_scatter_matches_reference_and_graph_replay(heads, padding, offset, noncontiguous_source, tokens):
    blocks = 64
    raw = torch.full((offset + blocks * _pitch(heads, padding),), 13, dtype=torch.int8, device="npu")
    views = _views_on(raw[offset:], heads, blocks, padding)
    expected = raw.cpu()
    cpu_views = _views_on(expected[offset:], heads, blocks, padding)
    slots = torch.arange(tokens, dtype=torch.int64) * 73 % (blocks * BLOCK_SIZE)
    k, v, kd = _payloads(tokens, heads, 0x51 + tokens)
    if noncontiguous_source:
        backing = torch.empty((tokens, heads, DIM * 2), dtype=torch.int8, device="npu")
        source_k, source_v = backing[..., :DIM], backing[..., DIM:]
        source_k.copy_(k.npu())
        source_v.copy_(v.npu())
        source_kd = torch.empty((tokens, heads, 2), dtype=torch.float32, device="npu")[..., 0]
        source_kd.copy_(kd.npu())
    else:
        source_k, source_v, source_kd = k.npu(), v.npu(), kd.npu()
    _write_reference(cpu_views, slots, k, v, kd)

    slot_index = tuple(index.npu() for index in (slots // BLOCK_SIZE, slots % BLOCK_SIZE))
    _scatter(views, source_k, source_v, source_kd, slot_index)
    torch.npu.synchronize()
    torch.testing.assert_close(raw.cpu(), expected, rtol=0, atol=0)

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        _scatter(views, source_k, source_v, source_kd, slot_index)
    new_slots = (slots * 7 + 31) % (blocks * BLOCK_SIZE)
    new_k, new_v, new_kd = _payloads(tokens, heads, 0xC3 + tokens)
    slot_index[0].copy_(new_slots // BLOCK_SIZE)
    slot_index[1].copy_(new_slots % BLOCK_SIZE)
    source_k.copy_(new_k.npu())
    source_v.copy_(new_v.npu())
    source_kd.copy_(new_kd.npu())
    _write_reference(cpu_views, new_slots, new_k, new_v, new_kd)
    graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(raw.cpu(), expected, rtol=0, atol=0)


@ASCEND_ONLY
@pytest.mark.parametrize("heads", [1, 2])
def test_token_counts_reuse_compilation(heads, monkeypatch):
    """Changing token counts must preserve bytes and reuse the compiled kernels."""
    blocks, max_tokens = 8, 257
    raw = torch.full((blocks * _pitch(heads),), 13, dtype=torch.int8, device="npu")
    views = _views_on(raw, heads, blocks)
    expected = raw.cpu()
    cpu_views = _views_on(expected, heads, blocks)
    slots = torch.arange(max_tokens, dtype=torch.int64) * 37 % (blocks * BLOCK_SIZE)
    k, v, kd = _payloads(max_tokens, heads, 7)
    compile_events = []
    originals = {
        _scatter_kv_payload_kernel: _scatter_kv_payload_kernel._do_compile,
        _scatter_k_descale_kernel: _scatter_k_descale_kernel._do_compile,
    }

    def make_recorder(kernel):
        original = originals[kernel]

        def record(*args, **kwargs):
            compile_events.append(args[0])
            return original(*args, **kwargs)

        return record

    monkeypatch.setattr(_scatter_kv_payload_kernel, "_do_compile", make_recorder(_scatter_kv_payload_kernel))
    monkeypatch.setattr(_scatter_k_descale_kernel, "_do_compile", make_recorder(_scatter_k_descale_kernel))
    first_count = None
    for tokens in (1, 2, 31, 32, 33, 64, 65, 128, 129, 257):
        slot_index = tuple(index.npu() for index in (slots[:tokens] // BLOCK_SIZE, slots[:tokens] % BLOCK_SIZE))
        _scatter(views, k[:tokens].npu(), v[:tokens].npu(), kd[:tokens].npu(), slot_index)
        torch.npu.synchronize()
        _write_reference(cpu_views, slots[:tokens], k[:tokens], v[:tokens], kd[:tokens])
        torch.testing.assert_close(raw.cpu(), expected, rtol=0, atol=0)
        if first_count is None:
            first_count = len(compile_events)
        assert len(compile_events) == first_count


@ASCEND_ONLY
@pytest.mark.parametrize("wrapper", ["payload", "descale"])
def test_empty_scatter_preserves_cache(wrapper):
    heads, blocks = 2, 4
    raw = torch.full((blocks * _pitch(heads),), 13, dtype=torch.int8, device="npu")
    views = _views_on(raw, heads, blocks)
    before = raw.cpu()
    empty_index = (torch.empty(0, dtype=torch.int64, device="npu"),) * 2
    if wrapper == "payload":
        scatter_kv_payload(torch.empty(0, heads, DIM, dtype=torch.int8, device="npu"), views[0], empty_index)
    else:
        scatter_k_descale(torch.empty(0, heads, dtype=torch.float32, device="npu"), views[1], empty_index)
    torch.npu.synchronize()
    torch.testing.assert_close(raw.cpu(), before, rtol=0, atol=0)


@ASCEND_ONLY
@pytest.mark.parametrize("heads", [1, 2])
@pytest.mark.parametrize("wrapper", ["payload", "descale"])
def test_mismatched_source_size_is_rejected_before_launch(heads, wrapper, monkeypatch):
    blocks, tokens = 2, 2
    raw = torch.full((blocks * _pitch(heads),), 13, dtype=torch.int8, device="npu")
    views = _views_on(raw, heads, blocks)
    before = raw.cpu()
    slot_index = tuple(index.npu() for index in (torch.tensor([1, 2]), torch.tensor([3, 4])))
    if wrapper == "payload":
        kernel = MagicMock()
        monkeypatch.setattr("vllm_ascend.ops.triton.qfa_kv_scatter._scatter_kv_payload_kernel", kernel)
        source = torch.ones(tokens + 1, heads, DIM, dtype=torch.int8, device="npu")
        message = f"KV payload contains {source.numel()} elements; expected {tokens * heads * DIM}."
        with pytest.raises(ValueError, match=message):
            scatter_kv_payload(source, views[0], slot_index)
    else:
        kernel = MagicMock()
        monkeypatch.setattr("vllm_ascend.ops.triton.qfa_kv_scatter._scatter_k_descale_kernel", kernel)
        source = torch.ones(tokens + 1, heads, dtype=torch.float32, device="npu")
        message = f"K descale contains {source.numel()} elements; expected {tokens * heads}."
        with pytest.raises(ValueError, match=message):
            scatter_k_descale(source, views[1], slot_index)
    kernel.__getitem__.assert_not_called()
    torch.testing.assert_close(raw.cpu(), before, rtol=0, atol=0)


@ASCEND_ONLY
def test_padded_rows_land_only_on_the_null_slot():
    """The caller clamps PAD slots to slot 0; the write must match the
    clamped reference exactly, so every slot a request can own is untouched."""
    heads, blocks = 2, 8
    raw = torch.full((blocks * _pitch(heads),), 13, dtype=torch.int8, device="npu")
    views = _views_on(raw, heads, blocks)
    expected = raw.cpu()
    cpu_views = _views_on(expected, heads, blocks)
    slots = torch.tensor([200, -1, 900], dtype=torch.int64)
    safe = slots.clamp(min=0)
    k, v, kd = _payloads(slots.shape[0], heads, 11)
    _write_reference(cpu_views, safe, k, v, kd)
    slot_index = tuple(index.npu() for index in (safe // BLOCK_SIZE, safe % BLOCK_SIZE))
    _scatter(views, k.npu(), v.npu(), kd.npu(), slot_index)
    torch.npu.synchronize()
    torch.testing.assert_close(raw.cpu(), expected, rtol=0, atol=0)


@ASCEND_ONLY
def test_back_to_back_launches_last_write_wins():
    heads, blocks = 2, 4
    raw = torch.full((blocks * _pitch(heads),), 13, dtype=torch.int8, device="npu")
    views = _views_on(raw, heads, blocks)
    expected = raw.cpu()
    cpu_views = _views_on(expected, heads, blocks)
    slots = torch.tensor([100], dtype=torch.int64)
    k1, v1, kd1 = _payloads(1, heads, 21)
    k2, v2, kd2 = _payloads(1, heads, 22)
    _write_reference(cpu_views, slots, k1, v1, kd1)
    _write_reference(cpu_views, slots, k2, v2, kd2)
    slot_index = tuple(index.npu() for index in (slots // BLOCK_SIZE, slots % BLOCK_SIZE))
    _scatter(views, k1.npu(), v1.npu(), kd1.npu(), slot_index)
    _scatter(views, k2.npu(), v2.npu(), kd2.npu(), slot_index)
    torch.npu.synchronize()
    torch.testing.assert_close(raw.cpu(), expected, rtol=0, atol=0)
