# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 The vllm-ascend contributors

from dataclasses import replace
from types import SimpleNamespace

import torch
from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
from vllm.v1.kv_cache_interface import MLAAttentionSpec

import vllm_ascend.worker.utils as worker_utils
from vllm_ascend.worker.utils import AscendKVBlockZeroer, copy_kv_cache_blocks_inplace

MANAGER_BLOCK_SIZE = 384
KERNEL_BLOCK_SIZE = 128
RATIO = MANAGER_BLOCK_SIZE // KERNEL_BLOCK_SIZE
NUM_KV_HEADS = 1
FUSED_DIM = 576
DTYPE_SIZE = 2
PAGE_BYTES = 488448
SLOT_BYTES = PAGE_BYTES // RATIO
SLOT_ELEMENTS = SLOT_BYTES // DTYPE_SIZE
SLOT_LOGICAL_BYTES = KERNEL_BLOCK_SIZE * FUSED_DIM * DTYPE_SIZE


def _make_k3_fused_cache(*, num_blocks: int = 2):
    logical_spec = MLAAttentionSpec(
        block_size=MANAGER_BLOCK_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        head_size=FUSED_DIM,
        dtype=torch.bfloat16,
    )
    spec = replace(logical_spec, page_size_padded=PAGE_BYTES)
    raw = torch.zeros(num_blocks * PAGE_BYTES, dtype=torch.uint8)
    typed_raw = raw.view(torch.bfloat16)
    fused = torch.as_strided(
        typed_raw,
        size=(num_blocks * RATIO, KERNEL_BLOCK_SIZE, NUM_KV_HEADS, FUSED_DIM),
        stride=(SLOT_ELEMENTS, FUSED_DIM, FUSED_DIM, 1),
        storage_offset=0,
    )
    return spec, fused


def _make_k3_component_cache(*, num_blocks: int = 2):
    logical_spec = MLAAttentionSpec(
        block_size=MANAGER_BLOCK_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        head_size=FUSED_DIM,
        dtype=torch.bfloat16,
    )
    spec = replace(logical_spec, page_size_padded=PAGE_BYTES)
    raw = torch.zeros(num_blocks * PAGE_BYTES, dtype=torch.uint8)
    typed_raw = raw.view(torch.bfloat16)
    nope = torch.as_strided(
        typed_raw,
        size=(num_blocks * RATIO, KERNEL_BLOCK_SIZE, NUM_KV_HEADS, 512),
        stride=(SLOT_ELEMENTS, 512, 512, 1),
        storage_offset=0,
    )
    rope = torch.as_strided(
        typed_raw,
        size=(num_blocks * RATIO, KERNEL_BLOCK_SIZE, NUM_KV_HEADS, 64),
        stride=(SLOT_ELEMENTS, 64, 64, 1),
        storage_offset=KERNEL_BLOCK_SIZE * NUM_KV_HEADS * 512,
    )
    return spec, (nope, rope)


def _install_cpu_h2d(monkeypatch):
    monkeypatch.setattr(
        worker_utils,
        "async_tensor_h2d",
        lambda array, device: torch.from_numpy(array).to(device),
    )


def test_zeroer_uses_single_fused_mla_physical_page():
    spec, fused = _make_k3_fused_cache()
    zeroer = AscendKVBlockZeroer(torch.device("cpu"), pin_memory=False)
    zeroer.init_meta(
        [SimpleNamespace(kv_cache_spec=spec, kv_cache_group_id=0, layer_names=["layer"])],
        [[KERNEL_BLOCK_SIZE]],
        "auto",
        set(),
        {"layer": SimpleNamespace(kv_cache=fused)},
    )

    assert zeroer._meta is not None
    seg_addrs, page_size_el, _, n_segs = zeroer._meta
    assert len(seg_addrs) == 1
    assert page_size_el == PAGE_BYTES // 4
    assert n_segs == 1


def test_zeroer_dedupes_component_mla_views_sharing_physical_page():
    spec, (nope, rope) = _make_k3_component_cache()
    zeroer = AscendKVBlockZeroer(torch.device("cpu"), pin_memory=False)
    zeroer.init_meta(
        [SimpleNamespace(kv_cache_spec=spec, kv_cache_group_id=0, layer_names=["layer"])],
        [[KERNEL_BLOCK_SIZE]],
        "auto",
        set(),
        {"layer": SimpleNamespace(kv_cache=(nope, rope))},
    )

    assert zeroer._meta is not None
    seg_addrs, page_size_el, _, n_segs = zeroer._meta
    assert len(seg_addrs) == 1
    assert seg_addrs[0] == nope.data_ptr()
    assert page_size_el == PAGE_BYTES // 4
    assert n_segs == 1


def test_component_mla_cow_copies_both_logical_components(monkeypatch):
    num_blocks = 2
    _, (nope, rope) = _make_k3_component_cache(num_blocks=num_blocks)
    nope_payload = torch.rand((3, KERNEL_BLOCK_SIZE, NUM_KV_HEADS, 512), dtype=torch.bfloat16)
    rope_payload = torch.rand((3, KERNEL_BLOCK_SIZE, NUM_KV_HEADS, 64), dtype=torch.bfloat16)
    nope[3:6].copy_(nope_payload)
    rope[3:6].copy_(rope_payload)

    _install_cpu_h2d(monkeypatch)
    copy_kv_cache_blocks_inplace(
        [(nope, rope)],
        num_blocks,
        [KVCacheBlockCopy(src_block_id=1, dst_block_id=0)],
    )

    torch.testing.assert_close(nope[0:3], nope_payload)
    torch.testing.assert_close(rope[0:3], rope_payload)


def test_fused_mla_cow_copies_complete_manager_block(monkeypatch):
    num_blocks = 2
    _, fused = _make_k3_fused_cache(num_blocks=num_blocks)
    payload = ((torch.arange(SLOT_LOGICAL_BYTES, dtype=torch.int64) * 31 + 17) % 251).to(torch.uint8)
    fused[3].copy_(payload.view(KERNEL_BLOCK_SIZE, NUM_KV_HEADS, FUSED_DIM))

    _install_cpu_h2d(monkeypatch)
    copy_kv_cache_blocks_inplace(
        [fused],
        num_blocks,
        [KVCacheBlockCopy(src_block_id=1, dst_block_id=0)],
    )

    expected = payload.view(KERNEL_BLOCK_SIZE, NUM_KV_HEADS, FUSED_DIM)
    torch.testing.assert_close(fused[0], expected)
    torch.testing.assert_close(fused[1], expected)
    torch.testing.assert_close(fused[2], expected)
    assert not torch.equal(fused[4], expected)
