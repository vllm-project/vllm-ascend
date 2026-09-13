# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 The vllm-ascend contributors

from dataclasses import replace
from types import SimpleNamespace

import torch
import vllm.v1.worker.utils as upstream_utils
from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
from vllm.v1.kv_cache_interface import MLAAttentionSpec

from vllm_ascend.patch.worker.patch_copy_kv_cache import (
    _fused_page_view,
    _is_fused_mla_pair,
    copy_kv_cache_blocks_inplace,
)
from vllm_ascend.worker.utils import AscendKVBlockZeroer

MANAGER_BLOCK_SIZE = 384
KERNEL_BLOCK_SIZE = 128
RATIO = MANAGER_BLOCK_SIZE // KERNEL_BLOCK_SIZE
NUM_KV_HEADS = 1
NOPE_DIM = 512
ROPE_DIM = 64
FUSED_DIM = NOPE_DIM + ROPE_DIM
DTYPE_SIZE = 2
PAGE_BYTES = 488448
SLOT_BYTES = PAGE_BYTES // RATIO
SLOT_ELEMENTS = SLOT_BYTES // DTYPE_SIZE


def _make_k3_fused_pair(*, num_blocks: int = 2):
    logical_spec = MLAAttentionSpec(
        block_size=MANAGER_BLOCK_SIZE,
        num_kv_heads=1,
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
    return spec, fused[..., :NOPE_DIM], fused[..., NOPE_DIM:]


def _install_cpu_h2d(monkeypatch):
    monkeypatch.setattr(
        upstream_utils,
        "async_tensor_h2d",
        lambda array, device: torch.from_numpy(array).to(device),
    )


def test_zeroer_recognizes_one_fused_mla_physical_page():
    spec, nope, rope = _make_k3_fused_pair()
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
    assert page_size_el == PAGE_BYTES // 4
    assert n_segs == 1


def test_fused_mla_cow_copies_complete_manager_page(monkeypatch):
    num_blocks = 2
    _, nope, rope = _make_k3_fused_pair(num_blocks=num_blocks)
    assert _is_fused_mla_pair((nope, rope))

    page_view = _fused_page_view(nope, rope)
    assert page_view.shape == (num_blocks * RATIO, SLOT_BYTES)
    payload = ((torch.arange(SLOT_BYTES, dtype=torch.int64) * 31 + 17) % 251).to(torch.uint8)
    page_view[3].copy_(payload)

    _install_cpu_h2d(monkeypatch)
    copy_kv_cache_blocks_inplace(
        [(nope, rope)],
        2,
        [KVCacheBlockCopy(src_block_id=1, dst_block_id=0)],
    )

    torch.testing.assert_close(page_view[0], payload)
    torch.testing.assert_close(page_view[1], payload)
    torch.testing.assert_close(page_view[2], payload)
    assert not torch.equal(page_view[4], payload)
