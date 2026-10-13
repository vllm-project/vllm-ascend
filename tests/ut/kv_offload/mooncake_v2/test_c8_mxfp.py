# SPDX-License-Identifier: Apache-2.0
"""C8 packet views must transfer dynamic payloads without copying static V scales."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec

from vllm_ascend.attention.attention_c8_mxfp import (
    fill_mxfp_v_scale_cache,
    mxfp_cache_spec,
    mxfp_cache_views_for_spec,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake import base_worker
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.base_worker import MooncakeBaseConnectorWorker
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_worker import MooncakePullRecvingThread

from .helpers import make_kv_cache_tensor, make_pp_metadata


def make_c8_cache(ratio: int, padding: int):
    spec = mxfp_cache_spec(
        FullAttentionSpec(block_size=512 * ratio, num_kv_heads=1, head_size=64, dtype=torch.float8_e4m3fn)
    )
    spec = replace(spec, page_size_padded=spec.page_size_bytes + ratio * padding)
    # Nonzero storage offset and guards catch writes outside a selected packet.
    backing = torch.full((128 + 4 * spec.page_size_bytes + 128,), 37, dtype=torch.int8)
    raw = backing[128:-128]
    return spec, backing, mxfp_cache_views_for_spec(raw, spec, 512)


def make_worker(spec, backing, *, tp_size: int, tp_rank: int, cache_dtype: str = "mxfp8"):
    worker = MooncakeBaseConnectorWorker.__new__(MooncakeBaseConnectorWorker)
    worker.vllm_config = SimpleNamespace(cache_config=SimpleNamespace(cache_dtype=cache_dtype))
    worker.ascend_config = SimpleNamespace(kvpp_config=SimpleNamespace(size=1))
    worker.pcp_rank, worker.pcp_size = 0, 1
    worker.tp_rank, worker.tp_size = tp_rank, tp_size
    worker.kv_cache_config = KVCacheConfig(
        num_blocks=4,
        kv_cache_tensors=[
            make_kv_cache_tensor(
                size=backing.numel() - 256,
                layers=["attn"],
                layer_stride=backing.numel() - 256,
                block_stride=spec.page_size_bytes,
            )
        ],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=["attn"], kv_cache_spec=spec)],
    )
    worker.engine_id, worker.te_rpc_port = f"engine-{tp_size}", 9000
    worker.block_size, worker.side_channel_host, worker.handshake_port = spec.block_size, "127.0.0.1", 5000
    return worker


@pytest.mark.parametrize("local_rank", range(4))
@pytest.mark.parametrize(("local_ratio", "remote_ratio"), [(1, 1), (2, 1), (1, 2), (4, 2)])
def test_tp8_to_tp4_c8_packets_preserve_local_v_scales(monkeypatch, local_rank, local_ratio, remote_ratio):
    local_spec, local_raw, local_caches = make_c8_cache(local_ratio, padding=256)
    remote_spec, remote_raw, remote_caches = make_c8_cache(remote_ratio, padding=512)
    engine = MagicMock()
    monkeypatch.setattr(base_worker, "global_te", engine)
    monkeypatch.setattr(base_worker, "validate_register_region_count", MagicMock())
    local_worker = make_worker(local_spec, local_raw, tp_size=4, tp_rank=local_rank)
    remote_worker = make_worker(remote_spec, remote_raw, tp_size=8, tp_rank=2 * local_rank)
    caches = {"attn": local_caches}
    local_worker.register_kv_caches(caches)
    remote_worker.register_kv_caches({"attn": remote_caches})
    local = local_worker.xfer_handshake_metadata
    remote = remote_worker.xfer_handshake_metadata

    assert local_worker.kv_caches is caches and len(caches["attn"]) == 4
    assert local.kv_caches_base_addr == [[cache.data_ptr() for cache in local_caches[:3]]]
    assert local.block_lens == [[32768, 32768, 1024]]
    assert local.block_strides == [[local_spec.page_size_bytes // local_ratio] * 3]
    assert local.block_size_scales == [[local_ratio] * 3]
    assert remote.block_size_scales == [[remote_ratio] * 3]
    assert local.block_shapes == [[tuple(cache.shape[1:]) for cache in local_caches[:3]]]
    # The registered backing still covers all four original views, including
    # scales and padding. Registration does not imply those bytes are copied.
    ptrs, lengths = engine.register_buffer.call_args_list[0].args
    for cache in local_caches:
        end = cache.data_ptr() + (cache.shape[0] - 1) * cache.stride(0) + cache[0].numel()
        assert any(start <= cache.data_ptr() and end <= start + size for start, size in zip(ptrs, lengths))

    remote_metadata = make_pp_metadata(
        layer_names=remote.layer_names,
        layer_block_sizes=remote.layer_block_sizes,
        block_shapes=remote.block_shapes,
        block_strides=remote.block_strides,
        block_lens=remote.block_lens,
        block_size_scales=remote.block_size_scales,
        tp_base_addrs={rank: remote.kv_caches_base_addr for rank in range(8)},
    )
    remote_metadata = replace(remote_metadata, block_size=remote_spec.block_size, num_blocks=4)
    thread = MooncakePullRecvingThread.__new__(MooncakePullRecvingThread)
    thread.tp_size, thread.tp_rank = 4, local_rank
    thread.pcp_size, thread.pcp_rank, thread.dcp_size = 1, 0, 1
    thread.kv_cache_specs = [local_spec]
    for name in ("layer_names", "kv_caches_base_addr", "block_shapes", "block_strides", "block_lens"):
        setattr(thread, name, getattr(local, name))
    candidates = thread._get_layer_remote_tp_rank_groups(0, 0, local_spec, remote_metadata, 1, 8, 1)
    assert len(candidates) == 1
    assert all(rank // 4 == local_rank // 2 for rank in candidates[0])

    # Simulate a reassigned D page, restoring its static checkpoint scales
    # before receiving. The P-side scale deliberately contains corrupt bytes:
    # an otherwise correct transfer must not overwrite D's restored values.
    local_raw[128:-128].fill_(73)
    checkpoint_scale = torch.full((64,), 126 + local_rank // 2, dtype=torch.uint8)
    fill_mxfp_v_scale_cache(checkpoint_scale, local_caches[3])
    remote_caches[3].fill_(255)
    for cache_index, cache in enumerate(remote_caches[:3]):
        for block_id in range(cache.shape[0]):
            cache.view(torch.uint8)[block_id].fill_(10 + cache_index * 32 + block_id)

    local_blocks = [local_ratio, local_ratio + 1]
    remote_blocks = [4 * remote_ratio - 2, 4 * remote_ratio - 1]
    before_source = remote_raw.clone()
    expected = local_raw.clone()
    expected_caches = mxfp_cache_views_for_spec(expected[128:-128], local_spec, 512)
    for cache_index in range(3):
        for local_block, remote_block in zip(local_blocks, remote_blocks):
            expected_caches[cache_index].view(torch.uint8)[local_block].copy_(
                remote_caches[cache_index].view(torch.uint8)[remote_block]
            )

    local_addrs, remote_addrs, sizes = [], [], []
    thread._append_spec_transfer_addresses(
        spec_index=0,
        remote_pcp_rank=0,
        remote_tp_rank=candidates[0][0],
        remote_pcp_size=1,
        remote_tp_size=8,
        remote_dcp_size=1,
        transfer_entries_by_layer={(0, 0): [("request", local_blocks, remote_blocks)]},
        remote_metadata=remote_metadata,
        src_list=local_addrs,
        dst_list=remote_addrs,
        length_list=sizes,
    )
    # Consecutive block IDs must remain separate for all three strided views.
    assert sizes == [32768, 32768, 32768, 32768, 1024, 1024]
    for local_addr, remote_addr, size in zip(local_addrs, remote_addrs, sizes):
        local_offset = local_addr - local_raw.data_ptr()
        remote_offset = remote_addr - remote_raw.data_ptr()
        assert 0 <= local_offset < local_offset + size <= local_raw.numel()
        assert 0 <= remote_offset < remote_offset + size <= remote_raw.numel()
        local_raw[local_offset : local_offset + size].copy_(remote_raw[remote_offset : remote_offset + size])
    # Check all bytes, not just payload values: V scales, other blocks,
    # padding and both guards must remain untouched by the transfer.
    assert torch.equal(local_raw, expected)
    assert torch.equal(remote_raw, before_source)


@pytest.mark.parametrize("count", [2, 3, 5])
def test_mxfp_registration_rejects_incomplete_or_extra_cache_views(monkeypatch, count):
    spec, backing, caches = make_c8_cache(1, 0)
    monkeypatch.setattr(base_worker, "global_te", MagicMock())
    worker = make_worker(spec, backing, tp_size=4, tp_rank=0)
    malformed = (caches + (caches[3],))[:count]
    with pytest.raises(ValueError, match="requires .*K_scale, V_scale"):
        worker.register_kv_caches({"attn": malformed})
    base_worker.global_te.register_buffer.assert_not_called()


def test_non_mxfp_four_tensor_cache_keeps_all_views(monkeypatch):
    spec, backing, caches = make_c8_cache(1, 0)
    monkeypatch.setattr(base_worker, "global_te", MagicMock())
    monkeypatch.setattr(base_worker, "validate_register_region_count", MagicMock())
    worker = make_worker(spec, backing, tp_size=1, tp_rank=0, cache_dtype="fp8")
    worker.register_kv_caches({"attn": caches})
    assert worker.xfer_handshake_metadata.kv_caches_base_addr == [[cache.data_ptr() for cache in caches]]
