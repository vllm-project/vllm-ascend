# SPDX-License-Identifier: Apache-2.0
"""Exercise partial first slots and aligned tails through READ and landing."""

import ctypes
from contextlib import nullcontext
from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_sfa import transfer_targets as registry_module

from .test_dsa_pd_mooncake_connector_v2 import make_reader

registry = registry_module.MooncakeKVRegistry()

ALIGNMENT = 2 * 1024 * 1024


@pytest.fixture(autouse=True)
def reset_registry(monkeypatch):
    registry.clear()
    monkeypatch.setattr(torch.npu, "stream", lambda _stream: nullcontext())
    monkeypatch.setattr(torch.npu, "Event", MagicMock())
    monkeypatch.setattr(torch.npu, "synchronize", MagicMock())
    yield
    registry.clear()


@pytest.mark.parametrize("chunk_units", [1, 2])
@pytest.mark.parametrize("page_offset", [0, 1, ALIGNMENT - 1])
@pytest.mark.parametrize("pipeline", [False, True])
@pytest.mark.parametrize("metadata", [False, True])
def test_partial_first_slot_and_aligned_tails_preserve_mixed_landing(chunk_units, page_offset, pipeline, metadata):
    chunk = chunk_units * ALIGNMENT
    length = 2 * chunk + 17
    target_offset = 13
    host = torch.zeros(length + 2 * target_offset, dtype=torch.int8)
    direct = torch.zeros(16, dtype=torch.int8)
    registry.register_offloaded_tensor(host)
    engine = MagicMock()
    reader = make_reader(engine, pipeline=pipeline, window=4, kv_registry=registry)
    reader.block_len_per_addr = [[chunk]]
    reader._dsa_pd_mooncake_staging_tensor = torch.empty(4 * chunk, dtype=torch.int8)
    reader._dsa_pd_mooncake_staging_bytes = 4 * chunk
    staging_ptr = reader._dsa_pd_mooncake_staging_tensor.data_ptr()
    remote_base = 8 * ALIGNMENT + page_offset
    pieces: list[tuple[int, int, int]] = []
    direct_reads = []

    def read(_session, local, remote, lengths):
        assert reader._dsa_pd_mooncake_staging_lock.locked()
        for dst, src, size in zip(local, remote, lengths):
            if dst == direct.data_ptr():
                direct_reads.append((src, size))
                ctypes.memset(dst, 7, size)
                continue
            slot_offset = (dst - staging_ptr) % chunk
            assert slot_offset == src % ALIGNMENT
            assert slot_offset + size <= chunk
            assert staging_ptr <= dst and dst + size <= staging_ptr + 4 * chunk
            value = len(pieces) + 1
            ctypes.memset(dst, value, size)
            pieces.append((src, size, value))
        return 0

    engine.batch_transfer_sync_read.side_effect = read
    kwargs = {}
    if metadata:
        table = registry.create_transfer_targets([[host], [direct]])
        kwargs["transfer_targets"] = [table[0][0], table[1][0]]
    assert (
        reader.batch_transfer_sync_read(
            "peer",
            [host.data_ptr() + target_offset, direct.data_ptr()],
            [remote_base, 1024],
            [length, direct.numel()],
            "request",
            **kwargs,
        )
        == 0
    )
    assert direct_reads == [(1024, direct.numel())]
    assert torch.all(direct == 7)
    assert len(pieces) == (length + page_offset + chunk - 1) // chunk
    consumed = 0
    for src, size, value in pieces:
        assert src == remote_base + consumed
        assert torch.all(host[target_offset + consumed : target_offset + consumed + size] == value)
        consumed += size
    assert consumed == length
    assert torch.count_nonzero(host[:target_offset]) == 0
    assert torch.count_nonzero(host[target_offset + length :]) == 0
    assert not reader._dsa_pd_mooncake_staging_lock.locked()
