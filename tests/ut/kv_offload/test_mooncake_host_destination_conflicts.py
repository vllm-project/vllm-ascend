# SPDX-License-Identifier: Apache-2.0
"""Invalid Host destinations must fail before any READ or landing is submitted."""

import ctypes
from contextlib import nullcontext
from unittest.mock import MagicMock

import pytest
import torch

from .test_dsa_pd_mooncake_connector_v2 import make_reader


@pytest.mark.parametrize("pipeline", [False, True])
@pytest.mark.parametrize("streams", [1, 4])
@pytest.mark.parametrize("metadata", [False, True])
@pytest.mark.parametrize("offsets", [(16, 16), (16, 24), (24, 16), (16, 20)])
def test_conflicting_host_ranges_reject_mixed_read_without_writing(monkeypatch, pipeline, streams, metadata, offsets):
    monkeypatch.setattr(torch.npu, "stream", lambda _stream: nullcontext())
    monkeypatch.setattr(torch.npu, "Event", MagicMock())
    monkeypatch.setattr(torch.npu, "Stream", MagicMock())
    monkeypatch.setattr(torch.npu, "synchronize", MagicMock())
    monkeypatch.setenv("VLLM_ASCEND_DSA_PD_MOONCAKE_COPY_STREAMS_PER_BANK", str(streams))
    reader = make_reader(MagicMock(), pipeline=pipeline, window=4, bank_workers=None)
    host, direct = torch.full((96,), -7, dtype=torch.int8), torch.full((16,), -9, dtype=torch.int8)
    reader.kv_registry.register_offloaded_tensor(host)
    reader.host_kv.manager.copy_sfa_kv = MagicMock()
    reader._get_dsa_pd_mooncake_bank_workers = MagicMock(side_effect=AssertionError("workers started"))
    sources = [torch.full((16,), value, dtype=torch.int8) for value in [1, 2, 3]]
    # Include HBM first so validation must precede the mixed direct READ.
    destinations = [direct.data_ptr(), *(host.data_ptr() + offset for offset in offsets)]
    kwargs = {}
    if metadata:
        table = reader.kv_registry.create_transfer_targets([[direct], [host]])
        kwargs["transfer_targets"] = [table[0][0], table[1][0], table[1][0]]
    with pytest.raises(ValueError, match="Overlapping Mooncake Host destination"):
        reader.batch_transfer_sync_read(
            "peer", destinations, [source.data_ptr() for source in sources], [16] * 3, "conflict", **kwargs
        )
    reader.engine.batch_transfer_sync_read.assert_not_called()
    reader.host_kv.manager.copy_sfa_kv.assert_not_called()
    assert torch.all(host == -7) and torch.all(direct == -9)
    assert not reader._dsa_pd_mooncake_staging_lock.locked()

    # A rejected request must leave the reader usable. Adjacent ranges are valid.
    def read(_session, local, remote, lengths):
        for dst, src, length in zip(local, remote, lengths):
            ctypes.memmove(dst, src, length)
        return 0

    def land(src, dst, lengths, count):
        for source, destination, length in zip(src.tolist(), dst.tolist(), lengths.tolist()):
            ctypes.memmove(destination, source, length)

    reader.engine.batch_transfer_sync_read.side_effect = read
    reader.host_kv.manager.copy_sfa_kv.side_effect = land
    reader._get_dsa_pd_mooncake_bank_workers = lambda _banks: 0
    destinations = [direct.data_ptr(), host.data_ptr() + 16, host.data_ptr() + 32]
    assert (
        reader.batch_transfer_sync_read(
            "peer", destinations, [source.data_ptr() for source in sources], [16] * 3, "valid", **kwargs
        )
        == 0
    )
    assert torch.all(direct == 1) and torch.all(host[16:32] == 2) and torch.all(host[32:48] == 3)
    assert torch.all(host[:16] == -7) and torch.all(host[48:] == -7)
