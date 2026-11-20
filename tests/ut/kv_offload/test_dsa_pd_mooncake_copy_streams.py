# SPDX-License-Identifier: Apache-2.0
"""Each bank owns one copy stream and retains buffers until its copies stop."""

from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_sfa import staging

from .test_dsa_pd_mooncake_bank_workers import BANKS, HOST_STRIDE, PIECE_BYTES, build


def assert_bytes(case):
    for index in range(case.count):
        start = index * HOST_STRIDE
        assert torch.equal(
            case.host[start + 16 : start + 16 + PIECE_BYTES],
            case.source[index * PIECE_BYTES : (index + 1) * PIECE_BYTES],
        )
        assert torch.count_nonzero(case.host[start : start + 16]) == 0
        assert torch.count_nonzero(case.host[start + 16 + PIECE_BYTES : start + HOST_STRIDE]) == 0


@pytest.mark.parametrize("metadata", [False, True])
def test_single_stream_per_bank_preserves_bytes_events_and_repeated_receives(monkeypatch, metadata):
    case = build(monkeypatch, metadata=metadata)
    contexts = None
    events = None
    for value in (1, 2):
        case.source.fill_(value)
        case.host.zero_()
        assert case.reader.batch_transfer_sync_read(*case.args, **case.kwargs) == 0
        assert_bytes(case)
        case.runtime.assert_drained()
        banks = case.reader._dsa_pd_mooncake_staging_banks
        if contexts is not None:
            assert banks is contexts
            assert [bank.completion_event for bank in banks] == events
        contexts = banks
        events = [bank.completion_event for bank in banks]
        for index, bank in enumerate(banks):
            submissions = [row for row in case.runtime.copy_submissions if row[0] == index]
            assert {row[1] for row in submissions} == {index}
            assert len({thread for bank_id, thread in case.runtime.batch_threads if bank_id == index}) == 1
        case.runtime.batch_threads.clear()
    assert len(case.runtime.reads) == 20
    assert len(case.runtime.events) == BANKS
    assert len(case.runtime.queues) == BANKS


@pytest.mark.parametrize("banks", [1, 2, 4, 8, 16])
def test_bank_count_changes_batching_without_changing_staging_capacity(monkeypatch, banks):
    case = build(monkeypatch, banks=banks, count=65)
    assert case.reader.batch_transfer_sync_read(*case.args, **case.kwargs) == 0
    assert_bytes(case)
    case.runtime.assert_drained()
    contexts = case.reader._dsa_pd_mooncake_staging_banks
    assert len(contexts) == banks
    assert len(case.runtime.queues) == banks
    assert len(case.runtime.reads) == (65 + 16 // banks - 1) // (16 // banks)
    assert case.reader._dsa_pd_mooncake_staging_bytes == 32 * 1024 * 1024


@pytest.mark.parametrize("failure", ["record", "copy"])
def test_partial_submission_failure_drains_banks_and_allows_retry(monkeypatch, failure):
    case = build(monkeypatch)
    failed = False

    def fail_once(_index):
        nonlocal failed
        if not failed:
            failed = True
            raise RuntimeError("injected landing failure")

    if failure == "record":
        case.runtime.event_hook = fail_once
    else:
        case.runtime.copy_hook = fail_once
    with pytest.raises(RuntimeError, match="injected landing failure"):
        case.reader.batch_transfer_sync_read(*case.args, **case.kwargs)
    case.runtime.assert_drained()
    assert case.reader.batch_transfer_sync_read(*case.args, **case.kwargs) == 0
    assert_bytes(case)
    case.runtime.assert_drained()


def test_failed_bank_drain_still_drains_peers_and_retains_tensor_owners():
    first, second = MagicMock(), MagicMock()
    first.synchronize.side_effect = RuntimeError("stream drain failed")
    owner = torch.empty(16)
    banks = tuple(
        staging._DSAPDMooncakeStagingBank(index, 4, stream, needs_stream_drain=True, inflight_owners=[(owner,)])
        for index, stream in enumerate((first, second))
    )
    reader = staging.MooncakeSFAStagingReader.__new__(staging.MooncakeSFAStagingReader)
    with pytest.raises(RuntimeError, match="stream drain failed"):
        reader._drain_dsa_pd_mooncake_staging_banks(banks)
    second.synchronize.assert_called_once()
    assert banks[0].needs_stream_drain and banks[0].inflight_owners[0][0] is owner
    assert not banks[1].needs_stream_drain and not banks[1].inflight_owners
    first.synchronize.side_effect = None
    reader._drain_dsa_pd_mooncake_staging_banks(banks)
    assert all(not bank.needs_stream_drain and not bank.inflight_owners for bank in banks)
