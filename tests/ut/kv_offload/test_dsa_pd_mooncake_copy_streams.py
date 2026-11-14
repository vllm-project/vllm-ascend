# SPDX-License-Identifier: Apache-2.0
"""Multi-stream landing must join every copy before reuse, return, or failure."""

from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend import envs
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


@pytest.mark.parametrize("streams", [1, 2, 4, 7])
@pytest.mark.parametrize("metadata", [False, True])
def test_stream_counts_preserve_bytes_batching_workers_and_reuse(monkeypatch, streams, metadata):
    case = build(monkeypatch, copy_streams=streams, metadata=metadata)
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
            assert [(bank.completion_event, bank.copy_events) for bank in banks] == events
        contexts = banks
        events = [(bank.completion_event, bank.copy_events) for bank in banks]
        for index, bank in enumerate(banks):
            expected = min(streams, bank.slot_count)
            assert len(bank.copy_streams) == expected
            submissions = [row for row in case.runtime.copy_submissions if row[0] == index]
            assert len({row[1] for row in submissions}) == expected
            assert len({thread for bank_id, thread in case.runtime.batch_threads if bank_id == index}) == 1
        case.runtime.batch_threads.clear()
    assert len(case.runtime.reads) == 2 * 10
    assert len(case.runtime.events) == BANKS * min(streams, 4)
    assert BANKS <= len(case.runtime.threads) <= 2 * BANKS


def test_stream_configuration_changes_only_after_prior_copies_drain(monkeypatch):
    case = build(monkeypatch)
    banks = None
    for streams in (1, 4, 2, 1):
        monkeypatch.setenv("VLLM_ASCEND_DSA_PD_MOONCAKE_COPY_STREAMS_PER_BANK", str(streams))
        case.source.add_(1)
        assert case.reader.batch_transfer_sync_read(*case.args, **case.kwargs) == 0
        assert_bytes(case)
        case.runtime.assert_drained()
        current = case.reader._dsa_pd_mooncake_staging_banks
        assert banks is None or current is banks
        banks = current
        assert all(len(bank.copy_streams) == streams for bank in banks)


@pytest.mark.parametrize("banks,streams", [(1, 16), (2, 8), (4, 4), (8, 2), (16, 1), (4, 2), (4, 1), (2, 1), (8, 1)])
def test_bank_and_stream_counts_are_independent_with_fixed_staging(monkeypatch, banks, streams):
    case = build(monkeypatch, banks=banks, copy_streams=streams, count=65)
    assert case.reader.batch_transfer_sync_read(*case.args, **case.kwargs) == 0
    assert_bytes(case)
    case.runtime.assert_drained()
    contexts = case.reader._dsa_pd_mooncake_staging_banks
    assert len(contexts) == banks
    assert sum(len(bank.copy_streams) for bank in contexts) == banks * streams
    assert len(case.runtime.reads) == (65 + 16 // banks - 1) // (16 // banks)
    assert case.reader._dsa_pd_mooncake_staging_bytes == 32 * 1024 * 1024


def test_single_copy_does_not_wait_on_unused_auxiliary_events(monkeypatch):
    case = build(monkeypatch, banks=1, copy_streams=16, count=1)
    assert case.reader.batch_transfer_sync_read(*case.args, **case.kwargs) == 0
    assert_bytes(case)
    assert len(case.runtime.copy_submissions) == 1
    assert sum(event.records for event in case.runtime.events) == 1
    case.runtime.assert_drained()


def test_overlapping_copies_are_rejected_before_any_read(monkeypatch):
    case = build(monkeypatch, copy_streams=4)
    case.runtime.workers = 0
    case.runtime.barrier = None
    args = (case.args[0], [case.host.data_ptr() + 16] * case.count, *case.args[2:])
    with pytest.raises(ValueError, match="Overlapping Mooncake Host destination"):
        case.reader.batch_transfer_sync_read(*args, **case.kwargs)
    assert not case.runtime.reads and not case.runtime.copy_submissions
    assert not torch.count_nonzero(case.host)
    case.runtime.assert_drained()


@pytest.mark.parametrize("reason", ["probe", "pipeline_off"])
def test_disabled_pipeline_and_probe_keep_one_copy_stream(monkeypatch, reason):
    case = build(monkeypatch, copy_streams=4)
    case.runtime.workers = 0
    case.runtime.barrier = None
    if reason == "probe":
        monkeypatch.setenv("VLLM_ASCEND_DSA_PD_MOONCAKE_READ_PROBE", "1")
    else:
        case.reader._pipeline_landing_enabled = False
    assert case.reader.batch_transfer_sync_read(*case.args, **case.kwargs) == 0
    assert all(len(bank.copy_streams) == 1 for bank in case.reader._dsa_pd_mooncake_staging_banks)
    assert_bytes(case)
    case.runtime.assert_drained()


@pytest.mark.parametrize("failure", ["auxiliary_record", "join", "copy"])
def test_partial_submission_failure_drains_all_streams_and_allows_retry(monkeypatch, failure):
    case = build(monkeypatch, copy_streams=4)
    failed = False

    def fail_once(index):
        nonlocal failed
        if not failed and (failure != "auxiliary_record" or index >= BANKS):
            failed = True
            raise RuntimeError("injected landing failure")

    if failure == "auxiliary_record":
        case.runtime.event_hook = fail_once
    elif failure == "join":
        case.runtime.wait_hook = fail_once
    else:
        case.runtime.copy_hook = fail_once
    with pytest.raises(RuntimeError, match="injected landing failure"):
        case.reader.batch_transfer_sync_read(*case.args, **case.kwargs)
    case.runtime.assert_drained()
    assert case.reader.batch_transfer_sync_read(*case.args, **case.kwargs) == 0
    assert_bytes(case)
    case.runtime.assert_drained()


def test_failed_stream_drain_still_drains_peers_and_retains_tensor_owners():
    first = MagicMock()
    first.synchronize.side_effect = RuntimeError("stream drain failed")
    second = MagicMock()
    owner = torch.empty(16)
    bank = staging._DSAPDMooncakeStagingBank(
        0, 4, first, copy_streams=(first, second), needs_stream_drain=True, inflight_owners=[(owner,)]
    )
    reader = staging.MooncakeSFAStagingReader.__new__(staging.MooncakeSFAStagingReader)
    with pytest.raises(RuntimeError, match="stream drain failed"):
        reader._drain_dsa_pd_mooncake_staging_banks((bank,))
    second.synchronize.assert_called_once()
    assert bank.needs_stream_drain and bank.inflight_owners[0][0] is owner
    first.synchronize.side_effect = None
    reader._drain_dsa_pd_mooncake_staging_banks((bank,))
    assert not bank.needs_stream_drain and not bank.inflight_owners


@pytest.mark.parametrize("value", ["0", "-1", "invalid"])
def test_invalid_stream_environment_is_rejected(monkeypatch, value):
    name = "VLLM_ASCEND_DSA_PD_MOONCAKE_COPY_STREAMS_PER_BANK"
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError):
        envs.env_variables[name]()


def test_stream_environment_defaults_to_one(monkeypatch):
    name = "VLLM_ASCEND_DSA_PD_MOONCAKE_COPY_STREAMS_PER_BANK"
    monkeypatch.delenv(name, raising=False)
    assert envs.env_variables[name]() == 1
