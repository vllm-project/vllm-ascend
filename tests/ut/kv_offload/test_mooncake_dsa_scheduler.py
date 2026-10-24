# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector import (
    _MooncakeDsaDecodeScheduler,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_dsa_metadata import (
    DsaLocalResult,
    DsaLocalResultKind,
    DsaTransferPhase,
    DsaWorkerResultMetadata,
)


def _scheduler():
    scheduler = object.__new__(_MooncakeDsaDecodeScheduler)
    scheduler._main_block_size = 2
    scheduler.block_size = [2, 2]
    scheduler.main_group_idx = 1
    scheduler.indexer_group_idx = 0
    scheduler._dsa_requests = {}
    scheduler._dsa_prefill_tp_size = 2
    scheduler._expected_tp_ranks = frozenset((0, 1))
    return scheduler


def _blocks(group0, group1, unhashed_group0=None):
    if unhashed_group0 is None:
        unhashed_group0 = group0
    return SimpleNamespace(
        get_block_ids=lambda: (group0, group1),
        get_unhashed_block_ids_all_groups=lambda: (
            unhashed_group0,
            group1,
        ),
    )


def _request():
    return SimpleNamespace(
        request_id="request",
        prompt_token_ids=[1, 2, 3, 4],
        num_computed_tokens=0,
        kv_transfer_params={
            "do_remote_prefill": True,
            "remote_request_id": "remote",
            "dsa_main_group_id": 1,
            "dsa_indexer_group_id": 0,
            "remote_block_ids": ((10, 11), (20, 21)),
            "remote_host": "127.0.0.1",
            "remote_port": 5000,
            "remote_engine_id": "prefill",
            "remote_multi_nodes_meta_mapping": {},
        },
    )


@pytest.mark.parametrize("group0_block_size,prefix_tokens", [(2, 2), (1, 2)])
def test_failure_reporting_excludes_untouched_cached_prefix(
    group0_block_size, prefix_tokens
):
    scheduler = _scheduler()
    scheduler.block_size[0] = group0_block_size
    request = _request()
    external_tokens, _ = scheduler.get_num_new_matched_tokens(request, prefix_tokens)
    group0_ids = list(range(30, 30 + 4 // group0_block_size))
    blocks = _blocks(
        group0_ids,
        [40, 41],
        group0_ids[prefix_tokens // group0_block_size :],
    )
    scheduler.update_state_after_alloc(request, blocks, external_tokens)
    (command,) = scheduler.build_connector_meta(None).requests
    first_transferred_block = prefix_tokens // group0_block_size
    assert command.invalid_block_ids == tuple(group0_ids[first_transferred_block:])


@pytest.mark.parametrize(
    "group0_block_size,prefix_tokens,external_tokens,expected_invalid_ids",
    [
        (2, 0, 4, (30, 31)),
        (2, 1, 3, (30, 31)),
        (4, 2, 2, (30,)),
        (2, 4, 0, ()),
    ],
)
def test_failure_reporting_includes_overlapping_block_and_handles_notification_only(
    group0_block_size, prefix_tokens, external_tokens, expected_invalid_ids
):
    scheduler = _scheduler()
    scheduler.block_size[0] = group0_block_size
    request = _request()
    scheduler.get_num_new_matched_tokens(request, prefix_tokens)
    group0_ids = list(range(30, 30 + 4 // group0_block_size))
    blocks = _blocks(group0_ids, [40, 41], expected_invalid_ids)
    scheduler.update_state_after_alloc(request, blocks, external_tokens)
    (command,) = scheduler.build_connector_meta(None).requests
    assert command.invalid_block_ids == expected_invalid_ids
    assert command.notify_only == (external_tokens == 0)


def test_dsa_scheduler_emits_once_and_waits_for_all_tp_results():
    scheduler = _scheduler()
    request = _request()
    assert scheduler.get_num_new_matched_tokens(request, 0) == (4, True)
    blocks = _blocks([30, 31], [40, 41])
    scheduler.update_state_after_alloc(request, blocks, 4)

    metadata = scheduler.build_connector_meta(None)
    assert len(metadata.requests) == 1
    assert scheduler.build_connector_meta(None).requests == ()
    command = metadata.requests[0]
    assert command.source.indexer_block_ids == (10, 11)
    assert command.source.main_block_ids == (20, 21)
    assert command.indexer_hbm_block_ids == (30, 31)
    assert command.main_host_block_ids == (40, 41)

    output = SimpleNamespace(
        kv_connector_worker_meta=DsaWorkerResultMetadata(
            (
                DsaLocalResult(
                    "request",
                    0,
                    DsaLocalResultKind.RECEIVE_COMPLETE,
                ),
            )
        ),
        finished_recving=set(),
    )
    scheduler.update_connector_output(output)
    assert output.finished_recving == set()

    output.kv_connector_worker_meta = DsaWorkerResultMetadata(
        (
            DsaLocalResult(
                "request",
                1,
                DsaLocalResultKind.RECEIVE_COMPLETE,
            ),
        )
    )
    scheduler.update_connector_output(output)
    assert output.finished_recving == {"request"}
    assert "request" not in scheduler._dsa_requests


def test_dsa_scheduler_failure_never_requests_local_recompute():
    scheduler = _scheduler()
    request = _request()
    scheduler.get_num_new_matched_tokens(request, 0)
    scheduler.update_state_after_alloc(
        request,
        _blocks([30, 31], [40, 41]),
        4,
    )
    output = SimpleNamespace(
        kv_connector_worker_meta=DsaWorkerResultMetadata(
            (
                DsaLocalResult(
                    "request",
                    0,
                    DsaLocalResultKind.TRANSFER_FAILED,
                    DsaTransferPhase.INDEXER_D2D,
                ),
                DsaLocalResult(
                    "request",
                    1,
                    DsaLocalResultKind.RECEIVE_COMPLETE,
                ),
            )
        ),
        finished_recving=set(),
    )
    request.num_computed_tokens = 3
    scheduler.update_connector_output(output)
    assert request.num_computed_tokens == 3
    assert output.finished_recving == {"request"}


def test_empty_external_receive_only_notifies_prefill():
    scheduler = _scheduler()
    request = _request()
    assert scheduler.get_num_new_matched_tokens(request, 4) == (0, False)
    scheduler.update_state_after_alloc(request, _blocks([], []), 0)
    (command,) = scheduler.build_connector_meta(None).requests
    assert command.notify_only
    assert command.main_host_block_ids == ()
    assert "request" not in scheduler._dsa_requests


def test_rejection_before_allocation_retains_notification():
    scheduler = _scheduler()
    request = _request()
    assert scheduler.request_finished(request, ()) == (False, None)
    (command,) = scheduler.build_connector_meta(None).requests
    assert command.notify_only
    assert not request.kv_transfer_params["do_remote_prefill"]


def test_cancel_keeps_delayed_free_until_rank_results():
    scheduler = _scheduler()
    request = _request()
    scheduler.get_num_new_matched_tokens(request, 0)
    scheduler.update_state_after_alloc(request, _blocks([30, 31], [40, 41]), 4)
    scheduler.build_connector_meta(None)
    assert scheduler.request_finished(request, ()) == (True, None)
    assert scheduler.build_connector_meta(None).cancelled_requests == ("request",)
    assert "request" in scheduler._dsa_requests


def test_source_group_order_is_independent_of_local_groups():
    scheduler = _scheduler()
    request = _request()
    request.kv_transfer_params.update(dsa_main_group_id=0, dsa_indexer_group_id=1)
    scheduler.get_num_new_matched_tokens(request, 0)
    source = scheduler._dsa_requests["request"].source
    assert source.main_block_ids == (10, 11)
    assert source.indexer_block_ids == (20, 21)
