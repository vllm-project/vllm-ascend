# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

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
    scheduler.main_group_idx = 1
    scheduler.indexer_group_idx = 0
    scheduler._dsa_requests = {}
    scheduler._dsa_prefill_tp_size = 2
    scheduler._expected_tp_ranks = frozenset((0, 1))
    return scheduler


def _request():
    return SimpleNamespace(
        request_id="request",
        prompt_token_ids=[1, 2, 3, 4],
        num_computed_tokens=0,
        kv_transfer_params={
            "do_remote_prefill": True,
            "remote_request_id": "remote",
            "remote_block_ids": ((10, 11), (20, 21)),
            "remote_host": "127.0.0.1",
            "remote_port": 5000,
            "remote_engine_id": "prefill",
            "remote_multi_nodes_meta_mapping": {},
        },
    )


def test_dsa_scheduler_emits_once_and_waits_for_all_tp_results():
    scheduler = _scheduler()
    request = _request()
    assert scheduler.get_num_new_matched_tokens(request, 0) == (4, True)
    blocks = SimpleNamespace(
        get_block_ids=lambda: ([30, 31], [40, 41])
    )
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


def test_dsa_scheduler_failure_requests_local_recompute():
    scheduler = _scheduler()
    request = _request()
    scheduler.get_num_new_matched_tokens(request, 0)
    scheduler.update_state_after_alloc(
        request,
        SimpleNamespace(get_block_ids=lambda: ([30, 31], [40, 41])),
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
    scheduler.update_connector_output(output)
    assert request.num_computed_tokens == 0
    assert output.finished_recving == {"request"}
