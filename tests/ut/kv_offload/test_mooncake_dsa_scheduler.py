# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.ascend_config import SparseKVOffloadConfig
from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, AscendSFAIndexerCacheSpec
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
    scheduler._prefill_pp_size = 1
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
def test_failure_reporting_excludes_untouched_cached_prefix(group0_block_size, prefix_tokens):
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


@pytest.mark.parametrize("prefill_pp_size", [1, 2])
def test_dsa_scheduler_initializes_main_groups_without_copy_sfa_state(prefill_pp_size):
    main = AscendMLAAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.bfloat16,
        cache_dtype_str="auto",
        store_on_host=True,
    )
    indexer = AscendSFAIndexerCacheSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
    )
    kv_config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(layer_names=["model.layers.0.self_attn.attn"], kv_cache_spec=main),
            SimpleNamespace(layer_names=["model.layers.0.indexer.attn.index_cache"], kv_cache_spec=indexer),
        ]
    )
    extra = {"prefill": {"tp_size": 4, "pp_size": prefill_pp_size}}
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            engine_id="decode",
            kv_port=5000,
            get_from_extra_config=lambda key, default: extra.get(key, default),
        ),
        parallel_config=SimpleNamespace(
            data_parallel_rank=2,
            data_parallel_size=4,
            tensor_parallel_size=2,
        ),
        speculative_config=None,
    )
    offload = SparseKVOffloadConfig(
        enabled=True,
        host_backend="mooncake",
        use_fused_overlap=True,
    )
    with patch(
        "vllm_ascend.ascend_config.get_ascend_config",
        return_value=SimpleNamespace(sparse_kv_offload_config=offload),
    ):
        scheduler = _MooncakeDsaDecodeScheduler(config, kv_config)
    try:
        assert scheduler.main_group_idx == 0
        assert scheduler.indexer_group_idx == 1
        assert scheduler._main_block_size == 128
        assert scheduler._prefill_pp_size == prefill_pp_size
        assert scheduler.side_channel_port == 5004
        assert scheduler._expected_tp_ranks == frozenset((0, 1))
        assert scheduler._copy_sfa_hot_tokens == 0
        assert scheduler._copy_sfa_slot_allocator is None
    finally:
        scheduler.shutdown()


def test_dsa_scheduler_gets_pp_topology_from_configuration_without_wire_field():
    scheduler = _scheduler()
    scheduler._prefill_pp_size = 2
    request = _request()
    peers = {
        str(rank): {"host": f"192.0.2.{rank + 1}", "handshake_port": 6000 + rank, "engine_id": f"prefill-{rank}"}
        for rank in range(4)
    }
    request.kv_transfer_params["remote_multi_nodes_meta_mapping"] = peers
    assert "remote_pp_size" not in request.kv_transfer_params
    assert scheduler.get_num_new_matched_tokens(request, 0) == (4, True)
    source = scheduler._dsa_requests[request.request_id].source
    assert source.remote_pp_size == 2
    assert len(source.endpoints_by_prefill_rank) == 4
    for rank, endpoint in enumerate(source.endpoints_by_prefill_rank):
        assert endpoint.remote_host == peers[str(rank)]["host"]
        assert endpoint.remote_port == peers[str(rank)]["handshake_port"]
        assert endpoint.remote_engine_id == peers[str(rank)]["engine_id"]


def test_dsa_scheduler_keeps_endpoint_coverage_check_for_configured_pp():
    scheduler = _scheduler()
    scheduler._prefill_pp_size = 2
    request = _request()
    request.kv_transfer_params["remote_multi_nodes_meta_mapping"] = {
        str(rank): {"host": "192.0.2.1", "handshake_port": 6000 + rank} for rank in range(2)
    }
    with pytest.raises(ValueError, match="remote endpoint mapping must cover Prefill TP ranks"):
        scheduler.get_num_new_matched_tokens(request, 0)


def test_on_new_request_is_a_noop_hook():
    """D-side requests must not be touched before they are scheduled.

    vLLM calls ``on_new_request`` on the connector for every new request, so
    the DSA decode scheduler has to expose it; the P-side prompt truncation it
    enables elsewhere does not apply here.
    """
    scheduler = _scheduler()
    request = _request()
    params_before = dict(request.kv_transfer_params)

    scheduler.on_new_request(request)

    assert scheduler._dsa_requests == {}
    assert request.kv_transfer_params == params_before
