# SPDX-License-Identifier: Apache-2.0

import queue
import threading
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.metadata import (
    MooncakeConnectorMetadata,
    ReqMeta,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_worker import (
    MooncakePullConnectorWorker,
    MooncakePullRecvingThread,
)

from .helpers import (
    make_full_spec,
    make_mamba_spec,
    make_metadata_groups,
    make_pp_metadata,
    make_sliding_spec,
    make_transfer_metadata,
)


def make_thread(**overrides: object) -> MooncakePullRecvingThread:
    thread = MooncakePullRecvingThread.__new__(MooncakePullRecvingThread)
    thread.tp_rank = 0
    thread.tp_size = 1
    thread.dcp_rank = 0
    thread.dcp_size = 1
    thread.num_speculative_tokens = 0
    thread.layer_names = ["model.layers.0.self_attn"]
    thread.group_indices = [0]
    thread.spec_indices = [0]
    thread.kv_cache_specs = [make_full_spec()]
    thread.local_metadata = make_transfer_metadata()
    thread.kv_caches_base_addr = [[1000]]
    thread.block_strides = [[128]]
    thread.block_lens = [[128]]
    thread.block_shapes = [[(1, 16, 4)]]
    thread.can_report_invalid_block_ids = True
    thread.invalid_block_ids = set()
    thread.invalid_block_ids_lock = threading.Lock()
    thread.finished_requests = queue.SimpleQueue()
    thread.request_queue = queue.Queue()
    for name, value in overrides.items():
        setattr(thread, name, value)
    return thread


def make_req_meta(
    *,
    local: tuple[list[int], ...] = ([10, 11],),
    remote: tuple[list[int], ...] = ([20, 21],),
    engine_id: str = "engine-p",
) -> ReqMeta:
    return ReqMeta(
        local_block_ids=local,
        local_num_prompt_tokens=32,
        num_external_tokens=16,
        num_computed_tokens=0,
        remote_block_ids=remote,
        remote_host="10.0.0.1",
        remote_port=6000,
        remote_engine_id=engine_id,
        remote_request_id="request-p",
        remote_num_prompt_tokens=32,
        local_full_block_ids=local,
    )


def test_remote_endpoint_requires_one_endpoint() -> None:
    assert MooncakePullRecvingThread._get_remote_endpoint({("host", 1)}) == ("host", 1)
    with pytest.raises(ValueError, match="must share one scheduler endpoint"):
        MooncakePullRecvingThread._get_remote_endpoint({("a", 1), ("b", 2)})


def test_result_queues_are_drained_atomically() -> None:
    thread = make_thread()
    thread.finished_requests.put("request-a")
    thread.finished_requests.put("request-b")
    thread.invalid_block_ids = {10, 11}

    assert thread.get_and_clear_finished_requests() == {"request-a", "request-b"}
    assert thread.get_and_clear_finished_requests() == set()
    assert thread.get_and_clear_invalid_block_ids() == {10, 11}
    assert thread.get_and_clear_invalid_block_ids() == set()


def test_failed_request_marks_all_local_groups_for_non_hybrid_cache() -> None:
    thread = make_thread()
    request = make_req_meta(local=([10, 11], [20]))

    thread._mark_request_failed(request)

    assert thread.invalid_block_ids == {10, 11, 20}


def test_hybrid_cache_failure_is_not_reported_as_invalid_blocks() -> None:
    thread = make_thread(can_report_invalid_block_ids=False)

    thread._mark_request_failed(make_req_meta(local=([10], [20])))

    assert thread.invalid_block_ids == set()


def test_attention_tp_groups_cover_replication_split_and_dcp() -> None:
    replicated = make_thread(tp_size=4, tp_rank=1)
    assert replicated._get_attention_remote_tp_rank_groups(8, 1, 1, 4) == [[2, 3]]

    split = make_thread(tp_size=2, tp_rank=0)
    assert split._get_attention_remote_tp_rank_groups(4, 1, 1, 8) == [[0], [1]]

    mla_dcp = make_thread(tp_size=4, tp_rank=2, dcp_size=4, dcp_rank=2)
    assert mla_dcp._get_attention_remote_tp_rank_groups(8, 4, 8, 1) == [list(range(8))]

    unequal_dcp = make_thread(tp_size=8, tp_rank=3, dcp_size=2, dcp_rank=1)
    assert unequal_dcp._get_attention_remote_tp_rank_groups(8, 2, 4, 4) == [[0, 1, 2, 3]]


def test_mamba_tp_groups_follow_non_replicated_tp_ratio() -> None:
    local_smaller = make_thread(tp_size=2, tp_rank=1)
    assert local_smaller._get_mamba_remote_tp_rank_groups(4) == [[2], [3]]

    local_larger = make_thread(tp_size=4, tp_rank=3)
    assert local_larger._get_mamba_remote_tp_rank_groups(2) == [[1]]

    with pytest.raises(ValueError, match="integer ratio"):
        local_larger._get_mamba_remote_tp_rank_groups(3)


def test_build_remote_layout_matches_layers_across_pp_ranks() -> None:
    thread = make_thread(
        layer_names=["layer.0", "layer.1"],
        spec_indices=[0, 0],
        kv_cache_specs=[make_full_spec()],
    )
    thread._get_layer_remote_tp_rank_groups = MagicMock(return_value=[[0]])  # type: ignore[method-assign]
    pp0 = make_pp_metadata(layer_names=["layer.0"])
    pp1 = make_pp_metadata(layer_names=["layer.1"], tp_base_addrs={0: [[6000]]})
    groups = make_metadata_groups()
    groups = SimpleNamespace(
        tp_size=1,
        dcp_size=1,
        metadata_by_pp_rank={0: pp0, 1: pp1},
    )

    rank_groups, pairs = thread._build_remote_transfer_layout(groups)  # type: ignore[arg-type]

    assert pairs == {0: [(0, 0)], 1: [(1, 0)]}
    assert rank_groups == {0: {(0, 0): [[0]]}, 1: {(1, 0): [[0]]}}


def test_build_remote_layout_rejects_missing_local_layer() -> None:
    thread = make_thread(layer_names=["layer.0", "missing"], spec_indices=[0, 0])
    thread._get_layer_remote_tp_rank_groups = MagicMock(return_value=[[0]])  # type: ignore[method-assign]
    groups = make_metadata_groups(pp_metadata=make_pp_metadata(layer_names=["layer.0"]))

    with pytest.raises(ValueError, match="missing layers.*missing"):
        thread._build_remote_transfer_layout(groups)


def test_expand_block_ids_and_round_robin_candidate_selection() -> None:
    assert MooncakePullRecvingThread._expand_block_ids([3, 5], 2) == [6, 7, 10, 11]
    assert MooncakePullRecvingThread._select_remote_tp_rank([2, 3, 4], 4) == 3


def test_compute_full_attention_blocks_skips_remote_prefix_and_balances_replica() -> None:
    thread = make_thread()

    result = thread._compute_group_block_ids(
        request_id="request",
        remote_tp_rank_groups=[[2, 3]],
        remote_dcp_size=1,
        spec_index=0,
        local_block_size=16,
        remote_block_size=16,
        local_group_block_ids=[10, 11],
        local_full_group_block_ids=[10, 11],
        remote_group_block_ids=[20, 21, 22],
        local_num_prompt_tokens=32,
        remote_num_prompt_tokens=32,
        num_computed_tokens=16,
        local_block_size_scale=1,
        remote_block_size_scale=1,
        spec=make_full_spec(),
        selection_index=1,
    )

    assert result == [(3, [10, 11], [21, 22])]


def test_compute_sliding_window_blocks_uses_unhashed_suffix() -> None:
    thread = make_thread()

    result = thread._compute_group_block_ids(
        "request",
        [[0]],
        1,
        0,
        16,
        16,
        [10, 11],
        [0, 0, 10, 11],
        [0, 0, 20, 21],
        64,
        64,
        32,
        1,
        1,
        make_sliding_spec(),
        0,
    )

    assert result == [(0, [10, 11], [20, 21])]


def test_compute_mamba_state_selects_pre_speculative_local_block() -> None:
    thread = make_thread(num_speculative_tokens=2)

    result = thread._compute_group_block_ids(
        "request",
        [[0]],
        1,
        0,
        16,
        16,
        [1, 2, 3, 4],
        [1, 2, 3, 4],
        [8, 9, 10],
        32,
        31,
        0,
        1,
        1,
        make_mamba_spec(),
        0,
    )

    assert result == [(0, [2], [10])]


def test_compute_dcp_blocks_maps_local_shards_to_remote_dcp_ranks() -> None:
    thread = make_thread(tp_size=4, tp_rank=1, dcp_size=2, dcp_rank=1)

    result = thread._compute_group_block_ids(
        "request",
        [[0, 1, 2, 3]],
        4,
        0,
        16,
        16,
        [10, 11],
        [10, 11],
        [20, 21],
        64,
        64,
        0,
        1,
        1,
        make_full_spec(),
        0,
    )

    assert result == [(1, [10], [20]), (3, [11], [20])]


def test_compute_dcp_blocks_supports_different_logical_block_sizes() -> None:
    thread = make_thread(dcp_size=2, dcp_rank=0)

    result = thread._compute_group_block_ids(
        "request",
        [[5]],
        1,
        0,
        16,
        32,
        [10, 11],
        [10, 11],
        [20, 21],
        65,
        64,
        0,
        1,
        2,
        make_full_spec(),
        0,
    )

    assert result == [(5, [10, 11], [40, 42])]


def test_transfer_bucket_reuses_block_mapping_for_layers_of_same_spec() -> None:
    thread = make_thread(
        layer_names=["layer.0", "layer.1"],
        group_indices=[0, 0],
        spec_indices=[0, 0],
        block_shapes=[[(1, 16, 4)], [(1, 16, 4)]],
        block_strides=[[128], [128]],
        block_lens=[[128], [128]],
        kv_caches_base_addr=[[1000], [2000]],
    )
    thread.local_metadata = make_transfer_metadata(
        layer_names=["layer.0", "layer.1"],
        spec_indices=[0, 0],
        spec_block_sizes=[16],
    )
    thread._compute_group_block_ids = MagicMock(return_value=[(0, [10], [20])])  # type: ignore[method-assign]
    remote = make_pp_metadata(
        layer_names=["layer.0", "layer.1"],
        spec_indices=[0, 0],
        spec_block_sizes=[16],
        tp_base_addrs={0: [[5000], [6000]]},
    )

    buckets, request_ids = thread._build_transfer_block_buckets(
        remote,
        [(0, 0), (1, 1)],
        {(0, 0): [[0]], (1, 1): [[0]]},
        1,
        {"request": make_req_meta()},
        {},
    )

    thread._compute_group_block_ids.assert_called_once()
    assert set(buckets[0][0]) == {(0, 0), (1, 1)}
    assert request_ids == {0: {"request"}}


def test_attention_address_generation_handles_partial_head_overlap() -> None:
    thread = make_thread(
        tp_size=1,
        tp_rank=0,
        kv_cache_specs=[make_full_spec(num_kv_heads=4)],
        block_shapes=[[(4, 16, 8)]],
        block_lens=[[128]],
        block_strides=[[128]],
        kv_caches_base_addr=[[1000]],
    )
    remote = make_pp_metadata(
        block_shapes=[[(2, 16, 8)]],
        block_lens=[[64]],
        block_strides=[[64]],
        tp_base_addrs={0: [[5000]], 1: [[6000]]},
    )
    src: list[int] = []
    dst: list[int] = []
    lengths: list[int] = []

    thread._append_spec_transfer_addresses(
        0,
        remote_tp_rank=1,
        remote_tp_size=2,
        remote_dcp_size=1,
        transfer_entries_by_layer={(0, 0): [("request", [1], [2])]},
        remote_metadata=remote,
        src_list=src,
        dst_list=dst,
        length_list=lengths,
    )

    assert src == [1000 + 128 + 64]
    assert dst == [6000 + 2 * 64]
    assert lengths == [64]


def test_mamba_equal_tp_address_generation_transfers_each_cache() -> None:
    thread = make_thread(
        kv_caches_base_addr=[[1000, 2000]],
        block_shapes=[[(3, 16), (2, 4, 4)]],
        block_strides=[[128, 256]],
        block_lens=[[96, 128]],
    )
    remote = make_pp_metadata(
        block_shapes=[[(3, 16), (2, 4, 4)]],
        block_strides=[[128, 256]],
        block_lens=[[96, 128]],
        tp_base_addrs={0: [[5000, 6000]]},
    )
    src: list[int] = []
    dst: list[int] = []
    lengths: list[int] = []

    thread._append_mamba_transfer_addresses(
        make_mamba_spec(),
        0,
        1,
        {(0, 0): [("request", [1], [2])]},
        remote,
        src,
        dst,
        lengths,
    )

    assert src == [1128, 2256]
    assert dst == [5256, 6512]
    assert lengths == [96, 128]


def test_execute_bucket_calls_mooncake_and_raises_on_negative_return() -> None:
    thread = make_thread(engine=MagicMock())
    thread._append_spec_transfer_addresses = MagicMock(  # type: ignore[method-assign]
        side_effect=lambda **kwargs: (
            kwargs["src_list"].append(1000),
            kwargs["dst_list"].append(2000),
            kwargs["length_list"].append(128),
        )
    )
    remote = make_pp_metadata()
    entries = {0: {(0, 0): [("request", [1], [2])]}}

    thread.engine.batch_transfer_sync_read.return_value = 0
    thread._execute_tp_transfer_bucket(0, 0, 1, 1, remote, entries)
    thread.engine.batch_transfer_sync_read.assert_called_once_with("10.0.0.1:9000", [1000], [2000], [128])

    thread.engine.batch_transfer_sync_read.return_value = -1
    with pytest.raises(RuntimeError, match="transfer failed"):
        thread._execute_tp_transfer_bucket(0, 0, 1, 1, remote, entries)


@pytest.mark.parametrize(("can_report", "expected_failed"), [(True, {"request-b"}), (False, set())])
def test_handle_requests_attributes_failed_tp_to_affected_requests(
    can_report: bool,
    expected_failed: set[str],
) -> None:
    thread = make_thread(can_report_invalid_block_ids=can_report)
    remote_pp = make_pp_metadata(tp_base_addrs={0: [[5000]], 1: [[6000]]})
    remote_groups = make_metadata_groups(tp_size=2, pp_metadata=remote_pp)
    thread._get_remote_metadata = MagicMock(return_value=remote_groups)  # type: ignore[method-assign]
    thread.remote_tp_rank_groups = {"engine-p": {0: {(0, 0): [[0], [1]]}}}
    thread.remote_layer_index_pairs = {"engine-p": {0: [(0, 0)]}}
    thread._build_transfer_block_buckets = MagicMock(  # type: ignore[method-assign]
        return_value=(
            {
                0: {0: {(0, 0): [("request-a", [1], [2])]}},
                1: {0: {(0, 0): [("request-b", [3], [4])]}},
            },
            {0: {"request-a"}, 1: {"request-b"}},
        )
    )

    def submit(_func: object, _pp: int, tp_rank: int, *_args: object) -> Future[None]:
        future: Future[None] = Future()
        if tp_rank == 1:
            future.set_exception(RuntimeError("remote TP failed"))
        else:
            future.set_result(None)
        return future

    thread.executor = MagicMock()
    thread.executor.submit.side_effect = submit
    requests = {"request-a": make_req_meta(), "request-b": make_req_meta()}

    assert thread._handle_requests("engine-p", "10.0.0.1", 6000, requests) == expected_failed


def test_connector_worker_groups_start_load_by_remote_engine() -> None:
    worker = MooncakePullConnectorWorker.__new__(MooncakePullConnectorWorker)
    worker.kv_transfer_config = SimpleNamespace(is_kv_consumer=True)
    worker._recving_thread = MagicMock()
    metadata = MooncakeConnectorMetadata()
    metadata.requests = {
        "a": make_req_meta(engine_id="engine-1"),
        "b": make_req_meta(engine_id="engine-2"),
        "c": make_req_meta(engine_id="engine-1"),
    }

    worker.start_load_kv(metadata)

    calls = worker._recving_thread.add_requests.call_args_list
    assert len(calls) == 2
    grouped = {call.args[0]: set(call.args[1]) for call in calls}
    assert grouped == {"engine-1": {"a", "c"}, "engine-2": {"b"}}


def test_connector_worker_exposes_finished_and_invalid_blocks() -> None:
    worker = MooncakePullConnectorWorker.__new__(MooncakePullConnectorWorker)
    worker._recving_thread = MagicMock()
    worker._recving_thread.get_and_clear_finished_requests.return_value = {"request"}
    worker._recving_thread.get_and_clear_invalid_block_ids.return_value = {10, 11}

    assert worker.get_finished() == (set(), {"request"})
    assert worker.get_block_ids_with_load_errors() == {10, 11}
