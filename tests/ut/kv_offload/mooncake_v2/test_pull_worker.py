# SPDX-License-Identifier: Apache-2.0

import queue
import threading
from concurrent.futures import Future
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import msgspec
import pytest
import torch
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import MLAAttentionSpec

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.metadata import (
    MooncakeConnectorMetadata,
    MooncakeTransferMetadataGroups,
    ReqMeta,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_worker import (
    MooncakePullConnectorWorker,
    MooncakePullRecvingThread,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.utils import SizedDict

from .helpers import (
    make_full_spec,
    make_mamba_spec,
    make_metadata_groups,
    make_pp_metadata,
    make_sfa_indexer_spec,
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
    thread.layer_block_sizes = [16]
    thread.kv_cache_specs = [make_full_spec()]
    thread.local_metadata = make_transfer_metadata()
    thread.kv_caches_base_addr = [[1000]]
    thread.block_strides = [[128]]
    thread.block_lens = [[128]]
    thread.block_shapes = [[(1, 16, 4)]]
    thread.block_size_scales = [[1]]
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
        use_kv_pp=False,
        metadata_by_pp_rank={0: pp0, 1: pp1},
    )

    rank_groups, pairs = thread._build_remote_transfer_layout(groups)  # type: ignore[arg-type]

    assert pairs == {0: [(0, 0)], 1: [(1, 0)]}
    assert rank_groups == {0: {(0, 0): [[0]]}, 1: {(1, 0): [[0]]}}
    assert thread._get_layer_remote_tp_rank_groups.call_count == 1


def test_build_remote_layout_filters_kv_parallel_owners_and_keeps_mtp_replicas() -> None:
    thread = make_thread(
        layer_names=["layer.0", "mtp.layer"],
        spec_indices=[0, 0],
        kv_cache_specs=[make_full_spec()],
    )
    thread._get_layer_remote_tp_rank_groups = MagicMock(return_value=[[0, 1]])  # type: ignore[method-assign]
    pp_metadata = make_pp_metadata(
        layer_names=["layer.0", "mtp.layer"],
        tp_base_addrs={
            0: [[5000], [6000]],
            1: [[], [7000]],
        },
        tp_layer_indices={0: [0, 1], 1: [1]},
    )
    groups = make_metadata_groups(
        tp_size=2,
        use_kv_pp=True,
        pp_metadata=pp_metadata,
    )

    rank_groups, pairs = thread._build_remote_transfer_layout(groups)

    assert pairs == {0: [(0, 0), (1, 1)]}
    assert rank_groups == {
        0: {
            (0, 0): [[0]],
            (1, 1): [[0, 1]],
        }
    }
    assert thread._get_layer_remote_tp_rank_groups.call_count == 1


def test_build_remote_layout_allows_local_dcp_with_remote_kv_parallel() -> None:
    thread = make_thread(tp_size=2, dcp_size=2)
    thread._get_layer_remote_tp_rank_groups = MagicMock(return_value=[[0, 1]])  # type: ignore[method-assign]
    pp_metadata = make_pp_metadata(
        tp_base_addrs={0: [[5000]], 1: [[]]},
        tp_layer_indices={0: [0], 1: []},
    )
    groups = make_metadata_groups(tp_size=2, use_kv_pp=True, pp_metadata=pp_metadata)

    rank_groups, pairs = thread._build_remote_transfer_layout(groups)

    assert pairs == {0: [(0, 0)]}
    assert rank_groups == {0: {(0, 0): [[0]]}}


def test_build_remote_layout_rejects_remote_kv_parallel_with_remote_dcp() -> None:
    thread = make_thread()
    groups = replace(make_metadata_groups(use_kv_pp=True), dcp_size=2)

    with pytest.raises(ValueError, match="producer cannot enable KV parallel and DCP together"):
        thread._build_remote_transfer_layout(groups)


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


@pytest.mark.parametrize(
    ("remote_block_size", "remote_scale", "expected_remote_blocks"),
    [
        (32, 2, [42, 43, 44, 45]),
        (64, 4, [82, 83, 84, 85]),
    ],
)
def test_compute_sfa_indexer_blocks_uses_virtual_block_size_for_prefix(
    remote_block_size: int,
    remote_scale: int,
    expected_remote_blocks: list[int],
) -> None:
    thread = make_thread(dcp_size=2)

    result = thread._compute_group_block_ids(
        request_id="request",
        remote_tp_rank_groups=[[0]],
        remote_dcp_size=remote_scale,
        spec_index=0,
        local_block_size=32,
        remote_block_size=remote_block_size,
        local_group_block_ids=[10, 11],
        local_full_group_block_ids=[10, 11],
        remote_group_block_ids=[20, 21, 22],
        local_num_prompt_tokens=96,
        remote_num_prompt_tokens=96,
        num_computed_tokens=32,
        local_block_size_scale=2,
        remote_block_size_scale=remote_scale,
        spec=make_sfa_indexer_spec(replication_size=2),
        selection_index=0,
    )

    assert result == [(0, [20, 21, 22, 23], expected_remote_blocks)]


def test_transfer_bucket_accepts_sfa_indexer_virtual_block_sizes() -> None:
    spec = make_sfa_indexer_spec(replication_size=2)
    thread = make_thread(
        dcp_size=2,
        kv_cache_specs=[spec],
        layer_block_sizes=[32],
        block_size_scales=[[2]],
    )
    remote = make_pp_metadata(
        layer_block_sizes=[64],
        block_size_scales=[[4]],
    )

    buckets, request_ids = thread._build_transfer_block_buckets(
        remote_metadata=remote,
        layer_pairs=[(0, 0)],
        tp_rank_groups_by_layer={(0, 0): [[0]]},
        remote_dcp_size=4,
        requests={"request": make_req_meta()},
        transfer_block_ids_by_spec={},
    )

    assert buckets[0][0][(0, 0)] == [("request", [20, 21, 22, 23], [80, 81, 82, 83])]
    assert request_ids == {0: {"request"}}


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
        layer_block_sizes=[16, 16],
        block_shapes=[[(1, 16, 4)], [(1, 16, 4)]],
        block_strides=[[128], [128]],
        block_lens=[[128], [128]],
        block_size_scales=[[1], [1]],
        kv_caches_base_addr=[[1000], [2000]],
    )
    thread.local_metadata = make_transfer_metadata(
        layer_names=["layer.0", "layer.1"],
        layer_block_sizes=[16, 16],
    )
    thread._compute_group_block_ids = MagicMock(return_value=[(0, [10], [20])])  # type: ignore[method-assign]
    remote = make_pp_metadata(
        layer_names=["layer.0", "layer.1"],
        layer_block_sizes=[16, 16],
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


def test_transfer_bucket_separates_same_spec_layers_with_different_tp_owners() -> None:
    thread = make_thread(
        layer_names=["layer.0", "layer.1"],
        group_indices=[0, 0],
        spec_indices=[0, 0],
        layer_block_sizes=[16, 16],
        block_shapes=[[(1, 16, 4)], [(1, 16, 4)]],
        block_strides=[[128], [128]],
        block_lens=[[128], [128]],
        block_size_scales=[[1], [1]],
        kv_caches_base_addr=[[1000], [2000]],
    )
    thread.local_metadata = make_transfer_metadata(
        layer_names=["layer.0", "layer.1"],
        layer_block_sizes=[16, 16],
    )

    def compute_block_ids(
        request_id: str,
        remote_tp_rank_groups: list[list[int]],
        *args: object,
    ) -> list[tuple[int, list[int], list[int]]]:
        del request_id, args
        return [(remote_tp_rank_groups[0][0], [10], [20])]

    thread._compute_group_block_ids = MagicMock(side_effect=compute_block_ids)  # type: ignore[method-assign]
    remote = make_pp_metadata(
        layer_names=["layer.0", "layer.1"],
        layer_block_sizes=[16, 16],
        tp_base_addrs={0: [[5000], []], 1: [[], [6000]]},
        tp_layer_indices={0: [0], 1: [1]},
    )

    buckets, request_ids = thread._build_transfer_block_buckets(
        remote,
        [(0, 0), (1, 1)],
        {(0, 0): [[0]], (1, 1): [[1]]},
        1,
        {"request": make_req_meta()},
        {},
    )

    assert set(buckets) == {0, 1}
    assert request_ids == {0: {"request"}, 1: {"request"}}
    assert thread._compute_group_block_ids.call_count == 2


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


class _StopWorkerLoop(BaseException):
    """Stop the worker's unbounded queue loop after one test iteration."""


def test_get_remote_metadata_fetches_once_and_populates_layout_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    thread = make_thread()
    thread.encoder = msgspec.msgpack.Encoder()
    thread.decoder = msgspec.msgpack.Decoder(MooncakeTransferMetadataGroups)
    thread.remote_metadata = SizedDict()
    thread.remote_tp_rank_groups = SizedDict()
    thread.remote_layer_index_pairs = SizedDict()
    expected = make_metadata_groups()
    socket = MagicMock()
    context = MagicMock()
    context.__enter__.return_value = socket
    send = MagicMock()
    recv = MagicMock(return_value=msgspec.msgpack.encode(expected))
    thread._build_remote_transfer_layout = MagicMock(  # type: ignore[method-assign]
        return_value=({0: {(0, 0): [[0]]}}, {0: [(0, 0)]})
    )
    monkeypatch.setattr(
        "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_worker.zmq_ctx",
        MagicMock(return_value=context),
    )
    monkeypatch.setattr(
        "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_worker.ensure_zmq_send",
        send,
    )
    monkeypatch.setattr(
        "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_worker.ensure_zmq_recv",
        recv,
    )

    first = thread._get_remote_metadata("engine-p", "10.0.0.1", 6000)
    second = thread._get_remote_metadata("engine-p", "10.0.0.1", 6000)

    assert first is second
    send.assert_called_once()
    recv.assert_called_once()
    thread._build_remote_transfer_layout.assert_called_once_with(expected)
    assert thread.remote_tp_rank_groups["engine-p"] == {0: {(0, 0): [[0]]}}
    assert thread.remote_layer_index_pairs["engine-p"] == {0: [(0, 0)]}


def test_get_remote_metadata_rejects_empty_response(monkeypatch: pytest.MonkeyPatch) -> None:
    thread = make_thread()
    thread.encoder = msgspec.msgpack.Encoder()
    thread.decoder = msgspec.msgpack.Decoder(MooncakeTransferMetadataGroups)
    thread.remote_metadata = SizedDict()
    socket = MagicMock()
    context = MagicMock()
    context.__enter__.return_value = socket
    monkeypatch.setattr(
        "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_worker.zmq_ctx",
        MagicMock(return_value=context),
    )
    monkeypatch.setattr(
        "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_worker.ensure_zmq_send",
        MagicMock(),
    )
    monkeypatch.setattr(
        "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_worker.ensure_zmq_recv",
        MagicMock(return_value=b""),
    )

    with pytest.raises(RuntimeError, match="returned no transfer metadata"):
        thread._get_remote_metadata("engine-p", "10.0.0.1", 6000)


@pytest.mark.parametrize(
    ("metadata", "expected_error"),
    [
        (replace(make_metadata_groups(), engine_id="wrong"), "engine ID mismatch"),
        (replace(make_metadata_groups(), pcp_size=2), "requires remote pcp_size=1"),
        (replace(make_metadata_groups(), metadata_by_pp_rank={}), "no PP metadata"),
        (
            replace(
                make_metadata_groups(),
                metadata_by_pp_rank={0: replace(make_pp_metadata(), metadata_by_tp_rank={})},
            ),
            "no TP metadata",
        ),
        (
            make_metadata_groups(
                pp_metadata=make_pp_metadata(tp_base_addrs={2: [[5000]]}),
            ),
            "invalid TP ranks",
        ),
    ],
)
def test_validate_remote_metadata_rejects_invalid_topology(
    metadata: MooncakeTransferMetadataGroups,
    expected_error: str,
) -> None:
    with pytest.raises(ValueError, match=expected_error):
        MooncakePullRecvingThread._validate_remote_metadata(metadata, "engine-p")


def test_worker_run_marks_only_failed_requests_and_finishes_the_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    thread = make_thread(device=0, ready_event=threading.Event())
    requests = {
        "request-a": make_req_meta(local=([10],)),
        "request-b": make_req_meta(local=([20],)),
    }
    request_queue = MagicMock()
    request_queue.get.side_effect = [("engine-p", requests), _StopWorkerLoop()]
    thread.request_queue = request_queue
    thread._handle_requests = MagicMock(return_value={"request-b"})  # type: ignore[method-assign]
    set_device = MagicMock()
    monkeypatch.setattr(torch.npu, "set_device", set_device)

    with pytest.raises(_StopWorkerLoop):
        thread.run()

    set_device.assert_called_once_with(0)
    assert thread.ready_event.is_set()
    assert thread.invalid_block_ids == {20}
    assert thread.get_and_clear_finished_requests() == {"request-a", "request-b"}
    request_queue.task_done.assert_called_once_with()


def test_whole_block_mla_address_generation_uses_independent_stride() -> None:
    spec = MLAAttentionSpec(block_size=16, num_kv_heads=1, head_size=64, dtype=torch.float16)
    thread = make_thread(
        kv_cache_specs=[spec],
        kv_caches_base_addr=[[1000]],
        block_strides=[[256]],
        block_lens=[[128]],
    )
    remote = make_pp_metadata(
        block_strides=[[512]],
        block_lens=[[128]],
        tp_base_addrs={0: [[5000]]},
    )
    src: list[int] = []
    dst: list[int] = []
    lengths: list[int] = []

    thread._append_spec_transfer_addresses(
        0,
        0,
        1,
        1,
        {(0, 0): [("request", [1, 2], [3, 4])]},
        remote,
        src,
        dst,
        lengths,
    )

    assert src == [1256, 1512]
    assert dst == [6536, 7048]
    assert lengths == [128, 128]


def test_mamba_unequal_tp_slices_conv_projections_and_state() -> None:
    spec = SimpleNamespace(
        dtypes=(torch.float16, torch.float16),
        mamba_type=MambaAttentionBackendEnum.MAMBA2,
    )
    thread = make_thread(
        tp_size=1,
        tp_rank=0,
        kv_caches_base_addr=[[1000, 2000]],
        block_shapes=[[(3, 16), (2, 4, 4)]],
        block_strides=[[96, 64]],
        block_lens=[[96, 64]],
    )
    remote = make_pp_metadata(
        block_shapes=[[(3, 8), (1, 4, 4)]],
        block_strides=[[48, 32]],
        block_lens=[[48, 32]],
        tp_base_addrs={0: [[5000, 6000]], 1: [[7000, 8000]]},
    )
    src: list[int] = []
    dst: list[int] = []
    lengths: list[int] = []

    thread._append_mamba_transfer_addresses(
        spec,  # type: ignore[arg-type]
        remote_tp_rank=1,
        remote_tp_size=2,
        transfer_entries_by_layer={(0, 0): [("request", [1], [2])]},
        remote_metadata=remote,
        src_list=src,
        dst_list=dst,
        length_list=lengths,
    )

    assert src == [1104, 1136, 1168, 1116, 1148, 1180, 1124, 1156, 1188, 2096]
    assert dst == [7096, 7112, 7128, 7104, 7120, 7136, 7108, 7124, 7140, 8064]
    assert lengths == [8, 8, 8, 4, 4, 4, 4, 4, 4, 32]


@pytest.mark.parametrize(
    (
        "local_dcp_size",
        "local_dcp_rank",
        "remote_dcp_size",
        "candidate_ranks",
        "remote_blocks",
        "expected",
    ),
    [
        (1, 0, 4, [0, 1, 2, 3], [20], [(0, [10], [20]), (1, [11], [20])]),
        (4, 2, 1, [0], list(range(20, 28)), [(0, [10, 11], [22, 26])]),
        (4, 2, 4, [0, 1, 2, 3], [20, 21], [(2, [10, 11], [20, 21])]),
    ],
)
def test_compute_same_block_size_dcp_matrix(
    local_dcp_size: int,
    local_dcp_rank: int,
    remote_dcp_size: int,
    candidate_ranks: list[int],
    remote_blocks: list[int],
    expected: list[tuple[int, list[int], list[int]]],
) -> None:
    thread = make_thread(
        tp_size=max(local_dcp_size, 1),
        tp_rank=local_dcp_rank,
        dcp_size=local_dcp_size,
        dcp_rank=local_dcp_rank,
    )

    result = thread._compute_group_block_ids(
        "request",
        [candidate_ranks],
        remote_dcp_size,
        0,
        16,
        16,
        [10, 11],
        [10, 11],
        remote_blocks,
        128,
        128,
        0,
        1,
        1,
        make_full_spec(),
        0,
    )

    assert result == expected


def test_compute_dcp_stops_when_remote_prompt_has_fewer_blocks() -> None:
    thread = make_thread(tp_size=4, tp_rank=2, dcp_size=4, dcp_rank=2)

    result = thread._compute_group_block_ids(
        "request",
        [[0]],
        1,
        0,
        16,
        16,
        [10, 11],
        [10, 11],
        [20, 21, 22],
        128,
        47,
        0,
        1,
        1,
        make_full_spec(),
        0,
    )

    assert result == [(0, [10], [22])]


def test_get_remote_metadata_propagates_network_error_without_caching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    thread = make_thread()
    thread.encoder = msgspec.msgpack.Encoder()
    thread.decoder = msgspec.msgpack.Decoder(MooncakeTransferMetadataGroups)
    thread.remote_metadata = SizedDict()
    context = MagicMock()
    context.__enter__.return_value = MagicMock()
    monkeypatch.setattr(
        "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_worker.zmq_ctx",
        MagicMock(return_value=context),
    )
    monkeypatch.setattr(
        "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_worker.ensure_zmq_send",
        MagicMock(),
    )
    monkeypatch.setattr(
        "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_worker.ensure_zmq_recv",
        MagicMock(side_effect=RuntimeError("timed out")),
    )

    with pytest.raises(RuntimeError, match="timed out"):
        thread._get_remote_metadata("engine-p", "10.0.0.1", 6000)

    assert thread.remote_metadata.get("engine-p") is None


@pytest.mark.parametrize(("is_consumer", "expected_thread_count"), [(True, 1), (False, 0)])
def test_register_kv_caches_starts_receiving_thread_only_for_consumer(
    monkeypatch: pytest.MonkeyPatch,
    is_consumer: bool,
    expected_thread_count: int,
) -> None:
    worker = MooncakePullConnectorWorker.__new__(MooncakePullConnectorWorker)
    worker.kv_transfer_config = SimpleNamespace(is_kv_consumer=is_consumer)
    worker.engine = MagicMock()
    worker.vllm_config = MagicMock()
    worker.kv_cache_config = MagicMock()
    worker.kv_cache_specs = [make_full_spec()]
    worker.layer_name_to_group_index = {"model.layers.0.self_attn": 0}
    worker.layer_name_to_spec_index = {"model.layers.0.self_attn": 0}
    worker.tp_rank = worker.pp_rank = worker.dp_rank = worker.pcp_rank = worker.dcp_rank = 0
    worker.tp_size = worker.pp_size = worker.dp_size = worker.pcp_size = worker.dcp_size = 1
    worker._recving_thread = None

    def fake_register(instance: MooncakePullConnectorWorker, _kv_caches: object) -> None:
        instance.xfer_handshake_metadata = make_transfer_metadata()

    created_threads: list[MagicMock] = []

    def fake_thread(**kwargs: object) -> MagicMock:
        ready_event = kwargs["ready_event"]
        assert isinstance(ready_event, threading.Event)
        thread = MagicMock()
        thread.start.side_effect = ready_event.set
        thread.is_alive.return_value = True
        created_threads.append(thread)
        return thread

    monkeypatch.setattr(
        "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.base_worker.MooncakeBaseConnectorWorker.register_kv_caches",
        fake_register,
    )
    monkeypatch.setattr(
        "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_worker.MooncakePullRecvingThread",
        fake_thread,
    )
    monkeypatch.setattr(torch.npu, "current_device", MagicMock(return_value=0))

    worker.register_kv_caches({})

    assert len(created_threads) == expected_thread_count
    if is_consumer:
        created_threads[0].start.assert_called_once_with()
        assert worker._recving_thread is created_threads[0]
    else:
        assert worker._recving_thread is None


def test_mamba_unequal_tp_slices_into_wider_remote_cache() -> None:
    spec = SimpleNamespace(
        dtypes=(torch.float16, torch.float16),
        mamba_type=MambaAttentionBackendEnum.MAMBA2,
    )
    thread = make_thread(
        tp_size=2,
        tp_rank=1,
        kv_caches_base_addr=[[1000, 2000]],
        block_shapes=[[(3, 8), (1, 4, 4)]],
        block_strides=[[48, 32]],
        block_lens=[[48, 32]],
    )
    remote = make_pp_metadata(
        block_shapes=[[(3, 16), (2, 4, 4)]],
        block_strides=[[96, 64]],
        block_lens=[[96, 64]],
        tp_base_addrs={0: [[5000, 6000]]},
    )
    src: list[int] = []
    dst: list[int] = []
    lengths: list[int] = []

    thread._append_mamba_transfer_addresses(
        spec,  # type: ignore[arg-type]
        remote_tp_rank=0,
        remote_tp_size=1,
        transfer_entries_by_layer={(0, 0): [("request", [1], [2])]},
        remote_metadata=remote,
        src_list=src,
        dst_list=dst,
        length_list=lengths,
    )

    assert src == [1048, 1064, 1080, 1056, 1072, 1088, 1060, 1076, 1092, 2032]
    assert dst == [5200, 5232, 5264, 5212, 5244, 5276, 5220, 5252, 5284, 6160]
    assert lengths == [8, 8, 8, 4, 4, 4, 4, 4, 4, 32]


@pytest.mark.parametrize(
    (
        "local_tp_size",
        "remote_tp_size",
        "local_heads",
        "remote_heads",
        "local_dcp_size",
        "remote_dcp_size",
        "fixed_heads",
        "expected_heads",
    ),
    [
        (8, 4, 1, 1, 1, 1, None, 4),
        (4, 8, 2, 1, 1, 1, None, 8),
        (16, 8, 1, 1, 4, 2, None, 4),
        (8, 16, 1, 1, 2, 4, None, 4),
        (16, 8, 1, 1, 4, 2, 1, 1),
    ],
)
def test_infer_total_kv_heads_across_tp_and_dcp_strategies(
    local_tp_size: int,
    remote_tp_size: int,
    local_heads: int,
    remote_heads: int,
    local_dcp_size: int,
    remote_dcp_size: int,
    fixed_heads: int | None,
    expected_heads: int,
) -> None:
    thread = make_thread(tp_size=local_tp_size)

    assert (
        thread._infer_total_num_kv_heads(
            local_num_kv_heads=local_heads,
            remote_num_kv_heads=remote_heads,
            remote_tp_size=remote_tp_size,
            local_dcp_size=local_dcp_size,
            remote_dcp_size=remote_dcp_size,
            fixed_total_num_kv_heads=fixed_heads,
        )
        == expected_heads
    )


def test_infer_total_kv_heads_rejects_inconsistent_topology() -> None:
    thread = make_thread(tp_size=4)

    with pytest.raises(ValueError, match="inconsistent total KV head counts"):
        thread._infer_total_num_kv_heads(
            local_num_kv_heads=2,
            remote_num_kv_heads=2,
            remote_tp_size=8,
            local_dcp_size=1,
            remote_dcp_size=1,
            fixed_total_num_kv_heads=None,
        )


@pytest.mark.parametrize(
    (
        "local_tp_size",
        "local_tp_rank",
        "local_dcp_size",
        "remote_tp_size",
        "remote_dcp_size",
        "total_heads",
        "expected_groups",
    ),
    [
        (4, 2, 1, 8, 1, 4, [[4, 5]]),
        (2, 0, 1, 4, 1, 8, [[0], [1]]),
        (16, 10, 4, 8, 2, 4, [[4, 5]]),
        (8, 5, 2, 16, 4, 4, [[8, 9, 10, 11]]),
        (8, 6, 4, 16, 4, 4, [[8, 9, 10, 11], [12, 13, 14, 15]]),
        (16, 13, 4, 8, 4, 4, [[4, 5, 6, 7]]),
        (4, 3, 4, 8, 8, 1, [list(range(8))]),
        (8, 6, 8, 4, 4, 1, [list(range(4))]),
    ],
)
def test_attention_remote_tp_rank_groups_cover_tp_and_dcp_matrix(
    local_tp_size: int,
    local_tp_rank: int,
    local_dcp_size: int,
    remote_tp_size: int,
    remote_dcp_size: int,
    total_heads: int,
    expected_groups: list[list[int]],
) -> None:
    thread = make_thread(tp_size=local_tp_size, tp_rank=local_tp_rank, dcp_size=local_dcp_size)

    assert (
        thread._get_attention_remote_tp_rank_groups(
            remote_tp_size=remote_tp_size,
            local_dcp_size=local_dcp_size,
            remote_dcp_size=remote_dcp_size,
            total_num_kv_heads=total_heads,
        )
        == expected_groups
    )


def test_layer_remote_tp_rank_groups_apply_spec_specific_dcp_rules() -> None:
    full_thread = make_thread(
        tp_size=16,
        tp_rank=10,
        dcp_size=4,
        block_shapes=[[(1, 16, 4)]],
    )
    full_remote = make_pp_metadata(block_shapes=[[(1, 16, 4)]])
    assert full_thread._get_layer_remote_tp_rank_groups(
        0, 0, make_full_spec(), full_remote, remote_tp_size=8, remote_dcp_size=2
    ) == [[4, 5]]

    mla_thread = make_thread(tp_size=4, tp_rank=2, dcp_size=4)
    mla_spec = MLAAttentionSpec(block_size=16, num_kv_heads=1, head_size=64, dtype=torch.float16)
    assert mla_thread._get_layer_remote_tp_rank_groups(
        0, 0, mla_spec, make_pp_metadata(), remote_tp_size=8, remote_dcp_size=8
    ) == [list(range(8))]

    swa_thread = make_thread(
        tp_size=4,
        tp_rank=2,
        dcp_size=4,
        block_shapes=[[(1, 16, 4)]],
    )
    swa_remote = make_pp_metadata(block_shapes=[[(1, 16, 4)]])
    assert swa_thread._get_layer_remote_tp_rank_groups(
        0, 0, make_sliding_spec(), swa_remote, remote_tp_size=4, remote_dcp_size=4
    ) == [[2]]


@pytest.mark.parametrize(
    (
        "local_dcp_size",
        "local_dcp_rank",
        "remote_dcp_size",
        "candidate_ranks",
        "local_blocks",
        "remote_blocks",
        "expected",
    ),
    [
        (
            1,
            0,
            4,
            [0, 1, 2, 3],
            [10, 11, 12, 13],
            [20],
            [(0, [10], [20]), (1, [11], [20]), (2, [12], [20]), (3, [13], [20])],
        ),
        (4, 3, 2, [4, 5], [10, 11], [20, 21, 22, 23], [(5, [10, 11], [21, 23])]),
        (
            2,
            1,
            4,
            [4, 5, 6, 7],
            [10, 11, 12, 13],
            [20, 21],
            [(5, [10, 12], [20, 21]), (7, [11, 13], [20, 21])],
        ),
        (4, 2, 4, [4, 5, 6, 7], [10, 11], [20, 21], [(6, [10, 11], [20, 21])]),
    ],
)
def test_compute_block_ids_across_equal_block_size_dcp_matrix(
    local_dcp_size: int,
    local_dcp_rank: int,
    remote_dcp_size: int,
    candidate_ranks: list[int],
    local_blocks: list[int],
    remote_blocks: list[int],
    expected: list[tuple[int, list[int], list[int]]],
) -> None:
    thread = make_thread(
        tp_size=max(local_dcp_size, 1),
        tp_rank=local_dcp_rank,
        dcp_size=local_dcp_size,
        dcp_rank=local_dcp_rank,
    )

    result = thread._compute_group_block_ids(
        request_id="request",
        remote_tp_rank_groups=[candidate_ranks],
        remote_dcp_size=remote_dcp_size,
        spec_index=0,
        local_block_size=16,
        remote_block_size=16,
        local_group_block_ids=local_blocks,
        local_full_group_block_ids=local_blocks,
        remote_group_block_ids=remote_blocks,
        local_num_prompt_tokens=256,
        remote_num_prompt_tokens=256,
        num_computed_tokens=0,
        local_block_size_scale=1,
        remote_block_size_scale=1,
        spec=make_full_spec(),
        selection_index=0,
    )

    assert result == expected


@pytest.mark.parametrize(
    (
        "local_dcp_size",
        "local_dcp_rank",
        "remote_dcp_size",
        "candidate_ranks",
        "local_block_size",
        "remote_block_size",
        "local_scale",
        "remote_scale",
        "local_blocks",
        "remote_blocks",
        "prompt_tokens",
        "expected",
    ),
    [
        (
            2,
            0,
            4,
            [0, 1, 2, 3],
            16,
            32,
            1,
            2,
            [10, 11, 12, 13],
            [20],
            128,
            [(0, [10], [40]), (1, [11], [40]), (2, [12], [40]), (3, [13], [40])],
        ),
        (
            4,
            1,
            2,
            [4, 5],
            32,
            16,
            2,
            1,
            [10, 11],
            [30, 31, 32, 33, 34, 35],
            256,
            [(4, [20, 22], [31, 35]), (5, [21, 23], [31, 35])],
        ),
    ],
)
def test_compute_block_ids_across_unequal_block_size_dcp_matrix(
    local_dcp_size: int,
    local_dcp_rank: int,
    remote_dcp_size: int,
    candidate_ranks: list[int],
    local_block_size: int,
    remote_block_size: int,
    local_scale: int,
    remote_scale: int,
    local_blocks: list[int],
    remote_blocks: list[int],
    prompt_tokens: int,
    expected: list[tuple[int, list[int], list[int]]],
) -> None:
    thread = make_thread(
        tp_size=max(local_dcp_size, 1),
        tp_rank=local_dcp_rank,
        dcp_size=local_dcp_size,
        dcp_rank=local_dcp_rank,
    )

    result = thread._compute_group_block_ids(
        request_id="request",
        remote_tp_rank_groups=[candidate_ranks],
        remote_dcp_size=remote_dcp_size,
        spec_index=0,
        local_block_size=local_block_size,
        remote_block_size=remote_block_size,
        local_group_block_ids=local_blocks,
        local_full_group_block_ids=local_blocks,
        remote_group_block_ids=remote_blocks,
        local_num_prompt_tokens=prompt_tokens,
        remote_num_prompt_tokens=prompt_tokens,
        num_computed_tokens=0,
        local_block_size_scale=local_scale,
        remote_block_size_scale=remote_scale,
        spec=make_full_spec(local_block_size),
        selection_index=0,
    )

    assert result == expected
