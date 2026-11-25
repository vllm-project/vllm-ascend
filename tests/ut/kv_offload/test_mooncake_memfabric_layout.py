# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("vllm")

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector import (  # noqa: E402
    MooncakeConnectorWorker,
)


def _make_worker(*, sparse_enabled: bool = True) -> MooncakeConnectorWorker:
    worker = MooncakeConnectorWorker.__new__(MooncakeConnectorWorker)
    worker.transfer_backend = "memfabric"
    worker.kv_role = "kv_consumer"
    worker.sparse_kv_offload_enabled = sparse_enabled
    worker.num_blocks = 16
    worker._is_index_cache_layer = lambda _: False
    return worker


def test_sparse_main_layout_uses_broadcast_host_gvas():
    worker = _make_worker()
    manager = SimpleNamespace(
        layer_name_to_offload_id={"model.layers.0.self_attn": 0},
        gvas_k_bases=[0x100000],
        gvas_v_bases=[0x200000],
        cpu_block_lens=[(1024, 512)],
    )

    with patch(
        "vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager.get_sparse_kv_offload_manager",
        return_value=manager,
    ):
        layout = worker._get_sparse_offload_main_layout("model.layers.0.self_attn")

    assert layout == [
        (0x100000, 1024, 1024, (16, 1024), 1),
        (0x200000, 512, 512, (16, 512), 1),
    ]


def test_sparse_main_layout_keeps_indexer_on_device():
    worker = _make_worker()
    worker._is_index_cache_layer = lambda _: True

    assert worker._get_sparse_offload_main_layout("model.layers.0.indexer") is None


def test_sparse_main_layout_is_disabled_for_mooncake_backend():
    worker = _make_worker()
    worker.transfer_backend = "mooncake"

    assert worker._get_sparse_offload_main_layout("model.layers.0.self_attn") is None


def test_sparse_main_layout_rejects_unready_gva_metadata():
    worker = _make_worker()
    manager = SimpleNamespace(
        layer_name_to_offload_id={"model.layers.0.self_attn": 0},
        gvas_k_bases=[0],
        gvas_v_bases=[0x200000],
        cpu_block_lens=[(1024, 512)],
    )

    with (
        patch(
            "vllm_ascend.distributed.kv_transfer.sparse_kv_offload."
            "sparse_kv_offload_manager.get_sparse_kv_offload_manager",
            return_value=manager,
        ),
        pytest.raises(RuntimeError, match="Invalid sparse KV offload Host layout"),
    ):
        worker._get_sparse_offload_main_layout("model.layers.0.self_attn")


@pytest.mark.parametrize(
    ("tp_rank", "expected"),
    [
        (0, [True, False, True, False, True, False, True, False]),
        (1, [False, True, False, True, False, True, False, True]),
    ],
)
def test_sparse_shared_main_routes_each_dcp_shard_to_one_decode_rank(tp_rank, expected):
    worker = _make_worker()
    worker.sparse_shared_main_group_ids = {0}
    worker.tp_size = 2
    worker.tp_rank = tp_rank

    actual = [
        worker._should_transfer_sparse_shared_main(
            shard_idx,
            remote_dcp_size=8,
            prefill_tp_size=8,
            remote_pcp_size=1,
        )
        for shard_idx in range(8)
    ]

    assert actual == expected


def test_sparse_shared_main_routing_keeps_legacy_paths_unchanged():
    worker = _make_worker()
    worker.tp_size = 2
    worker.tp_rank = 1
    worker.sparse_shared_main_group_ids = set()
    assert worker._should_transfer_sparse_shared_main(0, 8, 8, 1)

    worker.sparse_shared_main_group_ids = {0}
    assert worker._should_transfer_sparse_shared_main(0, 1, 1, 1)
    assert worker._should_transfer_sparse_shared_main(0, 8, 16, 1)
    assert worker._should_transfer_sparse_shared_main(0, 8, 8, 2)


def test_sparse_shared_main_syncs_completion_and_failures_across_tp():
    worker = _make_worker()
    worker.tp_size = 2
    worker.tp_rank = 0
    worker.tp_group = SimpleNamespace(cpu_group=object())
    worker.kv_send_thread = None
    worker.kv_recv_thread = MagicMock()
    worker._sync_sparse_shared_main_across_tp = True
    worker._recv_terminal_requests = set()
    worker._synced_invalid_block_ids = set()
    worker.kv_recv_thread.get_and_clear_finished_requests.side_effect = [
        {"request-1"},
        set(),
    ]
    worker.kv_recv_thread.get_and_clear_invalid_block_ids.side_effect = [
        set(),
        set(),
    ]
    remote_statuses = [
        (set(), {9}),
        ({"request-1"}, set()),
    ]

    def fake_all_gather_object(gathered, local_status, group):
        assert group is worker.tp_group.cpu_group
        gathered[:] = [local_status, remote_statuses.pop(0)]

    with patch(
        "torch.distributed.all_gather_object",
        side_effect=fake_all_gather_object,
    ):
        _, done_recving = worker.get_finished()
        assert done_recving == set()
        assert worker.get_block_ids_with_load_errors() == {9}
        assert worker.get_block_ids_with_load_errors() == set()

        _, done_recving = worker.get_finished()
        assert done_recving == {"request-1"}


def test_sparse_shared_main_handles_empty_gathered_statuses():
    worker = _make_worker()
    worker.tp_size = 2
    worker.tp_rank = 0
    worker.tp_group = SimpleNamespace(cpu_group=object())
    worker.kv_send_thread = None
    worker.kv_recv_thread = MagicMock()
    worker._sync_sparse_shared_main_across_tp = True
    worker._recv_terminal_requests = set()
    worker._synced_invalid_block_ids = set()
    worker.kv_recv_thread.get_and_clear_finished_requests.return_value = set()
    worker.kv_recv_thread.get_and_clear_invalid_block_ids.return_value = set()

    def fake_all_gather_object(gathered, local_status, group):
        assert group is worker.tp_group.cpu_group
        gathered[:] = []

    with patch(
        "torch.distributed.all_gather_object",
        side_effect=fake_all_gather_object,
    ):
        _, done_recving = worker.get_finished()

    assert done_recving == set()
    assert worker.get_block_ids_with_load_errors() == set()
