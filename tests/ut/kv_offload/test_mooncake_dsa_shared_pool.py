# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.distributed.kv_transfer.kv_p2p import (
    mooncake_connector,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector import (
    KVCacheRecvingThread,
    MooncakeConnectorWorker,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_dsa_metadata import (
    DsaLocalResultKind,
    DsaStepRequest,
    DsaTransferPhase,
    RemoteEndpoint,
    RemoteSource,
)


@pytest.mark.parametrize("main_owner", [True, False])
def test_blockwise_receive_writes_shared_main_only_on_owner(main_owner):
    thread = object.__new__(KVCacheRecvingThread)
    thread.tp_rank = 0
    thread._dsa_main_owner = main_owner
    thread._dsa_indexer_local_layout = [[(0, 100, 8, 8, 1)]]
    thread._dsa_main_local_layout = [[(0, 200, 8, 8, 1)]] if main_owner else [[]]
    thread.remote_metadata_lock = MagicMock()
    thread.kv_caches_base_addr = {"prefill": {5000: [[1000]]}}
    thread.remote_metadata_hosts = {"prefill": {5000: "10.0.0.1"}}
    thread.remote_block_stride_per_addr = {"prefill": {5000: [[8]]}}
    thread.remote_block_size_scale = {"prefill": {5000: [[1]]}}
    thread.remote_block_len_per_addr = {"prefill": {5000: [[8]]}}
    thread.remote_te_port = {"prefill": {5000: 6000}}
    thread._get_remote_metadata = MagicMock()
    thread._send_done_recv_signal = MagicMock()
    thread._log_dsa_transfer_phase_diag = MagicMock()

    calls = []
    thread.engine = SimpleNamespace(
        batch_transfer_sync_read=lambda session, *lists: calls.append(
            (session, lists)
        )
        or 0
    )
    thread._build_dsa_transfer_lists = MagicMock(
        side_effect=[([1], [11], [8]), ([2], [12], [8])]
        if main_owner
        else [([1], [11], [8])]
    )

    endpoint = RemoteEndpoint("10.0.0.1", 5000, "prefill")
    command = DsaStepRequest(
        request_id="req",
        source=RemoteSource("remote-req", (endpoint,), (1,), (2,)),
        main_host_block_ids=(12,),
        indexer_hbm_block_ids=(11,),
    )
    results = []

    thread._execute_dsa_receive(command, endpoint, results.append)

    assert len(calls) == (2 if main_owner else 1)
    assert results[0].kind is DsaLocalResultKind.RECEIVE_COMPLETE
    thread._send_done_recv_signal.assert_called_once()


def _dsa_receive_fixture():
    thread = object.__new__(KVCacheRecvingThread)
    thread.tp_rank = 0
    thread._dsa_main_owner = True
    thread._dsa_indexer_local_layout = [[(0, 100, 8, 8, 1)]]
    thread._dsa_main_local_layout = [[(0, 200, 8, 8, 1)]]
    thread.remote_metadata_lock = MagicMock()
    thread.kv_caches_base_addr = {"prefill": {5000: [[1000]]}}
    thread.remote_metadata_hosts = {"prefill": {5000: "10.0.0.1"}}
    thread.remote_block_stride_per_addr = {"prefill": {5000: [[8]]}}
    thread.remote_block_size_scale = {"prefill": {5000: [[1]]}}
    thread.remote_block_len_per_addr = {"prefill": {5000: [[8]]}}
    thread.remote_te_port = {"prefill": {5000: 6000}}
    thread._get_remote_metadata = MagicMock()
    thread._send_done_recv_signal = MagicMock()
    thread._build_dsa_transfer_lists = MagicMock(
        return_value=([1], [11], [8])
    )
    endpoint = RemoteEndpoint("10.0.0.1", 5000, "prefill")
    command = DsaStepRequest(
        request_id="req",
        source=RemoteSource("remote-req", (endpoint,), (1,), (2,)),
        main_host_block_ids=(12,),
        indexer_hbm_block_ids=(11,),
    )
    return thread, command, endpoint


def test_blockwise_receive_exception_returns_terminal_failure():
    thread, command, endpoint = _dsa_receive_fixture()
    thread.engine = SimpleNamespace(
        batch_transfer_sync_read=MagicMock(
            side_effect=RuntimeError("transfer failed")
        )
    )
    results = []

    thread._execute_dsa_receive(command, endpoint, results.append)

    assert len(results) == 1
    assert results[0].kind is DsaLocalResultKind.TRANSFER_FAILED
    assert results[0].failure_phase is DsaTransferPhase.INDEXER_D2D
    thread._send_done_recv_signal.assert_called_once()


def test_done_signal_exception_does_not_override_receive_success():
    thread, command, endpoint = _dsa_receive_fixture()
    thread.engine = SimpleNamespace(
        batch_transfer_sync_read=MagicMock(return_value=0)
    )
    thread._send_done_recv_signal.side_effect = ValueError(
        "done signal failed"
    )
    results = []

    thread._execute_dsa_receive(command, endpoint, results.append)

    assert len(results) == 1
    assert results[0].kind is DsaLocalResultKind.RECEIVE_COMPLETE
    assert results[0].failure_phase is None


@pytest.mark.parametrize("main_owner", [True, False])
def test_local_layout_uses_shared_pool_only_on_owner(
    main_owner, monkeypatch
):
    worker = object.__new__(MooncakeConnectorWorker)
    worker.num_blocks = 4
    worker.kv_caches_base_addr = [[], []]
    worker.engine = MagicMock()
    host_k = torch.empty((4, 1, 2), dtype=torch.bfloat16)
    host_v = torch.empty((4, 1, 2), dtype=torch.bfloat16)
    pool = SimpleNamespace(
        is_owner=main_owner,
        register=MagicMock(),
    )
    manager = SimpleNamespace(
        offload_layer_names=["model.layers.0.self_attn.attn"],
        get_mooncake_host_pool=lambda: pool,
        get_fused_overlap_cpu_kv_inputs=lambda _name: (
            host_k,
            host_v,
        ),
    )
    monkeypatch.setattr(
        mooncake_connector,
        "get_sparse_kv_offload_manager",
        lambda: manager,
    )
    indexer = torch.empty((8, 1, 2), dtype=torch.int8)
    kv_caches = {"model.layers.0.indexer.k_cache": (indexer,)}
    layer_name_to_idx = {
        "model.layers.0.self_attn.attn": 0,
        "model.layers.0.indexer.k_cache": 1,
    }

    indexer_layout, main_layout, is_owner = (
        worker._build_dsa_local_layouts(
            kv_caches, layer_name_to_idx
        )
    )

    assert len(indexer_layout[1]) == 1
    assert indexer_layout[1][0][0] == 0
    assert is_owner is main_owner
    if main_owner:
        assert [entry[0] for entry in main_layout[0]] == [0, 1]
        pool.register.assert_called_once_with(worker.engine)
    else:
        assert not any(main_layout)
        pool.register.assert_not_called()
