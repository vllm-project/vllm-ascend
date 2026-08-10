"""Regression tests for SFA PD transfer into SparseKVOffloadManager."""

import asyncio
import threading
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")

from vllm.distributed.kv_transfer.kv_connector.factory import (  # noqa: E402
    KVConnectorFactory,
)

from examples.disaggregated_prefill_v1 import (  # noqa: E402
    load_balance_proxy_layerwise_server_example as proxy_example,
)
from examples.disaggregated_prefill_v1.load_balance_proxy_layerwise_server_example import (  # noqa: E402
    get_api_request_ids,
)
from vllm_ascend.distributed.kv_transfer import register_connector  # noqa: E402
from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.connector import (  # noqa: E402
    SfaRemoteD2HConnector,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.protocol import (  # noqa: E402
    BATCH_KV_TRANSFER_PARAMS,
    READ_READY_BATCH,
    LayerMetadata,
    SendTask,
    SfaPDProducerReqMeta,
    infer_sfa_component_group_ids,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.read_thread import (  # noqa: E402
    ConsumerReadState,
    MembPullReadThread,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.scheduler import (  # noqa: E402
    SFAPDRD2HProducerScheduler,
    SFAPDRD2HScheduler,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.send_thread import (  # noqa: E402
    MembPullSendingThread,
    ProducerSendState,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.worker import (  # noqa: E402
    SFAPDRD2HConsumerWorker,
    SFAPDRD2HProducerWorker,
)
from vllm_ascend.distributed.kv_transfer.utils.memfabric_transfer_engine import (  # noqa: E402
    BACKEND_MEMFABRIC,
)


def test_sfa_remote_d2h_connector_is_registered():
    with (
        patch.object(KVConnectorFactory, "_registry", {}),
        patch.object(KVConnectorFactory, "register_connector") as mock_register,
    ):
        register_connector()

    mock_register.assert_any_call(
        "SfaRemoteD2HConnector",
        "vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.connector",
        "SfaRemoteD2HConnector",
    )


def test_infer_separate_main_and_indexer_groups():
    config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(layer_names=["model.layers.0.self_attn.indexer"]),
            SimpleNamespace(layer_names=["model.layers.0.self_attn"]),
        ]
    )

    assert infer_sfa_component_group_ids(config) == (1, 0)


def test_infer_uniform_group_uses_same_block_ids():
    config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(
                layer_names=[
                    "model.layers.0.self_attn",
                    "model.layers.0.self_attn.indexer",
                ]
            )
        ]
    )

    assert infer_sfa_component_group_ids(config) == (0, 0)


def test_completion_prompt_list_registers_each_vllm_child_request_id():
    assert get_api_request_ids(
        "/completions",
        "parent",
        {"prompt": ["first", "second", "third"]},
    ) == ["cmpl-parent-0", "cmpl-parent-1", "cmpl-parent-2"]


def test_token_id_prompt_list_remains_one_vllm_request():
    assert get_api_request_ids(
        "/completions",
        "parent",
        {"prompt": [1, 2, 3]},
    ) == ["cmpl-parent-0"]


def test_batch_metaserver_dispatches_prompt_list_once():
    async def run_test():
        parent_request_id = "parent"
        request_ids = ("cmpl-parent-0", "cmpl-parent-1", "cmpl-parent-2")
        request_record = ({"prompt": ["a", "b", "c"]}, 3, "/completions", parent_request_id)
        state = SimpleNamespace(
            req_data_dict={request_id: request_record for request_id in request_ids},
            metaserver_lock=asyncio.Lock(),
            metaserver_params={parent_request_id: {}},
            metaserver_expected_ids={parent_request_id: request_ids},
            metaserver_dispatch_tasks={},
            metaserver_ready_events={parent_request_id: asyncio.Event()},
        )
        requests = []
        for request_id in request_ids:
            request = MagicMock()
            request.json = AsyncMock(
                return_value={
                    "request_id": request_id,
                    "do_remote_decode": True,
                    "remote_port": 1234,
                }
            )
            requests.append(request)

        dispatch = AsyncMock()
        with (
            patch.object(proxy_example, "proxy_state", state),
            patch.object(proxy_example, "dispatch_prefill_batch", dispatch),
        ):
            responses = await asyncio.gather(*(proxy_example.metaserver(request) for request in requests))

        assert responses == [{"status": "ok"}] * len(request_ids)
        dispatch.assert_awaited_once()
        assert dispatch.await_args is not None
        dispatched_params = dispatch.await_args.args[-1]
        assert set(dispatched_params) == set(request_ids)

    asyncio.run(run_test())


def _make_read_thread() -> MembPullReadThread:
    thread = MembPullReadThread.__new__(MembPullReadThread)
    thread.tp_rank = 0
    thread._state = ConsumerReadState(
        num_blocks=16,
        tp_size=1,
        layer_metadata={},
        main_name_to_idx={},
        cpu_pools=[],
        main_gva_bases=[],
        main_block_lens=[],
        indexer_tensors=[],
        indexer_scale_tensors=[],
        dest_blocks_by_req={"req-0": ([3, 4], [8])},
        get_offload_layer_id=lambda _: 0,
    )
    return thread


def _make_layer(
    k_cpu_ptr: int | None,
    v_cpu_ptr: int | None,
    *,
    has_indexer: bool = True,
) -> dict:
    return {
        "layer_name": "model.layers.0.self_attn",
        "pool_idx": 0,
        "offload_id": 0,
        "p_k_base": 1000,
        "p_v_base": 2000,
        "p_k_len": 10,
        "p_v_len": 20,
        "k_cpu_ptr": k_cpu_ptr,
        "v_cpu_ptr": v_cpu_ptr,
        "indexer": (
            {
                "p_dsa_base": 7000,
                "block_len": 5,
                "d_base": 8000,
                "shape": (16, 1, 1, 5),
            }
            if has_indexer
            else None
        ),
        "scale": None,
    }


def test_read_descriptors_use_independent_main_and_indexer_block_ids():
    thread = _make_read_thread()

    local, peer, lengths, info = thread._build_req_descriptors(
        _make_layer(k_cpu_ptr=3000, v_cpu_ptr=4000),
        "req-0",
        p_main_block_ids=[1, 2],
        p_indexer_block_ids=[7],
        want_info=True,
    )

    assert local == [3030, 4060, 8040]
    assert peer == [1010, 2020, 7035]
    assert lengths == [20, 40, 5]
    assert info is not None
    assert info["n_main"] == 2
    assert info["n_indexer"] == 1


def test_non_tp0_read_descriptors_still_transfer_indexer():
    thread = _make_read_thread()

    local, peer, lengths, info = thread._build_req_descriptors(
        _make_layer(k_cpu_ptr=None, v_cpu_ptr=None),
        "req-0",
        p_main_block_ids=[1, 2],
        p_indexer_block_ids=[7],
        want_info=True,
    )

    assert local == [8040]
    assert peer == [7035]
    assert lengths == [5]
    assert info is not None
    assert info["n_main"] == 0
    assert info["n_indexer"] == 1


def test_non_tp0_resolves_broadcast_main_gva_without_cpu_tensor():
    thread = _make_read_thread()
    layer_name = "model.layers.0.self_attn"
    thread._state.main_name_to_idx = {layer_name: 0}
    thread._state.cpu_pools = [None]
    thread._state.main_gva_bases = [(3000, 4000)]
    thread._state.main_block_lens = [(10, 20)]
    thread._state.indexer_tensors = [None]
    thread._state.indexer_scale_tensors = [None]

    layer = thread._resolve_read_layer(
        layer_name,
        {
            layer_name: {
                "base_addrs": [1000, 2000],
                "block_len": [10, 20],
                "block_size_scale": [1, 1],
                "main_tensor_count": 2,
                "has_indexer": False,
            }
        },
    )

    assert layer is not None
    assert layer["k_cpu_ptr"] == 3000
    assert layer["v_cpu_ptr"] == 4000


@pytest.mark.parametrize(
    ("p_has_scale", "d_has_scale"),
    [(True, False), (False, True)],
)
def test_resolve_read_layer_rejects_asymmetric_indexer_scale_presence(p_has_scale, d_has_scale):
    thread = _make_read_thread()
    layer_name = "model.layers.0.self_attn"
    thread._state.main_name_to_idx = {layer_name: 0}
    thread._state.main_gva_bases = [(3000, 4000)]
    thread._state.main_block_lens = [(10, 20)]
    thread._state.indexer_tensors = [
        SimpleNamespace(
            shape=(16, 1, 1, 5),
            element_size=lambda: 1,
            data_ptr=lambda: 8000,
        )
    ]
    thread._state.indexer_scale_tensors = [
        SimpleNamespace(
            shape=(16, 1),
            element_size=lambda: 1,
            data_ptr=lambda: 9000,
        )
        if d_has_scale
        else None
    ]
    base_addrs = [1000, 2000, 7000]
    block_len = [10, 20, 5]
    block_size_scale = [1, 1, 1]
    if p_has_scale:
        base_addrs.append(9000)
        block_len.append(1)
        block_size_scale.append(1)

    with pytest.raises(
        RuntimeError,
        match=rf"indexer scale presence mismatch.*P={p_has_scale}, D={d_has_scale}",
    ):
        thread._resolve_read_layer(
            layer_name,
            {
                layer_name: {
                    "base_addrs": base_addrs,
                    "block_len": block_len,
                    "block_size_scale": block_size_scale,
                    "main_tensor_count": 2,
                    "has_indexer": True,
                }
            },
        )


@pytest.mark.parametrize(
    (
        "k_cpu_ptr",
        "v_cpu_ptr",
        "has_indexer",
        "expected_local",
        "expected_peer",
        "expected_lengths",
    ),
    [
        (None, None, True, [8040], [7035], [5]),
        (None, None, False, [], [], []),
        (3000, 4000, False, [3030, 4060], [1010, 2020], [20, 40]),
        (3000, 4000, True, [3030, 4060, 8040], [1010, 2020, 7035], [20, 40, 5]),
    ],
    ids=[
        "non-tp0-indexer",
        "non-tp0-main-only-noop",
        "tp0-main-only",
        "tp0-indexer",
    ],
)
def test_read_descriptors_cover_tp_ownership_and_optional_indexer(
    k_cpu_ptr,
    v_cpu_ptr,
    has_indexer,
    expected_local,
    expected_peer,
    expected_lengths,
):
    thread = _make_read_thread()

    local, peer, lengths, _ = thread._build_req_descriptors(
        _make_layer(
            k_cpu_ptr=k_cpu_ptr,
            v_cpu_ptr=v_cpu_ptr,
            has_indexer=has_indexer,
        ),
        "req-0",
        p_main_block_ids=[1, 2],
        p_indexer_block_ids=[7] if has_indexer else [],
        want_info=True,
    )

    assert local == expected_local
    assert peer == expected_peer
    assert lengths == expected_lengths


def test_non_tp0_main_only_layer_acknowledges_without_memfabric_read():
    thread = _make_read_thread()
    thread.engine = MagicMock()
    layer = _make_layer(k_cpu_ptr=None, v_cpu_ptr=None, has_indexer=False)
    thread._resolve_read_layer = MagicMock(return_value=layer)  # type: ignore[method-assign]

    thread._do_read_batch(
        layer["layer_name"],
        [("req-0", [1, 2], [], 0, 0)],
        p_session="p-session",
        p_layer_meta={},
    )

    thread.engine.batch_transfer_sync_read.assert_not_called()


def test_tp_ranks_split_main_blocks_into_disjoint_contiguous_ranges():
    layer = _make_layer(k_cpu_ptr=3000, v_cpu_ptr=4000, has_indexer=False)
    expected = [
        ([3000, 4000], [1000, 2000], [20, 40]),
        ([3020, 4040], [1020, 2040], [20, 40]),
    ]

    for tp_rank, (expected_local, expected_peer, expected_lengths) in enumerate(expected):
        thread = _make_read_thread()
        thread.tp_rank = tp_rank
        thread._state.tp_size = 2
        thread._state.dest_blocks_by_req["req-0"] = ([0, 1, 2, 3], [])

        local, peer, lengths, info = thread._build_req_descriptors(
            layer,
            "req-0",
            p_main_block_ids=[0, 1, 2, 3],
            p_indexer_block_ids=[],
            want_info=True,
        )

        assert local == expected_local
        assert peer == expected_peer
        assert lengths == expected_lengths
        assert info is not None
        assert info["n_main"] == 2


def test_tp_rank_without_blocks_in_small_chunk_acknowledges_without_read():
    thread = _make_read_thread()
    thread.tp_rank = 1
    thread._state.tp_size = 2
    thread.engine = MagicMock()
    layer = _make_layer(k_cpu_ptr=3000, v_cpu_ptr=4000, has_indexer=False)
    thread._resolve_read_layer = MagicMock(return_value=layer)  # type: ignore[method-assign]

    thread._do_read_batch(
        layer["layer_name"],
        [("req-0", [1], [], 0, 0)],
        p_session="p-session",
        p_layer_meta={},
    )

    thread.engine.batch_transfer_sync_read.assert_not_called()


def test_small_chunks_rotate_across_tp_ranks():
    layer = _make_layer(k_cpu_ptr=3000, v_cpu_ptr=4000, has_indexer=False)

    for chunk_start in range(4):
        for tp_rank in range(4):
            thread = _make_read_thread()
            thread.tp_rank = tp_rank
            thread._state.tp_size = 4
            thread._state.dest_blocks_by_req["req-0"] = ([0, 1, 2, 3], [])

            local, _, _, _ = thread._build_req_descriptors(
                layer,
                "req-0",
                p_main_block_ids=[chunk_start],
                p_indexer_block_ids=[],
                want_info=False,
                main_start_block=chunk_start,
            )

            assert bool(local) is (tp_rank == chunk_start)


def _make_consumer_worker_for_completion_test():
    worker = SFAPDRD2HConsumerWorker.__new__(SFAPDRD2HConsumerWorker)
    worker.tp_rank = 0
    worker.tp_size = 2
    worker.request_map = {"req-0": "req-0-internal"}
    worker._dest_blocks_by_req = {"req-0": ([1, 2], [3])}
    worker._cpu_blocks_by_req = {}
    worker._invalid_block_ids = set()
    worker._pending_done = set()
    worker._terminal_ext_ids = set()
    worker._mf_read_thread = MagicMock()
    worker._mf_read_thread.get_and_clear_failed.return_value = set()
    return worker


def test_consumer_completion_waits_for_every_tp_rank():
    worker = _make_consumer_worker_for_completion_test()
    worker._mf_read_thread.get_and_clear_done.side_effect = [{"req-0"}, set()]
    worker._gather_tp_read_status = MagicMock(
        side_effect=[
            [({"req-0"}, set()), (set(), set())],
            [({"req-0"}, set()), ({"req-0"}, set())],
        ]
    )

    assert worker.get_finished() == (set(), set())
    assert worker.get_finished() == (set(), {"req-0-internal"})


def test_consumer_load_errors_are_unioned_across_tp():
    worker = _make_consumer_worker_for_completion_test()
    worker._mf_read_thread.get_and_clear_done.return_value = set()
    worker._mf_read_thread.get_and_clear_failed.return_value = set()
    worker._gather_tp_read_status = MagicMock(return_value=[({"req-0"}, set()), ({"req-0"}, {"req-0"})])

    assert worker.get_finished() == (set(), {"req-0-internal"})

    assert worker.get_block_ids_with_load_errors() == {1, 2, 3}
    assert worker.get_block_ids_with_load_errors() == set()


def test_failed_tp_rank_remains_terminal_until_other_ranks_finish():
    worker = _make_consumer_worker_for_completion_test()
    worker._mf_read_thread.get_and_clear_done.return_value = set()
    worker._mf_read_thread.get_and_clear_failed.side_effect = [{"req-0"}, set()]
    worker._gather_tp_read_status = MagicMock(
        side_effect=[
            [({"req-0"}, {"req-0"}), (set(), set())],
            [({"req-0"}, set()), ({"req-0"}, set())],
        ]
    )

    assert worker.get_finished() == (set(), set())
    assert worker.get_finished() == (set(), {"req-0-internal"})
    assert worker._gather_tp_read_status.call_args_list[1].args[0] == {"req-0"}


def test_owned_component_without_descriptors_still_fails():
    thread = _make_read_thread()
    thread.engine = MagicMock()
    layer = _make_layer(k_cpu_ptr=3000, v_cpu_ptr=4000, has_indexer=False)
    thread._resolve_read_layer = MagicMock(return_value=layer)  # type: ignore[method-assign]
    thread._build_req_descriptors = MagicMock(  # type: ignore[method-assign]
        return_value=([], [], [], None)
    )

    with pytest.raises(RuntimeError, match="built no transfer descriptors"):
        thread._do_read_batch(
            layer["layer_name"],
            [("req-0", [1, 2], [], 0, 0)],
            p_session="p-session",
            p_layer_meta={},
        )

    thread.engine.batch_transfer_sync_read.assert_not_called()


def test_read_descriptor_rejects_missing_destination_blocks():
    thread = _make_read_thread()
    thread._state.dest_blocks_by_req.clear()

    with pytest.raises(RuntimeError, match="no destination blocks"):
        thread._build_req_descriptors(
            _make_layer(k_cpu_ptr=None, v_cpu_ptr=None, has_indexer=False),
            "req-0",
            p_main_block_ids=[1],
            p_indexer_block_ids=[],
            want_info=False,
        )


def test_read_descriptor_rejects_incomplete_indexer_transfer():
    thread = _make_read_thread()

    with pytest.raises(RuntimeError, match="indexer destination range is incomplete"):
        thread._build_req_descriptors(
            _make_layer(k_cpu_ptr=3000, v_cpu_ptr=4000),
            "req-0",
            p_main_block_ids=[1, 2],
            p_indexer_block_ids=[7, 9],
            want_info=False,
        )


def test_main_only_layer_uses_chunk_destination_slice():
    thread = _make_read_thread()

    local, peer, lengths, info = thread._build_req_descriptors(
        _make_layer(k_cpu_ptr=3000, v_cpu_ptr=4000, has_indexer=False),
        "req-0",
        p_main_block_ids=[2],
        p_indexer_block_ids=[],
        want_info=True,
        main_start_block=1,
    )

    assert local == [3040, 4080]
    assert peer == [1020, 2040]
    assert lengths == [10, 20]
    assert info is not None
    assert info["d_main_ids"] == [4]
    assert info["n_indexer"] == 0


def test_send_thread_wires_both_cache_group_block_lists():
    layer_name = "model.layers.0.self_attn"
    thread = MembPullSendingThread.__new__(MembPullSendingThread)
    thread._state = ProducerSendState(
        last_layer_idx=0,
        main_group_idx=1,
        indexer_group_idx=0,
        block_sizes=(32, 16),
        layer_metadata={
            layer_name: LayerMetadata(
                tensor_group_idx=[1, 1, 0],
                kv_caches_base_addr=[1000, 2000, 3000],
                block_len=[10, 20, 5],
                block_size_scale=[1, 1, 1],
                main_tensor_count=2,
                has_indexer=True,
            )
        },
        layer_storage_slots={0: (0, 1)},
        p_session="p-session",
    )
    thread.last_layer_idx = 0
    thread._p_save_events = {}
    thread._pending_reads_by_layer = {}
    thread._storage_read_errors = {}
    thread.storage_send_done_events = [threading.Event(), threading.Event()]
    for event in thread.storage_send_done_events:
        event.set()
    thread._mf_meta_sent_paths = set()
    thread._send_mf_meta = MagicMock()  # type: ignore[method-assign]
    dealer = MagicMock()
    thread._ensure_dealer = MagicMock(return_value=dealer)  # type: ignore[method-assign]
    encoder = MagicMock()
    encoder.encode.side_effect = lambda value: value
    req_meta = SfaPDProducerReqMeta(
        local_block_ids=[[7], [3, 4]],
        token_ids=[],
        remote_block_ids=[],
        remote_block_size=[],
        remote_engine_id=None,
        remote_host="127.0.0.1",
        remote_port=1234,
        remote_te_rpc_port=None,
        remote_layer_metadata=None,
        metaserver=None,
        remote_tp_size=None,
        remote_pcp_size=None,
        remote_dcp_size=None,
        chunk_finish=True,
        local_transed_tokens=0,
        local_computed_tokens=32,
    )

    thread._process_send_task(
        SendTask(
            send_request={"req-0": req_meta},
            layer_idx=0,
            layer_name=layer_name,
        ),
        encoder,
    )

    sent_message = dealer.send.call_args.args[0]
    assert sent_message[0] == READ_READY_BATCH
    assert sent_message[3] == [("req-0", [3, 4], [7], 0, 0)]
    assert sent_message[4] == ["req-0"]
    # Contributor identity defaults: single contributor group (member 0 of ratio 1).
    assert sent_message[5] == 0
    assert sent_message[6] == 1


def test_send_thread_slices_each_group_at_chunk_boundaries():
    layer_name = "model.layers.0.self_attn"
    thread = MembPullSendingThread.__new__(MembPullSendingThread)
    thread._state = ProducerSendState(
        last_layer_idx=1,
        main_group_idx=0,
        indexer_group_idx=1,
        block_sizes=(16, 32),
        layer_metadata={
            layer_name: LayerMetadata(
                tensor_group_idx=[0, 0, 1],
                kv_caches_base_addr=[1000, 2000, 3000],
                block_len=[10, 20, 5],
                block_size_scale=[1, 1, 1],
                main_tensor_count=2,
                has_indexer=True,
            )
        },
        layer_storage_slots={0: (0, 1)},
        p_session="p-session",
    )
    thread.last_layer_idx = 1
    thread._p_save_events = {}
    thread._pending_reads_by_layer = {}
    thread._storage_read_errors = {}
    thread.storage_send_done_events = [threading.Event(), threading.Event()]
    for event in thread.storage_send_done_events:
        event.set()
    thread._mf_meta_sent_paths = {"tcp://127.0.0.1:1234"}
    dealer = MagicMock()
    thread._ensure_dealer = MagicMock(return_value=dealer)  # type: ignore[method-assign]
    encoder = MagicMock()
    encoder.encode.side_effect = lambda value: value
    req_meta = SfaPDProducerReqMeta(
        local_block_ids=[[10, 11, 12], [20, 21]],
        token_ids=[],
        remote_block_ids=[],
        remote_block_size=[],
        remote_engine_id=None,
        remote_host="127.0.0.1",
        remote_port=1234,
        remote_te_rpc_port=None,
        remote_layer_metadata=None,
        metaserver=None,
        remote_tp_size=None,
        remote_pcp_size=None,
        remote_dcp_size=None,
        chunk_finish=False,
        local_transed_tokens=16,
        local_computed_tokens=40,
    )

    thread._process_send_task(
        SendTask(
            send_request={"req-0": req_meta},
            layer_idx=0,
            layer_name=layer_name,
        ),
        encoder,
    )

    sent_message = dealer.send.call_args.args[0]
    assert sent_message[3] == [("req-0", [11], [20], 1, 0)]
    assert sent_message[5] == 0
    assert sent_message[6] == 1


@pytest.mark.parametrize("all_groups", [False, True])
def test_producer_scheduler_cleans_request_state_on_finish(all_groups):
    scheduler = SFAPDRD2HProducerScheduler.__new__(SFAPDRD2HProducerScheduler)
    request = SimpleNamespace(request_id="req-0")
    scheduler._reqs_need_send_layerwise = {"req-0": MagicMock()}

    if all_groups:
        result = scheduler.request_finished_all_groups(request, ([1], [2]))
    else:
        result = scheduler.request_finished(request, [1])

    assert result == (False, None)
    assert "req-0" not in scheduler._reqs_need_send_layerwise


def test_producer_scheduler_resolves_batch_metadata_by_external_request_id():
    scheduler = SFAPDRD2HProducerScheduler.__new__(SFAPDRD2HProducerScheduler)
    scheduler._reqs_need_send_layerwise = {}
    child_params = {
        "do_remote_decode": True,
        "remote_cached_tokens": 16,
        "remote_host": "decode-1",
        "remote_port": 1234,
    }
    request = SimpleNamespace(
        request_id="cmpl-parent-1-12345678",
        kv_transfer_params={
            "do_remote_decode": True,
            BATCH_KV_TRANSFER_PARAMS: {
                "cmpl-parent-0": {
                    "do_remote_decode": True,
                    "remote_cached_tokens": 0,
                },
                "cmpl-parent-1": child_params,
            },
        },
    )
    blocks = MagicMock()
    blocks.get_block_ids.return_value = ([1, 2], [3])

    scheduler.update_state_after_alloc(request, blocks, num_external_tokens=0)

    assert request.kv_transfer_params is child_params
    send_req_info = scheduler._reqs_need_send_layerwise[request.request_id]
    assert send_req_info.local_transferred_tokens == 16
    assert send_req_info.local_block_ids == [[1, 2], [3]]


def test_producer_scheduler_rejects_missing_batch_metadata():
    scheduler = SFAPDRD2HProducerScheduler.__new__(SFAPDRD2HProducerScheduler)
    scheduler._reqs_need_send_layerwise = {}
    request = SimpleNamespace(
        request_id="cmpl-parent-1-12345678",
        kv_transfer_params={
            "do_remote_decode": True,
            BATCH_KV_TRANSFER_PARAMS: {
                "cmpl-parent-0": {
                    "do_remote_decode": True,
                    "remote_cached_tokens": 0,
                }
            },
        },
    )

    with pytest.raises(RuntimeError, match="cmpl-parent-1"):
        scheduler.update_state_after_alloc(request, MagicMock(), num_external_tokens=0)


def test_storage_slot_gate_is_shared_across_reuse_ring_boundary():
    thread = MembPullSendingThread.__new__(MembPullSendingThread)
    thread._state = ProducerSendState(
        last_layer_idx=5,
        layer_metadata={},
        p_session="p-session",
        main_group_idx=0,
        indexer_group_idx=0,
        block_sizes=(),
        layer_storage_slots={1: (0,), 5: (0,)},
    )
    thread.storage_send_done_events = [threading.Event()]
    thread.storage_send_done_events[0].set()
    thread._storage_read_errors = {}
    thread._pending_reads_by_layer = {}

    thread.mark_layer_pending(5)
    storage_event = thread.get_storage_send_event(0)
    assert storage_event is not None
    assert not storage_event.is_set()
    # Completing the last occupant opens the same gate observed by the first
    # occupant in the next scheduler step.
    thread._signal_layer_done(5)
    assert storage_event.is_set()
    thread.mark_layer_pending(1)
    assert not storage_event.is_set()


def test_pd_read_wait_continues_after_log_interval_until_read_done():
    event = MagicMock()
    event.wait.side_effect = [False, True]
    send_thread = MagicMock()
    send_thread.get_storage_send_event.return_value = event
    send_thread.get_storage_error.return_value = None
    send_thread.is_alive.return_value = True
    worker = SFAPDRD2HProducerWorker.__new__(SFAPDRD2HProducerWorker)
    worker.kv_send_layer_thread = send_thread
    worker.layer_storage_slots = {0: (0,)}

    worker.wait_for_layer_send(0)

    assert event.wait.call_count == 2
    send_thread.is_alive.assert_called_once_with()


def test_pd_read_wait_fails_when_send_thread_stops():
    event = MagicMock()
    event.wait.return_value = False
    send_thread = MagicMock()
    send_thread.get_storage_send_event.return_value = event
    send_thread.get_storage_error.return_value = None
    send_thread.is_alive.return_value = False
    send_thread.startup_error = None
    worker = SFAPDRD2HProducerWorker.__new__(SFAPDRD2HProducerWorker)
    worker.kv_send_layer_thread = send_thread
    worker.layer_storage_slots = {0: (0,)}

    with pytest.raises(RuntimeError, match="send thread stopped"):
        worker.wait_for_layer_send(0)


def test_pd_read_wait_propagates_read_failed():
    event = MagicMock()
    event.wait.return_value = True
    send_thread = MagicMock()
    send_thread.get_storage_send_event.return_value = event
    send_thread.get_storage_error.return_value = "memfabric read failed"
    worker = SFAPDRD2HProducerWorker.__new__(SFAPDRD2HProducerWorker)
    worker.kv_send_layer_thread = send_thread
    worker.layer_storage_slots = {0: (0,)}

    with pytest.raises(RuntimeError, match="memfabric read failed"):
        worker.wait_for_layer_send(0)


def test_save_kv_layer_requires_send_thread_without_marking_dispatched():
    worker = SFAPDRD2HProducerWorker.__new__(SFAPDRD2HProducerWorker)
    worker._backend = BACKEND_MEMFABRIC
    worker.kv_send_layer_thread = None
    worker._pd_dispatched_layers = set()
    worker.current_layer = 0

    with pytest.raises(RuntimeError, match=r"register_kv_caches\(\)"):
        worker.save_kv_layer(
            "model.layers.0.self_attn",
            [],
            MagicMock(),
            SimpleNamespace(requests={"req-0": MagicMock()}),
        )

    assert worker._pd_dispatched_layers == set()
    assert worker.current_layer == 0


def test_main_only_and_indexer_layers_share_their_physical_main_slot():
    main_only = LayerMetadata(
        tensor_group_idx=[0, 0],
        kv_caches_base_addr=[1000, 2000],
        block_len=[10, 20],
        block_size_scale=[1, 1],
        main_tensor_count=2,
        has_indexer=False,
    )
    with_indexer = LayerMetadata(
        tensor_group_idx=[0, 0, 0],
        kv_caches_base_addr=[1000, 2000, 3000],
        block_len=[10, 20, 5],
        block_size_scale=[1, 1, 1],
        main_tensor_count=2,
        has_indexer=True,
    )

    slots = SFAPDRD2HProducerWorker._infer_layer_storage_slots(
        {
            "model.layers.1.self_attn": main_only,
            "model.layers.5.self_attn": with_indexer,
        }
    )

    assert slots[1] == (0,)
    assert slots[5][0] == 0
    assert len(slots[5]) == 2


def test_consumer_scheduler_closes_remote_prefill_before_rendezvous():
    scheduler = SFAPDRD2HScheduler.__new__(SFAPDRD2HScheduler)
    scheduler.main_group_idx = 0
    scheduler.indexer_group_idx = 1
    scheduler.engine_id = "decode-engine"
    scheduler.side_channel_host = "decode-host"
    scheduler.side_channel_port = 1234
    scheduler.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        )
    )
    scheduler._request_trackers = {}
    scheduler._reqs_need_recv = set()
    scheduler._metaserver_lock = threading.Lock()
    scheduler._cancelled_metaserver_requests = set()
    scheduler._submit_metaserver_request = MagicMock()  # type: ignore[method-assign]
    params = {
        "do_remote_prefill": True,
        "metaserver": "http://metaserver",
    }
    request = SimpleNamespace(
        request_id="req-0",
        kv_transfer_params=params,
        num_computed_tokens=0,
    )
    blocks = MagicMock()
    blocks.get_block_ids.return_value = ([1], [2])

    scheduler.update_state_after_alloc(request, blocks, num_external_tokens=0)

    assert params["do_remote_prefill"] is False
    scheduler._submit_metaserver_request.assert_called_once()


def test_metaserver_treats_legacy_http_error_as_delivered():
    scheduler = SFAPDRD2HScheduler.__new__(SFAPDRD2HScheduler)
    response = MagicMock()
    response.is_error = True
    response.status_code = 500
    client = MagicMock()
    client.post.return_value = response

    with patch("vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.scheduler.httpx.Client") as client_cls:
        client_cls.return_value.__enter__.return_value = client
        scheduler._access_metaserver("http://metaserver", {"request_id": "req-0"})

    client.post.assert_called_once_with(
        "http://metaserver",
        json={"request_id": "req-0"},
    )
    assert client_cls.call_args.kwargs["timeout"] is None


def test_metaserver_retries_transport_errors():
    scheduler = SFAPDRD2HScheduler.__new__(SFAPDRD2HScheduler)
    response = MagicMock(is_error=False)
    client = MagicMock()
    client.post.side_effect = [
        httpx.ConnectError("connection refused"),
        response,
    ]

    with patch("vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.scheduler.httpx.Client") as client_cls:
        client_cls.return_value.__enter__.return_value = client
        scheduler._access_metaserver("http://metaserver", {"request_id": "req-0"})

    assert client.post.call_count == 2


def test_metaserver_callback_retries_without_changing_request_state():
    scheduler = SFAPDRD2HScheduler.__new__(SFAPDRD2HScheduler)
    scheduler._metaserver_lock = threading.Lock()
    scheduler._shutdown_event = threading.Event()
    scheduler._metaserver_retry_timers = {}
    scheduler._cancelled_metaserver_requests = set()
    failed_future: Future[None] = Future()
    scheduler._metaserver_futures = {"req-0": failed_future}
    failed_future.set_exception(RuntimeError("metaserver unavailable"))

    retry_timer = MagicMock()
    with patch(
        "vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.scheduler.threading.Timer",
        return_value=retry_timer,
    ):
        scheduler._on_metaserver_done(
            failed_future,
            request_id="req-0",
            url="http://metaserver",
            message={"request_id": "req-0"},
        )

    assert "req-0" not in scheduler._metaserver_futures
    assert scheduler._metaserver_retry_timers["req-0"] is retry_timer
    retry_timer.start.assert_called_once_with()


def test_metaserver_callback_clears_completed_future():
    scheduler = SFAPDRD2HScheduler.__new__(SFAPDRD2HScheduler)
    scheduler._metaserver_lock = threading.Lock()
    scheduler._shutdown_event = threading.Event()
    scheduler._metaserver_retry_timers = {}
    scheduler._cancelled_metaserver_requests = set()
    succeeded_future: Future[None] = Future()
    scheduler._metaserver_futures = {"req-0": succeeded_future}
    succeeded_future.set_result(None)

    scheduler._on_metaserver_done(
        succeeded_future,
        request_id="req-0",
        url="http://metaserver",
        message={"request_id": "req-0"},
    )

    assert "req-0" not in scheduler._metaserver_futures


@pytest.mark.parametrize(
    ("p_tp", "d_tp", "p_rank", "expected_d_rank"),
    [
        (8, 8, 7, 7),
        (8, 4, 0, 0),
        (8, 4, 3, 1),
        (8, 4, 7, 3),
        # 16 -> 4 fold (tp_ratio=4): the unequal-TP topology this connector targets.
        (16, 4, 0, 0),
        (16, 4, 3, 0),
        (16, 4, 4, 1),
        (16, 4, 15, 3),
    ],
)
def test_prefill_to_decode_tp_rank_mapping(p_tp, d_tp, p_rank, expected_d_rank):
    assert (
        SFAPDRD2HProducerWorker._map_prefill_rank_to_decode_rank(
            prefill_tp_size=p_tp,
            decode_tp_size=d_tp,
            prefill_tp_rank=p_rank,
        )
        == expected_d_rank
    )


@pytest.mark.parametrize(
    ("dp_rank", "tp_size", "expected_port"),
    [
        # Single-host DP4 (tp_size=4): global ranks 0..3 -> disjoint port ranges.
        (0, 4, 1000),
        (1, 4, 1004),
        (2, 4, 1008),
        (3, 4, 1012),
        # Multi-host DP4: same global-rank formula applies per host (host disambiguates).
        (3, 4, 1012),
        # Unequal-TP decode side (D tp_size=4), wider DP.
        (5, 4, 1020),
    ],
)
def test_d_side_control_plane_port_uses_global_dp_rank(dp_rank, tp_size, expected_port):
    """SFAPDRD2HScheduler must derive the TP control-plane port from the GLOBAL
    data_parallel_rank (never the per-host local rank) so single-host and
    multi-host DP both get disjoint per-engine port ranges."""
    kv_port = 1000
    kv_cache_config = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=16))],
        num_blocks=8,
    )
    vllm_config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_port=kv_port, engine_id="d0"),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp_size,
            data_parallel_rank=dp_rank,
            data_parallel_rank_local=0,  # SPMD local rank; must NOT drive the port
            data_parallel_size=8,
            data_parallel_size_local=4,
        ),
    )
    with (
        patch(
            "vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.scheduler.get_ip",
            return_value="decode-host",
        ),
        patch(
            "vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.scheduler.infer_sfa_component_group_ids",
            return_value=(0, 0),
        ),
        patch(
            "vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.scheduler.ThreadPoolExecutor",
            return_value=MagicMock(),
        ),
    ):
        scheduler = SFAPDRD2HScheduler(vllm_config, use_layerwise=True, kv_cache_config=kv_cache_config)

    assert scheduler.side_channel_port == expected_port
    # Guard: never derive the port from the per-host local rank (which is 0 here).
    assert scheduler.side_channel_port != kv_port + 0 * tp_size or dp_rank == 0


@pytest.mark.parametrize(
    ("p_tp", "d_tp", "p_rank"),
    [
        (4, 8, 0),
        (8, 3, 0),
        (0, 1, 0),
        (8, 4, 8),
    ],
)
def test_prefill_to_decode_tp_rank_mapping_rejects_invalid_topology(
    p_tp,
    d_tp,
    p_rank,
):
    with pytest.raises(ValueError):
        SFAPDRD2HProducerWorker._map_prefill_rank_to_decode_rank(
            prefill_tp_size=p_tp,
            decode_tp_size=d_tp,
            prefill_tp_rank=p_rank,
        )


def test_read_thread_propagates_bind_failure_without_hanging():
    state = _make_read_thread()._state
    thread = MembPullReadThread(
        tp_rank=0,
        side_channel_port=12345,
        engine=MagicMock(),
        state=state,
    )

    with patch(
        "vllm.utils.network_utils.make_zmq_socket",
        side_effect=OSError("address already in use"),
    ):
        thread.start()
        assert thread.ready_event.wait(timeout=1)
        thread.join(timeout=1)

    assert isinstance(thread.startup_error, OSError)
    assert not thread.is_alive()


def test_scheduler_shutdown_cancels_rendezvous_and_executor():
    scheduler = SFAPDRD2HScheduler.__new__(SFAPDRD2HScheduler)
    scheduler._metaserver_lock = threading.Lock()
    scheduler._shutdown_event = threading.Event()
    scheduler._metaserver_retry_timers = {}
    scheduler._cancelled_metaserver_requests = set()
    pending_future: Future[None] = Future()
    scheduler._metaserver_futures = {"req-0": pending_future}
    scheduler.executor = MagicMock()

    scheduler.shutdown()

    assert pending_future.cancelled()
    scheduler.executor.shutdown.assert_called_once_with(
        wait=False,
        cancel_futures=True,
    )


def test_connector_shutdown_delegates_to_active_components():
    connector = SfaRemoteD2HConnector.__new__(SfaRemoteD2HConnector)
    connector.connector_worker = MagicMock()
    connector.connector_scheduler = MagicMock()

    connector.shutdown()

    connector.connector_worker.shutdown.assert_called_once_with()
    connector.connector_scheduler.shutdown.assert_called_once_with()


# ---------------------------------------------------------------------------
# Contributor-group split (unequal P/D TP)
# ---------------------------------------------------------------------------


def _make_indexer_only_read_thread(indexer_dest: list[int], main_dest: list[int] | None = None) -> MembPullReadThread:
    thread = MembPullReadThread.__new__(MembPullReadThread)
    thread.tp_rank = 0
    thread._state = ConsumerReadState(
        num_blocks=64,
        tp_size=1,
        layer_metadata={},
        main_name_to_idx={},
        cpu_pools=[],
        main_gva_bases=[],
        main_block_lens=[],
        indexer_tensors=[],
        indexer_scale_tensors=[],
        dest_blocks_by_req={"req-0": (main_dest or [], indexer_dest)},
        get_offload_layer_id=lambda _: 0,
    )
    return thread


def test_member0_pulls_main_and_indexer_slice():
    # ratio=2: member 0 pulls main (replicated, no split) + its indexer half.
    thread = _make_indexer_only_read_thread([10, 11, 12, 13], main_dest=[20, 21])
    layer = _make_layer(k_cpu_ptr=3000, v_cpu_ptr=4000)

    local, peer, lengths, info = thread._build_req_descriptors(
        layer,
        "req-0",
        p_main_block_ids=[1, 2],
        p_indexer_block_ids=[7, 8, 9, 10],
        want_info=True,
        group_member_idx=0,
        ratio=2,
    )

    assert info is not None
    assert info["n_main"] == 2  # member 0 pulls the whole main share
    assert info["n_indexer"] == 2  # and the first half of the indexer
    # indexer half = d blocks [10, 11] -> d_base 8000 + {10,11}*5. Adjacent
    # indexer blocks coalesce into one transfer, so assert the pulled d-block
    # ids rather than individual local pointers.
    assert info["d_indexer_ids"] == [10, 11]
    assert 8050 in local  # coalesced head of [10, 11]
    assert 8060 not in local and 8065 not in local


def test_nonzero_member_skips_main_and_pulls_other_indexer_slice():
    thread = _make_indexer_only_read_thread([10, 11, 12, 13])
    layer = _make_layer(k_cpu_ptr=3000, v_cpu_ptr=4000)

    local, peer, lengths, info = thread._build_req_descriptors(
        layer,
        "req-0",
        p_main_block_ids=[1, 2],
        p_indexer_block_ids=[7, 8, 9, 10],
        want_info=True,
        group_member_idx=1,
        ratio=2,
    )

    assert info is not None
    assert info["n_main"] == 0  # main is contributor-0 only
    assert info["n_indexer"] == 2  # second half of the indexer
    # main K/V destinations absent
    assert 3000 not in local and 4000 not in local
    # indexer second half = d blocks [12, 13]; adjacent blocks coalesce, so
    # assert the pulled d-block ids rather than individual local pointers.
    assert info["d_indexer_ids"] == [12, 13]
    assert 8060 in local  # coalesced head of [12, 13]
    assert 8050 not in local and 8055 not in local


def test_indexer_slices_are_disjoint_and_cover_full_range():
    thread = _make_indexer_only_read_thread([10, 11, 12, 13])
    layer = _make_layer(k_cpu_ptr=None, v_cpu_ptr=None)
    covered: set[int] = set()
    for member in range(4):
        local, _, _, info = thread._build_req_descriptors(
            layer,
            "req-0",
            p_main_block_ids=[],
            p_indexer_block_ids=[7, 8, 9, 10],
            want_info=True,
            group_member_idx=member,
            ratio=4,
        )
        assert info is not None
        d_ids = set(info["d_indexer_ids"])
        assert covered.isdisjoint(d_ids)
        covered |= d_ids
    assert covered == {10, 11, 12, 13}


def test_ratio_one_degenerates_to_full_pull():
    thread = _make_indexer_only_read_thread([10, 11, 12, 13], main_dest=[20, 21])
    layer = _make_layer(k_cpu_ptr=3000, v_cpu_ptr=4000)

    _, _, _, info = thread._build_req_descriptors(
        layer,
        "req-0",
        p_main_block_ids=[1, 2],
        p_indexer_block_ids=[7, 8, 9, 10],
        want_info=True,
        group_member_idx=0,
        ratio=1,
    )

    assert info is not None
    assert info["n_main"] == 2
    assert info["n_indexer"] == 4  # full indexer, no split


def _make_completion_thread() -> MembPullReadThread:
    thread = MembPullReadThread.__new__(MembPullReadThread)
    thread._done_requests = set()
    thread._failed_requests = set()
    thread._done_contributors = {}
    thread._expected_ratio = {}
    thread._lock = threading.Lock()
    return thread


def test_request_done_requires_all_contributors():
    thread = _make_completion_thread()
    thread._record_chunk_done(["req-0"], 0, 4)
    thread._record_chunk_done(["req-0"], 1, 4)
    thread._record_chunk_done(["req-0"], 2, 4)
    assert "req-0" not in thread._done_requests
    thread._record_chunk_done(["req-0"], 3, 4)
    assert "req-0" in thread._done_requests
    # contributor state released once complete
    assert "req-0" not in thread._done_contributors
    assert "req-0" not in thread._expected_ratio


def test_duplicate_contributor_does_not_double_count():
    thread = _make_completion_thread()
    thread._record_chunk_done(["req-0"], 0, 3)
    thread._record_chunk_done(["req-0"], 0, 3)  # same member re-delivers
    thread._record_chunk_done(["req-0"], 1, 3)
    assert "req-0" not in thread._done_requests  # still missing member 2


def test_ratio_one_completes_immediately():
    thread = _make_completion_thread()
    thread._record_chunk_done(["req-0"], 0, 1)
    assert "req-0" in thread._done_requests


def test_discard_requests_clears_partial_contributor_state():
    thread = _make_completion_thread()
    thread._record_chunk_done(["req-0"], 0, 4)
    assert "req-0" in thread._done_contributors
    thread.discard_requests({"req-0"})
    assert "req-0" not in thread._done_contributors
    assert "req-0" not in thread._expected_ratio
