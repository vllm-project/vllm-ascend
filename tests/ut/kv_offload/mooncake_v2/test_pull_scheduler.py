# SPDX-License-Identifier: Apache-2.0

import threading
import time
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import msgspec
import pytest
import torch
import zmq
from vllm.sampling_params import SamplingParams
from vllm.v1.request import Request, RequestStatus

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake import heartbeat as heartbeat_module
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake import pull_scheduler
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.base_scheduler import (
    MooncakeBaseConnectorScheduler,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.connector import MooncakePullConnector
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.heartbeat import (
    HEARTBEAT_MSG,
    MooncakeHeartbeatThread,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.metadata import (
    MooncakeTransferMetadataGroups,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_scheduler import (
    ACK_MSG,
    DONE_RECVING_MSG,
    MooncakePullConnectorScheduler,
    MooncakeSchedulerRecvingThread,
    MooncakeSchedulerSendingThread,
)

from .helpers import (
    make_blocks,
    make_circular_spec,
    make_full_spec,
    make_mamba_spec,
    make_request,
    make_transfer_metadata,
)


def make_sending_thread(
    metadata: dict[int | tuple[int, ...], object] | None = None,
    *,
    tp_size: int = 1,
    pp_size: int = 1,
    pcp_size: int = 1,
    dcp_size: int = 1,
    use_kv_pp: bool = False,
) -> MooncakeSchedulerSendingThread:
    return MooncakeSchedulerSendingThread(
        host="127.0.0.1",
        port=6000,
        engine_id="engine-p",
        metadata=metadata or {0: make_transfer_metadata()},  # type: ignore[arg-type]
        tp_size=tp_size,
        pp_size=pp_size,
        pcp_size=pcp_size,
        dcp_size=dcp_size,
        use_kv_pp=use_kv_pp,
        ready_event=threading.Event(),
    )


def make_pull_scheduler() -> MooncakePullConnectorScheduler:
    scheduler = MooncakePullConnectorScheduler.__new__(MooncakePullConnectorScheduler)
    scheduler.need_truncate = False
    scheduler._logged_mamba_allocation = False
    scheduler.num_speculative_tokens = 0
    scheduler.pcp_size = 1
    scheduler.dcp_size = 1
    scheduler.ascend_config = SimpleNamespace(kvpp_config=SimpleNamespace(size=1))
    scheduler.group_block_size = [16]
    scheduler.group_unique_specs = [[make_full_spec()]]
    scheduler.engine_id = "engine-p"
    scheduler.side_channel_host = "10.0.0.10"
    scheduler.side_channel_port = 6000
    scheduler._reqs_need_recv = {}
    scheduler._reqs_need_send = {}
    scheduler._reqs_in_batch = set()
    scheduler._reqs_recv_info = {}
    scheduler._sending_thread = None
    scheduler._recving_thread = None
    scheduler._heartbeat_thread = None
    scheduler._kv_lease_duration = 30
    return scheduler


def test_sending_thread_merges_tp_private_and_pp_common_metadata() -> None:
    first = make_transfer_metadata(te_rpc_port=9000, local_ip="10.0.0.1", base_addrs=[[1000]])
    second = make_transfer_metadata(te_rpc_port=9001, local_ip="10.0.0.2", base_addrs=[[2000]])

    thread = make_sending_thread({0: first, 1: second}, tp_size=2)
    decoded = msgspec.msgpack.decode(thread.encoded_metadata, type=MooncakeTransferMetadataGroups)

    assert decoded.use_kv_pp is False
    pp_metadata = decoded.metadata_by_pp_rank[0]
    assert pp_metadata.layer_names == first.layer_names
    assert pp_metadata.block_shapes == first.block_shapes
    tp_metadata = pp_metadata.metadata_by_pcp_rank[0].metadata_by_tp_rank
    assert tp_metadata[0].kv_caches_base_addr == [[1000]]
    assert tp_metadata[1].kv_caches_base_addr == [[2000]]
    assert tp_metadata[1].te_rpc_port == 9001


def test_sending_thread_merges_layer_split_metadata_by_name() -> None:
    tp0 = make_transfer_metadata(
        layer_names=["layer.0", "layer.2"],
        group_indices=[0, 0],
        base_addrs=[[1000], [3000]],
        block_strides=[[128], [128]],
        block_lens=[[128], [128]],
        block_shapes=[[(1, 16, 4)], [(1, 16, 4)]],
        block_size_scales=[[1], [1]],
    )
    tp1 = make_transfer_metadata(
        te_rpc_port=9001,
        local_ip="10.0.0.2",
        layer_names=["layer.1", "layer.2"],
        group_indices=[0, 0],
        base_addrs=[[2000], [4000]],
        block_strides=[[128], [128]],
        block_lens=[[128], [128]],
        block_shapes=[[(1, 16, 4)], [(1, 16, 4)]],
        block_size_scales=[[1], [1]],
    )

    thread = make_sending_thread({0: tp0, 1: tp1}, tp_size=2, use_kv_pp=True)
    decoded = msgspec.msgpack.decode(thread.encoded_metadata, type=MooncakeTransferMetadataGroups)

    assert decoded.use_kv_pp is True
    pp_metadata = decoded.metadata_by_pp_rank[0]
    assert pp_metadata.layer_names == ["layer.0", "layer.1", "layer.2"]
    assert pp_metadata.layer_block_sizes == [16, 16, 16]
    tp_metadata = pp_metadata.metadata_by_pcp_rank[0].metadata_by_tp_rank
    assert tp_metadata[0].layer_indices == [0, 2]
    assert tp_metadata[0].kv_caches_base_addr == [[1000], [], [3000]]
    assert tp_metadata[1].layer_indices == [1, 2]
    assert tp_metadata[1].kv_caches_base_addr == [[], [2000], [4000]]


def test_sending_thread_rejects_layer_split_with_dcp() -> None:
    tp0 = make_transfer_metadata(layer_names=["layer.0"], base_addrs=[[1000]])
    tp1 = make_transfer_metadata(
        te_rpc_port=9001,
        local_ip="10.0.0.2",
        layer_names=["layer.1"],
        base_addrs=[[2000]],
    )

    with pytest.raises(ValueError, match="KVPP cannot be combined with DCP"):
        make_sending_thread({0: tp0, 1: tp1}, tp_size=2, dcp_size=2, use_kv_pp=True)


def test_sending_thread_aligns_different_tp_layer_orders() -> None:
    tp0 = make_transfer_metadata(
        layer_names=["layer.0", "layer.1"],
        group_indices=[0, 0],
        base_addrs=[[1000], [2000]],
        block_strides=[[128], [128]],
        block_lens=[[128], [128]],
        block_shapes=[[(1, 16, 4)], [(1, 16, 4)]],
        block_size_scales=[[1], [1]],
    )
    tp1 = make_transfer_metadata(
        te_rpc_port=9001,
        local_ip="10.0.0.2",
        layer_names=["layer.1", "layer.0"],
        group_indices=[0, 0],
        base_addrs=[[4000], [3000]],
        block_strides=[[128], [128]],
        block_lens=[[128], [128]],
        block_shapes=[[(1, 16, 4)], [(1, 16, 4)]],
        block_size_scales=[[1], [1]],
    )

    thread = make_sending_thread({0: tp0, 1: tp1}, tp_size=2)
    decoded = msgspec.msgpack.decode(thread.encoded_metadata, type=MooncakeTransferMetadataGroups)

    pp_metadata = decoded.metadata_by_pp_rank[0]
    assert pp_metadata.layer_names == ["layer.0", "layer.1"]
    tp_metadata = pp_metadata.metadata_by_pcp_rank[0].metadata_by_tp_rank
    assert tp_metadata[0].layer_indices == [0, 1]
    assert tp_metadata[1].layer_indices == [0, 1]
    assert tp_metadata[1].kv_caches_base_addr == [
        [3000],
        [4000],
    ]


def test_sending_thread_accepts_pp_aware_keys() -> None:
    pp0 = make_transfer_metadata(layer_names=["pp0.layer"])
    pp1 = make_transfer_metadata(layer_names=["pp1.layer"], base_addrs=[[3000]])

    thread = make_sending_thread({(0, 0): pp0, (1, 0): pp1}, pp_size=2)
    decoded = msgspec.msgpack.decode(thread.encoded_metadata, type=MooncakeTransferMetadataGroups)

    assert decoded.metadata_by_pp_rank[0].layer_names == ["pp0.layer"]
    assert decoded.metadata_by_pp_rank[1].layer_names == ["pp1.layer"]


def test_sending_thread_groups_kvpp_layers_across_all_pcp_tp_workers() -> None:
    metadata: dict[int | tuple[int, ...], object] = {}
    for pcp_rank, tp_rank, kvpp_rank in ((0, 0, 0), (0, 1, 1), (1, 0, 2), (1, 1, 3)):
        metadata[(0, pcp_rank, tp_rank)] = make_transfer_metadata(
            te_rpc_port=9000 + kvpp_rank,
            local_ip=f"10.0.{pcp_rank}.{tp_rank + 1}",
            layer_names=[f"layer.{kvpp_rank}", "mtp.layer"],
            base_addrs=[[1000 + kvpp_rank * 1000], [9000 + kvpp_rank * 100]],
            handshake_port=5000 + kvpp_rank,
        )

    thread = make_sending_thread(metadata, tp_size=2, pcp_size=2, use_kv_pp=True)
    decoded = msgspec.msgpack.decode(thread.encoded_metadata, type=MooncakeTransferMetadataGroups)

    pp_metadata = decoded.metadata_by_pp_rank[0]
    assert decoded.use_kv_pp
    assert pp_metadata.layer_names == ["layer.0", "layer.1", "layer.2", "layer.3", "mtp.layer"]
    assert set(pp_metadata.metadata_by_pcp_rank) == {0, 1}
    for pcp_rank, tp_rank, kvpp_rank in ((0, 0, 0), (0, 1, 1), (1, 0, 2), (1, 1, 3)):
        worker_metadata = pp_metadata.metadata_by_pcp_rank[pcp_rank].metadata_by_tp_rank[tp_rank]
        assert worker_metadata.layer_indices == [kvpp_rank, 4]
        assert worker_metadata.kv_caches_base_addr[kvpp_rank] == [1000 + kvpp_rank * 1000]
        assert worker_metadata.kv_caches_base_addr[4] == [9000 + kvpp_rank * 100]


def test_sending_thread_uses_configured_kvpp_when_worker_layers_match() -> None:
    metadata: dict[int | tuple[int, ...], object] = {
        (0, pcp_rank, tp_rank): make_transfer_metadata(
            te_rpc_port=9000 + pcp_rank * 2 + tp_rank,
            base_addrs=[[1000 + pcp_rank * 2000 + tp_rank * 1000]],
        )
        for pcp_rank in range(2)
        for tp_rank in range(2)
    }

    thread = make_sending_thread(metadata, tp_size=2, pcp_size=2, use_kv_pp=True)
    decoded = msgspec.msgpack.decode(thread.encoded_metadata, type=MooncakeTransferMetadataGroups)

    assert decoded.use_kv_pp


def test_sending_thread_rejects_layer_split_when_kvpp_is_disabled() -> None:
    tp0 = make_transfer_metadata(layer_names=["layer.0"])
    tp1 = make_transfer_metadata(layer_names=["layer.1"], te_rpc_port=9001)

    with pytest.raises(ValueError, match="different KV-cache layers while KVPP is disabled"):
        make_sending_thread({0: tp0, 1: tp1}, tp_size=2)


def test_sending_thread_rejects_incomplete_or_inconsistent_workers() -> None:
    metadata = make_transfer_metadata()
    with pytest.raises(ValueError, match="incomplete PCP ranks"):
        make_sending_thread({(0, 0, 0): metadata}, pcp_size=2)

    with pytest.raises(ValueError, match="incomplete TP ranks"):
        make_sending_thread({0: metadata}, tp_size=2)

    mismatched = replace(metadata, block_lens=[[256]], te_rpc_port=9001)
    with pytest.raises(ValueError, match="mismatch in.*block_lens"):
        make_sending_thread({0: metadata, 1: mismatched}, tp_size=2)


def test_sending_thread_handles_completion_once() -> None:
    thread = make_sending_thread()

    thread.add_delayed_request("normal")
    thread._handle_finished_request("normal")
    thread._handle_finished_request("normal")
    assert thread.get_and_clear_finished_requests() == {"normal"}
    assert not thread.delayed_free_requests

    thread._handle_finished_request("normal")
    assert thread.get_and_clear_finished_requests() == set()


def test_sending_thread_ignores_unknown_completion() -> None:
    thread = make_sending_thread()

    thread._handle_finished_request("unknown")
    assert thread.get_and_clear_finished_requests() == set()
    thread.add_delayed_request("unknown")
    assert thread.get_and_clear_finished_requests() == set()
    thread._handle_finished_request("unknown")
    assert thread.get_and_clear_finished_requests() == {"unknown"}


def test_sending_thread_acks_duplicate_and_unknown_completion() -> None:
    thread = make_sending_thread()
    thread.add_delayed_request("normal")
    sock = MagicMock()
    sock.recv_multipart.side_effect = [
        [b"consumer", b"", msgspec.msgpack.encode((DONE_RECVING_MSG, request_id))]
        for request_id in ("normal", "normal", "unknown")
    ] + [KeyboardInterrupt()]

    with pytest.raises(KeyboardInterrupt):
        thread._run_busy_loop(sock)

    assert sock.send_multipart.call_args_list == [call((b"consumer", b"", ACK_MSG))] * 3
    assert thread.get_and_clear_finished_requests() == {"normal"}


def test_sending_thread_does_not_retain_completed_request_ids() -> None:
    thread = make_sending_thread()

    for index in range(100):
        request_id = f"request-{index}"
        thread.add_delayed_request(request_id)
        thread._handle_finished_request(request_id)
        assert thread.get_and_clear_finished_requests() == {request_id}
        thread._handle_finished_request(request_id)

    assert thread.get_and_clear_finished_requests() == set()
    assert not thread.delayed_free_requests
    assert thread.finished_requests.empty()
    assert not getattr(thread, "finished_request_ids", set())
    assert not getattr(thread, "early_finished_requests", set())


def test_sending_thread_force_frees_expired_request(monkeypatch: pytest.MonkeyPatch) -> None:
    thread = make_sending_thread()
    clock = MagicMock(return_value=10.0)
    monkeypatch.setattr(
        pull_scheduler.time,
        "monotonic",
        clock,
    )
    thread.add_delayed_request("expired")
    clock.return_value = 40.0

    assert thread.get_and_clear_finished_requests() == {"expired"}
    assert not thread.delayed_free_requests

    thread._handle_finished_request("expired")
    assert thread.get_and_clear_finished_requests() == set()


def test_recving_thread_reuses_socket_after_ack_and_discards_it_on_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    thread = MooncakeSchedulerRecvingThread(threading.Event())
    socket = MagicMock()
    thread._get_remote_socket = MagicMock(return_value=socket)  # type: ignore[method-assign]
    thread._return_remote_socket = MagicMock()  # type: ignore[method-assign]
    send = MagicMock()
    recv = MagicMock(return_value=ACK_MSG)
    monkeypatch.setattr(
        pull_scheduler,
        "ensure_zmq_send",
        send,
    )
    monkeypatch.setattr(
        pull_scheduler,
        "ensure_zmq_recv",
        recv,
    )

    thread._send_done_recving("10.0.0.1", 6000, "request-p")

    path = "tcp://10.0.0.1:6000"
    thread._get_remote_socket.assert_called_once_with(path)
    thread._return_remote_socket.assert_called_once_with(path, socket)
    socket.close.assert_not_called()

    recv.return_value = b"not-ack"
    with pytest.raises(RuntimeError, match="Unexpected.*completion response"):
        thread._send_done_recving("10.0.0.1", 6000, "request-p")

    socket.close.assert_called_once_with(linger=0)
    assert thread._return_remote_socket.call_count == 1


def test_recving_thread_socket_pool_creates_once_and_reuses_by_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    thread = MooncakeSchedulerRecvingThread(threading.Event())
    context = MagicMock()
    socket = MagicMock()
    context_cls = MagicMock(return_value=context)
    make_socket = MagicMock(return_value=socket)
    monkeypatch.setattr(
        pull_scheduler.zmq,
        "Context",
        context_cls,
    )
    monkeypatch.setattr(
        pull_scheduler,
        "make_zmq_socket",
        make_socket,
    )
    path = "tcp://10.0.0.1:6000"

    first = thread._get_remote_socket(path)
    thread._return_remote_socket(path, first)
    second = thread._get_remote_socket(path)

    assert first is second is socket
    context_cls.assert_called_once_with()
    make_socket.assert_called_once_with(
        ctx=context,
        path=path,
        socket_type=zmq.REQ,  # type: ignore[attr-defined]
        bind=False,
    )
    assert socket.setsockopt.call_count == 2


def test_base_scheduler_clips_attention_and_keeps_mamba_state_blocks() -> None:
    scheduler = MooncakeBaseConnectorScheduler.__new__(MooncakeBaseConnectorScheduler)
    scheduler.pcp_size = 1
    scheduler.dcp_size = 1
    scheduler.num_speculative_tokens = 2
    scheduler.group_block_size = [16, 16]
    scheduler.group_unique_specs = [[make_full_spec()], [replace(make_mamba_spec(), num_speculative_blocks=2)]]

    result = scheduler._get_transfer_block_ids(
        ([10, 11, 12, 13], [20, 21, 22, 23]),
        prompt_len=17,
    )

    assert result == ([10, 11], [20, 21])


def test_get_num_new_matched_tokens_records_local_prefix() -> None:
    scheduler = make_pull_scheduler()
    request = make_request(kv_transfer_params={"do_remote_prefill": True})

    count, is_async = scheduler.get_num_new_matched_tokens(request, num_computed_tokens=16)

    assert (count, is_async) == (16, True)
    assert request.kv_transfer_params["num_computed_tokens"] == 16


def test_update_after_alloc_and_build_metadata() -> None:
    scheduler = make_pull_scheduler()
    request = make_request(
        request_id="request-d",
        kv_transfer_params={
            "do_remote_prefill": True,
            "remote_block_ids": ([20, 21],),
            "remote_engine_id": "engine-p",
            "remote_host": "10.0.0.1",
            "remote_port": 6000,
            "remote_request_id": "request-p",
            "remote_num_prompt_tokens": 31,
            "num_computed_tokens": 16,
        },
    )

    scheduler.update_state_after_alloc(request, make_blocks(), num_external_tokens=16)
    metadata = scheduler.build_connector_meta(MagicMock())

    assert request.kv_transfer_params["do_remote_prefill"] is False
    assert metadata.reqs_in_batch == {"request-d"}
    assert metadata.requests["request-d"].local_block_ids == ([10, 11],)
    assert metadata.requests["request-d"].local_full_block_ids == ([1, 2, 10, 11],)
    assert scheduler._reqs_need_recv == {}


def test_zero_external_tokens_acknowledges_without_worker_transfer() -> None:
    scheduler = make_pull_scheduler()
    scheduler._recving_thread = MagicMock()
    request = make_request(
        kv_transfer_params={
            "do_remote_prefill": True,
            "remote_block_ids": ([20],),
            "remote_engine_id": "engine-p",
            "remote_host": "10.0.0.1",
            "remote_port": 6000,
            "remote_request_id": "request-p",
        }
    )

    scheduler.update_state_after_alloc(request, make_blocks(), num_external_tokens=0)

    assert scheduler._reqs_need_recv == {}
    scheduler._recving_thread.add_request.assert_called_once_with("10.0.0.1", 6000, "request-p")


def test_request_finished_delays_blocks_and_builds_remote_params() -> None:
    scheduler = make_pull_scheduler()
    scheduler._sending_thread = MagicMock()
    request = make_request(
        request_id="request-p",
        status=RequestStatus.FINISHED_LENGTH_CAPPED,
        kv_transfer_params={"do_remote_decode": True},
        output_token_ids=[123],
    )

    delay_free, params = scheduler.request_finished(request, ([10, 11, 12],))

    assert delay_free is True
    assert params is not None
    assert params["remote_block_ids"] == ([10, 11],)
    assert params["remote_request_id"] == "request-p"
    assert params["last_token_id"] == 123
    scheduler._sending_thread.add_delayed_request.assert_called_once()


@pytest.mark.parametrize("failed", [False, True])
def test_update_connector_output_routes_worker_completion_and_scheduler_ack(failed: bool) -> None:
    scheduler = make_pull_scheduler()
    scheduler.kv_transfer_config = SimpleNamespace(kv_load_failure_policy="fail")
    scheduler._recving_thread = MagicMock()
    scheduler._sending_thread = MagicMock()
    scheduler._sending_thread.get_and_clear_finished_requests.return_value = {"request-p"}
    scheduler._reqs_recv_info["request-d"] = ("10.0.0.1", 6000, "request-p")
    scheduler._reqs_need_send["request-p"] = time.time()
    output = SimpleNamespace(
        finished_recving={"request-d"},
        finished_sending=None,
        failed_recving={"request-d"} if failed else set(),
    )

    scheduler.update_connector_output(output)  # type: ignore[arg-type]

    scheduler._recving_thread.add_request.assert_called_once_with("10.0.0.1", 6000, "request-p")
    assert output.finished_sending == {"request-p"}
    assert scheduler._reqs_need_send == {}


class _StopLoop(BaseException):
    """Stop an otherwise unbounded thread loop after one test iteration."""


def test_base_scheduler_detects_state_and_compressed_prefill_truncation() -> None:
    scheduler = MooncakeBaseConnectorScheduler.__new__(MooncakeBaseConnectorScheduler)
    scheduler.vllm_config = SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace()))
    scheduler.group_unique_specs = [[make_full_spec()]]
    assert scheduler._needs_prefill_token_truncation() is False

    scheduler.vllm_config.model_config.hf_config.compress_ratios = [2]
    assert scheduler._needs_prefill_token_truncation() is True

    scheduler.vllm_config.model_config.hf_config.compress_ratios = None
    scheduler.group_unique_specs = [[make_full_spec()], [make_mamba_spec()]]
    assert scheduler._needs_prefill_token_truncation() is True

    scheduler.group_unique_specs = [[make_full_spec()], [make_circular_spec()]]
    assert scheduler._needs_prefill_token_truncation() is False


def test_truncate_request_for_prefill_is_idempotent_for_token_ids() -> None:
    scheduler = MooncakeBaseConnectorScheduler.__new__(MooncakeBaseConnectorScheduler)
    scheduler.need_truncate = True
    request = make_request(
        prompt_token_ids=[1, 2, 3],
        num_prompt_tokens=3,
        _all_token_ids=[1, 2, 3],
        kv_transfer_params={},
        max_tokens=8,
    )

    scheduler._truncate_request_for_prefill(request)
    scheduler._truncate_request_for_prefill(request)

    assert request.prompt_token_ids == [1, 2]
    assert request._all_token_ids == [1, 2]
    assert request.num_prompt_tokens == 2
    assert request.max_tokens == 1
    assert request.kv_transfer_params["_p_side_truncated"] is True


def test_truncate_request_for_prefill_supports_prompt_embeddings() -> None:
    scheduler = MooncakeBaseConnectorScheduler.__new__(MooncakeBaseConnectorScheduler)
    scheduler.need_truncate = True
    prompt_embeds = torch.arange(12).reshape(3, 4)
    request = make_request(
        prompt_token_ids=None,
        prompt_embeds=prompt_embeds,
        num_prompt_tokens=3,
        _all_token_ids=[1, 2, 3],
        kv_transfer_params={},
    )

    scheduler._truncate_request_for_prefill(request)

    assert torch.equal(request.prompt_embeds, prompt_embeds[:2])
    assert request.num_prompt_tokens == 2


def test_sending_busy_loop_serves_metadata_and_completion_ack() -> None:
    thread = make_sending_thread()
    thread._handle_finished_request = MagicMock()  # type: ignore[method-assign]
    encoder = msgspec.msgpack.Encoder()
    socket = MagicMock()
    socket.recv_multipart.side_effect = [
        [b"worker", b"", encoder.encode((b"get_meta_msg",))],
        [b"worker", b"", encoder.encode((b"done_recving_msg", "request-p"))],
        _StopLoop(),
    ]

    with pytest.raises(_StopLoop):
        thread._run_busy_loop(socket)

    assert socket.send_multipart.call_args_list == [
        call((b"worker", b"", thread.encoded_metadata)),
        call((b"worker", b"", ACK_MSG)),
    ]
    thread._handle_finished_request.assert_called_once_with("request-p")


def test_recving_run_discards_failed_completion_and_cleans_queue() -> None:
    ready_event = threading.Event()
    thread = MooncakeSchedulerRecvingThread(ready_event)
    request = ("10.0.0.1", 6000, "request-p")
    thread.request_queue.put(request)
    thread.request_queue.put(None)
    # Pending work after the stop sentinel must also be cleared on exit.
    thread.request_queue.put(("10.0.0.2", 6000, "pending"))
    thread._send_done_recving = MagicMock(side_effect=RuntimeError("network error"))  # type: ignore[method-assign]

    thread.run()

    assert ready_event.is_set()
    thread._send_done_recving.assert_called_once_with(*request)
    assert thread.request_queue.empty()
    assert thread.request_queue.unfinished_tasks == 0


def test_set_worker_metadata_starts_only_one_producer_sending_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = make_pull_scheduler()
    scheduler.kv_transfer_config = SimpleNamespace(is_kv_producer=True)
    scheduler.side_channel_host = "127.0.0.1"
    scheduler.side_channel_port = 6000
    scheduler.tp_size = 2
    scheduler.pp_size = 1
    scheduler.pcp_size = 1
    scheduler.dcp_size = 1
    scheduler.ascend_config.kvpp_config.size = 2
    created: list[MagicMock] = []

    def make_fake_thread(*args, **_kwargs):
        fake = MagicMock()
        ready_event = args[-1]
        fake.start.side_effect = ready_event.set
        fake.is_alive.return_value = True
        created.append(fake)
        return fake

    thread_cls = MagicMock(side_effect=make_fake_thread)
    monkeypatch.setattr(
        pull_scheduler,
        "MooncakeSchedulerSendingThread",
        thread_cls,
    )
    metadata = {0: make_transfer_metadata()}

    scheduler.set_xfer_handshake_metadata_from_workers(metadata)
    scheduler.set_xfer_handshake_metadata_from_workers(metadata)

    thread_cls.assert_called_once()
    assert thread_cls.call_args.args[-2] is True
    created[0].start.assert_called_once_with()


def test_set_worker_metadata_is_ignored_on_consumer() -> None:
    scheduler = make_pull_scheduler()
    scheduler.kv_transfer_config = SimpleNamespace(is_kv_producer=False)

    scheduler.set_xfer_handshake_metadata_from_workers({0: make_transfer_metadata()})

    assert scheduler._sending_thread is None


@pytest.fixture
def prefix_connector():
    result = MooncakePullConnector.__new__(MooncakePullConnector)
    result.connector_scheduler = MooncakePullConnectorScheduler.__new__(MooncakePullConnectorScheduler)
    result.connector_scheduler.need_truncate = True
    result.connector_scheduler._heartbeat_thread = None
    return result


def make_prefix_request(num_tokens, params, use_embeds=False):
    request = Request(
        request_id="prefill",
        prompt_token_ids=None if use_embeds else list(range(num_tokens)),
        prompt_embeds=torch.zeros(num_tokens, 4) if use_embeds else None,
        sampling_params=SamplingParams(max_tokens=8),
        pooling_params=None,
    )
    request.kv_transfer_params = params
    return request


@pytest.mark.parametrize("num_tokens", [512, 513, 514])
@pytest.mark.parametrize("use_embeds", [False, True])
def test_truncation_precedes_prefix_lookup(prefix_connector, num_tokens, use_embeds):
    request = make_prefix_request(num_tokens, {"do_remote_decode": True}, use_embeds)

    # Scheduler.add_request calls this facade before querying the prefix cache.
    prefix_connector.on_new_request(request)
    assert request.num_prompt_tokens == num_tokens - 1
    assert len(request.all_token_ids) == num_tokens - 1
    if use_embeds:
        assert request.prompt_embeds.shape[0] == num_tokens - 1
    else:
        assert request.prompt_token_ids == list(range(num_tokens - 1))
    assert request.max_tokens == 1
    assert request.kv_transfer_params["_p_side_truncated"] is True

    # A warm cache can return only complete blocks before the final token.
    # In particular, raw 513 must yield a 384-token hit, not 512.
    local_hit = ((request.num_prompt_tokens - 1) // 128) * 128
    assert local_hit == {512: 384, 513: 384, 514: 512}[num_tokens]
    assert prefix_connector.get_num_new_matched_tokens(request, local_hit) == (0, False)
    assert request.num_prompt_tokens - local_hit > 0

    # Reentry/preemption must not remove another token.
    prefix_connector.on_new_request(request)
    assert prefix_connector.get_num_new_matched_tokens(request, local_hit) == (0, False)
    assert request.num_prompt_tokens == num_tokens - 1


@pytest.mark.parametrize("params", [None, {}, {"do_remote_prefill": True}, {"do_remote_decode": False}])
def test_non_producer_requests_are_not_truncated(prefix_connector, params):
    request = make_prefix_request(513, params)
    prefix_connector.on_new_request(request)
    assert request.num_prompt_tokens == 513
    assert len(request.all_token_ids) == 513
    assert request.max_tokens == 8
    if params and params.get("do_remote_prefill"):
        assert prefix_connector.get_num_new_matched_tokens(request, 128) == (384, True)
        assert request.num_prompt_tokens == 513


@pytest.mark.parametrize("need_truncate,num_tokens", [(False, 513), (True, 1)])
def test_truncation_guards(prefix_connector, need_truncate, num_tokens):
    prefix_connector.connector_scheduler.need_truncate = need_truncate
    request = make_prefix_request(num_tokens, {"do_remote_decode": True})
    prefix_connector.on_new_request(request)
    assert request.num_prompt_tokens == num_tokens
    assert request.max_tokens == 8
    assert "_p_side_truncated" not in request.kv_transfer_params


def test_matched_token_query_does_not_change_prompt_length(prefix_connector):
    request = make_prefix_request(513, {"do_remote_decode": True})
    # Protect against reintroducing the late mutation, after a 512-token hit.
    assert prefix_connector.get_num_new_matched_tokens(request, 512) == (0, False)
    assert request.num_prompt_tokens == 513
    assert len(request.all_token_ids) == 513


def test_recving_thread_does_not_requeue_failed_notification() -> None:
    thread = MooncakeSchedulerRecvingThread(threading.Event())
    thread.add_request("10.0.0.1", 6000, "failed")
    thread.add_request("10.0.0.2", 6000, "healthy")
    thread.request_queue.put(None)
    thread._send_done_recving = MagicMock(side_effect=[RuntimeError("unreachable"), None])

    thread.run()

    assert thread._send_done_recving.call_args_list == [
        call("10.0.0.1", 6000, "failed"),
        call("10.0.0.2", 6000, "healthy"),
    ]
    assert thread.request_queue.empty()
    assert thread.request_queue.unfinished_tasks == 0


def test_recving_thread_shutdown_wakes_idle_thread() -> None:
    thread = MooncakeSchedulerRecvingThread(threading.Event())
    thread.start()
    assert thread.ready_event.wait(timeout=5)

    thread.shutdown()
    thread.shutdown()

    assert not thread.is_alive()
    assert thread.request_queue.empty()
    assert thread.request_queue.unfinished_tasks == 0


def test_recving_thread_shutdown_closes_resources_in_owner_thread() -> None:
    thread = MooncakeSchedulerRecvingThread(threading.Event())
    socket = MagicMock()
    context = MagicMock()
    cleanup_threads = []
    socket.close.side_effect = lambda **kwargs: cleanup_threads.append(threading.get_ident())
    context.destroy.side_effect = lambda **kwargs: cleanup_threads.append(threading.get_ident())

    def send_done(*args):
        thread.remote_sockets["tcp://10.0.0.1:6000"].append(socket)
        thread.zmq_context = context
        thread.stop_event.set()

    thread._send_done_recving = MagicMock(side_effect=send_done)
    thread.add_request("10.0.0.1", 6000, "first")
    thread.add_request("10.0.0.1", 6000, "pending")
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive()
    thread.shutdown()

    socket.close.assert_called_once_with(linger=0)
    context.destroy.assert_called_once_with(linger=0)
    assert cleanup_threads == [thread.ident, thread.ident]
    assert not thread.remote_sockets
    assert thread.zmq_context is None
    assert thread.request_queue.empty()
    assert thread.request_queue.unfinished_tasks == 0
    assert thread._send_done_recving.call_count == 1


def test_sending_thread_shutdown_exits_receive_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    thread = make_sending_thread()
    socket = MagicMock()
    context = MagicMock()
    context.__enter__.return_value = socket
    monkeypatch.setattr(pull_scheduler, "zmq_ctx", MagicMock(return_value=context))

    def recv():
        thread.stop_event.wait(timeout=5)
        raise zmq.Again()  # type: ignore[attr-defined]

    socket.recv_multipart.side_effect = recv
    thread.add_delayed_request("pending")
    thread.start()
    assert thread.ready_event.wait(timeout=5)
    thread.shutdown()
    thread.shutdown()

    assert not thread.is_alive()
    context.__exit__.assert_called_once()
    assert not thread.delayed_free_requests
    assert thread.finished_requests.empty()


def test_scheduler_shutdown_stops_threads_and_clears_request_state() -> None:
    scheduler = make_pull_scheduler()
    scheduler._heartbeat_thread = MagicMock()
    scheduler._sending_thread = MagicMock()
    scheduler._recving_thread = MagicMock()
    scheduler._reqs_need_recv["recv"] = MagicMock()
    scheduler._reqs_need_send["send"] = time.time()
    scheduler._reqs_in_batch.add("recv")
    scheduler._reqs_recv_info["recv"] = ("10.0.0.1", 6000, "remote")
    connector = MooncakePullConnector.__new__(MooncakePullConnector)
    connector.connector_scheduler = scheduler

    connector.shutdown()

    scheduler._heartbeat_thread.shutdown.assert_called_once()
    scheduler._sending_thread.shutdown.assert_called_once()
    scheduler._recving_thread.shutdown.assert_called_once()
    assert not scheduler._reqs_need_recv
    assert not scheduler._reqs_need_send
    assert not scheduler._reqs_in_batch
    assert not scheduler._reqs_recv_info


def heartbeat_request(request_id: str = "request-d", **overrides):
    params = dict(
        do_remote_prefill=True,
        remote_engine_id="engine-p",
        remote_host="127.0.0.1",
        remote_port=6000,
        remote_request_id="request-p",
        remote_num_prompt_tokens=32,
        remote_block_ids=([20, 21],),
    )
    params.update(overrides)
    return make_request(request_id=request_id, kv_transfer_params=params)


def test_heartbeat_tracks_waiting_request_until_aggregated_receive() -> None:
    scheduler = make_pull_scheduler()
    heartbeat = MooncakeHeartbeatThread(5)
    scheduler._heartbeat_thread = heartbeat
    scheduler._recving_thread = MagicMock()
    request = heartbeat_request()
    scheduler.on_new_request(request)
    assert heartbeat.requests == {"request-d": ("127.0.0.1", 6000, "request-p")}
    assert not scheduler._reqs_need_recv  # No allocation is needed to register.
    scheduler.update_state_after_alloc(request, make_blocks(), 16)
    scheduler.build_connector_meta(MagicMock())
    assert "request-d" in heartbeat.requests
    scheduler.update_connector_output(SimpleNamespace(finished_recving={"request-d"}, failed_recving=set()))
    assert not heartbeat.requests
    scheduler._recving_thread.add_request.assert_called_once_with("127.0.0.1", 6000, "request-p")


@pytest.mark.parametrize("params", [{"do_remote_prefill": False}, {"remote_block_ids": ()}, {"remote_port": None}])
def test_heartbeat_ignores_requests_without_remote_kv(params) -> None:
    scheduler = make_pull_scheduler()
    scheduler._heartbeat_thread = MooncakeHeartbeatThread(5)
    request = heartbeat_request(**params)
    if params.get("remote_port", 1) is None:
        del request.kv_transfer_params["remote_port"]
    scheduler.on_new_request(request)
    assert not scheduler._heartbeat_thread.requests


def test_local_prefix_hit_stops_heartbeat() -> None:
    scheduler = make_pull_scheduler()
    scheduler._heartbeat_thread = MooncakeHeartbeatThread(5)
    scheduler._recving_thread = MagicMock()
    request = heartbeat_request()
    scheduler.on_new_request(request)
    scheduler.update_state_after_alloc(request, make_blocks(), 0)
    assert not scheduler._heartbeat_thread.requests
    scheduler._recving_thread.add_request.assert_called_once()


def test_heartbeat_registration_starts_at_scheduler_admission_and_is_idempotent() -> None:
    scheduler = make_pull_scheduler()
    scheduler._heartbeat_thread = MooncakeHeartbeatThread(5)
    request = heartbeat_request()
    assert not scheduler._heartbeat_thread.requests
    scheduler.on_new_request(request)
    assert scheduler._heartbeat_thread.requests == {"request-d": ("127.0.0.1", 6000, "request-p")}
    assert not scheduler._reqs_recv_info and not scheduler._reqs_need_recv
    snapshot = dict(scheduler._heartbeat_thread.requests)
    scheduler.on_new_request(request)
    assert scheduler._heartbeat_thread.requests == snapshot


@pytest.mark.parametrize("in_flight", [False, True])
def test_abort_only_releases_remote_kv_after_read_is_safe(in_flight: bool) -> None:
    scheduler = make_pull_scheduler()
    scheduler._heartbeat_thread = MooncakeHeartbeatThread(5)
    scheduler._recving_thread = MagicMock()
    request = heartbeat_request()
    scheduler.on_new_request(request)
    if in_flight:
        scheduler.update_state_after_alloc(request, make_blocks(), 16)
    request.status = RequestStatus.FINISHED_ABORTED
    scheduler.request_finished(request, ())
    assert bool(scheduler._heartbeat_thread.requests) == in_flight
    if in_flight:
        scheduler._recving_thread.add_request.assert_not_called()
    else:
        scheduler._recving_thread.add_request.assert_called_once()


def test_lease_renewal_preserves_initial_deadline_and_reaps_both_queues(monkeypatch) -> None:
    clock = MagicMock(return_value=0.0)
    monkeypatch.setattr(pull_scheduler.time, "monotonic", clock)
    thread = make_sending_thread()
    thread.add_delayed_request("older")  # initial expiry 30
    clock.return_value = 5.0
    thread.add_delayed_request("newer")  # initial expiry 35
    thread._handle_heartbeat(("older", "unknown"))
    assert thread.delayed_free_requests["older"] == 30.0
    clock.return_value = 15.0
    thread._handle_heartbeat(("older",))  # renewed expiry 35
    assert thread.renewed_requests["older"] == 35.0
    clock.return_value = 25.0
    thread._handle_heartbeat(("older",))  # renewed expiry 45
    clock.return_value = 35.0
    assert thread.get_and_clear_finished_requests() == {"newer"}
    assert thread.renewed_requests == {"older": 45.0}
    clock.return_value = 45.0
    thread._handle_heartbeat(("older",))  # expired even if not yet reaped
    assert thread.get_and_clear_finished_requests() == {"older"}
    thread._handle_heartbeat(("older",))
    assert not thread.renewed_requests and not thread.delayed_free_requests


def test_shorter_renewed_lease_is_not_blocked_by_new_initial_lease(monkeypatch) -> None:
    clock = MagicMock(return_value=0.0)
    monkeypatch.setattr(pull_scheduler.time, "monotonic", clock)
    thread = make_sending_thread()
    thread.add_delayed_request("renewed")
    clock.return_value = 15.0
    thread.add_delayed_request("initial")  # expiry 45
    thread._handle_heartbeat(("renewed",))  # expiry 35
    clock.return_value = 35.0
    assert thread.get_and_clear_finished_requests() == {"renewed"}
    assert "initial" in thread.delayed_free_requests


def test_done_and_late_heartbeat_complete_only_once(monkeypatch) -> None:
    clock = MagicMock(return_value=0.0)
    monkeypatch.setattr(pull_scheduler.time, "monotonic", clock)
    thread = make_sending_thread()
    thread.add_delayed_request("request")
    clock.return_value = 15.0
    thread._handle_heartbeat(("request",))
    thread._handle_finished_request("request")
    thread._handle_heartbeat(("request", "missing"))
    thread._handle_finished_request("request")
    assert thread.get_and_clear_finished_requests() == {"request"}
    assert thread.get_and_clear_finished_requests() == set()
    assert not thread.renewed_requests


def test_control_loop_accepts_heartbeat_without_completing_transfer(monkeypatch) -> None:
    clock = MagicMock(return_value=0.0)
    monkeypatch.setattr(pull_scheduler.time, "monotonic", clock)
    thread = make_sending_thread()
    thread.add_delayed_request("request")
    clock.return_value = 15.0
    sock = MagicMock()
    sock.recv_multipart.side_effect = [
        [b"decoder", b"", msgspec.msgpack.encode((HEARTBEAT_MSG, ("request",)))],
        KeyboardInterrupt(),
    ]
    with pytest.raises(KeyboardInterrupt):
        thread._run_busy_loop(sock)
    sock.send_multipart.assert_called_once_with((b"decoder", b"", ACK_MSG))
    assert thread.get_and_clear_finished_requests() == set()
    assert thread.renewed_requests["request"] == 35.0


def test_heartbeat_thread_renews_without_scheduler_steps_and_isolates_dead_peer() -> None:
    # Real control messages, without ever scheduling or running a model.
    stop = threading.Event()
    received = threading.Event()
    second = threading.Event()
    with zmq.Context() as context:  # type: ignore[attr-defined]
        router = context.socket(zmq.ROUTER)  # type: ignore[attr-defined]
        port = router.bind_to_random_port("tcp://127.0.0.1")
        router.setsockopt(zmq.RCVTIMEO, 100)  # type: ignore[attr-defined]
        seen = []

        def serve():
            try:
                while not stop.is_set():
                    try:
                        frames = router.recv_multipart()
                    except zmq.Again:  # type: ignore[attr-defined]
                        continue
                    message = msgspec.msgpack.decode(frames[-1])
                    seen.append(message)
                    router.send_multipart((frames[0], b"", ACK_MSG))
                    (received if len(seen) == 1 else second).set()
            finally:
                router.close(linger=0)

        server = threading.Thread(target=serve)
        server.start()
        heartbeat = MooncakeHeartbeatThread(0.1)
        heartbeat.add_request("dead", "127.0.0.1", 1, "dead-request")
        heartbeat.add_request("local", "127.0.0.1", port, "remote")
        heartbeat.start()
        try:
            assert received.wait(3)
            assert second.wait(3)
            assert all(message == [HEARTBEAT_MSG, ["remote"]] for message in seen)
            assert heartbeat.remove_request("local") == ("127.0.0.1", port, "remote")
        finally:
            heartbeat.shutdown()
            stop.set()
            server.join(3)
        assert not heartbeat.is_alive()
        assert not heartbeat.requests


@pytest.mark.parametrize("duration", [0, 5, -1, 6.0, True, "30"])
def test_invalid_lease_configuration_fails_before_starting_threads(monkeypatch, duration) -> None:
    config = SimpleNamespace(get_from_extra_config=lambda key, default: duration)
    monkeypatch.setattr(
        MooncakeBaseConnectorScheduler, "__init__", lambda self, *args: setattr(self, "kv_transfer_config", config)
    )
    with pytest.raises(ValueError, match="kv_lease_duration"):
        MooncakePullConnectorScheduler(MagicMock(), "engine", MagicMock())


def test_heartbeat_can_be_disabled_without_disabling_completion_notifications(monkeypatch) -> None:
    config = SimpleNamespace(
        is_kv_consumer=True,
        get_from_extra_config=lambda key, default: False if key == "enable_heartbeat" else default,
    )
    monkeypatch.setattr(
        MooncakeBaseConnectorScheduler, "__init__", lambda self, *args: setattr(self, "kv_transfer_config", config)
    )
    receiver = MagicMock()

    def create_receiver(ready):
        ready.set()
        return receiver

    monkeypatch.setattr(pull_scheduler, "MooncakeSchedulerRecvingThread", create_receiver)
    scheduler = MooncakePullConnectorScheduler(MagicMock(), "engine", MagicMock())
    assert scheduler._heartbeat_thread is None
    receiver.start.assert_called_once()
    scheduler.shutdown()


def test_heartbeat_batches_large_groups_and_reconnects_after_bad_ack(monkeypatch) -> None:
    monkeypatch.setattr(heartbeat_module, "HEARTBEAT_ACK_TIMEOUT", 0.05)
    stop = threading.Event()
    received_all = threading.Event()
    with zmq.Context() as context:  # type: ignore[attr-defined]
        router = context.socket(zmq.ROUTER)  # type: ignore[attr-defined]
        port = router.bind_to_random_port("tcp://127.0.0.1")
        router.setsockopt(zmq.RCVTIMEO, 100)  # type: ignore[attr-defined]
        seen = set()
        batch_sizes = []
        identities = set()

        def serve():
            first = True
            try:
                while not stop.is_set():
                    try:
                        frames = router.recv_multipart()
                    except zmq.Again:  # type: ignore[attr-defined]
                        continue
                    identities.add(frames[0])
                    _, ids = msgspec.msgpack.decode(frames[-1])
                    if not first:
                        seen.update(ids)
                        batch_sizes.append(len(ids))
                    router.send_multipart((frames[0], b"", b"bad-ACK" if first else ACK_MSG))
                    first = False
                    if len(seen) == 1025:
                        received_all.set()
            finally:
                router.close(linger=0)

        server = threading.Thread(target=serve)
        server.start()
        heartbeat = MooncakeHeartbeatThread(0.1)
        for index in range(1025):
            heartbeat.add_request(str(index), "127.0.0.1", port, "remote-" + str(index))
        heartbeat.start()
        try:
            assert received_all.wait(5)
            assert max(batch_sizes) <= 512
            assert len(identities) >= 2
        finally:
            heartbeat.shutdown()
            stop.set()
            server.join(3)


def test_slow_heartbeat_acks_do_not_starve_tail_or_retain_removed_requests() -> None:
    stop = threading.Event()
    received_all = threading.Event()
    received_new = threading.Event()
    with zmq.Context() as context:  # type: ignore[attr-defined]
        router = context.socket(zmq.ROUTER)  # type: ignore[attr-defined]
        port = router.bind_to_random_port("tcp://127.0.0.1")
        router.setsockopt(zmq.RCVTIMEO, 100)  # type: ignore[attr-defined]
        seen = set()
        batch_sizes = []
        new_snapshot_batches = []

        def serve():
            refreshed = False
            try:
                while not stop.is_set():
                    try:
                        frames = router.recv_multipart()
                    except zmq.Again:  # type: ignore[attr-defined]
                        continue
                    _, ids = msgspec.msgpack.decode(frames[-1])
                    seen.update(ids)
                    batch_sizes.append(len(ids))
                    if len(seen) >= 1025:
                        received_all.set()
                    if "new-remote" in ids:
                        refreshed = True
                        received_new.set()
                    if refreshed:
                        new_snapshot_batches.append(ids)
                    # Cross a refresh interval while each ACK is outstanding.
                    stop.wait(0.08)
                    router.send_multipart((frames[0], b"", ACK_MSG))
            finally:
                router.close(linger=0)

        server = threading.Thread(target=serve)
        server.start()
        heartbeat = MooncakeHeartbeatThread(0.05)
        for index in range(1025):
            heartbeat.add_request(str(index), "127.0.0.1", port, "remote-" + str(index))
        heartbeat.start()
        try:
            assert received_all.wait(3), "unsent tail must survive heartbeat refreshes"
            heartbeat.remove_request("0")
            heartbeat.add_request("new", "127.0.0.1", port, "new-remote")
            assert received_new.wait(3), "new requests must enter the next snapshot"
            assert max(batch_sizes) <= 512
            assert all("remote-0" not in ids for ids in new_snapshot_batches)
            assert heartbeat.is_alive()
        finally:
            heartbeat.shutdown()
            stop.set()
            server.join(3)
        assert not server.is_alive()


@pytest.mark.parametrize("failure_stage", ["create", "configure"])
def test_heartbeat_connection_failure_is_isolated_and_closes_partial_socket(monkeypatch, failure_stage) -> None:
    stop = threading.Event()
    received_twice = threading.Event()
    original_make_socket = heartbeat_module.make_zmq_socket
    failed_socket = MagicMock()
    failed_socket.setsockopt.side_effect = RuntimeError("socket configuration failed")

    def make_socket(context, path, *args, **kwargs):
        if path.endswith(":1"):
            if failure_stage == "create":
                raise ValueError("invalid heartbeat endpoint")
            return failed_socket
        return original_make_socket(context, path, *args, **kwargs)

    monkeypatch.setattr(heartbeat_module, "make_zmq_socket", make_socket)
    with zmq.Context() as context:  # type: ignore[attr-defined]
        router = context.socket(zmq.ROUTER)  # type: ignore[attr-defined]
        port = router.bind_to_random_port("tcp://127.0.0.1")
        router.setsockopt(zmq.RCVTIMEO, 100)  # type: ignore[attr-defined]
        seen = []

        def serve():
            try:
                while not stop.is_set():
                    try:
                        frames = router.recv_multipart()
                    except zmq.Again:  # type: ignore[attr-defined]
                        continue
                    seen.append(msgspec.msgpack.decode(frames[-1]))
                    router.send_multipart((frames[0], b"", ACK_MSG))
                    if len(seen) >= 2:
                        received_twice.set()
            finally:
                router.close(linger=0)

        server = threading.Thread(target=serve)
        server.start()
        heartbeat = MooncakeHeartbeatThread(0.05)
        heartbeat.add_request("bad", "127.0.0.1", 1, "bad-remote")
        heartbeat.add_request("good", "127.0.0.1", port, "good-remote")
        heartbeat.start()
        try:
            assert received_twice.wait(3), "a failed connection must not stop other producers"
            assert heartbeat.is_alive()
            assert all(message == [HEARTBEAT_MSG, ["good-remote"]] for message in seen)
            if failure_stage == "configure":
                failed_socket.close.assert_any_call(linger=0)
            else:
                failed_socket.close.assert_not_called()
        finally:
            heartbeat.shutdown()
            stop.set()
            server.join(3)
        assert not server.is_alive()
