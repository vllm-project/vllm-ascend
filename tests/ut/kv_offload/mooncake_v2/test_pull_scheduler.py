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
from vllm.v1.request import RequestStatus

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.base_scheduler import (
    MooncakeBaseConnectorScheduler,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.metadata import (
    MooncakeTransferMetadataGroups,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_scheduler import (
    ACK_MSG,
    MooncakePullConnectorScheduler,
    MooncakeSchedulerRecvingThread,
    MooncakeSchedulerSendingThread,
)

from .helpers import make_blocks, make_full_spec, make_mamba_spec, make_request, make_transfer_metadata


def make_sending_thread(
    metadata: dict[int | tuple[int, int], object] | None = None,
    *,
    tp_size: int = 1,
    pp_size: int = 1,
    dcp_size: int = 1,
) -> MooncakeSchedulerSendingThread:
    return MooncakeSchedulerSendingThread(
        host="127.0.0.1",
        port=6000,
        engine_id="engine-p",
        metadata=metadata or {0: make_transfer_metadata()},  # type: ignore[arg-type]
        tp_size=tp_size,
        pp_size=pp_size,
        pcp_size=1,
        dcp_size=dcp_size,
        ready_event=threading.Event(),
    )


def make_pull_scheduler() -> MooncakePullConnectorScheduler:
    scheduler = MooncakePullConnectorScheduler.__new__(MooncakePullConnectorScheduler)
    scheduler.need_truncate = False
    scheduler.num_speculative_tokens = 0
    scheduler.pcp_size = 1
    scheduler.dcp_size = 1
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
    assert pp_metadata.metadata_by_tp_rank[0].kv_caches_base_addr == [[1000]]
    assert pp_metadata.metadata_by_tp_rank[1].kv_caches_base_addr == [[2000]]
    assert pp_metadata.metadata_by_tp_rank[1].te_rpc_port == 9001


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

    thread = make_sending_thread({0: tp0, 1: tp1}, tp_size=2)
    decoded = msgspec.msgpack.decode(thread.encoded_metadata, type=MooncakeTransferMetadataGroups)

    assert decoded.use_kv_pp is True
    pp_metadata = decoded.metadata_by_pp_rank[0]
    assert pp_metadata.layer_names == ["layer.0", "layer.1", "layer.2"]
    assert pp_metadata.layer_block_sizes == [16, 16, 16]
    assert pp_metadata.metadata_by_tp_rank[0].layer_indices == [0, 2]
    assert pp_metadata.metadata_by_tp_rank[0].kv_caches_base_addr == [[1000], [], [3000]]
    assert pp_metadata.metadata_by_tp_rank[1].layer_indices == [1, 2]
    assert pp_metadata.metadata_by_tp_rank[1].kv_caches_base_addr == [[], [2000], [4000]]


def test_sending_thread_rejects_layer_split_with_dcp() -> None:
    tp0 = make_transfer_metadata(layer_names=["layer.0"], base_addrs=[[1000]])
    tp1 = make_transfer_metadata(
        te_rpc_port=9001,
        local_ip="10.0.0.2",
        layer_names=["layer.1"],
        base_addrs=[[2000]],
    )

    with pytest.raises(ValueError, match="LayerSplit cannot be combined with DCP"):
        make_sending_thread({0: tp0, 1: tp1}, tp_size=2, dcp_size=2)


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
    assert pp_metadata.metadata_by_tp_rank[0].layer_indices == [0, 1]
    assert pp_metadata.metadata_by_tp_rank[1].layer_indices == [0, 1]
    assert pp_metadata.metadata_by_tp_rank[1].kv_caches_base_addr == [
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


def test_sending_thread_rejects_incomplete_or_inconsistent_workers() -> None:
    metadata = make_transfer_metadata()
    with pytest.raises(ValueError, match="incomplete TP ranks"):
        make_sending_thread({0: metadata}, tp_size=2)

    mismatched = replace(metadata, block_lens=[[256]], te_rpc_port=9001)
    with pytest.raises(ValueError, match="mismatch in.*block_lens"):
        make_sending_thread({0: metadata, 1: mismatched}, tp_size=2)


def test_sending_thread_handles_early_and_normal_completion_once() -> None:
    thread = make_sending_thread()

    thread._handle_finished_request("early")
    assert thread.get_and_clear_finished_requests() == set()
    thread.add_delayed_request("early", time.time())
    assert thread.get_and_clear_finished_requests() == {"early"}

    thread.add_delayed_request("normal", time.time())
    thread._handle_finished_request("normal")
    thread._handle_finished_request("normal")
    assert thread.get_and_clear_finished_requests() == {"normal"}


def test_sending_thread_force_frees_expired_request(monkeypatch: pytest.MonkeyPatch) -> None:
    thread = make_sending_thread()
    thread.add_delayed_request("expired", 10.0)
    monkeypatch.setattr(
        "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_scheduler.envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT",
        5,
    )
    monkeypatch.setattr(
        "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_scheduler.time.time",
        MagicMock(return_value=20.0),
    )

    assert thread.get_and_clear_finished_requests() == {"expired"}
    assert not thread.delayed_free_requests


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
        "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_scheduler.ensure_zmq_send",
        send,
    )
    monkeypatch.setattr(
        "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_scheduler.ensure_zmq_recv",
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
        "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_scheduler.zmq.Context",
        context_cls,
    )
    monkeypatch.setattr(
        "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_scheduler.make_zmq_socket",
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
    scheduler.group_unique_specs = [[make_full_spec()], [make_mamba_spec()]]

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


def test_update_connector_output_routes_worker_completion_and_scheduler_ack() -> None:
    scheduler = make_pull_scheduler()
    scheduler._recving_thread = MagicMock()
    scheduler._sending_thread = MagicMock()
    scheduler._sending_thread.get_and_clear_finished_requests.return_value = {"request-p"}
    scheduler._reqs_recv_info["request-d"] = ("10.0.0.1", 6000, "request-p")
    scheduler._reqs_need_send["request-p"] = time.time()
    output = SimpleNamespace(finished_recving={"request-d"}, finished_sending=None)

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


def test_recving_run_requeues_failed_completion() -> None:
    ready_event = threading.Event()
    thread = MooncakeSchedulerRecvingThread(ready_event)
    request = ("10.0.0.1", 6000, "request-p")
    request_queue = MagicMock()
    request_queue.get.side_effect = [request, _StopLoop()]
    thread.request_queue = request_queue
    thread._send_done_recving = MagicMock(side_effect=RuntimeError("network error"))  # type: ignore[method-assign]

    with pytest.raises(_StopLoop):
        thread.run()

    assert ready_event.is_set()
    request_queue.put.assert_called_once_with(request)
    request_queue.task_done.assert_called_once_with()


def test_set_worker_metadata_starts_only_one_producer_sending_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = make_pull_scheduler()
    scheduler.kv_transfer_config = SimpleNamespace(is_kv_producer=True)
    scheduler.side_channel_host = "127.0.0.1"
    scheduler.side_channel_port = 6000
    scheduler.tp_size = 1
    scheduler.pp_size = 1
    scheduler.pcp_size = 1
    scheduler.dcp_size = 1
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
        "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_scheduler.MooncakeSchedulerSendingThread",
        thread_cls,
    )
    metadata = {0: make_transfer_metadata()}

    scheduler.set_xfer_handshake_metadata_from_workers(metadata)
    scheduler.set_xfer_handshake_metadata_from_workers(metadata)

    thread_cls.assert_called_once()
    created[0].start.assert_called_once_with()


def test_set_worker_metadata_is_ignored_on_consumer() -> None:
    scheduler = make_pull_scheduler()
    scheduler.kv_transfer_config = SimpleNamespace(is_kv_producer=False)

    scheduler.set_xfer_handshake_metadata_from_workers({0: make_transfer_metadata()})

    assert scheduler._sending_thread is None
