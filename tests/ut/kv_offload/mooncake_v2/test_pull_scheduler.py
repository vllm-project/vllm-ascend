# SPDX-License-Identifier: Apache-2.0

import threading
import time
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import msgspec
import pytest
import torch
import zmq
from vllm.sampling_params import SamplingParams
from vllm.v1.request import Request, RequestStatus

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake import heartbeat, pull_scheduler
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.base_scheduler import (
    MooncakeBaseConnectorScheduler,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.connector import MooncakePullConnector
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.metadata import (
    MooncakeTransferMetadataGroups,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_scheduler import (
    ACK_MSG,
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
    scheduler._kv_lease_duration = heartbeat.DEFAULT_KV_LEASE_DURATION
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


def test_sending_thread_handles_early_and_normal_completion_once() -> None:
    thread = make_sending_thread()

    thread._handle_finished_request("early")
    assert thread.get_and_clear_finished_requests() == set()
    thread.add_delayed_request("early", time.monotonic())
    assert thread.get_and_clear_finished_requests() == {"early"}

    thread.add_delayed_request("normal", time.monotonic())
    thread._handle_finished_request("normal")
    thread._handle_finished_request("normal")
    assert thread.get_and_clear_finished_requests() == {"normal"}


def test_sending_thread_force_frees_expired_request(monkeypatch: pytest.MonkeyPatch) -> None:
    thread = make_sending_thread()
    thread.kv_lease_duration = 6
    monkeypatch.setattr(
        pull_scheduler.time,
        "monotonic",
        MagicMock(return_value=20.0),
    )
    thread.add_delayed_request("expired", 10.0)

    assert thread.get_and_clear_finished_requests() == {"expired"}
    assert not thread.delayed_free_requests


@pytest.mark.parametrize("attempts", [3, 5])
def test_recving_thread_reuses_socket_after_ack_and_discards_it_on_error(
    monkeypatch: pytest.MonkeyPatch,
    attempts: int,
) -> None:
    thread = MooncakeSchedulerRecvingThread(threading.Event(), max_attempts=attempts)
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
    assert send.call_args.kwargs == {"max_retries": attempts}
    assert recv.call_args.kwargs == {"max_retries": attempts}

    recv.return_value = b"not-ack"
    with pytest.raises(RuntimeError, match="Unexpected.*control response"):
        thread._send_done_recving("10.0.0.1", 6000, "request-p")

    socket.close.assert_called_once_with(linger=0)
    assert thread._return_remote_socket.call_count == 1


@pytest.mark.parametrize("timeout", [1000, 2500])
def test_recving_thread_socket_pool_creates_once_and_reuses_by_endpoint(
    monkeypatch: pytest.MonkeyPatch,
    timeout: int,
) -> None:
    thread = MooncakeSchedulerRecvingThread(threading.Event(), io_timeout_ms=timeout)
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
    assert socket.setsockopt.call_args_list == [call(zmq.SNDTIMEO, timeout), call(zmq.RCVTIMEO, timeout)]


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
    scheduler._reqs_need_send["request-p"] = time.monotonic()
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
    result.connector_scheduler._recving_thread = None
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


@pytest.mark.parametrize("duration", [0, 5, -1, True, "30", float("nan"), float("inf")])
def test_invalid_lease_configuration_is_rejected(duration):
    with pytest.raises(ValueError, match="kv_lease_duration"):
        pull_scheduler.validate_lease_duration(duration)


def test_heartbeat_renews_without_shortening_and_expiry_ignores_insertion_order(monkeypatch):
    thread = make_sending_thread()
    thread.kv_lease_duration = 30
    monkeypatch.setattr(pull_scheduler.time, "monotonic", lambda: 0.0)
    thread.add_delayed_request("renewed", 0.0)
    thread.add_delayed_request("expired", 1.0)
    assert thread._handle_heartbeat("engine-p", ["renewed", "unknown"])
    assert thread.delayed_free_requests["renewed"] == 30
    assert "unknown" not in thread.delayed_free_requests
    monkeypatch.setattr(pull_scheduler.time, "monotonic", lambda: 20.0)
    thread._handle_heartbeat("engine-p", ["renewed"])
    assert thread.delayed_free_requests["renewed"] == 40
    monkeypatch.setattr(pull_scheduler.time, "monotonic", lambda: 32.0)
    assert thread.get_and_clear_finished_requests() == {"expired"}
    thread._handle_heartbeat("engine-p", ["expired"])
    assert "expired" not in thread.delayed_free_requests
    thread._handle_finished_request("renewed")
    thread._handle_heartbeat("engine-p", ["renewed"])
    assert thread.get_and_clear_finished_requests() == {"renewed"}
    assert thread.get_and_clear_finished_requests() == set()


def test_heartbeat_cannot_revive_expired_lease_before_reaper_runs(monkeypatch):
    thread = make_sending_thread()
    thread.kv_lease_duration = 30
    thread.add_delayed_request("expired", 0.0)
    monkeypatch.setattr(pull_scheduler.time, "monotonic", lambda: 30.0)
    thread._handle_heartbeat("engine-p", ["expired"])
    assert thread.get_and_clear_finished_requests() == {"expired"}


@pytest.mark.parametrize("engine, ids", [("old-engine", ["req"]), ("engine-p", "req"), ("engine-p", [1])])
def test_heartbeat_rejects_wrong_engine_or_malformed_payload(engine, ids):
    thread = make_sending_thread()
    thread.kv_lease_duration = 30
    thread.add_delayed_request("req", time.monotonic())
    before = dict(thread.delayed_free_requests)
    assert not thread._handle_heartbeat(engine, ids)
    assert dict(thread.delayed_free_requests) == before


def lease_request(local_id="request-d", remote_id="request-p", **overrides):
    params = {
        "do_remote_prefill": True,
        "remote_block_ids": ([20],),
        "remote_engine_id": "engine-p",
        "remote_host": "10.0.0.1",
        "remote_port": 6000,
        "remote_request_id": remote_id,
        "kv_lease_version": 1,
        "kv_lease_duration": 30,
    }
    params.update(overrides)
    return make_request(request_id=local_id, kv_transfer_params=params)


def start_heartbeat(thread, local_id="d", remote_id="p", **overrides):
    request = lease_request(local_id, remote_id, **overrides)
    thread.start_request(request.request_id, request.kv_transfer_params)


def heartbeat_tick(thread):
    with patch.object(thread._wakeup, "wait"):
        thread._run_once()


def test_waiting_requests_heartbeat_without_scheduler_steps(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(pull_scheduler.time, "monotonic", lambda: now[0])
    scheduler = make_pull_scheduler()
    thread = heartbeat.MooncakeHeartbeatThread(threading.Event())
    scheduler._heartbeat_thread = thread
    thread._send_control = MagicMock()
    scheduler.on_new_request(lease_request("d1", "p1"))
    scheduler.on_new_request(lease_request("d2", "p2"))
    assert not scheduler._reqs_need_recv
    heartbeat_tick(thread)
    heartbeat_tick(thread)
    assert thread._send_control.call_count == 1
    for moment in (5.0, 10.0, 30.0):
        now[0] = moment
        heartbeat_tick(thread)
        thread._send_control.assert_called_with(
            "10.0.0.1", 6000, (pull_scheduler.HEARTBEAT_MSG, "engine-p", ("p1", "p2"))
        )
    assert thread._send_control.call_count == 4


def test_legacy_request_does_not_track_heartbeats():
    scheduler = make_pull_scheduler()
    scheduler._heartbeat_thread = heartbeat.MooncakeHeartbeatThread(threading.Event())
    scheduler.on_new_request(lease_request(kv_lease_version=None))
    assert not scheduler._heartbeat_thread.requests


def test_producer_advertises_opt_in_lease(monkeypatch):
    scheduler = make_pull_scheduler()
    scheduler._sending_thread = MagicMock()
    scheduler._kv_lease_duration = 30
    monkeypatch.setattr(pull_scheduler.time, "monotonic", lambda: 123.0)
    request = make_request(
        request_id="p",
        status=RequestStatus.FINISHED_LENGTH_CAPPED,
        kv_transfer_params={"do_remote_decode": True},
        output_token_ids=[1],
    )
    delayed, params = scheduler.request_finished(request, ([10, 11, 12],))
    assert delayed and params["kv_lease_duration"] == 30 and params["kv_lease_version"] == 1
    scheduler._sending_thread.add_delayed_request.assert_called_once_with("p", 123.0)


def test_control_channel_dispatches_heartbeat_and_ack():
    thread = make_sending_thread()
    thread.kv_lease_duration = 30
    thread._handle_heartbeat = MagicMock(return_value=True)
    socket = MagicMock()
    socket.recv_multipart.side_effect = [
        [b"decoder", b"", msgspec.msgpack.encode((pull_scheduler.HEARTBEAT_MSG, "engine-p", ["p"]))],
        _StopLoop(),
    ]
    with pytest.raises(_StopLoop):
        thread._run_busy_loop(socket)
    thread._handle_heartbeat.assert_called_once_with("engine-p", ["p"])
    socket.send_multipart.assert_called_once_with((b"decoder", b"", ACK_MSG))


def test_legacy_producer_rejects_heartbeat_without_changing_deadline():
    thread = make_sending_thread()
    thread.add_delayed_request("p", time.monotonic())
    before = dict(thread.delayed_free_requests)
    assert not thread._handle_heartbeat("engine-p", ["p"])
    assert dict(thread.delayed_free_requests) == before


def test_same_engine_cannot_change_endpoint_or_lease_duration():
    tracker = heartbeat.MooncakeHeartbeatThread(threading.Event())
    start_heartbeat(tracker, "d1", "p1")
    with pytest.raises(ValueError, match="Inconsistent"):
        start_heartbeat(tracker, "d2", "p2", remote_port=7000)
    with pytest.raises(ValueError, match="Inconsistent"):
        start_heartbeat(tracker, "d3", "p3", kv_lease_duration=60)
    assert set(tracker.requests) == {"d1"}


def test_stopping_one_local_request_preserves_shared_remote_request(monkeypatch):
    monkeypatch.setattr(pull_scheduler.time, "monotonic", lambda: 0.0)
    tracker = heartbeat.MooncakeHeartbeatThread(threading.Event())
    start_heartbeat(tracker, "d1", "p")
    start_heartbeat(tracker, "d2", "p")
    tracker.stop_request("d1")
    snapshot, _ = tracker._next_heartbeat()
    assert snapshot is not None and snapshot[3] == ("p",)
    tracker.stop_request("d2")
    tracker.stop_request("d2")
    assert not tracker.requests and not tracker.last_sent
    assert tracker._next_heartbeat() == (None, None)


def test_duplicate_delayed_registration_cannot_resurrect_finished_request():
    thread = make_sending_thread()
    thread.add_delayed_request("p", time.monotonic())
    thread._handle_finished_request("p")
    assert thread.get_and_clear_finished_requests() == {"p"}
    thread.add_delayed_request("p", time.monotonic())
    assert not thread.delayed_free_requests
    assert thread.get_and_clear_finished_requests() == set()


def test_stop_is_processed_before_due_heartbeat(monkeypatch):
    monkeypatch.setattr(pull_scheduler.time, "monotonic", lambda: 10.0)
    thread = heartbeat.MooncakeHeartbeatThread(threading.Event())
    thread._send_control = MagicMock()
    start_heartbeat(thread)
    thread.stop_request("d")
    heartbeat_tick(thread)
    thread._send_control.assert_not_called()


def test_next_heartbeat_rebuilds_membership_after_failed_snapshot(monkeypatch):
    now = [10.0]
    monkeypatch.setattr(pull_scheduler.time, "monotonic", lambda: now[0])
    thread = heartbeat.MooncakeHeartbeatThread(threading.Event())
    thread._send_control = MagicMock(side_effect=[RuntimeError("timeout"), None])
    start_heartbeat(thread, "d1", "p1")
    start_heartbeat(thread, "d2", "p2")
    heartbeat_tick(thread)
    thread._send_control.assert_called_with("10.0.0.1", 6000, (pull_scheduler.HEARTBEAT_MSG, "engine-p", ("p1", "p2")))
    thread.stop_request("d1")
    start_heartbeat(thread, "d3", "p3")
    heartbeat_tick(thread)
    heartbeat_tick(thread)
    assert thread._send_control.call_count == 1
    now[0] = 15.0
    heartbeat_tick(thread)
    thread._send_control.assert_called_with("10.0.0.1", 6000, (pull_scheduler.HEARTBEAT_MSG, "engine-p", ("p2", "p3")))
    assert thread._send_control.call_count == 2


@pytest.mark.parametrize("cleanup", ["finished", "aborted", "zero_external"])
def test_separate_threads_stop_renewal_and_ack_only_ended_reads(monkeypatch, cleanup):
    monkeypatch.setattr(pull_scheduler.time, "monotonic", lambda: 0.0)
    scheduler = make_pull_scheduler()
    hb = heartbeat.MooncakeHeartbeatThread(threading.Event())
    ack = MooncakeSchedulerRecvingThread(threading.Event())
    scheduler._heartbeat_thread, scheduler._recving_thread = hb, ack
    hb._send_control = MagicMock()
    ack._send_done_recving = MagicMock()
    request = lease_request()
    scheduler.on_new_request(request)
    hb._run_once()
    if cleanup == "finished":
        scheduler._reqs_recv_info[request.request_id] = ("10.0.0.1", 6000, "request-p")
        scheduler.update_connector_output(
            SimpleNamespace(
                finished_recving={request.request_id},
                finished_sending=None,
            )
        )
    elif cleanup == "aborted":
        scheduler.request_finished(request, ())
    else:
        scheduler.update_state_after_alloc(request, make_blocks(), 0)
    assert not hb.requests
    if cleanup == "aborted":
        assert ack.request_queue.empty()
    else:
        ack._process_next_request()
        ack._send_done_recving.assert_called_once_with("10.0.0.1", 6000, "request-p")


def test_blocked_heartbeat_does_not_block_done(monkeypatch):
    hb = heartbeat.MooncakeHeartbeatThread(threading.Event())
    ack = MooncakeSchedulerRecvingThread(threading.Event())
    entered, release = threading.Event(), threading.Event()

    def blocked_send(*args):
        entered.set()
        assert release.wait(timeout=5)

    hb._send_control = blocked_send
    start_heartbeat(hb)
    runner = threading.Thread(target=hb._run_once)
    runner.start()
    try:
        assert entered.wait(timeout=5)
        hb.stop_request("d")
        start_heartbeat(hb, "new-d", "new-p")
        assert set(hb.requests) == {"new-d"}
        ack._send_done_recving = MagicMock(side_effect=RuntimeError("P unreachable"))
        ack.add_request("host", 6000, "p")
        ack._process_next_request()
        assert ack.request_queue.empty()
        ack._send_done_recving.assert_called_once()
    finally:
        release.set()
        runner.join(timeout=5)
    assert not runner.is_alive()


@pytest.mark.parametrize("timeout", [1000, 1800])
def test_heartbeat_network_uses_one_attempt_and_own_socket(monkeypatch, timeout):
    hb = heartbeat.MooncakeHeartbeatThread(threading.Event(), io_timeout_ms=timeout)
    sock = MagicMock()
    context = MagicMock()
    context.__enter__.return_value = sock
    monkeypatch.setattr(heartbeat, "zmq_ctx", MagicMock(return_value=context))
    send, recv = MagicMock(), MagicMock(side_effect=RuntimeError("timeout"))
    monkeypatch.setattr(heartbeat, "ensure_zmq_send", send)
    monkeypatch.setattr(heartbeat, "ensure_zmq_recv", recv)
    with pytest.raises(RuntimeError, match="timeout"):
        hb._send_control("host", 6000, (heartbeat.HEARTBEAT_MSG, "engine", ("p",)))
    assert send.call_args.kwargs == {"max_retries": 1}
    assert recv.call_args.kwargs == {"max_retries": 1}
    context.__exit__.assert_called_once()
    assert sock.setsockopt.call_args_list == [call(zmq.SNDTIMEO, timeout), call(zmq.RCVTIMEO, timeout)]


@pytest.mark.parametrize("active, expected", [(False, None), (True, 5.0)])
def test_heartbeat_event_waits_for_deadline_or_membership_change(monkeypatch, active, expected):
    monkeypatch.setattr(heartbeat.time, "monotonic", lambda: 10.0)
    thread = heartbeat.MooncakeHeartbeatThread(threading.Event())
    thread._send_control = MagicMock()
    if active:
        start_heartbeat(thread)
        thread._run_once()
    with patch.object(thread._wakeup, "wait") as wait:
        thread._run_once()
    wait.assert_called_once_with(expected)


def test_new_request_between_snapshot_and_wait_keeps_wakeup():
    thread = heartbeat.MooncakeHeartbeatThread(threading.Event())
    original_wait = thread._wakeup.wait

    def add_before_wait(timeout):
        assert timeout is None
        start_heartbeat(thread)
        assert original_wait(0)  # START must not be cleared after the snapshot.

    with patch.object(thread._wakeup, "wait", side_effect=add_before_wait):
        thread._run_once()
    thread._send_control = MagicMock()
    thread._run_once()
    thread._send_control.assert_called_once()


def test_request_tracking_copies_params_before_scheduler_mutation():
    thread = heartbeat.MooncakeHeartbeatThread(threading.Event())
    request = lease_request()
    thread.start_request(request.request_id, request.kv_transfer_params)
    request.kv_transfer_params["do_remote_prefill"] = False
    request.kv_transfer_params["remote_request_id"] = "changed"
    assert thread.requests["request-d"][3] == "request-p"


@pytest.mark.parametrize(
    "extra, expected",
    [
        ({}, (1000, 3, 1000, 3)),
        (
            {
                "control_io_timeout_ms": 2500,
                "done_max_attempts": 5,
                "lease_io_timeout_ms": 1800,
                "heartbeat_max_attempts": 5,
            },
            (2500, 5, 1800, 5),
        ),
    ],
)
def test_scheduler_passes_extra_config_to_threads(monkeypatch, extra, expected):
    config = SimpleNamespace(is_kv_consumer=True, get_from_extra_config=lambda key, default: extra.get(key, default))
    monkeypatch.setattr(
        MooncakeBaseConnectorScheduler, "__init__", lambda self, *args: setattr(self, "kv_transfer_config", config)
    )

    def ready_thread(event, **kwargs):
        event.set()
        return MagicMock()

    done_cls = MagicMock(side_effect=ready_thread)
    heartbeat_cls = MagicMock(side_effect=ready_thread)
    monkeypatch.setattr(pull_scheduler, "MooncakeSchedulerRecvingThread", done_cls)
    monkeypatch.setattr(pull_scheduler, "MooncakeHeartbeatThread", heartbeat_cls)
    MooncakePullConnectorScheduler(MagicMock(), "d", MagicMock())
    assert done_cls.call_args.kwargs == {"io_timeout_ms": expected[0], "max_attempts": expected[1]}
    assert heartbeat_cls.call_args.kwargs == {"io_timeout_ms": expected[2], "max_attempts": expected[3]}


@pytest.mark.parametrize(
    "field",
    [
        "control_io_timeout_ms",
        "done_max_attempts",
        "lease_io_timeout_ms",
        "heartbeat_max_attempts",
        "kv_lease_duration",
    ],
)
@pytest.mark.parametrize("value", [0, -1, True, "1000", 1.5, None])
def test_invalid_control_config_rejected_before_threads_start(monkeypatch, field, value):
    extra = {field: value}
    config = SimpleNamespace(is_kv_consumer=True, get_from_extra_config=lambda key, default: extra.get(key, default))
    monkeypatch.setattr(
        MooncakeBaseConnectorScheduler, "__init__", lambda self, *args: setattr(self, "kv_transfer_config", config)
    )
    done_cls, heartbeat_cls = MagicMock(), MagicMock()
    monkeypatch.setattr(pull_scheduler, "MooncakeSchedulerRecvingThread", done_cls)
    monkeypatch.setattr(pull_scheduler, "MooncakeHeartbeatThread", heartbeat_cls)
    with pytest.raises(ValueError, match=field):
        MooncakePullConnectorScheduler(MagicMock(), "d", MagicMock())
    done_cls.assert_not_called()
    heartbeat_cls.assert_not_called()


@pytest.mark.parametrize("extra, expected", [({}, 480.0), ({"kv_lease_duration": 60}, 60.0)])
def test_default_lease_and_explicit_override(monkeypatch, extra, expected):
    config = SimpleNamespace(is_kv_consumer=False, get_from_extra_config=lambda key, default: extra.get(key, default))
    monkeypatch.setattr(
        MooncakeBaseConnectorScheduler, "__init__", lambda self, *args: setattr(self, "kv_transfer_config", config)
    )
    scheduler = MooncakePullConnectorScheduler(MagicMock(), "p", MagicMock())
    assert scheduler._kv_lease_duration == expected


def test_default_lease_heartbeat_interval_and_renewal(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(heartbeat.time, "monotonic", lambda: now[0])
    thread = heartbeat.MooncakeHeartbeatThread(threading.Event())
    thread._send_control = MagicMock(side_effect=RuntimeError("unreachable"))
    start_heartbeat(thread, kv_lease_duration=heartbeat.DEFAULT_KV_LEASE_DURATION)
    heartbeat_tick(thread)
    assert thread._next_heartbeat() == (None, 80.0)
    for timestamp in (80.0, 160.0, 240.0, 480.0):
        now[0] = timestamp
        heartbeat_tick(thread)
    assert thread._send_control.call_count == 3
    assert not thread.requests
    assert not thread._failures
    assert not thread.last_sent
    deadlines = {"p": 500.0}
    assert heartbeat.renew_heartbeat_leases(deadlines, ["p"], heartbeat.DEFAULT_KV_LEASE_DURATION)
    assert deadlines["p"] == 800.0


def test_heartbeat_success_resets_consecutive_failures(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(heartbeat.time, "monotonic", lambda: now[0])
    thread = heartbeat.MooncakeHeartbeatThread(threading.Event())
    thread._send_control = MagicMock(
        side_effect=[RuntimeError(), RuntimeError(), None, RuntimeError(), RuntimeError(), RuntimeError()]
    )
    start_heartbeat(thread)
    for timestamp in (0.0, 5.0, 10.0, 15.0, 20.0):
        now[0] = timestamp
        heartbeat_tick(thread)
        assert "d" in thread.requests
        if timestamp == 10.0:
            assert not thread._failures
    now[0] = 25.0
    heartbeat_tick(thread)
    assert not thread.requests
    assert not thread._failures
    assert not thread.last_sent


@pytest.mark.parametrize("replace_request", [False, True])
def test_heartbeat_failure_does_not_charge_new_membership(monkeypatch, replace_request):
    now = [0.0]
    monkeypatch.setattr(heartbeat.time, "monotonic", lambda: now[0])
    thread = heartbeat.MooncakeHeartbeatThread(threading.Event())
    start_heartbeat(thread)
    start_heartbeat(thread, local_id="other", remote_engine_id="other-engine")
    thread._send_control = MagicMock(side_effect=RuntimeError())
    heartbeat_tick(thread)
    # Keep the other engine out of this test's next heartbeat round.
    thread.last_sent["other-engine"] = 5.0

    def fail_after_registering(*args):
        if replace_request:
            thread.stop_request("d")
        start_heartbeat(thread, local_id="d" if replace_request else "new")
        raise RuntimeError("unreachable")

    thread._send_control.side_effect = fail_after_registering
    now[0] = 5.0
    heartbeat_tick(thread)
    assert thread._failures == ({} if replace_request else {"d": 2})
    assert "other" in thread.requests
    assert ("d" if replace_request else "new") in thread.requests


@pytest.mark.parametrize("attempts", [1, 3, 5])
def test_configured_heartbeat_attempt_limit(monkeypatch, attempts):
    now = [0.0]
    monkeypatch.setattr(heartbeat.time, "monotonic", lambda: now[0])
    thread = heartbeat.MooncakeHeartbeatThread(threading.Event(), max_attempts=attempts)
    thread._send_control = MagicMock(side_effect=RuntimeError("unreachable"))
    start_heartbeat(thread)
    for attempt in range(attempts):
        now[0] = attempt * 5.0
        heartbeat_tick(thread)
        assert ("d" in thread.requests) == (attempt + 1 < attempts)
    now[0] += 5.0
    heartbeat_tick(thread)
    assert thread._send_control.call_count == attempts
    assert not thread._failures
    assert not thread.last_sent
