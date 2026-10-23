# SPDX-License-Identifier: Apache-2.0

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import msgspec
import pytest
import torch
import zmq
from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.sampling_params import SamplingParams
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request, RequestStatus

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake import pull_scheduler
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.base_scheduler import (
    MooncakeBaseConnectorScheduler,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.connector import MooncakePullConnector
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
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_worker import MooncakePullRecvingThread

from .helpers import (
    make_blocks,
    make_circular_spec,
    make_full_spec,
    make_mamba_spec,
    make_metadata_groups,
    make_pp_metadata,
    make_request,
    make_transfer_metadata,
)
from .test_pull_worker import make_req_meta, make_thread


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

    nonce = thread.add_delayed_request("normal", time.time())
    thread._handle_finished_request("normal", nonce)
    thread._handle_finished_request("normal", nonce)
    assert thread.get_and_clear_finished_requests() == {"normal"}
    assert not thread.delayed_free_requests

    thread._handle_finished_request("normal", nonce)
    assert thread.get_and_clear_finished_requests() == set()


def test_sending_thread_ignores_unknown_completion() -> None:
    thread = make_sending_thread()

    thread._handle_finished_request("unknown")
    assert thread.get_and_clear_finished_requests() == set()
    nonce = thread.add_delayed_request("unknown", time.time())
    assert thread.get_and_clear_finished_requests() == set()
    thread._handle_finished_request("unknown", nonce)
    assert thread.get_and_clear_finished_requests() == {"unknown"}


def test_sending_thread_acks_duplicate_and_unknown_completion() -> None:
    thread = make_sending_thread()
    nonce = thread.add_delayed_request("normal", time.time())
    sock = MagicMock()
    sock.recv_multipart.side_effect = [
        [b"consumer", b"", msgspec.msgpack.encode((DONE_RECVING_MSG, request_id, nonce))]
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
        nonce = thread.add_delayed_request(request_id, time.time())
        thread._handle_finished_request(request_id, nonce)
        assert thread.get_and_clear_finished_requests() == {request_id}
        thread._handle_finished_request(request_id, nonce)

    assert thread.get_and_clear_finished_requests() == set()
    assert not thread.delayed_free_requests
    assert thread.finished_requests.empty()
    assert not getattr(thread, "finished_request_ids", set())
    assert not getattr(thread, "early_finished_requests", set())


def test_sending_thread_force_frees_expired_request(monkeypatch: pytest.MonkeyPatch) -> None:
    thread = make_sending_thread()
    thread.add_delayed_request("expired", 10.0)
    monkeypatch.setattr(
        pull_scheduler.envs,
        "VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT",
        5,
    )
    monkeypatch.setattr(
        pull_scheduler.time,
        "time",
        MagicMock(return_value=20.0),
    )

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
            "remote_lease_nonce": "hold-1",
        }
    )

    scheduler.update_state_after_alloc(request, make_blocks(), num_external_tokens=0)

    assert scheduler._reqs_need_recv == {}
    scheduler._recving_thread.add_request.assert_called_once_with("10.0.0.1", 6000, "request-p", "hold-1")


def test_request_finished_delays_blocks_and_builds_remote_params() -> None:
    scheduler = make_pull_scheduler()
    scheduler._sending_thread = make_sending_thread()
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
    nonce = params["remote_lease_nonce"]
    assert nonce and scheduler._sending_thread._verify_leases((("request-p", nonce),)) == [True]
    consumer = make_pull_scheduler()
    consumer._recving_thread = MagicMock()
    receiving = make_request(request_id="request-d", kv_transfer_params=params)
    consumer.update_state_after_alloc(receiving, make_blocks(), num_external_tokens=16)
    metadata = consumer.build_connector_meta(MagicMock())
    assert metadata.requests["request-d"].remote_lease_nonce == nonce
    consumer.update_connector_output(KVConnectorOutput(finished_recving={"request-d"}))
    consumer._recving_thread.add_request.assert_called_once_with(
        params["remote_host"], params["remote_port"], "request-p", nonce
    )


@pytest.mark.parametrize("failed", [False, True])
def test_update_connector_output_routes_worker_completion_and_scheduler_ack(failed: bool) -> None:
    scheduler = make_pull_scheduler()
    scheduler.kv_transfer_config = SimpleNamespace(kv_load_failure_policy="fail")
    scheduler._recving_thread = MagicMock()
    scheduler._sending_thread = MagicMock()
    scheduler._sending_thread.get_and_clear_finished_requests.return_value = {"request-p"}
    scheduler._reqs_recv_info["request-d"] = ("10.0.0.1", 6000, "request-p", None)
    scheduler._reqs_need_send["request-p"] = time.time()
    output = SimpleNamespace(
        finished_recving={"request-d"},
        finished_sending=None,
        failed_recving={"request-d"} if failed else set(),
    )

    scheduler.update_connector_output(output)  # type: ignore[arg-type]

    scheduler._recving_thread.add_request.assert_called_once_with("10.0.0.1", 6000, "request-p", None)
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
        [b"worker", b"", encoder.encode((b"done_recving_msg", "request-p", "nonce"))],
        _StopLoop(),
    ]

    with pytest.raises(_StopLoop):
        thread._run_busy_loop(socket)

    assert socket.send_multipart.call_args_list == [
        call((b"worker", b"", thread.encoded_metadata)),
        call((b"worker", b"", ACK_MSG)),
    ]
    thread._handle_finished_request.assert_called_once_with("request-p", "nonce")


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
        call("10.0.0.1", 6000, "failed", None),
        call("10.0.0.2", 6000, "healthy", None),
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
    thread.add_delayed_request("pending", time.time())
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
    scheduler._sending_thread = MagicMock()
    scheduler._recving_thread = MagicMock()
    scheduler._reqs_need_recv["recv"] = MagicMock()
    scheduler._reqs_need_send["send"] = time.time()
    scheduler._reqs_in_batch.add("recv")
    scheduler._reqs_recv_info["recv"] = ("10.0.0.1", 6000, "remote")
    connector = MooncakePullConnector.__new__(MooncakePullConnector)
    connector.connector_scheduler = scheduler

    connector.shutdown()

    scheduler._sending_thread.shutdown.assert_called_once()
    scheduler._recving_thread.shutdown.assert_called_once()
    assert not scheduler._reqs_need_recv
    assert not scheduler._reqs_need_send
    assert not scheduler._reqs_in_batch
    assert not scheduler._reqs_recv_info


def test_expired_lease_is_invalid_before_finished_sending_is_consumed() -> None:
    thread = make_sending_thread()
    nonce = thread.add_delayed_request("request", 0.0)
    assert thread._verify_leases((("request", nonce),)) == [True]
    with thread.state_lock:
        thread._retrieve_expired_requests_locked()
    assert not thread.finished_requests.empty()
    assert thread._verify_leases((("request", nonce),)) == [False]


def test_stale_done_cannot_release_a_reused_request_id() -> None:
    thread = make_sending_thread()
    first = thread.add_delayed_request("request", 0.0)
    assert thread.get_and_clear_finished_requests() == {"request"}
    second = thread.add_delayed_request("request", time.time())
    assert first != second
    thread._handle_finished_request("request", first)
    thread._handle_finished_request("request")
    assert thread.get_and_clear_finished_requests() == set()
    assert thread._verify_leases((("request", first), ("request", second))) == [False, True]
    thread._handle_finished_request("request", second)
    thread._handle_finished_request("request", second)
    assert thread.get_and_clear_finished_requests() == {"request"}
    assert not thread.lease_nonces


@pytest.fixture
def lease_control_server(request):
    """Exercise the actual control protocol over loopback, without NPU IO."""
    producer = make_sending_thread()
    port_ready = threading.Event()
    ports = []
    host = getattr(request, "param", "127.0.0.1")

    def serve():
        with zmq.Context() as context, context.socket(zmq.ROUTER) as socket:  # type: ignore[attr-defined]
            socket.setsockopt(zmq.RCVTIMEO, 50)  # type: ignore[attr-defined]
            if host == "::1":
                socket.setsockopt(zmq.IPV6, 1)  # type: ignore[attr-defined]
            ports.append(socket.bind_to_random_port(f"tcp://[{host}]" if host == "::1" else f"tcp://{host}"))
            port_ready.set()
            producer._run_busy_loop(socket)

    server = threading.Thread(target=serve, daemon=True)
    server.start()
    assert port_ready.wait(5)
    try:
        yield producer, ports[0]
    finally:
        producer.stop_event.set()
        server.join(5)
        assert not server.is_alive()


@pytest.fixture
def lease_client_sockets(monkeypatch):
    """Track real per-exchange sockets while sharing one live context."""
    context = zmq.Context.instance()  # type: ignore[attr-defined]
    sockets = []
    create_socket = context.socket

    def socket(*args, **kwargs):
        result = create_socket(*args, **kwargs)
        sockets.append(result)
        return result

    monkeypatch.setattr(context, "socket", socket)
    yield sockets
    assert sockets and all(socket.closed for socket in sockets)
    assert not context.closed
    assert zmq.Context.instance() is context  # type: ignore[attr-defined]


@pytest.mark.parametrize("released", [False, True])
def test_post_read_verification_protocol_detects_released_hold(lease_control_server, released, lease_client_sockets):
    producer, port = lease_control_server
    nonce = producer.add_delayed_request("p", time.time())
    worker = MooncakePullRecvingThread.__new__(MooncakePullRecvingThread)
    worker.encoder = msgspec.msgpack.Encoder()
    meta = SimpleNamespace(remote_request_id="p", remote_lease_nonce=nonce)
    if released:
        with producer.state_lock:
            producer.delayed_free_requests["p"] = 0.0
            producer._retrieve_expired_requests_locked()
    assert worker._verify_remote_leases("127.0.0.1", port, {"d": meta}) == ({"d"} if released else set())
    assert worker._verify_remote_leases(
        "127.0.0.1", port, {"missing": SimpleNamespace(remote_request_id="p", remote_lease_nonce=None)}
    ) == {"missing"}


def test_late_completion_protocol_preserves_new_attempt(lease_control_server):
    producer, port = lease_control_server
    old = producer.add_delayed_request("p", 0.0)
    assert producer.get_and_clear_finished_requests() == {"p"}
    new = producer.add_delayed_request("p", time.time())
    receiver = MooncakeSchedulerRecvingThread(threading.Event())
    try:
        receiver._send_done_recving("127.0.0.1", port, "p", old)
        assert producer._verify_leases((("p", new),)) == [True]
        receiver._send_done_recving("127.0.0.1", port, "p", new)
        receiver._send_done_recving("127.0.0.1", port, "p", new)
        assert producer.get_and_clear_finished_requests() == {"p"}
    finally:
        for sockets in receiver.remote_sockets.values():
            for socket in sockets:
                socket.close(linger=0)
        if receiver.zmq_context is not None:
            receiver.zmq_context.destroy(linger=0)


def test_unresponsive_lease_peer_fails_instead_of_waiting_forever():
    worker = MooncakePullRecvingThread.__new__(MooncakePullRecvingThread)
    worker.encoder = msgspec.msgpack.Encoder()
    with zmq.Context() as context, context.socket(zmq.ROUTER) as socket:  # type: ignore[attr-defined]
        port = socket.bind_to_random_port("tcp://127.0.0.1")
        start = time.monotonic()
        with pytest.raises(zmq.Again):  # type: ignore[attr-defined]
            worker._verify_remote_leases(
                "127.0.0.1", port, {"d": SimpleNamespace(remote_request_id="p", remote_lease_nonce="lease")}
            )
        assert time.monotonic() - start < 5


def test_batch_verification_distinguishes_reused_and_live_holds(lease_control_server):
    producer, port = lease_control_server
    old = producer.add_delayed_request("reused", 0.0)
    assert producer.get_and_clear_finished_requests() == {"reused"}
    new = producer.add_delayed_request("reused", time.time())
    live = producer.add_delayed_request("live", time.time())
    worker = MooncakePullRecvingThread.__new__(MooncakePullRecvingThread)
    worker.encoder = msgspec.msgpack.Encoder()
    requests = {
        "stale": SimpleNamespace(remote_request_id="reused", remote_lease_nonce=old),
        "current": SimpleNamespace(remote_request_id="reused", remote_lease_nonce=new),
        "other": SimpleNamespace(remote_request_id="live", remote_lease_nonce=live),
    }
    assert worker._verify_remote_leases("127.0.0.1", port, requests) == {"stale"}
    assert producer.get_and_clear_finished_requests() == set()
    assert producer._verify_leases((("reused", new), ("live", live))) == [True, True]


def test_duplicate_registration_preserves_original_hold():
    producer = make_sending_thread()
    nonce = producer.add_delayed_request("p", time.time())
    with pytest.raises(ValueError):
        producer.add_delayed_request("p", time.time())
    assert producer._verify_leases((("p", nonce),)) == [True]
    producer._handle_finished_request("p", nonce)
    assert producer.get_and_clear_finished_requests() == {"p"}


@pytest.mark.parametrize("cancelled", [False, True])
@pytest.mark.parametrize("expired", [False, True])
@pytest.mark.parametrize("can_report", [False, True])
def test_read_inflight_expiry_reaches_worker_failure(
    lease_control_server, expired, can_report, cancelled, monkeypatch, lease_client_sockets
):
    producer, port = lease_control_server
    nonce = producer.add_delayed_request("p", time.time())
    worker = make_thread(device=0, ready_event=threading.Event(), can_report_invalid_block_ids=can_report)
    worker.encoder = msgspec.msgpack.Encoder()
    monkeypatch.setattr(torch.npu, "set_device", MagicMock())
    meta = replace(
        make_req_meta(local=([10], [20])),
        remote_request_id="p",
        remote_lease_nonce=nonce,
        remote_host="127.0.0.1",
        remote_port=port,
    )
    worker._get_remote_metadata = MagicMock(return_value=make_metadata_groups(pp_metadata=make_pp_metadata()))
    worker.remote_tp_rank_groups = {"engine-p": {0: {}}}
    worker.remote_layer_index_pairs = {"engine-p": {0: []}}
    worker._build_transfer_block_buckets = MagicMock(
        return_value=({(0, 0): {}, (0, 1): {}}, {(0, 0): {"r"}, (0, 1): {"r"}})
    )
    first_started, second_observed, release_first = threading.Event(), threading.Event(), threading.Event()
    scheduler = make_pull_scheduler()
    scheduler.kv_transfer_config = SimpleNamespace(kv_load_failure_policy="fail")
    scheduler._reqs_recv_info["r"] = ("127.0.0.1", port, "p", nonce)
    scheduler._recving_thread = MagicMock()
    scheduler._recving_thread.add_request.side_effect = (
        lambda _host, _port, req_id, hold: producer._handle_finished_request(req_id, hold)
    )
    observed_results: list[int] = []
    verified_results = []

    def transfer(_pp, _pcp, tp, *_args):
        if tp == 0:
            first_started.set()
            assert release_first.wait(5)

    worker._execute_tp_transfer_bucket = transfer
    verify_remote_leases = worker._verify_remote_leases

    def verify(*args):
        assert observed_results == [1, 0]
        result = verify_remote_leases(*args)
        # Record only a successful real protocol return, never an exception
        # converted by run() into an indistinguishable batch failure.
        verified_results.append(result)
        return result

    worker._verify_remote_leases = MagicMock(side_effect=verify)
    worker.request_queue = MagicMock()
    worker.request_queue.get.side_effect = [("engine-p", {"r": meta}), _StopLoop()]

    def run():
        with pytest.raises(_StopLoop):
            worker.run()

    with ThreadPoolExecutor(max_workers=2) as transfers, ThreadPoolExecutor(max_workers=1) as runner:

        def submit(func, pp, pcp, tp, *args):
            future = transfers.submit(func, pp, pcp, tp, *args)
            original_result = future.result

            def observe_result(*args, **kwargs):
                result = original_result(*args, **kwargs)
                observed_results.append(tp)
                if tp == 1:
                    second_observed.set()
                return result

            monkeypatch.setattr(future, "result", observe_result)
            return future

        worker.executor = SimpleNamespace(submit=submit)
        task = runner.submit(run)
        try:
            assert first_started.wait(5) and second_observed.wait(5)
            worker._verify_remote_leases.assert_not_called()
            assert worker.finished_requests.empty()
            if cancelled:
                request = make_request(request_id="r", status=RequestStatus.FINISHED_ABORTED)
                assert scheduler.request_finished(request, ([10], [20])) == (False, None)
                scheduler._recving_thread.add_request.assert_not_called()
                assert producer._verify_leases((("p", nonce),)) == [True]
            if expired:
                with producer.state_lock:
                    producer.delayed_free_requests["p"] = 0.0
                    producer._retrieve_expired_requests_locked()
        finally:
            release_first.set()
        task.result(timeout=5)
    worker._verify_remote_leases.assert_called_once_with("127.0.0.1", port, {"r": meta})
    assert observed_results == [1, 0]
    assert verified_results == [({"r"} if expired else set())]
    result = worker.get_and_clear_finished_requests()
    assert result.finished_recving == {"r"}
    assert result.failed_recving == ({"r"} if expired else set())
    output = KVConnectorOutput(finished_recving=result.finished_recving, failed_recving=result.failed_recving)
    scheduler.update_connector_output(output)
    scheduler.update_connector_output(output)
    scheduler._recving_thread.add_request.assert_called_once_with("127.0.0.1", port, "p", nonce)
    assert producer.get_and_clear_finished_requests() == {"p"}
    assert not producer.lease_nonces and not scheduler._reqs_recv_info
    assert worker.invalid_block_ids == ({10, 20} if expired and can_report else set())
    worker.request_queue.task_done.assert_called_once()


@pytest.mark.parametrize("can_report", [False, True])
def test_unresponsive_lease_peer_fails_run_then_processes_next_batch(
    lease_control_server, can_report, monkeypatch, lease_client_sockets
):
    producer, healthy_port = lease_control_server
    nonce = producer.add_delayed_request("healthy-p", time.time())
    worker = make_thread(device=0, ready_event=threading.Event(), can_report_invalid_block_ids=can_report)
    worker.encoder = msgspec.msgpack.Encoder()
    monkeypatch.setattr(torch.npu, "set_device", MagicMock())
    worker._get_remote_metadata = MagicMock(return_value=make_metadata_groups(pp_metadata=make_pp_metadata()))
    worker.remote_tp_rank_groups = {"engine-p": {0: {}}}
    worker.remote_layer_index_pairs = {"engine-p": {0: []}}
    worker._build_transfer_block_buckets = MagicMock(
        side_effect=lambda _meta, _pairs, _groups, _pcp, _dcp, requests, _cache: ({(0, 0): {}}, {(0, 0): set(requests)})
    )
    # Only device IO is replaced; both batches use real executor Futures.
    worker._execute_tp_transfer_bucket = MagicMock(return_value=None)
    verify_remote_leases = worker._verify_remote_leases
    verification_outcomes = []

    def verify(host, port, requests):
        try:
            result = verify_remote_leases(host, port, requests)
        except zmq.Again:  # type: ignore[attr-defined]
            verification_outcomes.append((set(requests), "timeout"))
            raise
        verification_outcomes.append((set(requests), result))
        return result

    worker._verify_remote_leases = MagicMock(side_effect=verify)
    worker.request_queue = MagicMock()
    run_finished = threading.Event()
    run_errors = []

    def run():
        try:
            with pytest.raises(_StopLoop):
                worker.run()
        except BaseException as exc:
            run_errors.append(exc)
        finally:
            run_finished.set()

    with zmq.Context() as context, context.socket(zmq.ROUTER) as silent_peer:  # type: ignore[attr-defined]
        silent_port = silent_peer.bind_to_random_port("tcp://127.0.0.1")
        stalled = replace(
            make_req_meta(local=([10],)),
            remote_host="127.0.0.1",
            remote_port=silent_port,
            remote_lease_nonce="silent-hold",
        )
        healthy = replace(
            make_req_meta(local=([20],)),
            remote_request_id="healthy-p",
            remote_host="127.0.0.1",
            remote_port=healthy_port,
            remote_lease_nonce=nonce,
        )
        worker.request_queue.get.side_effect = [
            ("engine-p", {"stalled": stalled}),
            ("engine-p", {"healthy": healthy}),
            _StopLoop(),
        ]
        with ThreadPoolExecutor(max_workers=1) as transfers:
            worker.executor = transfers
            runner = threading.Thread(target=run, daemon=True)
            runner.start()
            assert run_finished.wait(5), "worker did not finish the timed-out and healthy batches"
            runner.join(timeout=1)
            assert not runner.is_alive()
        assert not run_errors, run_errors
        # Prove the silent endpoint received a real VERIFY request, not a
        # synthetic exception or a failure before the protocol was reached.
        assert silent_peer.poll(1000, zmq.POLLIN)  # type: ignore[attr-defined]
        message = msgspec.msgpack.decode(silent_peer.recv_multipart()[-1])
        assert message == [pull_scheduler.VERIFY_LEASE_MSG, [[stalled.remote_request_id, "silent-hold"]]]

    assert verification_outcomes == [({"stalled"}, "timeout"), ({"healthy"}, set())]
    assert worker._verify_remote_leases.call_args_list == [
        call("127.0.0.1", silent_port, {"stalled": stalled}),
        call("127.0.0.1", healthy_port, {"healthy": healthy}),
    ]
    assert worker._execute_tp_transfer_bucket.call_count == 2
    result = worker.get_and_clear_finished_requests()
    assert result.finished_recving == {"stalled", "healthy"}
    assert result.failed_recving == {"stalled"}
    assert worker.get_and_clear_finished_requests().finished_recving == set()
    assert worker.invalid_block_ids == ({10} if can_report else set())
    assert worker.request_queue.task_done.call_count == 2
    assert worker.request_queue.get.call_count == 3


@pytest.mark.parametrize("failure_first", [False, True])
def test_rank_failure_survives_cross_step_aggregation_before_done(failure_first):
    aggregator = KVOutputAggregator(expected_finished_count=2)
    scheduler = make_pull_scheduler()
    scheduler.kv_transfer_config = SimpleNamespace(kv_load_failure_policy="fail")
    scheduler._recving_thread = MagicMock()
    scheduler._reqs_recv_info["r"] = ("127.0.0.1", 6000, "p", "hold-1")
    for step in range(2):
        failed = failure_first if step == 0 else not failure_first
        outputs = [SimpleNamespace(kv_connector_output=KVConnectorOutput()) for _ in range(2)]
        outputs[step].kv_connector_output = KVConnectorOutput(
            finished_recving={"r"}, failed_recving={"r"} if failed else set()
        )
        result = aggregator.aggregate(outputs).kv_connector_output
        scheduler.update_connector_output(result)
        if step == 0:
            assert not result.finished_recving
            scheduler._recving_thread.add_request.assert_not_called()
        else:
            assert result.finished_recving == {"r"} and result.failed_recving == {"r"}
            scheduler._recving_thread.add_request.assert_called_once_with("127.0.0.1", 6000, "p", "hold-1")


@pytest.mark.parametrize("lease_control_server", ["::1"], indirect=True)
def test_lease_verification_preserves_ipv6(lease_control_server, lease_client_sockets):
    producer, port = lease_control_server
    nonce = producer.add_delayed_request("p", time.time())
    worker = MooncakePullRecvingThread.__new__(MooncakePullRecvingThread)
    worker.encoder = msgspec.msgpack.Encoder()
    meta = SimpleNamespace(remote_request_id="p", remote_lease_nonce=nonce)
    assert worker._verify_remote_leases("::1", port, {"d": meta}) == set()
