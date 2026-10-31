# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import RequestStatus

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pool_metadata import (
    PoolObjectRef,
    PoolObjectSlice,
    PoolSourceProfile,
    PoolTransferMetadata,
    PoolWorkerHandshake,
    PoolWorkerReport,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pool_scheduler import LayerwisePoolScheduler
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.key_codec import PoolKeyIdentity
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.restore_planner import CacheSlice
from vllm_ascend.distributed.kv_transfer.utils.request_identity import pool_request_digest


def request(*, producer=True):
    return SimpleNamespace(
        request_id="request",
        prompt_token_ids=[1, 2, 3, 4, 5],
        _all_token_ids=[1, 2, 3, 4, 5],
        prompt_embeds=None,
        num_prompt_tokens=5,
        num_computed_tokens=0,
        max_tokens=1,
        mm_features=[],
        lora_request=None,
        cache_salt=None,
        block_hashes=[b"a", b"b"],
        output_token_ids=[8],
        status=RequestStatus.FINISHED_LENGTH_CAPPED,
        kv_transfer_params={"do_remote_decode": producer, "do_remote_prefill": not producer},
    )


def scheduler(*, producer=True, tp=1, pp=1, pcp=1, client=None):
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(is_kv_producer=producer, is_kv_consumer=not producer),
        model_config=SimpleNamespace(hf_config=SimpleNamespace()),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp,
            pipeline_parallel_size=pp,
            prefill_context_parallel_size=pcp,
            decode_context_parallel_size=1,
        ),
    )
    spec = FullAttentionSpec(block_size=4, num_kv_heads=1, head_size=2, dtype=torch.float16)
    cache = SimpleNamespace(kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec, layer_names=["layer.0"])])
    result = LayerwisePoolScheduler(config, cache, control_request=client or MagicMock())
    result.hash_block_size = 2
    return result


def handshake(worker="pp0/pcp0/tp0"):
    endpoint = "tcp://127.0.0.1:1234"
    profile = PoolSourceProfile(worker, "incarnation", PoolKeyIdentity("memcache-v1", "model", 1, 1), endpoint)
    return PoolWorkerHandshake(worker, profile, endpoint)


def output(*, count=5, blocks=([3, 4],)):
    return SimpleNamespace(
        finished_req_ids=set(),
        num_scheduled_tokens={"request": count},
        kv_connector_block_state=SimpleNamespace(get_block_ids=lambda req_id: blocks),
    )


def committed(state, worker="pp0/pcp0/tp0"):
    cache = CacheSlice("layer.0", "0", "f16", 0, 5, 0, 1, 4, 4, 20)
    ref = PoolObjectRef(worker, 0, "private", 0, 0, 20, (PoolObjectSlice(cache, 0),))
    return PoolWorkerReport(state.transfer_id, state.attempt_id, worker, True, (ref,))


def make_transfer(req):
    cache = CacheSlice("layer.0", "0", "f16", 0, 5, 0, 1, 4, 4, 20)
    return PoolTransferMetadata(
        "transfer",
        "attempt",
        pool_request_digest(req),
        0,
        5,
        "attempt",
        (handshake().profile,),
        (PoolObjectRef("pp0/pcp0/tp0", 0, "private", 0, 0, 20, (PoolObjectSlice(cache, 0),)),),
    )


def test_each_chunk_uses_authoritative_tables_and_source_hash_granularity():
    current = scheduler()
    try:
        current.set_xfer_handshake_metadata({0: handshake()})
        req = request()
        current.on_new_request(req)
        first = current.build_connector_meta(output(count=3)).saves["request"]
        assert first.token_end == 3 and first.chunk_start == 0 and not first.is_final
        req.num_computed_tokens = 3
        saved = current.build_connector_meta(output(count=2)).saves["request"]
        assert saved.block_ids == ((3, 4),)
        assert saved.group_hashes == ((b"b".hex(),),)
        assert saved.token_end == 5
        assert saved.chunk_start == 3 and saved.is_final
        assert not current.build_connector_meta(output(count=2)).saves
    finally:
        current.shutdown()


def test_preemption_releases_chunk_pins_and_restarts_the_attempt():
    current = scheduler()
    try:
        current.set_xfer_handshake_metadata({0: handshake()})
        req = request()
        current.on_new_request(req)
        first = current.build_connector_meta(output(count=3)).saves["request"]
        preempted = output(count=0)
        preempted.preempted_req_ids = {"request"}
        released = current.build_connector_meta(preempted)
        assert released.released_sources == {"request"}
        restarted = current.build_connector_meta(output(count=2)).saves["request"]
        assert restarted.transfer_id == first.transfer_id
        assert restarted.attempt_id != first.attempt_id
        assert restarted.chunk_start == 0 and restarted.token_end == 2
        assert restarted.request_digest == first.request_digest
    finally:
        current.shutdown()


def test_p_prefix_hit_becomes_an_async_restore_before_suffix_compute():
    client = MagicMock()
    current = scheduler(client=client)
    try:
        worker = handshake()
        current.set_xfer_handshake_metadata({0: worker})
        req = request()
        current.on_new_request(req)

        def reply(endpoint, query):
            assert query["op"] == "prefix"
            if query["mode"] == "probe":
                return {"worker_id": worker.worker_id, "ready_ends": [4]}
            data = query["request"]
            cache = CacheSlice("layer.0", "0", "f16", 0, 4, 0, 1, 4, 4, 16)
            obj = PoolObjectRef(worker.profile.profile_id, 0, "prefix-hash", 0, 0, 16, (PoolObjectSlice(cache, 0),))
            return PoolWorkerReport(data["transfer_id"], data["attempt_id"], worker.worker_id, True, (obj,)).to_wire()

        client.side_effect = reply
        assert current.get_num_new_matched_tokens(req, 0) == (4, True)
        current.update_state_after_alloc(req, SimpleNamespace(get_block_ids=lambda: ([2, 3],)), 4)
        meta = current.build_connector_meta(output(count=0))
        assert meta.loads["request"].token_end == 4
        assert meta.local_block_ids["request"] == ([2, 3],)
        assert meta.prefix_group_ids["request"] == (0,)
        assert not meta.saves  # Loading alone cannot publish a new PD handoff.
        current.update_connector_output(KVConnectorOutput(finished_recving={"request"}))
        assert "request" not in current._prefix_receiving
    finally:
        current.shutdown()


def test_p_prefix_requires_every_cp_worker_and_rechecks_after_a_miss():
    current = scheduler(pcp=2)
    try:
        current.set_xfer_handshake_metadata({0: handshake(), 1: handshake("pp0/pcp1/tp0")})
        req = request()
        current.on_new_request(req)
        # Missing or mismatched rank replies must not count as a global hit.
        current._control_request = MagicMock(return_value={"worker_id": "pp0/pcp0/tp0", "ready_ends": [4]})
        assert current.get_num_new_matched_tokens(req, 0) == (0, False)
        assert not current._prefix_hits
    finally:
        current.shutdown()


def test_prefill_digest_precedes_state_truncation_and_ignores_mtp_configuration():
    current = scheduler()
    try:
        current.need_truncate = True
        req = request()
        original_digest = pool_request_digest(req)
        current.on_new_request(req)
        assert current._source["request"].digest == original_digest
        assert req.num_prompt_tokens == 4
        assert req.prompt_token_ids == [1, 2, 3, 4]
        assert req._all_token_ids == [1, 2, 3, 4]
    finally:
        current.shutdown()


def test_source_hashes_use_worker_logical_block_size_instead_of_scheduler_dcp_guess():
    current = scheduler()
    try:
        worker = handshake()
        worker = replace(worker, profile=replace(worker.profile, group_block_sizes=(8,)))
        current.set_xfer_handshake_metadata({0: worker})
        req = request()
        req.block_hashes = [b"a", b"b", b"c", b"d"]
        assert current._source_hashes(req, 0) == (b"d".hex(),)
        assert current._prefix_granularity == 8
    finally:
        current.shutdown()


def test_response_queries_all_commits_before_worker_metadata_update_and_frees_source_hbm():
    client = MagicMock()
    current = scheduler(tp=2, client=client)
    try:
        current.set_xfer_handshake_metadata({0: handshake(), 1: handshake("pp0/pcp0/tp1")})
        req = request()
        current.on_new_request(req)
        current.build_connector_meta(output())
        state = current._source["request"]
        client.side_effect = [committed(state).to_wire(), committed(state, "pp0/pcp0/tp1").to_wire()]
        delay_free, params = current.request_finished(req, ([3, 4],))
        assert not delay_free
        transfer = PoolTransferMetadata.from_wire(params["pool_transfer"])
        assert len(transfer.objects) == len(transfer.profiles) == 2
        assert client.call_count == 2
        assert "remote_block_ids" not in params
    finally:
        current.shutdown()


@pytest.mark.parametrize("failure", ["native_copy", "wrong_attempt", "wrong_worker", "control_timeout"])
def test_failed_source_never_exposes_a_partial_handoff(failure):
    client = MagicMock()
    current = scheduler(client=client)
    try:
        current.set_xfer_handshake_metadata({0: handshake()})
        req = request()
        current.on_new_request(req)
        current.build_connector_meta(output())
        report = committed(current._source["request"])
        if failure == "native_copy":
            report = replace(report, committed=False, error="copy failed", objects=())
        elif failure == "wrong_attempt":
            report = replace(report, attempt_id="old-attempt")
        elif failure == "wrong_worker":
            report = replace(report, worker_id="other-worker")
        else:
            client.side_effect = TimeoutError("control timed out")
        client.return_value = report.to_wire()
        delay_free, params = current.request_finished(req, ([3, 4],))
        assert not delay_free
        assert params == {"do_remote_prefill": False, "do_remote_decode": False}
    finally:
        current.shutdown()


def test_d_restore_excludes_shared_local_prefix_and_releases_only_after_aggregated_result():
    client = MagicMock(return_value={"released": True})
    current = scheduler(producer=False, client=client)
    try:
        req = request(producer=False)
        req.kv_transfer_params["pool_transfer"] = make_transfer(req).to_wire()
        assert current.get_num_new_matched_tokens(req, 4) == (1, True)
        blocks = SimpleNamespace(get_block_ids=lambda: ([7, 8],))
        current.update_state_after_alloc(req, blocks, 1)
        meta = current.build_connector_meta(output(count=0))
        assert meta.loads["request"].token_start == 4
        assert meta.local_block_ids["request"] == ([7, 8],)
        assert current.get_num_new_matched_tokens(req, 4) == (0, False)
        client.assert_not_called()
        current.update_connector_output(KVConnectorOutput(finished_recving={"request"}))
        current._notifications.submit(lambda: None).result(timeout=5)
        assert client.call_args.args[1]["op"] == "release"
    finally:
        current.shutdown()


@pytest.mark.parametrize("lookup_started", [False, True])
def test_cancel_before_allocation_releases_committed_source_once(lookup_started):
    client = MagicMock(return_value={"released": True})
    current = scheduler(producer=False, client=client)
    try:
        req = request(producer=False)
        req.kv_transfer_params["pool_transfer"] = make_transfer(req).to_wire()
        if lookup_started:
            assert current.get_num_new_matched_tokens(req, 0) == (5, True)
        req.status = RequestStatus.FINISHED_ABORTED
        assert current.request_finished(req, ([],)) == (False, None)
        assert current.request_finished(req, ([],)) == (False, None)
        current._notifications.submit(lambda: None).result(timeout=5)
        client.assert_called_once_with(
            handshake().control_endpoint,
            {
                "op": "release",
                "incarnation": "incarnation",
                "transfer_id": "transfer",
                "attempt_id": "attempt",
            },
        )
        assert not current._loads and not current._receiving and not current._cancelled
        assert not req.kv_transfer_params["do_remote_prefill"]
    finally:
        current.shutdown()


def test_cancel_before_allocation_does_not_release_a_mismatched_handoff():
    client = MagicMock()
    current = scheduler(producer=False, client=client)
    try:
        req = request(producer=False)
        req.kv_transfer_params["pool_transfer"] = make_transfer(req).to_wire()
        req.prompt_token_ids[-1] = 99
        req.status = RequestStatus.FINISHED_ABORTED
        assert current.request_finished(req, ([],)) == (False, None)
        current._notifications.submit(lambda: None).result(timeout=5)
        client.assert_not_called()
    finally:
        current.shutdown()


@pytest.mark.parametrize("dispatched", [False, True])
def test_cancel_after_allocation_still_waits_for_aggregated_worker_completion(dispatched):
    client = MagicMock(return_value={"released": True})
    current = scheduler(producer=False, client=client)
    try:
        req = request(producer=False)
        req.kv_transfer_params["pool_transfer"] = make_transfer(req).to_wire()
        current.update_state_after_alloc(req, SimpleNamespace(get_block_ids=lambda: ([7, 8],)), 5)
        if dispatched:
            assert "request" in current.build_connector_meta(output(count=0)).loads
        req.status = RequestStatus.FINISHED_ABORTED
        assert current.request_finished(req, ([7, 8],)) == (False, None)
        metadata = current.build_connector_meta(output(count=0))
        assert metadata.cancelled == {"request"}
        assert ("request" in metadata.loads) is not dispatched
        current._notifications.submit(lambda: None).result(timeout=5)
        client.assert_not_called()
        current.update_connector_output(KVConnectorOutput(finished_recving={"request"}))
        current._notifications.submit(lambda: None).result(timeout=5)
        assert client.call_count == 1
        assert not current._receiving
    finally:
        current.shutdown()


@pytest.mark.parametrize("changed", ["tokens", "salt", "adapter", "multimodal"])
def test_mismatched_input_cannot_be_advertised_as_pool_hit(changed):
    current = scheduler(producer=False)
    try:
        req = request(producer=False)
        req.kv_transfer_params["pool_transfer"] = make_transfer(req).to_wire()
        if changed == "tokens":
            req.prompt_token_ids[-1] = 99
        elif changed == "salt":
            req.cache_salt = "other-tenant"
        elif changed == "adapter":
            req.lora_request = SimpleNamespace(lora_name="adapter", lora_int_id=1)
        else:
            req.mm_features = [
                SimpleNamespace(
                    mm_hash="image-hash",
                    identifier="image",
                    modality="image",
                    mm_position=SimpleNamespace(offset=0, length=1),
                )
            ]
        assert current.get_num_new_matched_tokens(req, 0) == (0, False)
        assert req.kv_transfer_params["do_remote_prefill"] is False
    finally:
        current.shutdown()


def test_handshake_requires_exact_worker_set():
    current = scheduler(tp=2)
    try:
        with pytest.raises(ValueError, match="every"):
            current.set_xfer_handshake_metadata({0: handshake()})
        with pytest.raises(ValueError, match="Duplicate"):
            current.set_xfer_handshake_metadata({0: handshake(), 1: handshake()})
    finally:
        current.shutdown()


@pytest.mark.parametrize("wrong_worker", ["pp2/pcp0/tp0", "pp0/pcp2/tp0", "pp0/pcp0/tp2", "worker0"])
def test_handshake_rejects_same_count_with_missing_topology_shard(wrong_worker):
    current = scheduler(tp=2, pp=2, pcp=2)
    expected = {f"pp{pp}/pcp{pcp}/tp{tp}" for pp in range(2) for pcp in range(2) for tp in range(2)}
    workers = {index: handshake(worker) for index, worker in enumerate(sorted(expected, reverse=True))}
    try:
        current.set_xfer_handshake_metadata(workers)
        assert current._workers.keys() == expected
        workers[0] = handshake(wrong_worker)
        with pytest.raises(ValueError, match="missing=.*unexpected="):
            current.set_xfer_handshake_metadata(workers)
        # A rejected replacement cannot corrupt the previous complete mapping.
        assert current._workers.keys() == expected
    finally:
        current.shutdown()
