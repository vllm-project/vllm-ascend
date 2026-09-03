from types import SimpleNamespace

import pytest
from vllm.distributed.kv_events import BlockStored

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    AscendConnectorMetadata,
    AscendStoreKVConnectorWorkerMetadata,
    LoadSpec,
    ReqMeta,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.kv_cache import protocol as kv_cache_protocol
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.kv_cache.npu_ipc import (
    KVCacheStorageSpec,
    KVCacheTensorSpec,
    NPUEventSpec,
    WorkerKVCacheSpec,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.kv_cache.protocol import (
    ACK_RESPONSE,
    KVCacheMethod,
    decode_ack_response,
    decode_build_connector_meta_request,
    decode_build_connector_meta_response,
    decode_build_connector_worker_meta_request,
    decode_build_connector_worker_meta_response,
    decode_get_block_ids_with_load_errors_request,
    decode_get_block_ids_with_load_errors_response,
    decode_get_finished_request,
    decode_get_finished_response,
    decode_get_kv_events_request,
    decode_get_kv_events_response,
    decode_lookup_request,
    decode_lookup_response,
    decode_register_kv_caches_request,
    decode_registration,
    decode_registration_request,
    decode_request_finished,
    decode_request_finished_response,
    decode_save_kv_layer_request,
    decode_scheduler_session,
    decode_start_load_kv_request,
    decode_update_connector_output,
    decode_update_connector_output_response,
    decode_update_state_after_alloc,
    decode_wait_for_layer_load_request,
    decode_wait_for_save_request,
    decode_worker_session,
    encode_build_connector_meta_request,
    encode_build_connector_meta_response,
    encode_build_connector_worker_meta_request,
    encode_build_connector_worker_meta_response,
    encode_get_block_ids_with_load_errors_request,
    encode_get_block_ids_with_load_errors_response,
    encode_get_finished_request,
    encode_get_finished_response,
    encode_get_kv_events_request,
    encode_get_kv_events_response,
    encode_lookup_request,
    encode_lookup_response,
    encode_register_kv_caches_request,
    encode_registration,
    encode_registration_request,
    encode_request_finished,
    encode_request_finished_response,
    encode_save_kv_layer_request,
    encode_scheduler_session,
    encode_start_load_kv_request,
    encode_update_connector_output,
    encode_update_connector_output_response,
    encode_update_state_after_alloc,
    encode_wait_for_layer_load_request,
    encode_wait_for_save_request,
    encode_worker_session,
    scheduler_affinity_key,
    worker_affinity_key,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.kv_cache.registration import (
    SchedulerIdentity,
    SchedulerRegistration,
    WorkerIdentity,
    WorkerRegistration,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.rpc import MPProtocolError


def _make_vllm_config():
    hf_config = SimpleNamespace(num_hidden_layers=2, model_type="llama")
    return SimpleNamespace(
        model_config=SimpleNamespace(
            model="org/model",
            max_model_len=1024,
            hf_text_config=hf_config,
            hf_config=hf_config,
            use_mla=False,
            get_num_layers=lambda _parallel_config: 2,
            get_total_num_kv_heads=lambda: 1,
        ),
        parallel_config=SimpleNamespace(
            rank=2,
            world_size=4,
            data_parallel_rank=1,
            data_parallel_index=1,
            data_parallel_size=2,
            tensor_parallel_size=2,
            pipeline_parallel_size=2,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        kv_transfer_config=SimpleNamespace(
            engine_id="engine-0",
            kv_role="kv_both",
            kv_connector="AscendStoreConnector",
            kv_connector_extra_config={},
        ),
        cache_config=SimpleNamespace(block_size=16, prefix_match_unit=None),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        speculative_config=None,
        kv_events_config=None,
    )


def _make_worker_kv_cache_spec() -> WorkerKVCacheSpec:
    storage = KVCacheStorageSpec(
        size_bytes=4096,
        device_type="npu",
        device_uuid="host-0",
        handle_type="torch_npu_ipc",
        handle_version=1,
        handle=b"ipc-handle",
    )
    tensor = KVCacheTensorSpec(
        storage_index=0,
        storage_offset_bytes=0,
        shape=(16, 2, 8),
        stride=(16, 8, 1),
        dtype="torch.float16",
    )
    return WorkerKVCacheSpec(
        caches={"layer.0": (tensor,)},
        storages=(storage,),
    )


def test_registration_round_trip_and_type_validation() -> None:
    scheduler_registration = SchedulerRegistration.create(
        _make_vllm_config(),
        kv_cache_config=None,
        page_size_bytes=4096,
        session_id="scheduler-session",
    )
    worker_registration = WorkerRegistration.create(
        _make_vllm_config(),
        kv_cache_config=None,
        session_id="worker-session",
    )
    payload = encode_registration(scheduler_registration)

    assert decode_registration((payload,), SchedulerRegistration) == scheduler_registration
    with pytest.raises(MPProtocolError, match="Expected WorkerRegistration"):
        decode_registration((payload,), WorkerRegistration)

    scheduler_payloads = encode_registration_request(scheduler_registration)
    worker_payloads = encode_registration_request(worker_registration)
    serialized_scheduler = encode_registration(scheduler_registration)
    serialized_worker = encode_registration(worker_registration)
    assert decode_registration_request(scheduler_payloads, SchedulerRegistration) == (
        scheduler_registration,
        serialized_scheduler,
    )
    assert decode_registration_request(worker_payloads, WorkerRegistration) == (
        worker_registration,
        serialized_worker,
    )
    assert len(scheduler_payloads) == 4
    assert len(worker_payloads) == 5
    assert scheduler_affinity_key(b"client", scheduler_payloads) == scheduler_registration.identity
    assert worker_affinity_key(b"client", worker_payloads) == worker_registration.identity

    mismatched_payloads = (b"other-engine", *scheduler_payloads[1:])
    with pytest.raises(MPProtocolError, match="identity does not match request header"):
        decode_registration_request(mismatched_payloads, SchedulerRegistration)


def test_registration_size_is_checked_before_encoding_and_decoding(monkeypatch) -> None:
    registration = SchedulerRegistration.create(_make_vllm_config(), None, 0)
    payload = encode_registration(registration)
    monkeypatch.setattr(kv_cache_protocol, "_MAX_REGISTRATION_BYTES", len(payload) - 1)

    with pytest.raises(MPProtocolError, match="registration limit"):
        encode_registration(registration)
    with pytest.raises(MPProtocolError, match="registration limit"):
        decode_registration((payload,), SchedulerRegistration)


def test_service_session_round_trip() -> None:
    scheduler_identity = SchedulerIdentity("engine-0", data_parallel_rank=1)
    worker_identity = WorkerIdentity("engine-0", rank=2, data_parallel_rank=1)
    scheduler_payloads = encode_scheduler_session(scheduler_identity, "scheduler-session")
    worker_payloads = encode_worker_session(worker_identity, "worker-session")

    assert len(scheduler_payloads) == 4
    assert len(worker_payloads) == 5
    assert decode_scheduler_session(scheduler_payloads) == (
        scheduler_identity,
        "scheduler-session",
    )
    assert decode_worker_session(worker_payloads) == (
        worker_identity,
        "worker-session",
    )
    assert scheduler_affinity_key(b"client", scheduler_payloads) == scheduler_identity
    assert worker_affinity_key(b"client", worker_payloads) == worker_identity


def test_register_kv_caches_round_trip() -> None:
    registration = WorkerRegistration.create(
        _make_vllm_config(),
        kv_cache_config=None,
        session_id="worker-session",
    )
    spec = _make_worker_kv_cache_spec()

    payloads = encode_register_kv_caches_request(registration, spec)

    assert len(payloads) == 5
    assert worker_affinity_key(b"client", payloads) == registration.identity
    assert decode_register_kv_caches_request(payloads) == (
        registration.identity,
        registration.session_id,
        spec,
    )


def test_wait_for_save_round_trip() -> None:
    registration = WorkerRegistration.create(
        _make_vllm_config(),
        kv_cache_config=None,
        session_id="worker-session",
    )
    metadata = AscendConnectorMetadata(set(), set())
    metadata.add_request(ReqMeta("request-0", can_save=True))
    event = NPUEventSpec("host-0", b"event-handle")

    payloads = encode_wait_for_save_request(registration, metadata, event)
    identity, session_id, decoded_metadata, decoded_event = decode_wait_for_save_request(payloads)

    assert worker_affinity_key(b"client", payloads) == registration.identity
    assert (identity, session_id) == (registration.identity, registration.session_id)
    assert decoded_metadata.requests[0].req_id == "request-0"
    assert decoded_metadata.requests[0].can_save is True
    assert decoded_event == event


def test_layerwise_requests_round_trip() -> None:
    registration = WorkerRegistration.create(
        _make_vllm_config(),
        kv_cache_config=None,
        session_id="worker-session",
    )
    event = NPUEventSpec("host-0", b"event-handle")

    wait_payloads = encode_wait_for_layer_load_request(registration)
    save_payloads = encode_save_kv_layer_request(registration, event)

    assert decode_wait_for_layer_load_request(wait_payloads) == (
        registration.identity,
        registration.session_id,
    )
    assert decode_save_kv_layer_request(save_payloads) == (
        registration.identity,
        registration.session_id,
        event,
    )
    assert worker_affinity_key(b"client", wait_payloads) == registration.identity
    assert worker_affinity_key(b"client", save_payloads) == registration.identity


def test_get_finished_round_trip() -> None:
    registration = WorkerRegistration.create(
        _make_vllm_config(),
        kv_cache_config=None,
        session_id="worker-session",
    )
    metadata = AscendConnectorMetadata(
        preempted_req_ids={"preempted-0"},
        loading_req_ids={"loading-0"},
        delayed_free_req_ids={"saving-0"},
    )

    payloads = encode_get_finished_request(registration, {"saving-0", "loading-0"}, metadata)
    identity, session_id, finished_req_ids, decoded_metadata = decode_get_finished_request(payloads)

    assert worker_affinity_key(b"client", payloads) == registration.identity
    assert (identity, session_id) == (registration.identity, registration.session_id)
    assert finished_req_ids == {"saving-0", "loading-0"}
    assert decoded_metadata.preempted_req_ids == {"preempted-0"}
    assert decoded_metadata.loading_req_ids == {"loading-0"}
    assert decoded_metadata.delayed_free_req_ids == {"saving-0"}
    response = encode_get_finished_response({"saving-0"}, {"loading-0"})
    assert decode_get_finished_response(response) == ({"saving-0"}, {"loading-0"})


def test_build_connector_worker_meta_round_trip() -> None:
    registration = WorkerRegistration.create(
        _make_vllm_config(),
        kv_cache_config=None,
        session_id="worker-session",
    )

    payloads = encode_build_connector_worker_meta_request(registration)

    assert worker_affinity_key(b"client", payloads) == registration.identity
    assert decode_build_connector_worker_meta_request(payloads) == (
        registration.identity,
        registration.session_id,
    )
    metadata = AscendStoreKVConnectorWorkerMetadata({7: 1, 9: 1})
    assert (
        decode_build_connector_worker_meta_response(encode_build_connector_worker_meta_response(metadata)) == metadata
    )
    assert decode_build_connector_worker_meta_response(encode_build_connector_worker_meta_response(None)) is None


def test_get_kv_events_round_trip() -> None:
    registration = WorkerRegistration.create(
        _make_vllm_config(),
        kv_cache_config=None,
        session_id="worker-session",
    )
    event = BlockStored(
        block_hashes=[b"hash-0"],
        parent_block_hash=None,
        token_ids=[1, 2],
        block_size=2,
        lora_id=None,
        medium="CPU",
        lora_name=None,
    )

    payloads = encode_get_kv_events_request(registration)

    assert worker_affinity_key(b"client", payloads) == registration.identity
    assert decode_get_kv_events_request(payloads) == (
        registration.identity,
        registration.session_id,
    )
    assert decode_get_kv_events_response(encode_get_kv_events_response([event])) == [event]
    assert decode_get_kv_events_response(encode_get_kv_events_response([])) == []


def test_start_load_kv_and_load_errors_round_trip() -> None:
    registration = WorkerRegistration.create(
        _make_vllm_config(),
        kv_cache_config=None,
        session_id="worker-session",
    )
    metadata = AscendConnectorMetadata(set(), set())
    metadata.add_request(
        ReqMeta(
            "request-0",
            token_len_chunk=16,
            block_ids=[7],
            block_hashes=[b"hash-0"],
            load_spec=LoadSpec(0, 16, True),
        )
    )

    payloads = encode_start_load_kv_request(registration, metadata)
    identity, session_id, decoded_metadata = decode_start_load_kv_request(payloads)

    assert worker_affinity_key(b"client", payloads) == registration.identity
    assert (identity, session_id) == (registration.identity, registration.session_id)
    assert decoded_metadata.requests[0].req_id == "request-0"
    assert decoded_metadata.requests[0].block_ids == [7]

    error_payloads = encode_get_block_ids_with_load_errors_request(registration)
    assert decode_get_block_ids_with_load_errors_request(error_payloads) == (
        registration.identity,
        registration.session_id,
    )
    assert decode_get_block_ids_with_load_errors_response(encode_get_block_ids_with_load_errors_response({7, 9})) == {
        7,
        9,
    }


def test_lookup_request_preserves_required_fields_and_response_round_trip() -> None:
    registration = SchedulerRegistration.create(
        _make_vllm_config(),
        kv_cache_config=None,
        page_size_bytes=4096,
        session_id="scheduler-session",
    )
    request = SimpleNamespace(
        request_id="request-0",
        prompt_token_ids=[1, 2, 3],
        block_hashes=[b"hash-0", b"hash-1"],
        num_tokens=3,
    )

    payloads = encode_lookup_request(registration, request, num_computed_tokens=2)
    identity, session_id, decoded_request, num_computed_tokens = decode_lookup_request(payloads)

    assert scheduler_affinity_key(b"client", payloads) == registration.identity
    assert identity == registration.identity
    assert session_id == registration.session_id
    assert decoded_request.request_id == request.request_id
    assert decoded_request.prompt_token_ids == range(len(request.prompt_token_ids))
    assert decoded_request.block_hashes == request.block_hashes
    assert decoded_request.num_tokens == request.num_tokens
    assert num_computed_tokens == 2
    assert decode_lookup_response(encode_lookup_response(16, True)) == (16, True)


def test_lookup_protocol_rejects_malformed_payloads() -> None:
    with pytest.raises(MPProtocolError, match="expects at least 2 payloads"):
        scheduler_affinity_key(b"client", ())
    with pytest.raises(MPProtocolError, match="expects 4 payloads"):
        decode_lookup_request(())
    with pytest.raises(MPProtocolError, match="expects 1 response payload"):
        decode_lookup_response([])


def test_ack_response_validation() -> None:
    decode_ack_response((ACK_RESPONSE,), KVCacheMethod.RENEW_SCHEDULER)

    with pytest.raises(MPProtocolError, match="expects 1 response payload"):
        decode_ack_response((), KVCacheMethod.RENEW_SCHEDULER)
    with pytest.raises(MPProtocolError, match="expects an OK response"):
        decode_ack_response((b"invalid",), KVCacheMethod.RENEW_SCHEDULER)


def test_update_state_after_alloc_round_trip() -> None:
    registration = SchedulerRegistration.create(
        _make_vllm_config(),
        kv_cache_config=None,
        page_size_bytes=0,
        session_id="scheduler-session",
    )
    request = SimpleNamespace(
        request_id="request-0",
        prompt_token_ids=list(range(64)),
        block_hashes=[bytes([idx]) * 32 for idx in range(4)],
        num_tokens=64,
    )
    blocks = SimpleNamespace(get_block_ids=lambda: ([7, 8], [9]))

    payloads = encode_update_state_after_alloc(registration, request, blocks, num_external_tokens=48)
    identity, session_id, view, decoded_blocks, num_external_tokens = decode_update_state_after_alloc(payloads)

    assert scheduler_affinity_key(b"client", payloads) == registration.identity
    assert (identity, session_id) == (registration.identity, registration.session_id)
    assert num_external_tokens == 48
    assert view.request_id == request.request_id
    assert view.prompt_token_ids == request.prompt_token_ids
    assert view.block_hashes == request.block_hashes
    assert view.num_prompt_tokens == 64
    assert view.num_tokens == 64
    assert decoded_blocks.get_block_ids() == ([7, 8], [9])


def test_update_state_after_alloc_zero_external_carries_no_block_ids() -> None:
    registration = SchedulerRegistration.create(
        _make_vllm_config(),
        kv_cache_config=None,
        page_size_bytes=0,
        session_id="scheduler-session",
    )
    request = SimpleNamespace(
        request_id="request-0",
        prompt_token_ids=[1, 2, 3],
        block_hashes=[b"hash-0", b"hash-1"],
        num_tokens=3,
    )
    blocks = SimpleNamespace(get_block_ids=lambda: ([7],))

    payloads = encode_update_state_after_alloc(registration, request, blocks, num_external_tokens=0)
    _, _, _, decoded_blocks, num_external_tokens = decode_update_state_after_alloc(payloads)

    assert num_external_tokens == 0
    assert decoded_blocks.get_block_ids() == ()

    with pytest.raises(MPProtocolError, match="expects 4 payloads"):
        decode_update_state_after_alloc(payloads[:3])


def test_build_connector_meta_request_round_trip() -> None:
    registration = SchedulerRegistration.create(
        _make_vllm_config(),
        kv_cache_config=None,
        page_size_bytes=0,
        session_id="scheduler-session",
    )
    scheduler_output = SimpleNamespace(
        finished_req_ids={"done-0"},
        preempted_req_ids=set(),
        num_scheduled_tokens={"request-0": 48},
        scheduled_new_reqs=[SimpleNamespace(req_id="request-0", num_computed_tokens=16, block_ids=([7, 8], [9]))],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["request-1"],
            new_block_ids=[([10],)],
            num_computed_tokens=[64],
        ),
    )
    new_token_ids = {"request-1": [101, 102]}

    payloads = encode_build_connector_meta_request(registration, scheduler_output, new_token_ids)
    identity, session_id, view = decode_build_connector_meta_request(payloads)

    assert scheduler_affinity_key(b"client", payloads) == registration.identity
    assert (identity, session_id) == (registration.identity, registration.session_id)
    assert view.finished_req_ids == {"done-0"}
    assert view.preempted_req_ids == set()
    assert view.num_scheduled_tokens == {"request-0": 48}
    new_req = view.scheduled_new_reqs[0]
    assert (new_req.req_id, new_req.num_computed_tokens) == ("request-0", 16)
    assert new_req.block_ids_by_group == [[7, 8], [9]]
    cached = view.scheduled_cached_reqs
    assert cached.req_ids == ["request-1"]
    assert cached.new_block_ids == [[[10]]]
    assert cached.num_computed_tokens == [64]
    assert cached.new_token_ids == {"request-1": [101, 102]}

    with pytest.raises(MPProtocolError, match="expects 4 payloads"):
        decode_build_connector_meta_request(payloads[:3])


def test_build_connector_meta_response_round_trip() -> None:
    marker = SimpleNamespace(name="metadata-marker")
    payload = encode_build_connector_meta_response(marker, [5, 8])
    metadata, touch_block_ids = decode_build_connector_meta_response(payload)

    assert metadata == marker
    assert touch_block_ids == [5, 8]


def test_request_finished_round_trip() -> None:
    registration = SchedulerRegistration.create(
        _make_vllm_config(),
        kv_cache_config=None,
        page_size_bytes=0,
        session_id="scheduler-session",
    )
    for all_groups, block_ids in ((False, [7, 8]), (True, ([7], [8]))):
        payloads = encode_request_finished(registration, "request-0", block_ids, all_groups)
        identity, session_id, request_id, decoded_block_ids, decoded_all_groups = decode_request_finished(payloads)

        assert scheduler_affinity_key(b"client", payloads) == registration.identity
        assert identity == registration.identity
        assert session_id == registration.session_id
        assert request_id == "request-0"
        assert decoded_block_ids == block_ids
        assert decoded_all_groups is all_groups

    response = encode_request_finished_response(True, None)
    assert decode_request_finished_response(response) == (True, None)


def test_request_finished_rejects_malformed_block_ids() -> None:
    registration = SchedulerRegistration.create(
        _make_vllm_config(),
        kv_cache_config=None,
        page_size_bytes=0,
        session_id="scheduler-session",
    )

    def malformed_payload(block_ids, all_groups: bool) -> tuple[bytes, ...]:
        body = {"request_id": "request-0", "block_ids": block_ids, "all_groups": all_groups}
        return kv_cache_protocol._encode_scheduler_request(registration, body, KVCacheMethod.REQUEST_FINISHED)

    with pytest.raises(MPProtocolError, match="block_ids must be a sequence of block ids"):
        decode_request_finished(malformed_payload("7,8", False))
    with pytest.raises(MPProtocolError, match="block_ids must contain integers only"):
        decode_request_finished(malformed_payload([7, "8"], False))
    with pytest.raises(MPProtocolError, match="block_ids must be a sequence of per-group block id sequences"):
        decode_request_finished(malformed_payload(7, True))
    with pytest.raises(MPProtocolError, match="block_ids groups must be integer sequences"):
        decode_request_finished(malformed_payload([7, 8], True))
    with pytest.raises(MPProtocolError, match="block_ids groups must be integer sequences"):
        decode_request_finished(malformed_payload(([7], "8"), True))
    with pytest.raises(MPProtocolError, match="block_ids groups must contain integers only"):
        decode_request_finished(malformed_payload(([7], [True]), True))

    with pytest.raises(TypeError, match="block_ids must contain integers only"):
        encode_request_finished(registration, "request-0", [7, "8"], False)
    with pytest.raises(TypeError, match="block_ids groups must be integer sequences"):
        encode_request_finished(registration, "request-0", [7, 8], True)


def test_update_connector_output_round_trip() -> None:
    registration = SchedulerRegistration.create(
        _make_vllm_config(),
        kv_cache_config=None,
        page_size_bytes=0,
        session_id="scheduler-session",
    )

    payloads = encode_update_connector_output(registration, {7: 1, 9: 1})
    identity, session_id, completed_events = decode_update_connector_output(payloads)

    assert scheduler_affinity_key(b"client", payloads) == registration.identity
    assert (identity, session_id) == (registration.identity, registration.session_id)
    assert completed_events == {7: 1, 9: 1}

    assert decode_update_connector_output_response(encode_update_connector_output_response([5, 8])) == [5, 8]
