"""Define the cross-process contract for KV cache RPCs.

Requests expose the logical service identity and session before their
serialized body, allowing the server to choose the service's executor thread
without decoding business data. Codecs project live vLLM objects into
metadata-only views and validate the process boundary; service behavior remains
in the client, server, and manager.
"""

import enum
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeVar

import cloudpickle
from vllm.distributed.kv_events import BlockStored
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import Request

from ...metadata import AscendConnectorMetadata, AscendStoreKVConnectorWorkerMetadata
from ..rpc import MPProtocolError
from .npu_ipc import NPUEventSpec, WorkerKVCacheSpec
from .registration import SchedulerIdentity, SchedulerRegistration, WorkerIdentity, WorkerRegistration
from .scheduler_view import BlocksView, CachedReqsView, RequestView, ScheduledNewReqPayload, SchedulerOutputView

ACK_RESPONSE = b"OK"

_INTEGER_BYTES = 8
_BYTE_ORDER = "big"
_MAX_REGISTRATION_BYTES = 8 * 1024 * 1024
_SCHEDULER_REQUEST_PAYLOADS = 4
_WORKER_REQUEST_PAYLOADS = 5

_Registration = SchedulerRegistration | WorkerRegistration
RegistrationT = TypeVar("RegistrationT", bound=_Registration)
_ValidatedT = TypeVar("_ValidatedT")


class KVCacheMethod(str, enum.Enum):
    REGISTER_SCHEDULER = "REGISTER_SCHEDULER"
    REGISTER_WORKER = "REGISTER_WORKER"
    REGISTER_KV_CACHES = "REGISTER_KV_CACHES"
    UNREGISTER_SCHEDULER = "UNREGISTER_SCHEDULER"
    UNREGISTER_WORKER = "UNREGISTER_WORKER"
    RENEW_SCHEDULER = "RENEW_SCHEDULER"
    RENEW_WORKER = "RENEW_WORKER"
    LOOKUP = "LOOKUP"
    UPDATE_STATE_AFTER_ALLOC = "UPDATE_STATE_AFTER_ALLOC"
    BUILD_CONNECTOR_META = "BUILD_CONNECTOR_META"
    REQUEST_FINISHED = "REQUEST_FINISHED"
    UPDATE_CONNECTOR_OUTPUT = "UPDATE_CONNECTOR_OUTPUT"
    WAIT_FOR_SAVE = "WAIT_FOR_SAVE"
    GET_FINISHED = "GET_FINISHED"
    BUILD_CONNECTOR_WORKER_META = "BUILD_CONNECTOR_WORKER_META"
    GET_KV_EVENTS = "GET_KV_EVENTS"
    START_LOAD_KV = "START_LOAD_KV"
    WAIT_FOR_LAYER_LOAD = "WAIT_FOR_LAYER_LOAD"
    SAVE_KV_LAYER = "SAVE_KV_LAYER"
    GET_BLOCK_IDS_WITH_LOAD_ERRORS = "GET_BLOCK_IDS_WITH_LOAD_ERRORS"


# ==============================
# Service registration and session lifecycle
# ==============================

# Registration binds a configuration to one identity and session. Identity and
# session are repeated outside the serialized registration so routing happens
# before decoding and the decoded body can be checked against its header. Worker
# recovery also restores its fixed cache mapping before the client reports the
# service as registered; renew and unregister reuse the same header without a body.


def encode_registration(registration: _Registration) -> bytes:
    try:
        payload = cloudpickle.dumps(registration)
    except Exception as exc:
        raise MPProtocolError(f"Failed to encode {type(registration).__name__}: {exc}") from exc
    _validate_registration_size(payload, type(registration).__name__)
    return payload


def decode_registration(payloads: Sequence[bytes], expected_type: type[RegistrationT]) -> RegistrationT:
    payload = _single_response(payloads, expected_type.__name__)
    _validate_registration_size(payload, expected_type.__name__)
    try:
        registration = cloudpickle.loads(payload)
    except Exception as exc:
        # Preserve the pickle root cause because the client cannot inspect a
        # payload that failed inside the server process.
        raise MPProtocolError(f"Failed to decode {expected_type.__name__}: {exc}") from exc

    if not isinstance(registration, expected_type):
        raise MPProtocolError(f"Expected {expected_type.__name__}, got {type(registration).__name__}")
    return registration


def _validate_registration_size(payload: bytes, name: str) -> None:
    if len(payload) > _MAX_REGISTRATION_BYTES:
        raise MPProtocolError(
            f"{name} exceeds the {_MAX_REGISTRATION_BYTES}-byte registration limit: {len(payload)} bytes"
        )


def encode_registration_request(registration: _Registration) -> tuple[bytes, ...]:
    payload = encode_registration(registration)
    if isinstance(registration, SchedulerRegistration):
        return _encode_scheduler_envelope(registration.identity, registration.session_id, payload)
    if isinstance(registration, WorkerRegistration):
        return _encode_worker_envelope(registration.identity, registration.session_id, payload)
    raise TypeError(f"Unsupported registration type: {type(registration).__name__}")


def decode_registration_request(
    payloads: tuple[bytes, ...],
    expected_type: type[RegistrationT],
) -> tuple[RegistrationT, bytes]:
    if expected_type is SchedulerRegistration:
        identity, session_id, payload = _decode_scheduler_envelope(payloads, KVCacheMethod.REGISTER_SCHEDULER.value)
    elif expected_type is WorkerRegistration:
        identity, session_id, payload = _decode_worker_envelope(payloads, KVCacheMethod.REGISTER_WORKER.value)
    else:
        raise TypeError(f"Unsupported registration type: {expected_type.__name__}")

    registration = decode_registration((payload,), expected_type)
    if registration.identity != identity:
        raise MPProtocolError(
            f"{expected_type.__name__} identity does not match request header: "
            f"{registration.identity!r} != {identity!r}"
        )
    if registration.session_id != session_id:
        raise MPProtocolError(
            f"{expected_type.__name__} session does not match request header: "
            f"{registration.session_id!r} != {session_id!r}"
        )
    return registration, payload


def encode_scheduler_session(identity: SchedulerIdentity, session_id: str) -> tuple[bytes, ...]:
    return _encode_scheduler_envelope(identity, session_id, b"")


def decode_scheduler_session(payloads: tuple[bytes, ...]) -> tuple[SchedulerIdentity, str]:
    identity, session_id, body = _decode_scheduler_envelope(payloads, "Scheduler session")
    _require_empty_body(body, "Scheduler session")
    return identity, session_id


def encode_worker_session(identity: WorkerIdentity, session_id: str) -> tuple[bytes, ...]:
    return _encode_worker_envelope(identity, session_id, b"")


def decode_worker_session(payloads: tuple[bytes, ...]) -> tuple[WorkerIdentity, str]:
    identity, session_id, body = _decode_worker_envelope(payloads, "Worker session")
    _require_empty_body(body, "Worker session")
    return identity, session_id


def encode_register_kv_caches_request(registration: WorkerRegistration, spec: WorkerKVCacheSpec) -> tuple[bytes, ...]:
    _validate_type(spec, WorkerKVCacheSpec, "spec")
    return _encode_worker_request(registration, {"spec": spec}, KVCacheMethod.REGISTER_KV_CACHES)


def decode_register_kv_caches_request(payloads: tuple[bytes, ...]) -> tuple[WorkerIdentity, str, WorkerKVCacheSpec]:
    method = KVCacheMethod.REGISTER_KV_CACHES
    identity, session_id, body = _decode_worker_request(payloads, method)
    (spec,) = _body_fields(body, method.value, "spec")
    _require_type(spec, WorkerKVCacheSpec, "spec")
    return identity, session_id, spec


# ==============================
# Scheduler request lifecycle
# ==============================

# Scheduler RPCs turn live vLLM request and scheduling objects into the
# metadata-only views kept by the server-side Scheduler. Together they carry
# one request from cache lookup through allocation and completion, while
# responses return only cache decisions and local block-pool updates.


@dataclass
class LookupRequestView:
    """Request fields used by lookup; prompt_token_ids preserves length only."""

    request_id: str
    prompt_token_ids: range
    block_hashes: list[BlockHash]
    num_tokens: int


def encode_lookup_request(
    registration: SchedulerRegistration,
    request: Request,
    num_computed_tokens: int,
) -> tuple[bytes, ...]:
    body = {
        "request": LookupRequestView(
            request_id=_validate_text(request.request_id, "request_id"),
            prompt_token_ids=range(len(request.prompt_token_ids)),
            block_hashes=[_validate_block_hash(value) for value in request.block_hashes],
            num_tokens=_validate_non_negative_int(request.num_tokens, "num_tokens"),
        ),
        "num_computed_tokens": _validate_non_negative_int(num_computed_tokens, "num_computed_tokens"),
    }
    return _encode_scheduler_request(registration, body, KVCacheMethod.LOOKUP)


def decode_lookup_request(payloads: tuple[bytes, ...]) -> tuple[SchedulerIdentity, str, LookupRequestView, int]:
    method = KVCacheMethod.LOOKUP
    identity, session_id, body = _decode_scheduler_request(payloads, method)
    request, num_computed_tokens = _body_fields(body, method.value, "request", "num_computed_tokens")
    _require_type(request, LookupRequestView, "request")
    return identity, session_id, request, _decode_non_negative_int_value(num_computed_tokens, "num_computed_tokens")


def encode_lookup_response(matched_tokens: int, is_async: bool) -> tuple[bytes, ...]:
    return _encode_response(
        KVCacheMethod.LOOKUP,
        {
            "matched_tokens": _validate_non_negative_int(matched_tokens, "matched_tokens"),
            "is_async": _validate_bool(is_async, "is_async"),
        },
    )


def decode_lookup_response(responses: Sequence[bytes]) -> tuple[int, bool]:
    body = _decode_response(responses, KVCacheMethod.LOOKUP)
    matched_tokens, is_async = _body_fields(body, "LOOKUP response", "matched_tokens", "is_async")
    return (_decode_non_negative_int_value(matched_tokens, "matched_tokens"), _decode_bool_value(is_async, "is_async"))


def encode_update_state_after_alloc(
    registration: SchedulerRegistration,
    request: Request,
    blocks: KVCacheBlocks,
    num_external_tokens: int,
) -> tuple[bytes, ...]:
    prompt_token_ids = list(request.prompt_token_ids)
    num_external_tokens = _validate_non_negative_int(num_external_tokens, "num_external_tokens")
    body = {
        "request": RequestView(
            request_id=_validate_text(request.request_id, "request_id"),
            prompt_token_ids=prompt_token_ids,
            block_hashes=[_validate_block_hash(value) for value in request.block_hashes],
            num_prompt_tokens=len(prompt_token_ids),
            num_tokens=_validate_non_negative_int(request.num_tokens, "num_tokens"),
            all_token_ids=list(prompt_token_ids),
        ),
        "blocks": BlocksView([list(group) for group in blocks.get_block_ids()] if num_external_tokens > 0 else []),
        "num_external_tokens": num_external_tokens,
    }
    return _encode_scheduler_request(registration, body, KVCacheMethod.UPDATE_STATE_AFTER_ALLOC)


def decode_update_state_after_alloc(
    payloads: tuple[bytes, ...],
) -> tuple[SchedulerIdentity, str, RequestView, BlocksView, int]:
    method = KVCacheMethod.UPDATE_STATE_AFTER_ALLOC
    identity, session_id, body = _decode_scheduler_request(payloads, method)
    request, blocks, num_external_tokens = _body_fields(body, method.value, "request", "blocks", "num_external_tokens")
    _require_type(request, RequestView, "request")
    _require_type(blocks, BlocksView, "blocks")
    return (
        identity,
        session_id,
        request,
        blocks,
        _decode_non_negative_int_value(num_external_tokens, "num_external_tokens"),
    )


def encode_build_connector_meta_request(
    registration: SchedulerRegistration,
    scheduler_output: SchedulerOutput,
    new_token_ids: dict[str, list[int]],
) -> tuple[bytes, ...]:
    cached = scheduler_output.scheduled_cached_reqs
    output = SchedulerOutputView(
        finished_req_ids=set(scheduler_output.finished_req_ids or ()),
        preempted_req_ids=set(scheduler_output.preempted_req_ids or ()),
        num_scheduled_tokens=dict(scheduler_output.num_scheduled_tokens),
        scheduled_new_reqs=[
            ScheduledNewReqPayload(
                req.req_id,
                req.num_computed_tokens,
                [list(group) for group in (req.block_ids or ())],
            )
            for req in scheduler_output.scheduled_new_reqs
        ],
        scheduled_cached_reqs=CachedReqsView(
            req_ids=list(cached.req_ids),
            new_block_ids=[
                None if blocks is None else [list(group) for group in blocks] for blocks in cached.new_block_ids
            ],
            num_computed_tokens=list(cached.num_computed_tokens),
            new_token_ids={req_id: list(tokens) for req_id, tokens in new_token_ids.items()},
        ),
    )
    return _encode_scheduler_request(registration, {"output": output}, KVCacheMethod.BUILD_CONNECTOR_META)


def decode_build_connector_meta_request(
    payloads: tuple[bytes, ...],
) -> tuple[SchedulerIdentity, str, SchedulerOutputView]:
    method = KVCacheMethod.BUILD_CONNECTOR_META
    identity, session_id, body = _decode_scheduler_request(payloads, method)
    (output,) = _body_fields(body, method.value, "output")
    _require_type(output, SchedulerOutputView, "output")
    return identity, session_id, output


def encode_build_connector_meta_response(metadata, touch_block_ids: list[int]) -> tuple[bytes, ...]:
    return _encode_response(
        KVCacheMethod.BUILD_CONNECTOR_META,
        {"metadata": metadata, "touch_block_ids": list(touch_block_ids)},
    )


def decode_build_connector_meta_response(responses: Sequence[bytes]) -> tuple:
    body = _decode_response(responses, KVCacheMethod.BUILD_CONNECTOR_META)
    return _body_fields(body, "BUILD_CONNECTOR_META response", "metadata", "touch_block_ids")


def encode_request_finished(
    registration: SchedulerRegistration,
    request_id: str,
    block_ids: list[int] | tuple[list[int], ...],
    all_groups: bool,
) -> tuple[bytes, ...]:
    all_groups = _validate_bool(all_groups, "all_groups")
    body = {
        "request_id": _validate_text(request_id, "request_id"),
        "block_ids": _validate_block_ids(block_ids, all_groups),
        "all_groups": all_groups,
    }
    return _encode_scheduler_request(registration, body, KVCacheMethod.REQUEST_FINISHED)


def decode_request_finished(
    payloads: tuple[bytes, ...],
) -> tuple[SchedulerIdentity, str, str, list[int] | tuple[list[int], ...], bool]:
    method = KVCacheMethod.REQUEST_FINISHED
    identity, session_id, body = _decode_scheduler_request(payloads, method)
    request_id, block_ids, all_groups = _body_fields(body, method.value, "request_id", "block_ids", "all_groups")
    decoded_all_groups = _decode_bool_value(all_groups, "all_groups")
    return (
        identity,
        session_id,
        _decode_text_value(request_id, "request_id"),
        _decode_block_ids(block_ids, decoded_all_groups),
        decoded_all_groups,
    )


def encode_request_finished_response(delay_free: bool, extra: dict | None) -> tuple[bytes, ...]:
    return _encode_response(
        KVCacheMethod.REQUEST_FINISHED,
        {"delay_free": _validate_bool(delay_free, "delay_free"), "extra": extra},
    )


def decode_request_finished_response(responses: Sequence[bytes]) -> tuple[bool, dict | None]:
    body = _decode_response(responses, KVCacheMethod.REQUEST_FINISHED)
    delay_free, extra = _body_fields(body, "REQUEST_FINISHED response", "delay_free", "extra")
    return _decode_bool_value(delay_free, "delay_free"), extra


def encode_update_connector_output(
    registration: SchedulerRegistration,
    completed_events: dict[int, int],
) -> tuple[bytes, ...]:
    return _encode_scheduler_request(
        registration,
        {"completed_events": dict(completed_events)},
        KVCacheMethod.UPDATE_CONNECTOR_OUTPUT,
    )


def decode_update_connector_output(payloads: tuple[bytes, ...]) -> tuple[SchedulerIdentity, str, dict[int, int]]:
    method = KVCacheMethod.UPDATE_CONNECTOR_OUTPUT
    identity, session_id, body = _decode_scheduler_request(payloads, method)
    (completed_events,) = _body_fields(body, method.value, "completed_events")
    if not isinstance(completed_events, dict):
        raise MPProtocolError(f"completed_events must be a dict, got {type(completed_events).__name__}")
    return identity, session_id, completed_events


def encode_update_connector_output_response(free_block_ids: list[int]) -> tuple[bytes, ...]:
    return _encode_response(KVCacheMethod.UPDATE_CONNECTOR_OUTPUT, {"free_block_ids": list(free_block_ids)})


def decode_update_connector_output_response(responses: Sequence[bytes]) -> list[int]:
    body = _decode_response(responses, KVCacheMethod.UPDATE_CONNECTOR_OUTPUT)
    (free_block_ids,) = _body_fields(body, "UPDATE_CONNECTOR_OUTPUT response", "free_block_ids")
    return _decode_list(free_block_ids, "free_block_ids")


# ==============================
# Worker transfer lifecycle
# ==============================

# Worker RPCs coordinate transfer work after the fixed cache mapping is active.
# Messages carry serializable metadata and NPU IPC descriptions across the
# process boundary; imported tensors, events, transfer threads, and progress
# remain owned by the server-side Worker.


def encode_wait_for_save_request(
    registration: WorkerRegistration,
    metadata: AscendConnectorMetadata,
    event_spec: NPUEventSpec,
) -> tuple[bytes, ...]:
    _validate_type(metadata, AscendConnectorMetadata, "metadata")
    _validate_type(event_spec, NPUEventSpec, "event_spec")
    return _encode_worker_request(
        registration,
        {"metadata": metadata, "event_spec": event_spec},
        KVCacheMethod.WAIT_FOR_SAVE,
    )


def decode_wait_for_save_request(
    payloads: tuple[bytes, ...],
) -> tuple[WorkerIdentity, str, AscendConnectorMetadata, NPUEventSpec]:
    method = KVCacheMethod.WAIT_FOR_SAVE
    identity, session_id, body = _decode_worker_request(payloads, method)
    metadata, event_spec = _body_fields(body, method.value, "metadata", "event_spec")
    _require_type(metadata, AscendConnectorMetadata, "metadata")
    _require_type(event_spec, NPUEventSpec, "event_spec")
    return identity, session_id, metadata, event_spec


def encode_get_finished_request(
    registration: WorkerRegistration,
    finished_req_ids: set[str],
    metadata: AscendConnectorMetadata,
) -> tuple[bytes, ...]:
    _validate_type(metadata, AscendConnectorMetadata, "metadata")
    return _encode_worker_request(
        registration,
        {"finished_req_ids": _validate_text_set(finished_req_ids, "finished_req_ids"), "metadata": metadata},
        KVCacheMethod.GET_FINISHED,
    )


def decode_get_finished_request(
    payloads: tuple[bytes, ...],
) -> tuple[WorkerIdentity, str, set[str], AscendConnectorMetadata]:
    method = KVCacheMethod.GET_FINISHED
    identity, session_id, body = _decode_worker_request(payloads, method)
    finished_req_ids, metadata = _body_fields(body, method.value, "finished_req_ids", "metadata")
    _require_type(metadata, AscendConnectorMetadata, "metadata")
    return identity, session_id, _decode_text_set(finished_req_ids, "finished_req_ids"), metadata


def encode_get_finished_response(done_sending: set[str], done_recving: set[str]) -> tuple[bytes, ...]:
    return _encode_response(
        KVCacheMethod.GET_FINISHED,
        {
            "done_sending": _validate_text_set(done_sending, "done_sending"),
            "done_recving": _validate_text_set(done_recving, "done_recving"),
        },
    )


def decode_get_finished_response(responses: Sequence[bytes]) -> tuple[set[str], set[str]]:
    body = _decode_response(responses, KVCacheMethod.GET_FINISHED)
    done_sending, done_recving = _body_fields(body, "GET_FINISHED response", "done_sending", "done_recving")
    return _decode_text_set(done_sending, "done_sending"), _decode_text_set(done_recving, "done_recving")


def encode_build_connector_worker_meta_request(registration: WorkerRegistration) -> tuple[bytes, ...]:
    return _encode_empty_worker_request(registration, KVCacheMethod.BUILD_CONNECTOR_WORKER_META)


def decode_build_connector_worker_meta_request(payloads: tuple[bytes, ...]) -> tuple[WorkerIdentity, str]:
    return _decode_empty_worker_request(payloads, KVCacheMethod.BUILD_CONNECTOR_WORKER_META)


def encode_build_connector_worker_meta_response(
    metadata: AscendStoreKVConnectorWorkerMetadata | None,
) -> tuple[bytes, ...]:
    if metadata is not None and not isinstance(metadata, AscendStoreKVConnectorWorkerMetadata):
        raise TypeError(f"metadata must be AscendStoreKVConnectorWorkerMetadata or None, got {type(metadata).__name__}")
    return _encode_response(KVCacheMethod.BUILD_CONNECTOR_WORKER_META, {"metadata": metadata})


def decode_build_connector_worker_meta_response(
    responses: Sequence[bytes],
) -> AscendStoreKVConnectorWorkerMetadata | None:
    body = _decode_response(responses, KVCacheMethod.BUILD_CONNECTOR_WORKER_META)
    (metadata,) = _body_fields(body, "BUILD_CONNECTOR_WORKER_META response", "metadata")
    if metadata is not None:
        _require_type(metadata, AscendStoreKVConnectorWorkerMetadata, "metadata")
    return metadata


def encode_get_kv_events_request(registration: WorkerRegistration) -> tuple[bytes, ...]:
    return _encode_empty_worker_request(registration, KVCacheMethod.GET_KV_EVENTS)


def decode_get_kv_events_request(payloads: tuple[bytes, ...]) -> tuple[WorkerIdentity, str]:
    return _decode_empty_worker_request(payloads, KVCacheMethod.GET_KV_EVENTS)


def encode_get_kv_events_response(events: list[BlockStored]) -> tuple[bytes, ...]:
    _validate_type(events, list, "events")
    for event in events:
        _validate_type(event, BlockStored, "event")
    return _encode_response(KVCacheMethod.GET_KV_EVENTS, {"events": events})


def decode_get_kv_events_response(responses: Sequence[bytes]) -> list[BlockStored]:
    body = _decode_response(responses, KVCacheMethod.GET_KV_EVENTS)
    (events,) = _body_fields(body, "GET_KV_EVENTS response", "events")
    if not isinstance(events, list):
        raise MPProtocolError(f"events must be a list, got {type(events).__name__}")
    for event in events:
        _require_type(event, BlockStored, "event")
    return events


def encode_start_load_kv_request(
    registration: WorkerRegistration,
    metadata: AscendConnectorMetadata,
) -> tuple[bytes, ...]:
    _validate_type(metadata, AscendConnectorMetadata, "metadata")
    return _encode_worker_request(registration, {"metadata": metadata}, KVCacheMethod.START_LOAD_KV)


def decode_start_load_kv_request(payloads: tuple[bytes, ...]) -> tuple[WorkerIdentity, str, AscendConnectorMetadata]:
    method = KVCacheMethod.START_LOAD_KV
    identity, session_id, body = _decode_worker_request(payloads, method)
    (metadata,) = _body_fields(body, method.value, "metadata")
    _require_type(metadata, AscendConnectorMetadata, "metadata")
    return identity, session_id, metadata


def encode_wait_for_layer_load_request(registration: WorkerRegistration) -> tuple[bytes, ...]:
    return _encode_empty_worker_request(registration, KVCacheMethod.WAIT_FOR_LAYER_LOAD)


def decode_wait_for_layer_load_request(payloads: tuple[bytes, ...]) -> tuple[WorkerIdentity, str]:
    return _decode_empty_worker_request(payloads, KVCacheMethod.WAIT_FOR_LAYER_LOAD)


def encode_save_kv_layer_request(registration: WorkerRegistration, event_spec: NPUEventSpec) -> tuple[bytes, ...]:
    _validate_type(event_spec, NPUEventSpec, "event_spec")
    return _encode_worker_request(registration, {"event_spec": event_spec}, KVCacheMethod.SAVE_KV_LAYER)


def decode_save_kv_layer_request(payloads: tuple[bytes, ...]) -> tuple[WorkerIdentity, str, NPUEventSpec]:
    method = KVCacheMethod.SAVE_KV_LAYER
    identity, session_id, body = _decode_worker_request(payloads, method)
    (event_spec,) = _body_fields(body, method.value, "event_spec")
    _require_type(event_spec, NPUEventSpec, "event_spec")
    return identity, session_id, event_spec


def encode_get_block_ids_with_load_errors_request(registration: WorkerRegistration) -> tuple[bytes, ...]:
    return _encode_empty_worker_request(registration, KVCacheMethod.GET_BLOCK_IDS_WITH_LOAD_ERRORS)


def decode_get_block_ids_with_load_errors_request(payloads: tuple[bytes, ...]) -> tuple[WorkerIdentity, str]:
    return _decode_empty_worker_request(payloads, KVCacheMethod.GET_BLOCK_IDS_WITH_LOAD_ERRORS)


def encode_get_block_ids_with_load_errors_response(block_ids: set[int]) -> tuple[bytes, ...]:
    return _encode_response(
        KVCacheMethod.GET_BLOCK_IDS_WITH_LOAD_ERRORS,
        {"block_ids": _validate_non_negative_int_set(block_ids, "block_ids")},
    )


def decode_get_block_ids_with_load_errors_response(responses: Sequence[bytes]) -> set[int]:
    body = _decode_response(responses, KVCacheMethod.GET_BLOCK_IDS_WITH_LOAD_ERRORS)
    (block_ids,) = _body_fields(body, "GET_BLOCK_IDS_WITH_LOAD_ERRORS response", "block_ids")
    return _decode_non_negative_int_set(block_ids, "block_ids")


# ==============================
# Identity-keyed RPC envelopes
# ==============================

# Scheduler and Worker envelopes expose their logical identity in fixed leading
# frames, followed by the session and serialized body. Route key factories read
# only that prefix on the RPC I/O thread, letting the affinity executor run work
# for one service in order before a handler decodes the full request.


def scheduler_affinity_key(_client_identity: bytes, payloads: tuple[bytes, ...]) -> SchedulerIdentity:
    return _decode_scheduler_identity(payloads)


def worker_affinity_key(_client_identity: bytes, payloads: tuple[bytes, ...]) -> WorkerIdentity:
    return _decode_worker_identity(payloads)


def _encode_scheduler_request(
    registration: SchedulerRegistration, body: dict, method: KVCacheMethod
) -> tuple[bytes, ...]:
    return _encode_scheduler_envelope(
        registration.identity,
        registration.session_id,
        _encode_body(body, method.value),
    )


def _decode_scheduler_request(
    payloads: tuple[bytes, ...], method: KVCacheMethod
) -> tuple[SchedulerIdentity, str, dict]:
    identity, session_id, payload = _decode_scheduler_envelope(payloads, method.value)
    return identity, session_id, _decode_body(payload, method.value)


def _encode_worker_request(registration: WorkerRegistration, body: dict, method: KVCacheMethod) -> tuple[bytes, ...]:
    return _encode_worker_envelope(registration.identity, registration.session_id, _encode_body(body, method.value))


def _decode_worker_request(payloads: tuple[bytes, ...], method: KVCacheMethod) -> tuple[WorkerIdentity, str, dict]:
    identity, session_id, payload = _decode_worker_envelope(payloads, method.value)
    return identity, session_id, _decode_body(payload, method.value)


def _encode_empty_worker_request(registration: WorkerRegistration, method: KVCacheMethod) -> tuple[bytes, ...]:
    return _encode_worker_request(registration, {}, method)


def _decode_empty_worker_request(payloads: tuple[bytes, ...], method: KVCacheMethod) -> tuple[WorkerIdentity, str]:
    identity, session_id, body = _decode_worker_request(payloads, method)
    if body:
        raise MPProtocolError(f"{method.value} body must be empty")
    return identity, session_id


def _encode_scheduler_envelope(identity: SchedulerIdentity, session_id: str, body: bytes) -> tuple[bytes, ...]:
    return *_encode_scheduler_identity(identity), _encode_text(session_id, "session_id"), body


def _decode_scheduler_envelope(payloads: tuple[bytes, ...], method: str) -> tuple[SchedulerIdentity, str, bytes]:
    _require_payload_count(payloads, _SCHEDULER_REQUEST_PAYLOADS, method)
    return _decode_scheduler_identity(payloads), _decode_text(payloads[2], "session_id"), payloads[3]


def _encode_worker_envelope(identity: WorkerIdentity, session_id: str, body: bytes) -> tuple[bytes, ...]:
    return *_encode_worker_identity(identity), _encode_text(session_id, "session_id"), body


def _decode_worker_envelope(payloads: tuple[bytes, ...], method: str) -> tuple[WorkerIdentity, str, bytes]:
    _require_payload_count(payloads, _WORKER_REQUEST_PAYLOADS, method)
    return _decode_worker_identity(payloads), _decode_text(payloads[3], "session_id"), payloads[4]


def _require_payload_count(payloads: Sequence[bytes], expected: int, method: str) -> None:
    if len(payloads) != expected:
        raise MPProtocolError(f"{method} expects {expected} payloads, got {len(payloads)}")


def _encode_scheduler_identity(identity: SchedulerIdentity) -> tuple[bytes, ...]:
    return (
        _encode_text(identity.engine_id, "engine_id"),
        _encode_non_negative_int(identity.data_parallel_rank, "data_parallel_rank"),
    )


def _decode_scheduler_identity(payloads: Sequence[bytes]) -> SchedulerIdentity:
    if len(payloads) < 2:
        raise MPProtocolError(f"Scheduler identity expects at least 2 payloads, got {len(payloads)}")
    return SchedulerIdentity(
        engine_id=_decode_text(payloads[0], "engine_id"),
        data_parallel_rank=_decode_non_negative_int(payloads[1], "data_parallel_rank"),
    )


def _encode_worker_identity(identity: WorkerIdentity) -> tuple[bytes, ...]:
    return (
        _encode_text(identity.engine_id, "engine_id"),
        _encode_non_negative_int(identity.rank, "rank"),
        _encode_non_negative_int(identity.data_parallel_rank, "data_parallel_rank"),
    )


def _decode_worker_identity(payloads: Sequence[bytes]) -> WorkerIdentity:
    if len(payloads) < 3:
        raise MPProtocolError(f"Worker identity expects at least 3 payloads, got {len(payloads)}")
    return WorkerIdentity(
        engine_id=_decode_text(payloads[0], "engine_id"),
        rank=_decode_non_negative_int(payloads[1], "rank"),
        data_parallel_rank=_decode_non_negative_int(payloads[2], "data_parallel_rank"),
    )


# ==============================
# Serialized bodies and boundary validation
# ==============================

# Every business body is one cloudpickled dictionary frame and every structured
# response is one frame. Local validators raise ordinary Python errors; decoders
# reuse those constraints where fields are shared and report malformed peer data
# as MPProtocolError, keeping protocol failures distinct from service failures.


def decode_ack_response(responses: Sequence[bytes], method: KVCacheMethod) -> None:
    response = _single_response(responses, method.value)
    if response != ACK_RESPONSE:
        raise MPProtocolError(f"{method.value} expects an OK response, got {response!r}")


def _encode_response(method: KVCacheMethod, body: dict) -> tuple[bytes, ...]:
    return (_encode_body(body, f"{method.value} response"),)


def _decode_response(responses: Sequence[bytes], method: KVCacheMethod) -> dict:
    name = f"{method.value} response"
    return _decode_body(_single_response(responses, method.value), name)


def _single_response(responses: Sequence[bytes], method: str) -> bytes:
    if len(responses) != 1:
        raise MPProtocolError(f"{method} expects 1 response payload, got {len(responses)}")
    return responses[0]


def _encode_body(value: dict, method: str) -> bytes:
    try:
        return cloudpickle.dumps(value)
    except Exception as exc:
        raise MPProtocolError(f"Failed to encode {method} body") from exc


def _decode_body(payload: bytes, method: str) -> dict:
    try:
        value = cloudpickle.loads(payload)
    except Exception as exc:
        raise MPProtocolError(f"Failed to decode {method} body") from exc
    if not isinstance(value, dict):
        raise MPProtocolError(f"{method} body must be a dict, got {type(value).__name__}")
    return value


def _body_fields(body: dict, method: str, *fields: str) -> tuple:
    missing = [field for field in fields if field not in body]
    if missing:
        raise MPProtocolError(f"{method} body is missing fields: {', '.join(missing)}")
    return tuple(body[field] for field in fields)


def _decode_validated(validator: Callable[..., _ValidatedT], *args: object) -> _ValidatedT:
    try:
        return validator(*args)
    except (TypeError, ValueError) as exc:
        raise MPProtocolError(str(exc)) from exc


def _validate_type(value: object, expected_type: type[_ValidatedT], field_name: str) -> _ValidatedT:
    if not isinstance(value, expected_type):
        raise TypeError(f"{field_name} must be {expected_type.__name__}, got {type(value).__name__}")
    return value


def _require_type(value: object, expected_type: type[_ValidatedT], field_name: str) -> None:
    _decode_validated(_validate_type, value, expected_type, field_name)


def _require_empty_body(body: bytes, method: str) -> None:
    if body:
        raise MPProtocolError(f"{method} body must be empty")


def _encode_non_negative_int(value: int, field_name: str) -> bytes:
    value = _validate_non_negative_int(value, field_name)
    try:
        return value.to_bytes(_INTEGER_BYTES, byteorder=_BYTE_ORDER)
    except OverflowError as exc:
        raise ValueError(f"{field_name} is too large: {value}") from exc


def _decode_non_negative_int(payload: bytes, field_name: str) -> int:
    if not isinstance(payload, bytes):
        raise MPProtocolError(f"{field_name} payload must be bytes, got {type(payload).__name__}")
    if len(payload) != _INTEGER_BYTES:
        raise MPProtocolError(f"{field_name} payload must contain {_INTEGER_BYTES} bytes, got {len(payload)}")
    return int.from_bytes(payload, byteorder=_BYTE_ORDER)


def _validate_non_negative_int(value: int, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{field_name} must not be negative, got {value}")
    return value


def _decode_non_negative_int_value(value, field_name: str) -> int:
    return _decode_validated(_validate_non_negative_int, value, field_name)


def _encode_text(value: str, field_name: str) -> bytes:
    return _validate_text(value, field_name).encode()


def _decode_text(payload: bytes, field_name: str) -> str:
    try:
        value = payload.decode()
    except (AttributeError, UnicodeDecodeError) as exc:
        raise MPProtocolError(f"{field_name} payload must be valid UTF-8 bytes") from exc
    return _decode_text_value(value, field_name)


def _validate_text(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string, got {type(value).__name__}")
    if not value:
        raise ValueError(f"{field_name} must not be empty")
    return value


def _decode_text_value(value, field_name: str) -> str:
    return _decode_validated(_validate_text, value, field_name)


def _validate_block_hash(value: BlockHash) -> BlockHash:
    if not isinstance(value, bytes):
        raise TypeError(f"block_hash must be bytes, got {type(value).__name__}")
    if not value:
        raise ValueError("block_hash must not be empty")
    return value


def _validate_text_set(value: set[str], field_name: str) -> set[str]:
    if not isinstance(value, set):
        raise TypeError(f"{field_name} must be a set, got {type(value).__name__}")
    return {_validate_text(item, f"{field_name} item") for item in value}


def _decode_text_set(value, field_name: str) -> set[str]:
    return _decode_validated(_validate_text_set, value, field_name)


def _validate_non_negative_int_set(value: set[int], field_name: str) -> set[int]:
    if not isinstance(value, set):
        raise TypeError(f"{field_name} must be a set, got {type(value).__name__}")
    return {_validate_non_negative_int(item, f"{field_name} item") for item in value}


def _decode_non_negative_int_set(value, field_name: str) -> set[int]:
    return _decode_validated(_validate_non_negative_int_set, value, field_name)


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_block_ids(
    block_ids: list[int] | tuple[list[int], ...], all_groups: bool
) -> list[int] | tuple[list[int], ...]:
    expected_shape = "a sequence of per-group block id sequences" if all_groups else "a sequence of block ids"
    if not isinstance(block_ids, Sequence) or isinstance(block_ids, (str, bytes)):
        raise TypeError(f"block_ids must be {expected_shape}, got {type(block_ids).__name__}")
    if all_groups:
        for group_ids in block_ids:
            if not isinstance(group_ids, Sequence) or isinstance(group_ids, (str, bytes)):
                raise TypeError(f"block_ids groups must be integer sequences, got {type(group_ids).__name__}")
            if any(not _is_int(block_id) for block_id in group_ids):
                raise TypeError("block_ids groups must contain integers only")
    elif any(not _is_int(block_id) for block_id in block_ids):
        raise TypeError("block_ids must contain integers only")
    return block_ids


def _decode_block_ids(
    block_ids: list[int] | tuple[list[int], ...], all_groups: bool
) -> list[int] | tuple[list[int], ...]:
    return _decode_validated(_validate_block_ids, block_ids, all_groups)


def _validate_bool(value: bool, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a boolean, got {type(value).__name__}")
    return value


def _decode_bool_value(value, field_name: str) -> bool:
    return _decode_validated(_validate_bool, value, field_name)


def _decode_list(value, field_name: str) -> list:
    if not isinstance(value, (list, tuple)):
        raise MPProtocolError(f"{field_name} must be a list, got {type(value).__name__}")
    return list(value)
