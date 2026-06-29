# SPDX-License-Identifier: Apache-2.0

import contextlib
import hashlib
import json
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import torch
from vllm.logger import logger

from vllm_ascend import envs as ascend_envs

MAX_DUMP_ITEMS = 16
MAX_STATE_HISTORY = 32
MAX_KV_SAMPLE_BLOCKS = 2
MAX_KV_SAMPLE_TENSORS = 4
MAX_KV_SAMPLE_VALUES = 16


class MooncakeDFXEvent(str, Enum):
    STATE_TRANSITION = "state_transition"
    METADATA_DUMP = "metadata_dump"
    CONSISTENCY_CHECK = "consistency_check"
    TRANSFER_PLAN = "transfer_plan"
    TRANSFER_RESULT = "transfer_result"
    KV_CONTENT_CHECK = "kv_content_check"
    LIFECYCLE = "lifecycle"
    FAILURE = "failure"


class MooncakePDState(str, Enum):
    SCHEDULED = "scheduled"
    PREFILL_FINISHED = "prefill_finished"
    DELAYED_FREE = "delayed_free"
    FORCE_FREED = "force_freed"
    RECV_QUEUED = "recv_queued"
    METADATA_REQUESTED = "metadata_requested"
    METADATA_RECEIVED = "metadata_received"
    TRANSFER_STARTED = "transfer_started"
    TRANSFER_SUCCEEDED = "transfer_succeeded"
    TRANSFER_FAILED = "transfer_failed"
    DONE_RECVING_SENT = "done_recving_sent"
    DONE_RECVING_ACKED = "done_recving_acked"
    REMOTE_RELEASED = "remote_released"
    FINISHED = "finished"


class MooncakeFailureReason(str, Enum):
    METADATA_MISSING = "metadata_missing"
    BLOCK_MISMATCH = "block_mismatch"
    TRANSFER_ENGINE_ERROR = "transfer_engine_error"
    SOCKET_ERROR = "socket_error"
    INVALID_METADATA = "invalid_metadata"
    UNEXPECTED_MESSAGE = "unexpected_message"
    KV_CONTENT_MISMATCH = "kv_content_mismatch"
    KV_CONTENT_CHECKSUM_ERROR = "kv_content_checksum_error"


@dataclass
class MooncakeDFXRecord:
    event: MooncakeDFXEvent
    request_id: str | None = None
    remote_request_id: str | None = None
    role: str | None = None
    state: MooncakePDState | None = None
    previous_state: MooncakePDState | None = None
    reason: str | None = None
    elapsed_ms: float | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class RequestStateHistory:
    current_state: MooncakePDState | None = None
    transitions: list[tuple[float, MooncakePDState]] = field(default_factory=list)


def is_mooncake_transfer_dfx_enabled() -> bool:
    return bool(ascend_envs.VLLM_ASCEND_MOONCAKE_TRANSFER_DFX)


def _shorten_sequence(value: Sequence[Any]) -> dict[str, Any]:
    items = list(value[:MAX_DUMP_ITEMS])
    return {
        "len": len(value),
        "items": [_sanitize_for_dump(item) for item in items],
        "truncated": len(value) > MAX_DUMP_ITEMS,
    }


def _sanitize_for_dump(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _sanitize_for_dump(item) for key, item in list(value.items())[:MAX_DUMP_ITEMS]}
    if isinstance(value, (list, tuple)):
        return _shorten_sequence(value)
    if isinstance(value, set):
        return _shorten_sequence(sorted(value, key=str))
    if hasattr(value, "__dict__"):
        return _sanitize_for_dump(vars(value))
    return value


def _record_to_dict(record: MooncakeDFXRecord) -> dict[str, Any]:
    data = asdict(record)
    return {key: _sanitize_for_dump(value) for key, value in data.items() if value is not None and value != {}}


class MooncakeTransferDFX:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._request_states: dict[str, RequestStateHistory] = {}

    def record(self, record: MooncakeDFXRecord) -> None:
        if not is_mooncake_transfer_dfx_enabled():
            return
        try:
            logger.info("[MooncakeTransferDFX] %s", json.dumps(_record_to_dict(record), sort_keys=True, default=str))
        except Exception as err:
            logger.debug("Failed to emit Mooncake transfer DFX record: %s", err)

    def update_state(
        self,
        request_id: str,
        state: MooncakePDState,
        *,
        remote_request_id: str | None = None,
        role: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        if not is_mooncake_transfer_dfx_enabled():
            return
        with self._lock:
            history = self._request_states.setdefault(request_id, RequestStateHistory())
            previous_state = history.current_state
            history.current_state = state
            history.transitions.append((time.time(), state))
            if len(history.transitions) > MAX_STATE_HISTORY:
                history.transitions = history.transitions[-MAX_STATE_HISTORY:]

        self.record(
            MooncakeDFXRecord(
                event=MooncakeDFXEvent.STATE_TRANSITION,
                request_id=request_id,
                remote_request_id=remote_request_id,
                role=role,
                state=state,
                previous_state=previous_state,
                details=details or {},
            )
        )

    def dump_metadata(
        self,
        *,
        label: str,
        request_id: str | None = None,
        remote_request_id: str | None = None,
        metadata: Any,
        role: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        if not is_mooncake_transfer_dfx_enabled():
            return
        details = {"label": label, "metadata": _sanitize_for_dump(metadata)}
        if extra:
            details.update(extra)
        self.record(
            MooncakeDFXRecord(
                event=MooncakeDFXEvent.METADATA_DUMP,
                request_id=request_id,
                remote_request_id=remote_request_id,
                role=role,
                details=details,
            )
        )

    def validate_block_mapping(
        self,
        *,
        request_id: str,
        local_block_ids: Sequence[Any],
        remote_block_ids: Sequence[Any],
        remote_request_id: str | None = None,
        role: str | None = None,
        allow_remote_superset: bool = True,
    ) -> list[str]:
        if not is_mooncake_transfer_dfx_enabled():
            return []

        issues: list[str] = []
        local_count = len(local_block_ids)
        remote_count = len(remote_block_ids)
        if local_count < 0 or remote_count < 0:
            issues.append("negative_block_count")
        if allow_remote_superset:
            if local_count > remote_count:
                issues.append("local_blocks_exceed_remote_blocks")
        elif local_count != remote_count:
            issues.append("local_remote_block_count_mismatch")
        if any(block_id is None for block_id in local_block_ids):
            issues.append("local_block_id_none")
        if any(block_id is None for block_id in remote_block_ids):
            issues.append("remote_block_id_none")

        self.record(
            MooncakeDFXRecord(
                event=MooncakeDFXEvent.CONSISTENCY_CHECK,
                request_id=request_id,
                remote_request_id=remote_request_id,
                role=role,
                reason=",".join(issues) if issues else None,
                details={
                    "check": "block_mapping",
                    "local_block_count": local_count,
                    "remote_block_count": remote_count,
                    "passed": not issues,
                },
            )
        )
        return issues

    def validate_agent_metadata(
        self,
        *,
        metadata: Any,
        request_id: str | None = None,
        role: str | None = None,
    ) -> list[str]:
        if not is_mooncake_transfer_dfx_enabled():
            return []

        issues: list[str] = []
        if not getattr(metadata, "engine_id", None):
            issues.append("missing_engine_id")
        if getattr(metadata, "te_rpc_port", None) is None:
            issues.append("missing_te_rpc_port")
        base_addrs = getattr(metadata, "kv_caches_base_addr", None)
        if base_addrs is None:
            issues.append("missing_kv_caches_base_addr")
        elif not isinstance(base_addrs, list):
            issues.append("invalid_kv_caches_base_addr_type")
        block_lens = getattr(metadata, "block_lens", None)
        if block_lens is not None and base_addrs is not None and isinstance(base_addrs, list):
            if len(block_lens) != len(base_addrs):
                issues.append("block_lens_base_addr_count_mismatch")

        self.record(
            MooncakeDFXRecord(
                event=MooncakeDFXEvent.CONSISTENCY_CHECK,
                request_id=request_id,
                role=role,
                reason=",".join(issues) if issues else None,
                details={"check": "agent_metadata", "passed": not issues},
            )
        )
        return issues

    def validate_kv_cache_registration(
        self,
        *,
        num_blocks: int,
        block_lens: Sequence[int],
        base_addrs: Sequence[int],
        role: str | None = None,
    ) -> list[str]:
        if not is_mooncake_transfer_dfx_enabled():
            return []

        issues: list[str] = []
        if num_blocks <= 0:
            issues.append("non_positive_num_blocks")
        if not block_lens:
            issues.append("empty_block_lens")
        if not base_addrs:
            issues.append("empty_base_addrs")
        if any(block_len <= 0 for block_len in block_lens):
            issues.append("non_positive_block_len")
        if len(set(base_addrs)) != len(base_addrs):
            issues.append("duplicate_base_addr")
        if block_lens and base_addrs and len(base_addrs) % len(block_lens) != 0:
            issues.append("base_addr_count_not_multiple_of_block_lens")

        self.record(
            MooncakeDFXRecord(
                event=MooncakeDFXEvent.CONSISTENCY_CHECK,
                role=role,
                reason=",".join(issues) if issues else None,
                details={
                    "check": "kv_cache_registration",
                    "num_blocks": num_blocks,
                    "block_lens": block_lens,
                    "base_addr_count": len(base_addrs),
                    "passed": not issues,
                },
            )
        )
        return issues

    def record_transfer_plan(
        self,
        *,
        request_id: str,
        remote_request_id: str,
        session_id: str,
        src_list: Sequence[int],
        dst_list: Sequence[int],
        length_list: Sequence[int],
        role: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        if not is_mooncake_transfer_dfx_enabled():
            return
        total_bytes = sum(length_list)
        details = {
            "session_id": session_id,
            "transfer_count": len(length_list),
            "total_bytes": total_bytes,
            "src": src_list,
            "dst": dst_list,
            "length": length_list,
        }
        if extra:
            details.update(extra)
        self.record(
            MooncakeDFXRecord(
                event=MooncakeDFXEvent.TRANSFER_PLAN,
                request_id=request_id,
                remote_request_id=remote_request_id,
                role=role,
                details=details,
            )
        )

    @staticmethod
    def _iter_cache_tensors(kv_caches: Mapping[str, Any]) -> list[Any]:
        cache_tensors: list[Any] = []
        for cache_or_caches in kv_caches.values():
            if hasattr(cache_or_caches, "data_ptr"):
                cache_tensors.append(cache_or_caches)
                continue
            for cache in cache_or_caches:
                cache_tensors.append(cache)
        return cache_tensors

    @staticmethod
    def _block_nbytes(cache: Any) -> int:
        return int(cache[0].numel() * cache.element_size())

    @staticmethod
    def _tensor_to_raw_bytes(tensor: Any) -> bytes:
        return tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()

    @staticmethod
    def _normalize_block_groups(block_groups: Sequence[Any]) -> list[list[int]]:
        normalized_groups: list[list[int]] = []
        for block_group in block_groups:
            if isinstance(block_group, (list, tuple)):
                normalized_groups.append([int(block_id) for block_id in block_group])
            elif hasattr(block_group, "tolist"):
                normalized_groups.append([int(block_id) for block_id in block_group.tolist()])
            else:
                normalized_groups.append([int(block_group)])
        return normalized_groups

    def compute_kv_cache_checksum(
        self,
        *,
        kv_caches: Mapping[str, Any],
        block_groups: Sequence[Any],
        tp_num_need_pulls: int,
        inner_offset: int,
        block_lens: Sequence[int],
        cache_start: int = 0,
        cache_end: int | None = None,
    ) -> dict[str, Any] | None:
        if not is_mooncake_transfer_dfx_enabled():
            return None

        if tp_num_need_pulls <= 0:
            raise ValueError(f"tp_num_need_pulls must be positive, got {tp_num_need_pulls}")

        normalized_groups = self._normalize_block_groups(block_groups)
        cache_tensors = self._iter_cache_tensors(kv_caches)
        selected_tensors = cache_tensors[cache_start:cache_end]
        if not selected_tensors:
            raise ValueError("no KV cache tensors selected for checksum")
        digest = hashlib.sha256()
        total_bytes = 0
        num_segments = 0

        for absolute_cache_idx, cache in enumerate(selected_tensors, start=cache_start):
            block_len = (
                int(block_lens[absolute_cache_idx % len(block_lens)])
                if block_lens
                else self._block_nbytes(cache)
            )
            inner_block_len = block_len // tp_num_need_pulls
            byte_start = inner_offset * inner_block_len
            byte_end = byte_start + inner_block_len
            for block_group in normalized_groups:
                for block_id in block_group:
                    block_bytes = self._tensor_to_raw_bytes(cache[block_id])
                    segment = block_bytes[byte_start:byte_end]
                    digest.update(segment)
                    total_bytes += len(segment)
                    num_segments += 1

        return {
            "algorithm": "sha256",
            "digest": digest.hexdigest(),
            "bytes": total_bytes,
            "segments": num_segments,
            "cache_tensors": len(selected_tensors),
            "block_groups": len(normalized_groups),
            "tp_num_need_pulls": tp_num_need_pulls,
            "inner_offset": inner_offset,
        }

    def sample_kv_cache_blocks(
        self,
        *,
        kv_caches: Mapping[str, Any],
        block_groups: Sequence[Any],
        tp_num_need_pulls: int,
        inner_offset: int,
        block_lens: Sequence[int],
        cache_start: int = 0,
        cache_end: int | None = None,
        max_tensors: int = MAX_KV_SAMPLE_TENSORS,
        max_blocks: int = MAX_KV_SAMPLE_BLOCKS,
        max_values: int = MAX_KV_SAMPLE_VALUES,
    ) -> list[dict[str, Any]]:
        if not is_mooncake_transfer_dfx_enabled():
            return []

        normalized_groups = self._normalize_block_groups(block_groups)
        block_ids = [block_id for block_group in normalized_groups for block_id in block_group][:max_blocks]
        cache_tensors = self._iter_cache_tensors(kv_caches)
        selected_tensors = cache_tensors[cache_start:cache_end][:max_tensors]
        samples: list[dict[str, Any]] = []

        for absolute_cache_idx, cache in enumerate(selected_tensors, start=cache_start):
            block_len = (
                int(block_lens[absolute_cache_idx % len(block_lens)])
                if block_lens
                else self._block_nbytes(cache)
            )
            inner_block_len = block_len // tp_num_need_pulls
            byte_start = inner_offset * inner_block_len
            byte_end = byte_start + inner_block_len
            for block_id in block_ids:
                block_tensor = cache[block_id].detach().contiguous().cpu()
                block_nbytes = int(block_tensor.numel() * block_tensor.element_size())
                segment_tensor = block_tensor.reshape(-1)
                element_size = int(block_tensor.element_size())
                elem_start = byte_start // element_size
                elem_end = min(byte_end // element_size, segment_tensor.numel())
                selected_values = segment_tensor[elem_start:elem_end]
                try:
                    sample_values = selected_values[:max_values].tolist()
                except TypeError:
                    sample_values = selected_values[:max_values].float().tolist()
                if hasattr(selected_values, "float") and selected_values.numel() > 0:
                    selected_float = selected_values.float()
                    min_value = float(selected_float.min().item())
                    max_value = float(selected_float.max().item())
                    mean_value = float(selected_float.mean().item())
                else:
                    min_value = max_value = mean_value = None
                samples.append(
                    {
                        "cache_index": absolute_cache_idx,
                        "block_id": block_id,
                        "dtype": str(block_tensor.dtype),
                        "shape": list(block_tensor.shape),
                        "byte_range": [byte_start, min(byte_end, block_nbytes)],
                        "element_range": [elem_start, elem_end],
                        "numel": int(selected_values.numel()),
                        "values": sample_values,
                        "min": min_value,
                        "max": max_value,
                        "mean": mean_value,
                    }
                )

        return samples

    def record_kv_content_check(
        self,
        *,
        request_id: str,
        remote_request_id: str,
        source_checksum: Mapping[str, Any] | None,
        target_checksum: Mapping[str, Any] | None,
        role: str | None = None,
        extra: dict[str, Any] | None = None,
        source_samples: Sequence[Mapping[str, Any]] | None = None,
        target_samples: Sequence[Mapping[str, Any]] | None = None,
    ) -> bool:
        if not is_mooncake_transfer_dfx_enabled():
            return True

        match = (
            source_checksum is not None
            and target_checksum is not None
            and source_checksum.get("algorithm") == target_checksum.get("algorithm")
            and source_checksum.get("digest") == target_checksum.get("digest")
            and source_checksum.get("bytes") == target_checksum.get("bytes")
            and source_checksum.get("segments") == target_checksum.get("segments")
        )
        mismatch_fields = []
        if source_checksum is None:
            mismatch_fields.append("source_checksum_missing")
        if target_checksum is None:
            mismatch_fields.append("target_checksum_missing")
        if source_checksum is not None and target_checksum is not None:
            for field_name in ("algorithm", "digest", "bytes", "segments"):
                if source_checksum.get(field_name) != target_checksum.get(field_name):
                    mismatch_fields.append(field_name)
        details: dict[str, Any] = {
            "passed": match,
            "source_checksum": source_checksum,
            "target_checksum": target_checksum,
        }
        if not match:
            details["mismatch_fields"] = mismatch_fields
            details["source_samples"] = source_samples or []
            details["target_samples"] = target_samples or []
        if extra:
            details.update(extra)
        self.record(
            MooncakeDFXRecord(
                event=MooncakeDFXEvent.KV_CONTENT_CHECK,
                request_id=request_id,
                remote_request_id=remote_request_id,
                role=role,
                reason=None if match else MooncakeFailureReason.KV_CONTENT_MISMATCH.value,
                details=details,
            )
        )
        return match

    @contextlib.contextmanager
    def transfer_span(
        self,
        *,
        request_id: str,
        remote_request_id: str,
        session_id: str,
        role: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> Iterator[None]:
        if not is_mooncake_transfer_dfx_enabled():
            yield
            return

        start_time = time.perf_counter()
        self.update_state(
            request_id,
            MooncakePDState.TRANSFER_STARTED,
            remote_request_id=remote_request_id,
            role=role,
            details={"session_id": session_id, **(details or {})},
        )
        try:
            yield
        except Exception as err:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self.update_state(
                request_id,
                MooncakePDState.TRANSFER_FAILED,
                remote_request_id=remote_request_id,
                role=role,
                details={"session_id": session_id, "error": str(err)},
            )
            self.record(
                MooncakeDFXRecord(
                    event=MooncakeDFXEvent.FAILURE,
                    request_id=request_id,
                    remote_request_id=remote_request_id,
                    role=role,
                    reason=MooncakeFailureReason.TRANSFER_ENGINE_ERROR.value,
                    elapsed_ms=elapsed_ms,
                    details={"session_id": session_id, "error": str(err)},
                )
            )
            raise
        else:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self.update_state(
                request_id,
                MooncakePDState.TRANSFER_SUCCEEDED,
                remote_request_id=remote_request_id,
                role=role,
                details={"session_id": session_id, "elapsed_ms": elapsed_ms},
            )
            self.record(
                MooncakeDFXRecord(
                    event=MooncakeDFXEvent.TRANSFER_RESULT,
                    request_id=request_id,
                    remote_request_id=remote_request_id,
                    role=role,
                    elapsed_ms=elapsed_ms,
                    details={"session_id": session_id, "success": True},
                )
            )


mooncake_transfer_dfx = MooncakeTransferDFX()
