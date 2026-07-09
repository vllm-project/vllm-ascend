# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import torch
from vllm.logger import logger

from vllm_ascend import envs as ascend_envs

MAX_DUMP_ITEMS = 16
MAX_KV_SAMPLE_BLOCKS = 2
MAX_KV_SAMPLE_TENSORS = 4
MAX_KV_SAMPLE_VALUES = 16


class MooncakeDFXEvent(str, Enum):
    KV_CONTENT_CHECK = "kv_content_check"
    FAILURE = "failure"


class MooncakeFailureReason(str, Enum):
    KV_CONTENT_MISMATCH = "kv_content_mismatch"
    KV_CONTENT_CHECKSUM_ERROR = "kv_content_checksum_error"


@dataclass
class MooncakeDFXRecord:
    event: MooncakeDFXEvent
    request_id: str | None = None
    remote_request_id: str | None = None
    role: str | None = None
    reason: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


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
    return value


def _record_to_dict(record: MooncakeDFXRecord) -> dict[str, Any]:
    data = asdict(record)
    return {key: _sanitize_for_dump(value) for key, value in data.items() if value is not None and value != {}}


class MooncakeTransferDFX:
    def record(self, record: MooncakeDFXRecord) -> None:
        if not is_mooncake_transfer_dfx_enabled():
            return
        try:
            logger.info("[MooncakeTransferDFX] %s", json.dumps(_record_to_dict(record), sort_keys=True, default=str))
        except Exception as err:
            logger.debug("Failed to emit Mooncake transfer DFX record: %s", err)

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


mooncake_transfer_dfx = MooncakeTransferDFX()
