# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any

import torch
from vllm.logger import logger

from vllm_ascend import envs as ascend_envs

MAX_DUMP_ITEMS = 16


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
    if hasattr(value, "__struct_fields__"):
        return {
            field: _sanitize_for_dump(getattr(value, field))
            for field in value.__struct_fields__
        }
    return value


def _record(event: str, *, role: str | None = None, details: dict[str, Any] | None = None) -> None:
    if not is_mooncake_transfer_dfx_enabled():
        return
    payload = {"event": event, "role": role, "details": details or {}}
    try:
        logger.info("[MooncakeTransferDFX] %s", json.dumps(_sanitize_for_dump(payload), sort_keys=True, default=str))
    except Exception as err:
        logger.debug("Failed to emit Mooncake transfer DFX record: %s", err)


def dump_metadata(
    *,
    label: str,
    metadata: Any,
    role: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    details = {
        "label": label,
        "metadata": _sanitize_for_dump(metadata),
    }
    if extra:
        details.update(extra)
    _record("metadata_dump", role=role, details=details)


def _iter_cache_tensors(kv_caches: Mapping[str, Any]) -> list[Any]:
    cache_tensors: list[Any] = []
    for cache_or_caches in kv_caches.values():
        if hasattr(cache_or_caches, "data_ptr"):
            cache_tensors.append(cache_or_caches)
            continue
        for cache in cache_or_caches:
            cache_tensors.append(cache)
    return cache_tensors


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


def _block_nbytes(cache: Any) -> int:
    return int(cache[0].numel() * cache.element_size())


def _tensor_to_raw_bytes(tensor: Any) -> bytes:
    return tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()


def compute_kv_cache_checksum(
    *,
    kv_caches: Mapping[str, Any],
    block_groups: Sequence[Any],
    tp_num_need_pulls: int = 1,
    inner_offset: int = 0,
    cache_start: int = 0,
    cache_end: int | None = None,
) -> dict[str, Any] | None:
    if not is_mooncake_transfer_dfx_enabled():
        return None
    if tp_num_need_pulls <= 0:
        raise ValueError(f"tp_num_need_pulls must be positive, got {tp_num_need_pulls}")

    normalized_groups = _normalize_block_groups(block_groups)
    selected_tensors = _iter_cache_tensors(kv_caches)[cache_start:cache_end]
    digest = hashlib.sha256()
    total_bytes = 0
    segments = 0

    for cache in selected_tensors:
        block_len = _block_nbytes(cache)
        inner_block_len = block_len // tp_num_need_pulls
        byte_start = inner_offset * inner_block_len
        byte_end = byte_start + inner_block_len
        for block_group in normalized_groups:
            for block_id in block_group:
                block_bytes = _tensor_to_raw_bytes(cache[block_id])
                segment = block_bytes[byte_start:byte_end]
                digest.update(segment)
                total_bytes += len(segment)
                segments += 1

    return {
        "algorithm": "sha256",
        "digest": digest.hexdigest(),
        "bytes": total_bytes,
        "segments": segments,
        "cache_tensors": len(selected_tensors),
        "block_groups": len(normalized_groups),
        "tp_num_need_pulls": tp_num_need_pulls,
        "inner_offset": inner_offset,
        "cache_start": cache_start,
        "cache_end": cache_end,
    }


def record_kv_content_check(
    *,
    request_id: str,
    remote_request_id: str,
    source_checksum: Mapping[str, Any] | None,
    target_checksum: Mapping[str, Any] | None,
    role: str | None = None,
    extra: dict[str, Any] | None = None,
) -> bool:
    if not is_mooncake_transfer_dfx_enabled():
        return True

    passed = (
        source_checksum is not None
        and target_checksum is not None
        and source_checksum.get("digest") == target_checksum.get("digest")
        and source_checksum.get("bytes") == target_checksum.get("bytes")
        and source_checksum.get("segments") == target_checksum.get("segments")
    )
    details: dict[str, Any] = {
        "request_id": request_id,
        "remote_request_id": remote_request_id,
        "passed": passed,
        "source_checksum": source_checksum,
        "target_checksum": target_checksum,
    }
    if extra:
        details.update(extra)
    _record("kv_content_checksum", role=role, details=details)
    return passed
