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


class MooncakeDFXErrorCode(str, Enum):
    OK = "ErrorCode_0000"
    SOURCE_CHECKSUM_UNAVAILABLE = "ErrorCode_1001"
    TARGET_CHECKSUM_UNAVAILABLE = "ErrorCode_1002"
    SOURCE_CHECKSUM_ERROR = "ErrorCode_1003"
    TARGET_CHECKSUM_ERROR = "ErrorCode_1004"
    CHECKSUM_ALGORITHM_MISMATCH = "ErrorCode_1101"
    CHECKSUM_DIGEST_MISMATCH = "ErrorCode_1102"
    CHECKSUM_BYTES_MISMATCH = "ErrorCode_1103"
    CHECKSUM_SEGMENTS_MISMATCH = "ErrorCode_1104"
    METADATA_ENCODE_ERROR = "ErrorCode_2001"
    METADATA_SEND_ERROR = "ErrorCode_2002"
    METADATA_RECV_ERROR = "ErrorCode_2003"
    METADATA_DECODE_ERROR = "ErrorCode_2004"
    METADATA_VALIDATE_ERROR = "ErrorCode_2005"


ERROR_CODE_REASONS = {
    MooncakeDFXErrorCode.OK.value: "ok",
    MooncakeDFXErrorCode.SOURCE_CHECKSUM_UNAVAILABLE.value: "source checksum is unavailable",
    MooncakeDFXErrorCode.TARGET_CHECKSUM_UNAVAILABLE.value: "target checksum is unavailable",
    MooncakeDFXErrorCode.SOURCE_CHECKSUM_ERROR.value: "source checksum computation failed",
    MooncakeDFXErrorCode.TARGET_CHECKSUM_ERROR.value: "target checksum computation failed",
    MooncakeDFXErrorCode.CHECKSUM_ALGORITHM_MISMATCH.value: "source and target checksum algorithms mismatch",
    MooncakeDFXErrorCode.CHECKSUM_DIGEST_MISMATCH.value: "source and target checksum digests mismatch",
    MooncakeDFXErrorCode.CHECKSUM_BYTES_MISMATCH.value: "source and target checksum byte counts mismatch",
    MooncakeDFXErrorCode.CHECKSUM_SEGMENTS_MISMATCH.value: "source and target checksum segment counts mismatch",
    MooncakeDFXErrorCode.METADATA_ENCODE_ERROR.value: "metadata encode failed",
    MooncakeDFXErrorCode.METADATA_SEND_ERROR.value: "metadata send failed",
    MooncakeDFXErrorCode.METADATA_RECV_ERROR.value: "metadata receive failed",
    MooncakeDFXErrorCode.METADATA_DECODE_ERROR.value: "metadata decode failed",
    MooncakeDFXErrorCode.METADATA_VALIDATE_ERROR.value: "metadata validation failed",
}


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
    error_codes = [MooncakeDFXErrorCode.OK.value]
    details = {
        "label": label,
        "passed": True,
        "error_code": error_codes[0],
        "error_codes": error_codes,
        "error_reasons": {code: ERROR_CODE_REASONS.get(code, "unknown") for code in error_codes},
        "metadata": _sanitize_for_dump(metadata),
    }
    if extra:
        details.update(extra)
    _record("metadata_dump", role=role, details=details)


def record_metadata_error(
    *,
    label: str,
    error_code: MooncakeDFXErrorCode,
    error: Any,
    role: str | None = None,
    metadata: Any | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    error_codes = [error_code.value]
    details: dict[str, Any] = {
        "label": label,
        "passed": False,
        "error_code": error_codes[0],
        "error_codes": error_codes,
        "error_reasons": {code: ERROR_CODE_REASONS.get(code, "unknown") for code in error_codes},
        "error": str(error),
    }
    if metadata is not None:
        details["metadata"] = _sanitize_for_dump(metadata)
    if extra:
        details.update(extra)
    _record("metadata_error", role=role, details=details)


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


def make_checksum_error(error_code: MooncakeDFXErrorCode, error: Any) -> dict[str, Any]:
    return {
        "error_code": error_code.value,
        "error": str(error),
    }


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


def _append_checksum_error_code(
    checksum: Mapping[str, Any] | None,
    *,
    unavailable_code: MooncakeDFXErrorCode,
    default_error_code: MooncakeDFXErrorCode,
    error_codes: list[str],
) -> bool:
    if checksum is None:
        error_codes.append(unavailable_code.value)
        return True
    if checksum.get("error") is not None:
        error_codes.append(str(checksum.get("error_code", default_error_code.value)))
        return True
    return False


def _get_kv_content_error_codes(
    source_checksum: Mapping[str, Any] | None,
    target_checksum: Mapping[str, Any] | None,
) -> list[str]:
    error_codes: list[str] = []
    source_failed = _append_checksum_error_code(
        source_checksum,
        unavailable_code=MooncakeDFXErrorCode.SOURCE_CHECKSUM_UNAVAILABLE,
        default_error_code=MooncakeDFXErrorCode.SOURCE_CHECKSUM_ERROR,
        error_codes=error_codes,
    )
    target_failed = _append_checksum_error_code(
        target_checksum,
        unavailable_code=MooncakeDFXErrorCode.TARGET_CHECKSUM_UNAVAILABLE,
        default_error_code=MooncakeDFXErrorCode.TARGET_CHECKSUM_ERROR,
        error_codes=error_codes,
    )
    if source_failed or target_failed:
        return error_codes

    assert source_checksum is not None
    assert target_checksum is not None
    if source_checksum.get("algorithm") != target_checksum.get("algorithm"):
        error_codes.append(MooncakeDFXErrorCode.CHECKSUM_ALGORITHM_MISMATCH.value)
    if source_checksum.get("digest") != target_checksum.get("digest"):
        error_codes.append(MooncakeDFXErrorCode.CHECKSUM_DIGEST_MISMATCH.value)
    if source_checksum.get("bytes") != target_checksum.get("bytes"):
        error_codes.append(MooncakeDFXErrorCode.CHECKSUM_BYTES_MISMATCH.value)
    if source_checksum.get("segments") != target_checksum.get("segments"):
        error_codes.append(MooncakeDFXErrorCode.CHECKSUM_SEGMENTS_MISMATCH.value)
    return error_codes or [MooncakeDFXErrorCode.OK.value]


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

    error_codes = _get_kv_content_error_codes(source_checksum, target_checksum)
    passed = error_codes == [MooncakeDFXErrorCode.OK.value]
    details: dict[str, Any] = {
        "request_id": request_id,
        "remote_request_id": remote_request_id,
        "passed": passed,
        "error_code": error_codes[0],
        "error_codes": error_codes,
        "error_reasons": {code: ERROR_CODE_REASONS.get(code, "unknown") for code in error_codes},
        "source_checksum": source_checksum,
        "target_checksum": target_checksum,
    }
    if extra:
        details.update(extra)
    _record("kv_content_checksum", role=role, details=details)
    return passed
