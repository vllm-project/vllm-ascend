#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""KV-dump on-disk writers (skip markers, request_info) shared across runtime_guard."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from vllm_ascend.logger import init_logger_ascend

logger = init_logger_ascend(__name__)


def kv_dump_wave_dirname(wave: int | None) -> str:
    """Subdir under ``{type}/{req_id}/`` separating dumps across steps."""
    if wave is None:
        return "wave_unknown"
    return f"wave_{int(wave)}"


def write_kv_dump_skipped(
    dump_root: str | Path,
    *,
    req_id: str,
    incident_type: str,
    reason: str,
    stage: str = "arm",
    rank_tag: str = "",
    wave: int | None = None,
    detail: dict[str, Any] | None = None,
) -> Path | None:
    """Record why KV dump did not fully succeed on this rank.

    Layout (same tree as ``.pt`` when ``rank_tag`` is set)::

      {dump_root}/{incident_type}/{req_id}/wave_<N>/[rank_tag/]dump_skipped.json

    Arm skips (leader) and drain skips (each last-PP TP) all use this helper.
    ``dump_skipped`` means incomplete / failed — ``request_info`` or some
    ``.pt`` files may already exist beside it.
    """
    if not req_id:
        return None
    out_dir = Path(dump_root) / str(incident_type or "unknown") / str(req_id) / kv_dump_wave_dirname(wave)
    if rank_tag:
        out_dir = out_dir / str(rank_tag)
    path = out_dir / "dump_skipped.json"
    payload: dict[str, Any] = {
        "reason": str(reason or "unknown"),
        "req_id": str(req_id),
        "incident_type": str(incident_type or "unknown"),
        "stage": str(stage or "arm"),
        "rank_tag": str(rank_tag or ""),
        "dump_arm_wave": int(wave) if wave is not None else None,
        "ts": time.time(),
    }
    if detail:
        payload["detail"] = dict(detail)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        logger.warning(
            "[runtime_guard dump_kv] skipped reason=%s stage=%s req_id=%s type=%s rank=%s marker=%s",
            reason,
            stage,
            req_id,
            incident_type,
            rank_tag or "-",
            path,
        )
        return path
    except OSError as exc:
        logger.warning(
            "[runtime_guard dump_kv] failed to write skip marker req_id=%s path=%s: %s",
            req_id,
            path,
            exc,
        )
        return None


def write_kv_dump_request_info(
    dump_root: str | Path,
    *,
    req_id: str,
    incident_type: str,
    detail: dict[str, Any] | None,
    rank_tag: str = "",
    wave: int | None = None,
    block_ids: list[int] | None = None,
    tokenizer: Any | None = None,
    save_sensitive_info: bool = False,
    decode_token_ids: bool = True,
    max_prompt_token_ids: int = 1000,
    max_output_token_ids: int = 1000,
) -> Path | None:
    """Last-PP TP0: write report-like request metadata next to KV ``.pt`` shards.

    Path: ``{dump_root}/{incident_type}/{req_id}/{wave_N}/request_info.json``
    """
    if not req_id:
        return None
    from vllm_ascend.observability.runtime_guard.report import dumps_report_json, sanitize_report_detail

    out_dir = Path(dump_root) / str(incident_type or "unknown") / str(req_id) / kv_dump_wave_dirname(wave)
    path = out_dir / "request_info.json"
    safe_detail = sanitize_report_detail(
        detail,
        save_sensitive_info=save_sensitive_info,
        max_prompt_token_ids=max_prompt_token_ids,
        max_output_token_ids=max_output_token_ids,
        decode_token_ids=decode_token_ids and save_sensitive_info,
        tokenizer=tokenizer if (decode_token_ids and save_sensitive_info) else None,
    )
    payload = {
        "ts": time.time(),
        "incident_type": str(incident_type or "unknown"),
        "req_id": str(req_id),
        "rank": str(rank_tag or ""),
        "dump_arm_wave": int(wave) if wave is not None else None,
        "block_ids": list(block_ids) if block_ids is not None else safe_detail.get("block_ids"),
        "decode_token_ids": bool(decode_token_ids and save_sensitive_info),
        "max_prompt_token_ids": int(max_prompt_token_ids),
        "max_output_token_ids": int(max_output_token_ids),
        "detail": safe_detail,
    }
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(dumps_report_json(payload, indent=2) + "\n", encoding="utf-8")
        logger.info(
            "[runtime_guard dump_kv] request_info req_id=%s type=%s path=%s",
            req_id,
            incident_type,
            path,
        )
        return path
    except OSError as exc:
        logger.warning(
            "[runtime_guard dump_kv] failed to write request_info req_id=%s path=%s: %s",
            req_id,
            path,
            exc,
        )
        return None
