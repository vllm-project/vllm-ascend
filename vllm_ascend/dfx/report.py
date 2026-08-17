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

"""Short anomaly reports under ``dfx/report``."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from vllm_ascend.dfx.util import decode_token_ids, is_int_list, is_list_of_int_lists
from vllm_ascend.logger import init_logger_ascend

logger = init_logger_ascend(__name__)

# Keys that often carry raw token id lists (content / PII risk).
_TOKEN_ID_DETAIL_KEYS = frozenset(
    {
        "window_token_ids",
        "prompt_token_ids",
        "output_token_ids",
        "window_sampled_token_ids",
        "window_accepted_token_ids",
        "current_sampled_token_ids",
        "current_accepted_token_ids",
    }
)

_PROMPT_TOKEN_ID_KEYS = frozenset({"prompt_token_ids"})


def _token_list_len(value: Any) -> int:
    """Length for count derivation (flat list or list-of-lists)."""
    if not isinstance(value, list):
        return 0
    if value and isinstance(value[0], list):
        return sum(len(step) for step in value if isinstance(step, list))
    return len(value)


def _count_key_for_token_ids(key: str) -> str:
    """``prompt_token_ids`` → ``prompt_token_count``."""
    if key.endswith("_token_ids"):
        return f"{key[: -len('_ids')]}_count"
    return f"{key}_count"


def _truncate_token_ids_value(value: Any, max_len: int) -> tuple[Any, bool]:
    """Truncate a flat or nested token-id list. ``max_len<=0`` means unlimited."""
    if max_len <= 0 or not isinstance(value, list):
        return value, False
    if is_int_list(value):
        if len(value) <= max_len:
            return value, False
        return value[:max_len], True
    if is_list_of_int_lists(value):
        truncated = False
        out: list[list[int]] = []
        for step in value:
            if len(step) > max_len:
                out.append(step[:max_len])
                truncated = True
            else:
                out.append(list(step))
        return out, truncated
    return value, False


def _is_token_ids_key(key: Any) -> bool:
    s = str(key)
    return s in _TOKEN_ID_DETAIL_KEYS or s.endswith("_token_ids")


def _is_list_of_dicts(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(x, dict) for x in value)


def truncate_token_id_fields(
    detail: dict[str, Any],
    *,
    max_prompt_token_ids: int = 1000,
    max_output_token_ids: int = 1000,
) -> dict[str, Any]:
    """Cap prompt/output-like ``*_token_ids`` lists; keep full ``*_token_count``.

    Recurses into nested dicts and list-of-dicts (e.g. manual_trigger
    ``detail.requests[]``).
    """
    out = dict(detail)
    for key, value in list(out.items()):
        if isinstance(value, dict):
            out[key] = truncate_token_id_fields(
                value,
                max_prompt_token_ids=max_prompt_token_ids,
                max_output_token_ids=max_output_token_ids,
            )
            continue
        if _is_list_of_dicts(value):
            out[key] = [
                truncate_token_id_fields(
                    item,
                    max_prompt_token_ids=max_prompt_token_ids,
                    max_output_token_ids=max_output_token_ids,
                )
                for item in value
            ]
            continue
        if not _is_token_ids_key(key) or not isinstance(value, list):
            continue
        count_key = _count_key_for_token_ids(str(key))
        if count_key not in out:
            out[count_key] = _token_list_len(value)
        if key in _PROMPT_TOKEN_ID_KEYS or str(key).startswith("prompt_"):
            max_len = max_prompt_token_ids
        else:
            max_len = max_output_token_ids
        new_val, truncated = _truncate_token_ids_value(value, max_len)
        out[key] = new_val
        if truncated:
            out[f"{key}_truncated"] = True
            out[f"{key}_max"] = max_len
    return out


def _text_key_for_token_ids(ids_key: str, *, nested: bool) -> str:
    """``window_token_ids`` → ``window_text``; nested → ``window_sampled_texts``."""
    if ids_key.endswith("_token_ids"):
        base = ids_key[: -len("_token_ids")]
    else:
        base = ids_key
    return f"{base}_texts" if nested else f"{base}_text"


def decode_token_id_texts(
    detail: dict[str, Any],
    tokenizer: Any | None,
) -> dict[str, Any]:
    """Decode prompt/output/window/current ``*_token_ids`` into text fields.

    - Flat int list → ``*_text`` (string)
    - List of int lists (e.g. per-step window) → ``*_texts`` (list[str])
    - Nested dict / list-of-dicts (manual_trigger ``requests``) are walked.
    """
    if tokenizer is None:
        return detail
    out = dict(detail)
    for key, value in list(out.items()):
        if isinstance(value, dict):
            out[key] = decode_token_id_texts(value, tokenizer)
            continue
        if _is_list_of_dicts(value):
            out[key] = [decode_token_id_texts(item, tokenizer) for item in value]
            continue
        if not _is_token_ids_key(key):
            continue
        try:
            if is_int_list(value) and value:
                out[_text_key_for_token_ids(str(key), nested=False)] = decode_token_ids(tokenizer, value)
            elif is_list_of_int_lists(value) and value:
                texts: list[str] = []
                for step in value:
                    texts.append(decode_token_ids(tokenizer, step) if step else "")
                out[_text_key_for_token_ids(str(key), nested=True)] = texts
        except Exception as exc:
            logger.warning("[DFX report] decode %s failed error=%s", key, exc)
    return out


def sanitize_report_detail(
    detail: dict[str, Any] | None,
    *,
    save_sensitive_info: bool = False,
    max_prompt_token_ids: int = 1000,
    max_output_token_ids: int = 1000,
    decode_token_ids: bool = True,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    """Prepare anomaly detail for disk.

    - ``save_sensitive_info=false``: drop all token-id lists; keep / derive
      ``*_token_count`` (and non-token fields). No ``<redacted len=N>`` stubs.
    - ``save_sensitive_info=true``: keep token-id lists (truncated by max_*),
      optionally decode prompt/output/window/current ids to text.

    Nested dicts and list-of-dicts (e.g. ``detail.requests``) follow the same
    policy.
    """
    if not detail:
        return {}
    if save_sensitive_info:
        out = truncate_token_id_fields(
            detail,
            max_prompt_token_ids=max_prompt_token_ids,
            max_output_token_ids=max_output_token_ids,
        )
        if decode_token_ids:
            out = decode_token_id_texts(out, tokenizer)
        return out

    out: dict[str, Any] = {}
    for key, value in detail.items():
        if isinstance(value, dict):
            out[key] = sanitize_report_detail(
                value,
                save_sensitive_info=False,
                max_prompt_token_ids=max_prompt_token_ids,
                max_output_token_ids=max_output_token_ids,
                decode_token_ids=False,
                tokenizer=None,
            )
            continue
        if _is_list_of_dicts(value):
            out[key] = [
                sanitize_report_detail(
                    item,
                    save_sensitive_info=False,
                    max_prompt_token_ids=max_prompt_token_ids,
                    max_output_token_ids=max_output_token_ids,
                    decode_token_ids=False,
                    tokenizer=None,
                )
                for item in value
            ]
            continue
        if not _is_token_ids_key(key):
            out[key] = value
            continue
        count_key = _count_key_for_token_ids(str(key))
        if count_key not in detail and count_key not in out and isinstance(value, list):
            out[count_key] = _token_list_len(value)
        # Drop the token-id list itself (count already present or just derived).
    return out


def dumps_report_json(obj: Any, *, indent: int = 2) -> str:
    """Pretty-print JSON, but keep int arrays (token ids) on one line."""

    def _format(value: Any, level: int) -> str:
        sp = " " * (indent * level)
        sp_in = " " * (indent * (level + 1))
        if isinstance(value, dict):
            if not value:
                return "{}"
            parts = []
            for k, v in value.items():
                parts.append(f"{sp_in}{json.dumps(k, ensure_ascii=False)}: {_format(v, level + 1)}")
            return "{\n" + ",\n".join(parts) + f"\n{sp}}}"
        if is_int_list(value):
            # Compact: [1, 2, 3] on a single line.
            return json.dumps(value, ensure_ascii=False, separators=(", ", ": "))
        if is_list_of_int_lists(value):
            # Each inner token-id row compact; rows stacked for readability.
            if not value:
                return "[]"
            inner = ",\n".join(
                f"{sp_in}{json.dumps(row, ensure_ascii=False, separators=(', ', ': '))}" for row in value
            )
            return "[\n" + inner + f"\n{sp}]"
        if isinstance(value, list):
            if not value:
                return "[]"
            parts = [f"{sp_in}{_format(v, level + 1)}" for v in value]
            return "[\n" + ",\n".join(parts) + f"\n{sp}]"
        return json.dumps(value, ensure_ascii=False)

    return _format(obj, 0)


class DfxReportWriter:
    """Write short anomaly records under ``dfx/report``.

    Filenames include millisecond + pid so concurrent ranks do not collide on
    the same second-granularity stamp.
    """

    def __init__(
        self,
        report_dir: str | Path,
        *,
        save_sensitive_info: bool = False,
        max_prompt_token_ids: int = 1000,
        max_output_token_ids: int = 1000,
        decode_token_ids: bool = True,
    ) -> None:
        self.report_dir = Path(report_dir)
        self.save_sensitive_info = bool(save_sensitive_info)
        self.max_prompt_token_ids = int(max_prompt_token_ids)
        self.max_output_token_ids = int(max_output_token_ids)
        self.decode_token_ids = bool(decode_token_ids)
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        *,
        anomaly_type: str,
        req_id: str | None = None,
        detail: dict[str, Any] | None = None,
        rank_tag: str | None = None,
        tokenizer: Any | None = None,
        dump_attempted: bool = False,
        dump_armed: bool = False,
        dump_count: int | None = None,
        dump_max_times: int | None = None,
        dump_arm_wave: int | None = None,
    ) -> Path | None:
        """Write one pretty-printed anomaly JSON file. Returns path or None on failure.

        ``dump_armed=True`` (msprobe dump successfully armed for this event)
        adds a ``_dump`` marker in the filename so ops can grep dump-linked
        reports without opening each file. The report itself is still written
        immediately at detect / trigger time. When armed, ``dump_arm_wave``
        records the real-step wave index at arm (correlate with dump_finish).
        """
        try:
            self.report_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            dump_tag = "_dump" if dump_armed else ""
            report_path = self.report_dir / f"anomaly_{stamp}{dump_tag}_pid{os.getpid()}.log"
            safe_detail = sanitize_report_detail(
                detail,
                save_sensitive_info=self.save_sensitive_info,
                max_prompt_token_ids=self.max_prompt_token_ids,
                max_output_token_ids=self.max_output_token_ids,
                decode_token_ids=self.decode_token_ids,
                tokenizer=tokenizer if self.decode_token_ids else None,
            )
            attempted = bool(dump_attempted or dump_armed)
            armed = bool(dump_armed)
            record = {
                "ts": datetime.now().isoformat(timespec="milliseconds"),
                "anomaly_type": anomaly_type,
                "req_id": req_id,
                "rank": rank_tag,
                "dump_attempted": attempted,
                "dump_armed": armed,
                "dump_arm_wave": int(dump_arm_wave) if dump_arm_wave is not None else None,
                "dump_count": int(dump_count) if dump_count is not None else None,
                "dump_max_times": int(dump_max_times) if dump_max_times is not None else None,
                "detail": safe_detail,
                "decode_token_ids": self.decode_token_ids and self.save_sensitive_info,
                "max_prompt_token_ids": self.max_prompt_token_ids,
                "max_output_token_ids": self.max_output_token_ids,
            }
            text = dumps_report_json(record, indent=2)
            with report_path.open("w", encoding="utf-8") as f:
                f.write(text + "\n")
            logger.info(
                "[DFX report] anomaly_type=%s req_id=%s path=%s dump_attempted=%s dump_armed=%s "
                "dump_count=%s/%s save_sensitive_info=%s decode_token_ids=%s "
                "max_prompt=%d max_output=%d",
                anomaly_type,
                req_id,
                report_path,
                attempted,
                armed,
                record["dump_count"],
                record["dump_max_times"],
                self.save_sensitive_info,
                self.decode_token_ids and self.save_sensitive_info,
                self.max_prompt_token_ids,
                self.max_output_token_ids,
            )
            return report_path
        except Exception as exc:
            logger.error("[DFX report] write failed dir=%s error=%s", self.report_dir, exc)
            return None

    def write_dump_finish(
        self,
        *,
        req_id: str,
        detail: dict[str, Any] | None = None,
        rank_tag: str | None = None,
        tokenizer: Any | None = None,
        anomaly_type: str | None = None,
        source: str | None = None,
        dump_arm_wave: int | None = None,
        dump_activate_wave: int | None = None,
        dump_waves_after_report: int | None = None,
        dump_count: int | None = None,
        finish_wave: int | None = None,
    ) -> Path | None:
        """Write dump-linked finish sidecar with output + wave stamps.

        Called from reap (``_reap_finished_requests``) for reqs that successfully activated a
        dump. Does not rewrite the immediate anomaly report. Token ids follow
        the same ``save_sensitive_info`` / ``max_*`` / ``decode_token_ids``
        policy as anomaly reports.
        """
        try:
            self.report_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            safe_req = "".join(c if c.isalnum() or c in "-_" else "_" for c in (req_id or "unknown"))[:64]
            report_path = self.report_dir / f"dump_finish_{stamp}_{safe_req}_pid{os.getpid()}.log"
            raw_detail = dict(detail or {})
            safe_detail = sanitize_report_detail(
                raw_detail,
                save_sensitive_info=self.save_sensitive_info,
                max_prompt_token_ids=self.max_prompt_token_ids,
                max_output_token_ids=self.max_output_token_ids,
                decode_token_ids=self.decode_token_ids,
                tokenizer=tokenizer if self.decode_token_ids and self.save_sensitive_info else None,
            )
            record = {
                "ts": datetime.now().isoformat(timespec="milliseconds"),
                "kind": "dump_finish",
                "anomaly_type": anomaly_type,
                "source": source,
                "req_id": req_id,
                "rank": rank_tag,
                "dump_arm_wave": dump_arm_wave,
                "dump_activate_wave": dump_activate_wave,
                "dump_waves_after_report": dump_waves_after_report,
                "dump_finish_wave": finish_wave,
                "dump_count": dump_count,
                "detail": safe_detail,
                "decode_token_ids": self.decode_token_ids and self.save_sensitive_info,
                "max_prompt_token_ids": self.max_prompt_token_ids,
                "max_output_token_ids": self.max_output_token_ids,
            }
            text = dumps_report_json(record, indent=2)
            with report_path.open("w", encoding="utf-8") as f:
                f.write(text + "\n")
            logger.info(
                "[DFX dump_finish] req_id=%s path=%s arm_wave=%s activate_wave=%s "
                "waves_after_report=%s finish_wave=%s dump_count=%s",
                req_id,
                report_path,
                dump_arm_wave,
                dump_activate_wave,
                dump_waves_after_report,
                finish_wave,
                dump_count,
            )
            return report_path
        except Exception as exc:
            logger.error("[DFX dump_finish] write failed dir=%s req_id=%s error=%s", self.report_dir, req_id, exc)
            return None
