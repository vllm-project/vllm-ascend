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

"""Short anomaly reports under ``runtime/report``."""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from vllm_ascend.runtime_guard.token_utils import decode_token_ids, is_int_list, is_list_of_int_lists
from vllm_ascend.logger import init_logger_ascend

logger = init_logger_ascend(__name__)

# Same (incident_type, req_id): wave-based backoff (not wall-clock).
# After the 1st write, next needs +64 waves; then +128, +256, ... (doubles).
# Cap is ``report.max_per_req`` (default 1); at cap, detection for that req stops.
_SAME_PAIR_BASE_WAVES = 64
_SAME_PAIR_MAX_DEFAULT = 1


def _same_pair_gap_waves(writes_so_far: int) -> int:
    """Waves required before the next write, given how many are already on disk."""
    if writes_so_far <= 0:
        return 0
    # 1→64, 2→128, 3→256, … ; saturate shift to avoid huge ints.
    shift = min(int(writes_so_far) - 1, 20)
    return int(_SAME_PAIR_BASE_WAVES) << shift


def _kv_dump_req_ids_for_report(
    *,
    req_id: str | None,
    detail: dict[str, Any] | None,
) -> list[str]:
    """Req ids under which ``dump_kv`` lands ``.pt`` / ``request_info``.

    Auto incidents dump under ``incident.req_id``. ``manual_trigger`` uses a
    synthetic incident id but ``scope=all_requests``, so KV lands under each
    real ``detail.requests[].req_id`` — report ``dump_dir`` must follow that.
    """
    from vllm_ascend.runtime_guard.manual_trigger import MANUAL_TRIGGER_REQ_ID

    out: list[str] = []
    if isinstance(detail, dict):
        rows = detail.get("requests")
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                rid = str(row.get("req_id") or "").strip()
                if rid and rid != MANUAL_TRIGGER_REQ_ID and rid not in out:
                    out.append(rid)
    if out:
        return out
    rid = str(req_id or "").strip()
    if rid:
        return [rid]
    return ["unknown"]


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
            logger.warning("[runtime_guard report] decode %s failed error=%s", key, exc)
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


def _json_default(value: Any) -> Any:
    # Forensic reports must never be lost to a stray np.int64 / torch scalar:
    # degrade via .item()/.tolist(), else fall back to repr().
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            pass
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return tolist()
        except Exception:
            pass
    return repr(value)


# Large / array-ish detail keys: keep after scalars so the top of the JSON is
# skimmable. Only keys still emitted by current report enrichment / IO.
_DETAIL_TAIL_ORDER = (
    "prompt_token_ids",
    "output_token_ids",
    "prompt_text",
    "output_text",
    "block_ids",
)


def _order_detail_tail(detail: dict[str, Any]) -> dict[str, Any]:
    """Scalars first; known large arrays last (missing keys are skipped)."""
    tail = {k: detail[k] for k in _DETAIL_TAIL_ORDER if k in detail}
    if not tail:
        return detail
    head = {k: v for k, v in detail.items() if k not in tail}
    return {**head, **tail}


def dumps_report_json(obj: Any, *, indent: int = 2) -> str:
    """Pretty-print JSON, but keep int arrays (token ids) compact on one line."""

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
            # Compact, no space after comma: [1,2,3] — keeps long token-id
            # walls short.
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        if is_list_of_int_lists(value):
            # Each inner token-id row compact; rows stacked for readability.
            if not value:
                return "[]"
            inner = ",\n".join(
                f"{sp_in}{json.dumps(row, ensure_ascii=False, separators=(',', ':'))}" for row in value
            )
            return "[\n" + inner + f"\n{sp}]"
        if isinstance(value, list):
            if not value:
                return "[]"
            parts = [f"{sp_in}{_format(v, level + 1)}" for v in value]
            return "[\n" + ",\n".join(parts) + f"\n{sp}]"
        return json.dumps(value, ensure_ascii=False, default=_json_default)

    return _format(obj, 0)


class ReportWriter:
    """Write short anomaly records under ``runtime/report``.

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
        max_per_req: int = _SAME_PAIR_MAX_DEFAULT,
        dump_root_provider: Callable[[], str | Path] | None = None,
    ) -> None:
        self.report_dir = Path(report_dir)
        self.save_sensitive_info = bool(save_sensitive_info)
        self.max_prompt_token_ids = int(max_prompt_token_ids)
        self.max_output_token_ids = int(max_output_token_ids)
        self.decode_token_ids = bool(decode_token_ids)
        self.max_per_req = max(1, int(max_per_req))
        # Called per write so a hot-reloaded dump.dump_dir takes effect at once.
        self._dump_root_provider = dump_root_provider
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self._pair_lock = threading.Lock()
        # (incident_type, req_id) → [last_wave, write_count]
        self._pair_state: dict[tuple[str, str], list[int]] = {}

    @staticmethod
    def _stop_detect_for_req(req_id: str | None) -> None:
        if not req_id:
            return
        try:
            from vllm_ascend.runtime_guard.request_state import RequestGuardStore

            RequestGuardStore.get().mark_detection_stopped(req_id)
        except Exception as exc:
            logger.debug(
                "[runtime_guard report] stop-detect mark failed req_id=%s error=%s",
                req_id,
                exc,
            )

    def write(
        self,
        *,
        incident_type: str,
        req_id: str | None = None,
        detail: dict[str, Any] | None = None,
        rank_tag: str | None = None,
        tokenizer: Any | None = None,
        dump_attempted: bool = False,
        dump_count: int | None = None,
        dump_max_times: int | None = None,
        dump_arm_wave: int | None = None,
    ) -> Path | None:
        """Write one pretty-printed anomaly JSON file. Returns path or None on failure.

        ``dump_attempted`` means ``dump_kv`` was in this incident's action list
        (not that D2H already succeeded — dump runs on a later ``sync_for_step``).
        ``dump_arm_wave`` is the real-step wave when the incident was handled;
        same-pair rate limiting uses it (64 waves, then doubles) up to
        ``max_per_req``.
        """
        pair = (str(incident_type or "unknown"), str(req_id or ""))
        wave_i: int | None = int(dump_arm_wave) if dump_arm_wave is not None else None
        max_per = max(1, int(self.max_per_req))
        wrote_count = 0
        prev_rec: list[int] | None = None
        with self._pair_lock:
            rec = self._pair_state.get(pair)
            if rec is not None:
                last_wave, count = int(rec[0]), int(rec[1])
                if count >= max_per:
                    logger.debug(
                        "[runtime_guard report] per-req cap hit (%d) type=%s req_id=%s; skip",
                        max_per,
                        pair[0],
                        pair[1],
                    )
                    # Cap already reached — keep detectors off for this req.
                    self._stop_detect_for_req(req_id)
                    return None
                gap = _same_pair_gap_waves(count)
                if wave_i is None:
                    logger.debug(
                        "[runtime_guard report] same-pair backoff needs wave "
                        "type=%s req_id=%s count=%d; skip",
                        pair[0],
                        pair[1],
                        count,
                    )
                    return None
                if wave_i - last_wave < gap:
                    logger.debug(
                        "[runtime_guard report] same-pair wave backoff "
                        "type=%s req_id=%s wave=%d last=%d need_gap=%d; skip",
                        pair[0],
                        pair[1],
                        wave_i,
                        last_wave,
                        gap,
                    )
                    return None
                prev_rec = [last_wave, count]
                rec[0] = wave_i
                rec[1] = count + 1
                wrote_count = rec[1]
            else:
                if len(self._pair_state) >= 8192:
                    # Bound the tracking dict on long-running servers.
                    for stale in sorted(self._pair_state.items(), key=lambda kv: kv[1][0])[: -4096]:
                        self._pair_state.pop(stale[0], None)
                # First write: record wave (0 if unknown) so later gaps are measurable.
                self._pair_state[pair] = [wave_i if wave_i is not None else 0, 1]
                wrote_count = 1
        try:
            type_dir = self.report_dir / str(incident_type or "unknown")
            type_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            req_tag = f"_{req_id}" if req_id else ""
            report_path = type_dir / f"report_{stamp}{req_tag}_pid{os.getpid()}.json"
            safe_detail = _order_detail_tail(
                sanitize_report_detail(
                    detail,
                    save_sensitive_info=self.save_sensitive_info,
                    max_prompt_token_ids=self.max_prompt_token_ids,
                    max_output_token_ids=self.max_output_token_ids,
                    decode_token_ids=self.decode_token_ids,
                    tokenizer=tokenizer if self.decode_token_ids else None,
                )
            )
            dump_root = (
                Path(self._dump_root_provider())
                if self._dump_root_provider is not None
                else self.report_dir / "kv_cache"
            )
            itype = str(incident_type or "unknown")
            dump_req_ids = _kv_dump_req_ids_for_report(
                req_id=req_id,
                detail=detail if isinstance(detail, dict) else None,
            )
            # Req root only (no wave_*). Arm wave ≠ drain wave when dump is
            # deferred (manual / next-flush); tools must pick wave_* with .pt.
            # ``dump_arm_wave`` remains the arm-time hint.
            dump_dirs = [str(dump_root / itype / rid) for rid in dump_req_ids]
            record = {
                "ts": datetime.now().isoformat(timespec="milliseconds"),
                "incident_type": incident_type,
                "req_id": req_id,
                "rank": rank_tag,
                "dump_attempted": bool(dump_attempted),
                "dump_arm_wave": int(dump_arm_wave) if dump_arm_wave is not None else None,
                "dump_count": int(dump_count) if dump_count is not None else None,
                "dump_max_times": int(dump_max_times) if dump_max_times is not None else None,
                # Req-id landing root(s); ``wave_*`` appears after flush.
                "dump_dir": dump_dirs[0],
                "decode_token_ids": self.decode_token_ids and self.save_sensitive_info,
                "max_prompt_token_ids": self.max_prompt_token_ids,
                "max_output_token_ids": self.max_output_token_ids,
                "detail": safe_detail,
            }
            # Surface block_ids next to req_id/dump_dir when present (K-13).
            if "block_ids" in safe_detail:
                record["block_ids"] = safe_detail["block_ids"]
            if len(dump_dirs) > 1:
                record["dump_dirs"] = dump_dirs
            text = dumps_report_json(record, indent=2)
            with report_path.open("w", encoding="utf-8") as f:
                f.write(text + "\n")
            logger.info(
                "[runtime_guard report] incident_type=%s req_id=%s path=%s dump_attempted=%s "
                "dump_count=%s/%s save_sensitive_info=%s decode_token_ids=%s "
                "max_prompt=%d max_output=%d",
                incident_type,
                req_id,
                report_path,
                record["dump_attempted"],
                record["dump_count"],
                record["dump_max_times"],
                self.save_sensitive_info,
                self.decode_token_ids and self.save_sensitive_info,
                self.max_prompt_token_ids,
                self.max_output_token_ids,
            )
            if wrote_count >= max_per:
                self._stop_detect_for_req(req_id)
            return report_path
        except Exception as exc:
            logger.error("[runtime_guard report] write failed dir=%s error=%s", self.report_dir, exc)
            # Roll the pair quota back: the count was committed before the write;
            # a failed write must not burn ``max_per_req`` (with the default 1 that
            # would permanently stop detection with zero reports on disk). Only
            # undo when our increment is still the latest (a concurrent later
            # write may have re-based the counter).
            with self._pair_lock:
                rec = self._pair_state.get(pair)
                if rec is not None and int(rec[1]) == wrote_count:
                    if prev_rec is None:
                        self._pair_state.pop(pair, None)
                    else:
                        rec[0] = prev_rec[0]
                        rec[1] = prev_rec[1]
            return None
