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

"""Output substring / token-id sequence anomaly detector."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from vllm_ascend.logger import init_logger_ascend
from vllm_ascend.runtime_config._validate import normalize_raw_patterns
from vllm_ascend.runtime_guard.detector.base import ConfigBackedDetector, resolve_batch_req_ids
from vllm_ascend.runtime_guard.incident import ILL_TYPE_NONE, Incident
from vllm_ascend.runtime_guard.io_snapshot import RequestIoSnapshotManager, output_token_count_for_request
from vllm_ascend.runtime_guard.token_utils import (
    decode_token_ids,
    load_model_tokenizer,
    normalize_token_ids,
)

if TYPE_CHECKING:
    from vllm_ascend.runtime_config.config import RuntimeConfig

logger = init_logger_ascend(__name__)

SourceKind = Literal["text", "token_ids"]


@dataclass(frozen=True)
class CompiledOutputPattern:
    """One pattern with both text and token-id views after config refresh."""

    index: int
    source: SourceKind
    text: str
    token_ids: tuple[int, ...]
    raw: Any  # original JSON entry (str or list)


def contains_token_subsequence(haystack: list[int], needle: list[int] | tuple[int, ...]) -> bool:
    """True if ``needle`` appears as a contiguous subsequence of ``haystack``."""
    n = len(needle)
    if n == 0 or n > len(haystack):
        return False
    if n == 1:
        return needle[0] in haystack
    limit = len(haystack) - n + 1
    return any(all(haystack[start + j] == needle[j] for j in range(n)) for start in range(limit))


def contains_prefix(haystack: list[int], needle: list[int] | tuple[int, ...]) -> bool:
    """True if ``needle`` is a prefix (contiguous, at position 0) of ``haystack``."""
    n = len(needle)
    if n == 0 or n > len(haystack):
        return False
    return all(haystack[i] == needle[i] for i in range(n))


def _encode_text(tokenizer: Any, text: str, *, add_special_tokens: bool) -> list[int]:
    ids = tokenizer.encode(text, add_special_tokens=add_special_tokens)
    return [int(x) for x in ids]


class OutputSubstringDetector(ConfigBackedDetector):
    """Detect configured substrings / token-id sequences in generated output.

    Patterns may be authored as text (``str``) or token ids (``list[int]``).
    On config refresh the detector encode/decodes each pattern and logs both
    views. Matching is on contiguous **token ids** in the cumulative output
    (no per-step decode). Each ``req_id`` alerts at most once.
    """

    incident_type = "output_substring"
    section_key = "output_substring"

    def __init__(
        self,
        *,
        runtime_config: RuntimeConfig | None = None,
        runner: Any | None = None,
        tokenizer_provider: Callable[[], Any | None] | None = None,
    ) -> None:
        super().__init__(runtime_config=runtime_config, runner=runner, enabled=False)
        self._tokenizer_provider = tokenizer_provider
        self._raw_patterns: list[Any] = []
        self._add_special_tokens = False
        # true: patterns match only at the start of the cumulative output (prefix);
        # false (default): match anywhere as a contiguous token-id subsequence.
        self._match_prefix = False
        self._compiled: list[CompiledOutputPattern] = []
        self._compile_fp: tuple[Any, ...] | None = None
        self._alerted: set[str] = set()
        self._tokenizer: Any | None = None
        self._tokenizer_failed = False
        if runtime_config is not None:
            self.refresh_from_config()

    def _apply_detector_values(self, getter: Callable[[str, Any], Any]) -> None:
        # Retry the tokenizer on every config apply: a latch set during an early
        # provider/load failure (runner not yet ready) must not silence text
        # patterns for the whole process lifetime.
        self._tokenizer_failed = False
        raw = getter("patterns", self._raw_patterns)
        try:
            self._raw_patterns = normalize_raw_patterns(raw)
        except ValueError as exc:
            logger.error("[runtime_guard: output_substring] invalid patterns: %s; keeping previous", exc)
        self._add_special_tokens = bool(getter("add_special_tokens", self._add_special_tokens))
        self._match_prefix = bool(getter("match_prefix", self._match_prefix))
        # Rebuild / re-log when enable flips on or patterns / encode knobs change.
        self._maybe_compile_patterns(force_log=False)

    def clear_finished(self, req_id: str) -> None:
        self._alerted.discard(req_id)

    def _get_tokenizer(self) -> Any | None:
        if self._tokenizer is not None:
            return self._tokenizer
        if self._tokenizer_failed:
            return None
        tok: Any | None = None
        if self._tokenizer_provider is not None:
            try:
                tok = self._tokenizer_provider()
            except Exception as exc:
                logger.warning("[runtime_guard: output_substring] tokenizer_provider failed error=%s", exc)
                self._tokenizer_failed = True
                return None
        if tok is None:
            try:
                tok = load_model_tokenizer(self._runner)
            except Exception as exc:
                logger.warning("[runtime_guard: output_substring] tokenizer load failed error=%s", exc)
                self._tokenizer_failed = True
                return None
            if tok is None:
                # runner / model_config missing; retry on next refresh/check.
                return None
        self._tokenizer = tok
        return tok

    def _maybe_compile_patterns(self, *, force_log: bool) -> None:
        fp = (
            self._enabled,
            self._add_special_tokens,
            tuple(p if isinstance(p, str) else tuple(p) for p in self._raw_patterns),
        )
        if not force_log and fp == self._compile_fp and self._compiled:
            return
        self._compile_fp = fp
        self._compiled = []
        if not self._enabled or not self._raw_patterns:
            return
        tokenizer = self._get_tokenizer()
        if tokenizer is None:
            logger.info_once(
                "[runtime_guard: output_substring] patterns pending: tokenizer not ready yet "
                "(will retry on next refresh/check)"
            )
            self._compile_fp = None  # retry when tokenizer appears
            return

        compiled: list[CompiledOutputPattern] = []
        for index, raw in enumerate(self._raw_patterns):
            try:
                if isinstance(raw, str):
                    token_ids = _encode_text(tokenizer, raw, add_special_tokens=self._add_special_tokens)
                    text = raw
                    source: SourceKind = "text"
                else:
                    token_ids = [int(x) for x in raw]
                    text = decode_token_ids(tokenizer, token_ids)
                    source = "token_ids"
            except Exception as exc:
                logger.warning(
                    "[runtime_guard: output_substring] skip pattern[%d] encode/decode failed error=%s raw=%r",
                    index,
                    exc,
                    raw,
                )
                continue
            if not token_ids:
                logger.warning(
                    "[runtime_guard: output_substring] skip pattern[%d] empty token_ids after encode raw=%r",
                    index,
                    raw,
                )
                continue
            pat = CompiledOutputPattern(
                index=index,
                source=source,
                text=text,
                token_ids=tuple(token_ids),
                raw=raw,
            )
            compiled.append(pat)
            logger.info(
                "[runtime_guard: output_substring] pattern[%d] source=%s text=%r token_ids=%s",
                pat.index,
                pat.source,
                pat.text,
                list(pat.token_ids),
            )
        self._compiled = compiled
        if not compiled:
            logger.warning("[runtime_guard: output_substring] no usable patterns after encode/decode")

    def check_all(
        self,
        sampled_token_ids: list[list[int]] | Any | None = None,
        req_ids: list[str] | None = None,
        skip_req_ids: set[str] | None = None,
    ) -> list[Incident]:
        """Batch entry: return alerts for requests whose cumulative output matches.

        ``sampled_token_ids=None`` means the caller
        (``DetectorManager.run_after_sample_cpu``) already appended this step's
        tokens to the cumulative IO buffer; this method then matches against
        the buffer only. When provided, tokens are appended here first
        (direct / standalone callers).

        ``skip_req_ids``: requests to skip (stopped after ``report.max_per_req``).
        Batch index alignment is preserved for the rest.
        """
        if not self._precheck():
            return []
        # Ensure compile once tokenizer is available (first enabled check).
        if self._enabled and self._raw_patterns and not self._compiled:
            self._maybe_compile_patterns(force_log=True)
        if not self._compiled:
            return []

        runner = self._runner
        req_ids = resolve_batch_req_ids(runner, req_ids)
        if not req_ids:
            return []

        # Ensure this step's tokens are in the cumulative IO buffer when provided.
        if sampled_token_ids is not None:
            RequestIoSnapshotManager.get().append_batch(req_ids, sampled_token_ids)

        alerts: list[Incident] = []
        io_mgr = RequestIoSnapshotManager.get()
        # Text-authored patterns are tokenization-context sensitive (a leading
        # space in the generated text re-encodes the same substring to different
        # token ids). Fall back to a decoded-text substring match when enabled.
        need_text = any(pat.source == "text" for pat in self._compiled)
        tokenizer = self._get_tokenizer() if need_text else None
        for batch_idx, req_id in enumerate(req_ids):
            if not req_id or req_id in self._alerted:
                continue
            if skip_req_ids and req_id in skip_req_ids:
                continue
            snap = io_mgr.snapshot(runner, req_id, batch_idx, include_token_ids=True)
            output_ids = list(snap.output_token_ids or [])
            if not output_ids and sampled_token_ids is not None and batch_idx < len(sampled_token_ids):
                output_ids = normalize_token_ids(sampled_token_ids[batch_idx])
            if not output_ids:
                continue
            output_text: str | None = None
            if tokenizer is not None and need_text:
                try:
                    output_text = decode_token_ids(tokenizer, output_ids)
                except Exception:
                    output_text = None
            for pat in self._compiled:
                if self._matches(pat, output_ids, output_text):
                    self._alerted.add(req_id)
                    alerts.append(self._make_alert(req_id, batch_idx, pat, output_ids))
                    break
        return alerts

    def _matches(
        self,
        pat: CompiledOutputPattern,
        output_ids: list[int],
        output_text: str | None = None,
    ) -> bool:
        """Prefix (start-of-output) or anywhere-subsequence match per ``match_prefix``.

        Token-id matching stays primary; for text-authored patterns a decoded
        text fallback covers re-tokenization drift (e.g. a leading space).
        """
        if self._match_prefix:
            if contains_prefix(output_ids, pat.token_ids):
                return True
            return bool(output_text and pat.text and output_text.startswith(pat.text))
        if contains_token_subsequence(output_ids, pat.token_ids):
            return True
        if pat.source != "text":
            return False
        return bool(output_text and pat.text and pat.text in output_text)

    def _make_alert(
        self,
        req_id: str,
        req_idx: int,
        pat: CompiledOutputPattern,
        output_ids: list[int],
    ) -> Incident:
        output_token_count = output_token_count_for_request(self._runner, req_id, req_idx)
        if output_token_count <= 0:
            output_token_count = len(output_ids)
        # "prefix" (match_prefix:true, start-of-output) vs "subsequence" (default, anywhere).
        match_mode = "prefix" if self._match_prefix else "subsequence"
        detail = {
            "matched_pattern_index": pat.index,
            "matched_source": pat.source,
            "matched_text": pat.text,
            "matched_token_ids": list(pat.token_ids),
            "match_mode": match_mode,
            "output_token_count": output_token_count,
        }
        logger.error(
            "[runtime_guard: output_substring] hit req_id=%s pattern[%d] source=%s mode=%s "
            "text=%r token_ids=%s output_token_count=%d",
            req_id,
            pat.index,
            pat.source,
            match_mode,
            pat.text,
            list(pat.token_ids),
            output_token_count,
        )
        return Incident(
            incident_type=self.incident_type,
            req_id=req_id,
            req_idx=req_idx,
            is_ill=True,
            ill_type=ILL_TYPE_NONE,
            detail=detail,
        )
