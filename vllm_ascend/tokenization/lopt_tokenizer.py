#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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

from __future__ import annotations

import logging
import threading
import time
from bisect import bisect_left
from collections import defaultdict
from collections.abc import Sequence
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from dataclasses import dataclass
from itertools import islice
from numbers import Integral
from typing import Any, Protocol, overload

logger = logging.getLogger(__name__)

_BATCH_PROBE_CHARS = 256


class _Tokenizer(Protocol):
    is_fast: bool

    def __call__(self, text: str | list[str], **kwargs: Any) -> Any: ...

    def encode(self, text: str, **kwargs: Any) -> list[int]: ...


class _PositionEncoding(Protocol):
    @property
    def index(self) -> int: ...

    @property
    def global_start(self) -> int: ...

    @property
    def global_end(self) -> int: ...

    @property
    def token_ids(self) -> Sequence[int]: ...

    @property
    def global_offsets(self) -> Sequence[tuple[int, int]]: ...


@dataclass(frozen=True)
class LoptConfig:
    enabled: bool = False
    thread_workers: int = 4
    min_chars: int = 32768
    chunk_chars: int = 32768
    overlap_chars: int = 512
    min_match_tokens: int = 2
    max_retries: int = 3
    verify: bool = False

    def __post_init__(self) -> None:
        if self.thread_workers < 1:
            raise ValueError("LoPT thread_workers must be at least 1")
        if self.min_chars < 1:
            raise ValueError("LoPT min_chars must be at least 1")
        if self.chunk_chars < 1:
            raise ValueError("LoPT chunk_chars must be at least 1")
        if not 0 < self.overlap_chars <= self.chunk_chars:
            raise ValueError("LoPT overlap_chars must be in [1, chunk_chars]")
        if self.min_match_tokens < 1:
            raise ValueError("LoPT min_match_tokens must be at least 1")
        if self.max_retries < 0:
            raise ValueError("LoPT max_retries cannot be negative")


@dataclass(frozen=True)
class TextChunk:
    index: int
    global_start: int
    global_end: int
    text: str


@dataclass(frozen=True)
class ChunkEncoding:
    index: int
    global_start: int
    global_end: int
    token_ids: tuple[int, ...]
    local_offsets: tuple[tuple[int, int], ...]
    global_offsets: tuple[tuple[int, int], ...]


class _GlobalOffsetView(Sequence[tuple[int, int]]):
    """Translate tokenizer-local offsets to global positions on demand."""

    def __init__(self, local_offsets: Sequence[Any], global_start: int) -> None:
        self._local_offsets = local_offsets
        self._global_start = global_start

    def __len__(self) -> int:
        return len(self._local_offsets)

    @overload
    def __getitem__(self, index: int) -> tuple[int, int]: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[tuple[int, int]]: ...

    def __getitem__(self, index: int | slice) -> tuple[int, int] | Sequence[tuple[int, int]]:
        if isinstance(index, slice):
            indices = range(*index.indices(len(self)))
            return [self[item_index] for item_index in indices]
        start, end = self._local_offsets[index]
        return self._global_start + start, self._global_start + end


@dataclass(frozen=True)
class _ChunkEncodingView:
    """Batch-tokenizer output that avoids eagerly copying IDs and offsets."""

    index: int
    global_start: int
    global_end: int
    token_ids: Sequence[int]
    global_offsets: Sequence[tuple[int, int]]


@dataclass
class _ChunkEncodingBuilder:
    """Mutable merge state that avoids copying accumulated tuple prefixes."""

    index: int
    global_start: int
    global_end: int
    token_ids: list[int]
    global_offsets: list[tuple[int, int]]


@dataclass(frozen=True)
class OverlapMatch:
    left_start: int
    right_start: int
    token_count: int
    char_start: int
    char_end: int


class LoptError(RuntimeError):
    """Base exception for a safely recoverable LoPT failure."""


class ChunkEncodingError(LoptError):
    """A tokenizer failed to produce a valid offset-aware chunk encoding."""


class OverlapMatchError(LoptError):
    """Adjacent chunks could not be joined without ambiguity."""


def _split_overlapping(text: str, chunk_chars: int, overlap_chars: int) -> list[TextChunk]:
    if chunk_chars < 1:
        raise ValueError("chunk_chars must be positive")
    if not 0 < overlap_chars <= chunk_chars:
        raise ValueError("overlap_chars must be in [1, chunk_chars]")

    chunks: list[TextChunk] = []
    start = 0
    index = 0
    while start < len(text):
        end = min(len(text), start + chunk_chars + overlap_chars)
        chunks.append(
            TextChunk(
                index=index,
                global_start=start,
                global_end=end,
                text=text[start:end],
            )
        )
        start += chunk_chars
        index += 1
    return chunks


def _as_token_id_list(value: Any) -> list[int]:
    try:
        values = value if isinstance(value, list) else list(value)
    except TypeError as exc:
        raise ChunkEncodingError("tokenizer input_ids is not a sequence") from exc

    if values and isinstance(values[0], (list, tuple)):
        raise ChunkEncodingError("tokenizer unexpectedly returned batched input_ids")

    converted_values: list[int] | None = None
    for index, token_id in enumerate(values):
        if not isinstance(token_id, Integral):
            raise ChunkEncodingError("tokenizer returned a non-integer token ID")
        if not isinstance(token_id, int):
            if converted_values is None:
                converted_values = list(values)
            converted_values[index] = int(token_id)
    return converted_values if converted_values is not None else values


def _as_token_ids(value: Any) -> tuple[int, ...]:
    return tuple(_as_token_id_list(value))


def _as_offsets(value: Any, text_length: int) -> tuple[tuple[int, int], ...]:
    try:
        values = list(value)
    except TypeError as exc:
        raise ChunkEncodingError("tokenizer offset_mapping is not a sequence") from exc

    offsets: list[tuple[int, int]] = []
    previous_start = 0
    previous_end = 0
    for raw_offset in values:
        if not isinstance(raw_offset, (list, tuple)) or len(raw_offset) != 2:
            raise ChunkEncodingError("tokenizer returned an invalid offset pair")
        start, end = raw_offset
        if not isinstance(start, Integral) or not isinstance(end, Integral):
            raise ChunkEncodingError("tokenizer returned a non-integer offset")
        start = int(start)
        end = int(end)
        if not 0 <= start <= end <= text_length:
            raise ChunkEncodingError("tokenizer returned an out-of-range offset")
        if start < previous_start or end < previous_end:
            raise ChunkEncodingError("tokenizer offsets are not monotonic")
        offsets.append((start, end))
        previous_start = start
        previous_end = end
    return tuple(offsets)


def _encoding_value(encoded: Any, key: str) -> Any:
    try:
        return encoded[key]
    except (KeyError, TypeError):
        value = getattr(encoded, key, None)
        if value is None:
            raise ChunkEncodingError(f"tokenizer result does not contain {key}") from None
        return value


def _build_chunk_encoding(chunk: TextChunk, input_ids: Any, offset_mapping: Any) -> ChunkEncoding:
    token_ids = _as_token_ids(input_ids)
    local_offsets = _as_offsets(offset_mapping, len(chunk.text))
    if len(token_ids) != len(local_offsets):
        raise ChunkEncodingError("input_ids and offset_mapping lengths differ")

    global_offsets = tuple(
        (chunk.global_start + start, chunk.global_start + end) for start, end in local_offsets
    )
    return ChunkEncoding(
        index=chunk.index,
        global_start=chunk.global_start,
        global_end=chunk.global_end,
        token_ids=token_ids,
        local_offsets=local_offsets,
        global_offsets=global_offsets,
    )


def _as_sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return value
    try:
        return list(value)
    except TypeError as exc:
        raise ChunkEncodingError(f"tokenizer {name} is not a sequence") from exc


def _build_chunk_encoding_view(chunk: TextChunk, input_ids: Any, offset_mapping: Any) -> _ChunkEncodingView:
    token_ids = _as_sequence(input_ids, "input_ids")
    local_offsets = _as_sequence(offset_mapping, "offset_mapping")
    if len(token_ids) != len(local_offsets):
        raise ChunkEncodingError("input_ids and offset_mapping lengths differ")
    if token_ids and isinstance(token_ids[0], (list, tuple)):
        raise ChunkEncodingError("tokenizer unexpectedly returned batched input_ids")

    return _ChunkEncodingView(
        index=chunk.index,
        global_start=chunk.global_start,
        global_end=chunk.global_end,
        token_ids=token_ids,
        global_offsets=_GlobalOffsetView(local_offsets, chunk.global_start),
    )


def _encode_chunk(tokenizer: _Tokenizer, chunk: TextChunk) -> ChunkEncoding:
    try:
        encoded = tokenizer(
            chunk.text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
    except Exception as exc:
        raise ChunkEncodingError(f"chunk {chunk.index} tokenization failed") from exc

    return _build_chunk_encoding(
        chunk,
        _encoding_value(encoded, "input_ids"),
        _encoding_value(encoded, "offset_mapping"),
    )


def _batch_encode_chunks(
    tokenizer: _Tokenizer,
    chunks: list[TextChunk],
    *,
    validate: bool = True,
) -> list[_PositionEncoding]:
    """Encode all chunks in one call so fast tokenizers can parallelize in native code."""
    try:
        encoded = tokenizer(
            [chunk.text for chunk in chunks],
            add_special_tokens=False,
            return_offsets_mapping=True,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        input_ids = list(_encoding_value(encoded, "input_ids"))
        offset_mappings = list(_encoding_value(encoded, "offset_mapping"))
    except Exception as exc:
        raise ChunkEncodingError("batch chunk tokenization failed") from exc

    if len(input_ids) != len(chunks) or len(offset_mappings) != len(chunks):
        raise ChunkEncodingError("batch tokenizer result size differs from chunk count")
    builder = _build_chunk_encoding if validate else _build_chunk_encoding_view
    return [
        builder(chunk, chunk_input_ids, chunk_offsets)
        for chunk, chunk_input_ids, chunk_offsets in zip(chunks, input_ids, offset_mappings, strict=True)
    ]


def _parallel_encode_chunks(
    tokenizer: _Tokenizer,
    chunks: list[TextChunk],
    executor: Executor,
) -> list[ChunkEncoding]:
    from concurrent.futures import as_completed
    futures: list[Future[ChunkEncoding]] = [executor.submit(_encode_chunk, tokenizer, chunk) for chunk in chunks]
    encodings: list[ChunkEncoding] = []
    try:
        for future in as_completed(futures):
            encodings.append(future.result())
    except Exception as exc:
        for future in futures:
            future.cancel()
        if isinstance(exc, LoptError):
            raise
        raise ChunkEncodingError("parallel chunk tokenization failed") from exc

    encodings.sort(key=lambda encoding: encoding.index)
    return encodings


def _record(encoding: _PositionEncoding, index: int) -> tuple[int, int, int]:
    start, end = encoding.global_offsets[index]
    return start, end, encoding.token_ids[index]


def _in_overlap(record: tuple[int, int, int], overlap_start: int, overlap_end: int) -> bool:
    start, end, _ = record
    return end > start and start >= overlap_start and end <= overlap_end


def _overlap_token_range(encoding: _PositionEncoding, overlap_start: int, overlap_end: int) -> range:
    """Return token indices whose starts may fall within the overlap."""
    first = bisect_left(encoding.global_offsets, (overlap_start, overlap_start))
    stop = bisect_left(encoding.global_offsets, (overlap_end, overlap_end), lo=first)
    return range(first, stop)


def _find_position_overlap(
    left: _PositionEncoding,
    right: _PositionEncoding,
    min_match_tokens: int,
) -> OverlapMatch | None:
    overlap_start = max(left.global_start, right.global_start)
    overlap_end = min(left.global_end, right.global_end)
    if overlap_start >= overlap_end:
        return None

    right_candidates: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for right_index in _overlap_token_range(right, overlap_start, overlap_end):
        record = _record(right, right_index)
        if _in_overlap(record, overlap_start, overlap_end):
            right_candidates[record].append(right_index)

    def rank(match: OverlapMatch) -> tuple[int, int, int]:
        safety_margin = min(match.char_start - overlap_start, overlap_end - match.char_end)
        return match.token_count, safety_margin, match.char_end - match.char_start

    best_match: OverlapMatch | None = None
    best_rank: tuple[int, int, int] | None = None
    best_is_ambiguous = False
    # A later start on the same matched diagonal can only produce a shorter
    # suffix, so do not expand that suffix a second time.
    matched_pairs: set[tuple[int, int]] = set()
    for left_index in _overlap_token_range(left, overlap_start, overlap_end):
        first_record = _record(left, left_index)
        if not _in_overlap(first_record, overlap_start, overlap_end):
            continue
        for right_index in right_candidates.get(first_record, ()):
            if (left_index, right_index) in matched_pairs:
                continue

            token_count = 0
            char_end = first_record[1]
            while left_index + token_count < len(left.token_ids) and right_index + token_count < len(right.token_ids):
                left_record = _record(left, left_index + token_count)
                right_record = _record(right, right_index + token_count)
                if left_record != right_record or not _in_overlap(left_record, overlap_start, overlap_end):
                    break
                matched_pairs.add((left_index + token_count, right_index + token_count))
                char_end = max(char_end, left_record[1])
                token_count += 1

            if token_count >= min_match_tokens and char_end > first_record[0]:
                candidate = OverlapMatch(
                    left_start=left_index,
                    right_start=right_index,
                    token_count=token_count,
                    char_start=first_record[0],
                    char_end=char_end,
                )
                candidate_rank = rank(candidate)
                if best_rank is None or candidate_rank > best_rank:
                    best_match = candidate
                    best_rank = candidate_rank
                    best_is_ambiguous = False
                elif candidate_rank == best_rank:
                    best_is_ambiguous = True

    if best_match is None or best_is_ambiguous:
        return None
    return best_match


def _new_encoding_builder(encoding: _PositionEncoding) -> _ChunkEncodingBuilder:
    return _ChunkEncodingBuilder(
        index=encoding.index,
        global_start=encoding.global_start,
        global_end=encoding.global_end,
        token_ids=list(encoding.token_ids),
        global_offsets=list(encoding.global_offsets),
    )


def _freeze_encoding_builder(builder: _ChunkEncodingBuilder) -> ChunkEncoding:
    token_ids = tuple(builder.token_ids)
    global_offsets = tuple(builder.global_offsets)
    if not token_ids or len(token_ids) != len(global_offsets):
        raise OverlapMatchError("merged chunk encoding is empty or inconsistent")

    local_offsets = tuple(
        (start - builder.global_start, end - builder.global_start) for start, end in global_offsets
    )
    return ChunkEncoding(
        index=builder.index,
        global_start=builder.global_start,
        global_end=builder.global_end,
        token_ids=token_ids,
        local_offsets=local_offsets,
        global_offsets=global_offsets,
    )


def _merge_pair_into(
    merged: _ChunkEncodingBuilder,
    right: _PositionEncoding,
    min_match_tokens: int,
) -> None:
    """Merge ``right`` into reusable mutable buffers."""

    match = _find_position_overlap(merged, right, min_match_tokens)
    if match is None:
        raise OverlapMatchError(
            f"no unambiguous position-aware overlap between chunks {merged.index} and {right.index}"
        )

    left_end = match.left_start + match.token_count
    right_end = match.right_start + match.token_count
    del merged.token_ids[left_end:]
    del merged.global_offsets[left_end:]
    merged.token_ids.extend(islice(right.token_ids, right_end, None))
    merged.global_offsets.extend(islice(right.global_offsets, right_end, None))
    if not merged.token_ids or len(merged.token_ids) != len(merged.global_offsets):
        raise OverlapMatchError("merged chunk encoding is empty or inconsistent")

    merged.index = right.index
    merged.global_start = min(merged.global_start, right.global_start)
    merged.global_end = max(merged.global_end, right.global_end)


def _merge_pair(left: ChunkEncoding, right: ChunkEncoding, min_match_tokens: int) -> ChunkEncoding:
    builder = _new_encoding_builder(left)
    _merge_pair_into(builder, right, min_match_tokens)
    return _freeze_encoding_builder(builder)


def _merge_chunk_encoding_builder(
    encodings: Sequence[_PositionEncoding], min_match_tokens: int
) -> _ChunkEncodingBuilder:
    if not encodings:
        raise OverlapMatchError("cannot merge an empty chunk list")

    merged = _new_encoding_builder(encodings[0])
    for encoding in encodings[1:]:
        _merge_pair_into(merged, encoding, min_match_tokens)
    return merged


def _merge_chunk_encodings(encodings: list[ChunkEncoding], min_match_tokens: int) -> ChunkEncoding:
    if len(encodings) == 1:
        return encodings[0]
    merged = _merge_chunk_encoding_builder(encodings, min_match_tokens)
    return _freeze_encoding_builder(merged)


def _find_adjacent_matches(
    encodings: Sequence[_PositionEncoding], min_match_tokens: int
) -> list[OverlapMatch]:
    matches: list[OverlapMatch] = []
    for left, right in zip(encodings, encodings[1:]):
        match = _find_position_overlap(left, right, min_match_tokens)
        if match is None:
            raise OverlapMatchError(
                f"no unambiguous position-aware overlap between chunks {left.index} and {right.index}"
            )
        matches.append(match)
    return matches


def _merge_chunk_token_ids(
    encodings: Sequence[_PositionEncoding], min_match_tokens: int
) -> list[int]:
    """Join token ranges after independently locating every adjacent seam."""
    if not encodings:
        raise OverlapMatchError("cannot merge an empty chunk list")
    if len(encodings) == 1:
        return list(encodings[0].token_ids)

    matches = _find_adjacent_matches(encodings, min_match_tokens)
    token_ids: list[int] = []
    for index, encoding in enumerate(encodings):
        start = 0 if index == 0 else matches[index - 1].right_start + matches[index - 1].token_count
        stop = (
            len(encoding.token_ids)
            if index == len(encodings) - 1
            else matches[index].left_start + matches[index].token_count
        )
        if start > stop:
            raise OverlapMatchError(f"overlap seams cross within chunk {encoding.index}")
        token_ids.extend(islice(encoding.token_ids, start, stop))

    if not token_ids:
        raise OverlapMatchError("merged chunk encoding is empty")
    return token_ids


class LosslessParallelTokenizer:
    """Offset-aware overlapping parallel tokenization with safe fallback."""

    _SUPPORTED_ENCODE_KWARGS = frozenset({"add_special_tokens", "truncation", "max_length"})

    def __init__(
        self,
        tokenizer: _Tokenizer,
        config: LoptConfig,
        executor: Executor | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.config = config
        self._executor = executor or ThreadPoolExecutor(
            max_workers=config.thread_workers,
            thread_name_prefix="vllm-ascend-lopt",
        )
        self._owns_executor = executor is None
        self._state_lock = threading.Lock()
        self._shutdown = False
        self._compatibility: dict[tuple[bool, bool, int | None], bool] = {}
        self._warned_fallbacks: set[str] = set()
        self._batch_encoding_supported: bool | None = None

    def can_use(self, text: str, encode_kwargs: dict[str, Any] | None = None) -> bool:
        encode_kwargs = encode_kwargs or {}
        with self._state_lock:
            is_shutdown = self._shutdown
        if is_shutdown or not self.config.enabled:
            return False
        if getattr(self.tokenizer, "is_fast", False) is not True:
            return False
        if len(text) < self.config.min_chars:
            return False
        if len(text) <= self.config.chunk_chars + self.config.overlap_chars:
            return False
        if set(encode_kwargs) - self._SUPPORTED_ENCODE_KWARGS:
            return False
        if (
            encode_kwargs.get("truncation", False)
            and encode_kwargs.get("add_special_tokens", True)
            and not callable(getattr(self.tokenizer, "prepare_for_model", None))
        ):
            return False
        if not callable(getattr(self.tokenizer, "encode", None)) or not callable(self.tokenizer):
            return False
        return self._is_compatible(encode_kwargs)

    def encode(self, text: str, **encode_kwargs: Any) -> list[int]:
        if not self.can_use(text, encode_kwargs):
            return self._standard_encode(text, encode_kwargs)

        lopt_started_at = time.perf_counter() if self.config.verify else None
        try:
            token_ids, attempts, chunk_count = self._encode_with_retries(text, encode_kwargs)
        except Exception as exc:
            failed_lopt_stopped_at = time.perf_counter() if self.config.verify else None
            self._log_fallback_once(
                type(exc).__name__,
                "LoPT tokenization failed (%s); using standard tokenization",
                exc,
            )
            serial_started_at = time.perf_counter() if self.config.verify else None
            standard_ids = self._standard_encode(text, encode_kwargs)
            if self.config.verify:
                assert lopt_started_at is not None
                assert failed_lopt_stopped_at is not None
                assert serial_started_at is not None
                failed_lopt_ms = (failed_lopt_stopped_at - lopt_started_at) * 1000
                serial_ms = (time.perf_counter() - serial_started_at) * 1000
                logger.warning(
                    "LoPT timing comparison failed: characters=%d tokens=%d serial_ms=%.3f "
                    "failed_lopt_ms=%.3f error=%s",
                    len(text),
                    len(standard_ids),
                    serial_ms,
                    failed_lopt_ms,
                    exc,
                )
            return standard_ids

        if self.config.verify:
            assert lopt_started_at is not None
            lopt_ms = (time.perf_counter() - lopt_started_at) * 1000
            serial_started_at = time.perf_counter()
            standard_ids = self._standard_encode(text, encode_kwargs)
            serial_ms = (time.perf_counter() - serial_started_at) * 1000
            matched = token_ids == standard_ids
            speedup = serial_ms / lopt_ms if lopt_ms > 0 else 0.0
            logger.info(
                "LoPT timing comparison: characters=%d tokens=%d serial_ms=%.3f "
                "lopt_ms=%.3f speedup=%.2fx matched=%s chunks=%d attempts=%d",
                len(text),
                len(standard_ids),
                serial_ms,
                lopt_ms,
                speedup,
                matched,
                chunk_count,
                attempts,
            )
            if not matched:
                self._log_fallback_once("verification", "LoPT verification failed; using standard tokenization")
                return standard_ids

        return token_ids

    def shutdown(self, wait: bool = True) -> None:
        with self._state_lock:
            if self._shutdown:
                return
            self._shutdown = True
        if self._owns_executor:
            self._executor.shutdown(wait=wait, cancel_futures=True)

    def _standard_encode(self, text: str, encode_kwargs: dict[str, Any]) -> list[int]:
        return list(self.tokenizer.encode(text, **encode_kwargs))

    def _log_fallback_once(self, key: str, message: str, *args: Any) -> None:
        with self._state_lock:
            first_occurrence = key not in self._warned_fallbacks
            self._warned_fallbacks.add(key)
        if first_occurrence:
            logger.warning(message, *args)

    def _is_compatible(self, encode_kwargs: dict[str, Any]) -> bool:
        add_special_tokens = bool(encode_kwargs.get("add_special_tokens", True))
        truncation = bool(encode_kwargs.get("truncation", False))
        max_length = encode_kwargs.get("max_length")
        key = (add_special_tokens, truncation, max_length)
        with self._state_lock:
            cached = self._compatibility.get(key)
        if cached is not None:
            return cached

        probe = "LoPT compatibility probe: ASCII 中文 🙂 e\u0301 0123456789. " * 2
        probe_kwargs: dict[str, Any] = {
            "add_special_tokens": add_special_tokens,
            "truncation": truncation,
            "max_length": None,
        }
        if truncation:
            probe_kwargs["max_length"] = min(max_length, 16) if isinstance(max_length, int) else 16

        try:
            chunk = TextChunk(index=0, global_start=0, global_end=len(probe), text=probe)
            content_ids = _encode_chunk(self.tokenizer, chunk).token_ids
            prepared_ids = self._prepare_for_model(content_ids, probe_kwargs)
            standard_ids = self._standard_encode(probe, probe_kwargs)
            compatible = prepared_ids == standard_ids
        except Exception:
            compatible = False

        with self._state_lock:
            self._compatibility[key] = compatible
        if not compatible:
            self._log_fallback_once(
                f"compatibility-{key}",
                "LoPT compatibility probe failed for add_special_tokens=%s, truncation=%s; using standard tokenization",
                add_special_tokens,
                truncation,
            )
        return compatible

    def _encode_chunks(self, chunks: list[TextChunk]) -> list[_PositionEncoding]:
        with self._state_lock:
            batch_encoding_supported = self._batch_encoding_supported

        if batch_encoding_supported is None:
            probe_chunks = [
                TextChunk(
                    index=chunk.index,
                    global_start=chunk.global_start,
                    global_end=chunk.global_start + min(len(chunk.text), _BATCH_PROBE_CHARS),
                    text=chunk.text[:_BATCH_PROBE_CHARS],
                )
                for chunk in chunks[:2]
            ]
            try:
                _batch_encode_chunks(self.tokenizer, probe_chunks, validate=True)
            except ChunkEncodingError:
                batch_encoding_supported = False
            else:
                batch_encoding_supported = True
            with self._state_lock:
                self._batch_encoding_supported = batch_encoding_supported

        if batch_encoding_supported:
            try:
                encodings = _batch_encode_chunks(self.tokenizer, chunks, validate=False)
            except ChunkEncodingError:
                with self._state_lock:
                    self._batch_encoding_supported = False
            else:
                return encodings

        return _parallel_encode_chunks(self.tokenizer, chunks, self._executor)

    def _encode_with_retries(
        self,
        text: str,
        encode_kwargs: dict[str, Any],
    ) -> tuple[list[int], int, int]:
        chunk_chars = self.config.chunk_chars
        last_error: OverlapMatchError | None = None

        for attempt in range(self.config.max_retries + 1):
            chunks = _split_overlapping(text, chunk_chars, self.config.overlap_chars)
            if len(chunks) < 2:
                raise LoptError("prompt does not produce multiple chunks")

            encodings = self._encode_chunks(chunks)
            try:
                merged_token_ids = _merge_chunk_token_ids(encodings, self.config.min_match_tokens)
            except OverlapMatchError as exc:
                last_error = exc
                chunk_chars *= 2
                continue

            token_ids = self._prepare_for_model(merged_token_ids, encode_kwargs)
            return token_ids, attempt + 1, len(chunks)

        raise last_error or OverlapMatchError("LoPT overlap matching failed")

    def _prepare_for_model(self, token_ids: Sequence[int], encode_kwargs: dict[str, Any]) -> list[int]:
        add_special_tokens = bool(encode_kwargs.get("add_special_tokens", True))
        truncation = bool(encode_kwargs.get("truncation", False))
        max_length = encode_kwargs.get("max_length")
        ids = list(token_ids)

        # DeepSeekV4 chat templates already contain their control tokens and
        # vLLM encodes them with add_special_tokens=False. In this case,
        # truncating the complete merged token sequence is equivalent to
        # tokenizer-level truncation.
        if not add_special_tokens:
            if not truncation:
                return ids

            if (
                not isinstance(max_length, int)
                or isinstance(max_length, bool)
                or max_length < 0
            ):
                raise LoptError("truncation requires a non-negative integer max_length")

            if len(ids) <= max_length:
                return ids

            truncation_side = getattr(self.tokenizer, "truncation_side", "right")
            if truncation_side == "right":
                return ids[:max_length]
            if truncation_side == "left":
                # ids[-0:] returns the complete list instead of an empty list.
                return ids[len(ids) - max_length :]

            raise LoptError(f"unsupported truncation_side: {truncation_side!r}")

        prepare_for_model = getattr(self.tokenizer, "prepare_for_model", None)
        if callable(prepare_for_model):
            prepared = prepare_for_model(
                ids,
                add_special_tokens=True,
                truncation=truncation,
                max_length=max_length,
                padding=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            return _as_token_id_list(_encoding_value(prepared, "input_ids"))

        if truncation:
            raise LoptError("tokenizer cannot reproduce truncation with special tokens")

        build_inputs = getattr(self.tokenizer, "build_inputs_with_special_tokens", None)
        if not callable(build_inputs):
            raise LoptError("tokenizer cannot add special tokens to merged IDs")
        return list(build_inputs(ids))
