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
from typing import Any

import pytest

import vllm_ascend.tokenization.lopt_tokenizer as lopt_tokenizer_module
from vllm_ascend.tokenization.lopt_tokenizer import (
    ChunkEncoding,
    LoptConfig,
    LosslessParallelTokenizer,
    _encode_chunk,
    _find_position_overlap,
    _merge_chunk_encodings,
    _merge_chunk_token_ids,
    _split_overlapping,
)


class CharacterTokenizer:
    is_fast = True
    truncation_side = "right"
    bos_token_id = 1
    eos_token_id = 2

    def __init__(self) -> None:
        self.chunk_call_count = 0

    @staticmethod
    def _content(text: str) -> tuple[list[int], list[tuple[int, int]]]:
        return [ord(char) + 10 for char in text], [(index, index + 1) for index in range(len(text))]

    def __call__(self, text: str, **kwargs: Any) -> dict[str, Any]:
        self.chunk_call_count += 1
        token_ids, offsets = self._content(text)
        return {"input_ids": token_ids, "offset_mapping": offsets}

    def prepare_for_model(
        self,
        token_ids: list[int],
        *,
        add_special_tokens: bool,
        truncation: bool,
        max_length: int | None,
        **kwargs: Any,
    ) -> dict[str, list[int]]:
        ids = list(token_ids)
        special_count = 2 if add_special_tokens else 0
        if truncation and max_length is not None:
            content_length = max(0, max_length - special_count)
            ids = ids[-content_length:] if self.truncation_side == "left" else ids[:content_length]
        if add_special_tokens:
            ids = [self.bos_token_id, *ids, self.eos_token_id]
        return {"input_ids": ids}

    def encode(self, text: str, **kwargs: Any) -> list[int]:
        token_ids, _ = self._content(text)
        return self.prepare_for_model(token_ids, **kwargs)["input_ids"]


class TokenizersBackendLikeTokenizer(CharacterTokenizer):
    """Fast-tokenizer stub without the Transformers v4 post-processing API."""

    prepare_for_model = None

    def encode(self, text: str, **kwargs: Any) -> list[int]:
        ids, _ = self._content(text)
        add_special_tokens = bool(kwargs.get("add_special_tokens", True))
        truncation = bool(kwargs.get("truncation", False))
        max_length = kwargs.get("max_length")

        if truncation and isinstance(max_length, int) and not isinstance(max_length, bool):
            special_count = 2 if add_special_tokens else 0
            content_length = max(0, max_length - special_count)
            if len(ids) > content_length:
                if self.truncation_side == "left":
                    ids = ids[len(ids) - content_length :]
                else:
                    ids = ids[:content_length]
        if add_special_tokens:
            ids = [self.bos_token_id, *ids, self.eos_token_id]
        return ids


class PairTokenizer(CharacterTokenizer):
    @staticmethod
    def _content(text: str) -> tuple[list[int], list[tuple[int, int]]]:
        token_ids: list[int] = []
        offsets: list[tuple[int, int]] = []
        for start in range(0, len(text), 2):
            piece = text[start : start + 2]
            token_ids.append(sum((index + 1) * ord(char) for index, char in enumerate(piece)) + 10)
            offsets.append((start, start + len(piece)))
        return token_ids, offsets


class FailingOffsetTokenizer(CharacterTokenizer):
    def __call__(self, text: str, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("offset mapping unavailable")


class InvalidOffsetTokenizer(CharacterTokenizer):
    def __call__(self, text: str, **kwargs: Any) -> dict[str, Any]:
        token_ids, offsets = self._content(text)
        offsets[-1] = (len(text), len(text) + 1)
        return {"input_ids": token_ids, "offset_mapping": offsets}


class IncompatibleSpecialTokenizer(CharacterTokenizer):
    def encode(self, text: str, **kwargs: Any) -> list[int]:
        token_ids, _ = self._content(text)
        if kwargs.get("add_special_tokens", True):
            return [101, *token_ids, 102]
        return token_ids


class ConcurrentTokenizer(CharacterTokenizer):
    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._active = 0
        self.max_active = 0

    def __call__(self, text: str, **kwargs: Any) -> dict[str, Any]:
        with self._lock:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
        try:
            time.sleep(0.02)
            return super().__call__(text, **kwargs)
        finally:
            with self._lock:
                self._active -= 1


class BatchTokenizer(CharacterTokenizer):
    def __init__(self) -> None:
        super().__init__()
        self.batch_call_count = 0

    def __call__(self, text: str | list[str], **kwargs: Any) -> dict[str, Any]:
        if isinstance(text, str):
            return super().__call__(text, **kwargs)

        self.batch_call_count += 1
        encodings = [super(BatchTokenizer, self).__call__(item, **kwargs) for item in text]
        return {
            "input_ids": [encoding["input_ids"] for encoding in encodings],
            "offset_mapping": [encoding["offset_mapping"] for encoding in encodings],
        }


class ChunkMismatchTokenizer(CharacterTokenizer):
    def __call__(self, text: str, **kwargs: Any) -> dict[str, Any]:
        result = super().__call__(text, **kwargs)
        if len(text) <= 6 and result["input_ids"]:
            result["input_ids"][0] += 1
        return result


def make_config(**kwargs: Any) -> LoptConfig:
    values = {
        "enabled": True,
        "thread_workers": 4,
        "min_chars": 1,
        "chunk_chars": 4,
        "overlap_chars": 2,
        "min_match_tokens": 2,
        "max_retries": 2,
        "verify": False,
    }
    values.update(kwargs)
    return LoptConfig(**values)


def encode_kwargs(**kwargs: Any) -> dict[str, Any]:
    values = {"add_special_tokens": True, "truncation": False, "max_length": None}
    values.update(kwargs)
    return values


def test_split_overlapping_uses_global_character_coordinates():
    chunks = _split_overlapping("abcdefghij", chunk_chars=4, overlap_chars=2)

    assert [(chunk.index, chunk.global_start, chunk.global_end, chunk.text) for chunk in chunks] == [
        (0, 0, 6, "abcdef"),
        (1, 4, 10, "efghij"),
        (2, 8, 10, "ij"),
    ]


def test_encode_chunk_converts_unicode_offsets_to_global_coordinates():
    tokenizer = CharacterTokenizer()
    chunk = _split_overlapping("前缀🙂中文", chunk_chars=3, overlap_chars=2)[1]

    encoding = _encode_chunk(tokenizer, chunk)

    assert encoding.local_offsets == ((0, 1), (1, 2))
    assert encoding.global_offsets == ((3, 4), (4, 5))


def test_position_match_uses_offsets_instead_of_repeated_token_ids():
    left = ChunkEncoding(
        index=0,
        global_start=0,
        global_end=8,
        token_ids=(7, 7, 7, 7, 7, 7, 7, 7),
        local_offsets=tuple((index, index + 1) for index in range(8)),
        global_offsets=tuple((index, index + 1) for index in range(8)),
    )
    right = ChunkEncoding(
        index=1,
        global_start=5,
        global_end=10,
        token_ids=(7, 7, 7, 7, 7),
        local_offsets=tuple((index, index + 1) for index in range(5)),
        global_offsets=tuple((index + 5, index + 6) for index in range(5)),
    )

    match = _find_position_overlap(left, right, min_match_tokens=2)

    assert match is not None
    assert (match.left_start, match.right_start, match.token_count) == (5, 0, 3)


def test_position_match_supports_multiple_byte_tokens_with_the_same_character_offset():
    left = ChunkEncoding(
        index=0,
        global_start=0,
        global_end=4,
        token_ids=(1, 20, 21, 30),
        local_offsets=((0, 1), (2, 3), (2, 3), (3, 4)),
        global_offsets=((0, 1), (2, 3), (2, 3), (3, 4)),
    )
    right = ChunkEncoding(
        index=1,
        global_start=2,
        global_end=6,
        token_ids=(20, 21, 30, 40),
        local_offsets=((0, 1), (0, 1), (1, 2), (3, 4)),
        global_offsets=((2, 3), (2, 3), (3, 4), (5, 6)),
    )

    match = _find_position_overlap(left, right, min_match_tokens=2)

    assert match is not None
    assert (match.left_start, match.right_start, match.token_count) == (1, 0, 3)


def test_position_match_keeps_longest_run_after_earlier_run_stops():
    offsets = tuple((index, index + 1) for index in range(6))
    left = ChunkEncoding(
        index=0,
        global_start=0,
        global_end=6,
        token_ids=(10, 20, 30, 40, 50, 60),
        local_offsets=offsets,
        global_offsets=offsets,
    )
    right = ChunkEncoding(
        index=1,
        global_start=0,
        global_end=6,
        token_ids=(10, 99, 30, 40, 50, 98),
        local_offsets=offsets,
        global_offsets=offsets,
    )

    match = _find_position_overlap(left, right, min_match_tokens=2)

    assert match is not None
    assert (match.left_start, match.right_start, match.token_count) == (2, 2, 3)


def test_position_match_does_not_reexpand_matched_suffixes(monkeypatch):
    token_count = 100
    offsets = tuple((index, index + 1) for index in range(token_count))
    left = ChunkEncoding(
        index=0,
        global_start=0,
        global_end=token_count,
        token_ids=tuple(range(token_count)),
        local_offsets=offsets,
        global_offsets=offsets,
    )
    right = ChunkEncoding(
        index=1,
        global_start=0,
        global_end=token_count,
        token_ids=tuple(range(token_count)),
        local_offsets=offsets,
        global_offsets=offsets,
    )
    original_record = lopt_tokenizer_module._record
    left_record_calls = 0

    def tracking_record(encoding: ChunkEncoding, index: int) -> tuple[int, int, int]:
        nonlocal left_record_calls
        if encoding is left:
            left_record_calls += 1
        return original_record(encoding, index)

    monkeypatch.setattr(lopt_tokenizer_module, "_record", tracking_record)

    match = _find_position_overlap(left, right, min_match_tokens=2)

    assert match is not None
    assert match.token_count == token_count
    assert left_record_calls <= token_count * 2


def test_position_match_does_not_scan_tokens_before_overlap(monkeypatch):
    left_token_count = 1_000
    overlap_start = 997
    right_token_count = 6
    left = ChunkEncoding(
        index=0,
        global_start=0,
        global_end=left_token_count,
        token_ids=tuple(range(left_token_count)),
        local_offsets=tuple((index, index + 1) for index in range(left_token_count)),
        global_offsets=tuple((index, index + 1) for index in range(left_token_count)),
    )
    right = ChunkEncoding(
        index=1,
        global_start=overlap_start,
        global_end=overlap_start + right_token_count,
        token_ids=tuple(range(overlap_start, overlap_start + right_token_count)),
        local_offsets=tuple((index, index + 1) for index in range(right_token_count)),
        global_offsets=tuple(
            (index, index + 1) for index in range(overlap_start, overlap_start + right_token_count)
        ),
    )
    original_record = lopt_tokenizer_module._record
    visited_left_indices: list[int] = []

    def tracking_record(encoding: ChunkEncoding, index: int) -> tuple[int, int, int]:
        if encoding is left:
            visited_left_indices.append(index)
        return original_record(encoding, index)

    monkeypatch.setattr(lopt_tokenizer_module, "_record", tracking_record)

    match = _find_position_overlap(left, right, min_match_tokens=2)

    assert match is not None
    assert (match.left_start, match.right_start, match.token_count) == (overlap_start, 0, 3)
    assert visited_left_indices
    assert min(visited_left_indices) >= overlap_start


def test_merge_chunk_encodings_reuses_mutable_buffers(monkeypatch):
    encodings = [
        ChunkEncoding(
            index=chunk_index,
            global_start=start,
            global_end=end,
            token_ids=tuple(range(start, end)),
            local_offsets=tuple((index, index + 1) for index in range(end - start)),
            global_offsets=tuple((index, index + 1) for index in range(start, end)),
        )
        for chunk_index, start, end in ((0, 10, 16), (1, 14, 20), (2, 18, 22))
    ]
    original_merge_pair_into = lopt_tokenizer_module._merge_pair_into
    token_buffer_ids: list[int] = []
    offset_buffer_ids: list[int] = []

    def tracking_merge_pair_into(merged, right, min_match_tokens):
        token_buffer_ids.append(id(merged.token_ids))
        offset_buffer_ids.append(id(merged.global_offsets))
        original_merge_pair_into(merged, right, min_match_tokens)
        assert id(merged.token_ids) == token_buffer_ids[-1]
        assert id(merged.global_offsets) == offset_buffer_ids[-1]

    monkeypatch.setattr(lopt_tokenizer_module, "_merge_pair_into", tracking_merge_pair_into)

    merged = _merge_chunk_encodings(encodings, min_match_tokens=2)

    assert merged.token_ids == tuple(range(10, 22))
    assert merged.global_offsets == tuple((index, index + 1) for index in range(10, 22))
    assert merged.local_offsets == tuple((index, index + 1) for index in range(12))
    assert len(set(token_buffer_ids)) == 1
    assert len(set(offset_buffer_ids)) == 1


def test_direct_token_merge_matches_full_position_aware_merge():
    encodings = [
        ChunkEncoding(
            index=chunk_index,
            global_start=start,
            global_end=end,
            token_ids=tuple(range(start, end)),
            local_offsets=tuple((index, index + 1) for index in range(end - start)),
            global_offsets=tuple((index, index + 1) for index in range(start, end)),
        )
        for chunk_index, start, end in ((0, 10, 16), (1, 14, 20), (2, 18, 22))
    ]

    expected = _merge_chunk_encodings(encodings, min_match_tokens=2)
    actual = _merge_chunk_token_ids(encodings, min_match_tokens=2)

    assert actual == list(expected.token_ids)


def test_lossless_merge_matches_standard_tokenization_for_unicode_and_repetition():
    tokenizer = CharacterTokenizer()
    lopt = LosslessParallelTokenizer(tokenizer, make_config())
    text = "中文🙂abcabc   e\u0301" * 4

    try:
        actual = lopt.encode(text, **encode_kwargs())
    finally:
        lopt.shutdown()

    assert actual == tokenizer.encode(text, **encode_kwargs())


def test_special_tokens_and_truncation_are_applied_once_after_merge():
    tokenizer = CharacterTokenizer()
    lopt = LosslessParallelTokenizer(tokenizer, make_config())
    text = "abcdefghijklmnop"
    kwargs = encode_kwargs(truncation=True, max_length=9)

    try:
        actual = lopt.encode(text, **kwargs)
    finally:
        lopt.shutdown()

    assert actual == tokenizer.encode(text, **kwargs)
    assert actual[0] == tokenizer.bos_token_id
    assert actual[-1] == tokenizer.eos_token_id
    assert actual.count(tokenizer.bos_token_id) == 1
    assert actual.count(tokenizer.eos_token_id) == 1


@pytest.mark.parametrize(
    ("truncation_side", "max_length"),
    [("right", 5), ("left", 5), ("right", 0), ("left", 0)],
)
def test_truncation_without_special_tokens_does_not_require_prepare_for_model(
    truncation_side: str,
    max_length: int,
):
    tokenizer = TokenizersBackendLikeTokenizer()
    tokenizer.truncation_side = truncation_side
    lopt = LosslessParallelTokenizer(tokenizer, make_config())
    text = "abcdefghijklmnop"
    kwargs = encode_kwargs(
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )

    try:
        actual = lopt.encode(text, **kwargs)
    finally:
        lopt.shutdown()

    assert actual == tokenizer.encode(text, **kwargs)
    assert tokenizer.chunk_call_count > 0


def test_truncation_with_special_tokens_still_requires_prepare_for_model():
    tokenizer = TokenizersBackendLikeTokenizer()
    lopt = LosslessParallelTokenizer(tokenizer, make_config())
    text = "abcdefghijklmnop"
    kwargs = encode_kwargs(truncation=True, max_length=9)

    try:
        assert not lopt.can_use(text, kwargs)
        actual = lopt.encode(text, **kwargs)
    finally:
        lopt.shutdown()

    assert actual == tokenizer.encode(text, **kwargs)
    assert tokenizer.chunk_call_count == 0


def test_dynamic_chunk_retry_recovers_from_boundary_phase_mismatch():
    tokenizer = PairTokenizer()
    config = make_config(chunk_chars=3, overlap_chars=3, min_match_tokens=1)
    lopt = LosslessParallelTokenizer(tokenizer, config)
    text = "abcdefghijklmnopqrstuvwxyz"
    kwargs = encode_kwargs(add_special_tokens=False)

    try:
        actual, attempts, _ = lopt._encode_with_retries(text, kwargs)
    finally:
        lopt.shutdown()

    assert actual == tokenizer.encode(text, **kwargs)
    assert attempts == 2
    assert tokenizer.chunk_call_count > len(_split_overlapping(text, 3, 3))


@pytest.mark.parametrize("tokenizer_cls", [FailingOffsetTokenizer, InvalidOffsetTokenizer])
def test_offset_failures_fall_back_to_standard_tokenization(tokenizer_cls):
    tokenizer = tokenizer_cls()
    lopt = LosslessParallelTokenizer(tokenizer, make_config())
    text = "abcdefghijklmno"

    try:
        actual = lopt.encode(text, **encode_kwargs())
    finally:
        lopt.shutdown()

    assert actual == tokenizer.encode(text, **encode_kwargs())


def test_incompatible_special_token_post_processor_disables_lopt():
    tokenizer = IncompatibleSpecialTokenizer()
    lopt = LosslessParallelTokenizer(tokenizer, make_config())
    text = "abcdefghijklmno"

    try:
        actual = lopt.encode(text, **encode_kwargs())
    finally:
        lopt.shutdown()

    assert actual == tokenizer.encode(text, **encode_kwargs())
    assert actual[0] == 101
    assert actual[-1] == 102


def test_verify_mode_discards_a_non_lossless_chunk_result():
    tokenizer = ChunkMismatchTokenizer()
    lopt = LosslessParallelTokenizer(tokenizer, make_config(verify=True))
    text = "abcdefghijklmno"

    try:
        actual = lopt.encode(text, **encode_kwargs(add_special_tokens=False))
    finally:
        lopt.shutdown()

    assert actual == tokenizer.encode(text, **encode_kwargs(add_special_tokens=False))


def test_verify_mode_logs_serial_and_lopt_timings(monkeypatch, caplog):
    tokenizer = CharacterTokenizer()
    lopt = LosslessParallelTokenizer(tokenizer, make_config(verify=True))
    text = "abcdefghijklmno"
    clock = iter([1.0, 1.01, 2.0, 2.025])
    monkeypatch.setattr(lopt_tokenizer_module.time, "perf_counter", lambda: next(clock))
    caplog.set_level(logging.INFO, logger=lopt_tokenizer_module.logger.name)

    try:
        actual = lopt.encode(text, **encode_kwargs(add_special_tokens=False))
    finally:
        lopt.shutdown()

    assert actual == tokenizer.encode(text, **encode_kwargs(add_special_tokens=False))
    assert (
        "LoPT timing comparison: characters=15 tokens=15 serial_ms=25.000 "
        "lopt_ms=10.000 speedup=2.50x matched=True chunks=4 attempts=1"
    ) in caplog.messages


def test_chunk_tokenization_runs_concurrently():
    tokenizer = ConcurrentTokenizer()
    lopt = LosslessParallelTokenizer(tokenizer, make_config(thread_workers=4))

    try:
        actual = lopt.encode("abcdefghijklmnopqrstuvwxyz", **encode_kwargs(add_special_tokens=False))
    finally:
        lopt.shutdown()

    assert actual == tokenizer.encode("abcdefghijklmnopqrstuvwxyz", **encode_kwargs(add_special_tokens=False))
    assert tokenizer.max_active > 1


def test_fast_tokenizer_uses_native_batch_encoding_when_available():
    tokenizer = BatchTokenizer()
    lopt = LosslessParallelTokenizer(tokenizer, make_config(thread_workers=4))
    text = "abcdefghijklmnopqrstuvwxyz"
    kwargs = encode_kwargs(add_special_tokens=False)

    try:
        actual = lopt.encode(text, **kwargs)
        repeated = lopt.encode(text, **kwargs)
    finally:
        lopt.shutdown()

    assert actual == tokenizer.encode(text, **kwargs)
    assert repeated == actual
    assert tokenizer.batch_call_count == 3


def test_slow_and_short_prompts_use_standard_path_without_chunk_calls():
    tokenizer = CharacterTokenizer()
    tokenizer.is_fast = False
    lopt = LosslessParallelTokenizer(tokenizer, make_config(min_chars=20))

    try:
        assert lopt.encode("short", **encode_kwargs()) == tokenizer.encode("short", **encode_kwargs())
    finally:
        lopt.shutdown()

    assert tokenizer.chunk_call_count == 0


def test_merge_rejects_position_mismatch():
    left = ChunkEncoding(0, 0, 4, (1, 2), ((0, 1), (1, 2)), ((0, 1), (1, 2)))
    right = ChunkEncoding(1, 2, 6, (1, 2), ((0, 1), (1, 2)), ((2, 3), (3, 4)))

    with pytest.raises(RuntimeError, match="position-aware overlap"):
        _merge_chunk_encodings([left, right], min_match_tokens=1)
