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
# This file is a part of the vllm-ascend project.
#

"""Warm up the in-tree CPU pattern-repeat backend from the served tokenizer."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

from vllm_ascend.logger import init_logger_ascend

logger = init_logger_ascend(__name__)


class PatternResult(Protocol):
    is_ill: bool
    fired_at: int
    reason: str  # short on-call title
    rule: str  # long rule id


class PatternBackend(Protocol):
    def start(self) -> Any: ...
    def feed_events(self, states: list[Any], inputs: list[list[int]]) -> list[tuple[int, int, str, str]]: ...
    def progress(self, state: Any) -> tuple[int, int, int, str, str] | None: ...
    def finish(self, state: Any) -> PatternResult: ...


def load_pattern_backend(runner: Any, think_end: int) -> PatternBackend:
    """Build tables once during detector warmup; never read a separately deployed NPZ."""
    if isinstance(think_end, bool) or not isinstance(think_end, int) or think_end < -1:
        raise ValueError("think_end must be -1 (chat) or a token ID")
    model_config = getattr(runner, "model_config", None)
    served = getattr(model_config, "tokenizer", None)
    if not served:
        raise ValueError("token_repeat requires runner.model_config.tokenizer")
    source = Path(served) / "tokenizer.json"
    if not source.is_file():
        from transformers.utils.hub import cached_file

        cached = cached_file(
            served,
            "tokenizer.json",
            local_files_only=True,
            revision=getattr(model_config, "tokenizer_revision", None),
        )
        if cached is None:
            raise ValueError("token_repeat requires the served tokenizer.json in the local model cache")
        source = Path(cached)
    from vllm_ascend.observability.runtime_guard.native_pattern import PatternDetector
    from vllm_ascend.observability.runtime_guard.native_pattern import vocab as pattern_vocab

    started = perf_counter()
    try:
        tables = pattern_vocab.build_tables(str(source))
    except ImportError as exc:
        raise RuntimeError(
            "token_repeat pattern backend requires a compiled vllm-ascend install (native_pattern / tokenizers)"
        ) from exc
    if think_end >= len(tables["norm"]):
        raise ValueError("think_end is outside the served tokenizer vocabulary")
    try:
        backend = PatternDetector(tables, think_end=None if think_end == -1 else think_end)
    except ImportError as exc:
        raise RuntimeError(
            "token_repeat pattern backend requires a compiled vllm-ascend install (native_pattern)"
        ) from exc
    elapsed = perf_counter() - started
    logger.info(
        "[runtime_guard: token_repeat] pattern table build and native init: %d tokens in %.3fs",
        len(tables["norm"]),
        elapsed,
    )
    if elapsed > 5.0:
        logger.warning(
            "[runtime_guard: token_repeat] pattern table build and native init exceeded the 5s target: %.3fs",
            elapsed,
        )
    return backend
