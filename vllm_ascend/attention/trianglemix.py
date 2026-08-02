#
# Copyright (c) 2026 TriangleMix contributors.
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
#
"""Optional TriangleMix sparse-prefill routing for the Ascend backend."""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import regex as re
import torch
from vllm.logger import logger

TRIANGLE_QUERY_HEADS = 32
TRIANGLE_KV_HEADS = 8
TRIANGLE_HEAD_SIZE = 128
TRIANGLE_CACHE_BLOCK_SIZE = 128
TRIANGLE_SINK_TOKENS = 8
TRIANGLE_LOCAL_WINDOW = 512
TRIANGLE_DENSE_TAIL = 128

_LAYER_INDEX_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
_PREFILL_STATES = frozenset({"PrefillNoCache", "PrefillCacheHit", "ChunkedPrefill"})


class TriangleMixFallbackReason(str, Enum):
    NONE = "none"
    DISABLED = "disabled"
    LAYER_NOT_SELECTED = "layer_not_selected"
    STATE_UNSUPPORTED = "state_unsupported"
    MIXED_DECODE = "mixed_decode"
    BATCH_UNSUPPORTED = "batch_unsupported"
    MISSING_METADATA = "missing_metadata"
    INVALID_LENGTHS = "invalid_lengths"
    NO_SPARSE_MIDDLE = "no_sparse_middle"
    BELOW_MIN_SPARSE_ROWS = "below_min_sparse_rows"
    BELOW_MIN_SAVED_QK = "below_min_saved_qk"
    GEOMETRY_UNSUPPORTED = "geometry_unsupported"
    GRAPH_CAPTURE = "graph_capture"
    CONTEXT_PARALLEL = "context_parallel"
    TENSOR_PARALLEL = "tensor_parallel"
    NON_CAUSAL = "non_causal"
    MODEL_UNSUPPORTED = "model_unsupported"
    QUERY_UNSUPPORTED = "query_unsupported"
    KV_CACHE_UNSUPPORTED = "kv_cache_unsupported"
    BLOCK_TABLE_UNSUPPORTED = "block_table_unsupported"
    OPERATOR_UNAVAILABLE = "operator_unavailable"
    OPERATOR_ERROR = "operator_error"


def parse_layer_indices(value: object) -> frozenset[int]:
    if isinstance(value, (list, tuple, set, frozenset)):
        value = ",".join(str(item) for item in value)
    text = str(value or "").strip()
    if not text:
        return frozenset()
    indices: set[int] = set()
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start < 0 or end < start:
                raise ValueError(f"Invalid TriangleMix layer range: {item!r}")
            indices.update(range(start, end + 1))
        else:
            index = int(item)
            if index < 0:
                raise ValueError(f"Invalid TriangleMix layer index: {item!r}")
            indices.add(index)
    return frozenset(indices)


def _as_bool(value: object, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


@dataclass(frozen=True)
class TriangleMixConfig:
    enabled: bool = False
    layer_indices: frozenset[int] = frozenset()
    strict: bool = False
    min_sparse_rows: int = 128
    min_saved_qk: int = 913_152
    split_min_sparse_rows: int = 192
    split_min_saved_qk: int = 1_299_264
    stats_log_interval: int = 0

    @classmethod
    def from_mapping(
        cls,
        value: object,
    ) -> TriangleMixConfig:
        section: Mapping[str, Any] = value if isinstance(value, Mapping) else {}
        config = cls(
            enabled=_as_bool(section.get("enabled", False), name="trianglemix.enabled"),
            layer_indices=parse_layer_indices(section.get("layers", "")),
            strict=_as_bool(section.get("strict", False), name="trianglemix.strict"),
            min_sparse_rows=int(section.get("min_sparse_rows", 128)),
            min_saved_qk=int(section.get("min_saved_qk", 913_152)),
            split_min_sparse_rows=int(section.get("split_min_sparse_rows", 192)),
            split_min_saved_qk=int(section.get("split_min_saved_qk", 1_299_264)),
            stats_log_interval=int(section.get("stats_log_interval", 0)),
        )
        if (
            config.min_sparse_rows < 0
            or config.min_saved_qk < 0
            or config.split_min_sparse_rows < 0
            or config.split_min_saved_qk < 0
            or config.stats_log_interval < 0
        ):
            raise ValueError("TriangleMix thresholds must be non-negative")
        if config.enabled and not config.layer_indices:
            raise ValueError("trianglemix.enabled requires at least one selected layer")
        return config

    def uses_layer(self, layer_name: str) -> bool:
        match = _LAYER_INDEX_RE.search(layer_name)
        return self.enabled and match is not None and int(match.group(1)) in self.layer_indices


@dataclass(frozen=True)
class TriangleMixRequestPlan:
    query_len: int
    seq_len: int
    prompt_len: int
    query_start: int
    sparse_start: int
    sparse_end: int
    saved_qk: int
    reason: TriangleMixFallbackReason

    @property
    def direct(self) -> bool:
        return self.reason is TriangleMixFallbackReason.NONE


def _saved_qk(sparse_start: int, sparse_end: int) -> int:
    if sparse_end <= sparse_start:
        return 0
    rows = sparse_end - sparse_start
    return rows * (sparse_start + sparse_end - 1041) // 2


def build_trianglemix_plan(
    *,
    state_name: str,
    cumulative_query_ends: Sequence[int] | None,
    seq_lens: Sequence[int] | None,
    prompt_lens: Sequence[int] | None,
    num_decodes: int,
    num_prefills: int,
    config: TriangleMixConfig,
) -> TriangleMixRequestPlan:
    query_ends = tuple(int(value) for value in cumulative_query_ends or ())
    sequences = tuple(int(value) for value in seq_lens or ())
    prompts = tuple(int(value) for value in prompt_lens or ())
    if state_name not in _PREFILL_STATES:
        reason = TriangleMixFallbackReason.STATE_UNSUPPORTED
    elif num_decodes:
        reason = TriangleMixFallbackReason.MIXED_DECODE
    elif len(sequences) != 1 or len(query_ends) != 1 or len(prompts) != 1 or num_prefills != 1:
        reason = (
            TriangleMixFallbackReason.BATCH_UNSUPPORTED if sequences else TriangleMixFallbackReason.MISSING_METADATA
        )
    else:
        reason = TriangleMixFallbackReason.NONE

    query_len = query_ends[0] if len(query_ends) == 1 else 0
    seq_len = sequences[0] if len(sequences) == 1 else 0
    prompt_len = prompts[0] if len(prompts) == 1 else 0
    if reason is TriangleMixFallbackReason.NONE and (query_len <= 0 or seq_len < query_len or prompt_len < seq_len):
        reason = TriangleMixFallbackReason.INVALID_LENGTHS

    query_start = max(0, seq_len - query_len)
    sparse_start = max(query_start, TRIANGLE_SINK_TOKENS + TRIANGLE_LOCAL_WINDOW + 1)
    sparse_end = min(seq_len, max(0, prompt_len - TRIANGLE_DENSE_TAIL))
    saved_qk = _saved_qk(sparse_start, sparse_end)
    if reason is TriangleMixFallbackReason.NONE:
        if sparse_end <= sparse_start:
            reason = TriangleMixFallbackReason.NO_SPARSE_MIDDLE
        elif sparse_end - sparse_start < config.min_sparse_rows:
            reason = TriangleMixFallbackReason.BELOW_MIN_SPARSE_ROWS
        elif saved_qk < config.min_saved_qk:
            reason = TriangleMixFallbackReason.BELOW_MIN_SAVED_QK
        elif (
            sparse_start > query_start or sparse_end < seq_len
        ) and sparse_end - sparse_start < config.split_min_sparse_rows:
            reason = TriangleMixFallbackReason.BELOW_MIN_SPARSE_ROWS
        elif (sparse_start > query_start or sparse_end < seq_len) and saved_qk < config.split_min_saved_qk:
            reason = TriangleMixFallbackReason.BELOW_MIN_SAVED_QK
    return TriangleMixRequestPlan(
        query_len=query_len,
        seq_len=seq_len,
        prompt_len=prompt_len,
        query_start=query_start,
        sparse_start=sparse_start,
        sparse_end=sparse_end,
        saved_qk=saved_qk,
        reason=reason,
    )


def trianglemix_dispatch_reason(
    *,
    config: TriangleMixConfig,
    plan: TriangleMixRequestPlan | None,
    layer_name: str,
    query: torch.Tensor,
    output: torch.Tensor,
    key_cache: torch.Tensor | None,
    value_cache: torch.Tensor | None,
    block_table: torch.Tensor | None,
    causal: bool,
    capturing: bool,
    tensor_parallel_size: int,
    context_parallel_enabled: bool,
    sliding_window: int | None,
    sinks: torch.Tensor | None,
    alibi_slopes: torch.Tensor | None,
    enable_c8_quant: bool,
) -> TriangleMixFallbackReason:
    if not config.enabled:
        return TriangleMixFallbackReason.DISABLED
    if not config.uses_layer(layer_name):
        return TriangleMixFallbackReason.LAYER_NOT_SELECTED
    if plan is None:
        return TriangleMixFallbackReason.MISSING_METADATA
    if capturing:
        return TriangleMixFallbackReason.GRAPH_CAPTURE
    if not plan.direct:
        return plan.reason
    if tensor_parallel_size != 1:
        return TriangleMixFallbackReason.TENSOR_PARALLEL
    if context_parallel_enabled:
        return TriangleMixFallbackReason.CONTEXT_PARALLEL
    if not causal:
        return TriangleMixFallbackReason.NON_CAUSAL
    if sliding_window is not None or sinks is not None or alibi_slopes is not None or enable_c8_quant:
        return TriangleMixFallbackReason.MODEL_UNSUPPORTED
    if (
        query.ndim != 3
        or tuple(query.shape)
        != (
            plan.query_len,
            TRIANGLE_QUERY_HEADS,
            TRIANGLE_HEAD_SIZE,
        )
        or query.dtype != torch.bfloat16
        or not query.is_contiguous()
        or output.dtype != torch.bfloat16
        or output.shape[0] < plan.query_len
        or not output.is_contiguous()
    ):
        return TriangleMixFallbackReason.QUERY_UNSUPPORTED
    expected_cache_tail = (
        TRIANGLE_CACHE_BLOCK_SIZE,
        TRIANGLE_KV_HEADS,
        TRIANGLE_HEAD_SIZE,
    )
    if (
        key_cache is None
        or value_cache is None
        or key_cache.ndim != 4
        or tuple(key_cache.shape[1:]) != expected_cache_tail
        or value_cache.shape != key_cache.shape
        or key_cache.dtype != torch.bfloat16
        or value_cache.dtype != torch.bfloat16
        or not key_cache.is_contiguous()
        or not value_cache.is_contiguous()
    ):
        return TriangleMixFallbackReason.KV_CACHE_UNSUPPORTED
    required_pages = (plan.seq_len + TRIANGLE_CACHE_BLOCK_SIZE - 1) // TRIANGLE_CACHE_BLOCK_SIZE
    if (
        block_table is None
        or block_table.ndim != 2
        or block_table.shape[0] != 1
        or block_table.shape[1] < required_pages
        or block_table.dtype != torch.int32
        or not block_table.is_contiguous()
    ):
        return TriangleMixFallbackReason.BLOCK_TABLE_UNSUPPORTED
    if any(tensor.device != query.device for tensor in (key_cache, value_cache, block_table, output)):
        return TriangleMixFallbackReason.KV_CACHE_UNSUPPORTED
    try:
        _ = torch.ops._C_ascend.npu_triangle_paged_sparse_attention
    except AttributeError:
        return TriangleMixFallbackReason.OPERATOR_UNAVAILABLE
    return TriangleMixFallbackReason.NONE


def run_trianglemix(
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    plan: TriangleMixRequestPlan,
    scale: float,
    output: torch.Tensor,
) -> torch.Tensor:
    output_view = output[: plan.query_len].view_as(query)
    torch.ops._C_ascend.npu_triangle_paged_sparse_attention(
        query,
        key_cache,
        value_cache,
        block_table,
        plan.query_start,
        plan.seq_len,
        plan.prompt_len,
        scale,
        output_view,
    )
    return output


@dataclass
class TriangleMixRuntimeStats:
    """Per-backend counters; no request data is retained."""

    config: TriangleMixConfig
    counters: Counter[str] = field(default_factory=Counter)

    def record(
        self,
        *,
        layer_name: str,
        reason: TriangleMixFallbackReason,
        saved_qk: int = 0,
        enqueue_ns: int = 0,
    ) -> None:
        self.counters["calls"] += 1
        if reason is TriangleMixFallbackReason.NONE:
            self.counters["hits"] += 1
            self.counters["estimated_saved_qk"] += saved_qk
            self.counters["host_enqueue_ns"] += enqueue_ns
        else:
            self.counters["fallbacks"] += 1
            self.counters[f"fallback:{reason.value}"] += 1
        interval = self.config.stats_log_interval
        if interval and self.counters["calls"] % interval == 0:
            logger.info(
                "TriangleMix stats layer=%s calls=%d hits=%d fallbacks=%d "
                "reason=%s estimated_saved_qk=%d host_enqueue_ns=%d",
                layer_name,
                self.counters["calls"],
                self.counters["hits"],
                self.counters["fallbacks"],
                reason.value,
                self.counters["estimated_saved_qk"],
                self.counters["host_enqueue_ns"],
            )


def timed_trianglemix_launch(**kwargs: Any) -> tuple[torch.Tensor, int]:
    started = time.perf_counter_ns()
    result = run_trianglemix(**kwargs)
    return result, time.perf_counter_ns() - started
