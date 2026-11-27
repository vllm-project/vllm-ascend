# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Minimally invasive HiSparse on-demand & heterogeneous prefill pipeline.

Key properties:
1. Zero vLLM Core / Scheduler / Request-ordering modifications:
   - Tracks per-request Host block IDs and Resident shadow bitmap on CPU via
     HiSparseConnector worker hooks.
   - Fixes mixed Decode + 100%-Resident Prefill batches so Tier-0 prefill
     keeps zero-copy PA_BSND on Resident.
2. Single unsplit persistent staging_pool with foolproof hard floor:
   - Enforces capacity >= 2 * max(max_model_len, 1_048_576) * 656 B
     (default >= 2,097,152 rows = 1,312 MiB for C8).
3. Pre-forward 3-tier elastic classification:
   - Tier 0: H_host == 0 -> On-Card Resident.
   - Tier 1: H_host > 32K and Q / H_host <= 1% -> On-Demand AIV gather.
   - Tier 2: H_host > 32K and 1% < Q / H_host <= 5% -> Elastic (Full-Load SDMA
     when staging_pool fits; progressively downgraded smallest-ratio-first to
     On-Demand when staging_pool is pressured).
   - Tier 3: H_host <= 32K or Q / H_host > 5% -> Full-Load SDMA.
4. Dual-channel transfer + post-SFA follower overlap + whole-request wave guard:
   - Layer 0 Full-Load SDMA starts in prepare_forward (before Input Embedding).
   - Group leaders compute 32-AIV bitmap union + unified physical TopK remap
     once per HiSparseMLAIndexGroup.
   - Post-SFA launches next layer's Wave-0 Host prefetch (SDMA across all
     layers + AIV gather on group followers) behind V-up + O-proj + MoE.
   - Two-beat overflow guard: zero NPU->CPU sync when Worst_Max <= C_stage;
     checks actual union counts only when Worst_Max > C_stage, splitting into
     whole-request waves only if actual total exceeds C_stage.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from functools import wraps
from typing import Any

import numpy as np
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import logger
from vllm.utils.math_utils import cdiv, round_up

from vllm_ascend.hisparse.c8 import C8_ROW_BYTES

DEFAULT_FLOOR_MODEL_LEN = 1_048_576
MAX_BITMAP_HISTORY = 1_048_576
MAX_BITMAP_WORDS = MAX_BITMAP_HISTORY // 32
MAX_FULL_PAGE_COLUMNS = MAX_BITMAP_WORDS
MAX_SUPER_BLOCKS = MAX_BITMAP_HISTORY // 256
MAX_KERNEL_TOKENS = 65_536
MAX_KERNEL_REQS = 256
BLOCK_SIZE = 128
TOPK_WIDTH = 2048
SDMA_BATCH_LIMIT = 2047

TIER_ON_CARD = 0
TIER_HARD_DEMAND = 1
TIER_ELASTIC = 2
TIER_FULL_LOAD = 3

MODE_FULL = 0
MODE_DEMAND = 1


def compute_staging_floor_rows(max_model_len: int, block_size: int = BLOCK_SIZE) -> int:
    """Hard floor in rows: >= 2 * max(max_model_len, 1_048_576), aligned to block_size."""
    if max_model_len <= 0 or block_size <= 0:
        raise ValueError(f"max_model_len ({max_model_len}) and block_size ({block_size}) must be positive")
    effective_model_len = max(int(max_model_len), DEFAULT_FLOOR_MODEL_LEN)
    return round_up(2 * effective_model_len, block_size)


def compute_staging_floor_bytes(max_model_len: int, row_bytes: int = C8_ROW_BYTES, block_size: int = BLOCK_SIZE) -> int:
    """Hard floor in bytes: >= 2 * max(max_model_len, 1_048_576) * 656 B."""
    if row_bytes <= 0:
        raise ValueError(f"row_bytes must be positive, got {row_bytes}")
    return compute_staging_floor_rows(max_model_len, block_size) * int(row_bytes)


def _to_signed_int64(unsigned_addr: int) -> int:
    if unsigned_addr < 0 or unsigned_addr >= (1 << 64):
        raise ValueError(f"Address out of uint64 range: {unsigned_addr}")
    return unsigned_addr - (1 << 64) if unsigned_addr >= (1 << 63) else unsigned_addr


@dataclass(frozen=True)
class HiSparsePrefillConfig:
    enabled: bool = False
    min_history_tokens: int = 32_768
    tier1_max_ratio: float = 0.01
    tier2_max_ratio: float = 0.05
    staging_rows: int = 2 * DEFAULT_FLOOR_MODEL_LEN

    @classmethod
    def from_vllm_config(cls, vllm_config: Any) -> HiSparsePrefillConfig:
        additional = getattr(vllm_config, "additional_config", None) or {}
        raw = additional.get("hisparse_prefill_config") if isinstance(additional, dict) else None
        model_cfg = getattr(vllm_config, "model_config", None)
        max_model_len = int(getattr(model_cfg, "max_model_len", DEFAULT_FLOOR_MODEL_LEN) or DEFAULT_FLOOR_MODEL_LEN)
        if max_model_len <= 0:
            raise ValueError(f"max_model_len ({max_model_len}) must be positive")
        raw_floor_rows = 2 * max(max_model_len, DEFAULT_FLOOR_MODEL_LEN)
        raw_floor_bytes = raw_floor_rows * C8_ROW_BYTES
        floor_rows = compute_staging_floor_rows(max_model_len, BLOCK_SIZE)

        if raw is None or raw is False:
            return cls(enabled=False, staging_rows=floor_rows)
        if raw is True:
            raw = {"enabled": True}
        if not isinstance(raw, dict):
            raise ValueError(f"additional_config.hisparse_prefill_config must be a dict or bool, got {type(raw)!r}")

        allowed_keys = {
            "enabled",
            "min_history_tokens",
            "tier1_max_ratio",
            "tier2_max_ratio",
            "staging_rows",
            "staging_bytes",
        }
        unknown = sorted(set(raw) - allowed_keys)
        if unknown:
            raise ValueError(f"Unknown keys in hisparse_prefill_config: {unknown}")

        enabled = raw.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ValueError("hisparse_prefill_config.enabled must be a boolean")

        min_hist = raw.get("min_history_tokens", 32_768)
        if type(min_hist) is not int or min_hist < 0:
            raise ValueError("hisparse_prefill_config.min_history_tokens must be a non-negative integer")

        t1 = raw.get("tier1_max_ratio", 0.01)
        t2 = raw.get("tier2_max_ratio", 0.05)
        if type(t1) not in (int, float) or type(t2) not in (int, float):
            raise ValueError("hisparse_prefill_config ratios must be numbers")
        t1_f, t2_f = float(t1), float(t2)
        if not (math.isfinite(t1_f) and math.isfinite(t2_f) and 0.0 <= t1_f <= t2_f <= 1.0):
            raise ValueError("hisparse_prefill_config requires 0 <= tier1_max_ratio <= tier2_max_ratio <= 1")

        req_rows = raw.get("staging_rows")
        req_bytes = raw.get("staging_bytes")
        if req_rows is not None and req_bytes is not None:
            raise ValueError("Specify at most one of staging_rows or staging_bytes in hisparse_prefill_config")
        if req_rows is not None:
            if type(req_rows) is not int or req_rows < raw_floor_rows:
                raise ValueError(
                    f"hisparse_prefill_config.staging_rows={req_rows!r} violates hard floor >= {raw_floor_rows} rows "
                    f"(2 * max(max_model_len={max_model_len}, {DEFAULT_FLOOR_MODEL_LEN}))"
                )
            resolved_rows = round_up(req_rows, BLOCK_SIZE)
        elif req_bytes is not None:
            if type(req_bytes) is not int or req_bytes < raw_floor_bytes:
                raise ValueError(
                    f"hisparse_prefill_config.staging_bytes={req_bytes!r} violates hard floor >= {raw_floor_bytes} B "
                    f"(2 * max(max_model_len={max_model_len}, {DEFAULT_FLOOR_MODEL_LEN}) * {C8_ROW_BYTES} B)"
                )
            resolved_rows = round_up(cdiv(req_bytes, C8_ROW_BYTES), BLOCK_SIZE)
        else:
            resolved_rows = floor_rows

        if resolved_rows >= (1 << 31):
            raise ValueError(f"hisparse_prefill_config staging_rows={resolved_rows} exceeds int32 physical row limit")

        cfg = cls(
            enabled=enabled,
            min_history_tokens=min_hist,
            tier1_max_ratio=t1_f,
            tier2_max_ratio=t2_f,
            staging_rows=resolved_rows,
        )
        if cfg.enabled:
            cfg._validate_deployment(vllm_config)
        return cfg

    def _validate_deployment(self, vllm_config: Any) -> None:
        attn_cfg = getattr(vllm_config, "attention_config", None)
        if getattr(attn_cfg, "hisparse_config", None) is None:
            raise ValueError("hisparse_prefill_config.enabled requires attention_config.hisparse_config")
        cache_cfg = getattr(vllm_config, "cache_config", None)
        if getattr(cache_cfg, "cache_dtype", None) not in ("int8", "fp8"):
            raise ValueError("hisparse_prefill_config.enabled requires C8 cache_dtype ('int8' or 'fp8')")
        if getattr(cache_cfg, "block_size", BLOCK_SIZE) != BLOCK_SIZE:
            raise ValueError(f"hisparse_prefill_config.enabled requires block_size == {BLOCK_SIZE}")
        model_cfg = getattr(vllm_config, "model_config", None)
        hf_text = getattr(model_cfg, "hf_text_config", None)
        index_topk = getattr(hf_text, "index_topk", None)
        if type(index_topk) is int and index_topk != TOPK_WIDTH:
            raise ValueError(f"hisparse_prefill_config.enabled requires index_topk == {TOPK_WIDTH}, got {index_topk}")
        par_cfg = getattr(vllm_config, "parallel_config", None)
        if par_cfg is not None:
            if (
                getattr(par_cfg, "pipeline_parallel_size", 1) != 1
                or getattr(par_cfg, "prefill_context_parallel_size", 1) != 1
                or getattr(par_cfg, "decode_context_parallel_size", 1) != 1
            ):
                raise ValueError("hisparse_prefill_config.enabled requires PP == 1, PCP == 1, and DCP == 1")


def validate_hisparse_prefill_config(vllm_config: Any) -> HiSparsePrefillConfig:
    return HiSparsePrefillConfig.from_vllm_config(vllm_config)


def is_hisparse_prefill_enabled(vllm_config: Any) -> bool:
    try:
        return HiSparsePrefillConfig.from_vllm_config(vllm_config).enabled
    except Exception:
        return False


@dataclass
class PrefillRequestSpec:
    prefill_idx: int
    batch_req_idx: int
    req_id: str
    history_tokens: int
    query_tokens: int
    q_batch_start: int
    host_blocks: tuple[int, ...]
    # Shape: [num_resident_groups, num_pages], int32 (>0 resident block, 0 host-only)
    residency_by_group: np.ndarray
    host_miss_pages: int
    host_history_tokens: int
    tier: int = TIER_ON_CARD
    mode: int = MODE_FULL
    downgraded_from_tier2: bool = False
    demand_slot: int = -1

    @property
    def seq_len(self) -> int:
        return self.history_tokens + self.query_tokens

    @property
    def num_pages(self) -> int:
        return cdiv(self.seq_len, BLOCK_SIZE)

    @property
    def sealed_history_pages(self) -> int:
        return self.history_tokens // BLOCK_SIZE

    @property
    def hit_ratio(self) -> float:
        if self.host_history_tokens <= 0:
            return float("inf")
        return self.query_tokens / self.host_history_tokens

    @property
    def worst_union_rows(self) -> int:
        return min(self.history_tokens, self.query_tokens * TOPK_WIDTH)


@dataclass
class SDMAPageEntry:
    host_block_id: int
    dst_page: int


@dataclass
class WaveExecutionPlan:
    wave_idx: int
    req_start: int
    req_end: int
    q_start: int
    q_end: int
    total_staging_rows: int
    full_pages: int
    # Per resident group g: list of (host_block_id, dst_page) for H2D SDMA
    sdma_entries_by_group: tuple[tuple[SDMAPageEntry, ...], ...]
    # Per resident group g: int32 array [N_spans, 8] for Resident/new-Q D2D copy
    d2d_spans_by_group: tuple[np.ndarray, ...]
    # int32 array [num_reqs_in_wave, 8] for hisparse_prefill_remap_topk
    remap_desc: np.ndarray
    # int32 array [num_full_reqs_in_wave, max_pages] for staged_page_table
    staged_page_table: np.ndarray
    # int32 array [num_demand_reqs_in_wave, 4] for hisparse_prefill_gather_union
    gather_meta: np.ndarray
    # GPU mirrors lazily uploaded once per wave
    remap_desc_gpu: torch.Tensor | None = None
    staged_page_table_gpu: torch.Tensor | None = None
    gather_meta_gpu: torch.Tensor | None = None
    d2d_spans_gpu_by_group: list[tuple[torch.Tensor, int]] = field(default_factory=list)


@dataclass
class PrefillStepPlan:
    step_id: int
    all_prefill_resident: bool
    requests: list[PrefillRequestSpec]
    demand_requests: list[PrefillRequestSpec]
    staging_rows: int
    num_resident_groups: int
    worst_total_rows: int
    needs_union_count_check: bool
    # Pre-built wave(s). When needs_union_count_check is False, waves is final (len == 1).
    waves: list[WaveExecutionPlan]
    # Conservative Wave-0 plan under worst-case union bounds (safe across index-group boundaries).
    conservative_wave0: WaveExecutionPlan | None = None
    # Per-index-group state reset at index_group_index == 0
    active_group_id: int | None = None
    group_waves: list[WaveExecutionPlan] | None = None
    last_union_counts: tuple[int, ...] = ()


def _count_dedup_full_pages(requests: list[PrefillRequestSpec]) -> int:
    """Count staging pages needed by FULL/ON_CARD requests with host_block_id deduplication."""
    seen_sealed_host_blocks: set[int] = set()
    tail_pages = 0
    for req in requests:
        if req.mode != MODE_FULL:
            continue
        sealed = req.sealed_history_pages
        for col in range(sealed):
            seen_sealed_host_blocks.add(int(req.host_blocks[col]))
        tail_pages += req.num_pages - sealed
    return len(seen_sealed_host_blocks) + tail_pages


def classify_prefill_requests(
    requests: list[PrefillRequestSpec],
    config: HiSparsePrefillConfig,
    *,
    topk: int = TOPK_WIDTH,
) -> tuple[bool, int, bool]:
    """Apply 3-tier pre-forward classification and elastic Tier-2 downgrade in place.

    Returns:
        (all_prefill_resident, worst_total_rows, needs_union_count_check)
    """
    if not requests:
        return True, 0, False

    has_any_host_miss = False
    tier2_candidates: list[PrefillRequestSpec] = []

    for req in requests:
        if req.host_history_tokens <= 0:
            req.tier = TIER_ON_CARD
            req.mode = MODE_FULL
            continue
        has_any_host_miss = True
        ratio = req.query_tokens / req.host_history_tokens
        can_use_bitmap = req.history_tokens <= MAX_BITMAP_HISTORY
        if can_use_bitmap and req.host_history_tokens > config.min_history_tokens and ratio <= config.tier1_max_ratio:
            req.tier = TIER_HARD_DEMAND
            req.mode = MODE_DEMAND
        elif can_use_bitmap and req.host_history_tokens > config.min_history_tokens and ratio <= config.tier2_max_ratio:
            req.tier = TIER_ELASTIC
            req.mode = MODE_FULL
            tier2_candidates.append(req)
        else:
            req.tier = TIER_FULL_LOAD
            req.mode = MODE_FULL

    if not has_any_host_miss:
        return True, 0, False

    def current_pressure_rows() -> int:
        full_rows = _count_dedup_full_pages(requests) * BLOCK_SIZE
        demand_rows = 0
        for req in requests:
            if req.mode == MODE_DEMAND:
                if req.downgraded_from_tier2:
                    # Elastic Tier-2 expected footprint (~25% upper estimate for admission;
                    # exact union count is verified in Beat 2 at group leader).
                    demand_rows += req.query_tokens + min(req.history_tokens, max(topk, req.history_tokens // 4))
                else:
                    demand_rows += req.query_tokens + min(req.history_tokens, req.query_tokens * topk)
        return full_rows + demand_rows

    # Downgrade Tier-2 requests (lowest Q/H_host first, breaking ties by largest H_host)
    # whenever staging_pool cannot comfortably hold Full-Load + Tier-1 worst-case.
    tier2_candidates.sort(key=lambda r: (r.hit_ratio, -r.host_history_tokens, r.prefill_idx))
    for cand in tier2_candidates:
        if current_pressure_rows() <= config.staging_rows:
            break
        cand.mode = MODE_DEMAND
        cand.downgraded_from_tier2 = True

    demand_slot = 0
    for req in requests:
        if req.mode == MODE_DEMAND:
            req.demand_slot = demand_slot
            demand_slot += 1
        else:
            req.demand_slot = -1

    full_rows = _count_dedup_full_pages(requests) * BLOCK_SIZE
    worst_demand_rows = sum(
        req.query_tokens + min(req.history_tokens, req.query_tokens * topk)
        for req in requests
        if req.mode == MODE_DEMAND
    )
    worst_total_rows = full_rows + worst_demand_rows
    needs_union_check = worst_total_rows > config.staging_rows and demand_slot > 0
    return False, worst_total_rows, needs_union_check


def _build_single_wave(
    wave_idx: int,
    all_requests: list[PrefillRequestSpec],
    req_start: int,
    req_end: int,
    num_resident_groups: int,
    union_capacities: dict[int, int],
    staging_rows: int,
) -> WaveExecutionPlan:
    """Pack contiguous whole-request slice [req_start, req_end) into staging_pool."""
    sub_reqs = all_requests[req_start:req_end]
    assert sub_reqs, "Wave must contain at least one request"
    q_start = sub_reqs[0].q_batch_start
    q_end = sub_reqs[-1].q_batch_start + sub_reqs[-1].query_tokens

    # Zone A: FULL / ON_CARD pages (deduplicating sealed host_block_ids within this wave)
    sealed_map: dict[int, tuple[int, PrefillRequestSpec, int]] = {}
    full_reqs = [r for r in sub_reqs if r.mode == MODE_FULL]
    max_pages = max((r.num_pages for r in full_reqs), default=1)
    staged_page_table = np.zeros((max(len(full_reqs), 1), max_pages), dtype=np.int32)

    next_dst_page = 0
    tail_allocs: list[tuple[PrefillRequestSpec, int, int, int]] = []
    full_slot_by_req: dict[int, int] = {}

    for f_slot, req in enumerate(full_reqs):
        full_slot_by_req[req.prefill_idx] = f_slot
        sealed = req.sealed_history_pages
        for col in range(req.num_pages):
            if col < sealed:
                h_blk = int(req.host_blocks[col])
                existing = sealed_map.get(h_blk)
                if existing is None:
                    dst_pg = next_dst_page
                    next_dst_page += 1
                    sealed_map[h_blk] = (dst_pg, req, col)
                else:
                    dst_pg = existing[0]
                staged_page_table[f_slot, col] = dst_pg
            else:
                dst_pg = next_dst_page
                next_dst_page += 1
                staged_page_table[f_slot, col] = dst_pg
                valid_rows = min(BLOCK_SIZE, req.seq_len - col * BLOCK_SIZE)
                tail_allocs.append((req, col, dst_pg, valid_rows))

    full_pages = next_dst_page
    cursor_row = full_pages * BLOCK_SIZE

    # Zone B: ON_DEMAND new query rows [H_r, H_r + Q_r)
    demand_reqs = [r for r in sub_reqs if r.mode == MODE_DEMAND]
    q_dst_base_by_req: dict[int, int] = {}
    for req in demand_reqs:
        q_dst_base_by_req[req.prefill_idx] = cursor_row
        cursor_row += req.query_tokens

    # Zone C: ON_DEMAND deduplicated historical union rows [0, U_r)
    u_dst_base_by_req: dict[int, int] = {}
    for req in demand_reqs:
        u_cap = int(union_capacities[req.demand_slot])
        u_dst_base_by_req[req.prefill_idx] = cursor_row
        cursor_row += u_cap

    if cursor_row > staging_rows:
        raise ValueError(
            f"Wave {wave_idx} staging allocation ({cursor_row} rows) exceeds staging_pool ({staging_rows} rows)"
        )

    # Build per-resident-group SDMA (res_blk == 0) & D2D spans (res_blk > 0 + tail + demand Q)
    sdma_by_group: list[tuple[SDMAPageEntry, ...]] = []
    spans_by_group: list[np.ndarray] = []

    for g in range(num_resident_groups):
        sdma_list: list[SDMAPageEntry] = []
        span_rows: list[tuple[int, int, int, int, int, int, int, int]] = []

        for h_blk, (dst_pg, rep_req, rep_col) in sealed_map.items():
            res_blk = (
                int(rep_req.residency_by_group[g, rep_col]) if rep_col < rep_req.residency_by_group.shape[1] else 0
            )
            if res_blk > 0:
                span_rows.append((rep_req.batch_req_idx, rep_col, 0, BLOCK_SIZE, dst_pg * BLOCK_SIZE, 0, 0, 0))
            else:
                sdma_list.append(SDMAPageEntry(host_block_id=h_blk, dst_page=dst_pg))

        for tail_req, col, dst_pg, valid_rows in tail_allocs:
            span_rows.append((tail_req.batch_req_idx, col, 0, valid_rows, dst_pg * BLOCK_SIZE, 0, 0, 0))

        for req in demand_reqs:
            q_dst = q_dst_base_by_req[req.prefill_idx]
            pos = req.history_tokens
            end_pos = req.seq_len
            written = 0
            while pos < end_pos:
                pg_col, row_off = divmod(pos, BLOCK_SIZE)
                take = min(BLOCK_SIZE - row_off, end_pos - pos)
                span_rows.append((req.batch_req_idx, pg_col, row_off, take, q_dst + written, 0, 0, 0))
                pos += take
                written += take

        sdma_by_group.append(tuple(sdma_list))
        if span_rows:
            spans_by_group.append(np.asarray(span_rows, dtype=np.int32))
        else:
            spans_by_group.append(np.zeros((0, 8), dtype=np.int32))

    # Build remap_desc [len(sub_reqs), 8] and gather_meta [len(demand_reqs), 4]
    remap_desc = np.zeros((len(sub_reqs), 8), dtype=np.int32)
    for i, req in enumerate(sub_reqs):
        q_rel = req.q_batch_start - q_start
        if req.mode == MODE_FULL:
            remap_desc[i] = (
                MODE_FULL,
                req.history_tokens,
                req.query_tokens,
                req.q_batch_start,
                q_rel,
                full_slot_by_req[req.prefill_idx],
                0,
                req.num_pages,
            )
        else:
            remap_desc[i] = (
                MODE_DEMAND,
                req.history_tokens,
                req.query_tokens,
                req.q_batch_start,
                q_rel,
                req.demand_slot,
                u_dst_base_by_req[req.prefill_idx],
                q_dst_base_by_req[req.prefill_idx],
            )

    if demand_reqs:
        gather_meta = np.asarray(
            [
                (
                    req.batch_req_idx,
                    req.history_tokens,
                    u_dst_base_by_req[req.prefill_idx],
                    req.demand_slot,
                )
                for req in demand_reqs
            ],
            dtype=np.int32,
        )
    else:
        gather_meta = np.zeros((0, 4), dtype=np.int32)

    return WaveExecutionPlan(
        wave_idx=wave_idx,
        req_start=req_start,
        req_end=req_end,
        q_start=q_start,
        q_end=q_end,
        total_staging_rows=cursor_row,
        full_pages=full_pages,
        sdma_entries_by_group=tuple(sdma_by_group),
        d2d_spans_by_group=tuple(spans_by_group),
        remap_desc=remap_desc,
        staged_page_table=staged_page_table,
        gather_meta=gather_meta,
    )


def plan_whole_request_waves(
    requests: list[PrefillRequestSpec],
    num_resident_groups: int,
    staging_rows: int,
    union_capacities: dict[int, int],
) -> list[WaveExecutionPlan]:
    """Greedily pack contiguous whole requests into >= 1 waves within staging_rows."""
    if not requests:
        return []

    def slice_rows(start: int, end: int) -> int:
        sub = requests[start:end]
        full_r = _count_dedup_full_pages(sub) * BLOCK_SIZE
        dem_r = sum(r.query_tokens + int(union_capacities[r.demand_slot]) for r in sub if r.mode == MODE_DEMAND)
        return full_r + dem_r

    total = len(requests)
    if slice_rows(0, total) <= staging_rows:
        return [_build_single_wave(0, requests, 0, total, num_resident_groups, union_capacities, staging_rows)]

    waves: list[WaveExecutionPlan] = []
    start = 0
    while start < total:
        single_need = slice_rows(start, start + 1)
        if single_need > staging_rows:
            raise ValueError(
                f"Single request {requests[start].req_id!r} needs {single_need} rows > staging_rows={staging_rows}; "
                "check staging_pool >= 2 * max_model_len hard floor."
            )
        end = start + 1
        while end < total and slice_rows(start, end + 1) <= staging_rows:
            end += 1
        waves.append(
            _build_single_wave(len(waves), requests, start, end, num_resident_groups, union_capacities, staging_rows)
        )
        start = end
    return waves


def coalesce_sdma_descriptors(
    host_base_ptr: int,
    staging_base_ptr: int,
    entries: tuple[SDMAPageEntry, ...],
    page_bytes: int = BLOCK_SIZE * C8_ROW_BYTES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Coalesce contiguous (host_block_id, dst_page) runs into minimal H2D descriptors."""
    if not entries:
        empty = np.empty((0,), dtype=np.int64)
        return empty, empty, empty
    host_ids = np.fromiter((e.host_block_id for e in entries), dtype=np.int64, count=len(entries))
    dst_pages = np.fromiter((e.dst_page for e in entries), dtype=np.int64, count=len(entries))
    src = host_base_ptr + host_ids * page_bytes
    dst = staging_base_ptr + dst_pages * page_bytes
    if len(entries) == 1:
        return src, dst, np.array([page_bytes], dtype=np.int64)
    breaks = np.flatnonzero(((host_ids[1:] - host_ids[:-1]) != 1) | ((dst_pages[1:] - dst_pages[:-1]) != 1)) + 1
    starts = np.concatenate(([0], breaks))
    ends = np.concatenate((breaks, [len(entries)]))
    counts = ends - starts
    return src[starts], dst[starts], counts * page_bytes


class HiSparsePrefillWorkspace:
    """Persistent NPU buffers profiled once via WorkspaceManager."""

    def __init__(
        self,
        *,
        device: torch.device,
        kv_dtype: torch.dtype,
        staging_rows: int,
        max_num_seqs: int,
        max_batched_tokens: int,
        max_model_len: int,
    ) -> None:
        self.device = device
        self.kv_dtype = kv_dtype
        self.staging_rows = staging_rows
        self.staging_pages = staging_rows // BLOCK_SIZE
        self.staging_pool = torch.empty(
            (self.staging_pages, BLOCK_SIZE, C8_ROW_BYTES),
            dtype=kv_dtype,
            device=device,
        )
        self.part_bits = torch.empty((MAX_BITMAP_HISTORY,), dtype=torch.int32, device=device)
        self.status_words = torch.empty((64 * 8,), dtype=torch.int32, device=device)

        self.max_reqs = min(max(int(max_num_seqs), 1), MAX_KERNEL_REQS)
        self.max_tokens = min(max(int(max_batched_tokens), 1), MAX_KERNEL_TOKENS)
        capped_len = min(int(max_model_len), MAX_BITMAP_HISTORY)
        self.union_stride = min(MAX_BITMAP_HISTORY, max(BLOCK_SIZE, round_up(capped_len, BLOCK_SIZE)))

        self.union_ids = torch.empty((self.max_reqs, self.union_stride), dtype=torch.int32, device=device)
        self.union_status = torch.empty((self.max_reqs, 8), dtype=torch.int32, device=device)
        pin = device.type == "npu"
        self.union_status_cpu = torch.empty((self.max_reqs, 8), dtype=torch.int32, device="cpu", pin_memory=pin)
        self.bitmap_buf = torch.empty((self.max_reqs, MAX_BITMAP_WORDS), dtype=torch.int32, device=device)
        self.super_prefix_buf = torch.empty((self.max_reqs, MAX_SUPER_BLOCKS), dtype=torch.int32, device=device)
        self.mapped_topk = torch.empty((self.max_tokens, TOPK_WIDTH), dtype=torch.int32, device=device)
        self.dummy_page_table = torch.zeros((1, 1), dtype=torch.int32, device=device)

        if device.type == "npu":
            self.prefill_stream = torch.npu.Stream(device=device)
            self.sfa_done_event = torch.npu.Event()
            self.prefetch_ready_event = torch.npu.Event()
        else:
            self.prefill_stream = None
            self.sfa_done_event = None
            self.prefetch_ready_event = None


class HiSparseResidencyShadowTracker:
    """Zero-Core-invasion CPU tracker for per-request Host blocks and Resident pages."""

    def __init__(self, source_group_id: int, num_resident_groups: int) -> None:
        self.source_group_id = source_group_id
        self.num_resident_groups = max(int(num_resident_groups), 1)
        self.host_blocks_by_req: dict[str, list[int]] = {}
        self.residency_by_req: dict[str, np.ndarray] = {}
        self.computed_tokens_by_req: dict[str, int] = {}

    def clear(self) -> None:
        self.host_blocks_by_req.clear()
        self.residency_by_req.clear()
        self.computed_tokens_by_req.clear()

    def drop_requests(self, req_ids: Any) -> None:
        if not req_ids:
            return
        for req_id in req_ids:
            self.host_blocks_by_req.pop(req_id, None)
            self.residency_by_req.pop(req_id, None)
            self.computed_tokens_by_req.pop(req_id, None)

    def update_from_step(self, scheduler_output: Any, connector_meta: Any) -> None:
        if scheduler_output is not None:
            self.drop_requests(getattr(scheduler_output, "finished_req_ids", None))
            self.drop_requests(getattr(scheduler_output, "preempted_req_ids", None))

            for new_req in getattr(scheduler_output, "scheduled_new_reqs", ()) or ():
                req_id = new_req.req_id
                self.residency_by_req.pop(req_id, None)
                self.host_blocks_by_req[req_id] = list(new_req.block_ids[self.source_group_id])
                self.computed_tokens_by_req[req_id] = int(new_req.num_computed_tokens)

            cached = getattr(scheduler_output, "scheduled_cached_reqs", None)
            if cached is not None:
                resumed = getattr(cached, "resumed_req_ids", ()) or ()
                for idx, req_id in enumerate(cached.req_ids):
                    self.computed_tokens_by_req[req_id] = int(cached.num_computed_tokens[idx])
                    new_blocks = cached.new_block_ids[idx]
                    if req_id in resumed:
                        self.residency_by_req.pop(req_id, None)
                        self.host_blocks_by_req[req_id] = (
                            list(new_blocks[self.source_group_id]) if new_blocks is not None else []
                        )
                    elif new_blocks is not None:
                        cur = self.host_blocks_by_req.setdefault(req_id, [])
                        cur.extend(new_blocks[self.source_group_id])

        updates = getattr(connector_meta, "residency_updates", None)
        if updates:
            for req_id, upd in updates.items():
                pages = upd.pages
                if not pages:
                    continue
                needed = max(pages) + 1
                table = self.residency_by_req.get(req_id)
                if table is None:
                    table = np.zeros((self.num_resident_groups, needed), dtype=np.int32)
                    self.residency_by_req[req_id] = table
                elif table.shape[1] < needed:
                    grown = np.zeros((self.num_resident_groups, max(needed, table.shape[1] * 2)), dtype=np.int32)
                    grown[:, : table.shape[1]] = table
                    table = grown
                    self.residency_by_req[req_id] = table
                table[:, pages] = np.asarray(upd.block_ids, dtype=np.int32)


class HiSparsePrefillExecutor:
    """Coordinates pre-forward classification, SDMA/AIV prefetch, and wave attention."""

    def __init__(
        self,
        worker: Any,
        config: HiSparsePrefillConfig,
        workspace: HiSparsePrefillWorkspace,
    ) -> None:
        self.worker = worker
        self.config = config
        self.workspace = workspace

        target_indices = [i for i, h in enumerate(worker.cache_handles) if not h.draft_layer]
        self.target_handles = tuple(worker.cache_handles[i] for i in target_indices)
        self.target_host_caches = tuple(worker.host_caches[i] for i in target_indices)
        self.target_resident_caches = tuple(worker.resident_caches[i] for i in target_indices)
        self.layer_index_by_handle_id = {id(h): idx for idx, h in enumerate(self.target_handles)}
        leader_names = [
            layer_name
            for layer_name, handle in (getattr(worker, "_group_leaders", None) or ())
            if not getattr(handle, "draft_layer", False)
        ]
        if not leader_names:
            leader_names = list(getattr(worker, "cache_layer_names", None) or ())
        self.target_leader_names = tuple(leader_names)

        num_res_groups = max((int(h.runtime.resident_source_index) for h in self.target_handles), default=0) + 1
        (source_group_id,) = worker.kv_cache_config.host_group_ids
        self.tracker = HiSparseResidencyShadowTracker(source_group_id, num_res_groups)

        self.step_counter = 0
        self.active_step: PrefillStepPlan | None = None
        self.prefetched_layer_idx: int | None = None
        self.prefetched_sdma_pages: set[tuple[int, int]] = set()
        self.prefetched_has_demand: bool = False
        self._pending_sdma_descriptors: deque[tuple[Any, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = deque()
        self.step_summaries: list[dict[str, Any]] = []

    def _reap_completed_sdma_descriptors(self) -> None:
        while self._pending_sdma_descriptors:
            ev, _ = self._pending_sdma_descriptors[0]
            if ev is not None and not ev.query():
                break
            self._pending_sdma_descriptors.popleft()

    def on_finish_forward(self) -> None:
        if self.prefetched_layer_idx is not None and self.workspace.prefetch_ready_event is not None:
            torch.npu.current_stream().wait_event(self.workspace.prefetch_ready_event)
        self.active_step = None
        self.prefetched_layer_idx = None
        self.prefetched_sdma_pages.clear()
        self.prefetched_has_demand = False
        for handle in self.target_handles:
            handle._ascend_all_prefill_pages_resident = None
        self._reap_completed_sdma_descriptors()

    def reset(self) -> None:
        self.drain_stream()
        self.on_finish_forward()
        self.tracker.clear()

    def drain_stream(self) -> None:
        if self.workspace.prefill_stream is not None:
            self.workspace.prefill_stream.synchronize()
        for ev, _ in self._pending_sdma_descriptors:
            if ev is not None:
                ev.synchronize()
        self._pending_sdma_descriptors.clear()

    def _find_first_attn_metadata(self, attn_metadata_map: Any) -> Any | None:
        if not isinstance(attn_metadata_map, dict):
            return None
        for layer_name in self.target_leader_names:
            meta = attn_metadata_map.get(layer_name)
            if meta is not None:
                return meta
        return None

    def on_pre_forward(
        self,
        connector_meta: Any,
        scheduler_output: Any,
        request_ids: list[str] | None,
        attn_metadata_map: Any | None,
    ) -> None:
        self.on_finish_forward()
        self.tracker.update_from_step(scheduler_output, connector_meta)
        for handle in self.target_handles:
            handle._ascend_prefill_executor = self

        if (
            scheduler_output is None
            or not request_ids
            or not self.target_handles
            or self.worker.backend.is_current_stream_capturing()
            or (is_forward_context_available() and get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.FULL)
        ):
            return

        meta = self._find_first_attn_metadata(attn_metadata_map)
        if meta is None or getattr(meta, "hisparse_for_capture", False) or meta.num_prefills <= 0:
            return
        if meta.num_reqs != meta.num_decodes + meta.num_prefills or len(request_ids) < meta.num_reqs:
            return
        seq_lens_cpu = meta.seq_lens_cpu
        if seq_lens_cpu is None:
            return

        num_Scheduled = getattr(scheduler_output, "num_scheduled_tokens", None) or {}
        prefill_specs: list[PrefillRequestSpec] = []
        q_cursor = 0
        num_prefill_tokens = meta.num_actual_tokens - meta.num_decode_tokens

        for p_idx in range(meta.num_prefills):
            batch_idx = meta.num_decodes + p_idx
            req_id = request_ids[batch_idx]
            q_tok = int(num_Scheduled.get(req_id, 0))
            h_tok = int(self.tracker.computed_tokens_by_req.get(req_id, -1))
            expected_seq = int(seq_lens_cpu[batch_idx])
            if q_tok <= 0 or h_tok < 0 or h_tok + q_tok != expected_seq:
                return
            host_blocks = self.tracker.host_blocks_by_req.get(req_id)
            needed_pages = cdiv(expected_seq, BLOCK_SIZE)
            if host_blocks is None or len(host_blocks) < needed_pages:
                return
            res_table = self.tracker.residency_by_req.get(req_id)
            if res_table is None or res_table.shape[1] < needed_pages:
                return

            sealed_pages = h_tok // BLOCK_SIZE
            if sealed_pages > 0:
                miss_mask = np.any(res_table[:, :sealed_pages] <= 0, axis=0)
                miss_pages = int(np.count_nonzero(miss_mask))
            else:
                miss_pages = 0
            # Verify all tail/active pages [sealed_pages, needed_pages) are resident
            if np.any(res_table[:, sealed_pages:needed_pages] <= 0):
                return

            prefill_specs.append(
                PrefillRequestSpec(
                    prefill_idx=p_idx,
                    batch_req_idx=batch_idx,
                    req_id=req_id,
                    history_tokens=h_tok,
                    query_tokens=q_tok,
                    q_batch_start=q_cursor,
                    host_blocks=tuple(host_blocks[:needed_pages]),
                    residency_by_group=res_table[:, :needed_pages],
                    host_miss_pages=miss_pages,
                    host_history_tokens=miss_pages * BLOCK_SIZE,
                )
            )
            q_cursor += q_tok

        if q_cursor != num_prefill_tokens or q_cursor > self.workspace.max_tokens:
            return
        if len(prefill_specs) > self.workspace.max_reqs:
            return

        all_resident, worst_rows, needs_union_check = classify_prefill_requests(
            prefill_specs,
            self.config,
            topk=TOPK_WIDTH,
        )

        for handle in self.target_handles:
            handle._ascend_all_prefill_pages_resident = all_resident

        if all_resident:
            self._record_summary(prefill_specs, num_waves=0, all_resident=True, union_counts=())
            return

        demand_reqs = [r for r in prefill_specs if r.mode == MODE_DEMAND]
        if any(r.history_tokens > self.workspace.union_stride for r in demand_reqs) or any(
            r.num_pages > MAX_FULL_PAGE_COLUMNS for r in prefill_specs if r.mode == MODE_FULL
        ):
            # Fall back safely if demand union stride or full-page table exceeds kernel UB bounds.
            for handle in self.target_handles:
                handle._ascend_all_prefill_pages_resident = False
            return

        worst_caps = {r.demand_slot: min(r.history_tokens, r.query_tokens * TOPK_WIDTH) for r in demand_reqs}
        conservative_waves = plan_whole_request_waves(
            prefill_specs,
            self.tracker.num_resident_groups,
            self.workspace.staging_rows,
            worst_caps,
        )
        waves = conservative_waves if not needs_union_check else []

        self.step_counter += 1
        step = PrefillStepPlan(
            step_id=self.step_counter,
            all_prefill_resident=False,
            requests=prefill_specs,
            demand_requests=demand_reqs,
            staging_rows=self.workspace.staging_rows,
            num_resident_groups=self.tracker.num_resident_groups,
            worst_total_rows=worst_rows,
            needs_union_count_check=needs_union_check,
            waves=waves,
            conservative_wave0=conservative_waves[0] if conservative_waves else None,
        )
        self.active_step = step

        # Start Layer-0 Wave-0 Full-Load SDMA immediately before model.forward()!
        self._launch_layer0_early_sdma(step)

    def _submit_sdma_entries(self, layer_idx: int, entries: tuple[SDMAPageEntry, ...]) -> bool:
        if not entries:
            return False
        host_cache = self.target_host_caches[layer_idx]
        src_np, dst_np, sizes_np = coalesce_sdma_descriptors(
            host_cache.data_ptr(),
            self.workspace.staging_pool.data_ptr(),
            entries,
        )
        count = int(src_np.shape[0])
        if count == 0:
            return False
        src_t = torch.from_numpy(src_np)
        dst_t = torch.from_numpy(dst_np)
        sizes_t = torch.from_numpy(sizes_np)
        shared_region = getattr(self.worker, "shared_host_region", None)
        if shared_region is not None and not getattr(self.worker, "is_host_writer", True):
            mapping = shared_region.base_tensor._hisparse_registration
            with self.worker.backend.host_to_device_dma(mapping):
                self.worker.backend.swap_blocks_batch(src_t, dst_t, sizes_t)
        else:
            submit = torch.ops._C_ascend.swap_blocks_batch
            small_async = count <= 16
            if count <= SDMA_BATCH_LIMIT:
                submit(src_t, dst_t, sizes_t, 0, prefer_small_async=small_async)
            else:
                for s in range(0, count, SDMA_BATCH_LIMIT):
                    e = min(s + SDMA_BATCH_LIMIT, count)
                    submit(src_t[s:e], dst_t[s:e], sizes_t[s:e], 0, prefer_small_async=False)
        ev = None
        if self.workspace.device.type == "npu":
            ev = torch.npu.Event()
            ev.record(torch.npu.current_stream())
        self._pending_sdma_descriptors.append((ev, (src_t, dst_t, sizes_t)))
        return True

    def _launch_layer0_early_sdma(self, step: PrefillStepPlan) -> None:
        if self.workspace.prefill_stream is None or step.conservative_wave0 is None:
            return
        g0 = int(self.target_handles[0].runtime.resident_source_index)
        entries = step.conservative_wave0.sdma_entries_by_group[g0]
        if not entries:
            return

        compute_stream = torch.npu.current_stream()
        p_stream = self.workspace.prefill_stream
        p_stream.wait_stream(compute_stream)
        with torch.npu.stream(p_stream):
            if self._submit_sdma_entries(0, entries):
                assert self.workspace.prefetch_ready_event is not None
                self.workspace.prefetch_ready_event.record(p_stream)
                self.prefetched_layer_idx = 0
                self.prefetched_sdma_pages = {(e.host_block_id, e.dst_page) for e in entries}
                self.prefetched_has_demand = False

    def _ensure_wave_gpu_metadata(self, wave: WaveExecutionPlan) -> None:
        dev = self.workspace.device
        if wave.remap_desc_gpu is None:
            wave.remap_desc_gpu = torch.from_numpy(wave.remap_desc).to(device=dev, non_blocking=True)
        if wave.staged_page_table_gpu is None:
            wave.staged_page_table_gpu = torch.from_numpy(wave.staged_page_table).to(device=dev, non_blocking=True)
        if wave.gather_meta_gpu is None and wave.gather_meta.shape[0] > 0:
            wave.gather_meta_gpu = torch.from_numpy(wave.gather_meta).to(device=dev, non_blocking=True)
        if not wave.d2d_spans_gpu_by_group:
            for arr in wave.d2d_spans_by_group:
                if arr.shape[0] > 0:
                    gpu_arr = torch.from_numpy(arr).to(device=dev, non_blocking=True)
                    wave.d2d_spans_gpu_by_group.append((gpu_arr, int(arr.shape[0])))
                else:
                    wave.d2d_spans_gpu_by_group.append((self.workspace.dummy_page_table, 0))

    def _launch_gather_union(self, layer_idx: int, wave: WaveExecutionPlan) -> None:
        if wave.gather_meta_gpu is None or wave.gather_meta.shape[0] == 0:
            return
        handle = self.target_handles[layer_idx]
        res_cache = self.target_resident_caches[layer_idx]
        host_cache = self.target_host_caches[layer_idx]
        mapping = handle.runtime.cache_ops.host_mapping
        dva_base = _to_signed_int64(int(mapping.address_of(host_cache)))
        host_bytes = int(host_cache.numel() * host_cache.element_size())
        assert handle.source_block_table is not None and handle.block_table is not None
        assert handle.runtime.request_state_indices is not None
        torch.ops._C_ascend.hisparse_prefill_gather_union(
            self.workspace.union_ids,
            self.workspace.union_status,
            wave.gather_meta_gpu,
            handle.runtime.request_state_indices,
            handle.source_block_table,
            handle.block_table,
            res_cache,
            self.workspace.staging_pool,
            dva_base,
            host_bytes,
        )

    def _launch_d2d_spans(self, layer_idx: int, wave: WaveExecutionPlan) -> None:
        handle = self.target_handles[layer_idx]
        g = int(handle.runtime.resident_source_index)
        spans_gpu, count = wave.d2d_spans_gpu_by_group[g]
        if count <= 0:
            return
        res_cache = self.target_resident_caches[layer_idx]
        assert handle.block_table is not None and handle.runtime.request_state_indices is not None
        torch.ops._C_ascend.hisparse_prefill_copy_spans(
            spans_gpu,
            count,
            handle.runtime.request_state_indices,
            handle.block_table,
            res_cache,
            self.workspace.staging_pool,
        )

    def _prepare_group_at_leader(
        self,
        step: PrefillStepPlan,
        group: Any,
        logical_prefill: torch.Tensor,
    ) -> list[WaveExecutionPlan]:
        ws = self.workspace
        logical_contig = logical_prefill.contiguous()

        if step.demand_requests:
            histories = [r.history_tokens for r in step.demand_requests]
            starts = [r.q_batch_start for r in step.demand_requests]
            queries = [r.query_tokens for r in step.demand_requests]
            torch.ops._C_ascend.hisparse_prefill_union_plan(
                logical_contig,
                [ws.part_bits, ws.status_words],
                [ws.union_ids, ws.union_status, ws.bitmap_buf, ws.super_prefix_buf],
                histories,
                starts,
                queries,
                ws.union_stride,
            )

        if step.needs_union_count_check:
            m = len(step.demand_requests)
            ws.union_status_cpu[:m].copy_(ws.union_status[:m], non_blocking=False)
            status_np = ws.union_status_cpu[:m].numpy()
            if np.any(status_np[:, 3] != 0) or np.any(status_np[:, 4] != 0):
                raise RuntimeError(
                    f"HiSparse prefill union reported invalid/overflow status: {status_np[:, :5].tolist()}"
                )
            actual_caps = {slot: int(status_np[slot, 0]) for slot in range(m)}
            step.last_union_counts = tuple(int(status_np[slot, 0]) for slot in range(m))
            waves = plan_whole_request_waves(
                step.requests,
                step.num_resident_groups,
                step.staging_rows,
                actual_caps,
            )
        else:
            waves = step.waves

        for wave in waves:
            self._ensure_wave_gpu_metadata(wave)
            q_len = wave.q_end - wave.q_start
            mapped_slice = ws.mapped_topk[wave.q_start : wave.q_end]
            assert mapped_slice.shape[0] == q_len
            assert wave.remap_desc_gpu is not None and wave.staged_page_table_gpu is not None
            torch.ops._C_ascend.hisparse_prefill_remap_topk(
                logical_contig,
                wave.remap_desc_gpu,
                ws.bitmap_buf,
                ws.super_prefix_buf,
                wave.staged_page_table_gpu,
                mapped_slice,
                0,
                ws.staging_rows,
            )

        step.active_group_id = id(group)
        step.group_waves = waves
        return waves

    def _launch_post_sfa_next_layer_prefetch(
        self,
        step: PrefillStepPlan,
        cur_layer_idx: int,
        is_next_same_group_follower: bool,
        waves: list[WaveExecutionPlan],
    ) -> None:
        next_layer_idx = cur_layer_idx + 1
        if next_layer_idx >= len(self.target_handles) or not waves or self.workspace.prefill_stream is None:
            self.prefetched_layer_idx = None
            self.prefetched_sdma_pages.clear()
            self.prefetched_has_demand = False
            return

        next_handle = self.target_handles[next_layer_idx]
        next_g = int(next_handle.runtime.resident_source_index)
        if is_next_same_group_follower or not step.needs_union_count_check:
            sdma_wave = waves[0]
        else:
            # Across an index-group boundary under worst-case overflow guard, prefetch only
            # the guaranteed Wave-0 prefix so Zone A never overlaps next group's Zone B/C.
            sdma_wave = step.conservative_wave0
        sdma_entries = sdma_wave.sdma_entries_by_group[next_g] if sdma_wave is not None else ()
        wave0 = waves[0]
        do_demand = is_next_same_group_follower and wave0.gather_meta.shape[0] > 0

        if not sdma_entries and not do_demand:
            self.prefetched_layer_idx = None
            self.prefetched_sdma_pages.clear()
            self.prefetched_has_demand = False
            return

        compute_stream = torch.npu.current_stream()
        p_stream = self.workspace.prefill_stream
        sfa_done = self.workspace.sfa_done_event
        ready = self.workspace.prefetch_ready_event
        assert sfa_done is not None and ready is not None

        sfa_done.record(compute_stream)
        p_stream.wait_event(sfa_done)
        with torch.npu.stream(p_stream):
            did_sdma = self._submit_sdma_entries(next_layer_idx, sdma_entries) if sdma_entries else False
            if do_demand:
                self._ensure_wave_gpu_metadata(wave0)
                self._launch_gather_union(next_layer_idx, wave0)
            ready.record(p_stream)

        self.prefetched_layer_idx = next_layer_idx
        self.prefetched_sdma_pages = {(e.host_block_id, e.dst_page) for e in sdma_entries} if did_sdma else set()
        self.prefetched_has_demand = do_demand

    def execute_layer_prefill(
        self,
        impl: Any,
        query: torch.Tensor,
        query_rope: torch.Tensor | None,
        logical_prefill: torch.Tensor,
        metadata: Any,
    ) -> torch.Tensor | None:
        step = self.active_step
        if (
            step is None
            or not impl.enable_sparse_sfa_c8
            or query_rope is None
            or metadata.hisparse_for_capture
            or get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.FULL
            or torch.npu.is_current_stream_capturing()
            or logical_prefill.ndim != 2
            or logical_prefill.shape[1] != TOPK_WIDTH
        ):
            return None

        layer_idx = self.layer_index_by_handle_id.get(id(impl.hisparse_cache))
        if layer_idx is None:
            return None

        num_prefill_tokens = metadata.num_actual_tokens - metadata.num_decode_tokens
        if query.shape[0] < num_prefill_tokens or logical_prefill.shape[0] < num_prefill_tokens:
            return None
        q_actual = query[:num_prefill_tokens]
        q_rope_actual = query_rope[:num_prefill_tokens]
        logical_actual = logical_prefill[:num_prefill_tokens]

        if impl.index_group_index == 0 or step.active_group_id != id(impl.index_group):
            waves = self._prepare_group_at_leader(step, impl.index_group, logical_actual)
        else:
            assert step.group_waves is not None
            waves = step.group_waves

        compute_stream = torch.npu.current_stream()
        cur_g = int(self.target_handles[layer_idx].runtime.resident_source_index)
        wave_outputs: list[torch.Tensor] = []

        for w_idx, wave in enumerate(waves):
            self._ensure_wave_gpu_metadata(wave)
            sdma_entries = wave.sdma_entries_by_group[cur_g]
            if w_idx == 0 and self.prefetched_layer_idx == layer_idx:
                remaining = tuple(
                    e for e in sdma_entries if (e.host_block_id, e.dst_page) not in self.prefetched_sdma_pages
                )
                if remaining:
                    self._submit_sdma_entries(layer_idx, remaining)
                if wave.gather_meta.shape[0] > 0 and not self.prefetched_has_demand:
                    self._launch_gather_union(layer_idx, wave)
                self._launch_d2d_spans(layer_idx, wave)
                assert self.workspace.prefetch_ready_event is not None
                compute_stream.wait_event(self.workspace.prefetch_ready_event)
            else:
                if sdma_entries:
                    self._submit_sdma_entries(layer_idx, sdma_entries)
                if wave.gather_meta.shape[0] > 0:
                    self._launch_gather_union(layer_idx, wave)
                self._launch_d2d_spans(layer_idx, wave)

            active_pages = max(1, cdiv(max(1, wave.total_staging_rows), BLOCK_SIZE))
            staged_slice = self.workspace.staging_pool[:active_pages]
            mapped_slice = self.workspace.mapped_topk[wave.q_start : wave.q_end]
            wave_out = impl._attention_physical(
                q_actual[wave.q_start : wave.q_end],
                q_rope_actual[wave.q_start : wave.q_end],
                staged_slice,
                mapped_slice,
                indices_are_mapped=True,
            )
            wave_outputs.append(wave_out)

        is_next_follower = (impl.index_group_index + 1) < impl.index_group.num_layers
        self._launch_post_sfa_next_layer_prefetch(step, layer_idx, is_next_follower, waves)

        if layer_idx == len(self.target_handles) - 1:
            self._record_summary(
                step.requests,
                num_waves=len(waves),
                all_resident=False,
                union_counts=step.last_union_counts,
            )
            self.active_step = None

        return torch.cat(wave_outputs, dim=0) if len(wave_outputs) > 1 else wave_outputs[0]

    def _record_summary(
        self,
        requests: list[PrefillRequestSpec],
        *,
        num_waves: int,
        all_resident: bool,
        union_counts: tuple[int, ...],
    ) -> None:
        summary = {
            "step_id": self.step_counter,
            "all_prefill_resident": all_resident,
            "num_requests": len(requests),
            "num_waves": num_waves,
            "tier_counts": {
                "tier0_on_card": sum(1 for r in requests if r.tier == TIER_ON_CARD),
                "tier1_hard_demand": sum(1 for r in requests if r.tier == TIER_HARD_DEMAND),
                "tier2_elastic_full": sum(1 for r in requests if r.tier == TIER_ELASTIC and r.mode == MODE_FULL),
                "tier2_elastic_downgraded": sum(1 for r in requests if r.downgraded_from_tier2),
                "tier3_full_load": sum(1 for r in requests if r.tier == TIER_FULL_LOAD),
            },
            "modes": [
                {
                    "prefill_idx": r.prefill_idx,
                    "history_tokens": r.history_tokens,
                    "host_history_tokens": r.host_history_tokens,
                    "query_tokens": r.query_tokens,
                    "tier": r.tier,
                    "mode": "demand" if r.mode == MODE_DEMAND else "full",
                    "downgraded_from_tier2": r.downgraded_from_tier2,
                }
                for r in requests
            ],
            "union_counts": list(union_counts),
        }
        if len(self.step_summaries) >= 256:
            self.step_summaries.pop(0)
        self.step_summaries.append(summary)


def check_prefill_kernels_available() -> None:
    ns = torch.ops._C_ascend
    required = (
        "hisparse_prefill_union_plan",
        "hisparse_prefill_remap_topk",
        "hisparse_prefill_gather_union",
        "hisparse_prefill_copy_spans",
    )
    compiled = not hasattr(ns, "hisparse_prefill_kernels_compiled") or bool(ns.hisparse_prefill_kernels_compiled())
    if not compiled or not all(hasattr(ns, name) for name in required):
        raise RuntimeError(
            "hisparse_prefill_config.enabled requires compiled HiSparse prefill AscendC kernels in vllm_ascend_C."
        )


@torch.inference_mode(False)
def prepare_hisparse_prefill_workspace(forward_context: dict[str, Any]) -> None:
    """Allocate the persistent staging_pool (>= 2 * max(max_model_len, 1M) * 656B) before memory profiling."""
    from vllm.config import get_current_vllm_config_or_none
    from vllm.v1.worker.workspace import current_workspace_manager

    from vllm_ascend import ascend_config

    vllm_config = get_current_vllm_config_or_none() or ascend_config._INIT_VLLM_CONFIG
    if vllm_config is None:
        return
    cfg = HiSparsePrefillConfig.from_vllm_config(vllm_config)
    if not cfg.enabled:
        return

    target_runtime = None
    for layer in forward_context.values():
        cache = getattr(layer, "hisparse_cache", None)
        if cache is not None and not getattr(cache, "draft_layer", False):
            target_runtime = cache.runtime
            break
    if target_runtime is None or target_runtime.device.type != "npu":
        return

    check_prefill_kernels_available()
    sched = vllm_config.scheduler_config
    model_cfg = vllm_config.model_config
    key = (
        "ascend_hisparse_prefill",
        cfg.staging_rows,
        target_runtime.kv_dtype,
        sched.max_num_seqs,
        sched.max_num_batched_tokens,
        model_cfg.max_model_len,
    )
    workspace = current_workspace_manager().get_persistent_resource(
        key,
        lambda: HiSparsePrefillWorkspace(
            device=target_runtime.device,
            kv_dtype=target_runtime.kv_dtype,
            staging_rows=cfg.staging_rows,
            max_num_seqs=sched.max_num_seqs,
            max_batched_tokens=sched.max_num_batched_tokens,
            max_model_len=model_cfg.max_model_len,
        ),
    )
    for layer in forward_context.values():
        cache = getattr(layer, "hisparse_cache", None)
        if cache is not None and not getattr(cache, "draft_layer", False):
            cache.runtime._ascend_prefill_workspace = (cfg, workspace)
    logger.info_once(
        "[hisparse] Persistent prefill staging_pool ready: rows=%d (%.1f MiB), tier1<=%.2f%%, tier2<=%.2f%%, min_H=%d",
        cfg.staging_rows,
        cfg.staging_rows * C8_ROW_BYTES / (1024**2),
        cfg.tier1_max_ratio * 100.0,
        cfg.tier2_max_ratio * 100.0,
        cfg.min_history_tokens,
        scope="process",
    )


def bind_worker_prefill_executor(worker: Any) -> None:
    vllm_config = getattr(worker, "vllm_config", None)
    if vllm_config is None:
        return
    cfg = HiSparsePrefillConfig.from_vllm_config(vllm_config)
    if not cfg.enabled:
        worker._ascend_prefill_executor = None
        return
    target = next((h for h in worker.cache_handles if not h.draft_layer), None)
    if target is None or target.runtime.device.type != "npu":
        worker._ascend_prefill_executor = None
        return
    check_prefill_kernels_available()
    prepared = getattr(target.runtime, "_ascend_prefill_workspace", None)
    if prepared is not None:
        _, workspace = prepared
    else:
        sched = vllm_config.scheduler_config
        workspace = HiSparsePrefillWorkspace(
            device=target.runtime.device,
            kv_dtype=target.runtime.kv_dtype,
            staging_rows=cfg.staging_rows,
            max_num_seqs=sched.max_num_seqs,
            max_batched_tokens=sched.max_num_batched_tokens,
            max_model_len=vllm_config.model_config.max_model_len,
        )
    worker._ascend_prefill_executor = HiSparsePrefillExecutor(worker, cfg, workspace)


def install_hisparse_prefill_hooks() -> None:
    """Install lightweight pre-forward / post-forward / teardown hooks on HiSparseConnector & Worker."""
    from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.connector import HiSparseConnector
    from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.worker import HiSparseConnectorWorker

    orig_start_load = HiSparseConnector.start_load_kv
    if getattr(orig_start_load, "_ascend_prefill_hook", False):
        return

    orig_init_dma = HiSparseConnectorWorker._init_dma

    @wraps(orig_init_dma)
    def init_dma(worker):
        orig_init_dma(worker)
        if getattr(worker, "_ascend_prefill_executor", None) is None:
            bind_worker_prefill_executor(worker)

    HiSparseConnectorWorker._init_dma = init_dma

    @wraps(orig_start_load)
    def start_load_kv(connector, forward_context, **kwargs):
        orig_start_load(connector, forward_context, **kwargs)
        worker = connector.connector_worker
        executor: HiSparsePrefillExecutor | None = getattr(worker, "_ascend_prefill_executor", None)
        if executor is None:
            return
        attn_meta = kwargs.get("attn_metadata")
        if attn_meta is None and forward_context is not None:
            attn_meta = getattr(forward_context, "attn_metadata", None)
        executor.on_pre_forward(
            connector._get_connector_metadata(),
            kwargs.get("scheduler_output"),
            kwargs.get("request_ids"),
            attn_meta,
        )

    start_load_kv.__dict__["_ascend_prefill_hook"] = True
    HiSparseConnector.start_load_kv = start_load_kv

    orig_get_results = HiSparseConnector.get_transfer_results

    @wraps(orig_get_results)
    def get_transfer_results(connector, finished_req_ids):
        worker = getattr(connector, "connector_worker", None)
        executor: HiSparsePrefillExecutor | None = getattr(worker, "_ascend_prefill_executor", None)
        if executor is not None and finished_req_ids:
            executor.tracker.drop_requests(finished_req_ids)
        return orig_get_results(connector, finished_req_ids)

    HiSparseConnector.get_transfer_results = get_transfer_results

    orig_finish = HiSparseConnectorWorker.finish_forward

    @wraps(orig_finish)
    def finish_forward(worker):
        executor: HiSparsePrefillExecutor | None = getattr(worker, "_ascend_prefill_executor", None)
        if executor is not None:
            executor.on_finish_forward()
        return orig_finish(worker)

    HiSparseConnectorWorker.finish_forward = finish_forward

    orig_reset = HiSparseConnectorWorker.reset_hot_state

    @wraps(orig_reset)
    def reset_hot_state(worker):
        executor: HiSparsePrefillExecutor | None = getattr(worker, "_ascend_prefill_executor", None)
        if executor is not None:
            executor.drain_stream()
            executor.reset()
        return orig_reset(worker)

    HiSparseConnectorWorker.reset_hot_state = reset_hot_state

    orig_shutdown = HiSparseConnectorWorker.shutdown

    @wraps(orig_shutdown)
    def shutdown(worker):
        executor: HiSparsePrefillExecutor | None = getattr(worker, "_ascend_prefill_executor", None)
        if executor is not None:
            executor.drain_stream()
            executor.reset()
        return orig_shutdown(worker)

    HiSparseConnectorWorker.shutdown = shutdown

    try:
        from vllm.v1.worker.gpu.kv_connector import ActiveKVConnector
    except Exception:
        return

    def _iter_active_executors(active_conn: Any):
        root = getattr(active_conn, "kv_connector", None) or getattr(active_conn, "connector", None)
        if root is None:
            return
        stack = [root]
        seen: set[int] = set()
        while stack:
            cur = stack.pop()
            if cur is None or id(cur) in seen:
                continue
            seen.add(id(cur))
            worker = getattr(cur, "connector_worker", None)
            executor: HiSparsePrefillExecutor | None = getattr(worker, "_ascend_prefill_executor", None)
            if executor is not None:
                yield executor
            for attr in ("_connector", "connector", "kv_connector"):
                inner = getattr(cur, attr, None)
                if inner is not None:
                    stack.append(inner)
            for list_attr in ("_connectors", "connectors"):
                children = getattr(cur, list_attr, None)
                if isinstance(children, (list, tuple)):
                    stack.extend(children)

    orig_no_forward = ActiveKVConnector.no_forward

    @wraps(orig_no_forward)
    def no_forward(active_conn, scheduler_output):
        if scheduler_output is not None:
            for executor in _iter_active_executors(active_conn):
                executor.tracker.drop_requests(getattr(scheduler_output, "finished_req_ids", None))
                executor.tracker.drop_requests(getattr(scheduler_output, "preempted_req_ids", None))
                executor.on_finish_forward()
        return orig_no_forward(active_conn, scheduler_output)

    ActiveKVConnector.no_forward = no_forward

    orig_set_disabled = ActiveKVConnector.set_disabled

    @wraps(orig_set_disabled)
    def set_disabled(active_conn, disabled: bool) -> None:
        if disabled:
            for executor in _iter_active_executors(active_conn):
                executor.on_finish_forward()
        return orig_set_disabled(active_conn, disabled)

    ActiveKVConnector.set_disabled = set_disabled


def all_prefill_pages_resident(cache: Any) -> bool:
    override = getattr(cache, "_ascend_all_prefill_pages_resident", None)
    if override is not None:
        return bool(override)
    return bool(cache.all_context_pages_resident)


def try_execute_hisparse_prefill(
    impl: Any,
    query: torch.Tensor,
    query_rope: torch.Tensor | None,
    logical_prefill: torch.Tensor,
    metadata: Any,
) -> torch.Tensor | None:
    executor: HiSparsePrefillExecutor | None = getattr(impl.hisparse_cache, "_ascend_prefill_executor", None)
    if executor is None:
        return None
    return executor.execute_layer_prefill(impl, query, query_rope, logical_prefill, metadata)
