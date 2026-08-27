# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import bisect
import json
import math
import os
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from vllm.distributed import get_pp_group, get_tp_group
from vllm.logger import init_logger
from .verify_adaptive_config import (
    ENV_COST_TABLE_OVERRIDE,
    VerifyAdaptiveConfig,
)

logger = init_logger(__name__)


def _parse_cost_table_override(
    payload: Any,
    *,
    num_spec_tokens: int,
    max_batch_size: int,
    batch_size_levels: list[int],
    query_len_levels: list[int],
) -> tuple[
    dict[tuple[int, int], float],
    dict[tuple[int, int], float],
]:
    """Validate and decode an exported schema-v2 cost table."""
    if not isinstance(payload, dict):
        raise ValueError("cost table override must be a JSON object")
    if payload.get("schema_version") != 2:
        raise ValueError("cost table override schema_version must be 2")
    if payload.get("num_spec_tokens") != num_spec_tokens:
        raise ValueError(
            "cost table override num_spec_tokens mismatch: "
            f"file={payload.get('num_spec_tokens')} runtime={num_spec_tokens}"
        )
    if payload.get("max_batch_size") != max_batch_size:
        raise ValueError(
            "cost table override max_batch_size mismatch: "
            f"file={payload.get('max_batch_size')} runtime={max_batch_size}"
        )
    if payload.get("batch_size_levels") != batch_size_levels:
        raise ValueError(
            "cost table override batch_size_levels do not match runtime config"
        )
    if payload.get("query_len_levels") != query_len_levels:
        raise ValueError(
            "cost table override query_len_levels do not match runtime config"
        )

    rows = payload.get("cost_table")
    if not isinstance(rows, list) or not rows:
        raise ValueError("cost table override must contain at least one row")

    valid_bs = set(batch_size_levels)
    valid_ql = set(query_len_levels)
    target_table: dict[tuple[int, int], float] = {}
    draft_table: dict[tuple[int, int], float] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"cost table override row {index} must be an object")
        bs = row.get("batch_size")
        sum_query_len = row.get("sum_query_len")
        if (
            not isinstance(bs, int)
            or isinstance(bs, bool)
            or not isinstance(sum_query_len, int)
            or isinstance(sum_query_len, bool)
            or bs not in valid_bs
            or sum_query_len <= 0
            or sum_query_len % bs != 0
            or sum_query_len // bs not in valid_ql
        ):
            raise ValueError(
                f"cost table override row {index} has an invalid batch/query shape"
            )

        query_len_per_req = row.get("query_len_per_req")
        if (
            query_len_per_req is not None
            and query_len_per_req != sum_query_len // bs
        ):
            raise ValueError(
                f"cost table override row {index} has inconsistent "
                "query_len_per_req"
            )
        try:
            target_cost = float(row["target_cost_s"])
            draft_cost = float(row["draft_cost_s"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"cost table override row {index} has invalid component costs"
            ) from exc
        if (
            not math.isfinite(target_cost)
            or target_cost <= 0.0
            or not math.isfinite(draft_cost)
            or draft_cost < 0.0
        ):
            raise ValueError(
                f"cost table override row {index} has invalid component costs"
            )

        total_cost = row.get("cost_s")
        if total_cost is not None:
            try:
                total_cost = float(total_cost)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"cost table override row {index} has an invalid total cost"
                ) from exc
            if not math.isclose(
                total_cost,
                target_cost + draft_cost,
                rel_tol=1e-6,
                abs_tol=1e-9,
            ):
                raise ValueError(
                    f"cost table override row {index} total does not match "
                    "its components"
                )

        key = (bs, sum_query_len)
        if key in target_table:
            raise ValueError(f"cost table override contains duplicate shape {key}")
        target_table[key] = target_cost
        draft_table[key] = draft_cost

    return target_table, draft_table


# ---------------------------------------------------------------------------
# Core algorithm — pure function, stateless, unit-testable independently.
# ---------------------------------------------------------------------------

def choose_query_lens_discrete(
    probs: "list[list[float]] | np.ndarray",
    base_batch_size: int,
    q_levels: list[int],
    cost_lookup: Callable[[int], float],
    max_draft_len: int,
    collect_records: bool = False,
    draft_cost_lookup: Callable[[int], float] | None = None,
) -> dict[str, Any]:
    """Discrete marginal-gain scan over the *measured* sum_query_len levels.

    Since verifier cost depends only on ``(batch_size, sum_query_len)``, the
    candidate Q values are exactly the profiled sum_query_len levels for the
    fixed batch size (e.g. ``bs*2, bs*4, …``).  For each level Q we greedily
    fill the ``S = Q - base_batch_size`` highest marginal gains and divide the
    expected progress by ``target_cost(Q) + draft_cost(Q)``, keeping the best
    end-to-end Q.

    Args:
        probs: per-active-sequence accept probs; ``probs[i][t]`` is the
            predicted accept prob of draft position ``t`` for sequence ``i``.
        base_batch_size: full verifier batch size B.  Every sequence always
            contributes one anchor token, so ``sum_query_len = B + S``.
        q_levels: candidate sum_query_len values; must be real cost-table keys.
        cost_lookup: ``Q -> verifier ITL cost`` (batch size already fixed).
        max_draft_len: max draft tokens per sequence (``max_query_len - 1``).
        collect_records: if True, also return per-level debug records.
    """
    A = len(probs)

    # Marginal gains m[i,t] = prod_{k<=t} p[i,k], vectorised over the batch.
    mat = np.asarray(probs, dtype=np.float64).reshape(A, -1)[:, :max_draft_len]
    gains = np.cumprod(mat, axis=1)

    seq_ids = np.repeat(np.arange(A), gains.shape[1])
    flat_gains = gains.ravel()
    order = np.argsort(-flat_gains, kind="stable")
    sorted_seq = seq_ids[order]
    # prefix_gain[S] = sum of the top-S marginal gains.
    prefix_gain = np.concatenate(([0.0], np.cumsum(flat_gains[order])))
    total_available = flat_gains.shape[0]

    best_score = -math.inf
    best_Q, best_S = base_batch_size, 0
    records: list[dict[str, Any]] | None = [] if collect_records else None

    for Q in q_levels:
        S = Q - base_batch_size
        if S < 0:
            continue
        S = min(S, total_available)
        target_cost = cost_lookup(Q)
        draft_cost = (
            draft_cost_lookup(Q)
            if draft_cost_lookup is not None
            else 0.0
        )
        cost = target_cost + draft_cost
        if target_cost <= 0.0 or draft_cost < 0.0 or cost <= 0.0:
            continue
        score = (base_batch_size + prefix_gain[S]) / cost
        if records is not None:
            records.append({
                "Q": Q,
                "S": int(S),
                "score": score,
                "cost": cost,
                "target_cost": target_cost,
                "draft_cost": draft_cost,
            })
        if score > best_score:
            best_score, best_Q, best_S = score, Q, S

    # Reconstruct per-sequence draft lengths from the top-best_S marginals.
    draft_lens = np.bincount(sorted_seq[:best_S], minlength=A).tolist()

    return {
        "draft_lens": draft_lens,
        "best_Q": best_Q,
        "best_S": int(best_S),
        "best_score": best_score,
        "records": records,
    }


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class VerifyAdaptiveController:
    """Per-request draft-length selector for the verifier step.

    Call order: ``__init__`` → ``profile_cost_table`` (once, after CUDA
    graph capture / JIT warmup) → ``process_draft_output`` (each step) →
    ``get_adaptive_draft_len`` (inside ``_prepare_inputs``).
    Call ``invalidate`` on request completion.
    """

    def __init__(
        self,
        config: VerifyAdaptiveConfig,
        num_spec_tokens: int,
        max_batch_size: int,
        device: torch.device,
    ) -> None:
        config.validate(num_spec_tokens)

        self.config = config
        self.num_spec_tokens = num_spec_tokens
        self.max_batch_size = max_batch_size
        self.device = device
        self.max_query_len_per_req: int = (
            config.max_query_len_per_req
            if config.max_query_len_per_req is not None
            else num_spec_tokens + 1
        )

        self._batch_size_levels: list[int] = self._build_batch_size_levels()
        self._query_len_levels: list[int] = self._build_query_len_levels()

        # (batch_size, sum_query_len) → ITL in seconds
        self._cost_table: dict[tuple[int, int], float] = {}
        # DFlash full proposer cost. Although its draft query shape is fixed by
        # (B, K), context-KV preparation still consumes Q target hidden states.
        self._draft_cost_table: dict[tuple[int, int], float] = {}
        self._cost_records: list[dict[str, Any]] = []
        self._sorted_bs: list[int] = []
        self._sorted_sql_per_bs: dict[int, list[int]] = {}

        # req_id → recommended draft_len for the next verifier step
        self._adaptive_draft_lens: dict[str, int] = {}
        self._adaptive_decision_req_ids: frozenset[str] = frozenset()
        self._adaptive_decision_batch_size = 0
        self._decision_step = 0

        if get_tp_group().rank_in_group == 0 and get_pp_group().is_first_rank:
            logger.info(
                "VerifyAdaptiveController: bs_levels=%s  ql_levels=%s",
                self._batch_size_levels,
                self._query_len_levels,
            )

    def _build_batch_size_levels(self) -> list[int]:
        """Step-2 range from min_warmup_batch_size to cap."""
        if self.config.warmup_batch_sizes:
            return sorted(set(self.config.warmup_batch_sizes))
        cap = (
            self.config.max_warmup_batch_size
            if self.config.max_warmup_batch_size is not None
            else self.max_batch_size
        )
        start = self.config.min_warmup_batch_size
        levels = list(range(start, cap + 1, 2))
        if not levels or levels[-1] < cap:
            levels.append(cap)
        return levels

    def _build_query_len_levels(self) -> list[int]:
        """``{min_q, min_q+step, …, max_q}`` with max_q forced in."""
        min_q = self.config.min_query_len_per_req
        max_q = self.max_query_len_per_req
        step = self.config.query_len_step_per_req

        levels = list(range(min_q, max_q + 1, step))
        if not levels or levels[-1] < max_q:
            levels.append(max_q)
        return sorted(set(levels))

    def profile_cost_table(self, runner: Any) -> None:
        """Measure target and full DFlash proposer cost at every (B, Q).

        The two component tables are retained separately and summed only by
        the decision objective, so profiling output remains diagnosable.
        """
        if not self.config.enabled:
            return

        # Random cut mode: skip profiling
        if os.getenv("VLLM_DCUT_RANDOM_CUT"):
            logger.info("VerifyAdaptiveController: random cut mode enabled, skipping profiling")
            return

        if get_tp_group().rank_in_group == 0 and get_pp_group().is_first_rank:
            logger.info(
                "VerifyAdaptiveController: profiling %d ITL cost points "
                "(%d bs × %d ql).",
                len(self._batch_size_levels) * len(self._query_len_levels),
                len(self._batch_size_levels),
                len(self._query_len_levels),
            )

        max_tokens = getattr(runner, "max_num_tokens", None)

        for bs in self._batch_size_levels:
            self._sorted_sql_per_bs[bs] = []

            for ql in self._query_len_levels:
                num_tokens = bs * ql
                if max_tokens is not None and num_tokens > max_tokens:
                    logger.info(
                        "profile skip: bs=%d ql=%d num_tokens=%d > %d",
                        bs, ql, num_tokens, max_tokens,
                    )
                    continue

                sched_tokens = [ql] * bs
                logger.info('D-Cut profile: starting bs=%d ql=%d num_tokens=%d', bs, ql, num_tokens)

                runtime_mode, avg_ms, padded_tokens = runner._adaptive_profile_run(
                    sched_tokens,
                    self.config.warmup_seq_lens,
                    self.config.n_warmup_iters,
                    self.config.n_measure_iters,
                )
                elapsed_s = avg_ms / 1e3

                (
                    draft_runtime_mode,
                    draft_avg_ms,
                    draft_padded_tokens,
                ) = runner._adaptive_profile_draft_run(
                    batch_size=bs,
                    context_tokens=num_tokens,
                    n_warmup=self.config.n_warmup_iters,
                    n_measure=self.config.n_measure_iters,
                )
                draft_elapsed_s = draft_avg_ms / 1e3

                self._cost_table[(bs, num_tokens)] = elapsed_s
                self._draft_cost_table[(bs, num_tokens)] = draft_elapsed_s
                self._cost_records.append({
                    "batch_size": bs,
                    "query_len_per_req": ql,
                    "sum_query_len": num_tokens,
                    "padded_tokens": padded_tokens,
                    "seq_lens": self.config.warmup_seq_lens,
                    "runtime_mode": runtime_mode,
                    "avg_ms": avg_ms + draft_avg_ms,
                    "cost_s": elapsed_s + draft_elapsed_s,
                    "target_avg_ms": avg_ms,
                    "target_cost_s": elapsed_s,
                    "draft_runtime_mode": draft_runtime_mode,
                    "draft_padded_tokens": draft_padded_tokens,
                    "draft_avg_ms": draft_avg_ms,
                    "draft_cost_s": draft_elapsed_s,
                    "total_avg_ms": avg_ms + draft_avg_ms,
                    "total_cost_s": elapsed_s + draft_elapsed_s,
                })
                self._sorted_sql_per_bs[bs].append(num_tokens)
                if (
                    get_tp_group().rank_in_group == 0
                    and get_pp_group().is_first_rank
                ):
                    logger.info(
                        "profile  bs=%-4d  ql=%-4d  sql=%-6d  padded=%-6d  "
                        "seq_lens=%-6d  target=%-6s %.3f ms  "
                        "draft=%-6s %.3f ms  total=%.3f ms",
                        bs, ql, num_tokens, padded_tokens,
                        self.config.warmup_seq_lens,
                        runtime_mode,
                        avg_ms,
                        draft_runtime_mode,
                        draft_avg_ms,
                        avg_ms + draft_avg_ms,
                    )

        override_path = os.getenv(ENV_COST_TABLE_OVERRIDE)
        if override_path:
            # Read and validate before dumping so source==destination cannot
            # change the values that will be used by this process.
            with open(override_path, encoding="utf-8") as f:
                override_payload = json.load(f)
            override_target, override_draft = _parse_cost_table_override(
                override_payload,
                num_spec_tokens=self.num_spec_tokens,
                max_batch_size=self.max_batch_size,
                batch_size_levels=self._batch_size_levels,
                query_len_levels=self._query_len_levels,
            )

            # VLLM_DCUT_COST_TABLE_OUT remains the table measured by this run.
            # Only the in-memory decision table is replaced afterwards.
            self._dump_cost_table_if_requested()
            self._cost_table = override_target
            self._draft_cost_table = override_draft

        # TP correctness: GPU timings differ slightly per rank, which can
        # flip the argmax and cause divergent draft_lens -> NCCL deadlock.
        # Broadcast rank-0's table so all ranks decide identically.
        tp_group = get_tp_group()
        if tp_group.world_size > 1:
            self._cost_table = tp_group.broadcast_object(self._cost_table, src=0)
            self._draft_cost_table = tp_group.broadcast_object(
                self._draft_cost_table,
                src=0,
            )

        self._refresh_cost_indices()

        if get_tp_group().rank_in_group == 0 and get_pp_group().is_first_rank:
            if override_path:
                logger.warning(
                    "VerifyAdaptiveController: replaced the profiled cost "
                    "table with %d entries from %s via %s.",
                    len(self._cost_table),
                    override_path,
                    ENV_COST_TABLE_OVERRIDE,
                )
            logger.info(
                "VerifyAdaptiveController: cost table ready (%d entries).",
                len(self._cost_table),
            )
            self._log_cost_table()
        if not override_path:
            self._dump_cost_table_if_requested()

    def _refresh_cost_indices(self) -> None:
        """Rebuild lookup axes from the active profiled or overridden table."""
        self._sorted_sql_per_bs = {
            bs: [] for bs in self._batch_size_levels
        }
        for bs, sum_query_len in self._cost_table:
            self._sorted_sql_per_bs[bs].append(sum_query_len)

        # Keep only buckets with at least one available query length. An empty
        # table leaves the controller dormant instead of returning zero drafts.
        self._sorted_bs = [
            bs
            for bs in sorted(self._sorted_sql_per_bs)
            if self._sorted_sql_per_bs[bs]
        ]
        for bs in self._sorted_bs:
            self._sorted_sql_per_bs[bs].sort()

    def process_draft_output(
        self,
        selected_probs: torch.Tensor,  # [B, T] on CPU (pinned), already transferred
        req_ids: list[str],
        active_draft_req_ids: set[str],
        batch_size: int,
    ) -> None:
        """Compute and cache adaptive draft_lens from this step's drafter probs."""
        if not self.config.enabled or not active_draft_req_ids:
            self.clear_adaptive_decision()
            return

        # A decision is valid only as one coherent request-set snapshot. Clear
        # the previous one before any mode or cost-table lookup so an early
        # return can never reuse stale per-request caps.
        self.clear_adaptive_decision()

        # Random cut mode: assign random draft_lens (must be BEFORE _sorted_bs
        # check because random cut skips profiling -> _sorted_bs is empty)
        if os.getenv("VLLM_DCUT_RANDOM_CUT"):
            max_draft_len = self.max_query_len_per_req - 1
            n_rows = min(selected_probs.shape[0], len(req_ids), batch_size)
            active_req_ids = [req_ids[i] for i in range(n_rows) if req_ids[i] in active_draft_req_ids]
            draft_lens = []
            for req_id in active_req_ids:
                draft_lens.append(
                    int(np.random.randint(2, max_draft_len + 1))
                )
            self.set_adaptive_decision(
                active_req_ids,
                draft_lens,
                batch_size,
            )
            _dbg = getattr(self, "_dcut_rand_dbg_cnt", 0)
            if _dbg < 20:
                self._dcut_rand_dbg_cnt = _dbg + 1
                for rid in active_req_ids:
                    pass  # DCUT_DBG disabled
            # Periodic distribution statistics: track how many times each
            # draft_len value is assigned, print every 200 steps.
            _dist = getattr(self, "_dcut_rand_dist", None)
            if _dist is None:
                _dist = {}
                self._dcut_rand_dist = _dist
            for rid in active_req_ids:
                _dl = int(self._adaptive_draft_lens[rid])
                _dist[_dl] = _dist.get(_dl, 0) + 1
            _step = getattr(self, "_dcut_rand_step_cnt", 0) + 1
            self._dcut_rand_step_cnt = _step
            if _step % 200 == 0:
                _items = sorted(_dist.items())
                _total = sum(_dist.values())
                _parts = " ".join(f"{k}:{v}" for k, v in _items)
                print(f"[DCUT_RAND_DIST] step={_step} total={_total} dist({_parts})", flush=True)
            logger.debug(
                "random_cut: assigned random draft_lens to %d active requests (max_draft_len=%d)",
                len(active_req_ids), max_draft_len
            )
            return

        n_rows = min(selected_probs.shape[0], len(req_ids), batch_size)
        all_probs_np: np.ndarray = selected_probs[:n_rows].numpy()

        active_indices: list[int] = [
            i for i in range(n_rows) if req_ids[i] in active_draft_req_ids
        ]
        if not active_indices:
            return
        active_probs: np.ndarray = all_probs_np[active_indices]
        active_req_ids: list[str] = [req_ids[i] for i in active_indices]

        bs_key = _ceil_lookup(batch_size, self._sorted_bs)
        q_levels = self._sorted_sql_per_bs.get(bs_key) or []
        if not q_levels:
            return

        decision_dump_path = os.getenv("VLLM_DCUT_DECISION_STATS_OUT")
        result = choose_query_lens_discrete(
            probs=active_probs,
            base_batch_size=batch_size,
            q_levels=q_levels,
            cost_lookup=lambda q: self._cost_table[(bs_key, q)],
            max_draft_len=self.max_query_len_per_req - 1,
            collect_records=bool(decision_dump_path),
            draft_cost_lookup=lambda q: self._draft_cost_table[(bs_key, q)],
        )

        draft_lens = result["draft_lens"]
        self.set_adaptive_decision(
            active_req_ids,
            draft_lens,
            batch_size,
        )

        self._dump_decision_if_requested(
            decision_dump_path,
            batch_size=batch_size,
            active_count=len(active_req_ids),
            bs_key=bs_key,
            result=result,
            draft_lens=draft_lens,
        )

        logger.debug(
            "adaptive: bs_key=%d best_Q=%d best_S=%d score=%.4f draft_lens=%s",
            bs_key, result["best_Q"], result["best_S"],
            result["best_score"], draft_lens,
        )

    def get_adaptive_draft_len(self, req_id: str) -> int | None:
        """Cached draft_len for *req_id*, or None (→ use full spec tokens)."""
        return self._adaptive_draft_lens.get(req_id)

    def set_adaptive_decision(
        self,
        req_ids: list[str],
        draft_lens: list[int],
        batch_size: int,
    ) -> None:
        """Atomically replace the per-request caps and their source batch."""
        if len(req_ids) != len(draft_lens):
            raise ValueError(
                "D-Cut decision length mismatch: "
                f"req_ids={len(req_ids)} draft_lens={len(draft_lens)}"
            )
        self._adaptive_draft_lens = dict(zip(req_ids, draft_lens))
        self._adaptive_decision_req_ids = frozenset(req_ids)
        self._adaptive_decision_batch_size = int(batch_size)

    def clear_adaptive_decision(self) -> None:
        """Invalidate the complete cached decision snapshot."""
        self._adaptive_draft_lens.clear()
        self._adaptive_decision_req_ids = frozenset()
        self._adaptive_decision_batch_size = 0

    def matches_adaptive_request_set(self, req_ids) -> bool:
        """Whether cached caps were optimized for exactly this spec batch."""
        current_req_ids = frozenset(req_ids)
        return (
            len(current_req_ids) == self._adaptive_decision_batch_size
            and current_req_ids == self._adaptive_decision_req_ids
        )

    def invalidate(self, req_id: str) -> None:
        """Drop cached state for a completed or evicted request."""
        self._adaptive_draft_lens.pop(req_id, None)

    def _dump_decision_if_requested(
        self,
        dump_path: str | None,
        *,
        batch_size: int,
        active_count: int,
        bs_key: int,
        result: dict[str, Any],
        draft_lens: list[int],
    ) -> None:
        if not dump_path:
            return
        if get_tp_group().rank_in_group != 0 or not get_pp_group().is_first_rank:
            return

        self._decision_step += 1
        controller_cap_draft_len = self.max_query_len_per_req - 1
        draft_sum = int(sum(draft_lens))
        cap_sum = int(active_count * controller_cap_draft_len)
        best_Q = int(result["best_Q"])
        records = result.get("records") or []
        scores = []
        for record in records:
            Q = int(record["Q"])
            scores.append({
                "Q": Q,
                "query_len_per_req": (
                    Q // bs_key if bs_key > 0 and Q % bs_key == 0 else None
                ),
                "S": int(record["S"]),
                "score": float(record["score"]),
                "cost_ms": float(record["cost"]) * 1e3,
                "target_cost_ms": float(record["target_cost"]) * 1e3,
                "draft_cost_ms": float(record["draft_cost"]) * 1e3,
            })

        payload = {
            "step": self._decision_step,
            "batch_size": int(batch_size),
            "active_count": int(active_count),
            "bs_key": int(bs_key),
            "best_Q": best_Q,
            "best_query_len_per_req": (
                best_Q // bs_key if bs_key > 0 and best_Q % bs_key == 0 else None
            ),
            "best_S": int(result["best_S"]),
            "best_score": float(result["best_score"]),
            "controller_cap_draft_len": int(controller_cap_draft_len),
            "draft_len_sum": draft_sum,
            "controller_cap_draft_len_sum": cap_sum,
            "trimmed_vs_controller_cap": cap_sum - draft_sum,
            "avg_draft_len": draft_sum / active_count if active_count else 0.0,
            "min_draft_len": int(min(draft_lens)) if draft_lens else 0,
            "max_draft_len": int(max(draft_lens)) if draft_lens else 0,
            "cap_like_reqs": int(sum(
                d >= controller_cap_draft_len for d in draft_lens
            )),
            "scores": scores,
        }

        dirname = os.path.dirname(dump_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(dump_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, sort_keys=True) + "\n")

    def _log_cost_table(self) -> None:
        """Log the full profiled cost table as a bs x query_len grid (ms).

        Runs once after profiling on rank 0.  This is purely observability — the
        per-point ``profile bs=...`` lines and the JSON dump already carry the
        same numbers; this gives an at-a-glance view in the server log.
        """
        if not self._cost_table:
            return
        qls = list(self._query_len_levels)
        bss = list(self._batch_size_levels)
        logger.info(
            "D-Cut total cost table (ms/target+draft, seq_lens=%d; rows=batch_size, cols=query_len/req):",
            self.config.warmup_seq_lens,
        )
        header = "  bs\\ql |" + "".join(f"{q:>9d}" for q in qls)
        logger.info("%s", header)
        logger.info("  %s", "-" * (len(header) - 2))
        for bs in bss:
            cells = []
            for ql in qls:
                key = (bs, bs * ql)
                target_cost_s = self._cost_table.get(key)
                draft_cost_s = self._draft_cost_table.get(key)
                cost_s = (
                    target_cost_s + draft_cost_s
                    if target_cost_s is not None
                    and draft_cost_s is not None
                    else None
                )
                cells.append(
                    f"{cost_s * 1e3:>9.2f}" if cost_s is not None else f"{'-':>9}"
                )
            logger.info("  %5d |%s", bs, "".join(cells))

    def _dump_cost_table_if_requested(self) -> None:
        dump_path = (
            os.getenv("VLLM_DCUT_COST_TABLE_OUT")
            or self.config.cost_table_dump_path
        )
        if not dump_path:
            return
        if get_tp_group().rank_in_group != 0 or not get_pp_group().is_first_rank:
            return

        rows = []
        for (bs, sum_query_len), target_cost_s in sorted(self._cost_table.items()):
            draft_cost_s = self._draft_cost_table.get((bs, sum_query_len), 0.0)
            cost_s = target_cost_s + draft_cost_s
            rows.append({
                "batch_size": bs,
                "sum_query_len": sum_query_len,
                "query_len_per_req": (
                    sum_query_len // bs if bs > 0 and sum_query_len % bs == 0
                    else None
                ),
                "cost_s": cost_s,
                "cost_ms": cost_s * 1e3,
                "target_cost_s": target_cost_s,
                "target_cost_ms": target_cost_s * 1e3,
                "draft_cost_s": draft_cost_s,
                "draft_cost_ms": draft_cost_s * 1e3,
            })

        payload = {
            "schema_version": 2,
            "num_spec_tokens": self.num_spec_tokens,
            "max_batch_size": self.max_batch_size,
            "warmup_seq_lens": self.config.warmup_seq_lens,
            "n_warmup_iters": self.config.n_warmup_iters,
            "n_measure_iters": self.config.n_measure_iters,
            "batch_size_levels": self._batch_size_levels,
            "query_len_levels": self._query_len_levels,
            "cost_table": rows,
            "profile_records": self._cost_records,
        }

        dirname = os.path.dirname(dump_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        tmp_path = f"{dump_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp_path, dump_path)
        logger.info("VerifyAdaptiveController: dumped cost table to %s",
                    dump_path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ceil_lookup(val: int, sorted_keys: list[int]) -> int:
    """Smallest key ≥ val; falls back to max key when val is out of range."""
    idx = bisect.bisect_left(sorted_keys, val)
    return sorted_keys[min(idx, len(sorted_keys) - 1)]
