# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deferred vs immediate RecoverSSM verify/commit device-kernel latency table.

Protocol from the PR #22516 description ("device kernel performance"),
mirroring ``benchmarks/benchmark_kda_recoverssm_graph.py``; both paths come
from this checkout's operators instead of two checkouts:

* ``immediate``: a plain verify per layer plus the per-step state commit
  (the commit kernel folds the accepted records into the checkpoint).
* ``deferred``: a fused-replay verify per layer (it folds the previous
  step's pending records in-kernel) plus the one-warp defer kernel (the
  commit becomes a pending count).

* Each case runs 24 executions per path in 12 order-balanced pairs; within
  a pair each path runs once first and once second (an ABBA block), so
  ordering cannot bias the medians.
* Before timing, each case settles the card with sustained full steps for
  ``--settle`` seconds, so the frequency governor reaches steady state
  instead of being measured on its boost ramp.
* Durations are the pure device execution times of the kernels, read from
  the ``kernel_details.csv`` that ``torch_npu.profiler`` (Level0) exports
  for the run — the same number ``msprof`` shows. Host dispatch and
  launch-queue gaps cannot enter the timing. Traces land under
  ``npu_profile_trace``.
* Verify is the median of the per-execution mean over the L layer calls,
  commit runs once across the layers, and the total is the median of each
  execution's sum of L verify calls plus one commit.
* Timing excludes the commit plan, the conv compaction and the state
  reset: the plan and conv kernels are no-op'd after warmup, and the
  deferred path restores the warmup replay plan from a snapshot instead
  of rebuilding it (the defer kernel zeroes the plan it consumes), so both
  paths replay/commit the same counts every execution.
* ``rate k/Q``: every request accepts k of its Q draft tokens — a uniform
  acceptance-rate sweep, the axis production actually moves on (the bimodal
  zero/full splits production never produces are gone). K=V=128 throughout.

The deferred path runs in steady state (every verify replays the previous
step's accepted counts), which is what the engine does between flushes.

The ``Gate`` / ``Gated reduction`` columns run the production acceptance
gate over each pattern's counts: when its EMA of the mean acceptance rate
crosses the threshold the engine folds immediately instead of deferring,
so the gated reduction is the arm the gate picks (gate-on steps cost the
immediate path plus a one-warp defer kernel).

Run under ``ASCEND_RT_VISIBLE_DEVICES`` to pick the card (device npu:0).
"""

import argparse
import contextlib
import csv
import gc
import os
import statistics
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import torch
import torch_npu  # noqa: F401
from torch_npu.profiler import (
    ProfilerActivity,
    profile,
    tensorboard_trace_handler,
)

from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first

import vllm_ascend.ops.triton.kda.recoverssm as recoverssm_module
from vllm_ascend.models.glm5next.recoverssm_deferred import RecoverSSMAcceptanceGate
from vllm_ascend.ops.triton.kda.recoverssm import (
    KDARecoverSSMCommitContext,
    kda_recoverssm_verify,
)
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

init_device_properties_triton()

SHAPES = (
    # B, L, H, K, V, Q
    (30, 23, 12, 128, 128, 4),
    (12, 11, 32, 128, 128, 6),
    (12, 11, 64, 128, 128, 6),
)
LOWER_BOUND = -5.0
# Recorded serving page strides for the H32/H64 checkpoint pools.
SERVING_PAGE_STRIDE = {32: 625152, 64: 1250304}
PLAN_KERNELS = ("_prepare_commit_plan_kernel", "_compact_conv_state_kernel")
TRACE_ROOT = "npu_profile_trace"
TRACE_CSVS = ("kernel_details.csv", "op_summary.csv")
# Name fragments identifying the measured kernels in the trace CSVs.
VERIFY_MARK = "recoverssm_verify"
COMMIT_MARK = "commit_kda_state"
DEFER_MARK = "defer_commit"


def _profiler_options() -> dict:
    """Level0 (kernel durations only) where this torch_npu has it."""
    try:
        from torch_npu.profiler import AiCMetrics, ProfilerLevel, _ExperimentalConfig

        return {
            "experimental_config": _ExperimentalConfig(
                profiler_level=ProfilerLevel.Level0,
                aic_metrics=AiCMetrics.AiCoreNone,
            )
        }
    except (ImportError, AttributeError, TypeError, ValueError):
        return {}


# The acceptance sweep: every request accepts k of its Q draft tokens, one
# bench case per level (a uniform rate, the axis production actually moves on).
RATE_PATTERN = "rate {accepted}/{spec_query_len}"


def acceptance_counts(pattern: str, batch: int, spec_query_len: int) -> list[int]:
    prefix, _, tail = pattern.partition(" ")
    if prefix != "rate" or not tail:
        raise ValueError(f"unknown acceptance pattern: {pattern}")
    accepted = min(int(tail.partition("/")[0]), spec_query_len)
    return [accepted] * batch


def gate_arm(shape: tuple, pattern: str) -> str:
    """The arm the acceptance gate settles on for a pattern's counts.

    Runs the production gate object over the pattern's acceptance counts
    until its EMA converges, then reports the arm an engine step would take.
    """
    batch, _, _, _, _, spec_query_len = shape
    counts = torch.tensor(
        acceptance_counts(pattern, batch, spec_query_len), dtype=torch.int32, device="npu"
    )
    gate = RecoverSSMAcceptanceGate(batch, "npu")
    for _ in range(16):
        gate.update(counts, None, spec_query_len, batch)
    torch.npu.synchronize()  # land the host mirror before reading the gate
    return "immediate" if gate.engaged() else "deferred"


@dataclass
class Case:
    label: str
    layers: list
    context: Any
    starts: torch.Tensor
    indices: torch.Tensor
    counts: torch.Tensor
    outputs: list
    resets: list
    # Deferred-only: per-request record slots, the pending counts, and the
    # steady-state replay plan snapshot (the plan kernels are skipped while
    # measuring, so each execution restores the plan from the snapshot).
    record_rows: torch.Tensor | None = None
    pending: torch.Tensor | None = None
    plan: tuple[torch.Tensor, ...] | None = field(default=None, repr=False)

    def launch_verify_layer(self, layer) -> None:
        kwargs = {}
        if self.label == "deferred":
            context = self.context
            kwargs = dict(
                record_indices=self.record_rows,
                replay_commit_lens=context.commit_lens,
                replay_final_state_indices=context.final_state_indices,
                replay_boundary_state_indices=context.boundary_state_indices,
                replay_boundary_recovery_lens=context.boundary_recovery_lens,
            )
        kda_recoverssm_verify(
            *layer.verify_inputs,
            layer.gate_lower_bound,
            layer.kv_cache[1],
            layer.kv_cache[2],
            layer.kv_cache[3],
            self.starts,
            self.indices,
            layer.spec_query_len,
            out=layer.verify_out,
            **kwargs,
        )

    def launch_verify(self) -> None:
        for layer in self.layers:
            self.launch_verify_layer(layer)

    def commit(self) -> None:
        if self.label == "deferred":
            self.context.commit(
                self.counts,
                self.indices,
                self.starts,
                record_rows=self.record_rows,
                pending=self.pending,
            )
        else:
            self.context.commit(self.counts, self.indices, self.starts)

    def step(self) -> None:
        """One engine step (warmup only): compact, verify, commit."""
        if self.label == "deferred":
            self.context.compact_conv(self.indices, self.record_rows, self.pending)
        self.launch_verify()
        self.commit()

    def snapshot_plan(self) -> None:
        context = self.context
        self.plan = (
            context.commit_lens.clone(),
            context.final_state_indices.clone(),
            context.boundary_state_indices.clone(),
            context.boundary_recovery_lens.clone(),
        )

    def arm_replay(self) -> None:
        context = self.context
        live = (
            context.commit_lens,
            context.final_state_indices,
            context.boundary_state_indices,
            context.boundary_recovery_lens,
        )
        for saved, current in zip(self.plan, live):
            current.copy_(saved)

    def reset_state(self) -> None:
        for layer, (initial_state, initial_conv) in zip(self.layers, self.resets):
            layer.kv_cache[1].copy_(initial_state)
            layer.kv_cache[0].copy_(initial_conv)


def build_case(shape: tuple, pattern: str, label: str) -> Case:
    batch, num_layers, heads, key_dim, value_dim, spec_query_len = shape
    if key_dim != value_dim:
        raise ValueError("the table only covers square K/V heads")
    device, dtype = "npu", torch.bfloat16
    slots = batch + 2
    page_stride = SERVING_PAGE_STRIDE.get(heads)
    wide_stride = None if page_stride is None else 2 * page_stride
    torch.manual_seed(20261126 + num_layers)

    def allocate(shape, dtype, strides):
        if page_stride is None:
            return torch.empty(shape, dtype=dtype, device=device)
        return torch.empty_strided(shape, strides, dtype=dtype, device=device)

    starts = torch.arange(batch + 1, dtype=torch.int32, device=device) * spec_query_len
    indices = torch.arange(1, batch + 1, dtype=torch.int32, device=device)
    counts = torch.tensor(
        acceptance_counts(pattern, batch, spec_query_len),
        dtype=torch.int32,
        device=device,
    )
    layers, outputs, resets = [], [], []
    for _ in range(num_layers):
        state = allocate(
            (slots, heads, value_dim, key_dim),
            torch.float32,
            (page_stride, value_dim * key_dim, key_dim, 1),
        )
        state.copy_(torch.randn(state.shape, dtype=torch.float32, device=device) * 0.05)
        correction = allocate(
            (slots, heads, spec_query_len, value_dim),
            torch.float32,
            (page_stride, spec_query_len * value_dim, value_dim, 1),
        )
        kg = allocate(
            (slots, heads, spec_query_len, 2 * key_dim),
            dtype,
            (wide_stride, spec_query_len * 2 * key_dim, 2 * key_dim, 1),
        )
        conv = allocate(
            (slots, 3 * heads * key_dim, spec_query_len + 2),
            dtype,
            (wide_stride, 1, 3 * heads * key_dim),
        )
        conv.copy_(torch.randn(conv.shape, dtype=dtype, device=device))
        stored_conv = conv if is_conv_state_dim_first() else conv.transpose(-1, -2)
        projections = torch.randn(
            1, batch * spec_query_len, 4 * heads * key_dim, dtype=dtype, device=device
        )
        q, k, v, raw_g = (
            part.view(1, batch * spec_query_len, heads, key_dim)
            for part in projections.split(heads * key_dim, dim=-1)
        )
        raw_beta = torch.randn(
            1, batch * spec_query_len, heads, dtype=dtype, device=device
        )
        a_log = torch.randn(heads, device=device) * 0.1
        dt_bias = torch.randn(heads * key_dim, device=device) * 0.1
        layer = SimpleNamespace(
            kv_cache=(stored_conv, state, correction, kg),
            A_log=a_log,
            dt_bias=dt_bias,
            gate_lower_bound=LOWER_BOUND,
            local_num_heads=heads,
            head_dim=key_dim,
            spec_query_len=spec_query_len,
            verify_inputs=(q, k, v, raw_g, raw_beta, a_log, dt_bias),
            verify_out=torch.empty_like(v),
        )
        layers.append(layer)
        outputs.append(layer.verify_out)
        resets.append((state.clone(), stored_conv.clone()))
    context = KDARecoverSSMCommitContext.create(
        layers, spec_query_len=spec_query_len, max_num_reqs=batch
    )
    case = Case(
        label=label,
        layers=layers,
        context=context,
        starts=starts,
        indices=indices,
        counts=counts,
        outputs=outputs,
        resets=resets,
    )
    if label == "deferred":
        case.record_rows = torch.arange(batch, dtype=torch.int32, device=device)
        case.pending = torch.zeros(slots, dtype=torch.int32, device=device)
    return case


@contextlib.contextmanager
def skipped_kernels(module, names):
    class Skip:
        def __getitem__(self, grid):
            return lambda *args, **kwargs: None

    originals = [(name, getattr(module, name, None)) for name in names]
    try:
        for name, original in originals:
            if original is not None:
                setattr(module, name, Skip())
        yield
    finally:
        for name, original in originals:
            if original is not None:
                setattr(module, name, original)


def _trace_csvs(root: str) -> set[str]:
    """Every kernel-duration CSV under the trace root, any nesting."""
    found = set()
    for dirpath, _, names in os.walk(root):
        for name in names:
            if name in TRACE_CSVS:
                found.add(os.path.join(dirpath, name))
    return found


def _new_kernel_rows(root: str, known: set[str]) -> list[tuple[str, float, float]]:
    """(name, duration us, start us) rows from the trace this case wrote."""
    fresh: dict[str, list[str]] = {}
    for path in _trace_csvs(root) - known:
        fresh.setdefault(os.path.dirname(path), []).append(path)
    picked = []
    for _, paths in sorted(fresh.items()):
        for preferred in TRACE_CSVS:
            match = [p for p in paths if os.path.basename(p) == preferred]
            if match:
                picked.append(match[0])
                break
    if not picked:
        raise RuntimeError(
            f"torch_npu.profiler wrote no trace CSV under {root!r};"
            " check that profiling (MSPTI) is available on this system"
        )
    rows = []
    for path in picked:
        rows.extend(_read_kernel_rows(path))
    rows.sort(key=lambda row: row[2])
    return rows


def _read_kernel_rows(path: str) -> list[tuple[str, float, float]]:
    with open(path, newline="", encoding="utf-8") as handle:
        records = list(csv.reader(handle))
    if len(records) < 2:
        return []
    header = [column.strip().lower() for column in records[0]]
    marks = (VERIFY_MARK, COMMIT_MARK, DEFER_MARK)
    # Header spellings differ across CANN versions, so find the kernel
    # name column by the data it carries.
    name_col = None
    for col in range(len(records[0])):
        if any(
            len(record) > col and any(mark in record[col] for mark in marks)
            for record in records[1:]
        ):
            name_col = col
            break
    duration_col = start_col = None
    for col, column in enumerate(header):
        if duration_col is None and "duration" in column:
            duration_col = col
        elif start_col is None and "start" in column:
            start_col = col
    if name_col is None or duration_col is None:
        raise RuntimeError(f"unrecognized trace CSV {path!r}: header {header[:8]}")
    rows = []
    for position, record in enumerate(records[1:]):
        if len(record) <= max(name_col, duration_col):
            continue
        name = record[name_col]
        if not any(mark in name for mark in marks):
            continue
        start = (
            float(record[start_col])
            if start_col is not None and len(record) > start_col
            else float(position)
        )
        rows.append((name, float(record[duration_col]), start))
    return rows


def _assign_rows(
    rows: list[tuple[str, float, float]],
    sequence: list[str],
    num_layers: int,
) -> dict[str, list]:
    """Walk the trace in execution order: L verify kernels, then the
    path's single commit/defer kernel, per execution."""
    records = {label: [] for label in ("immediate", "deferred")}
    position = 0
    for label in sequence:
        verify = []
        for _ in range(num_layers):
            name, duration, _ = rows[position]
            if VERIFY_MARK not in name:
                raise RuntimeError(
                    f"expected a verify kernel at trace row {position},"
                    f" got {name!r}"
                )
            verify.append(duration)
            position += 1
        name, duration, _ = rows[position]
        expected = DEFER_MARK if label == "deferred" else COMMIT_MARK
        if expected not in name:
            raise RuntimeError(
                f"expected a {expected} kernel at trace row {position},"
                f" got {name!r}"
            )
        records[label].append((verify, duration))
        position += 1
    if position != len(rows):
        raise RuntimeError(
            f"the trace holds {len(rows)} measured kernels but the run"
            f" sequence accounts for {position}"
        )
    return records


def _medians(records: list) -> tuple:
    verify_means, commit_times, totals = [], [], []
    for verify_us, commit_us in records:
        verify_means.append(statistics.mean(verify_us))
        commit_times.append(commit_us)
        totals.append(sum(verify_us) + commit_us)
    return (
        statistics.median(verify_means),
        statistics.median(commit_times),
        statistics.median(totals),
    )


@torch.inference_mode()
def measure_case(args: argparse.Namespace, shape: tuple, pattern: str) -> dict:
    cases = (
        build_case(shape, pattern, "immediate"),
        build_case(shape, pattern, "deferred"),
    )
    for case in cases:
        # Warmup with the real plan kernels: sustained steps settle the
        # card's frequency governor before the timed window, and put the
        # deferred path in steady state (pending = counts) with a valid
        # plan for the measured executions, where the plan and conv kernels
        # are skipped and the replay plan is restored from the snapshot.
        deadline = time.perf_counter() + args.settle
        while time.perf_counter() < deadline:
            case.step()
        if case.label == "deferred":
            case.context.compact_conv(case.indices, case.record_rows, case.pending)
            case.snapshot_plan()
        torch.npu.synchronize()
        if not all(torch.isfinite(output).all().item() for output in case.outputs):
            raise RuntimeError(
                f"non-finite verify output for {shape} {pattern} ({case.label})"
            )

    def queue_execution(case: Case) -> None:
        case.reset_state()
        if case.label == "deferred":
            case.arm_replay()
        case.launch_verify()
        case.commit()

    # ABBA: each path runs once first and once second per pair, so ordering
    # cannot bias the medians. The kernels land in the trace in exactly
    # this order, which is how rows are attributed to a path.
    sequence = []
    for _ in range(args.pairs):
        sequence += ["immediate", "deferred", "deferred", "immediate"]
    by_label = {case.label: case for case in cases}
    known = _trace_csvs(TRACE_ROOT)
    with profile(
        activities=[ProfilerActivity.NPU],
        on_trace_ready=tensorboard_trace_handler(TRACE_ROOT),
        **_profiler_options(),
    ):
        with skipped_kernels(recoverssm_module, PLAN_KERNELS):
            for label in sequence:
                queue_execution(by_label[label])
        torch.npu.synchronize()
    rows = _new_kernel_rows(TRACE_ROOT, known)
    records = _assign_rows(rows, sequence, num_layers=len(by_label["immediate"].layers))
    del cases
    gc.collect()
    torch.npu.empty_cache()
    return {label: _medians(entries) for label, entries in records.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs", type=int, default=12, help="order-balanced ABBA pairs per case"
    )
    parser.add_argument(
        "--settle",
        type=float,
        default=1.0,
        help="seconds of warmup per case to settle the frequency governor",
    )
    parser.add_argument("--filter", type=str, help="substring filter for case labels")
    return parser.parse_args()


def shape_label(shape: tuple) -> str:
    batch, num_layers, heads, _, _, spec_query_len = shape
    return f"{batch} / {num_layers} / {heads} / {spec_query_len}"


def main() -> None:
    args = parse_args()
    if not torch.npu.is_available():
        raise SystemExit("requires an Ascend NPU")
    print(f"operators: {recoverssm_module.__file__}")
    print("immediate: plain verify + per-step state commit")
    print("deferred : fused-replay verify + one-warp defer commit")
    print("gated    : the arm the acceptance gate settles on for the pattern;")
    print("           gate-on steps add the one-warp defer kernel to immediate")
    print(f"timing: torch_npu.profiler kernel durations (traces in {TRACE_ROOT}/)")
    print(
        f"{args.pairs} ABBA pairs ({2 * args.pairs} executions per path),"
        f" {args.settle:.1f}s settle per case"
    )
    rows = []
    for shape in SHAPES:
        spec_query_len = shape[5]
        for accepted in range(1, spec_query_len + 1):
            pattern = RATE_PATTERN.format(accepted=accepted, spec_query_len=spec_query_len)
            label = f"{shape_label(shape)} {pattern}"
            if args.filter and args.filter not in label:
                continue
            stats = measure_case(args, shape, pattern)
            imm, deferred = stats["immediate"], stats["deferred"]
            picked = gate_arm(shape, pattern)
            gated = imm if picked == "immediate" else deferred
            reduction = (imm[2] - deferred[2]) / imm[2] * 100
            gated_reduction = (imm[2] - gated[2]) / imm[2] * 100
            rows.append(
                (shape_label(shape), pattern, imm, deferred, reduction, picked, gated_reduction)
            )
            print(
                f"{label}: immediate total {imm[2]:.2f} µs |"
                f" deferred total {deferred[2]:.2f} µs |"
                f" reduction {reduction:.2f}% |"
                f" gate picks {picked} (gated reduction {gated_reduction:.2f}%)",
                flush=True,
            )
    print(
        "| B / L / H / Q | Acceptance | immediate verify / layer"
        " | deferred verify / layer | immediate commit | deferred commit"
        " | immediate total | deferred total | Total reduction | Gate | Gated reduction |"
    )
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|")
    for label, pattern, imm, deferred, reduction, picked, gated_reduction in rows:
        print(
            f"| {label} | {pattern} | {imm[0]:.2f} | {deferred[0]:.2f}"
            f" | {imm[1]:.2f} | {deferred[1]:.2f}"
            f" | {imm[2]:.2f} | {deferred[2]:.2f}"
            f" | {reduction:.2f}% | {picked} | {gated_reduction:.2f}% |"
        )


if __name__ == "__main__":
    main()
