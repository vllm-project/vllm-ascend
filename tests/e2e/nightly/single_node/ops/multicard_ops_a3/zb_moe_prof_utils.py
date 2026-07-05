# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""NPU timing helpers for ZB MoE e2e tests (adapted from deepep_standalone/test/utils.py)."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import uuid
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path

import numpy as np
import torch
import torch_npu

ZB_MOE_KERNELS = (
    "ZbMoeDistributeDispatch",
    "ZbMoeDistributeCombine",
)

V2_MOE_KERNELS = (
    "MoeDistributeDispatchV2",
    "MoeDistributeCombineV2",
)

# Kernel names as they appear in Kineto chrome traces (lowercase op entry symbols).
FUSED_MC2_FFNC_KERNEL = ("dispatch_ffn_combine",)
FUSED_MC2_GMMCD_KERNEL = ("dispatch_gmm_combine_decode",)


class _EmptySuppress:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


class _SuppressStdoutStderr:
    def __enter__(self):
        self.outnull_file = open(os.devnull, "w")
        self.errnull_file = open(os.devnull, "w")
        self.old_stdout_fileno_undup = sys.stdout.fileno()
        self.old_stderr_fileno_undup = sys.stderr.fileno()
        self.old_stdout_fileno = os.dup(sys.stdout.fileno())
        self.old_stderr_fileno = os.dup(sys.stderr.fileno())
        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr
        os.dup2(self.outnull_file.fileno(), self.old_stdout_fileno_undup)
        os.dup2(self.errnull_file.fileno(), self.old_stderr_fileno_undup)
        sys.stdout = self.outnull_file
        sys.stderr = self.errnull_file
        return self

    def __exit__(self, *_):
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
        os.dup2(self.old_stdout_fileno, self.old_stdout_fileno_undup)
        os.dup2(self.old_stderr_fileno, self.old_stderr_fileno_undup)
        os.close(self.old_stdout_fileno)
        os.close(self.old_stderr_fileno)
        self.outnull_file.close()
        self.errnull_file.close()


def _stdout_stderr_context(suppress_output: bool) -> AbstractContextManager[None]:
    if suppress_output:
        return _SuppressStdoutStderr()
    return _EmptySuppress()


def _load_trace_events(trace_path: Path) -> list:
    profile_data = json.loads(trace_path.read_text())
    if isinstance(profile_data, dict):
        return profile_data.get("traceEvents", [])
    return profile_data


def _read_event_times(
    starts: list[torch.npu.Event],
    ends: list[torch.npu.Event],
) -> list[float]:
    """Read NPU event pairs after a single device sync."""
    torch.npu.synchronize()
    return [start.elapsed_time(end) / 1e3 for start, end in zip(starts, ends)]


def bench(
    fn: Callable[[], None],
    num_warmups: int = 50,
    num_tests: int = 100,
    post_fn: Callable[[], None] | None = None,
) -> tuple[float, float, float]:
    """Return (avg, min, max) wall time in seconds using NPU events.

    Runs ``num_warmups + num_tests`` iterations back-to-back in one session.
    Only the last ``num_tests`` iterations are timed (e.g. 50 + 100 = 150 total).
    No per-iteration ``torch.npu.synchronize()`` — events are batch-read once
    at the end.
    """
    torch.npu.synchronize()

    total_iters = num_warmups + num_tests
    starts: list[torch.npu.Event] = []
    ends: list[torch.npu.Event] = []
    for i in range(total_iters):
        record = i >= num_warmups
        if record:
            start = torch.npu.Event(enable_timing=True)
            start.record()
        fn()
        if record:
            end = torch.npu.Event(enable_timing=True)
            end.record()
            if post_fn is not None:
                post_fn()
            starts.append(start)
            ends.append(end)

    times = _read_event_times(starts, ends)
    samples = np.array(times, dtype=np.float64)
    return float(np.average(samples)), float(np.min(samples)), float(np.max(samples))


def bench_moe_dispatch(
    run_dispatch: Callable[[], None],
    run_combine: Callable[[], None],
    num_warmups: int = 50,
    num_tests: int = 100,
) -> tuple[float, float, float]:
    """Time dispatch only on the last ``num_tests`` of a continuous session.

    Each iteration runs dispatch; combine always follows (untimed) so combine
    metadata stays warm, matching steady-state serving.
    """
    torch.npu.synchronize()
    total_iters = num_warmups + num_tests
    starts: list[torch.npu.Event] = []
    ends: list[torch.npu.Event] = []
    for i in range(total_iters):
        record = i >= num_warmups
        if record:
            start = torch.npu.Event(enable_timing=True)
            start.record()
        run_dispatch()
        if record:
            end = torch.npu.Event(enable_timing=True)
            end.record()
            starts.append(start)
            ends.append(end)
        run_combine()

    times = _read_event_times(starts, ends)
    samples = np.array(times, dtype=np.float64)
    return float(np.average(samples)), float(np.min(samples)), float(np.max(samples))


def bench_moe_combine(
    run_dispatch: Callable[[], None],
    run_combine: Callable[[], None],
    num_warmups: int = 50,
    num_tests: int = 100,
) -> tuple[float, float, float]:
    """Time combine only on the last ``num_tests`` of a continuous session.

    Each iteration runs dispatch (untimed) then combine, keeping dispatch output
    fresh before every combine — same as production per-token flow.
    """
    torch.npu.synchronize()
    total_iters = num_warmups + num_tests
    starts: list[torch.npu.Event] = []
    ends: list[torch.npu.Event] = []
    for i in range(total_iters):
        run_dispatch()
        record = i >= num_warmups
        if record:
            start = torch.npu.Event(enable_timing=True)
            start.record()
        run_combine()
        if record:
            end = torch.npu.Event(enable_timing=True)
            end.record()
            starts.append(start)
            ends.append(end)

    times = _read_event_times(starts, ends)
    samples = np.array(times, dtype=np.float64)
    return float(np.average(samples)), float(np.min(samples)), float(np.max(samples))


def _kernel_event_matches(event_name: object, kernel_name: str) -> bool:
    if not isinstance(event_name, str):
        return False
    event_lower = event_name.lower()
    kernel_lower = kernel_name.lower()
    return event_lower == kernel_lower or kernel_lower in event_lower or event_lower in kernel_lower


def _msprof_experimental_config():
    """Match vLLM-Ascend serving torch profiler (full msprof export)."""
    return torch_npu.profiler._ExperimentalConfig(
        export_type=torch_npu.profiler.ExportType.Text,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        msprof_tx=False,
        aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
        l2_cache=False,
        op_attr=False,
        data_simplification=True,
        record_op_args=False,
        gc_detect_threshold=None,
    )


def profile_msprof(
    fn: Callable[[], None],
    trace_root: str,
    worker_name: str,
    num_warmups: int = 50,
    num_tests: int = 100,
    suppress_output: bool = False,
    run_analyse: bool = True,
) -> str:
    """Capture a full Ascend/msprof trace (CPU + NPU) under ``trace_root``.

    Runs ``num_warmups + num_tests`` iterations in one continuous session.
    Warmup iterations execute outside the profiler; the last ``num_tests``
    iterations are recorded (full msprof bundle).
    """
    trace_root_path = Path(trace_root)
    trace_root_path.mkdir(parents=True, exist_ok=True)

    for _ in range(num_warmups):
        fn()
    torch.npu.synchronize()

    suppress = _stdout_stderr_context(suppress_output)
    with (
        suppress,
        torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU,
            ],
            with_stack=False,
            profile_memory=False,
            with_modules=False,
            experimental_config=_msprof_experimental_config(),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                str(trace_root_path),
                worker_name=worker_name,
            ),
        ),
    ):
        for _ in range(num_tests):
            fn()
        torch.npu.synchronize()

    if run_analyse:
        _analyse_msprof_bundles(trace_root_path)
    return str(trace_root_path)


def _analyse_msprof_bundles(trace_root: Path) -> list[Path]:
    """Run ascend analyse on profiler bundles; return analysed ascend_pt dirs."""
    try:
        from torch_npu.profiler.profiler import analyse
    except ImportError:
        return []

    analysed: list[Path] = []
    bundles = sorted(trace_root.rglob("*_ascend_pt"))
    if not bundles:
        bundles = sorted(p for p in trace_root.rglob("*") if p.is_dir() and p.name.endswith("_ascend_pt"))
    for bundle in bundles:
        if not bundle.is_dir():
            continue
        out_dir = bundle / "ASCEND_PROFILER_OUTPUT"
        if out_dir.is_dir():
            analysed.append(bundle)
            continue
        with _EmptySuppress():
            analyse(str(bundle))
        analysed.append(bundle)
    return analysed


def msprof_kernel_summary(
    trace_root: str,
    kernel_names: str | tuple[str, ...],
) -> tuple[float, ...] | None:
    """Return per-kernel average seconds from msprof op_statistic.csv (best effort)."""
    names = (kernel_names,) if isinstance(kernel_names, str) else kernel_names
    return _kernel_durations_from_msprof(Path(trace_root), names)


def _kernel_durations_from_msprof(
    trace_root: Path,
    kernel_names: tuple[str, ...],
) -> tuple[float, ...] | None:
    """Best-effort dispatch/combine averages from msprof ``op_statistic.csv``."""
    import csv

    for bundle in _analyse_msprof_bundles(trace_root):
        csv_path = bundle / "ASCEND_PROFILER_OUTPUT" / "op_statistic.csv"
        if not csv_path.is_file():
            continue
        with csv_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
        if not rows:
            continue

        header = {name.lower(): name for name in rows[0]}
        name_key = header.get("op_name") or header.get("name") or header.get("op type")
        time_key = (
            header.get("total_time(us)")
            or header.get("total time(us)")
            or header.get("total_time")
            or header.get("duration(us)")
        )
        if name_key is None or time_key is None:
            continue

        durations: list[float] = []
        for kernel_name in kernel_names:
            matched = [row for row in rows if kernel_name.lower() in str(row.get(name_key, "")).lower()]
            if not matched:
                break
            total_us = sum(float(row[time_key]) for row in matched)
            durations.append(total_us / len(matched) / 1e6)
        if len(durations) == len(kernel_names):
            return tuple(durations)
    return None


def print_msprof_trace_info(
    *,
    rank: int,
    label: str,
    trace_root: str,
    num_warmups: int,
    num_tests: int,
    kernel_names: tuple[str, ...] | None = None,
    kernel_durations: tuple[float, ...] | None = None,
) -> None:
    if rank != 0:
        return

    lines = [
        f"\n{'=' * 80}",
        f"  msprof Trace — {label}",
        f"{'=' * 80}",
        f"  warmup={num_warmups} (untimed), profile_iters={num_tests} (recorded, continuous session)",
        f"  trace_root -> {trace_root}",
        "  Bundles: <trace_root>/*_ascend_pt/",
        "  After analyse: ASCEND_PROFILER_OUTPUT/{op_statistic,kernel_details,trace_view}.csv/json",
        "  View trace_view.json in MindStudio Insight or chrome://tracing",
    ]
    if kernel_names is not None and kernel_durations is not None:
        for name, duration in zip(kernel_names, kernel_durations):
            lines.append(
                f"  {name} (from op_statistic.csv): {duration * 1e3:.4f} ms",
            )
        if len(kernel_durations) > 1:
            lines.append(f"  Total: {sum(kernel_durations) * 1e3:.4f} ms")
    elif kernel_names is not None:
        lines.append(
            "  (dispatch/combine summary unavailable — inspect op_statistic.csv in trace bundle)",
        )
    lines.append(f"{'=' * 80}\n")
    print("\n".join(lines), flush=True)


def bench_kineto(
    fn: Callable[[], None],
    kernel_names: str | tuple[str, ...],
    num_warmups: int = 50,
    num_tests: int = 100,
    suppress_kineto_output: bool = False,
    trace_path: str | None = None,
) -> tuple[float, ...]:
    """Profile ``fn`` and return average kernel durations in seconds.

    Runs ``num_warmups + num_tests`` iterations in one continuous session inside
    the profiler. Kernel ``dur`` values are averaged over the **last**
    ``num_tests`` matching events per kernel (warmup iterations excluded).
    """
    total_iters = num_warmups + num_tests
    suppress = _stdout_stderr_context(suppress_kineto_output)
    with (
        suppress,
        torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.NPU],
        ) as prof,
    ):
        for _ in range(total_iters):
            fn()
        torch.npu.synchronize()

    is_tuple = isinstance(kernel_names, tuple)
    names = (kernel_names,) if isinstance(kernel_names, str) else kernel_names

    temp_path = Path(tempfile.gettempdir()) / f"trace_{uuid.uuid4().hex}.json"
    try:
        prof.export_chrome_trace(temp_path)
        profile_events = _load_trace_events(temp_path)

        kernel_durations = []
        for kernel_name in names:
            events = [
                event
                for event in profile_events
                if isinstance(event, dict) and _kernel_event_matches(event.get("name"), kernel_name)
            ]
            if not events:
                sample_names = sorted(
                    {name for e in profile_events if isinstance(e, dict) and isinstance(name := e.get("name"), str)}
                )[:8]
                raise AssertionError(
                    f"Kernel '{kernel_name}' not found in chrome trace (available sample names: {sample_names}...)"
                )
            events = sorted(events, key=lambda event: event["ts"])
            timed_events = events[-num_tests:]
            if len(timed_events) < num_tests:
                raise AssertionError(
                    f"Kernel '{kernel_name}': expected {num_tests} timed events, "
                    f"got {len(timed_events)} (total matched={len(events)}, "
                    f"warmup={num_warmups})"
                )
            durations = [event["dur"] / 1e6 for event in timed_events]
            kernel_durations.append(sum(durations) / len(durations))

        if trace_path is not None:
            output_path = Path(trace_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            prof.export_chrome_trace(output_path)
    finally:
        if temp_path.exists():
            os.unlink(temp_path)

    return tuple(kernel_durations) if is_tuple else kernel_durations[0]


def print_wallclock_table(
    *,
    rank: int,
    num_tokens: int,
    hidden: int,
    num_topk: int,
    num_experts: int,
    num_ranks: int,
    zb_dispatch_avg: float,
    zb_combine_avg: float,
    pta_dispatch_avg: float,
    pta_combine_avg: float,
    zb_roundtrip_avg: float,
    pta_roundtrip_avg: float,
    num_warmups: int,
    num_tests: int,
) -> None:
    if rank != 0:
        return

    def _speedup(baseline: float, optimized: float) -> str:
        if optimized <= 0:
            return "N/A"
        return f"{baseline / optimized:.2f}x"

    zb_d, zb_c = zb_dispatch_avg * 1e3, zb_combine_avg * 1e3
    pta_d, pta_c = pta_dispatch_avg * 1e3, pta_combine_avg * 1e3
    zb_rt, pta_rt = zb_roundtrip_avg * 1e3, pta_roundtrip_avg * 1e3
    row = "  {:<28s} {:>14s} {:>14s} {:>12s}"
    sep = "  " + "-" * 72
    total_iters = num_warmups + num_tests
    lines = [
        f"\n{'=' * 80}",
        "  ZB vs PTA MC2 Wall-Clock (NPU events, dispatch/combine only, no GMM)",
        f"{'=' * 80}",
        f"  num_tokens={num_tokens}, hidden={hidden}, num_topk={num_topk}, "
        f"num_experts={num_experts}, world_size={num_ranks}",
        f"  continuous iters={total_iters} (warmup={num_warmups}, timed={num_tests})",
        "  per-stage: dispatch timed with untimed combine; combine timed with untimed dispatch",
        "  timing: NPU events, one device sync after the full loop (no per-iter sync)",
        sep,
        row.format("Stage", "ZB SHMEM (ms)", "PTA V2 (ms)", "ZB speedup"),
        sep,
        row.format("Dispatch", f"{zb_d:.4f}", f"{pta_d:.4f}", _speedup(pta_d, zb_d)),
        row.format("Combine", f"{zb_c:.4f}", f"{pta_c:.4f}", _speedup(pta_c, zb_c)),
        row.format(
            "Sum (dispatch+combine)",
            f"{zb_d + zb_c:.4f}",
            f"{pta_d + pta_c:.4f}",
            _speedup(pta_d + pta_c, zb_d + zb_c),
        ),
        row.format(
            "Round-trip (1 session)",
            f"{zb_rt:.4f}",
            f"{pta_rt:.4f}",
            _speedup(pta_rt, zb_rt),
        ),
        f"{'=' * 80}\n",
    ]
    print("\n".join(lines), flush=True)


def print_kernel_table(
    *,
    rank: int,
    label: str,
    kernel_names: tuple[str, str],
    dispatch_t: float,
    combine_t: float,
    num_warmups: int,
    num_tests: int,
    trace_path: str | None = None,
) -> None:
    if rank != 0:
        return
    lines = [
        f"\n{'=' * 80}",
        f"  Kineto Kernel Timing — {label}",
        f"{'=' * 80}",
        f"  continuous iters={num_warmups + num_tests} "
        f"(warmup={num_warmups}, timed={num_tests}; kernel dur, last N events)",
        f"  {kernel_names[0]}: {dispatch_t * 1e3:.4f} ms",
        f"  {kernel_names[1]}: {combine_t * 1e3:.4f} ms",
        f"  Total: {(dispatch_t + combine_t) * 1e3:.4f} ms",
    ]
    if trace_path is not None:
        lines.append(f"  trace -> {trace_path}")
    lines.append(f"{'=' * 80}\n")
    print("\n".join(lines), flush=True)


def print_pta_baseline_wallclock_table(
    *,
    rank: int,
    num_tokens: int,
    hidden: int,
    num_topk: int,
    num_experts: int,
    num_ranks: int,
    dispatch_avg: float,
    combine_avg: float,
    num_warmups: int,
    num_tests: int,
) -> None:
    if rank != 0:
        return

    dispatch_ms = dispatch_avg * 1e3
    combine_ms = combine_avg * 1e3
    row = "  {:<28s} {:>16s}"
    sep = "  " + "-" * 48
    lines = [
        f"\n{'=' * 80}",
        "  PTA MC2 V2 Baseline Wall-Clock (dispatch/combine only, no GMM)",
        f"{'=' * 80}",
        f"  num_tokens={num_tokens}, hidden={hidden}, num_topk={num_topk}, "
        f"num_experts={num_experts}, world_size={num_ranks}",
        f"  continuous iters={num_warmups + num_tests} (warmup={num_warmups}, timed={num_tests}; NPU events)",
        sep,
        row.format("Stage", "PTA V2 (ms)"),
        sep,
        row.format("MoeDistributeDispatchV2", f"{dispatch_ms:.4f}"),
        row.format("MoeDistributeCombineV2", f"{combine_ms:.4f}"),
        sep,
        row.format("Total", f"{dispatch_ms + combine_ms:.4f}"),
        f"{'=' * 80}\n",
    ]
    print("\n".join(lines), flush=True)


def print_fused_mc2_wallclock_table(
    *,
    rank: int,
    variant: int,
    kernel_label: str,
    num_tokens: int,
    hidden: int,
    num_topk: int,
    num_experts: int,
    num_ranks: int,
    moe_intermediate: int,
    fused_avg: float,
    num_warmups: int,
    num_tests: int,
) -> None:
    if rank != 0:
        return

    fused_ms = fused_avg * 1e3
    row = "  {:<36s} {:>16s}"
    sep = "  " + "-" * 56
    lines = [
        f"\n{'=' * 80}",
        f"  Fused MC2 Baseline (VLLM_ASCEND_ENABLE_FUSED_MC2={variant})",
        f"{'=' * 80}",
        f"  kernel={kernel_label}",
        f"  num_tokens={num_tokens}, hidden={hidden}, moe_intermediate={moe_intermediate}, "
        f"num_topk={num_topk}, num_experts={num_experts}, world_size={num_ranks}",
        f"  continuous iters={num_warmups + num_tests} (warmup={num_warmups}, timed={num_tests}; NPU events)",
        sep,
        row.format("Fused op (dispatch+GMM+combine)", f"{fused_ms:.4f} ms"),
        f"{'=' * 80}\n",
    ]
    print("\n".join(lines), flush=True)


def print_single_kernel_table(
    *,
    rank: int,
    label: str,
    kernel_name: str,
    duration_t: float,
    num_warmups: int,
    num_tests: int,
    trace_path: str | None = None,
) -> None:
    if rank != 0:
        return
    lines = [
        f"\n{'=' * 80}",
        f"  Kineto Kernel Timing — {label}",
        f"{'=' * 80}",
        f"  continuous iters={num_warmups + num_tests} "
        f"(warmup={num_warmups}, timed={num_tests}; kernel dur, last N events)",
        f"  {kernel_name}: {duration_t * 1e3:.4f} ms",
    ]
    if trace_path is not None:
        lines.append(f"  trace -> {trace_path}")
    lines.append(f"{'=' * 80}\n")
    print("\n".join(lines), flush=True)
