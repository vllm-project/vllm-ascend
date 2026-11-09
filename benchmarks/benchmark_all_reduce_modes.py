# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare requested AI_CPU/AIV HCCL AllReduce modes in separate torchrun jobs.

Only torch and torch_npu are required on the benchmark host. See the companion
Markdown file for examples, timing semantics, and HCCL fallback limitations.
"""

import argparse
import csv
import json
import math
import os
import shlex
import socket
import statistics
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

_MODES = ("AI_CPU", "AIV")
_DTYPE_BYTES = {"bfloat16": 2, "float16": 2, "float32": 4}
_DEFAULT_SIZES = ("1K", "4K", "16K", "64K", "256K", "1M", "4M", "16M", "64M")
_TRACKED_ENV = (
    "HCCL_OP_EXPANSION_MODE",
    "HCCL_BUFFSIZE",
    "HCCL_ALGO",
    "HCCL_DETERMINISTIC",
    "HCCL_OP_RETRY_ENABLE",
    "ASCEND_RT_VISIBLE_DEVICES",
    "ASCEND_VISIBLE_DEVICES",
    "ASCEND_LAUNCH_BLOCKING",
    "TASK_QUEUE_ENABLE",
    "TORCH_DISTRIBUTED_DEBUG",
)
_MIB = 1024**2
_MODE_NOTE = "Requested HCCL mode; the actual engine is not verified and HCCL may fall back."


def parse_size(value: str) -> int:
    """Parse a payload size in bytes; K/M/G and KiB/MiB/GiB are binary units."""
    number = value.strip().upper()
    multiplier = 1
    for suffix, factor in (
        ("GIB", 1024**3),
        ("MIB", 1024**2),
        ("KIB", 1024),
        ("GB", 1024**3),
        ("MB", 1024**2),
        ("KB", 1024),
        ("G", 1024**3),
        ("M", 1024**2),
        ("K", 1024),
        ("B", 1),
    ):
        if number.endswith(suffix):
            number = number[: -len(suffix)].strip()
            multiplier = factor
            break
    if not number.isascii() or not number.isdecimal() or int(number) <= 0:
        raise argparse.ArgumentTypeError(f"Invalid size {value!r}; use bytes, 4K, 1MiB, etc.")
    return int(number) * multiplier


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("both", *_MODES), default="both")
    parser.add_argument("--nproc-per-node", type=int, default=2, help="Local launcher NPU count (default: 2).")
    parser.add_argument("--dtype", choices=_DTYPE_BYTES, default="bfloat16")
    parser.add_argument("--sizes", type=parse_size, nargs="+", default=[parse_size(size) for size in _DEFAULT_SIZES])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100, help="Samples per round, per size.")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument(
        "--timing",
        choices=("event", "wall"),
        default="event",
        help="event: NPU event interval; wall: synchronized single-call host latency.",
    )
    parser.add_argument("--hccl-buffsize-mib", type=int, help="Set the same HCCL_BUFFSIZE for both jobs.")
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--output-dir", type=Path, help="Default: all_reduce_results/<timestamp>.")
    parser.add_argument(
        "--profile", action="store_true", help="Capture separate torch_npu profiler traces after timing."
    )
    parser.add_argument("--profile-steps", type=int, default=5, help="Profiled AllReduce calls per size (default: 5).")
    parser.add_argument(
        "--profile-with-stack", action="store_true", help="Collect Python/module call stacks in traces."
    )
    parser.add_argument("--profile-memory", action="store_true", help="Collect memory information in traces.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print launch commands without importing torch or running."
    )
    args = parser.parse_args(argv)
    for name in ("iterations", "rounds", "timeout_seconds", "profile_steps"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.nproc_per_node < 2:
        parser.error("--nproc-per-node must be at least 2; a rank-1 AllReduce is not a communication benchmark")
    if args.warmup < 0:
        parser.error("--warmup must be nonnegative")
    if args.hccl_buffsize_mib is not None and args.hccl_buffsize_mib < 1:
        parser.error("--hccl-buffsize-mib must be positive")
    if len(set(args.sizes)) != len(args.sizes):
        parser.error("--sizes must not contain duplicate payload sizes")
    if any(size % _DTYPE_BYTES[args.dtype] for size in args.sizes):
        parser.error(f"All payload sizes must be divisible by {_DTYPE_BYTES[args.dtype]} for {args.dtype}")
    return args


def summarize(samples_us: list[float]) -> dict:
    if not samples_us or any(not math.isfinite(value) or value <= 0 for value in samples_us):
        raise ValueError("Timing samples must be finite and positive; increase message size if event timing is zero.")
    ordered = sorted(samples_us)
    return {
        "median_us": statistics.median(ordered),
        "mean_us": statistics.mean(ordered),
        "p95_us": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "min_us": ordered[0],
        "max_us": ordered[-1],
    }


def slowest_rank_samples(samples_by_rank: list[list[float]]) -> list[float]:
    if not samples_by_rank or not samples_by_rank[0]:
        raise ValueError("No timing samples collected")
    if any(len(values) != len(samples_by_rank[0]) for values in samples_by_rank):
        raise ValueError("Ranks returned different sample counts")
    for values in samples_by_rank:
        summarize(values)
    return [max(values) for values in zip(*samples_by_rank, strict=True)]


def comparison_rows(reports: dict[str, dict]) -> list[dict]:
    baseline, candidate = reports["AI_CPU"], reports["AIV"]
    for key in ("world_size", "dtype", "timing", "warmup", "iterations", "rounds"):
        if baseline[key] != candidate[key]:
            raise ValueError(f"Cannot compare runs with different {key}")
    ai_cpu_rows = {row["payload_bytes"]: row for row in baseline["results"]}
    aiv_rows = {row["payload_bytes"]: row for row in candidate["results"]}
    if ai_cpu_rows.keys() != aiv_rows.keys():
        raise ValueError("Runs measured different payload sizes")
    rows = []
    for size, base in ai_cpu_rows.items():
        aiv = aiv_rows[size]
        rows.append(
            {
                "payload_bytes": size,
                "numel": base["numel"],
                "ai_cpu_mean_us": base["mean_us"],
                "ai_cpu_median_us": base["median_us"],
                "ai_cpu_p95_us": base["p95_us"],
                "aiv_mean_us": aiv["mean_us"],
                "aiv_median_us": aiv["median_us"],
                "aiv_p95_us": aiv["p95_us"],
                "speedup_ai_cpu_over_aiv": base["median_us"] / aiv["median_us"],
                "aiv_latency_reduction_pct": (1 - aiv["median_us"] / base["median_us"]) * 100,
            }
        )
    return rows


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _launch_command(args: argparse.Namespace, mode: str, output_dir: Path) -> tuple[list[str], dict[str, str]]:
    env = os.environ.copy()
    # External HCCL settings belong to this standalone process launcher; avoid
    # importing vLLM (and initializing a runtime) just to switch benchmark modes.
    env["HCCL_OP_EXPANSION_MODE"] = mode
    if args.hccl_buffsize_mib is not None:
        env["HCCL_BUFFSIZE"] = str(args.hccl_buffsize_mib)
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node",
        str(args.nproc_per_node),
        "--max-restarts",
        "0",
        str(Path(__file__).resolve()),
        "--mode",
        mode,
        "--dtype",
        args.dtype,
        "--sizes",
        *(str(size) for size in args.sizes),
        "--warmup",
        str(args.warmup),
        "--iterations",
        str(args.iterations),
        "--rounds",
        str(args.rounds),
        "--timing",
        args.timing,
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--output-dir",
        str(output_dir),
    ]
    if args.hccl_buffsize_mib is not None:
        command.extend(("--hccl-buffsize-mib", str(args.hccl_buffsize_mib)))
    if args.profile:
        command.extend(("--profile", "--profile-steps", str(args.profile_steps)))
        if args.profile_with_stack:
            command.append("--profile-with-stack")
        if args.profile_memory:
            command.append("--profile-memory")
    return command, env


def _launch(args: argparse.Namespace) -> None:
    output_dir = (args.output_dir or Path("all_reduce_results") / datetime.now().strftime("%Y%m%d_%H%M%S_%f")).resolve()
    modes = _MODES if args.mode == "both" else (args.mode,)
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        for name in (*(f"{mode}.json" for mode in modes), "comparison.json", "comparison.csv"):
            if (output_dir / name).exists():
                raise FileExistsError(f"Refusing to replace {output_dir / name}; choose another --output-dir.")
        if args.profile:
            for mode in modes:
                if (output_dir / "profiles" / mode).exists():
                    raise FileExistsError(f"Profiling output already exists for {mode}; choose another --output-dir.")
    reports = {}
    for mode in modes:
        command, env = _launch_command(args, mode, output_dir)
        setting = f"HCCL_OP_EXPANSION_MODE={mode}"
        if args.hccl_buffsize_mib is not None:
            setting += f" HCCL_BUFFSIZE={args.hccl_buffsize_mib}"
        print(f"\n{setting} {shlex.join(command)}", flush=True)
        if args.dry_run:
            continue
        # A fresh interpreter/communicator is essential: HCCL can cache env
        # settings, so changing os.environ between collectives is not a comparison.
        subprocess.run(command, env=env, check=True)
        reports[mode] = json.loads((output_dir / f"{mode}.json").read_text(encoding="utf-8"))
    if args.dry_run:
        return
    if args.mode == "both":
        rows = comparison_rows(reports)
        _write_json(output_dir / "comparison.json", {"mode_note": _MODE_NOTE, "runs": reports, "comparison": rows})
        with (output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nRequested modes; {args.timing} timing, avg/median/p95 of per-sample slowest rank (us)")
        print(
            f"{'Payload':>12} {'AI_CPU avg':>12} {'AI_CPU med':>12} {'AI_CPU p95':>12} "
            f"{'AIV avg':>12} {'AIV med':>12} {'AIV p95':>12} {'Speedup(med)':>12}"
        )
        for row in rows:
            print(
                f"{row['payload_bytes']:>10} B {row['ai_cpu_mean_us']:>12.3f} "
                f"{row['ai_cpu_median_us']:>12.3f} {row['ai_cpu_p95_us']:>12.3f} "
                f"{row['aiv_mean_us']:>12.3f} {row['aiv_median_us']:>12.3f} {row['aiv_p95_us']:>12.3f} "
                f"{row['speedup_ai_cpu_over_aiv']:>11.3f}x"
            )
    print(f"\nResults: {output_dir}\n{_MODE_NOTE}", flush=True)


def _measure(source, working, hccl_group, args: argparse.Namespace) -> list[float]:
    # Lazy imports keep launcher/help/dry-run usable without torch or an NPU.
    import torch
    import torch.distributed as dist

    events = []
    if args.timing == "event":
        events = [
            (torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)) for _ in range(args.iterations)
        ]
        # Materialize lazy event handles outside the measurement.
        for start, end in events:
            start.record()
            end.record()
    torch.npu.synchronize()
    dist.barrier()  # Default group is CPU/Gloo; no extra HCCL collective.
    samples_us = []
    for index in range(args.iterations):
        # In-place SUM would otherwise grow the input exponentially. Reset on
        # the same stream, outside the measured interval, for both modes.
        working.copy_(source)
        if args.timing == "event":
            start, end = events[index]
            start.record()
            # Default synchronous semantics return the completion dependency
            # to the current stream before the end marker is recorded.
            dist.all_reduce(working, op=dist.ReduceOp.SUM, group=hccl_group, async_op=False)
            end.record()
        else:
            torch.npu.synchronize()  # Exclude input reset from wall timing.
            start_wall = time.perf_counter_ns()
            dist.all_reduce(working, op=dist.ReduceOp.SUM, group=hccl_group, async_op=False)
            torch.npu.synchronize()
            samples_us.append((time.perf_counter_ns() - start_wall) / 1000)
    torch.npu.synchronize()
    if args.timing == "event":
        samples_us = [start.elapsed_time(end) * 1000 for start, end in events]
    return samples_us


def _check_result(working, expected: float, world_size: int, dtype: str) -> None:
    import torch
    import torch.distributed as dist

    rtol = {"bfloat16": 0.01, "float16": 0.001, "float32": 1e-5}[dtype]
    reference = torch.tensor(expected, dtype=working.dtype, device=working.device)
    # Outside all timing intervals. Check every element and propagate failures
    # over Gloo, so every rank exits instead of peers entering the next size.
    correct = bool(torch.isclose(working, reference, rtol=rtol, atol=0).all().item())
    checks = [None] * world_size
    dist.all_gather_object(checks, correct)
    failures = [rank for rank, passed in enumerate(checks) if not passed]
    if failures:
        raise AssertionError(f"AllReduce SUM check failed on ranks {failures}; expected {expected}.")


def _create_profiler(trace_dir: Path, worker_name: str, args: argparse.Namespace):
    import torch_npu

    # Match vllm_ascend/profiler/torch_npu_profiler.py without importing vLLM.
    # Stack/memory collection are opt-in, as they add substantial overhead.
    profiler = torch_npu.profiler
    experimental_config = profiler._ExperimentalConfig(
        export_type=profiler.ExportType.Text,
        profiler_level=profiler.ProfilerLevel.Level1,
        msprof_tx=False,
        aic_metrics=profiler.AiCMetrics.PipeUtilization,
        l2_cache=False,
        op_attr=False,
        data_simplification=True,
        record_op_args=False,
        gc_detect_threshold=None,
    )
    return profiler.profile(
        activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU],
        with_stack=False,
        with_modules=args.profile_with_stack,
        profile_memory=args.profile_memory,
        experimental_config=experimental_config,
        on_trace_ready=profiler.tensorboard_trace_handler(
            str(trace_dir), worker_name=worker_name, analyse_flag=True, async_mode=False
        ),
    )


def _capture_profiles(args: argparse.Namespace, hccl_group, rank: int, world_size: int, output_dir: Path) -> str:
    import torch
    import torch.distributed as dist

    profile_dir = output_dir / "profiles" / args.mode / f"rank_{rank}"
    profile_dir.mkdir(parents=True, exist_ok=False)
    worker_name = f"{socket.gethostname()}_rank{rank}"
    for size in args.sizes:
        numel = size // _DTYPE_BYTES[args.dtype]
        source = torch.full(
            (numel,), float(rank + 1), dtype=getattr(torch, args.dtype), device=f"npu:{torch.npu.current_device()}"
        )
        working = torch.empty_like(source)
        for _ in range(max(1, args.warmup)):
            working.copy_(source)
            dist.all_reduce(working, op=dist.ReduceOp.SUM, group=hccl_group)
        torch.npu.synchronize()
        dist.barrier()  # Coordinate capture over Gloo, outside the trace.
        trace_dir = profile_dir / f"{size}B"
        with _create_profiler(trace_dir, worker_name, args) as profiler:
            for _ in range(args.profile_steps):
                with torch.profiler.record_function("benchmark_input_reset"):
                    working.copy_(source)
                with torch.profiler.record_function(f"all_reduce/{args.mode}/{size}B"):
                    dist.all_reduce(working, op=dist.ReduceOp.SUM, group=hccl_group, async_op=False)
                profiler.step()
            # Drain the NPU before stopping/exporting the capture.
            torch.npu.synchronize()
        _check_result(working, world_size * (world_size + 1) / 2, world_size, args.dtype)
        if rank == 0:
            print(f"Profiled {args.mode} {size} B: {args.profile_steps} calls per rank", flush=True)
        del working, source
    return str(profile_dir)


def _worker(args: argparse.Namespace) -> None:
    if args.mode == "both":
        raise ValueError("Under torchrun use --mode AI_CPU or --mode AIV; plain python launches both automatically.")
    # Set before torch_npu import, communicator creation, or the first collective.
    os.environ["HCCL_OP_EXPANSION_MODE"] = args.mode
    if args.hccl_buffsize_mib is not None:
        os.environ["HCCL_BUFFSIZE"] = str(args.hccl_buffsize_mib)
    import torch
    import torch.distributed as dist
    import torch_npu

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size < 2:
        raise ValueError("At least two ranks are required")
    if not torch.npu.is_available():
        raise RuntimeError("No Ascend NPU is available")
    if local_rank >= torch.npu.device_count():
        raise ValueError(f"LOCAL_RANK={local_rank} exceeds the visible NPU count")
    torch.npu.set_device(local_rank)
    device = torch.device(f"npu:{local_rank}")
    timeout = timedelta(seconds=args.timeout_seconds)
    dist.init_process_group(backend="gloo", timeout=timeout)
    # Only one HCCL group; control barriers and gathering timing samples are CPU-only.
    hccl_group = dist.new_group(ranks=list(range(world_size)), backend="hccl", timeout=timeout)
    local_metadata = {
        "rank": rank,
        "local_rank": local_rank,
        "hostname": socket.gethostname(),
        "device": torch.npu.get_device_name(local_rank),
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "environment": {name: os.environ.get(name) for name in _TRACKED_ENV},
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }
    metadata = [None] * world_size
    dist.all_gather_object(metadata, local_metadata)
    # Rank 0 chooses one timestamp/path for externally launched workers too.
    output_paths = [
        str(args.output_dir or Path("all_reduce_results") / datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        if rank == 0
        else None
    ]
    dist.broadcast_object_list(output_paths, src=0)
    output_dir = Path(output_paths[0])
    for item in metadata:
        if item["environment"] != metadata[0]["environment"]:
            # Visibility masks can legitimately differ across nodes, but HCCL
            # mode/algorithm/buffer/determinism settings must match.
            for name in _TRACKED_ENV:
                if name.startswith("HCCL_") and item["environment"][name] != metadata[0]["environment"][name]:
                    raise ValueError(f"Ranks have different {name} settings")
    report = {
        "requested_mode": args.mode,
        "mode_note": _MODE_NOTE,
        "world_size": world_size,
        "dtype": args.dtype,
        "timing": args.timing,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "rounds": args.rounds,
        "rank_metadata": metadata,
        "profiling": None,
        "results": [],
    }
    if rank == 0:
        print(f"\n{args.mode}: {world_size} ranks, {args.dtype}, {args.timing} timing\n{_MODE_NOTE}", flush=True)
    for size in args.sizes:
        numel = size // _DTYPE_BYTES[args.dtype]
        source = torch.full((numel,), float(rank + 1), dtype=getattr(torch, args.dtype), device=device)
        working = torch.empty_like(source)
        expected = world_size * (world_size + 1) / 2
        # The first collective initializes/warm-starts the HCCL communicator even
        # when --warmup=0, and validates the mode before measuring it.
        for _ in range(max(1, args.warmup)):
            working.copy_(source)
            dist.all_reduce(working, op=dist.ReduceOp.SUM, group=hccl_group)
        torch.npu.synchronize()
        _check_result(working, expected, world_size, args.dtype)
        all_samples = []
        round_medians = []
        for _ in range(args.rounds):
            local_samples = _measure(source, working, hccl_group, args)
            gathered = [None] * world_size
            dist.all_gather_object(gathered, local_samples)
            samples = slowest_rank_samples(gathered)
            all_samples.extend(samples)
            round_medians.append(statistics.median(samples))
            _check_result(working, expected, world_size, args.dtype)
        row = {
            "payload_bytes": size,
            "payload_mib": size / _MIB,
            "numel": numel,
            "correct": True,
            **summarize(all_samples),
            "round_medians_us": round_medians,
            "slowest_rank_samples_us": all_samples,
        }
        report["results"].append(row)
        if rank == 0:
            print(
                f"{args.mode:>6} {size:>10} B  avg={row['mean_us']:.3f} us  "
                f"median={row['median_us']:.3f} us  p95={row['p95_us']:.3f} us  PASS",
                flush=True,
            )
        del working, source
    if args.profile:
        # All timing samples have already been collected. Do not feed profiler
        # calls into the benchmark statistics or include export/analysis time.
        local_profile_dir = _capture_profiles(args, hccl_group, rank, world_size, output_dir)
        profile_directories = [None] * world_size
        dist.all_gather_object(profile_directories, local_profile_dir)
        report["profiling"] = {
            "steps_per_size": args.profile_steps,
            "rank_directories": profile_directories,
            "with_stack": args.profile_with_stack,
            "profile_memory": args.profile_memory,
        }
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{args.mode}.json"
        if output_path.exists():
            raise FileExistsError(f"Refusing to replace {output_path}")
        _write_json(output_path, report)
        print(f"Saved {output_path}", flush=True)
    # Successful runs clean up explicitly. On an error, torchrun terminates the
    # remaining workers; do not add a final barrier that could hide the failure.
    dist.destroy_process_group(hccl_group)
    dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        if args.dry_run:
            raise ValueError("Use --dry-run with plain python, outside torchrun")
        _worker(args)
    else:
        _launch(args)


if __name__ == "__main__":
    main()
