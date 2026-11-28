# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Profile paired QSFA graph replays with CANN PipeUtilization or HBM counters."""

import argparse
import csv
import gc
import json
import math
import shlex
import sqlite3
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

import benchmark
import extract_hbm

HBM_SAMPLE_HZ = 50
PMU_MATCH_NS = 1000
PIPES = {
    "aiv": ("vec", "scalar", "mte2", "mte3"),
    "aic": ("mac", "scalar", "mte1", "mte2", "fixpipe"),
}
TOTAL_COUNTERS = {f"{engine}_total_time" for engine in PIPES}
PIPE_COUNTERS = {f"{engine}_{pipe}_time" for engine, pipes in PIPES.items() for pipe in pipes}
GRAPH_SENTINELS = {"MODEL_EXECUTE", "MODEL_WAIT_COMPLETE"}
HBM_FIELDS = (
    "samples",
    "min_counter_samples",
    "forwards_per_s",
    "hbm_read_gb_s",
    "hbm_write_gb_s",
    "hbm_total_gb_s",
    "physical_mbu_pct",
    "hbm_bytes_per_forward_est",
    "window_range_pp",
)
PIPE_FIELDS = (
    "samples",
    "profile_device_span_us",
    "profile_kernel_tasks",
    "profile_tasks_with_pmu",
    "profile_tasks_with_aiv",
    "profile_tasks_with_aic",
    *(f"{engine}_total_us" for engine in PIPES),
    *(f"{engine}_{pipe}_pct" for engine, pipes in PIPES.items() for pipe in pipes),
)


def variants(args):
    return ("native",) if args.native_only else ("native", "candidate")


def ordered_samples(args, batch):
    for sample in range(args.profile_samples):
        order = variants(args)
        if (batch + sample) % 2 == 0:
            order = order[::-1]
        for implementation in order:
            yield sample, implementation, f"QSFA_{args.mode.upper()}_B{batch}_{implementation}_{sample}"


def hbm_window(runner, stream, mstx, label, window_ns, chunk):
    mstx.mark(label + "_begin", stream)
    start_real, start_mono = time.time_ns(), time.monotonic_ns()
    replays = 0
    while time.monotonic_ns() < start_mono + window_ns:
        for _ in range(chunk):
            runner.graph.replay()
        benchmark.torch.npu.synchronize()
        replays += chunk
    end_mono, end_real = time.monotonic_ns(), time.time_ns()
    mstx.mark(label + "_end", stream)
    benchmark.torch.npu.synchronize()
    return {
        "start_realtime_ns": start_real,
        "end_realtime_ns": end_real,
        "start_monotonic_ns": start_mono,
        "end_monotonic_ns": end_mono,
        "forwards": replays * runner.forwards,
    }


@benchmark.torch.inference_mode()
def capture(args):
    import torch_npu

    torch = benchmark.torch
    device_settings = benchmark.initialize_npu(args)
    if "9362" not in device_settings["device"]:
        raise ValueError(f"expected Ascend 910_9362, got {device_settings['device']}")
    mstx = torch_npu.npu.mstx
    max_pages = math.ceil(args.context_length / benchmark.PAGE_SIZE) * max(args.batch)
    storage = benchmark.prepare_storage(max_pages, args.device, args.seed, args.rope_dim)
    functions = {"native": benchmark.load_baseline(args)}
    if not args.native_only:
        functions["candidate"] = benchmark.load_candidate(
            args.candidate, args.block_i[0] if args.block_i else None, args.packed_gather
        )
    forwards = args.forwards if args.mode == "hbm" else 1
    stream = torch.npu.current_stream()
    with (args.output / "markers.jsonl").open("x", encoding="utf-8") as manifest:
        for batch in args.batch:
            inputs, _ = benchmark.make_inputs(
                batch, args.context_length, args.heads, args.topk, args.seed, storage, args.index_pattern
            )
            runners = {
                name: benchmark.Graph(function, inputs, args.scale, args.warmup, forwards)
                for name, function in functions.items()
            }
            if args.mode == "hbm":
                chunks = {
                    name: min(16, max(1, round(10_000 / (runner.time_us() * forwards))))
                    for name, runner in runners.items()
                }
            for sample, implementation, label in ordered_samples(args, batch):
                runner = runners[implementation]
                row = {
                    "batch": batch,
                    "tokens": batch * benchmark.QUERY_TOKENS,
                    "label": label,
                    "sample": sample,
                    "implementation": implementation,
                }
                if args.mode == "pipe":
                    marker = mstx.range_start(label)
                    try:
                        runner.graph.replay()
                        torch.npu.synchronize()
                    finally:
                        mstx.range_end(marker)
                else:
                    runner.graph.replay()
                    torch.npu.synchronize()
                    row.update(
                        hbm_window(runner, stream, mstx, label, round(args.window_s * 1e9), chunks[implementation])
                    )
                manifest.write(json.dumps(row) + "\n")
                manifest.flush()
            # NPU graph objects hold private allocator-pool references.
            del runner, runners, inputs
            gc.collect()
            print(f"profile capture complete for B{batch}", flush=True)


def profile_database(directory):
    databases = list(directory.rglob("msprof_*.db"))
    if not databases:
        captures = list(directory.glob("PROF_*"))
        if len(captures) != 1:
            raise ValueError("msprof did not produce exactly one PROF capture")
        subprocess.run(["msprof", "--export=on", "--type=db", f"--output={captures[0]}"], check=True)
        databases = list(directory.rglob("msprof_*.db"))
    if len(databases) != 1:
        raise ValueError("msprof did not export exactly one SQLite capture")
    return databases[0]


def read_manifest(args):
    with (args.output / "markers.jsonl").open(encoding="utf-8") as source:
        rows = [json.loads(line) for line in source]
    expected = [
        (batch, batch * benchmark.QUERY_TOKENS, sample, implementation, label)
        for batch in args.batch
        for sample, implementation, label in ordered_samples(args, batch)
    ]
    actual = [tuple(row[key] for key in ("batch", "tokens", "sample", "implementation", "label")) for row in rows]
    if actual != expected:
        raise ValueError("the completed capture manifest does not match the requested grid")
    return rows


def read_replay_tasks(db, connection_id):
    sentinels = db.execute(
        "SELECT t.startNs,t.endNs,t.deviceId,t.modelId,t.streamId,k.value "
        "FROM TASK t JOIN STRING_IDS k ON k.id=t.taskType WHERE t.connectionId=? "
        "AND k.value IN ('MODEL_EXECUTE','MODEL_WAIT_COMPLETE') ORDER BY t.startNs",
        (connection_id,),
    ).fetchall()
    if len(sentinels) != 2:
        raise ValueError(f"replay {connection_id} does not have two model sentinels")
    execute, wait = sentinels
    start, _, device, model, _, _ = execute
    _, end, _, _, _, _ = wait
    if (
        execute[2:5] != wait[2:5]
        or (execute[5], wait[5]) != ("MODEL_EXECUTE", "MODEL_WAIT_COMPLETE")
        or start >= end
        or model in (None, -1, 4294967295)
    ):
        raise ValueError(f"replay {connection_id} has inconsistent model sentinels")

    tasks = db.execute(
        "SELECT t.rowid,t.startNs,t.endNs,t.globalTaskId,t.connectionId,k.value,COALESCE(n.value,k.value) "
        "FROM TASK t JOIN STRING_IDS k ON k.id=t.taskType "
        "LEFT JOIN COMPUTE_TASK_INFO c ON c.globalTaskId=t.globalTaskId "
        "LEFT JOIN STRING_IDS n ON n.id=c.name "
        "WHERE t.deviceId=? AND t.modelId=? AND t.startNs>=? AND t.startNs<? ORDER BY t.startNs,t.rowid",
        (device, model, start, end),
    )
    results = {}
    for row_id, begin, stop, task_id, child_id, kind, name in tasks:
        if kind in GRAPH_SENTINELS:
            if child_id != connection_id:
                raise ValueError(f"replay {connection_id} overlaps another replay of model {model}")
            continue
        if not start <= begin <= stop <= end:
            raise ValueError(f"task {row_id} crosses replay {connection_id} bounds")
        item = {"start_ns": begin, "end_ns": stop, "task_id": task_id, "kind": kind, "name": name}
        if results.setdefault(row_id, item) != item:
            raise ValueError(f"ambiguous details for task {row_id} of replay {connection_id}")
    return list(results.values())


def read_pipes(db, task):
    end = task["end_ns"]
    rows = db.execute(
        "SELECT p.timestampNs,n.value,p.value FROM TASK_PMU_INFO p "
        "JOIN STRING_IDS n ON n.id=p.name WHERE p.globalTaskId=? AND p.timestampNs BETWEEN ? AND ?",
        (task["task_id"], end - PMU_MATCH_NS, end + PMU_MATCH_NS),
    )
    groups = defaultdict(dict)
    for timestamp, name, value in rows:
        if name not in TOTAL_COUNTERS and name not in PIPE_COUNTERS:
            continue
        if name in groups[timestamp] or not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid {name} for task {task['task_id']} at {timestamp}")
        groups[timestamp][name] = value
    if not groups:
        return None
    timestamps = sorted(groups, key=lambda timestamp: abs(timestamp - end))
    closest = timestamps[0]
    distance = abs(closest - end)
    if len(timestamps) > 1 and abs(timestamps[1] - end) == distance:
        raise ValueError(f"ambiguous PMU match for task {task['task_id']} at {end}")
    return {"match_error_ns": distance, **groups[closest]}


def pipe_samples(db, manifest):
    marks = db.execute(
        "SELECT m.startNs,m.endNs,s.value FROM MSTX_EVENTS m JOIN STRING_IDS s ON s.id=m.message "
        "WHERE s.value GLOB 'QSFA_PIPE_*' ORDER BY m.startNs"
    ).fetchall()
    if [label for _, _, label in marks] != [row["label"] for row in manifest]:
        raise ValueError("MSTX timing markers do not match the capture manifest")
    if any(before[1] > after[0] for before, after in zip(marks, marks[1:])):
        raise ValueError("MSTX timing markers overlap")
    pmu_table = bool(extract_hbm.table_schema(db, "TASK_PMU_INFO"))
    for (start, stop, label), row in zip(marks, manifest, strict=True):
        replays = db.execute(
            "SELECT DISTINCT c.connectionId FROM CANN_API c JOIN STRING_IDS n ON n.id=c.name "
            "WHERE c.startNs>=? AND c.startNs<? AND n.value='aclmdlRIExecuteAsync'",
            (start, stop),
        ).fetchall()
        if len(replays) != 1:
            raise ValueError(f"{label} has {len(replays)} graph replay API correlations")
        tasks = read_replay_tasks(db, replays[0][0])
        kernels = [task for task in tasks if task["kind"].upper().startswith("KERNEL")]
        if not kernels:
            raise ValueError(f"{label} has no correlated device kernel tasks")
        for task in kernels:
            task["pmu"] = read_pipes(db, task) if pmu_table else None
        report = {
            "profile_device_span_us": (max(task["end_ns"] for task in tasks) - min(task["start_ns"] for task in tasks))
            / 1000,
            "profile_kernel_tasks": len(kernels),
            "profile_tasks_with_pmu": sum(task["pmu"] is not None for task in kernels),
        }
        counters = [task["pmu"] for task in kernels if task["pmu"] is not None]
        for engine, pipes in PIPES.items():
            total_key = f"{engine}_total_time"
            active = [sample for sample in counters if sample.get(total_key, 0) > 0]
            report[f"profile_tasks_with_{engine}"] = len(active)
            if active:
                report[f"{engine}_total_us"] = sum(sample[total_key] for sample in active) / 1000
            for pipe in pipes:
                pipe_key = f"{engine}_{pipe}_time"
                if active and all(pipe_key in sample for sample in active):
                    report[f"{engine}_{pipe}_pct"] = 100 * (
                        sum(sample[pipe_key] for sample in active) / sum(sample[total_key] for sample in active)
                    )
        yield {**row, **report, "kernels": kernels}


def hbm_samples(db, manifest, args):
    options = SimpleNamespace(
        read_type=None,
        write_type=None,
        clock="auto",
        clock_offset_ns=0,
        min_samples=args.min_hbm_samples,
        max_gap_periods=3,
        peak_gb_s=args.peak_hbm_gb_s,
        trim_ms=args.trim_ms,
    )
    description = extract_hbm.hbm_description(db)
    device = extract_hbm.choose_device(description, args.hbm_device_id)
    series = extract_hbm.load_hbm(db, description, device, options)
    units = {unit for unit, _ in series}
    if len(units) != 4:
        raise ValueError(f"the A3 die has {len(units)} HBM counter units, expected four")
    clock, bounds, trim = extract_hbm.align_windows(manifest, None, series, options)
    for index, (row, interval) in enumerate(zip(manifest, bounds, strict=True), start=1):
        sample = extract_hbm.summarize_window(row, index, interval, series, trim, options)
        rate = 1e9 * row["forwards"] / (row["end_monotonic_ns"] - row["start_monotonic_ns"])
        yield {
            **row,
            **sample,
            "device_id": device,
            "clock": clock,
            "forwards_per_s": rate,
            "min_counter_samples": sample["min_series_samples"],
            **{
                f"hbm_{direction}_gb_s": sample[f"{direction}_bytes_per_second"] / 1e9
                for direction in ("read", "write", "total")
            },
            "hbm_bytes_per_forward_est": sample["total_bytes_per_second"] / rate,
        }


def summary_rows(samples, fields, mode):
    groups = defaultdict(list)
    for row in samples:
        groups[row["batch"], row["tokens"], row["implementation"]].append(row)
    for (batch, tokens, implementation), group in sorted(groups.items()):
        record = {"batch": batch, "tokens": tokens, "implementation": implementation, "samples": len(group)}
        for key in fields:
            available = [row[key] for row in group if key in row]
            if len(available) >= len(group) // 2 + 1:
                record[key] = statistics.median(available)
        if mode == "hbm":
            mbu = [row["physical_mbu_pct"] for row in group]
            record["min_counter_samples"] = min(row["min_counter_samples"] for row in group)
            record["window_range_pp"] = max(mbu) - min(mbu)
        yield record


def extract(args):
    manifest = read_manifest(args)
    path = args.database or profile_database(args.output / "profiler")
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        rows = list(pipe_samples(db, manifest) if args.mode == "pipe" else hbm_samples(db, manifest, args))
    fields = PIPE_FIELDS if args.mode == "pipe" else HBM_FIELDS
    with (args.output / f"{args.mode}.csv").open("x", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=("batch", "tokens", "implementation", *fields))
        writer.writeheader()
        writer.writerows(summary_rows(rows, fields, args.mode))
    with (args.output / "profile_samples.jsonl").open("x", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, allow_nan=False) + "\n")


def run(args):
    if args.extract_only:
        extract(args)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    options = {key: value for key, value in vars(args).items() if key not in {"output", "device", "database"}}
    (args.output / "profile_settings.json").write_text(json.dumps(options, indent=2) + "\n", encoding="utf-8")
    command = shlex.join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--capture"])
    metrics = (
        ["--task-time=on", "--ascendcl=on", "--aicpu=off", "--ai-core=on", "--aic-metrics=PipeUtilization"]
        if args.mode == "pipe"
        else [
            "--task-time=off",
            "--ascendcl=off",
            "--ai-core=off",
            "--sys-hardware-mem=on",
            f"--sys-hardware-mem-freq={HBM_SAMPLE_HZ}",
        ]
    )
    subprocess.run(
        [
            "msprof",
            f"--output={args.output.resolve() / 'profiler'}",
            f"--application={command}",
            "--msproftx=on",
            *metrics,
        ],
        check=True,
    )
    extract(args)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pipe", "hbm"), required=True)
    parser.add_argument("--profile-samples", type=int, help="per implementation; defaults to 5 for pipe or 2 for HBM")
    parser.add_argument("--window-s", type=float, default=0.45, help="wall time of each HBM replay window")
    parser.add_argument("--trim-ms", type=float, default=50, help="discard both edges of each HBM window")
    parser.add_argument("--min-hbm-samples", type=int, default=10)
    parser.add_argument("--hbm-device-id", type=int, help="physical profiler die ID if DB has multiple devices")
    parser.add_argument("--database", type=Path, help="read this exported database when extracting")
    parser.add_argument("--extract-only", action="store_true", help="re-extract an existing capture with the same grid")
    parser.add_argument("--capture", action="store_true", help=argparse.SUPPRESS)
    args = benchmark.arguments(parser)
    if args.profile_samples is None:
        args.profile_samples = 5 if args.mode == "pipe" else 2
    if (
        args.profile_samples < 1
        or args.min_hbm_samples < 2
        or not math.isfinite(args.window_s)
        or not math.isfinite(args.trim_ms)
        or not 0 <= args.trim_ms * 2 < args.window_s * 1000
    ):
        parser.error("provide positive sample counts and finite HBM windows longer than two edge trims")
    if args.cpu_fixture or args.accuracy_only or len(args.block_i or []) > 1:
        parser.error("profiling requires an NPU and at most one candidate tile per run")
    if args.database is not None and not args.extract_only:
        parser.error("--database requires --extract-only")
    return args


if __name__ == "__main__":
    options = arguments()
    if options.capture:
        capture(options)
    else:
        run(options)
