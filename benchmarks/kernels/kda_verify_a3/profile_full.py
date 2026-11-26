# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capture and extract msprof device timings for complete A3 KDA forwards.

Run ``capture`` under msprof with --task-time=on --msproftx=on, then use
``extract`` on its msprof_*.db and the capture manifest. Pass
``capture --execution graph`` to profile one forward per ACL Graph replay.
The primary metric spans the first through last device tasks, including gaps.
"""

import argparse
import bisect
import importlib
import json
import re
import sqlite3
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import torch
from compare_full import (
    MIDDLE_VARIANTS,
    GraphRunner,
    TorchReference,
    accepted_cases,
    accuracy_metrics,
    baseline_function,
    candidate_function,
    host_result,
    integer_sweep,
    reset,
    validate_runner,
)
from full_reference import Config, make_inputs, make_rollback

MARKER_MODES = "|".join(("native", "torch", "candidate", *MIDDLE_VARIANTS))
MARKER_PATTERN = re.compile(rf"A3KDA_B(\d+)_L(\d+)_({MARKER_MODES})_(\d+)")
A5_PROFILER_SOURCE = (
    "https://github.com/Biuapha/vllm-ascend/blob/81b44ab3e73d4ae89c564ec400d1eebafc1efd7b/"
    "benchmarks/kernels/kda_verify/optimized_replay/extract_whole.py"
)
GRAPH_PROFILE_MODE = "graph-replay"
GRAPH_SENTINELS = {"MODEL_EXECUTE", "MODEL_WAIT_COMPLETE"}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def selected_parameters(args, batch):
    if args.selection is None:
        return {"row_tile": args.row_tile, "projection_pad": args.projection_pad}
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    matches = [case for case in selection["cases"] if case["batch"] == batch and case["tokens"] == args.tokens]
    if len(matches) != 1 or matches[0].get("status") != "passed":
        raise ValueError(f"selection must contain exactly one passing B={batch}, L={args.tokens} case")
    case = matches[0]
    for name in ("heads", "hidden"):
        if case["config"][name] != getattr(args, name):
            raise ValueError(f"selection {name} does not match this profiler invocation")
    return case["selected"]


def variant_parameters(args, parameters):
    variants = {}
    for mode in args.variant:
        options = {**parameters, **MIDDLE_VARIANTS[mode]}
        if options.get("cv_slots"):
            options["core_count"] = args.core_count
        variants[mode] = options
    return variants


def validate_eager(args, config, accepted, baseline, candidates, record):
    validation_cases = accepted_cases(config, accepted.device) if args.variant else {"one": accepted}
    if args.variant:
        record["accuracy_by_variant"] = {mode: {} for mode in candidates}
    for case_name, accepted_lengths in validation_cases.items():
        reset(baseline)
        expected = host_result(baseline(accepted_lengths))
        for mode, candidate in candidates.items():
            reset(candidate)
            accuracy = accuracy_metrics(candidate(accepted_lengths), expected, args.protocol)
            if args.variant:
                record["accuracy_by_variant"][mode][case_name] = accuracy
            else:
                record["accuracy"] = accuracy
            if not accuracy["passed"]:
                raise AssertionError(
                    f"B={config.batch}: {mode} failed {args.protocol} accuracy for accepted={case_name}"
                )


def prepare_graph_runners(args, config, inputs, snapshots, accepted, functions, record):
    runners = {
        mode: GraphRunner(function, accepted, repeat=1, warmup=args.warmup) for mode, function in functions.items()
    }
    baseline = functions[args.baseline]
    record["accuracy_by_variant"] = {}
    for mode, runner in runners.items():
        reference = TorchReference(config, inputs) if mode == args.baseline else baseline
        accuracy = validate_runner(runner, reference, config, inputs, args.protocol, snapshots)
        record["accuracy_by_variant"][mode] = accuracy
        if not accuracy["passed"]:
            raise AssertionError(f"B={config.batch}: {mode} failed {args.protocol} validation")
        runner.warm_replays(args.warmup)
        torch.npu.synchronize()
    return runners


def capture_batch(args, batch, document, torch_npu):
    """Validate forwards, then mark individual timings with resets outside."""
    config = Config(batch=batch, tokens=args.tokens, heads=args.heads, hidden=args.hidden)
    inputs = make_inputs(config, device="npu", seed=args.seed + batch)
    inputs["rollback"] = make_rollback(config, inputs)
    snapshots = {name: inputs[name].clone() for name in ("state", "history", "rollback")}
    parameters = selected_parameters(args, batch)
    variants = variant_parameters(args, parameters) if args.variant else {"candidate": parameters}
    candidates = {mode: candidate_function(config, inputs, **options) for mode, options in variants.items()}
    baseline = baseline_function(args.baseline, config, inputs)
    functions = {args.baseline: baseline, **candidates}
    accepted = torch.ones(batch, dtype=torch.int32, device="npu")
    record = {
        "config": asdict(config),
        "seed": args.seed + batch,
        "parameters": parameters,
        "markers": [],
        "implementations": {
            mode: type(function).__module__ + "." + type(function).__name__ for mode, function in functions.items()
        },
    }
    document["cases"].append(record)
    if args.variant:
        record["variant_parameters"] = variants

    runners = None
    if args.execution == "graph":
        runners = prepare_graph_runners(args, config, inputs, snapshots, accepted, functions, record)
        stream = torch.npu.current_stream()
    else:
        validate_eager(args, config, accepted, baseline, candidates, record)
        for function in functions.values():
            for _ in range(args.warmup):
                reset(function)
                function(accepted)
            torch.npu.synchronize()

    # Alternate AB/BA order while retaining exactly 11 invocations per backend
    # by default. Compilation, reset, validation and warmup have no markers.
    mstx = torch_npu.npu.mstx
    for sample in range(args.samples):
        order = tuple(functions) if sample % 2 == 0 else tuple(reversed(functions))
        for mode in order:
            function = functions[mode]
            reset(function)
            torch.npu.synchronize()
            label = f"A3KDA_B{batch}_L{args.tokens}_{mode}_{sample}"
            mark = mstx.range_start(label)
            try:
                if runners is None:
                    function(accepted)
                else:
                    # Keep stream markers for inspecting the timeline. Extraction uses the model sentinels.
                    mstx.mark(f"A3KDADEV_{label}_begin", stream)
                    runners[mode].graph.replay()
                    mstx.mark(f"A3KDADEV_{label}_end", stream)
                torch.npu.synchronize()
            finally:
                mstx.range_end(mark)
            record["markers"].append(label)
    for name, original in snapshots.items():
        if not torch.equal(inputs[name], original):
            raise AssertionError(f"B={batch}: input {name} was modified")
    for mode, function in functions.items():
        if hasattr(function, "checkpoint_unchanged") and not function.checkpoint_unchanged():
            raise AssertionError(f"B={batch}: {mode} mutated its checkpoint")
    record["status"] = "captured"


def capture(args):
    torch_npu = importlib.import_module("torch_npu")
    name = torch.npu.get_device_name(torch.npu.current_device())
    if "910_93" not in name and "910C" not in name:
        raise RuntimeError(f"expected A3 (910_93 or 910C), got {name}")
    document = {
        "schema_version": 1,
        "device": name,
        "a5_profiler_source": A5_PROFILER_SOURCE,
        "baseline": args.baseline,
        "fixture": "A5 synthetic make_inputs plus rollback rows; not the captured serving trace.pt fixture",
        "scope": "one eager complete hidden-state-to-output forward; all projections, preparation and cache writes",
        "excluded": ["compilation", "warmup", "cache reset", "accuracy checks"],
        "execution": "eager split launches; no graph replay; synchronize before marker and before marker end",
        "samples_per_backend": args.samples,
        "warmup": args.warmup,
        "accepted": 1,
        "selection_source": str(args.selection) if args.selection else "fixed command-line parameters",
        "protocol": args.protocol,
        "cases": [],
    }
    if args.variant:
        document["candidate_modes"] = args.variant
    if args.execution == "graph":
        document.update(
            {
                "profile_mode": GRAPH_PROFILE_MODE,
                "scope": "one complete hidden-state-to-output forward in a captured ACL Graph",
                "execution": "one GraphRunner repeat=1 replay; same-stream MSTX points",
                "excluded": ["graph construction", *document["excluded"]],
            }
        )
    try:
        with torch.inference_mode():
            for batch in args.batches:
                capture_batch(args, batch, document, torch_npu)
                write_json(args.out, document)
                names = ", ".join((args.baseline, *args.variant)) if args.variant else f"{args.baseline}, candidate"
                print(f"B={batch}: captured {args.samples} forwards per mode ({names})", flush=True)
    except Exception as error:
        document["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        write_json(args.out, document)


def database_path(path):
    if path.is_file():
        return path
    candidates = sorted(path.rglob("msprof_*.db"))
    if len(candidates) != 1:
        raise ValueError(f"expected one msprof_*.db; found {len(candidates)}; pass the exact database path")
    return candidates[0]


def union_ns(intervals):
    elapsed, end = 0, None
    for begin, stop in sorted(intervals):
        elapsed += max(0, stop - max(begin, end)) if end is not None else stop - begin
        end = max(stop, end) if end is not None else stop
    return elapsed


def task_metrics(tasks):
    """Preserve spans, busy time and overlaps as distinct device measurements."""
    if not tasks:
        raise ValueError("a marked invocation has no correlated device tasks")
    all_intervals = [(task["start_ns"], task["end_ns"]) for task in tasks]
    kernels = [task for task in tasks if task["task_type"].upper().startswith("KERNEL")]
    if not kernels:
        raise ValueError("a marked invocation has no compute kernels")
    kernel_intervals = [(task["start_ns"], task["end_ns"]) for task in kernels]
    first = min(start for start, _ in all_intervals)
    last = max(end for _, end in all_intervals)
    busy = union_ns(all_intervals)
    return {
        "device_elapsed_us": (last - first) / 1000,
        "device_busy_union_us": busy / 1000,
        "device_gap_us": (last - first - busy) / 1000,
        "kernel_span_us": (max(end for _, end in kernel_intervals) - min(start for start, _ in kernel_intervals))
        / 1000,
        "kernel_busy_union_us": union_ns(kernel_intervals) / 1000,
        "kernel_duration_sum_us": sum(end - start for start, end in kernel_intervals) / 1000,
        "kernel_count": len(kernels),
        "device_task_count": len(tasks),
        "kernel_names": dict(Counter(task["name"] for task in kernels)),
        "task_types": dict(Counter(task["task_type"] for task in tasks)),
        "tasks": sorted(tasks, key=lambda task: task["start_ns"]),
    }


def read_markers(connection):
    marks = connection.execute(
        "SELECT m.startNs,m.endNs,s.value FROM MSTX_EVENTS m "
        "JOIN STRING_IDS s ON s.id=m.message WHERE s.value GLOB 'A3KDA_*' ORDER BY m.startNs"
    ).fetchall()
    if not marks or any(MARKER_PATTERN.fullmatch(mark[2]) is None for mark in marks):
        raise ValueError("no valid A3KDA markers; enable msprof --msproftx=on")
    if len({mark[2] for mark in marks}) != len(marks):
        raise ValueError("duplicate marker names; use a fresh profiler directory for every capture")
    if any(start >= end for start, end, _ in marks):
        raise ValueError("invalid marker duration")
    if any(left[1] > right[0] for left, right in zip(marks, marks[1:])):
        raise ValueError("overlapping markers cannot be assigned unambiguously")
    return marks


def read_groups(connection):
    """Use host API correlation; host and device timestamps need not share a clock."""
    marks = read_markers(connection)
    starts = [mark[0] for mark in marks]
    groups = {mark[2]: {} for mark in marks}
    # Include every task correlated to an API inside a marker, not only compute
    # tasks: copies can contribute to the first-to-last full-forward span too.
    rows = connection.execute(
        "SELECT p.startNs,t.startNs,t.endNs,t.globalTaskId,COALESCE(n.value,kind.value),kind.value "
        "FROM CANN_API p JOIN TASK t ON p.connectionId=t.connectionId "
        "JOIN STRING_IDS kind ON kind.id=t.taskType "
        "LEFT JOIN COMPUTE_TASK_INFO k ON k.globalTaskId=t.globalTaskId "
        "LEFT JOIN STRING_IDS n ON n.id=k.name ORDER BY t.startNs"
    )
    for launch, start, end, task_id, name, kind in rows:
        position = bisect.bisect_right(starts, launch) - 1
        if position < 0 or launch > marks[position][1]:
            continue
        # Stream synchronization can be recorded as an instant device event.
        if start > end or (start == end and kind not in {"EVENT RECORD", "EVENT WAIT"}):
            raise ValueError(f"invalid device task interval: {task_id}")
        task = {"start_ns": start, "end_ns": end, "task_id": task_id, "name": name, "task_type": kind}
        previous = groups[marks[position][2]].setdefault(task_id, task)
        if previous != task:
            raise ValueError(f"conflicting device rows for task {task_id}")
    return {label: list(tasks.values()) for label, tasks in groups.items()}


def read_replay_tasks(connection, replay_id):
    sentinels = connection.execute(
        "SELECT t.startNs,t.endNs,t.deviceId,t.modelId,t.streamId,k.value "
        "FROM TASK t JOIN STRING_IDS k ON k.id=t.taskType WHERE t.connectionId=? "
        "AND k.value IN ('MODEL_EXECUTE','MODEL_WAIT_COMPLETE') ORDER BY t.startNs",
        (replay_id,),
    ).fetchall()
    if len(sentinels) != 2:
        raise ValueError(f"replay {replay_id} has {len(sentinels)} model sentinels, expected two")
    execute, wait = sentinels
    start, _, device, model, stream, execute_kind = execute
    _, stop, wait_device, wait_model, wait_stream, wait_kind = wait
    if (
        (execute_kind, wait_kind) != ("MODEL_EXECUTE", "MODEL_WAIT_COMPLETE")
        or (device, model, stream) != (wait_device, wait_model, wait_stream)
        or model in (None, -1, 4294967295)
        or start >= stop
    ):
        raise ValueError(f"replay {replay_id} has inconsistent model sentinels")

    # Child tasks use different CANN connection IDs. Their global task IDs can
    # repeat across replays, so select by the device window and deduplicate by row ID.
    rows = connection.execute(
        "SELECT t.rowid,t.startNs,t.endNs,t.streamId,t.globalTaskId,t.connectionId,COALESCE(n.value,k.value),k.value "
        "FROM TASK t JOIN STRING_IDS k ON k.id=t.taskType "
        "LEFT JOIN COMPUTE_TASK_INFO c ON c.globalTaskId=t.globalTaskId "
        "LEFT JOIN STRING_IDS n ON n.id=c.name "
        "WHERE t.deviceId=? AND t.modelId=? AND t.startNs>=? AND t.startNs<? ORDER BY t.startNs,t.rowid",
        (device, model, start, stop),
    )
    tasks = {}
    for row_id, begin, end, stream_id, global_id, connection_id, name, kind in rows:
        if kind in GRAPH_SENTINELS:
            if connection_id != replay_id:
                raise ValueError(f"replay {replay_id} overlaps another invocation of model {model}")
            continue
        instant_event = kind in {"EVENT RECORD", "EVENT WAIT", "NOTIFY RECORD"}
        if end > stop or begin > end or (begin == end and not instant_event):
            raise ValueError(f"replay {replay_id} has invalid or cross-boundary task row {row_id}")
        task = {
            "row_id": row_id,
            "start_ns": begin,
            "end_ns": end,
            "stream_id": stream_id,
            "device_id": device,
            "model_id": model,
            "task_id": global_id,
            "replay_connection_id": replay_id,
            "name": name,
            "task_type": kind,
        }
        if tasks.setdefault(row_id, task) != task:
            raise ValueError(f"conflicting names for replay {replay_id} task row {row_id}")
    return list(tasks.values())


def read_graph_groups(connection):
    groups = {}
    for start, end, label in read_markers(connection):
        replays = connection.execute(
            "SELECT DISTINCT p.connectionId FROM CANN_API p JOIN STRING_IDS n ON n.id=p.name "
            "WHERE p.startNs>=? AND p.startNs<? AND n.value='aclmdlRIExecuteAsync'",
            (start, end),
        ).fetchall()
        if len(replays) != 1:
            raise ValueError(f"{label} contained {len(replays)} aclmdlRIExecuteAsync correlations, expected one")
        groups[label] = read_replay_tasks(connection, replays[0][0])
    return groups


def extract(args):
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("error") or any(case.get("status") != "captured" for case in manifest["cases"]):
        raise ValueError("capture manifest contains failed or incomplete batches")
    graph_profile = manifest.get("profile_mode") == GRAPH_PROFILE_MODE
    path = database_path(args.profile)
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        groups = read_graph_groups(connection) if graph_profile else read_groups(connection)
    expected = {label for case in manifest["cases"] for label in case["markers"]}
    if set(groups) != expected:
        raise ValueError(
            f"marker mismatch: missing={sorted(expected - set(groups))}, extra={sorted(set(groups) - expected)}"
        )
    samples = defaultdict(list)
    for label, tasks in groups.items():
        batch, tokens, mode, sample = MARKER_PATTERN.fullmatch(label).groups()
        item = task_metrics(tasks)
        item.update({"marker": label, "sample": int(sample)})
        samples[(int(batch), int(tokens), mode)].append(item)
    candidate_modes = manifest.get("candidate_modes", ["candidate"])
    results = []
    for case in manifest["cases"]:
        batch, tokens = case["config"]["batch"], case["config"]["tokens"]
        result = {"batch": batch, "tokens": tokens, "parameters": case["parameters"], "backends": {}}
        for mode in (manifest["baseline"], *candidate_modes):
            values = sorted(samples[(batch, tokens, mode)], key=lambda sample: sample["sample"])
            if [value["sample"] for value in values] != list(range(manifest["samples_per_backend"])):
                raise ValueError(f"B={batch} L={tokens} {mode}: missing, duplicate or extra samples")
            result["backends"][mode] = {
                "median_device_elapsed_us": statistics.median(value["device_elapsed_us"] for value in values),
                "kernel_counts": sorted({value["kernel_count"] for value in values}),
                "samples": values,
            }
        base = result["backends"][manifest["baseline"]]["median_device_elapsed_us"]
        if "variant_parameters" in case:
            result["variant_parameters"] = case["variant_parameters"]
            result["speedup_vs_baseline"] = {
                mode: base / result["backends"][mode]["median_device_elapsed_us"] for mode in candidate_modes
            }
        else:
            result["speedup"] = base / result["backends"]["candidate"]["median_device_elapsed_us"]
        results.append(result)
    document = {
        "schema_version": 1,
        "database": str(path),
        "manifest": manifest,
        "primary_metric": "first-to-last correlated device task span, including split-launch gaps",
        "limits": "excludes host time before the first device task and after the last; not serving latency",
        "a5_difference": "A5 single-launch extractor reports compute busy union; retained here as a secondary metric",
        "cases": results,
    }
    if graph_profile:
        document["primary_metric"] = "first-to-last correlated ACL Graph child task span"
        document["replay_attribution"] = (
            "unique aclmdlRIExecuteAsync in each host marker -> paired MODEL_EXECUTE/MODEL_WAIT_COMPLETE -> "
            "same-device, same-model physical TASK rows within the sentinel interval"
        )
    write_json(args.out, document)
    print(f"Extracted {len(results)} batches from {path}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    capture_parser = commands.add_parser("capture", help="run under msprof --task-time=on --msproftx=on")
    capture_parser.add_argument(
        "--execution", choices=("eager", "graph"), default="eager", help="capture eager launches or ACL Graph replays"
    )
    capture_parser.add_argument("--batches", type=integer_sweep, default=integer_sweep("1:50"))
    capture_parser.add_argument("--tokens", type=int, default=6)
    capture_parser.add_argument("--heads", type=int, default=64)
    capture_parser.add_argument("--hidden", type=int, default=4096)
    capture_parser.add_argument("--baseline", choices=("native", "torch"), default="native")
    candidate_options = capture_parser.add_mutually_exclusive_group()
    candidate_options.add_argument("--selection", type=Path, help="compare_full JSON; use each batch's frozen winner")
    capture_parser.add_argument("--row-tile", type=int, default=0)
    capture_parser.add_argument("--projection-pad", type=int, default=0)
    candidate_options.add_argument(
        "--variant", action="append", choices=MIDDLE_VARIANTS, help="named TileLang variant; may be repeated"
    )
    capture_parser.add_argument("--core-count", type=int, default=24, help="AIC groups for the named CV variants")
    capture_parser.add_argument("--samples", type=int, default=11)
    capture_parser.add_argument("--warmup", type=int, default=4)
    capture_parser.add_argument("--seed", type=int, default=724311)
    capture_parser.add_argument("--protocol", choices=("relative-l2", "elementwise", "both"), default="relative-l2")
    capture_parser.add_argument("--out", type=Path, required=True, help="capture manifest")
    extract_parser = commands.add_parser(
        "extract", help="extract device spans and kernel counts from a msprof SQLite DB"
    )
    extract_parser.add_argument(
        "--profile", type=Path, required=True, help="exact msprof_*.db or directory containing one"
    )
    extract_parser.add_argument("--manifest", type=Path, required=True)
    extract_parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "capture":
        if not 1 <= args.tokens <= 16 or min(args.heads, args.hidden, args.samples) < 1:
            parser.error("require 1 <= tokens <= 16 and positive dimensions/sample count")
        if min(args.row_tile, args.projection_pad, args.warmup) < 0:
            parser.error("candidate parameters and warmup must be nonnegative")
        if args.core_count < 1:
            parser.error("core count must be positive")
        if args.variant:
            args.variant = list(dict.fromkeys(args.variant))
    return args


def main():
    args = parse_args()
    if args.command == "capture":
        capture(args)
    else:
        extract(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
