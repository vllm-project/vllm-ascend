#!/usr/bin/env python3
"""
Ascend NPU Profiling Analyzer for vllm-ascend DBO Overlap Template Development.

Parses Ascend Profiler output directories (produced by torch_npu profiler) and
generates structured reports covering:
  - Kernel breakdown by OP type and core type
  - Communication operator analysis (AllReduce, AllGather, AlltoAll, Broadcast)
  - Compute vs Communication overlap ratio
  - Per-layer DBO hook timing estimation
  - Overlap opportunity identification for DBO template design
  - Communication call stack extraction (trace Python source → comm op mapping)

Usage:
    python analyze_ascend_profiling.py triage   --input /path/to/profiling_dir
    python analyze_ascend_profiling.py breakdown --input /path/to/profiling_dir
    python analyze_ascend_profiling.py comm      --input /path/to/profiling_dir
    python analyze_ascend_profiling.py stack     --input /path/to/profiling_dir
    python analyze_ascend_profiling.py compare   --input-a /path/to/dir_a --input-b /path/to/dir_b
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class KernelOp:
    """A single kernel execution on the NPU device."""
    device_id: int
    name: str
    op_type: str
    core_type: str
    start_us: float
    duration_us: float
    wait_us: float
    stream_id: int
    input_shapes: str = ""
    input_dtypes: str = ""
    category: str = ""  # filled by classify_kernel


@dataclass
class CommOp:
    """A single communication operation."""
    name: str
    comm_type: str  # allreduce, allgather, alltoallv, broadcast
    start_us: float
    elapse_ms: float
    transit_ms: float
    wait_ms: float
    sync_ms: float
    idle_ms: float
    group_id: str = ""


@dataclass
class TraceCommEvent:
    """A communication event extracted from trace_view.json with call stack."""
    name: str
    comm_type: str
    ts_us: float
    dur_us: float
    tid: int
    call_stack: List[str] = field(default_factory=list)  # from innermost to outermost
    code_location: str = ""  # the most relevant vllm/vllm-ascend source location


@dataclass
class OpStatRow:
    """Aggregated OP statistics from op_statistic.csv."""
    device_id: int
    op_type: str
    core_type: str
    count: int
    total_us: float
    min_us: float
    avg_us: float
    max_us: float
    ratio_pct: float
    category: str = ""


@dataclass
class StepTraceInfo:
    """Step-level trace timing from step_trace_time.csv."""
    device_id: int
    computing_us: float
    comm_not_overlapped_us: float
    overlapped_us: float
    communication_us: float
    free_us: float
    stage_us: float
    bubble_us: float
    comm_not_overlapped_excl_recv_us: float
    preparing_us: float


@dataclass
class CommBandwidthInfo:
    """Communication bandwidth summary."""
    transport_type: str  # HCCS, SIO, RDMA, PCIE, SDMA
    transit_size_mb: float
    transit_time_ms: float
    bandwidth_gbps: float


# ---------------------------------------------------------------------------
# Kernel classification (Ascend NPU specific)
# ---------------------------------------------------------------------------

KERNEL_CATEGORY_RULES: List[Tuple[str, List[str]]] = [
    ("communication", [
        "allreduce", "allgather", "alltoall", "broadcast",
        "hcom_", "Notify_Record", "Notify_Wait",
        "AivKernel",  # communication helper kernels
    ]),
    ("attention", [
        "FusedInferAttentionScore", "AttentionUpdate",
        "_fwd_diag_kernel", "_fwd_none_diag_kernel",
        "_fwd_kv_parallel", "_fwd_kv_reduce",
        "PagedCacheLoad",
    ]),
    ("gemm", [
        "MatMulV3", "MatMulV2", "GroupedMatmul",
        "QuantBatchMatmulV3", "BatchMatMul",
    ]),
    ("norm", [
        "layer_norm", "RmsNorm", "AddRmsNormBias",
        "KvRmsNormRopeCache",
    ]),
    ("rope", [
        "_triton_rope", "InterleaveRope", "Rope",
    ]),
    ("moe", [
        "MoeTokenPermute", "MoeTokenUnpermute",
        "MoeGatingTopK", "HistogramV2",
    ]),
    ("activation", [
        "SwiGlu", "_swiglu_quant_kernel", "Sigmoid",
        "muls_add_kernel",
    ]),
    ("quantize", [
        "DynamicQuant", "Quant",
    ]),
    ("memory", [
        "TensorMove", "ConcatD", "Transpose", "AsStrided",
        "Slice", "MemSet", "GatherV2", "Index",
        "RepeatInterleave", "BroadcastTo",
    ]),
    ("elementwise", [
        "Add", "Sub", "Mul", "Neg", "Exp", "Log",
        "Cast", "Fill", "ZerosLike", "Range",
        "Equal", "Less", "GreaterEqual", "LogicalAnd", "LogicalNot",
        "MaskedFill", "RealDiv",
    ]),
    ("sampling", [
        "SoftmaxV2", "ArgMaxV2", "DSARandomUniform",
        "ReduceSum",
    ]),
]


def classify_kernel(name: str) -> str:
    """Classify a kernel/op name into a category."""
    name_lower = name.lower()
    for category, patterns in KERNEL_CATEGORY_RULES:
        for pattern in patterns:
            if pattern.lower() in name_lower:
                return category
    return "other"


def classify_comm_type(name: str) -> str:
    """Extract communication type from hcom op name.

    Note: AlltoAll ops are classified as "alltoallv" (not "alltoall") throughout
    this script. This is intentional: Ascend NPU's hcom library uses the name
    "HcclAlltoAllV" (variable-length AlltoAll) for MoE token dispatch/combine,
    and the op names in communication.json / trace_view.json contain "alltoall"
    which maps to "alltoallv" here. Using a single canonical key "alltoallv"
    avoids splitting the same op type across two keys ("alltoall" vs "alltoallv").
    """
    name_lower = name.lower()
    if "allreduce" in name_lower:
        return "allreduce"
    if "allgather" in name_lower:
        return "allgather"
    if "alltoall" in name_lower:
        # Unified as "alltoallv": covers both HcclAlltoAll and HcclAlltoAllV
        return "alltoallv"
    if "broadcast" in name_lower:
        return "broadcast"
    return "other"


# ---------------------------------------------------------------------------
# File discovery and loading
# ---------------------------------------------------------------------------

def discover_profiling_dirs(root: Path) -> List[Path]:
    """Find all Ascend profiling output directories under root."""
    dirs = []
    if (root / "ASCEND_PROFILER_OUTPUT").is_dir():
        dirs.append(root)
    else:
        for child in sorted(root.iterdir()):
            if child.is_dir() and (child / "ASCEND_PROFILER_OUTPUT").is_dir():
                dirs.append(child)
    return dirs


def parse_rank_from_dirname(dirname: str) -> Dict[str, Any]:
    """Parse dp/pp/tp/ep/rank info from directory name like
    dp0_pp0_tp0_dcp0_ep0_rank0_25596_20260402090434918_ascend_pt"""
    info = {}
    parts = dirname.split("_")
    for part in parts:
        for prefix in ("dp", "pp", "tp", "dcp", "ep", "rank"):
            if part.startswith(prefix) and part[len(prefix):].isdigit():
                info[prefix] = int(part[len(prefix):])
    return info


def load_op_statistic(prof_dir: Path) -> List[OpStatRow]:
    """Load op_statistic.csv."""
    csv_path = prof_dir / "ASCEND_PROFILER_OUTPUT" / "op_statistic.csv"
    if not csv_path.exists():
        return []
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            row = OpStatRow(
                device_id=int(r.get("Device_id", 0)),
                op_type=r.get("OP Type", ""),
                core_type=r.get("Core Type", ""),
                count=int(r.get("Count", 0)),
                total_us=float(r.get("Total Time(us)", 0)),
                min_us=float(r.get("Min Time(us)", 0)),
                avg_us=float(r.get("Avg Time(us)", 0)),
                max_us=float(r.get("Max Time(us)", 0)),
                ratio_pct=float(r.get("Ratio(%)", 0)),
            )
            row.category = classify_kernel(row.op_type)
            rows.append(row)
    return rows


def load_step_trace(prof_dir: Path) -> Optional[StepTraceInfo]:
    """Load step_trace_time.csv."""
    csv_path = prof_dir / "ASCEND_PROFILER_OUTPUT" / "step_trace_time.csv"
    if not csv_path.exists():
        return None
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            return StepTraceInfo(
                device_id=int(r.get("Device_id", 0)),
                computing_us=float(r.get("Computing", 0)),
                comm_not_overlapped_us=float(r.get("Communication(Not Overlapped)", 0)),
                overlapped_us=float(r.get("Overlapped", 0)),
                communication_us=float(r.get("Communication", 0)),
                free_us=float(r.get("Free", 0)),
                stage_us=float(r.get("Stage", 0)),
                bubble_us=float(r.get("Bubble", 0)),
                comm_not_overlapped_excl_recv_us=float(
                    r.get("Communication(Not Overlapped and Exclude Receive)", 0)
                ),
                preparing_us=float(r.get("Preparing", 0)),
            )
    return None


def load_communication_json(prof_dir: Path) -> List[CommOp]:
    """Load communication.json and extract individual comm ops.

    The JSON may be a flat dict where keys are op names like
    ``hcom_allReduce__503_956_1@...`` and values contain
    ``Communication Time Info``.  There may also be a nested structure
    with a ``Total Op Info`` summary key.
    """
    json_path = prof_dir / "ASCEND_PROFILER_OUTPUT" / "communication.json"
    if not json_path.exists():
        return []
    with open(json_path, "r") as f:
        data = json.load(f)

    ops = []

    def _extract(d: dict) -> None:
        for key, val in d.items():
            if key == "Total Op Info":
                continue
            if not isinstance(val, dict):
                continue
            time_info = val.get("Communication Time Info")
            if time_info is None:
                # Might be a nested wrapper; recurse one level
                _extract(val)
                continue
            ops.append(CommOp(
                name=key,
                comm_type=classify_comm_type(key),
                start_us=float(time_info.get("Start Timestamp(us)", 0)),
                elapse_ms=float(time_info.get("Elapse Time(ms)", 0)),
                transit_ms=float(time_info.get("Transit Time(ms)", 0)),
                wait_ms=float(time_info.get("Wait Time(ms)", 0)),
                sync_ms=float(time_info.get("Synchronization Time(ms)", 0)),
                idle_ms=float(time_info.get("Idle Time(ms)", 0)),
            ))

    _extract(data)
    return ops


def load_communication_matrix(prof_dir: Path) -> Dict[str, Any]:
    """Load communication_matrix.json."""
    json_path = prof_dir / "ASCEND_PROFILER_OUTPUT" / "communication_matrix.json"
    if not json_path.exists():
        return {}
    with open(json_path, "r") as f:
        return json.load(f)


def load_profiler_metadata(prof_dir: Path) -> Dict[str, Any]:
    """Load profiler_metadata.json."""
    meta_path = prof_dir / "profiler_metadata.json"
    if not meta_path.exists():
        return {}
    with open(meta_path, "r") as f:
        return json.load(f)


def load_profiler_info(prof_dir: Path) -> Dict[str, Any]:
    """Load profiler_info_*.json (first found)."""
    for f in prof_dir.iterdir():
        if f.name.startswith("profiler_info_") and f.name.endswith(".json"):
            with open(f, "r") as fh:
                return json.load(fh)
    return {}


# ---------------------------------------------------------------------------
# trace_view.json loading and call-stack extraction
# ---------------------------------------------------------------------------

# Communication op names in trace_view.json (cpu_op category)
TRACE_COMM_OP_NAMES = {
    "c10d::allreduce_", "c10d::allgather_", "c10d::_allgather_base_",
    "c10d::alltoall_", "c10d::broadcast_", "c10d::_reduce_scatter_base_",
    "c10d::reduce_scatter_", "c10d::reduce_scatter_tensor_",
    "HcclAllreduce", "HcclAllgather", "HcclAlltoAll",
    "HcclBroadcast", "HcclReduceScatter",
}

# Patterns to identify communication ops by substring match
TRACE_COMM_PATTERNS = [
    "allreduce", "allgather", "alltoall", "broadcast",
    "reduce_scatter", "HcclAll", "HcclBroadcast", "HcclReduce",
]

# Source path prefixes that indicate vllm or vllm-ascend code (most relevant)
RELEVANT_SOURCE_PREFIXES = [
    "vllm_ascend/", "vllm/", "/vllm-ascend/", "/vllm/",
]

# Source path prefixes to skip (framework internals)
SKIP_SOURCE_PREFIXES = [
    "torch/", "torch_npu/", "threading.py", "tqdm/",
    "<built-in method", "<frozen", "importlib",
]


def _classify_trace_comm_type(name: str) -> str:
    """Classify a trace event name into communication type."""
    name_lower = name.lower()
    if "reduce_scatter" in name_lower:
        return "reduce_scatter"
    if "allreduce" in name_lower:
        return "allreduce"
    if "allgather" in name_lower:
        return "allgather"
    if "alltoall" in name_lower:
        return "alltoallv"
    if "broadcast" in name_lower:
        return "broadcast"
    return "other"


def _is_comm_event(ev: dict) -> bool:
    """Check if a trace event is a communication operation."""
    name = ev.get("name", "")
    if name in TRACE_COMM_OP_NAMES:
        return True
    name_lower = name.lower()
    return any(p.lower() in name_lower for p in TRACE_COMM_PATTERNS)


def _is_relevant_source(name: str) -> bool:
    """Check if a python_function event name points to vllm/vllm-ascend source."""
    if name.startswith("<built-in"):
        return False
    return any(prefix in name for prefix in RELEVANT_SOURCE_PREFIXES)


def _should_skip_source(name: str) -> bool:
    """Check if a python_function event name should be skipped in stack display."""
    return any(name.startswith(prefix) or prefix in name
               for prefix in SKIP_SOURCE_PREFIXES)


def _determine_target_comm_types(
    comm_ops: List[CommOp],
    top_k: int = 3,
    min_count: int = 10,
) -> Set[str]:
    """Determine the most frequent communication types from communication.json data.

    Analyzes the frequency of each communication type and returns the top-K
    types that appear at least ``min_count`` times.  This is used to focus
    stack extraction on the most relevant communication operations.

    Args:
        comm_ops: List of CommOp loaded from communication.json.
        top_k: Number of top communication types to return.
        min_count: Minimum occurrence count for a type to be considered.

    Returns:
        Set of communication type strings (e.g. {"allreduce", "alltoallv", "allgather"}).
        Returns an empty set if no types meet the criteria.
    """
    from collections import Counter

    counter = Counter(op.comm_type for op in comm_ops)
    result: Set[str] = set()
    for comm_type, count in counter.most_common(top_k):
        if count >= min_count:
            result.add(comm_type)
    return result


def _extract_source_location(name: str) -> str:
    """Extract file:line from a python_function event name.

    Format: ``path/to/file.py(123): function_name``
    Returns: ``path/to/file.py:123 function_name``
    """
    # Match pattern: path(line): func
    m = re.match(r"(.+?)\((\d+)\):\s*(.+)", name)
    if m:
        filepath, line, func = m.group(1), m.group(2), m.group(3)
        # Normalize path: strip leading deployment prefix
        for marker in ["/vllm-ascend/", "/vllm/"]:
            idx = filepath.find(marker)
            if idx >= 0:
                filepath = filepath[idx + 1:]  # keep "vllm-ascend/..." or "vllm/..."
                break
        return f"{filepath}:{line} {func}"
    return name


def load_trace_comm_with_stacks(
    prof_dir: Path,
    *,
    max_depth: int = 20,
    dedup: bool = True,
    target_comm_types: Optional[Set[str]] = None,
    max_stacks: int = 100,
    comm_ops: Optional[List[CommOp]] = None,
) -> List[TraceCommEvent]:
    """Load trace_view.json and extract communication events with Python call stacks.

    Uses a two-phase strategy to minimise memory usage on large trace files:

    **Phase 1 – Determine targets** (from communication.json):
      If *target_comm_types* is ``None`` and *comm_ops* (or a fresh load) is
      available, the function calls ``_determine_target_comm_types()`` to pick
      the top-K most frequent communication types so that Phase 2 can skip
      irrelevant events early.

    **Phase 2 – Single-pass extraction** (from trace_view.json):
      Iterates over the events list **once**, simultaneously building a
      ``py_by_id`` index (for ``python_function`` events) and collecting
      target ``cpu_op`` communication events.  After the scan, call stacks
      are reconstructed by tracing the ``Python parent id`` chain.

    The events list is explicitly deleted after processing to free memory.

    Args:
        prof_dir: Profiling directory containing ASCEND_PROFILER_OUTPUT/trace_view.json
        max_depth: Maximum call stack depth to trace.
        dedup: If True, deduplicate events with identical call stacks.
        target_comm_types: If provided, only extract stacks for these comm types.
            When ``None``, the function will auto-determine targets from
            *comm_ops* / communication.json (top-3 by default).
        max_stacks: Maximum number of unique stacks to collect (enables early
            termination).  Set to 0 to disable the limit.
        comm_ops: Pre-loaded CommOp list.  Passed in to avoid re-loading
            communication.json when the caller already has it.

    Returns:
        List of TraceCommEvent with populated call_stack and code_location.
    """
    json_path = prof_dir / "ASCEND_PROFILER_OUTPUT" / "trace_view.json"
    if not json_path.exists():
        return []

    # ------------------------------------------------------------------
    # Phase 1: determine target comm types (cheap – from communication.json)
    # ------------------------------------------------------------------
    if target_comm_types is None:
        if comm_ops is None:
            comm_ops = load_communication_json(prof_dir)
        if comm_ops:
            target_comm_types = _determine_target_comm_types(comm_ops)
        # If still empty / None, we will accept all comm types below.

    # ------------------------------------------------------------------
    # Phase 2: single-pass scan of trace_view.json
    # ------------------------------------------------------------------
    with open(json_path, "r") as f:
        events = json.load(f)

    if isinstance(events, dict):
        events = events.get("traceEvents", events.get("events", []))

    # Single-pass: build python_function index AND collect comm events
    py_by_id: Dict[int, dict] = {}
    py_by_tid: Dict[int, List[dict]] = defaultdict(list)
    raw_comm_events: List[dict] = []

    for ev in events:
        if not isinstance(ev, dict):
            continue
        cat = ev.get("cat")
        if cat == "python_function":
            pid = ev.get("args", {}).get("Python id")
            if pid is not None:
                py_by_id[pid] = ev
                py_by_tid[ev.get("tid", 0)].append(ev)
        elif cat == "cpu_op" and _is_comm_event(ev):
            comm_type = _classify_trace_comm_type(ev.get("name", ""))
            if target_comm_types is None or comm_type in target_comm_types:
                raw_comm_events.append(ev)

    # Free the (potentially huge) events list as early as possible
    del events

    # Sort each thread's python_function events by start timestamp for
    # efficient binary-search based containment lookup.
    for tid in py_by_tid:
        py_by_tid[tid].sort(key=lambda e: float(e.get("ts", 0)))

    # ------------------------------------------------------------------
    # Reconstruct call stacks for collected comm events
    # ------------------------------------------------------------------
    results: List[TraceCommEvent] = []
    seen_stacks: set = set()

    for ev in raw_comm_events:
        comm_name = ev.get("name", "")
        comm_type = _classify_trace_comm_type(comm_name)
        comm_ts = float(ev.get("ts", 0))
        comm_dur = float(ev.get("dur", 0))
        comm_end = comm_ts + comm_dur
        comm_tid = ev.get("tid", 0)

        # Find the most specific (smallest duration) python_function event
        # on the same thread that contains this comm event.
        # Use a linear scan over the (sorted-by-ts) list; for each candidate
        # check containment and track the tightest fit.
        best_py = None
        best_dur = float("inf")
        for py_ev in py_by_tid.get(comm_tid, []):
            py_ts = float(py_ev.get("ts", 0))
            py_dur = float(py_ev.get("dur", 0))
            py_end = py_ts + py_dur
            if py_ts <= comm_ts and py_end >= comm_end and py_dur < best_dur:
                best_py = py_ev
                best_dur = py_dur

        # Trace the call stack via Python parent id
        call_stack: List[str] = []
        code_location = ""
        if best_py:
            current = best_py
            depth = 0
            while current and depth < max_depth:
                name = current.get("name", "")
                call_stack.append(name)
                if not code_location and _is_relevant_source(name):
                    code_location = _extract_source_location(name)
                parent_id = current.get("args", {}).get("Python parent id")
                if parent_id is not None and parent_id in py_by_id:
                    current = py_by_id[parent_id]
                else:
                    break
                depth += 1

        # Deduplication: skip if we've seen this exact stack before
        if dedup:
            stack_key = (comm_name, tuple(call_stack[:5]))  # use top 5 frames as key
            if stack_key in seen_stacks:
                continue
            seen_stacks.add(stack_key)

        results.append(TraceCommEvent(
            name=comm_name,
            comm_type=comm_type,
            ts_us=comm_ts,
            dur_us=comm_dur,
            tid=comm_tid,
            call_stack=call_stack,
            code_location=code_location,
        ))

        # Early termination: stop once we have enough unique stacks
        if max_stacks > 0 and len(results) >= max_stacks:
            break

    return results


def render_comm_stacks(
    trace_comms: List[TraceCommEvent],
    *,
    show_full_stack: bool = False,
    max_stack_depth: int = 15,
) -> str:
    """Render communication events with their call stacks.

    Args:
        trace_comms: List of TraceCommEvent from load_trace_comm_with_stacks
        show_full_stack: If True, show full call stack; otherwise show only relevant frames
        max_stack_depth: Maximum number of stack frames to display
    """
    if not trace_comms:
        return "  (No communication events with call stacks found in trace_view.json)\n"

    lines: List[str] = []

    # Group by comm_type
    by_type: Dict[str, List[TraceCommEvent]] = defaultdict(list)
    for ev in trace_comms:
        by_type[ev.comm_type].append(ev)

    lines.append("\n### Communication Call Stacks (from trace_view.json)\n")

    for comm_type in ["allreduce", "reduce_scatter", "allgather", "alltoallv", "broadcast", "other"]:
        evts = by_type.get(comm_type, [])
        if not evts:
            continue

        lines.append(f"\n#### {comm_type.upper()} ({len(evts)} unique call sites)\n")

        for i, ev in enumerate(evts, 1):
            lines.append(f"**{i}. `{ev.name}`** (dur={ev.dur_us:.1f}μs)")
            if ev.code_location:
                lines.append(f"   → **Code location**: `{ev.code_location}`")

            # Show call stack
            lines.append("   ```")
            displayed = 0
            for frame in ev.call_stack:
                if not show_full_stack and _should_skip_source(frame):
                    continue
                loc = _extract_source_location(frame)
                lines.append(f"   {'  ' * min(displayed, 10)}{loc}")
                displayed += 1
                if displayed >= max_stack_depth:
                    lines.append(f"   {'  ' * 10}... (truncated)")
                    break
            lines.append("   ```")
            lines.append("")

    # Summary table: code location → comm types
    lines.append("\n### Communication Code Location Summary\n")
    lines.append("| # | Comm Type | Op Name | Code Location | Duration(μs) |")
    lines.append("| --- | --- | --- | --- | ---: |")
    for i, ev in enumerate(trace_comms, 1):
        loc = ev.code_location or "(unknown)"
        lines.append(f"| {i} | {ev.comm_type} | `{ev.name}` | `{loc}` | {ev.dur_us:.1f} |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def analyze_op_breakdown(op_stats: List[OpStatRow]) -> Dict[str, Any]:
    """Analyze kernel breakdown by category and individual ops."""
    total_us = sum(r.total_us for r in op_stats)
    if total_us == 0:
        return {"total_us": 0, "categories": {}, "top_ops": []}

    # Category aggregation
    cat_stats: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"total_us": 0.0, "count": 0, "ratio_pct": 0.0}
    )
    for r in op_stats:
        cat = r.category
        cat_stats[cat]["total_us"] += r.total_us
        cat_stats[cat]["count"] += r.count
    for cat in cat_stats:
        cat_stats[cat]["ratio_pct"] = cat_stats[cat]["total_us"] / total_us * 100

    # Sort categories by total time
    sorted_cats = dict(
        sorted(cat_stats.items(), key=lambda x: x[1]["total_us"], reverse=True)
    )

    # Top ops by total time
    sorted_ops = sorted(op_stats, key=lambda r: r.total_us, reverse=True)

    return {
        "total_us": total_us,
        "categories": sorted_cats,
        "top_ops": sorted_ops,
    }


def analyze_communication(
    comm_ops: List[CommOp],
    comm_matrix: Dict[str, Any],
) -> Dict[str, Any]:
    """Analyze communication patterns and bandwidth."""
    if not comm_ops:
        return {"total_ops": 0}

    # Group by comm type
    by_type: Dict[str, List[CommOp]] = defaultdict(list)
    for op in comm_ops:
        by_type[op.comm_type].append(op)

    type_summary = {}
    for ctype, ops in sorted(by_type.items()):
        durations = [op.elapse_ms for op in ops]
        type_summary[ctype] = {
            "count": len(ops),
            "total_ms": sum(durations),
            "avg_ms": sum(durations) / len(durations) if durations else 0,
            "min_ms": min(durations) if durations else 0,
            "max_ms": max(durations) if durations else 0,
        }

    # Extract bandwidth info from matrix
    bandwidth_info = {}
    step_data = comm_matrix.get("step", {})
    collective = step_data.get("collective", {})
    for key, val in collective.items():
        if "total" in key:
            for link, info in val.items():
                transport = info.get("Transport Type", "")
                if transport not in bandwidth_info:
                    bandwidth_info[transport] = []
                bandwidth_info[transport].append({
                    "op": key,
                    "link": link,
                    "size_mb": info.get("Transit Size(MB)", 0),
                    "time_ms": info.get("Transit Time(ms)", 0),
                    "bandwidth_gbps": info.get("Bandwidth(GB/s)", 0),
                })

    return {
        "total_ops": len(comm_ops),
        "by_type": type_summary,
        "bandwidth": bandwidth_info,
    }


def estimate_per_layer_comm(comm_ops: List[CommOp], num_layers: int = 10) -> Dict[str, Any]:
    """Estimate per-layer communication cost by dividing total comm ops by num_layers.

    Filters out initialization ops (first ~10% of ops) to get steady-state
    estimates, then divides the remaining ops evenly across layers.

    Args:
        comm_ops: List of CommOp from load_communication_json.
        num_layers: Estimated number of decoder layers in the model.

    Returns:
        Dict mapping comm_type -> {ops_per_layer, avg_per_op_ms, total_per_layer_ms}.
        Only includes comm types with at least 2 ops total.
    """
    by_type: Dict[str, List[float]] = defaultdict(list)
    for op in comm_ops:
        by_type[op.comm_type].append(op.elapse_ms)

    # Filter out initialization ops (first few) and estimate steady-state
    per_layer = {}
    for ctype, durations in by_type.items():
        if len(durations) < 2:
            continue
        # Skip first batch (likely init), use remaining
        steady = durations[max(1, len(durations) // 10):]
        if num_layers > 0 and len(steady) >= num_layers:
            # Estimate ops per layer
            ops_per_layer = len(steady) / num_layers
            avg_ms = sum(steady) / len(steady)
            per_layer[ctype] = {
                "ops_per_layer": round(ops_per_layer, 1),
                "avg_per_op_ms": round(avg_ms, 4),
                "total_per_layer_ms": round(ops_per_layer * avg_ms, 4),
            }

    return per_layer


def compute_overlap_metrics(step_info: StepTraceInfo) -> Dict[str, Any]:
    """Compute overlap efficiency metrics from step trace."""
    total = step_info.stage_us
    if total == 0:
        return {}

    comm_total = step_info.communication_us
    overlap = step_info.overlapped_us
    comm_exposed = step_info.comm_not_overlapped_us
    compute = step_info.computing_us
    free = step_info.free_us

    overlap_ratio = overlap / comm_total * 100 if comm_total > 0 else 0
    comm_exposed_ratio = comm_exposed / total * 100
    compute_ratio = compute / total * 100
    free_ratio = free / total * 100

    return {
        "total_us": total,
        "compute_us": compute,
        "compute_ratio_pct": round(compute_ratio, 2),
        "comm_total_us": comm_total,
        "comm_exposed_us": comm_exposed,
        "comm_exposed_ratio_pct": round(comm_exposed_ratio, 2),
        "overlap_us": overlap,
        "overlap_ratio_pct": round(overlap_ratio, 2),
        "free_us": free,
        "free_ratio_pct": round(free_ratio, 2),
        "bubble_us": step_info.bubble_us,
    }


def identify_dbo_opportunities(
    op_stats: List[OpStatRow],
    comm_ops: List[CommOp],
    step_info: Optional[StepTraceInfo],
) -> List[Dict[str, Any]]:
    """Identify DBO overlap opportunities based on profiling data."""
    opportunities = []

    # 1. Check communication exposure
    if step_info:
        comm_exposed = step_info.comm_not_overlapped_us
        total = step_info.stage_us
        if total > 0 and comm_exposed / total > 0.1:
            opportunities.append({
                "type": "high_comm_exposure",
                "severity": "high",
                "description": (
                    f"Communication not overlapped: {comm_exposed/1000:.1f}ms "
                    f"({comm_exposed/total*100:.1f}% of total). "
                    f"DBO overlap can hide this behind compute."
                ),
                "metric_us": comm_exposed,
            })

    # 2. Check for AlltoAll (indicates MoE A3 mode)
    alltoall_ops = [op for op in comm_ops if op.comm_type == "alltoallv"]
    allreduce_ops = [op for op in comm_ops if op.comm_type == "allreduce"]
    allgather_ops = [op for op in comm_ops if op.comm_type == "allgather"]

    if alltoall_ops:
        avg_alltoall_ms = sum(op.elapse_ms for op in alltoall_ops) / len(alltoall_ops)
        opportunities.append({
            "type": "moe_alltoall_overlap",
            "severity": "high" if avg_alltoall_ms > 0.5 else "medium",
            "description": (
                f"Detected {len(alltoall_ops)} AlltoAll ops (MoE A3 mode), "
                f"avg {avg_alltoall_ms*1000:.1f}us. "
                f"Use MoEAlltoallTemplate with dbo_moe_prepare/finalize hooks."
            ),
            "metric_us": avg_alltoall_ms * 1000,
        })

    if allreduce_ops:
        avg_allreduce_ms = sum(op.elapse_ms for op in allreduce_ops) / len(allreduce_ops)
        opportunities.append({
            "type": "allreduce_overlap",
            "severity": "high" if avg_allreduce_ms > 1.0 else "medium",
            "description": (
                f"Detected {len(allreduce_ops)} AllReduce ops, "
                f"avg {avg_allreduce_ms*1000:.1f}us. "
                f"Overlap with next layer's compute via dbo_linear_column/row hooks."
            ),
            "metric_us": avg_allreduce_ms * 1000,
        })

    if allgather_ops:
        avg_allgather_ms = sum(op.elapse_ms for op in allgather_ops) / len(allgather_ops)
        opportunities.append({
            "type": "allgather_overlap",
            "severity": "medium",
            "description": (
                f"Detected {len(allgather_ops)} AllGather ops, "
                f"avg {avg_allgather_ms*1000:.1f}us. "
                f"Overlap with attention/MLP compute."
            ),
            "metric_us": avg_allgather_ms * 1000,
        })

    # 3. Check free time (device idle)
    if step_info and step_info.free_us > 0:
        free_ratio = step_info.free_us / step_info.stage_us * 100
        if free_ratio > 5:
            opportunities.append({
                "type": "device_idle",
                "severity": "medium",
                "description": (
                    f"Device idle time: {step_info.free_us/1000:.1f}ms "
                    f"({free_ratio:.1f}%). May indicate CPU bottleneck or "
                    f"synchronization overhead."
                ),
                "metric_us": step_info.free_us,
            })

    # 4. Check for heavy norm ops (fusion opportunity)
    norm_ops = [r for r in op_stats if r.category == "norm"]
    total_norm_us = sum(r.total_us for r in norm_ops)
    total_us = sum(r.total_us for r in op_stats)
    if total_us > 0 and total_norm_us / total_us > 0.15:
        opportunities.append({
            "type": "norm_fusion",
            "severity": "medium",
            "description": (
                f"Norm ops consume {total_norm_us/1000:.1f}ms "
                f"({total_norm_us/total_us*100:.1f}%). "
                f"Consider AddRmsNorm fusion or comm+norm overlap."
            ),
            "metric_us": total_norm_us,
        })

    return sorted(opportunities, key=lambda x: x.get("metric_us", 0), reverse=True)


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def render_header(title: str) -> str:
    return f"\n{'='*70}\n  {title}\n{'='*70}\n"


def render_metadata(prof_dir: Path) -> str:
    """Render profiling metadata."""
    lines = []
    rank_info = parse_rank_from_dirname(prof_dir.name)
    meta = load_profiler_metadata(prof_dir)
    info = load_profiler_info(prof_dir)

    lines.append(f"  Directory: {prof_dir.name}")
    if rank_info:
        lines.append(f"  Rank info: {rank_info}")
    if info:
        lines.append(f"  torch_npu: {info.get('torch_npu_version', 'N/A')}")
        lines.append(f"  CANN:      {info.get('cann_version', 'N/A')}")
        lines.append(f"  Rank ID:   {info.get('rank_id', 'N/A')}")
    if meta:
        env = meta.get("ENV_VARIABLES", {})
        pg = meta.get("parallel_group_info", {})
        if pg:
            for gname, ginfo in pg.items():
                ranks = ginfo.get("global_ranks", [])
                lines.append(f"  {gname}: group_rank={ginfo.get('group_rank')}, "
                             f"world_size={len(ranks)}")
    return "\n".join(lines)


def render_op_breakdown_table(analysis: Dict[str, Any], limit: int = 30) -> str:
    """Render kernel breakdown as markdown table."""
    lines = []

    # Category summary
    lines.append("\n### Category Summary\n")
    lines.append("| Category | Total Time(ms) | Count | Share(%) |")
    lines.append("| --- | ---: | ---: | ---: |")
    for cat, stats in analysis["categories"].items():
        lines.append(
            f"| {cat} | {stats['total_us']/1000:.2f} | "
            f"{int(stats['count'])} | {stats['ratio_pct']:.1f} |"
        )

    # Top ops
    lines.append(f"\n### Top {limit} Kernel Ops\n")
    lines.append("| OP Type | Core Type | Count | Total(ms) | Avg(us) | Max(us) | Share(%) | Category |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for r in analysis["top_ops"][:limit]:
        lines.append(
            f"| {r.op_type} | {r.core_type} | {r.count} | "
            f"{r.total_us/1000:.2f} | {r.avg_us:.1f} | {r.max_us:.1f} | "
            f"{r.ratio_pct:.1f} | {r.category} |"
        )

    return "\n".join(lines)


def render_comm_analysis(comm_analysis: Dict[str, Any]) -> str:
    """Render communication analysis."""
    lines = []
    lines.append("\n### Communication Summary\n")

    by_type = comm_analysis.get("by_type", {})
    if by_type:
        lines.append("| Comm Type | Count | Total(ms) | Avg(ms) | Min(ms) | Max(ms) |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for ctype, stats in by_type.items():
            lines.append(
                f"| {ctype} | {stats['count']} | {stats['total_ms']:.2f} | "
                f"{stats['avg_ms']:.4f} | {stats['min_ms']:.4f} | {stats['max_ms']:.4f} |"
            )

    bandwidth = comm_analysis.get("bandwidth", {})
    if bandwidth:
        lines.append("\n### Bandwidth by Transport\n")
        lines.append("| Transport | Op | Link | Size(MB) | Time(ms) | BW(GB/s) |")
        lines.append("| --- | --- | --- | ---: | ---: | ---: |")
        for transport, entries in bandwidth.items():
            for entry in entries:
                lines.append(
                    f"| {transport} | {entry['op']} | {entry['link']} | "
                    f"{entry['size_mb']:.1f} | {entry['time_ms']:.2f} | "
                    f"{entry['bandwidth_gbps']:.1f} |"
                )

    return "\n".join(lines)


def render_overlap_metrics(metrics: Dict[str, Any]) -> str:
    """Render overlap metrics."""
    if not metrics:
        return "  (No step trace data available)\n"
    lines = []
    total_ms = metrics["total_us"] / 1000
    lines.append(f"\n### Step-Level Overlap Metrics\n")
    lines.append(f"  Total step time:           {total_ms:.2f} ms")
    lines.append(f"  Computing:                 {metrics['compute_us']/1000:.2f} ms ({metrics['compute_ratio_pct']:.1f}%)")
    lines.append(f"  Communication (total):     {metrics['comm_total_us']/1000:.2f} ms")
    lines.append(f"  Communication (exposed):   {metrics['comm_exposed_us']/1000:.2f} ms ({metrics['comm_exposed_ratio_pct']:.1f}%)")
    lines.append(f"  Overlapped:                {metrics['overlap_us']/1000:.2f} ms ({metrics['overlap_ratio_pct']:.1f}%)")
    lines.append(f"  Free (idle):               {metrics['free_us']/1000:.2f} ms ({metrics['free_ratio_pct']:.1f}%)")
    lines.append(f"  Bubble:                    {metrics['bubble_us']/1000:.2f} ms")

    # Visual bar
    if metrics["total_us"] > 0:
        bar_width = 60
        compute_w = int(metrics["compute_ratio_pct"] / 100 * bar_width)
        comm_exp_w = int(metrics["comm_exposed_ratio_pct"] / 100 * bar_width)
        overlap_w = max(1, int(metrics["overlap_ratio_pct"] / 100 * bar_width)) if metrics["overlap_us"] > 0 else 0
        free_w = bar_width - compute_w - comm_exp_w - overlap_w
        if free_w < 0:
            free_w = 0
        lines.append(f"\n  [{'#' * compute_w}{'!' * comm_exp_w}{'~' * overlap_w}{'.' * free_w}]")
        lines.append(f"   # = compute  ! = comm(exposed)  ~ = overlap  . = free")

    return "\n".join(lines)


def render_opportunities(opportunities: List[Dict[str, Any]]) -> str:
    """Render DBO overlap opportunities."""
    if not opportunities:
        return "  No significant overlap opportunities identified.\n"
    lines = []
    lines.append("\n### DBO Overlap Opportunities\n")
    lines.append("| # | Severity | Type | Description | Metric |")
    lines.append("| --- | --- | --- | --- | ---: |")
    for i, opp in enumerate(opportunities, 1):
        lines.append(
            f"| {i} | **{opp['severity']}** | {opp['type']} | "
            f"{opp['description']} | {opp['metric_us']/1000:.2f}ms |"
        )
    return "\n".join(lines)


def render_model_type_inference(
    comm_ops: List[CommOp],
    op_stats: List[OpStatRow],
) -> str:
    """Infer model type and communication mode from profiling data."""
    lines = []

    has_alltoall = any(op.comm_type == "alltoallv" for op in comm_ops)
    has_allgather = any(op.comm_type == "allgather" for op in comm_ops)
    has_allreduce = any(op.comm_type == "allreduce" for op in comm_ops)
    has_broadcast = any(op.comm_type == "broadcast" for op in comm_ops)
    has_moe_ops = any(r.category == "moe" for r in op_stats)
    has_attention = any(r.category == "attention" for r in op_stats)

    # Infer model type
    if has_moe_ops and has_alltoall:
        model_type = "MoE (with AlltoAll / A3 mode)"
        template_suggestion = "MoEAlltoallTemplate"
    elif has_moe_ops and has_allgather:
        model_type = "MoE (with AllGather / A2 mode)"
        template_suggestion = "MoEAllgatherTemplate"
    elif has_moe_ops:
        model_type = "MoE (mode unclear)"
        template_suggestion = "MoEAllgatherTemplate or MoEAlltoallTemplate"
    else:
        model_type = "Dense"
        template_suggestion = "DenseAllgatherTemplate"

    # Check for MLA indicators
    has_fwd_diag = any("_fwd_diag" in r.op_type for r in op_stats)
    has_fwd_kv = any("_fwd_kv" in r.op_type for r in op_stats)
    if has_fwd_diag and has_fwd_kv:
        model_type += " + MLA attention"
        if "MoE" in model_type:
            template_suggestion = "DeepseekAlltoallTemplate" if has_alltoall else "DeepseekAllgatherTemplate"

    lines.append(f"\n### Model Type Inference\n")
    lines.append(f"  Inferred model type:     {model_type}")
    lines.append(f"  Suggested DBO template:  {template_suggestion}")
    lines.append(f"  Communication patterns:  "
                 f"AllReduce={'Yes' if has_allreduce else 'No'}, "
                 f"AllGather={'Yes' if has_allgather else 'No'}, "
                 f"AlltoAll={'Yes' if has_alltoall else 'No'}, "
                 f"Broadcast={'Yes' if has_broadcast else 'No'}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_triage(args: argparse.Namespace) -> None:
    """Compact triage: metadata + breakdown + comm + overlap + opportunities."""
    root = Path(args.input).resolve()
    prof_dirs = discover_profiling_dirs(root)
    if not prof_dirs:
        print(f"ERROR: No Ascend profiling directories found under {root}")
        sys.exit(1)

    for prof_dir in prof_dirs:
        print(render_header(f"Triage Report: {prof_dir.name}"))
        print(render_metadata(prof_dir))

        op_stats = load_op_statistic(prof_dir)
        comm_ops = load_communication_json(prof_dir)
        step_info = load_step_trace(prof_dir)
        comm_matrix = load_communication_matrix(prof_dir)

        # Model type inference
        print(render_model_type_inference(comm_ops, op_stats))

        # Overlap metrics
        if step_info:
            metrics = compute_overlap_metrics(step_info)
            print(render_overlap_metrics(metrics))

        # Op breakdown
        analysis = analyze_op_breakdown(op_stats)
        print(render_op_breakdown_table(analysis, limit=args.top_k))

        # Communication
        comm_analysis = analyze_communication(comm_ops, comm_matrix)
        print(render_comm_analysis(comm_analysis))

        # Per-layer estimation
        per_layer = estimate_per_layer_comm(comm_ops, num_layers=args.num_layers)
        if per_layer:
            print("\n### Estimated Per-Layer Communication\n")
            print("| Comm Type | Ops/Layer | Avg/Op(ms) | Total/Layer(ms) |")
            print("| --- | ---: | ---: | ---: |")
            for ctype, stats in per_layer.items():
                print(
                    f"| {ctype} | {stats['ops_per_layer']} | "
                    f"{stats['avg_per_op_ms']:.4f} | {stats['total_per_layer_ms']:.4f} |"
                )

        # Opportunities
        opportunities = identify_dbo_opportunities(op_stats, comm_ops, step_info)
        print(render_opportunities(opportunities))

        print()


def cmd_breakdown(args: argparse.Namespace) -> None:
    """Detailed kernel breakdown."""
    root = Path(args.input).resolve()
    prof_dirs = discover_profiling_dirs(root)
    if not prof_dirs:
        print(f"ERROR: No Ascend profiling directories found under {root}")
        sys.exit(1)

    for prof_dir in prof_dirs:
        print(render_header(f"Kernel Breakdown: {prof_dir.name}"))
        print(render_metadata(prof_dir))

        op_stats = load_op_statistic(prof_dir)
        analysis = analyze_op_breakdown(op_stats)
        print(render_op_breakdown_table(analysis, limit=args.top_k))


def cmd_comm(args: argparse.Namespace) -> None:
    """Detailed communication analysis."""
    root = Path(args.input).resolve()
    prof_dirs = discover_profiling_dirs(root)
    if not prof_dirs:
        print(f"ERROR: No Ascend profiling directories found under {root}")
        sys.exit(1)

    for prof_dir in prof_dirs:
        print(render_header(f"Communication Analysis: {prof_dir.name}"))
        print(render_metadata(prof_dir))

        comm_ops = load_communication_json(prof_dir)
        comm_matrix = load_communication_matrix(prof_dir)
        comm_analysis = analyze_communication(comm_ops, comm_matrix)
        print(render_comm_analysis(comm_analysis))

        per_layer = estimate_per_layer_comm(comm_ops, num_layers=args.num_layers)
        if per_layer:
            print("\n### Estimated Per-Layer Communication\n")
            print("| Comm Type | Ops/Layer | Avg/Op(ms) | Total/Layer(ms) |")
            print("| --- | ---: | ---: | ---: |")
            for ctype, stats in per_layer.items():
                print(
                    f"| {ctype} | {stats['ops_per_layer']} | "
                    f"{stats['avg_per_op_ms']:.4f} | {stats['total_per_layer_ms']:.4f} |"
                )


def cmd_compare(args: argparse.Namespace) -> None:
    """Compare two profiling runs (e.g., different ranks or configs)."""
    root_a = Path(args.input_a).resolve()
    root_b = Path(args.input_b).resolve()

    dirs_a = discover_profiling_dirs(root_a)
    dirs_b = discover_profiling_dirs(root_b)

    if not dirs_a or not dirs_b:
        print("ERROR: Need profiling directories in both inputs.")
        sys.exit(1)

    # Compare first dir from each
    dir_a, dir_b = dirs_a[0], dirs_b[0]

    print(render_header("Comparison Report"))
    print(f"  A: {dir_a.name}")
    print(f"  B: {dir_b.name}")

    stats_a = load_op_statistic(dir_a)
    stats_b = load_op_statistic(dir_b)
    step_a = load_step_trace(dir_a)
    step_b = load_step_trace(dir_b)

    # Compare overlap metrics
    if step_a and step_b:
        metrics_a = compute_overlap_metrics(step_a)
        metrics_b = compute_overlap_metrics(step_b)
        print("\n### Overlap Metrics Comparison\n")
        print("| Metric | A | B | Delta |")
        print("| --- | ---: | ---: | ---: |")
        for key in ["total_us", "compute_us", "comm_exposed_us", "overlap_us", "free_us"]:
            va = metrics_a.get(key, 0)
            vb = metrics_b.get(key, 0)
            delta = vb - va
            print(f"| {key} | {va/1000:.2f}ms | {vb/1000:.2f}ms | {delta/1000:+.2f}ms |")

    # Compare category breakdown
    analysis_a = analyze_op_breakdown(stats_a)
    analysis_b = analyze_op_breakdown(stats_b)
    all_cats = set(analysis_a["categories"].keys()) | set(analysis_b["categories"].keys())
    print("\n### Category Comparison\n")
    print("| Category | A Time(ms) | A Share(%) | B Time(ms) | B Share(%) | Delta(ms) |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for cat in sorted(all_cats):
        sa = analysis_a["categories"].get(cat, {"total_us": 0, "ratio_pct": 0})
        sb = analysis_b["categories"].get(cat, {"total_us": 0, "ratio_pct": 0})
        delta = sb["total_us"] - sa["total_us"]
        print(
            f"| {cat} | {sa['total_us']/1000:.2f} | {sa['ratio_pct']:.1f} | "
            f"{sb['total_us']/1000:.2f} | {sb['ratio_pct']:.1f} | {delta/1000:+.2f} |"
        )


def cmd_stack(args: argparse.Namespace) -> None:
    """Extract communication call stacks from trace_view.json.

    Uses a two-phase approach for efficiency:
      1. Load communication.json to determine the most frequent comm types.
      2. Pass the target types into ``load_trace_comm_with_stacks()`` so that
         only relevant events are collected from the (potentially huge)
         trace_view.json.

    This enables precise identification of which source code location
    triggers each communication operation, which is critical for:
    1. Verifying static analysis results (Step 2 of SKILL.md)
    2. Detecting missing DBO hooks (Step 2.5)
    3. Understanding the exact communication flow in the model
    """
    root = Path(args.input).resolve()
    prof_dirs = discover_profiling_dirs(root)
    if not prof_dirs:
        print(f"ERROR: No Ascend profiling directories found under {root}")
        sys.exit(1)

    top_comm_types_k = args.top_comm_types
    max_stacks = args.max_stacks

    for prof_dir in prof_dirs:
        print(render_header(f"Communication Call Stacks: {prof_dir.name}"))
        print(render_metadata(prof_dir))

        # Phase 1: Load communication.json and determine target comm types
        comm_ops = load_communication_json(prof_dir)

        target_comm_types: Optional[Set[str]] = None
        if top_comm_types_k > 0 and comm_ops:
            target_comm_types = _determine_target_comm_types(
                comm_ops, top_k=top_comm_types_k,
            )
            if target_comm_types:
                print(f"\n### Filtering by top-{top_comm_types_k} communication types: "
                      f"{sorted(target_comm_types)}\n")
            else:
                print(f"\n### WARNING: No comm types with sufficient frequency found; "
                      f"extracting all types.\n")
                target_comm_types = None

        # Phase 2: Extract stacks from trace_view.json
        trace_comms = load_trace_comm_with_stacks(
            prof_dir,
            max_depth=args.max_depth,
            dedup=not args.no_dedup,
            target_comm_types=target_comm_types,
            max_stacks=max_stacks,
            comm_ops=comm_ops,
        )

        if not trace_comms:
            print("\n  WARNING: No communication events found in trace_view.json.")
            print("  Possible reasons:")
            print("    1. trace_view.json does not exist or is empty")
            print("    2. Profiling was collected without python tracing enabled")
            print("    3. No communication ops were executed during profiling")
            continue

        print(render_comm_stacks(
            trace_comms,
            show_full_stack=args.full_stack,
            max_stack_depth=args.max_depth,
        ))

        # Cross-reference with communication.json for timing data
        if comm_ops:
            print("\n### Cross-Reference: Stack Locations vs communication.json Timing\n")
            print("| Code Location | Comm Type | Avg Elapse(ms) | Count | Overlap Priority |")
            print("| --- | --- | ---: | ---: | --- |")

            # Group trace_comms by code_location
            by_loc: Dict[str, TraceCommEvent] = {}
            for ev in trace_comms:
                loc = ev.code_location or "(unknown)"
                if loc not in by_loc:
                    by_loc[loc] = ev

            # Match with communication.json timing by comm_type
            comm_by_type: Dict[str, List[CommOp]] = defaultdict(list)
            for op in comm_ops:
                comm_by_type[op.comm_type].append(op)

            for loc, ev in by_loc.items():
                # Map trace comm_type to communication.json comm_type
                ctype = ev.comm_type
                matching_ops = comm_by_type.get(ctype, [])
                if matching_ops:
                    avg_ms = sum(op.elapse_ms for op in matching_ops) / len(matching_ops)
                    count = len(matching_ops)
                    if avg_ms * 1000 > 1000:
                        priority = "🔴 P0 (>1000μs)"
                    elif avg_ms * 1000 > 500:
                        priority = "🟠 P1 (>500μs)"
                    elif avg_ms * 1000 > 100:
                        priority = "🟡 P2 (>100μs)"
                    elif avg_ms * 1000 > 50:
                        priority = "⚪ P3 (>50μs)"
                    else:
                        priority = "⬜ P4 (<50μs)"
                    print(f"| `{loc}` | {ctype} | {avg_ms:.4f} | {count} | {priority} |")
                else:
                    print(f"| `{loc}` | {ctype} | N/A | 0 | — |")

        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="analyze_ascend_profiling.py",
        description=(
            "Ascend NPU Profiling Analyzer for vllm-ascend DBO overlap template development. "
            "Parses Ascend Profiler output and generates structured reports."
        ),
    )
    sub = parser.add_subparsers(dest="command")

    # triage
    p_triage = sub.add_parser("triage", help="Compact triage report with all analyses.")
    p_triage.add_argument("--input", required=True, help="Profiling directory path.")
    p_triage.add_argument("--top-k", type=int, default=30, help="Number of top ops to show.")
    p_triage.add_argument("--num-layers", type=int, default=10, help="Estimated number of decoder layers for per-layer analysis.")

    # breakdown
    p_break = sub.add_parser("breakdown", help="Detailed kernel breakdown.")
    p_break.add_argument("--input", required=True, help="Profiling directory path.")
    p_break.add_argument("--top-k", type=int, default=50, help="Number of top ops to show.")

    # comm
    p_comm = sub.add_parser("comm", help="Communication analysis.")
    p_comm.add_argument("--input", required=True, help="Profiling directory path.")
    p_comm.add_argument("--num-layers", type=int, default=10, help="Estimated number of decoder layers.")

    # compare
    p_compare = sub.add_parser("compare", help="Compare two profiling runs.")
    p_compare.add_argument("--input-a", required=True, help="First profiling directory.")
    p_compare.add_argument("--input-b", required=True, help="Second profiling directory.")

    # stack
    p_stack = sub.add_parser(
        "stack",
        help="Extract communication call stacks from trace_view.json.",
    )
    p_stack.add_argument("--input", required=True, help="Profiling directory path.")
    p_stack.add_argument(
        "--max-depth", type=int, default=20,
        help="Maximum call stack depth to trace (default: 20).",
    )
    p_stack.add_argument(
        "--full-stack", action="store_true",
        help="Show full call stack including framework internals (torch, torch_npu).",
    )
    p_stack.add_argument(
        "--no-dedup", action="store_true",
        help="Do not deduplicate events with identical call stacks.",
    )
    p_stack.add_argument(
        "--top-comm-types", type=int, default=3,
        help="Only extract stacks for top-K most frequent comm types (0 = all, default: 3). "
             "This optimization significantly reduces memory usage and processing time "
             "for large trace_view.json files by filtering early.",
    )
    p_stack.add_argument(
        "--max-stacks", type=int, default=50,
        help="Maximum number of unique stacks to collect per extraction (default: 50). "
             "Enables early termination once enough unique stacks are found. "
             "Set to 0 to disable the limit.",
    )

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    dispatch = {
        "triage": cmd_triage,
        "breakdown": cmd_breakdown,
        "comm": cmd_comm,
        "compare": cmd_compare,
        "stack": cmd_stack,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
