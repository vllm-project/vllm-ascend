#!/usr/bin/env python3
# ruff: noqa: E402
"""Run one frozen 310P DFlash eager or FULL_DECODE_ONLY acceptance group."""

from __future__ import annotations

import os
import sys

# Direct execution adds ``tools/`` ahead of the standard library. The repository
# also has ``tools/bisect``, which must not shadow Python's ``bisect`` module.
_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if sys.path and os.path.abspath(sys.path[0]) == _TOOLS_DIR:
    sys.path.pop(0)
_REPO_ROOT = os.path.dirname(_TOOLS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import argparse
import concurrent.futures
import json
import re
import subprocess
import threading
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tools.run_310p_dflash_piecewise_acceptance import (
    DATASET,
    EXPECTED_UPSTREAM_HEAD,
    REPO,
    UPSTREAM,
    assert_cards_idle,
    build_benchmark_command,
    git_head,
    git_status,
    parse_devices,
    port_is_open,
    run_checked,
    stop_process_group,
    tokenize_outputs,
    wait_for_health,
)

MODEL_PRESETS = {
    "4b": {
        "target": "/home/models/Qwen3.5-4B",
        "draft": "/home/models/Qwen3.5-4B-DFlash",
        "served_name": "Qwen3.5-4B",
        "quantization": None,
    },
    "35b": {
        "target": "/home/models/Qwen3.6-35B-A3B-w8a8",
        "draft": "/home/models/Qwen3.6-35B-A3B-DFlash",
        "served_name": "Qwen3.6-35B-A3B-w8a8",
        "quantization": "ascend",
    },
}
CAPTURE_SIZES = (160, 16)
C10_CAPTURE_SIZES = CAPTURE_SIZES
GRAPH_PROOF_WARMUPS = 4
GRAPH_PROOF_OUTPUT_LENGTHS = (128, 80, 16, 16)
SOURCE_PYTHONPATH = f"{REPO}:{UPSTREAM}"


@dataclass(frozen=True)
class GraphEvidence:
    manifest_descriptors_by_component_rank: dict[str, tuple[int, ...]]
    replay_descriptors_by_component_rank: dict[str, tuple[int, ...]]
    native_graph_descriptors_by_component_rank: dict[str, tuple[int, ...]]
    manifest_complete_ranks: tuple[int, ...]
    expected_none_dispatches: int
    runtime_full_dispatches: int
    safety_errors: tuple[str, ...]


class PairComparisonError(RuntimeError):
    def __init__(self, message: str, comparison: dict[str, Any]) -> None:
        super().__init__(message)
        self.comparison = comparison


def _source_environment() -> dict[str, str]:
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = SOURCE_PYTHONPATH + (f":{existing}" if existing else "")
    return environment


def source_runtime_identity() -> dict[str, str]:
    code = """
import importlib.metadata
import json
import acl
import vllm
import vllm_ascend
print(json.dumps({
    "acl_origin": acl.__file__,
    "vllm_version": importlib.metadata.version("vllm"),
    "vllm_origin": vllm.__file__,
    "vllm_ascend_version": importlib.metadata.version("vllm-ascend"),
    "vllm_ascend_origin": vllm_ascend.__file__,
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path("/home/whn"),
        env=_source_environment(),
        check=True,
        text=True,
        capture_output=True,
    )
    identity = json.loads(completed.stdout.splitlines()[-1])
    if not str(identity["vllm_ascend_origin"]).startswith(str(REPO)):
        raise RuntimeError(f"vllm-ascend is not imported from the source tree: {identity}")
    if not str(identity["vllm_origin"]).startswith(str(UPSTREAM)):
        raise RuntimeError(f"vLLM is not imported from the frozen source tree: {identity}")
    return identity


def _descriptors_by_component_rank(log_text: str, event_pattern: str, tp: int) -> dict[str, tuple[int, ...]]:
    evidence: dict[str, tuple[int, ...]] = {}
    for component in ("target", "draft"):
        for rank in range(tp):
            pattern = re.compile(
                event_pattern
                + rf".*?component={component} rank={rank}\b.*?"
                + r"descriptor=BatchDescriptor\(num_tokens=(\d+),"
            )
            evidence[f"{component}-rank{rank}"] = tuple(sorted({int(value) for value in pattern.findall(log_text)}))
    return evidence


def collect_graph_evidence(log_text: str, tp: int) -> GraphEvidence:
    operational_log = log_text.partition("[shutdown] EngineCore: trigger received signal=SIGTERM")[0]
    safety_markers = (
        "GraphInputContractError",
        "graph input contract failed",
        "eligible uniform decode selected NONE",
        "eligible uniform decode has no validated FULL descriptor",
        "Traceback (most recent call last):",
    )
    return GraphEvidence(
        manifest_descriptors_by_component_rank=_descriptors_by_component_rank(
            log_text,
            r"\[310p-dflash-full-decode-only/manifest\] event=record",
            tp,
        ),
        replay_descriptors_by_component_rank=_descriptors_by_component_rank(
            log_text,
            r"\[310p-dflash-graph\] event=replay",
            tp,
        ),
        native_graph_descriptors_by_component_rank=_descriptors_by_component_rank(
            log_text,
            r"\[310p-dflash-graph\] event=native-graph-dump",
            tp,
        ),
        manifest_complete_ranks=tuple(
            rank
            for rank in range(tp)
            if re.search(
                rf"\[310p-dflash-full-decode-only/manifest\] event=complete rank={rank}\b",
                log_text,
            )
        ),
        expected_none_dispatches=len(
            re.findall(
                r"\[310p-dflash-full-decode-only/dispatch\].*?"
                r"expected=NONE selected=NONE.*?"
                r"reason=(?:prefill|chunked_prefill|prefix_cache_transition|"
                r"mixed_or_nonuniform_decode)",
                log_text,
            )
        ),
        runtime_full_dispatches=len(
            re.findall(
                r"\[310p-dflash-full-decode-only/dispatch\].*?"
                r"expected=FULL selected=FULL.*?reason=uniform_dflash_decode",
                log_text,
            )
        ),
        safety_errors=tuple(marker for marker in safety_markers if marker in operational_log),
    )


def require_full_decode_graphs(
    evidence: GraphEvidence,
    capture_sizes: tuple[int, ...] = CAPTURE_SIZES,
) -> None:
    expected = set(capture_sizes)
    for key, descriptors in evidence.manifest_descriptors_by_component_rank.items():
        if set(descriptors) != expected:
            raise RuntimeError(f"incomplete manifest for {key}: {descriptors}")
    for key, descriptors in evidence.replay_descriptors_by_component_rank.items():
        if set(descriptors) != expected:
            raise RuntimeError(f"incomplete runtime replay for {key}: {descriptors}")
    for key, descriptors in evidence.native_graph_descriptors_by_component_rank.items():
        if set(descriptors) != expected:
            raise RuntimeError(f"incomplete native graph dump for {key}: {descriptors}")
    expected_ranks = tuple(range(len(evidence.manifest_descriptors_by_component_rank) // 2))
    if evidence.manifest_complete_ranks != expected_ranks:
        raise RuntimeError(
            "incomplete manifest-complete evidence: "
            f"expected={expected_ranks} actual={evidence.manifest_complete_ranks}"
        )
    if evidence.expected_none_dispatches <= 0:
        raise RuntimeError("missing expected NONE dispatch evidence")
    if evidence.runtime_full_dispatches <= 0:
        raise RuntimeError("missing runtime FULL dispatch evidence")
    if evidence.safety_errors:
        raise RuntimeError(f"graph safety failures: {evidence.safety_errors}")


def require_native_graph_files(
    native_graph_dir: Path,
    tp: int,
    capture_sizes: tuple[int, ...] = CAPTURE_SIZES,
) -> list[str]:
    paths = sorted(native_graph_dir.glob("*.json"))
    missing = []
    for component in ("target", "draft"):
        for rank in range(tp):
            for descriptor in capture_sizes:
                prefix = f"{component}-rank{rank}-tokens{descriptor}-"
                if not any(path.name.startswith(prefix) for path in paths):
                    missing.append(prefix)
    if missing:
        raise RuntimeError(f"missing native graph JSON files: {missing}")
    return [str(path) for path in paths]


def build_server_command(
    preset: dict[str, str | None],
    mode: str,
    tp: int,
    port: int,
    capture_sizes: tuple[int, ...] = CAPTURE_SIZES,
) -> list[str]:
    from tools.run_310p_dflash_piecewise_acceptance import build_server_command as build_piecewise_server

    command = build_piecewise_server(preset, "eager", tp, port)
    if mode == "full_decode_only":
        command[-1] = json.dumps(
            {
                "cudagraph_mode": "FULL_DECODE_ONLY",
                "cudagraph_capture_sizes": list(capture_sizes),
                "cudagraph_num_of_warmups": 0,
            },
            separators=(",", ":"),
        )
    return command


def _first_token_difference(eager_ids: list[int], current_ids: list[int]) -> int:
    common_length = min(len(eager_ids), len(current_ids))
    return next(
        (index for index in range(common_length) if eager_ids[index] != current_ids[index]),
        common_length,
    )


def compare_pair(current: dict[str, Any], eager_path: Path) -> dict[str, Any]:
    eager = json.loads(eager_path.read_text(encoding="utf-8"))
    eager_outputs = eager["generated_token_ids"]
    current_outputs = current["generated_token_ids"]
    mismatch_details = []
    if len(eager_outputs) != len(current_outputs):
        mismatch_details.append(
            {
                "request_count": {"eager": len(eager_outputs), "full_decode_only": len(current_outputs)},
            }
        )
    for request_index, (eager_ids, current_ids) in enumerate(zip(eager_outputs, current_outputs)):
        if eager_ids == current_ids:
            continue
        token_index = _first_token_difference(eager_ids, current_ids)
        mismatch_details.append(
            {
                "request_index": request_index,
                "first_differing_token_index": token_index,
                "eager_token": eager_ids[token_index] if token_index < len(eager_ids) else None,
                "full_decode_only_token": current_ids[token_index] if token_index < len(current_ids) else None,
                "eager_length": len(eager_ids),
                "full_decode_only_length": len(current_ids),
            }
        )

    eager_metrics = eager["benchmark"]
    current_metrics = current["benchmark"]
    request_ratio = current_metrics["request_throughput"] / eager_metrics["request_throughput"]
    output_ratio = current_metrics["output_throughput"] / eager_metrics["output_throughput"]
    acceptance_length = current_metrics["spec_decode_acceptance_length"]
    acceptance_ratio = acceptance_length / eager_metrics["spec_decode_acceptance_length"]
    acceptance_rate_delta = (
        current_metrics["spec_decode_acceptance_rate"] - eager_metrics["spec_decode_acceptance_rate"]
    )
    comparison = {
        "token_mismatch_details": mismatch_details,
        "request_throughput_ratio": request_ratio,
        "output_throughput_ratio": output_ratio,
        "acceptance_length": acceptance_length,
        "acceptance_length_ratio": acceptance_ratio,
        "acceptance_rate_delta_pp": acceptance_rate_delta,
    }
    if mismatch_details:
        first = mismatch_details[0]
        if "request_index" in first:
            message = (
                f"generated token mismatch at request {first['request_index']} "
                f"token {first['first_differing_token_index']}"
            )
        else:
            message = "generated output request count mismatch"
        raise PairComparisonError(message, comparison)
    if request_ratio < 0.85:
        raise PairComparisonError(f"request throughput ratio {request_ratio:.3f} is below 0.85", comparison)
    if output_ratio < 0.85:
        raise PairComparisonError(f"output throughput ratio {output_ratio:.3f} is below 0.85", comparison)
    if acceptance_length < 5.0:
        raise PairComparisonError(f"mean accepted length {acceptance_length:.3f} is below 5.0", comparison)
    if acceptance_ratio < 0.90:
        raise PairComparisonError(f"accepted length ratio {acceptance_ratio:.3f} is below 0.90", comparison)
    if acceptance_rate_delta < -5.0:
        raise PairComparisonError(
            f"acceptance rate delta {acceptance_rate_delta:.3f}pp is below -5pp",
            comparison,
        )
    return comparison


def run_graph_proof_warmups(port: int, served_name: str) -> list[dict[str, Any]]:
    barrier = threading.Barrier(GRAPH_PROOF_WARMUPS)

    def send_request(index: int, max_tokens: int) -> dict[str, Any]:
        payload = json.dumps(
            {
                "model": served_name,
                "prompt": f"310P FULL_DECODE_ONLY graph proof request {index}: solve 2+2.",
                "max_tokens": max_tokens,
                "temperature": 0,
                "ignore_eos": True,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        barrier.wait(timeout=20)
        with urllib.request.urlopen(request, timeout=180) as response:
            result = json.load(response)
        return {
            "request_index": index,
            "max_tokens": max_tokens,
            "completion_tokens": result["usage"]["completion_tokens"],
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=GRAPH_PROOF_WARMUPS) as executor:
        warmup_requests = enumerate(GRAPH_PROOF_OUTPUT_LENGTHS)
        futures = [executor.submit(send_request, index, max_tokens) for index, max_tokens in warmup_requests]
        results = [future.result() for future in futures]
    if any(result["completion_tokens"] != result["max_tokens"] for result in results):
        raise RuntimeError(f"graph proof warmup returned incomplete output: {results}")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_PRESETS, required=True)
    parser.add_argument("--mode", choices=("eager", "full_decode_only"), required=True)
    parser.add_argument("--tp", type=int, choices=(1, 2, 4), required=True)
    parser.add_argument("--concurrency", type=int, choices=(1, 10), required=True)
    parser.add_argument("--devices", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--health-timeout", type=int, default=600)
    parser.add_argument("--compare-to", type=Path)
    return parser.parse_args()


def _validate_scenario(model: str, tp: int, concurrency: int) -> None:
    valid = {("4b", 1), ("4b", 2), ("35b", 2), ("35b", 4)}
    if (model, tp) not in valid or concurrency not in (1, 10):
        raise ValueError(
            f"scenario is outside the frozen acceptance matrix: model={model} tp={tp} concurrency={concurrency}"
        )


def capture_sizes_for_scenario(
    model: str,
    tp: int,
    concurrency: int,
) -> tuple[int, ...]:
    _validate_scenario(model, tp, concurrency)
    return CAPTURE_SIZES


def main() -> int:
    args = parse_args()
    capture_sizes = capture_sizes_for_scenario(
        args.model,
        args.tp,
        args.concurrency,
    )
    expected_requests = 4 if args.concurrency == 1 else 20
    preset = MODEL_PRESETS[args.model]
    devices = parse_devices(args.devices, args.tp)
    run_name = f"{args.model}-tp{args.tp}-{args.mode}-c{args.concurrency}"
    result_dir = args.result_root / run_name
    result_dir.mkdir(parents=True, exist_ok=False)
    server_log = result_dir / "server.log"
    benchmark_log = result_dir / "benchmark.out"
    benchmark_path = result_dir / "benchmark.json"
    native_graph_dir = result_dir / "native-graphs"

    for key in ("target", "draft"):
        if not Path(str(preset[key])).is_dir():
            raise RuntimeError(f"model path does not exist: {preset[key]}")
    if not DATASET.is_file():
        raise RuntimeError(f"dataset does not exist: {DATASET}")
    if git_head(UPSTREAM) != EXPECTED_UPSTREAM_HEAD or git_status(UPSTREAM):
        raise RuntimeError("frozen upstream vLLM checkout changed")
    if port_is_open(args.port):
        raise RuntimeError(f"port {args.port} is already in use")

    runtime_identity = source_runtime_identity()
    npu_before = assert_cards_idle(devices)
    (result_dir / "npu-before.txt").write_text(npu_before, encoding="utf-8")
    server_command = build_server_command(
        preset,
        args.mode,
        args.tp,
        args.port,
        capture_sizes,
    )
    benchmark_command = build_benchmark_command(
        preset,
        args.port,
        args.concurrency,
        result_dir,
        "benchmark.json",
        0,
    )
    benchmark_command[benchmark_command.index("--num-prompts") + 1] = str(expected_requests)
    environment = _source_environment()
    environment.update(
        {
            "VLLM_LOGGING_LEVEL": "DEBUG",
            "ASCEND_RT_VISIBLE_DEVICES": args.devices,
        }
    )
    if args.mode == "full_decode_only":
        native_graph_dir.mkdir()
        environment["VLLM_ASCEND_DFLASH_GRAPH_DUMP_DIR"] = str(native_graph_dir)

    manifest = {
        "run_name": run_name,
        "repo_head": git_head(REPO),
        "repo_status": git_status(REPO),
        "upstream_head": git_head(UPSTREAM),
        "runtime_identity": runtime_identity,
        "dataset": {"path": str(DATASET), "examples": [0, expected_requests - 1]},
        "generation": {
            "num_prompts": expected_requests,
            "output_length": 256,
            "temperature": 0,
            "ignore_eos": True,
            "num_speculative_tokens": 15,
        },
        "graph": {
            "capture_sizes": capture_sizes,
            "acl_internal_warmups": 0,
            "excluded_graph_proof_warmups": GRAPH_PROOF_WARMUPS,
        },
        "devices": devices,
        "server_command": server_command,
        "benchmark_command": benchmark_command,
        "environment": {
            key: environment[key]
            for key in (
                "PYTHONPATH",
                "VLLM_LOGGING_LEVEL",
                "ASCEND_RT_VISIBLE_DEVICES",
                "VLLM_ASCEND_DFLASH_GRAPH_DUMP_DIR",
            )
            if key in environment
        },
    }
    (result_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    server_process: subprocess.Popen[str] | None = None
    benchmark_process: subprocess.Popen[str] | None = None
    failure: BaseException | None = None
    try:
        with server_log.open("w", encoding="utf-8") as server_output:
            server_process = subprocess.Popen(
                server_command,
                cwd=result_dir,
                env=environment,
                text=True,
                stdout=server_output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            (result_dir / "server.pid").write_text(str(server_process.pid), encoding="utf-8")
            wait_for_health(args.port, server_process, args.health_timeout)
            (result_dir / "npu-ready.txt").write_text(run_checked(["npu-smi", "info"]), encoding="utf-8")
            warmups = run_graph_proof_warmups(args.port, str(preset["served_name"]))
            (result_dir / "warmup.json").write_text(json.dumps(warmups, indent=2), encoding="utf-8")
            (result_dir / "npu-after-warmup.txt").write_text(
                run_checked(["npu-smi", "info"]),
                encoding="utf-8",
            )
            with benchmark_log.open("w", encoding="utf-8") as benchmark_output:
                benchmark_process = subprocess.Popen(
                    benchmark_command,
                    cwd=result_dir,
                    env=environment,
                    text=True,
                    stdout=benchmark_output,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                benchmark_return_code = benchmark_process.wait()
                if benchmark_return_code != 0:
                    raise RuntimeError(f"benchmark failed: rc={benchmark_return_code}")
            (result_dir / "npu-after-benchmark.txt").write_text(
                run_checked(["npu-smi", "info"]),
                encoding="utf-8",
            )
    except BaseException as exc:
        failure = exc
    finally:
        if benchmark_process is not None:
            stop_process_group(benchmark_process)
        if server_process is not None:
            stop_process_group(server_process)
        (result_dir / "npu-after-cleanup.txt").write_text(
            run_checked(["npu-smi", "info"]),
            encoding="utf-8",
        )

    if port_is_open(args.port):
        cleanup_error = RuntimeError(f"port {args.port} remains open after cleanup")
        if failure is None:
            failure = cleanup_error
    if failure is not None:
        (result_dir / "failure.json").write_text(
            json.dumps({"type": type(failure).__name__, "message": str(failure)}, indent=2),
            encoding="utf-8",
        )
        raise failure

    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    if benchmark["completed"] != expected_requests or benchmark["failed"] != 0:
        completed = benchmark["completed"]
        failed = benchmark["failed"]
        raise RuntimeError(f"formal requests failed: completed={completed} failed={failed}")
    generated_token_ids = tokenize_outputs(benchmark, str(preset["target"]))
    log_text = server_log.read_text(encoding="utf-8", errors="replace")
    evidence = collect_graph_evidence(log_text, args.tp)
    native_graph_files: list[str] = []
    if args.mode == "full_decode_only":
        require_full_decode_graphs(evidence, capture_sizes)
        native_graph_files = require_native_graph_files(
            native_graph_dir,
            args.tp,
            capture_sizes,
        )
        if benchmark["spec_decode_acceptance_length"] < 5.0:
            raise RuntimeError("FULL_DECODE_ONLY mean accepted length is below 5.0")

    summary = {
        "run_name": run_name,
        "status": "passed",
        "benchmark": benchmark,
        "generated_token_ids": generated_token_ids,
        "graph_evidence": asdict(evidence),
        "native_graph_files": native_graph_files,
        "comparison": None,
    }
    comparison_error: PairComparisonError | None = None
    if args.compare_to is not None:
        try:
            summary["comparison"] = compare_pair(summary, args.compare_to)
        except PairComparisonError as exc:
            comparison_error = exc
            summary["status"] = "failed"
            summary["failure"] = str(exc)
            summary["comparison"] = exc.comparison
    (result_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if comparison_error is not None:
        raise comparison_error
    print(json.dumps(summary["comparison"] or summary["graph_evidence"], indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"acceptance failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
