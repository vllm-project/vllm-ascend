#!/usr/bin/env python3
"""Run one reproducible 310P DFlash eager or Piecewise acceptance group."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

REPO = Path("/home/whn/vllm-ascend-openspec-fix")
UPSTREAM = Path("/home/whn/vllm-v0.24.0-clean")
DATASET = Path("/home/qzh/datasets/gsm8k/test_vllm.jsonl")
EXPECTED_UPSTREAM_HEAD = "ee0da84ab9e04ac7610e28580af62c365e898389"
CAPTURE_SIZES = (64, 32)

MODEL_PRESETS = {
    "9b": {
        "target": "/home/models/Qwen3.5-9B",
        "draft": "/home/models/Qwen3.5-9B-DFlash",
        "served_name": "Qwen3.5-9B-DFlash",
        "quantization": None,
    },
    "35b": {
        "target": "/home/models/Qwen3.6-35B-A3B-w8a8",
        "draft": "/home/models/Qwen3.6-35B-A3B-DFlash",
        "served_name": "Qwen3.6-35B-A3B-w8a8",
        "quantization": "ascend",
    },
}


@dataclass(frozen=True)
class GraphEvidence:
    target_capture_by_rank: dict[str, int]
    draft_capture_by_rank: dict[str, int]
    target_replay_by_rank: dict[str, int]
    draft_replay_by_rank: dict[str, int]
    contract_errors: int
    requested_capture_sizes_present: bool


class PairComparisonError(RuntimeError):
    def __init__(self, message: str, comparison: dict[str, Any]) -> None:
        super().__init__(message)
        self.comparison = comparison


def run_checked(command: list[str], *, cwd: Path = REPO) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return completed.stdout.strip()


def git_head(path: Path) -> str:
    return run_checked(["git", "rev-parse", "HEAD"], cwd=path)


def git_status(path: Path) -> list[str]:
    output = run_checked(["git", "status", "--short"], cwd=path)
    return output.splitlines() if output else []


def installed_vllm_ascend_origin() -> str:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import vllm_ascend; print(vllm_ascend.__file__)",
        ],
        cwd=Path("/home/whn"),
        env=environment,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    return completed.stdout.strip()


def parse_devices(raw: str, tp: int) -> tuple[int, ...]:
    devices = tuple(int(value) for value in raw.split(","))
    if len(devices) != tp or len(set(devices)) != tp:
        raise ValueError(f"--devices must contain exactly {tp} unique card IDs")
    return devices


def assert_cards_idle(devices: tuple[int, ...]) -> str:
    output = run_checked(["npu-smi", "info"])
    usage: dict[int, int] = {}
    pattern = re.compile(r"\|\s+\d+\s+(\d+)\s+\|\s+[0-9a-fA-F:.]+\s+\|\s+\d+\s+(\d+)\s*/\s*\d+")
    for match in pattern.finditer(output):
        usage[int(match.group(1))] = int(match.group(2))
    missing = [device for device in devices if device not in usage]
    busy = {device: usage[device] for device in devices if usage.get(device, 1 << 30) >= 4096}
    if missing or busy:
        raise RuntimeError(f"selected cards are not idle: missing={missing}, memory_mib={busy}")
    return output


def port_is_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def wait_for_health(port: int, process: subprocess.Popen[str], timeout: int) -> None:
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{port}/health"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited before health check: rc={process.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except OSError:
            pass
        time.sleep(2)
    raise TimeoutError(f"server did not become healthy within {timeout}s")


def stop_process_group(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=25)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def count_by_rank(log_text: str, event: str, component: str, tp: int) -> dict[str, int]:
    return {
        str(rank): len(
            re.findall(
                rf"event={event} component={component} rank={rank}\b",
                log_text,
            )
        )
        for rank in range(tp)
    }


def collect_graph_evidence(log_text: str, tp: int) -> GraphEvidence:
    return GraphEvidence(
        target_capture_by_rank=count_by_rank(log_text, "capture", "target", tp),
        draft_capture_by_rank=count_by_rank(log_text, "capture", "draft", tp),
        target_replay_by_rank=count_by_rank(log_text, "replay", "target", tp),
        draft_replay_by_rank=count_by_rank(log_text, "replay", "draft", tp),
        contract_errors=log_text.count("graph input contract failed"),
        requested_capture_sizes_present=(
            "cudagraph_capture_sizes': [32, 64]" in log_text or '"cudagraph_capture_sizes":[64,32]' in log_text
        ),
    )


def tokenize_outputs(benchmark: dict[str, Any], tokenizer_path: str) -> list[list[int]]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    return [tokenizer.encode(text, add_special_tokens=False) for text in benchmark["generated_texts"]]


def build_server_command(
    preset: dict[str, str | None],
    mode: str,
    tp: int,
    port: int,
) -> list[str]:
    command = [
        "vllm",
        "serve",
        str(preset["target"]),
        "--served-model-name",
        str(preset["served_name"]),
        "--dtype",
        "float16",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--tensor-parallel-size",
        str(tp),
        "--gpu-memory-utilization",
        "0.85",
        "--max-num-seqs",
        "16",
        "--max-num-batched-tokens",
        "1280",
        "--max-model-len",
        "8192",
        "--trust-remote-code",
        "--enable-prefix-caching",
        "--enable-chunked-prefill",
        "--async-scheduling",
        "--safetensors-load-strategy",
        "eager",
        "--limit-mm-per-prompt",
        '{"image":0,"video":0}',
        "--speculative-config",
        json.dumps(
            {
                "method": "dflash",
                "model": preset["draft"],
                "draft_tensor_parallel_size": tp,
                "num_speculative_tokens": 15,
            },
            separators=(",", ":"),
        ),
        "--additional-config",
        '{"ascend_compilation_config":{"enable_npugraph_ex":false,"fuse_norm_quant":false}}',
    ]
    if preset["quantization"]:
        command.extend(["--quantization", str(preset["quantization"])])
    compilation = {"cudagraph_mode": "NONE"}
    if mode == "piecewise":
        compilation = {
            "cudagraph_mode": "PIECEWISE",
            "cudagraph_capture_sizes": list(CAPTURE_SIZES),
        }
    command.extend(
        [
            "--compilation-config",
            json.dumps(compilation, separators=(",", ":")),
        ]
    )
    return command


def build_benchmark_command(
    preset: dict[str, str | None],
    port: int,
    concurrency: int,
    result_dir: Path,
    result_name: str,
    warmups: int,
) -> list[str]:
    return [
        "vllm",
        "bench",
        "serve",
        "--backend",
        "openai",
        "--base-url",
        f"http://127.0.0.1:{port}",
        "--endpoint",
        "/v1/completions",
        "--model",
        str(preset["served_name"]),
        "--tokenizer",
        str(preset["target"]),
        "--dataset-name",
        "custom",
        "--dataset-path",
        str(DATASET),
        "--num-prompts",
        "16",
        "--custom-output-len",
        "256",
        "--max-concurrency",
        str(concurrency),
        "--temperature",
        "0",
        "--ignore-eos",
        "--disable-shuffle",
        "--num-warmups",
        str(warmups),
        "--save-result",
        "--save-detailed",
        "--result-dir",
        str(result_dir),
        "--result-filename",
        result_name,
    ]


def require_piecewise_graphs(evidence: GraphEvidence) -> None:
    counters = (
        evidence.target_capture_by_rank,
        evidence.draft_capture_by_rank,
        evidence.target_replay_by_rank,
        evidence.draft_replay_by_rank,
    )
    if any(value <= 0 for counter in counters for value in counter.values()):
        raise RuntimeError(f"missing target/draft capture/replay evidence: {evidence}")
    if evidence.contract_errors:
        raise RuntimeError(f"graph input contract failures: {evidence.contract_errors}")
    if not evidence.requested_capture_sizes_present:
        raise RuntimeError("the server log does not preserve capture sizes [64,32]")


def compare_pair(current: dict[str, Any], eager_path: Path) -> dict[str, Any]:
    eager = json.loads(eager_path.read_text(encoding="utf-8"))
    token_mismatch_details = []
    for index, (eager_ids, current_ids) in enumerate(
        zip(eager["generated_token_ids"], current["generated_token_ids"], strict=True)
    ):
        if eager_ids == current_ids:
            continue
        common_length = min(len(eager_ids), len(current_ids))
        first_difference = next(
            (token_index for token_index in range(common_length) if eager_ids[token_index] != current_ids[token_index]),
            common_length,
        )
        token_mismatch_details.append(
            {
                "request_index": index,
                "first_differing_token_index": first_difference,
                "eager_token": (eager_ids[first_difference] if first_difference < len(eager_ids) else None),
                "piecewise_token": (current_ids[first_difference] if first_difference < len(current_ids) else None),
                "eager_length": len(eager_ids),
                "piecewise_length": len(current_ids),
            }
        )
    token_mismatches = [detail["request_index"] for detail in token_mismatch_details]
    eager_benchmark = eager["benchmark"]
    current_benchmark = current["benchmark"]
    throughput_delta = current_benchmark["output_throughput"] / eager_benchmark["output_throughput"] - 1
    acceptance_ratio = (
        current_benchmark["spec_decode_acceptance_length"] / eager_benchmark["spec_decode_acceptance_length"]
    )
    comparison = {
        "token_mismatch_indices": token_mismatches,
        "token_mismatch_details": token_mismatch_details,
        "throughput_delta": throughput_delta,
        "acceptance_length_ratio": acceptance_ratio,
        "acceptance_rate_delta": current_benchmark["spec_decode_acceptance_rate"]
        - eager_benchmark["spec_decode_acceptance_rate"],
    }
    if token_mismatches:
        first = token_mismatch_details[0]
        raise PairComparisonError(
            "generated token mismatch at "
            f"request {first['request_index']} token "
            f"{first['first_differing_token_index']}",
            comparison,
        )
    if throughput_delta < -0.15:
        raise PairComparisonError(
            f"Piecewise throughput delta {throughput_delta:.3f} is below -0.15",
            comparison,
        )
    if acceptance_ratio < 0.90:
        raise PairComparisonError(
            f"Piecewise acceptance ratio {acceptance_ratio:.3f} is below 0.90",
            comparison,
        )
    if comparison["acceptance_rate_delta"] < -5.0:
        raise PairComparisonError(
            "Piecewise acceptance rate is more than 5 percentage points below eager",
            comparison,
        )
    return comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_PRESETS, required=True)
    parser.add_argument("--mode", choices=("eager", "piecewise"), required=True)
    parser.add_argument("--tp", type=int, choices=(1, 2), required=True)
    parser.add_argument("--concurrency", type=int, choices=(1, 4), required=True)
    parser.add_argument("--devices", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--warmups", type=int, choices=range(0, 5), default=1)
    parser.add_argument("--health-timeout", type=int, default=420)
    parser.add_argument("--compare-to", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    preset = MODEL_PRESETS[args.model]
    devices = parse_devices(args.devices, args.tp)
    run_name = f"{args.model}-tp{args.tp}-{args.mode}-c{args.concurrency}"
    result_dir = args.result_root / run_name
    result_dir.mkdir(parents=True, exist_ok=False)
    server_log = result_dir / "server.log"
    benchmark_log = result_dir / "benchmark.out"
    benchmark_name = "benchmark.json"
    benchmark_path = result_dir / benchmark_name

    if git_head(UPSTREAM) != EXPECTED_UPSTREAM_HEAD or git_status(UPSTREAM):
        raise RuntimeError("frozen upstream vLLM checkout changed")
    if port_is_open(args.port):
        raise RuntimeError(f"port {args.port} is already in use")
    npu_snapshot = assert_cards_idle(devices)
    (result_dir / "npu-before.txt").write_text(npu_snapshot, encoding="utf-8")

    server_command = build_server_command(preset, args.mode, args.tp, args.port)
    benchmark_command = build_benchmark_command(
        preset,
        args.port,
        args.concurrency,
        result_dir,
        benchmark_name,
        args.warmups,
    )
    manifest = {
        "repo_head": git_head(REPO),
        "upstream_head": git_head(UPSTREAM),
        "vllm_ascend_version": importlib.metadata.version("vllm-ascend"),
        "vllm_ascend_import_origin": installed_vllm_ascend_origin(),
        "vllm_version": importlib.metadata.version("vllm"),
        "dataset": str(DATASET),
        "devices": devices,
        "server_command": server_command,
        "benchmark_command": benchmark_command,
        "environment": {
            "VLLM_LOGGING_LEVEL": "DEBUG",
            "ASCEND_RT_VISIBLE_DEVICES": args.devices,
        },
    }
    (result_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    environment = os.environ.copy()
    environment.update(
        {
            "VLLM_LOGGING_LEVEL": "DEBUG",
            "ASCEND_RT_VISIBLE_DEVICES": args.devices,
        }
    )
    process: subprocess.Popen[str] | None = None
    try:
        with server_log.open("w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                server_command,
                cwd=result_dir,
                env=environment,
                text=True,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            (result_dir / "server.pid").write_text(str(process.pid), encoding="utf-8")
            wait_for_health(args.port, process, args.health_timeout)
            with benchmark_log.open("w", encoding="utf-8") as output_file:
                subprocess.run(
                    benchmark_command,
                    cwd=result_dir,
                    env=environment,
                    check=True,
                    text=True,
                    stdout=output_file,
                    stderr=subprocess.STDOUT,
                )
    finally:
        if process is not None:
            stop_process_group(process)

    if port_is_open(args.port):
        raise RuntimeError(f"port {args.port} remains open after cleanup")
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    if benchmark["completed"] != 16 or benchmark["failed"] != 0:
        raise RuntimeError(f"formal requests failed: completed={benchmark['completed']} failed={benchmark['failed']}")
    log_text = server_log.read_text(encoding="utf-8", errors="replace")
    evidence = collect_graph_evidence(log_text, args.tp)
    if args.mode == "piecewise":
        require_piecewise_graphs(evidence)
        if benchmark["spec_decode_acceptance_length"] < 5.0:
            raise RuntimeError("Piecewise acceptance length is below 5.0")

    summary = {
        "run_name": run_name,
        "status": "passed",
        "benchmark": benchmark,
        "generated_token_ids": tokenize_outputs(benchmark, str(preset["target"])),
        "graph_evidence": asdict(evidence),
        "comparison": None,
    }
    comparison_error = None
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
    (result_dir / "npu-after.txt").write_text(
        run_checked(["npu-smi", "info"]),
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
