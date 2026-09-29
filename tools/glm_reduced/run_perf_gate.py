# SPDX-License-Identifier: Apache-2.0
"""Run the registered 11-layer GLM5.x offline performance gate once.

One launch per configuration. A fixed warmup is excluded from timing and each
workload is measured a fixed number of times; the median is compared against the
committed baseline with explicit thresholds. Nothing is recalibrated here, no
baseline is ever written, and evidence directories are never overwritten.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import os
import platform
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

from .logits_gate import DATA as LOGITS_DATA
from .perf_gate import DATA, FORMAT, compare, load_registered, read_json
from .run_logits_gate import provision, validate_checkpoint, write_json

VOCAB = 154880  # token ids must stay below the checkpoint vocabulary


def build_prompts(input_tokens: int, concurrency: int, seed: int) -> list[dict]:
    return [
        {"prompt_token_ids": [1 + (seed + row * 131 + i * 17) % (VOCAB - 1) for i in range(input_tokens)]}
        for row in range(concurrency)
    ]


def runtime_identity() -> dict:
    packages = {}
    for name in ("vllm", "vllm-ascend", "torch", "torch-npu", "transformers", "tokenizers"):
        try:
            packages[name] = importlib.metadata.version(name)
        except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
            packages[name] = "UNAVAILABLE: " + str(exc)
    identity = {
        "machine": platform.machine(),
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "python": sys.version,
        "packages": packages,
        "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
    }
    for name in ("vllm", "vllm_ascend"):
        module = sys.modules.get(name)
        if module and getattr(module, "__file__", None):
            root = Path(module.__file__).resolve().parent.parent
            commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True)
            identity[name] = {
                "path": str(root),
                "commit": commit.stdout.strip() if commit.returncode == 0 else None,
            }
    return identity


def measure_once(engine, prompts, sampling, output_tokens, label):
    """Submit every request, drive the engine synchronously, time each token arrival."""
    events = {}
    started = time.perf_counter()
    for row, prompt in enumerate(prompts):
        request_id = label + "-" + str(row)
        events[request_id] = {
            "submitted": time.perf_counter(),
            "first": None,
            "last": None,
            "finished": None,
            "tokens": 0,
        }
        engine.add_request(request_id, prompt, sampling)
    while engine.has_unfinished_requests():
        for item in engine.step():
            now = time.perf_counter()
            event = events.get(item.request_id)
            if event is None:
                raise AssertionError("engine returned unknown request " + repr(item.request_id))
            count = len(item.outputs[0].token_ids)
            if count > event["tokens"]:
                if event["first"] is None:
                    event["first"] = now
                event["last"] = now
                event["tokens"] = count
            if item.finished:
                event["finished"] = now
    elapsed = time.perf_counter() - started

    requests = []
    for request_id, event in events.items():
        if event["tokens"] != output_tokens or event["finished"] is None or event["first"] is None:
            raise AssertionError(
                request_id + ": produced " + str(event["tokens"]) + " tokens, expected exactly " + str(output_tokens)
            )
        tpot = (event["last"] - event["first"]) / (output_tokens - 1) if output_tokens > 1 else 0.0
        requests.append(
            {
                "request_id": request_id,
                "ttft_seconds": event["first"] - event["submitted"],
                "tpot_seconds": tpot,
                "request_seconds": event["finished"] - event["submitted"],
            }
        )
    total_tokens = len(prompts) * output_tokens
    if not all(all(v >= 0 for v in (r["ttft_seconds"], r["tpot_seconds"])) for r in requests):
        raise AssertionError("negative latency sample")
    return {
        "seconds": elapsed,
        "output_tokens_per_second": total_tokens / elapsed,
        "total_output_tokens": total_tokens,
        "requests": requests,
    }


def _median(values):
    ordered = sorted(values)
    count = len(ordered)
    if count == 0:
        raise ValueError("cannot aggregate an empty sample list")
    if count % 2:
        return ordered[count // 2]
    return (ordered[count // 2 - 1] + ordered[count // 2]) / 2


def _aggregate(iterations) -> dict:
    elapsed = [item["seconds"] for item in iterations]
    throughput = [item["output_tokens_per_second"] for item in iterations]
    ttfts = [r["ttft_seconds"] for item in iterations for r in item["requests"]]
    tpots = [r["tpot_seconds"] for item in iterations for r in item["requests"]]
    return {
        "seconds": {"median": _median(elapsed), "min": min(elapsed), "max": max(elapsed)},
        "output_tokens_per_second": {
            "median": _median(throughput),
            "min": min(throughput),
            "max": max(throughput),
        },
        "ttft_seconds": {"median": _median(ttfts), "min": min(ttfts), "max": max(ttfts)},
        "tpot_seconds": {"median": _median(tpots), "min": min(tpots), "max": max(tpots)},
    }


def collect(model: Path, baseline: dict, *, hardware: str, report: Path) -> dict:
    """Run the baseline's fixed workloads once and return a candidate record."""
    from vllm import LLM, SamplingParams  # noqa: PLC0415 - NPU runtime boundary

    identity = baseline["identity"]
    settings = dict(identity["engine_settings"])
    settings["model"] = str(model)
    workloads = [tuple(item) for item in identity["workloads"]]
    warmup = identity["warmup_iterations"]
    measured = identity["measured_iterations"]

    llm = LLM(**settings)
    try:
        runs = []
        for input_tokens, output_tokens, concurrency in workloads:
            prompts = build_prompts(input_tokens, concurrency, int(settings.get("seed", 1024)))
            sampling = SamplingParams(temperature=0, max_tokens=output_tokens, ignore_eos=True, detokenize=False)
            label = "perf-" + str(input_tokens) + "-" + str(output_tokens) + "-" + str(concurrency)
            for _ in range(warmup):
                measure_once(llm.llm_engine, prompts, sampling, output_tokens, label + "-warmup")
            iterations = [
                measure_once(llm.llm_engine, prompts, sampling, output_tokens, label + "-m" + str(index))
                for index in range(measured)
            ]
            runs.append(
                {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "concurrency": concurrency,
                    "warmup_iterations": warmup,
                    "measured_iterations": measured,
                    "iterations": iterations,
                    "aggregate": _aggregate(iterations),
                }
            )
        payload = {
            "format": FORMAT,
            "status": "SINGLE_ROUND_OBSERVATION",
            "hardware": hardware,
            "identity": {
                "model": identity["model"],
                "hardware": hardware,
                "engine_settings": settings,
                "workloads": [list(item) for item in workloads],
                "warmup_iterations": warmup,
                "measured_iterations": measured,
                "runtime": runtime_identity(),
                "run_id": uuid.uuid4().hex,
            },
            "runs": runs,
        }
        write_json(report / "candidate.json", payload)
        return payload
    finally:
        with contextlib.suppress(Exception):
            llm.llm_engine.engine_core.shutdown()


def _run(baseline: dict, model: Path, report: Path, hardware: str, options) -> dict:
    machine = platform.machine()
    if machine not in hardware:
        raise ValueError(
            "Runner architecture " + machine + " does not match hardware tag " + hardware + "; refusing to compare"
        )
    if hardware != baseline["identity"]["hardware"]:
        raise ValueError(
            "Hardware tag "
            + hardware
            + " differs from the pinned baseline "
            + baseline["identity"]["hardware"]
            + "; a mismatched runner fails closed"
        )
    candidate = collect(model, baseline, hardware=hardware, report=report)
    result = compare(
        baseline,
        candidate,
        min_throughput_fraction=options.get("min_throughput_fraction"),
        max_latency_fraction=options.get("max_latency_fraction"),
    )
    write_json(report / "result.json", result)
    if not result["ok"]:
        raise AssertionError("GLM5X_PERF_GATE_FAILED: " + "; ".join(result["problems"]))
    print("GLM5X_PERF_GATE_OK workloads=" + str(len(candidate["runs"])), flush=True)
    return result


def run_nightly(config) -> None:
    """SingleNodeConfigLoader entry point; never starts an OpenAI server."""
    options = config.extra_config["glm5x_perf_gate"]
    case = read_json(LOGITS_DATA / "case.json")
    if config.model != case["model"]:
        raise ValueError("This model requires its own registered baseline")
    report = Path("benchmark_results") / config.name / uuid.uuid4().hex
    report.mkdir(parents=True, exist_ok=False)
    try:
        model = provision(options, config.model)
        baseline = load_registered(options["baseline"])
        validate_checkpoint(Path(model), case)
        _run(baseline, Path(model), report, str(options["hardware"]), options)
    except Exception as exc:
        write_json(report / "failure.json", {"error": str(exc), "retry": "disabled"})
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--baseline", default=str(DATA / "baseline-167-eager.json"))
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--hardware", required=True)
    parser.add_argument("--min-throughput-fraction", type=float, required=True)
    parser.add_argument("--max-latency-fraction", type=float, required=True)
    args = parser.parse_args(argv)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    if any((args.report_dir / name).exists() for name in ("candidate.json", "result.json", "failure.json")):
        raise ValueError("Refusing to overwrite evidence; use a fresh report directory")
    options = {
        "min_throughput_fraction": args.min_throughput_fraction,
        "max_latency_fraction": args.max_latency_fraction,
    }
    try:
        baseline = load_registered(args.baseline)
        validate_checkpoint(args.model, read_json(LOGITS_DATA / "case.json"))
        _run(baseline, args.model, args.report_dir, args.hardware, options)
        return 0
    except Exception as exc:
        write_json(args.report_dir / "failure.json", {"error": str(exc), "retry": "disabled"})
        raise


if __name__ == "__main__":
    sys.exit(main())
