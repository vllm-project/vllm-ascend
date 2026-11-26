# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate and tune complete A3 KDA forwards from hidden states to output projection.

Candidate selection uses training timings only. The selected configuration is
then measured against the baseline with new, held-out timing samples. Neither
the winning training timing nor the best held-out sample is reported as speedup.
"""

import argparse
import gc
import importlib
import importlib.metadata
import json
import math
import platform
import statistics
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from full_reference import Config, full_reference, make_cache, make_inputs, make_rollback

ELEMENTWISE_LIMITS = {
    "output": (0.025, 0.002),
    "normalized": (0.025, 0.002),
    "correction": (1e-4, 1e-6),
    "kg": (0.01, 0.0001),
    "history": (0.0, 0.0),
}
RELATIVE_LIMITS = {"output": 0.012, "normalized": 0.012, "correction": 0.003, "kg": 0.003}
DEFAULT_CANDIDATES = "0:0,16:0,32:0,32:16,native:16:0,native:32:0,native:64:0,native:0:16"
MIDDLE_VARIANTS = {
    "legacy": {},
    "fused-prep": {"prepare_fused": True},
    "fused-prep-cv1": {"prepare_fused": True, "cv_slots": 1},
    "fused-prep-cv2": {"prepare_fused": True, "cv_slots": 2},
    "a5-style": {"middle": "a5-style"},
    "a5-style-parallel": {"middle": "a5-style-parallel"},
}
VARIANT_PAIRS = {
    "cv2_vs_cv1": ("fused-prep-cv1", "fused-prep-cv2"),
    "a5_style_vs_fused_prep": ("fused-prep", "a5-style"),
    "a5_style_parallel_vs_serial": ("a5-style", "a5-style-parallel"),
}
A5_REFERENCE = "https://github.com/Biuapha/vllm-ascend/tree/f16f61a9509a4a609d88942dc7109c671d9197ac"


def integer_sweep(value):
    """Parse comma-separated integers and inclusive start:stop[:step] ranges."""
    result = []
    for group in value.split(","):
        parts = [int(item) for item in group.split(":")]
        if len(parts) == 1:
            result.extend(parts)
        elif len(parts) in (2, 3):
            start, stop = parts[:2]
            step = parts[2] if len(parts) == 3 else 1
            if step < 1 or stop < start:
                raise argparse.ArgumentTypeError("ranges require start <= stop and step >= 1")
            result.extend(range(start, stop + 1, step))
        else:
            raise argparse.ArgumentTypeError("expected an integer or start:stop[:step]")
    if not result or min(result) < 1:
        raise argparse.ArgumentTypeError("values must be positive")
    return list(dict.fromkeys(result))


def candidate_sweep(value):
    """Select gate row tile, input row alignment, and optionally native middle."""
    result = []
    for pair in value.split(","):
        fields = pair.split(":")
        native_middle = len(fields) == 3 and fields[0] == "native"
        if native_middle:
            fields = fields[1:]
        if len(fields) != 2:
            raise argparse.ArgumentTypeError("expected row_tile:projection_pad or native:row_tile:projection_pad")
        tile, pad = (int(field) for field in fields)
        if min(tile, pad) < 0:
            raise argparse.ArgumentTypeError("candidate parameters must be nonnegative")
        item = {"row_tile": tile, "projection_pad": pad}
        if native_middle:
            item["middle"] = "native"
        if item not in result:
            result.append(item)
    return result


def accepted_cases(config, device):
    """Accepted lengths select the previous replay's history, not current commits."""
    choices = {"one": 1, "two": min(2, config.tokens), "full": config.tokens}
    cases = {
        name: torch.full((config.batch,), count, dtype=torch.int32, device=device) for name, count in choices.items()
    }
    cases["mixed"] = torch.arange(config.batch, dtype=torch.int32, device=device) % config.tokens + 1
    return cases


def host_result(result):
    return {
        name: tensor.detach().to(device="cpu", dtype=torch.float64).clone()
        for name, tensor in zip(result._fields, result, strict=True)
    }


def accuracy_metrics(result, expected, protocol):
    """Record both A5 protocols even when only one is used as the pass criterion."""
    observed = host_result(result)
    metrics = {}
    for name, golden in expected.items():
        actual = observed[name]
        if actual.shape != golden.shape:
            metrics[name] = {"passed": False, "shape": list(actual.shape), "expected_shape": list(golden.shape)}
            continue
        finite = bool(torch.isfinite(actual).all() and torch.isfinite(golden).all())
        error = (actual - golden).abs()
        rtol, atol = ELEMENTWISE_LIMITS[name]
        mismatches = int((error > atol + rtol * golden.abs()).sum()) if finite else actual.numel()
        denominator = float(torch.linalg.vector_norm(golden))
        numerator = float(torch.linalg.vector_norm(error))
        relative_l2 = numerator / denominator if denominator else (0.0 if numerator == 0 else None)
        exact = bool(torch.equal(actual, golden))
        relative_pass = (
            exact if name == "history" else (relative_l2 is not None and relative_l2 < RELATIVE_LIMITS[name])
        )
        elementwise_pass = mismatches == 0
        selected = {
            "elementwise": elementwise_pass,
            "relative-l2": relative_pass,
            "both": elementwise_pass and relative_pass,
        }[protocol]
        metrics[name] = {
            "finite": finite,
            "max_abs": float(error.max()) if finite and error.numel() else None,
            "relative_l2": relative_l2 if relative_l2 is not None and math.isfinite(relative_l2) else None,
            "rtol": rtol,
            "atol": atol,
            "elementwise_mismatches": mismatches,
            "elementwise_pass": finite and elementwise_pass,
            "exact": exact,
            "relative_l2_limit": RELATIVE_LIMITS.get(name, 0.0),
            "relative_l2_pass": finite and relative_pass,
            "passed": finite and selected,
        }
    return {"passed": all(item["passed"] for item in metrics.values()), "outputs": metrics}


def reset(function):
    if hasattr(function, "reset"):
        function.reset()


class TorchReference:
    """Use the same advancing convolution-cache semantics as native graph replay."""

    def __init__(self, config, inputs):
        self.config = config
        self.inputs = inputs
        self.initial_cache = make_cache(config, inputs)
        self.cache = self.initial_cache.clone()

    def reset(self):
        self.cache.copy_(self.initial_cache)

    def __call__(self, accepted):
        result = full_reference(self.config, self.inputs, accepted, cache=self.cache)
        self.cache.copy_(result.history)
        return result


class GraphRunner:
    """Capture repeated full forwards once; exclude reset and input copies from timing."""

    def __init__(self, function, accepted, repeat, warmup):
        self.function = function
        self.accepted = accepted.clone()
        self.repeat = repeat
        self.start = torch.npu.Event(enable_timing=True)
        self.end = torch.npu.Event(enable_timing=True)
        reset(function)
        for _ in range(warmup):
            function(self.accepted)
        torch.npu.synchronize()
        reset(function)
        self.graph = torch.npu.NPUGraph()
        with torch.npu.graph(self.graph):
            for _ in range(repeat):
                self.output = function(self.accepted)
        torch.npu.synchronize()
        reset(function)

    def run(self, accepted):
        self.accepted.copy_(accepted)
        reset(self.function)
        self.graph.replay()
        torch.npu.synchronize()
        return self.output

    def warm_replays(self, count):
        self.accepted.fill_(1)
        reset(self.function)
        for _ in range(count):
            self.graph.replay()

    def time_us(self):
        # Replay continues from the history left by the previous timed group.
        self.start.record()
        self.graph.replay()
        self.end.record()
        self.end.synchronize()
        return self.start.elapsed_time(self.end) * 1000 / self.repeat


def paired_timings(baseline, candidate, samples, warmup):
    """Return every ABBA observation and pair means, without choosing a minimum."""
    for runner in (baseline, candidate):
        runner.warm_replays(warmup)
    torch.npu.synchronize()
    records = []
    for sample in range(samples):
        # Alternate ABBA and BAAB as well, distributing the outer positions.
        order = (
            (baseline, candidate, candidate, baseline)
            if sample % 2 == 0
            else (candidate, baseline, baseline, candidate)
        )
        measured = [runner.time_us() for runner in order]
        indices = (0, 3) if sample % 2 == 0 else (1, 2)
        base_us = statistics.mean(measured[index] for index in indices)
        candidate_us = statistics.mean(measured[index] for index in range(4) if index not in indices)
        records.append(
            {
                "order": "ABBA" if sample % 2 == 0 else "BAAB",
                "raw_us": measured,
                "baseline_us": base_us,
                "candidate_us": candidate_us,
                "speedup": base_us / candidate_us,
            }
        )
    base_median = statistics.median(item["baseline_us"] for item in records)
    candidate_median = statistics.median(item["candidate_us"] for item in records)
    return {
        "baseline_median_us": base_median,
        "candidate_median_us": candidate_median,
        "speedup": base_median / candidate_median,
        "samples": records,
    }


def validate_runner(runner, expected_function, config, inputs, protocol, snapshots):
    """Exercise eager and captured forwards with dynamic accepted-length inputs."""
    reports = {}
    for name, accepted in accepted_cases(config, inputs["x"].device).items():
        reset(expected_function)
        expected = host_result(expected_function(accepted))
        reset(runner.function)
        eager = accuracy_metrics(runner.function(accepted), expected, protocol)
        # Reuse the timing graph and change accepted values after capture.
        # Its final output must match the same number of advancing forwards.
        reset(expected_function)
        for _ in range(runner.repeat):
            expected_replay = expected_function(accepted)
        expected_graph = host_result(expected_replay)
        captured = accuracy_metrics(runner.run(accepted), expected_graph, protocol)
        reports[name] = {"eager": eager, "graph": captured}
        for tensor_name, original in snapshots.items():
            if not torch.equal(inputs[tensor_name], original):
                raise AssertionError(f"{tensor_name} was modified during {name} validation")
        if hasattr(runner.function, "checkpoint_unchanged") and not runner.function.checkpoint_unchanged():
            raise AssertionError(f"native checkpoint was modified during {name} validation")
    return {
        "passed": all(case[mode]["passed"] for case in reports.values() for mode in ("eager", "graph")),
        "checkpoint_unchanged": True,
        "history_input_unchanged": True,
        "cases": reports,
    }


def baseline_function(name, config, inputs):
    if name == "torch":
        return TorchReference(config, inputs)
    # Keep optional vLLM/fla_npu imports out of the portable reference path.
    return importlib.import_module("native_adapter").Native(config, inputs)


def candidate_function(config, inputs, *, middle="tilelang", **parameters):
    if middle == "native":
        return importlib.import_module("native_adapter").Native(config, inputs, **parameters)
    candidates = importlib.import_module("full_candidate")
    if middle in ("a5-style", "a5-style-parallel"):
        if parameters.keys() - {"row_tile", "projection_pad"} or any(parameters.values()):
            raise ValueError(f"{middle} supports only its fixed schedule")
        if middle == "a5-style-parallel":
            return candidates.ParallelA5StyleCandidate(config, inputs)
        return candidates.A5StyleCandidate(config, inputs)
    return candidates.Candidate(config, inputs, **parameters)


def run_case(args, batch, tokens, record):
    """Validate and time a tuned winner or each requested fixed variant."""
    config = Config(batch=batch, tokens=tokens, heads=args.heads, hidden=args.hidden)
    inputs = make_inputs(config, device=args.device, seed=args.seed + batch)
    inputs["rollback"] = make_rollback(config, inputs)
    snapshots = {name: inputs[name].clone() for name in ("state", "history", "rollback")}
    record.update({"config": asdict(config), "seed": args.seed + batch})
    if args.reference_only:
        record["reference_cases"] = {}
        for name, accepted in accepted_cases(config, args.device).items():
            output = host_result(full_reference(config, inputs, accepted))
            finite = all(bool(torch.isfinite(value).all()) for value in output.values())
            if not finite or any(not torch.equal(inputs[key], value) for key, value in snapshots.items()):
                raise AssertionError("portable reference produced non-finite output or mutated inputs")
            record["reference_cases"][name] = {
                "finite": True,
                "inputs_unchanged": True,
                "shapes": {key: list(value.shape) for key, value in output.items()},
            }
        record["status"] = "reference_smoke_only"
        return
    accepted = accepted_cases(config, args.device)["one"]
    baseline = baseline_function(args.baseline, config, inputs)
    baseline_runner = GraphRunner(baseline, accepted, args.repeat, args.warmup)
    portable = TorchReference(config, inputs)
    record["baseline_accuracy"] = validate_runner(baseline_runner, portable, config, inputs, args.protocol, snapshots)
    if not record["baseline_accuracy"]["passed"]:
        raise AssertionError("baseline failed the portable reference accuracy protocol")
    record["candidates"] = []
    winning_score = math.inf
    winner = None
    requested_variants = dict.fromkeys(args.variant or ())
    pairs = {key: pair for key, pair in VARIANT_PAIRS.items() if all(name in requested_variants for name in pair)}
    paired_variants = {name for pair in pairs.values() for name in pair}
    paired_runners = {}
    if args.variant:
        variants = []
        for name in requested_variants:
            parameters = MIDDLE_VARIANTS[name].copy()
            if "cv_slots" in parameters:
                parameters["core_count"] = args.core_count
            variants.append((name, parameters))
    else:
        variants = [(None, parameters) for parameters in args.candidates]
    for name, parameters in variants:
        item = {"parameters": parameters}
        if name is not None:
            item["variant"] = name
        record["candidates"].append(item)
        runner = None
        try:
            function = candidate_function(config, inputs, **parameters)
            item["projection_rows"] = function.projection_rows
            runner = GraphRunner(function, accepted, args.repeat, args.warmup)
            item["accuracy"] = validate_runner(runner, baseline, config, inputs, args.protocol, snapshots)
            if not item["accuracy"]["passed"]:
                item["status"] = "accuracy_failed"
                continue
            if name is not None:
                item["held_out"] = paired_timings(baseline_runner, runner, args.samples, args.warmup)
                item["status"] = "passed"
                if name in paired_variants:
                    paired_runners[name] = runner
                continue
            item["training"] = paired_timings(baseline_runner, runner, args.tune_samples, args.warmup)
            item["status"] = "passed"
            score = item["training"]["candidate_median_us"]
            if score < winning_score:
                winning_score, winner = score, parameters
        except Exception as error:
            item.update({"status": "error", "error": f"{type(error).__name__}: {error}"})
        finally:
            del runner
    for key, (baseline_name, candidate_name) in pairs.items():
        if baseline_name in paired_runners and candidate_name in paired_runners:
            record[key] = paired_timings(
                paired_runners[baseline_name], paired_runners[candidate_name], args.samples, args.warmup
            )
    paired_runners.clear()
    if not args.variant:
        if winner is None:
            raise AssertionError("no candidate passed compilation, capture and accuracy checks")
        record["selected"] = winner
        winning_runner = GraphRunner(candidate_function(config, inputs, **winner), accepted, args.repeat, args.warmup)
        # These fresh samples do not influence the selected parameters.
        record["held_out"] = paired_timings(baseline_runner, winning_runner, args.samples, args.warmup)
    for name, original in snapshots.items():
        if not torch.equal(inputs[name], original):
            raise AssertionError(f"{name} was modified during timing")
    record["status"] = (
        "candidate_failed"
        if args.variant and any(item["status"] != "passed" for item in record["candidates"])
        else "passed"
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=integer_sweep, default=integer_sweep("1:80"))
    parser.add_argument("--tokens", type=integer_sweep, default=[6], help="1:16 sweeps all A5 verify lengths")
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--device", choices=("cpu", "npu"), default="npu")
    parser.add_argument("--baseline", choices=("torch", "native"), default="torch")
    candidates = parser.add_mutually_exclusive_group()
    candidates.add_argument("--candidates", type=candidate_sweep, default=candidate_sweep(DEFAULT_CANDIDATES))
    candidates.add_argument(
        "--variant", action="append", choices=MIDDLE_VARIANTS, help="measure each fixed variant with no tuning"
    )
    parser.add_argument("--core-count", type=int, default=24, help="Cube groups used by persistent variants")
    parser.add_argument("--protocol", choices=("relative-l2", "elementwise", "both"), default="relative-l2")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=9)
    parser.add_argument("--tune-samples", type=int, default=5)
    parser.add_argument("--samples", type=int, default=23, help="held-out ABBA/BAAB timing samples")
    parser.add_argument("--seed", type=int, default=724311)
    parser.add_argument(
        "--reference-only", action="store_true", help="only check portable reference finiteness and purity"
    )
    parser.add_argument("--out", type=Path, default=Path("a3-full-kda.json"))
    args = parser.parse_args()
    if min(args.heads, args.hidden, args.warmup, args.repeat, args.tune_samples, args.samples, args.core_count) < 1:
        parser.error("dimensions, warmup, repeats and sample counts must be positive")
    if max(args.tokens) > 16:
        parser.error("the A5 comparison scope is 1 through 16 tokens")
    if args.device == "cpu" and not args.reference_only:
        parser.error("CPU supports --reference-only; candidate and graph timing require an A3 NPU")
    if args.reference_only and args.baseline != "torch":
        parser.error("--reference-only uses the portable torch reference")
    return args


def main():
    args = parse_args()
    torch.set_num_threads(2)
    device_name = "CPU"
    if args.device == "npu":
        importlib.import_module("torch_npu")
        device_name = torch.npu.get_device_name(torch.npu.current_device())
        if "910_93" not in device_name and "910C" not in device_name:
            raise RuntimeError(f"expected Ascend A3 (910_93 or 910C), got {device_name}")
    document = {
        "schema_version": 1,
        "a5_reference": A5_REFERENCE,
        "baseline": "portable PyTorch full path" if args.baseline == "torch" else "A5 native operator chain on A3",
        "timing_boundary": "hidden states through output projection, including convolution/history and correction/KG",
        "timing_method": "NPU stream-event time per full forward in a repeated NPU graph; ABBA/BAAB pairs",
        "timing_accepted": 1,
        "reset_timed": False,
        "cache_semantics": "each backend advances its private convolution cache inside every full forward",
        "selection": (
            "fixed variants measured independently against the baseline"
            if args.variant
            else "minimum training candidate median; frozen winner measured on fresh held-out timing samples"
        ),
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": device_name},
        "options": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "cases": [],
    }
    for distribution in ("tilelang", "torch-npu", "vllm-ascend"):
        try:
            document["environment"][distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            document["environment"][distribution] = None
    if args.device == "npu":
        document["environment"]["tilelang"] = importlib.import_module("tilelang").__version__
    failed = False
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for tokens in args.tokens:
        for batch in args.batches:
            record = {"batch": batch, "tokens": tokens}
            document["cases"].append(record)
            try:
                with torch.inference_mode():
                    run_case(args, batch, tokens, record)
                failed |= record["status"] == "candidate_failed"
            except Exception as error:
                record.update({"status": "error", "error": f"{type(error).__name__}: {error}"})
                failed = True
            args.out.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            print(f"B={batch} L={tokens}: {record['status']}", flush=True)
            if args.device == "npu":
                gc.collect()
                torch.npu.empty_cache()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
