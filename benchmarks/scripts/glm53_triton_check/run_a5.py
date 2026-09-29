# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""PR #17542: pinned A/B correctness and NPU-graph operator latency."""

import argparse
import csv
import hashlib
import importlib
import importlib.metadata
import json
import os
import shlex
import statistics
import subprocess
import sys
import traceback
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent


def verify_bundle():
    manifest = json.loads((ROOT / "manifest.json").read_text())
    for name, expected in manifest["sha256"].items():
        actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Bundle file differs from pinned source: {name}")
    return manifest


def snapshot_command(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        return {"command": command, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": command, "error": str(error)}


def capture(fn, copies):
    for _ in range(3):
        fn()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        outputs = [fn() for _ in range(copies)]
    for _ in range(3):
        graph.replay()
    torch.npu.synchronize()
    return graph, outputs


def measure_pair(before, after, copies, options):
    """Compile both graphs first; alternate A/B timing order to reduce drift."""
    graphs = [capture(fn, copies) for fn in (before, after)]
    samples = [[], []]
    for sample in range(options.samples):
        for arm in [0, 1] if sample % 2 == 0 else [1, 0]:
            start, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
            start.record()
            for _ in range(options.replays):
                graphs[arm][0].replay()
            end.record()
            end.synchronize()
            samples[arm].append(start.elapsed_time(end) * 1000 / (options.replays * copies))
    old, new = map(statistics.median, samples)
    if old <= 0 or new <= 0:
        raise RuntimeError("NPU event timing returned a nonpositive duration")
    reduction = (old - new) / old * 100
    return {
        "baseline_us": old,
        "candidate_us": new,
        "reduction_pct": reduction,
        "baseline_samples_us": samples[0],
        "candidate_samples_us": samples[1],
        "status": "REGRESSION"
        if reduction < -options.regression_pct
        else "IMPROVED"
        if reduction > options.regression_pct
        else "NEUTRAL",
    }


@contextmanager
def observe_scores():
    saved = torch.topk
    values = []

    def topk(scores, *args, **kwargs):
        values.append(scores.detach().cpu().clone())
        return saved(scores, *args, **kwargs)

    with patch.object(torch, "topk", topk):
        yield values


def indexer_inputs(tokens, requests, pools, live=None):
    live = tokens if live is None else live
    pages = (pools + 15) // 16
    query = torch.randn(tokens, 32, 128, dtype=torch.bfloat16, device="npu")
    cache = torch.randn(requests * pages, 16, 1, 128, dtype=torch.bfloat16, device="npu")
    weights = torch.randn(tokens, 32, dtype=torch.bfloat16, device="npu")
    ends = torch.arange(1, requests + 1, dtype=torch.int32, device="npu") * (live // requests)
    lengths = torch.full((requests,), pools, dtype=torch.int32, device="npu")
    table = torch.randperm(requests * pages, device="npu").int().reshape(requests, pages)
    # Consecutive positions per request, including an incomplete causal tail.
    positions = torch.full((tokens,), pools * 4 + 2, dtype=torch.int64, device="npu")
    per_request = live // requests
    positions[:live].copy_(torch.arange(pools * 4 + 3 - per_request, pools * 4 + 3, device="npu").repeat(requests))
    return query, cache, weights, ends, lengths, table, positions


def assert_indexer_scores(baseline_chunks, candidate_chunks, live):
    baseline = torch.cat(baseline_chunks, dim=0)
    candidate = torch.cat(candidate_chunks, dim=0)
    assert baseline.shape == candidate.shape
    assert 0 <= live <= baseline.shape[0]
    # Mainline still computes padded rows; only the candidate masks them here.
    # Compare every valid score exactly, independently of score chunk boundaries.
    torch.testing.assert_close(baseline[:live], candidate[:live], rtol=0, atol=0)
    lowest = torch.finfo(candidate.dtype).min
    assert bool((candidate[live:] == lowest).all()), "Candidate padding scores must use the lowest finite value"


def indexer_pair(modules, args, pools, allow):
    kwargs = dict(index_topk=2048, index_kpool=4, max_pool_seq_len=pools)
    before = lambda: modules[0].glm5_next_lightning_indexer_triton(*args, **kwargs)
    after = lambda: modules[1].glm5_next_lightning_indexer_triton(*args, **kwargs, allow_cache_packing=allow)
    live = int(args[3][-1].item())
    with observe_scores() as a:
        expected = before()
    with observe_scores() as b:
        actual = after()
    assert_indexer_scores(a, b, live)
    # Baseline's caller clears padding; the candidate does it in the operator.
    torch.testing.assert_close(expected[:live], actual[:live], rtol=0, atol=0)
    assert bool((actual[live:] == -1).all())
    return before, after


def padding_replay(modules):
    """Fixed bucket: 8 one-token requests, then 8 two-token requests on replay."""
    args = indexer_inputs(64, 8, 875, live=8)
    output_backing = torch.full((128, 2080), 77, dtype=torch.int32, device="npu")
    output = output_backing[::2]
    kwargs = dict(
        index_topk=2048,
        index_kpool=4,
        max_pool_seq_len=875,
        allow_cache_packing=False,
        output_buffer=output,
        pack_tail=True,
    )
    # The real gather must never be dispatched when FULL-mode policy is false.
    with patch.object(modules[1], "_gather_pool_cache", side_effect=AssertionError("Unexpected packed-cache dispatch")):
        graph, captured = capture(lambda: modules[1].glm5_next_lightning_indexer_triton(*args, **kwargs), 1)
        for count in (1, 2, 1):
            args[3].copy_(torch.arange(1, 9, dtype=torch.int32, device="npu") * count)
            args[0].copy_(torch.randn_like(args[0]))
            args[4].fill_(875 - count)
            args[6].fill_((875 - count) * 4 + 2)
            graph.replay()
            torch.npu.synchronize()
            actual = captured[0].cpu().clone()
            expected = (
                modules[0]
                .glm5_next_lightning_indexer_triton(*args, index_topk=2048, index_kpool=4, max_pool_seq_len=875)
                .cpu()
            )
            torch.testing.assert_close(actual[: 8 * count], expected[: 8 * count], rtol=0, atol=0)
            assert bool((actual[8 * count :] == -1).all())
            assert captured[0].data_ptr() == output.data_ptr()
            assert bool((output[:, 2051:] == -1).all())
            assert bool((output_backing[1::2] == 77).all())
    return {"bucket_rows": 64, "requests": 8, "live_rows_on_replay": [8, 16, 8], "packing": False, "result": "PASS"}


def compressor_pair(modules, tokens):
    keys = torch.randn(tokens, 128, device="npu")
    gates, ape = torch.randn_like(keys), torch.randn(4, 128, device="npu")
    positions = torch.arange(tokens, device="npu")
    ends = torch.tensor([tokens], dtype=torch.int32, device="npu")
    slots = torch.where((positions + 1) % 4 == 0, positions // 4, -1)
    shared = (
        keys,
        gates,
        ape,
        positions,
        ends,
        ends,
        positions % 4,
        torch.zeros(1, 1, dtype=torch.int32, device="npu"),
        slots,
        4,
    )
    tails = [torch.zeros(1, 2, 4, 128, device="npu") for _ in range(2)]
    caches = [torch.zeros((tokens + 63) // 64 + 1, 16, 1, 128, dtype=torch.bfloat16, device="npu") for _ in range(2)]
    funcs = [
        lambda arm=arm: modules[arm].glm5_next_kpool_tail_compress_and_write_cache_triton(
            tails[arm], caches[arm], *shared
        )
        for arm in range(2)
    ]
    for fn in funcs:
        fn()
    torch.testing.assert_close(tails[0], tails[1], rtol=0, atol=0)
    torch.testing.assert_close(caches[0], caches[1], rtol=0, atol=0)
    # Every pool reads current input starting at position zero: replay is idempotent.
    return funcs


def run_existing_tests(output, device):
    names = ["test_glm5next_pool_key_indexer_triton.py", "test_glm5next_kpool_tail_triton.py"]
    bootstrap = (
        "import sys, torch, torch_npu, pytest; "
        "torch.npu.set_device(int(sys.argv[1])); sys.exit(pytest.main(sys.argv[2:]))"
    )
    command = [
        sys.executable,
        "-c",
        bootstrap,
        str(device),
        "-q",
        "-x",
        "-o",
        "addopts=",
        "--confcutdir=" + str(ROOT),
        "--junitxml=" + str(output / "operators.xml"),
    ] + [str(ROOT / name) for name in names]
    print("Existing operator cases:", shlex.join(command), flush=True)
    result = subprocess.run(command, cwd=ROOT)
    if result.returncode:
        raise RuntimeError(f"Existing operator tests failed (exit {result.returncode})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0, help="Logical NPU index after device visibility mapping")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--suite", choices=("quick", "full"), default="quick")
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--regression-pct", type=float, default=5.0)
    parser.add_argument("--check-only", action="store_true", help="Run correctness checks without latency measurements")
    parser.add_argument(
        "--check-bundle", action="store_true", help="Check pinned files without importing NPU libraries"
    )
    options = parser.parse_args()
    if options.samples < 3 or options.replays < 1 or options.regression_pct < 0:
        parser.error("samples >= 3, replays >= 1 and regression-pct >= 0 are required")
    manifest = verify_bundle()
    if options.check_bundle:
        print("Bundle hashes OK", manifest["revisions"])
        return 0
    output = options.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "results.json").exists():
        raise RuntimeError("Output already has results.json; select a new output directory")
    results = {
        "complete": False,
        "scope": "isolated operator graph latency, not model throughput or GSM8K",
        "manifest": manifest,
        "command": shlex.join(sys.argv),
        "options": vars(options) | {"output": str(output)},
        "cases": [],
    }

    def save():
        (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")

    def record(name, pair, copies=8):
        row = {"case": name, "accuracy": "exact_match_to_baseline"}
        if not options.check_only:
            row.update(measure_pair(*pair, copies, options))
        results["cases"].append(row)
        print(json.dumps(row), flush=True)
        save()

    save()
    try:
        global torch
        import torch
        import torch_npu

        torch.npu.set_device(options.device)
        torch.manual_seed(17542)
        results["environment"] = {
            "python": sys.version,
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
            "device": torch.npu.get_device_name(options.device),
            "environment_variables": {
                name: os.environ.get(name)
                for name in ("ASCEND_RT_VISIBLE_DEVICES", "ASCEND_VISIBLE_DEVICES", "TRITON_CACHE_DIR")
            },
            "npu_smi": snapshot_command(["npu-smi", "info"]),
        }
        results["environment"]["packages"] = {
            p.metadata["Name"]: p.version
            for p in importlib.metadata.distributions()
            if any(s in (p.metadata["Name"] or "").lower() for s in ("torch", "triton", "vllm", "cann", "pytest"))
        }
        print(json.dumps(results["environment"], indent=2), flush=True)
        save()
        modules = {
            name: [importlib.import_module(f"{role}.{name}") for role in ("baseline", "candidate")]
            for name in ("glm5_next_lightning_indexer", "glm5_next_kpool_tail_compress", "gated_norm")
        }
        # Launch pytest before compiling parent graphs to keep memory use bounded.
        run_existing_tests(output, options.device)
        results["existing_operator_cases"] = "PASS"
        results["norm_branches"] = importlib.import_module("norm_checks").run(*modules["gated_norm"])
        results["padding_replay"] = padding_replay(modules["glm5_next_lightning_indexer"])
        save()
        shapes = [(1, 1, 875), (8, 1, 8192), (64, 64, 8192), (128, 8, 875), (512, 1, 875)]
        if options.suite == "full":
            shapes += [
                (1, 1, 8192),
                (32, 32, 875),
                (32, 32, 8192),
                (128, 1, 512),
                (256, 1, 875),
                (2048, 1, 875),
                (1024, 1, 876),
            ]
        for tokens, requests, pools in shapes:
            print(f"CHECK indexer T={tokens} R={requests} P={pools}, packing allowed", flush=True)
            args = indexer_inputs(tokens, requests, pools)
            pair = indexer_pair(modules["glm5_next_lightning_indexer"], args, pools, allow=True)
            record(f"indexer/T{tokens}/R{requests}/P{pools}/packing_allowed", pair, copies=4)
        print("CHECK indexer T=64 live=8 R=8 P=875, FULL paged", flush=True)
        args = indexer_inputs(64, 8, 875, live=8)
        pair = indexer_pair(modules["glm5_next_lightning_indexer"], args, 875, allow=False)
        record("indexer/T64/live8/R8/P875/FULL_paged", pair, copies=4)
        for tokens in [1, 8, 128] if options.suite == "quick" else [1, 8, 128, 2048]:
            record(f"compress/T{tokens}", compressor_pair(modules["glm5_next_kpool_tail_compress"], tokens))
        for rows in [4, 8, 32, 1024] if options.suite == "quick" else [4, 8, 32, 64, 1024, 8192]:
            x = torch.randn(rows, 128, dtype=torch.bfloat16, device="npu")
            gate, weight = torch.randn_like(x), torch.randn(128, dtype=x.dtype, device="npu")
            pair = [
                partial(module.rms_norm_gated, x, gate, weight, None, activation="sigmoid")
                for module in modules["gated_norm"]
            ]
            torch.testing.assert_close(pair[0](), pair[1](), rtol=0, atol=0)
            record(f"gated_norm/flattened_rows{rows}/D128", pair, copies=32)
        results["complete"] = True
        results["regressions"] = [row["case"] for row in results["cases"] if row.get("status") == "REGRESSION"]
        results["exit_code"] = 2 if results["regressions"] else 0
        fields = ["case", "accuracy", "baseline_us", "candidate_us", "reduction_pct", "status"]
        with (output / "summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results["cases"])
        save()
        print("DONE", "correctness PASS", "regressions:", results["regressions"], "results:", output, flush=True)
        return results["exit_code"]
    except Exception:
        results["error"] = traceback.format_exc()
        results["exit_code"] = 1
        save()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
