# SPDX-License-Identifier: Apache-2.0
"""Compare native and TileLang QLI with independent parity and ACL Graph timing."""

import json
import math
import random
import statistics
from itertools import repeat

import torch
from qli_fixture import HEADS, WIDTH, make_fixture, reference_scores


def build_backend_call(fixture, backend):
    if backend == "native":
        return fixture.native_call()
    if backend != "tilelang-v29":
        raise ValueError(f"Unknown QLI backend {backend!r}")

    from qli_tilelang_v29 import build_tilelang_call

    return build_tilelang_call(
        fixture.query,
        fixture.key,
        fixture.weights,
        fixture.query_scale,
        fixture.key_scale,
        fixture.query_ends,
        fixture.key_lengths,
        fixture.block_table,
        sparse_count=fixture.topk,
        sparse_mode=3,
        verified_lengths=(fixture.cpu_query_ends, fixture.cpu_key_lengths),
    )


def output_on_cpu(result, fixture):
    if not isinstance(result, torch.Tensor):
        raise TypeError("QLI should return one indices tensor")
    expected = (fixture.query.size(0), 1, fixture.topk)
    if result.shape != expected or result.dtype != torch.int32 or result.device != fixture.query.device:
        raise ValueError(f"Expected int32 indices of shape {expected} on the input device")
    return result.cpu().squeeze(1).clone()


def verify_fixture(
    fixture, eager_outputs, graph_outputs, *, strict_indices=False, check_reference=True, allow_index_mismatch=False
):
    for name, output in eager_outputs.items():
        torch.testing.assert_close(graph_outputs[name], output, atol=0, rtol=0, msg=f"{name}: eager vs graph")
    tilelang_backend = next((name for name in eager_outputs if name != "native"), None)
    has_pair = "native" in eager_outputs and tilelang_backend is not None
    if not check_reference and not (has_pair and strict_indices):
        raise ValueError("Native ordered-index parity is required without the CPU score oracle")
    report = {
        "rows": fixture.query.size(0),
        "oracle_rows": 0,
        "cross_backend_exact_rows": 0 if has_pair else None,
        "cross_backend_differing_slots": 0 if has_pair else None,
        "eager_graph_exact": True,
    }
    visible_lengths = fixture.visible_lengths()
    score_rows = reference_scores(fixture) if check_reference else repeat(None, len(visible_lengths))
    for row, (visible, scores) in enumerate(zip(visible_lengths, score_rows, strict=True)):
        count = min(visible, fixture.topk)
        if scores is not None:
            expected = scores.topk(count, sorted=True).values
        for name, matrix in eager_outputs.items():
            valid = matrix[row, :count].long()
            if bool(((valid < 0) | (valid >= visible)).any()) or valid.unique().numel() != count:
                raise AssertionError(f"{name} row {row}: repeated or out-of-range logical K index")
            if not bool((matrix[row, count:] == -1).all()):
                raise AssertionError(f"{name} row {row}: expected -1 padding")
            if scores is not None:
                torch.testing.assert_close(
                    scores[valid],
                    expected,
                    atol=2e-7,
                    rtol=5e-6,
                    msg=lambda msg, name=name, row=row: (
                        f"{name} row {row}: ranked values differ from the CPU oracle\n{msg}"
                    ),
                )
        report["oracle_rows"] += int(scores is not None)
        if has_pair:
            native_row = eager_outputs["native"][row, :count]
            tilelang_row = eager_outputs[tilelang_backend][row, :count]
            different = native_row != tilelang_row
            differences = int(different.sum())
            report["cross_backend_differing_slots"] += differences
            report["cross_backend_exact_rows"] += int(differences == 0)
            if differences:
                if strict_indices:
                    if allow_index_mismatch:
                        continue
                    slot = int(different.nonzero()[0, 0])
                    raise AssertionError(
                        f"{tilelang_backend} row {row}: {differences} indices differ from native order, "
                        f"first at slot {slot} ({tilelang_row[slot].item()} vs {native_row[slot].item()})"
                    )
                torch.testing.assert_close(
                    scores[native_row[different].long()],
                    scores[tilelang_row[different].long()],
                    atol=2e-7,
                    rtol=5e-6,
                    msg=lambda msg,
                    row=row: f"{tilelang_backend} row {row}: different indices with distinct scores\n{msg}",
                )
    return report


class GraphRunner:
    def __init__(self, calls, warmup, forwards):
        if not calls or forwards < 1 or forwards % len(calls):
            raise ValueError("Graph forward count must be a positive multiple of the fixture replica count")
        first, *replicas = calls
        self.forwards = forwards
        self.graph = torch.npu.NPUGraph()
        self.start = torch.npu.Event(enable_timing=True)
        self.end = torch.npu.Event(enable_timing=True)
        for _ in range(warmup):
            for call in calls:
                call()
        torch.npu.synchronize()
        with torch.npu.graph(self.graph):
            for _ in range(forwards // len(calls)):
                self.output = first()
                for replica in replicas:
                    replica()
        for _ in range(warmup):
            self.graph.replay()
        torch.npu.synchronize()

    def time_us(self):
        self.start.record()
        self.graph.replay()
        self.end.record()
        self.end.synchronize()
        elapsed = self.start.elapsed_time(self.end) * 1000 / self.forwards
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError(f"Bad NPU event duration: {elapsed} us")
        return elapsed


def collect_timings(graphs, count):
    if len(graphs) == 1:
        backend, graph = next(iter(graphs.items()))
        for sample in range(count):
            yield {"sample": sample, "order": "A", "duration_us": {backend: graph.time_us()}}
        return
    (tilelang_backend,) = set(graphs) - {"native"}
    pair = ("native", tilelang_backend)
    for sample in range(count):
        outer, inner = pair if sample % 2 == 0 else pair[::-1]
        order = (outer, inner, inner, outer)
        observations = [graphs[backend].time_us() for backend in order]
        paired = {outer: statistics.mean(observations[::3]), inner: statistics.mean(observations[1:3])}
        yield {
            "sample": sample,
            "order": "ABBA" if outer == "native" else "BAAB",
            "replays_us": observations,
            "duration_us": paired,
            "native_over_tilelang": paired["native"] / paired[tilelang_backend],
        }


def quantile(values, probability):
    values = sorted(values)
    index = probability * (len(values) - 1)
    left = math.floor(index)
    right = math.ceil(index)
    return values[left] + (index - left) * (values[right] - values[left])


def run_case(case, backends, args, perf, *, check_reference=True, allow_index_mismatch=False, dispatch=None):
    replicas = args.fixture_replicas if perf else 1
    fixtures = tuple(make_fixture(case) for _ in range(replicas))
    fixture = fixtures[0]
    eager = {}
    graphs = {}
    for backend in backends:
        print(f"QUALIFY {case['case']} {backend}", flush=True)
        calls = tuple(build_backend_call(replica, backend) for replica in fixtures)
        if dispatch is not None:
            observed = getattr(calls[0], "qli_dispatch", backend)
            dispatch[backend] = backend if observed == "packed" else observed
        eager[backend] = output_on_cpu(calls[0](), fixture)
        graphs[backend] = GraphRunner(calls, args.warmup, args.forwards if perf else 1)
    captured = {name: output_on_cpu(graph.output, fixture) for name, graph in graphs.items()}
    parity = verify_fixture(
        fixture,
        eager,
        captured,
        strict_indices=args.strict_indices,
        check_reference=check_reference,
        allow_index_mismatch=allow_index_mismatch,
    )
    print(f"PARITY {case['case']} {json.dumps(parity)}", flush=True)
    if not perf:
        return parity, [], [], None

    samples = list(collect_timings(graphs, args.samples))
    counts = {
        "visible_pairs": sum(fixture.visible_lengths()),
        "logical_bytes": fixture.logical_bytes(),
    }
    counts["useful_int8_ops"] = 2 * HEADS * WIDTH * counts["visible_pairs"]
    rows = []
    for name in backends:
        times = [sample["duration_us"][name] for sample in samples]
        rows.append(
            {
                "case": case["case"],
                "backend": name,
                "p50_us": statistics.median(times),
                "p90_us": quantile(times, 0.9),
                "num_samples": len(samples),
                **counts,
            }
        )
    comparison = None
    if len(backends) == 2:
        speedups = [sample["native_over_tilelang"] for sample in samples]
        rng = random.Random(case["seed"])
        bootstrap = [statistics.median(rng.choices(speedups, k=len(speedups))) for _ in range(4000)]
        comparison = {
            "case": case["case"],
            "tilelang_backend": next(name for name in backends if name != "native"),
            "native_over_tilelang": statistics.median(speedups),
            "ci95_low": quantile(bootstrap, 0.025),
            "ci95_high": quantile(bootstrap, 0.975),
        }
    print(f"TIMING {case['case']} {json.dumps(rows)}", flush=True)
    return parity, rows, samples, comparison
