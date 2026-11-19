# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare complete QKV-to-output graph latency and speculative state on Ascend 950."""

import argparse
import json
import statistics
from pathlib import Path

import torch
import torch_npu  # noqa: F401
from fused import FusedMiddle
from inputs import Config, make_inputs
from native import Native


def capture(fn, repeat):
    for _ in range(5):
        fn()
    torch.npu.synchronize()
    g = torch.npu.NPUGraph()
    with torch.npu.graph(g):
        for _ in range(repeat):
            out = fn()
    return g, out


def paired(graphs, repeat, *, samples=20, warmup=15):
    # Preallocate events, warm replay, alternate execution order.
    events = [
        (torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)) for _ in range(samples * len(graphs))
    ]
    for _ in range(warmup):
        for g in graphs:
            g.replay()
    torch.npu.synchronize()
    results = [[] for _ in graphs]
    for t in range(samples):
        for offset in range(len(graphs)):
            i = (t + (offset if t % 2 == 0 else -offset)) % len(graphs)
            s, e = events[t * len(graphs) + i]
            s.record()
            graphs[i].replay()
            e.record()
            e.synchronize()
            results[i].append(s.elapsed_time(e) * 1000 / repeat)
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batches", default="1,28,80")
    p.add_argument("--out", default="comparison.json")
    p.add_argument("--seed", type=int, default=100, help="Base RNG seed, incremented by batch size")
    p.add_argument("--check-only", action="store_true", help="Validate outputs and state without timing graphs")
    p.add_argument("--repeat", type=int, default=10)
    a = p.parse_args()
    torch.npu.set_device(0)
    torch.set_num_threads(2)
    report = {
        "scope": "Complete QKV-to-O: unfused, original fusion, tuned fusion",
        "tokens_per_request": 6,
        "hidden": 4096,
        "heads": 64,
        "dim": 128,
        "repeat": a.repeat,
        "seed": a.seed,
        "device": torch.npu.get_device_name(0),
        "torch": torch.__version__,
        "input_distribution": "reference.make_inputs; synthetic values, original model shapes",
        "rows": [],
    }
    batches = range(1, 81) if a.batches == "1:80" else map(int, a.batches.split(","))
    if a.repeat < 1:
        p.error("repeat must be positive")
    for b in batches:
        if not 1 <= b <= 80:
            p.error("batch must be in [1, 80]")
        c = Config(batch=b)
        i = make_inputs(c, device="npu", seed=a.seed + b)
        n = Native(c, i)
        previous = None if a.check_only else FusedMiddle(n)
        f = FusedMiddle(n, tuned=True)
        # Exercise rollback offsets against nonzero speculative history.
        n.cs_base[1:, 3:] = torch.randn_like(n.cs_base[1:, 3:]) * 0.25
        checkpoint = n.state.clone()
        checks = []
        for accept in [1, 2, 6] + (["mixed"] if b > 1 else []):
            if accept == "mixed":
                n.accepted.copy_(torch.arange(b, device="npu", dtype=torch.int32) % 6 + 1)
            else:
                n.accepted.fill_(accept)
            n.reset()
            expected = n().clone()
            ec = n.cs.clone()
            er = n.correction[1:].clone()
            ek = n.kg[1:].clone()
            en = n.normalized.clone()
            n.reset()
            actual = f()
            torch.npu.synchronize()
            pairs = [
                ("output", actual, expected, 0.025, 0.002),
                ("cache", n.cs, ec, 0, 0),
                ("correction", n.correction[1:], er, 1e-4, 1e-6),
                ("kg", n.kg[1:], ek, 0.01, 0.0001),
                ("norm", n.normalized, en, 0.025, 0.002),
                ("checkpoint", n.state, checkpoint, 0, 0),
            ]
            errors = {}
            for name, x, y, rtol, atol in pairs:
                delta = (x.float() - y.float()).abs()
                errors[name] = float(delta.max())
            for name, x, y, rtol, atol in pairs:
                torch.testing.assert_close(x, y, rtol=rtol, atol=atol, msg=lambda msg, name=name: name + ": " + msg)
            checks.append({"accepted": accept, "max_abs": errors})
        print("validated", b, "output max_abs", max(s["max_abs"]["output"] for s in checks), flush=True)
        if a.check_only:
            report["rows"].append({"batch": b, "checks": checks})
            Path(a.out).write_text(json.dumps(report, indent=2))
            del n, f, i
            torch.npu.empty_cache()
            continue
        n.accepted.fill_(1)
        n.reset()
        ng, _ = capture(n, a.repeat)
        n.reset()
        fg, _ = capture(f, a.repeat)
        n.reset()
        pg, _ = capture(previous, a.repeat)
        ns, ps, fs = paired([ng, pg, fg], a.repeat)
        row = {
            "batch": b,
            "tokens": b * 6,
            "native_us": statistics.median(ns),
            "fused_us": statistics.median(fs),
            "previous_us": statistics.median(ps),
            "previous_samples_us": ps,
            "native_samples_us": ns,
            "fused_samples_us": fs,
            "checks": checks,
        }
        row["speedup"] = row["native_us"] / row["fused_us"]
        row["speedup_vs_previous"] = row["previous_us"] / row["fused_us"]
        report["rows"].append(row)
        Path(a.out).write_text(json.dumps(report, indent=2))
        print(
            json.dumps(
                {k: row[k] for k in ["batch", "native_us", "previous_us", "fused_us", "speedup", "speedup_vs_previous"]}
            ),
            flush=True,
        )
        del ng, fg, pg, n, f, previous, i
        torch.npu.empty_cache()


if __name__ == "__main__":
    main()
