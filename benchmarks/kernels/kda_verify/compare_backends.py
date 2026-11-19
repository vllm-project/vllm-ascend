# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare unfused and TileLang QKV-to-O paths on identical examples and a single NPU."""

import argparse
import gc
import json
import statistics
from pathlib import Path

import torch
import torch_npu
from benchmark import capture, paired
from fused import WEIGHT_REUSE_BATCHES, FusedMiddle
from inputs import Config, make_inputs
from native import Native

# Reference peaks and logical traffic model for the fixed six-token GLM5Next fixture.
BF16_FLOPS_PER_SECOND = 432e12
HBM_BYTES_PER_SECOND = 4e12
PROJECTION_FLOPS_PER_REQUEST = 1_651_507_200
TOLERANCES = {
    "output": (0.025, 0.002),
    "cache": (0, 0),
    "correction": (1e-4, 1e-6),
    "kg": (0.01, 0.0001),
    "norm": (0.025, 0.002),
    "checkpoint": (0, 0),
}


def model_utilization(batch, latency_us, *, fused, projection_rows=None):
    """Pass the measured input projection height when comparing padded schedules."""
    if fused:
        logical_bytes = 275_481_088 + 6_020_612 * batch
        if projection_rows is None:
            projection_rows = 54 if batch in (6, 7, 8) else 6 * batch
        if projection_rows > 6 * batch:
            logical_bytes += 98_304 * batch
    else:
        logical_bytes = 275_481_096 + 7_790_100 * batch
    seconds = latency_us * 1e-6
    mfu = 100 * (PROJECTION_FLOPS_PER_REQUEST * batch) / (BF16_FLOPS_PER_SECOND * seconds)
    mbu = 100 * logical_bytes / (HBM_BYTES_PER_SECOND * seconds)
    return {"mfu_pct": mfu, "logical_mbu_pct": mbu, "sum_points": mfu + mbu}


def observe(n, output):
    return {
        "output": output,
        "cache": n.cs,
        "correction": n.correction[1:],
        "kg": n.kg[1:],
        "norm": n.normalized,
        "checkpoint": n.state,
    }


def validate(n, output, expected, reference_state):
    result = {}
    for key, actual in observe(n, output).items():
        target = reference_state if key == "checkpoint" else expected[key]
        rtol, atol = TOLERANCES[key]
        torch.testing.assert_close(actual, target, rtol=rtol, atol=atol, msg=lambda msg, key=key: key + " " + msg)
        if key != "checkpoint":
            result[key] = (actual.float() - target.float()).abs().max().item()
    return result


def run_batch(b, args):
    c = Config(batch=b)
    n = Native(c, make_inputs(c, device="npu", seed=args.seed + b))
    n.cs_base[1:, 3:] = torch.randn_like(n.cs_base[1:, 3:]) * 0.25
    paths = {"unfused": n, "tilelang": FusedMiddle(n, tuned=True, reuse_weights=True)}
    checks = []
    state = n.state.clone()
    for accept in [1, 2, 6, "mixed"] if b > 1 else [1, 2, 6]:
        if accept == "mixed":
            n.accepted.copy_(torch.arange(b, device="npu", dtype=torch.int32) % 6 + 1)
        else:
            n.accepted.fill_(accept)
        n.reset()
        ref = {k: v.clone() for k, v in observe(n, n()).items() if k != "checkpoint"}
        n.reset()
        checks.append({"accepted": accept, "tilelang": validate(n, paths["tilelang"](), ref, state)})
    del ref
    n.accepted.fill_(1)
    graphs = {}
    for k, f in paths.items():
        n.reset()
        graphs[k] = capture(f, args.repeat)
    # Change history after capture and poison all public output buffers. This
    # checks that replay recomputes outputs and writes back the state.
    n.accepted.fill_(2)
    n.reset()
    # Capture-time allocations are shared within each graph. Compute the native
    # reference by replaying the same number of steps from identical history.
    for _ in range(args.repeat):
        prediction = n()
    ref = {k: v.clone() for k, v in observe(n, prediction).items() if k != "checkpoint"}
    for k, (g, out) in graphs.items():
        n.reset()
        n.correction[1:].fill_(float("nan"))
        n.kg[1:].fill_(float("nan"))
        n.normalized.fill_(float("nan"))
        out.fill_(float("nan"))
        g.replay()
        validate(n, out, ref, state)
    n.accepted.fill_(1)
    n.reset()
    timings = paired([g for g, _ in graphs.values()], args.repeat, samples=args.samples, warmup=5)
    samples = dict(zip(graphs, timings))
    medians = {k: statistics.median(x) for k, x in samples.items()}
    return {
        "batch": b,
        "tilelang_projection_rows": paths["tilelang"].projection_rows,
        "tilelang_weight_reuse": b in WEIGHT_REUSE_BATCHES,
        "checks": checks,
        "graph_replay_validated": True,
        "median_us": medians,
        "model_utilization": {
            k: model_utilization(b, us, fused=k == "tilelang", projection_rows=paths["tilelang"].projection_rows)
            for k, us in medians.items()
        },
        "samples_us": samples,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batches", default="1:80")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=83700)
    p.add_argument("--repeat", type=int, default=5)
    p.add_argument("--samples", type=int, default=15)
    args = p.parse_args()
    torch.npu.set_device(0)
    torch.set_num_threads(2)
    batches = list(range(1, 81)) if args.batches == "1:80" else [int(k) for k in args.batches.split(",")]
    if args.repeat < 1 or args.samples < 1 or any(not 1 <= b <= 80 for b in batches):
        p.error("repeat and samples must be positive and batches must be in [1, 80]")
    dest = Path(args.out)
    report = {
        "seed": args.seed,
        "repeat": args.repeat,
        "samples": args.samples,
        "device": torch.npu.get_device_name(0),
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "reference_peaks": {
            "device": "Ascend950DT_9581",
            "bf16_flops_per_second": BF16_FLOPS_PER_SECOND,
            "hbm_bytes_per_second": HBM_BYTES_PER_SECOND,
        },
        "rows": [],
    }
    for b in batches:
        row = run_batch(b, args)
        report["rows"].append(row)
        tmp = dest.with_suffix(".tmp")
        tmp.write_text(json.dumps(report))
        tmp.replace(dest)
        print(b, row["median_us"], flush=True)
        gc.collect()
        torch.npu.empty_cache()


if __name__ == "__main__":
    main()
