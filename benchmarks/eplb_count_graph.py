"""Main versus final two-stage wrapper, Graph only, synthetic contract cases."""

import gc
import hashlib
import json
import statistics
import traceback
from pathlib import Path

import baseline_tensor_ops as main_module
import torch
import torch_npu  # noqa: F401

import vllm_ascend.distributed.eplb.tensor_ops as candidate_module
import vllm_ascend.ops.triton.eplb_count as kernel_module

SHAPES = [
    (1, 1, 1),
    (1, 8, 128),
    (32, 2, 8),
    (64, 4, 32),
    (65, 8, 17),
    (128, 3, 64),
    (256, 8, 64),
    (511, 1, 128),
    (512, 1, 128),
    (513, 1, 129),
    (512, 8, 128),
    (2048, 8, 128),
    (10240, 1, 128),
    (10240, 4, 128),
    (10240, 8, 32),
    (10240, 8, 64),
    (10240, 8, 128),
    (10240, 8, 256),
    (10240, 16, 128),
    (32768, 8, 128),
]
CASES = [(s, "int64", 1) for s in SHAPES]
CASES += [(s, "int32", 1) for s in [(65, 8, 17), (10240, 8, 128), (32768, 8, 128)]]
CASES += [((10240, 8, 128), "int64", 2)]
CALLS, SAMPLES = 20, 10
outdir = Path("results")
outdir.mkdir(exist_ok=True)
identity = [
    {
        "module": m.__name__,
        "path": m.__file__,
        "sha256": hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest(),
    }
    for m in [main_module, candidate_module, kernel_module]
]
(outdir / "import-identity.json").write_text(json.dumps(identity, indent=2))
rows = []


def measure(shape, dtype_name, stride, pattern):
    t, k, c = shape
    ids = (torch.arange(t * k).reshape(t, k) % c + 7).to(getattr(torch, dtype_name))
    mask = torch.ones(t, dtype=torch.bool)
    if pattern == "hot":
        ids.fill_(7)
    elif pattern == "mixed":
        ids = (torch.arange(t * k).reshape(t, k) % (c + 4) + 5).to(getattr(torch, dtype_name))
        mask = torch.arange(t) % 3 != 0
    expected = main_module.count_allgather_tokens(ids, mask, 7, c)
    ids = ids.npu()
    storage = torch.empty(t * stride, dtype=torch.bool, device="npu")
    mask_npu = storage[::stride]
    mask_npu.copy_(mask)
    graphs, outputs = {}, {}
    for name, fn in [
        ("main", main_module.count_allgather_tokens),
        ("two_stage", candidate_module.count_allgather_tokens),
    ]:
        for _ in range(5):
            value = fn(ids, mask_npu, 7, c)
        torch.testing.assert_close(value.cpu(), expected, rtol=0, atol=0)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            for _ in range(CALLS):
                value = fn(ids, mask_npu, 7, c)
        graphs[name], outputs[name] = graph, value
        for _ in range(5):
            graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(value.cpu(), expected, rtol=0, atol=0)
    samples = {name: [] for name in graphs}
    for trial in range(SAMPLES):
        order = list(graphs) if trial % 2 == 0 else list(reversed(graphs))
        for name in order:
            begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
            begin.record()
            graphs[name].replay()
            end.record()
            end.synchronize()
            samples[name].append(begin.elapsed_time(end) * 1000 / CALLS)
    for name, value in outputs.items():
        torch.testing.assert_close(value.cpu(), expected, rtol=0, atol=0, msg=lambda m, name=name: name + ": " + m)
    medians = {n: statistics.median(v) for n, v in samples.items()}
    return dict(
        status="pass",
        samples_us=samples,
        median_us=medians,
        min_us={n: min(v) for n, v in samples.items()},
        max_us={n: max(v) for n, v in samples.items()},
        speedup=medians["main"] / medians["two_stage"],
    )


for shape, dtype, stride in CASES:
    for pattern in ["uniform", "hot", "mixed"]:
        row = dict(
            shape=shape,
            dtype=dtype,
            mask_stride=stride,
            pattern=pattern,
            source="synthetic contract/boundary case",
            baseline_commit="3ae9de590",
            calls_per_graph=CALLS,
            samples=SAMPLES,
        )
        print("START", json.dumps(row), flush=True)
        try:
            row.update(measure(shape, dtype, stride, pattern))
        except Exception:
            row.update(status="failed", error=traceback.format_exc(), median_us=None, speedup=None)
        rows.append(row)
        (outdir / "expanded-graph.json").write_text(json.dumps(rows, indent=2))
        print("RESULT", json.dumps(row), flush=True)
        gc.collect()
print("COMPLETE", len(rows), sum(r["status"] == "pass" for r in rows), flush=True)
