# SPDX-License-Identifier: Apache-2.0
"""Compare owner-prefix output and NPUGraph latency for two source files."""

import argparse
import importlib.util
import json
import statistics
import sys

import torch
import torch_npu  # noqa: F401


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


parser = argparse.ArgumentParser()
parser.add_argument("--base", required=True, help="Path to the baseline sharded_indexer.py")
parser.add_argument("--candidate", required=True, help="Path to the candidate sharded_indexer.py")
args = parser.parse_args()
base = load("owner_prefix_base", args.base)
changed = load("owner_prefix_changed", args.candidate)
torch.npu.set_device(0)
print("versions", torch.__version__, torch_npu.__version__, flush=True)
K, INTERLEAVE = 2048, 768


def make_case(world, rows, mode):
    gen = torch.Generator().manual_seed(20883)
    local = torch.arange(K, dtype=torch.int32).view(1, 1, K).expand(world, rows, K).clone()
    scores = torch.randn(world, rows, K, generator=gen)
    owner = world // 2
    if mode == "skew":
        scores[:owner] += 10
        scores[owner + 1 :] -= 10
    elif mode == "partial":
        scores[:owner] += 10
        local[:owner, :, K // 2 :] = -1
        scores[owner + 1 :] -= 10
    elif mode == "ties":
        scores.fill_(0)
        scores[0, :, ::2] = -0.0
    elif mode == "empty":
        local[0] = -1
    valid = local >= 0
    gids = local + ((local // INTERLEAVE) * (world - 1) + torch.arange(world).view(-1, 1, 1)) * INTERLEAVE
    gids = torch.where(valid, gids, -1).int()
    scores = torch.where(valid, scores, -torch.inf)
    # Independently sort the concatenated global candidates for the reference.
    flat_ids = gids.permute(1, 0, 2).reshape(rows, -1)
    flat_scores = scores.permute(1, 0, 2).reshape(rows, -1)
    by_id = torch.where(flat_ids >= 0, flat_ids, torch.iinfo(torch.int32).max).argsort(dim=-1, stable=True)
    by_score = flat_scores.gather(-1, by_id).argsort(dim=-1, descending=True, stable=True)[:, :K]
    sel = flat_ids.gather(-1, by_id.gather(-1, by_score))
    owned = (sel >= 0) & ((sel // INTERLEAVE) % world == owner)
    local_sel = sel // (world * INTERLEAVE) * INTERLEAVE + sel % INTERLEAVE
    expected = torch.full((rows, K), -1, dtype=torch.int32)
    for row in range(rows):
        vals = local_sel[row][owned[row]]
        expected[row, : len(vals)] = vals.int()
    # Build the sorted per-rank packed representation on CPU.
    by_id = torch.where(gids >= 0, gids, torch.iinfo(torch.int32).max).argsort(dim=-1, stable=True)
    sorted_scores = scores.gather(-1, by_id)
    by_score = sorted_scores.argsort(dim=-1, descending=True, stable=True)
    sorted_ids = gids.gather(-1, by_id).gather(-1, by_score).contiguous()
    sorted_scores = sorted_scores.gather(-1, by_score).contiguous()
    packed = torch.stack((sorted_ids, sorted_scores.view(torch.int32)), dim=-1).reshape(world * rows, K, 2).npu()
    return packed, expected, owner


def run(module, packed, rows, world, owner, out):
    return module.merge_packed_candidates(
        packed, rows, world, K, local_rank=owner, interleave=INTERLEAVE, out=out, presorted=True, owner_sharded=True
    )


for world, rows in [(8, 32), (8, 128), (16, 48)]:
    for mode in ("balanced", "skew", "partial", "ties", "empty"):
        packed, expected, owner = make_case(world, rows, mode)
        outputs, graphs = [], []
        for module in (base, changed):
            out = torch.empty((rows, 1, K), dtype=torch.int32, device="npu")
            for _ in range(3):
                run(module, packed, rows, world, owner, out)
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                for _ in range(20):
                    run(module, packed, rows, world, owner, out)
            graph.replay()
            assert torch.equal(out.cpu().view(rows, K), expected), (world, rows, mode)
            outputs.append(out)
            graphs.append(graph)
        samples = [[], []]
        for group in range(10):
            for i in [0, 1] if group % 2 == 0 else [1, 0]:
                start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                start.record()
                graphs[i].replay()
                end.record()
                end.synchronize()
                samples[i].append(start.elapsed_time(end) * 1000 / 20)
        for out in outputs:
            assert torch.equal(out.cpu().view(rows, K), expected)
        print(
            json.dumps(
                {
                    "world": world,
                    "rows": rows,
                    "mode": mode,
                    "base_us": statistics.median(samples[0]),
                    "new_us": statistics.median(samples[1]),
                    "samples": samples,
                }
            ),
            flush=True,
        )
        for graph in graphs:
            graph.reset()
