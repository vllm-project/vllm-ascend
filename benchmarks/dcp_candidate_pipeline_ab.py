# SPDX-License-Identifier: Apache-2.0
"""Compare exact pack, owner merge and dependent compute; excludes real HCCL."""

import argparse
import importlib.util
import inspect
import json
from pathlib import Path

import torch
import torch_npu  # noqa: F401
from dcp_four_point_ab import candidate_inputs, local_reference, measure

from vllm_ascend.ops.triton import sharded_indexer as candidate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=48)
    parser.add_argument("--world", type=int, default=16)
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--pattern", choices=["random", "ties", "owner_wins", "peer_wins"], default="random")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument(
        "--fixed-merge", action="store_true", help="Use the candidate merge for every preparation variant"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, default=Path("src/baseline"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.npu.set_device(0)
    path = (args.reference_root / "vllm_ascend/ops/triton/sharded_indexer.py").resolve()
    spec = importlib.util.spec_from_file_location("frozen_pipeline", path)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    ids, scores, _, winners = candidate_inputs(
        args.rows, args.world, score_dtype=torch.float32 if args.dtype == "fp32" else torch.bfloat16
    )
    owner = args.world - 1
    if args.pattern != "random":
        if args.pattern == "ties":
            scores = torch.ones_like(scores)
        elif args.pattern == "owner_wins":
            scores[owner] += 100
        elif args.world > 1:
            scores[0] += 100
        ranks = torch.arange(args.world, dtype=torch.int32).view(args.world, 1, 1, 1)
        mapped = (ids // 768 * args.world + ranks) * 768 + ids % 768
        valid = (ids >= 0) & torch.isfinite(scores)
        keys = torch.where(valid, mapped, torch.iinfo(torch.int32).max)
        values = torch.where(valid, scores.float(), -torch.inf)
        keys = keys[:, :, 0].permute(1, 0, 2).reshape(args.rows, -1)
        values = values[:, :, 0].permute(1, 0, 2).reshape(args.rows, -1)
        order = keys.argsort(stable=True)
        ordered_values = values.gather(-1, order)
        selection = ordered_values.argsort(descending=True, stable=True)[:, :2048]
        selected_keys = keys.gather(-1, order).gather(-1, selection)
        winners = torch.where(torch.isfinite(ordered_values.gather(-1, selection)), selected_keys, -1)[:, None]
    local_ids, local_scores = ids[owner].npu(), scores[owner].npu()
    gathered = torch.cat(
        [
            baseline.pack_local_candidates(ids[r].npu(), scores[r].npu(), args.world, r, 768, sort_candidates=True)
            for r in range(args.world)
        ]
    )
    packed_expected = gathered[owner * args.rows : (owner + 1) * args.rows].cpu()
    expected = local_reference(winners, owner, args.world)
    modules = [("baseline", baseline), ("candidate", candidate)]
    if args.fixed_merge:
        modules.insert(0, ("native", baseline))
    if args.reverse:
        modules.reverse()
    records = []
    for name, module in modules:
        merge_module = candidate if args.fixed_merge else module
        extra = (
            {"owner_sharded": True}
            if "owner_sharded" in inspect.signature(merge_module.merge_packed_candidates).parameters
            else {}
        )
        pack_extra = (
            {"fuse_sort": name != "native"}
            if "fuse_sort" in inspect.signature(module.pack_local_candidates).parameters
            else {}
        )

        def pack(module=module, pack_extra=pack_extra):
            return module.pack_local_candidates(
                local_ids, local_scores, args.world, owner, 768, sort_candidates=True, **pack_extra
            )

        def merge(merge_module=merge_module, extra=extra):
            return merge_module.merge_packed_candidates(
                gathered, args.rows, args.world, 2048, local_rank=owner, interleave=768, presorted=True, **extra
            )

        def compute(pack=pack, merge=merge):
            packed = pack()
            # Peer data is preloaded. Include a dependent copy of this rank's
            # actual pack output; this is a communication surrogate, not HCCL.
            gathered[owner * args.rows : (owner + 1) * args.rows].copy_(packed)
            return merge()

        for section, run, oracle in [
            ("pack", pack, packed_expected),
            ("merge", merge, expected),
            ("compute", compute, expected),
        ]:
            torch.testing.assert_close(run().cpu(), oracle, rtol=0, atol=0)
            latest = []

            def checked_run(run=run, latest=latest):
                value = run()
                latest[:] = [value]
                return value

            timing = measure(checked_run, "graph", args.iterations, args.repeats)
            torch.testing.assert_close(latest[0].cpu(), oracle, rtol=0, atol=0)
            entry = dict(
                name=name,
                section=section,
                rows=args.rows,
                world=args.world,
                dtype=args.dtype,
                pattern=args.pattern,
                fixed_merge=args.fixed_merge,
                correctness="exact",
                iterations=args.iterations,
                repeats=args.repeats,
                timing=timing,
            )
            records.append(entry)
            args.output.write_text(json.dumps(records, indent=2) + "\n")
            print(json.dumps(entry), flush=True)


if __name__ == "__main__":
    main()
