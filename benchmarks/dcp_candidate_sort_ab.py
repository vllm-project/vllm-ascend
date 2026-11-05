# SPDX-License-Identifier: Apache-2.0
"""Measure independent pack and merge sections, excluding HCCL."""

import argparse
import importlib.util
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--section", choices=["pack", "merge"], default="pack")
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--score-dtype", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--reference-root", type=Path, default=Path("src/baseline"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.npu.set_device(0)
    path = (args.reference_root / "vllm_ascend/ops/triton/sharded_indexer.py").resolve()
    spec = importlib.util.spec_from_file_location("frozen_indexer", path)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    ids, scores, _, winners = candidate_inputs(
        args.rows, args.world, score_dtype=torch.float32 if args.score_dtype == "fp32" else torch.bfloat16
    )
    owner = args.world - 1
    local_ids, local_scores = ids[owner].npu(), scores[owner].npu()
    gathered = torch.cat(
        [
            baseline.pack_local_candidates(ids[r].npu(), scores[r].npu(), args.world, r, 768, sort_candidates=True)
            for r in range(args.world)
        ]
    )
    expected = (
        gathered.reshape(args.world, args.rows, 2048, 2)[owner].cpu()
        if args.section == "pack"
        else local_reference(winners, owner, args.world)
    )
    records = []
    cases = [("baseline", baseline), ("candidate", candidate)]
    if args.reverse:
        cases.reverse()
    for name, module in cases:

        def run(module=module):
            if args.section == "pack":
                return module.pack_local_candidates(
                    local_ids, local_scores, args.world, owner, 768, sort_candidates=True
                )
            return module.merge_packed_candidates(
                gathered, args.rows, args.world, 2048, local_rank=owner, interleave=768, presorted=True
            )

        for _ in range(5):
            actual = run()
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
        entry = dict(name=name, section=args.section, graph=measure(run, "graph", 20, 7), correctness="exact")
        records.append(entry)
        args.output.write_text(json.dumps(records, indent=2) + "\n")
        print(json.dumps(entry), flush=True)


if __name__ == "__main__":
    main()
