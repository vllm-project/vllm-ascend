# SPDX-License-Identifier: Apache-2.0
"""Compare exact DCP candidate preparation with a frozen reference module."""

import argparse
import hashlib
import importlib.util
import json
import traceback
from pathlib import Path

import torch
from dcp_four_point_ab import candidate_inputs, check, local_reference, measure

from vllm_ascend.ops.triton import sharded_indexer as candidate
from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num, init_device_properties_triton


def grouped_merge(packed, rows, world, topk, rank, group_size):
    if not group_size:
        init_device_properties_triton()
        output = torch.empty((rows, 1, topk), dtype=torch.int32, device=packed.device)
        candidate._merge_candidates[(min(rows, get_vectorcore_num()),)](
            packed,
            output,
            rows,
            topk,
            topk,
            world,
            rank,
            768,
            topk.bit_length() - 1,
            True,
            OWNER_WORLD=world,
            multibuffer=False,
        )
        return output
    init_device_properties_triton()
    groups = (world + group_size - 1) // group_size
    partial = torch.empty((groups * rows, topk, 2), dtype=torch.int32, device=packed.device)
    output = torch.empty((rows, 1, topk), dtype=torch.int32, device=packed.device)
    candidate._merge_candidates[(min(rows * groups, get_vectorcore_num()),)](
        packed,
        partial,
        rows,
        topk,
        topk,
        world,
        -1,
        768,
        topk.bit_length() - 1,
        True,
        GROUP_SIZE=group_size,
        OWNER_WORLD=world,
        multibuffer=False,
    )
    candidate._merge_candidates[(min(rows, get_vectorcore_num()),)](
        partial,
        output,
        rows,
        topk,
        topk,
        groups,
        rank,
        768,
        topk.bit_length() - 1,
        True,
        GROUP_SIZE=0,
        OWNER_WORLD=world,
        multibuffer=False,
    )
    return output


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", required=True)
    parser.add_argument("--expected-source-root", required=True)
    parser.add_argument("--expected-source-sha", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--world", type=int, default=16)
    parser.add_argument("--score-dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--group-sizes", type=int, nargs="+", default=[0, 2, 4, 8])
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--only", nargs="+")
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    if min(args.rows, args.world, args.iterations, args.repeats) <= 0 or any(size < 0 for size in args.group_sizes):
        parser.error("rows/world/iterations/repeats must be positive and group sizes must be nonnegative")
    root = Path(args.expected_source_root).resolve()
    relative = Path("vllm_ascend/ops/triton/sharded_indexer.py")
    if Path(candidate.__file__).resolve() != root / relative:
        raise RuntimeError(f"Unexpected candidate import: {candidate.__file__}")
    source_hash = hashlib.sha256((root / relative).read_bytes()).hexdigest()
    if source_hash != args.expected_source_sha:
        raise RuntimeError(f"Candidate source hash mismatch: {source_hash}")
    if "GROUP_SIZE" not in candidate._merge_candidates.arg_names:
        raise RuntimeError("Loaded candidate does not contain the grouped merge kernel")
    reference_path = Path(args.reference_root).resolve() / relative
    spec = importlib.util.spec_from_file_location("reference_dcp_candidates", reference_path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    torch.npu.set_device(0)
    dtype = torch.float32 if args.score_dtype == "fp32" else torch.bfloat16
    ids, scores, packed, winners = candidate_inputs(args.rows, args.world, score_dtype=dtype)
    packed = packed.view(args.world, args.rows, 2048, 2)
    valid = (packed[..., 0] >= 0) & torch.isfinite(packed[..., 1].contiguous().view(torch.float32))
    keys = torch.where(valid, packed[..., 0].float(), torch.inf)
    keys, id_order = torch.sort(keys, stable=True)
    values = torch.gather(torch.where(valid, packed[..., 1].contiguous().view(torch.float32), -torch.inf), -1, id_order)
    values, score_order = torch.sort(values, descending=True, stable=True)
    keys = torch.gather(keys, -1, score_order)
    keys = torch.where(torch.isfinite(values), keys, -1).int()
    sorted_packed = torch.stack((keys, values.contiguous().view(torch.int32)), -1)
    gathered = sorted_packed.reshape(-1, 2048, 2).npu()
    rank = args.world - 1
    local_ids, local_scores = ids[rank].npu(), scores[rank].npu()
    expected = local_reference(winners, rank, args.world)

    def pack(module):
        return module.pack_local_candidates(local_ids, local_scores, args.world, rank, 768, sort_candidates=True)

    def merge_reference():
        return reference.merge_packed_candidates(
            gathered, args.rows, args.world, 2048, local_rank=rank, interleave=768, presorted=True
        )

    def merge_candidate():
        return candidate.merge_packed_candidates(
            gathered, args.rows, args.world, 2048, local_rank=rank, interleave=768, presorted=True
        )

    def compute(module, merge):
        # Fixed independently prepared peers exclude HCCL, matching the earlier
        # production-compute benchmark. Retain the local pack until merge ends.
        local = pack(module)
        output = merge()
        del local
        return output

    cases = [
        ("pack_reference", lambda: pack(reference), sorted_packed[rank]),
        ("pack_candidate", lambda: pack(candidate), sorted_packed[rank]),
        ("merge_reference", merge_reference, expected),
        ("compute_reference", lambda: compute(reference, merge_reference), expected),
        ("compute_candidate", lambda: compute(candidate, merge_candidate), expected),
    ]
    for size in args.group_sizes:

        def merge(size=size):
            return grouped_merge(gathered, args.rows, args.world, 2048, rank, size)

        cases.append((f"merge_group{size}", merge, expected))
        cases.append((f"compute_group{size}", lambda merge=merge: compute(candidate, merge), expected))
    result = {
        "args": vars(args),
        "device": torch.npu.get_device_name(),
        "source_hash": source_hash,
        "reference_hash": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        "cases": [],
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    if args.reverse:
        cases.reverse()
    if args.only and set(args.only) - {name for name, _, _ in cases}:
        parser.error("--only contains an unknown benchmark case")
    for name, fn, oracle in cases:
        if args.only and name not in args.only:
            continue
        entry = {"name": name}
        try:
            check(fn(), oracle, exact=True)
            graph = torch.npu.NPUGraph()
            try:
                with torch.npu.graph(graph):
                    captured = fn()
                graph.replay()
                torch.npu.synchronize()
                check(captured, oracle, exact=True)
            finally:
                graph.reset()
            entry["validation"] = "pass"
            for mode in ("eager", "graph"):
                entry[mode] = measure(fn, mode, args.iterations, args.repeats)
        except Exception:
            entry["error"] = traceback.format_exc()
        result["cases"].append(entry)
        target.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(entry), flush=True)
    if any("error" in case for case in result["cases"]):
        raise RuntimeError(f"Candidate comparison failed; see {target}")


if __name__ == "__main__":
    main()
