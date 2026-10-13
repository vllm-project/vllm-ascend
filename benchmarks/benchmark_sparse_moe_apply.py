"""Compare grouped Ascend MoE writes with vLLM checkpoint staging on one NPU.

This measures receiver apply only: transport, training diff/gather, model
forward and dynamic expert placement are excluded. Both paths include sparse
input validation. The reference loader uses the same transposed w13/w2 layout.
"""

import argparse
import json
import statistics
import time

import torch
import torch_npu  # noqa: F401
from vllm.distributed.weight_transfer.sparse_nccl_engine import SparseWeightPatch

from vllm_ascend.distributed.weight_transfer.sparse_weight_patch import (
    _apply_moe_group,
    _validate_sparse_hf_patches,
    apply_sparse_hf_patches_with_loader,
)


class ReferenceMoELoader(torch.nn.Module):
    """Final same-shaped copies after expert selection and HF transpose."""

    def __init__(self, experts, hidden, intermediate, device):
        super().__init__()
        self.intermediate = intermediate
        self.w13 = torch.nn.Parameter(
            torch.zeros(experts, hidden, 2 * intermediate, dtype=torch.bfloat16, device=device)
        )
        self.w2 = torch.nn.Parameter(torch.zeros(experts, intermediate, hidden, dtype=torch.bfloat16, device=device))

    def load_weights(self, weights):
        loaded = set()
        for name, source in weights:
            expert, projection = name.split(".")
            expert = int(expert)
            if projection == "down_proj":
                self.w2[expert].copy_(source.t())
            else:
                start = self.intermediate if projection == "up_proj" else 0
                self.w13[expert, :, start : start + self.intermediate].copy_(source.t())
            loaded.add(name)
        return loaded


@torch.no_grad()
def run(args):
    torch.npu.set_device(0)
    device = torch.device("npu:0")
    direct = ReferenceMoELoader(args.experts, args.hidden, args.intermediate, device)
    reference = ReferenceMoELoader(args.experts, args.hidden, args.intermediate, device)
    patches, w13_entries, w2_entries = [], [], []
    for expert in range(args.experts):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            shape = (args.hidden, args.intermediate) if projection == "down_proj" else (args.intermediate, args.hidden)
            indices = torch.arange(
                expert % args.stride, shape[0] * shape[1], args.stride, device=device, dtype=torch.int32
            )
            values = (indices % 97).to(torch.bfloat16) - 48
            patch = SparseWeightPatch(f"{expert}.{projection}", indices, values, shape)
            patches.append((patch, list(shape)))
            (w2_entries if projection == "down_proj" else w13_entries).append((patch, expert, projection))

    def direct_apply():
        _validate_sparse_hf_patches(patches)
        _apply_moe_group(direct.w13, w13_entries, (args.intermediate, args.hidden), 0, 1, False)
        _apply_moe_group(direct.w2, w2_entries, (args.hidden, args.intermediate), 0, 1, True)

    def reference_apply():
        apply_sparse_hf_patches_with_loader(reference, patches)

    results = {}
    for name, apply in (("grouped_moe", direct_apply), ("public_checkpoint_loader", reference_apply)):
        for _ in range(args.warmup):
            apply()
        torch.npu.synchronize()
        durations = []
        peak_bytes = 0
        for _ in range(args.repeat):
            torch.npu.reset_peak_memory_stats()
            baseline = torch.npu.memory_allocated()
            start = time.perf_counter()
            apply()
            torch.npu.synchronize()
            durations.append((time.perf_counter() - start) * 1000)
            peak_bytes = max(peak_bytes, torch.npu.max_memory_allocated() - baseline)
        results[name] = {
            "median_ms": statistics.median(durations),
            "samples_ms": durations,
            "incremental_peak_mib": peak_bytes / 2**20,
        }
    assert torch.equal(direct.w13.view(torch.int16), reference.w13.view(torch.int16))
    assert torch.equal(direct.w2.view(torch.int16), reference.w2.view(torch.int16))
    results.update(
        {
            "scope": "receiver apply plus validation; no transport/diff/gather/forward",
            "config": vars(args),
            "patches": len(patches),
            "sparse_entries": sum(patch.indices.numel() for patch, _ in patches),
            "full_runtime_bitwise_equal": True,
        }
    )
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experts", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--intermediate", type=int, default=768)
    parser.add_argument("--stride", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    run(parser.parse_args())
