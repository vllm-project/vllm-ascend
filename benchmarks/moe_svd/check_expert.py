"""Compare converted projections with the source W4 checkpoint on random inputs.

This is a local approximation diagnostic, not a model accuracy evaluation.
"""

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

from vllm_ascend.ops.fused_moe.dataclass.fused_experts import MoELowRankLinear
from vllm_ascend.ops.fused_moe.moe_low_rank import low_rank_linear
from vllm_ascend.quantization.moe_svd import PROJECTIONS, dequantize_modelslim


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--converted", required=True)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--expert", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(123)
    source, converted = Path(args.source), Path(args.converted)
    index = json.loads((source / "quant_model_weights.safetensors.index.json").read_text())["weight_map"]

    def load(key):
        with safe_open(source / index[key], framework="pt", device="cpu") as handle:
            return handle.get_tensor(key)

    results = {}
    shard = converted / f"svd-layer{args.layer:03d}-expert{args.expert:03d}.safetensors"
    with safe_open(shard, framework="pt", device="cpu") as handle:
        for projection in PROJECTIONS:
            prefix = f"model.layers.{args.layer}.mlp.experts.{args.expert}.{projection}"
            original = dequantize_modelslim(
                load(prefix + ".weight"), load(prefix + ".weight_scale"), load(prefix + ".weight_offset")
            )
            payload = {}
            for side in ("left", "right"):
                payload[side] = (
                    handle.get_tensor(f"{prefix}.{side}_weight").T.contiguous().view(torch.int32)[None].npu()
                )
                payload[side + "_scale"] = (
                    handle.get_tensor(f"{prefix}.{side}_scale")
                    .T.contiguous()
                    .view(torch.int32)
                    .to(torch.int64)[None]
                    .npu()
                )
                payload[side + "_bias"] = handle.get_tensor(f"{prefix}.{side}_bias")[None].npu()
            x = torch.randn(16, original.shape[1], dtype=torch.bfloat16)
            output = (
                low_rank_linear(
                    x.npu(), MoELowRankLinear(**payload), torch.tensor([16], device="npu", dtype=torch.int64)
                )
                .cpu()
                .float()
            )
            reference = x.float() @ original.T
            results[projection] = {
                "relative_output_rmse": float((output - reference).norm() / reference.norm()),
                "cosine": float(torch.nn.functional.cosine_similarity(output.flatten(), reference.flatten(), dim=0)),
            }
    print(json.dumps({"layer": args.layer, "expert": args.expert, "random_input_diagnostic": results}, indent=2))


if __name__ == "__main__":
    main()
