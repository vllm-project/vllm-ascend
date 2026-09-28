"""Read one real expert and validate packed INT4 GMM plus SVD conversion cost."""

import argparse
import json
import time
from pathlib import Path

import torch
import torch_npu
from safetensors import safe_open

from vllm_ascend.quantization.moe_svd import dequantize_modelslim, quantize_factor, svd_factors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--rank", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    root = Path(args.model)
    index = json.loads((root / "quant_model_weights.safetensors.index.json").read_text())["weight_map"]
    prefix = "model.layers.3.mlp.experts.0.gate_proj"

    def load(suffix):
        key = prefix + "." + suffix
        with safe_open(root / index[key], framework="pt", device="cpu") as f:
            return f.get_tensor(key)

    packed, scale = load("weight"), load("weight_scale")
    dense = dequantize_modelslim(packed, scale, load("weight_offset"))
    stored_bias = load("scale_bias").flatten()
    expected_bias = 8 * dense.sum(dim=1)
    print("source_bias_max_error", (stored_bias - expected_bias).abs().max().item(), flush=True)
    start = time.monotonic()
    left, right, energy = svd_factors(dense, args.rank)
    print("svd_seconds", time.monotonic() - start, "energy", energy, flush=True)
    for label, factor in (("original", None), ("left", quantize_factor(left)), ("right", quantize_factor(right))):
        p, s = (packed, scale) if factor is None else (factor.weight, factor.scale)
        w = dense if factor is None else factor.dequantize()
        inputs = torch.randn(16, w.shape[1], dtype=torch.bfloat16).npu()
        quantized, input_scale = torch_npu.npu_dynamic_quant(inputs)
        weight = p.T.contiguous().view(torch.int32).unsqueeze(0).npu()
        scale_bits = s.flatten().contiguous().view(torch.int32).to(torch.int64)[None, None, :].npu()
        groups = torch.tensor([16], dtype=torch.int64, device="npu")
        reference = (quantized.cpu().float() * input_scale.cpu()[:, None]) @ w.T
        for bias_multiplier in (0, 1, -1, 8):
            bias = (bias_multiplier * w.sum(dim=1)).unsqueeze(0).npu()
            active_input = quantized.clone()
            output = (
                torch_npu.npu_grouped_matmul(
                    x=[active_input],
                    weight=[weight],
                    scale=[scale_bits],
                    bias=[bias],
                    per_token_scale=[input_scale],
                    group_list=groups,
                    split_item=2,
                    group_type=0,
                    group_list_type=1,
                    output_dtype=torch.bfloat16,
                )[0]
                .cpu()
                .float()
            )
            error = (output - reference).norm() / reference.norm()
            print(
                label,
                "bias_multiplier",
                bias_multiplier,
                "relative_error",
                error.item(),
                "input_mutated",
                not torch.equal(active_input.cpu(), quantized.cpu()),
                flush=True,
            )
        output = (
            torch_npu.npu_grouped_matmul(
                x=[inputs],
                weight=[weight],
                antiquant_scale=[s.T.unsqueeze(0).to(torch.bfloat16).npu()],
                antiquant_offset=[torch.zeros(1, 1, w.shape[0], dtype=torch.bfloat16, device="npu")],
                group_list=groups,
                split_item=2,
                group_type=0,
                group_list_type=1,
                output_dtype=torch.bfloat16,
            )[0]
            .cpu()
            .float()
        )
        reference = inputs.cpu().float() @ w.T
        print(label, "W4A16_error", ((output - reference).norm() / reference.norm()).item(), flush=True)


if __name__ == "__main__":
    main()
