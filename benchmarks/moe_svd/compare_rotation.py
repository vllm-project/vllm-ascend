"""Isolate SVD truncation and factor INT4 quantization errors on a real expert."""

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

from vllm_ascend.quantization.moe_svd import (
    PROJECTIONS,
    dequantize_modelslim,
    quantize_factor,
    rotate_factors,
    svd_factors,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--expert", type=int, default=0)
    parser.add_argument("--rank", type=int, default=1024)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(123)
    root = Path(args.model)
    index = json.loads((root / "quant_model_weights.safetensors.index.json").read_text())["weight_map"]
    results = {}
    for projection in PROJECTIONS:
        prefix = f"model.layers.{args.layer}.mlp.experts.{args.expert}.{projection}"
        values = {}
        for suffix in ("weight", "weight_scale", "weight_offset"):
            key = prefix + "." + suffix
            with safe_open(root / index[key], framework="pt", device="cpu") as handle:
                values[suffix] = handle.get_tensor(key)
        dense = dequantize_modelslim(values["weight"], values["weight_scale"], values["weight_offset"])
        left, right, energy = svd_factors(dense, args.rank)
        rotated_left, rotated_right = rotate_factors(left, right)
        inputs = torch.randn(32, dense.shape[1])
        expected = inputs @ dense.T
        norm = expected.norm()
        metrics = {"svd_energy": energy}
        for label, factor_left, factor_right in (
            ("svd_only", left, right),
            ("plain_int4", left, right),
            ("rotated_int4", rotated_left, rotated_right),
        ):
            if label != "svd_only":
                factor_left = quantize_factor(factor_left).dequantize()
                factor_right = quantize_factor(factor_right).dequantize()
            actual = (inputs @ factor_right.T) @ factor_left.T
            metrics[label] = float((actual - expected).norm() / norm)
        results[projection] = metrics
        print(json.dumps({"projection": projection, **metrics}), flush=True)
    print(json.dumps({"layer": args.layer, "expert": args.expert, "results": results}, indent=2))


if __name__ == "__main__":
    main()
