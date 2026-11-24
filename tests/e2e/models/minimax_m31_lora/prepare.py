# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Freeze real M3 Q/K/V/O mappings for the external MindSpeed LoRA runner."""

import argparse
import inspect
import json
import sys
from pathlib import Path


def build_mappings(model_config, ascend):
    text = model_config.get("text_config", model_config)
    hidden = text["hidden_size"]
    head = text["head_dim"]
    q_width = text["num_attention_heads"] * head
    kv_width = text["num_key_value_heads"] * head
    modules, mappings = [], []
    for layer in range(text["num_hidden_layers"]):
        for target, width, inputs in (
            ("q", q_width, hidden),
            ("k", kv_width, hidden),
            ("v", kv_width, hidden),
            ("o", hidden, q_width),
        ):
            name = f"language_model.model.layers.{layer}.self_attn.{target}_proj"
            identity = f"l{layer}-{target}"
            modules.append(
                {
                    "id": identity,
                    "module": name,
                    "layout": "linear_2d",
                    "in_features": inputs,
                    "out_features": width,
                    "a_key": "base_model.model." + name + ".lora_A.weight",
                    "b_key": "base_model.model." + name + ".lora_B.weight",
                }
            )
            mapping = {
                "id": identity,
                "runtime_module": f"model.language_model.model.layers.{layer}.self_attn."
                + ("o_proj" if target == "o" else "qkv_proj"),
                "features": [] if target == "o" else ["fused"],
                "source_evidence": [
                    {
                        "path": str(ascend / "vllm_ascend/models/minimax_m3/minimax_m3.py"),
                        "symbol": "self.o_proj" if target == "o" else "self.qkv_proj",
                    }
                ],
            }
            if target != "o":
                mapping["observer"] = {
                    "kind": "qkv",
                    "slice_sizes": [q_width, kv_width, kv_width],
                    "slice_index": ["q", "k", "v"].index(target),
                    "buffer_layout": "slices",
                }
            mappings.append(mapping)
    return modules, mappings


def source_files(ascend, vllm):
    from transformers.models.minimax_m3_vl import modeling_minimax_m3_vl

    files = {
        ascend: [
            "vllm_ascend/worker/worker.py",
            "vllm_ascend/worker/v2/model_runner.py",
            "vllm_ascend/lora/utils.py",
            "vllm_ascend/lora/lora_ops.py",
            "vllm_ascend/lora/punica_npu.py",
            "vllm_ascend/models/minimax_m3/minimax_m3.py",
            "vllm_ascend/models/minimax_m3/minimax_m3_vl.py",
            "vllm_ascend/models/minimax_m3/msa_m3.py",
            "vllm_ascend/ops/linear.py",
        ],
        vllm: [
            "vllm/lora/worker_manager.py",
            "vllm/lora/model_manager.py",
            "vllm/lora/lora_model.py",
            "vllm/lora/layers/base.py",
            "vllm/lora/layers/base_linear.py",
            "vllm/lora/layers/column_parallel_linear.py",
            "vllm/lora/layers/row_parallel_linear.py",
        ],
    }
    return [str(root / name) for root, names in files.items() for name in names] + [
        str(Path(__file__).resolve()),
        str(Path(__file__).with_name("m31_lora_driver.py").resolve()),
        str(Path(__file__).with_name("m31_observation.py").resolve()),
        inspect.getsourcefile(modeling_minimax_m3_vl),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--vllm", required=True, type=Path)
    parser.add_argument("--engine-root", required=True, type=Path)
    parser.add_argument("--directory", required=True, type=Path)
    args = parser.parse_args()
    model, vllm, engine = (p.resolve() for p in (args.model, args.vllm, args.engine_root))
    ascend = Path(__file__).resolve().parents[4]
    modules, mappings = build_mappings(json.loads((model / "config.json").read_text()), ascend)
    spec = {
        "schema_version": 1,
        "base_model_name_or_path": str(model),
        "layout": "linear_2d",
        "r": 8,
        "lora_alpha": 16,
        "seed": 20261124,
        "amplitude": 1.5,
        "modules": modules,
    }
    config = {
        "schema_version": 1,
        "model": str(model),
        "spec": "spec.json",
        "adapter_format": "safetensors",
        "tolerances": {"atol": 1 / 64, "rtol": 1 / 64},
        "combinations": [{"id": "npu-eager", "backend": "npu"}],
        "mappings": mappings,
        "model_files": [
            str(p.resolve())
            for p in model.rglob("*")
            if p.is_file()
            and (
                p.suffix in {".safetensors", ".bin"}
                or p.name in {"config.json", "reduced_checkpoint.json"}
                or p.name.endswith(".index.json")
            )
        ],
        "source_files": source_files(ascend, vllm),
        "prompt_token_ids": [1, 3, 4],
        "observation_max_elements": 80_000_000,
        "vllm_options": {
            "dtype": "bfloat16",
            "gpu_memory_utilization": 0.45,
            "max_model_len": 64,
            "max_num_batched_tokens": 64,
        },
        "drivers": {
            "npu-eager": {
                "argv": [
                    sys.executable,
                    str(Path(__file__).with_name("m31_lora_driver.py").resolve()),
                    "--engine-root",
                    str(engine),
                    "--request",
                    "{request}",
                    "--output",
                    "{output}",
                ],
                "timeout_seconds": 900,
            }
        },
        "timeout_seconds": 900,
    }
    args.directory.mkdir(parents=True, exist_ok=False)
    for name, value in (("spec", spec), ("config", config)):
        (args.directory / (name + ".json")).write_text(json.dumps(value, indent=2))
    print(f"Frozen {len(modules)} targets in {args.directory}")


if __name__ == "__main__":
    main()
