import json
import os
from argparse import ArgumentParser

import optiquant.faquant as faquant
import optiquant.qwen_int8 as qint8

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input-bf16-hf-path", type=str, required=True, help="bf16 weight path")
    parser.add_argument("--output-path", type=str, required=True, help="quantized weight path")
    parser.add_argument("--device", type=str, required=True, help="support cpu and npu")
    parser.add_argument("--file_count", type=int, default=0, help="File count when loading model")
    parser.add_argument("--model-name", type=str, default="deepseek-ai/DeepSeek-R1", help="Huggingface repo name")

    args = parser.parse_args()

    qint8.main(args, args.input_bf16_hf_path, args.output_path, args.model_name)
    num_bits = 8

    ignores = []
    for i in range(94):
        qkv = f"model.layers.{i}.self_attn.qkv_proj"
        o = f"model.layers.{i}.self_attn.o_proj"
        gate = f"model.layers.{i}.mlp.gate"
        ignores.extend([qkv, o, gate])
    for i in range(86, 94):
        experts = f"model.layers.{i}.mlp.experts"
        ignores.append(experts)

    quant_config = {"config_groups": {"group_0": {}}, "format": "int-quantized",
                    "global_compression_ratio": None, "ignore": ignores, "kv_cache_scheme": None,
                    "quant_method": "npu_w4a8_dynamic", "quantization_status": "compressed"}
    quant_config["config_groups"]["group_0"]["input_activations"] = {"actorder": None, "block_structure": None,
                                                                     "dynamic": True, "group_size": None, "num_bits": 8,
                                                                     "observer": "memoryless", "observer_kwargs": {},
                                                                     "strategy": "token", "symmetric": True,
                                                                     "type": "int"}
    quant_config["config_groups"]["group_0"]["output_activations"] = None
    quant_config["config_groups"]["group_0"]["targets"] = ["Linear"]
    quant_config["config_groups"]["group_0"]["weights"] = {"actorder": None, "block_structure": None, "dynamic": False,
                                                           "group_size": None, "num_bits": 8,
                                                           "observer": "minmax", "observer_kwargs": {},
                                                           "strategy": "channel", "symmetric": True, "type": "int"}

    config_path = os.path.join(args.output_path, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)

    config["quantization_config"] = quant_config

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
