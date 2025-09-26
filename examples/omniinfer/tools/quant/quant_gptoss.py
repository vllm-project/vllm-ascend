import optiquant.gpt_oss_int8 as qint8
import optiquant.faquant as faquant
from argparse import ArgumentParser
import json
import os

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input-bf16-hf-path", type=str, required=True, help="bf16 weight path")
    parser.add_argument("--output-path", type=str, required=True, help="quantized weight path")
    parser.add_argument("--device", type=str, required=True, help="support cpu and npu")
    parser.add_argument("--file_count", type=int, default=0, help="File count when loading model")
    parser.add_argument("--model-type", choices=["120b", "20b"], required=True, help="support gpt-oss-120b and gpt-oss-20b")

    args = parser.parse_args()

    qint8.main(args, args.input_bf16_hf_path, args.output_path, args.model_type)
    num_bits = 8

    quant_config = {
        "modules_to_not_convert": [
            "model.layer.*.self_attn",
            "model.layer.*.mlp.router",
            "model.embed_tokens",
            "lm_head",
        ],
        "quant_method": "mxfp4"
    }

    config_path = os.path.join(args.output_path, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)

    config["quantization_config"] = quant_config

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
