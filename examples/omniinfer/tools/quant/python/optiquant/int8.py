import json
import os
from glob import glob

import torch
from tqdm import tqdm

try:
    import torch_npu
except:
    pass

from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file


def weight_quant(tensor: torch.Tensor):
    assert tensor.dim() == 2
    qmax = 127.0
    abs_max = torch.abs(tensor).max(dim=1, keepdim=True)[0]
    scale = abs_max / qmax
    assert scale.shape == (tensor.shape[0], 1)
    quantized = torch.round(tensor / scale)
    quantized = torch.clamp(quantized, -qmax, qmax)
    return quantized.to(torch.int8), scale.to(torch.float32)


def main(args, bf16_path, output_path, next, pangu_mode, model_name="deepseek-ai/DeepSeek-R1"):
    quant_prefix = "quant_model_weight_w8a8_dynamic"
    disable_names = []
    for i in range(62):
        disable_names.append(f"model.layers.{i}.self_attn.kv_b_proj.weight")
        disable_names.append(f"model.layers.{i}.mlp.gate.weight")
        disable_names.append(f"model.layers.{i}.mlp.gate.e_score_correction_bias")

    disable_names.append("lm_head")
    disable_names.append("model.embed_tokens.weight")

    torch.set_default_dtype(torch.bfloat16)
    os.makedirs(output_path, exist_ok=True)
    model_index_file = os.path.join(output_path, "model.safetensors.index.json")
    config_file = os.path.join(output_path, "config.json")

    if not os.path.exists(model_index_file) or not os.path.exists(config_file):
        snapshot_download(
            repo_id=model_name,
            ignore_patterns=["*.safetensors"],
            local_dir=output_path,
            local_dir_use_symlinks=False
        )
        print(f"model index file and config file download to {output_path}")

    with open(model_index_file, "r") as f:
        model_index = json.load(f)
    weight_map = model_index["weight_map"]
    scale_count = len([key for key in weight_map.keys() if key.endswith("_scale_inv")])

    safetensor_files = list(glob(os.path.join(bf16_path, "*.safetensors")))
    safetensor_files.sort()
    if args.file_count:
        safetensor_files = safetensor_files[:args.file_count]

    quant_count = 0
    new_weight_map = {}

    for safetensor_file in tqdm(safetensor_files):
        file_name = os.path.basename(safetensor_file)
        file_name = file_name.replace("model", quant_prefix)

        state_dict = load_file(safetensor_file, device=args.device)
        new_state_dict = {}
        for weight_name, weight in state_dict.items():
            if weight_name in disable_names or "norm" in weight_name or "indexer" in weight_name:
                print(weight_name, "bf16")
                new_state_dict[weight_name] = weight
                new_weight_map[weight_name] = file_name
                continue
            scale_inv_name = f"{weight_name}_scale_inv"
            if scale_inv_name in weight_map or pangu_mode:
                assert weight.element_size() == 2
                quant_count += 1
                print(weight_name, "int8")
                int8_weight, scale_inv = weight_quant(weight)
                new_state_dict[weight_name] = int8_weight
                new_scale_name = scale_inv_name.replace("_scale_inv", "_scale")
                new_state_dict[new_scale_name] = scale_inv

                new_weight_map[weight_name] = file_name
                new_weight_map[new_scale_name] = file_name
            else:
                new_state_dict[weight_name] = weight
                new_weight_map[weight_name] = file_name

        new_safetensor_file = os.path.join(output_path, file_name)
        save_file(new_state_dict, new_safetensor_file)

    print(quant_count, scale_count)
    print(f"{quant_count} weights are quantized")

    with open(model_index_file, "r") as f:
        model_index = json.load(f)
    model_index["weight_map"] = new_weight_map
    with open(model_index_file, "w", encoding="utf-8") as f:
        json.dump(model_index, f, indent=2, ensure_ascii=False, sort_keys=True)
    print(f"model.safetensors.index.json modified and saved to {model_index_file}")
