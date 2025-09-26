import os
import json
from glob import glob
from tqdm import tqdm
import torch

try:
    import torch_npu
except:
    pass

from safetensors.torch import load_file, save_file
from huggingface_hub import snapshot_download


def weight_quant(tensor: torch.Tensor):
    assert tensor.dim() == 3
    qmax = 127.0
    abs_max = torch.abs(tensor).max(dim=-1, keepdim=True)[0]
    scale = abs_max / qmax
    assert scale.shape == (tensor.shape[0], 1)
    quantized = torch.round(tensor / scale)
    quantized = torch.clamp(quantized, -qmax, qmax)
    return quantized.to(torch.int8), scale.to(torch.float32)


def main(args, bf16_path, output_path, model_type):
    quant_prefix = "quant_model_weight_w8a8_dynamic"
    int8_names = []
    layer_num = 36 if model_type == "120b" else 24
    for i in range(layer_num):
        int8_names.append(f"model.layers.{i}.mlp.experts.down_proj.weight")
        int8_names.append(f"model.layers.{i}.mlp.experts.gate_up_proj.weight")

    torch.set_default_dtype(torch.bfloat16)
    os.makedirs(output_path, exist_ok=True)
    model_index_file = os.path.join(output_path, "model.safetensors.index.json")

    with open(model_index_file, "r") as f:
        model_index = json.load(f)
    weight_map = model_index["weight_map"]

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
            if weight_name in int8_names:
                assert weight.element_size() == 2
                quant_count += 1
                # print(weight_name, "int8")
                int8_weight, scale_inv = weight_quant(weight)
                new_state_dict[weight_name] = int8_weight
                new_scale_name = weight_name.replace(".weight", ".scale")
                new_state_dict[new_scale_name] = scale_inv

                new_weight_map[weight_name] = file_name
                new_weight_map[new_scale_name] = file_name
            else:
                new_state_dict[weight_name] = weight
                new_weight_map[weight_name] = file_name

        new_safetensor_file = os.path.join(output_path, file_name)
        save_file(new_state_dict, new_safetensor_file)

    print(f"{quant_count} weights are quantized")

    with open(model_index_file, "r") as f:
        model_index = json.load(f)
    model_index["weight_map"] = new_weight_map
    with open(model_index_file, "w", encoding="utf-8") as f:
        json.dump(model_index, f, indent=2, ensure_ascii=False, sort_keys=True)
    print(f"model.safetensors.index.json modified and saved to {model_index_file}")
