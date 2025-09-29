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


def cal_scale(faquant_path, layer_idx, method='max'):
    tensors = []

    # 遍历路径下的所有 .pth 文件
    for fname in os.listdir(faquant_path):
        if fname.endswith(f"_{layer_idx}pth"):
            fpath = os.path.join(faquant_path, fname)
            t = torch.load(fpath, map_location="cpu")

            if isinstance(t, torch.Tensor):
                tensors.append(t)
            # 如果文件里是dict, 取第一个tensosr, 也可以根据key修改
            elif isinstance(t, dict):
                for v in t.values():
                    if isinstance(v, torch.Tensor):
                        tensors.append(v)
                        break

    if not tensors:
        raise ValueError(f"没有找到匹配 _{layer_idx}.pth 的tensor")

    merged = torch.cat(tensors, dim=-1)
    scale = (merged.max() / 127).clamp(min=1e-5).cpu()
    return scale


def main(args, model_path, faquant_path, kvs_safetensor_name, layer_num=61):
    model_config = os.path.join(model_path, "model.safetensors.index.json")
    with open(model_config, "r") as f:
        model_index = json.load(f)
    weight_map = model_index["weight_map"]
    faquant_scale = {}

    for layer_idx in range(layer_num):
        kvs = cal_scale(faquant_path, layer_idx)
        print(f"the kvscale of layer_idx={layer_idx} is {kvs}")
        faquant_scale[f'model.layers.{layer_idx}.self_attn.kv_scale'] = kvs
        weight_map[f'model.layers.{layer_idx}.self_attn.kv_scale'] = kvs_safetensor_name

    file = os.path.join(model_path, f"{kvs_safetensor_name}")

    if not os.path.exists(file):
        init_dict = {}
        save_file(init_dict, file)

    state_dict = load_file(file)
    model_index["weight_map"] = weight_map
    save_file(state_dict, file+'.bak')
    state_dict.update(faquant_scale)
    save_file(state_dict, file)
    with open(model_config, "w", encoding="utf-8") as f:
        json.dump(model_index, f, indent=2, ensure_ascii=False, sort_keys=True)
    print(f"save to {file}")
