import json
import os
import re
from copy import deepcopy
from glob import glob

import torch
from tqdm import tqdm

try:
    import torch_npu
except:
    pass

from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file


class QType:
    desc: str
    exp_bits: int = -1
    man_bits: int = -1
    k_bits: int = -1
    k_outer_bits: int = 0
    blk_size: int = -1
    exp_offset: int = 0
    numbits: int = 4
    is_act_integer: bool = False

    def __init__(self, desc: str):
        self.desc = desc
        if desc.lower()[:3] == 'ssz':
            res = re.match(
                r'sszs([0-9]+)g([0-9]+)a([0-9]+)b([0-9]+)sym([0-9]+)$', desc)
            self.num_step = int(res.group(1))
            self.blk_size = int(res.group(2))
            self.is_act_integer = True if int(res.group(3)) > 0 else False
            self.numbits = int(res.group(4))
            self.ssz_sym = True if int(res.group(5)) > 0 else False

    def dim_(self, dim: int):
        self.q_dim = dim
        return self

    def dim(self, dim: int):
        out = deepcopy(self)
        out.q_dim = dim
        return out

    def copy(self):
        return deepcopy(self)

    def __repr__(self) -> str:
        return str(
            f'QType: {self.desc}   Dim: {self.q_dim}   ExpOffset: {self.exp_offset}'
        )


def get_qbits_minmax(numbits, is_sym):
    BIT_MAX = 2**(numbits - 1) - 1
    BIT_MIN = -BIT_MAX if is_sym else -BIT_MAX - 1
    return BIT_MIN, BIT_MAX


def get_scale_offset(x, qW_min, qW_max, is_sym, is_act_integer):
    scale = None
    offset = None
    if is_sym:
        xmax = torch.abs(x).max(dim=-1, keepdim=True)[0]
        if is_act_integer:
            scale = torch.round(xmax / qW_max).clamp(min=1)
        else:
            scale = (xmax / qW_max).clamp(min=1e-5)
    else:
        xmax = x.max(dim=-1, keepdim=True)[0]
        xmin = x.min(dim=-1, keepdim=True)[0]
        compare = ((xmax - xmin) < 1e-5).to(torch.int32)
        xmax = xmax * (1 - compare) + torch.max(torch.abs(xmax),
                                                torch.abs(xmin)) * compare
        xmin = xmin * (1 - compare)
        scale = (xmax - xmin).clamp(min=1e-5) / (qW_max - qW_min)
        if is_act_integer:
            scale = torch.round(scale).clamp(min=1)
        else:
            scale = scale.clamp(min=1e-5)
        offset = torch.round(-xmin / scale) + qW_min
    return scale, offset


def get_quant(x, qW_min, qW_max, scale, offset=None):
    if offset is not None:
        return torch.round(x / scale + offset).clamp(min=qW_min, max=qW_max)
    else:
        return torch.round(x / scale).clamp(min=qW_min, max=qW_max)


def get_dequant(x_quant, qW_min, qW_max, scale, offset=None):
    if offset is not None:
        return (x_quant - offset) * scale
    else:
        return x_quant * scale


L_mode = 2


def get_mseloss(x,
                qW_min,
                qW_max,
                scale=None,
                offset=None,
                quant=None,
                dequant=None,
                mode=L_mode):
    if quant is None and dequant is None:
        quant = get_quant(x, qW_min, qW_max, scale, offset)
    if dequant is None:
        dequant = get_dequant(quant, qW_min, qW_max, scale, offset)
    return torch.mean(torch.pow(torch.abs(x - dequant), mode),
                      dim=-1,
                      keepdim=True)


loss_function = get_mseloss


def quant_ssz(x: torch.Tensor,
              Q: QType,
              qdim: int,
              init_scale=None,
              init_offset=None,
              init_quant=None,
              w8=False,
              w4=False):
    num_step = Q.num_step
    groupsize = Q.blk_size
    is_act_integer = Q.is_act_integer
    numbits = Q.numbits
    is_ssz_sym = Q.ssz_sym
    shape = x.shape
    if groupsize != 0:
        shaped_x = x.view(-1, groupsize)
    else:
        shaped_x = x
        groupsize = shaped_x.shape[-1]
    qW_min, qW_max = get_qbits_minmax(numbits, is_sym=is_ssz_sym)

    if init_offset is not None and init_scale is not None and init_quant is not None:
        scale, offset, quant = init_scale, init_offset, init_quant
    else:
        scale, offset = get_scale_offset(shaped_x,
                                         qW_min,
                                         qW_max,
                                         is_sym=is_ssz_sym,
                                         is_act_integer=is_act_integer)
        scale = scale.clamp(min=1e-5)
        quant = get_quant(shaped_x, qW_min, qW_max, scale, offset=offset)

    dequant = get_dequant(quant, qW_min, qW_max, scale, offset=offset)
    bestScale = scale
    if is_ssz_sym:
        offset = 0
    bestOffset = offset
    bestQuant = quant
    bestMse = loss_function(shaped_x,
                            qW_min,
                            qW_max,
                            scale=scale,
                            offset=offset,
                            quant=quant,
                            dequant=dequant)
    for i in range(num_step):
        a = quant - offset
        if is_act_integer:
            scale = torch.round(
                torch.sum(a * shaped_x, dim=-1, keepdim=True) /
                torch.sum(a * a, dim=-1, keepdim=True)).clamp(min=1)
        else:
            scale = torch.sum(a * shaped_x, dim=-1, keepdim=True) / torch.sum(
                a * a, dim=-1, keepdim=True).clamp(min=1e-5)
        offset = torch.round(
            torch.sum(quant * scale - shaped_x, dim=-1, keepdim=True) /
            groupsize / scale)
        if is_ssz_sym:
            offset = 0
        quant = get_quant(shaped_x, qW_min, qW_max, scale, offset=offset)
        dequant = get_dequant(quant, qW_min, qW_max, scale, offset=offset)
        currentMse = loss_function(shaped_x,
                                   qW_min,
                                   qW_max,
                                   scale=scale,
                                   offset=offset,
                                   quant=quant,
                                   dequant=dequant)
        mask1 = (bestMse - currentMse) / bestMse.clamp(min=1e-4) < 1e-10
        mask2 = torch.abs(bestMse - currentMse) < 1e-10
        if torch.sum(
                torch.logical_and(torch.logical_not(mask1),
                                  torch.logical_not(mask2))) == 0:
            break
        mask = (currentMse < bestMse).to(torch.int32)
        bestMse = currentMse * mask + bestMse * (1 - mask)
        bestScale = scale * mask + bestScale * (1 - mask)
        bestOffset = offset * mask + bestOffset * (1 - mask)
        bestQuant = quant * mask + bestQuant * (1 - mask)

    recovered = get_dequant(bestQuant, qW_min, qW_max, bestScale, bestOffset)
    recovered = recovered.view(shape)
    if w8:
        return bestQuant.to(torch.int8), bestScale

    if w4:
        bestQuantInt4 = bestQuant.view(shape).to(torch.int8)
        bestScale = bestScale.view(shape[0],
                                   -1).T.to(torch.float32).view(torch.int32)
        bestScaleInt64 = torch.zeros(bestScale.shape, dtype=torch.int64).view(
            torch.int32).reshape(-1, 2)
        bestScaleInt64[:, 0] = bestScale.reshape(-1)
        bestScaleInt64 = bestScaleInt64.view(torch.int64).reshape(
            bestScale.shape)
        bias = (8 * recovered.to(torch.float32)).sum(dim=-1)
        return bestQuantInt4, bestScaleInt64, bias
    return recovered


def weight_quant(tensor: torch.Tensor):
    assert tensor.dim() == 2
    qmax = 127.0
    abs_max = torch.abs(tensor).max(dim=1, keepdim=True)[0]
    scale = abs_max / qmax
    assert scale.shape == (tensor.shape[0], 1)
    quantized = torch.round(tensor / scale)
    quantized = torch.clamp(quantized, -qmax, qmax)
    return quantized.to(torch.int8), scale.to(torch.float32)


def pack_4bit(x):
    x = x.T.contiguous()
    shape = x.shape
    x = x.view(-1, 2)
    x1 = x[:, 0]
    x2 = x[:, 1]
    y_x2 = torch.bitwise_left_shift(x2, 4)
    y_x1 = x1 & 0b00001111
    y = torch.bitwise_or(y_x1, y_x2)
    y = y.view(shape[0], shape[1] // 2)
    return y.T.contiguous()


def main(args,
         bf16_path,
         output_path,
         next,
         pangu_mode,
         model_name="deepseek-ai/DeepSeek-R1"):
    quant_prefix = "quant_model_weight_w4a8_dynamic"
    disable_names = []
    for i in range(62):
        disable_names.append(f"model.layers.{i}.self_attn.kv_b_proj.weight")
        disable_names.append(f"model.layers.{i}.mlp.gate.weight")
        disable_names.append(
            f"model.layers.{i}.mlp.gate.e_score_correction_bias")

    disable_names.append("lm_head")
    disable_names.append("model.embed_tokens.weight")

    w4_type = QType(args.qtype)
    torch.set_default_dtype(torch.bfloat16)
    os.makedirs(output_path, exist_ok=True)
    model_index_file = os.path.join(output_path,
                                    "model.safetensors.index.json")
    config_file = os.path.join(output_path, "config.json")

    if not os.path.exists(model_index_file) or not os.path.exists(config_file):
        snapshot_download(repo_id=model_name,
                          ignore_patterns=["*.safetensors"],
                          local_dir=output_path,
                          local_dir_use_symlinks=False)
        print(f"model index file and config file download to {output_path}")

    with open(model_index_file, "r") as f:
        model_index = json.load(f)
    weight_map = model_index["weight_map"]
    scale_count = len(
        [key for key in weight_map.keys() if key.endswith("_scale_inv")])

    safetensor_files = list(glob(os.path.join(bf16_path, "*.safetensors")))
    safetensor_files.sort()
    if args.file_count:
        safetensor_files = safetensor_files[:args.file_count]

    quant_count = 0
    new_weight_map = {}

    def weight_is_w4(weight_name):
        if 'experts' not in weight_name:
            return False
        if 'shared_experts' in weight_name:
            return False
        w4_list = ['up_proj', 'gate_proj', 'down_proj']
        for name in w4_list:
            if name in weight_name:
                return True
        return False

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
            if next:
                if "shared" in weight_name or "self_attn" in weight_name or "mlp.gate_proj" in weight_name or "mlp.up_proj" in weight_name or "mlp.down_proj" in weight_name:
                    print(weight_name, "bf16")
                    new_state_dict[weight_name] = weight
                    new_weight_map[weight_name] = file_name
                    continue
            scale_inv_name = f"{weight_name}_scale_inv"
            if scale_inv_name in weight_map or pangu_mode:
                assert weight.element_size() == 2
                quant_count += 1
                if weight_is_w4(weight_name):
                    print(weight_name, "int4")
                    int4_weight, int4_scale, bias = quant_ssz(weight,
                                                              w4_type,
                                                              -1,
                                                              w4=True)

                    new_state_dict[weight_name] = pack_4bit(int4_weight)
                    new_scale_int4 = scale_inv_name.replace(
                        "_scale_inv", "_int4_scale")
                    new_bias = scale_inv_name.replace("_scale_inv", "_bias")

                    new_state_dict[new_scale_int4] = int4_scale
                    new_state_dict[new_bias] = bias

                    new_weight_map[weight_name] = file_name
                    new_weight_map[new_scale_int4] = file_name
                    new_weight_map[new_bias] = file_name
                else:
                    print(weight_name, "int8")
                    int8_weight, scale_inv = weight_quant(weight)
                    new_state_dict[weight_name] = int8_weight
                    new_scale_name = scale_inv_name.replace(
                        "_scale_inv", "_scale")
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
    print(
        f"model.safetensors.index.json modified and saved to {model_index_file}"
    )
