#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from collections.abc import Iterable

import torch
from vllm.distributed import (get_tensor_model_parallel_rank,
                               get_tensor_model_parallel_world_size)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name)
from vllm.model_executor.models.qwen2 import Qwen2Model
from vllm.model_executor.models.utils import is_pp_missing_parameter


def qwen2_load_weights_with_kv_cache_remap(
        self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
    """Modified load_weights for Qwen2Model with KV cache scale remapping.

    This version handles the case where quantized models have separate
    k_proj.kv_cache_scale and v_proj.kv_cache_scale parameters, but
    vLLM's Qwen2Model uses fused qkv_proj and expects the scales to be
    registered under attn.key_antiquant_scale and attn.value_antiquant_scale.
    """
    stacked_params_mapping = [
        # (param_name, shard_name, shard_id)
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
    ]
    params_dict = dict(self.named_parameters(remove_duplicate=False))
    loaded_params: set[str] = set()

    for name, loaded_weight in weights:
        if "rotary_emb.inv_freq" in name:
            continue

        # KV cache scale remapping: k_proj.kv_cache_scale -> attn.key_antiquant_scale
        if name.endswith("k_proj.kv_cache_scale"):
            remapped_name = name.replace("k_proj.kv_cache_scale", "attn.key_antiquant_scale")
            if remapped_name in params_dict:
                param = params_dict[remapped_name]

                # Handle Tensor Parallel sharding for KV cache scales
                # Shape of loaded_weight: [total_num_kv_heads * head_size]
                # Shape of param: [num_kv_heads_per_partition * head_size]
                tp_size = get_tensor_model_parallel_world_size()
                tp_rank = get_tensor_model_parallel_rank()

                if tp_size > 1:
                    # Calculate shard size
                    shard_size = loaded_weight.shape[0] // tp_size
                    start_idx = tp_rank * shard_size
                    end_idx = (tp_rank + 1) * shard_size
                    loaded_weight = loaded_weight[start_idx:end_idx]

                # Convert dtype to match the parameter's dtype
                # Quantized models use float32, but runtime may use bfloat16/float16
                if loaded_weight.dtype != param.dtype:
                    loaded_weight = loaded_weight.to(param.dtype)

                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(remapped_name)
            continue

        # KV cache scale remapping: v_proj.kv_cache_scale -> attn.value_antiquant_scale
        if name.endswith("v_proj.kv_cache_scale"):
            remapped_name = name.replace("v_proj.kv_cache_scale", "attn.value_antiquant_scale")
            if remapped_name in params_dict:
                param = params_dict[remapped_name]

                # Handle Tensor Parallel sharding for KV cache scales
                # Shape of loaded_weight: [total_num_kv_heads * head_size]
                # Shape of param: [num_kv_heads_per_partition * head_size]
                tp_size = get_tensor_model_parallel_world_size()
                tp_rank = get_tensor_model_parallel_rank()

                if tp_size > 1:
                    # Calculate shard size
                    shard_size = loaded_weight.shape[0] // tp_size
                    start_idx = tp_rank * shard_size
                    end_idx = (tp_rank + 1) * shard_size
                    loaded_weight = loaded_weight[start_idx:end_idx]

                # Convert dtype to match the parameter's dtype
                # Quantized models use float32, but runtime may use bfloat16/float16
                if loaded_weight.dtype != param.dtype:
                    loaded_weight = loaded_weight.to(param.dtype)

                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(remapped_name)
            continue

        # Skip kv_cache_offset (not needed for C8 quantization)
        if name.endswith("k_proj.kv_cache_offset") or name.endswith("v_proj.kv_cache_offset"):
            continue

        # Handle pipeline parallelism missing parameters
        if is_pp_missing_parameter(name, self):
            continue

        for (param_name, weight_name, shard_id) in stacked_params_mapping:
            if weight_name not in name:
                continue
            name = name.replace(weight_name, param_name)
            # Skip loading extra bias for GPTQ models.
            if name.endswith(".bias") and name not in params_dict:
                continue
            if is_pp_missing_parameter(name, self):
                continue
            if name.endswith("scale"):
                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            if weight_loader == default_weight_loader:
                weight_loader(param, loaded_weight)
            else:
                weight_loader(param, loaded_weight, shard_id)
            break
        else:
            # Skip loading extra bias for GPTQ models.
            if name.endswith(".bias") and name not in params_dict:
                continue
            # Remapping the name of FP8 kv-scale.
            name = maybe_remap_kv_scale_name(name, params_dict)
            if name is None:
                continue
            if is_pp_missing_parameter(name, self):
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader(param, loaded_weight)
        loaded_params.add(name)
    return loaded_params


Qwen2Model.load_weights = qwen2_load_weights_with_kv_cache_remap
