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
import torch
import torch_npu
from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size

from .base import AscendAttentionScheme
from .registry import register_scheme


@register_scheme("C8_FP", "attention")
class AscendC8KVCacheMethod(AscendAttentionScheme):
    """kvcache quant method"""

    def __init__(self):
        self.kv_cache_type = torch.float8_e4m3fn
        self.kv_cache_type_real = torch.float8_e4m3fn
        self.key_cache = None
        self.value_cache = None

    def create_weights(self, layer: torch.nn.Module) -> None:
        kv_channel_dim = layer.num_kv_heads * layer.head_size
        params_dtype = torch.get_default_dtype()

        for name in ["k_scale", "v_scale"]:
            param = torch.nn.Parameter(torch.empty(kv_channel_dim, dtype=params_dtype), requires_grad=False)
            param.weight_loader = self.weight_loader
            layer.register_parameter(name, param)

        return

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        pass

    def weight_loader(self, param: torch.Tensor, loaded_weight: torch.Tensor, *args, **kwargs) -> None:
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
        shard_dim = 0
        shard_size = loaded_weight.shape[shard_dim] // tp_size
        start_idx = tp_rank * shard_size

        # scale需要切head维
        loaded_weight = loaded_weight.narrow(shard_dim, start_idx, shard_size)
        assert param.size() == loaded_weight.size(), (
            f"Attempted to load weight ({loaded_weight.size()}) into parameter ({param.size()})"
        )
        param.data.copy_(loaded_weight)

    def apply(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache,
        attn_metadata,
        scale,
        output,
        block_tables,
    ) -> torch.Tensor:
        key_scale = layer.k_scale
        key_offset = None
        value_scale = layer.v_scale
        value_offset = None
        self.num_kv_heads = layer.num_kv_heads
        self.num_heads = layer.num_heads
        self.scale = scale
        quant_key = torch_npu.npu_quantize(
            key.view([key.shape[0], -1]), key_scale, key_offset, self.kv_cache_type_real, -1, True
        ).view(key.shape)
        quant_value = torch_npu.npu_quantize(
            value.view([value.shape[0], -1]), value_scale, value_offset, self.kv_cache_type_real, -1, True
        ).view(value.shape)
        return quant_key, quant_value, key_scale, value_scale
