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
from __future__ import annotations

from collections.abc import Callable

import torch
import vllm.model_executor.model_loader.base_loader as model_loader_base
import vllm.model_executor.model_loader.utils as model_loader_utils
from torch import nn
from vllm.triton_utils import triton

from vllm_ascend.ops.triton.fla.fused_qkvzba_split_reshape import fused_qkvzba_split_reshape_cat

PostLoadProcessor = Callable[[nn.Module, torch.device], None]


def register_post_load_processor(processor: PostLoadProcessor) -> None:
    current = model_loader_utils.process_weights_after_loading
    processors = getattr(current, "_ascend_extra_post_load_processors", None)
    if processors is None:
        original = current

        def patched_process_weights_after_loading(model, model_config, target_device):
            original(model, model_config, target_device)
            for post_load_processor in patched_process_weights_after_loading._ascend_extra_post_load_processors:
                post_load_processor(model, target_device)

        patched_process_weights_after_loading._ascend_extra_post_load_processors = []
        model_loader_utils.process_weights_after_loading = patched_process_weights_after_loading
        model_loader_base.process_weights_after_loading = patched_process_weights_after_loading
        processors = patched_process_weights_after_loading._ascend_extra_post_load_processors

    if processor not in processors:
        processors.append(processor)


def process_modules_after_loading(
    model: nn.Module,
    target_device: torch.device,
    module_cls: type[nn.Module] | tuple[type[nn.Module], ...],
    processor: Callable[[nn.Module], None],
) -> None:
    for _, module in model.named_modules():
        if isinstance(module, module_cls):
            with model_loader_utils.device_loading_context(module, target_device):
                processor(module)


def process_qwen_gdn_conv1d_weight_after_loading(module: nn.Module) -> None:
    conv_weight = module.conv1d.weight
    conv_width = module.conv_kernel_size
    if conv_weight.dim() != 3 or conv_weight.size(1) != 1:
        raise RuntimeError(
            f"Unexpected conv1d weight shape {tuple(conv_weight.shape)}; expected 3D (dim_or_width, 1, width_or_dim)"
        )
    if conv_weight.size(0) == conv_width:
        module.conv1d.weight.data = conv_weight.contiguous()
        module._ascend_conv1d_weight_is_packed = True
        return
    if conv_weight.size(2) != conv_width:
        raise RuntimeError(
            f"Unexpected conv1d weight layout: shape={tuple(conv_weight.shape)}, expected conv width {conv_width}"
        )
    module.conv1d.weight.data = conv_weight.squeeze(1).transpose(0, 1).contiguous().unsqueeze(1)
    module._ascend_conv1d_weight_is_packed = True


def get_packed_qwen_gdn_conv1d_weights(module: nn.Module) -> torch.Tensor:
    conv_weight = module.conv1d.weight
    if not getattr(module, "_ascend_conv1d_weight_is_packed", False):
        raise RuntimeError("conv1d weight must be packed during process_weights_after_loading before forward")
    if conv_weight.dim() != 3 or conv_weight.size(1) != 1 or conv_weight.size(0) != module.conv_kernel_size:
        raise RuntimeError(
            "Packed conv1d weight has invalid shape "
            f"{tuple(conv_weight.shape)} for conv width {module.conv_kernel_size}"
        )
    return conv_weight.view(conv_weight.size(0), conv_weight.size(2))


def process_qwen_gdn_module_after_loading(module: nn.Module) -> None:
    process_qwen_gdn_conv1d_weight_after_loading(module)


def _run_qwen_gdn_linear(linear: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    projected_states = linear(hidden_states)
    if isinstance(projected_states, tuple):
        return projected_states[0]
    return projected_states


def get_qwen_gdn_projected_states(
    module: nn.Module,
    hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    projected_states_qkvz = _run_qwen_gdn_linear(module.in_proj_qkvz, hidden_states)
    projected_states_ba = _run_qwen_gdn_linear(module.in_proj_ba, hidden_states)
    return projected_states_qkvz, projected_states_ba


def run_qwen3_5_gdn_input_projection(
    module: nn.Module,
    hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    projected_states_qkvz, projected_states_ba = get_qwen_gdn_projected_states(module, hidden_states)
    qkv_size = (module.key_dim * 2 + module.value_dim) // module.tp_size
    z_size = module.value_dim // module.tp_size
    mixed_qkv, z = projected_states_qkvz.split([qkv_size, z_size], dim=-1)
    z = z.reshape(z.size(0), -1, module.head_v_dim)
    b, a = projected_states_ba.chunk(2, dim=-1)
    return mixed_qkv, z, b.contiguous(), a.contiguous()


def run_qwen3_next_gdn_input_projection(
    module: nn.Module,
    hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    projected_states_qkvz, projected_states_ba = get_qwen_gdn_projected_states(module, hidden_states)
    return fused_qkvzba_split_reshape_cat(
        projected_states_qkvz,
        projected_states_ba,
        triton.cdiv(module.num_k_heads, module.tp_size),
        triton.cdiv(module.num_v_heads, module.tp_size),
        module.head_k_dim,
        module.head_v_dim,
    )
