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


def _copy_tensor_attrs(target: torch.Tensor, source: torch.Tensor) -> None:
    for attr_name, attr_value in vars(source).items():
        setattr(target, attr_name, attr_value)


def _fuse_tensor_data(lhs_meta: torch.Tensor, rhs_meta: torch.Tensor) -> torch.Tensor:
    lhs = lhs_meta.data
    rhs = rhs_meta.data
    lhs_output_dim = getattr(lhs_meta, "output_dim", None)
    rhs_output_dim = getattr(rhs_meta, "output_dim", None)

    if lhs_output_dim is not None or rhs_output_dim is not None:
        if lhs_output_dim != rhs_output_dim:
            raise RuntimeError("mismatched output_dim")
        cat_dim = lhs_output_dim
        if lhs.dim() != rhs.dim() or any(lhs.size(i) != rhs.size(i) for i in range(lhs.dim()) if i != cat_dim):
            raise RuntimeError("output_dim tensor shapes do not align for fusion")
        return torch.cat((lhs, rhs), dim=cat_dim).contiguous()

    if lhs.shape == rhs.shape and torch.equal(lhs, rhs):
        return lhs.contiguous()

    if lhs.dim() == rhs.dim() and lhs.dim() > 0 and lhs.shape[1:] == rhs.shape[1:]:
        return torch.cat((lhs, rhs), dim=0).contiguous()

    raise RuntimeError("tensor shapes do not align for fusion")


def _replace_parameter_with_fused_data(module: nn.Module, name: str, fused_data: torch.Tensor) -> None:
    old_param = module._parameters[name]
    if old_param is None:
        raise RuntimeError(f"parameter {name} is unexpectedly None")
    new_param = nn.Parameter(fused_data, requires_grad=old_param.requires_grad)
    _copy_tensor_attrs(new_param, old_param)
    module._parameters[name] = new_param


def _replace_buffer_with_fused_data(module: nn.Module, name: str, fused_data: torch.Tensor) -> None:
    old_buffer = module._buffers[name]
    if old_buffer is None:
        raise RuntimeError(f"buffer {name} is unexpectedly None")
    _copy_tensor_attrs(fused_data, old_buffer)
    module._buffers[name] = fused_data


def _fuse_linear_modules_inplace(dst_module: nn.Module, src_module: nn.Module) -> bool:
    if type(dst_module) is not type(src_module):
        return False
    if getattr(dst_module, "input_size", None) != getattr(src_module, "input_size", None):
        return False
    if getattr(dst_module, "tp_size", None) != getattr(src_module, "tp_size", None):
        return False
    if type(getattr(dst_module, "quant_method", None)) is not type(getattr(src_module, "quant_method", None)):
        return False

    dst_params = dict(dst_module.named_parameters(recurse=False))
    src_params = dict(src_module.named_parameters(recurse=False))
    if dst_params.keys() != src_params.keys():
        return False

    dst_buffers = dict(dst_module.named_buffers(recurse=False))
    src_buffers = dict(src_module.named_buffers(recurse=False))
    if dst_buffers.keys() != src_buffers.keys():
        return False

    try:
        for name, dst_param in dst_params.items():
            fused_data = _fuse_tensor_data(dst_param, src_params[name])
            _replace_parameter_with_fused_data(dst_module, name, fused_data)
        for name, dst_buffer in dst_buffers.items():
            fused_data = _fuse_tensor_data(dst_buffer, src_buffers[name])
            _replace_buffer_with_fused_data(dst_module, name, fused_data)
    except RuntimeError:
        return False

    dst_module.output_sizes = list(dst_module.output_sizes) + list(src_module.output_sizes)
    dst_module.output_size = sum(dst_module.output_sizes)
    if hasattr(dst_module, "output_partition_sizes"):
        dst_module.output_partition_sizes = [size // dst_module.tp_size for size in dst_module.output_sizes]
    if hasattr(dst_module, "update_param_tp_status"):
        dst_module.update_param_tp_status()
    return True


def process_qwen_gdn_input_proj_after_loading(module: nn.Module) -> None:
    qkvz_proj = module.in_proj_qkvz
    ba_proj = module.in_proj_ba
    module._ascend_qkvzba_is_fused = False
    module._ascend_qkvz_local_output_size = qkvz_proj.output_size // qkvz_proj.tp_size
    module._ascend_ba_local_output_size = ba_proj.output_size // ba_proj.tp_size

    if not _fuse_linear_modules_inplace(qkvz_proj, ba_proj):
        return

    module._ascend_qkvzba_is_fused = True
    module.in_proj_ba = nn.Identity()


def process_qwen_gdn_module_after_loading(module: nn.Module) -> None:
    process_qwen_gdn_conv1d_weight_after_loading(module)
    process_qwen_gdn_input_proj_after_loading(module)


def _run_qwen_gdn_linear(linear: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    projected_states = linear(hidden_states)
    if isinstance(projected_states, tuple):
        return projected_states[0]
    return projected_states


def get_qwen_gdn_projected_states(
    module: nn.Module,
    hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if getattr(module, "_ascend_qkvzba_is_fused", False):
        projected_states_qkvzba = _run_qwen_gdn_linear(module.in_proj_qkvz, hidden_states)
        projected_states_qkvz, projected_states_ba = projected_states_qkvzba.split(
            [module._ascend_qkvz_local_output_size, module._ascend_ba_local_output_size],
            dim=-1,
        )
        return projected_states_qkvz, projected_states_ba

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
