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
