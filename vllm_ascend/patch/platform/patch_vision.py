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

# Patch vllm's FusedInputNorm.forward to replace eps=0.0 with eps=1e-5.
#
# Upstream vllm-project/vllm#50411 added FusedInputNorm with eps=0.0 in
# F.batch_norm. Ascend torch_npu still validates eps > 0, while CUDA
# PyTorch >= 2.7 accepts eps=0.0. This patch replaces eps to avoid
# "batch_norm eps must be positive" on Ascend.

import contextlib

import torch


def _patched_fused_input_norm_forward(self, grid_thw, visual_dtype):
    if self.is_identity:
        return grid_thw.to(visual_dtype)

    assert grid_thw.ndim == 2
    patches, size = grid_thw.shape
    patch_size = size // self.channel

    grid_thw = grid_thw.view(patches, self.channel, patch_size)
    grid_thw = torch.nn.functional.batch_norm(
        grid_thw.to(self.dtype),
        running_mean=self.running_mean,
        running_var=self.running_var,
        weight=self.weight,
        bias=self.bias,
        training=False,
        eps=1e-5,
    )
    return grid_thw.view(patches, size).to(visual_dtype)


def install_patch():
    from vllm.model_executor.models.vision import FusedInputNorm

    FusedInputNorm.forward = _patched_fused_input_norm_forward


with contextlib.suppress(ImportError):
    install_patch()
