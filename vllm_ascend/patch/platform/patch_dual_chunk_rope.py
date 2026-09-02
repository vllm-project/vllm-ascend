# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

# Patch vllm's DualChunkRotaryEmbedding to use the current platform device
# instead of the hard-coded "cuda" device string.
#
# Upstream dual_chunk_rope.py builds `self.device = torch.device(f"cuda:{idx}")`
# using the portable `torch.accelerator.current_device_index()` but a hard-coded
# "cuda" type. On Ascend the subsequent `.to(device=self.device)` inside
# `_compute_cos_sin_cache` triggers CUDA lazy-init and crashes with
# "Torch not compiled with CUDA enabled" during model loading (e.g.
# llava-onevision-qwen2). `self.device` is only ever read inside
# `_compute_cos_sin_cache`, so we wrap that method to correct `self.device` to
# `current_platform.device_type` (== "npu" on Ascend) before delegating to the
# original implementation. No init/cache body is duplicated.
#
# Future Plan:
#   Remove this patch once upstream uses `current_platform.device_type` (or
#   `torch.accelerator.current_device_type()`) instead of the "cuda" literal.

import contextlib

import torch


def install_patch():
    from vllm.model_executor.layers.rotary_embedding.dual_chunk_rope import (
        DualChunkRotaryEmbedding,
    )
    from vllm.platforms import current_platform

    _orig_compute_cos_sin_cache = DualChunkRotaryEmbedding._compute_cos_sin_cache

    def _patched_compute_cos_sin_cache(self):
        # Upstream hard-codes "cuda"; use the active platform's device type
        # (e.g. "npu" on Ascend) so `.to(device=self.device)` targets the NPU.
        # The device index follows torch.accelerator.current_device_index(),
        # which is per-process/rank-correct (same source as upstream).
        self.device = torch.device(
            current_platform.device_type,
            torch.accelerator.current_device_index(),
        )
        return _orig_compute_cos_sin_cache(self)

    DualChunkRotaryEmbedding._compute_cos_sin_cache = _patched_compute_cos_sin_cache


with contextlib.suppress(ImportError):
    install_patch()
