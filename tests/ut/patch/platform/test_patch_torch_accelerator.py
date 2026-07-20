#
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
#

from importlib import reload

import torch

import vllm_ascend.patch.platform.patch_torch_accelerator as patch_torch_accelerator


def test_patch_torch_accelerator_redirects_memory_apis(monkeypatch):
    def npu_empty_cache():
        empty_cache_calls.append(True)

    def npu_memory_stats():
        return {"allocated_bytes.all.peak": 1024}

    def npu_memory_reserved():
        return 2048

    def npu_reset_peak_memory_stats():
        reset_calls.append(True)

    empty_cache_calls: list[bool] = []
    reset_calls: list[bool] = []

    with monkeypatch.context() as patch:
        patch.setattr(torch.npu, "empty_cache", npu_empty_cache, raising=False)
        patch.setattr(torch.npu, "memory_stats", npu_memory_stats, raising=False)
        patch.setattr(torch.npu, "memory_reserved", npu_memory_reserved, raising=False)
        patch.setattr(
            torch.npu,
            "reset_peak_memory_stats",
            npu_reset_peak_memory_stats,
            raising=False,
        )

        reload(patch_torch_accelerator)

        assert torch.accelerator.memory_stats is npu_memory_stats
        assert torch.accelerator.memory_reserved is npu_memory_reserved
        assert torch.accelerator.reset_peak_memory_stats is npu_reset_peak_memory_stats

        torch.accelerator.empty_cache()
        assert empty_cache_calls == [True]

    reload(patch_torch_accelerator)
