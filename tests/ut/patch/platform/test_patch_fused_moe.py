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

from unittest.mock import MagicMock

import pytest


@pytest.mark.parametrize(
    "moe_kwargs",
    [
        {"is_act_and_mul": False},
        {"activation": "silu_no_mul"},
    ],
)
def test_fused_moe_factory_allows_nongated_moe_constructor(monkeypatch, moe_kwargs):
    import vllm.model_executor.layers.fused_moe.config as vllm_moe_config

    import vllm_ascend.patch.platform.patch_fused_moe as patch_fused_moe

    calls = []

    def fake_fused_moe(*args, runner_cls=None, runner_args=None, **kwargs):
        calls.append(
            {
                "is_cuda_alike": vllm_moe_config.current_platform.is_cuda_alike(),
                "runner_cls": runner_cls,
                "runner_args": runner_args,
                "kwargs": kwargs,
            }
        )
        return "moe"

    monkeypatch.setattr(vllm_moe_config.current_platform, "is_cuda_alike", lambda: False)
    monkeypatch.setattr(patch_fused_moe, "_original_FusedMoE", fake_fused_moe)
    monkeypatch.setattr(patch_fused_moe, "_DefaultAscendMoERunner", MagicMock())

    assert not vllm_moe_config.current_platform.is_cuda_alike()

    result = patch_fused_moe._ascend_FusedMoE(**moe_kwargs, tid2eid="map")

    assert result == "moe"
    assert calls == [
        {
            "is_cuda_alike": True,
            "runner_cls": patch_fused_moe._DefaultAscendMoERunner,
            "runner_args": {"tid2eid": "map"},
            "kwargs": moe_kwargs,
        }
    ]
    assert not vllm_moe_config.current_platform.is_cuda_alike()
