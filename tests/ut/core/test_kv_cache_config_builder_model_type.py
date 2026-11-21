#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
#

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm_ascend.core import kv_cache_config_builder


@pytest.mark.parametrize(
    "model_config",
    [
        SimpleNamespace(hf_config=SimpleNamespace()),
        SimpleNamespace(hf_config=None),
        SimpleNamespace(),
    ],
)
def test_kv_cache_config_builder_handles_mock_config_without_model_type(model_config):
    vllm_config = SimpleNamespace(
        model_config=model_config,
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        kv_transfer_config=None,
    )
    builder = object.__new__(kv_cache_config_builder.AscendKVCacheConfigBuilder)

    with (
        patch.object(kv_cache_config_builder, "get_step5_kv_cache_groups", return_value=None),
        patch.object(kv_cache_config_builder, "get_qwen4_exp_kv_cache_groups", return_value=None),
        patch.object(kv_cache_config_builder, "is_deepseek_v41_cache", return_value=False),
        patch.object(kv_cache_config_builder, "group_and_unify_kv_cache_specs", return_value=None),
        patch.object(kv_cache_config_builder, "_get_kimi_k3_dspark_mixed_kv_cache_groups", return_value=None),
        patch.object(kv_cache_config_builder, "_get_replicated_draft_pools", return_value=None),
        patch.object(kv_cache_config_builder, "is_step5_groups", return_value=False),
        patch.object(kv_cache_config_builder, "_get_glm5_next_cache_layout", return_value=None),
        patch.object(kv_cache_config_builder, "get_qwen4_exp_pool_bytes_per_block", return_value=None),
        patch.object(kv_cache_config_builder, "get_qwen4_exp_kv_cache_config", return_value=None),
        patch.object(kv_cache_config_builder.tq_cache, "uses_turboquant_groups", return_value=False),
        patch.object(kv_cache_config_builder, "_is_deepseek_v4_groups", return_value=False),
        patch.object(kv_cache_config_builder, "_attach_transfer_and_speculation_metadata", side_effect=lambda _, c: c),
        patch.object(
            kv_cache_config_builder,
            "_upstream_get_kv_cache_config_from_groups",
            return_value=SimpleNamespace(num_blocks=4),
        ) as upstream_cfg,
        patch.object(
            kv_cache_config_builder.AscendKVCacheConfigBuilder.__mro__[1],
            "get_kv_cache_groups",
            return_value=["base_group"],
        ) as base_groups,
        patch.object(
            kv_cache_config_builder.AscendKVCacheConfigBuilder.__mro__[1],
            "_get_pool_bytes_per_block",
            return_value=64,
        ) as base_pool,
    ):
        assert builder.get_kv_cache_groups(vllm_config, {}) == ["base_group"]
        assert builder._get_pool_bytes_per_block(vllm_config, []) == 64
        cfg = kv_cache_config_builder.get_ascend_kv_cache_config_from_groups(vllm_config, [], 1024)
        assert cfg.num_blocks == 4
        base_groups.assert_called_once()
        base_pool.assert_called_once()
        upstream_cfg.assert_called_once()
