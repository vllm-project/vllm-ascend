# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from unittest.mock import patch

from tests.ut.kv_cache_test_utils import is_v034, make_full_attention_layer_config, make_kv_cache_group

from vllm_ascend.core import kv_cache_config_builder


def test_packed_groups_leave_eagle_annotation_to_upstream():
    """Ascend grouping never tags draft groups; get_kv_cache_configs does."""
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="qwen3")),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
    )
    specs = {"layer": make_full_attention_layer_config()}
    groups = [make_kv_cache_group(["layer"], specs["layer"])]
    if is_v034():
        with (
            patch.object(kv_cache_config_builder, "is_deepseek_v41_cache", return_value=True),
            patch.object(kv_cache_config_builder, "get_qwen4_exp_kv_cache_groups", return_value=None),
            patch.object(kv_cache_config_builder, "group_cache_specs", return_value=specs),
            patch.object(kv_cache_config_builder, "make_cache_groups", return_value=groups),
        ):
            assert kv_cache_config_builder.AscendKVCacheConfigBuilder().get_kv_cache_groups(config, specs) is groups
    else:
        with (
            patch.object(kv_cache_config_builder, "is_deepseek_v41_cache", return_value=True),
            patch.object(kv_cache_config_builder, "get_qwen4_exp_kv_cache_groups", return_value=None),
            patch.object(kv_cache_config_builder, "group_deepseek_v41_layer_configs", return_value=groups),
        ):
            assert kv_cache_config_builder.AscendKVCacheConfigBuilder().get_kv_cache_groups(config, specs) is groups

    # ``is_eagle_group`` only exists on the lanes that annotate EAGLE groups.
    assert not any(getattr(group, "is_eagle_group", False) for group in groups)
