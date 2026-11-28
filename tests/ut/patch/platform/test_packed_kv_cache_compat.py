# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from unittest.mock import patch

from vllm_ascend.core import kv_cache_config_builder
from vllm_ascend.core.kv_cache_compat import KVCacheGroupSpec
from vllm_ascend.utils import vllm_version_is


def test_packed_groups_leave_eagle_annotation_to_upstream():
    """Ascend grouping never tags draft groups; get_kv_cache_configs does."""
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="qwen3")),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
    )
    specs = {"layer": object()}
    builder = kv_cache_config_builder.AscendKVCacheConfigBuilder()
    with patch.object(kv_cache_config_builder, "is_deepseek_v41_cache", return_value=True):
        if vllm_version_is("0.34.0"):
            groups = [KVCacheGroupSpec(layer_names=["layer"], kv_cache_spec=specs["layer"])]
            with (
                patch.object(kv_cache_config_builder, "get_qwen4_exp_kv_cache_groups", return_value=None),
                patch.object(kv_cache_config_builder, "group_cache_specs", return_value=specs),
                patch.object(kv_cache_config_builder, "make_cache_groups", return_value=groups),
            ):
                assert builder.get_kv_cache_groups(config, specs) is groups
        else:
            from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
            from vllm.v1.kv_cache_interface import KVCacheGroup

            groups = [KVCacheGroup(specs, FullAttentionManager.config(), block_size=16)]
            configs = {"layer": SimpleNamespace(spec=specs["layer"])}
            with patch.object(kv_cache_config_builder, "group_layer_configs", return_value=groups):
                assert builder.get_kv_cache_groups(config, configs) is groups

    # ``is_eagle_group`` only exists on the lanes that annotate EAGLE groups.
    assert not any(getattr(group, "is_eagle_group", False) for group in groups)
