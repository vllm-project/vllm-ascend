# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from vllm.v1.core import kv_cache_planning

from vllm_ascend.core import kv_cache_config_builder


def test_packed_groups_leave_eagle_annotation_to_upstream():
    """Both supported planners annotate EAGLE groups in get_kv_cache_configs."""
    config = SimpleNamespace(scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False))
    specs = {"layer": object()}
    groups = [object()]
    annotator = MagicMock()
    with (
        patch.object(kv_cache_planning, "_annotate_eagle_groups", annotator),
        patch.object(kv_cache_config_builder, "is_deepseek_v41_cache", return_value=True),
        patch.object(kv_cache_config_builder, "group_cache_specs", return_value=specs),
        patch.object(kv_cache_config_builder, "make_cache_groups", return_value=groups),
    ):
        assert kv_cache_config_builder.AscendKVCacheConfigBuilder().get_kv_cache_groups(config, specs) is groups

    annotator.assert_not_called()
