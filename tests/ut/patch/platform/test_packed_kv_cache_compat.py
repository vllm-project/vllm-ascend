# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from vllm_ascend.patch.platform import patch_kv_cache_utils as cache_patch


def test_packed_groups_leave_eagle_annotation_to_upstream():
    """Both supported planners annotate EAGLE groups in get_kv_cache_configs."""
    config = SimpleNamespace()
    specs = {"layer": object()}
    groups = [object()]
    annotator = MagicMock()
    warning = MagicMock()
    upstream = SimpleNamespace(
        _annotate_eagle_groups=annotator,
        _warn_if_unannotated_eagle_mamba=warning,
        _uses_trailing_mtp_layers=MagicMock(),
    )
    with (
        patch.object(cache_patch, "_kv_cache_planning", upstream),
        patch.object(cache_patch, "is_deepseek_v41_cache", return_value=True),
        patch.object(cache_patch, "group_cache_specs", return_value=specs),
        patch.object(cache_patch, "make_cache_groups", return_value=groups),
    ):
        assert cache_patch._ascend_get_packed_kv_cache_groups(config, specs) is groups

    annotator.assert_not_called()
    warning.assert_not_called()
