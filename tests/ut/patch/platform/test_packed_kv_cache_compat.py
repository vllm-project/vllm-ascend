# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from vllm_ascend.patch.platform import patch_kv_cache_utils as cache_patch


@pytest.mark.parametrize("legacy,use_fallback", [(True, True), (False, False), (False, True)])
def test_packed_groups_use_matching_eagle_annotation_api(legacy, use_fallback):
    config = SimpleNamespace()
    specs = {"layer": object()}
    groups = [object()]

    def annotate(config_arg, specs_arg, groups_arg, **kwargs):
        keyword = "use_deepseek_v4_fallback" if legacy else "use_trailing_layer_fallback"
        assert kwargs == {keyword: use_fallback}

    predicate = MagicMock(return_value=use_fallback)
    annotator = MagicMock(side_effect=annotate)
    warning = MagicMock()
    upstream = SimpleNamespace(
        _annotate_eagle_groups=annotator,
        _warn_if_unannotated_eagle_mamba=warning,
    )
    if not legacy:
        upstream._uses_trailing_mtp_layers = predicate
    with (
        patch.object(cache_patch.vllm.v1.core, "kv_cache_utils", upstream),
        patch.object(cache_patch, "is_deepseek_v41_cache", return_value=True),
        patch.object(cache_patch, "group_cache_specs", return_value=specs),
        patch.object(cache_patch, "make_cache_groups", return_value=groups),
    ):
        assert cache_patch._ascend_get_packed_kv_cache_groups(config, specs) is groups

    if legacy:
        predicate.assert_not_called()
    else:
        predicate.assert_called_once_with(config)
    assert annotator.call_args.args == (config, specs, groups)
    warning.assert_called_once_with(config, groups)
