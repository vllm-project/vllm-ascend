# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from vllm_ascend.patch.platform import patch_kv_cache_utils as cache_patch


@pytest.mark.parametrize("use_fallback", [False, True])
def test_packed_groups_use_matching_eagle_annotation_api(use_fallback):
    config = SimpleNamespace()
    specs = {"layer": object()}
    groups = [object()]

    def annotate(config_arg, specs_arg, groups_arg, *, use_trailing_layer_fallback):
        assert use_trailing_layer_fallback is use_fallback

    predicate = MagicMock(return_value=use_fallback)
    annotator = MagicMock(side_effect=annotate)
    warning = MagicMock()
    upstream = SimpleNamespace(
        _annotate_eagle_groups=annotator,
        _warn_if_unannotated_eagle_mamba=warning,
        _uses_trailing_mtp_layers=predicate,
    )
    with (
        patch.object(cache_patch, "_kv_cache_planning", upstream),
        patch.object(cache_patch, "vllm_version_is", lambda version: version == "0.32.0"),
        patch.object(cache_patch, "is_deepseek_v41_cache", return_value=True),
        patch.object(cache_patch, "group_cache_specs", return_value=specs),
        patch.object(cache_patch, "make_cache_groups", return_value=groups),
    ):
        assert cache_patch._ascend_get_packed_kv_cache_groups(config, specs) is groups

    predicate.assert_called_once_with(config)
    assert annotator.call_args.args == (config, specs, groups)
    warning.assert_called_once_with(config, groups)


def test_packed_groups_skip_eagle_annotation_on_main():
    """The version gate leaves the API main no longer ships untouched.

    vLLM #42309 drops ``_annotate_eagle_groups``/``_uses_trailing_mtp_layers``
    from main and annotates EAGLE groups in ``get_kv_cache_configs`` instead.
    The packed-group hook must therefore stay hands-off on the main lane even
    when the live module exposes the helpers.
    """
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
        patch.object(cache_patch, "vllm_version_is", lambda version: False),
        patch.object(cache_patch, "is_deepseek_v41_cache", return_value=True),
        patch.object(cache_patch, "group_cache_specs", return_value=specs),
        patch.object(cache_patch, "make_cache_groups", return_value=groups),
    ):
        assert cache_patch._ascend_get_packed_kv_cache_groups(config, specs) is groups

    annotator.assert_not_called()
    warning.assert_not_called()


def test_main_builder_uses_ascend_pool_accounting():
    builder_cls = getattr(
        cache_patch._kv_cache_planning,
        "DefaultKVCacheConfigBuilder",
        None,
    )
    if builder_cls is None:
        pytest.skip("vLLM lane predates KVCacheConfigBuilder")

    builder = builder_cls()
    config = SimpleNamespace()
    groups = [object()]
    with patch.object(
        cache_patch,
        "_ascend_pool_bytes_per_block",
        return_value=123,
    ) as pool_bytes:
        assert builder._get_pool_bytes_per_block(config, groups) == 123

    pool_bytes.assert_called_once_with(config, groups)
