# SPDX-License-Identifier: Apache-2.0

import pytest

from vllm_ascend.draft_config_context import is_draft_config_loading
from vllm_ascend.patch.worker.patch_v2 import patch_dflash_speculator


@pytest.mark.parametrize("fail", [False, True])
def test_dflash_draft_config_context_is_scoped(monkeypatch, fail):
    assert (
        patch_dflash_speculator.dflash_utils.load_dflash_model
        is patch_dflash_speculator._load_dflash_model_with_draft_context
    )
    assert (
        patch_dflash_speculator.speculator_module.load_dflash_model
        is patch_dflash_speculator._load_dflash_model_with_draft_context
    )
    target = object()
    config = object()

    def load(received_target, received_config):
        assert is_draft_config_loading()
        assert received_target is target
        assert received_config is config
        if fail:
            raise RuntimeError("draft load failed")
        return target

    monkeypatch.setattr(patch_dflash_speculator, "_original_load_dflash_model", load)

    if fail:
        with pytest.raises(RuntimeError, match="draft load failed"):
            patch_dflash_speculator._load_dflash_model_with_draft_context(target, config)
    else:
        assert patch_dflash_speculator._load_dflash_model_with_draft_context(target, config) is target

    assert not is_draft_config_loading()
