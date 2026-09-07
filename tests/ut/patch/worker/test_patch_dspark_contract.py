# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import vllm_ascend.patch.worker.patch_v2.patch_dspark as dspark_patch


@pytest.mark.parametrize("same_checkpoint", [False, True])
@pytest.mark.parametrize("use_pp", [False, True])
@pytest.mark.parametrize("fail_loading", [False, True])
def test_dspark_loader_does_not_read_removed_pp_alias(same_checkpoint, use_pp, fail_loading):
    original_quant = lambda config: "draft-quant"
    original_share = lambda *args: True
    model_utils = SimpleNamespace(get_draft_quant_config=original_quant)
    eagle_utils = SimpleNamespace(_should_share=original_share)
    # vLLM #50514 removes get_pp_group from this module.
    dspark_utils = SimpleNamespace()
    config = SimpleNamespace(
        model_config=SimpleNamespace(model="target"),
        quant_config=object(),
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(model="target" if same_checkpoint else "draft"),
        ),
    )
    target, loaded = object(), object()

    def load(model, vllm_config):
        assert model is target and vllm_config is config
        expected = config.quant_config if same_checkpoint else "draft-quant"
        assert model_utils.get_draft_quant_config(config) == expected
        assert eagle_utils._should_share(None, "has_own_embed_tokens", None, None) is not use_pp
        assert eagle_utils._should_share(None, "has_own_lm_head", None, None)
        if fail_loading:
            raise RuntimeError("load failure")
        return loaded

    with patch.multiple(
        dspark_patch,
        model_utils=model_utils,
        eagle_utils=eagle_utils,
        dspark_utils=dspark_utils,
        _original_get_draft_quant_config=original_quant,
        _original_load_dspark_model=load,
        resolve_spec_pp_support=lambda config: object() if use_pp else None,
        bypass_upstream_spec_pp_guard=lambda *args: nullcontext(),
    ):
        if fail_loading:
            with pytest.raises(RuntimeError, match="load failure"):
                dspark_patch._load_dspark_model_with_target_quant(target, config)
        else:
            assert dspark_patch._load_dspark_model_with_target_quant(target, config) is loaded
    assert model_utils.get_draft_quant_config is original_quant
    assert eagle_utils._should_share is original_share
    assert vars(dspark_utils) == {}
