# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import vllm_ascend.patch.worker.patch_v2.patch_dspark as dspark_patch


@dataclass
class _DraftParallelConfig:
    pipeline_parallel_size: int = 1
    tensor_parallel_size: int = 4
    enable_expert_parallel: bool = False


@pytest.mark.parametrize("same_checkpoint", [False, True])
@pytest.mark.parametrize("use_pp", [False, True])
@pytest.mark.parametrize("fail_loading", [False, True])
@pytest.mark.parametrize("use_ep", [False, True])
def test_dspark_loader_does_not_read_removed_pp_alias(same_checkpoint, use_pp, fail_loading, use_ep):
    original_quant = lambda config: "draft-quant"
    original_share = lambda *args: True
    model_utils = SimpleNamespace(get_draft_quant_config=original_quant)
    eagle_utils = SimpleNamespace(_should_share=original_share)
    # vLLM #50514 removes get_pp_group from this module.
    dspark_utils = SimpleNamespace()
    draft_parallel_config = _DraftParallelConfig()
    config = SimpleNamespace(
        model_config=SimpleNamespace(model="target"),
        parallel_config=SimpleNamespace(enable_expert_parallel=use_ep),
        quant_config=object(),
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(model="target" if same_checkpoint else "draft"),
            draft_parallel_config=draft_parallel_config,
        ),
    )
    target, loaded = object(), object()

    def load(model, vllm_config):
        assert model is target and vllm_config is config
        loaded_parallel = vllm_config.speculative_config.draft_parallel_config
        assert loaded_parallel.enable_expert_parallel is (use_ep and same_checkpoint)
        assert loaded_parallel.pipeline_parallel_size == 1
        assert loaded_parallel.tensor_parallel_size == 4
        assert not draft_parallel_config.enable_expert_parallel
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
    assert config.speculative_config.draft_parallel_config is draft_parallel_config
