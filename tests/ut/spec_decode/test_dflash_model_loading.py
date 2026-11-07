# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.model_executor.models.utils import PPMissingLayer
from vllm.v1.worker.gpu.spec_decode.dflash import utils as loader
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.eagle import utils as eagle_utils

from vllm_ascend.worker.v2.spec_decode.dflash.speculator import AscendDFlashSpeculator, load_missing_pp_embedding
from vllm_ascend.worker.v2.spec_decode.dflash2.speculator import AscendDFlash2Speculator


@pytest.mark.parametrize("speculator_cls", [AscendDFlashSpeculator, AscendDFlash2Speculator])
@pytest.mark.parametrize("outcome", ["success", "no_hook", "load_error", "hook_error"])
def test_draft_post_process_uses_restored_target_config(monkeypatch, speculator_cls, outcome):
    additional_config = {"scheduler_config": {"profiling_chunk_config": {"enabled": True}}}
    config = SimpleNamespace(
        additional_config=additional_config,
        quant_config=object(),
        parallel_config=SimpleNamespace(pipeline_parallel_size=2),
    )
    spec = speculator_cls.__new__(speculator_cls)
    spec.vllm_config = config
    target = object()
    names = {"target.attention"}
    draft = SimpleNamespace()

    def post_process(received):
        assert received is config
        assert received.additional_config is additional_config
        assert additional_config["scheduler_config"]["profiling_chunk_config"]["enabled"] is True
        if outcome == "hook_error":
            raise ValueError("post process failed")

    hook = Mock(side_effect=post_process)
    if outcome != "no_hook":
        draft.post_process = hook

    def load(self, received_target, received_names):
        assert self is spec
        assert received_target is target
        assert received_names is names
        assert config.additional_config is not additional_config
        assert config.additional_config["scheduler_config"]["profiling_chunk_config"]["enabled"] is False
        if outcome == "load_error":
            raise ValueError("load failed")
        return draft

    monkeypatch.setattr(DFlashSpeculator, "load_draft_model", load)
    if outcome in ("load_error", "hook_error"):
        message = "load failed" if outcome == "load_error" else "post process failed"
        with pytest.raises(ValueError, match=message):
            spec.load_draft_model(target, names)
    else:
        assert spec.load_draft_model(target, names) is draft

    assert config.additional_config is additional_config
    assert additional_config["scheduler_config"]["profiling_chunk_config"]["enabled"] is True
    if outcome in ("success", "hook_error"):
        hook.assert_called_once_with(config)
    else:
        hook.assert_not_called()


@pytest.mark.parametrize("speculator_cls", [AscendDFlashSpeculator, AscendDFlash2Speculator])
@pytest.mark.parametrize("outcome", ["success", "reload_error", "no_hook", "own_embedding", "target_present"])
def test_pp_dflash_embedding_is_loaded_before_upstream_sharing(monkeypatch, speculator_cls, outcome):
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(pipeline_parallel_size=2),
        additional_config={},
    )
    spec = speculator_cls.__new__(speculator_cls)
    spec.vllm_config = config
    original_share = loader.maybe_share_target_embed
    original_eagle_share = eagle_utils.maybe_share_target_embed
    monkeypatch.setattr(eagle_utils, "get_pp_group", lambda: SimpleNamespace(world_size=2))
    target = SimpleNamespace(model=SimpleNamespace(embed_tokens=PPMissingLayer()))
    draft = SimpleNamespace(
        model=SimpleNamespace(embed_tokens=torch.nn.Embedding(4, 2)),
        has_own_embed_tokens=outcome == "own_embedding",
    )
    initial_embed = draft.model.embed_tokens
    if outcome == "target_present":
        target.model.embed_tokens = torch.nn.Embedding(4, 2)

    events = []

    def reload_embedding(received):
        assert received is config
        events.append("reload")
        if outcome == "reload_error":
            raise ValueError("embedding load failed")
        draft.model.embed_tokens = torch.nn.Embedding(4, 2)
        draft.has_own_embed_tokens = True

    if outcome != "no_hook":
        draft.load_missing_target_embedding = Mock(side_effect=reload_embedding)

    def post_process(received):
        assert received is config
        assert loader.maybe_share_target_embed is original_share
        events.append("post_process")

    draft.post_process = Mock(side_effect=post_process)

    def load(self, received_target, names):
        assert received_target is target
        # Keep the real upstream check: a no-op bypass would leave the draft
        # without a loaded embedding and fail the ownership assertions below.
        loader.maybe_share_target_embed(draft, draft.model, target.model)
        events.append("shared")
        return draft

    monkeypatch.setattr(DFlashSpeculator, "load_draft_model", load)
    if outcome in ("reload_error", "no_hook"):
        error, message = (
            (ValueError, "embedding load failed")
            if outcome == "reload_error"
            else (RuntimeError, "needs the target input embedding")
        )
        with pytest.raises(error, match=message):
            spec.load_draft_model(target, set())
        draft.post_process.assert_not_called()
        assert draft.model.embed_tokens is initial_embed
        assert not draft.has_own_embed_tokens
    else:
        assert spec.load_draft_model(target, set()) is draft
        assert events == (["reload"] if outcome == "success" else []) + ["shared", "post_process"]
        if outcome == "success":
            assert draft.has_own_embed_tokens
            assert draft.model.embed_tokens is not initial_embed
        elif outcome == "target_present":
            assert draft.model.embed_tokens is target.model.embed_tokens
        else:
            assert draft.model.embed_tokens is initial_embed
    assert loader.maybe_share_target_embed is original_share
    assert eagle_utils.maybe_share_target_embed is original_eagle_share


def test_single_pp_stage_keeps_dflash_sharing_helper():
    config = SimpleNamespace(parallel_config=SimpleNamespace(pipeline_parallel_size=1))
    original = loader.maybe_share_target_embed
    with load_missing_pp_embedding(config):
        assert loader.maybe_share_target_embed is original
    assert loader.maybe_share_target_embed is original
