# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from vllm.config import AttentionConfig, ModelConfig, ParallelConfig, VllmConfig, replace
from vllm.v1.worker.gpu.spec_decode.dspark import utils as dspark_utils

from vllm_ascend.patch.worker import patch_dflash_speculator
from vllm_ascend.patch.worker.patch_dflash_speculator import _replace_draft_parallel_config


@pytest.mark.parametrize("tp,pcp,dcp", [(1, 2, 2), (1, 16, 16), (2, 8, 8), (2, 8, 16), (6, 4, 4)])
@pytest.mark.parametrize("pp", [1, 2])
def test_mla_draft_keeps_pcp_backed_dcp(tp, pcp, dcp, pp):
    target = ParallelConfig(
        tensor_parallel_size=tp,
        prefill_context_parallel_size=pcp,
        decode_context_parallel_size=dcp,
        pipeline_parallel_size=pp,
        dcp_q_replicate=False,
    )
    with pytest.raises(ValueError, match="must be divisible"):
        replace(target, prefill_context_parallel_size=1)
    draft = _replace_draft_parallel_config(target, prefill_context_parallel_size=1, decode_context_parallel_size=dcp)
    assert draft is not target
    assert draft.prefill_context_parallel_size == 1
    assert draft.decode_context_parallel_size == dcp
    assert draft.dcp_q_replicate is False
    assert draft.world_size == target.world_size // pcp
    assert target.prefill_context_parallel_size == pcp
    assert target.decode_context_parallel_size == dcp


@pytest.mark.parametrize("tp,pcp,dcp", [(1, 16, 1), (8, 1, 8), (8, 2, 2)])
def test_supported_upstream_clones_are_unchanged(tp, pcp, dcp):
    target = ParallelConfig(
        tensor_parallel_size=tp, prefill_context_parallel_size=pcp, decode_context_parallel_size=dcp
    )
    changes = dict(prefill_context_parallel_size=1, decode_context_parallel_size=dcp)
    expected = replace(target, **changes)
    actual = _replace_draft_parallel_config(target, **changes)
    assert vars(actual) == vars(expected)


def test_other_parallel_validation_still_runs():
    target = ParallelConfig(prefill_context_parallel_size=16, decode_context_parallel_size=16)
    with pytest.raises(ValueError, match="must be <="):
        _replace_draft_parallel_config(
            target, prefill_context_parallel_size=1, decode_context_parallel_size=16, data_parallel_size_local=2
        )


def test_attention_config_copy_stays_upstream():
    target = AttentionConfig()
    actual = _replace_draft_parallel_config(target, use_non_causal=True)
    assert vars(actual) == vars(replace(target, use_non_causal=True))


@pytest.mark.parametrize("tp,pcp,dcp", [(1, 2, 2), (1, 16, 16), (2, 8, 8), (2, 8, 16), (6, 4, 4)])
@pytest.mark.parametrize("pp", [1, 2])
def test_dspark_load_clones_existing_dcp_draft(monkeypatch, tp, pcp, dcp, pp):
    monkeypatch.setattr(patch_dflash_speculator, "get_pcp_group", lambda: SimpleNamespace(world_size=pcp))
    target = ParallelConfig(
        tensor_parallel_size=tp,
        prefill_context_parallel_size=pcp,
        decode_context_parallel_size=dcp,
        pipeline_parallel_size=pp,
        dcp_q_replicate=False,
    )
    draft = _replace_draft_parallel_config(target, prefill_context_parallel_size=1)
    # Exercise the actual upstream model-loading clone that failed on NPU.
    loaded = dspark_utils._get_dspark_parallel_config(draft, tp, object())
    assert loaded.prefill_context_parallel_size == loaded.pipeline_parallel_size == 1
    assert loaded.decode_context_parallel_size == dcp
    assert loaded.tensor_parallel_size == loaded.world_size == tp
    assert loaded.dcp_q_replicate is False
    assert draft.pipeline_parallel_size == pp
    assert target.prefill_context_parallel_size == pcp


def test_missing_worker_pcp_does_not_bypass_topology_checks(monkeypatch):
    monkeypatch.setattr(patch_dflash_speculator, "get_pcp_group", lambda: SimpleNamespace(world_size=1))
    target = ParallelConfig(prefill_context_parallel_size=16, decode_context_parallel_size=16)
    draft = _replace_draft_parallel_config(target, prefill_context_parallel_size=1)
    with pytest.raises(ValueError, match="must be divisible"):
        dspark_utils._get_dspark_parallel_config(draft, 1, object())


@pytest.mark.parametrize("tp,pcp,dcp", [(1, 16, 16), (2, 8, 8), (2, 8, 16), (6, 4, 4)])
def test_dspark_load_revalidates_nested_parallel_config(monkeypatch, tp, pcp, dcp):
    monkeypatch.setattr(patch_dflash_speculator, "get_pcp_group", lambda: SimpleNamespace(world_size=pcp))
    target = ParallelConfig(
        tensor_parallel_size=tp,
        prefill_context_parallel_size=pcp,
        decode_context_parallel_size=dcp,
        pipeline_parallel_size=2,
    )
    with (
        patch.object(VllmConfig, "__post_init__", return_value=None),
        patch.object(ModelConfig, "__post_init__", return_value=None),
    ):
        config = VllmConfig(model_config=ModelConfig(tokenizer="test", max_model_len=2048), parallel_config=target)
        draft = _replace_draft_parallel_config(target, prefill_context_parallel_size=1)
        loaded = dspark_utils._get_dspark_parallel_config(draft, tp, object())
        with pytest.raises(ValueError, match="must be divisible"):
            replace(config, parallel_config=loaded)
        actual = dspark_utils.replace(config, parallel_config=loaded)
    assert actual.parallel_config is loaded
    assert actual.parallel_config.prefill_context_parallel_size == 1
    assert actual.parallel_config.decode_context_parallel_size == dcp
    assert actual.parallel_config.world_size == tp
    assert config.parallel_config is target
    assert target.prefill_context_parallel_size == pcp
