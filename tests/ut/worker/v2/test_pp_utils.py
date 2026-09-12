# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import os
from types import SimpleNamespace

import pytest
import vllm.envs as vllm_envs
from vllm.distributed.utils import get_pp_indices

from vllm_ascend.worker.v2.pp_utils import SpecPPSupport, bypass_upstream_spec_pp_guard


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("partition", [None, "42,36"])
@pytest.mark.parametrize("fail", [False, True])
def test_unsharded_draft_preserves_target_partition(monkeypatch, cached, partition, fail):
    was_cached = vllm_envs._is_envs_cache_enabled()
    vllm_envs.disable_envs_cache()
    if partition is None:
        monkeypatch.delenv("VLLM_PP_LAYER_PARTITION", raising=False)
    else:
        monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", partition)
    config = SimpleNamespace(parallel_config=SimpleNamespace(pipeline_parallel_size=2))
    support = SpecPPSupport(bypass_upstream_pp_guard=True)

    def initialize():
        with bypass_upstream_spec_pp_guard(config, support) as bypassed:
            assert bypassed
            assert config.parallel_config.pipeline_parallel_size == 1
            assert vllm_envs.VLLM_PP_LAYER_PARTITION is None
            assert os.environ.get("VLLM_PP_LAYER_PARTITION") == partition
            assert get_pp_indices(78, 0, 1) == (0, 78)
            with bypass_upstream_spec_pp_guard(config, support):
                assert get_pp_indices(78, 0, 1) == (0, 78)
            assert vllm_envs.VLLM_PP_LAYER_PARTITION is None
            if fail:
                raise RuntimeError("draft initialization failed")

    try:
        if cached:
            vllm_envs.enable_envs_cache()
        if fail:
            with pytest.raises(RuntimeError, match="draft initialization failed"):
                initialize()
        else:
            initialize()
        assert config.parallel_config.pipeline_parallel_size == 2
        assert partition == vllm_envs.VLLM_PP_LAYER_PARTITION
        if partition is not None:
            assert get_pp_indices(78, 0, 2) == (0, 42)
            assert get_pp_indices(78, 1, 2) == (42, 78)
    finally:
        vllm_envs.disable_envs_cache()
        monkeypatch.undo()
        if was_cached:
            vllm_envs.enable_envs_cache()


@pytest.mark.parametrize("support", [None, SpecPPSupport()])
def test_pp_guard_noop_preserves_partition(monkeypatch, support):
    monkeypatch.setattr(vllm_envs, "VLLM_PP_LAYER_PARTITION", "42,36")
    config = SimpleNamespace(parallel_config=SimpleNamespace(pipeline_parallel_size=2))
    with bypass_upstream_spec_pp_guard(config, support) as bypassed:
        assert not bypassed
        assert config.parallel_config.pipeline_parallel_size == 2
        assert vllm_envs.VLLM_PP_LAYER_PARTITION == "42,36"
