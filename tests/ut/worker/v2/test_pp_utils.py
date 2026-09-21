# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import os
from types import SimpleNamespace

import pytest
import vllm
import vllm.envs as vllm_envs
from vllm.distributed.utils import get_pp_indices

from vllm_ascend import utils
from vllm_ascend.worker.v2 import pp_utils
from vllm_ascend.worker.v2.pp_utils import SpecPPSupport, bypass_upstream_spec_pp_guard


@pytest.fixture(autouse=True)
def _clear_partition_cache():
    """Drop a cached VLLM_PP_LAYER_PARTITION left by other test modules."""
    vllm_envs.__dict__.pop("VLLM_PP_LAYER_PARTITION", None)
    yield
    vllm_envs.__dict__.pop("VLLM_PP_LAYER_PARTITION", None)


@pytest.mark.parametrize(
    "version, legacy",
    [
        ("0.28.0", False),
        ("0.28.1+empty", False),
        ("0.29.0", True),
        ("0.29.1rc1", False),
        ("0.29.0+empty", True),
        ("0.1.dev1+g84030bbe3.empty", False),
        ("0.30.0", False),
        ("0.31.0.dev12", False),
    ],
)
@pytest.mark.parametrize("override", [False, True])
def test_spec_pp_version_routing(monkeypatch, version, legacy, override):
    monkeypatch.setattr(vllm, "__version__", "0.30.0" if override else version)
    monkeypatch.setattr(utils.envs_ascend, "VLLM_VERSION", version if override else None)
    monkeypatch.setattr(pp_utils, "vllm_version_is", utils.vllm_version_is.__wrapped__)
    assert pp_utils.use_legacy_spec_pp() is legacy


def test_untagged_release_uses_existing_version_detection(monkeypatch):
    monkeypatch.setattr(vllm, "__version__", "0.1.dev1+g123.empty")
    monkeypatch.setattr(utils.envs_ascend, "VLLM_VERSION", None)
    monkeypatch.setattr(pp_utils, "vllm_version_is", utils.vllm_version_is.__wrapped__)
    assert not pp_utils.use_legacy_spec_pp()
    # Untagged builds require an explicit override; do not infer a release SHA.
    monkeypatch.setattr(utils.envs_ascend, "VLLM_VERSION", "0.29.0")
    assert pp_utils.use_legacy_spec_pp()


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("partition", [None, "42,36"])
@pytest.mark.parametrize("fail", [False, True])
def test_unsharded_draft_preserves_target_partition(monkeypatch, cached, partition, fail):
    monkeypatch.setattr(pp_utils, "use_legacy_spec_pp", lambda: True)
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


@pytest.mark.parametrize(
    "legacy,support",
    [(True, None), (True, SpecPPSupport()), (False, SpecPPSupport(bypass_upstream_pp_guard=True))],
)
def test_pp_guard_noop_preserves_partition(monkeypatch, legacy, support):
    monkeypatch.setattr(pp_utils, "use_legacy_spec_pp", lambda: legacy)
    monkeypatch.setattr(vllm_envs, "VLLM_PP_LAYER_PARTITION", "42,36")
    config = SimpleNamespace(parallel_config=SimpleNamespace(pipeline_parallel_size=2))
    with bypass_upstream_spec_pp_guard(config, support) as bypassed:
        assert not bypassed
        assert config.parallel_config.pipeline_parallel_size == 2
        assert vllm_envs.VLLM_PP_LAYER_PARTITION == "42,36"
