# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import os
from types import SimpleNamespace

import pytest
import vllm.envs as vllm_envs
from vllm.distributed.utils import get_pp_indices

from vllm_ascend.worker.v2 import pp_utils
from vllm_ascend.worker.v2.pp_utils import (
    SpecPPSupport,
    bypass_upstream_spec_pp_guard,
    resolve_spec_pp_support,
)


@pytest.fixture(autouse=True)
def _clear_partition_cache():
    """Drop a cached VLLM_PP_LAYER_PARTITION left by other test modules."""
    vllm_envs.__dict__.pop("VLLM_PP_LAYER_PARTITION", None)
    yield
    vllm_envs.__dict__.pop("VLLM_PP_LAYER_PARTITION", None)


def test_spec_pp_uses_native_protocol():
    """The supported release shares the upstream sampled-token protocol."""
    assert pp_utils.use_legacy_spec_pp() is False


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

from types import SimpleNamespace

import pytest

from vllm_ascend.worker.v2.pp_utils import resolve_spec_pp_support


def _make_config(
    architecture: str,
    *,
    method: str = "dspark",
    pipeline_parallel_size: int = 2,
):
    return SimpleNamespace(
        speculative_config=SimpleNamespace(method=method),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=pipeline_parallel_size,
        ),
        model_config=SimpleNamespace(architecture=architecture),
    )


@pytest.mark.parametrize(
    "architecture",
    [
        "KimiLinearForCausalLM",
        "KimiK3ForCausalLM",
        "KimiK3ForConditionalGeneration",
    ],
)
def test_kimi_k3_dspark_pp_supports_all_target_aliases(architecture):
    support = resolve_spec_pp_support(_make_config(architecture))

    assert support is not None
    assert support.needs_aux_hidden_states
    assert support.bypass_upstream_pp_guard


@pytest.mark.parametrize(
    "config",
    [
        _make_config("KimiLinearForCausalLM", pipeline_parallel_size=1),
        _make_config("UnsupportedForPP"),
        _make_config("KimiLinearForCausalLM", method="eagle"),
    ],
)
def test_kimi_k3_dspark_pp_support_stays_scoped(config):
    assert resolve_spec_pp_support(config) is None


def test_pp_boundary_shards_never_use_all_gather_transport():
    """SP-sharded PP boundary tensors must take the full per-pair send/recv.

    The slice-send + receive-side all-gather transport is only valid for
    tensors replicated across the TP group; with rank-distinct shards it
    would rebuild every receiver from pieces of different shards.
    """
    from vllm_ascend.patch.worker.patch_distributed import GroupCoordinatorPatch
    from vllm_ascend.utils import pp_boundary_sp_sharded, set_pp_boundary_sp_sharded

    coordinator = object.__new__(GroupCoordinatorPatch)
    tp_group = SimpleNamespace(world_size=4)

    set_pp_boundary_sp_sharded(True)
    try:
        assert pp_boundary_sp_sharded()
        # Overrides both the numel-divisibility default and an explicit
        # per-key opt-in from callers such as the spec-PP send path.
        assert not GroupCoordinatorPatch._should_use_all_gather(
            coordinator, "hidden_states", 4 * 7168, tp_group, None
        )
        assert not GroupCoordinatorPatch._should_use_all_gather(
            coordinator, "residual", 4 * 7168, tp_group, {"residual": True}
        )
    finally:
        set_pp_boundary_sp_sharded(False)

    # With a replicated boundary, upstream semantics are untouched.
    assert GroupCoordinatorPatch._should_use_all_gather(
        coordinator, "hidden_states", 4 * 7168, tp_group, None
    )
    assert not GroupCoordinatorPatch._should_use_all_gather(
        coordinator, "hidden_states", 7169, tp_group, None
    )
    assert not GroupCoordinatorPatch._should_use_all_gather(
        coordinator, "residual", 4 * 7168, tp_group, {"residual": False}
    )
