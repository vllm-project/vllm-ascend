from dataclasses import replace
from unittest.mock import patch

import pytest
import torch
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.worker.v2.model_states.mamba_hybrid import (
    AscendMambaHybridModelState,
)


def _mamba_spec() -> MambaSpec:
    return MambaSpec(
        block_size=16,
        shapes=((2, 3), (2, 2)),
        dtypes=(torch.float16, torch.float32),
    )


@pytest.mark.parametrize("wrapped_group_ids", [(), (1, 3), (1,)], ids=["plain", "uniform", "mixed"])
def test_get_mamba_group_info_preserves_group_indices(wrapped_group_ids):
    mamba_spec = _mamba_spec()
    attention_spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float16,
    )
    groups = []
    for group_id, spec in enumerate([attention_spec, mamba_spec, attention_spec, mamba_spec]):
        layer_specs = {f"group.{group_id}.layer.{i}": spec for i in range(2)}
        # Include a wrapped attention group to check that it is skipped.
        group_spec = UniformTypeKVCacheSpecs.from_specs(layer_specs) if group_id in (0, *wrapped_group_ids) else spec
        assert group_spec is not None
        groups.append(KVCacheGroupSpec(layer_names=list(layer_specs), kv_cache_spec=group_spec))
    kv_cache_config = KVCacheConfig(num_blocks=8, kv_cache_tensors=[], kv_cache_groups=groups)
    state = AscendMambaHybridModelState.__new__(AscendMambaHybridModelState)
    state._mamba_spec = None
    state._mamba_group_ids = []

    with patch(
        "vllm.v1.worker.mamba_utils.get_mamba_groups",
        side_effect=AssertionError("MRV2 group lookup must not call the shared resolver"),
    ) as shared_resolver:
        group_ids, spec = state._get_mamba_group_info(kv_cache_config)
        shared_resolver.assert_not_called()

    assert group_ids == [1, 3]
    assert spec is mamba_spec
    assert state._mamba_group_ids is group_ids
    assert state._mamba_spec is spec


@pytest.mark.parametrize("case", ["missing", "inconsistent", "inconsistent_wrapped", "inconsistent_within_wrapper"])
def test_get_mamba_group_info_rejects_invalid_cache_groups(case):
    spec = _mamba_spec()
    groups = [KVCacheGroupSpec(layer_names=["linear_attn"], kv_cache_spec=spec)]
    if case == "missing":
        groups.clear()
    elif case == "inconsistent":
        groups.append(KVCacheGroupSpec(layer_names=["other_mamba"], kv_cache_spec=replace(spec, block_size=32)))
    else:
        other_spec = replace(spec, shapes=((2, 4), (2, 2)))
        layer_specs = {
            "wrapped.0": spec if case == "inconsistent_within_wrapper" else other_spec,
            "wrapped.1": other_spec,
        }
        wrapped_spec = UniformTypeKVCacheSpecs.from_specs(layer_specs)
        assert wrapped_spec is not None
        wrapped_group = KVCacheGroupSpec(layer_names=list(layer_specs), kv_cache_spec=wrapped_spec)
        if case == "inconsistent_within_wrapper":
            groups = [wrapped_group]
        else:
            groups.append(wrapped_group)
    kv_cache_config = KVCacheConfig(num_blocks=8, kv_cache_tensors=[], kv_cache_groups=groups)
    state = AscendMambaHybridModelState.__new__(AscendMambaHybridModelState)
    state._mamba_spec = None
    state._mamba_group_ids = []

    with pytest.raises(AssertionError, match="no mamba layers in the model" if case == "missing" else None):
        state._get_mamba_group_info(kv_cache_config)

    assert state._mamba_spec is None
    assert state._mamba_group_ids == []