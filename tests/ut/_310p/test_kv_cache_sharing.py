# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import pytest
import torch
from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec
from vllm.v1.kv_cache_layout import KVCacheLayout

from tests.ut.kv_cache_test_utils import get_group_spec, is_v034, make_kv_cache_group
from vllm_ascend._310p.kv_cache_sharing import get_310p_shared_cache_slots


def _make_attention_spec():
    if is_v034():
        from vllm.v1.kv_cache_interface import FullAttentionSpec

        return FullAttentionSpec(block_size=4, num_kv_heads=1, head_size=4, dtype=torch.float16)
    return AttentionSpec(block_size=4, num_kv_heads=1, head_size=4, dtype=torch.float16)


def _make_groups() -> list:
    attention = _make_attention_spec()
    mamba = MambaSpec(block_size=4, shapes=((16,),), dtypes=(torch.float32,))
    assert attention.page_size_bytes == mamba.page_size_bytes
    return [
        make_kv_cache_group(["L3", "L7"], attention),
        make_kv_cache_group(["L0", "L4"], mamba),
        make_kv_cache_group(["L1", "L5"], mamba),
        make_kv_cache_group(["L2", "L6"], mamba),
    ]


def test_shares_matching_mamba_slots_across_groups_only() -> None:
    assert get_310p_shared_cache_slots(_make_groups(), KVCacheLayout.LBHNC) == {
        "L0": 0,
        "L4": 1,
        "L1": 0,
        "L5": 1,
        "L2": 0,
        "L6": 1,
    }


def test_mtp_attention_layer_does_not_disable_mamba_slot_sharing() -> None:
    groups = _make_groups()
    expected_slots = get_310p_shared_cache_slots(groups, KVCacheLayout.LBNHC)
    assert expected_slots
    groups[0] = make_kv_cache_group(
        [*groups[0].layer_names, "mtp.layers.0.self_attn.attn"],
        get_group_spec(groups[0]),
    )

    assert get_310p_shared_cache_slots(groups, KVCacheLayout.LBNHC) == expected_slots


@pytest.mark.parametrize("layout", [KVCacheLayout.LHBNC, KVCacheLayout.BLHNC])
def test_rejects_layout_without_contiguous_per_layer_pages(layout: KVCacheLayout) -> None:
    assert get_310p_shared_cache_slots(_make_groups(), layout) == {}


def test_rejects_nonuniform_group_sizes_and_mamba_specs() -> None:
    groups = _make_groups()
    assert get_310p_shared_cache_slots(groups[:1], KVCacheLayout.LBHNC) == {}

    groups[2] = make_kv_cache_group(["L1"], get_group_spec(groups[2]))
    assert get_310p_shared_cache_slots(groups, KVCacheLayout.LBHNC) == {}

    groups = _make_groups()
    different_mamba = MambaSpec(block_size=4, shapes=((32,),), dtypes=(torch.float16,))
    assert different_mamba.page_size_bytes == get_group_spec(groups[1]).page_size_bytes
    groups[2] = make_kv_cache_group(["L1", "L5"], different_mamba)
    assert get_310p_shared_cache_slots(groups, KVCacheLayout.LBHNC) == {}


def test_rejects_mismatched_page_sizes_or_multiple_attention_groups() -> None:
    groups = _make_groups()
    smaller_mamba = MambaSpec(block_size=4, shapes=((8,),), dtypes=(torch.float32,))
    for index in range(1, len(groups)):
        groups[index] = make_kv_cache_group(groups[index].layer_names, smaller_mamba)
    assert get_310p_shared_cache_slots(groups, KVCacheLayout.LBHNC) == {}

    groups = _make_groups()
    groups[2] = make_kv_cache_group(groups[2].layer_names, get_group_spec(groups[0]))
    assert get_310p_shared_cache_slots(groups, KVCacheLayout.LBHNC) == {}


def test_uses_per_layer_specs_from_uniform_type_group() -> None:
    groups = _make_groups()
    mamba = get_group_spec(groups[1])
    if is_v034():
        from vllm.v1.kv_cache_interface import KVCacheGroupSpec, UniformTypeKVCacheSpecs

        groups[1] = KVCacheGroupSpec(
            ["L0", "L4"],
            UniformTypeKVCacheSpecs(
                block_size=mamba.block_size,
                kv_cache_specs={"L0": mamba, "L4": mamba},
            ),
        )
    else:
        groups[1] = make_kv_cache_group({"L0": mamba, "L4": mamba})
    assert get_310p_shared_cache_slots(groups, KVCacheLayout.LBNHC)["L4"] == 1
