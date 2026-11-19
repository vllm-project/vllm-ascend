# SPDX-License-Identifier: Apache-2.0
"""Exercise the real vLLM concurrency formula with mixed DS cache geometry.

PageSpec supplies physical byte sizes independently of version-specific MLA
storage fields. The upstream grouping, admission and concurrency methods are
used without mocks.
"""

from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.core.kv_cache_utils import get_max_concurrency_for_kv_cache_config
from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager
from vllm.v1.kv_cache_interface import MLAAttentionSpec

from tests.kv_cache_groups import cache_group, layer_config, window_mla_spec_type
from vllm_ascend.core.kv_cache_layers import HAS_LAYER_CONFIG, group_layer_specs, group_max_blocks
from vllm_ascend.quantization.methods.kv_cache.turboquant import cache


@dataclass(frozen=True, kw_only=True)
class PageSpec(MLAAttentionSpec):
    payload_bytes: int
    scale_dim: int = 0
    scale_dtype: torch.dtype = torch.float16

    @property
    def real_page_size_bytes(self):
        return self.payload_bytes

    @property
    def unpadded_page_size_bytes(self):
        return self.payload_bytes


def mixed_specs(tq, block_size=128):
    cache_dtype = "turboquant_4bit_nc" if tq else "bfloat16"
    c4 = {}
    for i in range(21):
        c4[f"c4.{i}"] = PageSpec(
            block_size=block_size * 4,
            num_kv_heads=1,
            head_size=258 if tq else 512,
            dtype=torch.uint8 if tq else torch.bfloat16,
            cache_dtype_str=cache_dtype,
            payload_bytes=block_size * (258 if tq else 1024),
        )
        c4[f"index.{i}"] = PageSpec(
            block_size=block_size * 4,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.int8,
            cache_dtype_str=cache_dtype,
            payload_bytes=block_size * 130,
            scale_dim=1,
        )
    c128 = {
        f"c128.{i}": PageSpec(
            block_size=block_size * 128,
            num_kv_heads=1,
            head_size=512,
            dtype=torch.bfloat16,
            cache_dtype_str=cache_dtype,
            payload_bytes=block_size * 1024,
        )
        for i in range(20)
    }
    swa = {
        f"swa.{i}": window_mla_spec_type(
            block_size=block_size,
            num_kv_heads=1,
            head_size=512,
            dtype=torch.bfloat16,
            **({} if HAS_LAYER_CONFIG else {"sliding_window": 128}),
        )
        for i in range(43)
    }
    state4 = {
        f"state4.{i}": window_mla_spec_type(
            block_size=block_size // 16,
            num_kv_heads=1,
            head_size=2048,
            dtype=torch.float32,
            **({} if HAS_LAYER_CONFIG else {"sliding_window": 8}),
            page_size_padded=block_size * 1024,
        )
        for i in range(21)
    }
    index_state = {
        f"index_state.{i}": window_mla_spec_type(
            block_size=block_size // 16,
            num_kv_heads=1,
            head_size=512,
            dtype=torch.float32,
            **({} if HAS_LAYER_CONFIG else {"sliding_window": 8}),
            page_size_padded=block_size * 130,
        )
        for i in range(21)
    }
    state128 = {
        f"state128.{i}": window_mla_spec_type(
            block_size=block_size // 4,
            num_kv_heads=1,
            head_size=1024,
            dtype=torch.float32,
            **({} if HAS_LAYER_CONFIG else {"sliding_window": 128}),
        )
        for i in range(20)
    }
    grouped = [specs for specs in (c4, c128, swa, {**state4, **index_state}, state128)]
    assert all(group is not None for group in grouped)
    return grouped


def requirements(members):
    if not HAS_LAYER_CONFIG:
        return members
    return {
        n: SlidingWindowManager.layer_config(s, sliding_window=8 if n.startswith(("state4.", "index_state.")) else 128)
        if type(s) is window_mla_spec_type
        else layer_config(s)
        for n, s in members.items()
    }


def group_specs(grouped):
    if HAS_LAYER_CONFIG:
        return cache.group_layers([requirements(members) for members in grouped])
    from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs

    return cache.group_specs([UniformTypeKVCacheSpecs.from_specs(members) for members in grouped])


def test_tq_planner_detects_pipeline_rank_without_c4():
    c128_group = mixed_specs(tq=True)[1]
    groups = group_specs([c128_group])
    assert cache.uses_turboquant_groups(groups)
    assert all(spec.head_size == 512 for group in groups for spec in group_layer_specs(group).values())
    baseline = group_specs([mixed_specs(tq=False)[1]])
    assert not cache.uses_turboquant_groups(baseline)


def test_tq_planner_ignores_unmaterialized_groups():
    c128_group = group_specs([mixed_specs(tq=True)[1]])[0]
    baseline_group = group_specs([mixed_specs(tq=False)[1]])[0]

    assert cache.uses_turboquant_groups([None, c128_group, None])
    assert not cache.uses_turboquant_groups([None, baseline_group, None])


def test_c4_grouping_preserves_nonquantized_specs_and_state_padding():
    grouped = mixed_specs(tq=True)
    states = {
        name: replace(spec, page_size_padded=128 * (130 if name.startswith("index_state") else 1024))
        for name, spec in grouped[3].items()
    }
    grouped[3] = states
    before = {name: spec for group in grouped for name, spec in group.items()}
    groups = group_specs(grouped)
    after = {name: spec for group in groups for name, spec in group_layer_specs(group).items()}
    assert before.keys() == after.keys()
    assert all(after[name] is spec for name, spec in before.items())
    c4_names = set(grouped[0])
    assert sum(bool(c4_names.intersection(group.layer_names)) for group in groups) == 1
    assert len(groups) == 16


def test_common_stride_uses_largest_group_without_page_bin_fragmentation():
    groups = group_specs(mixed_specs(tq=True))
    largest_group = max(sum(s.page_size_bytes for s in group_layer_specs(g).values()) for g in groups)
    # lcm(256-byte PA alignment, 258-byte compact row) = 33024 bytes.
    stride = cache.pool_bytes_per_block(groups)
    assert stride == ((largest_group + 33023) // 33024) * 33024
    assert stride % 256 == stride % 258 == 0


@pytest.mark.parametrize("block_size", [32, 64, 128])
@pytest.mark.parametrize("max_model_len", [131072, 1048576])
def test_mixed_cache_capacity_and_normalization(block_size, max_model_len):
    config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=max_model_len),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        cache_config=SimpleNamespace(num_gpu_blocks_override=None),
        max_in_flight_tokens=8192,
    )
    available_memory = 32 * 1024**3
    capacities = []
    for tq in (False, True):
        grouped = mixed_specs(tq, block_size)
        if tq:
            groups = group_specs(grouped)
            divisor = cache.pool_bytes_per_block(groups)
        else:
            # Existing BF16 planner: 21 tuples, each with a main and index
            # page. State pages are padded to those two canonical widths.
            groups = [cache_group(layer_names=list(g), kv_cache_spec=requirements(g)) for g in grouped[:2]]
            for index, group in enumerate(grouped[2:], start=2):
                names = list(group)
                chunk_size = len(names) if index == 3 else 21
                for start in range(0, len(names), chunk_size):
                    specs = {
                        name: replace(
                            group[name],
                            page_size_padded=block_size
                            * (130 if index == 3 and name.startswith("index_state") else 1024),
                        )
                        for name in names[start : start + chunk_size]
                    }
                    groups.append(cache_group(layer_names=list(specs), kv_cache_spec=requirements(specs)))
            divisor = 21 * block_size * (1024 + 130)
        num_blocks = available_memory // divisor
        runtime = SimpleNamespace(num_blocks=num_blocks, kv_cache_groups=groups)
        capacities.append(get_max_concurrency_for_kv_cache_config(config, runtime))
        blocks_per_request = sum(group_max_blocks(group, config) for group in groups)
        assert capacities[-1] == num_blocks / blocks_per_request
        if tq:
            assert cache.max_memory_usage(config, groups) == divisor * blocks_per_request
            # Compare the physical pool with identical groups and non-C4
            # specs, restoring only the C4 main cache to BF16. This isolates
            # quantization from changes to state padding or group counts.
            unquantized_groups = []
            for group in groups:
                specs = {
                    name: replace(
                        spec,
                        head_size=512,
                        dtype=torch.bfloat16,
                        cache_dtype_str="bfloat16",
                        payload_bytes=block_size * 1024,
                    )
                    if name.startswith("c4.")
                    else spec
                    for name, spec in group_layer_specs(group).items()
                }
                unquantized_groups.append(cache_group(layer_names=list(specs), kv_cache_spec=requirements(specs)))
            assert divisor < cache.pool_bytes_per_block(unquantized_groups)
        # Rank normalization must reproduce the chosen block count exactly.
        if tq:
            assert (num_blocks * divisor) // cache.pool_bytes_per_block(groups) == num_blocks
    # State precision/padding is retained. The total capacity gain includes
    # pool slack and cannot be inferred from the C4-only 1024/258 ratio.
    assert capacities[1] > capacities[0]
    print(f"length={max_model_len}, block={block_size}: baseline={capacities[0]:.3f}x, TQ={capacities[1]:.3f}x")
