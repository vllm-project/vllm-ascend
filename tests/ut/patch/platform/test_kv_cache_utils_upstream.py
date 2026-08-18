# SPDX-License-Identifier: Apache-2.0

import math
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.core import kv_cache_utils
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    SlidingWindowMLASpec,
)

from vllm_ascend.core.kv_cache_interface import (
    AscendMLAAttentionSpec,
    AscendSlidingWindowMLASpec,
)

DSV4_BLOCK_SIZE = 128
NUM_BLOCKS = 5


class _StubSpeculativeConfig:
    @staticmethod
    def use_eagle() -> bool:
        return True


def _new_mla_spec(
    compress_ratio: int,
    *,
    head_size: int,
) -> AscendMLAAttentionSpec:
    return AscendMLAAttentionSpec(
        block_size=DSV4_BLOCK_SIZE,
        num_kv_heads=1,
        head_size=head_size,
        dtype=torch.float16,
        compress_ratio=compress_ratio,
        model_version="deepseek_v4",
        indexes_kv_by_block_stride=True,
    )


def _new_swa_spec(*, block_size: int = DSV4_BLOCK_SIZE) -> AscendSlidingWindowMLASpec:
    return AscendSlidingWindowMLASpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float16,
        sliding_window=4096,
        compress_ratio=1,
        model_version="deepseek_v4",
        indexes_kv_by_block_stride=True,
    )


def _make_dsv4_specs() -> dict[str, AscendMLAAttentionSpec | AscendSlidingWindowMLASpec]:
    return {
        "model.layers.0.c4_attn": _new_mla_spec(4, head_size=4),
        "model.layers.1.c4_attn": _new_mla_spec(4, head_size=4),
        "model.layers.2.c128_attn": _new_mla_spec(128, head_size=2),
        "model.layers.3.swa_attn": _new_swa_spec(),
        "model.layers.4.swa_attn": _new_swa_spec(),
        # Upstream's EAGLE annotation marks the group containing the final
        # DeepSeek-V4 attention layer.
        "model.mtp.layers.0.c128_attn": _new_mla_spec(128, head_size=2),
    }


def _make_dsv4_groups() -> tuple[
    dict[str, AscendMLAAttentionSpec | AscendSlidingWindowMLASpec],
    list[KVCacheGroupSpec],
]:
    specs = _make_dsv4_specs()
    grouped_specs = kv_cache_utils.group_and_unify_kv_cache_specs(specs)
    assert grouped_specs is not None
    groups = kv_cache_utils._get_kv_cache_groups_uniform_groups(grouped_specs)
    return specs, groups


@pytest.mark.parametrize(
    "compress_ratio",
    [
        pytest.param(1, id="ordinary"),
        pytest.param(4, id="c4"),
        pytest.param(128, id="c128"),
    ],
)
def test_ascend_dsv4_block_size_metadata(compress_ratio: int) -> None:
    spec = _new_mla_spec(compress_ratio, head_size=4)

    assert spec.block_size == DSV4_BLOCK_SIZE
    assert spec.storage_block_size == DSV4_BLOCK_SIZE
    assert spec.page_size_bytes == DSV4_BLOCK_SIZE * 4 * 2
    assert spec.compress_ratio == compress_ratio
    assert spec.model_version == "deepseek_v4"
    assert spec.indexes_kv_by_block_stride


def test_ascend_dsv4_page_size_padding_metadata() -> None:
    mla_spec = AscendMLAAttentionSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=3,
        dtype=torch.float16,
        page_size_padded=32,
    )
    swa_spec = AscendSlidingWindowMLASpec(
        block_size=4,
        num_kv_heads=1,
        head_size=3,
        dtype=torch.float16,
        sliding_window=128,
        alignment=16,
    )

    assert mla_spec.real_page_size_bytes == 24
    assert mla_spec.page_size_bytes == 32
    assert swa_spec.real_page_size_bytes == 24
    assert swa_spec.page_size_bytes == 32


def test_upstream_dsv4_cache_group_structure() -> None:
    specs = _make_dsv4_specs()

    grouped_specs = kv_cache_utils.group_and_unify_kv_cache_specs(specs)

    assert grouped_specs is not None
    assert len(grouped_specs) == 2
    groups = kv_cache_utils._get_kv_cache_groups_uniform_groups(grouped_specs)
    assert [group.layer_names for group in groups] == [
        [
            "model.layers.0.c4_attn",
            "model.layers.1.c4_attn",
            "model.layers.2.c128_attn",
            "model.mtp.layers.0.c128_attn",
        ],
        ["model.layers.3.swa_attn", "model.layers.4.swa_attn"],
    ]


def test_upstream_dsv4_packed_layout_size_offset_and_block_stride() -> None:
    specs, groups = _make_dsv4_groups()
    expected_offsets = {
        "model.layers.0.c4_attn": 0,
        "model.layers.3.swa_attn": 0,
        "model.layers.4.swa_attn": specs["model.layers.3.swa_attn"].page_size_bytes,
        "model.layers.1.c4_attn": specs["model.layers.0.c4_attn"].page_size_bytes,
        "model.layers.2.c128_attn": 2 * specs["model.layers.0.c4_attn"].page_size_bytes,
        "model.mtp.layers.0.c128_attn": (
            2 * specs["model.layers.0.c4_attn"].page_size_bytes + specs["model.layers.2.c128_attn"].page_size_bytes
        ),
    }
    expected_block_stride = sum(specs[layer_name].page_size_bytes for layer_name in groups[0].layer_names)
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(num_gpu_blocks_override=None),
    )

    num_blocks, tensors = kv_cache_utils._get_kv_cache_config_packed(
        vllm_config,
        groups,
        expected_block_stride * NUM_BLOCKS,
    )

    assert num_blocks == NUM_BLOCKS
    assert {tensor.size for tensor in tensors} == {expected_block_stride * NUM_BLOCKS}
    assert {tensor.block_stride for tensor in tensors} == {expected_block_stride}
    assert all(tensor.block_stride > 0 for tensor in tensors)
    actual_offsets = {layer_name: tensor.offset for tensor in tensors for layer_name in tensor.shared_by}
    assert actual_offsets == expected_offsets
    assert len(set(actual_offsets.values())) > 1


@pytest.mark.parametrize(
    ("enable_prefix_caching", "expected_hash_block_size"),
    [
        pytest.param(False, 256, id="dcp-without-prefix-cache"),
        pytest.param(True, 64, id="dcp-with-prefix-cache"),
    ],
)
def test_upstream_dcp_and_prefix_cache_block_sizes(
    enable_prefix_caching: bool,
    expected_hash_block_size: int,
) -> None:
    c4_spec = _new_mla_spec(4, head_size=4)
    swa_spec: SlidingWindowMLASpec = _new_swa_spec(block_size=32)
    kv_cache_config = KVCacheConfig(
        num_blocks=NUM_BLOCKS,
        kv_cache_tensors=[],
        # Scheduler configs contain a representative per-layer spec rather
        # than the worker's UniformTypeKVCacheSpecs wrapper.
        kv_cache_groups=[
            KVCacheGroupSpec(["c4_attn"], c4_spec),
            KVCacheGroupSpec(["swa_attn"], swa_spec),
        ],
    )
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=DSV4_BLOCK_SIZE,
            enable_prefix_caching=enable_prefix_caching,
            prefix_match_unit=None,
        ),
        parallel_config=SimpleNamespace(decode_context_parallel_size=2),
        kv_transfer_config=None,
    )

    scheduler_block_size, hash_block_size = kv_cache_utils.resolve_kv_cache_block_sizes(
        kv_cache_config,
        vllm_config,
    )

    assert scheduler_block_size == math.lcm(DSV4_BLOCK_SIZE * 2, 32 * 2)
    assert hash_block_size == expected_hash_block_size


def test_upstream_marks_the_mtp_group_as_eagle() -> None:
    specs, groups = _make_dsv4_groups()
    vllm_config = SimpleNamespace(
        speculative_config=_StubSpeculativeConfig(),
    )

    kv_cache_utils._annotate_eagle_groups_deepseek_v4(
        vllm_config,
        specs,
        groups,
    )

    assert [group.is_eagle_group for group in groups] == [True, False]
    assert "model.mtp.layers.0.c128_attn" in groups[0].layer_names
