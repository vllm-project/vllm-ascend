# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for GLM-Next model-runner pooled cache views."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.core.single_type_kv_cache_manager import (
    register_all_kvcache_specs,
)
from vllm.v1.kv_cache_interface import KVCacheGroupSpec, MambaSpec, UniformTypeKVCacheSpecs

from vllm_ascend.core.kv_cache_interface import (
    AscendIndexerKPoolTailSpec,
    AscendMLAAttentionSpec,
    requires_padded_page_layout,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.layerwise_cache_layout import (
    apply_layerwise_kv_cache_plan,
)
from vllm_ascend.models.glm5next.cache_config import (
    get_glm5_next_kv_cache_config,
    get_glm5_next_kv_cache_groups,
    get_glm5_next_pool_bytes_per_block,
)
from vllm_ascend.models.glm5next.cache_views import view_kpool_tail_cache
from vllm_ascend.utils import get_kv_cache_tensor_layers
from vllm_ascend.worker.v2 import attn_utils

MAIN = "model.layers.1.attn"
INDEXER = "model.layers.1.indexer.k_cache"
STATE = "model.layers.1.indexer.tail_cache"
MAMBA = "model.layers.0.linear_attn"


def _ratio_kwargs(ratio: int) -> dict[str, int]:
    return {"tokens_per_state": ratio}


class _AttentionBackend:
    @staticmethod
    def get_kv_cache_shape(
        num_blocks,
        block_size,
        num_kv_heads,
        head_size,
        **_kwargs,
    ):
        return num_blocks, block_size, num_kv_heads, head_size


class _StateBackend:
    @staticmethod
    def get_kv_cache_shape(
        num_blocks,
        block_size,
        _num_kv_heads,
        head_size,
        **_kwargs,
    ):
        return num_blocks, block_size, 2 * head_size


def _make_config():
    return SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=64),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        max_in_flight_tokens=8,
        cache_config=SimpleNamespace(
            num_gpu_blocks_override=None,
            mamba_cache_mode="none",
            enable_prefix_caching=False,
            prefix_cache_retention_interval=0,
            get_resolved_kv_cache_layout=lambda: SimpleNamespace(
                is_layer_compact=True,
                is_block_compact=True,
            ),
        ),
        kv_transfer_config=None,
        compilation_config=SimpleNamespace(static_forward_context={}),
    )


def _make_specs(main_head_size=4):
    return {
        MAIN: AscendMLAAttentionSpec(
            block_size=8,
            num_kv_heads=1,
            head_size=main_head_size,
            dtype=torch.bfloat16,
            model_version="glm5_next",
            indexes_kv_by_block_stride=True,
        ),
        INDEXER: AscendMLAAttentionSpec(
            block_size=8,
            num_kv_heads=1,
            head_size=4,
            dtype=torch.bfloat16,
            model_version="glm5_next",
            indexes_kv_by_block_stride=True,
            **_ratio_kwargs(2),
        ),
        STATE: AscendIndexerKPoolTailSpec(
            block_size=2,
            ring_capacity=2,
            sliding_window=2,
            compress_ratio=2,
            num_kv_heads=1,
            head_size=1,
            dtype=torch.float32,
            model_version="glm5_next",
            indexes_kv_by_block_stride=True,
        ),
        MAMBA: MambaSpec(
            block_size=8,
            shapes=((2, 2), (1, 2, 2)),
            dtypes=(torch.bfloat16, torch.float32),
        ),
    }


def _make_plan(num_blocks=3, main_head_size=4):
    # Match production: vLLM registers built-in specs before the Ascend hook.
    register_all_kvcache_specs(None)
    config = _make_config()
    specs = _make_specs(main_head_size)
    groups = get_glm5_next_kv_cache_groups(config, specs)
    bytes_per_block = get_glm5_next_pool_bytes_per_block(groups)
    plan = get_glm5_next_kv_cache_config(
        config,
        groups,
        num_blocks * bytes_per_block,
    )
    return config, groups, plan


@pytest.fixture(params=[True, False], ids=["glm-marker", "layout-capability"])
def v2_glm_cache_plan(request, monkeypatch):
    config, _, plan = _make_plan()
    if not request.param:
        for group in plan.kv_cache_groups:
            spec = group.kv_cache_spec
            if isinstance(spec, UniformTypeKVCacheSpecs):
                group.kv_cache_spec = replace(
                    spec,
                    kv_cache_specs={
                        name: replace(layer_spec, model_version=None)
                        for name, layer_spec in spec.kv_cache_specs.items()
                    },
                )
    config.model_config.get_num_layers = lambda _: 2
    monkeypatch.setattr(attn_utils, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(attn_utils.KVPPConfig, "from_vllm_config", lambda _: SimpleNamespace(size=1))
    monkeypatch.setattr(attn_utils, "_is_dsv4_model", lambda _: False)
    monkeypatch.setattr(attn_utils, "enable_sfa", lambda _: False)
    return config, plan


@pytest.mark.parametrize(
    ("use_layerwise", "independent_layers"),
    [(None, None), (False, None), (True, None), (True, "all")],
    ids=["no-transfer", "non-layerwise", "layerwise", "independent"],
)
def test_v2_glm_shared_slots_preserve_backings(monkeypatch, v2_glm_cache_plan, use_layerwise, independent_layers):
    config, plan = v2_glm_cache_plan
    if use_layerwise is not None:
        extra_config = {"backend": "memcache", "use_layerwise": use_layerwise}
        if independent_layers is not None:
            extra_config["layerwise_independent_layers"] = independent_layers
        config.kv_transfer_config = SimpleNamespace(
            kv_connector="AscendStoreConnector", kv_connector_extra_config=extra_config
        )
    apply_layerwise_kv_cache_plan(plan, config)
    allocated_sizes = []
    allocate = attn_utils._allocate_int8_cache_tensor

    def allocate_slot(size, alignment, device):
        allocated_sizes.append(size)
        return allocate(size, alignment, device)

    monkeypatch.setattr(attn_utils, "_allocate_int8_cache_tensor", allocate_slot)

    raw_caches = attn_utils._allocate_kv_cache(plan, {}, torch.device("cpu"))

    assert set(raw_caches) == {MAIN, INDEXER, STATE, MAMBA}
    assert raw_caches[MAIN] is raw_caches[MAMBA]
    assert raw_caches[INDEXER] is raw_caches[STATE]
    assert raw_caches[MAIN] is not raw_caches[INDEXER]
    assert allocated_sizes == [descriptor.size for descriptor in plan.kv_cache_tensors]
    for descriptor in plan.kv_cache_tensors:
        backing = raw_caches[descriptor.layers[0]]
        assert backing.numel() == descriptor.size
        if use_layerwise is not None:
            assert backing.data_ptr() % (2 * 1024 * 1024) == 0


@pytest.mark.parametrize(
    "descriptor_changes",
    [
        {"offset": 1},
        {"layer_stride": 1},
        {"block_stride": 0},
        {"size": 1},
        {"layers": []},
        {"block_stride": 1, "size": 3},
    ],
)
def test_v2_glm_shared_slots_reject_invalid_geometry(v2_glm_cache_plan, descriptor_changes):
    _, plan = v2_glm_cache_plan
    plan.kv_cache_tensors[0] = replace(plan.kv_cache_tensors[0], **descriptor_changes)

    with pytest.raises(ValueError, match="Invalid padded shared-slot KV cache descriptor geometry"):
        attn_utils._allocate_kv_cache(plan, {}, torch.device("cpu"))


def test_v2_glm_shared_slots_reject_missing_layers(v2_glm_cache_plan):
    _, plan = v2_glm_cache_plan
    plan.kv_cache_tensors.pop()

    with pytest.raises(ValueError, match="Padded shared-slot KV cache descriptors must cover every cache spec"):
        attn_utils._allocate_kv_cache(plan, {}, torch.device("cpu"))


@pytest.mark.parametrize("kernel_size", [4, 8])
def test_glm_fp8_indexer_keys_and_scales_share_padded_pages_without_overlapping(monkeypatch, kernel_size):
    config = _make_config()
    specs = _make_specs()
    specs[INDEXER] = replace(
        specs[INDEXER],
        head_size=128,
        dtype=torch.float8_e4m3fn,
        scale_dim=1,
        scale_dtype=torch.float32,
        page_size_padded=640,
    )
    register_all_kvcache_specs(None)
    groups = get_glm5_next_kv_cache_groups(config, specs)
    plan = get_glm5_next_kv_cache_config(config, groups, 3 * get_glm5_next_pool_bytes_per_block(groups))
    config.model_config.get_num_layers = lambda _: 2
    monkeypatch.setattr(attn_utils, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(attn_utils.KVPPConfig, "from_vllm_config", lambda _: SimpleNamespace(size=1))
    monkeypatch.setattr(attn_utils, "_is_dsv4_model", lambda _: False)
    monkeypatch.setattr(attn_utils, "enable_sfa", lambda _: False)
    raw = attn_utils._allocate_kv_cache(plan, {}, torch.device("cpu"))
    group = SimpleNamespace(
        kv_cache_group_id=0, kv_cache_spec=specs[INDEXER], layer_names=[INDEXER], backend=_AttentionBackend
    )
    key, scale = attn_utils._reshape_kv_cache_v2([group], raw, "fp8", [kernel_size], {}, plan)[INDEXER]
    page_bytes = next(
        d.size // plan.num_blocks for d in plan.kv_cache_tensors if INDEXER in get_kv_cache_tensor_layers(d)
    )
    rows = kernel_size // 2
    assert key.shape == (3 * 8 // kernel_size, rows, 1, 128)
    assert scale.shape == (3 * 8 // kernel_size, rows, 1, 1)
    assert key.dtype == torch.float8_e4m3fn and scale.dtype == torch.float32
    assert key.stride(0) == scale.stride(0) * scale.element_size() == page_bytes // (8 // kernel_size)
    assert scale.data_ptr() - key.data_ptr() == rows * 128
    key[1].view(torch.uint8).fill_(23)
    scale[2].fill_(7)
    assert torch.count_nonzero(key[2].view(torch.uint8)) == 0
    assert torch.count_nonzero(scale[:2]) == 0
    assert torch.all(key[1].view(torch.uint8) == 23)


def test_cann_tail_views_keep_interleaved_rows_and_shared_page_stride(monkeypatch):
    config, _, plan = _make_plan()
    config.model_config.get_num_layers = lambda _: 2
    monkeypatch.setattr(attn_utils, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(attn_utils.KVPPConfig, "from_vllm_config", lambda _: SimpleNamespace(size=1))
    monkeypatch.setattr(attn_utils, "_is_dsv4_model", lambda _: False)
    monkeypatch.setattr(attn_utils, "enable_sfa", lambda _: False)
    raw_caches = attn_utils._allocate_kv_cache(plan, {}, torch.device("cpu"))
    spec = replace(_make_specs()[STATE], head_size=2)
    plan.kv_cache_groups = [KVCacheGroupSpec(layer_names=[STATE], kv_cache_spec=spec)]
    raw = raw_caches[STATE]
    expected = view_kpool_tail_cache(STATE, spec, raw, plan.num_blocks)[0]
    group = SimpleNamespace(kv_cache_group_id=0, kv_cache_spec=spec, layer_names=[STATE], backend=_StateBackend)
    actual = attn_utils._reshape_kv_cache_v2([group], {STATE: raw}, "auto", [2], {}, plan)[STATE][0]
    assert actual.shape == expected.shape == (plan.num_blocks, 2, 4)
    assert actual.data_ptr() == expected.data_ptr() == raw.data_ptr()
    page_bytes = raw.numel() // plan.num_blocks
    assert actual.stride(0) * actual.element_size() == page_bytes
    expected[1, :, :2] = 5
    expected[1, :, 2:] = 7
    torch.testing.assert_close(actual[1], torch.tensor([[5, 5, 7, 7], [5, 5, 7, 7]], dtype=torch.float32))
    torch.testing.assert_close(actual[0], torch.zeros_like(actual[0]))
    torch.testing.assert_close(actual[2], torch.zeros_like(actual[2]))


def test_speculative_tail_view_preserves_all_nine_physical_rows():
    spec = AscendIndexerKPoolTailSpec(
        block_size=4,
        ring_capacity=9,
        sliding_window=4,
        compress_ratio=4,
        num_kv_heads=1,
        head_size=2,
        dtype=torch.float32,
        page_size_padded=256,
    )
    raw = torch.zeros(3 * spec.page_size_bytes, dtype=torch.int8)
    view = view_kpool_tail_cache(STATE, spec, raw, 3)[0]
    assert view.shape == (3, 9, 4)
    assert view.stride() == (64, 4, 1)
    view[1, 8].fill_(7)
    torch.testing.assert_close(raw.view(torch.float32)[96:100], torch.full((4,), 7.0))
    assert not torch.count_nonzero(view[0])
    assert not torch.count_nonzero(view[2])


def test_padded_page_layout_rejected_without_tail_caches():
    # A standalone MTP runner has the same attention specs but no recurrent
    # state caches, so no state view needs the padded page stride.
    specs = {name: spec for name, spec in _make_specs().items() if not isinstance(spec, MambaSpec)}
    assert not requires_padded_page_layout(specs.values())


def test_padded_page_layout_rejected_for_packed_hybrid_pool():
    # Other hybrid models pad Mamba pages to the attention page size
    # (``cache_config.mamba_page_size_padded``) but keep the packed contiguous
    # state layout, so they must not take the shared padded page path.
    specs = [
        AscendMLAAttentionSpec(
            block_size=8,
            num_kv_heads=1,
            head_size=4,
            dtype=torch.bfloat16,
        ),
        MambaSpec(
            block_size=8,
            shapes=((2, 2), (1, 2, 2)),
            dtypes=(torch.bfloat16, torch.float32),
            page_size_padded=64,
        ),
    ]
    assert not requires_padded_page_layout(specs)
