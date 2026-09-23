# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for GLM-Next model-runner pooled cache views."""

from itertools import permutations
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.v1.core.single_type_kv_cache_manager import (
    register_all_kvcache_specs,
)
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, KVCacheTensor, MambaSpec

from vllm_ascend.attention.indexer_kpool import AscendIndexerKPoolBackend, AscendIndexerKPoolTailBackend
from vllm_ascend.core.kv_cache_interface import (
    AscendIndexerKPoolTailSpec,
    AscendMLAAttentionSpec,
    requires_padded_page_layout,
)
from vllm_ascend.models.glm5next.cache_config import (
    get_glm5_next_kv_cache_config,
    get_glm5_next_kv_cache_groups,
    get_glm5_next_pool_bytes_per_block,
)
from vllm_ascend.utils import get_kv_cache_tensor_layers
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
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


def _make_runner(config, main_cache_dims=(4, 0)):
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.device = torch.device("cpu")
    runner.vllm_config = config
    runner.ascend_config = SimpleNamespace(kvpp_config=SimpleNamespace(size=1))
    runner.compilation_config = config.compilation_config
    runner.runner_only_attn_layers = set()
    runner.shared_kv_cache_layers = {}
    runner.kv_caches = []
    runner.use_sparse = False
    # GLM-Next exposes its layout through KV specs and does not define the
    # legacy ``compress_ratios`` config used to derive this runner flag.
    runner.use_compress = False
    runner.use_hybrid_blocks = True
    runner.sparse_kv_offload_enabled = False
    runner.sparse_kv_offload_config = SimpleNamespace(enabled=False)
    runner.tp_rank = 0
    runner.attn_backend = _AttentionBackend
    runner.kernel_block_sizes = [[8], [2], [0]]
    # The runner gates GLM-Next reshape views on the spec-carried
    # model_version marker, not on model_config.
    runner.model_config = SimpleNamespace()

    specs = _make_specs()
    attn_groups = [
        SimpleNamespace(
            backend=_AttentionBackend,
            kv_cache_spec=specs[MAIN],
            layer_names=[MAIN],
            kv_cache_group_id=0,
        ),
        SimpleNamespace(
            backend=_AttentionBackend,
            kv_cache_spec=specs[INDEXER],
            layer_names=[INDEXER],
            kv_cache_group_id=0,
        ),
        SimpleNamespace(
            backend=_StateBackend,
            kv_cache_spec=specs[STATE],
            layer_names=[STATE],
            kv_cache_group_id=1,
        ),
        SimpleNamespace(
            backend=None,
            kv_cache_spec=specs[MAMBA],
            layer_names=[MAMBA],
            kv_cache_group_id=2,
        ),
    ]
    runner._kv_cache_spec_attn_group_iterator = lambda: iter(attn_groups)
    runner._get_attention_kv_cache_dims = lambda _name, _spec: main_cache_dims
    return runner


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


def test_glm5_next_runner_allocates_contiguous_slot_backings():
    config, _, plan = _make_plan()
    runner = _make_runner(config)

    raw_caches = runner._allocate_kv_cache_tensors(plan)
    assert raw_caches[MAIN] is raw_caches[MAMBA]
    assert raw_caches[INDEXER] is raw_caches[STATE]
    assert raw_caches[MAIN] is not raw_caches[INDEXER]

    caches = runner._reshape_kv_cache_tensors(plan, raw_caches)
    descriptors = {
        name: descriptor for descriptor in plan.kv_cache_tensors for name in get_kv_cache_tensor_layers(descriptor)
    }
    main_cache, main_rope_cache = caches[MAIN]
    (indexer_cache,) = caches[INDEXER]
    (tail_cache,) = caches[STATE]
    assert main_cache.shape == (3, 8, 1, 4)
    assert main_rope_cache.shape == (3, 8, 1, 0)
    assert main_cache.is_contiguous()
    assert indexer_cache.shape == (3, 4, 1, 4)
    assert tail_cache.shape == (3, 2, 2)
    assert [cache.shape for cache in caches[MAMBA]] == [
        (3, 2, 2),
        (3, 1, 2, 2),
    ]

    assert indexer_cache.is_contiguous()
    assert indexer_cache.data_ptr() == raw_caches[INDEXER].data_ptr()
    tail_packed_bytes = tail_cache.numel() * tail_cache.element_size()
    slot = raw_caches[STATE]
    slot_bytes = slot.numel() * slot.element_size()
    assert tail_cache.data_ptr() + tail_packed_bytes == slot.data_ptr() + slot_bytes
    for cache in caches[MAMBA]:
        assert cache.stride(0) * cache.element_size() == descriptors[MAMBA].size // plan.num_blocks

    mamba_second_offset = caches[MAMBA][0][0].numel() * caches[MAMBA][0].element_size()
    assert caches[MAMBA][1].data_ptr() - raw_caches[MAMBA].data_ptr() == mamba_second_offset
    mamba_payload_size = sum(cache.numel() * cache.element_size() for cache in caches[MAMBA])
    assert mamba_payload_size < descriptors[MAMBA].size

    tail_cache[2].fill_(7)
    tail_packed_els = tail_packed_bytes // slot.element_size()
    block2_els = tail_cache[2].numel() * tail_cache[2].element_size() // slot.element_size()
    assert torch.all(slot[slot.numel() - block2_els :].view(torch.float32) == 7)
    assert torch.count_nonzero(slot[: slot.numel() - tail_packed_els]) == 0
    assert torch.count_nonzero(slot[slot.numel() - tail_packed_els : slot.numel() - block2_els]) == 0


@pytest.mark.parametrize("runner_version", ["v1", "v2"])
@pytest.mark.parametrize("storage_offset", [0, 64])
def test_cann_tail_backend_preserves_packed_layout(runner_version, storage_offset, monkeypatch):
    # A speculative tail has a different capacity from the pool size. Use
    # D > 1 so [blocks, 2, capacity, D] cannot masquerade as interleaved K/G.
    num_blocks, logical_block_size, pool_size, capacity, head_dim = 3, 256, 4, 7, 5
    page_bytes = 4096
    specs = {
        INDEXER: AscendMLAAttentionSpec(
            block_size=logical_block_size,
            num_kv_heads=1,
            head_size=head_dim,
            dtype=torch.bfloat16,
            page_size_padded=page_bytes,
            model_version="glm5_next",
            indexes_kv_by_block_stride=True,
            **_ratio_kwargs(pool_size),
        ),
        STATE: AscendIndexerKPoolTailSpec(
            block_size=capacity,
            sliding_window=pool_size,
            compress_ratio=pool_size,
            num_kv_heads=1,
            head_size=head_dim,
            dtype=torch.float32,
            page_size_padded=page_bytes,
        ),
    }
    plan = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=num_blocks * page_bytes,
                layers=[INDEXER, STATE],
                layer_stride=page_bytes,
                block_stride=page_bytes,
                offset=0,
            )
        ],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=[name], kv_cache_spec=spec) for name, spec in specs.items()],
    )
    groups = [
        SimpleNamespace(
            kv_cache_group_id=gid,
            kv_cache_spec=specs[name],
            layer_names=[name],
            backend=backend,
        )
        for gid, (name, backend) in enumerate(
            [(INDEXER, AscendIndexerKPoolBackend), (STATE, AscendIndexerKPoolTailBackend)]
        )
    ]
    backing = torch.full((storage_offset + num_blocks * page_bytes + 64,), 17, dtype=torch.int8)
    raw = backing[storage_offset : storage_offset + num_blocks * page_bytes]
    raw_caches = {INDEXER: raw, STATE: raw}
    config = _make_config()
    if runner_version == "v1":
        runner = _make_runner(config)
        runner.kernel_block_sizes = [[128], [capacity]]
        runner._kv_cache_spec_attn_group_iterator = lambda: iter(groups)
        caches = runner._reshape_kv_cache_tensors(plan, raw_caches)
    else:
        monkeypatch.setattr(attn_utils, "get_current_vllm_config", lambda: config)
        monkeypatch.setattr(attn_utils, "enable_sfa", lambda _config: False)
        caches = attn_utils._reshape_kv_cache_v2(groups, raw_caches, "auto", [128, capacity], {}, plan)

    (indexer_cache,) = caches[INDEXER]
    (tail_cache,) = caches[STATE]
    assert indexer_cache.shape == (num_blocks * 2, 128 // pool_size, 1, head_dim)
    assert tail_cache.shape == (num_blocks, capacity, 2 * head_dim)
    assert tail_cache.dtype == torch.float32
    assert indexer_cache.is_contiguous() and tail_cache.is_contiguous()
    assert indexer_cache.data_ptr() == raw.data_ptr()
    tail_bytes = tail_cache.numel() * tail_cache.element_size()
    assert tail_cache.data_ptr() == raw.data_ptr() + raw.numel() - tail_bytes
    indexer_bytes = indexer_cache.numel() * indexer_cache.element_size()
    assert indexer_cache.data_ptr() + indexer_bytes <= tail_cache.data_ptr()

    indexer_cache.fill_(11)
    expected = backing.clone()
    # K and gate occupy adjacent columns of every ring row, never two planes.
    expected_tail = expected[storage_offset + raw.numel() - tail_bytes : storage_offset + raw.numel()].view(
        torch.float32
    )
    expected_rows = expected_tail.view(num_blocks, capacity, 2 * head_dim)
    expected_rows[:, :, :head_dim] = 5
    expected_rows[:, :, head_dim:] = 7
    tail_cache[:, :, :head_dim] = 5
    tail_cache[:, :, head_dim:] = 7
    torch.testing.assert_close(backing, expected, rtol=0, atol=0)
    torch.testing.assert_close(indexer_cache, torch.full_like(indexer_cache, 11), rtol=0, atol=0)


def test_glm5_next_runner_splits_main_mla_components_within_each_page():
    config, _, plan = _make_plan(main_head_size=6)
    runner = _make_runner(config, main_cache_dims=(4, 2))

    raw_caches = runner._allocate_kv_cache_tensors(plan)
    caches = runner._reshape_kv_cache_tensors(plan, raw_caches)

    kv_c_cache, k_pe_cache = caches[MAIN]
    assert kv_c_cache.shape == (3, 8, 1, 4)
    assert k_pe_cache.shape == (3, 8, 1, 2)
    page_size = next(
        descriptor.size // plan.num_blocks
        for descriptor in plan.kv_cache_tensors
        if MAIN in get_kv_cache_tensor_layers(descriptor)
    )
    assert kv_c_cache.stride(0) * kv_c_cache.element_size() == page_size
    assert k_pe_cache.stride(0) * k_pe_cache.element_size() == page_size
    assert k_pe_cache.data_ptr() - raw_caches[MAIN].data_ptr() == kv_c_cache[0].numel() * kv_c_cache.element_size()


def test_standalone_mtp_uses_existing_compressed_cache_allocator():
    config = _make_config()
    specs = {name: spec for name, spec in _make_specs().items() if not isinstance(spec, MambaSpec)}
    groups = get_glm5_next_kv_cache_groups(config, specs)
    bytes_per_block = get_glm5_next_pool_bytes_per_block(groups)
    plan = get_glm5_next_kv_cache_config(config, groups, 3 * bytes_per_block)

    raw_caches = _make_runner(config)._allocate_kv_cache_tensors(plan)

    assert set(raw_caches) == {MAIN, INDEXER, STATE}
    assert raw_caches[INDEXER] is raw_caches[STATE]


def test_glm5_next_initialize_passes_all_pooled_views_to_cache_binding():
    config, _, plan = _make_plan()
    runner = _make_runner(config)
    runner.model_config = SimpleNamespace(hf_text_config=SimpleNamespace(model_type="glm5_next"))

    with patch("vllm.v1.worker.utils.bind_kv_cache") as bind_kv_cache:
        caches = runner.initialize_kv_cache_tensors(plan)

    assert set(caches) == {MAIN, INDEXER, STATE, MAMBA}
    bind_kv_cache.assert_called_once_with(
        caches,
        runner.compilation_config.static_forward_context,
        runner.kv_caches,
        1,
    )


@pytest.mark.parametrize("offset", [0, 64])
def test_glm_shared_mla_mamba_pages_preserve_other_block_ids(offset):
    config, _, plan = _make_plan(num_blocks=4)
    runner = _make_runner(config)
    raw_tensors = runner._allocate_kv_cache_tensors(plan)
    assert raw_tensors[MAIN] is raw_tensors[MAMBA]
    page_bytes = raw_tensors[MAIN].numel() // plan.num_blocks
    backing = torch.zeros(offset + raw_tensors[MAIN].numel() + 64, dtype=torch.int8)
    raw = backing[offset : offset + plan.num_blocks * page_bytes]
    raw_tensors[MAIN] = raw_tensors[MAMBA] = raw
    caches = runner._reshape_kv_cache_tensors(plan, raw_tensors)
    latent, _ = caches[MAIN]
    conv, ssm = caches[MAMBA]
    for state in (conv, ssm):
        assert state.untyped_storage().data_ptr() == raw.untyped_storage().data_ptr()
        assert state.stride(0) * state.element_size() == page_bytes
    assert ssm.data_ptr() - conv.data_ptr() == conv[0].numel() * conv.element_size()
    state_bytes = sum(state[0].numel() * state.element_size() for state in (conv, ssm))
    for mla_id, state_id in permutations(range(plan.num_blocks), 2):
        raw.zero_()
        latent[mla_id].fill_(7)
        conv[state_id].fill_(11)
        ssm[state_id].fill_(13)
        torch.testing.assert_close(latent[mla_id], torch.full_like(latent[mla_id], 7))
        latent[mla_id].fill_(17)
        torch.testing.assert_close(conv[state_id], torch.full_like(conv[state_id], 11))
        torch.testing.assert_close(ssm[state_id], torch.full_like(ssm[state_id], 13))
        assert torch.count_nonzero(raw.view(plan.num_blocks, page_bytes)[state_id, state_bytes:]) == 0
    assert torch.count_nonzero(backing[:offset]) == 0
    assert torch.count_nonzero(backing[offset + raw.numel() :]) == 0


def test_padded_page_layout_detected_for_shared_state_pages():
    # The pooled layout pads the state caches to the page size of the
    # block-stride addressed MLA/indexer caches they share physical pages with.
    # Probe the specs the runner itself derives from the KV cache config.
    config, _, plan = _make_plan()
    layer_specs = _make_runner(config)._get_layer_kv_cache_specs(plan)
    assert requires_padded_page_layout(layer_specs.values())


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
