# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for GLM-Next model-runner pooled cache views."""

from types import SimpleNamespace
from unittest.mock import patch

import torch
from vllm.v1.core.single_type_kv_cache_manager import (
    register_all_kvcache_specs,
)
from vllm.v1.kv_cache_interface import MambaSpec

from vllm_ascend.core.kv_cache_interface import (
    AscendIndexerKPoolStateSpec,
    AscendMLAAttentionSpec,
)
from vllm_ascend.models.glm5next.cache_config import (
    get_glm5_next_kv_cache_config,
    get_glm5_next_kv_cache_groups,
    get_glm5_next_pool_bytes_per_block,
)
from vllm_ascend.utils import get_kv_cache_tensor_layers, vllm_version_is
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

MAIN = "model.layers.1.attn"
INDEXER = "model.layers.1.indexer.k_cache"
STATE = "model.layers.1.indexer.state_cache"
MAMBA = "model.layers.0.linear_attn"


def _ratio_kwargs(ratio: int) -> dict[str, int]:
    return {"compress_ratio": ratio} if vllm_version_is("0.28.0") else {"tokens_per_state": ratio}


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
        return num_blocks, block_size, head_size


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
        STATE: AscendIndexerKPoolStateSpec(
            block_size=2,
            sliding_window=2,
            num_kv_heads=1,
            head_size=3,
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
    # The runner must consume the descriptor/spec contract without inspecting
    # a model type.
    runner.model_config = SimpleNamespace()

    specs = _make_specs()
    attn_groups = [
        SimpleNamespace(
            backend=_AttentionBackend,
            kv_cache_spec=specs[MAIN],
            layer_names=[MAIN],
        ),
        SimpleNamespace(
            backend=_AttentionBackend,
            kv_cache_spec=specs[INDEXER],
            layer_names=[INDEXER],
        ),
        SimpleNamespace(
            backend=_StateBackend,
            kv_cache_spec=specs[STATE],
            layer_names=[STATE],
        ),
        SimpleNamespace(
            backend=None,
            kv_cache_spec=specs[MAMBA],
            layer_names=[MAMBA],
        ),
    ]
    runner._kv_cache_spec_attn_group_iterator = lambda: iter(attn_groups)
    runner._get_attention_kv_cache_dims = lambda _name, _spec: main_cache_dims
    return runner


def _make_plan(num_blocks=3, main_head_size=4):
    # Match production: vLLM registers built-in specs before the Ascend hook.
    register_all_kvcache_specs(None)
    config = _make_config()
    groups = get_glm5_next_kv_cache_groups(config, _make_specs(main_head_size))
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
    (state_cache,) = caches[STATE]
    assert main_cache.shape == (3, 8, 1, 4)
    assert main_rope_cache.shape == (3, 8, 1, 0)
    assert main_cache.is_contiguous()
    assert indexer_cache.shape == (3, 4, 1, 4)
    assert state_cache.shape == (3, 2, 3)
    assert [cache.shape for cache in caches[MAMBA]] == [
        (3, 2, 2),
        (3, 1, 2, 2),
    ]

    for name, cache in ((INDEXER, indexer_cache), (STATE, state_cache)):
        page_size = descriptors[name].size // plan.num_blocks
        assert cache.stride(0) * cache.element_size() == page_size
        assert cache.data_ptr() == raw_caches[name].data_ptr()
    assert all(cache.is_contiguous() for cache in caches[MAMBA])

    mamba_second_offset = caches[MAMBA][0].numel() * caches[MAMBA][0].element_size()
    assert caches[MAMBA][1].data_ptr() - raw_caches[MAMBA].data_ptr() == mamba_second_offset
    mamba_payload_size = sum(cache.numel() * cache.element_size() for cache in caches[MAMBA])
    assert mamba_payload_size < descriptors[MAMBA].size

    state_cache[2].fill_(7)
    state_payload_size = state_cache[0].numel() * state_cache.element_size()
    state_padding = 2 * (descriptors[STATE].size // plan.num_blocks) + state_payload_size
    assert raw_caches[STATE][state_padding].item() == 0


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
