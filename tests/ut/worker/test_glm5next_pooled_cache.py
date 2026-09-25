# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for GLM-Next model-runner pooled cache views."""

from contextlib import nullcontext
from dataclasses import replace
from itertools import permutations
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.model_executor.layers.attention.mla_attention import MLAAttention
from vllm.v1.core.single_type_kv_cache_manager import (
    register_all_kvcache_specs,
)
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, KVCacheTensor, MambaSpec, MLAAttentionSpec
from vllm.v1.worker import utils as worker_utils
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

from vllm_ascend.core.kv_cache_interface import (
    AscendIndexerKPoolTailSpec,
    AscendMLAAttentionSpec,
    requires_padded_page_layout,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.utils import collect_configured_register_regions
from vllm_ascend.models.glm5next.cache_config import (
    get_glm5_next_kv_cache_config,
    get_glm5_next_kv_cache_groups,
    get_glm5_next_pool_bytes_per_block,
)
from vllm_ascend.utils import get_kv_cache_tensor_layers
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
from vllm_ascend.worker.v2 import attn_utils
from vllm_ascend.worker.v2.block_table import AscendBlockTables
from vllm_ascend.worker.v2.model_runner import NPUModelRunner as NPUModelRunnerV2

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
        cache_dtype=None,
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
        return num_blocks, 2, block_size, head_size


def _make_config():
    return SimpleNamespace(
        additional_config={},
        model_config=SimpleNamespace(max_model_len=64, dtype=torch.bfloat16),
        attention_config=SimpleNamespace(indexer_kv_dtype="auto"),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        max_in_flight_tokens=8,
        cache_config=SimpleNamespace(
            cache_dtype="auto",
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


@pytest.fixture(params=["v1", "v2"])
def runner_factory(request, monkeypatch):
    def make_runner(config, main_cache_dims=(4, 0), kernel_block_size=8):
        runner = _make_runner(config, main_cache_dims)
        runner.kernel_block_sizes[0] = [kernel_block_size]
        if request.param == "v2":
            layers = {name: SimpleNamespace() for name in _make_specs()}
            main_layer = MLAAttention.__new__(MLAAttention)
            torch.nn.Module.__init__(main_layer)
            main_layer.kv_lora_rank, main_layer.qk_rope_head_dim = main_cache_dims
            layers[MAIN] = main_layer
            monkeypatch.setattr(attn_utils, "get_current_vllm_config", lambda: config)
            monkeypatch.setattr(attn_utils, "get_layers_from_vllm_config", lambda *_args: layers)
            monkeypatch.setattr(attn_utils, "enable_sfa", lambda _config: False)
            monkeypatch.setattr(attn_utils, "enable_fa_quant", lambda _config: False)
            runner._allocate_kv_cache_tensors = lambda plan: attn_utils._allocate_kv_cache(plan, {}, runner.device)

            def reshape(plan, raw):
                specs = attn_utils._get_layer_kv_cache_specs(plan)
                groups = [
                    SimpleNamespace(
                        kv_cache_group_id=i,
                        kv_cache_spec=specs[name],
                        layer_names=[name],
                        backend=_StateBackend if name == STATE else _AttentionBackend,
                    )
                    for i, group in enumerate(plan.kv_cache_groups)
                    for name in group.layer_names
                ]
                return attn_utils._reshape_kv_cache_v2(
                    groups,
                    raw,
                    "auto",
                    [
                        kernel_block_size if i == 0 else group.kv_cache_spec.block_size
                        for i, group in enumerate(plan.kv_cache_groups)
                    ],
                    {},
                    plan,
                )

            runner._reshape_kv_cache_tensors = reshape
        return runner

    return make_runner


def test_glm5_next_runner_allocates_contiguous_slot_backings(runner_factory):
    config, _, plan = _make_plan()
    runner = runner_factory(config)

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
    assert tail_cache.shape == (3, 2, 2, 1)
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


def test_glm5_next_runner_splits_main_mla_components_within_each_page(runner_factory):
    config, _, plan = _make_plan(main_head_size=6)
    runner = runner_factory(config, main_cache_dims=(4, 2))

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


@pytest.mark.parametrize("kv_transfer", [False, True])
@pytest.mark.parametrize("reverse_bindings", [False, True])
@pytest.mark.parametrize("main_cache_dims", [(4, 0), (4, 2)])
@pytest.mark.parametrize("kernel_block_size", [4, 8])
def test_mrv2_copy_on_write_preserves_pooled_pages_and_layer_bindings(
    runner_factory, monkeypatch, kv_transfer, reverse_bindings, main_cache_dims, kernel_block_size
):
    config, _, plan = _make_plan(main_head_size=sum(main_cache_dims))
    if kv_transfer:
        config.kv_transfer_config = SimpleNamespace(kv_connector="MooncakeConnectorV2")
    allocator = runner_factory(config, main_cache_dims, kernel_block_size)
    raw_caches = allocator._allocate_kv_cache_tensors(plan)
    caches = allocator._reshape_kv_cache_tensors(plan, raw_caches)
    if kv_transfer:
        regions = collect_configured_register_regions(plan, caches)
        assert set(zip(regions.ptrs, regions.lengths)) == {
            (raw_caches[name].data_ptr(), raw_caches[name].numel()) for name in (MAIN, INDEXER)
        }
        assert regions.logical_tensor_count == len(plan.kv_cache_tensors)
    bindings = {name: SimpleNamespace(kv_cache=cache) for name, cache in caches.items()}
    config.compilation_config.static_forward_context = bindings

    runner = NPUModelRunnerV2.__new__(NPUModelRunnerV2)
    runner.device = torch.device("cpu")
    runner.vllm_config = config
    runner.compilation_config = config.compilation_config
    runner.model_config = SimpleNamespace(enable_return_routed_experts=False)
    runner.model_state = SimpleNamespace()
    runner.pcp_manager = None
    runner.speculator = None
    runner.attn_groups = []

    def initialize(_runner, cache_config, kv_cache_allocation_context=None):
        _runner.kv_cache_config = cache_config
        _runner.block_tables = AscendBlockTables.__new__(AscendBlockTables)
        _runner.block_tables.cp_size = 1
        _runner.block_tables.block_sizes = [group.kv_cache_spec.block_size for group in cache_config.kv_cache_groups]
        _runner.block_tables.kernel_block_sizes = _runner.block_tables.block_sizes
        _runner.block_tables.is_circular = None
        _runner.kv_caches = list(caches.values())
        if reverse_bindings:
            _runner.kv_caches.reverse()

    monkeypatch.setattr(GPUModelRunner, "initialize_kv_cache", initialize)
    with (
        patch("vllm_ascend.worker.v2.model_runner.graph_manager_wrapper", return_value=nullcontext()),
        patch("vllm_ascend.worker.v2.model_runner.KVPPRuntime.create_from_kv_cache"),
    ):
        runner.initialize_kv_cache(plan)

    assert all(isinstance(cache, torch.Tensor) and cache.numel() > 0 for cache in runner.kv_caches)
    for name, cache in caches.items():
        assert bindings[name].kv_cache is cache

    # Copy a chain using the original source pages. Copying an aliased backing
    # twice would incorrectly propagate page 0 into page 2.
    for name in (MAIN, INDEXER):
        raw = raw_caches[name]
        torch.empty(0, dtype=torch.int8).set_(raw.untyped_storage()).fill_(99)
    main_pages = raw_caches[MAIN].view(plan.num_blocks, -1)
    for page_id, value in enumerate((11, 22, 33)):
        main_pages[page_id].fill_(value)
    # Each small-slot component has its own block stride. Filling raw bytes
    # by descriptor.block_stride would conceal wrong whole-slot copying.
    (indexer,) = caches[INDEXER]
    (tail,) = caches[STATE]
    indexer_pages = indexer.view(plan.num_blocks, -1)
    for page_id, value in enumerate((3, 5, 7)):
        indexer_pages[page_id].fill_(value)
        tail[page_id].fill_(value + 10)
    expected_small = raw_caches[INDEXER].clone()
    indexer_bytes = indexer.numel() * indexer.element_size()
    tail_bytes = tail.numel() * tail.element_size()
    expected_small[:indexer_bytes].view(indexer.dtype).view(plan.num_blocks, -1)[[1, 2]] = indexer_pages[[0, 1]]
    expected_small[-tail_bytes:].view(tail.dtype).view(tail.shape)[[1, 2]] = tail[[0, 1]]
    with patch.object(
        worker_utils, "async_tensor_h2d", side_effect=lambda data, device: torch.as_tensor(data, device=device)
    ):
        worker_utils.copy_kv_cache_blocks_inplace(runner.kv_caches, plan.num_blocks, [(0, 1), (1, 2)])

    for page_id, value in enumerate((11, 11, 22)):
        assert torch.all(main_pages[page_id] == value)
    torch.testing.assert_close(raw_caches[INDEXER], expected_small)
    for name in (MAIN, INDEXER):
        raw = raw_caches[name]
        storage = torch.empty(0, dtype=torch.int8).set_(raw.untyped_storage())
        assert torch.all(storage[: raw.storage_offset()] == 99)
        assert torch.all(storage[raw.storage_offset() + raw.numel() :] == 99)


def test_standalone_mtp_uses_existing_compressed_cache_allocator(runner_factory):
    config = _make_config()
    specs = {name: spec for name, spec in _make_specs().items() if not isinstance(spec, MambaSpec)}
    groups = get_glm5_next_kv_cache_groups(config, specs)
    bytes_per_block = get_glm5_next_pool_bytes_per_block(groups)
    plan = get_glm5_next_kv_cache_config(config, groups, 3 * bytes_per_block)

    runner = runner_factory(config)
    raw_caches = runner._allocate_kv_cache_tensors(plan)

    assert set(raw_caches) == {MAIN, INDEXER, STATE}
    assert raw_caches[INDEXER] is raw_caches[STATE]
    runner.runner_only_attn_layers.add(MAMBA)
    caches = runner._reshape_kv_cache_tensors(plan, raw_caches)
    assert caches[INDEXER][0].shape == (3, 4, 1, 4)
    assert caches[STATE][0].shape == (3, 2, 2, 1)


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
def test_glm_shared_mla_mamba_pages_preserve_other_block_ids(offset, runner_factory):
    config, _, plan = _make_plan(num_blocks=4)
    runner = runner_factory(config)
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


def test_mrv2_preserves_glm_cache_roles_and_auxiliary_page_sizes(monkeypatch):
    specs = _make_specs()
    main = MLAAttention.__new__(MLAAttention)
    torch.nn.Module.__init__(main)
    main.impl = SimpleNamespace(fa_quant_layer=False)
    main.model_version = "glm5_next"
    main.indexes_kv_by_block_stride = True
    main.get_kv_cache_spec = lambda _config: MLAAttentionSpec(
        block_size=8,
        num_kv_heads=1,
        head_size=4,
        dtype=torch.bfloat16,
        non_causal_multi_token_decode=True,
    )
    layers = {
        name: SimpleNamespace(
            get_kv_cache_spec=lambda _config, spec=spec: spec,
            align_kv_cache_with_mamba=name not in (INDEXER, STATE),
        )
        for name, spec in specs.items()
    }
    layers[MAIN] = main
    monkeypatch.setattr(attn_utils, "get_layers_from_vllm_config", lambda *_args: layers)
    monkeypatch.setattr(attn_utils, "enable_sfa", lambda _config: False)
    monkeypatch.setattr(attn_utils, "enable_sfa_dcp_replicated_indexer", lambda _config: False)

    discovered = attn_utils.get_kv_cache_spec(_make_config())

    assert discovered[MAIN].model_version == "glm5_next"
    assert discovered[MAIN].indexes_kv_by_block_stride
    assert discovered[MAIN].non_causal_multi_token_decode
    assert discovered[INDEXER] == specs[INDEXER]
    assert discovered[STATE] == specs[STATE]
    assert discovered[MAMBA].page_size_bytes == discovered[MAIN].page_size_bytes
    assert discovered[INDEXER].page_size_bytes < discovered[MAIN].page_size_bytes


def test_mrv2_block_stride_capability_preserves_generic_backing(monkeypatch):
    config = _make_config()
    attention_spec = AscendMLAAttentionSpec(
        block_size=8, num_kv_heads=1, head_size=4, dtype=torch.bfloat16, indexes_kv_by_block_stride=True
    )
    state_spec = replace(_make_specs()[MAMBA], page_size_padded=attention_spec.page_size_bytes)
    num_blocks = 3
    layer_size = num_blocks * attention_spec.page_size_bytes
    plan = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=2 * layer_size,
                layers=names,
                offset=0,
                layer_stride=layer_size,
                block_stride=attention_spec.page_size_bytes,
            )
            for names in (["attention.0", "attention.1"], ["state.0"])
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(["attention.0", "attention.1"], attention_spec),
            KVCacheGroupSpec(["state.0"], state_spec),
        ],
    )
    monkeypatch.setattr(attn_utils, "get_current_vllm_config", lambda: config)

    raw = attn_utils._allocate_kv_cache(plan, {}, torch.device("cpu"))

    backing = raw["attention.0"].untyped_storage()
    assert backing.nbytes() == 2 * layer_size
    assert all(cache.untyped_storage().data_ptr() == backing.data_ptr() for cache in raw.values())
    assert raw["state.0"].data_ptr() == raw["attention.0"].data_ptr()
    assert raw["attention.1"].data_ptr() - raw["attention.0"].data_ptr() == layer_size

    monkeypatch.setattr(attn_utils, "_get_attention_kv_cache_dims", lambda _name, _spec: (4, 0))
    attn_groups = [
        SimpleNamespace(
            kv_cache_group_id=i,
            kv_cache_spec=group.kv_cache_spec,
            layer_names=group.layer_names,
            backend=_AttentionBackend,
        )
        for i, group in enumerate(plan.kv_cache_groups)
    ]
    views = attn_utils._reshape_kv_cache_v2(attn_groups, raw, "auto", [4, 8], {}, plan)
    cache, rope = views["attention.0"]
    assert cache.shape == (6, 4, 1, 4)
    assert rope.numel() == 0
    # Kernel block 2 starts at scheduler page 1, without a model marker.
    cache[2].fill_(7)
    physical_pages = raw["attention.0"].view(torch.bfloat16).view(num_blocks, 8, 1, 4)
    assert torch.all(physical_pages[1, :4] == 7)
    assert torch.count_nonzero(physical_pages[[0, 2]]) == 0
    assert torch.count_nonzero(physical_pages[1, 4:]) == 0
    assert torch.count_nonzero(raw["attention.1"]) == 0


@pytest.mark.parametrize("geometry", [{"offset": 1}, {"layer_stride": 64}, {"block_stride": 32}, {"size": 1}])
def test_mrv2_rejects_invalid_shared_slot_geometry(monkeypatch, geometry):
    config, _, plan = _make_plan()
    monkeypatch.setattr(attn_utils, "get_current_vllm_config", lambda: config)
    plan.kv_cache_tensors[0] = replace(plan.kv_cache_tensors[0], **geometry)
    with pytest.raises(ValueError, match="shared-slot"):
        attn_utils._allocate_kv_cache(plan, {}, torch.device("cpu"))


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
