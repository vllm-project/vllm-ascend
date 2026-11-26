# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu import model_runner as upstream

from vllm_ascend.patch.worker import patch_model_runner  # noqa: F401


def test_initialize_hisparse_slab_padding_survives_both_runner_copies():
    pytest.importorskip("vllm.v1.hisparse.backend")
    pytest.importorskip("vllm.v1.hisparse.binding")
    from vllm.v1.hisparse import binding
    from vllm.v1.kv_cache_interface import KVCacheGroupSpec, KVCacheLayout, KVCacheTensor, SparseCacheRole

    from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, AscendSFAIndexerCacheSpec
    from vllm_ascend.hisparse.backend import AscendHiSparseBackend
    from vllm_ascend.worker.v2.model_runner import NPUModelRunner

    alignment = 2 * 1024 * 1024
    page = 128 * 128 * torch.bfloat16.itemsize
    stride = 3 * page
    size = 2 * stride
    source = AscendMLAAttentionSpec(block_size=128, num_kv_heads=1, head_size=128, dtype=torch.bfloat16)
    indexer = AscendSFAIndexerCacheSpec(
        block_size=128, num_kv_heads=1, head_size=128, dtype=torch.bfloat16, cache_role=SparseCacheRole.INDEXER
    )
    groups = [
        KVCacheGroupSpec(["source"], source, host_resident=True),
        KVCacheGroupSpec(["indexer"], indexer),
    ]
    tensors = [
        KVCacheTensor(3 * page, ["source"], page, page, host_resident=True),
        KVCacheTensor(size, ["indexer"], page, stride, offset=page),
    ]
    config = KVCacheConfig(2, tensors, groups, hisparse_host_num_blocks=3)
    context = {"indexer": SimpleNamespace()}
    runner = MagicMock()
    runner.device = torch.device("cpu")
    runner.speculator = runner.pcp_manager = None
    runner.is_encoder_decoder = runner.is_last_pp_rank = False
    runner.max_model_len = 256
    runner.vocab_size = 16
    runner.cache_config.kv_sharing_fast_prefill = False
    runner.jit_warmup_registry.activate.side_effect = nullcontext
    runner.compilation_config.static_forward_context = context
    runner.vllm_config.compilation_config = runner.compilation_config
    runner.vllm_config.cache_config.get_resolved_kv_cache_layout.return_value = KVCacheLayout.BLHNC
    runner.vllm_config.aux_output_config.enabled = False
    runner.vllm_config.attention_config.hisparse_config = None
    runner.vllm_config.kv_transfer_config.has_connector = lambda name: name == "AscendStoreConnector"
    runner.ascend_config.sparse_kv_offload_config.enabled = False
    runner._kv_trace = None
    host = torch.empty(3 * page, dtype=torch.int8)
    backing = torch.empty(size + alignment - 1, dtype=torch.int8)
    host_pool = SimpleNamespace(allocate=MagicMock(return_value=host))
    allocate = MagicMock(return_value=backing)

    def init_cache(
        forward_context, core_config, device, kernel_block_sizes, vllm_config, *_args, block_tables, **_kwargs
    ):
        assert core_config is runner.kv_cache_config
        assert core_config is not config and core_config.kv_cache_tensors is not tensors
        assert [tensor.size for tensor in core_config.kv_cache_tensors] == [3 * page, size + alignment - 1]
        return binding.init_hisparse_kv_cache(
            core_config, device, kernel_block_sizes, vllm_config, forward_context, block_tables, allocate=allocate
        )

    def bind_caches(*, kv_caches, **_kwargs):
        (indexer_view,) = kv_caches["indexer"]
        assert indexer_view.untyped_storage().data_ptr() % alignment == 0
        return []

    def initialize_upstream(runner_config, allocation_context):
        assert runner_config is not config
        patch_model_runner.initialize_kv_cache(runner, runner_config, kv_cache_allocation_context=allocation_context)
        assert runner.kv_cache_config is not runner_config

    runner._initialize_kv_cache_upstream.side_effect = initialize_upstream
    with (
        patch.object(binding, "get_hisparse_backend", return_value=AscendHiSparseBackend()),
        patch.object(binding, "get_hisparse_host_pool", return_value=host_pool),
        patch.object(binding, "bind_hisparse_kv_caches", side_effect=bind_caches) as bind,
        patch.object(binding, "initialize_hisparse_runtime_buffers"),
        patch.object(upstream, "init_attn_backend", return_value=([], MagicMock(), [128, 128])),
        patch.object(upstream, "maybe_create_adaptive_verification_manager", return_value=None),
        patch.object(upstream, "BlockTables"),
        patch.object(upstream.pcp, "maybe_build_pcp_manager", return_value=None),
        patch.object(upstream, "maybe_build_ubatch_runner", return_value=None),
        patch.object(upstream, "initialize_mamba_ssu_backend"),
        patch.object(upstream, "has_compiled_submodule", return_value=False),
        patch.object(upstream, "check_attention_cp_compatibility"),
        patch.object(upstream, "init_kv_cache", side_effect=init_cache),
        patch.object(upstream, "get_replayssm_block_copy_tensors", create=True),
        patch.object(upstream, "get_kv_connector") as register,
        patch("vllm_ascend.worker.v2.model_runner.ModelAclGraphManager"),
        patch("vllm_ascend.worker.v2.model_runner.prepare_v41_source_rope"),
        patch("vllm_ascend.worker.v2.model_runner.KVPPRuntime.create_from_kv_cache"),
        patch("vllm_ascend.ops.kda_state_copy_plan.initialize_kda_state_copy"),
    ):
        NPUModelRunner.initialize_kv_cache(runner, config)

    allocate.assert_called_once_with(size + alignment - 1)
    bind.assert_called_once()
    host_pool.allocate.assert_called_once_with(3 * page)
    assert config.kv_cache_tensors is tensors
    assert [tensor.size for tensor in runner.kv_cache_config.kv_cache_tensors] == [3 * page, size]
    views = register.call_args.args[1]
    assert views["source"].data_ptr() == host.data_ptr()
    assert views["source"].shape == (3, 1, 128, 128)
    (view,) = views["indexer"]
    base = backing.data_ptr() + (-backing.data_ptr() % alignment)
    assert view.untyped_storage().data_ptr() == base
    assert view.data_ptr() == base + page
    assert view.storage_offset() == page // view.element_size()
    assert view.shape == (4, 128, 1, 128) and view.stride() == (128 * 128, 128, 128, 1)
    assert context["indexer"].hisparse_block_stride == 3


@pytest.mark.parametrize(
    ("mode", "pcp_size", "expected_num_tokens"),
    [
        (CUDAGraphMode.NONE, 1, 8),
        (CUDAGraphMode.PIECEWISE, 1, 5),
        (CUDAGraphMode.NONE, 2, 5),
    ],
)
def test_dispatch_pads_only_non_pcp_eager(mode, pcp_size, expected_num_tokens):
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mode=mode),
        parallel_config=SimpleNamespace(
            pcp_shard_decode_requests=False,
            prefill_context_parallel_size=pcp_size,
        ),
    )
    manager = SimpleNamespace(vllm_config=config)
    with (
        patch.object(patch_model_runner, "pad_num_tokens", return_value=8) as pad,
        patch.object(
            patch_model_runner,
            "get_ascend_config",
            return_value=SimpleNamespace(enable_shared_expert_dp=True),
        ),
        patch.object(patch_model_runner, "_dispatch_cg_and_sync_dp", return_value=("desc", None)) as parent,
    ):
        assert patch_model_runner.dispatch_cg_and_sync_dp(manager, 2, 5, None, 1, 0) == ("desc", None)

    assert parent.call_args.args[:4] == (manager, 2, expected_num_tokens, None)
    if mode == CUDAGraphMode.NONE and pcp_size == 1:
        pad.assert_called_once_with(5, config, True)
    else:
        pad.assert_not_called()


def test_dispatch_preserves_sharded_pcp_request_count():
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE),
        parallel_config=SimpleNamespace(
            pcp_shard_decode_requests=True,
            prefill_context_parallel_size=2,
        ),
    )
    manager = SimpleNamespace(vllm_config=config)
    with (
        patch.object(patch_model_runner, "pad_num_tokens") as pad,
        patch.object(patch_model_runner, "_dispatch_cg_and_sync_dp", return_value=("desc", None)) as parent,
    ):
        patch_model_runner.dispatch_cg_and_sync_dp(manager, 16, 6, 2, 1, 0)

    assert parent.call_args.args[:4] == (manager, 3, 6, 2)
    pad.assert_not_called()


@pytest.mark.parametrize("is_profiling", [False, True])
@pytest.mark.parametrize("has_trackers", [False, True])
def test_initialize_preserves_connector_containers_and_flattens_runner_cache(is_profiling, has_trackers):
    k, v, conv, ssm, single = [torch.empty(3, 4) for _ in range(5)]
    other = torch.empty(3, 4, device="meta")
    caches = {"attention": (k, None, v), "mamba": [conv, ssm], "single": single, "other": other}
    runner = MagicMock()
    runner.device = torch.device("cpu")
    runner.is_encoder_decoder = False
    runner.speculator = None
    runner.vocab_size = 16
    runner.max_model_len = 128
    runner.cache_config.kv_sharing_fast_prefill = False
    # MagicMock makes aux_output_config.enabled truthy by default; disable it so
    # the test exercises the KV-connector path only (main-only branch).
    runner.vllm_config.aux_output_config.enabled = False
    runner.jit_warmup_registry.activate.side_effect = nullcontext
    stale_trackers = [torch.empty(3, dtype=torch.int32)]
    runner.replayssm_block_copy_tensors = stale_trackers
    trackers = [torch.zeros(3, dtype=torch.int32), torch.ones(3, dtype=torch.int32)] if has_trackers else []
    config = KVCacheConfig(num_blocks=3, kv_cache_tensors=[], kv_cache_groups=[])
    persistent_tables = MagicMock(name="persistent_block_tables")
    registration_order = []
    runner._register_sparse_kv_caches.side_effect = lambda _: registration_order.append("offload")

    def allocate_caches(*_args, **_kwargs):
        registration_order.append("allocate")
        return caches

    def initialize_model_state(resolved_config, block_tables):
        # Align-mode Mamba copies index persistent request slots. Passing the
        # batch-ordered input tables here would silently corrupt reordered
        # requests, while skipping this hook leaves the first request unready.
        assert resolved_config is runner.kv_cache_config
        assert block_tables is persistent_tables
        assert block_tables is not persistent_tables.input_block_tables
        assert [id(tensor) for tensor in runner.kv_caches] == [id(tensor) for tensor in (k, v, conv, ssm, single)]
        registration_order.append("model_state")

    runner.model_state.initialize_kv_cache.side_effect = initialize_model_state

    def register_connector(*_args):
        registration_order.append("connector")

    def collect_initialized_trackers(*_args):
        # Model-state initialization can replace the tracker tensors.
        assert registration_order[-1] == "model_state"
        return trackers

    tracker_context = patch.object(
        upstream, "get_replayssm_block_copy_tensors", side_effect=collect_initialized_trackers
    )
    with (
        patch.object(upstream, "init_attn_backend", return_value=([], MagicMock(), [])),
        patch.object(upstream, "maybe_create_adaptive_verification_manager", return_value=None),
        patch.object(upstream, "BlockTables", return_value=persistent_tables),
        patch.object(upstream.pcp, "maybe_build_pcp_manager", return_value=None),
        patch.object(upstream, "maybe_build_ubatch_runner", return_value=None),
        patch.object(upstream, "initialize_mamba_ssu_backend"),
        patch.object(upstream, "has_compiled_submodule", return_value=False),
        patch.object(upstream, "ModelCudaGraphManager"),
        patch.object(upstream, "check_attention_cp_compatibility"),
        patch.object(upstream, "init_kv_cache", side_effect=allocate_caches),
        tracker_context as collect_trackers,
        patch.object(upstream, "get_kv_connector", side_effect=register_connector) as connector,
    ):
        upstream.GPUModelRunner.initialize_kv_cache(runner, config, is_profiling=is_profiling)

    assert [id(tensor) for tensor in runner.kv_caches] == [id(tensor) for tensor in (k, v, conv, ssm, single)]
    collect_trackers.assert_called_once_with(runner.compilation_config.static_forward_context)
    assert runner.replayssm_block_copy_tensors is trackers
    assert type(runner.replayssm_block_copy_tensors) is list
    runner._register_sparse_kv_caches.assert_called_once_with(caches)
    runner.model_state.initialize_kv_cache.assert_called_once_with(runner.kv_cache_config, persistent_tables)
    if is_profiling:
        assert registration_order == ["allocate", "offload", "model_state"]
        connector.assert_not_called()
        assert runner.kv_connector is upstream.NO_OP_KV_CONNECTOR
    else:
        assert registration_order == ["allocate", "offload", "model_state", "connector"]
        assert connector.call_args.args[1] is caches
        assert type(caches["attention"]) is tuple
        assert type(caches["mamba"]) is list
        assert caches["attention"][2] is v
        assert caches["mamba"][1] is ssm
        assert caches["other"] is other


def test_initialize_builds_aux_output_connector_when_enabled():
    """The vendored initializer keeps upstream's AuxOutput connector wiring."""
    caches = {"attention": torch.empty(3, 4)}
    runner = MagicMock()
    runner.device = torch.device("cpu")
    runner.is_encoder_decoder = False
    runner.speculator = None
    runner.vocab_size = 16
    runner.max_model_len = 128
    runner.cache_config.kv_sharing_fast_prefill = False
    # Mirror of the case above: exercise the AuxOutput branch only.
    runner.vllm_config.aux_output_config.enabled = True
    connector = MagicMock()
    runner.aux_output_connector = None
    runner.jit_warmup_registry.activate.side_effect = nullcontext
    config = KVCacheConfig(num_blocks=3, kv_cache_tensors=[], kv_cache_groups=[])

    with (
        patch.object(upstream, "init_attn_backend", return_value=([], MagicMock(), [])),
        patch.object(upstream, "maybe_create_adaptive_verification_manager", return_value=None),
        patch.object(upstream, "BlockTables"),
        patch.object(upstream.pcp, "maybe_build_pcp_manager", return_value=None),
        patch.object(upstream, "maybe_build_ubatch_runner", return_value=None),
        patch.object(upstream, "initialize_mamba_ssu_backend"),
        patch.object(upstream, "has_compiled_submodule", return_value=False),
        patch.object(upstream, "ModelCudaGraphManager"),
        patch.object(upstream, "check_attention_cp_compatibility"),
        patch.object(upstream, "init_kv_cache", return_value=caches),
        patch.object(upstream, "get_kv_connector"),
        patch.object(upstream, "get_aux_output_connector", return_value=connector) as get_aux,
    ):
        upstream.GPUModelRunner.initialize_kv_cache(runner, config, is_profiling=False)

    # The initializer deep-copies the config, so assert against the resolved one
    # (identity shortcut) instead of comparing the original object by value.
    # Main creates the connector without the KV-cache config and
    # initializes its storage once the layout is resolved.
    get_aux.assert_called_once_with(runner.model, runner.vllm_config)
    connector.initialize_output_storage.assert_called_once_with(runner.kv_cache_config)
    assert runner.aux_output_connector is connector


def test_mrv2_block_copy_preserves_segmented_mamba_storage():
    storage = torch.arange(36, dtype=torch.float32)
    conv = storage[:12].view(3, 4)
    ssm = storage[12:].view(3, 8)
    before_conv, before_ssm = conv.clone(), ssm.clone()
    ring_start = torch.tensor([10, 20, 30], dtype=torch.int32)
    num_committed = torch.tensor([1, 2, 3], dtype=torch.int32)
    upstream.copy_kv_cache_blocks_inplace(
        [conv, ssm, conv, ssm, ring_start, num_committed], 3, [KVCacheBlockCopy(src_block_id=0, dst_block_id=2)]
    )
    torch.testing.assert_close(conv[2], before_conv[0])
    torch.testing.assert_close(ssm[2], before_ssm[0])
    torch.testing.assert_close(conv[:2], before_conv[:2])
    torch.testing.assert_close(ssm[:2], before_ssm[:2])
    torch.testing.assert_close(ring_start, torch.tensor([10, 20, 10], dtype=torch.int32))
    torch.testing.assert_close(num_committed, torch.tensor([1, 2, 1], dtype=torch.int32))
