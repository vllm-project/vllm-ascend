# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Unit tests for ``kernel_warmup`` orchestration."""

import ast
import importlib
import inspect
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from tests.ut.model_executor.warmup.helpers import (
    make_mock_vllm_config,
    make_registry,
)
from vllm_ascend.ops.triton import rms_norm
from vllm_ascend.ops.triton.dcp import dcp_warmup
from vllm_ascend.ops.triton.dcp.dcp_jit import DcpJitKernel

# The package __init__ re-exports the function of the same name, so resolve
# the module explicitly.
kw = importlib.import_module("vllm_ascend.model_executor.warmup.kernel_warmup")


def _make_worker(*, enable_jit_warmup=True, speculative_config=None):
    config = make_mock_vllm_config(
        enable_jit_warmup=enable_jit_warmup,
        speculative_config=speculative_config,
    )
    registry = make_registry(config)
    worker = SimpleNamespace(
        vllm_config=config,
        model_runner=SimpleNamespace(
            vllm_config=config,
            jit_warmup_registry=registry,
            drafter=None,
            input_batch=None,
        ),
    )
    return worker, registry


@patch.object(kw, "HAS_TRITON", False)
def test_skips_without_triton():
    worker, registry = _make_worker()
    registry.warmup = MagicMock()
    kw.kernel_warmup(worker)
    registry.warmup.assert_not_called()


@patch.object(kw, "HAS_TRITON", True)
def test_skips_when_jit_warmup_disabled():
    worker, registry = _make_worker(enable_jit_warmup=False)
    registry.warmup = MagicMock()
    kw.kernel_warmup(worker)
    registry.warmup.assert_not_called()


@patch.object(kw, "HAS_TRITON", True)
@pytest.mark.parametrize("enabled", [False, True])
def test_indexer_cache_registration_precedes_compilation(enabled):
    from vllm_ascend.attention.indexer import AscendSFAIndexerBackend
    from vllm_ascend.ops.triton import indexer_quant_cache_int8 as int8_ops

    worker, registry = _make_worker(enable_jit_warmup=enabled)
    indexer = MagicMock(spec=AscendSFAIndexerBackend)
    indexer.register_k_cache_warmup.side_effect = lambda: int8_ops._INDEXER_QUANT_CACHE_INT8_KERNEL.register_warmup(
        num_slots=1024,
        page_size=128,
        key_page_stride=16384,
        scale_page_stride=128,
    )
    worker.vllm_config.compilation_config.static_forward_context = {
        "attention": SimpleNamespace(impl=SimpleNamespace(indexer=indexer)),
        "unrelated": SimpleNamespace(),
    }
    registry.warmup = MagicMock(
        side_effect=lambda: int8_ops._INDEXER_QUANT_CACHE_INT8_KERNEL in registry._registrations or pytest.fail()
    )
    with patch.object(kw, "_register_dspark_swa_indices_warmup", return_value=0):
        kw.kernel_warmup(worker)
    assert indexer.register_k_cache_warmup.call_count == int(enabled)
    assert registry.warmup.call_count == int(enabled)


@patch.object(kw, "HAS_TRITON", True)
def test_prunes_non_ascend_registrations_and_warms():
    worker, registry = _make_worker()
    # One real Ascend registration...
    with registry.activate():
        rms_norm._TRITON_Q_RMS_KERNEL.register_warmup(worker.vllm_config)
    # ...and one foreign registration, as the upstream base RejectionSampler
    # leaves behind on NPU.
    registry._registrations[object()] = [((), {})]
    assert len(registry) == 2

    registry.warmup = MagicMock()
    kw.kernel_warmup(worker)

    assert len(registry) == 1
    assert rms_norm._TRITON_Q_RMS_KERNEL in registry._registrations
    registry.warmup.assert_called_once()


def _make_dspark_builder(*, device_metadata=True):
    """A (group, builder) pair shaped like the DSA draft groups."""
    from vllm_ascend.attention.dsa_v1 import AscendDSAMetadataBuilder

    builder = MagicMock()
    builder.__class__ = AscendDSAMetadataBuilder  # isinstance() passes
    builder._device_metadata_enabled = device_metadata
    builder._dspark_index_width = 4608
    builder.storage_block_size = 128
    builder.model_config.hf_config.sliding_window = 4096
    builder.speculative_config.num_speculative_tokens = 3
    group = MagicMock()
    group.get_metadata_builder.return_value = builder
    group.kv_cache_group_id = 0
    return group, builder


@patch(
    "vllm_ascend.ops.triton.dspark_swa_indices.register_dspark_swa_indices_warmup",
    return_value=True,
)
@patch.object(kw, "HAS_TRITON", True)
def test_dspark_walker_v2_speculator_path_registers_nested_groups(mock_register):
    """The v2 runner keeps draft groups on the speculator (possibly nested);
    the walker must find them and register one key per DSA group."""
    worker, _ = _make_worker()
    group, _ = _make_dspark_builder()
    nested_group, _ = _make_dspark_builder()
    worker.model_runner.speculator = SimpleNamespace(attn_groups=[[group], nested_group])
    block_table = MagicMock()
    block_table.get_device_tensor.return_value.shape = [0, 1024]
    worker.model_runner.input_batch = SimpleNamespace(block_table=[block_table])

    assert kw._register_dspark_swa_indices_warmup(worker) == 2
    assert mock_register.call_count == 2
    assert mock_register.call_args_list[0].kwargs["block_table_width"] == 1024
    assert mock_register.call_args_list[0].kwargs["index_width"] == 4608


@patch("vllm_ascend.ops.triton.dspark_swa_indices.register_dspark_swa_indices_warmup")
@patch.object(kw, "HAS_TRITON", True)
def test_dspark_walker_skips_eager_builders(mock_register):
    """Eager-path builders compile lazily at first call; no pre-capture key."""
    worker, _ = _make_worker()
    group, _ = _make_dspark_builder(device_metadata=False)
    worker.model_runner.speculator = SimpleNamespace(attn_groups=[group])

    assert kw._register_dspark_swa_indices_warmup(worker) == 0
    mock_register.assert_not_called()


def test_v2_initialize_kv_cache_upstream_holds_registry_window():
    """Builder construction inside upstream initialize_kv_cache registers
    into the registry; a runner without one (tests, partial mocks) still
    works."""
    # The v2 runner's base is the MRv2 GPUModelRunner, a different class
    # from the v1-path vllm.v1.worker.gpu_model_runner.GPUModelRunner.
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner as V2BaseRunner

    from vllm_ascend.worker.v2.model_runner import NPUModelRunner as V2Runner

    config = make_mock_vllm_config()
    registry = make_registry(config)
    runner = V2Runner.__new__(V2Runner)
    runner.jit_warmup_registry = registry

    with patch.object(V2BaseRunner, "initialize_kv_cache") as super_init:
        # init_attn_backend constructs the DSA builder, which registers here.
        super_init.side_effect = lambda *a, **k: rms_norm._TRITON_Q_RMS_KERNEL.register_warmup(config)
        runner._initialize_kv_cache_upstream(MagicMock(), None)
    super_init.assert_called_once()
    assert len(registry) == 1

    bare = V2Runner.__new__(V2Runner)  # no jit_warmup_registry attribute
    with patch.object(V2BaseRunner, "initialize_kv_cache") as super_init:
        bare._initialize_kv_cache_upstream(MagicMock(), None)
    super_init.assert_called_once()


@pytest.fixture(autouse=True)
def dcp_kernel_signatures(monkeypatch):
    # The no-Triton shim returns plain functions; emulate only JIT signature metadata.
    for owner in dcp_warmup._DCP_KERNELS:
        if not hasattr(owner.kernel, "arg_names"):
            monkeypatch.setattr(
                owner.kernel, "arg_names", list(inspect.signature(owner.kernel).parameters), raising=False
            )


def _register_dcp_warmup(**kwargs):
    registry = make_registry()
    with (
        registry.activate(),
        patch.object(dcp_warmup, "init_device_properties_triton"),
        patch.object(dcp_warmup, "get_vectorcore_num", return_value=24),
    ):
        dcp_warmup.register_dcp_warmup(**kwargs)
    return registry


def _dcp_warmup_keys(registry):
    return {
        owner: {key for args, kwargs in registrations for key in owner.get_warmup_keys(*args, **kwargs)}
        for owner, registrations in registry._registrations.items()
    }


@pytest.mark.parametrize(
    "dtype,local_heads,head_dim,scatter_size,pcp_size,from_stats,mla,max_tokens",
    [
        (torch.bfloat16, 8, 512, 8, 1, True, False, 4097),
        (torch.bfloat16, 12, 512, 8, 1, True, False, 257),
        (torch.bfloat16, 8, 256, 4, 2, True, False, 257),
        (torch.float16, 8, 256, 4, 1, True, False, 257),
        (torch.float32, 8, 256, 4, 1, False, True, 257),
        (torch.bfloat16, 8, 512, 8, 1, False, True, 4097),
        (torch.bfloat16, 128, 512, 2, 1, True, False, 33),
        (torch.bfloat16, 1, 2, 2, 1, False, False, 33),
    ],
)
def test_finite_warmup_covers_every_token_count(
    dtype, local_heads, head_dim, scatter_size, pcp_size, from_stats, mla, max_tokens
):
    # Exhaustive token enumeration belongs in the test, never startup. This
    # exercises non-capture sizes, threshold neighbours, scalar residues,
    # native T1, full/tiled pack, all stats tiles and MLA's local contribution.
    registry = _register_dcp_warmup(
        dtype=dtype,
        local_heads=local_heads,
        head_dim=head_dim,
        scatter_size=scatter_size,
        pcp_size=pcp_size,
        max_tokens=max_tokens,
        from_stats=from_stats,
        mla=mla,
    )
    registered = _dcp_warmup_keys(registry)
    owners = {owner.kernel: owner for owner in dcp_warmup._DCP_KERNELS}
    exhaustive = defaultdict(set)

    def record(kernel, grid, *args, **kwargs):
        owner = owners[kernel]
        key = owner.key_from_launch(*args, **kwargs)
        assert key in registered[owner]
        exhaustive[owner].add(key)

    layouts = ("token_major", "head_major_lse", "head_major") if mla else ("token_major",)
    for tokens in range(1, max_tokens + 1):
        for layout in layouts:
            dcp_warmup._record_dcp_layout(
                dtype, tokens, local_heads, head_dim, scatter_size, pcp_size, from_stats, mla, layout, 24, record
            )
    assert dict(exhaustive) == registered


def test_registration_is_scoped_deduplicated_and_allocates_nothing():
    kwargs = dict(
        dtype=torch.bfloat16,
        local_heads=8,
        head_dim=512,
        scatter_size=8,
        pcp_size=1,
        max_tokens=4097,
        from_stats=False,
        mla=True,
    )
    registry = make_registry()
    with (
        patch.object(dcp_warmup, "init_device_properties_triton"),
        patch.object(dcp_warmup, "get_vectorcore_num", return_value=24),
        patch.object(torch, "empty", side_effect=AssertionError("warmup must not allocate payloads")),
        patch.object(torch.distributed, "all_to_all_single", side_effect=AssertionError("warmup must not communicate")),
    ):
        dcp_warmup.register_dcp_warmup(**kwargs)
        assert len(registry) == 0
        with registry.activate():
            dcp_warmup.register_dcp_warmup(**kwargs)
            dcp_warmup.register_dcp_warmup(**kwargs)
    assert len(registry) == 8
    assert all(len(registrations) == 1 for registrations in registry._registrations.values())
    for original, keys in _dcp_warmup_keys(registry).items():
        kernel = MagicMock()
        kernel.arg_names = original.kernel.arg_names
        owner = DcpJitKernel(kernel, original.dynamic_scalars)
        with patch.object(torch, "empty", side_effect=AssertionError("compile must remain metadata-only")):
            for key in keys:
                owner.compile(key)
        assert kernel.warmup.call_count == len(keys)
        kernel.__getitem__.assert_not_called()
        assert all(call.kwargs["grid"] == (1,) for call in kernel.warmup.call_args_list)


def test_merge_rank_stride_is_runtime_and_offset_alignment_is_covered():
    registry = _register_dcp_warmup(
        dtype=torch.bfloat16,
        local_heads=1,
        head_dim=2,
        scatter_size=2,
        pcp_size=1,
        max_tokens=33,
        from_stats=True,
    )
    owner = next(owner for owner in registry._registrations if owner.kernel is dcp_warmup._fused_merge_kernel)
    inputs = [dict(key.inputs) for key in _dcp_warmup_keys(registry)[owner]]
    assert {case["lse_ptr"].aligned for case in inputs} == {False, True}
    assert {case["lse_stride_rank"] for case in inputs} == {2}
    assert {case["NATIVE_LAYOUT"] for case in inputs} == {False, True}
    # The no-Triton shim drops JIT metadata; inspect the actual declaration.
    kernel = dcp_warmup._fused_merge_kernel
    source = ast.parse(inspect.getsource(getattr(kernel, "fn", kernel)))
    function = source.body[0]
    assert isinstance(function, ast.FunctionDef)
    jit = function.decorator_list[0]
    assert isinstance(jit, ast.Call)
    do_not_specialize = next(ast.literal_eval(kw.value) for kw in jit.keywords if kw.arg == "do_not_specialize")
    assert "lse_stride_rank" in do_not_specialize


def test_dcp_target_and_draft_owners_register_before_compilation():
    from vllm_ascend.attention.context_parallel.mla_cp import AscendMlaDCPImpl
    from vllm_ascend.attention.context_parallel.sfa_cp import AscendSFADCPImpl

    worker, registry = _make_worker()
    target = object.__new__(AscendSFADCPImpl)
    draft = object.__new__(AscendMlaDCPImpl)
    for impl, dtype in ((target, torch.bfloat16), (draft, torch.float16)):
        impl.vllm_config = make_mock_vllm_config(dtype=dtype)
        impl.dcp_size = 8
        impl.pcp_group = SimpleNamespace(world_size=2)
        impl.tp_group = SimpleNamespace(world_size=4)
        impl.local_num_heads = impl.num_heads = 12
        impl.kv_lora_rank = 512
    target.enable_sparse_sfa_turboquant = False
    worker.vllm_config.compilation_config.static_forward_context = {
        "target": SimpleNamespace(impl=target),
        "same_target_layout": SimpleNamespace(impl=target),
        "draft": SimpleNamespace(impl=draft),
        "other": SimpleNamespace(impl=object()),
    }
    registered = []

    def register(**kwargs):
        registered.append(kwargs)
        dcp_warmup._DCP_KERNELS[0].register_warmup(())

    registry.warmup = MagicMock(side_effect=lambda: len(registered) == 2 or pytest.fail())
    with (
        patch.object(kw, "HAS_TRITON", True),
        patch.object(kw, "_register_dspark_swa_indices_warmup", return_value=0),
        patch.object(dcp_warmup, "register_dcp_warmup", side_effect=register),
    ):
        kw.kernel_warmup(worker)
    assert len(registered) == 2
    registry.warmup.assert_called_once()
    assert [(case["dtype"], case["mla"], case["from_stats"]) for case in registered] == [
        (torch.bfloat16, False, True),
        (torch.float16, True, False),
    ]
    assert all(case["scatter_size"] == 4 and case["pcp_size"] == 2 for case in registered)
