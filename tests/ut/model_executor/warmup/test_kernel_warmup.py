# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Unit tests for ``kernel_warmup`` orchestration."""

import importlib
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from tests.ut.model_executor.warmup.helpers import (
    make_mock_spec_config,
    make_mock_vllm_config,
    make_registry,
)
from vllm_ascend.ops.triton import rms_norm

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


@patch.object(kw, "HAS_TRITON", True)
def test_register_v1_sampling_warmup_registers_four_kernel_families():
    from tests.ut.model_executor.warmup.helpers import make_mock_ascend_config
    from vllm_ascend.ops.triton import bincount, penalty, reject_sample

    spec = make_mock_spec_config(num_speculative_tokens=3)
    config = make_mock_vllm_config(speculative_config=spec)
    registry = make_registry(config)

    with (
        patch.object(penalty, "get_tensor_model_parallel_world_size", return_value=1),
        patch.object(bincount, "get_tensor_model_parallel_world_size", return_value=1),
        patch.object(reject_sample, "get_vectorcore_num", return_value=4),
        patch(
            "vllm_ascend.ascend_config.get_ascend_config",
            return_value=make_mock_ascend_config(),
        ),
        patch("vllm.distributed.get_tensor_model_parallel_world_size", return_value=1),
    ):
        kw.register_v1_sampling_warmup(config)
        assert len(registry) == 0  # outside a window: no-op

        with registry.activate():
            kw.register_v1_sampling_warmup(config)
    # penalties + bincount + prepare_inputs + six rejection wrappers.
    assert len(registry) == 9
