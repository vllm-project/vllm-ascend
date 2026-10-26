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
def test_missing_kernel_config_defaults_to_enabled():
    # Older vllm pins expose no kernel_config; default is enabled.
    worker, registry = _make_worker()
    del worker.vllm_config.kernel_config
    registry.warmup = MagicMock()
    with patch.object(kw.logger, "warning") as mock_warning:
        kw.kernel_warmup(worker)
    # Empty registry: warn and skip the compile pass.
    mock_warning.assert_called_once()
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


@patch.object(kw, "HAS_TRITON", True)
def test_dspark_walker_without_drafter_registers_nothing():
    worker, _ = _make_worker()
    assert kw._register_dspark_swa_indices_warmup(worker) == 0


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
