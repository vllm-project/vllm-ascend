# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from contextlib import AbstractContextManager
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend._310p.model_runner_310p import NPUModelRunner310
from vllm_ascend.utils import vllm_version_is
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
from vllm_ascend.worker.worker import NPUWorker


class _AllocationScope(AbstractContextManager):
    def __init__(self) -> None:
        self.active = False

    def __enter__(self):
        assert not self.active
        self.active = True
        return self

    def __exit__(self, *args: Any) -> None:
        assert self.active
        self.active = False


def test_mrv1_kv_pool_only_wraps_backing_allocation() -> None:
    scope = _AllocationScope()
    raw_tensors = {"layer": object()}
    reshaped = {"layer": object()}
    runner = object.__new__(NPUModelRunner)
    runner.shared_kv_cache_layers = {}
    runner.kv_caches = []
    runner.model_config = SimpleNamespace(hf_text_config=SimpleNamespace(model_type="llama"))
    runner.compilation_config = SimpleNamespace(static_forward_context={})

    def allocate(_kv_cache_config):
        assert scope.active
        return raw_tensors

    def reshape(_kv_cache_config, tensors):
        assert not scope.active
        assert tensors is raw_tensors
        return reshaped

    def bind(*_args, **_kwargs):
        assert not scope.active

    runner._allocate_kv_cache_tensors = allocate
    runner._reshape_kv_cache_tensors = reshape

    with patch("vllm.v1.worker.utils.bind_kv_cache", side_effect=bind):
        result = NPUModelRunner.initialize_kv_cache_tensors(
            runner,
            SimpleNamespace(),
            kv_cache_allocation_context=scope,
        )

    assert result is reshaped
    assert not scope.active


def test_310p_kv_pool_only_wraps_backing_allocation() -> None:
    scope = _AllocationScope()
    caches = {"layer": object()}
    runner = object.__new__(NPUModelRunner310)
    runner.vllm_config = SimpleNamespace(kv_transfer_config=None)
    runner.use_sparse = False
    runner.model_config = SimpleNamespace(use_mla=False)
    runner.shared_kv_cache_layers = {}
    runner.kv_caches = []
    runner.compilation_config = SimpleNamespace(static_forward_context={})

    def allocate(_kv_cache_config):
        assert scope.active
        return caches

    def bind(*_args, **_kwargs):
        assert not scope.active

    runner._allocate_kv_cache_tensors = allocate

    with patch("vllm.v1.worker.utils.bind_kv_cache", side_effect=bind):
        result = NPUModelRunner310.initialize_kv_cache_tensors(
            runner,
            SimpleNamespace(),
            kv_cache_allocation_context=scope,
        )

    assert result is caches
    assert not scope.active


@pytest.mark.skipif(
    vllm_version_is("0.28.0"),
    reason="v0.28.0 still reconstructs metadata via post_kv_cache_wake_up",
)
def test_kv_wake_does_not_run_model_runner_recovery() -> None:
    model = torch.nn.Module()
    model.register_buffer("_k_scale", torch.tensor(0.5))
    model.register_buffer("_v_scale", torch.tensor(0.25))

    class Runner:
        def __init__(self) -> None:
            self.model = model
            self.recovery_calls = 0

        def post_kv_cache_wake_up(self) -> None:
            self.recovery_calls += 1
            self.model.get_buffer("_k_scale").fill_(1.0)
            self.model.get_buffer("_v_scale").fill_(1.0)

    runner = Runner()
    with patch.object(NPUWorker, "__init__", lambda *_args, **_kwargs: None):
        worker = NPUWorker()
    worker.model_runner = runner
    worker._sleep_saved_buffers = {}
    worker.sleep_wakeup_manager = MagicMock()

    with (
        patch("vllm_ascend.worker.worker.CaMemAllocator") as mock_allocator_class,
        patch("vllm_ascend.worker.worker.get_ascend_config") as mock_get_config,
    ):
        mock_get_config.return_value = SimpleNamespace(
            weight_nz_mode=0,
            rl_config=SimpleNamespace(enabled=False, sleep_mode_extra_cleanup=False),
        )
        mock_allocator_class.get_instance.return_value = MagicMock()
        NPUWorker.wake_up(worker, tags=["kv_cache"])

    assert runner.recovery_calls == 0
    assert model.get_buffer("_k_scale").item() == 0.5
    assert model.get_buffer("_v_scale").item() == 0.25
