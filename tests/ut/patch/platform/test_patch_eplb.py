# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from vllm.config import EPLBConfig, ParallelConfig, VllmConfig
from vllm.config import parallel as parallel_module
from vllm.platforms import current_platform

from vllm_ascend.patch.platform import patch_eplb


class _FakeNpuPlatform:
    device_type = "npu"

    def __getattr__(self, name):
        return getattr(current_platform, name)


@contextmanager
def _npu_parallel_config_platform():
    proxy = parallel_module.current_platform
    assert isinstance(proxy, patch_eplb._CudaAlikeEplbPlatformProxy)
    original_platform = proxy._platform
    proxy._platform = _FakeNpuPlatform()
    try:
        yield
    finally:
        proxy._platform = original_platform


def test_parallel_and_vllm_config_keep_upstream_validation():
    with (
        _npu_parallel_config_platform(),
        patch("vllm_ascend.logger.configure_ascend_file_logging"),
        patch("vllm_ascend.logger.configure_ascend_logging"),
        patch("vllm.distributed.nixl_utils.is_nixl_available", return_value=False),
    ):
        parallel_config = ParallelConfig(
            tensor_parallel_size=2,
            enable_expert_parallel=True,
            enable_eplb=True,
            eplb_config=EPLBConfig(use_async=True),
        )
        vllm_config = VllmConfig(parallel_config=parallel_config)

    assert vllm_config.parallel_config.enable_eplb
    assert vllm_config.parallel_config.eplb_config.communicator == "torch_gloo"


def test_parallel_config_keeps_upstream_nixl_auto_selection():
    with (
        _npu_parallel_config_platform(),
        patch(
            "vllm.distributed.nixl_utils.is_nixl_available",
            return_value=True,
        ) as is_nixl_available,
    ):
        parallel_config = ParallelConfig(
            tensor_parallel_size=2,
            enable_expert_parallel=True,
            enable_eplb=True,
            eplb_config=EPLBConfig(use_async=True),
        )

    assert parallel_config.eplb_config.communicator == "nixl"
    is_nixl_available.assert_called_once_with()


def test_parallel_config_platform_patch_is_idempotent():
    proxy = parallel_module.current_platform

    patch_eplb._patch_parallel_config()

    assert parallel_module.current_platform is proxy


def test_communicator_factory_creates_ascend_gloo_communicator(monkeypatch):
    communicator = object()
    gloo_cls = MagicMock(return_value=communicator)
    monkeypatch.setattr(patch_eplb, "AscendGlooEplbCommunicator", gloo_cls)
    coordinator = MagicMock()

    result = patch_eplb._eplb_communicator.create_eplb_communicator(
        coordinator,
        "torch_gloo",
        [[object()]],
        [object()],
    )

    assert result is communicator
    gloo_cls.assert_called_once_with(cpu_group=coordinator.cpu_group)


def test_communicator_factory_accepts_additive_parameters(monkeypatch):
    communicator = object()
    gloo_cls = MagicMock(return_value=communicator)
    monkeypatch.setattr(patch_eplb, "AscendGlooEplbCommunicator", gloo_cls)

    def original_factory(
        group_coordinator,
        backend,
        expert_weights,
        expert_buffer,
        *,
        transport_options=None,
    ):
        raise AssertionError("The upstream factory should not be called on Ascend.")

    wrapped_factory = patch_eplb._wrap_communicator_factory(original_factory)
    coordinator = MagicMock()
    result = wrapped_factory(
        group_coordinator=coordinator,
        backend="torch_gloo",
        expert_weights=[[object()]],
        expert_buffer=[object()],
        transport_options={"mode": "future"},
    )

    assert result is communicator
    gloo_cls.assert_called_once_with(cpu_group=coordinator.cpu_group)


def test_communicator_factory_requires_group_coordinator_parameter():
    def original_factory(backend, expert_weights, expert_buffer):
        raise AssertionError("The upstream factory should not be called on Ascend.")

    with pytest.raises(RuntimeError, match="group_coordinator"):
        patch_eplb._wrap_communicator_factory(original_factory)


def test_async_workspace_wrapper_refreshes_committed_layer(monkeypatch):
    call_order: list[str] = []
    consumed_event = MagicMock()
    consumed_event.record.side_effect = lambda _stream=None: call_order.append("ack")
    pending_result = SimpleNamespace(
        layer_idx=3,
        transfer_metadata=object(),
        consumed_event=consumed_event,
    )
    model_state = SimpleNamespace(
        pending_result=pending_result,
        rebalanced=True,
        model=SimpleNamespace(num_moe_layers=4),
        model_name="model",
    )
    refresh = MagicMock(side_effect=lambda *_args: call_order.append("refresh"))
    monkeypatch.setattr(patch_eplb, "refresh_model_routing_tables", refresh)
    log_info = MagicMock()
    monkeypatch.setattr(patch_eplb.logger, "info", log_info)

    def original_move(model_state, ep_rank, *, future_option=None):
        assert ep_rank == 0
        assert future_option == "future"
        call_order.append("move")
        model_state.pending_result.consumed_event.record()
        model_state.pending_result = None
        return "moved"

    wrapped_move = patch_eplb._wrap_move_to_workspace(original_move)
    result = wrapped_move(model_state, 0, future_option="future")

    assert result == "moved"
    refresh.assert_called_once_with(model_state, 3)
    log_info.assert_called_once_with(
        "%s: model=%s",
        patch_eplb.ASYNC_EPLB_CYCLE_COMMITTED_LOG,
        "model",
    )
    assert call_order == ["move", "refresh", "ack"]
