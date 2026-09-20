# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest
import torch
from vllm.config import EPLBConfig, ParallelConfig, VllmConfig
from vllm.config import parallel as parallel_module
from vllm.distributed.eplb import eplb_state as upstream_eplb_state
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
        layer_idx=1,
        transfer_metadata=object(),
        consumed_event=consumed_event,
        is_last_result=True,
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
    refresh.assert_called_once_with(model_state, 1)
    log_info.assert_called_once_with(
        "%s: model=%s",
        patch_eplb.ASYNC_EPLB_CYCLE_COMMITTED_LOG,
        "model",
    )
    assert call_order == ["move", "refresh", "ack"]


def test_distributed_initial_expert_map_spreads_redundant_slots():
    mapping = patch_eplb._build_distributed_initial_expert_map(8, 4, 4)

    assert mapping == [0, 1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 0]
    for rank in range(4):
        local_mapping = mapping[rank * 3 : (rank + 1) * 3]
        assert len(local_mapping) == 3


def test_distributed_initial_expert_map_validates_divisibility():
    with pytest.raises(ValueError, match="divisible by ep_size"):
        patch_eplb._build_distributed_initial_expert_map(8, 3, 4)


def test_get_changed_layer_indices():
    old_mapping = torch.tensor([[0, 1], [2, 3], [4, 5]])
    new_mapping = torch.tensor([[0, 1], [3, 2], [4, 5]])

    assert patch_eplb._get_changed_layer_indices(old_mapping, new_mapping) == [1]


def test_backported_noop_result_finishes_cycle():
    consumed_event = MagicMock()
    pending_result = patch_eplb._BackportedAsyncEplbLayerResult(
        layer_idx=None,
        new_physical_to_logical_map=None,
        transfer_metadata=None,
        consumed_event=consumed_event,
        is_last_result=True,
    )
    model_state = SimpleNamespace(
        pending_result=pending_result,
        rebalanced=True,
    )
    owner = SimpleNamespace(
        model_states={"model": model_state},
        _async_cycle_in_progress=True,
        expert_rearrangement_step=50,
    )
    model_state._ascend_eplb_owner = owner

    patch_eplb._backported_move_to_workspace(model_state, ep_rank=0)

    assert model_state.pending_result is None
    assert not model_state.rebalanced
    assert not owner._async_cycle_in_progress
    assert owner.expert_rearrangement_step == 0
    consumed_event.record.assert_called_once_with()


def test_backported_workspace_move_commits_changed_layer(monkeypatch):
    move_from_buffer = MagicMock()
    commit = MagicMock()
    monkeypatch.setattr(patch_eplb._rebalance_execute, "move_from_buffer", move_from_buffer)
    monkeypatch.setattr(upstream_eplb_state, "_commit_eplb_maps_for_layer", commit)
    consumed_event = MagicMock()
    new_mapping = torch.tensor([1, 0])
    model_state = SimpleNamespace(
        pending_result=patch_eplb._BackportedAsyncEplbLayerResult(
            layer_idx=1,
            new_physical_to_logical_map=new_mapping,
            transfer_metadata="metadata",
            consumed_event=consumed_event,
            is_last_result=False,
        ),
        model=SimpleNamespace(expert_weights=[["l0"], ["l1"]]),
        expert_buffer=["buffer"],
        rebalanced=True,
    )

    patch_eplb._backported_move_to_workspace(model_state, ep_rank=2)

    assert move_from_buffer.call_args_list == [
        call(
            expert_weights=["l1"],
            expert_weights_buffers=["buffer"],
            transfer_metadata="metadata",
            new_indices=new_mapping.numpy(),
            ep_rank=2,
        )
    ]
    commit.assert_called_once_with(
        model_state, new_physical_to_logical_map=new_mapping, layer=1
    )
    consumed_event.record.assert_called_once_with()
