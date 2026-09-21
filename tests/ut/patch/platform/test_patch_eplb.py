# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
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


def _explicit_target():
    target = torch.tensor([[1, 0]])
    target.source_rank_ids = np.array([[[1], [0]]])
    target.source_slot_ids = np.zeros((1, 2, 1), dtype=np.int64)
    return target


def test_async_rebalance_wrapper_stashes_explicit_target_on_communicator():
    target = _explicit_target()
    communicator = SimpleNamespace()
    model_state = SimpleNamespace(
        communicator=communicator,
        eplb_stats=SimpleNamespace(global_expert_load_window=torch.ones((1, 2))),
    )

    def original_rebalance(model_state, eplb_state, physical_to_logical_map_cpu, stream):
        return target

    wrapped = patch_eplb._wrap_async_rebalance(original_rebalance)

    assert wrapped(model_state, object(), torch.tensor([[0, 1]]), object()) is target
    assert getattr(communicator, patch_eplb._EXPLICIT_TRANSFER_TARGET_ATTR) is target


def test_async_rebalance_wrapper_requires_current_upstream_contract():
    def original_rebalance(model_state):
        raise AssertionError("unsupported contract must be rejected before use")

    with pytest.raises(RuntimeError, match="asynchronous rebalance signature"):
        patch_eplb._wrap_async_rebalance(original_rebalance)


def test_async_rebalance_ignores_stale_prepared_stats():
    current_values = torch.tensor([[3, 4]])
    model_state = SimpleNamespace(
        communicator=SimpleNamespace(),
        eplb_stats=SimpleNamespace(global_expert_load_window=current_values),
        _policy_load_stats=patch_eplb.PreparedLoadStats(torch.tensor([[[1, 2]]]), np.array([1])),
    )
    physical_map = torch.tensor([[0, 1]])

    def original_rebalance(model_state, eplb_state, physical_to_logical_map_cpu, stream):
        assert model_state.eplb_stats.global_expert_load_window is current_values
        return physical_to_logical_map_cpu

    wrapped = patch_eplb._wrap_async_rebalance(original_rebalance)

    assert wrapped(model_state, object(), physical_map, object()) is physical_map


def test_async_rebalance_passes_prepared_stats_to_policy_on_worker_stream(monkeypatch):
    worker_stream = object()
    stream_active = False
    cpu_values = torch.tensor([[[1, 2]], [[3, 4]]])
    device_values = MagicMock()

    def copy_to_cpu():
        assert stream_active
        return cpu_values

    device_values.cpu.side_effect = copy_to_cpu

    @contextmanager
    def device_stream(stream):
        nonlocal stream_active
        assert stream is worker_stream
        stream_active = True
        try:
            yield
        finally:
            stream_active = False

    monkeypatch.setattr(patch_eplb._async_worker, "device_stream", device_stream)
    stats = SimpleNamespace(
        global_expert_load_window=device_values,
        num_replicas=2,
        num_groups=1,
        num_nodes=2,
        num_gpus=2,
    )
    prepared_stats = patch_eplb.PreparedLoadStats(device_values, np.array([1, 3]))
    model_state = SimpleNamespace(
        communicator=SimpleNamespace(),
        eplb_stats=stats,
        _policy_load_stats=prepared_stats,
        _last_committed_mean_ratios=np.array([1.2]),
    )
    target = _explicit_target()
    policy = MagicMock()
    policy.rebalance_experts.return_value = target
    rank_node_ids = np.array([0, 1])
    eplb_state = SimpleNamespace(policy=policy, get_rank_node_ids=MagicMock(return_value=rank_node_ids))

    def original_rebalance(model_state, eplb_state, physical_to_logical_map_cpu, stream):
        raise AssertionError("prepared statistics must bypass the legacy runner")

    wrapped = patch_eplb._wrap_async_rebalance(original_rebalance)
    physical_map = torch.tensor([[0, 1]])

    assert wrapped(model_state, eplb_state, physical_map, worker_stream) is target
    planned_stats = policy.rebalance_experts.call_args.args[0]
    assert isinstance(planned_stats, patch_eplb.PreparedLoadStats)
    assert planned_stats.values is cpu_values
    np.testing.assert_array_equal(planned_stats.sample_counts, [1, 3])
    assert policy.rebalance_experts.call_args.args[1:] == (2, 1, 2, 2, physical_map)
    np.testing.assert_array_equal(
        policy.rebalance_experts.call_args.kwargs["last_committed_mean_ratios"],
        [1.2],
    )
    np.testing.assert_array_equal(policy.rebalance_experts.call_args.kwargs["rank_node_ids"], rank_node_ids)


def test_async_transfer_wrapper_executes_explicit_sources(monkeypatch):
    target = _explicit_target()
    communicator = SimpleNamespace(**{patch_eplb._EXPLICIT_TRANSFER_TARGET_ATTR: target})
    metadata = object()
    stage = MagicMock(return_value=metadata)
    monkeypatch.setattr(patch_eplb, "stage_explicit_layer_transfer", stage)

    def original_transfer(
        old_layer_indices,
        new_layer_indices,
        expert_weights,
        expert_weights_buffer,
        ep_group,
        communicator,
        is_profile=False,
        stream=None,
        rank_mapping=None,
        layer_idx=0,
    ):
        raise AssertionError("explicit source target must bypass the upstream transfer")

    stream = object()
    wrapped = patch_eplb._wrap_async_transfer(original_transfer)
    result = wrapped(
        torch.tensor([0, 1]),
        target[0],
        [object()],
        [object()],
        MagicMock(),
        communicator,
        stream=stream,
    )

    assert result is metadata
    assert stage.call_args.kwargs["stream"] is stream
    assert getattr(communicator, patch_eplb._EXPLICIT_TRANSFER_TARGET_ATTR) is target


@pytest.mark.parametrize(
    ("is_profile", "rank_mapping"),
    [(True, None), (False, {0: 0})],
)
def test_async_explicit_transfer_delegates_profile_and_rank_mapping(is_profile, rank_mapping):
    target = _explicit_target()
    communicator = SimpleNamespace(**{patch_eplb._EXPLICIT_TRANSFER_TARGET_ATTR: target})

    def original_transfer(
        old_layer_indices,
        new_layer_indices,
        expert_weights,
        expert_weights_buffer,
        ep_group,
        communicator,
        is_profile=False,
        stream=None,
        rank_mapping=None,
        layer_idx=0,
    ):
        return "upstream"

    wrapped = patch_eplb._wrap_async_transfer(original_transfer)

    result = wrapped(
        torch.tensor([0, 1]),
        target[0],
        None,
        None,
        None,
        communicator,
        is_profile=is_profile,
        rank_mapping=rank_mapping,
    )

    assert result == "upstream"


@pytest.mark.parametrize(("layer_idx", "is_last_layer"), [(2, False), (3, True)])
def test_async_workspace_refreshes_layer_and_clears_target_after_last(monkeypatch, layer_idx, is_last_layer):
    call_order: list[str] = []
    target = _explicit_target()
    target.predicted_mean_ratios = np.full(4, np.nan)
    target.predicted_mean_ratios[layer_idx] = 1.2
    consumed_event = MagicMock()
    consumed_event.record.side_effect = lambda _stream=None: call_order.append("ack")
    pending_result = SimpleNamespace(
        layer_idx=layer_idx,
        transfer_metadata=object(),
        consumed_event=consumed_event,
    )
    model_state = SimpleNamespace(
        pending_result=pending_result,
        rebalanced=True,
        communicator=SimpleNamespace(**{patch_eplb._EXPLICIT_TRANSFER_TARGET_ATTR: target}),
        model=SimpleNamespace(num_moe_layers=4),
        model_name="model",
        _last_committed_mean_ratios=np.full(4, np.nan),
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
    assert model_state._last_committed_mean_ratios[layer_idx] == 1.2
    refresh.assert_called_once_with(model_state, layer_idx)
    if is_last_layer:
        log_info.assert_called_once_with(
            "%s: model=%s",
            patch_eplb.ASYNC_EPLB_CYCLE_COMMITTED_LOG,
            "model",
        )
    else:
        log_info.assert_not_called()
    assert call_order == ["move", "refresh", "ack"]
    assert hasattr(model_state.communicator, patch_eplb._EXPLICIT_TRANSFER_TARGET_ATTR) == (not is_last_layer)


def test_async_workspace_refresh_failure_keeps_target_and_defers_ack(monkeypatch):
    target = _explicit_target()
    target.predicted_mean_ratios = np.array([1.2])
    consumed_event = MagicMock()
    pending_result = SimpleNamespace(
        layer_idx=0,
        transfer_metadata=object(),
        consumed_event=consumed_event,
    )
    communicator = SimpleNamespace(**{patch_eplb._EXPLICIT_TRANSFER_TARGET_ATTR: target})
    model_state = SimpleNamespace(
        pending_result=pending_result,
        communicator=communicator,
        model=SimpleNamespace(num_moe_layers=1),
        model_name="model",
        _last_committed_mean_ratios=np.array([np.nan]),
    )
    monkeypatch.setattr(
        patch_eplb,
        "refresh_model_routing_tables",
        MagicMock(side_effect=RuntimeError("refresh failed")),
    )

    def original_move(model_state, ep_rank):
        model_state.pending_result.consumed_event.record()
        model_state.pending_result = None

    wrapped_move = patch_eplb._wrap_move_to_workspace(original_move)
    with pytest.raises(RuntimeError, match="refresh failed"):
        wrapped_move(model_state, 0)

    assert getattr(communicator, patch_eplb._EXPLICIT_TRANSFER_TARGET_ATTR) is target
    consumed_event.record.assert_not_called()
    assert np.isnan(model_state._last_committed_mean_ratios[0])


def test_async_workspace_keeps_anchor_for_unchanged_layer(monkeypatch):
    target = _explicit_target()
    target.predicted_mean_ratios = np.array([np.nan])
    pending_result = SimpleNamespace(
        layer_idx=0,
        transfer_metadata=object(),
        consumed_event=MagicMock(),
    )
    model_state = SimpleNamespace(
        pending_result=pending_result,
        communicator=SimpleNamespace(**{patch_eplb._EXPLICIT_TRANSFER_TARGET_ATTR: target}),
        model=SimpleNamespace(num_moe_layers=1),
        model_name="model",
        _last_committed_mean_ratios=np.array([1.3]),
    )
    monkeypatch.setattr(patch_eplb, "refresh_model_routing_tables", MagicMock())

    def original_move(model_state, ep_rank):
        model_state.pending_result.consumed_event.record()
        model_state.pending_result = None

    patch_eplb._wrap_move_to_workspace(original_move)(model_state, 0)

    np.testing.assert_array_equal(model_state._last_committed_mean_ratios, [1.3])
