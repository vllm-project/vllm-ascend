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
    model_state = SimpleNamespace(communicator=communicator)

    def original_rebalance(self, model_state, context):
        return target

    wrapped = patch_eplb._wrap_async_rebalance(original_rebalance)

    assert wrapped(object(), model_state, object()) is target
    assert getattr(communicator, patch_eplb._EXPLICIT_TRANSFER_TARGET_ATTR) is target


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


@pytest.mark.parametrize("changed_layer", [None, 1])
def test_async_worker_only_publishes_changed_layers(monkeypatch, changed_layer):
    old_mapping = torch.tensor([[0, 1], [0, 1], [0, 1]])
    new_mapping = old_mapping.clone()
    if changed_layer is not None:
        new_mapping[changed_layer] = torch.tensor([1, 0])
    published = []
    model_state = SimpleNamespace(
        communicator=MagicMock(),
        physical_to_logical_map=old_mapping,
        model=SimpleNamespace(expert_weights=[[object()] for _ in range(3)]),
        expert_buffer=[object()],
        rebalanced=True,
        pending_result=None,
    )

    def wait_for_cycle(*, stream):
        if published:
            raise StopIteration

    def consume_result(*, stream):
        published.append(model_state.pending_result)
        model_state.rebalanced = False
        model_state.pending_result = None

    @contextmanager
    def device_stream(_stream):
        yield

    monkeypatch.setattr(patch_eplb, "AscendEplbState", SimpleNamespace)
    monkeypatch.setattr(
        patch_eplb._async_worker,
        "get_eplb_group",
        lambda: SimpleNamespace(device_group=MagicMock(), cpu_group=MagicMock(size=lambda: 1)),
    )
    monkeypatch.setattr(patch_eplb._async_worker, "run_rebalance_experts", lambda *_args: new_mapping)
    monkeypatch.setattr(patch_eplb._async_worker, "CpuGpuEvent", lambda: SimpleNamespace(wait=consume_result))
    monkeypatch.setattr(patch_eplb._async_worker, "transfer_layer", MagicMock(return_value=object()))
    monkeypatch.setattr(patch_eplb.torch.distributed, "all_reduce", MagicMock())
    monkeypatch.setattr(patch_eplb.torch.cuda, "stream", device_stream)
    state = SimpleNamespace(
        rearrange_event=SimpleNamespace(wait=wait_for_cycle),
        model_states={"model": model_state},
    )
    worker = patch_eplb._wrap_async_worker(MagicMock())

    with pytest.raises(StopIteration):
        worker(state, MagicMock())

    assert len(published) == 1
    assert published[0].layer_idx == changed_layer
    assert published[0].is_last_result
    assert patch_eplb._async_worker.transfer_layer.call_count == int(changed_layer is not None)


def test_async_noop_result_finishes_without_moving_weights(monkeypatch):
    move_from_buffer = MagicMock()
    monkeypatch.setattr(patch_eplb._eplb_state, "move_from_buffer", move_from_buffer)
    consumed_event = MagicMock()
    model_state = SimpleNamespace(
        pending_result=patch_eplb._AscendAsyncLayerResult(None, None, None, consumed_event, True),
        rebalanced=True,
    )

    patch_eplb._move_changed_layer_to_workspace(model_state, 0)

    assert not model_state.rebalanced
    assert model_state.pending_result is None
    move_from_buffer.assert_not_called()
    consumed_event.record.assert_called_once_with()


@pytest.mark.parametrize(("layer_idx", "is_last_layer"), [(2, False), (3, True)])
def test_async_workspace_refreshes_layer_and_clears_target_after_last(monkeypatch, layer_idx, is_last_layer):
    call_order: list[str] = []
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
        communicator=SimpleNamespace(**{patch_eplb._EXPLICIT_TRANSFER_TARGET_ATTR: _explicit_target()}),
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
