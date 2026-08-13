# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

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
    with _npu_parallel_config_platform():
        parallel_config = ParallelConfig(
            tensor_parallel_size=2,
            enable_expert_parallel=True,
            enable_eplb=True,
            eplb_config=EPLBConfig(use_async=True),
        )
        vllm_config = VllmConfig(parallel_config=parallel_config)

    assert vllm_config.parallel_config.enable_eplb
    assert not getattr(ParallelConfig.__post_init__, patch_eplb._PATCH_MARKER, False)


def test_parallel_config_platform_patch_is_idempotent():
    proxy = parallel_module.current_platform

    patch_eplb._patch_parallel_config()

    assert parallel_module.current_platform is proxy


def test_communicator_factory_maps_gloo_to_staged_on_npu(monkeypatch):
    communicator = object()
    gloo_cls = MagicMock(return_value=communicator)
    monkeypatch.setattr(patch_eplb, "AscendGlooEplbCommunicator", gloo_cls)
    coordinator = MagicMock()

    with _npu_parallel_config_platform():
        result = patch_eplb._eplb_communicator.create_eplb_communicator(
            coordinator,
            "torch_gloo",
            [[object()]],
            [object()],
        )

    assert result is communicator
    gloo_cls.assert_called_once_with(cpu_group=coordinator.cpu_group)


def test_communicator_factory_forwards_other_backends_and_additive_parameters():
    sentinel = object()
    calls = []

    def original_factory(
        group_coordinator,
        backend,
        expert_weights,
        expert_buffer,
        *,
        transport_options=None,
    ):
        calls.append(
            (
                group_coordinator,
                backend,
                expert_weights,
                expert_buffer,
                transport_options,
            )
        )
        return sentinel

    wrapped_factory = patch_eplb._wrap_communicator_factory(original_factory)
    coordinator = object()
    expert_weights = object()
    expert_buffer = object()

    with _npu_parallel_config_platform():
        result = wrapped_factory(
            coordinator,
            "nixl",
            expert_weights,
            expert_buffer,
            transport_options={"mode": "future"},
        )

    assert result is sentinel
    assert calls == [
        (
            coordinator,
            "nixl",
            expert_weights,
            expert_buffer,
            {"mode": "future"},
        )
    ]


def test_async_workspace_wrapper_refreshes_committed_layer(monkeypatch):
    pending_result = SimpleNamespace(layer_idx=3, transfer_metadata=object())
    state = SimpleNamespace(commit_policy_layer=MagicMock())
    model_state = SimpleNamespace(
        pending_result=pending_result,
        rebalanced=True,
        _ascend_eplb_state=state,
    )
    refresh = MagicMock()
    monkeypatch.setattr(patch_eplb, "refresh_model_routing_tables", refresh)

    def original_move(model_state, ep_rank, *, future_option=None):
        assert ep_rank == 2
        assert future_option == "future"
        model_state.pending_result = None
        return "moved"

    wrapped_move = patch_eplb._wrap_move_to_workspace(original_move)
    result = wrapped_move(model_state, 2, future_option="future")

    assert result == "moved"
    refresh.assert_called_once_with(model_state, 3)
    state.commit_policy_layer.assert_called_once_with(model_state, 3)


def test_async_workspace_wrapper_acknowledges_no_transfer_cycle(monkeypatch):
    consumed_event = MagicMock()
    pending_result = SimpleNamespace(
        layer_idx=1,
        transfer_metadata=patch_eplb.NO_TRANSFER_CYCLE_COMPLETE,
        consumed_event=consumed_event,
    )
    model_state = SimpleNamespace(
        pending_result=pending_result,
        rebalanced=True,
        model=SimpleNamespace(num_moe_layers=2),
    )
    original_move_called = False

    def original_move(model_state, ep_rank):
        nonlocal original_move_called
        original_move_called = True

    refresh = MagicMock()
    monkeypatch.setattr(patch_eplb, "refresh_model_routing_tables", refresh)

    wrapped_move = patch_eplb._wrap_move_to_workspace(original_move)
    result = wrapped_move(model_state, 0)

    assert result is None
    assert model_state.rebalanced is False
    assert model_state.pending_result is None
    consumed_event.record.assert_called_once_with()
    assert original_move_called is False
    refresh.assert_not_called()
