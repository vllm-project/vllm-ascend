# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

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
    with _npu_parallel_config_platform():
        parallel_config = ParallelConfig(
            tensor_parallel_size=2,
            enable_expert_parallel=True,
            enable_eplb=True,
            eplb_config=EPLBConfig(use_async=False),
        )
        vllm_config = VllmConfig(parallel_config=parallel_config)

    assert vllm_config.parallel_config.enable_eplb
    assert vllm_config.parallel_config.eplb_config.communicator == "torch_nccl"


def test_communicator_factory_maps_torch_distributed_to_hccl(monkeypatch):
    communicator = object()
    communicator_cls = MagicMock(return_value=communicator)
    monkeypatch.setattr(patch_eplb, "HcclEplbCommunicator", communicator_cls)
    coordinator = MagicMock()

    with _npu_parallel_config_platform():
        result = patch_eplb._eplb_communicator.create_eplb_communicator(
            coordinator,
            "torch_nccl",
            [],
            [],
        )

    assert result is communicator
    communicator_cls.assert_called_once_with(coordinator.device_group)


def test_parallel_config_platform_patch_is_idempotent():
    proxy = parallel_module.current_platform

    patch_eplb._patch_parallel_config()

    assert parallel_module.current_platform is proxy


def test_parallel_config_post_init_forwards_additive_parameters():
    sentinel = object()
    calls = []

    def original_post_init(config, init_value, *, init_mode="default"):
        calls.append((config, init_value, init_mode))
        return sentinel

    wrapped_post_init = patch_eplb._wrap_parallel_config_post_init(original_post_init)
    config = SimpleNamespace(enable_eplb=False)

    result = wrapped_post_init(config, "value", init_mode="custom")

    assert result is sentinel
    assert calls == [(config, "value", "custom")]


def test_communicator_factory_forwards_additive_parameters():
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

    result = wrapped_factory(
        coordinator,
        "torch_gloo",
        expert_weights,
        expert_buffer,
        transport_options={"mode": "future"},
    )

    assert result is sentinel
    assert calls == [
        (
            coordinator,
            "torch_gloo",
            expert_weights,
            expert_buffer,
            {"mode": "future"},
        )
    ]


def test_router_patch_calls_npu_custom_op(monkeypatch):
    topk_ids = torch.tensor([[0, 1]], dtype=torch.int32)
    physical_ids = torch.tensor([[2, 1]], dtype=torch.int32)
    custom_op = MagicMock(return_value=physical_ids)
    monkeypatch.setattr(
        patch_eplb.torch.ops.vllm,
        "ascend_eplb_map_to_physical",
        custom_op,
    )
    layer_state = SimpleNamespace(
        logical_to_physical_map=torch.tensor([[0], [1]], dtype=torch.int32),
        logical_replica_count=torch.ones(2, dtype=torch.int32),
        physical_id_lookup=torch.tensor([[0, 1]], dtype=torch.int32),
        expert_load_view=torch.zeros(3, dtype=torch.int64),
        should_record_tensor=torch.tensor(True),
        num_unpadded_tokens_tensors=[torch.tensor(1, dtype=torch.int32)],
    )
    router = SimpleNamespace(
        eplb_state=layer_state,
        _validate_eplb_state=MagicMock(),
    )

    result = patch_eplb.BaseRouter._apply_eplb_mapping(router, topk_ids)

    assert result is physical_ids
    router._validate_eplb_state.assert_called_once_with()
    custom_op.assert_called_once_with(topk_ids, layer_state.physical_id_lookup)


def test_layer_state_wrapper_builds_lookup_after_forwarding(monkeypatch):
    sentinel = object()
    calls = []

    def original_set_layer_state(
        self,
        moe_layer_idx,
        expert_load_view,
        logical_to_physical_map,
        logical_replica_count,
        *,
        future_option=None,
    ):
        calls.append((self, moe_layer_idx, future_option))
        return sentinel

    refresh = MagicMock()
    monkeypatch.setattr(patch_eplb, "_refresh_layer_lookup", refresh)
    wrapped = patch_eplb._wrap_set_layer_state(original_set_layer_state)
    layer_state = object()

    result = wrapped(
        layer_state,
        3,
        object(),
        object(),
        object(),
        future_option="future",
    )

    assert result is sentinel
    assert calls == [(layer_state, 3, "future")]
    refresh.assert_called_once_with(layer_state)


def test_refresh_layer_lookup_preserves_captured_tensor(monkeypatch):
    old_lookup = torch.full((2, 2), -1, dtype=torch.int32)
    new_lookup = torch.tensor([[0, 3], [2, 1]], dtype=torch.int32)
    layer_state = SimpleNamespace(
        logical_to_physical_map=torch.tensor([[0, 2], [1, 3]], dtype=torch.int32),
        logical_replica_count=torch.tensor([2, 2], dtype=torch.int32),
        physical_id_lookup=old_lookup,
    )
    monkeypatch.setattr(
        patch_eplb,
        "get_ep_group",
        lambda: SimpleNamespace(rank_in_group=1),
    )
    monkeypatch.setattr(
        patch_eplb._eplb_ops,
        "build_physical_id_lookup",
        MagicMock(return_value=new_lookup),
    )

    patch_eplb._refresh_layer_lookup(layer_state)

    assert layer_state.physical_id_lookup is old_lookup
    torch.testing.assert_close(old_lookup, new_lookup)


def test_from_mapping_wrapper_preserves_classmethod_binding(monkeypatch):
    state = SimpleNamespace(model_states={})
    calls = []

    def original_from_mapping(
        cls,
        model,
        model_config,
        device,
        parallel_config,
        expanded_physical_to_logical,
        num_valid_physical_experts,
        *,
        future_option=None,
    ):
        calls.append((cls, model, future_option))
        return state

    class TestState:
        pass

    monkeypatch.setattr(
        TestState,
        "from_mapping",
        classmethod(patch_eplb._wrap_from_mapping(original_from_mapping)),
        raising=False,
    )

    result = TestState.from_mapping(  # type: ignore[attr-defined]
        "model",
        "config",
        "device",
        "parallel",
        "mapping",
        6,
        future_option="future",
    )

    assert result is state
    assert calls == [(TestState, "model", "future")]


def test_eplb_state_step_forwards_additive_parameters():
    sentinel = object()
    calls = []

    def original_step(
        self,
        is_dummy=False,
        is_profile=False,
        log_stats=False,
        *,
        future_option=None,
    ):
        calls.append((self, is_dummy, is_profile, log_stats, future_option))
        return sentinel

    wrapped_step = patch_eplb._wrap_eplb_state_step(original_step)
    state = SimpleNamespace(_ascend_scope_matched=False)

    result = wrapped_step(state, future_option="future")

    assert result is sentinel
    assert calls == [(state, True, False, False, "future")]


def test_eplb_state_step_preserves_upstream_defaults():
    calls = []

    def original_step(
        self,
        is_dummy=False,
        is_profile=True,
        log_stats=True,
        *,
        future_option=None,
    ):
        calls.append((self, is_dummy, is_profile, log_stats, future_option))

    wrapped_step = patch_eplb._wrap_eplb_state_step(original_step)
    state = SimpleNamespace(_ascend_scope_matched=False)

    wrapped_step(state)

    assert calls == [(state, False, True, True, None)]


def test_non_matching_scope_discards_pass_without_advancing_load_window(monkeypatch):
    model_state = SimpleNamespace(expert_load_pass=torch.ones(2, dtype=torch.int64))
    eplb_state = patch_eplb._eplb_state.EplbState.__new__(patch_eplb._eplb_state.EplbState)
    eplb_state.model_states = {"model": model_state}
    eplb_state.parallel_config = SimpleNamespace(
        eplb_config=SimpleNamespace(
            log_balancedness_interval=1,
        )
    )
    eplb_state.expert_rearrangement_step = 0
    eplb_state.expert_rearrangement_step_interval = 10
    eplb_state.expert_load_window_step = 0
    eplb_state.expert_load_window_size = 2
    eplb_state.should_record_tensor = None
    eplb_state.is_async = False
    eplb_state._ascend_scope_matched = False
    ep_group = SimpleNamespace(device_group=MagicMock())
    monkeypatch.setattr(patch_eplb._eplb_state, "get_ep_group", lambda: ep_group)

    eplb_state.step()

    torch.testing.assert_close(model_state.expert_load_pass, torch.zeros(2, dtype=torch.int64))
    assert eplb_state.expert_load_window_step == 0
    assert eplb_state.expert_rearrangement_step == 1
