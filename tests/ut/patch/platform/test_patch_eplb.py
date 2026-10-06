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

from vllm_ascend.ascend_config import EplbConfig
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


@contextmanager
def _without_any_hixl_binding():
    """Make both the official package and the ctypes fallback unavailable."""
    with (
        patch.dict("sys.modules", {"hixl": None}),
        patch(
            "vllm_ascend.distributed.eplb.hixl_compat.ensure_available",
            side_effect=RuntimeError("No CANN HIXL libraries found"),
        ),
    ):
        yield


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
    # Provisional auto-selection happens at config stage (no HIXL here);
    # the worker-stage consensus may still correct it group-wide.
    assert vllm_config.parallel_config.eplb_config.communicator == "torch_gloo"
    assert getattr(
        vllm_config.parallel_config.eplb_config,
        patch_eplb._AUTO_SELECTED_ATTRIBUTE,
        False,
    )


def test_eplb_policy_config_supports_stair_and_default():
    assert EPLBConfig().policy == "stair"
    assert EPLBConfig(policy="stair", use_async=True).policy == "stair"
    assert EPLBConfig(policy="default").policy == "default"

    with pytest.raises(ValueError, match="Input should be 'default' or 'stair'"):
        EPLBConfig(policy="other")
    with pytest.raises(ValueError, match="torch_nccl communicator is incompatible"):
        EPLBConfig(policy="stair", communicator="torch_nccl")

    patch_eplb._patch_eplb_policy_config()
    assert EPLBConfig().policy == "stair"


def test_eplb_communicator_config_supports_hixl():
    assert EPLBConfig().communicator is None
    assert EPLBConfig(communicator="hixl").communicator == "hixl"
    assert EPLBConfig(communicator="torch_gloo").communicator == "torch_gloo"

    patch_eplb._patch_eplb_communicator_config()
    assert EPLBConfig(communicator="hixl").communicator == "hixl"


def _resolver_fixtures(communicator=None, auto_selected=False, additional_config=None):
    """Build configs for the resolver: provisional decision + parsed ascend config.

    Mirrors production: ``ascend_eplb_config`` is parsed from the raw
    ``additional_config`` dict, and the provisional communicator (if any) is
    marked auto-selected so the consensus may correct it.
    """
    parallel_config = MagicMock()
    parallel_config.eplb_config.communicator = communicator
    vllm_config = MagicMock()
    vllm_config.additional_config = additional_config or {}
    raw_eplb_config = vllm_config.additional_config.get("eplb_config", {})
    ascend_eplb_config = EplbConfig(**raw_eplb_config) if raw_eplb_config else EplbConfig()
    if auto_selected:
        patch_eplb._mark_auto_selected(parallel_config.eplb_config)
    else:
        # MagicMock auto-creates truthy attributes on getattr; the marker
        # must be explicitly False to simulate "not auto-selected".
        setattr(parallel_config.eplb_config, patch_eplb._AUTO_SELECTED_ATTRIBUTE, False)
    return parallel_config, ascend_eplb_config, vllm_config


def test_resolve_ascend_communicator_selects_hixl_when_group_agrees():
    for binding in ("official", "ctypes"):
        for provisional in (None, "torch_gloo", "hixl"):
            parallel_config, ascend_eplb_config, vllm_config = _resolver_fixtures(
                communicator=provisional, auto_selected=True
            )
            with (
                patch.object(patch_eplb, "_group_hixl_binding_consensus", return_value=binding),
                patch("vllm_ascend.patch.platform.patch_eplb.get_current_vllm_config", return_value=vllm_config),
            ):
                stair_config = patch_eplb.resolve_ascend_eplb_communicator(parallel_config, ascend_eplb_config)

            assert parallel_config.eplb_config.communicator == "hixl"
            assert stair_config is ascend_eplb_config.stair_config
            assert vllm_config.additional_config == {}


def test_resolve_ascend_communicator_falls_back_to_gloo_and_clamps_unset_limits():
    parallel_config, ascend_eplb_config, vllm_config = _resolver_fixtures(
        communicator="hixl",
        auto_selected=True,
        additional_config={"eplb_config": {"stair_config": {"load_window_bins": 32}}},
    )
    with (
        patch.object(patch_eplb, "_group_hixl_binding_consensus", return_value="none"),
        patch("vllm_ascend.patch.platform.patch_eplb.get_current_vllm_config", return_value=vllm_config),
    ):
        stair_config = patch_eplb.resolve_ascend_eplb_communicator(parallel_config, ascend_eplb_config)

    assert parallel_config.eplb_config.communicator == "torch_gloo"
    assert vllm_config.additional_config["eplb_config"]["stair_config"] == {
        "load_window_bins": 32,
        "rank_transfer_limit": 1,
        "cross_node_transfer_limit": 1,
    }
    assert (stair_config.rank_transfer_limit, stair_config.cross_node_transfer_limit) == (1, 1)
    assert stair_config.load_window_bins == 32


def test_resolve_ascend_communicator_keeps_explicit_limits_on_gloo_fallback():
    parallel_config, ascend_eplb_config, vllm_config = _resolver_fixtures(
        communicator="hixl",
        auto_selected=True,
        additional_config={
            "eplb_config": {"stair_config": {"rank_transfer_limit": -1, "cross_node_transfer_limit": 0}}
        },
    )
    with (
        patch.object(patch_eplb, "_group_hixl_binding_consensus", return_value="mixed"),
        patch("vllm_ascend.patch.platform.patch_eplb.get_current_vllm_config", return_value=vllm_config),
        patch.object(patch_eplb.logger, "warning") as warning,
    ):
        stair_config = patch_eplb.resolve_ascend_eplb_communicator(parallel_config, ascend_eplb_config)

    assert parallel_config.eplb_config.communicator == "torch_gloo"
    assert vllm_config.additional_config["eplb_config"]["stair_config"] == {
        "rank_transfer_limit": -1,
        "cross_node_transfer_limit": 0,
    }
    assert (stair_config.rank_transfer_limit, stair_config.cross_node_transfer_limit) == (-1, 0)
    assert any("keeping the explicitly configured" in call.args[0] for call in warning.call_args_list)


def test_resolve_ascend_communicator_skips_consensus_for_explicit_communicator():
    parallel_config, ascend_eplb_config, _ = _resolver_fixtures(communicator="hixl")
    with patch.object(
        patch_eplb,
        "_group_hixl_binding_consensus",
        side_effect=AssertionError("consensus must not run for an explicit communicator"),
    ):
        stair_config = patch_eplb.resolve_ascend_eplb_communicator(parallel_config, ascend_eplb_config)

    assert parallel_config.eplb_config.communicator == "hixl"
    assert stair_config is ascend_eplb_config.stair_config


def test_group_hixl_binding_consensus_combines_rank_bindings(monkeypatch):
    def consensus_with_rank_bindings(bindings):
        monkeypatch.setattr(
            patch_eplb,
            "_probe_local_hixl_binding",
            lambda: (bindings[0], None if bindings[0] != "none" else "no hixl here"),
        )
        ranks = [patch_eplb._BINDING_CONSENSUS_RANK[binding] for binding in bindings]

        def fake_all_reduce(flag, *, group, op):
            if op == torch.distributed.ReduceOp.MIN:
                flag[0] = min(ranks)
            else:
                flag[0] = max(ranks)

        monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
        monkeypatch.setattr(patch_eplb, "get_eplb_group", lambda: SimpleNamespace(cpu_group=MagicMock()))
        return patch_eplb._group_hixl_binding_consensus()

    assert consensus_with_rank_bindings(["official", "official", "official"]) == "official"
    assert consensus_with_rank_bindings(["ctypes", "ctypes", "ctypes"]) == "ctypes"
    assert consensus_with_rank_bindings(["official", "ctypes"]) == "mixed"
    assert consensus_with_rank_bindings(["ctypes", "none"]) == "none"


def test_group_hixl_binding_consensus_degrades_when_probe_fails(monkeypatch):
    with _without_any_hixl_binding():
        monkeypatch.setattr(patch_eplb, "get_eplb_group", lambda: SimpleNamespace(cpu_group=MagicMock()))
        monkeypatch.setattr(torch.distributed, "all_reduce", lambda flag, *, group, op: flag.fill_(0))
        assert patch_eplb._group_hixl_binding_consensus() == "none"


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


def test_communicator_factory_creates_ascend_hixl_communicator(monkeypatch):
    communicator = object()
    hixl_cls = MagicMock(return_value=communicator)
    monkeypatch.setattr(patch_eplb, "AscendHixlEplbCommunicator", hixl_cls)
    coordinator = MagicMock()
    weights = [[object()]]
    buffer = [object()]

    result = patch_eplb._eplb_communicator.create_eplb_communicator(
        coordinator,
        "hixl",
        weights,
        buffer,
    )

    assert result is communicator
    hixl_cls.assert_called_once_with(
        cpu_group=coordinator.cpu_group,
        all_expert_weights=weights,
        expert_buffer=buffer,
    )


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
    current_values = torch.ones((1, 2))
    model_state = SimpleNamespace(
        communicator=communicator,
        eplb_stats=SimpleNamespace(global_expert_load_window=current_values),
    )

    def original_rebalance(model_state, eplb_state, physical_to_logical_map_cpu, stream):
        return target

    wrapped = patch_eplb._wrap_async_rebalance(original_rebalance)

    assert wrapped(model_state, object(), torch.tensor([[0, 1]]), object()) is target
    assert getattr(communicator, patch_eplb._EXPLICIT_TRANSFER_TARGET_ATTR) is target


def test_async_rebalance_passes_current_prepared_stats_to_policy():
    worker_stream = MagicMock()
    cpu_values = torch.tensor([[[1, 2]], [[3, 4]]])
    device_values = MagicMock()
    device_values.cpu.return_value = cpu_values

    stats = SimpleNamespace(
        global_expert_load_window=device_values,
        num_replicas=2,
        num_groups=1,
        num_nodes=2,
        num_gpus=2,
    )
    model_state = SimpleNamespace(
        communicator=SimpleNamespace(),
        eplb_stats=stats,
        _policy_load_stats=patch_eplb.PreparedLoadStats(device_values, np.array([1, 3])),
        _last_committed_mean_ratios=np.array([1.2]),
    )
    target = _explicit_target()
    policy = MagicMock()
    policy.rebalance_experts.return_value = target
    rank_node_ids = np.array([0, 1])
    eplb_state = SimpleNamespace(
        policy=policy,
        get_rank_node_ids=MagicMock(return_value=rank_node_ids),
    )

    def original_rebalance(
        model_state,
        eplb_state,
        physical_to_logical_map_cpu,
        cuda_stream,
    ):
        raise AssertionError("prepared statistics must bypass the legacy runner")

    physical_map = torch.tensor([[0, 1]])
    result = patch_eplb._wrap_async_rebalance(original_rebalance)(
        model_state,
        eplb_state,
        physical_map,
        worker_stream,
    )

    assert result is target
    worker_stream.__enter__.assert_called_once_with()
    worker_stream.__exit__.assert_called_once()
    np.testing.assert_array_equal(result.rank_node_ids, rank_node_ids)
    planned_stats = policy.rebalance_experts.call_args.args[0]
    assert isinstance(planned_stats, patch_eplb.PreparedLoadStats)
    assert planned_stats.values is cpu_values
    np.testing.assert_array_equal(planned_stats.sample_counts, [1, 3])
    assert policy.rebalance_experts.call_args.args[1:] == (2, 1, 2, 2, physical_map)
    np.testing.assert_array_equal(
        policy.rebalance_experts.call_args.kwargs["last_committed_mean_ratios"],
        [1.2],
    )
    np.testing.assert_array_equal(
        policy.rebalance_experts.call_args.kwargs["rank_node_ids"],
        rank_node_ids,
    )


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
    published: list[SimpleNamespace] = []
    model_state = SimpleNamespace(
        communicator=MagicMock(),
        physical_to_logical_map=old_mapping,
        logical_to_physical_map=torch.empty((3, 2, 1), dtype=torch.int32),
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
    state = SimpleNamespace(
        rearrange_event=SimpleNamespace(wait=wait_for_cycle),
        model_states={"model": model_state},
    )

    def original_worker(state, cuda_stream, is_profile=False):
        raise AssertionError("Ascend state must use the patched worker")

    worker = patch_eplb._wrap_async_worker(original_worker)

    with pytest.raises(StopIteration):
        worker(state=state, cuda_stream=MagicMock())

    assert len(published) == 1
    assert published[0].layer_idx == changed_layer
    assert published[0].is_last_result
    assert patch_eplb._async_worker.transfer_layer.call_count == int(changed_layer is not None)


def test_async_noop_result_finishes_without_moving_weights(monkeypatch):
    move_from_buffer = MagicMock()
    monkeypatch.setattr(patch_eplb._eplb_state, "move_from_buffer", move_from_buffer)
    consumed_event = MagicMock()
    model_state = SimpleNamespace(
        pending_result=patch_eplb._AscendAsyncLayerResult(
            layer_idx=None,
            new_physical_to_logical_map=None,
            new_logical_to_physical_map=None,
            new_logical_replica_count=None,
            transfer_metadata=None,
            consumed_event=consumed_event,
            is_last_result=True,
        ),
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
    target = _explicit_target()
    target.predicted_mean_ratios = np.full(4, np.nan)
    target.predicted_mean_ratios[layer_idx] = 1.2
    target.predicted_imbalance_summary = (1.4, 1.6, 1.1, 1.2)
    target.changed_layer_count = 1
    target.rank_node_ids = np.array([0, 1])
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
    refresh.assert_called_once_with(model_state, layer_idx)
    assert model_state._last_committed_mean_ratios[layer_idx] == 1.2
    if is_last_layer:
        log_info.assert_called_once_with(
            "%s: model=%s mean=%.4f->%.4f p95=%.4f->%.4f changed_layers=%d rank_transfers=%d cross_node_transfers=%d",
            patch_eplb.ASYNC_EPLB_CYCLE_COMMITTED_LOG,
            "model",
            1.4,
            1.1,
            1.6,
            1.2,
            1,
            2,
            2,
        )
    else:
        log_info.assert_not_called()
    assert call_order == ["move", "refresh", "ack"]
    assert hasattr(model_state.communicator, patch_eplb._EXPLICIT_TRANSFER_TARGET_ATTR) == (not is_last_layer)


def test_hixl_workspace_logs_background_transfer_span(monkeypatch):
    call_order: list[str] = []
    communicator = object.__new__(patch_eplb.AscendHixlEplbCommunicator)
    communicator._eplb_hixl_phase_timings = [
        SimpleNamespace(
            launch_ms=1.0,
            transfer_ms=2.0,
            confirmation_ms=3.0,
            request_count=5,
            transfer_bytes=6,
        )
    ]
    consumed_event = MagicMock()
    model_state = SimpleNamespace(
        pending_result=patch_eplb._AscendAsyncLayerResult(
            layer_idx=0,
            new_physical_to_logical_map=torch.tensor([0]),
            new_logical_to_physical_map=torch.tensor([[0]]),
            new_logical_replica_count=torch.tensor([1]),
            transfer_metadata=object(),
            consumed_event=consumed_event,
            is_last_result=True,
        ),
        rebalanced=True,
        communicator=communicator,
        model=SimpleNamespace(num_moe_layers=1, expert_weights=[[object()]]),
        expert_buffer=[object()],
        physical_to_logical_map=torch.full((1, 1), -1, dtype=torch.int32),
        logical_to_physical_map=torch.full((1, 1, 1), -1, dtype=torch.int32),
        logical_replica_count=torch.zeros((1, 1), dtype=torch.int32),
        model_name="model",
        _eplb_migration_span_steps=7,
        _eplb_migration_deferred_steps=2,
        _eplb_foreground_wait_ms=0.25,
    )
    monkeypatch.setattr(
        patch_eplb._eplb_state,
        "move_from_buffer",
        MagicMock(side_effect=lambda **_kwargs: call_order.append("move")),
    )
    monkeypatch.setattr(patch_eplb, "refresh_model_routing_tables", MagicMock())
    log_info = MagicMock()
    monkeypatch.setattr(patch_eplb.logger, "info", log_info)

    def original_move(model_state, ep_rank):
        raise AssertionError("Ascend results must use the patched move path")

    patch_eplb._wrap_move_to_workspace(original_move)(model_state, 0)

    assert call_order == ["move"]
    assert model_state.physical_to_logical_map.tolist() == [[0]]
    assert model_state.logical_to_physical_map.tolist() == [[[0]]]
    assert model_state.logical_replica_count.tolist() == [[1]]
    consumed_event.record.assert_called_once_with(None)
    assert "_eplb_hixl_phase_timings" not in communicator.__dict__
    assert not hasattr(model_state, "_eplb_migration_span_steps")
    assert not hasattr(model_state, "_eplb_migration_deferred_steps")
    assert not hasattr(model_state, "_eplb_foreground_wait_ms")
    hixl_log = next(call for call in log_info.call_args_list if call.args[0].startswith("HIXL EPLB transfer:"))
    assert hixl_log.args[6] == 0.25
    assert hixl_log.args[-2:] == (7, 2)


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
