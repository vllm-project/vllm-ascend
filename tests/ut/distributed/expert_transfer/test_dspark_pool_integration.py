# SPDX-License-Identifier: Apache-2.0
"""Expert-pool ownership, configuration and dispatch contracts."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from pydantic import TypeAdapter

from vllm_ascend import ascend_config
from vllm_ascend.ascend_config import ExpertOffloadConfig
from vllm_ascend.distributed.expert_transfer import expert_offload_manager as manager_module
from vllm_ascend.distributed.expert_transfer import expert_offload_utils as expert_offload
from vllm_ascend.distributed.expert_transfer.dspark_expert_pool import DSparkExpertPool
from vllm_ascend.distributed.expert_transfer.expert_offload_manager import ExpertOffloadManager
from vllm_ascend.ops.fused_moe.router import fused_topk_router as router_module
from vllm_ascend.ops.fused_moe.router.fused_topk_router import AscendFusedTopKRouter


@pytest.mark.parametrize("enabled", [False, "false", "true"])
@pytest.mark.parametrize("suffix", ["0", "2"])
def test_flat_pool_config_uses_standard_bool_and_int_coercion(enabled, suffix):
    config = TypeAdapter(ExpertOffloadConfig).validate_python(
        {
            "enabled": enabled,
            "num_device_experts": "84",
            "dspark_protected_positions": "2",
            "dspark_suffix_experts": suffix,
        }
    )
    assert config.enabled is (enabled == "true")
    assert config.num_device_experts == 84
    assert config.dspark_protected_positions == 2
    assert config.dspark_suffix_experts == int(suffix)


@pytest.mark.parametrize("options", [{}, {"enabled": False}, {"enabled": True}])
@pytest.mark.parametrize("nested", [False, True])
def test_legacy_pool_key_reports_the_flat_fields(monkeypatch, options, nested):
    def unexpected_config_parsing(*args, **kwargs):
        pytest.fail("The legacy pool key must be rejected before unrelated config parsing")

    monkeypatch.setattr(ascend_config.SchedulerConfig, "from_additional_config", unexpected_config_parsing)
    additional_config = {"dspark_expert_pool": options}
    if nested:
        additional_config = {"expert_offload_config": additional_config}
    config = SimpleNamespace(additional_config=additional_config)
    with pytest.raises(ValueError, match=r"expert_offload_config\.dspark_protected_positions.*dspark_suffix_experts"):
        ascend_config.init_ascend_config(config)


def test_schema_rejects_the_removed_nested_pool_key():
    with pytest.raises(ValueError, match="dspark_expert_pool"):
        TypeAdapter(ExpertOffloadConfig).validate_python({"dspark_expert_pool": {"enabled": True}})


def _make_pool_runner(expert_factory):
    layers = [expert_factory(experts=128, top_k=6, slots=84, layer_name=f"layer.{i}")[0] for i in range(2)]
    router = AscendFusedTopKRouter(top_k=6, global_num_experts=128, scoring_func="sqrtsoftplus")
    layers[0].router, layers[1].router = router, None
    layers[0].log2phy = None
    manager = ExpertOffloadManager()
    manager.bindings = {layer.layer_name: layer._expert_offload_binding for layer in layers}
    for binding in manager.bindings.values():
        binding.manager = manager
    model = torch.nn.ModuleList(layers)
    model._expert_offload_manager = manager
    runner = SimpleNamespace(
        model=model,
        device=torch.device("cpu"),
        max_num_tokens=12,
        speculative_config=SimpleNamespace(num_speculative_tokens=5),
        ascend_config=SimpleNamespace(
            expert_offload_config=ExpertOffloadConfig(
                enabled=True,
                num_device_experts=84,
                dspark_protected_positions=2,
                dspark_suffix_experts=2,
            )
        ),
    )
    return manager, runner, layers


@pytest.mark.parametrize("suffix", [0, 2])
def test_manager_owns_one_pool_and_only_compatible_bindings_expose_it(expert_factory, suffix):
    manager, runner, layers = _make_pool_runner(expert_factory)
    runner.ascend_config.expert_offload_config = ExpertOffloadConfig(
        enabled=True, num_device_experts=84, dspark_protected_positions=2, dspark_suffix_experts=suffix
    )
    initialize = getattr(manager, "initialize_dspark_expert_pool", None)
    assert callable(initialize), "The offload manager must initialize its shared pool"
    initialize(runner)
    pool = manager.dspark_expert_pool
    assert isinstance(pool, DSparkExpertPool)
    assert pool.protected_positions == 2 and pool.suffix_experts == suffix
    assert pool is manager.dspark_expert_pool
    assert layers[0]._expert_offload_binding.dspark_expert_pool is pool
    assert layers[1]._expert_offload_binding.dspark_expert_pool is None
    assert not hasattr(runner, "dspark_expert_pool")
    assert all(not hasattr(layer, "dspark_expert_pool") for layer in layers)


@pytest.mark.parametrize("offload_enabled", [False, True])
def test_omitted_pool_parameters_do_not_allocate_a_pool(expert_factory, offload_enabled):
    manager, runner, layers = _make_pool_runner(expert_factory)
    runner.ascend_config.expert_offload_config = ExpertOffloadConfig(
        enabled=offload_enabled, num_device_experts=84 if offload_enabled else None
    )
    manager.initialize_dspark_expert_pool(runner)
    assert manager.dspark_expert_pool is None
    assert all(layer._expert_offload_binding.dspark_expert_pool is None for layer in layers)


def test_pool_shutdown_clears_binding_access(expert_factory, native_runtime):
    manager, runner, layers = _make_pool_runner(expert_factory)
    initialize = getattr(manager, "initialize_dspark_expert_pool", None)
    assert callable(initialize), "The offload manager must own the pool lifecycle"
    initialize(runner)
    assert layers[0]._expert_offload_binding.dspark_expert_pool is not None
    expert_offload.shutdown(runner)
    assert expert_offload.get_manager(runner.model) is None
    assert manager.dspark_expert_pool is None
    assert layers[0]._expert_offload_binding.dspark_expert_pool is None


def test_pool_reinitialization_does_not_discover_routers_as_expert_layers(expert_factory):
    manager, runner, layers = _make_pool_runner(expert_factory)
    manager.initialize_dspark_expert_pool(runner)
    first = manager.dspark_expert_pool
    manager.initialize_dspark_expert_pool(runner)
    assert manager.dspark_expert_pool is not first
    assert layers[0]._expert_offload_binding.dspark_expert_pool is manager.dspark_expert_pool
    assert layers[1]._expert_offload_binding.dspark_expert_pool is None


def test_initial_pool_allocation_is_counted_by_offload_loading(expert_factory, monkeypatch):
    manager, runner, _ = _make_pool_runner(expert_factory)
    runner.model_memory_usage = 1000
    # Weight storage is outside this test; keep the real pool allocation and accounting.
    monkeypatch.setattr(manager, "initialize", lambda model: None)
    monkeypatch.setattr(manager_module, "ExpertOffloadManager", lambda: manager)
    observed = []

    def allocated(device):
        has_pool = getattr(manager, "dspark_expert_pool", None) is not None
        observed.append(has_pool)
        return 164 if has_pool else 100

    monkeypatch.setattr(torch.npu, "memory_allocated", allocated)
    manager_module.initialize_expert_offload(runner)
    assert observed == [False, True]
    assert runner.model_memory_usage == 1064


@pytest.mark.parametrize(
    "rows,drafts,structured,invalid,width,expected",
    [
        ([6, 6], [5, 5], False, False, 6, 6),
        ([6], [5], True, False, 6, None),
        ([6], [5], False, True, 6, None),
        ([3], [2], False, False, 3, None),
        ([1, 6], [0, 5], False, False, None, None),
        ([1], None, False, False, 1, 1),
    ],
)
def test_prepare_batch_returns_dispatch_width_without_changing_offload_phase(
    rows, drafts, structured, invalid, width, expected
):
    manager = ExpertOffloadManager()
    manager.dspark_expert_pool = DSparkExpertPool(
        torch.device("cpu"), 12, 2, 2, num_speculative_tokens=5, num_device_experts=84
    )
    model = torch.nn.Module()
    model._expert_offload_manager = manager
    batch = SimpleNamespace(
        has_prefill=False,
        num_scheduled_tokens=rows,
        num_draft_tokens_np=drafts,
        is_prefilling_np=[False] * len(rows),
    )
    scheduled = SimpleNamespace(has_structured_output_requests=structured, num_invalid_spec_tokens=int(invalid))
    actual = expert_offload.prepare_batch(SimpleNamespace(model=model), batch, width, scheduled)
    assert actual == expected
    assert manager.has_prefill is False


@pytest.mark.parametrize("layout", ["verify", "decode_verify", "prefix_verify", "padding"])
def test_router_captures_final_ids_once_and_preserves_requested_dtype(expert_factory, monkeypatch, layout):
    manager, runner, layers = _make_pool_runner(expert_factory)
    manager.initialize_dspark_expert_pool(runner)
    router, pool = layers[0].router, manager.dspark_expert_pool
    mixed = layout == "decode_verify"
    real = 7 if layout in ("decode_verify", "prefix_verify") else 6
    rows = 12 if layout == "padding" else real
    batch = SimpleNamespace(
        num_reqs=2 if mixed else 1,
        num_tokens=real,
        num_tokens_after_padding=rows,
        query_start_loc_np=[0, 1, 7] if mixed else [0, real],
        num_draft_tokens_per_req=[0, 5] if mixed else [5],
        is_prefilling_np=[False] * (2 if mixed else 1),
        has_structured_output_reqs=False,
        has_prefill=False,
    )
    pool.prepare(batch)
    pooled = (torch.full((rows, 6), 0.25), torch.full((rows, 6), 11, dtype=torch.int32))
    pool.route = Mock(return_value=pooled)
    native_calls, captured = [], []

    def native_kernel(**kwargs):
        native_calls.append(kwargs)
        return torch.full((rows, 6), 0.5), torch.full((rows, 6), 23, dtype=torch.int32), None

    monkeypatch.setattr(torch.ops._C_ascend, "moe_gating_top_k_hash", native_kernel, raising=False)
    monkeypatch.setattr(router_module, "_EXTRA_CTX", SimpleNamespace(is_draft_model=False, in_profile_run=False))
    router.set_capture_fn(lambda ids: captured.append(ids.clone()))
    weights, ids = router._select_experts(torch.zeros(rows, 16), torch.zeros(rows, 128), torch.int64)
    expected_ids = [[11] * 6 for _ in range(rows)]
    expected_weights = torch.full((rows, 6), 0.25)
    if layout == "prefix_verify":
        expected_ids[0] = [23] * 6
        expected_weights[0] = 0.5
    assert ids.dtype == torch.int64 and ids.tolist() == expected_ids
    assert weights.equal(expected_weights)
    assert len(captured) == 1 and captured[0].tolist() == expected_ids
    assert len(native_calls) == int(layout == "prefix_verify")


@pytest.mark.parametrize("reason", ["uninitialized", "incompatible", "inactive", "draft", "profile"])
def test_router_retains_native_fallback_and_capture(expert_factory, monkeypatch, reason):
    manager, runner, layers = _make_pool_runner(expert_factory)
    manager.initialize_dspark_expert_pool(runner)
    router, pool = layers[0].router, manager.dspark_expert_pool
    pool.active = reason != "inactive"
    pool.route = Mock(side_effect=AssertionError("fallback must not launch the pool"))
    if reason == "uninitialized":
        router._expert_pool_binding = None
    elif reason == "incompatible":
        layers[0]._expert_offload_binding._uses_dspark_expert_pool = False
    monkeypatch.setattr(
        router_module,
        "_EXTRA_CTX",
        SimpleNamespace(is_draft_model=reason == "draft", in_profile_run=reason == "profile"),
    )
    native = torch.arange(6, dtype=torch.int32).reshape(1, 6)
    monkeypatch.setattr(
        torch.ops._C_ascend,
        "moe_gating_top_k_hash",
        lambda **kwargs: (torch.full((1, 6), 0.5), native, None),
        raising=False,
    )
    captured = []
    router.set_capture_fn(lambda ids: captured.append(ids.clone()))
    weights, ids = router._select_experts(torch.zeros(1, 16), torch.zeros(1, 128), torch.int64)
    assert weights.tolist() == [[0.5] * 6]
    assert ids.dtype == torch.int64 and ids.tolist() == [[0, 1, 2, 3, 4, 5]]
    assert len(captured) == 1 and captured[0].equal(native)


@pytest.mark.parametrize("force_balance", [False, True])
def test_moe_routing_keeps_the_common_router_entry_and_postprocessing(load_hook_class, monkeypatch, force_balance):
    moe = load_hook_class(
        "vllm_ascend/ops/fused_moe/routed_experts.py",
        "AscendRoutedExperts",
        {"_select_experts"},
        object,
        {
            "torch": torch,
            "get_ascend_config": lambda: SimpleNamespace(enable_force_eplb=False),
            "get_moe_num_logical_experts": lambda *args, **kwargs: 8,
        },
    )
    weights, ids = torch.full((1, 6), 0.5), torch.tensor([[0, 1, 2, 3, 4, 5]])
    router = SimpleNamespace(_select_experts=Mock(return_value=(weights, ids)))
    binding = SimpleNamespace(select_experts=Mock(side_effect=AssertionError("dispatch belongs inside the router")))
    layer = SimpleNamespace(
        router=router,
        _expert_offload_binding=binding,
        log2phy=torch.arange(20) + 10,
        n_shared_experts=0,
        moe_config=SimpleNamespace(num_experts=8),
        global_redundant_expert_num=0,
    )
    if force_balance:
        monkeypatch.setattr(torch, "rand", lambda *a, **k: torch.tensor([[7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0]]))
    actual_weights, actual_ids = moe._select_experts(layer, torch.zeros(1, 16), torch.zeros(1, 8), force_balance)
    assert actual_weights is weights
    assert actual_ids.tolist() == ([[7, 6, 5, 4, 3, 2]] if force_balance else [[10, 11, 12, 13, 14, 15]])


@pytest.mark.parametrize("has_manager", [False, True])
def test_lifecycle_hooks_reset_and_prepare_roles_without_a_pool_reference(has_manager):
    manager = ExpertOffloadManager()
    pool = DSparkExpertPool(torch.device("cpu"), 12, 2, 2, num_speculative_tokens=5, num_device_experts=84)
    manager.dspark_expert_pool = pool
    model = torch.nn.Module()
    if has_manager:
        model._expert_offload_manager = manager
    runner = SimpleNamespace(model=model)
    batch = SimpleNamespace(
        num_reqs=2,
        num_tokens=12,
        num_tokens_after_padding=12,
        query_start_loc_np=[0, 6, 12],
        num_draft_tokens_per_req=[5, 5],
        is_prefilling_np=[False, False],
        has_structured_output_reqs=False,
        has_prefill=False,
    )
    scheduled = SimpleNamespace(num_invalid_spec_tokens=0)
    address = pool.roles.data_ptr()
    for real_rows in (12, 6, 12):
        batch.num_tokens = real_rows
        batch.num_reqs = real_rows // 6
        batch.num_draft_tokens_per_req = [5] * batch.num_reqs
        batch.is_prefilling_np = [False] * batch.num_reqs
        pool.active = True
        expert_offload.before_execute(runner)
        assert pool.active is (not has_manager)
        expert_offload.prepare_capture(model, batch)
        expert_offload.prepare_inputs(runner, batch, scheduled)
        if has_manager:
            assert pool.active
            assert pool.roles.tolist() == [1, 1, 2, 2, 2, 2] * batch.num_reqs + [0] * (12 - real_rows)
            assert not pool.has_native_rows
            assert pool.roles.data_ptr() == address
    if has_manager:
        scheduled.num_invalid_spec_tokens = 1
        expert_offload.prepare_inputs(runner, batch, scheduled)
        assert not pool.active
    assert expert_offload.prepare_batch(runner, None, 1, scheduled) == 1
    assert expert_offload.get_manager(model) is (manager if has_manager else None)


def test_execute_clears_stale_pool_before_an_empty_or_failed_parent(load_hook_class):
    manager = ExpertOffloadManager()
    pool = DSparkExpertPool(torch.device("cpu"), 12, 2, 2, num_speculative_tokens=5, num_device_experts=84)
    pool.active = True
    manager.dspark_expert_pool = pool

    class Parent:
        def execute_model(self, *args, **kwargs):
            assert not pool.active
            raise RuntimeError("parent failure after reset")

    runner_class = load_hook_class(
        "vllm_ascend/worker/v2/model_runner.py",
        "NPUModelRunner",
        {"execute_model"},
        Parent,
        {
            "expert_offload": expert_offload,
            "has_kv_transfer_group": lambda: False,
            "should_skip_allreduce_across_dp_group": lambda _: False,
            "nullcontext": nullcontext,
            "HostMirrorConnector": type("HostMirrorConnector", (), {}),
            # This isolated runner owns no delayed-D2H buffers.
            "d2h_retire": lambda *args: None,
        },
    )
    runner = runner_class()
    runner.model = torch.nn.Module()
    runner.model._expert_offload_manager = manager
    runner.model_state = SimpleNamespace(finish_execution=Mock())
    runner.vllm_config = object()
    scheduled = SimpleNamespace(total_num_scheduled_tokens=0)
    with pytest.raises(RuntimeError, match="parent failure after reset"):
        runner.execute_model(scheduled)
    runner.model_state.finish_execution.assert_called_once_with(failed=True)
    assert not pool.active
