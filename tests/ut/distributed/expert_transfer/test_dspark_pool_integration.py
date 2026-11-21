# SPDX-License-Identifier: Apache-2.0
"""Expert-pool ownership, configuration and dispatch contracts."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from pydantic import TypeAdapter

from vllm_ascend import ascend_config
from vllm_ascend.ascend_config import DSparkExpertPoolConfig, ExpertOffloadConfig
from vllm_ascend.distributed.expert_transfer import expert_offload_manager as manager_module
from vllm_ascend.distributed.expert_transfer import expert_offload_utils as expert_offload
from vllm_ascend.distributed.expert_transfer.dspark_expert_pool import DSparkExpertPool
from vllm_ascend.distributed.expert_transfer.expert_offload_manager import ExpertOffloadManager
from vllm_ascend.ops.fused_moe.router.fused_topk_router import AscendFusedTopKRouter


@pytest.mark.parametrize("enabled", [False, "false", "true"])
def test_pool_config_uses_standard_bool_and_int_coercion(enabled):
    config = TypeAdapter(DSparkExpertPoolConfig).validate_python(
        {"enabled": enabled, "protected_positions": "2", "suffix_experts": "2"}
    )
    assert config.enabled is (enabled == "true")
    assert config.protected_positions == config.suffix_experts == 2


def test_pool_config_is_nested_under_expert_offload():
    config = TypeAdapter(ExpertOffloadConfig).validate_python(
        {
            "enabled": "true",
            "num_device_experts": "84",
            "dspark_expert_pool": {"enabled": "true", "protected_positions": "2", "suffix_experts": "2"},
        }
    )
    pool_config = getattr(config, "dspark_expert_pool", None)
    assert pool_config is not None, "ExpertOffloadConfig must own the pool options"
    assert pool_config.enabled and config.num_device_experts == 84
    assert pool_config.protected_positions == pool_config.suffix_experts == 2


@pytest.mark.parametrize("options", [{}, {"enabled": False}, {"enabled": True}])
def test_legacy_top_level_pool_key_reports_its_new_location(monkeypatch, options):
    def unexpected_config_parsing(*args, **kwargs):
        pytest.fail("The legacy pool key must be rejected before unrelated config parsing")

    monkeypatch.setattr(ascend_config.SchedulerConfig, "from_additional_config", unexpected_config_parsing)
    config = SimpleNamespace(additional_config={"dspark_expert_pool": options})
    with pytest.raises(ValueError, match=r"expert_offload_config\.dspark_expert_pool"):
        ascend_config.init_ascend_config(config)


def _make_pool_runner(expert_factory):
    layers = [expert_factory(experts=128, top_k=6, slots=84, layer_name=f"layer.{i}")[0] for i in range(2)]
    router = object.__new__(AscendFusedTopKRouter)
    for name, value in {
        "scoring_func": "sqrtsoftplus",
        "top_k": 6,
        "num_logical_experts": 128,
        "renormalize": True,
        "tid2eid": None,
        "bias_vl": None,
        "custom_routing_function": None,
        "num_expert_group": None,
        "topk_group": None,
    }.items():
        setattr(router, name, value)
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
            expert_offload_config=SimpleNamespace(
                enabled=True,
                num_device_experts=84,
                dspark_expert_pool=SimpleNamespace(enabled=True, protected_positions=2, suffix_experts=2),
            )
        ),
    )
    return manager, runner, layers


def test_manager_owns_one_pool_and_only_compatible_bindings_expose_it(expert_factory):
    manager, runner, layers = _make_pool_runner(expert_factory)
    initialize = getattr(manager, "initialize_dspark_expert_pool", None)
    assert callable(initialize), "The offload manager must initialize its shared pool"
    initialize(runner)
    get_pool = getattr(expert_offload, "get_dspark_expert_pool", None)
    assert callable(get_pool), "Runner and model-state hooks need one model-owned lookup"
    pool = get_pool(runner.model)
    assert isinstance(pool, DSparkExpertPool)
    assert pool is manager.dspark_expert_pool
    assert layers[0]._expert_offload_binding.dspark_expert_pool is pool
    assert layers[1]._expert_offload_binding.dspark_expert_pool is None
    assert not hasattr(runner, "dspark_expert_pool")
    assert all(not hasattr(layer, "dspark_expert_pool") for layer in layers)


def test_pool_shutdown_clears_binding_access(expert_factory, native_runtime):
    manager, runner, layers = _make_pool_runner(expert_factory)
    initialize = getattr(manager, "initialize_dspark_expert_pool", None)
    assert callable(initialize), "The offload manager must own the pool lifecycle"
    initialize(runner)
    assert layers[0]._expert_offload_binding.dspark_expert_pool is not None
    expert_offload.shutdown(runner)
    assert expert_offload.get_dspark_expert_pool(runner.model) is None
    assert manager.dspark_expert_pool is None
    assert layers[0]._expert_offload_binding.dspark_expert_pool is None


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
    manager.dspark_expert_pool = SimpleNamespace(num_speculative_tokens=5)
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


@pytest.mark.parametrize("mixed", [False, True])
def test_select_experts_preserves_pure_and_mixed_routing(mixed):
    rows = 7 if mixed else 6
    pool = DSparkExpertPool(torch.device("cpu"), rows, 2, 2, num_speculative_tokens=5, num_device_experts=84)
    pool.prepare(
        SimpleNamespace(
            num_reqs=2 if mixed else 1,
            num_tokens=rows,
            num_tokens_after_padding=rows,
            query_start_loc_np=[0, 1, 7] if mixed else [0, 6],
            num_draft_tokens_per_req=[0, 5] if mixed else [5],
            is_prefilling_np=[False, False] if mixed else [False],
            has_structured_output_reqs=False,
        )
    )
    pooled = (torch.full((rows, 6), 0.25), torch.full((rows, 6), 11, dtype=torch.int32))
    native = (torch.full((rows, 6), 0.5, dtype=torch.bfloat16), torch.full((rows, 6), 23, dtype=torch.int64))
    pool.route = Mock(return_value=pooled)
    router = SimpleNamespace(_select_experts=Mock(return_value=native))
    select = getattr(pool, "select_experts", None)
    assert callable(select), "Pool selection must encapsulate the native-row merge"
    hidden, logits = torch.zeros(rows, 4), torch.zeros(rows, 256)
    weights, ids = select(router, hidden, logits, None)
    if mixed:
        assert weights.dtype == native[0].dtype and ids.dtype == native[1].dtype
        assert weights[:1].equal(native[0][:1]) and ids[:1].equal(native[1][:1])
        assert weights[1:].equal(pooled[0][1:]) and ids[1:].equal(pooled[1][1:])
    else:
        assert weights is pooled[0] and ids is pooled[1]
    assert router._select_experts.call_count == int(mixed)
