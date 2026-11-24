# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import importlib.util
import sys
from dataclasses import dataclass
from importlib.metadata import EntryPoint
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import Mock

import pytest
import yaml

_REPOSITORY_ROOT = Path(__file__).parents[3]
_PACKAGE_ROOT = _REPOSITORY_ROOT / "vllm_ascend" / "observability" / "ms_metrics"
_HANDLER_PACKAGE = "vllm_ascend.observability.ms_metrics.handlers"
_COMMON_MODULE = f"{_HANDLER_PACKAGE}.common"


def _load_all_provider_configs(config_paths):
    configs = []
    for config_path in config_paths:
        configs.extend(yaml.safe_load(Path(config_path).read_text(encoding="utf-8")))
    return configs


def _load_source(module_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(module_name, _PACKAGE_ROOT / filename)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_provider_api(monkeypatch, **attributes):
    package = ModuleType("ms_service_metric")
    provider_api = ModuleType("ms_service_metric.provider_api")
    for name, value in attributes.items():
        setattr(provider_api, name, value)
    monkeypatch.setitem(sys.modules, "ms_service_metric", package)
    monkeypatch.setitem(sys.modules, "ms_service_metric.provider_api", provider_api)


def _load_feature_handlers(monkeypatch, module_name: str, feature: str):
    """Load ``handlers/<feature>.py`` freshly with ``handlers/common.py`` seeded.

    The common module is registered under its real absolute name so the
    feature module's import resolves without importing the whole package.
    """
    common = _load_source(f"{module_name}_common", "handlers/common.py")
    common.set_metric_level("basic")
    monkeypatch.setitem(sys.modules, _COMMON_MODULE, common)
    return _load_source(module_name, f"handlers/{feature}.py")


def test_get_metric_provider_returns_packaged_yaml(monkeypatch):
    @dataclass
    class FakeMetricProvider:
        name: str
        config_paths: tuple[str, ...]
        priority: int
        owned_symbol_prefixes: tuple[str, ...]
        framework_package: str | None = None
        handler_module_prefixes: tuple[str, ...] = ()
        ownership_mode: str = "overlay"

    _install_provider_api(monkeypatch, MetricProvider=FakeMetricProvider)
    provider_module = _load_source("test_vllm_ascend_metric_provider", "provider.py")

    provider = provider_module.get_metric_provider()

    assert provider.name == "vllm-ascend"
    assert provider.framework_package == "vllm_ascend"
    assert provider.ownership_mode == "overlay"
    assert provider.owned_symbol_prefixes == ("vllm_ascend.",)
    assert provider.handler_module_prefixes == ("vllm_ascend.observability.ms_metrics.",)
    assert provider.config_paths == tuple(sorted(provider.config_paths))
    assert [Path(path).name for path in provider.config_paths] == [
        "base_metrics.yaml",
        "eplb_metrics.yaml",
        "runtime_guard_metrics.yaml",
        "schedule_execute_metrics.yaml",
    ]
    assert all(Path(path).is_file() for path in provider.config_paths)
    config = _load_all_provider_configs(provider.config_paths)
    assert len(config) == 13
    assert all(item["symbol"].startswith("vllm_ascend.") for item in config)
    assert all("id" not in item for item in config)
    assert all(
        item.get("handler", "").startswith(
            (
                "ms_service_metric.provider_handlers:",
                "vllm_ascend.observability.ms_metrics.handlers:",
            )
        )
        for item in config
    )


def test_setup_registers_provider_entry_point_and_yaml_package_data():
    setup_path = _REPOSITORY_ROOT / "setup.py"
    syntax_tree = ast.parse(setup_path.read_text(encoding="utf-8"))
    setup_call = next(
        node
        for node in ast.walk(syntax_tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "setup"
    )
    keywords = {keyword.arg: keyword.value for keyword in setup_call.keywords}
    entry_points = ast.literal_eval(keywords["entry_points"])
    package_data = ast.literal_eval(keywords["package_data"])

    assert entry_points["ms_service_metric.providers"] == [
        "vllm-ascend = vllm_ascend.observability.ms_metrics:get_metric_provider"
    ]
    assert package_data["vllm_ascend.observability.ms_metrics"] == ["config/*.yaml"]


def test_provider_entry_point_loads_without_metric_core(monkeypatch):
    package = ModuleType("vllm_ascend")
    package.__path__ = [str(_REPOSITORY_ROOT / "vllm_ascend")]
    monkeypatch.setitem(sys.modules, "vllm_ascend", package)
    monkeypatch.delitem(sys.modules, "ms_service_metric", raising=False)
    monkeypatch.delitem(sys.modules, "ms_service_metric.provider_api", raising=False)

    entry_point = EntryPoint(
        name="vllm-ascend",
        value="vllm_ascend.observability.ms_metrics:get_metric_provider",
        group="ms_service_metric.providers",
    )

    assert callable(entry_point.load())


def test_ascend_handler_module_loads_from_yaml_path(monkeypatch):
    metric_type = type("MetricType", (), {"GAUGE": "gauge", "HISTOGRAM": "histogram", "COUNTER": "counter"})
    _install_provider_api(
        monkeypatch,
        MetricType=metric_type,
        get_metric_recorder=Mock(),
    )
    package = ModuleType("vllm_ascend")
    package.__path__ = [str(_REPOSITORY_ROOT / "vllm_ascend")]
    monkeypatch.setitem(sys.modules, "vllm_ascend", package)
    for module_name in (
        "vllm_ascend.observability",
        "vllm_ascend.observability.ms_metrics",
        "vllm_ascend.observability.ms_metrics.provider",
        "vllm_ascend.observability.ms_metrics.handlers",
        "vllm_ascend.observability.ms_metrics.handlers.common",
        "vllm_ascend.observability.ms_metrics.handlers.eplb",
        "vllm_ascend.observability.ms_metrics.handlers.runtime_guard",
        "vllm_ascend.observability.ms_metrics.handlers.schedule_execute",
    ):
        monkeypatch.delitem(sys.modules, module_name, raising=False)

    handler_module = __import__(
        _HANDLER_PACKAGE,
        fromlist=[
            "eplb_sync_load_handler",
            "runtime_guard_handle_alert_handler",
        ],
    )

    assert callable(handler_module.eplb_sync_load_handler)
    assert callable(handler_module.runtime_guard_handle_alert_handler)


def test_provider_yaml_symbols_exist_in_current_vllm_ascend_source():
    config_paths = sorted((_PACKAGE_ROOT / "config").glob("*.yaml"))
    assert len(config_paths) == 4
    config = _load_all_provider_configs(config_paths)

    for item in config:
        module_name, attribute_path = item["symbol"].split(":", 1)
        source_path = _REPOSITORY_ROOT / Path(*module_name.split(".")).with_suffix(".py")
        assert source_path.is_file(), f"Missing symbol module: {item['symbol']}"

        syntax_tree = ast.parse(source_path.read_text(encoding="utf-8"))
        owner_name, method_name = attribute_path.split(".", 1)
        owner = next(
            (node for node in syntax_tree.body if isinstance(node, ast.ClassDef) and node.name == owner_name),
            None,
        )
        assert owner is not None, f"Missing symbol owner: {item['symbol']}"
        assert any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name
            for node in owner.body
        ), f"Missing symbol method: {item['symbol']}"


def test_record_metric_skips_specs_above_active_level_and_registers_on_raise(monkeypatch):
    metrics = Mock()
    _install_provider_api(
        monkeypatch,
        MetricType=type("MetricType", (), {"GAUGE": "gauge", "HISTOGRAM": "histogram", "COUNTER": "counter"}),
        get_metric_recorder=lambda: metrics,
    )
    common = _load_source("test_metric_level_common", "handlers/common.py")
    common.set_metric_level("basic")
    low = common.MetricSpec("low", "gauge", ())
    high = common.MetricSpec("high", "gauge", ("layer",), level=common.METRIC_LEVEL_DETAIL)
    recorders: dict[str, object] = {}

    assert common.register_metrics((low, high), recorders, "group") is metrics
    assert metrics.get_or_create_metric.call_count == 1
    common.record_metric(metrics, high, 1.0, {"layer": "0"})
    metrics.record_metric.assert_not_called()
    common.record_metric(metrics, low, 1.0, {})
    metrics.record_metric.assert_called_once_with("low", value=1.0, labels={})

    common.set_metric_level("detail")
    common.register_metrics((low, high), recorders, "group")
    assert metrics.get_or_create_metric.call_count == 3
    common.record_metric(metrics, high, 2.0, {"layer": "0"})
    assert metrics.record_metric.call_count == 2

    common.set_metric_level("basic")
    common.record_metric(metrics, high, 3.0, {"layer": "0"})
    common.record_metric(metrics, low, 3.0, {})
    common.record_metric(None, low, 3.0, {})
    assert metrics.record_metric.call_count == 3


def test_set_metric_level_rejects_unknown_value(monkeypatch):
    _install_provider_api(
        monkeypatch,
        MetricType=type("MetricType", (), {"GAUGE": "gauge", "HISTOGRAM": "histogram", "COUNTER": "counter"}),
        get_metric_recorder=lambda: None,
    )
    common = _load_source("test_metric_level_reject", "handlers/common.py")

    with pytest.raises(ValueError):
        common.set_metric_level("nope")
    with pytest.raises(ValueError):
        common.set_metric_level(2)
    with pytest.raises(ValueError):
        common.set_metric_level("off")


def test_symbol_hook_handlers_honor_level_and_labels(monkeypatch):
    metric_type = type("MetricType", (), {"GAUGE": "gauge", "HISTOGRAM": "histogram", "COUNTER": "counter"})
    metrics = Mock()
    _install_provider_api(
        monkeypatch,
        MetricType=metric_type,
        get_metric_recorder=lambda: metrics,
    )
    handlers = _load_feature_handlers(monkeypatch, "test_schedule_execute_symbol_hooks", "schedule_execute")
    common = sys.modules[_COMMON_MODULE]
    common.set_metric_level("basic")

    assert handlers.sample_tokens_handler(lambda runner, grammar: "sampled", object(), None) == "sampled"
    metrics.record_metric.assert_not_called()


def test_custom_handlers_are_not_factory_created_from_yaml_metrics():
    """A custom handler entry must not declare a yaml ``metrics`` block.

    ``ms_service_metric`` treats ``handler`` + ``metrics`` as a factory
    contract: it calls ``handler(metrics_config, ...)`` and falls back to a
    pass-through hook on ``TypeError``, so the metric would silently
    disappear while "Hooked symbol" logs still claim success. Every
    ``vllm_ascend.*`` custom entry must declare only ``symbol`` and
    ``handler``; provider built-ins that follow the factory convention are
    the only entries allowed to keep a ``metrics`` block.
    """
    provider_handler_prefix = "ms_service_metric.provider_handlers:"
    custom_handler_prefix = "vllm_ascend.observability.ms_metrics.handlers:"

    config_paths = sorted((_PACKAGE_ROOT / "config").glob("*.yaml"))
    config = _load_all_provider_configs(config_paths)
    assert config, "no provider yaml entries found"

    for item in config:
        handler = item.get("handler", "")
        if handler.startswith(custom_handler_prefix):
            assert "metrics" not in item, (
                f"custom handler entry declares a metrics block and would be "
                f"factory-created (then silently fall back to a pass-through "
                f"hook): {item}"
            )
        elif handler.startswith(provider_handler_prefix):
            # Provider built-ins follow the factory convention; nothing to
            # assert here beyond the prefix check already done above.
            pass


# ---------------------------------------------------------------------------
# EPLB V2 load balance (AscendEplbState.step)
# ---------------------------------------------------------------------------


def _install_ep_group(monkeypatch, ep_group):
    package: Any = ModuleType("vllm")
    distributed: Any = ModuleType("vllm.distributed")
    parallel_state: Any = ModuleType("vllm.distributed.parallel_state")
    parallel_state.get_ep_group = lambda: type("Group", (), {"device_group": ep_group})()
    package.distributed = distributed
    distributed.parallel_state = parallel_state
    monkeypatch.setitem(sys.modules, "vllm", package)
    monkeypatch.setitem(sys.modules, "vllm.distributed", distributed)
    monkeypatch.setitem(sys.modules, "vllm.distributed.parallel_state", parallel_state)


def _load_balance_state(torch, *, load=None):
    """State shaped like ``AscendEplbState`` for the sync-load hook."""
    if load is None:
        load = torch.tensor([[1.0, 3.0], [2.0, 2.0]])
    return type(
        "State",
        (),
        {
            "model_states": {
                "main": type(
                    "ModelState",
                    (),
                    {
                        "model_name": "main",
                        "expert_load_pass": load,
                    },
                )()
            }
        },
    )()


def _sync_load_pass(state, *args, **kwargs):
    """Mirror upstream: return the globally reduced load pass."""
    return [model_state.expert_load_pass.clone() for model_state in state.model_states.values()]


def test_eplb_sync_load_handler_records_reduced_load(monkeypatch):
    torch = pytest.importorskip("torch")
    metrics = Mock()
    _install_provider_api(
        monkeypatch,
        MetricType=type("MetricType", (), {"GAUGE": "gauge", "HISTOGRAM": "histogram", "COUNTER": "counter"}),
        get_metric_recorder=lambda: metrics,
    )
    handlers = _load_feature_handlers(monkeypatch, "test_eplb_load_balance", "eplb")
    ep_group = Mock()
    ep_group.rank.return_value = 0
    ep_group.size.return_value = 2
    _install_ep_group(monkeypatch, ep_group)

    result = handlers.eplb_sync_load_handler(_sync_load_pass, _load_balance_state(torch))

    assert isinstance(result, list)
    assert metrics.record_metric.call_count == 2
    recorded = {call.args[0]: call.kwargs for call in metrics.record_metric.call_args_list}
    # The load is already reduced upstream. With two EP ranks, [[1, 3], [2, 2]]
    # gives rank means [2, 2] (avg 4) and rank maxima [3, 2] (max 5).
    assert recorded["eplb:load_balance:avg_tokens"]["value"] == 4.0
    assert recorded["eplb:load_balance:max_tokens"]["value"] == 5.0
    assert recorded["eplb:load_balance:avg_tokens"]["labels"] == {"rank": "0", "model": "main"}


def test_eplb_sync_load_handler_does_not_reduce_again(monkeypatch):
    """The hook reuses upstream's reduction instead of repeating the collective."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch.distributed")

    metrics = Mock()
    _install_provider_api(
        monkeypatch,
        MetricType=type("MetricType", (), {"GAUGE": "gauge", "HISTOGRAM": "histogram", "COUNTER": "counter"}),
        get_metric_recorder=lambda: metrics,
    )
    handlers = _load_feature_handlers(monkeypatch, "test_eplb_lb_no_reduce", "eplb")
    ep_group = Mock()
    ep_group.rank.return_value = 0
    ep_group.size.return_value = 2
    _install_ep_group(monkeypatch, ep_group)

    calls = []
    monkeypatch.setattr("torch.distributed.all_reduce", lambda *args, **kwargs: calls.append(args))

    handlers.eplb_sync_load_handler(_sync_load_pass, _load_balance_state(torch))

    assert calls == []
    assert metrics.record_metric.call_count == 2


def test_eplb_sync_load_handler_without_recorder_records_nothing(monkeypatch):
    torch = pytest.importorskip("torch")
    _install_provider_api(
        monkeypatch,
        MetricType=type("MetricType", (), {"GAUGE": "gauge", "HISTOGRAM": "histogram", "COUNTER": "counter"}),
        get_metric_recorder=lambda: None,
    )
    handlers = _load_feature_handlers(monkeypatch, "test_eplb_lb_no_recorder", "eplb")
    ep_group = Mock()
    ep_group.rank.return_value = 0
    ep_group.size.return_value = 2
    _install_ep_group(monkeypatch, ep_group)

    assert isinstance(handlers.eplb_sync_load_handler(_sync_load_pass, _load_balance_state(torch)), list)


def test_eplb_sync_load_handler_records_only_on_rank_zero(monkeypatch):
    torch = pytest.importorskip("torch")
    metrics = Mock()
    _install_provider_api(
        monkeypatch,
        MetricType=type("MetricType", (), {"GAUGE": "gauge", "HISTOGRAM": "histogram", "COUNTER": "counter"}),
        get_metric_recorder=lambda: metrics,
    )
    handlers = _load_feature_handlers(monkeypatch, "test_eplb_lb_rank", "eplb")
    ep_group = Mock()
    ep_group.rank.return_value = 1
    ep_group.size.return_value = 2
    _install_ep_group(monkeypatch, ep_group)

    handlers.eplb_sync_load_handler(_sync_load_pass, _load_balance_state(torch))
    metrics.record_metric.assert_not_called()


def test_eplb_sync_load_handler_drops_zero_load_steps(monkeypatch):
    """Idle or dummy logging steps still reach the hook but record nothing."""
    torch = pytest.importorskip("torch")
    metrics = Mock()
    _install_provider_api(
        monkeypatch,
        MetricType=type("MetricType", (), {"GAUGE": "gauge", "HISTOGRAM": "histogram", "COUNTER": "counter"}),
        get_metric_recorder=lambda: metrics,
    )
    handlers = _load_feature_handlers(monkeypatch, "test_eplb_lb_zero", "eplb")
    ep_group = Mock()
    ep_group.rank.return_value = 0
    ep_group.size.return_value = 2
    _install_ep_group(monkeypatch, ep_group)

    handlers.eplb_sync_load_handler(_sync_load_pass, _load_balance_state(torch, load=torch.zeros(2, 2)))
    metrics.record_metric.assert_not_called()


def test_eplb_sync_load_handler_passes_empty_result_through(monkeypatch):
    """A state without models yields an empty list and records nothing."""
    pytest.importorskip("torch")
    metrics = Mock()
    _install_provider_api(
        monkeypatch,
        MetricType=type("MetricType", (), {"GAUGE": "gauge", "HISTOGRAM": "histogram", "COUNTER": "counter"}),
        get_metric_recorder=lambda: metrics,
    )
    handlers = _load_feature_handlers(monkeypatch, "test_eplb_lb_empty", "eplb")
    ep_group = Mock()
    ep_group.rank.return_value = 0
    ep_group.size.return_value = 2
    _install_ep_group(monkeypatch, ep_group)
    empty = type("State", (), {"model_states": {}})()

    assert handlers.eplb_sync_load_handler(_sync_load_pass, empty) == []
    metrics.record_metric.assert_not_called()


# ---------------------------------------------------------------------------
# EPLB V2 weight transfer (AscendHixlEplbCommunicator)
# ---------------------------------------------------------------------------


def test_eplb_transfer_handlers_record_per_layer_totals(monkeypatch):
    metrics = Mock()
    _install_provider_api(
        monkeypatch,
        MetricType=type("MetricType", (), {"GAUGE": "gauge", "HISTOGRAM": "histogram", "COUNTER": "counter"}),
        get_metric_recorder=lambda: metrics,
    )
    handlers = _load_feature_handlers(monkeypatch, "test_eplb_transfer", "eplb")
    ep_group = Mock()
    ep_group.rank.return_value = 0
    ep_group.size.return_value = 2
    _install_ep_group(monkeypatch, ep_group)
    communicator = type("Communicator", (), {})()

    assert (
        handlers.eplb_transfer_context_handler(
            lambda *args, **kwargs: "context",
            communicator,
            None,
            3,
        )
        == "context"
    )
    send_tensors = [type("Tensor", (), {"nbytes": 8})(), type("Tensor", (), {"nbytes": 4})()]
    recv_tensors = [type("Tensor", (), {"nbytes": 16})()]
    for expert_id in (5, 6, 5):
        assert (
            handlers.eplb_transfer_send_handler(
                lambda *args, **kwargs: "sent",
                communicator,
                send_tensors,
                1,
                expert_id,
            )
            == "sent"
        )
    assert (
        handlers.eplb_transfer_recv_handler(
            lambda *args, **kwargs: "received",
            communicator,
            recv_tensors,
            0,
            7,
        )
        == "received"
    )
    assert handlers.eplb_transfer_execute_handler(lambda *args, **kwargs: "done", communicator) == "done"

    assert metrics.record_metric.call_count == 4
    recorded = {call.args[0]: call.kwargs for call in metrics.record_metric.call_args_list}
    assert recorded["eplb:transfer:send_experts"]["value"] == 2
    assert recorded["eplb:transfer:recv_experts"]["value"] == 1
    # Requests are counted per peer rank, not per descriptor: the three
    # sends target rank 1 and the receive reads from rank 0, so 2 requests.
    # Bytes still accumulate per tensor: 3 x (8 + 4) + 16 = 52.
    assert recorded["eplb:transfer:comm_requests"]["value"] == 2
    assert recorded["eplb:transfer:est_bytes"]["value"] == 52
    assert recorded["eplb:transfer:send_experts"]["labels"] == {"rank": "0", "phase": "all", "layer": "3"}


def test_eplb_transfer_batches_descriptors_per_peer(monkeypatch):
    """Two experts from one source rank still issue a single transfer request."""
    metrics = Mock()
    _install_provider_api(
        monkeypatch,
        MetricType=type("MetricType", (), {"GAUGE": "gauge", "HISTOGRAM": "histogram", "COUNTER": "counter"}),
        get_metric_recorder=lambda: metrics,
    )
    handlers = _load_feature_handlers(monkeypatch, "test_eplb_transfer_batch", "eplb")
    ep_group = Mock()
    ep_group.rank.return_value = 0
    ep_group.size.return_value = 2
    _install_ep_group(monkeypatch, ep_group)
    communicator = type("Communicator", (), {})()
    tensors = [type("Tensor", (), {"nbytes": 16})()]

    handlers.eplb_transfer_context_handler(lambda *args, **kwargs: "context", communicator, None, 5)
    for expert_id in (0, 1):
        handlers.eplb_transfer_recv_handler(
            lambda *args, **kwargs: "received",
            communicator,
            tensors,
            3,
            expert_id,
        )
    handlers.eplb_transfer_execute_handler(lambda *args, **kwargs: "done", communicator)

    recorded = {call.args[0]: call.kwargs for call in metrics.record_metric.call_args_list}
    assert recorded["eplb:transfer:recv_experts"]["value"] == 2
    assert recorded["eplb:transfer:comm_requests"]["value"] == 1
    assert recorded["eplb:transfer:est_bytes"]["value"] == 32


def test_eplb_transfer_execute_without_context_records_nothing(monkeypatch):
    metrics = Mock()
    _install_provider_api(
        monkeypatch,
        MetricType=type("MetricType", (), {"GAUGE": "gauge", "HISTOGRAM": "histogram", "COUNTER": "counter"}),
        get_metric_recorder=lambda: metrics,
    )
    handlers = _load_feature_handlers(monkeypatch, "test_eplb_transfer_no_ctx", "eplb")
    ep_group = Mock()
    ep_group.rank.return_value = 0
    ep_group.size.return_value = 2
    _install_ep_group(monkeypatch, ep_group)
    communicator = type("Communicator", (), {})()

    assert handlers.eplb_transfer_execute_handler(lambda *args, **kwargs: "done", communicator) == "done"

    metrics.record_metric.assert_not_called()


def test_eplb_transfer_execute_failure_clears_pending_accumulator(monkeypatch):
    metrics = Mock()
    _install_provider_api(
        monkeypatch,
        MetricType=type("MetricType", (), {"GAUGE": "gauge", "HISTOGRAM": "histogram", "COUNTER": "counter"}),
        get_metric_recorder=lambda: metrics,
    )
    handlers = _load_feature_handlers(monkeypatch, "test_eplb_transfer_fail", "eplb")
    ep_group = Mock()
    ep_group.rank.return_value = 0
    ep_group.size.return_value = 2
    _install_ep_group(monkeypatch, ep_group)
    communicator = type("Communicator", (), {})()

    handlers.eplb_transfer_context_handler(
        lambda *args, **kwargs: "context",
        communicator,
        None,
        3,
    )

    def failing_execute(*args, **kwargs):
        raise RuntimeError("execute failed")

    with pytest.raises(RuntimeError):
        handlers.eplb_transfer_execute_handler(failing_execute, communicator)

    # The accumulator is dropped, so nothing is recorded and a later execute
    # on the same communicator starts from a clean state.
    metrics.record_metric.assert_not_called()
    assert communicator not in handlers._pending


def test_eplb_transfer_handler_registers_metrics_once_per_recorder(monkeypatch):
    """Metric registration is cached per recorder and follows a recorder swap.

    ``register_metrics`` keeps one registration per (spec tuple, recorder)
    pair: the same recorder never registers twice, while a new recorder
    instance registers its own metrics and records exactly once.
    """
    metric_type = type("MetricType", (), {"GAUGE": "gauge", "HISTOGRAM": "histogram", "COUNTER": "counter"})
    first_metrics = Mock()
    current_metrics = [first_metrics]
    _install_provider_api(
        monkeypatch,
        MetricType=metric_type,
        get_metric_recorder=lambda: current_metrics[0],
    )
    handlers = _load_feature_handlers(monkeypatch, "test_eplb_transfer_recorder_cache", "eplb")
    ep_group = Mock()
    ep_group.rank.return_value = 0
    ep_group.size.return_value = 2
    _install_ep_group(monkeypatch, ep_group)
    communicator = type("Communicator", (), {})()
    tensors = [type("Tensor", (), {"nbytes": 8})()]

    def run_transfer():
        assert (
            handlers.eplb_transfer_context_handler(
                lambda *args, **kwargs: "context",
                communicator,
                None,
                3,
            )
            == "context"
        )
        assert (
            handlers.eplb_transfer_send_handler(
                lambda *args, **kwargs: "sent",
                communicator,
                tensors,
                1,
                5,
            )
            == "sent"
        )
        assert handlers.eplb_transfer_execute_handler(lambda *args, **kwargs: "done", communicator) == "done"

    run_transfer()
    run_transfer()

    # Both executes shared the first recorder's single registration.
    assert first_metrics.get_or_create_metric.call_count == 4
    assert first_metrics.record_metric.call_count == 8

    second_metrics = Mock()
    current_metrics[0] = second_metrics
    run_transfer()

    # A new recorder instance registers its own metrics and records once.
    assert second_metrics.get_or_create_metric.call_count == 4
    assert second_metrics.record_metric.call_count == 4


def test_runtime_guard_handler_counts_alerts_by_incident_type(monkeypatch):
    metric_type = type("MetricType", (), {"GAUGE": "gauge", "HISTOGRAM": "histogram", "COUNTER": "counter"})
    metrics = Mock()
    _install_provider_api(
        monkeypatch,
        MetricType=metric_type,
        get_metric_recorder=lambda: metrics,
    )
    handlers = _load_feature_handlers(
        monkeypatch,
        "test_vllm_ascend_metric_handlers_runtime_guard",
        "runtime_guard",
    )
    alert = type("Alert", (), {"incident_type": "token_repeat"})()

    result = handlers.runtime_guard_handle_alert_handler(
        lambda *_a, **_k: "armed",
        object(),
        alert,
    )

    assert result == "armed"
    metrics.get_or_create_metric.assert_called_once_with(
        metric_name="runtime_guard:anomaly:count",
        metric_type="counter",
        label_names=["incident_type"],
    )
    metrics.record_metric.assert_called_once_with(
        "runtime_guard:anomaly:count",
        value=1.0,
        labels={"incident_type": "token_repeat"},
    )


def test_runtime_guard_handler_skips_noop_early_return(monkeypatch):
    metric_type = type("MetricType", (), {"GAUGE": "gauge", "HISTOGRAM": "histogram", "COUNTER": "counter"})
    metrics = Mock()
    _install_provider_api(
        monkeypatch,
        MetricType=metric_type,
        get_metric_recorder=lambda: metrics,
    )
    handlers = _load_feature_handlers(
        monkeypatch,
        "test_vllm_ascend_metric_handlers_runtime_guard_noop",
        "runtime_guard",
    )
    alert = type("Alert", (), {"incident_type": "logits_finite"})()

    result = handlers.runtime_guard_handle_alert_handler(
        lambda *_a, **_k: None,
        object(),
        alert,
        write_report=False,
    )

    assert result is None
    metrics.record_metric.assert_not_called()


def test_runtime_guard_handler_preserves_result_when_recording_fails(monkeypatch):
    metric_type = type("MetricType", (), {"GAUGE": "gauge", "HISTOGRAM": "histogram", "COUNTER": "counter"})
    metrics = Mock()
    metrics.get_or_create_metric.side_effect = RuntimeError("registry unavailable")
    _install_provider_api(
        monkeypatch,
        MetricType=metric_type,
        get_metric_recorder=lambda: metrics,
    )
    handlers = _load_feature_handlers(
        monkeypatch,
        "test_vllm_ascend_metric_handlers_runtime_guard_fail",
        "runtime_guard",
    )
    alert = type("Alert", (), {"incident_type": "rare_chars"})()

    assert (
        handlers.runtime_guard_handle_alert_handler(
            lambda *_a, **_k: "armed",
            object(),
            alert,
        )
        == "armed"
    )
