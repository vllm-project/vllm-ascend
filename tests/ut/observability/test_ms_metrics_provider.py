# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import importlib.util
import logging
import sys
from dataclasses import dataclass
from importlib.metadata import EntryPoint
from pathlib import Path
from types import ModuleType
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
        "eplb_v2_metrics.yaml",
        "runtime_guard_metrics.yaml",
        "schedule_execute_metrics.yaml",
    ]
    assert all(Path(path).is_file() for path in provider.config_paths)
    config = _load_all_provider_configs(provider.config_paths)
    assert len(config) == 18
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
        "vllm_ascend.observability.ms_metrics.handlers.eplb_v2",
        "vllm_ascend.observability.ms_metrics.handlers.runtime_guard",
        "vllm_ascend.observability.ms_metrics.handlers.schedule_execute",
    ):
        monkeypatch.delitem(sys.modules, module_name, raising=False)

    handler_module = __import__(
        _HANDLER_PACKAGE,
        fromlist=[
            "eplb_do_update_hotness_handler",
            "runtime_guard_handle_alert_handler",
        ],
    )

    assert callable(handler_module.eplb_do_update_hotness_handler)
    assert callable(handler_module.runtime_guard_handle_alert_handler)


def test_provider_yaml_symbols_exist_in_current_vllm_ascend_source():
    config_paths = sorted((_PACKAGE_ROOT / "config").glob("*.yaml"))
    assert len(config_paths) == 5
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


def test_eplb_handler_records_rank_zero_hotness(monkeypatch):
    metric_type = type("MetricType", (), {"GAUGE": "gauge", "HISTOGRAM": "histogram", "COUNTER": "counter"})
    metrics = Mock()
    _install_provider_api(
        monkeypatch,
        MetricType=metric_type,
        get_metric_recorder=lambda: metrics,
    )
    handlers = _load_feature_handlers(monkeypatch, "test_vllm_ascend_metric_handlers", "eplb")
    worker = type(
        "Worker",
        (),
        {
            "rank_id": 0,
            "latest_expert_hotness": {
                "current_mean": 2.0,
                "current_max": 4.0,
                "update_mean": 3.0,
                "update_max": 6.0,
                "current_imbalance_list": [1.1, 1.2],
                "update_imbalance_list": [1.3, 1.4],
            },
        },
    )()

    result = handlers.eplb_do_update_hotness_handler(lambda _: "updated", worker)

    assert result == "updated"
    assert metrics.get_or_create_metric.call_count == 5
    metrics.get_or_create_metric.assert_any_call(
        metric_name="eplb:expert_hotness:current_mean",
        metric_type="gauge",
        label_names=["rank", "phase"],
    )
    metrics.get_or_create_metric.assert_any_call(
        metric_name="eplb:expert_hotness:imbalance",
        metric_type="gauge",
        label_names=["rank", "phase", "layer"],
    )
    assert metrics.record_metric.call_count == 8


def test_eplb_handler_registers_metrics_once_per_recorder(monkeypatch):
    metric_type = type("MetricType", (), {"GAUGE": "gauge", "HISTOGRAM": "histogram", "COUNTER": "counter"})
    first_metrics = Mock()
    current_metrics = [first_metrics]
    _install_provider_api(
        monkeypatch,
        MetricType=metric_type,
        get_metric_recorder=lambda: current_metrics[0],
    )
    handlers = _load_feature_handlers(monkeypatch, "test_vllm_ascend_metric_handlers_cache", "eplb")
    worker = type(
        "Worker",
        (),
        {
            "rank_id": 0,
            "latest_expert_hotness": {"current_mean": 1.0},
        },
    )()

    handlers.eplb_do_update_hotness_handler(lambda _: "updated", worker)
    handlers.eplb_do_update_hotness_handler(lambda _: "updated", worker)

    assert first_metrics.get_or_create_metric.call_count == 5
    assert first_metrics.record_metric.call_count == 2

    second_metrics = Mock()
    current_metrics[0] = second_metrics
    handlers.eplb_do_update_hotness_handler(lambda _: "updated", worker)

    assert second_metrics.get_or_create_metric.call_count == 5
    second_metrics.record_metric.assert_called_once()


def test_eplb_handler_skips_nonzero_rank(monkeypatch):
    metric_type = type("MetricType", (), {"GAUGE": "gauge", "HISTOGRAM": "histogram", "COUNTER": "counter"})
    metrics = Mock()
    _install_provider_api(
        monkeypatch,
        MetricType=metric_type,
        get_metric_recorder=lambda: metrics,
    )
    handlers = _load_feature_handlers(monkeypatch, "test_vllm_ascend_metric_handlers_nonzero", "eplb")
    worker = type("Worker", (), {"rank_id": 1})()

    assert handlers.eplb_do_update_hotness_handler(lambda _: "updated", worker) == "updated"
    metrics.record_metric.assert_not_called()


def test_eplb_handler_skips_when_recorder_unavailable(monkeypatch, caplog):
    metric_type = type("MetricType", (), {"GAUGE": "gauge", "HISTOGRAM": "histogram", "COUNTER": "counter"})
    _install_provider_api(
        monkeypatch,
        MetricType=metric_type,
        get_metric_recorder=lambda: None,  # ← when recorder is None
    )
    handlers = _load_feature_handlers(monkeypatch, "test_vllm_ascend_metric_handlers_no_recorder", "eplb")
    worker = type(
        "Worker",
        (),
        {
            "rank_id": 0,
            "latest_expert_hotness": {"current_mean": 1.0},
        },
    )()

    result = handlers.eplb_do_update_hotness_handler(lambda _: "updated", worker)

    assert result == "updated"
    with caplog.at_level(logging.WARNING):
        result = handlers.eplb_do_update_hotness_handler(lambda _: "updated", worker)

    assert result == "updated"
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_eplb_handler_given_metric_failure_then_preserves_inference_result(monkeypatch):
    metric_type = type("MetricType", (), {"GAUGE": "gauge", "HISTOGRAM": "histogram", "COUNTER": "counter"})
    metrics = Mock()
    metrics.get_or_create_metric.side_effect = RuntimeError("registry unavailable")
    _install_provider_api(
        monkeypatch,
        MetricType=metric_type,
        get_metric_recorder=lambda: metrics,
    )
    handlers = _load_feature_handlers(monkeypatch, "test_vllm_ascend_metric_handlers_failure", "eplb")
    worker = type(
        "Worker",
        (),
        {
            "rank_id": 0,
            "latest_expert_hotness": {"current_mean": 1.0},
        },
    )()

    assert (
        handlers.eplb_do_update_hotness_handler(
            lambda _: "updated",
            worker,
        )
        == "updated"
    )


def _hotness_worker():
    return type(
        "Worker",
        (),
        {
            "rank_id": 0,
            "latest_expert_hotness": {
                "current_mean": 2.0,
                "current_max": 4.0,
                "update_mean": 3.0,
                "update_max": 6.0,
                "current_imbalance_list": [1.1, 1.2],
                "update_imbalance_list": [1.3, 1.4],
            },
        },
    )()


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


def test_eplb_handler_skips_higher_level_series_until_level_rises(monkeypatch):
    metrics = Mock()
    _install_provider_api(
        monkeypatch,
        MetricType=type("MetricType", (), {"GAUGE": "gauge", "HISTOGRAM": "histogram", "COUNTER": "counter"}),
        get_metric_recorder=lambda: metrics,
    )
    handlers = _load_feature_handlers(monkeypatch, "test_vllm_ascend_metric_handlers_level_rise", "eplb")
    common = sys.modules[_COMMON_MODULE]
    imbalance = common.MetricSpec(
        handlers._IMBALANCE_METRIC,
        handlers.MetricType.GAUGE,
        ("rank", "phase", "layer"),
        level=common.METRIC_LEVEL_DETAIL,
    )
    handlers._IMBALANCE_SPEC = imbalance
    handlers._EPLB_METRIC_SPECS = (*handlers._SUMMARY_SPECS, imbalance)
    worker = _hotness_worker()

    handlers.eplb_do_update_hotness_handler(lambda _: "updated", worker)

    assert metrics.get_or_create_metric.call_count == 4
    recorded_names = [call.args[0] for call in metrics.record_metric.call_args_list]
    assert recorded_names == [
        "eplb:expert_hotness:current_mean",
        "eplb:expert_hotness:current_max",
        "eplb:expert_hotness:update_mean",
        "eplb:expert_hotness:update_max",
    ]

    common.set_metric_level("detail")
    handlers.eplb_do_update_hotness_handler(lambda _: "updated", worker)

    assert metrics.get_or_create_metric.call_count == 9
    recorded_names = [call.args[0] for call in metrics.record_metric.call_args_list]
    assert recorded_names.count("eplb:expert_hotness:imbalance") == 4


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
    package = ModuleType("vllm")
    distributed = ModuleType("vllm.distributed")
    parallel_state = ModuleType("vllm.distributed.parallel_state")
    parallel_state.get_ep_group = lambda: type("Group", (), {"device_group": ep_group})()
    package.distributed = distributed
    distributed.parallel_state = parallel_state
    monkeypatch.setitem(sys.modules, "vllm", package)
    monkeypatch.setitem(sys.modules, "vllm.distributed", distributed)
    monkeypatch.setitem(sys.modules, "vllm.distributed.parallel_state", parallel_state)


def _install_all_reduce(monkeypatch, factor: float = 1.0):
    """Stub the EP-group reduction; ``factor`` scales the reduced tensor."""
    import torch
    import torch.distributed

    def fake_all_reduce(tensor, op=None, group=None):
        if factor != 1.0:
            tensor.mul_(factor)
        return None

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)


def _load_balance_state(torch):
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
                        "expert_load_pass": torch.tensor([[1.0, 3.0], [2.0, 2.0]]),
                    },
                )()
            }
        },
    )()


def _clearing_step(state, *args, **kwargs):
    """Mirror upstream on a balancedness logging step: clear the buffer."""
    for model_state in state.model_states.values():
        model_state.expert_load_pass.zero_()
    return "stepped"


def test_eplb_load_balance_handler_records_reduced_load(monkeypatch):
    torch = pytest.importorskip("torch")
    metrics = Mock()
    _install_provider_api(
        monkeypatch,
        MetricType=type("MetricType", (), {"GAUGE": "gauge", "HISTOGRAM": "histogram", "COUNTER": "counter"}),
        get_metric_recorder=lambda: metrics,
    )
    handlers = _load_feature_handlers(monkeypatch, "test_eplb_v2_load_balance", "eplb_v2")
    ep_group = Mock()
    ep_group.rank.return_value = 0
    ep_group.size.return_value = 2
    _install_ep_group(monkeypatch, ep_group)
    # The reduction doubles the snapshot, so the recorded values prove they
    # come from the reduced tensor instead of the raw pre-step buffer.
    _install_all_reduce(monkeypatch, factor=2.0)

    result = handlers.eplb_load_balance_handler(
        _clearing_step,
        _load_balance_state(torch),
        False,
        False,
        True,
    )

    assert result == "stepped"
    assert metrics.record_metric.call_count == 3
    recorded = {call.args[0]: call.kwargs for call in metrics.record_metric.call_args_list}
    assert recorded["eplb:load_balance:avg_tokens"]["value"] == 8.0
    assert recorded["eplb:load_balance:max_tokens"]["value"] == 10.0
    assert recorded["eplb:load_balance:balancedness"]["value"] == pytest.approx(0.8)
    assert recorded["eplb:load_balance:avg_tokens"]["labels"] == {"rank": "0", "model": "main"}


def test_eplb_load_balance_handler_reduces_even_without_recorder(monkeypatch):
    """A rank without a recorder still joins the EP reduction.

    The recorder gate must sit after the collective loop: returning before
    the reduction on such a rank would leave the other EP ranks blocked
    inside all_reduce.
    """
    torch = pytest.importorskip("torch")

    _install_provider_api(
        monkeypatch,
        MetricType=type("MetricType", (), {"GAUGE": "gauge", "HISTOGRAM": "histogram", "COUNTER": "counter"}),
        get_metric_recorder=lambda: None,
    )
    handlers = _load_feature_handlers(monkeypatch, "test_eplb_v2_lb_no_recorder", "eplb_v2")
    ep_group = Mock()
    ep_group.rank.return_value = 0
    ep_group.size.return_value = 2
    _install_ep_group(monkeypatch, ep_group)

    calls = []
    monkeypatch.setattr(
        torch.distributed,
        "all_reduce",
        lambda tensor, op=None, group=None: calls.append(tensor),
    )

    assert (
        handlers.eplb_load_balance_handler(_clearing_step, _load_balance_state(torch), False, False, True) == "stepped"
    )
    # The reduction ran even though no recorder is available.
    assert len(calls) == 1


def test_eplb_load_balance_handler_skips_non_logging_steps(monkeypatch):
    """Only steps that consume ``expert_load_pass`` record, like upstream."""
    torch = pytest.importorskip("torch")
    metrics = Mock()
    _install_provider_api(
        monkeypatch,
        MetricType=type("MetricType", (), {"GAUGE": "gauge", "HISTOGRAM": "histogram", "COUNTER": "counter"}),
        get_metric_recorder=lambda: metrics,
    )
    handlers = _load_feature_handlers(monkeypatch, "test_eplb_v2_lb_skip", "eplb_v2")
    ep_group = Mock()
    ep_group.rank.return_value = 0
    ep_group.size.return_value = 2
    _install_ep_group(monkeypatch, ep_group)
    _install_all_reduce(monkeypatch)

    # log_stats=False: no snapshot is taken even though the step clears the
    # buffer.
    assert (
        handlers.eplb_load_balance_handler(_clearing_step, _load_balance_state(torch), True, False, False) == "stepped"
    )
    metrics.record_metric.assert_not_called()

    # log_stats=True but the step left the buffer intact: outside the
    # balancedness logging interval upstream logs nothing, and neither does
    # the handler.
    state = _load_balance_state(torch)

    def idle_step(state, *args, **kwargs):
        return "idle"

    assert handlers.eplb_load_balance_handler(idle_step, state, False, False, True) == "idle"
    metrics.record_metric.assert_not_called()


def test_eplb_load_balance_handler_records_only_on_rank_zero(monkeypatch):
    torch = pytest.importorskip("torch")
    metrics = Mock()
    _install_provider_api(
        monkeypatch,
        MetricType=type("MetricType", (), {"GAUGE": "gauge", "HISTOGRAM": "histogram", "COUNTER": "counter"}),
        get_metric_recorder=lambda: metrics,
    )
    handlers = _load_feature_handlers(monkeypatch, "test_eplb_v2_lb_rank", "eplb_v2")
    ep_group = Mock()
    ep_group.rank.return_value = 1
    ep_group.size.return_value = 2
    _install_ep_group(monkeypatch, ep_group)
    _install_all_reduce(monkeypatch)

    assert (
        handlers.eplb_load_balance_handler(_clearing_step, _load_balance_state(torch), False, False, True) == "stepped"
    )
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
    handlers = _load_feature_handlers(monkeypatch, "test_eplb_v2_transfer", "eplb_v2")
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
    # Both directions accumulate per call: 3 send calls x 2 tensors + 1 recv
    # tensor = 7 ops; 3 x (8 + 4) + 16 = 52 estimated bytes.
    assert recorded["eplb:transfer:comm_ops"]["value"] == 7
    assert recorded["eplb:transfer:est_bytes"]["value"] == 52
    assert recorded["eplb:transfer:send_experts"]["labels"] == {"rank": "0", "layer": "3"}


def test_eplb_transfer_execute_without_context_records_nothing(monkeypatch):
    metrics = Mock()
    _install_provider_api(
        monkeypatch,
        MetricType=type("MetricType", (), {"GAUGE": "gauge", "HISTOGRAM": "histogram", "COUNTER": "counter"}),
        get_metric_recorder=lambda: metrics,
    )
    handlers = _load_feature_handlers(monkeypatch, "test_eplb_v2_transfer_no_ctx", "eplb_v2")
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
    handlers = _load_feature_handlers(monkeypatch, "test_eplb_v2_transfer_fail", "eplb_v2")
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
