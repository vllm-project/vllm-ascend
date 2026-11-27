# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import importlib.util
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
        "runtime_guard_metrics.yaml",
        "schedule_execute_metrics.yaml",
    ]
    assert all(Path(path).is_file() for path in provider.config_paths)
    config = _load_all_provider_configs(provider.config_paths)
    assert len(config) == 8
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
        "vllm_ascend.observability.ms_metrics.handlers.runtime_guard",
        "vllm_ascend.observability.ms_metrics.handlers.schedule_execute",
    ):
        monkeypatch.delitem(sys.modules, module_name, raising=False)

    handler_module = __import__(
        _HANDLER_PACKAGE,
        fromlist=[
            "runtime_guard_handle_alert_handler",
        ],
    )

    assert callable(handler_module.runtime_guard_handle_alert_handler)


def test_provider_yaml_symbols_exist_in_current_vllm_ascend_source():
    config_paths = sorted((_PACKAGE_ROOT / "config").glob("*.yaml"))
    assert len(config_paths) == 3
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
