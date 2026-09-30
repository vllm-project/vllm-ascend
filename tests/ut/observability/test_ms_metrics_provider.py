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

import yaml

_REPOSITORY_ROOT = Path(__file__).parents[3]
_PACKAGE_ROOT = _REPOSITORY_ROOT / "vllm_ascend" / "observability"


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
    assert provider.handler_module_prefixes == ("vllm_ascend.observability.",)
    assert provider.config_paths == tuple(sorted(provider.config_paths))
    assert [Path(path).name for path in provider.config_paths] == [
        "base_metrics.yaml",
        "eplb_metrics.yaml",
    ]
    assert all(Path(path).is_file() for path in provider.config_paths)
    config = _load_all_provider_configs(provider.config_paths)
    assert len(config) == 12
    assert all(item["symbol"].startswith("vllm_ascend.") for item in config)
    assert all("id" not in item for item in config)
    assert all(
        item.get("handler", "").startswith(
            (
                "ms_service_metric.provider_handlers:",
                "vllm_ascend.observability.handlers:",
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
        "vllm-ascend = vllm_ascend.observability:get_metric_provider"
    ]
    assert package_data["vllm_ascend.observability"] == ["config/*.yaml"]


def test_provider_entry_point_loads_without_metric_core(monkeypatch):
    package = ModuleType("vllm_ascend")
    package.__path__ = [str(_REPOSITORY_ROOT / "vllm_ascend")]
    monkeypatch.setitem(sys.modules, "vllm_ascend", package)
    monkeypatch.delitem(sys.modules, "ms_service_metric", raising=False)
    monkeypatch.delitem(sys.modules, "ms_service_metric.provider_api", raising=False)

    entry_point = EntryPoint(
        name="vllm-ascend",
        value="vllm_ascend.observability:get_metric_provider",
        group="ms_service_metric.providers",
    )

    assert callable(entry_point.load())


def test_ascend_handler_module_loads_from_yaml_path(monkeypatch):
    metric_type = type("MetricType", (), {"GAUGE": "gauge"})
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
        "vllm_ascend.observability.provider",
        "vllm_ascend.observability.handlers",
    ):
        monkeypatch.delitem(sys.modules, module_name, raising=False)

    handler_module = __import__(
        "vllm_ascend.observability.handlers",
        fromlist=["eplb_do_update_hotness_handler"],
    )

    assert callable(handler_module.eplb_do_update_hotness_handler)


def test_provider_yaml_symbols_exist_in_current_vllm_ascend_source():
    config_paths = sorted((_PACKAGE_ROOT / "config").glob("*.yaml"))
    assert len(config_paths) == 2
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
    metric_type = type("MetricType", (), {"GAUGE": "gauge"})
    metrics = Mock()
    _install_provider_api(
        monkeypatch,
        MetricType=metric_type,
        get_metric_recorder=lambda: metrics,
    )
    handlers = _load_source("test_vllm_ascend_metric_handlers", "handlers.py")
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
        "eplb:expert_hotness:current_mean",
        metric_type="gauge",
        label_names=["rank", "phase"],
    )
    metrics.get_or_create_metric.assert_any_call(
        "eplb:expert_hotness:imbalance",
        metric_type="gauge",
        label_names=["rank", "phase", "layer"],
    )
    assert metrics.record_metric.call_count == 8


def test_eplb_handler_registers_metrics_once_per_recorder(monkeypatch):
    metric_type = type("MetricType", (), {"GAUGE": "gauge"})
    first_metrics = Mock()
    current_metrics = [first_metrics]
    _install_provider_api(
        monkeypatch,
        MetricType=metric_type,
        get_metric_recorder=lambda: current_metrics[0],
    )
    handlers = _load_source("test_vllm_ascend_metric_handlers_cache", "handlers.py")
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
    metric_type = type("MetricType", (), {"GAUGE": "gauge"})
    metrics = Mock()
    _install_provider_api(
        monkeypatch,
        MetricType=metric_type,
        get_metric_recorder=lambda: metrics,
    )
    handlers = _load_source("test_vllm_ascend_metric_handlers_nonzero", "handlers.py")
    worker = type("Worker", (), {"rank_id": 1})()

    assert handlers.eplb_do_update_hotness_handler(lambda _: "updated", worker) == "updated"
    metrics.record_metric.assert_not_called()


def test_eplb_handler_given_metric_failure_then_preserves_inference_result(monkeypatch):
    metric_type = type("MetricType", (), {"GAUGE": "gauge"})
    metrics = Mock()
    metrics.get_or_create_metric.side_effect = RuntimeError("registry unavailable")
    _install_provider_api(
        monkeypatch,
        MetricType=metric_type,
        get_metric_recorder=lambda: metrics,
    )
    handlers = _load_source("test_vllm_ascend_metric_handlers_failure", "handlers.py")
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
