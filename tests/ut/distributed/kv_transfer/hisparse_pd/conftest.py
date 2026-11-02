# SPDX-License-Identifier: Apache-2.0
"""CPU-load pure P/D modules without importing torch_npu or initializing vLLM.

The base/MultiConnector/aggregator classes are compiled from the actual paired
upstream source, not reimplemented. Only unrelated import-time dependencies are
replaced. Hardware transport remains a separately required test.
"""

import ast
import enum
import importlib
import logging
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[5]
UPSTREAM = REPO.parent / "vllm"


def load_classes(path, names, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    nodes += [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), "exec"), namespace)


@pytest.fixture(scope="session")
def modules():
    package = ModuleType("pd_under_test")
    package.__path__ = [str(REPO / "vllm_ascend/distributed/kv_transfer/kv_p2p/hisparse_pd")]
    sys.modules[package.__name__] = package
    root = UPSTREAM / "vllm/distributed/kv_transfer/kv_connector"
    base = ModuleType("vllm.distributed.kv_transfer.kv_connector.v1.base")
    base.__dict__.update(
        ABC=ABC,
        abstractmethod=abstractmethod,
        dataclass=dataclass,
        field=field,
        enum=enum,
        logger=logging.getLogger("pd-cpu"),
        CacheHitSource=SimpleNamespace(EXTERNAL_UNSPECIFIED="external"),
    )
    sys.modules[base.__name__] = base
    load_classes(
        root / "v1/base.py",
        {
            "KVConnectorTransferResults",
            "SupportsHMA",
            "KVConnectorRole",
            "KVConnectorMetadata",
            "KVConnectorWorkerMetadata",
            "KVConnectorBase_V1",
        },
        base.__dict__,
    )
    multi = ModuleType("pd_actual_multi")
    multi.__dict__.update(base.__dict__)
    multi.__dict__["__name__"] = multi.__name__
    sys.modules[multi.__name__] = multi
    load_classes(
        root / "v1/multi_connector.py",
        {
            "MultiKVConnectorMetadata",
            "MultiKVConnectorWorkerMetadata",
            "MultiConnector",
        },
        multi.__dict__,
    )
    aggregator = {"KVConnectorOutput": lambda **kwargs: SimpleNamespace(**kwargs)}
    load_classes(root / "utils.py", {"KVOutputAggregator"}, aggregator)
    result = SimpleNamespace(base=base, multi=multi, aggregator=aggregator["KVOutputAggregator"])
    scheduler = {}
    load_classes(root / "v1/hisparse/connector.py", {"HiSparseConnectorScheduler"}, scheduler)
    result.hisparse_scheduler = scheduler["HiSparseConnectorScheduler"]
    coordinator_path = UPSTREAM / "vllm/v1/hisparse/coordinator.py"
    coordinator = next(
        node
        for node in ast.parse(coordinator_path.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "HiSparseCoordinator"
    )
    methods = [
        node
        for node in coordinator.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"record_pending_host_import", "finish_host_import", "complete_host_import"}
    ]
    namespace = {}
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[])), str(coordinator_path), "exec"),
        namespace,
    )
    result.import_contract = type("ActualUpstreamImportContract", (), namespace)
    for name in ("protocol", "layout", "transport", "worker", "connector"):
        setattr(result, name, importlib.import_module(f"pd_under_test.{name}"))
    from unittest.mock import Mock

    class ChildConstructionOnly:
        def __init__(self, vllm_config, role, kv_cache_config):
            self._connectors = vllm_config.test_children

        def shutdown(self):
            for child in self._connectors:
                child.shutdown()

        def unregister_kv_caches(self):
            for child in self._connectors:
                child.unregister_kv_caches()

    ascend = {
        "MultiConnector": ChildConstructionOnly,
        "SupportsHMA": base.SupportsHMA,
        "uses_sfa_dspark_kv_transfer": lambda _: False,
        "supports_hma": lambda _: True,
        "mamba_utils": Mock(),
    }
    load_classes(
        REPO / "vllm_ascend/distributed/kv_transfer/ascend_multi_connector.py", {"AscendMultiConnector"}, ascend
    )
    result.ascend_multi = ascend["AscendMultiConnector"]
    return result
