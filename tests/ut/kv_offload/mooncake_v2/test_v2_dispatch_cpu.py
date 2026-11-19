# SPDX-License-Identifier: Apache-2.0
"""CPU regression tests for V2 dispatch and composite push allocations."""

import ast
import sys
from dataclasses import dataclass
from pathlib import Path
from types import MethodType, ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import pytest

ROOT = Path(__file__).resolve().parents[4]
PREFIX = "vllm_ascend/distributed/kv_transfer/"


def execute(relative, scope, *, owner=None, method=None):
    nodes = ast.parse((ROOT / (PREFIX + relative)).read_text(encoding="utf-8")).body
    if owner:
        nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == owner).body
    if method:
        nodes = [n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == method]
    else:
        nodes = [n for n in nodes if isinstance(n, (ast.ClassDef, ast.Assign))]
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[])), relative, "exec"), scope
    )
    return scope


def facade():
    return execute(
        "kv_p2p/mooncake/connector.py",
        {"KVConnectorBase_V1": type("Base", (), {}), "SupportsHMA": type("HMA", (), {})},
    )


def registration():
    registry: dict[str, tuple[str, str]] = {}
    factory = SimpleNamespace(
        _registry=registry,
        register_connector=lambda name, module, cls: registry.__setitem__(name, (module, cls)),
    )
    offloading = ModuleType("vllm.v1.kv_offload.factory")
    offloading.OffloadingSpecFactory = SimpleNamespace(_registry={}, register_spec=Mock())  # type: ignore[attr-defined]
    scope: dict[str, Any] = {"KVConnectorFactory": factory, "logger": Mock()}
    execute("__init__.py", scope, method="_register_with_class_name_alias")
    execute("__init__.py", scope, method="register_connector")
    with patch.dict(sys.modules, {offloading.__name__: offloading}):
        scope["register_connector"]()
    return registry


@pytest.mark.parametrize(
    "extra,expected",
    [
        ({}, "MooncakePullConnector"),
        ({"use_layerwise": False}, "MooncakePullConnector"),
        ({"use_layerwise": True}, "MooncakePushConnector"),
    ],
)
def test_registered_v2_selects_requested_protocol(extra, expected):
    classes = facade()
    module, name = registration()["MooncakeConnectorV2"]
    assert module == "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.connector"
    config = SimpleNamespace(kv_transfer_config=SimpleNamespace(kv_connector_extra_config=extra))
    role, cache = object(), object()
    with (
        patch.object(classes["MooncakePushConnector"], "__init__", return_value=None) as push_init,
        patch.object(classes["MooncakePullConnector"], "__init__", return_value=None) as pull_init,
    ):
        result = classes[name](config, role, cache)
    assert type(result) is classes[expected]
    selected = push_init if expected == "MooncakePushConnector" else pull_init
    selected.assert_called_once_with(config, role, cache)
    assert classes["MooncakeConnector"] is classes["MooncakePullConnector"]


@pytest.mark.parametrize("chosen", [-1, 0, 1])
@pytest.mark.parametrize("num_external_tokens", [0, 16])
def test_push_retains_source_blocks_and_prefix_origin(chosen, num_external_tokens):
    classes = facade()
    push = object.__new__(classes["MooncakePushConnector"])
    scheduler = SimpleNamespace(kv_transfer_config=SimpleNamespace(is_kv_producer=True), _reqs_need_send_layerwise={})
    scope = execute(
        "kv_p2p/mooncake/push_scheduler.py",
        {},
        owner="MooncakePushConnectorScheduler",
        method="update_state_after_alloc",
    )
    scheduler.update_state_after_alloc = MethodType(scope["update_state_after_alloc"], scheduler)
    push.connector_scheduler = scheduler
    request = SimpleNamespace(
        request_id="p",
        num_prompt_tokens=64,
        kv_transfer_params=dict(
            do_remote_decode=True,
            remote_request_id="d",
            remote_num_external_tokens=64,
            remote_num_computed_tokens=0,
            remote_engine_id="d-engine",
            remote_host="127.0.0.1",
            remote_port=9000,
            remote_block_ids=([20, 21], [22, 23]),
            remote_worker_addresses=(),
        ),
    )
    block_ids = ([2, 3], [8, 9])
    empty = SimpleNamespace(get_block_ids=lambda: ([], []))
    blocks = SimpleNamespace(get_block_ids=lambda: block_ids, new_empty=lambda: empty)
    multi = SimpleNamespace(_requests_to_connector={"p": chosen}, _connectors=[Mock(), push])
    scope = execute("ascend_multi_connector.py", {}, owner="AscendMultiConnector", method="update_state_after_alloc")
    scope["update_state_after_alloc"](multi, request, blocks, num_external_tokens)
    assert scheduler._reqs_need_send_layerwise["p"][1] == block_ids
    metadata = execute(
        "kv_p2p/mooncake/metadata.py",
        {
            "__name__": __name__,
            "dataclass": dataclass,
            "KVConnectorHandshakeMetadata": type("Handshake", (), {}),
            "KVConnectorMetadata": type("Metadata", (), {}),
        },
    )
    scope = execute(
        "kv_p2p/mooncake/push_scheduler.py",
        {"MooncakeConnectorMetadata": metadata["MooncakeConnectorMetadata"]},
        owner="MooncakePushConnectorScheduler",
        method="build_connector_meta",
    )
    scheduler._reqs_need_send = {}
    scheduler._reqs_sent_to_worker = set()
    scheduler._reqs_pending_send = {}
    scheduler._reqs_aborted = {}
    scheduler._get_transfer_block_ids = lambda ids, _: ids
    cached = SimpleNamespace(req_ids=[], num_computed_tokens=[], new_block_ids=[], resumed_req_ids=set())
    output = SimpleNamespace(
        scheduled_cached_reqs=cached,
        scheduled_new_reqs=[SimpleNamespace(req_id="p", num_computed_tokens=32)],
        num_scheduled_tokens={"p": 16},
    )
    meta = scope["build_connector_meta"](scheduler, output).push_send_requests["p"]
    assert meta.num_external_tokens == num_external_tokens
    assert meta.remote_num_external_tokens == 64
    # A later chunk appends allocations without losing the prefix's origin.
    output.scheduled_new_reqs = []
    cached.req_ids, cached.num_computed_tokens, cached.new_block_ids = ["p"], [48], [([4], [10])]
    meta = scope["build_connector_meta"](scheduler, output).push_send_requests["p"]
    assert meta.local_block_ids == ((2, 3, 4), (8, 9, 10))
    assert meta.num_external_tokens == num_external_tokens
