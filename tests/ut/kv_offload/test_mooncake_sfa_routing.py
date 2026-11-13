# SPDX-License-Identifier: Apache-2.0
"""Unified Pull dispatch and local metadata preserve the SFA contract."""

import ast
from multiprocessing import Pipe
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake import connector as ordinary
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.metadata import MooncakeConnectorMetadata
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_sfa import scheduler as sparse_scheduler
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_sfa import worker as sparse_worker
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_sfa.metadata import MooncakeSFAConnectorMetadata


@pytest.mark.parametrize("enabled", [None, False, True])
@pytest.mark.parametrize("component", ["worker", "scheduler"])
def test_unified_entry_selects_exactly_one_implementation_before_initialization(monkeypatch, enabled, component):
    monkeypatch.setattr(
        ordinary,
        "init_ascend_config",
        lambda _config: SimpleNamespace(
            sparse_kv_offload_config=None if enabled is None else SimpleNamespace(enabled=enabled)
        ),
    )
    suffix = "Worker" if component == "worker" else "Scheduler"
    original = MagicMock()
    specialized = MagicMock()
    monkeypatch.setattr(ordinary, "MooncakePullConnector" + suffix, original)
    sparse = sparse_worker if component == "worker" else sparse_scheduler
    monkeypatch.setattr(sparse, "MooncakeSFAConnector" + suffix, specialized)
    facade = ordinary.MooncakeConnector.__new__(ordinary.MooncakeConnector)
    facade.engine_id = "engine"
    config, caches = object(), object()
    selected = getattr(facade, "_create_" + component)(config, caches)
    chosen, unused = (specialized, original) if enabled else (original, specialized)
    assert selected is chosen.return_value
    chosen.assert_called_once_with(config, "engine", caches)
    unused.assert_not_called()
    assert facade.__class__.__name__ == "MooncakePullConnector"


def test_sparse_local_metadata_roundtrips_without_extending_the_wire_handshake():
    meta = MooncakeSFAConnectorMetadata({"r": 2, "local-hit": 3})
    meta.add_new_req(
        "r",
        ([1],),
        16,
        16,
        {
            "remote_block_ids": ([4],),
            "remote_host": "peer",
            "remote_port": 6000,
            "remote_engine_id": "p",
            "remote_request_id": "p-r",
            "remote_num_prompt_tokens": 16,
        },
    )
    receiver, sender = Pipe(duplex=False)
    try:
        sender.send(meta)
        restored = receiver.recv()
    finally:
        sender.close()
        receiver.close()
    assert isinstance(restored, MooncakeConnectorMetadata)
    assert restored.copy_sfa_slots == {"r": 2, "local-hit": 3}
    assert restored.requests["r"].copy_sfa_slot == 2
    assert restored.requests["r"].remote_block_ids == ([4],)
    assert not hasattr(MooncakeConnectorMetadata(), "copy_sfa_slots")


def test_only_pull_factories_lazily_import_offload_components():
    package = Path(ordinary.__file__).parent
    for source in package.glob("*.py"):
        tree = ast.parse(source.read_text())
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            assert "sparse_kv_offload" not in (node.module or ""), source.name
            if "mooncake_sfa" in (node.module or ""):
                assert source.name == "connector.py"
                ancestor = parents[node]
                while not isinstance(ancestor, ast.FunctionDef):
                    assert ancestor in parents, "SFA imports must remain inside factories"
                    ancestor = parents[ancestor]
                assert ancestor.name in {"_create_scheduler", "_create_worker"}


def test_ordinary_pull_has_no_sparse_state_or_preparation():
    facade = ordinary.MooncakePullConnector.__new__(ordinary.MooncakePullConnector)
    facade.connector_worker = SimpleNamespace()
    assert facade.get_copy_sfa_pd_state() is None
    assert facade.get_copy_sfa_slot_bindings() == {}
    facade.prepare_sparse_kv_offload(object())
