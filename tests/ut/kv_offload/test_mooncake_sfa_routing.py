# SPDX-License-Identifier: Apache-2.0
"""Legacy configuration dispatch and local metadata survive SFA isolation."""

import ast
from multiprocessing import Pipe
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake import connector as ordinary
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.metadata import MooncakeConnectorMetadata
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_sfa import compat
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_sfa import connector as sparse
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_sfa.metadata import MooncakeSFAConnectorMetadata


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("component", ["worker", "scheduler"])
def test_legacy_entry_selects_exactly_one_implementation_before_initialization(monkeypatch, enabled, component):
    monkeypatch.setattr(
        compat,
        "init_ascend_config",
        lambda _config: SimpleNamespace(sparse_kv_offload_config=SimpleNamespace(enabled=enabled)),
    )
    suffix = "Worker" if component == "worker" else "Scheduler"
    original = MagicMock()
    specialized = MagicMock()
    monkeypatch.setattr(ordinary, "MooncakePullConnector" + suffix, original)
    monkeypatch.setattr(sparse, "MooncakeSFAConnector" + suffix, specialized)
    facade = compat.MooncakeConnector.__new__(compat.MooncakeConnector)
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


def test_generic_mooncake_does_not_import_its_offload_specialization():
    package = Path(ordinary.__file__).parent
    for source in package.glob("*.py"):
        for node in ast.walk(ast.parse(source.read_text())):
            if isinstance(node, ast.ImportFrom):
                assert "mooncake_sfa" not in (node.module or ""), source.name
                assert "sparse_kv_offload" not in (node.module or ""), source.name
