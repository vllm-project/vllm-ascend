"""Worker depends on session contracts, not vendor implementations or native leases."""

from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import Mock

import pytest

from tests.ut.distributed.ascend_store.v1.helpers import FakeBackend, make_worker
from tests.ut.distributed.ascend_store.v1.worker.gva_fixtures import FakeGVABackend, make_gva_worker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.backend import GVAObjectSessions, ObjectSessions
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker import base as worker_module


@pytest.mark.parametrize("use_gva", (False, True), ids=("key_range", "gva"))
def test_worker_uses_backend_supplied_session_owner(monkeypatch, use_gva):
    sessions = Mock(spec=GVAObjectSessions if use_gva else ObjectSessions)
    sessions.has_load_sessions = False
    sessions.pending_store_keys = ()
    create = Mock(return_value=sessions)
    backend_type = FakeGVABackend if use_gva else FakeBackend
    monkeypatch.setattr(backend_type, "create_sessions", create)
    # A Worker constructing the old concrete class would bypass the factory.
    validate = Mock(side_effect=AssertionError("Worker must not choose a native session implementation"))
    monkeypatch.setattr(backend_type, "validate_gva_support" if use_gva else "validate_key_range_support", validate)

    worker, resources, _ = make_gva_worker() if use_gva else make_worker(layerwise=True)
    try:
        create.assert_called_once_with()
        assert worker._backend_sessions is sessions
        assert resources.backend_sessions is sessions
        if use_gva:
            assert worker._gva_backend_io._sessions is sessions
        validate.assert_not_called()
    finally:
        worker.close()
    sessions.close.assert_called_once_with()


@pytest.mark.parametrize("use_gva", (False, True), ids=("key_range", "gva"))
def test_worker_propagates_session_factory_failure(monkeypatch, use_gva):
    backend_type = FakeGVABackend if use_gva else FakeBackend
    create = Mock(side_effect=RuntimeError("native sessions unavailable"))
    monkeypatch.setattr(backend_type, "create_sessions", create)

    with pytest.raises(RuntimeError, match="native sessions unavailable"):
        if use_gva:
            make_gva_worker()
        else:
            make_worker(layerwise=True)
    create.assert_called_once_with()


def test_worker_has_no_vendor_session_or_native_lifecycle_dependency():
    root = Path(worker_module.__file__).parent
    native_operations = {
        "validate_gva_support",
        "validate_key_range_support",
        "add_gva_leases",
        "remove_gva_leases",
        "allocate_gva",
        "publish_gva",
        "revoke_gva",
        "start_key_range_load",
        "finish_key_range_load",
        "start_key_range_store",
        "commit_key_range_store",
        "revoke_key_range_store",
    }
    for source in root.rglob("*.py"):
        for node in ast.walk(ast.parse(source.read_text())):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                assert not any(part in {"memcache", "mooncake", "sessions"} for part in module.split(".")), source
                assert not {"MemcacheSessions", "MooncakeSessions", "GVASessions", "KeyRangeSessions"}.intersection(
                    item.name for item in node.names
                ), source
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr not in native_operations, source
