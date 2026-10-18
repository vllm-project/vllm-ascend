"""Public capability contracts and shared result validation remain vendor-independent."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1 import backend
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.backend import (
    GVABackend,
    GVAObjectSessions,
    KeyRangeBackend,
    ObjectSessions,
    resolve_backend_spec,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.backend.results import (
    BackendResultError,
    integer_result,
    integer_results,
)


@pytest.mark.parametrize("name", ("memcache", "mooncake"))
def test_registry_resolves_vendor_package(name):
    spec = resolve_backend_spec(name)
    assert spec.backend_factory.__module__ == f"{backend.__name__}.{name}.backend"
    assert callable(spec.backend_factory.create_sessions)


def test_public_contracts_expose_sessions_not_native_acquisition():
    assert callable(KeyRangeBackend.create_sessions)
    assert callable(GVABackend.create_sessions)
    assert ObjectSessions in GVAObjectSessions.__mro__
    assert callable(GVAObjectSessions.load_regions)
    assert callable(GVAObjectSessions.store_regions)
    for contract, internal_methods in (
        (
            KeyRangeBackend,
            (
                "validate_key_range_support",
                "start_key_range_load",
                "finish_key_range_load",
                "start_key_range_store",
                "commit_key_range_store",
                "revoke_key_range_store",
            ),
        ),
        (
            GVABackend,
            (
                "validate_gva_support",
                "add_gva_leases",
                "remove_gva_leases",
                "allocate_gva",
                "publish_gva",
                "revoke_gva",
            ),
        ),
    ):
        assert all(not hasattr(contract, name) for name in internal_methods)
    assert "MemcacheSessions" not in backend.__all__
    assert "MooncakeSessions" not in backend.__all__


def test_top_level_contracts_and_tools_do_not_import_vendor_implementations():
    for source in Path(backend.__file__).parent.glob("*.py"):
        for node in ast.walk(ast.parse(source.read_text())):
            if isinstance(node, ast.ImportFrom):
                assert not {"memcache", "mooncake"}.intersection((node.module or "").split(".")), source


def test_integer_validation_preserves_addresses_byte_counts_and_error_codes():
    assert integer_results("allocation", ["address", "rejected", "failed"], [1000, 0, -7]) == (1000, 0, -7)
    assert integer_result("copied bytes", 64) == 64


@pytest.mark.parametrize("values", (None, [], [1, 2], [True], [1.5]))
def test_integer_validation_rejects_malformed_per_object_results(values):
    with pytest.raises(BackendResultError):
        integer_results("operation", ["key"], values)
