# SPDX-License-Identifier: Apache-2.0

"""Tests for the context-parallel backend provider registry.

The adapters imported below register their built-in providers at import time,
which is exactly the production contract: a backend declares itself and the
registry resolves the (metadata builder, attention impl) pair from a topology.
"""

import ast
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackendImpl,
    AscendAttentionMetadataBuilder,
)
from vllm_ascend.attention.context_parallel import (
    attention_cp,
    dsa_cp,
    dsa_v41_cp,
    mla_cp,
    registry,
    sfa_cp,
)
from vllm_ascend.attention.context_parallel.topology import CPTopology
from vllm_ascend.attention.dsa_v1 import AscendDSAImpl, AscendDSAMetadataBuilder
from vllm_ascend.attention.dsa_v41 import AscendDSAV41Impl, AscendDSAV41MetadataBuilder
from vllm_ascend.attention.mla_v1 import AscendMLAImpl, AscendMLAMetadataBuilder
from vllm_ascend.attention.sfa_v1 import AscendSFAImpl, AscendSFAMetadataBuilder


class _SimBaseBuilder:
    pass


class _SimBaseImpl:
    pass


class _SimCPBuilder:
    pass


class _SimCPImpl:
    pass


DISABLED_TOPOLOGY = CPTopology(tp_size=2, pcp_size=1, dcp_size=1)


def _provider_snapshot(provider: registry.CPBackendProvider) -> tuple:
    return (
        provider.name,
        provider.base_builder_cls,
        provider.base_impl_cls,
        provider.cp_builder_cls,
        provider.cp_impl_cls,
        provider.supports_pcp,
        provider.supports_dcp,
        provider.validate,
        provider.cp_resolver,
    )


# Snapshot taken before any test registers the simulated provider, so the
# assertion in the isolation test compares against the built-in state only.
_BUILTIN_SNAPSHOT = {name: _provider_snapshot(registry.get_cp_backend(name)) for name in registry._CP_BACKEND_MODULES}


@pytest.fixture
def sim_provider():
    # The registry is a module-level dict; remove `sim` on teardown so the
    # simulated backend never leaks into the built-in provider set.
    provider = registry.CPBackendProvider(
        name="sim",
        base_builder_cls=_SimBaseBuilder,
        base_impl_cls=_SimBaseImpl,
        cp_builder_cls=_SimCPBuilder,
        cp_impl_cls=_SimCPImpl,
        supports_pcp=True,
        supports_dcp=True,
    )
    registry.register_cp_backend(provider)
    yield provider
    registry._CP_BACKEND_REGISTRY.pop("sim", None)


def test_register_and_resolve_provider(sim_provider) -> None:
    assert registry.get_cp_backend("sim") is sim_provider
    assert registry.resolve_cp_classes("sim", None, DISABLED_TOPOLOGY) == (_SimBaseBuilder, _SimBaseImpl)
    pcp_only = CPTopology(tp_size=2, pcp_size=2, dcp_size=1)
    assert registry.resolve_cp_classes("sim", None, pcp_only) == (_SimCPBuilder, _SimCPImpl)


def test_duplicate_registration_rejected(sim_provider) -> None:
    with pytest.raises(ValueError, match="sim"):
        registry.register_cp_backend(registry.CPBackendProvider(name="sim"))


def test_unknown_provider_raises_with_name() -> None:
    with pytest.raises(NotImplementedError, match="no_such_backend"):
        registry.resolve_cp_classes("no_such_backend", None, DISABLED_TOPOLOGY)


def test_builtin_providers_are_registered() -> None:
    expected_names = {"gqa", "mla", "sfa", "dsa", "dsa_v41"}
    assert set(registry._CP_BACKEND_MODULES) == expected_names
    for name in sorted(expected_names):
        registry.resolve_cp_classes(name, None, DISABLED_TOPOLOGY)
    assert set(registry.list_cp_backends()) >= expected_names
    for name in expected_names:
        provider = registry.get_cp_backend(name)
        assert provider is not None
        assert provider.name == name
    assert [field.name for field in fields(registry.CPBackendProvider)] == [
        "name",
        "base_builder_cls",
        "base_impl_cls",
        "cp_builder_cls",
        "cp_impl_cls",
        "supports_pcp",
        "supports_dcp",
        "validate",
        "cp_resolver",
    ]


@pytest.mark.parametrize(
    ("name", "builder_cls", "impl_cls"),
    [
        ("gqa", AscendAttentionMetadataBuilder, AscendAttentionBackendImpl),
        ("mla", AscendMLAMetadataBuilder, AscendMLAImpl),
        ("sfa", AscendSFAMetadataBuilder, AscendSFAImpl),
        ("dsa", AscendDSAMetadataBuilder, AscendDSAImpl),
        ("dsa_v41", AscendDSAV41MetadataBuilder, AscendDSAV41Impl),
    ],
)
def test_cp_disabled_returns_base_classes(name, builder_cls, impl_cls) -> None:
    assert registry.resolve_cp_classes(name, None, DISABLED_TOPOLOGY) == (builder_cls, impl_cls)


@pytest.mark.parametrize(
    ("name", "builder_cls", "impl_cls"),
    [
        ("gqa", AscendAttentionMetadataBuilder, AscendAttentionBackendImpl),
        ("mla", AscendMLAMetadataBuilder, AscendMLAImpl),
    ],
)
def test_gqa_mla_pcp_only_returns_base_classes(name, builder_cls, impl_cls) -> None:
    # PCP is inlined in the v1 backends, so a PCP-only topology keeps the base
    # classes and must not fall back to the DCP specialization.
    pcp_only = CPTopology(tp_size=2, pcp_size=2, dcp_size=1)
    assert registry.resolve_cp_classes(name, None, pcp_only) == (builder_cls, impl_cls)


@pytest.mark.parametrize(
    ("name", "builder_cls", "impl_cls"),
    [
        ("gqa", attention_cp.AscendAttentionDCPMetadataBuilder, attention_cp.AscendAttentionDCPImpl),
        ("mla", mla_cp.AscendMlaDCPMetadataBuilder, mla_cp.AscendMlaDCPImpl),
    ],
)
@pytest.mark.parametrize(
    "topology",
    [CPTopology(tp_size=2, pcp_size=1, dcp_size=2), CPTopology(tp_size=2, pcp_size=2, dcp_size=2)],
    ids=["dcp-only", "pcp-dcp"],
)
def test_dcp_enabled_returns_dcp_classes(name, builder_cls, impl_cls, topology) -> None:
    assert registry.resolve_cp_classes(name, None, topology) == (builder_cls, impl_cls)


@pytest.mark.parametrize(
    ("dcp_enabled", "topology"),
    [
        (False, CPTopology(tp_size=2, pcp_size=1, dcp_size=1)),
        (False, CPTopology(tp_size=2, pcp_size=2, dcp_size=1)),
        (True, CPTopology(tp_size=2, pcp_size=1, dcp_size=2)),
        (True, CPTopology(tp_size=2, pcp_size=2, dcp_size=2)),
    ],
    ids=["disabled", "pcp-only", "dcp-only", "pcp-dcp"],
)
def test_sfa_four_way_selection(monkeypatch, dcp_enabled, topology) -> None:
    monkeypatch.setattr(sfa_cp, "enable_sfa_dcp_replicated_indexer", lambda: dcp_enabled)
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(prefill_context_parallel_size=topology.pcp_size),
    )
    expected = (
        sfa_cp.resolve_sfa_metadata_builder(vllm_config),
        sfa_cp.resolve_sfa_impl(vllm_config),
    )
    assert registry.resolve_cp_classes("sfa", vllm_config, topology) == expected


@pytest.mark.parametrize(
    ("name", "cp_builder_cls", "cp_impl_cls", "base_builder_cls", "base_impl_cls"),
    [
        (
            "dsa",
            dsa_cp.AscendDSAPCPMetadataBuilder,
            dsa_cp.AscendDSAPCPImpl,
            AscendDSAMetadataBuilder,
            AscendDSAImpl,
        ),
        (
            "dsa_v41",
            dsa_v41_cp.AscendDSAV41PCPMetadataBuilder,
            dsa_v41_cp.AscendDSAV41PCPImpl,
            AscendDSAV41MetadataBuilder,
            AscendDSAV41Impl,
        ),
    ],
)
def test_dsa_and_v41_select_pcp_classes(name, cp_builder_cls, cp_impl_cls, base_builder_cls, base_impl_cls) -> None:
    for topology in (CPTopology(tp_size=2, pcp_size=2, dcp_size=1), CPTopology(tp_size=2, pcp_size=2, dcp_size=2)):
        assert registry.resolve_cp_classes(name, None, topology) == (cp_builder_cls, cp_impl_cls)
    assert registry.resolve_cp_classes(name, None, DISABLED_TOPOLOGY) == (base_builder_cls, base_impl_cls)


def test_provider_selection_is_stable() -> None:
    topology = CPTopology(tp_size=2, pcp_size=2, dcp_size=2)
    first = registry.resolve_cp_classes("gqa", None, topology)
    second = registry.resolve_cp_classes("gqa", None, topology)
    assert first[0] is second[0]
    assert first[1] is second[1]


def test_simulated_backend_registers_without_touching_existing_backends(sim_provider) -> None:
    # `sim` is fully described by this module's sentinel classes: registering
    # and resolving it never imports or edits an existing adapter, and the
    # built-in providers stay byte-identical to their pre-registration state.
    for name, snapshot in _BUILTIN_SNAPSHOT.items():
        assert _provider_snapshot(registry.get_cp_backend(name)) == snapshot
    assert sim_provider.name == "sim"
    assert registry.resolve_cp_classes("sim", None, CPTopology(tp_size=2, pcp_size=2, dcp_size=1)) == (
        _SimCPBuilder,
        _SimCPImpl,
    )


def test_registry_module_is_a_leaf() -> None:
    module_path = Path(__file__).resolve().parents[3] / "vllm_ascend/attention/context_parallel/registry.py"
    tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))

    runtime_imports: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            runtime_imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            runtime_imports.append(node.module or "")

    allowed_stdlib = {"__future__", "importlib", "collections.abc", "dataclasses", "typing"}
    assert set(runtime_imports) <= allowed_stdlib
    assert not any("context_parallel" in module for module in runtime_imports)
    assert not any(module.startswith(("vllm.", "vllm_ascend")) for module in runtime_imports)
