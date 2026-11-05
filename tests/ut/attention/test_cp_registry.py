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
from unittest.mock import patch

import pytest

from vllm_ascend.attention import attention_v1, dsa_v1, dsa_v41, mla_v1, sfa_v1
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
from vllm_ascend.attention.context_parallel.topology import CPTopology, resolve_cp_topology
from vllm_ascend.attention.dsa_v1 import AscendDSAImpl, AscendDSAMetadataBuilder
from vllm_ascend.attention.dsa_v41 import AscendDSAV41Impl, AscendDSAV41MetadataBuilder
from vllm_ascend.attention.mla_v1 import AscendMLAImpl, AscendMLAMetadataBuilder
from vllm_ascend.attention.sfa_v1 import AscendSFAImpl, AscendSFAMetadataBuilder
from vllm_ascend.device.hardware_profile import AttentionBackendFamily
from vllm_ascend.platform import NPUPlatform


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


def test_builtin_provider_lazy_import_registers_on_resolve(tmp_path, monkeypatch) -> None:
    # A module that is only reachable through the lazy-loading table must be
    # imported on first resolve and register its provider as a side effect.
    module_name = "test_cp_registry_lazy_probe_module"
    (tmp_path / f"{module_name}.py").write_text(
        "from vllm_ascend.attention.context_parallel.registry import CPBackendProvider, register_cp_backend\n"
        "\n"
        "\n"
        "class _LazyProbeBaseBuilder:\n"
        "    pass\n"
        "\n"
        "\n"
        "class _LazyProbeBaseImpl:\n"
        "    pass\n"
        "\n"
        "\n"
        "register_cp_backend(\n"
        "    CPBackendProvider(\n"
        '        name="lazy_probe",\n'
        "        base_builder_cls=_LazyProbeBaseBuilder,\n"
        "        base_impl_cls=_LazyProbeBaseImpl,\n"
        "    )\n"
        ")\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setitem(registry._CP_BACKEND_MODULES, "lazy_probe", module_name)
    assert registry.get_cp_backend("lazy_probe") is None
    try:
        builder_cls, impl_cls = registry.resolve_cp_classes(
            "lazy_probe", None, CPTopology(tp_size=1, pcp_size=1, dcp_size=1)
        )
    finally:
        registry._CP_BACKEND_REGISTRY.pop("lazy_probe", None)
    assert builder_cls.__name__ == "_LazyProbeBaseBuilder"
    assert impl_cls.__name__ == "_LazyProbeBaseImpl"


def test_platform_pcp_backend_selection_preserves_class_paths() -> None:
    cases = (
        ((True, False, False), "vllm_ascend.attention.mla_v1.AscendMLABackend"),
        ((False, False, False), "vllm_ascend.attention.attention_v1.AscendAttentionBackend"),
        ((True, True, False), "vllm_ascend.attention.sfa_v1.AscendSFABackend"),
        ((True, False, True), "vllm_ascend.attention.dsa_v1.AscendDSABackend"),
    )
    with (
        patch("vllm_ascend.platform._validate_fa3_backend", return_value=False),
        patch("vllm_ascend.platform._validate_flash_attn_backend", return_value=False),
        patch(
            "vllm_ascend.platform.get_current_hardware_profile",
            return_value=SimpleNamespace(attention_backend_family=AttentionBackendFamily.STANDARD),
        ),
    ):
        for (use_mla, use_sparse, use_compress), expected_path in cases:
            selector = SimpleNamespace(
                use_mla=use_mla,
                use_sparse=use_sparse,
                use_compress=use_compress,
                use_pcp=True,
            )
            assert NPUPlatform._select_attn_backend_cls(selector) == expected_path
        unsupported = SimpleNamespace(use_mla=False, use_sparse=True, use_compress=False, use_pcp=True)
        with pytest.raises(NotImplementedError, match="PCP does not support attention backend"):
            NPUPlatform._select_attn_backend_cls(unsupported)


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
def test_sfa_resolve_wrappers_match_provider(monkeypatch, dcp_enabled, topology) -> None:
    monkeypatch.setattr(sfa_cp, "enable_sfa_dcp_replicated_indexer", lambda: dcp_enabled)
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=topology.pcp_size,
            decode_context_parallel_size=topology.dcp_size,
        ),
    )
    expected_builder, expected_impl = registry.resolve_cp_classes("sfa", vllm_config, topology)
    assert sfa_cp.resolve_sfa_metadata_builder(vllm_config) is expected_builder
    assert sfa_cp.resolve_sfa_impl(vllm_config) is expected_impl

    # The no-arg call keeps the pre-registry PCP-only predicate.
    if dcp_enabled:
        assert (sfa_cp.resolve_sfa_metadata_builder(), sfa_cp.resolve_sfa_impl()) == (
            sfa_cp.AscendSFADCPMetadataBuilder,
            sfa_cp.AscendSFADCPImpl,
        )
    else:
        assert (sfa_cp.resolve_sfa_metadata_builder(), sfa_cp.resolve_sfa_impl()) == (
            AscendSFAMetadataBuilder,
            AscendSFAImpl,
        )

    # A legacy partial config that only declares the PCP size still resolves.
    partial_config = SimpleNamespace(parallel_config=SimpleNamespace(prefill_context_parallel_size=topology.pcp_size))
    assert sfa_cp.resolve_sfa_metadata_builder(partial_config) is expected_builder
    assert sfa_cp.resolve_sfa_impl(partial_config) is expected_impl


_V1_BACKEND_CASES = [
    (
        attention_v1.AscendAttentionBackend,
        attention_v1,
        "gqa",
        (1, 1),
        (1, 2),
        (AscendAttentionMetadataBuilder, AscendAttentionBackendImpl),
        (attention_cp.AscendAttentionDCPMetadataBuilder, attention_cp.AscendAttentionDCPImpl),
    ),
    (
        mla_v1.AscendMLABackend,
        mla_v1,
        "mla",
        (1, 1),
        (1, 2),
        (AscendMLAMetadataBuilder, AscendMLAImpl),
        (mla_cp.AscendMlaDCPMetadataBuilder, mla_cp.AscendMlaDCPImpl),
    ),
    (
        sfa_v1.AscendSFABackend,
        sfa_v1,
        "sfa",
        (1, 1),
        (2, 1),
        (AscendSFAMetadataBuilder, AscendSFAImpl),
        (AscendSFAMetadataBuilder, sfa_cp.AscendSFAPCPImpl),
    ),
    (
        dsa_v1.AscendDSABackend,
        dsa_v1,
        "dsa",
        (1, 1),
        (2, 1),
        (AscendDSAMetadataBuilder, AscendDSAImpl),
        (dsa_cp.AscendDSAPCPMetadataBuilder, dsa_cp.AscendDSAPCPImpl),
    ),
    (
        dsa_v41.DeepseekV41CacheBackend,
        dsa_v41,
        "dsa_v41",
        (1, 1),
        (2, 1),
        (AscendDSAV41MetadataBuilder, AscendDSAV41Impl),
        (dsa_v41_cp.AscendDSAV41PCPMetadataBuilder, dsa_v41_cp.AscendDSAV41PCPImpl),
    ),
]


@pytest.mark.parametrize(
    ("backend", "backend_module", "name", "off_sizes", "on_sizes", "base_pair", "cp_pair"),
    _V1_BACKEND_CASES,
    ids=["gqa", "mla", "sfa", "dsa", "dsa_v41"],
)
def test_v1_backends_delegate_to_registry(
    monkeypatch, backend, backend_module, name, off_sizes, on_sizes, base_pair, cp_pair
) -> None:
    # SFA also consults the KV-offload switch and the DCP indexer switch.
    monkeypatch.setattr(
        sfa_v1,
        "get_ascend_config",
        lambda: SimpleNamespace(sparse_kv_offload_config=SimpleNamespace(enabled=False)),
    )
    monkeypatch.setattr(sfa_cp, "enable_sfa_dcp_replicated_indexer", lambda: False)
    for sizes, expected_pair in ((off_sizes, base_pair), (on_sizes, cp_pair)):
        vllm_config = SimpleNamespace(
            parallel_config=SimpleNamespace(
                prefill_context_parallel_size=sizes[0],
                decode_context_parallel_size=sizes[1],
            ),
        )
        monkeypatch.setattr(backend_module, "get_current_vllm_config", lambda config=vllm_config: config)
        expected = registry.resolve_cp_classes(name, vllm_config, resolve_cp_topology(vllm_config))
        assert expected == expected_pair
        assert backend.get_builder_cls() is expected[0]
        assert backend.get_impl_cls() is expected[1]
