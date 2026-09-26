# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import logging
import sys
import types

import pytest
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory

# Connectors whose `kv_connector` config name differs from their class name.
# MultiConnector keys transfer stats by `__class__.__name__` and resolves that
# key through KVConnectorFactory, so the class name must be registered too.
CLASS_NAME_ALIASES = {
    "AscendOffloadingConnector": (
        "vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.native.offloading_connector",
        "AscendOffloadingConnector",
    ),
    "UCMConnectorV1": (
        "vllm_ascend.distributed.kv_transfer.kv_pool.ucm_connector.connector",
        "UCMConnectorV1",
    ),
}


def test_connectors_with_mismatched_names_register_class_name_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_ascend.distributed.kv_transfer import register_connector

    registrations: dict[str, tuple[str, str]] = {}

    def capture_registration(cls, name: str, module_path: str, class_name: str) -> None:
        registrations[name] = (module_path, class_name)

    # Keep the test independent of whether the vLLM plugin was already loaded
    # by the current pytest environment.
    monkeypatch.setattr(KVConnectorFactory, "_registry", {})
    monkeypatch.setattr(
        KVConnectorFactory,
        "register_connector",
        classmethod(capture_registration),
    )
    register_connector()

    for alias, expected in CLASS_NAME_ALIASES.items():
        assert registrations.get(alias) == expected


def test_stats_keys_are_resolvable_through_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exact keys MultiConnector stats carry must resolve through the factory.

    ``get_connector_class_by_name`` raises ``ValueError`` for any key missing
    from ``_registry``; that is the failure mode behind issue #16932. Asserting
    registry membership (not just that a registration call happened) is what
    guards the resolution path.
    """
    from vllm_ascend.distributed.kv_transfer import register_connector

    monkeypatch.setattr(KVConnectorFactory, "_registry", {})
    register_connector()

    for stats_key in CLASS_NAME_ALIASES:
        assert stats_key in KVConnectorFactory._registry, (
            f"stats key '{stats_key}' is not resolvable by KVConnectorFactory"
        )


def test_alias_helper_skips_alias_when_names_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When config name == class name, no redundant alias is registered."""
    from vllm_ascend.distributed.kv_transfer import _register_with_class_name_alias

    registrations: list[tuple[str, str, str]] = []

    def capture_registration(cls, name: str, module_path: str, class_name: str) -> None:
        registrations.append((name, module_path, class_name))

    monkeypatch.setattr(KVConnectorFactory, "_registry", {})
    monkeypatch.setattr(
        KVConnectorFactory,
        "register_connector",
        classmethod(capture_registration),
    )
    _register_with_class_name_alias("FooConnector", "some.module", "FooConnector")

    assert registrations == [("FooConnector", "some.module", "FooConnector")]


def test_alias_helper_registers_alias_when_names_differ(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When config name != class name, both names are registered."""
    from vllm_ascend.distributed.kv_transfer import _register_with_class_name_alias

    registrations: list[tuple[str, str, str]] = []

    def capture_registration(cls, name: str, module_path: str, class_name: str) -> None:
        registrations.append((name, module_path, class_name))

    monkeypatch.setattr(KVConnectorFactory, "_registry", {})
    monkeypatch.setattr(
        KVConnectorFactory,
        "register_connector",
        classmethod(capture_registration),
    )
    _register_with_class_name_alias("BarConnector", "some.module", "BarConnectorV1")

    assert ("BarConnector", "some.module", "BarConnectorV1") in registrations
    assert ("BarConnectorV1", "some.module", "BarConnectorV1") in registrations


def test_d2rh_connector_class_name_matches_its_config_name() -> None:
    """MultiConnector keys transfer stats by ``__class__.__name__`` and resolves
    that key through KVConnectorFactory, which is indexed by the registered
    config name. The D2RH connector's class must therefore carry the config
    name; the historical ``MooncakeConnector`` name resolves to the upstream
    registry entry and reconstructs an incompatible stats schema (KeyError on
    every stats interval under MultiConnector).
    """
    from vllm_ascend.distributed.kv_transfer.kv_p2p import mooncake_d2rh_connector as d2rh

    assert d2rh.MooncakeD2RHConnectorV1.__name__ == "MooncakeD2RHConnectorV1"
    assert d2rh.MooncakeConnector is d2rh.MooncakeD2RHConnectorV1
    assert d2rh.MooncakeD2RHConnector is d2rh.MooncakeD2RHConnectorV1


def _stub_ucm(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ucm package is not installed in the CPU UT environment."""
    connector_mod = types.ModuleType("ucm.integration.vllm.ucm_connector")

    class UCMConnector:
        @classmethod
        def build_kv_connector_stats(cls, data: dict | None = None):
            return None

    connector_mod.UCMConnector = UCMConnector  # type: ignore[attr-defined]
    for name in ("ucm", "ucm.integration", "ucm.integration.vllm"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "ucm.integration.vllm.ucm_connector", connector_mod)


def test_stats_keys_resolve_through_real_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_connector_class_by_name runs the real importlib loader; the
    membership tests above would pass with a typo in the registered module
    path or class name, so resolve every alias end-to-end.
    """
    from vllm_ascend.distributed.kv_transfer import register_connector

    _stub_ucm(monkeypatch)
    monkeypatch.setattr(KVConnectorFactory, "_registry", {})
    register_connector()

    for stats_key, (module_path, class_name) in CLASS_NAME_ALIASES.items():
        cls = KVConnectorFactory.get_connector_class_by_name(stats_key)
        assert cls.__name__ == class_name
        assert cls.__module__ == module_path


def test_multi_connector_stats_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    """The exact #16932 path: MultiConnector rebuilds per-connector stats from
    keys equal to the connector class names. Without the aliases this raises
    ValueError in get_connector_class_by_name and tears down EngineCore.
    """
    from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import MultiConnector

    from vllm_ascend.distributed.kv_transfer import register_connector

    _stub_ucm(monkeypatch)
    monkeypatch.setattr(KVConnectorFactory, "_registry", {})
    register_connector()

    stats = MultiConnector.build_kv_connector_stats(
        data={
            "UCMConnectorV1": {"dummy": [1.0]},
            "AscendOffloadingConnector": {"dummy": [1.0]},
        }
    )
    assert stats is not None
    # OffloadingConnector overrides build_kv_connector_stats, so this key
    # yields real reconstructed stats; UCMConnectorV1 delegates to the ucm
    # package (stubbed to the base default, None) and is skipped.
    assert "AscendOffloadingConnector" in stats.data
    assert "UCMConnectorV1" not in stats.data


def test_alias_override_pops_existing_entry_and_warns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A class name already registered under a different config name is
    overridden by the alias: the real register_connector raises on duplicate
    keys, so the pop is what makes the re-registration succeed, and the
    override is only a one-time warning.
    """
    from vllm.logger import logger as vllm_logger

    from vllm_ascend.distributed.kv_transfer import _register_with_class_name_alias
    from vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.native.offloading_connector import (
        AscendOffloadingConnector,
    )

    # Capture the warning directly on the vllm logger: in this environment
    # caplog.records stays empty (the live-logs plugin keeps a second
    # LogCaptureHandler on root).
    captured: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(record)

    monkeypatch.setattr(vllm_logger, "handlers", [_Capture()])

    module = "vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.native.offloading_connector"
    # A different connector claims the class name as its config name.
    other_module = "vllm_ascend.distributed.kv_transfer.kv_pool.ucm_connector.connector"
    monkeypatch.setattr(KVConnectorFactory, "_registry", {})
    KVConnectorFactory.register_connector("AscendOffloadingConnector", other_module, "UCMConnectorV1")
    old_entry = KVConnectorFactory._registry["AscendOffloadingConnector"]

    _register_with_class_name_alias("OffloadingConnector", module, "AscendOffloadingConnector")

    assert any("already registered" in record.getMessage() for record in captured)
    # The stale closure was replaced and the key now resolves to the alias.
    assert KVConnectorFactory._registry["AscendOffloadingConnector"] is not old_entry
    assert KVConnectorFactory._registry["AscendOffloadingConnector"]() is AscendOffloadingConnector
    assert "OffloadingConnector" in KVConnectorFactory._registry


def test_offloading_connector_config_name_resolves_to_ascend_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pre-existing pop of the upstream OffloadingConnector entry must
    leave the config name resolving to the Ascend class, and the alias must
    not disturb that ordering.
    """
    from vllm_ascend.distributed.kv_transfer import register_connector
    from vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.native.offloading_connector import (
        AscendOffloadingConnector,
    )

    monkeypatch.setattr(KVConnectorFactory, "_registry", {})
    register_connector()

    assert KVConnectorFactory._registry["OffloadingConnector"]() is AscendOffloadingConnector
    assert KVConnectorFactory._registry["AscendOffloadingConnector"]() is AscendOffloadingConnector


def test_no_alias_connectors_stay_out_of_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """The no-alias decision for the two stats-silent connectors is
    comment-only in the source; pin the negative case so a class-name key
    cannot appear in the registry without someone updating the comments.
    """
    from vllm_ascend.distributed.kv_transfer import register_connector

    monkeypatch.setattr(KVConnectorFactory, "_registry", {})
    register_connector()

    assert "AscendSimpleCPUOffloadConnector" not in KVConnectorFactory._registry
    assert "PreemptOffloadConnectorV1" not in KVConnectorFactory._registry


def test_no_alias_connectors_inherit_base_stats_default() -> None:
    """The no-alias decision rests on these connectors never emitting stats;
    pin the premise so a future get_kv_connector_stats override (upstream
    vllm-project/vllm#41790 would add one to SimpleCPUOffload) is caught here
    instead of reviving the #16932 crash.
    """
    from vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.preempt_offload.preempt_offload_connector import (
        PreemptOffloadConnectorV1,
    )
    from vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.simple.simple_cpu_offload_connector import (
        AscendSimpleCPUOffloadConnector,
    )

    for cls in (AscendSimpleCPUOffloadConnector, PreemptOffloadConnectorV1):
        # The defining class in the MRO must be the vllm base - compare by
        # qualified name, not identity, because the base module can be loaded
        # under a second path in the same process.
        for klass in cls.__mro__:
            if "get_kv_connector_stats" in klass.__dict__:
                assert klass.__name__ == "KVConnectorBase_V1", (
                    f"{cls.__name__} overrides get_kv_connector_stats in {klass.__name__}"
                )
                break
        else:
            raise AssertionError(f"{cls.__name__} does not define get_kv_connector_stats")
