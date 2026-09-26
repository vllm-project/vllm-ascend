# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

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
