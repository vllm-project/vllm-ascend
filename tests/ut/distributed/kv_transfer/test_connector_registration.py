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
