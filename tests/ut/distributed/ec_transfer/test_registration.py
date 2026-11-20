# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# SPDX-License-Identifier: Apache-2.0

import importlib
import subprocess
import sys

import pytest
from vllm.distributed.ec_transfer.ec_connector.factory import ECConnectorFactory

import vllm_ascend.distributed.ec_transfer as ec_registration


def test_tiering_package_import_does_not_load_local_cache_implementation():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; "
            "import vllm_ascend.distributed.ec_transfer.ec_connector.tiering; "
            "assert not any(name.startswith("
            "'vllm_ascend.distributed.ec_transfer.ec_connector.tiering.'"
            ") for name in sys.modules)",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.fixture
def isolated_registry(monkeypatch):
    registry = dict(ECConnectorFactory._registry)
    monkeypatch.setattr(ECConnectorFactory, "_registry", registry)
    return registry


def test_registers_ascend_eccpu_connector(isolated_registry):
    ec_registration.register_connector()

    connector_cls = isolated_registry["ECCPUConnector"]()
    assert connector_cls.__module__ == ("vllm_ascend.distributed.ec_transfer.ec_connector.cpu.connector")
    assert connector_cls.__name__ == "AscendECCPUConnector"


def test_missing_upstream_connector_is_skipped(monkeypatch, isolated_registry):
    before = dict(isolated_registry)
    upstream_module = "vllm.distributed.ec_transfer.ec_connector.cpu.connector"
    real_import_module = importlib.import_module

    def missing_upstream(name, package=None):
        if name == upstream_module:
            raise ModuleNotFoundError(
                "upstream EC CPU connector is unavailable",
                name="vllm.distributed.ec_transfer.ec_connector.cpu",
            )
        return real_import_module(name, package)

    monkeypatch.setattr(ec_registration.importlib, "import_module", missing_upstream)
    ec_registration.register_connector()

    assert isolated_registry == before


def test_nested_import_error_is_not_swallowed(monkeypatch, isolated_registry):
    del isolated_registry
    upstream_module = "vllm.distributed.ec_transfer.ec_connector.cpu.connector"

    real_import_module = importlib.import_module

    def broken_upstream(name, package=None):
        if name == upstream_module:
            raise ModuleNotFoundError(
                "unexpected_nested_dependency is missing",
                name="unexpected_nested_dependency",
            )
        return real_import_module(name, package)

    monkeypatch.setattr(ec_registration.importlib, "import_module", broken_upstream)
    with pytest.raises(ModuleNotFoundError, match="unexpected_nested_dependency"):
        ec_registration.register_connector()


def test_tiering_registration_is_lazy_and_repeatable(monkeypatch, isolated_registry):
    imported = []
    real_import = importlib.import_module

    def track_import(name, package=None):
        imported.append(name)
        return real_import(name, package)

    monkeypatch.setattr(ec_registration.importlib, "import_module", track_import)
    ec_registration.register_connector()
    ec_registration.register_connector()
    assert not any("tiering" in name or "worker" in name for name in imported)
    assert not any("mooncake.engine" in name for name in imported)

    connector_cls = isolated_registry["ECTieringConnector"]()
    assert connector_cls.__name__ == "AscendECTieringConnector"
    assert connector_cls.__module__.endswith("tiering.connector")


def test_missing_upstream_tiering_preserves_other_registrations(isolated_registry):
    isolated_registry.pop("ECTieringConnector", None)
    ec_registration.register_connector()
    assert "ECTieringConnector" not in isolated_registry
    assert isolated_registry["ECCPUConnector"]().__name__ == "AscendECCPUConnector"
    # Loading the Mooncake class would import optional runtime dependencies.
    assert "ECMooncakeConnector" in isolated_registry


def test_tiering_nested_dependency_error_propagates_on_load(monkeypatch, isolated_registry):
    ec_registration.register_connector()
    module = "vllm_ascend.distributed.ec_transfer.ec_connector.tiering.connector"
    real_import = importlib.import_module

    def broken_import(name, package=None):
        if name == module:
            raise ModuleNotFoundError("missing nested tiering dependency", name="nested_tiering_dependency")
        return real_import(name, package)

    monkeypatch.setattr(ec_registration.importlib, "import_module", broken_import)
    with pytest.raises(ModuleNotFoundError, match="missing nested tiering dependency"):
        isolated_registry["ECTieringConnector"]()
