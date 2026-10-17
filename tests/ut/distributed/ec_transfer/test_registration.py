# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# SPDX-License-Identifier: Apache-2.0

import importlib

import pytest
from vllm.distributed.ec_transfer.ec_connector.factory import ECConnectorFactory

import vllm_ascend.distributed.ec_transfer as ec_registration


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
