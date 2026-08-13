# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorRole
from vllm.distributed.ec_transfer.ec_connector.cpu.common import (
    ECCPUConnectorMetadata,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.connector import (
    ECCPUConnector,
)

import vllm_ascend.distributed.ec_transfer.ec_connector.cpu.connector as connector_mod
from vllm_ascend.distributed.ec_transfer.ec_connector.cpu.connector import (
    AscendECCPUConnector,
)


def _config(cpu_bytes):
    return SimpleNamespace(
        ec_transfer_config=SimpleNamespace(ec_connector_extra_config={"ec_cpu_bytes": cpu_bytes}),
        model_config=SimpleNamespace(dtype=torch.float32),
    )


def test_connector_requires_ec_transfer_config():
    config = SimpleNamespace(
        ec_transfer_config=None,
        model_config=SimpleNamespace(dtype=torch.float32),
    )

    with pytest.raises(ValueError, match="ec_transfer_config is required"):
        AscendECCPUConnector(config, ECConnectorRole.SCHEDULER)


@pytest.mark.parametrize("cpu_bytes", [None, "not-an-integer"])
def test_connector_rejects_invalid_cpu_bytes(cpu_bytes):
    with pytest.raises(ValueError, match="positive integer"):
        AscendECCPUConnector(_config(cpu_bytes), ECConnectorRole.SCHEDULER)


def test_connector_requires_capacity_for_one_block(monkeypatch):
    monkeypatch.setattr(connector_mod, "_get_encoder_cache_hidden_dim", lambda config: 16)

    with pytest.raises(ValueError, match="at least one encoder-cache block"):
        AscendECCPUConnector(_config(63), ECConnectorRole.SCHEDULER)


def test_valid_cpu_bytes_delegates_to_upstream_connector(monkeypatch):
    calls = []
    monkeypatch.setattr(connector_mod, "_get_encoder_cache_hidden_dim", lambda config: 16)
    monkeypatch.setattr(
        ECCPUConnector,
        "__init__",
        lambda self, config, role: calls.append((config, role)),
    )
    config = _config("64")

    AscendECCPUConnector(config, ECConnectorRole.SCHEDULER)

    assert calls == [(config, ECConnectorRole.SCHEDULER)]


def test_build_connector_meta_orders_blocks_for_dma_coalescing():
    metadata = ECCPUConnectorMetadata(
        saves={"save": [8, 7, 6]},
        loads={"load": [4, 2, 3]},
    )
    connector = AscendECCPUConnector.__new__(AscendECCPUConnector)
    connector.connector_scheduler = SimpleNamespace(build_connector_meta=lambda output: metadata)

    result = connector.build_connector_meta(None)

    assert result is metadata
    assert result.saves == {"save": [6, 7, 8]}
    assert result.loads == {"load": [2, 3, 4]}
