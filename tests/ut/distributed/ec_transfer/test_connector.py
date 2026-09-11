# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# SPDX-License-Identifier: Apache-2.0

from vllm_ascend.distributed.ec_transfer.ec_connector.cpu.connector import (
    AscendECCPUConnector,
)


def test_make_worker_uses_ascend_backend(monkeypatch):
    import vllm_ascend.distributed.ec_transfer.ec_connector.cpu.worker as worker_mod

    config = object()
    expected = object()
    monkeypatch.setattr(worker_mod, "AscendECCPUWorker", lambda cfg: expected)
    connector = AscendECCPUConnector.__new__(AscendECCPUConnector)

    assert connector._make_worker(config) is expected
