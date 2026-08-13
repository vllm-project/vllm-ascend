# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from unittest.mock import MagicMock

import pytest
from vllm.distributed.eplb.eplb_communicator import (
    EplbCommunicator,
    TorchDistGlooStagedEplbCommunicator,
)

from vllm_ascend.distributed.eplb.communicator import AscendGlooEplbCommunicator


@pytest.fixture
def communicator(monkeypatch):
    monkeypatch.setattr(EplbCommunicator, "_log_initialized", lambda self: None)
    return AscendGlooEplbCommunicator(cpu_group=MagicMock())


def test_communicator_reuses_upstream_gloo_staging(communicator):
    assert isinstance(communicator, TorchDistGlooStagedEplbCommunicator)
    assert communicator.needs_profile_buffer_reservation is False
