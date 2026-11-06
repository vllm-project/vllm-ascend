# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm_ascend.ascend_config import AscendConfig, SharedEngramRdmaConfig
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import get_hardware_profile


@pytest.mark.parametrize("device", [AscendDeviceType.A2, AscendDeviceType.A3, AscendDeviceType.A5])
@pytest.mark.parametrize("explicit", [None, False, True])
def test_engram_overlap_hardware_default_and_explicit_override(monkeypatch, device, explicit):
    monkeypatch.setattr("vllm_ascend.ascend_config.get_current_hardware_profile", lambda: get_hardware_profile(device))
    kwargs = {} if explicit is None else {"multistream_engram_overlap": explicit}
    config = AscendConfig(sparse_kv_offload_config=SimpleNamespace(enabled=False), **kwargs)
    expected = device == AscendDeviceType.A5 if explicit is None else explicit
    assert config.multistream_engram_overlap is expected


@pytest.mark.parametrize("endpoint", ["missing-port", "host:0", "host:65536", ":1234"])
def test_remote_engram_rejects_invalid_endpoint(endpoint):
    with pytest.raises(ValueError, match="host:port"):
        SharedEngramRdmaConfig(endpoint=endpoint)


def test_remote_engram_is_opt_in():
    config = AscendConfig(sparse_kv_offload_config=SimpleNamespace(enabled=False))
    assert config.shared_engram_rdma is None
    assert SharedEngramRdmaConfig(endpoint="127.0.0.1:29500").capacity == 65536


def test_shared_reader_rejects_immediate_shutdown(monkeypatch):
    monkeypatch.setattr(
        "vllm_ascend.ascend_config.get_current_hardware_profile", lambda: get_hardware_profile(AscendDeviceType.A5)
    )
    config = AscendConfig(
        shared_engram_rdma=SharedEngramRdmaConfig(endpoint="127.0.0.1:29500"),
        sparse_kv_offload_config=SimpleNamespace(enabled=False),
    )
    with pytest.raises(ValueError, match="shutdown-timeout"):
        config.derive_and_validate(SimpleNamespace(shutdown_timeout=0))
