# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import asdict
from types import SimpleNamespace

import pytest
from pydantic import TypeAdapter

from vllm_ascend.ascend_config import AscendConfig, SharedEngramConfig
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
        SharedEngramConfig(backend="a2", endpoint=endpoint)


def test_remote_engram_is_opt_in():
    config = AscendConfig(sparse_kv_offload_config=SimpleNamespace(enabled=False))
    assert config.shared_engram is None
    assert SharedEngramConfig(backend="a2", endpoint="127.0.0.1:29500").capacity == 65536


@pytest.mark.parametrize("backend", ["a2", "a3", "a5"])
def test_shared_engram_config_accepts_backend(backend):
    config = AscendConfig(
        shared_engram={"backend": backend, "endpoint": "owner:29500"},
        sparse_kv_offload_config=SimpleNamespace(enabled=False),
    )
    assert config.shared_engram is not None
    assert config.shared_engram.backend == backend
    assert config.shared_engram.endpoint == "owner:29500"


def test_shared_engram_config_rejects_unavailable_backend():
    with pytest.raises(ValueError):
        TypeAdapter(SharedEngramConfig).validate_python({"backend": "unknown", "endpoint": "owner:29500"})


def test_a3_rejects_rdma_options():
    with pytest.raises(ValueError, match="A3 does not accept"):
        SharedEngramConfig(backend="a3", endpoint="owner:29500", capacity=72000)


def test_a3_config_round_trips_after_serialization():
    config = SharedEngramConfig(backend="a3", endpoint="owner:29500")
    assert SharedEngramConfig(**asdict(config)) == config


@pytest.mark.parametrize("capacity", [65536, 72000, 262144])
def test_remote_engram_accepts_supported_capacity(capacity):
    assert SharedEngramConfig(backend="a2", endpoint="host:29500", capacity=capacity).capacity == capacity


@pytest.mark.parametrize("capacity", [0, 262145])
def test_remote_engram_rejects_unsupported_capacity(capacity):
    with pytest.raises(ValueError, match="capacity"):
        SharedEngramConfig(backend="a2", endpoint="host:29500", capacity=capacity)


@pytest.mark.parametrize("port", [0, 65536])
def test_remote_engram_rejects_invalid_listen_port(port):
    with pytest.raises(ValueError, match="listen_port"):
        SharedEngramConfig(backend="a2", endpoint="host:29500", listen_port=port)


def test_remote_engram_rejects_conflicting_hixl_options():
    with pytest.raises(ValueError, match="mutually exclusive"):
        SharedEngramConfig(backend="a2", endpoint="host:29500", listen_port=26666, local_comm_res="ub.json")


def test_shared_reader_rejects_immediate_shutdown(monkeypatch):
    monkeypatch.setattr(
        "vllm_ascend.ascend_config.get_current_hardware_profile", lambda: get_hardware_profile(AscendDeviceType.A5)
    )
    config = AscendConfig(
        shared_engram=SharedEngramConfig(backend="a2", endpoint="127.0.0.1:29500"),
        sparse_kv_offload_config=SimpleNamespace(enabled=False),
    )
    with pytest.raises(ValueError, match="shutdown-timeout"):
        config.derive_and_validate(SimpleNamespace(shutdown_timeout=0))
