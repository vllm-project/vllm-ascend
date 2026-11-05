#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

"""CPU-only unit tests for the EPD test-server infrastructure.

Subprocess launch and HTTP readiness waits are mocked, so these tests never
start a real vLLM/NPU process and never touch the network.
"""

import io
import json
import subprocess
import sys
import types

import pytest

from tests.e2e.common.epd import epd_servers
from tests.e2e.common.epd.epd_servers import DISAGG_EPD_PROXY_SCRIPT, DisaggEpdProxy, RemoteEPDServer
from tests.e2e.common.serving.remote_openai_server import RemoteOpenAIServer

ENCODE_PORT = "19534"
PREFILL_DECODE_PORT = "19535"
PROXY_PORT = "10001"


class _FakeProcess:
    """Stand-in for subprocess.Popen with a scripted exit behavior."""

    _next_pid = 1000

    def __init__(self, exit_code: int = 0, *, wait_times_out: bool = False) -> None:
        type(self)._next_pid += 1
        self.pid = type(self)._next_pid
        self.stdout = io.StringIO("")
        self.stderr = io.StringIO("")
        self.exit_code = exit_code
        self.wait_times_out = wait_times_out
        self.alive = True

    def wait(self, timeout: float | None = None) -> int:
        if self.wait_times_out:
            raise subprocess.TimeoutExpired(cmd="fake", timeout=timeout or 0)
        self.alive = False
        return self.exit_code

    def poll(self) -> int | None:
        return None if self.alive else self.exit_code


class _FakePopenFactory:
    """Records every launched command/env and returns scripted fake processes."""

    def __init__(self, exit_codes: list[int] | None = None, timeout_indexes: set[int] | None = None) -> None:
        self.calls: list[tuple[list[str], dict[str, str]]] = []
        self.exit_codes = list(exit_codes or [])
        self.timeout_indexes = timeout_indexes or set()

    def __call__(self, cmd, **kwargs):
        index = len(self.calls)
        self.calls.append((list(cmd), dict(kwargs.get("env") or {})))
        exit_code = self.exit_codes[index] if index < len(self.exit_codes) else 0
        return _FakeProcess(exit_code, wait_times_out=index in self.timeout_indexes)


def _install_fake_popen(monkeypatch, factory: _FakePopenFactory) -> None:
    monkeypatch.setattr(
        epd_servers,
        "subprocess",
        types.SimpleNamespace(
            Popen=factory,
            PIPE=subprocess.PIPE,
            TimeoutExpired=subprocess.TimeoutExpired,
        ),
    )


def _make_server(monkeypatch, **kwargs) -> tuple[RemoteEPDServer, _FakePopenFactory]:
    factory = _FakePopenFactory(
        exit_codes=kwargs.pop("exit_codes", None),
        timeout_indexes=kwargs.pop("timeout_indexes", None),
    )
    _install_fake_popen(monkeypatch, factory)
    vllm_serve_args = kwargs.pop(
        "vllm_serve_args",
        [["--model", "dummy", "--port", ENCODE_PORT], ["--model", "dummy", "--port", PREFILL_DECODE_PORT]],
    )
    server = RemoteEPDServer(vllm_serve_args=vllm_serve_args, max_wait_seconds=1, **kwargs)
    return server, factory


@pytest.fixture(autouse=True)
def _mock_waits_and_process_tree(monkeypatch):
    """Never wait on HTTP readiness and never kill real process trees."""
    monkeypatch.setattr(RemoteOpenAIServer, "_wait_for_multiple_servers", lambda self, *args, **kwargs: None)
    monkeypatch.setattr(RemoteOpenAIServer, "_terminate_process_tree", lambda self, proc: None)


def test_assigns_devices_and_environment(monkeypatch):
    server, factory = _make_server(monkeypatch)

    assert len(factory.calls) == 2
    expected_ports = [ENCODE_PORT, PREFILL_DECODE_PORT]
    for index, (cmd, env) in enumerate(factory.calls):
        assert cmd[:2] == ["vllm", "serve"]
        assert cmd[-2:] == ["--port", expected_ports[index]]
        assert env["ASCEND_RT_VISIBLE_DEVICES"] == str(index)
        assert env["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] == "1"
        assert env["PYTORCH_NPU_ALLOC_CONF"] == "expandable_segments:True"
        assert env["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"
    assert server.port == int(PREFILL_DECODE_PORT)
    assert server.exit_codes == []


def test_rejects_missing_port(monkeypatch):
    factory = _FakePopenFactory()
    _install_fake_popen(monkeypatch, factory)

    with pytest.raises(ValueError, match="port"):
        RemoteEPDServer(vllm_serve_args=[["--model", "dummy"]], max_wait_seconds=1)
    assert factory.calls == []


def test_per_instance_env_overrides(monkeypatch):
    server, factory = _make_server(
        monkeypatch,
        vllm_serve_args=[["--port", ENCODE_PORT], ["--port", PREFILL_DECODE_PORT]],
        per_instance_env=[
            {"VLLM_USE_V2_MODEL_RUNNER": "0"},
            {"VLLM_USE_V2_MODEL_RUNNER": "1", "EXTRA_FLAG": 7},
        ],
    )

    first_env = factory.calls[0][1]
    second_env = factory.calls[1][1]
    assert first_env["VLLM_USE_V2_MODEL_RUNNER"] == "0"
    assert second_env["VLLM_USE_V2_MODEL_RUNNER"] == "1"
    assert second_env["EXTRA_FLAG"] == "7"
    assert first_env["ASCEND_RT_VISIBLE_DEVICES"] == "0"
    assert second_env["ASCEND_RT_VISIBLE_DEVICES"] == "1"
    for env in (first_env, second_env):
        assert env["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"
    # Per-instance environments are copies; the shared env dict is not mutated.
    assert "ASCEND_RT_VISIBLE_DEVICES" not in server.env_dict

    with pytest.raises(ValueError, match="per_instance_env"):
        RemoteEPDServer(
            vllm_serve_args=[["--port", ENCODE_PORT]],
            per_instance_env=[{}, {}],
            max_wait_seconds=1,
        )


def test_delete_shm_parses_and_removes(monkeypatch, tmp_path):
    factory = _FakePopenFactory()
    _install_fake_popen(monkeypatch, factory)

    shm_dir = tmp_path / "ec_cache"
    shm_dir.mkdir()
    (shm_dir / "entry.safetensors").write_text("data")
    keep_dir = tmp_path / "keep"
    keep_dir.mkdir()
    ec_config = json.dumps(
        {
            "ec_connector": "ECExampleConnector",
            "ec_role": "ec_producer",
            "ec_connector_extra_config": {"shared_storage_path": str(shm_dir)},
        }
    )
    server = RemoteEPDServer(
        vllm_serve_args=[
            ["--port", ENCODE_PORT, "--ec-transfer-config", ec_config],
            ["--port", PREFILL_DECODE_PORT],
        ],
        max_wait_seconds=1,
    )

    assert server._delete_shm() == [str(shm_dir)]
    assert not shm_dir.exists()
    assert keep_dir.exists()
    # Idempotent: a repeated cleanup attempt is safe.
    assert server._delete_shm() == [str(shm_dir)]

    # An unparsable connector config warns but must not break teardown.
    bad_server = RemoteEPDServer(
        vllm_serve_args=[["--port", PREFILL_DECODE_PORT, "--ec-transfer-config", "{not-json"]],
        max_wait_seconds=1,
    )
    assert bad_server._delete_shm() == []


def test_exit_deletes_shm_and_asserts_clean_exit(monkeypatch):
    deleted: list[RemoteEPDServer] = []

    def _record_delete(self) -> list[str]:
        deleted.append(self)
        return []

    monkeypatch.setattr(RemoteEPDServer, "_delete_shm", _record_delete)

    server, _ = _make_server(monkeypatch, exit_codes=[0, -15])
    with server:
        pass
    assert server.exit_codes == [0, -15]
    assert deleted == [server]

    # Exit code outside {0, -SIGTERM} fails the teardown assertion.
    server, _ = _make_server(monkeypatch, exit_codes=[0, 1])
    with pytest.raises(AssertionError, match="unexpected server exit codes"), server:
        pass
    assert server.exit_codes == [0, 1]

    # A process that is still alive fails the teardown assertion.
    server, _ = _make_server(monkeypatch, exit_codes=[0], timeout_indexes={0})
    with pytest.raises(AssertionError, match="still alive"), server:
        pass


def test_exit_assertion_can_be_disabled(monkeypatch):
    server, _ = _make_server(monkeypatch, exit_codes=[0, 1], assert_clean_exit=False)
    with server:
        pass
    assert server.exit_codes == [0, 1]


def test_exit_does_not_mask_exception(monkeypatch):
    server, _ = _make_server(
        monkeypatch,
        vllm_serve_args=[["--model", "dummy", "--port", ENCODE_PORT]],
        exit_codes=[1],
    )
    with pytest.raises(RuntimeError, match="boom"), server:
        raise RuntimeError("boom")
    assert server.exit_codes == [1]


def test_disagg_epd_proxy_uses_sys_executable(monkeypatch):
    factory = _FakePopenFactory(exit_codes=[0])
    _install_fake_popen(monkeypatch, factory)

    proxy = DisaggEpdProxy(proxy_args=["--host", "127.0.0.1", "--port", PROXY_PORT], max_wait_seconds=1)
    cmd, _ = factory.calls[0]
    assert cmd[0] == sys.executable
    assert cmd[1] == str(DISAGG_EPD_PROXY_SCRIPT)
    assert DISAGG_EPD_PROXY_SCRIPT.exists()
    assert proxy.port == int(PROXY_PORT)
    with proxy:
        pass
    assert proxy.exit_codes == [0]

    # The proxy keeps the optional teardown-assertion switch too.
    factory = _FakePopenFactory(exit_codes=[1])
    _install_fake_popen(monkeypatch, factory)
    proxy = DisaggEpdProxy(
        proxy_args=["--port", PROXY_PORT],
        max_wait_seconds=1,
        assert_clean_exit=False,
    )
    with proxy:
        pass
    assert proxy.exit_codes == [1]

    with pytest.raises(ValueError, match="port"):
        DisaggEpdProxy(proxy_args=["--host", "127.0.0.1"], max_wait_seconds=1)
