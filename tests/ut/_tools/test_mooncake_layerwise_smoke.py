# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of the CI preflight; doubles do not establish native SDK correctness."""

import ast
import ctypes
import importlib.util
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / ".github/workflows/scripts/check_mooncake_layerwise.py"
LAYERWISE_TEST = "tests/e2e/pull_request/one_card/test_ascendstore_v1_layerwise.py"


@pytest.fixture
def smoke():
    spec = importlib.util.spec_from_file_location("check_mooncake_layerwise", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def store():
    objects: dict[str, bytearray] = {}
    client = Mock()
    client.setup.return_value = 0
    client.register_buffer.return_value = 0
    client.batch_put_session_end.side_effect = lambda keys: [0] * len(keys)
    client.batch_put_session_revoke.side_effect = lambda keys: [0] * len(keys)
    client.batch_get_session_start.side_effect = lambda keys: [0] * len(keys)
    client.batch_get_session_end.return_value = 0

    def start(keys, sizes):
        objects.update((key, bytearray(size)) for key, size in zip(keys, sizes, strict=True))
        return [0] * len(keys)

    def put(keys, buffers, sizes, offsets):
        for key, pointers, lengths, starts in zip(keys, buffers, sizes, offsets, strict=True):
            for pointer, length, offset in zip(pointers, lengths, starts, strict=True):
                objects[key][offset : offset + length] = ctypes.string_at(pointer, length)
        return [sum(lengths) for lengths in sizes]

    def get(keys, buffers, sizes, offsets):
        for key, pointers, lengths, starts in zip(keys, buffers, sizes, offsets, strict=True):
            for pointer, length, offset in zip(pointers, lengths, starts, strict=True):
                ctypes.memmove(pointer, bytes(objects[key][offset : offset + length]), length)
        return [sum(lengths) for lengths in sizes]

    client.batch_put_session_start.side_effect = start
    client.batch_put_from_multi_buffer_ranges.side_effect = put
    client.batch_get_into_multi_buffer_ranges.side_effect = get
    return SimpleNamespace(MooncakeDistributedStore=Mock(return_value=client)), client


def test_native_contract_matches_production_without_importing_vllm(smoke):
    backend = REPO_ROOT / "vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/mooncake_backend.py"
    source = ast.parse(backend.read_text())
    declaration = next(
        node
        for node in source.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "MOONCAKE_LAYERWISE_CLIENT_METHODS" for target in node.targets
        )
    )
    assert ast.literal_eval(declaration.value) == smoke.REQUIRED_METHODS


def test_smoke_writes_reverse_layers_reads_forward_and_revokes(smoke, store):
    module, client = store
    smoke.exercise_sessions(module, "127.0.0.1:1", "127.0.0.1:2")
    writes = client.batch_put_from_multi_buffer_ranges.call_args_list
    reads = client.batch_get_into_multi_buffer_ranges.call_args_list
    assert [call.args[3][0][0] for call in writes] == [256, 128, 0]
    assert [call.args[3][0][0] for call in reads] == [0, 128, 256]
    assert all(len(call.args[0]) == 2 and len(call.args[1][0]) == 2 for call in writes + reads)
    client.batch_put_session_revoke.assert_called_once_with(["layerwise-sdk-smoke-revoke"])
    client.batch_get_session_end.assert_called_once()
    client.close.assert_called_once()


@pytest.mark.parametrize("failure", ["setup", "put", "short_put", "get", "no_op_get", "revoke"])
def test_smoke_fails_closed_and_releases_client(smoke, store, failure):
    module, client = store
    if failure == "setup":
        client.setup.return_value = -1
    elif failure in ("put", "short_put"):
        client.batch_put_from_multi_buffer_ranges.side_effect = lambda *args: [-1, -1] if failure == "put" else []
    elif failure == "get":
        client.batch_get_into_multi_buffer_ranges.side_effect = RuntimeError("injected read failure")
    elif failure == "no_op_get":
        client.batch_get_into_multi_buffer_ranges.side_effect = lambda *args: [smoke.LAYER_BYTES] * smoke.KEY_COUNT
    elif failure == "revoke":
        client.batch_put_session_revoke.side_effect = lambda *args: [-1]
    with pytest.raises(RuntimeError):
        smoke.exercise_sessions(module, "127.0.0.1:1", "127.0.0.1:2")
    client.close.assert_called_once()
    if failure in ("put", "short_put"):
        client.batch_put_session_revoke.assert_called_once()
        client.batch_get_session_start.assert_not_called()
    if failure in ("get", "no_op_get"):
        client.batch_get_session_end.assert_called_once()


@pytest.mark.parametrize(
    "failure", ["version", "methods", "store_package", "master_package", "module_origin", "native"]
)
def test_preflight_rejects_missing_or_mixed_sdk_before_model_start(smoke, monkeypatch, failure):
    monkeypatch.setattr(smoke.sys, "argv", [str(SCRIPT_PATH), "--expected-version", "0.3.13.post1"])
    wheel = SimpleNamespace(version="0.3.13.post1", locate_file=lambda _: Path("/wheel/mooncake"))
    store_module = SimpleNamespace(__file__="/wheel/mooncake/store.so", MooncakeDistributedStore=Mock())
    cli_module = SimpleNamespace(__file__="/wheel/mooncake/cli.py")
    if failure == "version":
        wheel.version = "0.3.11.post1"
    elif failure == "methods":
        store_module.MooncakeDistributedStore = object
    elif failure == "store_package":
        store_module.__file__ = "/old/mooncake/store.so"
    elif failure == "master_package":
        cli_module.__file__ = "/old/mooncake/cli.py"
    elif failure == "module_origin":
        store_module.__file__ = None
    monkeypatch.setattr(smoke, "distribution", lambda _: wheel)
    imports = Mock(side_effect=[store_module, cli_module])
    if failure == "native":
        imports.side_effect = ImportError("missing native dependency")
    monkeypatch.setattr(smoke, "import_module", imports)
    launch = Mock()
    monkeypatch.setattr(smoke.subprocess, "Popen", launch)
    with pytest.raises((RuntimeError, ImportError)):
        smoke.main()
    launch.assert_not_called()


def test_exited_master_fails_without_waiting(smoke):
    master = Mock(returncode=127)
    master.poll.return_value = 127
    with pytest.raises(RuntimeError, match="127"):
        smoke.wait_for_master(master, 1)


def test_master_readiness_timeout_is_bounded(smoke, monkeypatch):
    monkeypatch.setattr(smoke.time, "monotonic", Mock(side_effect=[0, smoke.MASTER_READY_SECONDS + 1]))
    with pytest.raises(TimeoutError, match="did not become ready"):
        smoke.wait_for_master(Mock(), 1)


@pytest.mark.parametrize("failure", [False, True])
def test_main_stops_owned_master_on_success_or_range_failure(smoke, monkeypatch, failure):
    monkeypatch.setattr(smoke.sys, "argv", [str(SCRIPT_PATH), "--expected-version", "0.3.13.post1"])
    wheel = SimpleNamespace(version="0.3.13.post1", locate_file=lambda _: Path("/wheel/mooncake"))
    store_module = SimpleNamespace(__file__="/wheel/mooncake/store.so", MooncakeDistributedStore=Mock())
    cli_module = SimpleNamespace(__file__="/wheel/mooncake/cli.py")
    monkeypatch.setattr(smoke, "distribution", lambda _: wheel)
    monkeypatch.setattr(smoke, "import_module", Mock(side_effect=[store_module, cli_module]))
    reservations = [MagicMock() for _ in range(3)]
    for port, reserved in enumerate(reservations, start=10000):
        reserved.__enter__.return_value = reserved
        reserved.getsockname.return_value = (smoke.HOST, port)
    monkeypatch.setattr(smoke.socket, "socket", Mock(side_effect=reservations))
    master = MagicMock()
    master.__enter__.return_value = master
    if failure:
        master.wait.side_effect = [subprocess.TimeoutExpired("master", 5), 0]
    launch = Mock(return_value=master)
    monkeypatch.setattr(smoke.subprocess, "Popen", launch)
    monkeypatch.setattr(smoke, "wait_for_master", Mock())
    exercise = Mock(side_effect=RuntimeError("injected range failure") if failure else None)
    monkeypatch.setattr(smoke, "exercise_sessions", exercise)
    if failure:
        with pytest.raises(RuntimeError, match="injected range failure"):
            smoke.main()
        master.kill.assert_called_once()
    else:
        smoke.main()
        master.kill.assert_not_called()
    master.terminate.assert_called_once()
    exercise.assert_called_once_with(store_module, "127.0.0.1:10000", "127.0.0.1:10002")
    assert launch.call_args.args[0][:3] == [smoke.sys.executable, "-m", "mooncake.cli"]


def test_ci_scopes_new_wheel_and_native_preflight_to_layerwise():
    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/_selected_tests.yaml").read_text())
    steps = workflow["jobs"]["selected-tests"]["steps"]
    install = next(step for step in steps if step.get("name") == "Install Mooncake wheel")
    preflight = next(step for step in steps if step.get("name") == "Validate Mooncake Layerwise SDK")
    build = next(step for step in steps if step.get("name") == "Install vllm-project/vllm-ascend with device")
    assert install["env"]["MOONCAKE_WHEEL"] == (
        "${{ contains(matrix.group.tests, '" + LAYERWISE_TEST + "') && "
        "'mooncake-transfer-engine-npu==0.3.13.post1' || 'mooncake-transfer-engine-npu==0.3.11.post1' }}"
    )
    assert '"$MOONCAKE_WHEEL"' in install["run"]
    condition = "${{ matrix.group.npu_type != 'cpu' && contains(matrix.group.tests, '" + LAYERWISE_TEST + "') }}"
    assert preflight["if"] == condition
    assert preflight["timeout-minutes"] == 3
    assert steps.index(install) < steps.index(preflight) < steps.index(build)
    assert "check_mooncake_layerwise.py --expected-version 0.3.13.post1" in preflight["run"]
    subprocess.run(["bash", "-n", "-c", install["run"]], check=True, capture_output=True)
