"""CPU regressions for opt-in Mooncake producer startup recovery."""

import json
import socket
import subprocess
import sys
import time
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest

from tests.e2e.common.multi_node.external_dp import mooncake_ports as ports
from tests.e2e.common.multi_node.external_dp import runtime
from tests.e2e.common.multi_node.external_dp.external_dp_config import ExternalDPConfigLoader, RankResolver

M3_CONFIG = (
    Path(__file__).resolve().parents[3] / "tests/SIT/cases/models/configs/MiniMax/MiniMax-M3-w8a8-PP-prefix-PD.yaml"
)
BLOCK_SIZE = 16


@pytest.fixture
def blocks(monkeypatch):
    # Find real contiguous blocks without relying on bind(port=0) adjacency.
    monkeypatch.setattr(ports, "_kernel_port_ranges", lambda: (None, set()))
    monkeypatch.setattr(ports, "_socket_snapshot", lambda *_: "test socket snapshot")
    found = []
    for base in range(20000, 30000, BLOCK_SIZE):
        try:
            with ExitStack() as stack:
                for port in range(base, base + BLOCK_SIZE):
                    sock = stack.enter_context(socket.socket())
                    sock.bind(("127.0.0.1", port))
        except OSError:
            continue
        found.append(base)
        if len(found) == 3:
            return found
    pytest.fail("Could not find three local TCP port blocks")


def collision(host, port):
    return (
        "ERROR [mooncake_connector.py:334] Mooncake KVCacheSendingThread encountered exception. "
        f"Error: Address already in use (addr='tcp://{host}:{port}')\n"
    )


@pytest.mark.parametrize("offset", [1, BLOCK_SIZE - 1])
def test_busy_worker_port_skips_whole_block_and_releases_probes(blocks, offset):
    rejected = []
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", blocks[0] + offset))
        assert ports.select_port_base("127.0.0.1", blocks, BLOCK_SIZE, rejected) == blocks[1]
        assert rejected[0]["base"] == blocks[0]
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", blocks[0]))
    # The selected block is also released for the server to bind.
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", blocks[1] + BLOCK_SIZE - 1))


def test_dynamic_range_is_avoided_unless_entire_overlap_is_reserved(blocks, monkeypatch):
    low, high = blocks[0], blocks[0] + BLOCK_SIZE - 1
    monkeypatch.setattr(ports, "_kernel_port_ranges", lambda: ((low, high), {low}))
    rejected = []
    assert ports.select_port_base("127.0.0.1", blocks, BLOCK_SIZE, rejected) == blocks[1]
    assert "automatic TCP" in rejected[0]["reason"]
    monkeypatch.setattr(ports, "_kernel_port_ranges", lambda: ((low, high), set(range(low, high + 1))))
    assert ports.select_port_base("127.0.0.1", blocks, BLOCK_SIZE, []) == blocks[0]


def test_kernel_policy_uses_actual_ranges(tmp_path, monkeypatch):
    (tmp_path / "ip_local_port_range").write_text("30000 40000\n")
    (tmp_path / "ip_local_reserved_ports").write_text("30001-30003,31000\n")
    monkeypatch.setattr(ports, "Path", lambda _: tmp_path)
    assert ports._kernel_port_ranges() == ((30000, 40000), {30001, 30002, 30003, 31000})


@pytest.mark.parametrize(
    "raw",
    [
        {},
        {"bases": []},
        {"bases": [True]},
        {"bases": [0]},
        {"bases": [65536]},
        {"bases": [26000, 26000]},
        {"bases": [26000], "max_attempts": 0},
        {"bases": [26000], "max_attempts": True},
        {"bases": [26000], "max_attempts": 2},
        {"bases": [26000], "max_attempts": 1, "typo": 1},
    ],
)
def test_invalid_policy_fails_before_start(raw):
    with pytest.raises(ValueError, match="mooncake_port_retry"):
        ports.MooncakePortRetryConfig.parse(raw)


@pytest.fixture
def config():
    return ExternalDPConfigLoader.from_yaml(str(M3_CONFIG), cluster_ips=["127.0.0.1", "127.0.0.2", "127.0.0.3"])


@pytest.mark.parametrize("invalid", ["decoder", "dp", "multi_host", "cp", "overflow", "overlap", "consumer"])
def test_policy_is_restricted_to_supported_producer_topology(config, invalid):
    templates = list(config.launch_templates)
    nodes = list(config.nodes)
    if invalid == "decoder":
        templates[1] = replace(templates[1], mooncake_port_retry=templates[0].mooncake_port_retry)
    elif invalid in ("dp", "multi_host", "cp"):
        fields = {"dp": {"dp_size": 2}, "multi_host": {"nnodes": 2}, "cp": {"cp_size": 2}}[invalid]
        nodes[0] = replace(nodes[0], **fields)
    elif invalid in ("overflow", "overlap"):
        bases = (65530, 65531) if invalid == "overflow" else (26000, 26001)
        templates[0] = replace(templates[0], mooncake_port_retry=ports.MooncakePortRetryConfig(bases, 2))
    else:
        args = [arg.replace('"kv_producer"', '"kv_consumer"') for arg in templates[0].server_cmd_template]
        templates[0] = replace(templates[0], server_cmd_template=args)
    with pytest.raises(ValueError, match="mooncake_port_retry"):
        ExternalDPConfigLoader._validate_mooncake_port_retry(replace(config, nodes=nodes, launch_templates=templates))


def test_only_producer_kv_port_changes(config):
    assert [template.mooncake_port_retry is not None for template in config.launch_templates] == [True, False, False]
    original = config.launch_templates[0].server_cmd_template
    changed = ports.with_kv_port(original, 26100)
    index = original.index("--kv-transfer-config") + 1
    assert changed[:index] == original[:index] and changed[index + 1 :] == original[index + 1 :]
    expected = json.loads(original[index])
    expected["kv_port"] = "26100"
    assert json.loads(changed[index]) == expected
    assert json.loads(original[index])["kv_port"] == "36000"
    for template in config.launch_templates[1:]:
        args = template.server_cmd_template
        assert json.loads(args[args.index("--kv-transfer-config") + 1])["kv_port"] == "36100"


def test_monitor_ignores_old_logs_and_handles_split_lines(tmp_path):
    log = tmp_path / "rank.log"
    log.write_text(collision("127.0.0.1", 26001))
    monitor = ports.StartupLogMonitor(log, "127.0.0.1", 26000, BLOCK_SIZE)
    monitor.check()
    error = collision("127.0.0.1", 26015)
    with log.open("a") as out:
        out.write(error[:50])
    monitor.check()
    with log.open("a") as out:
        out.write(error[50:])
    with pytest.raises(ports.MooncakePortCollision, match="26015"):
        monitor.check()


@pytest.mark.parametrize("host,port", [("127.0.0.2", 26001), ("127.0.0.1", 25999), ("127.0.0.1", 26016)])
def test_unrelated_address_is_not_retryable(tmp_path, host, port):
    log = tmp_path / "rank.log"
    monitor = ports.StartupLogMonitor(log, "127.0.0.1", 26000, BLOCK_SIZE)
    log.write_text(collision(host, port))
    monitor.check()
    with log.open("a") as out:
        out.write("RuntimeError: EngineCore failed to start: unrelated error\n")
    with pytest.raises(RuntimeError, match="unrelated error") as error:
        monitor.check()
    assert not isinstance(error.value, ports.MooncakePortCollision)


@pytest.fixture
def manager(config, blocks, tmp_path, monkeypatch):
    templates = list(config.launch_templates)
    templates[0] = replace(templates[0], mooncake_port_retry=ports.MooncakePortRetryConfig(tuple(blocks), 3))
    config = replace(config, launch_templates=templates)
    monkeypatch.setattr(runtime, "build_dist_envs", lambda *_: {})
    monkeypatch.setattr(runtime.ProfileSpec, "from_env", lambda: runtime.ProfileSpec(enabled=False))
    monkeypatch.setattr(runtime.time, "sleep", Mock(side_effect=AssertionError("Unexpected readiness sleep")))
    manager = runtime.ExternalDPServerManager(
        config=config, ranks=RankResolver(config).resolve(), current_node_index=0, log_root=tmp_path
    )
    return manager


@pytest.mark.parametrize("failure_count", [0, 1, 3])
def test_manager_recovers_live_parent_bind_failure_with_bounded_owned_cleanup(manager, monkeypatch, failure_count):
    started = []
    stop = Mock()

    def launch(cmd, env, path):
        process = Mock(pid=10000 + len(started))
        process.poll.return_value = None  # Reproduce the surviving API parent.
        base = int(json.loads(cmd[cmd.index("--kv-transfer-config") + 1])["kv_port"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(collision("127.0.0.1", base + 1) if len(started) < failure_count else "startup complete\n")
        started.append((process, cmd, path))
        return process

    monkeypatch.setattr(runtime, "start_logged_process", launch)
    monkeypatch.setattr(runtime, "terminate_process_tree", stop)
    monkeypatch.setattr(runtime, "is_http_ready", lambda *_args, **_kwargs: True)
    if failure_count == 3:
        with pytest.raises(ports.MooncakePortCollision):
            manager.start_current_node()
        assert not manager.rank_processes
    else:
        manager.start_current_node()
        rank = manager.ranks[0]
        assert manager.effective_commands[rank].cmd == started[-1][1]
        assert str(manager.config.launch_templates[0].mooncake_port_retry.bases[failure_count]) in (
            manager.effective_commands[rank].display_cmd
        )
        assert manager.rank_processes[0][2] == started[-1][2]
        assert stop.call_count == failure_count
        manager.cleanup()
    assert len(started) == min(failure_count + 1, 3)
    assert [call.args[0] for call in stop.call_args_list] == [process.pid for process, _, _ in started]
    history = json.loads((manager.log_root / "node-0/rank-0.port-selection.json").read_text())
    assert [item["status"] for item in history["attempts"]] == (
        ["failed"] * failure_count + (["ready"] if failure_count < 3 else [])
    )
    assert len({path for _, _, path in started}) == len(started)
    assert all(path.exists() for _, _, path in started)


def test_busy_candidates_fail_without_loading_model(manager, monkeypatch, blocks):
    start = Mock()
    monkeypatch.setattr(runtime, "start_logged_process", start)
    with ExitStack() as sockets:
        for base in blocks:
            sock = sockets.enter_context(socket.socket())
            sock.bind(("127.0.0.1", base + 1))
        with pytest.raises(RuntimeError, match="No available Mooncake"):
            manager.start_current_node()
    start.assert_not_called()


def test_port_stolen_after_probe_retries_real_child_and_preserves_owner(blocks, tmp_path, monkeypatch):
    processes = {}
    selected = []
    select = ports.select_port_base
    child = """
import json
import socket
import sys
import time
base = int(json.loads(sys.argv[-1])["kv_port"])
sock = socket.socket()
try:
    sock.bind(("127.0.0.1", base + 1))
    print("READY", flush=True)
except OSError:
    print("Mooncake KVCacheSendingThread encountered exception. "
          "Address already in use (addr='tcp://127.0.0.1:%d')" % (base + 1), flush=True)
time.sleep(30)
"""
    with socket.socket() as owner:

        def steal_after_probe(host, bases, count, rejected):
            base = select(host, bases, count, rejected)
            if not selected:
                owner.bind((host, base + 1))
            selected.append(base)
            return base

        def launch(cmd, env, path):
            with path.open("wb") as log:
                process = subprocess.Popen(
                    [sys.executable, "-u", "-c", child, cmd[-1]], stdout=log, stderr=subprocess.STDOUT
                )
            processes[process.pid] = process
            return process

        def wait_ready(process, path, check):
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                check()
                assert process.poll() is None
                if "READY" in path.read_text():
                    return
                time.sleep(0.01)
            pytest.fail("Test child did not report a startup outcome")

        monkeypatch.setattr(ports, "select_port_base", steal_after_probe)
        cmd = ["--kv-transfer-config", '{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_producer"}']
        try:
            process, _, _ = ports.start_with_port_retry(
                cmd=cmd,
                env={},
                host="127.0.0.1",
                count=BLOCK_SIZE,
                policy=ports.MooncakePortRetryConfig(tuple(blocks), 3),
                log_file=tmp_path / "rank.log",
                start_process=launch,
                wait_ready=wait_ready,
                stop_process=lambda pid: processes[pid].terminate(),
            )
            assert selected == blocks[:2]
            assert [proc.poll() is None for proc in processes.values()] == [False, True]
            assert process.poll() is None
            assert owner.getsockname()[1] == blocks[0] + 1
        finally:
            for proc in processes.values():
                if proc.poll() is None:
                    proc.terminate()
                proc.wait(timeout=5)


@pytest.mark.parametrize("failure", ["RuntimeError: EngineCore failed to start: out of memory\n", "timeout"])
def test_other_startup_failures_are_not_retried(manager, monkeypatch, failure):
    start = Mock()
    process = Mock(pid=10000)
    process.poll.return_value = None

    def launch(cmd, env, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(failure)
        start()
        return process

    stop = Mock()
    monkeypatch.setattr(runtime, "start_logged_process", launch)
    monkeypatch.setattr(runtime, "terminate_process_tree", stop)
    if failure == "timeout":
        monkeypatch.setattr(runtime, "wait_ranks_ready", Mock(side_effect=TimeoutError("not ready")))
    with pytest.raises((RuntimeError, TimeoutError)):
        manager.start_current_node()
    assert start.call_count == 1
    stop.assert_called_once_with(process.pid)
    assert not manager.rank_processes


def test_disabled_policy_uses_original_startup_path(manager, monkeypatch):
    templates = list(manager.config.launch_templates)
    templates[0] = replace(templates[0], mooncake_port_retry=None)
    manager.config = replace(manager.config, launch_templates=templates)
    expected = manager.command_builder.build(manager.ranks[0], templates[0])
    start = Mock(return_value=Mock(pid=10000))
    monkeypatch.setattr(runtime, "start_logged_process", start)
    monkeypatch.setattr(runtime, "wait_ranks_ready", Mock())
    monkeypatch.setattr(runtime, "start_with_port_retry", Mock(side_effect=AssertionError("Opt-in only")))
    manager.start_current_node()
    start.assert_called_once_with(expected.cmd, expected.env, manager.log_root / "node-0/rank-0.log")
    assert not manager.effective_commands
