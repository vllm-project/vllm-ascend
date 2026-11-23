"""Opt-in startup recovery for a single-node Mooncake KV producer."""

import errno
import json
import logging
import socket
import subprocess
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import regex as re

logger = logging.getLogger(__name__)
MAX_TCP_PORT = 65535
DEFAULT_MAX_ATTEMPTS = 3


@dataclass(frozen=True)
class MooncakePortRetryConfig:
    """Candidate KV blocks and the total launch limit, including the first try."""

    bases: tuple[int, ...]
    max_attempts: int = DEFAULT_MAX_ATTEMPTS

    @classmethod
    def parse(cls, raw: Any) -> "MooncakePortRetryConfig | None":
        if raw is None:
            return None
        if not isinstance(raw, dict) or set(raw) - {"bases", "max_attempts"}:
            raise ValueError("mooncake_port_retry requires bases and optional max_attempts")
        bases = raw.get("bases")
        if not isinstance(bases, list) or not bases:
            raise ValueError("mooncake_port_retry.bases must be a non-empty list")
        if any(type(base) is not int or not 1 <= base <= MAX_TCP_PORT for base in bases):
            raise ValueError("mooncake_port_retry.bases must contain TCP port integers")
        if len(set(bases)) != len(bases):
            raise ValueError("mooncake_port_retry.bases must be unique")
        attempts = raw.get("max_attempts", DEFAULT_MAX_ATTEMPTS)
        if type(attempts) is not int or not 1 <= attempts <= len(bases):
            raise ValueError("mooncake_port_retry.max_attempts must be between 1 and the number of bases")
        return cls(tuple(bases), attempts)


def with_kv_port(cmd: list[str], base: int) -> list[str]:
    if cmd.count("--kv-transfer-config") != 1:
        raise ValueError("mooncake_port_retry requires one --kv-transfer-config")
    index = cmd.index("--kv-transfer-config") + 1
    config = json.loads(cmd[index])
    if config.get("kv_connector") != "MooncakeConnectorV1" or config.get("kv_role") != "kv_producer":
        raise ValueError("mooncake_port_retry only supports MooncakeConnectorV1 kv_producer")
    config["kv_port"] = str(base)
    return [*cmd[:index], json.dumps(config), *cmd[index + 1 :]]


def _kernel_port_ranges() -> tuple[tuple[int, int] | None, set[int]]:
    """Read this network namespace's allocation policy; never change sysctls."""
    root = Path("/proc/sys/net/ipv4")
    if not root.exists():
        return None, set()
    low, high = map(int, (root / "ip_local_port_range").read_text().split())
    reserved: set[int] = set()
    for item in (root / "ip_local_reserved_ports").read_text().strip().split(","):
        if item:
            bounds = list(map(int, item.split("-")))
            reserved.update(range(bounds[0], bounds[-1] + 1))
    return (low, high), reserved


def _socket_snapshot(base: int, count: int) -> str:
    try:
        result = subprocess.run(
            ["ss", "-H", "-tanpe", f"( sport >= :{base} and sport <= :{base + count - 1} )"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return result.stdout + result.stderr
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"Socket snapshot unavailable: {exc}"


def select_port_base(host: str, bases: list[int], count: int, rejected: list[dict[str, Any]]) -> int:
    automatic, reserved = _kernel_port_ranges()
    for base in bases:
        if not 1 <= base <= base + count - 1 <= MAX_TCP_PORT:
            raise ValueError("Mooncake port block exceeds TCP port range")
        if automatic is not None and any(
            automatic[0] <= port <= automatic[1] and port not in reserved for port in range(base, base + count)
        ):
            rejected.append({"base": base, "reason": "overlaps automatic TCP allocation range"})
            continue
        try:
            # Hold the entire block while probing; release before vLLM binds it.
            # A later collision is still possible and is handled during startup.
            with ExitStack() as sockets:
                for port in range(base, base + count):
                    sock = sockets.enter_context(socket.socket(socket.AF_INET, socket.SOCK_STREAM))
                    sock.bind((host, port))
        except OSError as exc:
            if exc.errno != errno.EADDRINUSE:
                raise
            rejected.append({"base": base, "reason": str(exc), "sockets": _socket_snapshot(base, count)})
            continue
        return base
    raise RuntimeError(f"No available Mooncake KV port block on {host}: {rejected}")


class MooncakePortCollision(RuntimeError):
    """Only this startup failure permits another port selection and launch."""


class StartupLogMonitor:
    def __init__(self, log_file: Path, host: str, base: int, count: int):
        self.log_file = log_file
        self.host = host
        self.base = base
        self.count = count
        self.offset = log_file.stat().st_size if log_file.exists() else 0
        self.pending = ""

    def check(self) -> None:
        if not self.log_file.exists():
            return
        with self.log_file.open("rb") as log:
            log.seek(self.offset)
            text = self.pending + log.read().decode("utf-8", errors="replace")
            self.offset = log.tell()
        lines = text.split("\n")
        self.pending = lines.pop()
        for line in lines:
            if "Mooncake KVCacheSendingThread encountered exception" in line and "Address already in use" in line:
                address = re.search(r"addr=['\"]tcp://([^:'\"]+):(\d+)", line)
                if address and address[1] == self.host and self.base <= int(address[2]) < self.base + self.count:
                    raise MooncakePortCollision(f"Mooncake KV port collision: {line.strip()}; log={self.log_file}")
            if "EngineCore failed to start" in line or "KV Cache sending/receiving thread failed to start" in line:
                raise RuntimeError(f"vLLM startup failed: {line.strip()}; log={self.log_file}")


def start_with_port_retry(
    *,
    cmd: list[str],
    env: dict[str, str],
    host: str,
    count: int,
    policy: MooncakePortRetryConfig,
    log_file: Path,
    start_process: Callable[[list[str], dict[str, str], Path], subprocess.Popen],
    wait_ready: Callable[[subprocess.Popen, Path, Callable[[], None]], None],
    stop_process: Callable[[int], None],
) -> tuple[subprocess.Popen, list[str], Path]:
    """Retry before readiness only; leave the successful process to its manager."""
    remaining = list(policy.bases)
    history: dict[str, Any] = {"host": host, "port_count": count, "rejected": [], "attempts": []}
    manifest = log_file.with_suffix(".port-selection.json")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    try:
        for attempt in range(1, policy.max_attempts + 1):
            base = select_port_base(host, remaining, count, history["rejected"])
            # Never reuse a failed block or revisit previously rejected blocks.
            remaining = remaining[remaining.index(base) + 1 :]
            attempt_cmd = with_kv_port(cmd, base)
            attempt_log = log_file.with_name(f"{log_file.stem}.kv-attempt-{attempt}.log")
            monitor = StartupLogMonitor(attempt_log, host, base, count)
            record = {"attempt": attempt, "base": base, "log": str(attempt_log), "status": "starting"}
            history["attempts"].append(record)
            manifest.write_text(json.dumps(history, indent=2), encoding="utf-8")
            logger.info("Mooncake startup attempt=%d ports=%d-%d log=%s", attempt, base, base + count - 1, attempt_log)
            process = start_process(attempt_cmd, env, attempt_log)
            try:
                wait_ready(process, attempt_log, monitor.check)
            except BaseException as exc:
                record.update(status="failed", error=str(exc))
                if isinstance(exc, MooncakePortCollision):
                    record["sockets"] = _socket_snapshot(base, count)
                # Only stop the process tree created by this launch attempt.
                stop_process(process.pid)
                process.wait(timeout=30)
                if not isinstance(exc, MooncakePortCollision) or attempt == policy.max_attempts:
                    raise
                logger.warning("%s; retrying producer startup with another KV port block", exc)
                continue
            record["status"] = "ready"
            return process, attempt_cmd, attempt_log
        raise RuntimeError("Mooncake startup attempts exhausted")
    finally:
        manifest.write_text(json.dumps(history, indent=2), encoding="utf-8")
