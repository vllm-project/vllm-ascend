import logging
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path

from tests.e2e.nightly.multi_node.external_dp.scripts.external_dp_config import (
    SERVER_HEALTH_PATH,
    SERVER_READY_TIMEOUT_SECONDS,
    ExternalDPConfig,
    ExternalDPConfigLoader,
    RankInfo,
    RankResolver,
    resolve_current_node_index,
)
from tests.e2e.nightly.multi_node.external_dp.scripts.utils import (
    ServerCommandBuilder,
    _is_http_ready,
    build_dist_envs,
    build_proxy_server_cmd,
    collect_logs,
    proxy_server_health_url,
    start_logged_process,
    terminate_process_tree,
    wait_http_ready,
    wait_http_unready,
    write_benchmark_results_json,
)
from tools.aisbench import run_aisbench_cases

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_LOG_ROOT = Path("/tmp/external_dp_logs")
READY_POLL_INTERVAL_SECONDS = 5
READY_STATUS_LOG_INTERVAL_SECONDS = 30
HEARTBEAT_LOG_INTERVAL_SECONDS = 30
POST_BENCHMARK_HEALTHCHECK_TIMEOUT_SECONDS = 30
RankProcess = tuple[subprocess.Popen, RankInfo, Path]


def _install_special_dependencies(config: ExternalDPConfig) -> None:
    for package, version in config.special_dependencies.items():
        command = [
            sys.executable,
            "-m",
            "pip",
            "install",
            f"{package}=={version}",
        ]
        subprocess.call(command)


class ExternalDPServerManager:
    """Start and stop the external DP ranks owned by the current node."""

    def __init__(
        self,
        *,
        config: ExternalDPConfig,
        ranks: list[RankInfo],
        current_node_index: int,
        log_root: Path,
    ):
        self.config = config
        self.ranks = ranks
        self.current_node_index = current_node_index
        self.log_root = log_root
        self.command_builder = ServerCommandBuilder(config)
        self.dist_envs = build_dist_envs(
            config.cluster_ips[current_node_index],
            config.cluster_ips[0],
        )
        self.rank_processes: list[RankProcess] = []

    def start_current_node(self) -> None:
        local_ranks = [rank for rank in self.ranks if rank.node_index == self.current_node_index]
        logger.info("Starting %d external DP ranks on node %d", len(local_ranks), self.current_node_index)
        try:
            for rank in local_ranks:
                template = self.config.launch_templates[rank.node_index]
                template = type(template)(
                    envs={**template.envs, **self.dist_envs},
                    server_cmd_template=template.server_cmd_template,
                )
                server_cmd = self.command_builder.build(rank, template)
                log_file = self._rank_log_file(rank)
                process = start_logged_process(server_cmd.cmd, server_cmd.env, log_file)
                self.rank_processes.append((process, rank, log_file))

            _wait_ranks_ready(
                local_ranks,
                timeout=SERVER_READY_TIMEOUT_SECONDS,
                rank_processes=self.rank_processes,
            )
        except Exception:
            self.cleanup()
            raise

    def __enter__(self):
        self.start_current_node()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.cleanup()

    def cleanup(self) -> None:
        for process, rank, _log_file in reversed(self.rank_processes):
            logger.info(
                "Stopping external DP rank node=%d rank=%d pid=%d",
                rank.node_index,
                rank.local_rank,
                process.pid,
            )
            terminate_process_tree(process.pid)
        self.rank_processes.clear()

    def _rank_log_file(self, rank: RankInfo) -> Path:
        return self.log_root / f"node-{rank.node_index}" / f"rank-{rank.local_rank}.log"


class ExternalDPProxyLauncher:
    """Launch the external DP proxy on the configured proxy node."""

    def __init__(
        self,
        *,
        config: ExternalDPConfig,
        ranks: list[RankInfo],
        current_node_index: int,
        log_root: Path,
    ):
        self.config = config
        self.ranks = ranks
        self.current_node_index = current_node_index
        self.log_root = log_root
        self.pid: int | None = None

    def start(self) -> None:
        if self.current_node_index != self.config.routing.proxy_node_index:
            logger.info("Current node is not proxy node, skip launching external DP proxy")
            return

        cmd = build_proxy_server_cmd(self.config, self.ranks)
        log_file = self.log_root / f"node-{self.current_node_index}" / "proxy.log"
        process = start_logged_process(cmd, {}, log_file)
        self.pid = process.pid
        logger.info("External DP proxy launched: %s", proxy_server_health_url(self.config))

    def wait_ready(self, timeout: int = 300) -> None:
        wait_http_ready(proxy_server_health_url(self.config), timeout=timeout)
        logger.info("External DP proxy ready: %s", proxy_server_health_url(self.config))

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.cleanup()

    def cleanup(self) -> None:
        if self.pid is None:
            return
        logger.info("Stopping external DP proxy pid=%d", self.pid)
        terminate_process_tree(self.pid)
        self.pid = None


def _build_all_server_commands(config: ExternalDPConfig, ranks: list[RankInfo]):
    builder = ServerCommandBuilder(config)
    commands = [builder.build(rank, config.launch_templates[rank.node_index]) for rank in ranks]
    return commands


@contextmanager
def _heartbeat(
    task_name: str,
    *,
    interval: int = HEARTBEAT_LOG_INTERVAL_SECONDS,
    status_fn: Callable[[], str] | None = None,
):
    start_time = time.monotonic()
    stop_event = threading.Event()

    def report_progress() -> None:
        while not stop_event.wait(interval):
            elapsed = int(time.monotonic() - start_time)
            status = ""
            if status_fn is not None:
                try:
                    status = f" {status_fn()}"
                except Exception as exc:  # pragma: no cover - diagnostic only
                    status = f" status_error={exc!r}"
            logger.info("%s still running: elapsed=%ds%s", task_name, elapsed, status)

    logger.info("%s started", task_name)
    thread = threading.Thread(target=report_progress, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop_event.set()
        thread.join(timeout=1)
        elapsed = int(time.monotonic() - start_time)
        logger.info("%s finished: elapsed=%ds", task_name, elapsed)


def _rank_health_url(rank: RankInfo) -> str:
    return f"http://{rank.host}:{rank.port}{SERVER_HEALTH_PATH}"


def _master_rank_health_url(ranks: list[RankInfo]) -> str:
    for rank in ranks:
        if rank.node_index == 0 and rank.local_rank == 0:
            return _rank_health_url(rank)
    raise RuntimeError("External DP master rank was not found")


def _rank_label(rank: RankInfo) -> str:
    return f"node={rank.node_index} rank={rank.local_rank} role={rank.role} url={_rank_health_url(rank)}"


def _format_http_status(label: str, url: str) -> str:
    status = "ready" if _is_http_ready(url, timeout=1.0) else "waiting"
    return f"{label}={status} url={url}"


def _format_benchmark_cases(config: ExternalDPConfig) -> str:
    names = [str(case.get("case_name", "<unnamed>")) for case in config.benchmark_cases]
    return ", ".join(names) if names else "<none>"


def _format_rank_statuses(
    ranks: list[RankInfo],
    rank_ready: dict[RankInfo, bool],
) -> str:
    parts = []
    for rank in ranks:
        status = "ready" if rank_ready[rank] else "waiting"
        parts.append(f"  {_rank_label(rank)} status={status}")
    return "\n".join(parts)


def _raise_if_rank_process_exited(rank_processes: list[RankProcess] | None) -> None:
    if not rank_processes:
        return

    exited = []
    for process, rank, log_file in rank_processes:
        returncode = process.poll()
        if returncode is not None:
            exited.append(f"{_rank_label(rank)} pid={process.pid} returncode={returncode} log={log_file}")

    if exited:
        raise RuntimeError("External DP rank process exited before ready: " + "; ".join(exited))


def _wait_ranks_ready(
    ranks,
    timeout: int,
    rank_processes: list[RankProcess] | None = None,
) -> None:
    ranks = list(ranks)
    rank_ready = {rank: False for rank in ranks}
    deadline = time.monotonic() + timeout
    last_log_time = 0.0

    while True:
        _raise_if_rank_process_exited(rank_processes)

        all_ready = True
        unhealthy_after_ready = []

        for rank in ranks:
            is_ready = _is_http_ready(_rank_health_url(rank), timeout=1.0)
            if is_ready:
                if not rank_ready[rank]:
                    logger.info("[READY] External DP rank %s", _rank_label(rank))
                rank_ready[rank] = True
                continue

            all_ready = False
            if rank_ready[rank]:
                unhealthy_after_ready.append(rank)

        if unhealthy_after_ready:
            failed = "; ".join(_rank_label(rank) for rank in unhealthy_after_ready)
            raise RuntimeError(f"External DP rank became unhealthy after ready: {failed}")

        if all_ready:
            return

        now = time.monotonic()
        if now - last_log_time >= READY_STATUS_LOG_INTERVAL_SECONDS:
            logger.info(
                "Polling external DP ranks: ready=%d/%d\n%s",
                sum(rank_ready.values()),
                len(ranks),
                _format_rank_statuses(ranks, rank_ready),
            )
            last_log_time = now

        if now >= deadline:
            pending = [rank for rank in ranks if not rank_ready[rank]]
            pending_labels = "; ".join(_rank_label(rank) for rank in pending)
            raise TimeoutError(f"Timed out waiting for external DP ranks ready: {pending_labels}")

        time.sleep(READY_POLL_INTERVAL_SECONDS)


def _wait_master_rank_stopped(ranks: list[RankInfo], timeout: int) -> None:
    url = _master_rank_health_url(ranks)
    wait_http_ready(url, timeout=SERVER_READY_TIMEOUT_SECONDS)
    logger.info("Hanging until master external DP rank stops: %s", url)
    wait_http_unready(url, timeout=timeout)


def _archive_rank_logs(log_root: Path, current_node_index: int) -> None:
    log_prefix = os.environ.get("LOG_PREFIX")
    if not log_prefix:
        return
    node_log_dir = log_root / f"node-{current_node_index}"
    output_tar = Path(log_prefix) / f"node_{current_node_index}_external_dp_logs.tar.gz"
    collect_logs(node_log_dir, output_tar)


def test_external_dp() -> None:
    config = ExternalDPConfigLoader.from_yaml()
    _install_special_dependencies(config)
    ranks = RankResolver(config).resolve()
    current_node_index = resolve_current_node_index(config)
    log_root = Path(os.environ.get("EXTERNAL_DP_LOG_DIR", str(DEFAULT_LOG_ROOT)))
    max_wait_seconds = int(os.environ.get("EXTERNAL_DP_MAX_WAIT_SECONDS", "3600"))
    is_master = current_node_index == 0

    server_manager = ExternalDPServerManager(
        config=config,
        ranks=ranks,
        current_node_index=current_node_index,
        log_root=log_root,
    )
    proxy_launcher = ExternalDPProxyLauncher(
        config=config,
        ranks=ranks,
        current_node_index=current_node_index,
        log_root=log_root,
    )

    try:
        with server_manager, proxy_launcher:
            if is_master:
                _wait_ranks_ready(ranks, timeout=max_wait_seconds)
                proxy_launcher.wait_ready()
                target = f"http://{config.routing.proxy_host}:{config.routing.proxy_port}"
                logger.info(
                    "Running AISBench cases: model=%s target=%s cases=[%s]",
                    config.model,
                    target,
                    _format_benchmark_cases(config),
                )
                with _heartbeat(
                    "Running AISBench",
                    status_fn=lambda: _format_http_status("proxy", proxy_server_health_url(config)),
                ):
                    results = run_aisbench_cases(
                        model=config.model,
                        port=config.routing.proxy_port,
                        aisbench_cases=config.benchmark_cases,
                        host_ip=config.routing.proxy_host,
                    )
                logger.info("AISBench completed: results=%d", len(results or []))
                all_commands = _build_all_server_commands(config, ranks)
                write_benchmark_results_json(
                    config=config,
                    ranks=ranks,
                    commands=all_commands,
                    results=results,
                )
                _wait_ranks_ready(ranks, timeout=POST_BENCHMARK_HEALTHCHECK_TIMEOUT_SECONDS)
            else:
                master_url = _master_rank_health_url(ranks)
                with _heartbeat(
                    "Waiting for master external DP rank to stop",
                    status_fn=lambda: _format_http_status("master", master_url),
                ):
                    _wait_master_rank_stopped(ranks, timeout=max_wait_seconds)
    finally:
        _archive_rank_logs(log_root, current_node_index)
