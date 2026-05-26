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
    BACKEND_HEALTHCHECK_PATH,
    BACKEND_READY_TIMEOUT,
    EndpointResolver,
    ExternalDPConfig,
    ExternalDPConfigLoader,
    ExternalDPEndpoint,
    resolve_current_node_index,
)
from tests.e2e.nightly.multi_node.external_dp.scripts.utils import (
    CommandBuilder,
    _is_http_ready,
    build_distributed_envs,
    build_proxy_command,
    collect_logs,
    proxy_health_url,
    start_process,
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
READY_POLL_INTERVAL = 5
READY_STATUS_LOG_INTERVAL = 30
LONG_TASK_LOG_INTERVAL = 30
POST_BENCHMARK_HEALTHCHECK_TIMEOUT = 30
EndpointProcess = tuple[subprocess.Popen, ExternalDPEndpoint, Path]


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
    """Start and stop the backend endpoints owned by the current node."""

    def __init__(
        self,
        *,
        config: ExternalDPConfig,
        endpoints: list[ExternalDPEndpoint],
        current_node_index: int,
        log_root: Path,
    ):
        self.config = config
        self.endpoints = endpoints
        self.current_node_index = current_node_index
        self.log_root = log_root
        self.command_builder = CommandBuilder(config)
        self.distributed_envs = build_distributed_envs(
            config.cluster_ips[current_node_index],
            config.cluster_ips[0],
        )
        self.processes: list[EndpointProcess] = []

    def start_current_node(self) -> None:
        local_endpoints = [endpoint for endpoint in self.endpoints if endpoint.config_index == self.current_node_index]
        logger.info("Starting %d external DP endpoints on node %d", len(local_endpoints), self.current_node_index)
        try:
            for endpoint in local_endpoints:
                template = self.config.templates[endpoint.config_index]
                template = type(template)(
                    envs={**template.envs, **self.distributed_envs},
                    server_cmd_template=template.server_cmd_template,
                )
                built_command = self.command_builder.build(endpoint, template)
                log_file = self._endpoint_log_file(endpoint)
                process = start_process(built_command.cmd, built_command.env, log_file)
                self.processes.append((process, endpoint, log_file))

            _wait_all_endpoints_ready(
                local_endpoints,
                timeout=BACKEND_READY_TIMEOUT,
                process_checks=self.processes,
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
        for process, endpoint, _log_file in reversed(self.processes):
            logger.info(
                "Stopping external DP endpoint node=%d rank=%d pid=%d",
                endpoint.config_index,
                endpoint.local_rank,
                process.pid,
            )
            terminate_process_tree(process.pid)
        self.processes.clear()

    def _endpoint_log_file(self, endpoint: ExternalDPEndpoint) -> Path:
        return self.log_root / f"node-{endpoint.config_index}" / f"rank-{endpoint.local_rank}.log"


class ExternalDPProxyLauncher:
    """Launch the external DP proxy on the configured proxy node."""

    def __init__(
        self,
        *,
        config: ExternalDPConfig,
        endpoints: list[ExternalDPEndpoint],
        current_node_index: int,
        log_root: Path,
    ):
        self.config = config
        self.endpoints = endpoints
        self.current_node_index = current_node_index
        self.log_root = log_root
        self.pid: int | None = None

    def start(self) -> None:
        if self.current_node_index != self.config.routing.proxy_node_index:
            logger.info("Current node is not proxy node, skip launching external DP proxy")
            return

        cmd = build_proxy_command(self.config, self.endpoints)
        log_file = self.log_root / f"node-{self.current_node_index}" / "proxy.log"
        process = start_process(cmd, {}, log_file)
        self.pid = process.pid
        logger.info("External DP proxy launched: %s", proxy_health_url(self.config))

    def wait_ready(self, timeout: int = 300) -> None:
        wait_http_ready(proxy_health_url(self.config), timeout=timeout)
        logger.info("External DP proxy ready: %s", proxy_health_url(self.config))

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


def _build_all_commands(config: ExternalDPConfig, endpoints: list[ExternalDPEndpoint]):
    builder = CommandBuilder(config)
    commands = [builder.build(endpoint, config.templates[endpoint.config_index]) for endpoint in endpoints]
    return commands


@contextmanager
def _heartbeat(
    task_name: str,
    *,
    interval: int = LONG_TASK_LOG_INTERVAL,
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


def _endpoint_health_url(endpoint: ExternalDPEndpoint) -> str:
    return f"http://{endpoint.host}:{endpoint.port}{BACKEND_HEALTHCHECK_PATH}"


def _master_endpoint_health_url(endpoints: list[ExternalDPEndpoint]) -> str:
    for endpoint in endpoints:
        if endpoint.config_index == 0 and endpoint.local_rank == 0:
            return _endpoint_health_url(endpoint)
    raise RuntimeError("External DP master endpoint was not found")


def _endpoint_label(endpoint: ExternalDPEndpoint) -> str:
    return (
        f"node={endpoint.config_index} rank={endpoint.local_rank} "
        f"role={endpoint.role} url={_endpoint_health_url(endpoint)}"
    )


def _http_status(label: str, url: str) -> str:
    status = "ready" if _is_http_ready(url, timeout=1.0) else "waiting"
    return f"{label}={status} url={url}"


def _benchmark_case_names(config: ExternalDPConfig) -> str:
    names = [str(case.get("case_name", "<unnamed>")) for case in config.benchmark_cases]
    return ", ".join(names) if names else "<none>"


def _format_endpoint_statuses(
    endpoints: list[ExternalDPEndpoint],
    ready_once: dict[ExternalDPEndpoint, bool],
) -> str:
    parts = []
    for endpoint in endpoints:
        status = "ready" if ready_once[endpoint] else "waiting"
        parts.append(f"  {_endpoint_label(endpoint)} status={status}")
    return "\n".join(parts)


def _raise_if_endpoint_process_exited(process_checks: list[EndpointProcess] | None) -> None:
    if not process_checks:
        return

    exited = []
    for process, endpoint, log_file in process_checks:
        returncode = process.poll()
        if returncode is not None:
            exited.append(f"{_endpoint_label(endpoint)} pid={process.pid} returncode={returncode} log={log_file}")

    if exited:
        raise RuntimeError("External DP endpoint process exited before ready: " + "; ".join(exited))


def _wait_all_endpoints_ready(
    endpoints,
    timeout: int,
    process_checks: list[EndpointProcess] | None = None,
) -> None:
    endpoints = list(endpoints)
    ready_once = {endpoint: False for endpoint in endpoints}
    deadline = time.monotonic() + timeout
    last_log_time = 0.0

    while True:
        _raise_if_endpoint_process_exited(process_checks)

        all_ready = True
        unhealthy_after_ready = []

        for endpoint in endpoints:
            is_ready = _is_http_ready(_endpoint_health_url(endpoint), timeout=1.0)
            if is_ready:
                if not ready_once[endpoint]:
                    logger.info("[READY] External DP endpoint %s", _endpoint_label(endpoint))
                ready_once[endpoint] = True
                continue

            all_ready = False
            if ready_once[endpoint]:
                unhealthy_after_ready.append(endpoint)

        if unhealthy_after_ready:
            failed = "; ".join(_endpoint_label(endpoint) for endpoint in unhealthy_after_ready)
            raise RuntimeError(f"External DP endpoint became unhealthy after ready: {failed}")

        if all_ready:
            return

        now = time.monotonic()
        if now - last_log_time >= READY_STATUS_LOG_INTERVAL:
            logger.info(
                "Polling external DP endpoints: ready=%d/%d\n%s",
                sum(ready_once.values()),
                len(endpoints),
                _format_endpoint_statuses(endpoints, ready_once),
            )
            last_log_time = now

        if now >= deadline:
            pending = [endpoint for endpoint in endpoints if not ready_once[endpoint]]
            pending_labels = "; ".join(_endpoint_label(endpoint) for endpoint in pending)
            raise TimeoutError(f"Timed out waiting for external DP endpoints ready: {pending_labels}")

        time.sleep(READY_POLL_INTERVAL)


def _wait_master_endpoint_terminated(endpoints: list[ExternalDPEndpoint], timeout: int) -> None:
    url = _master_endpoint_health_url(endpoints)
    wait_http_ready(url, timeout=BACKEND_READY_TIMEOUT)
    logger.info("Hanging until master external DP endpoint terminates: %s", url)
    wait_http_unready(url, timeout=timeout)


def _collect_external_dp_logs(log_root: Path, current_node_index: int) -> None:
    log_prefix = os.environ.get("LOG_PREFIX")
    if not log_prefix:
        return
    node_log_dir = log_root / f"node-{current_node_index}"
    output_tar = Path(log_prefix) / f"node_{current_node_index}_external_dp_logs.tar.gz"
    collect_logs(node_log_dir, output_tar)


def test_external_dp() -> None:
    config = ExternalDPConfigLoader.from_yaml()
    _install_special_dependencies(config)
    endpoints = EndpointResolver(config).resolve()
    current_node_index = resolve_current_node_index(config)
    log_root = Path(os.environ.get("EXTERNAL_DP_LOG_DIR", str(DEFAULT_LOG_ROOT)))
    max_wait_seconds = int(os.environ.get("EXTERNAL_DP_MAX_WAIT_SECONDS", "3600"))
    is_master = current_node_index == 0

    server_manager = ExternalDPServerManager(
        config=config,
        endpoints=endpoints,
        current_node_index=current_node_index,
        log_root=log_root,
    )
    proxy_launcher = ExternalDPProxyLauncher(
        config=config,
        endpoints=endpoints,
        current_node_index=current_node_index,
        log_root=log_root,
    )

    try:
        with server_manager, proxy_launcher:
            if is_master:
                _wait_all_endpoints_ready(endpoints, timeout=max_wait_seconds)
                proxy_launcher.wait_ready()
                target = f"http://{config.routing.proxy_host}:{config.routing.proxy_port}"
                logger.info(
                    "Running AISBench cases: model=%s target=%s cases=[%s]",
                    config.model,
                    target,
                    _benchmark_case_names(config),
                )
                with _heartbeat(
                    "Running AISBench",
                    status_fn=lambda: _http_status("proxy", proxy_health_url(config)),
                ):
                    results = run_aisbench_cases(
                        model=config.model,
                        port=config.routing.proxy_port,
                        aisbench_cases=config.benchmark_cases,
                        host_ip=config.routing.proxy_host,
                    )
                logger.info("AISBench completed: results=%d", len(results or []))
                all_commands = _build_all_commands(config, endpoints)
                write_benchmark_results_json(
                    config=config,
                    endpoints=endpoints,
                    commands=all_commands,
                    results=results,
                )
                _wait_all_endpoints_ready(endpoints, timeout=POST_BENCHMARK_HEALTHCHECK_TIMEOUT)
            else:
                master_url = _master_endpoint_health_url(endpoints)
                with _heartbeat(
                    "Waiting for master external DP endpoint to terminate",
                    status_fn=lambda: _http_status("master", master_url),
                ):
                    _wait_master_endpoint_terminated(endpoints, timeout=max_wait_seconds)
    finally:
        _collect_external_dp_logs(log_root, current_node_index)
