import logging
import os
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
from tests.e2e.nightly.multi_node.external_dp.scripts.external_dp_utils import (
    CommandBuilder,
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
POST_BENCHMARK_HEALTHCHECK_TIMEOUT = 30


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
        self.processes: list[tuple[int, ExternalDPEndpoint]] = []

    def start_current_node(self) -> None:
        local_endpoints = [endpoint for endpoint in self.endpoints if endpoint.config_index == self.current_node_index]
        logger.info("Starting %d external DP endpoints on node %d", len(local_endpoints), self.current_node_index)
        try:
            for endpoint in local_endpoints:
                template = self.config.templates[endpoint.config_index]
                built_command = self.command_builder.build(endpoint, template)
                log_file = self._endpoint_log_file(endpoint)
                process = start_process(built_command.cmd, built_command.env, log_file)
                self.processes.append((process.pid, endpoint))

            for _, endpoint in self.processes:
                url = _endpoint_health_url(endpoint)
                wait_http_ready(url, BACKEND_READY_TIMEOUT)
                logger.info("External DP endpoint ready: %s", url)
        except Exception:
            self.cleanup()
            raise

    def __enter__(self):
        self.start_current_node()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.cleanup()

    def cleanup(self) -> None:
        for pid, endpoint in reversed(self.processes):
            logger.info(
                "Stopping external DP endpoint node=%d rank=%d pid=%d",
                endpoint.config_index,
                endpoint.local_rank,
                pid,
            )
            terminate_process_tree(pid)
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


def _endpoint_health_url(endpoint: ExternalDPEndpoint) -> str:
    return f"http://{endpoint.host}:{endpoint.port}{BACKEND_HEALTHCHECK_PATH}"


def _master_endpoint_health_url(endpoints: list[ExternalDPEndpoint]) -> str:
    for endpoint in endpoints:
        if endpoint.config_index == 0 and endpoint.local_rank == 0:
            return _endpoint_health_url(endpoint)
    raise RuntimeError("External DP master endpoint was not found")


def _wait_all_endpoints_ready(endpoints, timeout: int) -> None:
    for endpoint in endpoints:
        url = _endpoint_health_url(endpoint)
        wait_http_ready(url, timeout=timeout)
        logger.info("External DP endpoint ready from master: %s", url)


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
                results = run_aisbench_cases(
                    model=config.model,
                    port=config.routing.proxy_port,
                    aisbench_cases=config.benchmark_cases(),
                    host_ip=config.routing.proxy_host,
                )
                all_commands = _build_all_commands(config, endpoints)
                write_benchmark_results_json(
                    config=config,
                    endpoints=endpoints,
                    commands=all_commands,
                    results=results,
                )
                _wait_all_endpoints_ready(endpoints, timeout=POST_BENCHMARK_HEALTHCHECK_TIMEOUT)
            else:
                _wait_master_endpoint_terminated(endpoints, timeout=max_wait_seconds)
    finally:
        _collect_external_dp_logs(log_root, current_node_index)
