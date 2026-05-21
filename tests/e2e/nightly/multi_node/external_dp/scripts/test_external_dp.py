import logging
import os
import time
from copy import deepcopy
from pathlib import Path

from tests.e2e.nightly.multi_node.external_dp.scripts.external_dp_config import (
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
    write_benchmark_results_json,
)
from tools.aisbench import maybe_download_from_modelscope, run_aisbench_cases

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_LOG_ROOT = Path("/tmp/external_dp_logs")
DEFAULT_SIGNAL_ROOT = Path("/root/.cache/external_dp_signals")


class ExternalDPServerManager:
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
        local_endpoints = [endpoint for endpoint in self.endpoints if endpoint.node_index == self.current_node_index]
        logger.info("Starting %d external DP endpoints on node %d", len(local_endpoints), self.current_node_index)
        try:
            for endpoint in local_endpoints:
                template = self.config.templates[endpoint.config_index]
                built_command = self.command_builder.build(endpoint, template)
                log_file = self._endpoint_log_file(endpoint)
                process = start_process(built_command.cmd, built_command.env, log_file)
                self.processes.append((process.pid, endpoint))

            for _, endpoint in self.processes:
                if endpoint.node_index != self.current_node_index:
                    continue
                url = f"http://{endpoint.host}:{endpoint.port}{endpoint.healthcheck_path}"
                wait_http_ready(url, endpoint.ready_timeout)
                logger.info("External DP endpoint ready: %s", url)
        except Exception:
            self.cleanup()
            raise

    def cleanup(self) -> None:
        for pid, endpoint in reversed(self.processes):
            logger.info(
                "Stopping external DP endpoint node=%d rank=%d pid=%d",
                endpoint.node_index,
                endpoint.local_rank,
                pid,
            )
            terminate_process_tree(pid)
        self.processes.clear()

    def _endpoint_log_file(self, endpoint: ExternalDPEndpoint) -> Path:
        return self.log_root / f"node-{endpoint.node_index}" / f"rank-{endpoint.local_rank}.log"


class ExternalDPProxyLauncher:
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
        wait_http_ready(proxy_health_url(self.config), timeout=300)
        logger.info("External DP proxy ready: %s", proxy_health_url(self.config))

    def cleanup(self) -> None:
        if self.pid is None:
            return
        logger.info("Stopping external DP proxy pid=%d", self.pid)
        terminate_process_tree(self.pid)
        self.pid = None


def _job_name(config: ExternalDPConfig) -> str:
    return os.environ.get("BENCHMARK_JOB_NAME", "") or config.test_name.replace(" ", "-")


def _done_signal_path(config: ExternalDPConfig) -> Path:
    return DEFAULT_SIGNAL_ROOT / f"{_job_name(config)}.done"


def _write_done_signal(config: ExternalDPConfig) -> None:
    signal_path = _done_signal_path(config)
    signal_path.parent.mkdir(parents=True, exist_ok=True)
    signal_path.write_text("done\n", encoding="utf-8")


def _wait_done_signal(config: ExternalDPConfig, timeout: int) -> None:
    signal_path = _done_signal_path(config)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if signal_path.exists():
            return
        time.sleep(5)
    raise TimeoutError(f"Timed out waiting for external DP done signal: {signal_path}")


def _prepare_aisbench_cases(config: ExternalDPConfig) -> list[dict]:
    cases = [deepcopy(case) for case in config.benchmark_cases()]
    if config.request_model == config.model:
        return cases

    model_path = maybe_download_from_modelscope(config.model)
    for case in cases:
        case.setdefault("model_path", model_path)
    return cases


def _build_all_commands(config: ExternalDPConfig):
    endpoints = EndpointResolver(config).resolve()
    builder = CommandBuilder(config)
    commands = [builder.build(endpoint, config.templates[endpoint.config_index]) for endpoint in endpoints]
    return endpoints, commands


def _wait_all_endpoints_ready(endpoints, timeout: int) -> None:
    for endpoint in endpoints:
        url = f"http://{endpoint.host}:{endpoint.port}{endpoint.healthcheck_path}"
        wait_http_ready(url, timeout=timeout)
        logger.info("External DP endpoint ready from master: %s", url)


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
    max_wait_seconds = int(os.environ.get("EXTERNAL_DP_MAX_WAIT_SECONDS", "7200"))
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
        server_manager.start_current_node()
        if is_master:
            _wait_all_endpoints_ready(endpoints, timeout=max_wait_seconds)
        proxy_launcher.start()

        if is_master:
            wait_http_ready(proxy_health_url(config), timeout=300)
            aisbench_cases = _prepare_aisbench_cases(config)
            results = run_aisbench_cases(
                model=config.request_model,
                port=config.routing.proxy_port,
                aisbench_cases=aisbench_cases,
                host_ip=config.routing.proxy_host,
            )
            all_endpoints, all_commands = _build_all_commands(config)
            write_benchmark_results_json(
                config=config,
                endpoints=all_endpoints,
                commands=all_commands,
                results=results,
            )
        else:
            _wait_done_signal(config, timeout=max_wait_seconds)
    finally:
        if is_master:
            _write_done_signal(config)
        proxy_launcher.cleanup()
        server_manager.cleanup()
        _collect_external_dp_logs(log_root, current_node_index)
