import json
import logging
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import requests

from tests.e2e.nightly.multi_node.external_dp.scripts.external_dp_config import (
    ExternalDPConfig,
    ExternalDPConfigLoader,
    RankResolver,
    RankInfo,
    resolve_current_node_index,
)
from tests.e2e.nightly.multi_node.external_dp.scripts.runtime import (
    ExternalDPProxyLauncher,
    ExternalDPServerManager,
    build_all_server_commands,
    create_kv_pool_manager,
    format_http_status,
    master_rank_health_url,
    proxy_server_health_url,
    wait_master_rank_stopped,
    wait_ranks_ready,
)
from tests.e2e.nightly.multi_node.external_dp.scripts.utils import (
    collect_logs,
    write_benchmark_results_json,
)
from tests.e2e.nightly.multi_node.scripts.utils import ProxyServer
from tools.aisbench import run_aisbench_cases

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_LOG_ROOT = Path("/tmp/external_dp_logs")


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


@contextmanager
def _heartbeat(
    task_name: str,
    *,
    interval: int = 30,
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


def _format_benchmark_cases(config: ExternalDPConfig) -> str:
    names = [str(case.get("case_name", "<unnamed>")) for case in config.benchmark_cases]
    return ", ".join(names) if names else "<none>"


def _archive_rank_logs(log_root: Path, current_node_index: int) -> None:
    log_prefix = os.environ.get("LOG_PREFIX")
    if not log_prefix:
        return
    node_log_dir = log_root / f"node-{current_node_index}"
    output_tar = Path(log_prefix) / f"node_{current_node_index}_external_dp_logs.tar.gz"
    collect_logs(node_log_dir, output_tar)


def _extract_cmd_value(cmd: list[str], flag: str) -> str | None:
    try:
        idx = cmd.index(flag)
        return cmd[idx + 1]
    except (ValueError, IndexError):
        return None


def _parse_json_flag_ext(cmd: list[str], flag: str) -> dict[str, Any]:
    val = _extract_cmd_value(cmd, flag)
    if not val:
        return {}
    try:
        return json.loads(val)
    except (json.JSONDecodeError, ValueError):
        return {}


def _get_first_server_cmd(
    config: ExternalDPConfig, ranks: list[RankInfo], all_commands: list
) -> list[str]:
    if all_commands:
        return all_commands[0].cmd
    return []


def _build_external_dp_servers(
    config: ExternalDPConfig, ranks: list[RankInfo]
) -> tuple[ProxyServer, ProxyServer, ProxyServer]:
    completion_server = ProxyServer(config.routing.proxy_host, config.routing.proxy_port)

    if config.is_disaggregated_prefill:
        prefill_ranks = [r for r in ranks if r.role == "prefiller"]
        decode_ranks = [r for r in ranks if r.role == "decoder"]
        tokenize_server = ProxyServer(prefill_ranks[0].host, prefill_ranks[0].port)
        metrics_server = ProxyServer(decode_ranks[0].host, decode_ranks[0].port)
    else:
        tokenize_server = completion_server
        metrics_server = completion_server

    return completion_server, tokenize_server, metrics_server


def _run_chat_completion_ext(
    config: ExternalDPConfig,
    completion_server: ProxyServer,
    tokenize_server: ProxyServer,
    first_server_cmd: list[str],
) -> None:
    from tools.send_request import resolve_prompt, send_v1_chat_completions

    prompts = config.chat_prompts or ["Hello!"]
    expected = config.expected_response or {}

    max_model_len_str = _extract_cmd_value(first_server_cmd, "--max-model-len")
    max_model_len = int(max_model_len_str) if max_model_len_str else None

    if isinstance(config.api_keyword_args, list):
        api_args_list = config.api_keyword_args
        if len(api_args_list) != len(prompts):
            raise ValueError(f"""
api_keyword_args list length ({len(api_args_list)}) must match prompts length ({len(prompts)})""")
    else:
        api_args_list = [config.api_keyword_args] * len(prompts)

    if isinstance(expected.get("per_prompt"), list):
        expected_list = expected["per_prompt"]
    else:
        expected_list = [expected] * len(prompts)

    for prompt_raw, api_args, exp in zip(prompts, api_args_list, expected_list):
        prompt, actual_prompt_tokens = resolve_prompt(tokenize_server, prompt_raw, use_chat=True)
        if actual_prompt_tokens is not None:
            exp = dict(exp) if exp else {}
            exp.setdefault("prompt_tokens", actual_prompt_tokens)
        send_v1_chat_completions(
            prompt,
            model=config.model,
            server=completion_server,
            request_args=api_args,
            expected=exp,
            max_model_len=max_model_len,
        )


def _run_spec_decode_acceptance_ext(
    config: ExternalDPConfig,
    metrics_server: ProxyServer,
    first_server_cmd: list[str],
    baseline: tuple[int, list[int]] | None = None,
) -> None:
    from tools.spec_decode_metrics import measure_acceptance_rate, validate_acceptance_rate

    spec_config = _parse_json_flag_ext(first_server_cmd, "--speculative-config")
    num_speculative_tokens = int(spec_config.get("num_speculative_tokens", 1))

    acceptance_cfg = config.acceptance_rate or {}
    baseline_val = acceptance_cfg.get("baseline")
    tolerance = acceptance_cfg.get("tolerance", 0.05)

    if baseline_val is None:
        logger.warning("acceptance_rate.baseline not set in config, skipping validation")
        return

    if baseline is None:
        baseline = (0, [0] * num_speculative_tokens)

    _, all_rates = measure_acceptance_rate(metrics_server, num_speculative_tokens, baseline)
    validate_acceptance_rate(all_rates[0], float(baseline_val), float(tolerance))


def test_external_dp() -> None:
    config = ExternalDPConfigLoader.from_yaml()
    _install_special_dependencies(config)
    ranks = RankResolver(config).resolve()
    current_node_index = resolve_current_node_index(config)
    log_root = Path(os.environ.get("EXTERNAL_DP_LOG_DIR", str(DEFAULT_LOG_ROOT)))
    max_wait_seconds = int(os.environ.get("EXTERNAL_DP_MAX_WAIT_SECONDS", "3600"))
    is_master = current_node_index == 0

    kv_pool_manager = create_kv_pool_manager(
        config=config,
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
        with (
            kv_pool_manager,
            ExternalDPServerManager(
                config=config,
                ranks=ranks,
                current_node_index=current_node_index,
                log_root=log_root,
                extra_envs=kv_pool_manager.server_envs,
            ),
            proxy_launcher,
        ):
            if is_master:
                wait_ranks_ready(ranks, timeout=max_wait_seconds)
                proxy_launcher.wait_ready()

                all_commands = build_all_server_commands(config, ranks)
                first_server_cmd = _get_first_server_cmd(config, ranks, all_commands)
                completion_server, tokenize_server, metrics_server = (
                    _build_external_dp_servers(config, ranks)
                )

                if "chat_completion" in config.test_content:
                    logger.info("Running chat_completion tests")
                    _run_chat_completion_ext(
                        config, completion_server, tokenize_server, first_server_cmd
                    )

                spec_baseline = None
                if "spec_decode_acceptance" in config.test_content:
                    from tools.spec_decode_metrics import capture_baseline

                    spec_cfg = _parse_json_flag_ext(first_server_cmd, "--speculative-config")
                    num_spec_tokens = int(spec_cfg.get("num_speculative_tokens", 1))

                    def warmup_fn():
                        requests.post(
                            completion_server.url_for("v1", "chat", "completions"),
                            json={
                                "model": config.model,
                                "messages": [{"role": "user", "content": "Hello!"}],
                                "max_tokens": 16,
                            },
                            timeout=120,
                        )

                    logger.info("Capturing spec_decode baseline")
                    spec_baseline = capture_baseline(metrics_server, num_spec_tokens, warmup_fn)

                target = f"http://{config.routing.proxy_host}:{config.routing.proxy_port}"
                logger.info(
                    "Running AISBench cases: model=%s target=%s cases=[%s]",
                    config.model,
                    target,
                    _format_benchmark_cases(config),
                )
                with _heartbeat(
                    "Running AISBench",
                    status_fn=lambda: format_http_status("proxy", proxy_server_health_url(config)),
                ):
                    results = run_aisbench_cases(
                        model=config.model,
                        port=config.routing.proxy_port,
                        aisbench_cases=config.benchmark_cases,
                        host_ip=config.routing.proxy_host,
                    )
                logger.info("AISBench completed: results=%d", len(results or []))
                write_benchmark_results_json(
                    config=config,
                    ranks=ranks,
                    commands=all_commands,
                    results=results,
                )

                if "spec_decode_acceptance" in config.test_content:
                    logger.info("Validating spec_decode acceptance rate")
                    _run_spec_decode_acceptance_ext(config, metrics_server, first_server_cmd, spec_baseline)

                wait_ranks_ready(ranks, timeout=30)
            else:
                master_url = master_rank_health_url(ranks)
                with _heartbeat(
                    "Waiting for master external DP rank to stop",
                    status_fn=lambda: format_http_status("master", master_url),
                ):
                    wait_master_rank_stopped(ranks, timeout=max_wait_seconds)
    finally:
        _archive_rank_logs(log_root, current_node_index)

