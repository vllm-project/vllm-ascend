# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""KV pool A/B watch driven by a PD deployment YAML (leg A: pool, leg B: local prefix cache)."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import socket
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import requests
import yaml
from prometheus_client.parser import text_string_to_metric_families
from transformers import AutoTokenizer

from tests.e2e.common.multi_node.dp_mode import resolve_config_path
from tests.e2e.common.multi_node.external_dp.external_dp_config import (
    ExternalDPConfig,
    ExternalDPConfigLoader,
    NodeTemplate,
    RankInfo,
    RankResolver,
    resolve_current_node_index,
)
from tests.e2e.common.multi_node.external_dp.runtime import (
    ExternalDPProxyLauncher,
    ExternalDPServerManager,
    ServerCommandBuilder,
    build_proxy_server_cmd,
    create_kv_pool_manager,
    wait_ranks_ready,
)
from tests.e2e.common.multi_node.external_dp.utils import collect_logs, start_logged_process
from tests.e2e.pull_request.kv_pool_watch import (
    DELAYED_BLOCKS_METRIC,
    HITS_METRIC,
    LOAD_KEYS_METRIC,
    PREFIX_HITS_METRIC,
    QUERIES_METRIC,
    build_pool_accuracy_prompt,
    extract_pool_answer,
    resolve_model_path,
    run_pool_ab,
    run_pool_hit_sequence,
    snapshot_metrics,
)
from tools.aisbench import run_aisbench_cases

BLOCK_SIZE = 128
POOL_SECRET = "7391"
SECRET_ACC_MAX_TOKENS = 32

# Shares KV_POOL_PD_READY_TIMEOUT_S with runtime.py so both waits stay inside the
# 360-minute nightly job budget.
READY_TIMEOUT_SECONDS = int(os.environ.get("KV_POOL_PD_READY_TIMEOUT_S", "3600"))
PROXY_READY_TIMEOUT_SECONDS = int(os.environ.get("KV_POOL_PD_PROXY_READY_TIMEOUT_S", "600"))
REQUEST_TIMEOUT_SECONDS = int(os.environ.get("KV_POOL_PD_REQUEST_TIMEOUT_S", "1800"))
# Per-run engine/proxy logs. Each leg is archived to LOG_PREFIX so the CI job
# collects it together with the other multi-node logs.
LOG_ROOT = Path(os.environ.get("KV_POOL_PD_LOG_DIR", "/tmp/kv_pool_pd_logs"))
POOL_METRICS = (
    HITS_METRIC,
    QUERIES_METRIC,
    LOAD_KEYS_METRIC,
    DELAYED_BLOCKS_METRIC,
    PREFIX_HITS_METRIC,
)


def _archive_leg_logs(case_name: str, node_index: int) -> None:
    """Pack one leg's raw logs into LOG_PREFIX; CI collects this directory only."""
    log_prefix = os.environ.get("LOG_PREFIX")
    if not log_prefix:
        return
    node_log_dir = LOG_ROOT / case_name / f"node-{node_index}"
    output_tar = Path(log_prefix) / f"node_{node_index}_kv_pool_pd_{case_name}.tar.gz"
    collect_logs(node_log_dir, output_tar)


@dataclass(frozen=True)
class PoolABCase:
    """Watch parameters from the deployment YAML's ``pool_ab`` block."""

    case_name: str
    prompt_tokens: int
    lcm_block_size: int
    control_port: int
    perf_gate: bool = True
    check_output: bool = True
    model_env: str | None = None
    block_size: int = BLOCK_SIZE
    secret: str = POOL_SECRET
    accuracy_max_tokens: int = SECRET_ACC_MAX_TOKENS
    perf_concurrency: int = 4
    perf_batch_requests: int = 8
    perf_sample_count: int = 3
    perf_warmup_count: int = 1
    perf_aisbench: dict[str, Any] | None = None
    token_identity: bool = False
    acceptance_ratio: float | None = None

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> PoolABCase:
        if not isinstance(mapping, dict):
            raise TypeError(f"pool_ab must be a mapping, got {type(mapping).__name__}")
        required = ("case_name", "prompt_tokens", "lcm_block_size", "control_port")
        missing = [key for key in required if key not in mapping]
        if missing:
            raise KeyError(f"Missing required pool_ab fields: {missing}")
        unknown = sorted(set(mapping) - set(cls.__dataclass_fields__))
        if unknown:
            raise KeyError(f"Unknown pool_ab fields: {unknown}")

        values = dict(mapping)
        for key in (
            "prompt_tokens",
            "lcm_block_size",
            "control_port",
            "block_size",
            "accuracy_max_tokens",
            "perf_concurrency",
            "perf_batch_requests",
            "perf_sample_count",
            "perf_warmup_count",
        ):
            if key in values and values[key] is not None:
                values[key] = int(values[key])
        for key in ("perf_gate", "check_output", "token_identity"):
            if key in values:
                values[key] = bool(values[key])
        if values.get("acceptance_ratio") is not None:
            values["acceptance_ratio"] = float(values["acceptance_ratio"])
        return cls(**values)


def load_case() -> tuple[ExternalDPConfig, PoolABCase]:
    """Load the deployment and its ``pool_ab`` block from ``CONFIG_YAML_PATH``."""
    path = resolve_config_path()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"multi-node config must be a mapping: {path}")
    block = raw.get("pool_ab")
    if block is None:
        raise KeyError(f"{path} does not declare a pool_ab block")
    case = PoolABCase.from_mapping(block)

    config = ExternalDPConfigLoader.from_yaml(str(path))
    override = os.environ.get(case.model_env) if case.model_env else None
    if override and override != config.model:
        config = replace(config, model=override)
    return config, case


def _role_for(config: ExternalDPConfig, node_index: int) -> str:
    if node_index in config.routing.groups.get("prefiller", []):
        return "prefill"
    if node_index in config.routing.groups.get("decoder", []):
        return "decode"
    raise RuntimeError(f"node {node_index} has no prefiller/decoder role in {config.routing.groups}")


def _ranks_by_role(ranks: list[RankInfo], role: str) -> list[RankInfo]:
    selected = [rank for rank in ranks if rank.role == role]
    if not selected:
        raise RuntimeError(f"no {role} ranks were resolved")
    return selected


def _http_prefill_ranks(ranks: list[RankInfo]) -> list[RankInfo]:
    """Prefill ranks that expose an API server (PP headless stages do not)."""
    selected = [rank for rank in _ranks_by_role(ranks, "prefiller") if not rank.is_headless]
    if not selected:
        raise RuntimeError("no HTTP prefiller ranks were resolved")
    return selected


def _template_value(arguments: list[str], flag: str) -> str:
    index = arguments.index(flag)
    return arguments[index + 1]


def _max_model_len(config: ExternalDPConfig) -> int:
    return int(_template_value(config.launch_templates[0].server_cmd_template, "--max-model-len"))


def _served_model_name(config: ExternalDPConfig) -> str:
    """Read the served name from the deployment template (single source of truth)."""
    return _template_value(config.launch_templates[0].server_cmd_template, "--served-model-name")


def _drop_store_connector(arguments: list[str]) -> list[str]:
    """B leg: keep only the PD transfer connector (no external pool)."""
    if "--kv-transfer-config" not in arguments:
        return arguments
    index = arguments.index("--kv-transfer-config")
    transfer_config = json.loads(arguments[index + 1])
    connectors = transfer_config.get("kv_connector_extra_config", {}).get("connectors")
    if not connectors:
        return arguments
    transfer_config["kv_connector_extra_config"]["connectors"] = [
        connector for connector in connectors if connector.get("kv_connector") != "AscendStoreConnector"
    ]
    updated = list(arguments)
    updated[index + 1] = json.dumps(transfer_config)
    return updated


def _uses_layerwise_push(config: ExternalDPConfig) -> bool:
    """True when the P->D link is the layerwise push connector."""
    for template in config.launch_templates:
        arguments = template.server_cmd_template
        if "--kv-transfer-config" not in arguments:
            continue
        transfer = json.loads(arguments[arguments.index("--kv-transfer-config") + 1])
        connectors = transfer.get("kv_connector_extra_config", {}).get("connectors", [])
        for connector in connectors:
            extra = connector.get("kv_connector_extra_config") or {}
            if connector.get("kv_connector") == "MooncakeConnectorV2" and extra.get("use_layerwise", False):
                return True
    return False


def _leg_config(base: ExternalDPConfig, *, use_pool: bool) -> ExternalDPConfig:
    """A leg keeps the pool templates; B leg drops the pool and uses local APC."""
    prefiller_nodes = set(base.routing.groups.get("prefiller", []))
    templates: list[NodeTemplate] = []
    for node_index, template in enumerate(base.launch_templates):
        arguments = list(template.server_cmd_template)
        if not use_pool:
            arguments = _drop_store_connector(arguments)
            if node_index in prefiller_nodes:
                arguments = [
                    "--enable-prefix-caching" if argument == "--no-enable-prefix-caching" else argument
                    for argument in arguments
                ]
        templates.append(NodeTemplate(envs=dict(template.envs), server_cmd_template=arguments))
    return replace(base, launch_templates=templates, kv_pool=base.kv_pool if use_pool else None)


# --- leg control channel ----------------------------------------------------


class _ControlChannel:
    """Line-delimited JSON channel between the driver and one follower node."""

    def __init__(self, sock: socket.socket):
        self._socket = sock
        self._stream = sock.makefile("rwb")

    def send(self, payload: dict[str, Any]) -> None:
        self._stream.write((json.dumps(payload) + "\n").encode())
        self._stream.flush()

    def recv(self) -> dict[str, Any]:
        line = self._stream.readline()
        if not line:
            raise ConnectionError("PD control channel closed by the peer")
        message = json.loads(line)
        if message.get("event") == "error":
            raise RuntimeError(f"decode node failed: {message.get('error')}")
        return message

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._stream.close()
        with contextlib.suppress(Exception):
            self._socket.close()


class _ControlServer:
    """Accept one control connection per non-driver node."""

    def __init__(self, port: int, expected_clients: int = 1, timeout: int = READY_TIMEOUT_SECONDS):
        self.port = port
        self.expected_clients = expected_clients
        self.timeout = timeout
        self._listener: socket.socket | None = None
        self.channels: dict[int, _ControlChannel] = {}

    def __enter__(self) -> _ControlServer:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("0.0.0.0", self.port))
        listener.listen(max(1, self.expected_clients))
        self._listener = listener
        deadline = time.monotonic() + float(self.timeout)
        print(
            f"[kv_pool_pd] wait {self.expected_clients} node(s) on control port {self.port}",
            flush=True,
        )
        while len(self.channels) < self.expected_clients:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"only {len(self.channels)}/{self.expected_clients} nodes connected "
                    f"on port {self.port} in {self.timeout}s"
                )
            listener.settimeout(remaining)
            try:
                connection, peer = listener.accept()
            except socket.timeout as exc:
                raise TimeoutError(
                    f"only {len(self.channels)}/{self.expected_clients} nodes connected "
                    f"on port {self.port} in {self.timeout}s"
                ) from exc
            connection.settimeout(None)
            channel = _ControlChannel(connection)
            # Bound only the handshake so a stray connection cannot stall the
            # driver before the followers announce themselves.
            connection.settimeout(60.0)
            try:
                hello = channel.recv()
            except socket.timeout as exc:
                raise TimeoutError(f"no hello from {peer} on port {self.port}") from exc
            finally:
                connection.settimeout(None)
            if hello.get("cmd") != "hello":
                raise RuntimeError(f"expected hello from {peer}, got {hello!r}")
            node_index = int(hello["node_index"])
            if node_index in self.channels:
                raise RuntimeError(f"node {node_index} connected twice on port {self.port}")
            self.channels[node_index] = channel
            print(f"[kv_pool_pd] node {node_index} connected from {peer}", flush=True)
        return self

    def broadcast(self, payload: dict[str, Any]) -> None:
        for node_index in sorted(self.channels):
            self.channels[node_index].send(payload)

    def wait_all(self, event: str) -> dict[int, dict[str, Any]]:
        messages = {node_index: self.channels[node_index].recv() for node_index in sorted(self.channels)}
        for node_index, message in messages.items():
            if message.get("event") != event:
                raise RuntimeError(f"node {node_index} replied {message!r}, expected {event!r}")
        return messages

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        for node_index in sorted(self.channels):
            with contextlib.suppress(Exception):
                self.channels[node_index].send({"cmd": "exit"})
            self.channels[node_index].close()
        self.channels.clear()
        if self._listener is not None:
            self._listener.close()
            self._listener = None


def _connect_control(control_host: str, port: int, timeout: int = READY_TIMEOUT_SECONDS) -> _ControlChannel:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            connection = socket.create_connection((control_host, port), timeout=10)
        except OSError as exc:
            last_error = exc
            time.sleep(2)
            continue
        # The connect timeout must not bound the leg waits: a follower blocks
        # for the whole leg until the driver sends stop.
        connection.settimeout(None)
        return _ControlChannel(connection)
    raise TimeoutError(f"cannot connect to the driver control port {control_host}:{port}: {last_error}")


# --- HTTP watch client ------------------------------------------------------


def _session_id(prompt_ids: list[int]) -> str:
    """Stable conversation key so equal prompts pin to one prefill rank."""
    digest = hashlib.sha256()
    digest.update(str(len(prompt_ids)).encode())
    for token in prompt_ids[:128]:
        digest.update(b",")
        digest.update(str(token).encode())
    return digest.hexdigest()


@dataclass
class _PDWatchClient:
    """Drive A/B traffic through the PD load-balance proxy; pool counters are summed over prefill ranks."""

    proxy_host: str
    proxy_port: int
    served_model_name: str
    metrics_prefill: list[RankInfo]

    def _completion(self, prompt_ids: list[int], *, max_tokens: int, min_tokens: int = 1) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.served_model_name,
            "prompt": prompt_ids,
            "max_tokens": max_tokens,
            "min_tokens": min_tokens,
            "temperature": 0,
            "return_token_ids": True,
        }
        response = requests.post(
            f"http://{self.proxy_host}:{self.proxy_port}/v1/completions",
            json=payload,
            headers={"X-Session-Id": _session_id(prompt_ids)},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return response.json()

    def _generate_one(self, prompt_ids: list[int], max_tokens: int) -> tuple[list[int], str]:
        completion = self._completion(prompt_ids, max_tokens=max_tokens)
        choice = completion["choices"][0]
        output_ids = list(choice.get("token_ids") or [])
        return list(prompt_ids) + output_ids, choice.get("text") or ""

    def generate_greedy(self, prompts: list[list[int]], max_tokens: int) -> list[tuple[list[int], str]]:
        with ThreadPoolExecutor(max_workers=max(1, len(prompts))) as executor:
            futures = [executor.submit(self._generate_one, prompt_ids, max_tokens) for prompt_ids in prompts]
            return [future.result() for future in futures]

    def get_metrics(self) -> dict[str, float]:
        totals = {name: 0.0 for name in POOL_METRICS}
        for rank in self.metrics_prefill:
            response = requests.get(f"http://{rank.host}:{rank.port}/metrics", timeout=30)
            response.raise_for_status()
            for family in text_string_to_metric_families(response.text):
                if family.name not in totals:
                    continue
                # Keep the counter value only; the sibling ``_created`` sample
                # is a timestamp and must not be summed into the hit counts.
                names = {family.name, f"{family.name}_total"}
                totals[family.name] += sum(sample.value for sample in family.samples if sample.name in names)
        return totals


class _AffinityProxyLauncher(ExternalDPProxyLauncher):
    """Start the load-balance proxy that matches the leg's PD transfer mode.

    Layerwise push needs the layerwise proxy (dispatch decode first); the pull
    connector keeps the plain proxy with ``--enable-prefill-affinity``.
    """

    def start(self) -> None:
        if self.current_node_index != self.config.routing.proxy_node_index:
            return
        cmd = build_proxy_server_cmd(self.config, self.ranks)
        if _uses_layerwise_push(self.config):
            if "layerwise" not in Path(self.config.routing.proxy_script).name:
                raise RuntimeError(
                    "the P->D link is layerwise push but routing.proxy_script is not the layerwise proxy: "
                    f"{self.config.routing.proxy_script}"
                )
        else:
            cmd.append("--enable-prefill-affinity")
        log_file = self.log_root / f"node-{self.current_node_index}" / "proxy.log"
        self.pid = start_logged_process(cmd, {}, log_file).pid


# --- PD legs ----------------------------------------------------------------


class _KVPoolCommandBuilder(ServerCommandBuilder):
    """Assign one lookup RPC port per engine (37000+P rank, 37100+D rank)."""

    def _build_variables(self, rank: RankInfo) -> dict[str, str]:
        variables = super()._build_variables(rank)
        base = 37000 if rank.role == "prefiller" else 37100
        variables["LOOKUP_RPC_PORT"] = str(base + rank.dp_rank)
        return variables


class _KVPoolServerManager(ExternalDPServerManager):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.command_builder = _KVPoolCommandBuilder(self.config)


@contextlib.contextmanager
def _driver_leg(
    base: ExternalDPConfig,
    control: _ControlServer,
    *,
    ranks: list[RankInfo],
    node_index: int,
    use_pool: bool,
    case_name: str,
):
    """Start every node and the proxy for one leg, yield the watch client, then tear down."""
    config = _leg_config(base, use_pool=use_pool)
    log_root = LOG_ROOT / case_name
    pool_manager = create_kv_pool_manager(config=config, current_node_index=node_index, log_root=log_root)
    try:
        with pool_manager:
            control.broadcast({"cmd": "start", "leg": case_name, "use_pool": use_pool})
            with _KVPoolServerManager(
                config=config,
                ranks=ranks,
                current_node_index=node_index,
                log_root=log_root,
                extra_envs=pool_manager.server_envs,
            ):
                wait_ranks_ready(ranks, timeout=READY_TIMEOUT_SECONDS)
                control.wait_all("ready")
                # The proxy is the only data plane: it owns the PD handshake,
                # so the watch never talks to a P or D rank directly.
                with _AffinityProxyLauncher(
                    config=config,
                    ranks=ranks,
                    current_node_index=node_index,
                    log_root=log_root,
                ) as proxy:
                    proxy.wait_ready(timeout=PROXY_READY_TIMEOUT_SECONDS)
                    print(
                        f"[kv_pool_pd] leg={case_name} pool={int(use_pool)} all nodes ready, "
                        f"proxy={config.routing.proxy_host}:{config.routing.proxy_port}",
                        flush=True,
                    )
                    yield _PDWatchClient(
                        proxy_host=config.routing.proxy_host,
                        proxy_port=config.routing.proxy_port,
                        served_model_name=_served_model_name(config),
                        metrics_prefill=_http_prefill_ranks(ranks),
                    )
    finally:
        body_failed = sys.exc_info()[0] is not None
        teardown_error: Exception | None = None
        try:
            control.broadcast({"cmd": "stop"})
            control.wait_all("stopped")
        except Exception as exc:  # noqa: BLE001 - keep the body error as the diagnosis
            teardown_error = exc
        with contextlib.suppress(Exception):
            _archive_leg_logs(case_name, node_index)
        if teardown_error is not None and not body_failed:
            raise RuntimeError(f"a follower node did not stop cleanly: {teardown_error}") from teardown_error


def _run_follower_legs(
    base: ExternalDPConfig,
    ranks: list[RankInfo],
    node_index: int,
    driver_ip: str,
    case: PoolABCase,
) -> None:
    """Serve local ranks for each leg the driver announces, until it exits."""
    channel = _connect_control(driver_ip, case.control_port)
    try:
        channel.send({"cmd": "hello", "node_index": node_index})
        while True:
            message = channel.recv()
            command = message.get("cmd")
            if command == "start":
                use_pool = bool(message.get("use_pool"))
                leg = str(message["leg"])
                config = _leg_config(base, use_pool=use_pool)
                log_root = LOG_ROOT / leg
                with create_kv_pool_manager(
                    config=config, current_node_index=node_index, log_root=log_root
                ) as pool_manager:
                    with _KVPoolServerManager(
                        config=config,
                        ranks=ranks,
                        current_node_index=node_index,
                        log_root=log_root,
                        extra_envs=pool_manager.server_envs,
                    ):
                        wait_ranks_ready(ranks, timeout=READY_TIMEOUT_SECONDS)
                        channel.send({"event": "ready"})
                        stop = channel.recv()
                        if stop.get("cmd") != "stop":
                            raise RuntimeError(f"expected stop, got {stop!r}")
                channel.send({"event": "stopped"})
                with contextlib.suppress(Exception):
                    _archive_leg_logs(leg, node_index)
            elif command == "exit":
                with contextlib.suppress(Exception):
                    channel.send({"event": "closed"})
                return
            else:
                raise RuntimeError(f"unknown control command: {command!r}")
    finally:
        channel.close()


def _aisbench_perf_case(
    config: ExternalDPConfig,
    spec: dict[str, Any],
    *,
    name: str,
    num_prompts: int,
    batch_size: int,
) -> dict[str, Any]:
    """Build one AISBench performance case that reads the 128K prefix90 dataset."""
    case = {
        "benchmark_name": name,
        "case_type": "performance",
        "dataset_path": spec["dataset_path"],
        "dataset_conf": spec["dataset_conf"],
        "request_conf": spec["request_conf"],
        "model_path": spec.get("model_path") or config.model,
        "num_prompts": int(num_prompts),
        "batch_size": int(batch_size),
        "max_out_len": int(spec.get("max_out_len", 1)),
        "request_rate": int(spec.get("request_rate", 0)),
    }
    if spec.get("seed") is not None:
        case["seed"] = int(spec["seed"])
    # CI has no shared dataset mount: only pin dataset_path_local when the case
    # YAML provides one, otherwise AISBench downloads the ModelScope dataset.
    local_path = spec.get("dataset_path_local") or os.environ.get("KV_POOL_PERF_DATASET_LOCAL")
    if local_path:
        case["dataset_path_local"] = local_path
    return case


def _aisbench_input_throughput(result: Any) -> float:
    """Extract the input token throughput (token/s) from an AISBench perf result."""
    payload = result[1] if isinstance(result, (list, tuple)) and len(result) > 1 else result
    if not isinstance(payload, dict):
        raise RuntimeError(f"unexpected AISBench perf result: {type(result).__name__}")
    value: Any = payload.get("Input Token Throughput")
    if isinstance(value, dict):
        value = value.get("total") or next(iter(value.values()), None)
    if value is None:
        raise RuntimeError(f"Input Token Throughput missing in AISBench result: {list(payload)}")
    return float(str(value).replace("token/s", "").strip())


def run_pool_ab_case() -> None:
    """Run the pool A/B watch described by ``CONFIG_YAML_PATH``."""
    config, case = load_case()
    ranks = RankResolver(config).resolve()
    node_index = resolve_current_node_index(config)
    driver_index = config.routing.proxy_node_index
    role = _role_for(config, node_index)
    print(
        f"[kv_pool_pd] case={case.case_name} node={node_index} role={role} "
        f"driver={driver_index} cluster={config.cluster_ips} model={config.model}",
        flush=True,
    )
    if node_index != driver_index:
        _run_follower_legs(config, ranks, node_index, config.cluster_ips[driver_index], case)
        return

    max_model_len = _max_model_len(config)
    tokenizer = AutoTokenizer.from_pretrained(
        resolve_model_path(config.model), trust_remote_code=True
    )
    prefix = build_pool_accuracy_prompt(
        tokenizer, case.prompt_tokens, max_model_len, secret=case.secret
    )
    prefill_ranks = _http_prefill_ranks(ranks)
    decode_ranks = _ranks_by_role(ranks, "decoder")
    print(
        f"[kv_pool_pd] prompt_tokens={len(prefix)} "
        f"max_model_len={max_model_len} lcm={case.lcm_block_size} "
        f"prefill_ranks={len(prefill_ranks)} decode_ranks={len(decode_ranks)}",
        flush=True,
    )

    def accuracy(runner: Any) -> list[float]:
        outputs = runner.generate_greedy([prefix], max_tokens=case.accuracy_max_tokens)
        answer = extract_pool_answer(outputs[0][1])
        assert answer == case.secret, answer
        print(f"[kv_pool_acc] secret answer={answer}", flush=True)
        return [1.0]

    def hit(runner: Any) -> dict[str, Any]:
        summary = run_pool_hit_sequence(
            runner,
            block_size=case.block_size,
            prompt_tokens=len(prefix),
            prompt_ids=prefix,
            # Prefill affinity sends every copy of one prompt to the same P
            # rank, so the expected pool hits do not scale with the rank count.
            replicas=1,
            lcm_block_size=case.lcm_block_size,
            answer_secret=case.secret,
            # DSpark cold/hit generation is not bit-identical (same class as P3);
            # plain MTP/MLA cases keep the H3 assertion.
            check_output=case.check_output,
            tag=case.case_name,
        )
        summary["hit_outputs"] = summary["hit_outputs"][:1]
        return summary

    def perf(runner: Any, use_pool: bool) -> float:
        spec = case.perf_aisbench
        if not spec:
            raise RuntimeError("pool_ab.perf_aisbench is required for the AISBench perf phase")
        warmup = spec.get("warmup") or {}
        if warmup:
            run_aisbench_cases(
                model=runner.served_model_name,
                port=runner.proxy_port,
                host_ip=runner.proxy_host,
                aisbench_cases=[
                    _aisbench_perf_case(
                        config,
                        spec,
                        name="kv_pool_prefix90_warmup",
                        num_prompts=int(warmup.get("num_prompts", 4)),
                        batch_size=int(warmup.get("batch_size", 4)),
                    )
                ],
            )
        before = snapshot_metrics(runner) if use_pool else None
        results = run_aisbench_cases(
            model=runner.served_model_name,
            port=runner.proxy_port,
            host_ip=runner.proxy_host,
            aisbench_cases=[
                _aisbench_perf_case(
                    config,
                    spec,
                    name="kv_pool_prefix90",
                    num_prompts=int(spec["num_prompts"]),
                    batch_size=int(spec["batch_size"]),
                )
            ],
        )
        if use_pool and before is not None:
            after = snapshot_metrics(runner)
            queries = after.queries - before.queries
            hits = after.hits - before.hits
            ratio = hits / queries if queries > 0 else 0.0
            target = float(spec.get("hit_rate_target") or 0.0)
            tolerance = float(spec.get("hit_rate_tolerance", 0.03))
            print(
                f"[kv_pool_hitrate] leg=a hit_tokens={hits:.0f} query_tokens={queries:.0f} "
                f"ratio={ratio:.4f} target={target:.4f}",
                flush=True,
            )
            if target and abs(ratio - target) > tolerance:
                raise AssertionError(
                    f"perf hit ratio {ratio:.4f} outside {target:.4f}±{tolerance:.4f}"
                )
        return _aisbench_input_throughput(results[-1])

    with _ControlServer(case.control_port, expected_clients=config.num_nodes - 1) as control:

        def build_leg(use_pool: bool, case_name: str):
            return _driver_leg(
                config,
                control,
                ranks=ranks,
                node_index=node_index,
                use_pool=use_pool,
                case_name=case_name,
            )

        run_pool_ab(
            case_name=case.case_name,
            pool_first=True,
            build_leg=build_leg,
            accuracy=accuracy,
            hit=hit,
            prompt_ids=prefix,
            # DSpark: A/B output identity is diagnostic only; answers and pool
            # hits are the gate.
            token_identity=case.token_identity,
            acceptance_ratio=case.acceptance_ratio,
            perf_concurrency=case.perf_concurrency,
            perf_batch_requests=case.perf_batch_requests,
            perf_sample_count=case.perf_sample_count,
            perf_warmup_count=case.perf_warmup_count,
            # Cases without pool_ab.perf_aisbench keep the legacy local sampling.
            perf_fn=perf if case.perf_aisbench else None,
            perf_gate=case.perf_gate,
        )
