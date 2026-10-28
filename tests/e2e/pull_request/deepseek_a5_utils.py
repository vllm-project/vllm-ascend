# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared utilities for full-weight DeepSeek A5 PR tests."""

import json
import math
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import requests
import torch
from prometheus_client.parser import text_string_to_metric_families
from vllm.utils.network_utils import get_open_port

from tests.e2e.conftest import RemoteOpenAIServer
from vllm_ascend.device.device_config import check_ascend_device_type, is_950

VOCAB_SIZE = 129280
OUTPUT_TOKENS = 8
SERVER_MODEL_NAME = "deepseek-a5"
REQUEST_TIMEOUT = 600
SERVER_STARTUP_TIMEOUT = 2800
METRICS_WAIT_SECONDS = 30
MAX_CONCURRENT_REQUESTS = 4
SWA_REPLAY_LOG = "SWA bounded replay is active"


def configure_a5(monkeypatch: pytest.MonkeyPatch) -> None:
    if not torch.npu.is_available():
        pytest.skip("DeepSeek A5 E2E requires Ascend 950")
    check_ascend_device_type()
    if not is_950():
        pytest.skip("DeepSeek A5 E2E requires Ascend 950")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    monkeypatch.setenv("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True")
    monkeypatch.setenv("HCCL_OP_EXPANSION_MODE", "CCU_SCHED")
    monkeypatch.setenv("HCCL_BUFFSIZE", "1024")
    # Selected-tests containers default to ERROR, which hides replay evidence.
    monkeypatch.setenv("VLLM_LOGGING_LEVEL", "INFO")
    # Full-weight cases exercise prefix caching and chunked prefill, which
    # are not supported with batch invariance.
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")


def engine_args() -> dict:
    return {
        "dtype": "bfloat16",
        "tensor_parallel_size": 4,
        "enable_expert_parallel": True,
        "distributed_executor_backend": "mp",
        "max_model_len": 2048,
        "max_num_seqs": 4,
        "max_num_batched_tokens": 512,
        "kv_cache_memory_bytes": 512 * 1024**2,
        "enable_chunked_prefill": True,
        "enable_prefix_caching": True,
        "disable_log_stats": False,
        "seed": 0,
        "additional_config": {"enable_cpu_binding": False},
    }


def prompt(length: int, salt: int = 0, *, tokenizer) -> dict:
    """Preserve exact boundary lengths using meaningful, checkpoint-tokenized text."""
    header = tokenizer.encode(f"Document {salt}.\n", add_special_tokens=False)
    body = tokenizer.encode("The library contains books about history, science and art. ", add_special_tokens=False)
    suffix = tokenizer.encode("\nContinue counting in English: one, two, three, four, five,", add_special_tokens=False)
    if length < len(header) + len(suffix):
        tokens = suffix[-length:]
    else:
        remaining = length - len(header) - len(suffix)
        tokens = header + (body * (remaining // len(body) + 1))[:remaining] + suffix
    assert len(tokens) == length
    return {"prompt_token_ids": tokens}


def serve_args(args: dict) -> list[str]:
    result = ["--served-model-name", SERVER_MODEL_NAME, "--enable-prompt-tokens-details"]
    for name, value in args.items():
        option = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            if value:
                result.append(option)
            elif name in {"enable_prefix_caching", "enable_chunked_prefill", "swa_bounded_replay", "async_scheduling"}:
                result.append("--no-" + name.replace("_", "-"))
        else:
            result.extend([option, json.dumps(value) if isinstance(value, dict) else str(value)])
    return result


class LoggedServer(RemoteOpenAIServer):
    """Keep worker logs as evidence and diagnostics for each server restart."""

    def __init__(self, model: str, args: dict, log_path: Path) -> None:
        self.log_path = log_path
        port = get_open_port()
        cli_args = [*serve_args(args), "--host", "127.0.0.1", "--port", str(port)]
        with log_path.open("w", encoding="utf-8") as log_file:
            self.log_file = log_file
            try:
                super().__init__(
                    model,
                    cli_args,
                    server_host="127.0.0.1",
                    server_port=port,
                    auto_port=False,
                    max_wait_seconds=SERVER_STARTUP_TIMEOUT,
                )
            except Exception:
                if hasattr(self, "proc"):
                    self._terminate_server()
                print(log_path.read_text(encoding="utf-8", errors="replace"), flush=True)
                raise

    def _start_server(self, model: str, server_cmd: list[str], env_dict: dict[str, str] | None) -> None:
        env = os.environ.copy()
        env.update(env_dict or {})
        self.proc = subprocess.Popen(server_cmd, env=env, stdout=self.log_file, stderr=subprocess.STDOUT)

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        finally:
            if exc_type is not None:
                # pytest tmp_path is outside the CI artifact directory. Include
                # runtime failures in captured output, as for startup failures.
                print(self.log_path.read_text(encoding="utf-8", errors="replace"), flush=True)

    def assert_log(self, message: str) -> None:
        logs = self.log_path.read_text(encoding="utf-8", errors="replace")
        assert message in logs, f"Missing execution evidence {message!r}:\n{logs}"

    def collective_rpc(self, method: str) -> list:
        response = requests.post(self.url_for("collective_rpc"), json={"method": method}, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json().get("results", []) if response.content else []


def completion(server: RemoteOpenAIServer, prompt: str | list[int], *, max_tokens: int = OUTPUT_TOKENS) -> dict:
    response = requests.post(
        server.url_for("v1", "completions"),
        json={
            "model": SERVER_MODEL_NAME,
            "prompt": prompt,
            "temperature": 0,
            "seed": 0,
            "max_tokens": max_tokens,
            "ignore_eos": True,
            "return_token_ids": True,
            "logprobs": 1,
        },
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    result = response.json()
    assert len(result["choices"]) == 1, result
    choice = result["choices"][0]
    assert choice["finish_reason"] == "length", result
    assert len(choice["token_ids"]) == max_tokens, result
    assert all(0 <= token < VOCAB_SIZE for token in choice["token_ids"]), result
    logprobs = choice["logprobs"]["token_logprobs"]
    assert len(logprobs) == max_tokens and all(value is not None and math.isfinite(value) for value in logprobs)
    assert result["usage"]["completion_tokens"] == max_tokens
    return result


def completions(server: RemoteOpenAIServer, prompts: list, *, max_tokens: int = OUTPUT_TOKENS) -> list[dict]:
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as pool:
        return list(pool.map(lambda value: completion(server, value, max_tokens=max_tokens), prompts))


def assert_invalid_request(server: RemoteOpenAIServer) -> None:
    response = requests.post(
        server.url_for("v1", "completions"),
        json={"model": SERVER_MODEL_NAME, "prompt": "Hello", "max_tokens": -1},
        timeout=REQUEST_TIMEOUT,
    )
    assert response.status_code == 400, response.text
    assert response.json()["error"]["message"], response.text


def cached_tokens(result: dict) -> int:
    return result["usage"]["prompt_tokens_details"]["cached_tokens"]


def assert_dp_engines(server: RemoteOpenAIServer, dp_size: int, *, speculative: bool = False) -> None:
    metric_name = "vllm:spec_decode_num_drafts_total" if speculative else "vllm:generation_tokens_total"
    deadline = time.monotonic() + METRICS_WAIT_SECONDS
    while True:
        response = requests.get(server.url_for("metrics"), timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        counts = {
            sample.labels["engine"]: sample.value
            for family in text_string_to_metric_families(response.text)
            for sample in family.samples
            if sample.name == metric_name
        }
        if len(counts) == dp_size and all(value > 0 for value in counts.values()):
            return
        assert time.monotonic() < deadline, counts
        time.sleep(0.5)


def assert_server_speculation(server: RemoteOpenAIServer, num_speculative_tokens: int) -> list[float]:
    response = requests.get(server.url_for("metrics"), timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    drafts = 0.0
    accepted = [0.0] * num_speculative_tokens
    positions = set()
    for family in text_string_to_metric_families(response.text):
        for sample in family.samples:
            if sample.name == "vllm:spec_decode_num_drafts_total":
                drafts += sample.value
            elif sample.name == "vllm:spec_decode_num_accepted_tokens_per_pos_total":
                position = int(sample.labels["position"])
                assert 0 <= position < num_speculative_tokens, sample
                accepted[position] += sample.value
                positions.add(position)
    assert drafts > 0, "Requests bypassed speculative decoding"
    assert positions == set(range(num_speculative_tokens)), positions
    rates = [value / drafts for value in accepted]
    print(f"Accepted tokens per draft position: {rates}, drafts: {drafts}", flush=True)
    return rates


class A5WorkerProbe:
    """Observe graph and SWA replay execution in spawned workers after warmup."""

    def install_swa_replay_probe(self) -> None:
        # Lazy import: this method runs in an initialized NPU worker only.
        from vllm_ascend.models.deepseek_v41.engram.model_state import EngramModelState

        self.a5_swa_replay_observed = False
        original = EngramModelState.prepare_attn

        def prepare_attn(state, input_batch, *args, **kwargs):
            metadata = original(state, input_batch, *args, **kwargs)
            # Read the CPU boundary buffer used to build real SWA metadata;
            # this adds no device synchronization and excludes startup warmup.
            if (
                not self.a5_swa_replay_observed
                and not input_batch.is_dummy
                and (state._replay_start.np[: input_batch.num_reqs] > 0).any()
            ):
                self.a5_swa_replay_observed = True
                print(f"A5 E2E worker probe: {SWA_REPLAY_LOG}", flush=True)
            return metadata

        EngramModelState.prepare_attn = prepare_attn

    def install_graph_probe(self) -> None:
        # Lazy import: this method runs in an initialized NPU worker only.
        from vllm_ascend.worker.v2.aclgraph_utils import ModelAclGraphManager

        self.a5_graph_replays = 0
        original = ModelAclGraphManager.run_fullgraph

        def run_fullgraph(manager, *args, **kwargs):
            result = original(manager, *args, **kwargs)
            self.a5_graph_replays += 1
            return result

        ModelAclGraphManager.run_fullgraph = run_fullgraph

    def read_graph_probe(self) -> int:
        return self.a5_graph_replays
