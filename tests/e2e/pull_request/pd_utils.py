# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared real Mooncake PD output, transfer and recovery contracts on 2/4 devices.

Use the same real weights for colocated and PD output comparisons; require
correlated transfer evidence and reject unavailable-transfer fallback.
"""

import json
import os
import socket
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import psutil
import requests
from vllm.transformers_utils.utils import maybe_model_redirect
from vllm.utils.network_utils import get_open_port

from tests.e2e.conftest import DisaggPDProxy, RemotePDServer
from tests.e2e.pull_request.pd_http import assert_prefix_reset_response
from tests.e2e.pull_request.pd_output import assert_distribution_match

SERVED_MODEL_NAME = "pd-e2e"
OUTPUT_TOKENS = 10
REQUEST_TIMEOUT = 120
CLEANUP_WAIT_SECONDS = 10
PROMPTS = (
    "The capital of France is",
    "The sequence of even numbers starts with",
    "Explain why leaves appear green in one sentence:",
    "Write a short greeting to a new colleague:",
)


class _BoundedServer(RemotePDServer):
    """Bound readiness and verify cleanup for both model servers and proxy."""

    def _terminate_server(self) -> None:
        # Record owned children before terminating their parents: otherwise a
        # surviving worker can be reparented and disappear from the tree.
        owned = {}
        for proc in self._proc_list:
            try:
                parent = psutil.Process(proc.pid)
                for process in [parent, *parent.children(recursive=True)]:
                    owned[process.pid] = process
            except psutil.NoSuchProcess:
                continue
        super()._terminate_server()
        _, alive = psutil.wait_procs(list(owned.values()), timeout=CLEANUP_WAIT_SECONDS)
        assert not alive, f"PD test cleanup left owned processes alive: {[process.pid for process in alive]}"

    def _wait_for_multiple_servers(
        self, targets, timeout: float, log_interval: float = 30.0, always_check_nodes: bool = False
    ):
        # Distinguish P/D endpoints on the same host. A non-200 health response
        # must not be treated as ready, and a stalled probe must be bounded.
        pending = {url for _, url in targets}
        deadline = time.monotonic() + timeout
        while pending:
            if any(proc.poll() is not None for proc in self._proc_list):
                raise RuntimeError(f"PD service exited before readiness: {sorted(pending)}")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"PD services did not become ready: {sorted(pending)}")
            for url in tuple(pending):
                try:
                    response = requests.get(url, timeout=min(3, remaining))
                    if response.status_code == 200:
                        pending.remove(url)
                except requests.RequestException:
                    pass
            if pending:
                time.sleep(1)


class _ObservedPDServer(_BoundedServer):
    """Record existing PD logs without replacing transport or worker behavior."""

    def __init__(self, *args, **kwargs):
        self.lines: list[str] = []
        self.output_changed = threading.Condition()
        visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
        self.assigned_devices = visible.split(",") if visible else None
        kwargs.setdefault("max_wait_seconds", 600)
        try:
            super().__init__(*args, **kwargs)
        except BaseException:
            # __exit__ is not invoked when __init__ raises after starting P/D.
            self._terminate_server()
            raise

    def _start_server_with_prefix(self, server_cmd, env_dict, log_prefix):
        # A3 communicators can share a device network address. Let HCCL choose
        # free ports instead of colliding on 16666; honor explicit runner ranges.
        env_dict = dict(env_dict)
        env_dict.setdefault("HCCL_NPU_SOCKET_PORT_RANGE", os.environ.get("HCCL_NPU_SOCKET_PORT_RANGE", "auto"))
        # All ranks in this single-node fixture share loopback. Avoid relying
        # on a container hostname being resolvable by the runner DNS.
        env_dict.setdefault("GLOO_SOCKET_IFNAME", os.environ.get("GLOO_SOCKET_IFNAME", "lo"))
        # RemotePDServer assigns logical slices starting at zero. Map those
        # slices into this test's allocation instead of taking physical card 0.
        if self.assigned_devices is not None:
            indices = [int(value) for value in env_dict["ASCEND_RT_VISIBLE_DEVICES"].split(",")]
            assert all(0 <= index < len(self.assigned_devices) for index in indices), (
                f"Insufficient assigned devices: {self.assigned_devices}, requested slice {indices}"
            )
            env_dict["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(self.assigned_devices[index] for index in indices)
        return super()._start_server_with_prefix(server_cmd, env_dict, log_prefix)

    def _read_output(self, pipe, prefix):
        with pipe:
            for line in pipe:
                with self.output_changed:
                    self.lines.append(f"{prefix}{line}")
                    self.output_changed.notify_all()
                print(f"{prefix}{line}", end="")

    def require_transfer(self, remote_request_id: str) -> None:
        # This INFO event is emitted only after a nonempty batch read succeeds.
        # The finished event follows descriptor processing; neither alone proves
        # consumption. Combine with D output and the unavailable-transfer control.
        success = f"KV cache transfer for request {remote_request_id} took "
        finished = f"Finished transferring KV cache for request {remote_request_id}."
        with self.output_changed:
            observed = self.output_changed.wait_for(
                lambda: any(success in line for line in self.lines) and any(finished in line for line in self.lines),
                timeout=10,
            )
        assert observed, f"Missing successful transfer evidence for {remote_request_id}"

    def require_transfer_failure(self, remote_request_id: str) -> None:
        failure = f"Failed to transfer KV cache for request {remote_request_id}:"
        with self.output_changed:
            observed = self.output_changed.wait_for(lambda: any(failure in line for line in self.lines), timeout=10)
        assert observed, f"Missing connector failure evidence for {remote_request_id}"

    def require_decode_graph(self) -> None:
        with self.output_changed:
            observed = self.output_changed.wait_for(
                lambda: any(line.startswith("[PD_1]") and "Replaying aclgraph" in line for line in self.lines),
                timeout=10,
            )
        assert observed, "No graph replay observed from the Decoder"


class _BoundedPDProxy(_BoundedServer, DisaggPDProxy):
    def __init__(self, *args, **kwargs):
        try:
            super().__init__(*args, **kwargs)
        except BaseException:
            self._terminate_server()
            raise


def _request(url: str, prompt: str | list[int], **overrides) -> dict:
    payload = {
        "model": SERVED_MODEL_NAME,
        "prompt": prompt,
        "temperature": 0,
        "seed": 0,
        "max_tokens": OUTPUT_TOKENS,
        "ignore_eos": True,
        "return_token_ids": True,
    }
    payload.update(overrides)
    response = requests.post(
        url,
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    result = response.json()
    assert len(result["choices"]) == 1, result
    choice = result["choices"][0]
    assert choice["finish_reason"] == "length", result
    assert result["usage"]["completion_tokens"] == payload["max_tokens"], result
    tokens = choice["token_ids"]
    assert len(tokens) == payload["max_tokens"], result
    return result


def _complete(url: str, prompt: str | list[int]) -> list[int]:
    return _request(url, prompt)["choices"][0]["token_ids"]


def _assert_transfer_failure_and_recovery(pd, prefill_url, decode_url, prompt, baseline):
    prefilled = _request(
        prefill_url,
        prompt,
        max_tokens=1,
        kv_transfer_params={"do_remote_decode": True, "do_remote_prefill": False},
    )
    transfer = dict(prefilled["kv_transfer_params"])
    assert transfer["do_remote_prefill"] and any(transfer["remote_block_ids"]), transfer
    # Own the port for the whole request, but do not listen: no unrelated peer
    # is contacted. Keep block addresses untouched. A fresh peer identity avoids
    # reusing the healthy peer's cached handshake metadata.
    with socket.socket() as unavailable:
        unavailable.bind(("0.0.0.0", 0))
        unavailable_engine = f"e2e-unavailable-{uuid.uuid4().hex}"
        unavailable_port = unavailable.getsockname()[1]
        transfer.update(
            remote_host="127.0.0.1",
            remote_port=unavailable_port,
            remote_engine_id=unavailable_engine,
        )
        # V1 can resolve peers through this mapping instead of top-level host
        # and engine fields. Redirect both representations to the owned endpoint.
        if "remote_multi_nodes_meta_mapping" in transfer:
            transfer["remote_multi_nodes_meta_mapping"] = {
                rank: {
                    **metadata,
                    "host": "127.0.0.1",
                    "engine_id": unavailable_engine,
                    "handshake_port": unavailable_port,
                }
                for rank, metadata in transfer["remote_multi_nodes_meta_mapping"].items()
            }
        response = requests.post(
            decode_url,
            json={
                "model": SERVED_MODEL_NAME,
                "prompt": prompt,
                "temperature": 0,
                "seed": 0,
                "max_tokens": OUTPUT_TOKENS,
                "ignore_eos": True,
                "return_token_ids": True,
                "kv_transfer_params": transfer,
            },
            timeout=REQUEST_TIMEOUT,
        )
        # A local client timeout or schema rejection does not prove the KV
        # failure contract. Require a server failure AND the correlated PD fault.
        assert response.status_code == 500, response.text
        assert "error" in response.json(), response.text
        pd.require_transfer_failure(transfer["remote_request_id"])

    recovered = _request(
        prefill_url,
        prompt,
        max_tokens=1,
        kv_transfer_params={"do_remote_decode": True, "do_remote_prefill": False},
    )["kv_transfer_params"]
    result = _request(decode_url, prompt, kv_transfer_params=recovered)
    assert result["choices"][0]["token_ids"] == baseline, result
    assert result["choices"][0].get("stop_reason") != "recomputed", result
    pd.require_transfer(recovered["remote_request_id"])


def _server_args(
    port: int, *, model: str, tp: int, graph: bool, prefix: bool, extra_args: tuple[str, ...], max_num_seqs: int = 4
) -> list[str]:
    args = [
        maybe_model_redirect(model),
        "--served-model-name",
        SERVED_MODEL_NAME,
        "--port",
        str(port),
        "--host",
        "127.0.0.1",
        "--seed",
        "0",
        "--dtype",
        "bfloat16",
        "--tensor-parallel-size",
        str(tp),
        "--max-model-len",
        "2048",
        "--max-num-seqs",
        str(max_num_seqs),
        "--max-num-batched-tokens",
        "512" if prefix else "2048",
        "--block-size",
        "128",
        "--enable-prefix-caching" if prefix else "--no-enable-prefix-caching",
        "--enable-prompt-tokens-details",
    ]
    if graph:
        args += [
            "--compilation-config",
            json.dumps(
                {
                    "cudagraph_mode": "FULL_DECODE_ONLY",
                    "cudagraph_capture_sizes": [size for size in (1, 2, 4) if size <= max_num_seqs],
                }
            ),
        ]
    else:
        args += ["--enforce-eager"]
    return args + list(extra_args)


def run_pd_contract(
    *,
    model: str,
    prefix: bool,
    tp: int = 1,
    extra_args: tuple[str, ...] = (),
    check_failure: bool = True,
    max_num_seqs: int = 4,
    aligned_reference: bool = False,
    compare_logprobs: bool = False,
    decode_prefix: bool | None = None,
    connector: str = "MooncakeConnectorV1",
    prompt_lengths: tuple[int, int, int, int] = (127, 128, 129, 1537),
) -> None:
    """E01/E03: graph decode and four requests retain the numerical contract.

    P is eager and D uses FULL_DECODE_ONLY. The baseline uses the same D graph
    configuration, so the primary comparison isolates the PD deployment change.
    Baseline and PD sequentially reuse devices; the total is 2 * tp, at most 4.
    """
    assert max_num_seqs in (1, 4)
    assert not aligned_reference or (prefix and max_num_seqs == 1)
    assert not compare_logprobs or prefix
    assert tp in (1, 2), "PD PR E2E must fit in four devices"
    assert connector in ("MooncakeConnectorV1", "MooncakeHybridConnector")
    assert len(prompt_lengths) == len(PROMPTS) and all(length > 0 for length in prompt_lengths)

    def server_args(port, graph, prefix_caching=None):
        args = _server_args(
            port,
            model=model,
            tp=tp,
            graph=graph,
            prefix=prefix,
            extra_args=extra_args,
            max_num_seqs=max_num_seqs,
        )
        if prefix_caching is not None:
            flag = "--enable-prefix-caching" if prefix else "--no-enable-prefix-caching"
            args[args.index(flag)] = "--enable-prefix-caching" if prefix_caching else "--no-enable-prefix-caching"
        return args

    probability_args = {"logprobs": 5, "return_tokens_as_token_ids": True} if compare_logprobs else {}

    def complete(url, prompt):
        return _request(url, prompt, **probability_args)

    def compare(url, prompt, reference, actual):
        if compare_logprobs:
            assert isinstance(prompt, list)
            assert_distribution_match(
                reference,
                actual,
                prompt,
                lambda history: _request(url, history, max_tokens=1, **probability_args),
            )
        else:
            assert actual["choices"][0]["token_ids"] == reference["choices"][0]["token_ids"], (
                f"PD={actual['choices'][0]['token_ids']}, colocated={reference['choices'][0]['token_ids']}"
            )

    def reference_response(url: str, prompt: str | list[int], max_tokens: int = OUTPUT_TOKENS) -> dict:
        if not aligned_reference:
            return _request(url, prompt, max_tokens=max_tokens, **probability_args)
        assert isinstance(prompt, list) and len(prompt) > 1
        # PD recomputes the last input token in decode. Match that boundary
        # locally: force only this known input token, then generate ten answers
        # without any constraint. No connector participates in the reference.
        response = _request(
            url,
            prompt[:-1],
            max_tokens=max_tokens + 1,
            vllm_xargs={"pd_input_suffix": prompt[-1]},
            **probability_args,
        )
        tokens = response["choices"][0]["token_ids"]
        assert tokens[0] == prompt[-1], "Reference did not reconstruct the known input suffix"
        assert len(tokens[1:]) == max_tokens
        response["choices"][0]["token_ids"] = tokens[1:]
        logprobs = response["choices"][0].get("logprobs")
        if logprobs is not None:
            for key in ("tokens", "token_logprobs", "top_logprobs", "text_offset"):
                assert len(logprobs[key]) == max_tokens + 1
                logprobs[key] = logprobs[key][1:]
        return response

    baseline_error = None
    baseline_port = get_open_port()
    baseline_args = server_args(baseline_port, True)
    baseline_env = {}
    if aligned_reference:
        baseline_args += ["--logits-processors", "tests.e2e.pull_request.pd_reference:InputSuffixProcessor"]
        # Make the reference processor importable by the separate API workers.
        baseline_env["PYTHONPATH"] = (
            str(Path(__file__).resolve().parents[3]) + os.pathsep + os.environ.get("PYTHONPATH", "")
        )
    with _ObservedPDServer([baseline_args], env_dict=baseline_env):
        baseline_url = f"http://127.0.0.1:{baseline_port}/v1/completions"
        prompts: list[str | list[int]] = list(PROMPTS)
        if prefix:
            prompts = []
            for text, length in zip(PROMPTS, prompt_lengths):
                response = requests.post(
                    f"http://127.0.0.1:{baseline_port}/tokenize",
                    json={"model": SERVED_MODEL_NAME, "prompt": text, "add_special_tokens": False},
                    timeout=REQUEST_TIMEOUT,
                )
                response.raise_for_status()
                tokens = response.json()["tokens"]
                assert tokens, response.text
                prompts.append((tokens * (length // len(tokens) + 1))[:length])
        reference_responses = [reference_response(baseline_url, prompt) for prompt in prompts]
        baseline = [response["choices"][0]["token_ids"] for response in reference_responses]
        if compare_logprobs:
            for prompt, reference, tokens in zip(prompts, reference_responses, baseline):
                assert isinstance(prompt, list)
                reference["_pd_replay_references"] = [
                    reference_response(baseline_url, prompt + tokens[:position], max_tokens=1)
                    for position in range(len(tokens))
                ]
        # Validate the same numerical contract on colocated cold/warm requests
        # before attributing a discrepancy to PD. The serial baseline remains
        # an independent control, never a golden updated from the PD result.
        with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
            concurrent_responses = list(pool.map(lambda prompt: reference_response(baseline_url, prompt), prompts))
        try:
            for prompt, reference, actual in zip(prompts, reference_responses, concurrent_responses):
                compare(baseline_url, prompt, reference, actual)
        except AssertionError as error:
            baseline_error = str(error) or "Colocated cold/warm comparison failed"
            print(f"Colocated numerical control failed: {baseline_error}", flush=True)

    # Model initialization takes minutes. Select each service's ports just
    # before starting it, rather than leaving them unreserved during baseline.
    prefill_port, decode_port = [get_open_port() for _ in range(2)]
    servers = []
    for port, role, graph in (
        (prefill_port, "kv_producer", False),
        (decode_port, "kv_consumer", True),
    ):
        config = {
            "kv_connector": connector,
            "kv_role": role,
            "kv_port": get_open_port(),
            "kv_load_failure_policy": "fail",
            "kv_connector_extra_config": {
                "prefill": {"dp_size": 1, "tp_size": tp},
                "decode": {"dp_size": 1, "tp_size": tp},
            },
        }
        servers.append(
            server_args(port, graph, decode_prefix if role == "kv_consumer" else None)
            + ["--kv-transfer-config", json.dumps(config)]
        )

    with (
        _ObservedPDServer(
            servers,
            # The cache-reset test API is gated by upstream's dev mode. These
            # test servers bind loopback only; no global runtime is changed.
            env_dict={"VLLM_LOGGING_LEVEL": "DEBUG", "VLLM_SERVER_DEV_MODE": "1" if prefix else "0"},
        ) as pd,
        _BoundedPDProxy(get_open_port(), [prefill_port], [decode_port]) as proxy,
    ):
        url = proxy.url_for("v1", "completions")
        with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
            responses = list(pool.map(lambda prompt: complete(url, prompt), prompts))
        outputs = [response["choices"][0]["token_ids"] for response in responses]
        assert len(outputs) == len(baseline)
        mismatches = [index for index, (actual, expected) in enumerate(zip(outputs, baseline)) if actual != expected]
        if mismatches:
            print(f"PD greedy token differences (checking numerical contract after transfer): {mismatches}", flush=True)

        # Bypass proxy retry/recompute for a correlated protocol-level request.
        # When testing prefix reuse, retain P's cache but reset D's local cache:
        # a full D-local hit would legitimately bypass the transfer under test.
        index = 3 if prefix else 0
        prompt = prompts[index]
        if prefix:
            reset = requests.post(f"http://127.0.0.1:{decode_port}/reset_prefix_cache", timeout=REQUEST_TIMEOUT)
            assert_prefix_reset_response(reset)
        prefilled = _request(
            f"http://127.0.0.1:{prefill_port}/v1/completions",
            prompt,
            max_tokens=1,
            kv_transfer_params={"do_remote_decode": True, "do_remote_prefill": False},
        )
        if prefix:
            assert prefilled["usage"]["prompt_tokens_details"]["cached_tokens"] > 0, prefilled
        transfer = prefilled["kv_transfer_params"]
        assert transfer["do_remote_prefill"], transfer
        assert any(transfer["remote_block_ids"]), transfer
        decoded = _request(
            f"http://127.0.0.1:{decode_port}/v1/completions",
            prompt,
            kv_transfer_params=transfer,
            **probability_args,
        )
        assert decoded["choices"][0].get("stop_reason") != "recomputed", decoded
        pd.require_transfer(transfer["remote_request_id"])
        pd.require_decode_graph()
        if not prefix and check_failure:
            _assert_transfer_failure_and_recovery(
                pd,
                f"http://127.0.0.1:{prefill_port}/v1/completions",
                f"http://127.0.0.1:{decode_port}/v1/completions",
                prompt,
                baseline[index],
            )
        # Keep numerical failures, but first collect the independent transport,
        # graph evidence needed to locate their origin.
        assert baseline_error is None, baseline_error
        compare(url, prompt, reference_responses[index], decoded)
        for prompt, reference, actual in zip(prompts, reference_responses, responses):
            compare(url, prompt, reference, actual)
