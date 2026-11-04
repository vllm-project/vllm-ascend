# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Qwen3 BF16 completion parity across retained 4 -> 2 -> 4 transitions."""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest
import regex as re
import requests

from tests.e2e.conftest import RemoteOpenAIServer

MODEL = "Qwen/Qwen3-30B-A3B-Base"


@pytest.mark.parametrize("graphs", [False, True])
def test_fasteep_alltoall_serving(tmp_path, graphs):
    observations = _run_serving(tmp_path, "1", "alltoall", graphs)
    assert all(outputs == observations[0] for outputs in observations[1:])


def test_fasteep_allgather_graph_parity(tmp_path):
    # BF16 reduction grouping changes with EP topology. Compare graph replay
    # with eager execution at each identical placement, including restored ranks.
    eager = _run_serving(tmp_path / "eager", "0", "allgather", False)
    graphs = _run_serving(tmp_path / "graphs", "0", "allgather", True)
    assert graphs == eager


def _run_serving(tmp_path, runner, communication, graphs):
    tmp_path.mkdir(parents=True, exist_ok=True)
    placement = tmp_path / "placement.json"
    devices = [
        {"device_expert": experts}
        for experts in [
            list(range(64)),
            list(range(32, 96)),
            list(range(64, 128)),
            list(range(32)) + list(range(96, 128)),
        ]
    ]
    placement.write_text(json.dumps({"moe_layer_count": 48, "layer_list": [{"device_list": devices}] * 48}))
    args = [
        "--data-parallel-size",
        "4",
        "--data-parallel-size-local",
        "4",
        "--enable-expert-parallel",
        "--api-server-count",
        "1",
        "--dtype",
        "bfloat16",
        "--enforce-eager",
        "--enable-sleep-mode",
        "--no-enable-prefix-caching",
        "--max-model-len",
        "1024",
        "--max-num-seqs",
        "8",
        "--max-num-batched-tokens",
        "512",
        "--kv-cache-memory-bytes",
        "2147483648",
        "--worker-extension-cls",
        "vllm_ascend.distributed.fasteep.runtime.FastEEPWorkerExtension",
        "--middleware",
        "vllm_ascend.distributed.fasteep.frontend.FastEEPMiddleware",
        "--additional-config",
        json.dumps(
            {
                "eplb_config": {"dynamic_eplb": False, "num_redundant_experts": 128, "expert_map_path": str(placement)},
                "enable_cpu_binding": False,
                "fasteep": {"communication": communication, "verify_migration": True},
            }
        ),
    ]
    if graphs:
        config_index = args.index("--additional-config") + 1
        additional = json.loads(args[config_index])
        additional["ascend_compilation_config"] = {"fuse_norm_quant": False}
        args[config_index] = json.dumps(additional)
        args.remove("--enforce-eager")
        args += [
            "--compilation-config",
            json.dumps({"mode": 3, "cudagraph_mode": "PIECEWISE", "cudagraph_capture_sizes": [1, 2, 4, 8]}),
        ]
    env = {
        "VLLM_USE_V2_MODEL_RUNNER": runner,
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:False",
        "HCCL_CONNECT_TIMEOUT": "120",
        "HCCL_EXEC_TIMEOUT": "120",
        "HCCL_BUFFSIZE": "64",
        "HCCL_DETERMINISTIC": "true",
        "OMP_NUM_THREADS": "4",
    }
    with RemoteOpenAIServer(MODEL, args, env_dict=env, max_wait_seconds=900) as server:

        def complete(prompt):
            response = requests.post(
                server.url_for("v1", "completions"),
                headers={"X-data-parallel-rank": "0"},
                json={"model": MODEL, "prompt": prompt, "temperature": 0, "seed": 0, "max_tokens": 32, "logprobs": 1},
                timeout=240,
            )
            response.raise_for_status()
            body = response.json()
            assert body["usage"]["completion_tokens"] > 0
            choice = body["choices"][0]
            return choice["logprobs"]["tokens"], choice["finish_reason"]

        prompts = [
            "The capital of France is",
            "One plus one equals",
            "The next prime after seven is",
            "Water freezes at",
        ]

        def resize(target):
            started = time.monotonic()
            response = requests.post(
                server.url_for("fasteep_scale_ep"),
                json={"new_data_parallel_size": target, "drain_timeout": 180},
                timeout=360,
            )
            response.raise_for_status()
            assert response.json()["active"] == target
            return started, time.monotonic()

        with ThreadPoolExecutor(max_workers=4) as pool:
            baseline = [complete(prompt) for prompt in prompts]
            assert [complete(prompt) for prompt in prompts] == baseline
            patterns = [r"(?i)\bparis\b", r"(?i)\b(two|2)\b", r"(?i)\b(eleven|11)\b", r"(?i)\b(zero|0|32)\b"]
            assert all(re.search(pattern, "".join(output[0])[:64]) for pattern, output in zip(patterns, baseline))
            observations = [baseline]
            for target in (3, 2, 4, 2, 4):
                resize(target)
                observations.append([complete(prompt) for prompt in prompts])

            started = Event()

            def stream():
                chunks = 0
                done = False
                with requests.post(
                    server.url_for("v1", "completions"),
                    json={
                        "model": MODEL,
                        "prompt": "List the positive integers in order: 1, 2, 3,",
                        "temperature": 0,
                        "max_tokens": 128,
                        "ignore_eos": True,
                        "stream": True,
                    },
                    stream=True,
                    timeout=240,
                ) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if not line.startswith(b"data: "):
                            continue
                        if line[6:] == b"[DONE]":
                            done = True
                            break
                        if json.loads(line[6:]).get("choices"):
                            chunks += 1
                            started.set()
                assert done and chunks > 1
                return time.monotonic()

            def arriving_request():
                output = complete(prompts[0])
                return output, time.monotonic()

            streaming = pool.submit(stream)
            assert started.wait(120)
            shrinking = pool.submit(resize, 2)
            time.sleep(0.1)
            arriving = pool.submit(arriving_request)
            stream_finished = streaming.result(timeout=300)
            resize_started, resize_finished = shrinking.result(timeout=360)
            arrival_output, arrival_finished = arriving.result(timeout=300)
            # Allow a small margin for HTTP response delivery to the clients.
            assert resize_started <= stream_finished <= resize_finished + 0.1
            assert arrival_finished >= resize_finished - 0.1
            assert arrival_output == observations[2][0]
            resize(4)
            assert complete(prompts[0]) == observations[-1][0]
            return observations
