# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import requests
from vllm.transformers_utils.utils import maybe_model_redirect
from vllm.utils.network_utils import get_open_port

from tests.e2e.conftest import RemotePDServer, VllmRunner, wait_until_npu_memory_free

MODEL = "Qwen/Qwen3.6-35B-A3B"


@pytest.mark.parametrize("cache_mode", ["none", "align"])
def test_qwen36_recoverssm_matches_mtp_baseline_and_graph(monkeypatch, cache_mode):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    prompts = ["What is 17 times 23?", "Explain why the sky is blue.", "Write a Python function to reverse a list."]
    if cache_mode == "align":
        prefix = "This is shared background context for the following question. " * 128
        prompts = [prefix + prompt for prompt in prompts]
    baseline = None
    for recovery, eager in [(False, True), (True, True), (True, False)]:
        with VllmRunner(
            "Qwen/Qwen3.6-35B-A3B",
            tensor_parallel_size=4,
            dtype="bfloat16",
            mamba_ssm_cache_dtype="float32",
            mamba_cache_mode=cache_mode,
            enable_prefix_caching=cache_mode == "align",
            use_replayssm=recovery,
            speculative_config={"method": "mtp", "num_speculative_tokens": 4},
            enforce_eager=eager,
            max_model_len=4096,
            max_num_seqs=8,
            max_num_batched_tokens=1024,
            gpu_memory_utilization=0.9,
        ) as runner:
            result = runner.generate_greedy(prompts, 64)
            # Second call exercises finished-request reuse and align prefix hits.
            reused = runner.generate_greedy(list(reversed(prompts)), 64)
        tokens = [item[0] for item in result]
        assert all(tokens)
        assert [item[0] for item in reused] == list(reversed(tokens))
        if baseline is None:
            baseline = tokens
        else:
            assert tokens == baseline


def _complete_pd_request(url, prompt, **kwargs):
    response = requests.post(
        url + "/v1/completions",
        json={
            "model": "qwen-gdn-recoverssm",
            "prompt": prompt,
            "temperature": 0,
            "max_tokens": 32,
            "ignore_eos": True,
            "return_token_ids": True,
            **kwargs,
        },
        timeout=180,
    )
    response.raise_for_status()
    return response.json()


def _generated_token_ids(result):
    choice = result["choices"][0]
    assert choice["finish_reason"] == "length"
    assert len(choice["token_ids"]) == 32
    return choice["token_ids"]


@pytest.mark.e2e_model(MODEL)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="spec_decode,mtp,chunked_prefill",
    parallel="TP",
    deploy="pd_disaggregation",
    hardware="A3",
    quantization="BF16",
    graph_mode="eager",
)
@pytest.mark.parametrize("connector", ["MooncakeConnectorV2", "MooncakePullConnector"])
@wait_until_npu_memory_free()
def test_qwen_recoverssm_pd_matches_local_generation(connector):
    model = maybe_model_redirect(MODEL)
    p_port, d_port = get_open_port(), get_open_port()
    p_url, d_url = f"http://127.0.0.1:{p_port}", f"http://127.0.0.1:{d_port}"
    common_args = [
        "--served-model-name",
        "qwen-gdn-recoverssm",
        "--tensor-parallel-size",
        "2",
        "--dtype",
        "bfloat16",
        "--mamba-ssm-cache-dtype",
        "float32",
        "--mamba-cache-mode",
        "none",
        "--no-enable-prefix-caching",
        "--use-replayssm",
        "--speculative-config",
        json.dumps({"method": "mtp", "num_speculative_tokens": 4}),
        "--enforce-eager",
        "--max-model-len",
        "2048",
        "--max-num-batched-tokens",
        "128",
        "--max-num-seqs",
        "4",
        "--enable-chunked-prefill",
        "--gpu-memory-utilization",
        "0.85",
        "--generation-config",
        "vllm",
        "--seed",
        "42",
    ]
    servers = [
        [
            model,
            *common_args,
            "--port",
            str(port),
            "--kv-transfer-config",
            json.dumps(
                {"kv_connector": connector, "kv_role": role, "kv_port": kv_port, "kv_load_failure_policy": "fail"}
            ),
        ]
        for port, role, kv_port in (
            (p_port, "kv_producer", 30000),
            (d_port, "kv_consumer", 31000),
        )
    ]
    prompts = [
        "What is 17 times 23?",
        "This is background information. " * 80 + "Explain why the sky is blue.",
    ]
    with RemotePDServer(servers, env_dict={"VLLM_USE_V2_MODEL_RUNNER": "1"}):
        expected = [_generated_token_ids(_complete_pd_request(p_url, prompt)) for prompt in prompts]

        def generate(prompt):
            prefill = _complete_pd_request(
                p_url, prompt, max_tokens=1, min_tokens=1, kv_transfer_params={"do_remote_decode": True}
            )
            transfer = prefill["kv_transfer_params"]
            assert transfer["do_remote_prefill"]
            assert any(transfer["remote_block_ids"])
            return _generated_token_ids(_complete_pd_request(d_url, prompt, kv_transfer_params=transfer))

        assert [generate(prompt) for prompt in prompts] == expected
        # Repeat with concurrent, reversed requests to exercise recycled state pages.
        with ThreadPoolExecutor(max_workers=2) as executor:
            assert list(executor.map(generate, reversed(prompts))) == list(reversed(expected))

        # Connector statistics can reach Prometheus after the response has finished.
        def metric_total(text, name):
            return sum(float(line.split()[-1]) for line in text.splitlines() if line.startswith(name + "{"))

        deadline = time.monotonic() + 30
        while True:
            metrics = requests.get(d_url + "/metrics", timeout=30)
            metrics.raise_for_status()

            drafts = metric_total(metrics.text, "vllm:spec_decode_num_drafts_total")
            transferred = metric_total(metrics.text, "vllm:mooncake_bytes_transferred_sum")
            assert metric_total(metrics.text, "vllm:mooncake_num_failed_transfers_total") == 0
            if (drafts > 0 and transferred > 0) or time.monotonic() >= deadline:
                break
            time.sleep(1)
        assert drafts > 0, "The decoder did not execute speculative decoding"
        assert transferred > 0, "No successful PD transfer was recorded"
