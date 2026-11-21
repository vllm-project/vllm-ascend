# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import os

import requests
from prometheus_client.parser import text_string_to_metric_families
from vllm.utils.network_utils import get_open_port

from tests.e2e.conftest import RemoteOpenAIServer


def test_qwen35_moe_online_dp2_tp2_ep4():
    model = os.environ.get("QWEN35_MOE_EP_MODEL", "Qwen/Qwen3.5-35B-A3B")
    quantization = os.environ.get("QWEN35_MOE_EP_QUANTIZATION")
    port = get_open_port()
    args = [
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--tensor-parallel-size",
        "2",
        "--data-parallel-size",
        "2",
        "--enable-expert-parallel",
        "--distributed-executor-backend",
        "mp",
        "--api-server-count",
        "1",
        "--dtype",
        "float16",
        "--mamba-ssm-cache-dtype",
        "float16",
        "--max-model-len",
        "2048",
        "--max-num-batched-tokens",
        "1024",
        "--max-num-seqs",
        "8",
        "--gpu-memory-utilization",
        "0.85",
        "--language-model-only",
        "--no-enable-prefix-caching",
        "--compilation-config",
        '{"cudagraph_mode":"FULL_DECODE_ONLY","decode_cudagraph_capture_request_sizes":[8]}',
        "--additional-config",
        '{"ascend_compilation_config":{"enable_npugraph_ex":false}}',
    ]
    if quantization:
        args.extend(("--quantization", quantization))

    with RemoteOpenAIServer(
        model,
        args,
        server_host="127.0.0.1",
        server_port=port,
        auto_port=False,
        env_dict={"VLLM_USE_V2_MODEL_RUNNER": "1"},
    ) as server:
        with server.get_client() as client:

            def complete_on(rank):
                output = client.completions.create(
                    model=model,
                    prompt=[1000] * 640,
                    temperature=0,
                    max_tokens=8,
                    extra_body={"ignore_eos": True},
                    extra_headers={"X-data-parallel-rank": str(rank)},
                )
                assert output.usage is not None
                assert output.usage.completion_tokens == 8

            complete_on(0)
            complete_on(1)
            with client.completions.create(
                model=model,
                prompt="Write a long explanation of prime numbers.",
                temperature=0,
                max_tokens=192,
                stream=True,
                extra_body={"ignore_eos": True},
                extra_headers={"X-data-parallel-rank": "0"},
            ) as stream:
                first = next(stream)
                assert first.choices
            complete_on(1)
            complete_on(0)

        metrics = requests.get(server.url_for("metrics"), timeout=10)
        metrics.raise_for_status()
        completed_ranks = {
            sample.labels["engine"]
            for family in text_string_to_metric_families(metrics.text)
            for sample in family.samples
            if sample.name == "vllm:request_success_total"
            and sample.labels["finished_reason"] in ("length", "stop")
            and sample.value >= 2
        }
        assert completed_ranks == {"0", "1"}
