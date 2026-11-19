# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import os

import requests
from prometheus_client.parser import text_string_to_metric_families

from tests.e2e.conftest import RemoteOpenAIServer

MODEL = os.environ.get("QWEN3_06B_MODEL", "Qwen/Qwen3-0.6B")


def test_dense_online_dp2_tp2():
    args = [
        "--tensor-parallel-size",
        "2",
        "--data-parallel-size",
        "2",
        "--distributed-executor-backend",
        "mp",
        "--api-server-count",
        "1",
        "--dtype",
        "float16",
        "--max-model-len",
        "512",
        "--max-num-seqs",
        "4",
        "--gpu-memory-utilization",
        "0.35",
        "--enforce-eager",
    ]
    with RemoteOpenAIServer(MODEL, args, env_dict={"VLLM_USE_V2_MODEL_RUNNER": "1"}) as server:
        with server.get_client() as client:
            for rank in range(2):
                output = client.completions.create(
                    model=MODEL,
                    prompt="The capital of France is",
                    temperature=0,
                    max_tokens=8,
                    extra_headers={"X-data-parallel-rank": str(rank)},
                )
                assert output.choices[0].text.strip()

        metrics = requests.get(server.url_for("metrics"), timeout=10)
        metrics.raise_for_status()
        ranks = {
            sample.labels["engine"]
            for family in text_string_to_metric_families(metrics.text)
            for sample in family.samples
            if sample.name == "vllm:request_success_total"
            and sample.labels["finished_reason"] in ("length", "stop")
            and sample.value > 0
        }
        assert ranks == {"0", "1"}
