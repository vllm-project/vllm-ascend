from __future__ import annotations

import json
import math
import os
from typing import Any

import requests
from vllm.utils.network_utils import get_open_port

from tests.e2e.conftest import RemoteOpenAIServer

MODEL = os.environ.get("VLLM_TEST_MODEL", "Qwen/Qwen3-0.6B")


def _validate_detailed_metrics(metrics: dict[str, Any]) -> None:
    histogram = metrics["acceptance_histogram"]
    accepted_per_step = metrics["per_step_accepted"]
    drafted_per_step = metrics["per_step_drafted"]

    assert metrics["num_spec_steps"] > 0
    assert metrics["num_draft_tokens"] > 0
    assert metrics["num_spec_steps"] == sum(histogram)
    assert metrics["num_accepted_draft_tokens"] == sum(accepted * count for accepted, count in enumerate(histogram))
    assert metrics["num_accepted_draft_tokens"] == sum(accepted_per_step)
    assert metrics["num_draft_tokens"] == sum(drafted_per_step)
    assert len(accepted_per_step) == metrics["num_spec_steps"]
    assert len(drafted_per_step) == metrics["num_spec_steps"]
    assert math.isclose(
        metrics["mean_acceptance_length"],
        1 + metrics["num_accepted_draft_tokens"] / metrics["num_spec_steps"],
    )
    assert math.isclose(
        metrics["draft_acceptance_rate"],
        metrics["num_accepted_draft_tokens"] / metrics["num_draft_tokens"],
    )


def test_per_request_spec_decode_metrics():
    port = get_open_port()
    server_args = [
        "--enforce-eager",
        "--max-model-len",
        "1024",
        "--gpu-memory-utilization",
        "0.35",
        "--speculative-config",
        json.dumps(
            {
                "method": "ngram",
                "num_speculative_tokens": 3,
                "prompt_lookup_min": 1,
                "prompt_lookup_max": 4,
            }
        ),
        "--per-request-spec-decode-metrics",
        "detailed",
        "--port",
        str(port),
    ]
    prompt = "alpha beta gamma alpha beta gamma alpha beta gamma alpha beta"
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "temperature": 0,
        "seed": 42,
        "max_tokens": 24,
    }

    with RemoteOpenAIServer(
        MODEL,
        server_args,
        server_host="127.0.0.1",
        server_port=port,
        auto_port=False,
    ) as server:
        response = requests.post(
            server.url_for("v1", "completions"),
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        metrics = response.json()["metrics"]["speculative_decoding"]
        _validate_detailed_metrics(metrics)

        stream_response = requests.post(
            server.url_for("v1", "completions"),
            json={
                **payload,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
            stream=True,
            timeout=120,
        )
        stream_response.raise_for_status()
        chunks = [
            json.loads(line.removeprefix("data: "))
            for line in stream_response.iter_lines(decode_unicode=True)
            if line and line != "data: [DONE]"
        ]
        metric_chunks = [chunk for chunk in chunks if chunk.get("metrics") is not None]
        assert len(metric_chunks) == 1
        assert metric_chunks[0] is chunks[-1]
        assert metric_chunks[0]["usage"] is not None
        _validate_detailed_metrics(metric_chunks[0]["metrics"]["speculative_decoding"])
