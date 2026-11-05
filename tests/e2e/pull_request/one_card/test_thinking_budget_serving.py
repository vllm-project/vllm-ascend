# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools
import json
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import requests
from transformers import AutoTokenizer
from vllm.utils.network_utils import get_open_port

from tests.e2e.conftest import RemoteOpenAIServer

MODEL = "Qwen/Qwen3-1.7B"
BUDGETS = [None, -1, 0, 1, 2, 3, 4, 16, 64, 128]
REQUEST_TIMEOUT = 120
KV_CACHE_BYTES = 2 * 1024**3


@pytest.mark.parametrize("enforce_eager", [True, False])
@pytest.mark.parametrize("speculative", [False, True])
def test_thinking_budget_serving(enforce_eager, speculative, tmp_path, runner_version="1"):
    response_index = itertools.count()
    port = get_open_port()
    args = [
        "--port",
        str(port),
        "--host",
        "127.0.0.1",
        "--max-model-len",
        "4096",
        "--max-num-seqs",
        "4",
        "--max-num-batched-tokens",
        "512",
        "--kv-cache-memory-bytes",
        str(KV_CACHE_BYTES),
        "--reasoning-parser",
        "qwen3",
        "--reasoning-config",
        json.dumps({"reasoning_start_str": "<think>", "reasoning_end_str": "</think>"}),
    ]
    if enforce_eager:
        args.append("--enforce-eager")
    if speculative:
        args.extend(
            [
                "--speculative-config",
                json.dumps(
                    {
                        "method": "ngram_gpu",
                        "num_speculative_tokens": 3,
                        "prompt_lookup_min": 2,
                        "prompt_lookup_max": 5,
                    }
                ),
            ]
        )
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    start_id = tokenizer.encode("<think>", add_special_tokens=False)[0]
    end_id = tokenizer.encode("</think>", add_special_tokens=False)[0]

    def thinking_count(ids):
        end = ids.index(end_id) if end_id in ids else len(ids)
        start = ids.index(start_id) + 1 if start_id in ids[:end] else 0
        return end - start

    with RemoteOpenAIServer(
        MODEL,
        args,
        server_host="127.0.0.1",
        server_port=port,
        auto_port=False,
        env_dict={"VLLM_USE_V2_MODEL_RUNNER": runner_version},
    ) as server:
        url = server.url_for("v1", "chat", "completions")
        base = {
            "model": MODEL,
            "messages": [{"role": "user", "content": "What is 2 + 3? Answer with the number."}],
            "max_tokens": 512,
            "seed": 42,
            "return_token_ids": True,
            "chat_template_kwargs": {"enable_thinking": True},
        }
        for temperature in [0.0, 0.6]:

            def run(budget, current_temperature=temperature):
                response = requests.post(
                    url,
                    json={**base, "temperature": current_temperature, "thinking_token_budget": budget},
                    timeout=REQUEST_TIMEOUT,
                )
                assert response.status_code == 200, response.text
                choice = response.json()["choices"][0]
                record = {
                    "budget": budget,
                    "temperature": current_temperature,
                    "eager": enforce_eager,
                    "speculative": speculative,
                    "choice": choice,
                }
                response_file = tmp_path / f"response-{next(response_index):03d}.json"
                response_file.write_text(json.dumps(record, ensure_ascii=False))
                thinking_tokens = thinking_count(choice["token_ids"])
                print(
                    json.dumps(
                        {
                            "budget": budget,
                            "thinking_tokens": thinking_tokens,
                            "finish_reason": choice["finish_reason"],
                            "response_file": str(response_file),
                        }
                    )
                )
                if budget is not None and budget >= 0:
                    assert end_id in choice["token_ids"], choice
                    assert thinking_tokens <= budget, choice
                    assert "5" in choice["message"]["content"], choice
                    assert choice["finish_reason"] == "stop", choice
                elif choice["finish_reason"] == "length":
                    # Unlimited thinking still obeys the total output limit.
                    assert len(choice["token_ids"]) == base["max_tokens"], choice
                else:
                    assert choice["finish_reason"] == "stop", choice
                    assert "5" in choice["message"]["content"], choice

            with ThreadPoolExecutor(max_workers=4) as executor:
                list(executor.map(run, BUDGETS))
        for invalid in [-2, True, 1.5, "3"]:
            response = requests.post(url, json={**base, "thinking_token_budget": invalid}, timeout=REQUEST_TIMEOUT)
            assert response.status_code == 400, response.text
            assert "thinking_token_budget" in response.text
            print(json.dumps({"invalid_budget": invalid, "status": response.status_code, "error": response.json()}))
        # Invalid requests must not poison the engine or a reused request slot.
        run(0)
        unlimited = []
        for extra in [{}, {"thinking_token_budget": None}, {"thinking_token_budget": -1}]:
            response = requests.post(url, json={**base, "temperature": 0, **extra}, timeout=REQUEST_TIMEOUT)
            assert response.status_code == 200, response.text
            unlimited.append(response.json()["choices"][0]["token_ids"])
        assert unlimited[0] == unlimited[1] == unlimited[2]
        print(json.dumps({"unlimited_equal": True, "eager": enforce_eager, "speculative": speculative}))

        stream_ids = []
        answer_parts = []
        with requests.post(
            url,
            json={**base, "temperature": 0, "thinking_token_budget": 16, "stream": True},
            stream=True,
            timeout=REQUEST_TIMEOUT,
        ) as response:
            assert response.status_code == 200, response.text
            for line in response.iter_lines():
                if not line.startswith(b"data: ") or line == b"data: [DONE]":
                    continue
                chunk = json.loads(line[6:])
                for choice in chunk.get("choices", []):
                    stream_ids.extend(choice.get("token_ids") or [])
                    answer_parts.append(choice.get("delta", {}).get("content") or "")
        assert thinking_count(stream_ids) <= 16
        assert "".join(answer_parts).strip()
        print(
            json.dumps(
                {
                    "stream_thinking_tokens": thinking_count(stream_ids),
                    "eager": enforce_eager,
                    "speculative": speculative,
                }
            )
        )

        if speculative:
            repeat_body = {
                **base,
                "temperature": 0,
                "thinking_token_budget": 64,
                "messages": [
                    {
                        "role": "user",
                        "content": "Repeat this list exactly twice: alpha beta gamma delta epsilon zeta eta theta. "
                        "alpha beta gamma delta epsilon zeta eta theta.",
                    }
                ],
            }
            response = requests.post(url, json=repeat_body, timeout=REQUEST_TIMEOUT)
            assert response.status_code == 200, response.text
            counters = {}
            for _ in range(15):
                metrics = requests.get(server.url_for("metrics"), timeout=10)
                metrics.raise_for_status()
                counters = {}
                for line in metrics.text.splitlines():
                    if line.startswith("vllm:spec_decode_num_") and not line.startswith("#"):
                        name = line.split("{", 1)[0].split()[0]
                        counters[name] = counters.get(name, 0) + float(line.split()[-1])
                drafted = counters.get("vllm:spec_decode_num_draft_tokens_total", 0)
                accepted = counters.get("vllm:spec_decode_num_accepted_tokens_total", 0)
                if drafted > accepted > 0:
                    break
                time.sleep(1)
            assert drafted > accepted > 0, counters
            print(json.dumps({"spec_metrics": counters, "eager": enforce_eager}))
