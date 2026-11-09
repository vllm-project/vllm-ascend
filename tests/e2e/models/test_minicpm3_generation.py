# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniCPM3-4B generation coverage for Ascend.

The config-driven cases run only when ``MiniCPM3-4B.yaml`` is selected via
``--config``. They cover offline and OpenAI-compatible serving, greedy and
non-greedy sampling, English and Chinese prompts, and prompts that are short
and long enough to exercise prefill/chunking. Numerical correctness against the
reference implementation is covered by the GSM8K case in
``test_lm_eval_correctness.py``.

This file is opt-in. The nightly accuracy job runs only
``test_lm_eval_correctness.py`` for each entry of
``tests/e2e/models/configs/accuracy.txt``, so the cases below execute when
they are requested explicitly:

Run on a single card::

    python -m pytest -sv tests/e2e/models \
        --config tests/e2e/models/configs/MiniCPM3-4B.yaml --tp-size 1

Run with tensor parallelism::

    python -m pytest -sv tests/e2e/models \
        --config tests/e2e/models/configs/MiniCPM3-4B.yaml --tp-size 2
"""

from pathlib import Path
from typing import Any

import pytest
import requests
import torch
import yaml
from transformers import AutoTokenizer
from vllm import SamplingParams
from vllm.utils.network_utils import get_open_port

from tests.e2e.conftest import (
    RemoteOpenAIServer,
    VllmRunner,
    wait_until_npu_memory_free,
)

MODEL_NAME = "openbmb/MiniCPM3-4B"
SERVED_MODEL_NAME = "minicpm3-4b"
SHORT_PROMPTS = [
    "Hello, my name is",
    "人工智能的未来是",
]
LONG_INSTRUCTIONS = [
    "Summarize the archive excerpt above in one sentence.",
    "请用一句话概括上面的档案内容。",
]


def _load_minicpm3_config(config_filename: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(config_filename).read_text(encoding="utf-8"))
    if config.get("model_name") != MODEL_NAME:
        pytest.skip(f"MiniCPM3-4B config not selected; got {config.get('model_name')!r}")
    return config


def _skip_if_insufficient_npu(tp_size: int) -> None:
    if not torch.npu.is_available():
        pytest.skip("MiniCPM3-4B E2E requires Ascend NPU")
    if torch.npu.device_count() < tp_size:
        pytest.skip(f"TP{tp_size} requires {tp_size} NPUs, found {torch.npu.device_count()}")


def _repeat_to_token_length(tokenizer, text: str, target_tokens: int) -> str:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    repeats = (target_tokens + len(token_ids) - 1) // len(token_ids)
    decoded = tokenizer.decode((token_ids * repeats)[:target_tokens], skip_special_tokens=True)
    while len(tokenizer.encode(decoded, add_special_tokens=False)) < target_tokens:
        decoded = f"{decoded} {text}"
    return decoded


def _build_prompts(tokenizer, long_prompt_tokens: int) -> list[str]:
    english_filler = _repeat_to_token_length(
        tokenizer,
        "The archive records routine maintenance, weather observations, and staff schedules. ",
        long_prompt_tokens,
    )
    chinese_filler = _repeat_to_token_length(
        tokenizer,
        "档案记录日常维护、天气观测和人员排班等例行事项，内容仅供内部检索和核对。",
        long_prompt_tokens,
    )
    return [
        *SHORT_PROMPTS,
        f"{english_filler}\n\n{LONG_INSTRUCTIONS[0]}",
        f"{chinese_filler}\n\n{LONG_INSTRUCTIONS[1]}",
    ]


def _tokens_text(outputs: list[tuple]) -> list[tuple[list[int], str]]:
    """Keep only the generated token IDs and text.

    ``VllmRunner.generate`` prepends the prompt to both the token IDs and the
    text, whereas the OpenAI-compatible server returns the completion only.
    The ``*_logprobs`` helpers expose the completion alone, which is what both
    sides of an offline/online comparison must use.
    """
    return [(list(item[0]), item[1]) for item in outputs]


def _sampling_params(config: dict[str, Any], seed: int | None = None) -> SamplingParams:
    generation = config["generation"]
    return SamplingParams(
        max_tokens=generation["max_tokens"],
        temperature=generation["temperature"],
        top_p=generation["top_p"],
        seed=generation["seed"] if seed is None else seed,
    )


def _assert_non_empty(outputs: list[tuple[list[int], str]], label: str) -> None:
    for index, (token_ids, text) in enumerate(outputs):
        assert token_ids, f"{label} prompt {index} returned no token IDs"
        assert text.strip(), f"{label} prompt {index} returned empty text"


def _run_offline(config: dict[str, Any], prompts: list[str], tp_size: int):
    serve = config["serve"]
    generation = config["generation"]
    with VllmRunner(
        MODEL_NAME,
        max_model_len=serve["max_model_len"],
        dtype=serve["dtype"],
        gpu_memory_utilization=serve["gpu_memory_utilization"],
        tensor_parallel_size=tp_size,
        enforce_eager=serve.get("enforce_eager", False),
    ) as runner:
        greedy = _tokens_text(runner.generate_greedy_logprobs(prompts, generation["max_tokens"], num_logprobs=None))
        _assert_non_empty(greedy, "offline greedy")

        sampling_params = _sampling_params(config)
        warmup = _tokens_text(runner.generate_w_logprobs(prompts, sampling_params))
        _assert_non_empty(warmup, "offline sampled")

        torch.npu.synchronize()
        baseline_bytes = torch.npu.memory_allocated()
        repeats = []
        for _ in range(generation["memory_rounds"]):
            repeat = _tokens_text(runner.generate_w_logprobs(prompts, sampling_params))
            _assert_non_empty(repeat, "offline sampled repeat")
            repeats.append(repeat)
        torch.npu.synchronize()
        growth_mb = (torch.npu.memory_allocated() - baseline_bytes) / 1024**2
        print(f"MiniCPM3-4B NPU memory growth over repeated generation: {growth_mb:.1f} MiB")
        assert growth_mb <= generation["max_memory_growth_mb"], (
            f"NPU memory grew by {growth_mb:.1f} MiB across "
            f"{generation['memory_rounds']} deterministic generation rounds"
        )

    for repeat in repeats:
        assert repeat == warmup, "Seeded non-greedy generation is not reproducible in the offline runner"
    return greedy, warmup


def _server_args(config: dict[str, Any], tp_size: int, port: int) -> list[str]:
    serve = config["serve"]
    args = [
        "--served-model-name",
        SERVED_MODEL_NAME,
        "--dtype",
        serve["dtype"],
        "--max-model-len",
        str(serve["max_model_len"]),
        "--gpu-memory-utilization",
        str(serve["gpu_memory_utilization"]),
        "--tensor-parallel-size",
        str(tp_size),
        "--trust-remote-code",
        "--port",
        str(port),
    ]
    if serve.get("enforce_eager", False):
        args.append("--enforce-eager")
    return args


def _online_generate(
    server: RemoteOpenAIServer,
    prompts: list[str],
    config: dict[str, Any],
    *,
    temperature: float,
    seed: int,
) -> list[tuple[list[int] | None, str]]:
    generation = config["generation"]
    results = []
    for prompt in prompts:
        payload: dict[str, Any] = {
            "model": SERVED_MODEL_NAME,
            "prompt": prompt,
            "max_tokens": generation["max_tokens"],
            "temperature": temperature,
            "seed": seed,
            "return_token_ids": True,
        }
        if temperature > 0:
            payload["top_p"] = generation["top_p"]
        response = requests.post(
            server.url_for("v1", "completions"),
            json=payload,
            timeout=600,
        )
        response.raise_for_status()
        choice = response.json()["choices"][0]
        text = choice["text"]
        assert text.strip(), f"OpenAI-compatible server returned empty text for {prompt[:32]!r}"
        results.append((choice.get("token_ids"), text))
    return results


@wait_until_npu_memory_free(target_free_percentage=0.5)
def test_minicpm3_offline_online(config_filename, tp_size):
    config = _load_minicpm3_config(config_filename)
    tp_size = int(tp_size)
    _skip_if_insufficient_npu(tp_size)

    sampling = config["generation"]
    max_model_len = config["serve"]["max_model_len"]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    prompts = _build_prompts(tokenizer, sampling["long_prompt_tokens"])
    long_token_counts = [len(tokenizer.encode(prompt)) for prompt in prompts[2:]]
    assert min(long_token_counts) >= sampling["long_prompt_tokens"]
    assert max(long_token_counts) < max_model_len

    offline_greedy, offline_sampled = _run_offline(config, prompts, tp_size)

    port = get_open_port()
    with RemoteOpenAIServer(
        MODEL_NAME,
        _server_args(config, tp_size, port),
        server_host="127.0.0.1",
        server_port=port,
        auto_port=False,
        seed=sampling["seed"],
    ) as server:
        online_greedy = _online_generate(server, prompts, config, temperature=0.0, seed=0)
        for index, ((offline_ids, offline_text), (online_ids, online_text)) in enumerate(
            zip(offline_greedy, online_greedy)
        ):
            assert online_text == offline_text, (
                f"Offline/online greedy output differs for prompt {index}: {offline_text!r} != {online_text!r}"
            )
            if online_ids is not None:
                assert online_ids == offline_ids, f"Offline/online greedy token IDs differ for prompt {index}"

        sampled_kwargs = {"temperature": sampling["temperature"], "seed": sampling["seed"]}
        online_sample_1 = _online_generate(server, prompts, config, **sampled_kwargs)
        online_sample_2 = _online_generate(server, prompts, config, **sampled_kwargs)
        for index, (first, second) in enumerate(zip(online_sample_1, online_sample_2)):
            assert first[1] == second[1], f"Seeded non-greedy output is not reproducible for prompt {index}"
            if first[0] is not None and second[0] is not None:
                assert first[0] == second[0], f"Seeded non-greedy token IDs differ for prompt {index}"
        assert len(online_sample_1) == len(offline_sampled) == len(prompts)
