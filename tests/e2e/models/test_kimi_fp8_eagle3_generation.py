# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Opt-in, eight-A5 generation regression for FP8 MLA with BF16 Eagle3.

Run explicitly with the existing full-model configuration::

    pytest -sv tests/e2e/models/test_kimi_fp8_eagle3_generation.py \
        --config tests/e2e/cases/models/configs/Kimi/KimiK26_w4a4c8_A5.yaml

This loads the full checkpoint twice, sequentially. It needs eight free A5
devices and the checkpoint paths declared in that configuration. Other model
configs skip this test; ordinary PR operator tests do not download this model.
The BF16 Eagle3 wide-input writer also requires companion PR #18897. Apply
both fixes before running this configuration; this test does not bypass or
skip a missing writer adapter, and does not establish standalone validation.
"""

import asyncio
import contextlib
import json
import time
from pathlib import Path

import pytest
import regex as re
import requests
import torch
import yaml
from prometheus_client.parser import text_string_to_metric_families
from transformers import AutoTokenizer
from vllm.utils.network_utils import get_open_port

from tests.e2e.conftest import RemoteOpenAIServer, wait_until_npu_memory_free

CAPTURE_SIZES = (4, 8, 16, 32)
FULL_DISPATCH_ROW = re.compile(r"\|\s*\d+\s*\|\s*(?:4|8|16|32)\s*\|\s*\d+\s*\|\s*FULL\s*\|\s*[1-9]\d*\s*\|")


class _LoggedServer(RemoteOpenAIServer):
    def __init__(self, model, args, env, log_file):
        self.log_file = log_file
        port = get_open_port()
        super().__init__(
            model,
            [*args, "--port", str(port)],
            env_dict=env,
            server_host="127.0.0.1",
            server_port=port,
            auto_port=False,
        )

    def _start_server(self, model, server_cmd, env_dict):
        with contextlib.redirect_stdout(self.log_file), contextlib.redirect_stderr(self.log_file):
            super()._start_server(model, server_cmd, env_dict)


def _replace_option(args, option, value):
    if option in args:
        index = args.index(option)
        del args[index : index + 2]
    args.extend((option, value))


def _load_case(config_filename):
    with Path(config_filename).open() as config_file:
        config = yaml.safe_load(config_file)
    cases = config.get("test_cases", [])
    case = next((case for case in cases if case["name"] == "KimiK26_A5_Single_Eagle3_mla_Step_01"), None)
    if case is None:
        pytest.skip("Select KimiK26_w4a4c8_A5.yaml with --config for this full-model regression")
    assert case is not None
    args = list(case["server_cmd"])
    extra_args = case["envs"]["server_cmd_extra"]
    draft_config = json.loads(extra_args[extra_args.index("--speculative-config") + 1])
    assert args[args.index("--kv-cache-dtype") + 1] == "fp8"
    assert args[args.index("--tensor-parallel-size") + 1] == "4"
    assert args[args.index("--data-parallel-size") + 1] == "2"
    assert "--enable-expert-parallel" in args
    port_index = args.index("--port")
    del args[port_index : port_index + 2]
    _replace_option(
        args,
        "--compilation-config",
        json.dumps(
            {
                "cudagraph_mode": "FULL_DECODE_ONLY",
                "cudagraph_capture_sizes": CAPTURE_SIZES,
            }
        ),
    )
    args.append("--cudagraph-metrics")
    env = {key: value for key, value in case["envs"].items() if key != "server_cmd_extra"}
    env["VLLM_LOG_STATS_INTERVAL"] = "1"
    return case["model"], args, env, draft_config


def _prompts(model):
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    filler_ids = tokenizer.encode("Unrelated background notes. " * 2048, add_special_tokens=False)[:4096]
    filler = tokenizer.decode(filler_ids)
    prompts = []
    for index in range(8):
        if index % 2:
            left, right = 17 + index, 29 + index
            expected = str(left + right)
            instruction = f"Calculate {left} + {right}. Output only the integer answer."
        else:
            expected = f"amber cobalt jade ivory plum cedar birch maple quartz coral {index}"
            instruction = f"Copy the following text exactly, without quotes or explanation: {expected}"
        prefix = f"Ignore these reference notes:\n{filler}\n\n" if index >= 4 else ""
        prompts.append((prefix + instruction, expected))
    return prompts


async def _generate_round(server, model, prompts, concurrency):
    semaphore = asyncio.Semaphore(concurrency)
    async with server.get_async_client() as client:

        async def generate(prompt, expected):
            async with semaphore:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=48,
                    seed=42,
                    extra_body={
                        "return_token_ids": True,
                        "chat_template_kwargs": {"thinking": False, "enable_thinking": False},
                    },
                )
            choice = response.choices[0]
            assert choice.message.content is not None
            assert choice.message.content.strip() == expected, choice.message.content
            token_ids = choice.model_dump().get("token_ids")
            assert token_ids, "Generation must return token IDs for baseline comparison"
            assert choice.finish_reason == "stop", "The answer was truncated"
            if len(prompt) > 4096:
                assert response.usage is not None and response.usage.prompt_tokens >= 4000
            return token_ids

        return await asyncio.gather(*(generate(*case) for case in prompts))


def _assert_speculation(server):
    response = requests.get(server.url_for("metrics"), timeout=30)
    response.raise_for_status()
    counters = {"vllm:spec_decode_num_drafts_total": 0.0, "vllm:spec_decode_num_accepted_tokens_total": 0.0}
    accepted_third = 0.0
    for family in text_string_to_metric_families(response.text):
        for sample in family.samples:
            if sample.name in counters:
                counters[sample.name] += sample.value
            if (
                sample.name == "vllm:spec_decode_num_accepted_tokens_per_pos_total"
                and sample.labels.get("position") == "2"
            ):
                accepted_third += sample.value
    assert all(value > 0 for value in counters.values()), counters
    assert accepted_third > 0, "The third Eagle3 draft position was never accepted"


@wait_until_npu_memory_free(target_free_percentage=0.8)
def _run_generation(model, args, env, prompts, log_path, baseline=None):
    with log_path.open("w") as log_file, _LoggedServer(model, args, env, log_file) as server:
        outputs: list[list[list[int]]] = []
        for _ in range(2):
            for concurrency in (1, 2, 8):
                result = asyncio.run(_generate_round(server, model, prompts, concurrency))
                expected = baseline if baseline is not None else (outputs[0] if outputs else result)
                assert result == expected, f"Token IDs differ from baseline at concurrency={concurrency}"
                outputs.append(result)
        # The API server publishes accumulated engine statistics periodically.
        # Match observed runtime rows, not the configured FULL_DECODE_ONLY mode.
        deadline = time.monotonic() + 5
        while not FULL_DISPATCH_ROW.search(log_path.read_text()) and time.monotonic() < deadline:
            time.sleep(0.1)
        assert FULL_DISPATCH_ROW.search(log_path.read_text()), (
            f"No actual target FULL dispatch was recorded; inspect {log_path}"
        )
        if baseline is not None:
            _assert_speculation(server)
    return outputs[0]


def test_kimi_fp8_eagle3_generation(config_filename, tmp_path):
    model, args, env, draft_config = _load_case(config_filename)
    if not torch.npu.is_available() or torch.npu.device_count() < 8:
        pytest.skip("Eight A5 devices are required")
    if "Ascend950" not in torch.npu.get_device_name():
        pytest.skip("This regression targets A5 FP8 MLA")
    assert Path(model).is_dir(), f"Install the target checkpoint declared in {config_filename}: {model}"
    assert Path(draft_config["model"]).is_dir(), f"Install the draft checkpoint: {draft_config['model']}"
    prompts = _prompts(model)
    baseline = _run_generation(model, args, env, prompts, tmp_path / "baseline.log")
    draft_config.update(num_speculative_tokens=3, kv_cache_dtype="bfloat16")
    spec_args = [*args, "--speculative-config", json.dumps(draft_config)]
    _run_generation(model, spec_args, env, prompts, tmp_path / "eagle3.log", baseline)
