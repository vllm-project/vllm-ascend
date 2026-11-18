import asyncio
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from tools import aisbench


@pytest.mark.parametrize("returncode", [0, 1])
def test_pd_waits_for_actual_aisbench_exit(tmp_path, monkeypatch, returncode):
    monkeypatch.chdir(tmp_path)
    runner = aisbench.AisbenchRunner.__new__(aisbench.AisbenchRunner)
    runner.__dict__.update(
        dataset_conf="gsm8k/perf",
        request_conf="vllm_api_stream_chat",
        task_type="performance",
        num_prompts=2,
        pd_range=(0, 2),
        pd_cache_client={"stage": "decode"},
        benchmark_artifact_name="decode_16k",
        profile_marker=None,
        result_line=None,
    )

    def run(args, stdout, **kwargs):
        assert args[args.index("--num-warmups") + 1] == "0"
        stdout.write("Performance Result files located in /tmp/result.\n")
        return SimpleNamespace(returncode=returncode)

    monkeypatch.setattr(aisbench.subprocess, "run", run)
    if returncode:
        with pytest.raises(RuntimeError, match="code 1"):
            runner._run_aisbench_task()
    else:
        runner._run_aisbench_task()
        runner._wait_for_task()
        assert runner.result_line == "Performance Result files located in /tmp/result."


@pytest.fixture
def client_cls(monkeypatch):
    class Chat:
        def __init__(self, retry):
            self.headers = {}
            self.logger = Mock()
            self.retry = retry

        async def get_request_body(self):
            return {}

        async def stream_infer(self, request_body, output):
            output.success = True

        async def _parse_usage(self, response, output):
            output.input_tokens = response["usage"]["prompt_tokens"]
            output.output_tokens = response["usage"]["completion_tokens"]

    models = ModuleType("ais_bench.benchmark.models")
    models.VLLMCustomAPIChat = Chat
    monkeypatch.setitem(sys.modules, models.__name__, models)
    spec = importlib.util.spec_from_file_location("pd_chat_test", Path(aisbench.__file__).with_name("aisbench_pd.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PDCacheChat


@pytest.mark.parametrize(
    ("stage", "cached", "accepted"),
    [
        ("prefill", 0, True),
        ("prefill", 16, False),
        ("decode", 1008, True),
        ("decode", 16, False),
        ("decode", None, False),
    ],
)
def test_request_cache_evidence(client_cls, stage, cached, accepted):
    client = client_cls(stage=stage, replay_tail_tokens=32)
    assert client.retry == 1
    if stage == "prefill":
        assert (
            asyncio.run(client.get_request_body())["cache_salt"] != asyncio.run(client.get_request_body())["cache_salt"]
        )
    output = SimpleNamespace(extra_details_data={})
    usage = {"prompt_tokens": 1024, "completion_tokens": 1, "prompt_tokens_details": {"cached_tokens": cached}}
    asyncio.run(client._parse_usage({"usage": usage}, output))
    if accepted:
        asyncio.run(client.stream_infer({"max_tokens": 1}, output))
    else:
        with pytest.raises(ValueError):
            asyncio.run(client.stream_infer({"max_tokens": 1}, output))


def test_real_kv_windows_cover_every_row_and_prefill_replica(monkeypatch):
    metrics = "".join(
        f'vllm:cache_config_info{{engine="{rank}",block_size="16",mamba_block_size="32"}} 1\n' for rank in range(2)
    )
    monkeypatch.setattr(aisbench.requests, "get", Mock(return_value=Mock(text=metrics)))
    runs = []

    class Runner:
        def __init__(self, model, port, case, host):
            client = case["_pd_cache_client"]
            self.result = (port, case["_pd_range"], client["stage"], client.get("prefill_rank"))
            assert case["num_prompts"] == case["_pd_range"][1] - case["_pd_range"][0]
            assert case["max_out_len"] == (1 if client["stage"] == "warmup" else 1024)
            runs.append(self.result)

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

    monkeypatch.setattr(aisbench, "AisbenchRunner", Runner)
    monkeypatch.setattr(aisbench, "aggregate_decode_windows", list)
    case = {
        "pd_phase": "decode",
        "case_type": "performance",
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "gsm8k/gsm8k_gen_0_shot_cot_str_perf",
        "num_prompts": 5,
        "batch_size": 2,
        "max_out_len": 1024,
    }
    result = aisbench._run_pd_performance("test-model", 2000, "localhost", case, [1000])
    expected = []
    for indices in [(0, 2), (2, 4), (4, 5)]:
        expected.extend([(1000, indices, "warmup", 0), (1000, indices, "warmup", 1), (2000, indices, "decode", None)])
    assert runs == expected
    assert result == expected[2::3]
