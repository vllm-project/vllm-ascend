# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared full-weight V4 A5 configuration and module-scoped reference fixtures."""

from pathlib import Path

import pytest
from vllm.tokenizers import get_tokenizer
from vllm.transformers_utils.utils import maybe_model_redirect

from tests.e2e.pull_request.deepseek_a5_utils import (
    LoggedServer,
    assert_dp_engines,
    assert_server_speculation,
    completions,
    configure_a5,
    engine_args,
    prompt,
)

MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
MTP_MODEL = "deepseek-ai/DeepSeek-V4-Flash"
PROBE_CLASS = "tests.e2e.pull_request.deepseek_a5_utils.A5WorkerProbe"


def _args() -> dict:
    args = engine_args()
    args.update(
        tensor_parallel_size=1,
        data_parallel_size=4,
        data_parallel_size_local=4,
        quantization="deepseek_v4_fp8",
        tokenizer_mode="deepseek_v4",
        block_size=32,
        kv_cache_dtype="fp8",
        attention_config={"indexer_kv_dtype": "fp8"},
        enable_prefix_caching=False,
        safetensors_load_strategy="lazy",
        model_loader_extra_config={"enable_multithread_load": True, "num_threads": 16},
    )
    return args


def _speculation(method: str = "dspark") -> dict:
    config = {"method": method, "num_speculative_tokens": 3 if method == "mtp" else 5, "enforce_eager": True}
    if method == "dspark":
        config["draft_sample_method"] = "greedy"
    return config


@pytest.fixture(scope="module", autouse=True)
def a5_runtime():
    with pytest.MonkeyPatch.context() as monkeypatch:
        configure_a5(monkeypatch)
        monkeypatch.setenv("VLLM_SERVER_DEV_MODE", "1")
        yield


@pytest.fixture(scope="module", name="v4_model")
def v4_model_fixture(a5_runtime):
    return _model_inputs(MODEL)


@pytest.fixture(scope="module", name="v4_mtp_model")
def v4_mtp_model_fixture(a5_runtime):
    return _model_inputs(MTP_MODEL)


def _model_inputs(model_name: str):
    model = maybe_model_redirect(model_name)
    tokenizer = get_tokenizer(model, tokenizer_mode="deepseek_v4")
    prompts = [
        prompt(length, salt=i * 137, tokenizer=tokenizer)["prompt_token_ids"]
        for i, length in enumerate((127, 128, 129, 769))
    ]
    return model, prompts, tokenizer


def _run_functional(model: str, args: dict, prompts: list, path: Path, *, dp_size: int | None = None) -> None:
    """Check short requests and real feature execution without a second server."""
    with LoggedServer(model, args, path) as server:
        server.collective_rpc("install_graph_probe")
        completions(server, prompts)
        assert_server_speculation(server, args["speculative_config"]["num_speculative_tokens"])
        if dp_size is not None:
            assert_dp_engines(server, dp_size, speculative=True)
        counts = server.collective_rpc("read_graph_probe")
        assert counts and any(int(count) > 0 for count in counts), counts
