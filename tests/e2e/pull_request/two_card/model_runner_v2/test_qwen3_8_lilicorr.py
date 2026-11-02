# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

"""Real-weight LiLiCorr correctness and acceptance checks on two NPUs."""

import os
from typing import TypedDict
from unittest.mock import patch

import pytest
from modelscope import snapshot_download  # type: ignore[import-untyped]
from vllm import SamplingParams
from vllm.v1.metrics.reader import Counter, Vector

from tests.e2e.conftest import VllmRunner
from tests.e2e.pull_request.utils import PROMPTS_SHORT

TARGET_MODEL = "Qwen/Qwen3.8-27B"
DRAFT_MODEL = "niannianxu/Qwen3.8-27B-LiLiCorr-block8-epoch0"
NUM_SPECULATIVE_TOKENS = 7
MAX_TOKENS = 64
GREEDY_NEAR_TIE_LOGPROB_GAP = 0.25
GREEDY_TOP_LOGPROBS = 5


class _RunnerKwargs(TypedDict):
    tensor_parallel_size: int
    dtype: str
    enforce_eager: bool
    max_model_len: int
    max_num_seqs: int
    gpu_memory_utilization: float
    enable_prefix_caching: bool
    async_scheduling: bool
    generation_config: str
    seed: int
    compilation_config: dict


RUNNER_KWARGS: _RunnerKwargs = {
    "tensor_parallel_size": 2,
    "dtype": "bfloat16",
    "enforce_eager": False,
    "max_model_len": 4096,
    "max_num_seqs": len(PROMPTS_SHORT),
    "gpu_memory_utilization": 0.8,
    "enable_prefix_caching": False,
    "async_scheduling": False,
    "generation_config": "vllm",
    "seed": 0,
    "compilation_config": {"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [8, 16, 32, 64]},
}


@pytest.fixture(scope="module", autouse=True)
def mrv2_environment():
    with patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "1", "VLLM_WORKER_MULTIPROC_METHOD": "spawn"}):
        yield


@pytest.fixture(scope="module")
def draft_model_path() -> str:
    return snapshot_download(DRAFT_MODEL, local_files_only=False)


@pytest.fixture(scope="module")
def greedy_reference(mrv2_environment):
    with VllmRunner(TARGET_MODEL, **RUNNER_KWARGS) as runner:
        outputs = runner.model.generate(
            PROMPTS_SHORT,
            SamplingParams(
                temperature=0.0,
                seed=0,
                max_tokens=MAX_TOKENS,
                ignore_eos=True,
                logprobs=GREEDY_TOP_LOGPROBS,
            ),
            use_tqdm=False,
        )
    return [output.outputs[0] for output in outputs]


def _track_draft_graph_replays(worker):
    """Instrument only the test worker; detect silent eager fallback."""
    manager = worker.model_runner.speculator.query_cudagraph_manager
    manager._test_replay_count = 0
    manager._test_padded_replay_count = 0
    original = manager.run_fullgraph

    def counted(desc):
        manager._test_replay_count += 1
        if desc.num_reqs > worker.model_runner.speculator.input_batch.num_reqs:
            manager._test_padded_replay_count += 1
        return original(desc)

    manager.run_fullgraph = counted
    return bool(manager.graphs)


def _read_draft_graph_replays(worker):
    manager = worker.model_runner.speculator.query_cudagraph_manager
    return manager._test_replay_count, manager._test_padded_replay_count


@pytest.mark.e2e_model("Qwen/Qwen3.8-27B")
@pytest.mark.e2e_coverage(
    arch="dense",
    feature="spec_decode",
    parallel="TP",
    deploy="",
    hardware="A3",
    quantization="BF16",
    graph_mode="eager,full_decode_only",
)
@pytest.mark.parametrize(
    "draft_sample_method,temperature",
    [("greedy", 0.0), ("probabilistic", 0.0), ("probabilistic", 0.8)],
)
@pytest.mark.parametrize("draft_enforce_eager", [True, False])
def test_qwen38_lilicorr_tp2(
    draft_sample_method, temperature, draft_enforce_eager, greedy_reference, draft_model_path, record_property
):
    # A block of eight contains one anchor and at most seven draft tokens.
    speculative_config = {
        "method": "dflash",
        "model": draft_model_path,
        "num_speculative_tokens": NUM_SPECULATIVE_TOKENS,
        "draft_tensor_parallel_size": 2,
        "enforce_eager": draft_enforce_eager,
        "draft_sample_method": draft_sample_method,
    }
    with VllmRunner(
        TARGET_MODEL, **RUNNER_KWARGS, disable_log_stats=False, speculative_config=speculative_config
    ) as runner:
        config = runner.model.llm_engine.vllm_config.speculative_config
        assert "LiLiCorrDraftModel" in config.draft_model_config.architectures
        captured = runner.model.collective_rpc(_track_draft_graph_replays)
        if not draft_enforce_eager:
            assert all(captured), "LiLiCorr must capture draft graphs on every TP rank"
        outputs = runner.model.generate(
            PROMPTS_SHORT,
            SamplingParams(
                temperature=temperature,
                seed=0,
                max_tokens=MAX_TOKENS,
                ignore_eos=True,
                logprobs=GREEDY_TOP_LOGPROBS if temperature == 0.0 else None,
            ),
            use_tqdm=False,
        )
        # A second call reuses request-state slots. Three live requests require
        # padding to the four-request capture bucket (32 draft query tokens).
        runner.model.generate(
            PROMPTS_SHORT[:3],
            SamplingParams(temperature=temperature, seed=1, max_tokens=16, ignore_eos=True),
            use_tqdm=False,
        )
        replay_counts = runner.model.collective_rpc(_read_draft_graph_replays)
        if draft_enforce_eager:
            assert all(count == 0 for count, _ in replay_counts)
        else:
            assert all(count > 0 and padded > 0 for count, padded in replay_counts)
        record_property("draft_graph_replays", replay_counts)
        metrics = runner.model.get_metrics()

    tokens = [list(output.outputs[0].token_ids) for output in outputs]
    assert len(tokens) == len(PROMPTS_SHORT)
    assert all(len(ids) == MAX_TOKENS for ids in tokens)
    if temperature == 0.0:
        for request_idx, (actual_ids, baseline) in enumerate(zip(tokens, greedy_reference, strict=True)):
            baseline_ids = list(baseline.token_ids)
            assert len(baseline_ids) == MAX_TOKENS
            first_diff = next(
                (i for i, (actual, expected) in enumerate(zip(actual_ids, baseline_ids)) if actual != expected), None
            )
            if first_diff is None:
                assert actual_ids == baseline_ids
                continue

            # Different verify batch shapes can change a BF16 argmax near a
            # tie. Check the first divergence while both histories still
            # agree; later tokens have different autoregressive contexts.
            actual = outputs[request_idx].outputs[0]
            for side, completion, selected, alternative in (
                ("baseline", baseline, baseline_ids[first_diff], actual_ids[first_diff]),
                ("speculative", actual, actual_ids[first_diff], baseline_ids[first_diff]),
            ):
                assert completion.logprobs is not None
                assert len(completion.logprobs) == MAX_TOKENS
                scores = completion.logprobs[first_diff]
                assert selected in scores
                assert alternative in scores, (
                    f"request {request_idx}, token {first_diff}, {side}: "
                    f"alternative token {alternative} is outside top {GREEDY_TOP_LOGPROBS}"
                )
                gap = scores[selected].logprob - scores[alternative].logprob
                assert -1e-5 <= gap <= GREEDY_NEAR_TIE_LOGPROB_GAP, (
                    f"request {request_idx}, token {first_diff}, {side}: logprob gap {gap:.4f} "
                    f"exceeds near-tie bound {GREEDY_NEAR_TIE_LOGPROB_GAP}"
                )

    num_drafts = num_accepted = 0
    accepted_per_pos = [0] * NUM_SPECULATIVE_TOKENS
    for metric in metrics:
        if metric.name == "vllm:spec_decode_num_drafts":
            assert isinstance(metric, Counter)
            num_drafts += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens":
            assert isinstance(metric, Counter)
            num_accepted += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            assert isinstance(metric, Vector)
            assert len(metric.values) == NUM_SPECULATIVE_TOKENS
            accepted_per_pos = [a + b for a, b in zip(accepted_per_pos, metric.values)]

    assert num_drafts > 0, "LiLiCorr must execute real draft steps"
    assert 0 < num_accepted <= NUM_SPECULATIVE_TOKENS * num_drafts
    assert sum(accepted_per_pos) == num_accepted
    assert all(0 <= count <= num_drafts for count in accepted_per_pos)
    record_property("acceptance_length", 1 + num_accepted / num_drafts)
    record_property("accepted_per_position", accepted_per_pos)
