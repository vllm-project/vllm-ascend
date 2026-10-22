# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

"""Opt-in A3 TP8 greedy equivalence for Flash-Next with a DFlash2 draft.

Set QWEN38_DFLASH_TARGET_MODEL to the Ascend-quantized target and
QWEN38_DFLASH_DRAFT_MODEL to its matching DFlash2 checkpoint before running.
"""

import os
from typing import Any
from unittest.mock import patch

import pytest
import torch
from vllm import SamplingParams
from vllm.v1.metrics.reader import Counter, Metric

from tests.e2e.conftest import VllmRunner
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

TARGET_MODEL = os.environ.get("QWEN38_DFLASH_TARGET_MODEL")
DRAFT_MODEL = os.environ.get("QWEN38_DFLASH_DRAFT_MODEL")
NUM_SPECULATIVE_TOKENS = 3
MAX_TOKENS = 64
PROMPTS = (
    "Continue this sequence: 1, 2, 3, 4,",
    "Explain how a compiler optimizes a loop. " + "Consider memory access, computation and dependency order. " * 12,
    "Summarize the following observations. "
    + "The server receives a request, checks its cache, computes an answer and releases the request slot. " * 32,
)


class DFlashGraphWorker:
    model_runner: Any

    def install_dflash_graph_probe(self) -> int:
        manager = self.model_runner.speculator.query_cudagraph_manager
        self.dflash_graph_replays = 0
        original = manager.run_fullgraph

        def replay(desc):
            self.dflash_graph_replays += 1
            return original(desc)

        manager.run_fullgraph = replay
        return len(manager.graphs)

    def read_dflash_graph_replays(self) -> int:
        return self.dflash_graph_replays


def _draft_count(metrics: list[Metric]) -> int:
    count = 0
    for metric in metrics:
        if metric.name == "vllm:spec_decode_num_drafts":
            assert isinstance(metric, Counter)
            count += metric.value
    return count


def _generate_rounds(runner: VllmRunner, *, speculative: bool, draft_graph: bool = False) -> list[list[list[int]]]:
    # Three requests round up to a graph size of four; the next batch reuses
    # request slots with a different order and batch size.
    batches = (PROMPTS, (PROMPTS[2], PROMPTS[0]))
    sampling = SamplingParams(temperature=0, max_tokens=MAX_TOKENS, ignore_eos=True)
    rounds = []
    for prompts in batches:
        before = _draft_count(runner.model.get_metrics()) if speculative else 0
        before_replays = runner.model.collective_rpc("read_dflash_graph_replays") if speculative else []
        outputs = runner.model.generate(list(prompts), sampling, use_tqdm=False)
        assert len(outputs) == len(prompts)
        token_ids = [list(output.outputs[0].token_ids) for output in outputs]
        assert all(output.finished for output in outputs)
        assert all(len(tokens) == MAX_TOKENS for tokens in token_ids)
        if speculative:
            assert _draft_count(runner.model.get_metrics()) > before, "DFlash must draft tokens in each reused batch"
            after_replays = runner.model.collective_rpc("read_dflash_graph_replays")
            if draft_graph:
                assert all(after > before for before, after in zip(before_replays, after_replays)), (
                    "Every TP rank must replay the DFlash graph in each reused batch"
                )
            else:
                assert after_replays == before_replays
        rounds.append(token_ids)
    return rounds


@pytest.mark.skipif(
    not TARGET_MODEL or not DRAFT_MODEL,
    reason="Set QWEN38_DFLASH_TARGET_MODEL and QWEN38_DFLASH_DRAFT_MODEL to run this eight-NPU test",
)
@pytest.mark.parametrize(
    ("cudagraph_mode", "draft_enforce_eager"),
    [
        pytest.param(
            "NONE",
            True,
            id="eager",
            marks=pytest.mark.e2e_coverage(
                arch="moe",
                feature="spec_decode,chunked_prefill,mixed_lengths",
                parallel="TP,EP",
                deploy="pd_mix",
                hardware="A3",
                quantization="W8A8",
                graph_mode="eager",
            ),
        ),
        pytest.param(
            "FULL_DECODE_ONLY",
            True,
            id="full_decode_only_eager_draft",
            marks=pytest.mark.e2e_coverage(
                arch="moe",
                feature="spec_decode,aclgraph,chunked_prefill,mixed_lengths",
                parallel="TP,EP",
                deploy="pd_mix",
                hardware="A3",
                quantization="W8A8",
                graph_mode="full_decode_only",
            ),
        ),
        pytest.param(
            "FULL_DECODE_ONLY",
            None,
            id="full_decode_only_graph_draft",
            marks=pytest.mark.e2e_coverage(
                arch="moe",
                feature="spec_decode,aclgraph,chunked_prefill,mixed_lengths",
                parallel="TP,EP",
                deploy="pd_mix",
                hardware="A3",
                quantization="W8A8",
                graph_mode="full_decode_only",
            ),
        ),
    ],
)
@patch.dict(
    os.environ,
    {
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "HCCL_BUFFSIZE": "1024",
        "LCCL_DETERMINISTIC": "1",
        "HCCL_DETERMINISTIC": "true",
        "ATB_MATMUL_SHUFFLE_K_ENABLE": "0",
        "CLOSE_MATMUL_K_SHIFT": "1",
    },
)
def test_qwen38_dflash_matches_target_greedy_tp8(cudagraph_mode: str, draft_enforce_eager: bool | None) -> None:
    if get_ascend_device_type() != AscendDeviceType.A3 or torch.npu.device_count() < 8:
        pytest.skip("Flash-Next DFlash TP8 validation requires eight A3 NPUs")
    assert TARGET_MODEL is not None and DRAFT_MODEL is not None
    runner_kwargs = {
        "dtype": "bfloat16",
        "quantization": "ascend",
        "tensor_parallel_size": 8,
        "enable_expert_parallel": True,
        "distributed_executor_backend": "mp",
        "block_size": 128,
        "max_model_len": 2048,
        "max_num_batched_tokens": 256,
        "max_num_seqs": 4,
        "enable_chunked_prefill": True,
        "enable_prefix_caching": False,
        "async_scheduling": True,
        "disable_log_stats": False,
        "generation_config": "vllm",
        "seed": 42,
        "enforce_eager": cudagraph_mode == "NONE",
        "compilation_config": {
            "cudagraph_mode": cudagraph_mode,
            "cudagraph_capture_sizes": [4, 8, 16],
        },
        "additional_config": {
            "ascend_compilation_config": {"fuse_norm_quant": False},
        },
    }
    with VllmRunner(TARGET_MODEL, **runner_kwargs) as baseline_runner:
        expected = _generate_rounds(baseline_runner, speculative=False)

    speculative_config = {
        "method": "dflash",
        "model": DRAFT_MODEL,
        "num_speculative_tokens": NUM_SPECULATIVE_TOKENS,
    }
    if draft_enforce_eager is not None:
        speculative_config["enforce_eager"] = draft_enforce_eager
    draft_graph = cudagraph_mode == "FULL_DECODE_ONLY" and draft_enforce_eager is not True
    with VllmRunner(
        TARGET_MODEL,
        **runner_kwargs,
        worker_extension_cls=f"{__name__}.DFlashGraphWorker",
        speculative_config=speculative_config,
    ) as speculative_runner:
        captures = speculative_runner.model.collective_rpc("install_dflash_graph_probe")
        assert len(captures) == 8
        assert all(count > 0 if draft_graph else count == 0 for count in captures), captures
        actual = _generate_rounds(speculative_runner, speculative=True, draft_graph=draft_graph)

    for round_idx, (target_ids, dflash_ids) in enumerate(zip(expected, actual)):
        assert dflash_ids == target_ids, f"DFlash changed greedy target token IDs in batch round {round_idx}"
