# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Exercise dynamic target verification widths with real MTP ACL graphs."""

import math
import os
from unittest.mock import patch

import pytest
from vllm import SamplingParams
from vllm.config.compilation import CUDAGraphMode

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free


def _record_graph_dispatch(worker):
    manager = worker.model_runner.cudagraph_manager
    original_dispatch = manager.dispatch
    worker.dynamic_sd_graph_shapes = set()

    def dispatch(num_reqs, num_tokens, uniform_token_count, *args, **kwargs):
        desc = original_dispatch(num_reqs, num_tokens, uniform_token_count, *args, **kwargs)
        if desc.cg_mode == CUDAGraphMode.FULL:
            assert desc.num_tokens == desc.num_reqs * desc.uniform_token_count
            worker.dynamic_sd_graph_shapes.add((num_reqs, desc.uniform_token_count))
        return desc

    manager.dispatch = dispatch


def _read_graph_shapes(worker):
    return sorted(worker.dynamic_sd_graph_shapes)


def _force_transition_drafts(worker):
    """Keep real MTP forward, but guarantee acceptance to stress a K decrease."""
    import torch

    runner = worker.model_runner
    original_propose = runner.speculator.propose
    original_prepare = runner.model_state.prepare_attn
    worker.dynamic_sd_transitions = []

    def propose(input_batch, *args, **kwargs):
        draft = original_propose(input_batch, *args, **kwargs)
        tokens = [100 + int(req_id.split("-")[1]) for req_id in input_batch.req_ids]
        return torch.tensor(tokens, device=draft.device, dtype=draft.dtype)[:, None].expand_as(draft).contiguous()

    def prepare(input_batch, *args, **kwargs):
        metadata = original_prepare(input_batch, *args, **kwargs)
        for meta in metadata.values():
            indices = getattr(meta, "spec_state_indices_tensor", None)
            if indices is not None:
                # Diagnostic synchronization is confined to this offline test.
                width = int(meta.spec_decode_metadata.actual_seq_lengths.max().cpu())
                accepted = int(meta.num_accepted_tokens.max().cpu())
                assert indices.shape[1] >= accepted
                worker.dynamic_sd_transitions.append((accepted, width))
                break
        return metadata

    runner.speculator.propose = propose
    runner.model_state.prepare_attn = prepare


def _read_transitions(worker):
    return worker.dynamic_sd_transitions


def _exercise_live_k_decrease(model):
    model.collective_rpc(_force_transition_drafts)
    engine = model.llm_engine

    def add(index):
        engine.add_request(
            f"dynamic-{index}",
            {"prompt_token_ids": [100 + index] * 32},
            SamplingParams(temperature=0, max_tokens=96, ignore_eos=True, allowed_token_ids=[100 + index], logprobs=1),
        )

    add(0)
    add(1)
    step = 0
    finished = set()
    while engine.has_unfinished_requests():
        for output in engine.step():
            if output.finished:
                index = int(output.request_id.split("-")[1])
                completion = output.outputs[0]
                assert completion.token_ids == [100 + index] * 96
                assert all(math.isfinite(p[100 + index].logprob) for p in completion.logprobs)
                finished.add(index)
        step += 1
        if step == 8:
            for index in range(2, 8):
                add(index)
    assert finished == set(range(8))
    for transitions in model.collective_rpc(_read_transitions):
        assert any(accepted > width for accepted, width in transitions)
        assert any(width == 1 for _, width in transitions)


@pytest.mark.parametrize("model_name,quantization", [("Qwen/Qwen3.6-27B", None)])
@patch.dict(
    os.environ,
    {
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        # Only trusted test-local callables are sent to workers by collective_rpc.
        "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
    },
)
@wait_until_npu_memory_free()
def test_dynamic_mtp_full_decode_graphs(model_name, quantization):
    with VllmRunner(
        model_name,
        quantization=quantization,
        tensor_parallel_size=2,
        language_model_only=True,
        max_model_len=1024,
        max_num_batched_tokens=1024,
        max_num_seqs=8,
        gpu_memory_utilization=0.75,
        enable_prefix_caching=False,
        async_scheduling=True,
        seed=42,
        logprobs_mode="raw_logprobs",
        speculative_config={
            "method": "mtp",
            "num_speculative_tokens": 3,
            "num_speculative_tokens_per_batch_size": [(1, 2, 3), (3, 4, 1), (5, 8, 0)],
        },
        compilation_config={
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [4, 8, 16, 32],
        },
    ) as runner:
        runner.model.collective_rpc(_record_graph_dispatch)
        # Revisit widths after switching away: all graphs share persistent
        # metadata buffers, whose rows must be refreshed at the active width.
        for batch_size in [2, 4, 8, 4, 2]:
            prompts = [{"prompt_token_ids": [100 + i] * 32} for i in range(batch_size)]
            params = [
                SamplingParams(
                    temperature=0,
                    max_tokens=32,
                    ignore_eos=True,
                    allowed_token_ids=[100 + i],
                    logprobs=1,
                )
                for i in range(batch_size)
            ]
            outputs = runner.model.generate(prompts, params, use_tqdm=False)
            for i, output in enumerate(outputs):
                completion = output.outputs[0]
                assert completion.token_ids == [100 + i] * 32
                assert all(math.isfinite(step[100 + i].logprob) for step in completion.logprobs)

        for shapes in runner.model.collective_rpc(_read_graph_shapes):
            assert {(2, 4), (4, 2), (8, 1)} <= set(shapes)
        _exercise_live_k_decrease(runner.model)
