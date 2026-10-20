# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

"""Regress deferred PP Mamba state copies with a real Qwen3.8-27B model."""

import os
from typing import Any
from unittest.mock import patch

import pytest
import torch
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free

MODEL = os.environ.get("QWEN38_DENSE_MODEL", "Qwen/Qwen3.8-27B")
NUM_REQUESTS = 8
MAX_MODEL_LEN = 4096


class MambaAlignWorker:
    model_runner: Any

    def install_align_probe(self):
        runner = self.model_runner
        assert runner.cache_config.mamba_cache_mode == "align"
        if runner.is_last_pp_rank:
            return None
        state = runner.model_state
        original = state.postprocess_state
        # Read-only device-side checks; no table writes or per-step D2H.
        # Only read these flags after generation finishes.
        self.align_probe = torch.zeros(2, dtype=torch.bool, device=runner.device)
        tables = runner.block_tables
        columns = [torch.arange(table.shape[1], device=runner.device) for table in tables.input_block_tables]

        def postprocess(idx_mapping, num_sampled, num_computed_tokens=None):
            if num_computed_tokens is not None and state._mamba_ctx is not None:
                indices = idx_mapping.clamp_min(0).long()
                block_size = state._mamba_spec.block_size
                computed = num_computed_tokens[indices]
                aligned = computed // block_size * block_size
                running = computed - num_sampled + 1
                # Match the align kernel's nontrivial-copy condition. A stale
                # row on a step that copies no state is not this regression.
                copies = (
                    (idx_mapping >= 0)
                    & (aligned >= running)
                    & ((state._mamba_state_idx_gpu[indices] != aligned // block_size - 1) | (aligned != running))
                )
                self.align_probe[0].logical_or_(copies.any())
                for group_id in state._mamba_group_ids:
                    expected = tables.block_tables[group_id].gpu[indices]
                    actual = tables.input_block_tables[group_id][: len(indices)]
                    valid = copies[:, None] & (
                        columns[group_id][None, :] < tables.num_blocks.gpu[group_id, indices, None]
                    )
                    self.align_probe[1].logical_or_(((actual != expected) & valid).any())
            return original(idx_mapping, num_sampled, num_computed_tokens)

        state.postprocess_state = postprocess
        return runner.cache_config.block_size

    def read_align_probe(self):
        if self.model_runner.is_last_pp_rank:
            return None
        result = self.align_probe.tolist()
        self.align_probe.zero_()
        return result


@pytest.mark.e2e_model("Qwen/Qwen3.8-27B")
@pytest.mark.e2e_coverage(
    arch="dense",
    feature="mtp,aclgraph,prefix_caching,chunked_prefill,mixed_lengths",
    parallel="PP,TP",
    deploy="pd_mix",
    hardware="A3",
    quantization="BF16",
    graph_mode="full_decode_only",
)
@patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "1", "VLLM_WORKER_MULTIPROC_METHOD": "spawn"})
@wait_until_npu_memory_free()
def test_qwen38_pp_mtp_prefix_cache():
    with VllmRunner(
        MODEL,
        dtype="bfloat16",
        tensor_parallel_size=2,
        pipeline_parallel_size=2,
        distributed_executor_backend="mp",
        worker_extension_cls=f"{__name__}.MambaAlignWorker",
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=4,
        max_num_batched_tokens=8192,
        gpu_memory_utilization=0.6,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        async_scheduling=True,
        seed=1024,
        speculative_config={"method": "mtp", "num_speculative_tokens": 3, "enforce_eager": True},
        compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY"},
    ) as runner:
        block_sizes = runner.model.collective_rpc("install_align_probe")
        block_size = next(size for size in block_sizes if size is not None)
        tokenizer = runner.model.get_tokenizer()
        prompts, params = [], []
        for index in range(NUM_REQUESTS):
            tokens = tokenizer.encode(f"Request {index}: continue counting 1, 2, 3, 4. ", add_special_tokens=False)
            prompt_len = block_size - 16 - index
            output_len = 64 + 8 * index
            assert prompt_len > 0 and prompt_len + output_len < MAX_MODEL_LEN
            prompts.append({"prompt_token_ids": (tokens * (prompt_len // len(tokens) + 1))[:prompt_len]})
            # Fixed, unequal lengths cross a state boundary and retire request
            # slots at different steps. No dependency on a math problem's CoT.
            params.append(SamplingParams(temperature=0, max_tokens=output_len, ignore_eos=True, detokenize=False))

        for _ in range(2):
            outputs = runner.model.generate(prompts, params, use_tqdm=False)
            assert len(outputs) == NUM_REQUESTS
            for output, sampling in zip(outputs, params):
                assert output.finished
                assert len(output.outputs[0].token_ids) == sampling.max_tokens
            probes = [probe for probe in runner.model.collective_rpc("read_align_probe") if probe is not None]
            assert len(probes) == 2, "Both TP workers on the first PP stage must check the deferred copies"
            for probe in probes:
                copied, mismatched = probe
                assert copied, "No deferred Mamba boundary copy was exercised"
                assert not mismatched, "Deferred PP state copy used another batch's block-table rows"

        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": "What is 4242 + 1717? Reply with only the integer."}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        answer = runner.model.generate([prompt], SamplingParams(temperature=0, max_tokens=32), use_tqdm=False)
        assert answer[0].outputs[0].text.strip() == "5959", answer
