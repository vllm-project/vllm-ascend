from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from transformers import AutoTokenizer
from vllm import SamplingParams
from vllm.config import CompilationConfig, CompilationMode
from vllm.v1.metrics.reader import Counter, Vector

from tests.e2e.conftest import VllmRunner
from tests.e2e.pull_request.one_card.spec_decode.utils import BASELINES, DFLASH, calculate_acceptance_per_pos

if TYPE_CHECKING:
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


@pytest.mark.parametrize("method", DFLASH.keys())
@pytest.mark.parametrize("num_speculative_tokens", [8])
def test_dflash_acceptance(
    method: str,
    num_speculative_tokens: int,
):
    main_model_name = DFLASH[method]["main"]
    spec_model_name = DFLASH[method]["spec"]

    tokenizer = AutoTokenizer.from_pretrained(
        main_model_name,
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(
        temperature=0,
        ignore_eos=False,
        max_tokens=256,
    )

    prompts = [{"role": "user", "content": "Hello, your name is"}]
    prompts = [
        tokenizer.apply_chat_template(
            [prompt],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for prompt in prompts
    ]

    speculative_config = {
        "method": "dflash",
        "model": spec_model_name,
        "num_speculative_tokens": num_speculative_tokens,
    }

    compilation_config = CompilationConfig(cudagraph_mode="FULL_DECODE_ONLY", cudagraph_capture_sizes=[9, 18])

    with VllmRunner(
        main_model_name,
        max_model_len=4096,
        disable_log_stats=False,
        tensor_parallel_size=1,
        max_num_seqs=256,
        distributed_executor_backend="mp",
        gpu_memory_utilization=0.8,
        speculative_config=speculative_config,
        compilation_config=compilation_config,
        enable_prefix_caching=False,
    ) as llm:
        outputs = llm.model.generate(prompts, sampling_params)
        metrics = llm.model.get_metrics()

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        output_tokens = output.outputs[0].token_ids
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        print(f"Output tokens: {output_tokens}")

    acceptance_per_pos = calculate_acceptance_per_pos(metrics, num_speculative_tokens, Counter, Vector)
    golden = BASELINES[method]

    match = all(abs(a - b) < 0.1 for a, b in zip(acceptance_per_pos, golden))
    if not match:
        print(f"acceptance_per_pos: {acceptance_per_pos}")
        print(f"golden: {golden}")

    assert match


class DFlashQueryBufferProbe:
    model_runner: NPUModelRunner

    def dflash_query_buffer_state(self):
        runner = self.model_runner
        runtime_mode, descriptor = runner.cudagraph_dispatcher.dispatch(4, uniform_decode=False)
        return {
            "drafter": type(runner.drafter).__name__,
            "use_graph": runner.drafter.use_cuda_graph,
            "runtime_mode": runtime_mode.name,
            "num_tokens": descriptor.num_tokens,
            "positions": runner.drafter.positions.numel(),
            "device": runner.drafter.positions.device.type,
        }


def test_dflash_padded_query_buffers():
    """Exercise an oversized graph bucket with one request and a 32-sequence limit."""
    sampling_params = SamplingParams(temperature=0, ignore_eos=True, max_tokens=4)
    compilation_config = CompilationConfig(
        mode=CompilationMode.VLLM_COMPILE,
        cudagraph_mode="PIECEWISE",
        cudagraph_capture_sizes=[256],
    )

    with VllmRunner(
        DFLASH["dflash"]["main"],
        dtype="bfloat16",
        max_model_len=256,
        max_num_seqs=32,
        max_num_batched_tokens=256,
        tensor_parallel_size=1,
        distributed_executor_backend="mp",
        gpu_memory_utilization=0.8,
        enable_prefix_caching=False,
        worker_extension_cls=f"{__name__}.DFlashQueryBufferProbe",
        compilation_config=compilation_config,
        speculative_config={
            "method": "dflash",
            "model": DFLASH["dflash"]["spec"],
            "num_speculative_tokens": 3,
            "draft_tensor_parallel_size": 1,
            "enforce_eager": False,
        },
    ) as llm:
        # The sole bucket pads even a single four-token draft query to 256.
        # The unpatched allocation is only 32 * (1 + 3) = 128 positions.
        assert llm.model.collective_rpc("dflash_query_buffer_state", timeout=30) == [
            {
                "drafter": "AscendDflashProposer",
                "use_graph": True,
                "runtime_mode": "PIECEWISE",
                "num_tokens": 256,
                "positions": 256,
                "device": "npu",
            }
        ]
        prompts = [{"prompt_token_ids": [42]}]
        outputs = llm.model.generate(prompts, sampling_params)

    assert len(outputs) == 1
    assert all(len(output.outputs[0].token_ids) == sampling_params.max_tokens for output in outputs)
