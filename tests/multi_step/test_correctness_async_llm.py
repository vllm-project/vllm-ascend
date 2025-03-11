# Test the AsyncLLMEngine with multi-step-decoding
from typing import List, Optional

import pytest

from tests.model_utils import check_logprobs_close
from tests.utils import (completions_with_server_args,
                         get_client_text_generations,
                         get_client_text_logprob_generations)

MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
]
NUM_SCHEDULER_STEPS = [8]  # Multi-step decoding steps
NUM_PROMPTS = [10]

DEFAULT_SERVER_ARGS: List[str] = [
    "--distributed-executor-backend",
    "ray",
    "--gpu-memory-utilization",
    "0.85",
    "--swap-space",
    "16",
]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize(("tp_size, pp_size"), [
    (1, 1),
])
@pytest.mark.parametrize("eager_mode", [True])
@pytest.mark.parametrize("num_scheduler_steps", NUM_SCHEDULER_STEPS)
@pytest.mark.parametrize("num_prompts", NUM_PROMPTS)
@pytest.mark.parametrize("num_logprobs", [5])
@pytest.mark.asyncio
async def test_multi_step(
    example_prompts,
    model: str,
    tp_size: int,
    pp_size: int,
    eager_mode: int,
    num_scheduler_steps: int,
    num_prompts: int,
    num_logprobs: Optional[int],
) -> None:
    """Test vLLM engine with multi-step scheduling in an OpenAI-protocol
    client/server environment.

    Set up an engine with single-step scheduling as a ground-truth reference.

    Send a completions API request to both engines with the same prompts.

    Validate:
    * Generated tokens match
    * Generated logprobs are all very close

    Args:
      example_prompts: test fixture providing example prompts
      model: model under test (same for single- and multi-step engines)
      tp_size: degree of tensor-parallelism
      pp_size: degree of pipeline-parallelism
      eager_mode
      num_scheduler_steps: for multi-step scheduling, GPU-side steps per
                           GPU -> CPU output transfer
      num_prompts: number of example prompts under test
      num_logprobs: corresponds to the `logprobs` argument to the OpenAI
                    completions endpoint; `None` -> no logprobs
    """

    prompts = example_prompts
    if len(prompts) < num_prompts:
        prompts = prompts * ((num_prompts // len(prompts)) + 1)
    prompts = prompts[:num_prompts]
    assert len(prompts) == num_prompts

    server_args = DEFAULT_SERVER_ARGS + ["--enforce-eager"]
    ms_server_args = DEFAULT_SERVER_ARGS + \
        ["--num-scheduler-steps", f"{num_scheduler_steps}"]

    if eager_mode:
        ms_server_args.append("--enforce-eager")

    distributed_args = [
        "--tensor-parallel-size",
        str(tp_size),
        "--pipeline-parallel-size",
        str(pp_size),
    ]

    # Spin up client/server & issue completion API requests.
    # Default `max_wait_seconds` is 240 but was empirically
    # was raised 5x to 1200 *just for this test* due to
    # observed timeouts in GHA CI
    ref_completions = await completions_with_server_args(
        prompts,
        model,
        server_args + distributed_args,
        num_logprobs,
        max_wait_seconds=5 * 240)
    test_completions = await completions_with_server_args(
        prompts,
        model,
        ms_server_args + distributed_args,
        num_logprobs,
        max_wait_seconds=5 * 240)

    # Assert multi-step scheduling produces identical tokens
    # to single-step scheduling.
    ref_generations = get_client_text_generations(ref_completions)
    test_generations = get_client_text_generations(test_completions)
    assert ref_generations == test_generations

    # Assert multi-step scheduling produces nearly-identical logprobs
    # to single-step scheduling.
    ref_text_logprobs = get_client_text_logprob_generations(ref_completions)
    test_text_logprobs = get_client_text_logprob_generations(test_completions)
    check_logprobs_close(
        outputs_0_lst=ref_text_logprobs,
        outputs_1_lst=test_text_logprobs,
        name_0="hf",
        name_1="vllm",
    )


@pytest.mark.parametrize(("tp_size, pp_size"), [
    (1, 1),
])
@pytest.mark.asyncio
async def test_multi_step_pp_smoke(
    tp_size: int,
    pp_size: int,
) -> None:
    """
    Smoke test for the vLLM engine with multi-step scheduling in an
    OpenAI-protocol client/server environment.

    This tests compares the outputs between multi-step scheduling and
    single-step scheduling. Notably, this test lets the engines generate
    more tokens (default is 5) and test for an exact match over all the
    tokens.

    Args:
      tp_size: degree of tensor-parallelism
      pp_size: degree of pipeline-parallelism
      eager_mode
    """

    model = "Qwen/Qwen2.5-0.5B-Instruct"
    num_scheduler_steps = 8
    max_num_seqs = 3

    # Prompt from the ShareGPT dataset
    prompts = [
        "in the jtbd context whats a push?",  # codespell:ignore
        "in the jtbd context whats a push?",  # codespell:ignore
        "in the jtbd context whats a push?",  # codespell:ignore
        "in the jtbd context whats a push?",  # codespell:ignore
    ]
    # Use varying max_tokens to introduce scheduling randomness.
    max_tokens = [10 * i for i in range(1, len(prompts) + 1)]
    assert len(prompts) == len(max_tokens)

    test_args = [
        "--tensor-parallel-size",
        str(tp_size), "--pipeline-parallel-size",
        str(pp_size), "--max-num-seqs",
        str(max_num_seqs)
    ]

    server_args = DEFAULT_SERVER_ARGS + test_args
    ms_server_args = DEFAULT_SERVER_ARGS + \
       ["--num-scheduler-steps", f"{num_scheduler_steps}"] + \
       test_args

    # Spin up client/server & issue completion API requests.
    # Default `max_wait_seconds` is 240 but was empirically
    # was raised 3x to 720 *just for this test* due to
    # observed timeouts in GHA CI
    ref_completions = await completions_with_server_args(
        prompts=prompts,
        model_name=model,
        server_cli_args=server_args,
        num_logprobs=None,
        max_wait_seconds=5 * 240,
        max_tokens=max_tokens)

    test_completions = await completions_with_server_args(
        prompts=prompts,
        model_name=model,
        server_cli_args=ms_server_args,
        num_logprobs=None,
        max_wait_seconds=5 * 240,
        max_tokens=max_tokens)

    # Assert multi-step scheduling produces identical tokens
    # to single-step scheduling.
    ref_generations = get_client_text_generations(ref_completions)
    test_generations = get_client_text_generations(test_completions)

    assert ref_generations == test_generations
