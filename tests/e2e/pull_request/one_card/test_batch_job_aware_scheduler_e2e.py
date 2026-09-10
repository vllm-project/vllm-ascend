"""
End-to-end tests for BatchJobAwareScheduler.

This test module verifies the correctness of the BatchJobAwareScheduler
by comparing outputs with the default scheduler.
"""

from typing import Any

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free
from tests.e2e.model_utils import check_outputs_equal

MODEL = "Qwen/Qwen3-0.6B"
MAX_TOKENS = 4
PROMPTS = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

# Workload is 4 prompts x 4 tokens. Unpinned capture enumerates graphs up to
# min(max_num_seqs * (1+K), 512); default max_num_batched_tokens=8192 also
# makes torch.compile use range (1, 8192). Together those dominate runtime.
_RUNNER_KWARGS = {
    "max_model_len": 2048,
    "gpu_memory_utilization": 0.7,
    "max_num_seqs": 8,
    "max_num_batched_tokens": 256,
    "compilation_config": {"cudagraph_capture_sizes": [4, 8]},
}

_CUSTOM_BATCH_JOB_CONFIG = {
    "enabled": True,
    "max_jobs": 10,
    "reserve_margin_blocks": 4,
    "reserve_max_blocks": 12,
    "low_available_tokens_threshold": 2048,
    "short_decode_token_threshold": 32,
}


def _batch_job_additional_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "scheduler_config": {
            "batch_job_sched_config": config or {"enabled": True},
        },
    }


def _generate(*, additional_config: dict[str, Any] | None = None, **kwargs: Any) -> list:
    runner_kwargs = {**_RUNNER_KWARGS, **kwargs}
    if additional_config is not None:
        runner_kwargs["additional_config"] = additional_config
    with VllmRunner(MODEL, **runner_kwargs) as vllm_model:
        return vllm_model.generate_greedy(PROMPTS, MAX_TOKENS)


@wait_until_npu_memory_free()
def test_batch_job_aware_scheduler_matches_default() -> None:
    """Compare sync, async, and custom-config schedulers against one default engine."""
    default_output = _generate(async_scheduling=False)

    batch_job_output = _generate(
        additional_config=_batch_job_additional_config(),
        async_scheduling=False,
    )
    check_outputs_equal(
        outputs_0_lst=default_output,
        outputs_1_lst=batch_job_output,
        name_0="default_scheduler",
        name_1="batch_job_aware_scheduler",
    )

    async_output = _generate(
        additional_config=_batch_job_additional_config(),
        async_scheduling=True,
    )
    check_outputs_equal(
        outputs_0_lst=default_output,
        outputs_1_lst=async_output,
        name_0="default_scheduler",
        name_1="batch_job_aware_async_scheduler",
    )

    custom_output = _generate(
        additional_config=_batch_job_additional_config(_CUSTOM_BATCH_JOB_CONFIG),
        async_scheduling=False,
    )
    check_outputs_equal(
        outputs_0_lst=default_output,
        outputs_1_lst=custom_output,
        name_0="default_scheduler",
        name_1="batch_job_aware_scheduler_custom_config",
    )


@wait_until_npu_memory_free()
def test_batch_job_aware_scheduler_with_chunked_prefill() -> None:
    """Tiny max_num_batched_tokens forces chunked prefill across scheduling steps."""
    chunked_kwargs: dict[str, Any] = {
        "max_num_seqs": 16,
        "max_num_batched_tokens": 16,
        "enable_chunked_prefill": True,
        "async_scheduling": False,
        "compilation_config": {"cudagraph_capture_sizes": [16]},
    }

    batch_job_output = _generate(
        additional_config=_batch_job_additional_config(),
        **chunked_kwargs,
    )
    default_output = _generate(**chunked_kwargs)

    check_outputs_equal(
        outputs_0_lst=default_output,
        outputs_1_lst=batch_job_output,
        name_0="default_scheduler_chunked_prefill",
        name_1="batch_job_aware_scheduler_chunked_prefill",
    )
