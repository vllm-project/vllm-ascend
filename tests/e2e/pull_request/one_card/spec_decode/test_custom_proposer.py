from __future__ import annotations

import torch
from vllm import SamplingParams
from vllm.config import VllmConfig

from tests.e2e.conftest import VllmRunner

MODEL_NAME = "Qwen/Qwen3-0.6B"
NUM_SPECULATIVE_TOKENS = 3


class RepeatLastTokenProposer:
    """Minimal CPU proposer used to exercise the custom-class interface."""

    def __init__(self, vllm_config: VllmConfig):
        assert vllm_config.speculative_config is not None
        self.num_speculative_tokens = vllm_config.speculative_config.num_speculative_tokens

    def propose(
        self,
        sampled_token_ids: list[list[int]],
        num_tokens_no_spec: torch.Tensor,
        token_ids_cpu: torch.Tensor,
        slot_mappings: torch.Tensor | None = None,
    ) -> list[list[int]]:
        del num_tokens_no_spec, token_ids_cpu, slot_mappings
        return [
            [sampled_ids[-1]] * self.num_speculative_tokens if sampled_ids else [] for sampled_ids in sampled_token_ids
        ]

    def dummy_run(self, *args, **kwargs):
        raise AssertionError("custom proposer dummy_run must not be called")


def test_custom_proposer_single_card_npu():
    prompts = [
        "The capital of France is",
        "Repeat the word hello several times:",
    ]
    sampling_params = SamplingParams(temperature=0, max_tokens=16)

    with VllmRunner(
        MODEL_NAME,
        speculative_config={
            "method": "custom_class",
            "model": f"{__name__}.RepeatLastTokenProposer",
            "num_speculative_tokens": NUM_SPECULATIVE_TOKENS,
        },
        max_model_len=1024,
        enforce_eager=True,
        disable_log_stats=False,
    ) as runner:
        outputs = runner.model.generate(prompts, sampling_params)
        runner.model.generate(prompts[:1], sampling_params)
        metrics = runner.model.get_metrics()

    assert all(output.outputs[0].token_ids for output in outputs)
    num_draft_tokens = sum(metric.value for metric in metrics if metric.name == "vllm:spec_decode_num_draft_tokens")
    assert num_draft_tokens > 0
