from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

from vllm_ascend.sample.rejection_sampler import (
    AscendRejectionSampler,
    MAX_SPEC_LEN,
    apply_sampling_constraints,
    rejection_sample,
)
from vllm_ascend.sample.sampler import AscendSampler


@dataclass
class PreparedSpecSamplingInputs:
    metadata: SpecDecodeMetadata
    sampling_metadata: SamplingMetadata
    logits: torch.Tensor
    draft_probs: torch.Tensor | None = None
    prepared_top_k: int | None = None


class SpecSamplingNPUExecutor:
    """Execute the speculative sampling pipeline with prepared inputs.

    This first version intentionally mirrors the current real sampling path,
    but makes the pipeline boundary explicit so later stages can migrate more
    work out of ``model_runner_v1`` without re-deriving the input contract.
    """

    def __init__(
        self,
        sampler: AscendSampler,
        rejection_sampler: AscendRejectionSampler,
    ) -> None:
        self.sampler = sampler
        self.rejection_sampler = rejection_sampler

    @staticmethod
    def build_inputs(
        *,
        metadata: SpecDecodeMetadata,
        sampling_metadata: SamplingMetadata,
        logits: torch.Tensor,
        draft_probs: torch.Tensor | None = None,
        prepared_top_k: int | None = None,
    ) -> PreparedSpecSamplingInputs:
        return PreparedSpecSamplingInputs(
            metadata=metadata,
            sampling_metadata=sampling_metadata,
            logits=logits,
            draft_probs=draft_probs,
            prepared_top_k=prepared_top_k,
        )

    @classmethod
    def build_inputs_from_runtime(
        cls,
        *,
        metadata: SpecDecodeMetadata,
        sampling_metadata: SamplingMetadata,
        logits: torch.Tensor,
        top_k_cpu: torch.Tensor | None,
        enable_reduce_sample: bool,
        draft_probs: torch.Tensor | None = None,
    ) -> PreparedSpecSamplingInputs:
        prepared_top_k = None
        if top_k_cpu is not None and enable_reduce_sample:
            valid_top_k = top_k_cpu[top_k_cpu < logits.shape[1]]
            if len(valid_top_k) > 0:
                prepared_top_k = int(valid_top_k.max())
        return cls.build_inputs(
            metadata=metadata,
            sampling_metadata=sampling_metadata,
            logits=logits,
            draft_probs=draft_probs,
            prepared_top_k=prepared_top_k,
        )

    def execute(self, inputs: PreparedSpecSamplingInputs) -> SamplerOutput:
        metadata = inputs.metadata
        sampling_metadata = inputs.sampling_metadata
        logits = inputs.logits

        assert metadata.max_spec_len <= MAX_SPEC_LEN
        bonus_logits_indices = metadata.bonus_logits_indices
        target_logits_indices = metadata.target_logits_indices

        bonus_logits = logits[bonus_logits_indices]
        bonus_sampler_output = self.sampler(
            logits=bonus_logits,
            sampling_metadata=replace(
                sampling_metadata,
                max_num_logprobs=-1,
            ),
            predict_bonus_token=True,
            logprobs_mode_override=(
                "processed_logits"
                if self.rejection_sampler.is_processed_logprobs_mode
                else "raw_logits"
            ),
        )
        bonus_token_ids = bonus_sampler_output.sampled_token_ids

        raw_target_logits = logits[target_logits_indices].to(torch.float32)
        target_logits = raw_target_logits
        if not self.rejection_sampler.is_processed_logprobs_mode:
            target_logits = target_logits.clone()

        target_logits = self.rejection_sampler.apply_logits_processors(
            target_logits,
            sampling_metadata,
            metadata,
        )
        target_logits = apply_sampling_constraints(
            target_logits,
            metadata.cu_num_draft_tokens,
            sampling_metadata,
            inputs.prepared_top_k,
        )

        output_token_ids = rejection_sample(
            metadata.draft_token_ids,
            metadata.num_draft_tokens,
            metadata.max_spec_len,
            metadata.cu_num_draft_tokens,
            inputs.draft_probs,
            target_logits,
            bonus_token_ids,
            sampling_metadata,
        )

        logprobs_tensors = None
        if sampling_metadata.max_num_logprobs is not None:
            logprobs_tensors = self.rejection_sampler.build_logprobs_tensors_from_prepared_inputs(
                sampling_metadata.max_num_logprobs,
                metadata,
                logits,
                target_logits if self.rejection_sampler.is_processed_logprobs_mode else raw_target_logits,
                bonus_sampler_output.logprobs_tensors.logprobs,
                output_token_ids,
            )

        return SamplerOutput(
            sampled_token_ids=output_token_ids,
            logprobs_tensors=logprobs_tensors,
        )
