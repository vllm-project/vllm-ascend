from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

from vllm_ascend.sample.rejection_sampler import (
    MAX_SPEC_LEN,
    PLACEHOLDER_TOKEN_ID,
    AscendRejectionSampler,
    apply_sampling_constraints,
    rejection_sample,
)
from vllm_ascend.sample.sampler import AscendSampler
from vllm_ascend.sample.spec_sampling_poc import write_spec_sampling_marker


@dataclass
class PreparedSpecSamplingInputs:
    metadata: SpecDecodeMetadata
    sampling_metadata: SamplingMetadata
    logits: torch.Tensor
    draft_probs: torch.Tensor | None = None
    prepared_top_k: int | None = None


@dataclass
class SpecSamplingExecutionResult:
    sampler_output: SamplerOutput
    num_output_tokens_per_req: torch.Tensor


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
    def _ensure_index_tensor(
        indices: torch.Tensor | list[int],
        device: torch.device,
    ) -> torch.Tensor:
        if not isinstance(indices, torch.Tensor):
            return torch.tensor(indices, device=device, dtype=torch.long)
        if indices.device != device:
            return indices.to(device=device, dtype=torch.long)
        if indices.dtype != torch.long:
            return indices.to(dtype=torch.long)
        return indices

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
        return self.execute_detailed(inputs).sampler_output

    def execute_detailed(self, inputs: PreparedSpecSamplingInputs) -> SpecSamplingExecutionResult:
        metadata = inputs.metadata
        sampling_metadata = inputs.sampling_metadata
        logits = inputs.logits

        assert metadata.max_spec_len <= MAX_SPEC_LEN
        bonus_logits_indices = self._ensure_index_tensor(
            metadata.bonus_logits_indices,
            logits.device,
        )
        target_logits_indices = self._ensure_index_tensor(
            metadata.target_logits_indices,
            logits.device,
        )

        bonus_logits = logits[bonus_logits_indices]
        bonus_sampler_output = self.sampler(
            logits=bonus_logits,
            sampling_metadata=replace(
                sampling_metadata,
                max_num_logprobs=-1,
            ),
            predict_bonus_token=True,
            logprobs_mode_override=(
                "processed_logits" if self.rejection_sampler.is_processed_logprobs_mode else "raw_logits"
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

        sampler_output = SamplerOutput(
            sampled_token_ids=output_token_ids,
            logprobs_tensors=logprobs_tensors,
        )
        num_output_tokens_per_req = output_token_ids.ne(PLACEHOLDER_TOKEN_ID).sum(dim=-1).to(torch.int32)
        return SpecSamplingExecutionResult(
            sampler_output=sampler_output,
            num_output_tokens_per_req=num_output_tokens_per_req,
        )

    def execute_from_runtime(
        self,
        *,
        metadata: SpecDecodeMetadata,
        sampling_metadata: SamplingMetadata,
        logits: torch.Tensor,
        top_k_cpu: torch.Tensor | None,
        enable_reduce_sample: bool,
        trim_logits_to_indices: bool = False,
        draft_probs: torch.Tensor | None = None,
        write_markers: bool = False,
        num_reqs: int | None = None,
        num_spec_tokens: int | None = None,
    ) -> SpecSamplingExecutionResult:
        if write_markers:
            write_spec_sampling_marker(
                "entered_mtp_sample",
                {
                    "logits_shape": list(logits.shape),
                    "num_reqs": num_reqs,
                    "num_spec_tokens": num_spec_tokens,
                },
            )
        if trim_logits_to_indices:
            logits = logits[: len(metadata.logits_indices)]
        inputs = self.build_inputs_from_runtime(
            metadata=metadata,
            sampling_metadata=sampling_metadata,
            logits=logits,
            top_k_cpu=top_k_cpu,
            enable_reduce_sample=enable_reduce_sample,
            draft_probs=draft_probs,
        )
        result = self.execute_detailed(inputs)
        if write_markers:
            write_spec_sampling_marker(
                "finished_mtp_sample",
                {
                    "sampled_token_ids_shape": list(result.sampler_output.sampled_token_ids.shape),
                    "num_output_tokens_per_req": result.num_output_tokens_per_req.tolist(),
                },
            )
        return result
