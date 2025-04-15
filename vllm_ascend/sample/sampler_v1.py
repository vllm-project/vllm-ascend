import torch
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.penalties import apply_min_token_penalties
from vllm.v1.sample.sampler import Sampler

from vllm_ascend.sample.ops.ascend_topk_topp_sampler import \
    AscendTopKTopPSampler
from vllm_ascend.sample.ops.penalties import apply_all_penalties


class AscendSampler(Sampler):

    def __init__(self):
        super().__init__()
        self.topk_topp_sampler = AscendTopKTopPSampler()

    def apply_penalties(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        if sampling_metadata.min_tokens:
            apply_min_token_penalties(logits,
                                      sampling_metadata.output_token_ids,
                                      sampling_metadata.min_tokens)
        if not sampling_metadata.no_penalties:
            assert sampling_metadata.prompt_token_ids is not None
            logits = apply_all_penalties(
                logits,
                sampling_metadata.prompt_token_ids,
                sampling_metadata.presence_penalties,
                sampling_metadata.frequency_penalties,
                sampling_metadata.repetition_penalties,
                sampling_metadata.output_token_ids,
            )
        return logits
