import torch
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.sample.metadata import SamplingMetadata
from vllm.v1.worker.gpu.sample.min_p import apply_min_p
from vllm.v1.worker.gpu.sample.sampler import Sampler

from vllm_ascend.worker.v2.sample.gumbel import gumbel_sample
from vllm_ascend.worker.v2.sample.penalties import \
    apply_penalties_and_temperature


class AscendSampler(Sampler):

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Override sample method because we need to override triton operators
        called in the method.
        """
        # Copy logits to a new FP32 tensor.
        logits = torch.empty_like(logits, dtype=torch.float32).copy_(logits)

        # Apply penalties and temperature in place.
        apply_penalties_and_temperature(logits, sampling_metadata)
        # Apply min_p in place.
        if sampling_metadata.min_p is not None:
            apply_min_p(logits, sampling_metadata.min_p)
        # Apply top_k and/or top_p. This might return a new tensor.
        logits = apply_top_k_top_p(logits, sampling_metadata.top_k,
                                   sampling_metadata.top_p)

        sampled = gumbel_sample(
            logits,
            sampling_metadata.temperature,
            sampling_metadata.seeds,
            sampling_metadata.pos,
            apply_temperature=False,
        )
        return sampled, logits
