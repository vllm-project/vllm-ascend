import torch
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.logger import init_logger
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.medusa import MedusaProposer as VllmMedusaProposer
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.spec_decode.interface import SpecDcodeType

logger = init_logger(__name__)


class MedusaProposer(VllmMedusaProposer):
    """
    Medusa proposer class for generating token sequences
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner,
    ):
        # Save config parameters
        self.name = SpecDcodeType.MEDUSA
        self.vllm_config = vllm_config
        self.device = device
        self.max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.hidden_size = vllm_config.speculative_config.draft_model_config.get_hidden_size()
        self.dtype = vllm_config.model_config.dtype
        self.runner = runner

    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens: int,
        with_prefill: bool = False,
        in_graph_capturing: bool = False,
        num_reqs: int = 0,
        num_tokens_across_dp: torch.Tensor | None = None,
        aclgraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        batch_descriptor=None,
        dummy_compute_logits=lambda hidden_states: None,
        is_profile=False,
    ):
        hidden_states = torch.zeros(
            (self.max_num_tokens, self.hidden_size),
            dtype=self.dtype,
            device=self.device,
        )
        with set_ascend_forward_context(
            None,
            self.vllm_config,
            num_tokens=num_tokens,
            num_actual_tokens=0,
            in_profile_run=is_profile,
            batch_descriptor=batch_descriptor,
            aclgraph_runtime_mode=aclgraph_runtime_mode,
            is_draft_model=True,
        ):
            self.model(hidden_states)
            dummy_compute_logits(hidden_states)

    def generate_token_ids(
        self,
        valid_sampled_token_ids: list[list[int]],
        sampling_metadata: SamplingMetadata,
        spec_decode_metadata: SpecDecodeMetadata,
        sample_hidden_states: torch.Tensor,
        *args,
        **kwargs,
    ):
        if sample_hidden_states.shape[0] == len(valid_sampled_token_ids):
            # The input to the target model does not include draft tokens.
            hidden_states = sample_hidden_states
        else:
            num_accepted_tokens = torch.tensor(
                [len(t) for t in valid_sampled_token_ids], device=self.device, dtype=torch.long
            )
            num_draft_tokens = torch.tensor(spec_decode_metadata.num_draft_tokens, device=self.device, dtype=torch.long)

            offsets = torch.cumsum(num_draft_tokens + 1, dim=0) - (num_draft_tokens + 1)
            indices = offsets + num_accepted_tokens - 1
            hidden_states = sample_hidden_states[indices]

        spec_token_ids = self.propose(
            target_hidden_states=hidden_states,
            sampling_metadata=sampling_metadata,
        )
        return spec_token_ids
