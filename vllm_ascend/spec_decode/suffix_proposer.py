import torch
from vllm.config import CUDAGraphMode
from vllm.v1.spec_decode.suffix_decoding import \
    SuffixDecodingProposer as VllmSuffixDecodingProposer

from vllm_ascend.spec_decode.interface import Proposer, SpecDcodeType


class SuffixDecodingProposer(VllmSuffixDecodingProposer, Proposer):

    def __init__(self, vllm_config, device, runner):
        super().__init__(vllm_config)
        self.name = SpecDcodeType.SUFFIX
        self.device = device
        self.runner = runner

    def load_model(self, *args, **kwargs):
        # No model to load.
        pass

    @torch.inference_mode()
    def dummy_run(self,
                  num_tokens,
                  with_prefill=None,
                  skip_attn=None,
                  num_reqs=None,
                  num_tokens_across_dp=None,
                  aclgraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
                  batch_descriptor=None,
                  dummy_compute_logits=lambda hidden_states: None):
        pass

    def generate_token_ids(self,
                           valid_sampled_token_ids,
                           sampling_metadata=None,
                           scheduler_output=None,
                           spec_decode_metadata=None,
                           positions=None,
                           num_scheduled_tokens=None,
                           hidden_states=None,
                           attn_metadata=None,
                           aux_hidden_states=None) -> list[list[int]]:
        draft_token_ids = self.propose(self.runner.input_batch,
                                       valid_sampled_token_ids)
        return draft_token_ids
