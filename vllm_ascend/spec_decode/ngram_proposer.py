import torch
from vllm.config import CUDAGraphMode
from vllm.v1.spec_decode.ngram_proposer import \
    NgramProposer as VllmNgramProposer

from vllm_ascend.spec_decode.interface import Proposer, SpecDcodeType


class NgramProposer(VllmNgramProposer, Proposer):

    def __init__(self, vllm_config, device, runner):
        super().__init__(vllm_config)
        self.name = SpecDcodeType.NGRAM
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
        valid_ngram_requests = []
        for i, sampled_ids in enumerate(valid_sampled_token_ids):
            num_sampled_ids = len(sampled_ids)
            if not num_sampled_ids:
                continue

            req_id = self.runner.input_batch.req_ids[i]
            if req_id in self.runner.input_batch.spec_decode_unsupported_reqs:
                continue

            num_tokens = self.runner.input_batch.num_tokens_no_spec[i]
            if num_tokens >= self.runner.input_batch.max_model_len:
                # Skip requests that have already reached the max model length.
                continue

            start_idx = self.runner.input_batch.num_tokens_no_spec[i]
            end_idx = start_idx + num_sampled_ids
            self.runner.input_batch.token_ids_cpu[
                i, start_idx:end_idx] = sampled_ids

            valid_ngram_requests.append(i)

        draft_token_ids = self.batch_propose(
            len(valid_sampled_token_ids),
            valid_ngram_requests,
            self.runner.input_batch.num_tokens_no_spec,
            self.runner.input_batch.token_ids_cpu,
        )

        return draft_token_ids