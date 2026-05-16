from vllm.v1.spec_decode.suffix_decoding import SuffixDecodingProposer
from vllm.config import CUDAGraphMode


class AscendSuffixDecodingProposer(SuffixDecodingProposer):
    def __init__(self, vllm_config, runner):
        super().__init__(vllm_config)
        self.runner = runner
        self.vllm_config = vllm_config

    def dummy_run(
        self,
        num_tokens,
        with_prefill=None,
        in_graph_capturing=None,
        num_reqs=None,
        num_tokens_across_dp=None,
        aclgraph_runtime_mode=None,
        batch_descriptor=None,
        dummy_compute_logits=lambda hidden_states: None,
        is_profile=False,
    ):
        pass

    def propose(self, valid_sampled_token_ids):
        draft_token_ids = super().propose(self.runner.input_batch, valid_sampled_token_ids)
        if self.vllm_config.compilation_config.cudagraph_mode.has_mode(CUDAGraphMode.FULL):
            target_len = self.num_speculative_tokens
            for i in range(len(draft_token_ids)):
                cur_len = len(draft_token_ids[i])
                if cur_len < target_len:
                    draft_token_ids[i].extend([0] * (target_len - cur_len))
        return draft_token_ids
