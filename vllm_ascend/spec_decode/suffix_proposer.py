from vllm.v1.spec_decode.suffix_decoding import SuffixDecodingProposer


class AscendSuffixDecodingProposer(SuffixDecodingProposer):
    def __init__(self, vllm_config, runner):
        super().__init__(vllm_config)
        self.runner = runner

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
        target_len = self.num_speculative_tokens
        draft_token_ids = super().propose(self.runner.input_batch, valid_sampled_token_ids)
        for i in range(len(draft_token_ids)):
            cur_len = len(draft_token_ids[i])
            if cur_len < target_len:
                draft_token_ids[i].extend([0] * (target_len - cur_len))
        return draft_token_ids
