from vllm.v1.spec_decode.suffix_decoding import SuffixDecodingProposer

from vllm_ascend.utils import vllm_version_is


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
        if vllm_version_is("0.23.0"):
            return super().propose(self.runner.input_batch, valid_sampled_token_ids)
        else:
            return super().propose(
                self.num_speculative_tokens, self.runner.input_batch, valid_sampled_token_ids, slot_mappings=None
            )
