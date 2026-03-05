import torch
from vllm.v1.spec_decode.ngram_proposer import NgramProposer


class AscendNgramProposer(NgramProposer):
    def __init__(self, vllm_config, runner):
        self.runner = runner
        super().__init__(vllm_config)

    def load_model(self, *args, **kwargs):
        # No model to load.
        pass

    @torch.inference_mode()
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

    def propose(
        self,
        sampled_token_ids: list[list[int]],
        num_tokens_no_spec=None,
        token_ids_cpu=None,
        slot_masks: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None = None,
    ) -> list[list[int]]:
        valid_ngram_requests = []
        for i, sampled_ids in enumerate(sampled_token_ids):
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
            self.runner.input_batch.token_ids_cpu[i, start_idx:end_idx] = sampled_ids

            valid_ngram_requests.append(i)

        draft_token_ids = self.batch_propose(
            len(sampled_token_ids),
            valid_ngram_requests,
            self.runner.input_batch.num_tokens_no_spec,
            self.runner.input_batch.token_ids_cpu,
        )

        return draft_token_ids
