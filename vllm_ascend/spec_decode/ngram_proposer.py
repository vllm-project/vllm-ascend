import torch
from vllm.v1.spec_decode.ngram_proposer import NgramProposer

from vllm_ascend.utils import vllm_version_is


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

    if vllm_version_is("0.23.0"):

        def propose(
            self,
            sampled_token_ids: list[list[int]],
            num_tokens_no_spec=None,
            token_ids_cpu=None,
            slot_mappings: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None = None,
        ) -> list[list[int]]:
            input_batch = self.runner.input_batch
            valid_ngram_requests = []
            for i, sampled_ids in enumerate(sampled_token_ids):
                num_sampled_ids = len(sampled_ids)
                if not num_sampled_ids:
                    continue

                req_id = input_batch.req_ids[i]
                if req_id in input_batch.spec_decode_unsupported_reqs:
                    continue

                num_tokens = input_batch.num_tokens_no_spec[i]
                if num_tokens >= input_batch.max_model_len:
                    # Skip requests that have already reached the max model length.
                    continue

                # NOTE: The sampled tokens are already written into
                # ``token_ids_cpu`` (and ``num_tokens_no_spec`` is already
                # advanced) by ``_bookkeeping_sync`` in the model runner, which
                # runs *before* this proposer. Mirroring upstream vLLM's
                # ``NgramProposer.propose``, we must NOT write them again here:
                # doing so would write at an incorrect offset
                # ``num_tokens_no_spec + num_sampled_ids`` (since
                # ``num_tokens_no_spec`` already reflects the just-sampled
                # tokens) and could overflow ``token_ids_cpu`` when a request
                # is close to ``max_model_len``.

                valid_ngram_requests.append(i)

            return self.batch_propose(
                len(sampled_token_ids),
                valid_ngram_requests,
                input_batch.num_tokens_no_spec,
                input_batch.token_ids_cpu,
            )

    else:

        def propose(  # type: ignore[misc]
            self,
            num_speculative_tokens: int,
            sampled_token_ids: list[list[int]],
            num_tokens_no_spec=None,
            token_ids_cpu=None,
            slot_mappings: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None = None,
        ) -> list[list[int]]:
            assert num_speculative_tokens <= self.k
            assert num_tokens_no_spec is not None
            assert token_ids_cpu is not None

            valid_ngram_requests = []
            for i, sampled_ids in enumerate(sampled_token_ids):
                num_sampled_ids = len(sampled_ids)
                if not num_sampled_ids:
                    continue

                num_tokens = num_tokens_no_spec[i]
                if num_tokens >= self.max_model_len:
                    # Skip requests that have already reached the max model length.
                    continue

                valid_ngram_requests.append(i)

            return self.batch_propose(
                len(sampled_token_ids),
                valid_ngram_requests,
                num_tokens_no_spec,
                token_ids_cpu,
                num_speculative_tokens,
            )
