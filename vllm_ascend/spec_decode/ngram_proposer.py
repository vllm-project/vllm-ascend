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

        # vllm PR #44597: when the speculative config sets
        # ``prompt_lookup_cache_scope='global'``, the upstream
        # ``NgramProposer.__init__`` (already chained via
        # ``super().__init__``) populates ``self.global_ngram_cache`` and
        # exposes ``batch_propose_global``; we dispatch to the upstream
        # method so the global LRU prompt-lookup cache is reused verbatim.
        # No NPU-side changes are needed because the optimization is
        # pure-Python / CPU-only and operates on the same ``input_batch``
        # arrays the legacy ``batch_propose`` consumes.
        # ``getattr`` keeps us compatible with older vllm versions that
        # predate PR #44597 (where neither the field nor the method
        # exists) — we silently fall back to the legacy ``batch_propose``.
        if getattr(self, "global_ngram_cache", None) is not None and hasattr(self, "batch_propose_global"):
            return self.batch_propose_global(
                len(sampled_token_ids),
                valid_ngram_requests,
                self.runner.input_batch.num_tokens_no_spec,
                self.runner.input_batch.token_ids_cpu,
                self.runner.input_batch.req_ids,
            )

        draft_token_ids = self.batch_propose(
            len(sampled_token_ids),
            valid_ngram_requests,
            self.runner.input_batch.num_tokens_no_spec,
            self.runner.input_batch.token_ids_cpu,
        )

        return draft_token_ids
