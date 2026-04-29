# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend adaptation of ExtractHiddenStatesProposer for extracting and caching
hidden states during speculative decoding."""

import torch
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import set_forward_context
from vllm.v1.spec_decode.extract_hidden_states import ExtractHiddenStatesProposer


class AscendExtractHiddenStatesProposer(ExtractHiddenStatesProposer):
    """Ascend-adapted ExtractHiddenStatesProposer for NPU devices.

    This proposer extracts hidden states from the target model and caches them
    in the KV cache without performing actual speculation. It's used with the
    ExampleHiddenStatesConnector for KV transfer.

    The main differences from the GPU version:
    - Uses ACL graphs instead of CUDA graphs
    - Implements dummy_run for ACL graph capture with Ascend-specific signature
    - Adapts prepare_next_token_ids_padded for Ascend's indices/count pattern
    """

    def __init__(self, vllm_config: VllmConfig, device: torch.device, runner=None):
        self.runner = runner
        super().__init__(vllm_config, device)

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
    ) -> None:
        """Dummy run for ACL graph capture.

        Same functional logic as GPU version but with Ascend's parameter signature.
        """
        assert self.model is not None, "Model must be initialized before dummy_run"

        if num_tokens_across_dp is not None:
            num_tokens_across_dp[self.dp_rank] = num_tokens

        with set_forward_context(
            None,
            self.vllm_config,
            num_tokens=num_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=aclgraph_runtime_mode or CUDAGraphMode.NONE,
            slot_mapping={},
        ):
            self.model(
                hidden_states=self.hidden_states[:num_tokens],
            )

    def prepare_next_token_ids_padded(
        self,
        sampled_token_ids: torch.Tensor,
        requests,
        gpu_input_batch,
        discard_request_indices: torch.Tensor,
        num_discarded_requests: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare next token IDs for speculative decoding.

        Since num_speculative_tokens == 1, sampled_token_ids has shape
        (batch_size, 1). For each request we either use the sampled token
        (if valid and not discarded) or a backup token from the request state.

        This adapts the GPU version for Ascend's indices/count pattern
        (discard_request_indices instead of boolean mask).
        """
        num_reqs = gpu_input_batch.num_reqs
        device = sampled_token_ids.device

        # Compute backup tokens for discarded / invalid requests
        seq_lens_list = (gpu_input_batch.num_tokens_no_spec[:num_reqs] - 1).tolist()
        backup_tokens = torch.tensor(
            [requests[gpu_input_batch.req_ids[i]].get_token_id(seq_lens_list[i]) for i in range(num_reqs)],
            dtype=torch.int32,
            device=device,
        )

        # Create discard mask from indices (Ascend uses indices/count pattern)
        discard_mask = torch.zeros(num_reqs, dtype=torch.bool, device=device)
        discard_mask[discard_request_indices[:num_discarded_requests]] = True

        # With num_speculative_tokens == 1, there is exactly one token
        sampled = sampled_token_ids[:, 0]
        is_valid = (sampled >= 0) & (sampled < gpu_input_batch.vocab_size)
        valid_sampled_tokens_count = is_valid.to(torch.int32)

        use_sampled = is_valid & ~discard_mask
        next_token_ids = torch.where(use_sampled, sampled.to(torch.int32), backup_tokens)

        return next_token_ids, valid_sampled_tokens_count
