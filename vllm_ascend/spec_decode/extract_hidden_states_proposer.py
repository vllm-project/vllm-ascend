#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
    def _determine_batch_execution_and_padding(
        self,
        num_tokens: int,
        use_cudagraphs: bool = True,
    ) -> tuple[CUDAGraphMode, int, torch.Tensor | None]:
        """Determine cudagraph mode and padded token count for this proposer step.

        Same contract as upstream ``ExtractHiddenStatesProposer``, but follows
        the Ascend model runner path: SP-pad ``num_tokens`` before dispatch,
        and use ``runner._sync_metadata_across_dp`` for DP coordination when
        data parallel size > 1.
        """
        assert self.runner is not None, (
            "AscendExtractHiddenStatesProposer requires a runner reference "
            "for _pad_for_sequence_parallelism / _sync_metadata_across_dp"
        )

        # SP padding must happen BEFORE DP sync (matches the main runner).
        # The v2 NPUModelRunner does not implement
        # _pad_for_sequence_parallelism; fail with a clear error instead of a
        # confusing AttributeError if this proposer is paired with it.
        if not hasattr(self.runner, "_pad_for_sequence_parallelism"):
            raise NotImplementedError(
                "The current model runner does not support sequence "
                "parallelism padding (_pad_for_sequence_parallelism) required "
                "for AscendExtractHiddenStatesProposer."
            )
        num_tokens = self.runner._pad_for_sequence_parallelism(num_tokens)

        cudagraph_mode, batch_desc = self.cudagraph_dispatcher.dispatch(
            num_tokens,
            valid_modes=({CUDAGraphMode.NONE} if not use_cudagraphs else None),
        )
        num_tokens_padded = batch_desc.num_tokens

        num_tokens_across_dp = None
        if self.vllm_config.parallel_config.data_parallel_size > 1:
            # The v2 NPUModelRunner does not implement
            # _sync_metadata_across_dp; fail with a clear error instead of a
            # confusing AttributeError when data parallel size > 1.
            if not hasattr(self.runner, "_sync_metadata_across_dp"):
                raise NotImplementedError(
                    "The current model runner does not support DP metadata "
                    "synchronization (_sync_metadata_across_dp) required for "
                    "data parallel size > 1."
                )
            # Reuse the runner's DP sync so the shape matches the main model
            # forward. ``is_draft_model=True`` short-circuits the all_reduce
            # (cache_only draft is not MoE); the SP-pad above keeps the value
            # TP-aligned.
            (
                _max_tokens_across_dp,
                num_tokens_across_dp,
                synced_cudagraph_mode,
            ) = self.runner._sync_metadata_across_dp(
                num_tokens=num_tokens_padded,
                is_draft_model=True,
                cudagraph_mode=cudagraph_mode,
                allow_dp_padding=use_cudagraphs,
            )

            if num_tokens_across_dp is not None:
                num_tokens_padded = int(num_tokens_across_dp[self.dp_rank].item())
                # Re-dispatch with DP-synced padding.
                cudagraph_mode, batch_desc = self.cudagraph_dispatcher.dispatch(
                    num_tokens_padded,
                    valid_modes={synced_cudagraph_mode},
                )
                assert batch_desc.num_tokens == num_tokens_padded

        return cudagraph_mode, num_tokens_padded, num_tokens_across_dp

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
