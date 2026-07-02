#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
#

import numpy as np
import torch
from typing import Any

from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.worker.gpu_input_batch import InputBatch
from vllm.v1.worker.gpu_model_runner import CachedRequestState

from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer


class AscendSpecDecodeBaseProposer310(AscendSpecDecodeBaseProposer):
    """310P proposer base: guard empty discard indices before NPU index_fill_."""

    def prepare_next_token_ids_padded(
        self,
        sampled_token_ids: torch.Tensor,
        requests: dict[str, CachedRequestState],
        gpu_input_batch: InputBatch,
        discard_request_indices: torch.Tensor,
        num_discarded_requests: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_reqs = gpu_input_batch.num_reqs
        seq_lens_list = (gpu_input_batch.num_tokens_no_spec[:num_reqs] - 1).tolist()
        self.backup_next_token_ids.np[:num_reqs] = np.array(
            [requests[gpu_input_batch.req_ids[i]].get_token_id(seq_lens_list[i]) for i in range(num_reqs)]
        )
        self.backup_next_token_ids.copy_to_gpu(num_reqs)

        discard_sampled_tokens_req_indices = discard_request_indices[:num_discarded_requests]
        valid_sampled_token_ids_gpu = sampled_token_ids.clone()
        if discard_sampled_tokens_req_indices.numel() != 0:
            valid_sampled_token_ids_gpu.index_fill_(0, discard_sampled_tokens_req_indices, -1)

        valid_mask = (valid_sampled_token_ids_gpu != -1) & (valid_sampled_token_ids_gpu < gpu_input_batch.vocab_size)
        valid_sampled_tokens_count = valid_mask.sum(dim=1)
        last_valid_indices = valid_sampled_tokens_count - 1
        last_valid_indices_safe = torch.clamp(last_valid_indices, min=0)
        selected_tokens = torch.gather(valid_sampled_token_ids_gpu, 1, last_valid_indices_safe.unsqueeze(1)).squeeze(1)
        batch_size = valid_sampled_token_ids_gpu.shape[0]
        next_token_ids = torch.where(
            last_valid_indices != -1,
            selected_tokens,
            self.backup_next_token_ids.gpu[:batch_size],
        )
        return next_token_ids, valid_sampled_tokens_count

    def set_inputs_first_pass(
        self,
        target_token_ids: torch.Tensor,
        next_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        cad: CommonAttentionMetadata,
        num_rejected_tokens_gpu: torch.Tensor | None,
        req_scheduled_tokens=None,
        long_seq_metadata=None,
        num_prefill_reqs=0,
        num_decode_reqs=0,
    ) -> tuple[int, torch.Tensor, CommonAttentionMetadata, tuple[Any, Any] | None]:
        if not self.needs_extra_input_slots:
            # 310P workaround for MTP:
            # The NPU implementation of the slice assign
            #   self.input_ids[:num_tokens-1] = target_token_ids[1:]
            # can corrupt the tail element (index num_tokens-1) of the
            # persistent drafter input_ids buffer. We save/restore it to
            # avoid feeding garbage to the draft model or later GatherV2.
            if token_indices_to_sample is None:
                token_indices_to_sample = cad.query_start_loc[1:] - 1

            num_tokens = target_token_ids.shape[0]

            # Protected shift (310P specific)
            tail_save = self.input_ids[num_tokens - 1].item()
            self.input_ids[: num_tokens - 1] = target_token_ids[1:]
            self.input_ids[num_tokens - 1] = tail_save

            # Replace the last token with the next token.
            self.input_ids[token_indices_to_sample] = next_token_ids

            assert self.runner is not None

            # 310P does not support PCP/DCP, so we skip all PCP handling.
            ori_token_indices_to_sample = None
            query_lens_d = None

            if self.uses_xdrope_dim > 0 and self.draft_uses_xdrope_dim == 0:
                target_positions = target_positions[0]

            self._set_positions(num_tokens, target_positions)
            self.hidden_states[:num_tokens] = target_hidden_states.view(num_tokens, -1)

            return num_tokens, token_indices_to_sample, cad, (query_lens_d, ori_token_indices_to_sample)
        else:
            # For extra-slots path, delegate to base (different code path).
            return super().set_inputs_first_pass(
                target_token_ids,
                next_token_ids,
                target_positions,
                target_hidden_states,
                token_indices_to_sample,
                cad,
                num_rejected_tokens_gpu,
                req_scheduled_tokens=req_scheduled_tokens,
                long_seq_metadata=long_seq_metadata,
                num_prefill_reqs=num_prefill_reqs,
                num_decode_reqs=num_decode_reqs,
            )
