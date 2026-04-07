# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/sample/spec_decode/eagle.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
from typing import Any

import torch
import vllm
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.dp_utils import sync_cudagraph_and_dp_padding
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.spec_decode.eagle.speculator import prepare_eagle_decode, prepare_eagle_inputs

from vllm_ascend.worker.v2.attn_utils import build_attn_metadata


@torch.inference_mode()
def propose(
    self,
    input_batch: InputBatch,
    attn_metadata: dict[str, Any],
    slot_mappings: dict[str, torch.Tensor],
    # [num_tokens, hidden_size]
    last_hidden_states: torch.Tensor,
    # num_layers x [num_tokens, hidden_size]
    aux_hidden_states: list[torch.Tensor] | None,
    # [num_reqs]
    num_sampled: torch.Tensor,
    # [num_reqs]
    num_rejected: torch.Tensor,
    # [max_num_reqs]
    last_sampled: torch.Tensor,
    # [max_num_reqs]
    next_prefill_tokens: torch.Tensor,
    # [max_num_reqs]
    temperature: torch.Tensor,
    # [max_num_reqs]
    seeds: torch.Tensor,
    num_tokens_across_dp: torch.Tensor | None = None,
    dummy_run: bool = False,
    skip_attn_for_dummy_run: bool = False,
    mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    # NOTE(woosuk): To avoid CPU-GPU synchronization without CPU knowing the
    # number of rejected tokens, we maintain the size of eagle's input_ids and
    # hidden_states the same as the target model's. This means, we pad each
    # request's query length to include any rejected positions. By doing so,
    # we can also reuse the attention metadata (e.g., query_start_loc,
    # seq_lens) of the target model.
    if aux_hidden_states:
        assert self.method == "eagle3"
        hidden_states = self.model.combine_hidden_states(torch.cat(aux_hidden_states, dim=-1))
    else:
        hidden_states = last_hidden_states
    num_tokens = input_batch.num_tokens_after_padding
    self.hidden_states[:num_tokens] = hidden_states

    # Get the input ids and last token indices for the speculator.
    last_token_indices = prepare_eagle_inputs(
        self.input_buffers,
        input_batch,
        num_sampled,
        num_rejected,
        last_sampled,
        next_prefill_tokens,
    )

    # Prefill: Run the eagle speculator with eager mode.
    # TODO(woosuk): Support CUDA graph for prefill.
    last_hidden_states, hidden_states = self.run_model(
        num_tokens,
        attn_metadata,
        slot_mappings,
        num_tokens_across_dp=num_tokens_across_dp,
        mm_inputs=mm_inputs,
    )
    sample_hidden_states = last_hidden_states[last_token_indices]
    logits = self.model.compute_logits(sample_hidden_states)

    num_reqs = input_batch.num_reqs
    num_reqs_padded = input_batch.num_reqs_after_padding
    # NOTE(woosuk): For draft sampling, we only consider the temperature
    # and ignore the other sampling parameters such as top_k and top_p,
    # for simplicity and performance.
    # While this may slightly degrade the acceptance rate, it does not
    # affect the output distribution after rejection sampling.
    idx_mapping = self.idx_mapping[:num_reqs]
    idx_mapping.copy_(input_batch.idx_mapping)
    self.temperature.copy_(temperature)
    self.seeds.copy_(seeds)

    # NOTE(Ronald1995): torch.gather will pollute the cache such as self.input_buffers.positions
    # the bug is reported to huawei CANN team, but not fixed yet.
    # So we clone the tensors before calling torch.gather to avoid the issue.

    # Gather the values and copy them to the pre-allocated buffers.
    pos = self.input_buffers.positions[:num_reqs].clone()
    torch.gather(input_batch.positions, 0, last_token_indices, out=pos)
    # NOTE(woosuk): We must add 1 to the positions to match the Gumbel noise
    # used for draft and target sampling.
    draft_tokens = gumbel_sample(
        logits,
        idx_mapping,
        self.temperature,
        self.seeds,
        pos + 1,
        apply_temperature=True,
        processed_logits_out=self.draft_logits[:, 0] if self.draft_logits is not None else None,
    )

    if self.num_speculative_steps == 1:
        # Early exit.
        return draft_tokens.view(-1, 1)

    # Save the draft tokens for the first step.
    self.draft_tokens[:num_reqs, 0] = draft_tokens
    # Prepare the inputs for the decode steps.
    prepare_eagle_decode(
        draft_tokens,
        hidden_states,
        last_token_indices,
        input_batch.seq_lens,
        num_rejected,
        self.input_buffers,
        self.hidden_states,
        self.max_model_len,
        self.max_num_reqs,
    )

    # Get batch descriptor and sync across DP ranks.
    # Eagle uses FULL-only mode, dispatch with uniform_token_count=1 for decode

    batch_desc = self.cudagraph_manager.dispatch(num_reqs, num_reqs, 1)
    num_tokens_across_dp = None

    if self.dp_size > 1:
        batch_desc, num_tokens_across_dp = sync_cudagraph_and_dp_padding(
            self.cudagraph_manager,
            batch_desc,
            num_reqs,
            num_reqs,
            1,  # uniform_token_count
            self.dp_size,
            self.dp_rank,
        )

    if not (dummy_run and skip_attn_for_dummy_run):
        query_start_loc = self.input_buffers.query_start_loc[: num_reqs + 1]
        slot_mappings = self.block_tables.compute_slot_mappings(
            idx_mapping, query_start_loc, pos, batch_desc.num_tokens
        )

    if batch_desc.cg_mode == CUDAGraphMode.FULL:
        return self.cudagraph_manager.run_fullgraph(batch_desc)[:num_reqs]

    # Run eager or piecewise CUDA graph.
    attn_metadata_updated = None
    slot_mappings_updated = None
    if not (dummy_run and skip_attn_for_dummy_run):
        query_start_loc_cpu = torch.arange(num_reqs_padded + 1, dtype=torch.int32, device="cpu")
        block_tables = [x[:num_reqs_padded] for x in self.block_tables.input_block_tables]

        # FIXME(woosuk): This is UNSAFE!!
        attn_metadata_updated = build_attn_metadata(
            attn_groups=self.attn_groups,
            num_reqs=num_reqs_padded,
            num_tokens=num_reqs_padded,
            query_start_loc_gpu=query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            max_query_len=1,
            seq_lens=self.input_buffers.seq_lens[:num_reqs_padded],
            max_seq_len=self.max_model_len,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=self.kv_cache_config,
        )
        slot_mappings_updated = build_slot_mappings_by_layer(slot_mappings, self.kv_cache_config)

    self.generate_draft(
        num_reqs,
        batch_desc.num_tokens,
        attn_metadata_updated,
        slot_mappings_updated,
        num_tokens_across_dp=num_tokens_across_dp,
        cudagraph_runtime_mode=batch_desc.cg_mode,
    )
    return self.draft_tokens[:num_reqs]


vllm.v1.worker.gpu.spec_decode.eagle.speculator.EagleSpeculator.propose = propose
