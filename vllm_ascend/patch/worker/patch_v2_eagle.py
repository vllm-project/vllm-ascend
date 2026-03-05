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
import torch
import vllm
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample

from vllm_ascend.utils import vllm_version_is

if vllm_version_is("v0.16.0"):
    from vllm.v1.worker.gpu.spec_decode.eagle import prepare_eagle_decode, prepare_eagle_inputs
else:
    from vllm.v1.worker.gpu.spec_decode.eagle.speculator import prepare_eagle_decode, prepare_eagle_inputs


from vllm_ascend.worker.v2.attn_utils import build_attn_metadata


@torch.inference_mode()
def propose(
    self,
    input_batch: InputBatch,
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
        input_batch.attn_metadata,
        input_batch.slot_mappings,
        num_tokens_across_dp=None,  # FIXME
    )
    sample_hidden_states = last_hidden_states[last_token_indices]
    logits = self.model.compute_logits(sample_hidden_states)

    num_reqs = input_batch.num_reqs
    # NOTE(woosuk): For draft sampling, we only consider the temperature
    # and ignore the other sampling parameters such as top_k and top_p,
    # for simplicity and performance.
    # While this may slightly degrade the acceptance rate, it does not
    # affect the output distribution after rejection sampling.
    # NOTE(Ronald1995): torch.gather will pollute the cache such as self.input_buffers.positions
    # the bug is reported to huawei CANN team, but not fixed yet.
    # So we clone the tensors before calling torch.gather to avoid the issue.
    idx_mapping = self.idx_mapping[:num_reqs]
    idx_mapping.copy_(input_batch.idx_mapping)
    self.temperature.copy_(temperature)
    self.seeds.copy_(seeds)
    pos = self.input_buffers.positions[:num_reqs].clone()
    # Gather the values and copy them to the pre-allocated buffers.
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
    query_start_loc = self.input_buffers.query_start_loc[: num_reqs + 1]
    slot_mappings = self.block_tables.compute_slot_mappings(
        idx_mapping,
        query_start_loc,
        pos,
    )

    cudagraph_size = self.cudagraph_manager.get_cudagraph_size(num_reqs)
    if cudagraph_size is not None:
        # Run CUDA graph.
        self.cudagraph_manager.run(cudagraph_size)
        return self.draft_tokens[:num_reqs]

    # Run eager mode.
    query_start_loc_cpu = torch.arange(num_reqs + 1, dtype=torch.int32, device="cpu")
    # HACK(woosuk)
    block_tables = [x[:num_reqs] for x in self.block_tables.input_block_tables]

    # FIXME(woosuk): This is UNSAFE!!
    attn_metadata = build_attn_metadata(
        attn_metadata_builders=self.attn_metadata_builders,
        num_reqs=num_reqs,
        num_tokens=num_reqs,
        query_start_loc_gpu=query_start_loc,
        query_start_loc_cpu=query_start_loc_cpu,
        max_query_len=1,
        seq_lens=self.input_buffers.seq_lens[:num_reqs],
        max_seq_len=self.max_model_len,
        block_tables=block_tables,
        slot_mappings=slot_mappings,
        kv_cache_config=self.kv_cache_config,
    )
    slot_mappings_by_layer = build_slot_mappings_by_layer(slot_mappings, self.kv_cache_config)
    self.generate_draft(
        num_reqs,
        attn_metadata,
        slot_mappings_by_layer,
        num_tokens_across_dp=None,
    )  # FIXME
    return self.draft_tokens[:num_reqs]


if vllm_version_is("v0.16.0"):
    vllm.v1.worker.gpu.spec_decode.eagle.EagleSpeculator.propose = propose
else:
    vllm.v1.worker.gpu.spec_decode.eagle.speculator.EagleSpeculator.propose = propose
