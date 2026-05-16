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
#

import torch

from vllm_ascend.utils import enable_custom_op


def quest_prefill_metadata(
    k_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    metadata_block_tables: torch.Tensor,
    maxblocks: torch.Tensor,
    minblocks: torch.Tensor,
) -> None:
    enable_custom_op()
    torch.ops._C_ascend.npu_quest_prefill_metadata(
        k_cache,
        block_tables,
        seq_lens,
        metadata_block_tables,
        maxblocks,
        minblocks,
    )


def quest_block_select_paged(
    query: torch.Tensor,
    maxblocks: torch.Tensor,
    minblocks: torch.Tensor,
    metadata_block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    k: int,
    tokens_since_metadata_update: int = -1,
) -> torch.Tensor:
    enable_custom_op()
    return torch.ops._C_ascend.npu_quest_block_select_paged(
        query,
        maxblocks,
        minblocks,
        metadata_block_tables,
        seq_lens,
        k,
        tokens_since_metadata_update,
    )


def quest_block_select_paged_out(
    query: torch.Tensor,
    maxblocks: torch.Tensor,
    minblocks: torch.Tensor,
    metadata_block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    out: torch.Tensor,
    tokens_since_metadata_update: int = -1,
) -> torch.Tensor:
    enable_custom_op()
    return torch.ops._C_ascend.npu_quest_block_select_paged_out(
        query,
        maxblocks,
        minblocks,
        metadata_block_tables,
        seq_lens,
        out,
        tokens_since_metadata_update,
    )


def paged_select_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    actual_seq_lengths: list[int],
    actual_seq_lengths_kv: list[int],
    block_table: torch.Tensor,
    selected_kv_indices: torch.Tensor,
    num_heads: int,
    scale_value: float,
    num_key_value_heads: int,
    block_size: int,
) -> torch.Tensor:
    enable_custom_op()
    return torch.ops._C_ascend.npu_paged_select_attention(
        query,
        key,
        value,
        actual_seq_lengths,
        actual_seq_lengths_kv,
        block_table,
        selected_kv_indices,
        num_heads,
        scale_value,
        num_key_value_heads,
        block_size,
    )


def paged_select_attention_get_workspace(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    actual_seq_lengths: list[int],
    actual_seq_lengths_kv: list[int],
    block_table: torch.Tensor,
    selected_kv_indices: torch.Tensor,
    num_heads: int,
    scale_value: float,
    num_key_value_heads: int,
    block_size: int,
    output: torch.Tensor,
) -> torch.Tensor:
    enable_custom_op()
    return torch.ops._C_ascend.npu_paged_select_attention_get_workspace(
        query,
        key,
        value,
        actual_seq_lengths,
        actual_seq_lengths_kv,
        block_table,
        selected_kv_indices,
        num_heads,
        scale_value,
        num_key_value_heads,
        block_size,
        output,
    )


def paged_select_attention_graph_out(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    actual_seq_lengths: list[int],
    actual_seq_lengths_kv: list[int],
    block_table: torch.Tensor,
    selected_kv_indices: torch.Tensor,
    num_heads: int,
    scale_value: float,
    num_key_value_heads: int,
    block_size: int,
    workspace: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    enable_custom_op()
    return torch.ops._C_ascend.npu_paged_select_attention_graph_out(
        query,
        key,
        value,
        actual_seq_lengths,
        actual_seq_lengths_kv,
        block_table,
        selected_kv_indices,
        num_heads,
        scale_value,
        num_key_value_heads,
        block_size,
        workspace,
        out,
    )


__all__ = [
    "paged_select_attention",
    "paged_select_attention_get_workspace",
    "paged_select_attention_graph_out",
    "quest_prefill_metadata",
    "quest_block_select_paged",
    "quest_block_select_paged_out",
]
