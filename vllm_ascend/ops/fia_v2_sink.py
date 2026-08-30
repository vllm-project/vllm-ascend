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

import torch

FIA_V2_SINK_METADATA_SIZE = 1024


def build_fia_v2_sink_metadata(
    *,
    actual_seq_qlen: torch.Tensor,
    actual_seq_kvlen: torch.Tensor,
    num_query_heads: int,
    num_key_value_heads: int,
    head_dim_qk: int,
    head_dim_v: int,
    input_layout: str,
    input_layout_kv: str,
    sparse_mode: int,
    block_size: int,
    pre_tokens: int = 2147483647,
    next_tokens: int = 2147483647,
    rope_head_dim: int = 0,
    output_buffer: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build FIA tiling metadata before attention graph capture or replay."""
    import vllm_ascend.vllm_ascend_C  # type: ignore[import-untyped]  # noqa: F401, PLC0415

    # The AICPU metadata operator reads these tensors asynchronously.  Keep
    # call-local storage, as the DSA QLI metadata path does, so a subsequent
    # metadata build cannot overwrite its inputs.
    actual_seq_qlen = actual_seq_qlen.clone()
    actual_seq_kvlen = actual_seq_kvlen.clone()
    stream_limit = torch.npu.get_stream_limit(torch.npu.current_stream())
    metadata = torch.ops._C_ascend._npu_fused_infer_attention_score_v2_sink_metadata(
        num_query_heads,
        num_key_value_heads,
        head_dim_qk,
        head_dim_v,
        actual_seq_lengths=actual_seq_qlen,
        actual_seq_lengths_kv=actual_seq_kvlen,
        batch_size=actual_seq_kvlen.shape[0],
        sparse_mode=sparse_mode,
        pre_tokens=pre_tokens,
        next_tokens=next_tokens,
        input_layout=input_layout,
        input_layout_kv=input_layout_kv,
        sink_num=0,
        k_sink_num=0,
        rope_head_dim=rope_head_dim,
        block_size=block_size,
        aic_core_num=stream_limit["cube_core_num"],
        aiv_core_num=stream_limit["vector_core_num"],
    )
    if output_buffer is None:
        return metadata
    output_buffer[:FIA_V2_SINK_METADATA_SIZE].copy_(metadata)
    return output_buffer


def fused_infer_attention_score_v2_sink(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    actual_seq_qlen: torch.Tensor,
    actual_seq_kvlen: torch.Tensor,
    block_table: torch.Tensor,
    metadata: torch.Tensor,
    num_query_heads: int,
    num_key_value_heads: int,
    softmax_scale: float,
    input_layout: str,
    sparse_mode: int,
    block_size: int,
    atten_mask: torch.Tensor | None = None,
    query_rope: torch.Tensor | None = None,
    key_rope: torch.Tensor | None = None,
    pre_tokens: int = 2147483647,
    next_tokens: int = 2147483647,
) -> torch.Tensor:
    """Run paged FIA with device-side sequence-length tiling.

    ``metadata`` is built by the attention metadata builder before graph replay,
    so the graph contains only the FIA kernel and reads a stable metadata buffer.
    """
    import vllm_ascend.vllm_ascend_C  # type: ignore[import-untyped]  # noqa: F401, PLC0415

    output, _ = torch.ops._C_ascend.npu_fused_infer_attention_score_v2_sink(
        query,
        key,
        value,
        query_rope=query_rope,
        key_rope=key_rope,
        atten_mask=atten_mask,
        actual_seq_qlen=actual_seq_qlen,
        actual_seq_kvlen=actual_seq_kvlen,
        block_table=block_table,
        meta_data=metadata,
        num_query_heads=num_query_heads,
        num_key_value_heads=num_key_value_heads,
        softmax_scale=softmax_scale,
        pre_tokens=pre_tokens,
        next_tokens=next_tokens,
        input_layout=input_layout,
        sparse_mode=sparse_mode,
        block_size=block_size,
    )
    return output
