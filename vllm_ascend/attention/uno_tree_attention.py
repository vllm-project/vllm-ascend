# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""UNO tree GQA attention using paged BSH FIA-v2 with an explicit mask."""

from dataclasses import dataclass

import torch
import torch_npu

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.compilation.acl_graph import get_graph_params
from vllm_ascend.utils import weak_ref_tensors


@dataclass
class UnoTreeAttentionGraph:
    kwargs: dict
    output: list
    workspace: torch.Tensor


def forward_uno_tree_attention(impl, query, key, value, metadata, output):
    key, value, block_size, table, seq_lens = impl._get_fia_params(key, value, metadata)
    rows = metadata.num_actual_tokens
    if len(seq_lens) != 1:
        raise ValueError("UNO tree attention currently requires a single request.")
    query_bsh = query[:rows].reshape(1, rows, -1)
    output_bsh = output[:rows].view(1, rows, -1)
    if table is None:
        key = key.reshape(1, rows, -1)
        value = value.reshape(1, rows, -1)
    kwargs = dict(
        query=query_bsh,
        key=key,
        value=value,
        block_table=table,
        atten_mask=metadata.uno_tree_mask,
        input_layout="BSH",
        block_size=block_size,
        actual_seq_qlen=[rows],
        actual_seq_kvlen=seq_lens,
        num_query_heads=impl.num_heads,
        num_key_value_heads=impl.num_kv_heads,
        softmax_scale=impl.scale,
        sparse_mode=0,
    )
    if not _EXTRA_CTX.capturing:
        result, _ = torch_npu.npu_fused_infer_attention_score_v2(**kwargs)
        output_bsh.copy_(result)
        return output

    params = get_graph_params()
    # All layers in the supported single GQA cache group share these shapes
    # and execute serially. Keep one workspace for this graph size.
    workspace = params.workspaces.get(rows)
    if workspace is None:
        workspace = torch_npu._npu_fused_infer_attention_score_v2_get_max_workspace(**kwargs)
        params.workspaces[rows] = workspace
    lse = torch.empty(1, dtype=query.dtype, device=query.device)
    stream = torch.npu.current_stream()
    event = torch.npu.ExternalEvent()
    event.wait(stream)
    event.reset(stream)
    params.events[rows].append(event)
    captured = {
        name: weak_ref_tensors(value) if isinstance(value, torch.Tensor) else value for name, value in kwargs.items()
    }
    entry = UnoTreeAttentionGraph(captured, [weak_ref_tensors(output_bsh), weak_ref_tensors(lse)], workspace)
    params.attn_params[rows].append(entry)
    torch.npu.graph_task_group_begin(stream)
    torch_npu.npu_fused_infer_attention_score_v2.out(**kwargs, workspace=workspace, out=[output_bsh, lse])
    params.handles[rows].append(torch.npu.graph_task_group_end(stream))
    return output


def update_uno_tree_attention_graph(update_stream, context, num_tokens) -> bool:
    if _EXTRA_CTX.is_draft_model:
        return False
    params = get_graph_params()
    if params is None:
        return False
    entries = params.attn_params.get(num_tokens, [])
    if not entries or not isinstance(entries[0], UnoTreeAttentionGraph):
        return False
    metadata = next(iter(context.attn_metadata.values()))
    with torch.npu.stream(update_stream):
        for entry, handle, event in zip(entries, params.handles[num_tokens], params.events[num_tokens]):
            kwargs = dict(entry.kwargs)
            kwargs["actual_seq_kvlen"] = metadata.seq_lens_list
            torch.npu.graph_task_update_begin(update_stream, handle)
            torch_npu.npu_fused_infer_attention_score_v2.out(**kwargs, workspace=entry.workspace, out=entry.output)
            torch.npu.graph_task_update_end(update_stream)
            event.record(update_stream)
    return True
