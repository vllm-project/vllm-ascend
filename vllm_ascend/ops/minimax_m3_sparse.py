# SPDX-License-Identifier: Apache-2.0
"""ACL graph helpers for MiniMax-M3 sparse attention on Ascend."""

from __future__ import annotations

import torch
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_ascend.compilation.breakable_acl_graph import (
    BreakableACLGraphCapture,
    breakable_acl_graph_eager,
)


def _in_breakable_graph_capture() -> bool:
    capture = BreakableACLGraphCapture.current()
    return capture is not None and capture._capturing


@breakable_acl_graph_eager
def minimax_m3_sparse_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    index_query: torch.Tensor,
    index_key: torch.Tensor,
    attn_output: torch.Tensor,
    layer_name: str,
) -> None:
    forward_context: ForwardContext = get_forward_context()

    attn_metadata = forward_context.attn_metadata
    if not isinstance(attn_metadata, dict):
        attn_output.zero_()
        return

    layer = forward_context.no_compile_layers[layer_name]
    layer._run_sparse_attention(
        query, key, value, index_query, index_key, attn_output
    )


def minimax_m3_sparse_forward_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    index_query: torch.Tensor,
    index_key: torch.Tensor,
    attn_output: torch.Tensor,
    layer_name: str,
) -> None:
    return


direct_register_custom_op(
    op_name="minimax_m3_sparse_forward",
    op_func=minimax_m3_sparse_forward,
    mutates_args=["attn_output"],
    fake_impl=minimax_m3_sparse_forward_fake,
    dispatch_key="PrivateUse1",
)
