# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opaque orchestration whose device work is captured by FULL ACLGraph."""

import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import get_forward_context
from vllm.utils.torch_utils import direct_register_custom_op


def engram_elastic_begin(
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    query: torch.Tensor,
    lookback: torch.Tensor,
    slots: torch.Tensor,
    blocks: torch.Tensor,
    valid: torch.Tensor,
    dead: torch.Tensor,
    lookback_dead: torch.Tensor,
    layer_name: str,
    layers: int,
    columns: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    context = get_forward_context()
    if context.cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE:
        raise RuntimeError("ElasticBuffer Engram overlap requires eager or FULL_DECODE_ONLY, not PIECEWISE")
    layer = context.no_compile_layers[layer_name]
    return layer.executor.begin(input_ids, positions, query, lookback, slots, blocks, valid, dead, lookback_dead)


def engram_elastic_begin_fake(
    input_ids, positions, query, lookback, slots, blocks, valid, dead, lookback_dead, layer_name, layers, columns
):
    return (
        input_ids.new_empty((input_ids.shape[0], layers, columns), dtype=torch.int32),
        input_ids.new_empty(input_ids.shape[0], dtype=torch.bool),
    )


def engram_elastic_project(
    source: torch.Tensor, dependency: torch.Tensor, layer_name: str, hc_mult: int, hidden_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    layer = get_forward_context().no_compile_layers[layer_name]
    return layer.executor.project(layer, source)


def engram_elastic_project_fake(source, dependency, layer_name, hc_mult, hidden_size):
    kv = source.new_empty((source.shape[0], (hc_mult + 1) * hidden_size), dtype=torch.bfloat16)
    key, value = kv.split([hc_mult * hidden_size, hidden_size], dim=-1)
    return (
        key.float().view(source.shape[0], hc_mult, hidden_size),
        value,
        source.new_empty((hc_mult, hidden_size), dtype=torch.float32),
    )


def engram_elastic_consume(
    hidden: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    weight: torch.Tensor,
    mask: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    layer = get_forward_context().no_compile_layers[layer_name]
    return layer.executor.consume(layer, hidden, key, value, weight, mask)


def engram_elastic_consume_fake(hidden, key, value, weight, mask, layer_name):
    return torch.empty_like(hidden)


direct_register_custom_op(
    op_name="engram_elastic_begin",
    op_func=engram_elastic_begin,
    mutates_args=[],
    fake_impl=engram_elastic_begin_fake,
    dispatch_key="PrivateUse1",
)
direct_register_custom_op(
    op_name="engram_elastic_project",
    op_func=engram_elastic_project,
    mutates_args=[],
    fake_impl=engram_elastic_project_fake,
    dispatch_key="PrivateUse1",
)
direct_register_custom_op(
    op_name="engram_elastic_consume",
    op_func=engram_elastic_consume,
    mutates_args=[],
    fake_impl=engram_elastic_consume_fake,
    dispatch_key="PrivateUse1",
)
