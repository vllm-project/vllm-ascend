# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Indexer QW prologue import and one-time weight preparation."""

from functools import cache
from importlib import import_module

import torch_npu


@cache
def get_indexer_prologue_qw():
    return import_module("ops.indexer_prologue_qw_vllm").indexer_prologue_qw


def prepare_indexer_query_weights(layer):
    """Consume the checkpoint NK weights and expanded NK paired scales."""
    layer.weight.data = torch_npu.npu_format_cast(layer.weight.data, 29)
    # DS post-load already produces the exact [N,K/64,2] scale layout.


def prepare_indexer_weights(indexer):
    """Pack QW weights after loading; preserve the native K weights."""
    indexer.weights_proj.weight.data = torch_npu.npu_format_cast(indexer.weights_proj.weight.data, 29)
