# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import abstractmethod
from typing import Generic, TypeVar

import torch
from vllm.attention.backends.abstract import AttentionImpl, AttentionLayer


class AttentionMetadata:
    pass


T = TypeVar("T", bound=AttentionMetadata)


class DSAAttentionImpl(AttentionImpl[T], Generic[T]):

    @abstractmethod
    def __init__(
        self,
        dim: int,
        n_heads: int,
        scale: float,
        n_local_heads: int,
        q_lora_rank: int,
        o_lora_rank: int,
        head_dim: int,
        rope_head_dim: int | None,
        nope_head_dim: int,
        n_groups: int,
        n_local_groups: int,
        window_size: int,
        compress_ratio: int,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        layer: AttentionLayer,
        hidden_states_or_cq: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: T,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError
