# SPDX-License-Identifier: Apache-2.0
"""Structural capabilities shared by speculative-decoding proposers."""

from typing import Protocol, runtime_checkable

import torch
from vllm.v1.kv_cache_interface import KVCacheConfig


@runtime_checkable
class SupportsPerGroupBlockTables(Protocol):
    """A proposer whose draft layers span more than one KV-cache group."""

    def set_per_group_block_table(
        self,
        gid: int,
        block_table: torch.Tensor,
    ) -> None: ...


@runtime_checkable
class SupportsAttentionBackendInitialization(Protocol):
    """A proposer that initializes metadata builders from the target cache."""

    def initialize_attn_backend(
        self,
        kv_cache_config: KVCacheConfig,
        kernel_block_sizes: list[int] | None = None,
    ) -> None: ...


@runtime_checkable
class SupportsPerGroupKernelBlockSizes(Protocol):
    """A proposer that consumes one resolved kernel size per cache group."""

    def uses_per_group_kernel_block_sizes(self) -> bool: ...


@runtime_checkable
class SupportsCUDAGraphInitialization(Protocol):
    """A proposer with Ascend graph-dispatch initialization."""

    def initialize_cudagraph_keys(self, cudagraph_mode) -> None: ...
