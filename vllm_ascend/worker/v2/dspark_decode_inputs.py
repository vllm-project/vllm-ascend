# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Read-only target inputs for uniform MiniMax DSpark decode batches."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.worker.utils import AttentionGroup


def supports_dspark_decode_reuse(config: "VllmConfig", *, sparse_offload_enabled: bool) -> bool:
    spec = config.speculative_config
    parallel = config.parallel_config
    return (
        spec is not None
        and spec.method == "dspark"
        and config.model_config.architecture == "MiniMaxM3SparseForCausalLM"
        and parallel.pipeline_parallel_size == 1
        and parallel.decode_context_parallel_size == 1
        and parallel.prefill_context_parallel_size == 1
        and not sparse_offload_enabled
        and not spec.enable_adaptive_verification
    )


class DSparkTargetAttention:
    """Keep KV group numbering while omitting independently built draft groups.

    Retain the source list to detect replacement after KV reinitialization. The
    full list still owns cache allocation, zeroing, capture and draft metadata.
    """

    def __init__(self, groups: list[list["AttentionGroup"]]):
        self.source = groups
        self.target = [[group for group in cache_groups if not group.kv_cache_spec.is_draft] for cache_groups in groups]

    def select(self, groups: list[list["AttentionGroup"]]) -> list[list["AttentionGroup"]]:
        return self.target if groups is self.source else groups


class DSparkDecodeIndexCache:
    """One immutable allocation per input; replace rather than overwrite on miss.

    Only request indices, logits boundaries and their expansion are retained.
    Consumers must treat the returned tensors as read-only. Dynamic query
    boundaries, sequence lengths, positions, block tables and KV are not cached.
    """

    def __init__(self):
        self._copies: dict[str, tuple[tuple[Any, ...], torch.Tensor]] = {}
        self._expanded: tuple[tuple[Any, ...], tuple[torch.Tensor, torch.Tensor]] | None = None

    def clear(self) -> None:
        self._copies.clear()
        self._expanded = None

    def copy(self, name: str, values: np.ndarray, copier: Callable[[np.ndarray], torch.Tensor]) -> torch.Tensor:
        key = (values.shape, values.dtype.str, values.tobytes())
        previous = self._copies.get(name)
        if previous is not None and previous[0] == key:
            return previous[1]
        result = copier(values)
        self._copies[name] = (key, result)
        return result

    def expand(
        self,
        idx_mapping: torch.Tensor,
        total_num_logits: int,
        cu_num_logits: torch.Tensor,
        decode_query_len: int,
        expander: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (self._copies["indices"][0], self._copies["logits"][0], total_num_logits, decode_query_len)
        if self._expanded is not None and self._expanded[0] == key:
            return self._expanded[1]
        result = expander(idx_mapping, total_num_logits, cu_num_logits, decode_query_len)
        self._expanded = (key, result)
        return result
