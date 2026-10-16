# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Thin facade for recipes DSA offload operators.

The DSA kernels are compiled and installed from ``cann-recipes-infer``. This
module owns only the stable Python call contract; it does not implement a
fallback gather kernel or duplicate operator input validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


DSA_BLOCK_SIZE = 128
DSA_TOPK = 2048


def get_dsa_config_value(
    model_config: Any, attribute: str, default: Any = None
) -> Any:
    """Read a DSA model attribute from the model's config variants.

    Hugging Face configurations are exposed through either ``hf_config`` or
    ``hf_text_config`` depending on the model adapter.  Keeping this lookup in
    one place prevents the runner, attention backend, and cache planner from
    silently using different variants.
    """

    for config in (
        getattr(model_config, "hf_config", None),
        getattr(model_config, "hf_text_config", None),
        model_config,
    ):
        if config is None:
            continue
        value = getattr(config, attribute, None)
        if value is not None:
            return value
    return default


def get_dsa_raw_seq(speculative_config: Any) -> int:
    """Return the number of query rows represented by one scheduler row."""

    num_speculative_tokens = getattr(
        speculative_config, "num_speculative_tokens", None
    )
    return (
        num_speculative_tokens + 1
        if isinstance(num_speculative_tokens, int)
        else 1
    )


@dataclass(frozen=True)
class DsaOffloadConfig:
    """Static DSA bucket attributes used by eager and ACL-graph paths."""

    raw_seq: int = 1
    topk: int = DSA_TOPK
    selection_block_size: int = DSA_BLOCK_SIZE
    compact_layout: int = 1

    def __post_init__(self) -> None:
        """Validate the currently supported recipes block contract.

        The installed DSA recipes kernels currently operate on complete
        selection blocks.  Keep this restriction at the offload operator
        boundary so every plan/serve/install call applies the same contract.
        This is intentionally local to DSA offload and does not constrain
        regular SFA attention.
        """

        if self.raw_seq <= 0:
            raise ValueError("DSA raw_seq must be positive")
        if self.topk <= 0:
            raise ValueError("DSA topk must be positive")
        if self.selection_block_size <= 0:
            raise ValueError("DSA selection_block_size must be positive")
        if self.topk % self.selection_block_size != 0:
            raise ValueError(
                "DSA topk must be an integer multiple of selection_block_size "
                "for the installed recipes kernels"
            )

    @property
    def selection_blocks_per_row(self) -> int:
        return (
            self.topk + self.selection_block_size - 1
        ) // self.selection_block_size

    def selection_rows(self, batch_size: int) -> int:
        if batch_size < 0:
            raise ValueError("DSA batch_size must be non-negative")
        return batch_size * self.raw_seq

    def selection_block_count(self, batch_size: int) -> int:
        return self.selection_rows(batch_size) * self.selection_blocks_per_row


def _load_custom_ops_package() -> None:
    """Import the independently built recipes operator package.

    Importing ``custom_ops`` loads its compiled extension and TorchAir
    converters. vLLM Ascend deliberately does not register, redefine, or
    provide a fallback implementation for any DSA operator.
    """

    try:
        import custom_ops  # noqa: F401
    except ImportError as error:
        raise RuntimeError(
            "The recipes custom_ops package is not installed; build and install "
            "it from cann-recipes-infer/ops/ascendc/torch_ops_extension first"
        ) from error


def _custom_op(name: str) -> Any:
    _load_custom_ops_package()
    custom = getattr(torch.ops, "custom", None)
    op = getattr(custom, name, None) if custom is not None else None
    if op is None:
        raise RuntimeError(
            f"torch.ops.custom.{name} is unavailable; install the recipes DSA extension "
            "built from cann-recipes-infer/ops/ascendc"
        )
    return op


def dsa_plan(
    selection_topk_indices: torch.Tensor,
    full_kv_actual_seq: torch.Tensor,
    pool_ids: torch.Tensor,
    id_to_slot: torch.Tensor,
    lru_counter: torch.Tensor,
    *,
    config: DsaOffloadConfig | None = None,
    group_id: int = 0,
    owner_layer: int = 0,
    group_kind: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    config = config or DsaOffloadConfig()
    result = _custom_op("dsa_plan")(
        selection_topk_indices,
        full_kv_actual_seq,
        pool_ids,
        id_to_slot,
        lru_counter,
        raw_seq=config.raw_seq,
        group_id=group_id,
        owner_layer=owner_layer,
        group_kind=group_kind,
    )
    if not isinstance(result, tuple) or len(result) != 3:
        raise RuntimeError("dsa_plan must return (plan, install_records, selection_kv_actual_seq)")
    return result


def dsa_serve(
    plan: torch.Tensor,
    full_kv_cache: torch.Tensor,
    full_k_rope: torch.Tensor,
    pool_kv_cache: torch.Tensor,
    pool_k_rope: torch.Tensor,
    selection_kv_cache: torch.Tensor,
    selection_k_rope: torch.Tensor,
    *,
    full_kv_block_table: torch.Tensor | None = None,
    full_kv_actual_seq: torch.Tensor | None = None,
    config: DsaOffloadConfig | None = None,
) -> None:
    config = config or DsaOffloadConfig()
    if full_kv_block_table is None:
        raise ValueError("full_kv_block_table is required for layerwise Host KV offload")
    if full_kv_cache.ndim == 4:
        full_kv_cache = full_kv_cache.squeeze(2)
    if full_k_rope.ndim == 4:
        full_k_rope = full_k_rope.squeeze(2)
    op = _custom_op("dsa_serve")
    # Recipes' PA form appends the physical block table after the selection
    # outputs. Keep it keyworded so future optional inputs cannot shift it.
    op(
        plan,
        full_kv_cache,
        full_k_rope,
        pool_kv_cache,
        pool_k_rope,
        selection_kv_cache,
        selection_k_rope,
        full_kv_block_table=full_kv_block_table,
        raw_seq=config.raw_seq,
        topk=config.topk,
        selection_block_size=config.selection_block_size,
        compact_layout=config.compact_layout,
    )
    return None


def dsa_install(
    install_records: torch.Tensor,
    selection_kv_cache: torch.Tensor,
    selection_k_rope: torch.Tensor,
    selection_kv_block_table: torch.Tensor,
    pool_kv_cache: torch.Tensor,
    pool_k_rope: torch.Tensor,
    pool_ids: torch.Tensor,
    id_to_slot: torch.Tensor,
    lru_counter: torch.Tensor,
    *,
    config: DsaOffloadConfig | None = None,
    metadata_update: int = 1,
) -> None:
    config = config or DsaOffloadConfig()
    _custom_op("dsa_install")(
        install_records,
        selection_kv_cache,
        selection_k_rope,
        selection_kv_block_table,
        pool_kv_cache,
        pool_k_rope,
        pool_ids,
        id_to_slot,
        lru_counter,
        raw_seq=config.raw_seq,
        topk=config.topk,
        selection_block_size=config.selection_block_size,
        metadata_update=metadata_update,
    )
    return None
