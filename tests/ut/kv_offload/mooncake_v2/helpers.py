# SPDX-License-Identifier: Apache-2.0
"""Shared builders for refactored Mooncake connector tests."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec, SlidingWindowSpec

from vllm_ascend.core.kv_cache_interface import AscendSFAIndexerCacheSpec
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.metadata import (
    MooncakePPTransferMetadata,
    MooncakeTPTransferMetadata,
    MooncakeTransferMetadata,
    MooncakeTransferMetadataGroups,
)


def make_full_spec(block_size: int = 16, num_kv_heads: int = 1) -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=8,
        head_size_v=8,
        dtype=torch.float16,
    )


def make_sliding_spec(block_size: int = 16) -> SlidingWindowSpec:
    return SlidingWindowSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.float16,
        sliding_window=64,
    )


def make_mamba_spec(block_size: int = 16) -> MambaSpec:
    return MambaSpec(
        block_size=block_size,
        shapes=((3, 16), (2, 4, 4)),
        dtypes=(torch.float16, torch.float16),
    )


def make_sfa_indexer_spec(
    block_size: int = 16,
    replication_size: int = 2,
) -> AscendSFAIndexerCacheSpec:
    return AscendSFAIndexerCacheSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.float16,
        sfa_dcp_replicated_indexer_size=replication_size,
    )


def make_transfer_metadata(
    *,
    engine_id: str = "engine-p",
    te_rpc_port: int = 9000,
    local_ip: str = "10.0.0.1",
    handshake_port: int = 5000,
    layer_names: list[str] | None = None,
    group_indices: list[int] | None = None,
    layer_block_sizes: list[int] | None = None,
    base_addrs: list[list[int]] | None = None,
    block_strides: list[list[int]] | None = None,
    block_lens: list[list[int]] | None = None,
    block_shapes: list[list[tuple[int, ...]]] | None = None,
    block_size_scales: list[list[int]] | None = None,
) -> MooncakeTransferMetadata:
    layer_names = layer_names or ["model.layers.0.self_attn"]
    num_layers = len(layer_names)
    return MooncakeTransferMetadata(
        engine_id=engine_id,
        te_rpc_port=te_rpc_port,
        block_size=16,
        num_blocks=32,
        layer_names=layer_names,
        layer_block_sizes=layer_block_sizes or [16] * num_layers,
        group_indices=group_indices or [0] * num_layers,
        kv_caches_base_addr=base_addrs or [[1000 + index * 1000] for index in range(num_layers)],
        block_strides=block_strides or [[128] for _ in range(num_layers)],
        block_lens=block_lens or [[128] for _ in range(num_layers)],
        block_shapes=block_shapes or [[(1, 16, 4)] for _ in range(num_layers)],
        block_size_scales=block_size_scales or [[1] for _ in range(num_layers)],
        local_ip=local_ip,
        handshake_port=handshake_port,
    )


def make_pp_metadata(
    *,
    layer_names: list[str] | None = None,
    layer_block_sizes: list[int] | None = None,
    block_shapes: list[list[tuple[int, ...]]] | None = None,
    block_strides: list[list[int]] | None = None,
    block_lens: list[list[int]] | None = None,
    block_size_scales: list[list[int]] | None = None,
    tp_base_addrs: dict[int, list[list[int]]] | None = None,
    tp_layer_indices: dict[int, list[int]] | None = None,
) -> MooncakePPTransferMetadata:
    layer_names = layer_names or ["model.layers.0.self_attn"]
    num_layers = len(layer_names)
    tp_base_addrs = tp_base_addrs or {0: [[5000 + index * 1000] for index in range(num_layers)]}
    tp_layer_indices = tp_layer_indices or {}
    return MooncakePPTransferMetadata(
        block_size=16,
        num_blocks=32,
        layer_names=layer_names,
        layer_block_sizes=layer_block_sizes or [16] * num_layers,
        group_indices=[0] * num_layers,
        block_strides=block_strides or [[128] for _ in range(num_layers)],
        block_lens=block_lens or [[128] for _ in range(num_layers)],
        block_shapes=block_shapes or [[(1, 16, 4)] for _ in range(num_layers)],
        block_size_scales=block_size_scales or [[1] for _ in range(num_layers)],
        metadata_by_tp_rank={
            tp_rank: MooncakeTPTransferMetadata(
                te_rpc_port=9000 + tp_rank,
                layer_indices=tp_layer_indices.get(tp_rank, list(range(num_layers))),
                kv_caches_base_addr=base_addrs,
                local_ip=f"10.0.0.{tp_rank + 1}",
                handshake_port=5000 + tp_rank,
            )
            for tp_rank, base_addrs in tp_base_addrs.items()
        },
    )


def make_metadata_groups(
    *,
    engine_id: str = "engine-p",
    tp_size: int = 1,
    use_kv_pp: bool = False,
    pp_metadata: MooncakePPTransferMetadata | None = None,
) -> MooncakeTransferMetadataGroups:
    return MooncakeTransferMetadataGroups(
        engine_id=engine_id,
        scheduler_host="10.0.0.10",
        scheduler_port=6000,
        pp_size=1,
        pcp_size=1,
        dcp_size=1,
        tp_size=tp_size,
        use_kv_pp=use_kv_pp,
        metadata_by_pp_rank={0: pp_metadata or make_pp_metadata()},
    )


def make_request(**overrides: Any) -> SimpleNamespace:
    values: dict[str, Any] = {
        "request_id": "request-0",
        "prompt_token_ids": list(range(32)),
        "prompt_embeds": None,
        "num_prompt_tokens": 32,
        "kv_transfer_params": {},
        "status": "running",
        "output_token_ids": [100],
        "_all_token_ids": list(range(32)),
        "max_tokens": 16,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def make_blocks(
    unhashed: tuple[list[int], ...] = ([10, 11],),
    full: tuple[list[int], ...] = ([1, 2, 10, 11],),
) -> MagicMock:
    blocks = MagicMock()
    blocks.get_unhashed_block_ids_all_groups.return_value = unhashed
    blocks.get_block_ids.return_value = full
    return blocks
