# SPDX-License-Identifier: Apache-2.0
"""Optional deployment metadata; recovery policy is owned by the subscriber."""

import json

from vllm.distributed.kv_events import BlockStored
from vllm.v1.kv_cache_interface import MambaSpec, UniformTypeKVCacheSpecs

from vllm_ascend.core.kv_cache_interface import AscendIndexerKPoolTailSpec, is_prefix_cacheable


class BlockStoredWithLayout(BlockStored, tag="BlockStored"):
    kv_cache_layout: str | None = None


def _group_object_namespaces(worker, group_id: int, ranks: int) -> dict:
    if worker.use_layerwise:
        # Use the same bound key builder as PUT and scheduler lookup. Layerwise
        # puts the TP coordinate after the hash, unlike the whole-key path.
        marker = "{block_hash}"
        keys = worker.layerwise_keys.make_hit_check_keys(group_id, marker, ranks)
        namespaces = [key.split(marker) for key in keys]
        if any(len(parts) != 2 for parts in namespaces):
            raise ValueError("Layerwise object keys must contain exactly one block hash")
        return {
            "key_prefixes": [prefix for prefix, _ in namespaces],
            "key_suffixes": [suffix for _, suffix in namespaces],
        }
    prefix = worker.token_database._get_key_prefix(group_id)
    local_rank = worker.metadata[group_id].head_or_tp_rank
    return {
        "key_prefixes": [
            prefix.replace(f"@head_or_tp_rank:{local_rank}@", f"@head_or_tp_rank:{rank}@", 1) for rank in range(ranks)
        ],
    }


def build_deployment_layout(worker) -> str | None:
    """Describe all required groups/TP objects from one resolved Mooncake pool.

    Unsupported layouts remain undeclared instead of claiming completeness.
    In particular, only the known GLM pool-aligned transient tail is omitted.
    """
    if (
        not worker.use_mamba
        or (worker.use_layerwise and not getattr(worker, "use_block_key_layerwise", False))
        or worker.backend_name != "mooncake"
        or worker.pp_size != 1
        or worker.pcp_size != 1
        or worker.dcp_size != 1
        or worker.kv_cache_config is None
    ):
        return None
    groups = []
    transient_alignment = 1
    for group_id, group in enumerate(worker.kv_cache_config.kv_cache_groups):
        spec = group.kv_cache_spec
        specs = list(spec.kv_cache_specs.values()) if isinstance(spec, UniformTypeKVCacheSpecs) else [spec]
        if not is_prefix_cacheable(spec):
            if not all(isinstance(inner, AscendIndexerKPoolTailSpec) for inner in specs):
                return None
            ratios = {inner.compress_ratio for inner in specs}
            if ratios != {4}:
                return None
            transient_alignment = 4
            continue
        kind, _ = worker.kv_event_group_metadata[group_id]
        if kind not in ("mamba", "full_attention", "mla_attention"):
            return None
        ranks = worker.tp_size if worker.group_uses_align_state[group_id] else worker.tp_size // worker.put_step
        entry = {
            "group_idx": group_id,
            "kind": kind,
            "block_tokens": worker.grouped_block_size[group_id],
            **_group_object_namespaces(worker, group_id, ranks),
        }
        if kind == "mamba":
            if not all(isinstance(inner, MambaSpec) and inner.mamba_cache_mode == "align" for inner in specs):
                return None
            entry["mamba_cache_mode"] = "align"
        groups.append(entry)
    return json.dumps(
        {
            "format": "ascend-store-v1",
            **({"object_key_format": "mooncake-layerwise-v1"} if worker.use_layerwise else {}),
            "model_name": worker.model_name,
            "hash_unit_tokens": worker.hash_block_size,
            "alignment_tokens": worker.cache_transfer_granularity,
            "storage_domain": "ascend://" + worker.m_store.config.master_server_address,
            "transient_alignment_tokens": transient_alignment,
            "groups": groups,
        },
        separators=(",", ":"),
        sort_keys=True,
    )


def attach_deployment_layout(events: list[BlockStored], declaration: str) -> list[BlockStored]:
    # Unknown fields are ignored by upstream msgspec decoders. Keep the exact
    # BlockStored tag and every original field, including future additions.
    return [
        BlockStoredWithLayout(
            **{name: getattr(event, name) for name in BlockStored.__struct_fields__},
            kv_cache_layout=declaration,
        )
        for event in events
    ]


def deployment_announcement(declaration: str, block_size: int) -> BlockStored:
    """A zero-block announcement declares layout, never data residency.

    It is needed even when all Store keys already exist or a write fails.
    No native BlockStored fields or event tags are reinterpreted.
    """
    return BlockStoredWithLayout(
        block_hashes=[],
        parent_block_hash=None,
        token_ids=[],
        block_size=block_size,
        lora_id=None,
        medium="cpu",
        lora_name=None,
        kv_cache_layout=declaration,
    )
