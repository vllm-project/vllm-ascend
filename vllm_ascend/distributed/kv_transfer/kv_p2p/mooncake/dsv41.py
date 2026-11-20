# SPDX-License-Identifier: Apache-2.0
"""DSV4.1 layout adapter for the existing Mooncake pull data plane."""

import hashlib
import json
import math
from dataclasses import asdict
from typing import Any

from vllm_ascend.distributed.kv_transfer.kv_p2p.dsv41_state import (
    STATE_PROTOCOL_VERSION,
    CachePlane,
    StateSnapshot,
    build_copy_plan,
    cache_kind,
    coalesce_copy_ranges,
    decode_planes,
    validate_ids,
    validate_pd_config,
)


def pool_capacity(config: Any, name: str) -> int:
    for tensor in config.kv_cache_tensors:
        if name in tensor.layers:
            if tensor.block_stride <= 0 or tensor.size % tensor.block_stride:
                raise ValueError(f"Invalid backing capacity for {name}")
            return tensor.size // tensor.block_stride
    raise ValueError(f"No backing for {name}")


def make_layout(worker: Any, metadata: Any) -> dict[str, Any]:
    """Describe actual persistent views, including separate sparse K/scales."""
    model_id = validate_pd_config(worker.vllm_config)
    if worker.ascend_config.kvpp_config.size != 1:
        raise ValueError("State PD does not support KVPP")
    planes = []
    for layer_index, name in enumerate(metadata.layer_names):
        group = worker.layer_name_to_group_index[name]
        spec = worker.kv_cache_specs[worker.layer_name_to_spec_index[name]]
        values = worker.kv_caches[name]
        caches = values if isinstance(values, (list, tuple)) else (values,)
        capacity = pool_capacity(worker.kv_cache_config, name)
        for cache_index, tensor in enumerate(caches):
            if tensor.shape[0] != capacity:
                raise ValueError(f"Unexpected tensor capacity for {name}")
            # The initial path supports contiguous rows and a possibly padded
            # page stride. It does not silently flatten arbitrary tensor views.
            expected = 1
            for size, stride in reversed(list(zip(tensor.shape[1:], tensor.stride()[1:]))):
                if size > 1 and stride != expected:
                    raise ValueError(f"Non-contiguous inner cache layout for {name}")
                expected *= size
            plane = CachePlane(
                name,
                cache_index,
                cache_kind(name),
                group,
                capacity,
                tensor.shape[1],
                math.prod(tensor.shape[2:]) * tensor.element_size(),
                tensor.stride(0) * tensor.element_size(),
                int(spec.tokens_per_state) if group == 0 else 1,
                str(tensor.dtype),
                tuple(tensor.shape[1:]),
            )
            plane.validate()
            if len(metadata.kv_caches_base_addr[layer_index]) != len(caches):
                raise ValueError("State PD requires independent metadata for every logical view")
            planes.append(asdict(plane))
    hf = worker.vllm_config.model_config.hf_config.to_dict()
    for key in ("_name_or_path", "_commit_hash", "transformers_version"):
        hf.pop(key, None)
    config = {
        "hf": hf,
        "model_id": model_id,
        "dtype": str(worker.vllm_config.model_config.dtype),
        "quantization": worker.vllm_config.model_config.quantization,
    }
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()
    return {
        "version": STATE_PROTOCOL_VERSION,
        "model_id": model_id,
        "fingerprint": fingerprint,
        "tp_size": worker.tp_size,
        "planes": planes,
    }


def transfer_addresses(
    snapshot: StateSnapshot, local: Any, remote: Any, rank: int, local_ids: Any, remote_ids: Any
) -> tuple[list[int], list[int], list[int]]:
    """Build all addresses first; caller submits one synchronous Mooncake READ."""
    plan = coalesce_copy_ranges(
        build_copy_plan(snapshot, local.state_layout, remote.state_layout, local_ids, remote_ids)
    )
    remote_tp = remote.metadata_by_pcp_rank[0].metadata_by_tp_rank[rank]
    local_indices = {n: i for i, n in enumerate(local.layer_names)}
    remote_indices = {n: i for i, n in enumerate(remote.layer_names)}
    sources, destinations, lengths = [], [], []
    for item in plan:
        src = remote_tp.kv_caches_base_addr[remote_indices[item.name]][item.cache_index]
        dst = local.kv_caches_base_addr[local_indices[item.name]][item.cache_index]
        if src <= 0 or dst <= 0:
            raise ValueError("State transfer has an invalid registered base address")
        sources.append(src + item.source_offset)
        destinations.append(dst + item.destination_offset)
        lengths.append(item.nbytes)
    return sources, destinations, lengths


def zero_state_destination(caches: dict[str, Any], local_ids: Any, layout: dict[str, Any]) -> None:
    """Clear only unused receiver ring contents before loading; never afterward."""
    destinations = []
    for plane in decode_planes(layout):
        if plane.kind == "history":
            continue
        destination = validate_ids(local_ids[plane.group], plane.capacity, 1, single=True)[0]
        values = caches[plane.name]
        tensor = values[plane.cache_index] if isinstance(values, (tuple, list)) else values
        destinations.append(tensor[destination])
    # Validate every state slot before submitting any reset kernel.
    for destination in destinations:
        destination.zero_()
