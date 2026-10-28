"""CPU Tensor fixtures: real storage, resolved topology facts, no Worker or Backend."""

import ctypes
import inspect
from dataclasses import dataclass
from types import SimpleNamespace

import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import KeyMetadata


@dataclass
class MappingCase:
    topology: SimpleNamespace
    registration_arrays: tuple[dict, ...]
    caches: dict
    layout: str
    head_slices: int

    def bind(self, module, *, group_id=None):
        options = {} if group_id is None else {"group_id": group_id}
        # Snapshot compatibility belongs only to the paired benchmark fixture.
        bind = getattr(module, "bind_mapping", module.BulkMapping)
        binding = bind(
            self.topology, *self.registration_arrays, kv_cache_layout=self.layout, kv_caches=self.caches, **options
        )
        return module.BulkMapping(binding) if hasattr(module, "bind_mapping") else binding

    def bind_layers(self, module, *, group_id=None, **options):
        if "topology" not in inspect.signature(module.LayerwiseMapping).parameters:
            # Compare complete initialization, including the historical local
            # binding, instead of timing only the old Layerwise view constructor.
            bind = getattr(module, "bind_mapping", module.BulkMapping)
            bound = bind(
                self.topology,
                *self.registration_arrays,
                group_id=group_id,
                kv_cache_layout=self.layout,
                kv_caches=self.caches,
            )
            return module.LayerwiseMapping(bound, **options)
        return module.LayerwiseMapping(
            self.topology,
            *self.registration_arrays,
            group_id=group_id,
            kv_cache_layout=self.layout,
            kv_caches=self.caches,
            **options,
        )


def make_case(
    *,
    layers=(0, 1, 2, 3),
    head_slices=None,
    pp_layer_counts=None,
    layout="NHD",
    block_size=8,
    stored_tokens=None,
    heads=4,
    vector=3,
    padding=0,
    kernels=1,
    blocks=3,
    tp_rank=0,
    replicas=1,
    pp_rank=0,
    pp_size=2,
    key_rank_count=4,
    fill=True,
):
    tokens_by_layer = stored_tokens or (block_size // kernels,) * len(layers)
    caches, entry_addresses, entry_block_payload_bytes, entry_block_strides_bytes, layer_entry_boundaries = (
        {},
        [],
        [],
        [],
        [0],
    )
    layer_facts = []
    for layer, tokens in zip(layers, tokens_by_layer, strict=True):
        name = f"layer{layer}"
        layer_facts.append(SimpleNamespace(physical_layer_id=layer, layer_names=(name,)))
        shape = (blocks * kernels, tokens, heads, vector + padding)
        if layout in ("HND", "LBHNC"):
            shape = (blocks * kernels, heads, tokens, vector + padding)
        tensors = []
        for kind in range(2):
            storage = torch.empty(shape, dtype=torch.int32)
            if fill:
                storage.copy_(torch.arange(storage.numel()).reshape(shape) + layer * 100000 + kind * 10000)
            tensor = storage[..., :vector]
            tensors.append(tensor)
            entry_addresses.append(tensor.data_ptr())
            entry_block_payload_bytes.append(tensor[0].numel() * tensor.element_size() * kernels)
            entry_block_strides_bytes.append(tensor.stride(0) * tensor.element_size() * kernels)
        # Match registration's empty-entry filtering and preserve K/V order.
        caches[name] = (tensors[0], torch.empty((0, 1, 1, 1)), tensors[1])
        layer_entry_boundaries.append(len(entry_addresses))
    group = SimpleNamespace(
        group_id=3,
        block_size=block_size,
        kv_cache_spec=FullAttentionSpec(block_size=block_size, num_kv_heads=heads, head_size=vector, dtype=torch.int32),
        uses_align_state=False,
        layers=tuple(layer_facts),
        key_metadata=KeyMetadata("mapping-test", tp_rank // replicas, 0, pp_rank, 3),
    )
    topology = SimpleNamespace(
        config=SimpleNamespace(
            pp_partition=SimpleNamespace(global_layer_count=None, target_layer_counts=pp_layer_counts),
            cp_partition=SimpleNamespace(object_shard_count=1, shard_alignment_bytes=1, dcp_size=1, pcp_size=1),
            hash_block_size=block_size,
            pp_size=pp_size,
            tp_size=key_rank_count // (head_slices or 1) * replicas,
            put_step=replicas,
            tp_partition=SimpleNamespace(
                tp_mismatch=head_slices is not None,
                key_rank_count=key_rank_count,
                key_slices_per_rank=head_slices or 1,
            ),
        ),
        stage_layer_start=0,
        stage_layer_count=len(layers),
        object_shard_index=0,
        transfer_groups=(group,),
        groups=(group,),
        pcp_rank=0,
        tp_rank=tp_rank,
    )
    return MappingCase(
        topology,
        tuple(
            {3: values}
            for values in (
                entry_addresses,
                entry_block_payload_bytes,
                entry_block_strides_bytes,
                layer_entry_boundaries,
            )
        ),
        caches,
        layout,
        head_slices or 1,
    )


def make_hybrid_case(
    *, attention_layers=(0, 2), state_layers=(1, 3), pp_layer_counts=None, tp_rank=0, pp_rank=0, pp_size=2
):
    """Interleaved groups, compressed attention and padded mixed-dtype state pages.

    Attention ranks share keys in pairs; state ranks retain distinct identities.
    Group 1 is private, and transfer order (9, 3) differs from original group IDs.
    """
    case = make_case(
        layers=attention_layers,
        block_size=64,
        stored_tokens=(4,) * len(attention_layers),
        heads=1,
        tp_rank=tp_rank,
        replicas=2,
        key_rank_count=2,
        pp_layer_counts=pp_layer_counts,
        pp_rank=pp_rank,
        pp_size=pp_size,
    )
    attention = case.topology.transfer_groups[0]
    attention.key_metadata.cache_family = "c16"
    attention.kv_cache_spec = AscendMLAAttentionSpec(
        block_size=64, num_kv_heads=1, head_size=3, dtype=torch.int32, tokens_per_state=16
    )
    entry_addresses, entry_block_payload_bytes, entry_block_strides_bytes, layer_entry_boundaries = [], [], [], [0]
    layers = []
    for layer in state_layers:
        names = (f"layer{layer}.conv", f"layer{layer}.ssm")
        layers.append(SimpleNamespace(physical_layer_id=layer, layer_names=names))
        for component, (name, dtype) in enumerate(zip(names, (torch.int16, torch.float32), strict=True)):
            # Padding is only between pages: each registered payload is contiguous.
            storage = torch.full((3, 11), -1, dtype=dtype)
            tensor = storage[:, :6].view(3, 2, 3)
            tensor.copy_(torch.arange(18).reshape(3, 2, 3) + layer * 100 + component * 10 + tp_rank)
            case.caches[name] = (torch.empty(0), tensor)
            entry_addresses.append(tensor.data_ptr())
            entry_block_payload_bytes.append(tensor[0].numel() * tensor.element_size())
            entry_block_strides_bytes.append(tensor.stride(0) * tensor.element_size())
        layer_entry_boundaries.append(len(entry_addresses))
    state = SimpleNamespace(
        group_id=9,
        block_size=32,
        kv_cache_spec=MambaSpec(
            block_size=32, shapes=((2, 3), (2, 3)), dtypes=(torch.int16, torch.float32), page_size_padded=66
        ),
        uses_align_state=True,
        layers=tuple(layers),
        key_metadata=KeyMetadata("mapping-test", tp_rank, 0, pp_rank, 9, cache_family="mamba"),
    )
    for array, values in zip(
        case.registration_arrays,
        (entry_addresses, entry_block_payload_bytes, entry_block_strides_bytes, layer_entry_boundaries),
        strict=True,
    ):
        array[9] = values
    case.topology.groups = (SimpleNamespace(group_id=1), attention, state)
    case.topology.transfer_groups = (state, attention)
    case.topology.config.hash_block_size = 8
    return case


def read_bytes(addresses, lengths):
    return b"".join(ctypes.string_at(address, length) for address, length in zip(addresses, lengths, strict=True))


def expected_head_bytes(case, layers, head_slice, block_id):
    """Independent complete tensor head slices in the existing NHD/HND wire order."""
    payload = []
    for layer in layers:
        layer_index = next(
            index
            for index, item in enumerate(case.topology.transfer_groups[0].layers)
            if item.physical_layer_id == layer
        )
        entry = case.registration_arrays[3][3][layer_index]
        for tensor in case.caches[f"layer{layer}"]:
            if not tensor.numel():
                continue
            head_major = case.layout in ("HND", "LBHNC")
            head_axis = 1 if head_major else 2
            heads = tensor.shape[head_axis] // case.head_slices
            kernels = case.registration_arrays[2][3][entry] // (tensor.stride(0) * tensor.element_size())
            for kernel in range(kernels):
                first_head = head_slice * heads
                block = tensor[block_id * kernels + kernel]
                selected = (
                    block[first_head : first_head + heads] if head_major else block[:, first_head : first_head + heads]
                )
                payload.append(selected.contiguous().numpy().tobytes())
    return b"".join(payload)


def expected_group_bytes(case, group_id, layers, block_id):
    """Read complete Tensor pages without consulting mapping's byte segments."""
    group = next(group for group in case.topology.transfer_groups if group.group_id == group_id)
    return b"".join(
        tensor[block_id].contiguous().numpy().tobytes()
        for layer in group.layers
        if layer.physical_layer_id in layers
        for name in layer.layer_names
        for tensor in case.caches[name]
        if tensor.numel()
    )
