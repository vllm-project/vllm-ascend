"""Check Tensor mappings against bytes, upstream cache specs and wire names."""

import ctypes
from dataclasses import replace
from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
    compute_layer_kv_cache_shape_bytes,
)

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import KeyMetadata
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.mapping import (
    build_gva_remote,
    build_key_range_remote,
    build_mappings,
    local_addresses,
    local_lengths,
    map_keys,
)

_ALL_SEGMENTS = slice(None)


def make_case(*, layout="NHD", layers=(0, 1, 2, 3), head_partition_count=2, pp_counts=None, kernels=1, padding=0):
    group = SimpleNamespace(
        group_id=3,
        key_metadata=KeyMetadata("tensor-test", 1, 1, 0, 3),
        kv_cache_spec=FullAttentionSpec(block_size=2 * kernels, num_kv_heads=4, head_size=3, dtype=torch.int32),
        layers=tuple(SimpleNamespace(physical_layer_id=layer, layer_names=(f"layer{layer}",)) for layer in layers),
    )
    topology = SimpleNamespace(
        config=SimpleNamespace(
            tp_partition=SimpleNamespace(
                tp_mismatch=head_partition_count > 1, key_slices_per_rank=head_partition_count
            ),
            pp_partition=SimpleNamespace(global_layer_count=None, target_layer_counts=pp_counts),
            cp_partition=SimpleNamespace(object_shard_count=1, shard_alignment_bytes=1, dcp_size=2, pcp_size=2),
        ),
        transfer_groups=(group,),
        stage_layer_start=0,
        stage_layer_count=len(layers),
        object_shard_index=0,
        pcp_rank=1,
    )
    caches = {}
    for layer in layers:
        shape = (3 * kernels, 2, 4, 3 + padding)
        if layout in ("HND", "LBHNC"):
            shape = (3 * kernels, 4, 2, 3 + padding)
        components = []
        for component in range(2):
            storage = torch.arange(np.prod(shape), dtype=torch.int32).reshape(shape)
            storage += layer * 10000 + component * 1000
            components.append(storage[..., :3])
        caches[f"layer{layer}"] = (components[0], torch.empty(0), components[1])
    return topology, caches


def expected_bytes(caches, layers, head, block, *, layout="NHD", head_partition_count=2, kernels=1):
    result = []
    head_axis = 1 if layout in ("HND", "LBHNC") else 2
    for layer in layers:
        for tensor in caches[f"layer{layer}"]:
            if not tensor.numel():
                continue
            selected = tensor[block * kernels : (block + 1) * kernels]
            heads = tensor.shape[head_axis] // head_partition_count
            result.append(selected.narrow(head_axis, head * heads, heads).contiguous().numpy().tobytes())
    return b"".join(result)


def read_local(segments, block, segment_range=_ALL_SEGMENTS):
    addresses = local_addresses(segments, (block,), segment_range)[0]
    lengths = np.broadcast_to(local_lengths(segments, segment_range), addresses.shape)
    return b"".join(
        ctypes.string_at(int(address), int(length)) for address, length in zip(addresses, lengths, strict=True)
    )


@pytest.mark.parametrize("layout", ("NHD", "HND", "LBHNC"))
@pytest.mark.parametrize("head_partition_count", (1, 2))
@pytest.mark.parametrize("pp_counts", (None, (1, 0, 3)))
def test_partition_keys_and_layer_payloads(layout, head_partition_count, pp_counts):
    kernels, padding = 2, 1
    topology, caches = make_case(
        layout=layout, head_partition_count=head_partition_count, pp_counts=pp_counts, kernels=kernels, padding=padding
    )
    keys, local, layer_slices = build_mappings(topology, caches, group_id=3, kv_cache_layout=layout, layerwise=True)
    remote = build_gva_remote(topology, local, layer_slices)
    expected_pp = ((0, (0, 1, 2, 3)),) if pp_counts is None else ((0, (0,)), (2, (1, 2, 3)))
    identities = [(pp, 1, head_partition_count + head) for pp, _ in expected_pp for head in range(head_partition_count)]
    assert list(keys) == list(local) == identities
    assert list(remote) == list(layer_slices) == [pp for pp, _ in expected_pp]
    for pp, layers in expected_pp:
        for head in range(head_partition_count):
            identity = (pp, 1, head_partition_count + head)
            prefix = (
                f"tensor-test@dcp:1@head_or_tp_rank:{head_partition_count + head}@pp_rank:{pp}"
                "@group:3@cache_role:kv@cache_family:default@"
            )
            assert keys[identity] == (prefix, "")
            assert map_keys(keys[identity], (b"a", "hash", b"a")) == (prefix + "61", prefix + "hash", prefix + "61")
            assert map_keys(keys[identity], ()) == ()
            segments = local[identity]
            rows = local_addresses(segments, (2, 0, 2))
            np.testing.assert_array_equal(rows[0], rows[2])
            assert local_addresses(segments, ()).shape == (0, len(segments["bases"]))
            for block in (2, 0, 2):
                expected = expected_bytes(
                    caches,
                    layers,
                    head,
                    block,
                    layout=layout,
                    head_partition_count=head_partition_count,
                    kernels=kernels,
                )
                assert read_local(segments, block) == expected
                target = remote[pp]
                assert target["size_bytes"] == len(expected)
                result = bytearray(target["size_bytes"])
                for address, length, offset in zip(
                    local_addresses(segments, (block,))[0],
                    np.broadcast_to(local_lengths(segments), target["offsets"].shape),
                    target["offsets"],
                    strict=True,
                ):
                    result[int(offset) : int(offset + length)] = ctypes.string_at(int(address), int(length))
                assert result == expected
                assert tuple(layer_slices[pp]) == layers
                for layer, selected in layer_slices[pp].items():
                    assert read_local(segments, block, selected) == expected_bytes(
                        caches,
                        (layer,),
                        head,
                        block,
                        layout=layout,
                        head_partition_count=head_partition_count,
                        kernels=kernels,
                    )


def test_external_builder_preserves_partition_identity_and_source_metadata():
    topology, caches = make_case(pp_counts=(1, 0, 3))
    seen = []

    def builder(metadata):
        seen.append(metadata)
        return (
            f"{metadata.model_name}@{metadata.kv_cache_group_id}@pp{metadata.pp_rank}@",
            f"@{metadata.head_or_tp_rank}",
        )

    keys, _, _ = build_mappings(topology, caches, group_id=3, key_builder=builder)
    assert [(meta.pp_rank, meta.dcp_rank, meta.head_or_tp_rank) for meta in seen] == list(keys)
    assert topology.transfer_groups[0].key_metadata.pp_rank == 0
    assert topology.transfer_groups[0].key_metadata.head_or_tp_rank == 1
    assert map_keys(keys[(2, 1, 3)], (b"a", "hash")) == ("tensor-test@3@pp2@61@3", "tensor-test@3@pp2@hash@3")


@pytest.mark.parametrize("vary_stride", (False, True))
@pytest.mark.parametrize("vary_length", (False, True))
def test_uniform_and_mixed_geometry_preserves_layer_bytes(vary_stride, vary_length):
    topology, caches = make_case(layers=(0, 1), head_partition_count=1)
    group = topology.transfer_groups[0]
    specs = {}
    for layer in (0, 1):
        width = 2 if layer and vary_length else 3
        block_stride = 40 if layer and vary_stride else 32
        tensor = torch.empty_strided((3, 2, 4, width), (block_stride, 4 * width, width, 1), dtype=torch.int32)
        tensor.copy_(torch.arange(tensor.numel()).reshape(tensor.shape) + layer * 1000)
        caches[f"layer{layer}"] = (tensor,)
        specs[f"layer{layer}"] = replace(group.kv_cache_spec, head_size=width, head_size_v=width)
    group.kv_cache_spec = UniformTypeKVCacheSpecs(block_size=2, kv_cache_specs=specs)
    _, local, layer_slices = build_mappings(topology, caches, group_id=3, layerwise=True)
    segments = local[(0, 1, 1)]
    for field, varying in (("block_strides", vary_stride), ("lengths", vary_length)):
        array = segments[field]
        assert len(array) == (2 if varying else 1)
    assert local_lengths(segments).tolist() == ([96, 64] if vary_length else [96])
    for layer, selected in layer_slices[0].items():
        for block in (0, 2):
            assert read_local(segments, block, selected) == caches[f"layer{layer}"][0][block].numpy().tobytes()
    remote = build_key_range_remote(local)[0]
    assert remote["offsets"].tolist() == [0, 96]
    assert remote["size_bytes"] == (160 if vary_length else 192)


@pytest.mark.parametrize("layout", ("NHD", "HND"))
def test_key_range_and_gva_have_distinct_remote_placement(layout):
    topology, caches = make_case(layout=layout, layers=(3, 4), kernels=2, padding=1)
    topology.config.pp_partition.global_layer_count = 7
    topology.stage_layer_start = 3
    topology.config.cp_partition.object_shard_count = 2
    topology.config.cp_partition.shard_alignment_bytes = 1024
    topology.object_shard_index = 1
    _, local, layer_slices = build_mappings(topology, caches, group_id=3, kv_cache_layout=layout, layerwise=True)
    for mode in ("key_range", "gva"):
        remote = (
            build_key_range_remote(local) if mode == "key_range" else build_gva_remote(topology, local, layer_slices)
        )
        size, origin = (384, 0) if mode == "key_range" else (4096, 2624)
        # 192 bytes/layer; 7 global layers reserve 1344 bytes, aligned to a 2048-byte CP slot.
        for identity, segments in local.items():
            target = remote[identity[0]]
            assert target["size_bytes"] == size
            assert target["offsets"][layer_slices[identity[0]][3]][0] == origin
            assert target["offsets"][layer_slices[identity[0]][4]][0] == origin + 192
            result = bytearray(b"\xcc" * size)
            for address, length, offset in zip(
                local_addresses(segments, (1,))[0],
                np.broadcast_to(local_lengths(segments), target["offsets"].shape),
                target["offsets"],
                strict=True,
            ):
                result[int(offset) : int(offset + length)] = ctypes.string_at(int(address), int(length))
            assert result[origin : origin + 384] == expected_bytes(
                caches, (3, 4), identity[2] - 2, 1, layout=layout, kernels=2
            )
            assert result[:origin] == b"\xcc" * origin
            assert result[origin + 384 :] == b"\xcc" * (size - origin - 384)


@pytest.mark.parametrize("layout", ("NHD", "HND", "LBHNC"))
def test_block_grouping_matches_upstream_shape_without_scheduler_count(layout):
    topology, _ = make_case(layers=(0,))
    spec = FullAttentionSpec(block_size=8, num_kv_heads=4, head_size=3, dtype=torch.int32)
    topology.transfer_groups[0].kv_cache_spec = spec
    # The runner has three local pages; the allocator may expose fewer. Only the
    # storage-row/kernel-row ratio determines grouping, not this allocator limit.
    topology.num_blocks = 1
    shape = compute_layer_kv_cache_shape_bytes(spec, num_blocks=3, kernel_block_size=2)
    assert shape[0] == 12
    tensor = torch.arange(np.prod(shape), dtype=torch.int32).to(torch.uint8).reshape(shape).view(torch.int32)
    if layout == "NHD":
        tensor = tensor.permute(0, 2, 1, 3)
    _, local, _ = build_mappings(topology, {"layer0": tensor}, group_id=3, kv_cache_layout=layout)
    head_axis = 2 if layout == "NHD" else 1
    for identity, segments in local.items():
        assert np.all(segments["block_strides"] == 4 * tensor.stride(0) * tensor.element_size())
        selected = tensor[8:12].narrow(head_axis, (identity[2] - 2) * 2, 2)
        assert read_local(segments, 2) == selected.contiguous().numpy().tobytes()


def test_compressed_mla_uses_storage_rows_instead_of_logical_tokens():
    topology, caches = make_case(layers=(0,), kernels=2)
    topology.transfer_groups[0].kv_cache_spec = AscendMLAAttentionSpec(
        block_size=64,
        num_kv_heads=4,
        head_size=3,
        dtype=torch.int32,
        tokens_per_state=16,
    )
    topology.transfer_groups[0].key_metadata.cache_family = "c16"
    keys, local, _ = build_mappings(topology, caches, group_id=3)
    for identity, segments in local.items():
        assert "@cache_family:c16@" in keys[identity][0]
        # 64 logical tokens -> 4 stored rows -> two 2-row kernel blocks.
        assert np.all(segments["block_strides"] == 2 * caches["layer0"][0].stride(0) * 4)
        assert read_local(segments, 2) == expected_bytes(caches, (0,), identity[2] - 2, 2, kernels=2)


def test_uniform_specs_use_each_layers_storage_layout():
    topology, caches = make_case(layers=(0, 1), kernels=2)
    first = topology.transfer_groups[0].kv_cache_spec
    second = replace(first, head_size=1, head_size_v=1)
    caches["layer1"] = tuple(tensor[..., :1] if tensor.numel() else tensor for tensor in caches["layer1"])
    topology.transfer_groups[0].kv_cache_spec = UniformTypeKVCacheSpecs(
        block_size=4, kv_cache_specs={"layer0": first, "layer1": second}
    )
    _, local, _ = build_mappings(topology, caches, group_id=3)
    for identity, segments in local.items():
        assert read_local(segments, 2) == expected_bytes(caches, (0, 1), identity[2] - 2, 2, kernels=2)


def test_mamba_pages_multiple_names_and_mixed_dtypes():
    topology, _ = make_case(layers=(1,), head_partition_count=1)
    group = topology.transfer_groups[0]
    group.group_id = 9
    group.key_metadata.cache_family = "mamba"
    group.layers[0].layer_names = ("conv", "ssm")
    group.kv_cache_spec = MambaSpec(
        block_size=64, shapes=((2, 3), (2, 3)), dtypes=(torch.int16, torch.float32), mamba_cache_mode="align"
    )
    caches = {}
    for name, dtype in (("conv", torch.int16), ("ssm", torch.float32)):
        storage = torch.arange(33, dtype=dtype).reshape(3, 11)
        tensor = storage[:, :6].view(3, 2, 3)
        caches[name] = tensor if name == "conv" else (torch.empty(0), tensor)
    keys, local, layer_slices = build_mappings(topology, caches, group_id=9, layerwise=True)
    remote = build_key_range_remote(local)
    identity = (0, 1, 1)
    assert "@group:9@cache_role:kv@cache_family:mamba@" in keys[identity][0]
    assert local[identity]["lengths"].tolist() == [12, 24]
    assert local[identity]["block_strides"].tolist() == [22, 44]
    assert remote[identity[0]]["offsets"].tolist() == [0, 12]
    for block in (0, 2):
        expected = caches["conv"][block].contiguous().numpy().tobytes() + caches["ssm"][1][block].numpy().tobytes()
        assert read_local(local[identity], block) == expected


def test_component_shapes_storage_aliases_and_sparse_pp_identity():
    topology, caches = make_case(layers=(4, 7), pp_counts=(2, 2, 1, 3))
    for name, (key, _, value) in caches.items():
        caches[name] = (key, value[..., :1])
    _, local, _ = build_mappings(topology, caches, group_id=3)
    assert list(local) == [(2, 1, 2), (2, 1, 3), (3, 1, 2), (3, 1, 3)]
    caches["layer7"][1][2, 1, 2, 0] = -123
    for (pp, dcp, head), segments in local.items():
        assert read_local(segments, 2) == expected_bytes(caches, (4,) if pp == 2 else (7,), head - 2, 2)
    topology.config.pp_partition.target_layer_counts = None
    topology.transfer_groups[0].key_metadata.pp_rank = 2
    keys, _, _ = build_mappings(topology, caches, group_id=3)
    assert list(keys) == [(2, 1, 2), (2, 1, 3)]


def test_empty_components_layers_and_groups():
    topology, caches = make_case(layers=(0, 1))
    caches["layer0"] = torch.empty(0)
    _, local, layer_slices = build_mappings(topology, caches, group_id=3, layerwise=True)
    remote = build_key_range_remote(local)
    for identity in local:
        empty = layer_slices[identity[0]][0]
        assert empty == slice(0, 0)
        assert local_addresses(local[identity], (0, 2), empty).shape == (2, 0)
        assert local_lengths(local[identity], empty).shape == (0,)
        assert remote[identity[0]]["offsets"][layer_slices[identity[0]][1]][0] == 0
    caches["layer1"] = torch.empty(0)
    _, local, layer_slices = build_mappings(topology, caches, group_id=3, layerwise=True)
    assert all(len(array) == 0 for segments in local.values() for array in segments.values())
    for segments in local.values():
        assert local_lengths(segments).shape == (0,)
        assert local_addresses(segments, (0, 2)).shape == (2, 0)
    for remote in (build_key_range_remote(local), build_gva_remote(topology, local, layer_slices)):
        assert remote[0]["size_bytes"] == 0
        assert remote[0]["offsets"].shape == (0,)
    topology.transfer_groups[0].layers = ()
    assert build_mappings(topology, {}, group_id=3, layerwise=True) == ({}, {}, {})
    assert build_key_range_remote({}) == build_gva_remote(topology, {}, {}) == {}


def test_nonuniform_layers_are_valid_without_uniform_global_reservation():
    topology, caches = make_case(layers=(0, 1))
    caches["layer1"] = tuple(tensor[..., :1] if tensor.numel() else tensor for tensor in caches["layer1"])
    topology.config.pp_partition.global_layer_count = 4
    _, local, layer_slices = build_mappings(topology, caches, group_id=3, layerwise=True)
    remote = build_key_range_remote(local)
    assert remote[0]["size_bytes"] == 128
    with pytest.raises(ValueError, match="uniform"):
        build_gva_remote(topology, local, layer_slices)
    topology.config.pp_partition.global_layer_count = None
    remote = build_gva_remote(topology, local, layer_slices)
    assert remote[0]["size_bytes"] == 128


def test_invalid_kernel_rows_and_block_extent_are_rejected():
    topology, caches = make_case(layers=(0,), kernels=2)
    topology.transfer_groups[0].kv_cache_spec = replace(topology.transfer_groups[0].kv_cache_spec, block_size=3)
    with pytest.raises(ValueError, match="Kernel rows"):
        build_mappings(topology, caches, group_id=3)
    topology.transfer_groups[0].kv_cache_spec = replace(topology.transfer_groups[0].kv_cache_spec, block_size=4)
    caches["layer0"] = (caches["layer0"][0][:-1],)
    with pytest.raises(RuntimeError, match="unflatten"):
        build_mappings(topology, caches, group_id=3)


def test_invalid_head_partition_and_pp_span_are_rejected():
    topology, caches = make_case(layers=(0,))
    caches["layer0"] = (caches["layer0"][0][:, :, :3],)
    with pytest.raises(RuntimeError, match="unflatten"):
        build_mappings(topology, caches, group_id=3)
    topology.config.pp_partition.target_layer_counts = (0, 0)
    with pytest.raises(ValueError, match="do not cover"):
        build_mappings(topology, caches, group_id=3)


@pytest.mark.parametrize("count,alignment,index", ((0, 1, 0), (1, 0, 0), (2, 1, 2)))
def test_invalid_gva_slots_are_rejected(count, alignment, index):
    topology, caches = make_case(layers=(0,))
    topology.config.cp_partition.object_shard_count = count
    topology.config.cp_partition.shard_alignment_bytes = alignment
    topology.object_shard_index = index
    _, local, layer_slices = build_mappings(topology, caches, group_id=3, layerwise=True)
    with pytest.raises(ValueError, match="Invalid remote shard"):
        build_gva_remote(topology, local, layer_slices)


def test_gva_reservations_compose_with_pp_and_head_partitions():
    topology, caches = make_case(pp_counts=(1, 0, 3))
    topology.config.pp_partition.global_layer_count = 7
    topology.config.cp_partition.object_shard_count = 2
    topology.config.cp_partition.shard_alignment_bytes = 1024
    topology.object_shard_index = 1
    keys, local, layer_slices = build_mappings(topology, caches, group_id=3, layerwise=True)
    remote = build_gva_remote(topology, local, layer_slices)
    assert list(keys) == [(0, 1, 2), (0, 1, 3), (2, 1, 2), (2, 1, 3)]
    for identity, segments in local.items():
        target = remote[identity[0]]
        # 96 bytes/layer, seven reserved layers, two aligned 1024-byte slots.
        assert target["size_bytes"] == 2048
        assert target["offsets"][0] == 1024
        layers = (0,) if identity[0] == 0 else (1, 2, 3)
        assert read_local(local[identity], 2) == expected_bytes(caches, layers, identity[2] - 2, 2)
        assert int(target["offsets"][-1] + local[identity]["lengths"][-1]) == 1024 + len(layers) * 96


@pytest.mark.parametrize("dtype", (torch.int16, torch.float32))
@pytest.mark.parametrize(
    "layout,strides",
    (
        pytest.param("NHD", (32, 8, 2, 1), id="nhd"),
        pytest.param("NHD", (40, 8, 2, 1), id="nhd-block-padding"),
        pytest.param("NHD", (40, 10, 2, 1), id="nhd-token-padding"),
        pytest.param("NHD", (48, 12, 3, 1), id="nhd-head-padding"),
        pytest.param("NHD", (64, 16, 4, 2), id="nhd-element-padding"),
        pytest.param("HND", (32, 8, 2, 1), id="hnd"),
        pytest.param("HND", (40, 8, 2, 1), id="hnd-block-padding"),
        pytest.param("HND", (40, 10, 2, 1), id="hnd-head-padding"),
        pytest.param("HND", (48, 12, 3, 1), id="hnd-token-padding"),
        pytest.param("HND", (64, 16, 4, 2), id="hnd-element-padding"),
    ),
)
def test_strided_views_map_only_logical_elements(dtype, layout, strides):
    shape = (3, 4, 4, 2)
    storage_offset = 5
    storage = torch.full((storage_offset + shape[0] * strides[0],), -999, dtype=dtype)
    tensor = torch.as_strided(storage, shape, strides, storage_offset)
    tensor.copy_(torch.arange(tensor.numel(), dtype=dtype).reshape(shape))
    head_axis = 1 if layout == "HND" else 2
    element_bytes = tensor.element_size()
    topology, _ = make_case(layers=(0,))
    topology.transfer_groups[0].kv_cache_spec = replace(topology.transfer_groups[0].kv_cache_spec, block_size=4)
    _, local, layer_slices = build_mappings(
        topology, {"layer0": tensor}, group_id=3, kv_cache_layout=layout, layerwise=True
    )
    remote = build_key_range_remote(local)

    for head in range(2):
        selected = tensor.narrow(head_axis, head * 2, 2)
        segments = local[(0, 1, 2 + head)]
        for block_id in (0, 2):
            block = selected[block_id]
            addresses = local_addresses(segments, (block_id,))[0].tolist()
            lengths = np.broadcast_to(local_lengths(segments), (len(addresses),)).tolist()
            assert remote[0]["offsets"].tolist() == [sum(lengths[:index]) for index in range(len(lengths))]
            # Enumerate logical elements independently of the recursive traversal.
            expected_addresses = [block[index].data_ptr() for index in product(*(range(size) for size in block.shape))]
            actual_addresses = [
                address + offset
                for address, length in zip(addresses, lengths, strict=True)
                for offset in range(0, length, element_bytes)
            ]
            assert actual_addresses == expected_addresses
            assert sum(lengths) == block.numel() * element_bytes
            actual_bytes = b"".join(
                ctypes.string_at(address, length) for address, length in zip(addresses, lengths, strict=True)
            )
            assert actual_bytes == block.contiguous().numpy().tobytes()
