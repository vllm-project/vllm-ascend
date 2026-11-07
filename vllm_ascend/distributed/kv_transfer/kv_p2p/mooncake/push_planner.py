# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Validate whole-request destinations before submitting any WRITE."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.utils import BlockIds

    from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.metadata import MooncakeTransferMetadata


def plan_push(
    source: "MooncakeTransferMetadata",
    destination: "MooncakeTransferMetadata",
    source_blocks: "BlockIds",
    destination_blocks: "BlockIds",
    num_tokens: int,
) -> tuple[list[int], list[int], list[int]]:
    if source.block_size != destination.block_size or num_tokens < 0:
        raise ValueError("Push requires matching positive block sizes and nonnegative token counts")
    if source.block_size <= 0:
        raise ValueError("Invalid push block size")
    if len(source_blocks) != 1 or len(destination_blocks) != 1:
        raise ValueError("Push initially supports one FullAttention cache group")
    expected = (num_tokens + source.block_size - 1) // source.block_size
    for metadata, groups in ((source, source_blocks), (destination, destination_blocks)):
        blocks = groups[0]
        if len(blocks) != expected or len(set(blocks)) != len(blocks):
            raise ValueError("Push block tables must contain exactly the requested pages without duplicates")
        if any(type(block) is not int or not 0 <= block < metadata.num_blocks for block in blocks):
            raise ValueError("Push block ID is outside its registered allocation")
    if len(set(source.layer_names)) != len(source.layer_names):
        raise ValueError("Duplicate source layer names")
    if len(set(destination.layer_names)) != len(destination.layer_names):
        raise ValueError("Duplicate destination layer names")
    if set(source.layer_names) != set(destination.layer_names):
        raise ValueError("Push source and destination must expose identical layers")

    src: list[int] = []
    dst: list[int] = []
    lengths: list[int] = []
    destination_layers = {name: index for index, name in enumerate(destination.layer_names)}
    for source_index, name in enumerate(source.layer_names):
        destination_index = destination_layers[name]
        for metadata, index in ((source, source_index), (destination, destination_index)):
            if metadata.group_indices[index] != 0 or metadata.layer_block_sizes[index] != metadata.block_size:
                raise ValueError("Unsupported push cache group or per-layer block size")
            scales = metadata.block_size_scales[index]
            components = len(metadata.kv_caches_base_addr[index])
            if not components or len(scales) != components or any(scale != 1 for scale in scales):
                raise ValueError("Push initially requires unscaled cache pages")
            for field in (metadata.block_lens, metadata.block_strides, metadata.block_shapes):
                if len(field[index]) != components:
                    raise ValueError("Inconsistent push component metadata")
            for length, stride in zip(metadata.block_lens[index], metadata.block_strides[index]):
                if length <= 0 or stride < length:
                    raise ValueError("Push requires contiguous nonoverlapping physical pages")
        if source.block_shapes[source_index] != destination.block_shapes[destination_index]:
            raise ValueError("Push component shapes differ")
        if source.block_lens[source_index] != destination.block_lens[destination_index]:
            raise ValueError("Push component byte lengths differ")
        if len(source.kv_caches_base_addr[source_index]) != len(destination.kv_caches_base_addr[destination_index]):
            raise ValueError("Push component counts differ")
        for component, length in enumerate(source.block_lens[source_index]):
            source_base = source.kv_caches_base_addr[source_index][component]
            destination_base = destination.kv_caches_base_addr[destination_index][component]
            if source_base <= 0 or destination_base <= 0:
                raise ValueError("Push cache base address must be positive")
            source_stride = source.block_strides[source_index][component]
            destination_stride = destination.block_strides[destination_index][component]
            for source_block, destination_block in zip(source_blocks[0], destination_blocks[0]):
                source_address = source_base + source_block * source_stride
                destination_address = destination_base + destination_block * destination_stride
                if src and src[-1] + lengths[-1] == source_address and dst[-1] + lengths[-1] == destination_address:
                    lengths[-1] += length
                else:
                    src.append(source_address)
                    dst.append(destination_address)
                    lengths.append(length)
    return src, dst, lengths
