"""Place an already-sharded local payload in a context object's reserved byte slots."""

from ...configuration.topology import CPPartitionSpec


def bind_context_object_span(
    partition: CPPartitionSpec, shard_bytes: int, layer_offset_bytes: int, *, shard_index: int
) -> tuple[int, int]:
    """Compute complete object size and local origin from resolved CP coordinates.

    Called once at Mapping initialization, after PP placement. This neither
    allocates an object nor slices already-sharded local tokens again.
    """
    count, alignment = partition.object_shard_count, partition.shard_alignment_bytes
    stride = ((shard_bytes + alignment - 1) // alignment) * alignment
    return stride * count, shard_index * stride + layer_offset_bytes
