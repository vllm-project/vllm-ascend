"""Reserve a global layer span without changing the selected local memory."""

from ...configuration.topology import PPPartitionSpec


def bind_layer_object_span(
    partition: PPPartitionSpec,
    local_bytes: int,
    group_layer_count: int,
    *,
    stage_layer_start: int,
    stage_layer_count: int,
) -> tuple[int, int]:
    """Return reserved bytes and local origin; the caller proves uniform layer payloads."""
    if partition.global_layer_count is None:
        return local_bytes, 0
    layer_count = max(partition.global_layer_count, stage_layer_count, group_layer_count)
    if stage_layer_start + group_layer_count > layer_count:
        raise ValueError("Stage payload exceeds the reserved global layer span")
    bytes_per_layer = local_bytes // group_layer_count
    return layer_count * bytes_per_layer, stage_layer_start * bytes_per_layer
