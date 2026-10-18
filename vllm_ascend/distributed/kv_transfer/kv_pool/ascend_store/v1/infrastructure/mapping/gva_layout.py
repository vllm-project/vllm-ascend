"""Compile the production GVA object's global-layer and context-shard coordinates.

This is byte geometry only. Allocation, writer election, and complete-object
publication remain outside Mapping. PP keys are distinct even though each key
reserves a global layer span; only DCP replicas actually share one object.
"""

from dataclasses import dataclass

GVA_SHARD_ALIGNMENT_BYTES = 2 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class GVAObjectLayout:
    total_layer_count: int
    local_layer_count: int
    stage_layer_offset: int
    shard_count: int = 1
    shard_index: int = 0

    def bind(self, local_bytes: int, group_layer_count: int) -> tuple[int, int]:
        """Return (object size, local payload offset), both in bytes, once per group."""
        if group_layer_count <= 0 or local_bytes <= 0 or local_bytes % group_layer_count:
            raise ValueError("GVA groups require nonempty, uniform per-layer payloads")
        if not 0 <= self.shard_index < self.shard_count or self.stage_layer_offset < 0:
            raise ValueError("Invalid GVA shard or stage coordinate")
        bytes_per_layer = local_bytes // group_layer_count
        layer_count = max(self.total_layer_count, self.local_layer_count, group_layer_count)
        shard_stride = bytes_per_layer * layer_count
        if self.shard_count > 1:
            shard_stride = (
                (shard_stride + GVA_SHARD_ALIGNMENT_BYTES - 1) // GVA_SHARD_ALIGNMENT_BYTES
            ) * GVA_SHARD_ALIGNMENT_BYTES
        offset = self.shard_index * shard_stride + self.stage_layer_offset * bytes_per_layer
        if self.stage_layer_offset + group_layer_count > layer_count:
            raise ValueError("Stage payload exceeds the production GVA layer span")
        return shard_stride * self.shard_count, offset
