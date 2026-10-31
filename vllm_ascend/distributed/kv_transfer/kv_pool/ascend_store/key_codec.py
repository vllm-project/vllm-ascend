# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Portable source identity for existing layerwise pool wire keys.

Resolve namespaces on the writer, once. Readers bind this identity instead of
rehashing their own model/parallel configuration. This describes key naming,
not object layout, readiness, or whether two caches can be restored together.
"""

from dataclasses import asdict, dataclass
from typing import Any

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.layerwise_keys import LayerwiseKeyBuilder


@dataclass(frozen=True)
class PoolKeyIdentity:
    format: str
    model_name: str
    num_groups: int
    pp_size: int
    namespace: str = ""
    layout_id: str = ""
    group_block_sizes: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.format not in ("memcache-v1", "mooncake-block-v1", "mooncake-hybrid-v1"):
            raise ValueError(f"Unsupported pool key format: {self.format!r}")
        if not isinstance(self.model_name, str) or not self.model_name:
            raise ValueError("Pool key identity requires a model name")
        for name in ("num_groups", "pp_size"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"Pool key {name} must be a positive integer")
        if not isinstance(self.namespace, str) or not isinstance(self.layout_id, str):
            raise ValueError("Pool key namespace and layout_id must be strings")
        if not isinstance(self.group_block_sizes, tuple):
            raise ValueError("Pool key group_block_sizes must be an immutable tuple")
        if self.format == "mooncake-hybrid-v1":
            if not self.layout_id or len(self.group_block_sizes) != self.num_groups:
                raise ValueError("Hybrid pool keys require a layout ID and one block size per group")
            if any(type(size) is not int or size <= 0 for size in self.group_block_sizes):
                raise ValueError("Hybrid pool key block sizes must be positive integers")
        elif self.layout_id or self.group_block_sizes:
            raise ValueError("Only hybrid pool keys accept layout_id and group_block_sizes")
        if self.format == "memcache-v1" and self.namespace:
            raise ValueError("Legacy MemCache keys do not encode a namespace")
        if self.format == "mooncake-block-v1" and self.pp_size > 1 and not self.namespace:
            raise ValueError("Mooncake PP keys require the source layout namespace")

    def to_wire(self) -> dict[str, Any]:
        result = asdict(self)
        result["group_block_sizes"] = list(self.group_block_sizes)
        return result

    @classmethod
    def from_wire(cls, data: dict[str, Any]) -> "PoolKeyIdentity":
        if not isinstance(data, dict):
            raise ValueError("Pool key identity must be an object")
        required = {"format", "model_name", "num_groups", "pp_size"}
        optional = {"namespace", "layout_id", "group_block_sizes"}
        if not required.issubset(data) or data.keys() - required - optional:
            raise ValueError("Invalid pool key identity fields")
        values = dict(data)
        sizes = values.get("group_block_sizes", [])
        if not isinstance(sizes, list):
            raise ValueError("Wire group_block_sizes must be an array")
        values["group_block_sizes"] = tuple(sizes)
        return cls(**values)

    def bind(self) -> LayerwiseKeyBuilder:
        # Backend imports are deferred: decoding metadata must not import or
        # initialize the store runtime. Use the deployed key functions verbatim.
        if self.format == "memcache-v1":
            from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.memcache_backend import make_full_key

            def make_key(group: int, block_hash: str, head: int, stage: int) -> str:
                return make_full_key(self.model_name, group, block_hash, head, self.num_groups, stage, self.pp_size)

        else:
            from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend import mooncake_layerwise

            if self.format == "mooncake-hybrid-v1":

                def make_key(group: int, block_hash: str, head: int, stage: int) -> str:
                    return mooncake_layerwise.hybrid_block_key(
                        self.model_name,
                        self.layout_id,
                        group,
                        self.group_block_sizes[group],
                        block_hash,
                        head,
                        pp_rank=stage if self.pp_size > 1 else None,
                    )

            else:

                def make_key(group: int, block_hash: str, head: int, stage: int) -> str:
                    return mooncake_layerwise.make_block_key(
                        self.model_name, block_hash, head, namespace=self.namespace, pp_rank=stage
                    )

        def checked_key(group: int, block_hash: str, head: int, stage: int) -> str:
            self._validate_coordinates(group, head, stage)
            if not isinstance(block_hash, str) or not block_hash:
                raise ValueError("Source block hash or object ID must be a nonempty string")
            return make_key(group, block_hash, head, stage)

        return LayerwiseKeyBuilder(checked_key, self.pp_size, self)

    def _validate_coordinates(self, group: int, head: int, stage: int) -> None:
        if type(group) is not int or not 0 <= group < self.num_groups:
            raise ValueError(f"Invalid source cache group: {group!r}")
        if type(stage) is not int or not 0 <= stage < self.pp_size:
            raise ValueError(f"Invalid source PP rank: {stage!r}")
        if type(head) is not int or head < 0:
            raise ValueError(f"Invalid source head/TP rank: {head!r}")

    def partial_key(
        self, request_id: str, group: int, block_index: int, end_token: int, head: int, stage: int = 0
    ) -> str:
        """Build the ordinary MemCache partial key, not a PD tail object ID."""
        if self.format != "memcache-v1":
            raise ValueError("Ordinary partial keys are only supported by MemCache")
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("Partial keys require a source request ID")
        if type(block_index) is not int or block_index < 0 or type(end_token) is not int or end_token <= 0:
            raise ValueError("Partial keys require a nonnegative block index and positive end token")
        self._validate_coordinates(group, head, stage)
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.memcache_backend import MemcacheBackend

        return MemcacheBackend.make_partial_key(
            self.model_name, request_id, group, block_index, end_token, head, stage, self.pp_size
        )
