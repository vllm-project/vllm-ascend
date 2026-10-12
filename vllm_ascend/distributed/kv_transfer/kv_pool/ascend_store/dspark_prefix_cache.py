# SPDX-License-Identifier: Apache-2.0
"""Persistent DSpark prefix pages in the existing AscendStore backend."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    ReqMeta,
    block_hash_to_str,
    get_block_hashes,
)
from vllm_ascend.spec_decode.dspark_utils import get_dspark_aux_layer_ids

_RPC_BLOCKS = 128
_LEASE_TTL_MS = 5 * 60 * 1000


def _model_identity(model: Any) -> dict[str, Any]:
    hf_config = getattr(model, "hf_config", None)
    config = (
        hf_config.to_dict()
        if hf_config is not None and hasattr(hf_config, "to_dict")
        else vars(hf_config)
        if hf_config
        else {}
    )
    return {
        "model": str(model.model),
        "revision": getattr(model, "revision", None),
        "code_revision": getattr(model, "code_revision", None),
        "dtype": str(getattr(model, "dtype", None)),
        "quantization": getattr(model, "quantization", None),
        "config": config,
    }


@dataclass(frozen=True)
class DSparkPrefixKeys:
    identity: str
    pp_size: int
    tp_size: int

    @classmethod
    def from_config(cls, vllm_config: Any) -> DSparkPrefixKeys | None:
        speculative = getattr(vllm_config, "speculative_config", None)
        if speculative is None or speculative.method != "dspark":
            return None
        draft = speculative.draft_model_config
        if draft is None:
            raise ValueError("DSpark prefix caching requires the loaded draft model configuration")
        parallel = vllm_config.parallel_config
        cache_config = getattr(vllm_config, "cache_config", None)
        identity = {
            "target": _model_identity(vllm_config.model_config),
            "draft": _model_identity(draft),
            "draft_width": speculative.num_speculative_tokens,
            "aux_layers": get_dspark_aux_layer_ids(vllm_config),
            "tp_size": parallel.tensor_parallel_size,
            "pp_size": parallel.pipeline_parallel_size,
            "block_size": getattr(cache_config, "block_size", None),
            "prefix_match_unit": getattr(cache_config, "prefix_match_unit", None),
        }
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()
        return cls(digest, parallel.pipeline_parallel_size, parallel.tensor_parallel_size)

    def make_key(self, block_hash: str, tp_rank: int) -> str:
        if not 0 <= tp_rank < self.tp_size or self.pp_size < 1:
            raise ValueError("Invalid DSpark prefix parallel rank")
        return f"dspark-prefix-v1@{self.identity}@pp{self.pp_size - 1}@tp{tp_rank}@{block_hash}"

    def make_hit_check_keys(self, block_hash: str) -> list[str]:
        return [self.make_key(block_hash, rank) for rank in range(self.tp_size)]


@dataclass(frozen=True)
class _Page:
    group_id: int
    address: int
    stride: int
    size: int
    region_size: int


class DSparkPrefixCache:
    """A companion object per token block; target scratch buffers stay separate."""

    def __init__(
        self,
        keys: DSparkPrefixKeys,
        backend: Any,
        tp_rank: int,
        block_size: int,
        hash_block_size: int,
        copy_fn: Callable[[np.ndarray, np.ndarray, np.ndarray, int], int],
    ) -> None:
        if block_size <= 0 or hash_block_size <= 0 or block_size % hash_block_size:
            raise ValueError("DSpark prefix pages require aligned logical and hash block sizes")
        keys.make_key("", tp_rank)
        self.keys, self.backend, self.tp_rank = keys, backend, tp_rank
        self.block_size, self.hash_block_size, self.copy_fn = block_size, hash_block_size, copy_fn
        self._pages: tuple[_Page, ...] = ()
        self._caches: dict[str, Any] = {}
        self.num_blocks = 0
        self.page_bytes = 0

    def register(
        self, caches: dict[str, Any], layer_group_ids: dict[str, int], block_sizes: list[int], num_blocks: int
    ) -> None:
        if not caches or set(caches) != set(layer_group_ids) or num_blocks <= 0:
            raise ValueError("DSpark prefix registration requires every loaded draft cache and its group")
        pages: list[_Page] = []
        for name in sorted(caches):
            first_page = len(pages)
            group_id = layer_group_ids[name]
            if not 0 <= group_id < len(block_sizes) or block_sizes[group_id] != self.block_size:
                raise ValueError("DSpark prefix cache groups must share one logical block size")
            tensors = caches[name] if isinstance(caches[name], (tuple, list)) else (caches[name],)
            for cache in tensors:
                if cache is None or not cache.numel():
                    continue
                if cache.ndim < 2 or cache.shape[0] % num_blocks or not cache[0].is_contiguous():
                    raise ValueError("DSpark prefix caches require contiguous inner kernel pages")
                scale = cache.shape[0] // num_blocks
                if scale * cache.shape[1] != self.block_size:
                    raise ValueError("DSpark kernel pages do not cover one logical token block")
                kernel_bytes = cache[0].numel() * cache.element_size()
                kernel_stride = cache.stride(0) * cache.element_size()
                if kernel_stride < kernel_bytes or (scale > 1 and kernel_stride != kernel_bytes):
                    raise ValueError("Scaled DSpark kernel pages must be adjacent within each logical page")
                size, stride = kernel_bytes * scale, kernel_stride * scale
                pages.append(_Page(group_id, cache.data_ptr(), stride, size, (num_blocks - 1) * stride + size))
            if len(pages) == first_page:
                raise ValueError(f"DSpark prefix layer {name} has no draft KV pages")
        if not pages:
            raise ValueError("DSpark prefix registration contains no draft KV pages")
        self.backend.ensure_initialized()
        if not callable(getattr(self.backend.store, "batch_write_finish", None)):
            raise RuntimeError("DSpark prefix caching requires MemCache atomic write publication")
        self._pages, self._caches, self.num_blocks = tuple(pages), dict(caches), num_blocks
        self.page_bytes = sum(page.size for page in pages)

    def registered_regions(self) -> tuple[list[int], list[int]]:
        return [page.address for page in self._pages], [page.region_size for page in self._pages]

    def _block_keys(self, request: ReqMeta, tokens: int, *, exact: bool) -> list[str]:
        prompt_tokens = request.num_prompt_tokens if request.num_prompt_tokens is not None else request.target_token_len
        if not self._pages or tokens < 0 or tokens > prompt_tokens:
            raise ValueError("DSpark prefix extent exceeds initialized prompt KV")
        hashes = get_block_hashes(request.block_hashes, self.block_size, self.hash_block_size)
        count = tokens // self.block_size
        if exact and (tokens % self.block_size or count > len(hashes)):
            raise ValueError("DSpark prefix restoration requires complete hashed blocks")
        return [
            self.keys.make_key(block_hash_to_str(hashes[index]), self.tp_rank)
            for index in range(min(count, len(hashes)))
        ]

    def _copy(
        self, indices: Sequence[int], gvas: Sequence[int], blocks_by_group: Mapping[int, Sequence[int]], direction: int
    ) -> None:
        remote, local, sizes = [], [], []
        for index, gva in zip(indices, gvas, strict=True):
            offset = 0
            for page in self._pages:
                blocks = blocks_by_group[page.group_id]
                if index >= len(blocks) or type(blocks[index]) is not int or not 0 <= blocks[index] < self.num_blocks:
                    raise ValueError("DSpark prefix block table does not cover the requested prompt extent")
                remote.append(gva + offset)
                local.append(page.address + blocks[index] * page.stride)
                sizes.append(page.size)
                offset += page.size
        if (
            self.copy_fn(
                np.asarray(remote, dtype=np.int64),
                np.asarray(local, dtype=np.int64),
                np.asarray(sizes, dtype=np.int64),
                direction,
            )
            != 0
        ):
            raise RuntimeError("DSpark prefix KV copy failed")

    def _gvas(self, keys: list[str], *, for_load: bool = False) -> list[int]:
        infos = self.backend.batch_get_key_info(keys, for_load=for_load)
        if infos is None or len(infos) != len(keys):
            raise RuntimeError("DSpark prefix key metadata response is incomplete")
        result = []
        for info in infos:
            if info is None or info.size() != self.page_bytes:
                raise RuntimeError("DSpark prefix KV object is missing or has a different page layout")
            addresses = info.gva_list()
            if not addresses or addresses[0] <= 0:
                raise RuntimeError("DSpark prefix KV object is not readable")
            result.append(addresses[0])
        return result

    def restore(self, request: ReqMeta, offset: int, blocks_by_group: Mapping[int, Sequence[int]]) -> int:
        keys = self._block_keys(request, offset, exact=True)
        if not keys:
            return offset
        load_spec = request.load_spec
        if load_spec is None or not load_spec.can_load or load_spec.kvpool_cached_tokens < offset:
            raise RuntimeError("DSpark prefix restoration requires the scheduler's validated pool hit")
        for start in range(0, len(keys), _RPC_BLOCKS):
            chunk = keys[start : start + _RPC_BLOCKS]
            self._gvas(chunk, for_load=True)
            leased: list[str] = []
            try:
                statuses = self.backend.batch_add_lease(chunk, _LEASE_TTL_MS)
                if statuses is not None:
                    leased = [key for key, result in zip(chunk, statuses) if result == 0]
                if statuses is None or len(statuses) != len(chunk) or any(result != 0 for result in statuses):
                    raise RuntimeError("DSpark prefix read lease failed")
                # Rewarm before leasing only: a leased object must not move.
                gvas = self._gvas(chunk)
                self._copy(range(start, start + len(chunk)), gvas, blocks_by_group, 1)
            finally:
                if leased and self.backend.batch_remove_lease(leased) != 0:
                    raise RuntimeError("DSpark prefix read lease release failed")
        return offset

    def save(self, request: ReqMeta, through_tokens: int, blocks_by_group: Mapping[int, Sequence[int]]) -> None:
        keys = self._block_keys(request, through_tokens, exact=False)
        for start in range(0, len(keys), _RPC_BLOCKS):
            chunk = keys[start : start + _RPC_BLOCKS]
            readable = self.backend.batch_is_readable(chunk)
            if len(readable) != len(chunk) or any(type(result) is not bool for result in readable):
                raise RuntimeError("DSpark prefix readability response is incomplete")
            # Existing objects need no write. They may be evicted after this
            # probe; only restore needs to validate and lease their metadata.
            indices = [start + index for index, hit in enumerate(readable) if not hit]
            missing = [key for key, hit in zip(chunk, readable, strict=True) if not hit]
            if not missing:
                continue
            allocated: list[str] = []
            try:
                gvas = self.backend.batch_alloc(missing, [self.page_bytes] * len(missing), _LEASE_TTL_MS)
                if gvas is not None:
                    allocated = [
                        key
                        for key, address in zip(missing, gvas)
                        if type(address) is int or isinstance(address, np.integer)
                        if address > 0
                    ]
                if (
                    gvas is None
                    or len(gvas) != len(missing)
                    or any(type(address) is not int and not isinstance(address, np.integer) for address in gvas)
                ):
                    raise RuntimeError("DSpark prefix allocation response is incomplete")
                collisions = [key for key, address in zip(missing, gvas, strict=True) if address <= 0]
                if collisions:
                    exists = self.backend.exists(collisions)
                    if exists is None or len(exists) != len(collisions) or any(result != 1 for result in exists):
                        raise RuntimeError("DSpark prefix allocation failed")
                    # Another DP writer owns these objects, possibly still WRITING.
                    # Joint prefix lookup will remain a miss until they are readable.
                owned_indices = [index for index, address in zip(indices, gvas, strict=True) if address > 0]
                owned_gvas = [address for address in gvas if address > 0]
                if not allocated:
                    continue
                # WRITING objects need not expose readable metadata until publication.
                self._copy(owned_indices, owned_gvas, blocks_by_group, 0)
                statuses = self.backend.batch_write_finish(allocated, [0] * len(allocated))
                if statuses is None or len(statuses) != len(allocated) or any(result != 0 for result in statuses):
                    raise RuntimeError("DSpark prefix publication failed")
            except Exception:
                # Never invalidate a colliding object owned by another writer.
                if allocated:
                    self.backend.batch_write_finish(allocated, [-1] * len(allocated))
                raise
            finally:
                if allocated and self.backend.batch_remove_lease(allocated) != 0:
                    raise RuntimeError("DSpark prefix write lease release failed")
