from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.logger import logger
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_utils import BlockHash, BlockHashList
from vllm.v1.core.sched.output import NewRequestData

_GROUPED_BLOCK_HASH_DOMAIN = b"vllm-ascend-grouped-block-hash-v1\0"
_GROUPED_BLOCK_HASH_LENGTH_PREFIX_BYTES = 4


# Parameters related to the key
@dataclass
class KeyMetadata:
    """name of the LLM model"""

    model_name: str
    """ worker id when running under a distributed setting """
    head_or_tp_rank: int
    """ Initialize the current prefill context model parallel rank """
    pcp_rank: int
    """ Initialize the current decode context model parallel rank """
    dcp_rank: int
    """ Initialize the current pipeline parallel rank """
    pp_rank: int
    """ Initialize the current kv cache group id """
    kv_cache_group_id: int = 0
    """ Differentiate kv/state keys that share the same chunk hash """
    cache_role: str = "kv"
    """ Family name for compress-aware hybrid cache layouts """
    cache_family: str = "default"


@dataclass(order=True)
class PoolKey:
    key_metadata: KeyMetadata
    chunk_hash: str
    chunk_hash_bytes: BlockHash | str | None = field(default=None, compare=False, kw_only=True)

    def __hash__(self):
        return hash(
            (
                self.key_metadata.model_name,
                self.key_metadata.head_or_tp_rank,
                self.key_metadata.pcp_rank,
                self.key_metadata.dcp_rank,
                self.key_metadata.pp_rank,
                self.key_metadata.kv_cache_group_id,
                self.key_metadata.cache_role,
                self.key_metadata.cache_family,
                self.chunk_hash,
            )
        )

    def to_string(self):
        return (
            f"{self.key_metadata.model_name}"
            f"@pcp{self.key_metadata.pcp_rank}@dcp{self.key_metadata.dcp_rank}"
            f"@head_or_tp_rank:{self.key_metadata.head_or_tp_rank}"
            f"@pp_rank:{self.key_metadata.pp_rank}"
            f"@group:{self.key_metadata.kv_cache_group_id}"
            f"@cache_role:{self.key_metadata.cache_role}"
            f"@cache_family:{self.key_metadata.cache_family}"
            f"@{self.chunk_hash}"
        )

    def split_layers(self, num_layers: int) -> list[LayerPoolKey]:
        """Split the key into multiple keys for each layer"""
        keys = []
        for layer_id in range(num_layers):
            keys.append(
                LayerPoolKey(
                    self.key_metadata,
                    self.chunk_hash,
                    layer_id,
                    chunk_hash_bytes=self.chunk_hash_bytes,
                )
            )
        return keys


@dataclass(order=True)
class LayerPoolKey(PoolKey):
    """A key for the layer cache engine"""

    layer_id: int

    def __hash__(self):
        return hash(
            (
                self.key_metadata.model_name,
                self.key_metadata.head_or_tp_rank,
                self.key_metadata.pcp_rank,
                self.key_metadata.dcp_rank,
                self.key_metadata.kv_cache_group_id,
                self.key_metadata.cache_role,
                self.key_metadata.cache_family,
                self.chunk_hash,
                self.layer_id,
            )
        )

    def to_string(self):
        return (
            f"{self.key_metadata.model_name}"
            f"@pcp{self.key_metadata.pcp_rank}@dcp{self.key_metadata.dcp_rank}"
            f"@head_or_tp_rank:{self.key_metadata.head_or_tp_rank}"
            f"@group:{self.key_metadata.kv_cache_group_id}"
            f"@cache_role:{self.key_metadata.cache_role}"
            f"@cache_family:{self.key_metadata.cache_family}"
            f"@layer_id:{self.layer_id}"
            f"@{self.chunk_hash}"
        )


def infer_cache_family_from_ratio(compress_ratio: int | None) -> str:
    if compress_ratio is None:
        return "default"
    if compress_ratio <= 1:
        return "c1"
    return f"c{compress_ratio}"


def infer_cache_family_ratio(cache_family: str | None) -> int:
    if not cache_family or not cache_family.startswith("c"):
        return 1
    ratio = cache_family[1:]
    return int(ratio) if ratio.isdigit() else 1


def get_cache_family_granularity(block_size: int, cache_family: str | None) -> int:
    return block_size * infer_cache_family_ratio(cache_family)


def _get_layer_compress_ratio(
    layer_name: str,
    compress_ratios: Sequence[int] | None,
    hf_config: Any | None = None,
) -> int | None:
    if compress_ratios is None:
        return None
    if getattr(hf_config, "model_type", None) == "deepseek_v4":
        from vllm_ascend.utils import extract_dsv4_layer_index, get_dsv4_compress_ratio

        return get_dsv4_compress_ratio(hf_config, extract_dsv4_layer_index(hf_config, layer_name))
    from vllm.model_executor.models.utils import extract_layer_index

    return compress_ratios[extract_layer_index(layer_name)]


def _get_group_spec_ratios(group: object) -> set[int | None]:
    kv_cache_spec = getattr(group, "kv_cache_spec", None)
    if kv_cache_spec is None:
        return set()
    kv_cache_specs = getattr(kv_cache_spec, "kv_cache_specs", None)
    if kv_cache_specs is not None:
        return {getattr(spec, "compress_ratio", None) for spec in kv_cache_specs.values()}
    return {getattr(kv_cache_spec, "compress_ratio", None)}


def infer_group_cache_families(
    kv_cache_groups: Sequence[object] | None,
    compress_ratios: Sequence[int] | None,
    hf_config: Any | None = None,
) -> list[str]:
    if kv_cache_groups is None:
        return ["default"]

    families: list[str] = []
    for group in kv_cache_groups:
        spec_ratios = _get_group_spec_ratios(group)
        if len(spec_ratios) == 1:
            families.append(infer_cache_family_from_ratio(next(iter(spec_ratios))))
            continue
        if len(spec_ratios) > 1:
            families.append("mixed")
            continue

        layer_names = list(getattr(group, "layer_names", []))
        if compress_ratios is None or not layer_names:
            families.append("default")
            continue

        group_ratios = {_get_layer_compress_ratio(layer_name, compress_ratios, hf_config) for layer_name in layer_names}
        if len(group_ratios) == 1:
            families.append(infer_cache_family_from_ratio(next(iter(group_ratios))))
        else:
            logger.debug(
                "KV cache group has mixed layer compress ratios %s for layers %s; using mixed cache family.",
                sorted(group_ratios, key=lambda ratio: -1 if ratio is None else ratio),
                layer_names,
            )
            families.append("mixed")
    return families


class ChunkedTokenDatabase:
    def __init__(
        self,
        metadata: KeyMetadata,
        block_size: int | list[int],
        partitions: list[int] | None,
        use_hybrid: bool = False,
        hash_block_size: int | None = None,
        kv_cache_groups: Sequence[Any] | None = None,
        alignment_tokens: int | None = None,
        retention_interval: int | None = None,
        use_eagle: bool = False,
    ):
        self.metadata = metadata
        self.block_size = block_size if isinstance(block_size, list) else [block_size]
        self.kv_cache_groups = list(kv_cache_groups or [])
        self.kv_caches_base_addr: list[int] = []
        self.block_len: list[int] = []
        self.block_stride: list[int] = []
        self.group_kv_caches_base_addr: dict[int, list[int]] = {}
        self.group_block_len: dict[int, list[int]] = {}
        self.group_block_stride: dict[int, list[int]] = {}
        self.group_cache_families: dict[str, dict[int, str]] = {
            "kv": {},
            "state": {},
        }
        self.group_num_layers: dict[str, dict[int, int]] = {
            "kv": {},
            "state": {},
        }
        self.partitions = partitions
        self.use_hybrid = use_hybrid
        self.hash_block_size = self.block_size[0] if hash_block_size is None else hash_block_size
        self.alignment_tokens = alignment_tokens if alignment_tokens is not None else max(self.block_size)
        self.retention_interval = retention_interval
        self.eagle_group_ids = {
            idx for idx, group in enumerate(self.kv_cache_groups) if getattr(group, "is_eagle_group", False)
        }
        if use_eagle and not self.eagle_group_ids:
            self.eagle_group_ids = set(range(len(self.kv_cache_groups)))
        self.cache_coordinator: Any | None = None
        self._key_prefix_cache = {}

    def set_cache_coordinator(self, cache_coordinator: Any | None) -> None:
        self.cache_coordinator = cache_coordinator

    def _get_key_prefix(self, kv_cache_group_id, cache_role="kv", cache_family=None):
        """Database-level prefix cache. Per group, constructed once."""
        if cache_family is None:
            cache_family = self.group_cache_families.get(cache_role, {}).get(
                kv_cache_group_id, "default"
            )
        cache_key = (kv_cache_group_id, cache_role, cache_family)
        prefix = self._key_prefix_cache.get(cache_key)
        if prefix is None:
            prefix = (
                f"{self.metadata.model_name}"
                f"@pcp{self.metadata.pcp_rank}@dcp{self.metadata.dcp_rank}"
                f"@head_or_tp_rank:{self.metadata.head_or_tp_rank}"
                f"@pp_rank:{self.metadata.pp_rank}"
                f"@group:{kv_cache_group_id}"
                f"@cache_role:{cache_role}"
                f"@cache_family:{cache_family}@"
            )
            self._key_prefix_cache[cache_key] = prefix
        return prefix

    def process_token_key_strings(
        self,
        token_len: int,
        block_hashes: BlockHashList | list[str],
        mask_num: int = 0,
        kv_cache_group_id: int = 0,
        cache_role: str = "kv",
        cache_family: str | None = None,
        max_num: int | None = None,
        chunk_filter: Callable[[int], bool] | None = None,
    ):
        """Fast-path: yield (start, end, key_string, chunk_hash) without creating PoolKey."""
        if not block_hashes:
            return
        base_block_size = self.get_block_size(kv_cache_group_id)
        if cache_family is None:
            cache_family = self.group_cache_families.get(cache_role, {}).get(
                kv_cache_group_id, "default"
            )
        cache_family_ratio = max(infer_cache_family_ratio(cache_family), 1)
        effective_block_size = base_block_size * cache_family_ratio

        grouped_hashes = get_block_hashes(block_hashes, effective_block_size, self.hash_block_size)
        if not grouped_hashes:
            return

        prefix = self._get_key_prefix(kv_cache_group_id, cache_role, cache_family)

        lookup_end = token_len if max_num is None else min(token_len, max_num)
        for chunk_id in range(len(grouped_hashes)):
            start_token = chunk_id * effective_block_size
            if start_token >= lookup_end:
                break
            end_token = min(start_token + effective_block_size, lookup_end)
            if start_token < mask_num:
                continue
            start_idx = start_token // cache_family_ratio
            end_idx = end_token // cache_family_ratio
            if end_idx <= start_idx:
                continue
            if chunk_filter is not None and not chunk_filter(start_idx):
                continue
            hash_val = grouped_hashes[chunk_id]
            chunk_hash = _block_hash_to_hex(hash_val)
            yield start_idx, end_idx, prefix + chunk_hash, hash_val

    def process_token_key_strings_with_block_ids(
        self,
        token_len: int,
        block_hashes: BlockHashList | list[str],
        block_ids: list[int],
        mask_num: int = 0,
        kv_cache_group_id: int = 0,
        skip_null_blocks: bool = False,
        cache_role: str = "kv",
        cache_family: str | None = None,
        chunk_filter: Callable[[int], bool] | None = None,
    ):
        """Fast-path: yield (start, end, key_string, chunk_hash, block_id) without materializing."""
        base_block_size = self.get_block_size(kv_cache_group_id)
        if cache_family is None:
            cache_family = self.group_cache_families.get(cache_role, {}).get(
                kv_cache_group_id, "default"
            )
        cache_family_ratio = max(infer_cache_family_ratio(cache_family), 1)
        effective_block_size = base_block_size * cache_family_ratio

        grouped_hashes = get_block_hashes(block_hashes, effective_block_size, self.hash_block_size)
        if not grouped_hashes:
            return

        num_by_hashes = len(grouped_hashes)
        num_by_token_len = cdiv(token_len, effective_block_size) if token_len > 0 else 0
        num_logical_blocks = min(num_by_hashes, num_by_token_len)
        block_id_offset = max(num_logical_blocks - len(block_ids), 0)

        prefix = self._get_key_prefix(kv_cache_group_id, cache_role, cache_family)

        for chunk_id in range(num_logical_blocks):
            start_token = chunk_id * effective_block_size
            if start_token >= token_len:
                break
            end_token = min(start_token + effective_block_size, token_len)
            if start_token < mask_num:
                continue
            start_idx = start_token // cache_family_ratio
            end_idx = end_token // cache_family_ratio
            if end_idx <= start_idx:
                continue
            if chunk_filter is not None and not chunk_filter(start_idx):
                continue
            block_idx = start_idx // base_block_size - block_id_offset
            if block_idx < 0 or block_idx >= len(block_ids):
                continue
            block_id = block_ids[block_idx]
            if skip_null_blocks and block_id <= 0:
                continue
            hash_val = grouped_hashes[chunk_id]
            chunk_hash = _block_hash_to_hex(hash_val)
            yield start_idx, end_idx, prefix + chunk_hash, hash_val, block_id

    def process_token_key_strings_with_block_ids_sparse_store_mask(
        self,
        token_len: int,
        block_hashes: BlockHashList | list[str],
        block_ids: list[int],
        store_mask: Sequence[bool],
        mask_num: int = 0,
        kv_cache_group_id: int = 0,
        skip_null_blocks: bool = False,
        cache_role: str = "kv",
        cache_family: str | None = None,
        shard_rank: int | None = None,
        shard_size: int | None = None,
    ):
        """Yield only chunks allowed by store_mask.

        This is equivalent to process_token_key_strings_with_block_ids()
        followed by mask_allows_chunk(), because the mask index is
        start // base_block_size, which is the chunk id after cache-family
        scaling.
        """
        if not block_hashes or not store_mask:
            return
        base_block_size = self.get_block_size(kv_cache_group_id)
        if cache_family is None:
            cache_family = self.group_cache_families.get(cache_role, {}).get(
                kv_cache_group_id, "default"
            )
        cache_family_ratio = max(infer_cache_family_ratio(cache_family), 1)
        effective_block_size = base_block_size * cache_family_ratio

        use_shard = shard_rank is not None and shard_size is not None and shard_size > 1
        if use_shard:
            shard_rank = shard_rank % shard_size
            num_by_hashes = get_num_block_hashes(block_hashes, effective_block_size, self.hash_block_size)
        else:
            grouped_hashes = get_block_hashes(block_hashes, effective_block_size, self.hash_block_size)
            if not grouped_hashes:
                return
            num_by_hashes = len(grouped_hashes)

        num_by_token_len = cdiv(token_len, effective_block_size) if token_len > 0 else 0
        num_logical_blocks = min(num_by_hashes, num_by_token_len)
        block_id_offset = max(num_logical_blocks - len(block_ids), 0)
        max_chunks = min(num_logical_blocks, len(store_mask))

        prefix = self._get_key_prefix(kv_cache_group_id, cache_role, cache_family)
        candidate_index = 0

        for chunk_id in range(max_chunks):
            if not store_mask[chunk_id]:
                continue
            start_token = chunk_id * effective_block_size
            if start_token >= token_len:
                break
            end_token = min(start_token + effective_block_size, token_len)
            if start_token < mask_num:
                continue
            start_idx = start_token // cache_family_ratio
            end_idx = end_token // cache_family_ratio
            if end_idx <= start_idx:
                continue
            block_idx = start_idx // base_block_size - block_id_offset
            if block_idx < 0 or block_idx >= len(block_ids):
                continue
            block_id = block_ids[block_idx]
            if skip_null_blocks and block_id <= 0:
                continue
            if use_shard:
                assert shard_size is not None and shard_rank is not None
                if candidate_index % shard_size != shard_rank:
                    candidate_index += 1
                    continue
                candidate_index += 1
                hash_val = get_block_hash_at(block_hashes, chunk_id, effective_block_size, self.hash_block_size)
                if hash_val is None:
                    continue
            else:
                hash_val = grouped_hashes[chunk_id]
            chunk_hash = _block_hash_to_hex(hash_val)
            yield start_idx, end_idx, prefix + chunk_hash, hash_val, block_id

    def store_mask(
        self,
        aligned_token_len: int,
        num_prompt_tokens: int | None = None,
    ) -> tuple[list[bool], ...] | None:
        if self.cache_coordinator is None:
            return None
        return self.cache_coordinator.store_mask(aligned_token_len, num_prompt_tokens)

    def load_mask(
        self,
        block_hashes: list[BlockHash],
        token_len: int,
    ) -> tuple[list[bool], ...] | None:
        if self.cache_coordinator is None:
            return None
        return self.cache_coordinator.load_mask(block_hashes, token_len)

    def mask_allows_chunk(
        self,
        masks: tuple[list[bool], ...] | None,
        kv_cache_group_id: int,
        start: int,
    ) -> bool:
        if masks is None or kv_cache_group_id >= len(masks):
            return True
        group_mask = masks[kv_cache_group_id]
        chunk_idx = start // self.get_block_size(kv_cache_group_id)
        return chunk_idx < len(group_mask) and group_mask[chunk_idx]

    def _make_key_by_hash(
        self,
        chunk_hash: str,
        kv_cache_group_id: int = 0,
        cache_role: str = "kv",
        cache_family: str | None = None,
        chunk_hash_bytes: BlockHash | str | None = None,
        layer_id: int | None = None,
    ):
        assert self.metadata is not None
        if cache_family is None:
            cache_family = self.group_cache_families.get(cache_role, {}).get(kv_cache_group_id, "default")
        return PoolKey(
            KeyMetadata(
                model_name=self.metadata.model_name,
                head_or_tp_rank=self.metadata.head_or_tp_rank,
                pcp_rank=self.metadata.pcp_rank,
                dcp_rank=self.metadata.dcp_rank,
                pp_rank=self.metadata.pp_rank,
                kv_cache_group_id=kv_cache_group_id,
                cache_role=cache_role,
                cache_family=cache_family,
            ),
            chunk_hash,
            chunk_hash_bytes=chunk_hash_bytes,
        )

    def set_kv_caches_base_addr(self, kv_caches_base_addr: list[int]):
        self.kv_caches_base_addr = kv_caches_base_addr

    def set_block_len(self, block_len: list[int]):
        self.block_len = block_len

    def set_block_stride(self, block_stride: list[int]):
        self.block_stride = block_stride

    def get_block_size(self, kv_cache_group_id: int) -> int:
        if kv_cache_group_id >= len(self.block_size):
            return self.block_size[0]
        return self.block_size[kv_cache_group_id]

    def set_group_buffers(
        self,
        group_kv_caches_base_addr: dict[int, list[int]],
        group_block_len: dict[int, list[int]],
        group_block_stride: dict[int, list[int]] | None = None,
        cache_role: str = "kv",
        group_cache_families: dict[int, str] | None = None,
        group_num_layers: dict[int, int] | None = None,
    ) -> None:
        if cache_role == "state":
            # Keep the interface for future explicit state groups, but this
            # DSV4 branch stores compressor/indexer states in kv_caches.
            pass
        else:
            self.group_kv_caches_base_addr = group_kv_caches_base_addr
            self.group_block_len = group_block_len
            self.group_block_stride = group_block_stride or {}
        if group_cache_families is not None:
            self.group_cache_families[cache_role] = group_cache_families.copy()
            self._key_prefix_cache.clear()
        if group_num_layers is not None:
            self.group_num_layers[cache_role] = group_num_layers.copy()

    def _get_group_cache_family(self, kv_cache_group_id: int, cache_role: str = "kv") -> str:
        return self.group_cache_families.get(cache_role, {}).get(kv_cache_group_id, "default")

    def _get_store_granularity(self, kv_cache_group_id: int, cache_role: str = "kv") -> int:
        return get_cache_family_granularity(
            self.get_block_size(kv_cache_group_id),
            self._get_group_cache_family(kv_cache_group_id, cache_role),
        )

    def _get_group_buffers(
        self, kv_cache_group_id: int, cache_role: str = "kv"
    ) -> tuple[list[int], list[int], list[int] | None]:
        if cache_role == "state":
            return [], [], []
        return (
            self.group_kv_caches_base_addr.get(kv_cache_group_id, self.kv_caches_base_addr),
            self.group_block_len.get(kv_cache_group_id, self.block_len),
            self.group_block_stride.get(kv_cache_group_id, self.block_stride),
        )

    def prepare_value(
        self,
        start: int,
        end: int,
        block_ids: list[int],
        kv_cache_group_id: int = 0,
        cache_role: str = "kv",
        block_id: int | None = None,
    ):
        addr_list: list[int] = []
        size_list: list[int] = []
        group_block_size = self.get_block_size(kv_cache_group_id)
        if block_id is None:
            block_id = block_ids[start // group_block_size]
        group_addrs, group_block_len, group_block_stride = self._get_group_buffers(kv_cache_group_id, cache_role)
        length = len(group_block_len)
        if length == 0:
            return addr_list, size_list, block_id
        for index, base_addr in enumerate(group_addrs):
            block_len = group_block_len[index % length]
            block_stride = group_block_stride[index % length] if group_block_stride else block_len
            addr = base_addr + block_id * block_stride
            size = int(block_len / group_block_size * (end - start))
            addr_list.append(addr)
            size_list.append(size)
        return addr_list, size_list, block_id

    def prepare_value_layer(self, start: int, end: int, block_ids: list[int], layer_id: int):
        group_block_size = self.get_block_size(0)
        block_id = block_ids[start // group_block_size]
        addr_list: list[int] = []
        size_list: list[int] = []
        length = len(self.block_len)
        for i in range(length):
            block_stride = self.block_stride[i] if self.block_stride else self.block_len[i]
            addr = self.kv_caches_base_addr[layer_id * length] + block_id * block_stride
            size = int(self.block_len[i] / group_block_size * (end - start))
            addr_list.append(addr)
            size_list.append(size)
        return addr_list, size_list

    def process_tokens(
        self,
        token_len: int,
        block_hashes: BlockHashList | list[str],
        mask_num: int = 0,
        kv_cache_group_id: int = 0,
        cache_role: str = "kv",
        cache_family: str | None = None,
        chunk_filter: Callable[[int], bool] | None = None,
    ) -> Iterable[tuple[int, int, PoolKey]]:
        """Process the tokens and return the corresponding cache engine keys."""
        if not block_hashes:
            return
        base_block_size = self.get_block_size(kv_cache_group_id)
        if cache_family is None:
            cache_family = self.group_cache_families.get(cache_role, {}).get(kv_cache_group_id, "default")
        cache_family_ratio = max(infer_cache_family_ratio(cache_family), 1)
        effective_block_size = base_block_size * cache_family_ratio
        grouped_hashes = get_block_hashes(
            block_hashes,
            effective_block_size,
            self.hash_block_size,
        )
        if not grouped_hashes:
            return
        for chunk_id in range(len(grouped_hashes)):
            start_token = chunk_id * effective_block_size
            if start_token >= token_len:
                break
            end_token = min(start_token + effective_block_size, token_len)
            if start_token < mask_num:
                continue
            start_idx = start_token // cache_family_ratio
            end_idx = end_token // cache_family_ratio
            if end_idx <= start_idx:
                continue
            if chunk_filter is not None and not chunk_filter(start_idx):
                continue
            hash_val = grouped_hashes[chunk_id]
            chunk_hash = _block_hash_to_hex(hash_val)
            yield (
                start_idx,
                end_idx,
                self._make_key_by_hash(
                    chunk_hash,
                    kv_cache_group_id=kv_cache_group_id,
                    cache_role=cache_role,
                    cache_family=cache_family,
                    chunk_hash_bytes=hash_val,
                ),
            )

    def process_tokens_with_block_ids(
        self,
        token_len: int,
        block_hashes: BlockHashList | list[str],
        block_ids: list[int],
        mask_num: int = 0,
        kv_cache_group_id: int = 0,
        skip_null_blocks: bool = False,
        cache_role: str = "kv",
        cache_family: str | None = None,
        chunk_filter: Callable[[int], bool] | None = None,
    ) -> Iterable[tuple[int, int, PoolKey, int]]:
        if not block_hashes:
            return
        base_block_size = self.get_block_size(kv_cache_group_id)
        if cache_family is None:
            cache_family = self.group_cache_families.get(cache_role, {}).get(kv_cache_group_id, "default")
        cache_family_ratio = max(infer_cache_family_ratio(cache_family), 1)
        effective_block_size = base_block_size * cache_family_ratio
        grouped_hashes = get_block_hashes(
            block_hashes,
            effective_block_size,
            self.hash_block_size,
        )
        if not grouped_hashes:
            return

        num_by_hashes = len(grouped_hashes)
        num_by_token_len = cdiv(token_len, effective_block_size) if token_len > 0 else 0
        num_logical_blocks = min(num_by_hashes, num_by_token_len)
        block_id_offset = max(num_logical_blocks - len(block_ids), 0)

        for chunk_id in range(num_logical_blocks):
            start_token = chunk_id * effective_block_size
            if start_token >= token_len:
                break
            end_token = min(start_token + effective_block_size, token_len)
            if start_token < mask_num:
                continue
            start_idx = start_token // cache_family_ratio
            end_idx = end_token // cache_family_ratio
            if end_idx <= start_idx:
                continue
            if chunk_filter is not None and not chunk_filter(start_idx):
                continue
            block_idx = start_idx // base_block_size - block_id_offset
            if block_idx < 0 or block_idx >= len(block_ids):
                continue
            block_id = block_ids[block_idx]
            if skip_null_blocks and block_id <= 0:
                continue
            hash_val = grouped_hashes[chunk_id]
            chunk_hash = _block_hash_to_hex(hash_val)
            key = self._make_key_by_hash(
                chunk_hash,
                kv_cache_group_id=kv_cache_group_id,
                cache_role=cache_role,
                cache_family=cache_family,
                chunk_hash_bytes=hash_val,
            )
            yield start_idx, end_idx, key, block_id

    def decode_adaptor_prefill_pp(self, key, addr, size, kv_cache_group_id: int = 0, cache_role: str = "kv"):
        if self.partitions is None or len(self.partitions) == 1:
            return key, addr, size

        new_key = []
        new_addr = []
        new_size = []

        group_num_layers = self.group_num_layers.get(cache_role, {}).get(kv_cache_group_id, 0)
        for i, (addr_list, size_list) in enumerate(zip(addr, size)):
            caches_per_layer = len(addr_list) // group_num_layers if group_num_layers else 2
            caches_per_layer = max(caches_per_layer, 1)
            start = 0
            for j, part in enumerate(self.partitions):
                end = len(addr_list) if j == len(self.partitions) - 1 else start + part * caches_per_layer
                new_str = key[i].replace(  # type: ignore[attr-defined]
                    "@pp_rank:0", f"@pp_rank:{j}", 1
                )
                new_key.append(new_str)
                new_addr.append(addr_list[start:end])
                new_size.append(size_list[start:end])
                start = end
        return new_key, new_addr, new_size


def normalize_block_ids_by_group(block_ids: tuple[list[int], ...] | list[int] | list[list[int]]) -> list[list[int]]:
    if isinstance(block_ids, tuple):
        return [group.copy() for group in block_ids]
    if isinstance(block_ids, list):
        if not block_ids:
            return [[]]
        if isinstance(block_ids[0], list):
            grouped_block_ids = cast(list[list[int]], block_ids)
            return [group.copy() for group in grouped_block_ids]
        flat_block_ids = cast(list[int], block_ids)
        return [flat_block_ids.copy()]
    raise ValueError(f"Unsupported block_ids type {type(block_ids)}")


def get_block_hashes(
    block_hashes: Sequence[BlockHash | str],
    group_block_size: int,
    hash_block_size: int,
) -> Sequence[BlockHash | str]:
    if group_block_size == hash_block_size:
        return block_hashes
    assert group_block_size % hash_block_size == 0, "block_size must be divisible by hash_block_size"
    return _LazyGroupedBlockHashList(block_hashes, group_block_size // hash_block_size)


def get_num_block_hashes(
    block_hashes: Sequence[BlockHash | str],
    group_block_size: int,
    hash_block_size: int,
) -> int:
    if group_block_size == hash_block_size:
        return len(block_hashes)
    assert group_block_size % hash_block_size == 0, "block_size must be divisible by hash_block_size"
    return len(block_hashes) // (group_block_size // hash_block_size)


def get_block_hash_at(
    block_hashes: Sequence[BlockHash | str],
    chunk_id: int,
    group_block_size: int,
    hash_block_size: int,
) -> BlockHash | str | None:
    if group_block_size == hash_block_size:
        if chunk_id >= len(block_hashes):
            return None
        return block_hashes[chunk_id]
    assert group_block_size % hash_block_size == 0, "block_size must be divisible by hash_block_size"
    scale_factor = group_block_size // hash_block_size
    start = chunk_id * scale_factor
    end = start + scale_factor
    if end > len(block_hashes):
        return None
    return _rehash_block_hash_group(block_hashes[start:end])


class _LazyGroupedBlockHashList(Sequence[BlockHash]):
    def __init__(self, block_hashes: Sequence[BlockHash | str], scale_factor: int) -> None:
        self._block_hashes = block_hashes
        self._scale_factor = scale_factor
        self._length = len(block_hashes) // scale_factor
        self._cache: dict[int, BlockHash] = {}

    def __len__(self) -> int:
        return self._length

    def __iter__(self):
        for idx in range(self._length):
            yield self[idx]

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[idx] for idx in range(*index.indices(self._length))]
        if index < 0:
            index += self._length
        if index < 0 or index >= self._length:
            raise IndexError(index)
        cached = self._cache.get(index)
        if cached is None:
            start = index * self._scale_factor
            end = start + self._scale_factor
            cached = _rehash_block_hash_group(self._block_hashes[start:end])
            self._cache[index] = cached
        return cached


def _rehash_block_hash_group(block_hashes: Sequence[BlockHash | str]) -> BlockHash:
    hasher = hashlib.sha256()
    hasher.update(_GROUPED_BLOCK_HASH_DOMAIN)
    hasher.update(len(block_hashes).to_bytes(_GROUPED_BLOCK_HASH_LENGTH_PREFIX_BYTES, "big"))
    for block_hash in block_hashes:
        hash_bytes = _block_hash_to_bytes(block_hash)
        hasher.update(len(hash_bytes).to_bytes(_GROUPED_BLOCK_HASH_LENGTH_PREFIX_BYTES, "big"))
        hasher.update(hash_bytes)
    return BlockHash(hasher.digest())


def _block_hash_to_hex(block_hash: BlockHash | str) -> str:
    if isinstance(block_hash, str):
        return block_hash
    return bytes(block_hash).hex()


def _block_hash_to_bytes(block_hash: BlockHash | str) -> bytes:
    if isinstance(block_hash, str):
        try:
            return bytes.fromhex(block_hash)
        except ValueError:
            pass
        return block_hash.encode("utf-8")
    return bytes(block_hash)


# Parameters related to the connector metadata
@dataclass
class LoadSpec:
    # Number of tokens cached in vLLM
    vllm_cached_tokens: int
    # Number of tokens that are cached in kvpool
    kvpool_cached_tokens: int
    # Whether the scheduler allow us to load the tokens
    can_load: bool

    token_len: int = 0


@dataclass(init=False)
class RequestTracker:
    # Request id
    req_id: str

    token_len: int

    # The block ids that has been allocated so far, grouped by KV cache group.
    # NOTE: allocated blocks could be more than the number of tokens.
    allocated_block_ids_by_group: list[list[int]]

    # The number of tokens that has been savd
    num_saved_tokens: int = 0

    # The token ids that has been scheduled so far
    # NOTE: This field will only be used when you enable kv-event
    token_ids: list[int] | None = None

    # Full prompt length used by retention-aware external store masks.
    num_prompt_tokens: int | None = None

    def __init__(
        self,
        req_id: str,
        token_len: int,
        allocated_block_ids_by_group: list[list[int]] | None = None,
        allocated_block_ids: list[int] | list[list[int]] | None = None,
        num_saved_tokens: int = 0,
        token_ids: list[int] | None = None,
        num_prompt_tokens: int | None = None,
    ) -> None:
        self.req_id = req_id
        self.token_len = token_len
        block_ids = allocated_block_ids_by_group
        if block_ids is None:
            block_ids = normalize_block_ids_by_group(allocated_block_ids or [])
        self.allocated_block_ids_by_group = block_ids
        self.num_saved_tokens = num_saved_tokens
        self.token_ids = token_ids
        self.num_prompt_tokens = num_prompt_tokens

    @property
    def allocated_block_ids(self) -> list[int]:
        return self.allocated_block_ids_by_group[0] if self.allocated_block_ids_by_group else []

    @allocated_block_ids.setter
    def allocated_block_ids(self, block_ids: list[int] | list[list[int]]) -> None:
        self.allocated_block_ids_by_group = normalize_block_ids_by_group(block_ids)

    @staticmethod
    def from_new_request(
        new_request: NewRequestData,
        num_tokens_to_compute: int,
    ) -> RequestTracker:
        """Create the request tracker from a new request."""
        return RequestTracker(
            req_id=new_request.req_id,
            token_ids=new_request.prompt_token_ids[:num_tokens_to_compute].copy(),
            token_len=num_tokens_to_compute,
            allocated_block_ids_by_group=normalize_block_ids_by_group(new_request.block_ids),
            num_saved_tokens=0,
            num_prompt_tokens=len(new_request.prompt_token_ids),
        )

    def update(
        self,
        new_block_ids: tuple[list[int], ...] | list[int],
    ) -> None:
        """Update the request tracker when a running request is scheduled again."""
        normalized = normalize_block_ids_by_group(new_block_ids)
        if len(normalized) > len(self.allocated_block_ids_by_group):
            self.allocated_block_ids_by_group.extend(
                [[] for _ in range(len(normalized) - len(self.allocated_block_ids_by_group))]
            )
        for group_id, ids in enumerate(normalized):
            self.allocated_block_ids_by_group[group_id].extend(ids)


@dataclass(init=False)
class ReqMeta:
    # Request id
    req_id: str
    # Number of tokens in this chunk
    token_len_chunk: int

    block_ids_by_group: list[list[int]]

    block_hashes: list[BlockHash]

    can_save: bool | None = None
    # load_spec
    load_spec: LoadSpec | None = None

    is_last_chunk: bool | None = None

    current_event: torch.npu.Event | None = None
    kv_cache_group_ids: list[int] | None = None
    kv_cache_families_by_group: list[str] | None = None
    skip_null_blocks_by_group: list[bool] | None = None
    disable_tp_key_sharding: bool = False

    # The following parameters are only used for kv event generation
    # TODO: add lora_request which used for gen lora_id/lora_name in kv event
    token_ids: list[int] | None = None
    original_block_size: list[int] | int | None = None
    num_prompt_tokens: int | None = None

    def __init__(
        self,
        req_id: str,
        token_len_chunk: int,
        block_ids_by_group: list[list[int]] | None = None,
        block_hashes: list[BlockHash] | None = None,
        can_save: bool | None = None,
        load_spec: LoadSpec | None = None,
        is_last_chunk: bool | None = None,
        current_event: torch.npu.Event | None = None,
        kv_cache_group_ids: list[int] | None = None,
        kv_cache_families_by_group: list[str] | None = None,
        skip_null_blocks_by_group: list[bool] | None = None,
        disable_tp_key_sharding: bool = False,
        token_ids: list[int] | None = None,
        original_block_size: list[int] | int | None = None,
        num_prompt_tokens: int | None = None,
        block_ids: list[int] | list[list[int]] | None = None,
    ) -> None:
        self.req_id = req_id
        self.token_len_chunk = token_len_chunk
        if block_ids_by_group is None:
            block_ids_by_group = normalize_block_ids_by_group(block_ids or [])
        self.block_ids_by_group = block_ids_by_group
        self.block_hashes = [] if block_hashes is None else block_hashes
        self.can_save = can_save
        self.load_spec = load_spec
        self.is_last_chunk = is_last_chunk
        self.current_event = current_event
        self.kv_cache_group_ids = kv_cache_group_ids
        self.kv_cache_families_by_group = kv_cache_families_by_group
        self.skip_null_blocks_by_group = skip_null_blocks_by_group
        self.disable_tp_key_sharding = disable_tp_key_sharding
        self.token_ids = token_ids
        self.original_block_size = original_block_size
        self.num_prompt_tokens = num_prompt_tokens

    @property
    def block_ids(self) -> list[int]:
        return self.block_ids_by_group[0] if self.block_ids_by_group else []

    @block_ids.setter
    def block_ids(self, block_ids: list[int] | list[list[int]]) -> None:
        self.block_ids_by_group = normalize_block_ids_by_group(block_ids)

    @staticmethod
    def from_request_tracker(
        tracker: RequestTracker,
        block_size: int,
        load_spec: LoadSpec | None = None,
        skip_save: bool | None = False,
        block_hashes: list[BlockHash] | None = None,
        is_last_chunk: bool | None = None,
        discard_partial_chunks: bool = True,
        original_block_size: list[int] | int | None = None,
        kv_cache_group_families: list[str] | None = None,
    ) -> ReqMeta | None:
        """Create the request metadata from a request tracker."""
        if block_hashes is None:
            block_hashes = []
        input_token_len = tracker.token_len

        # For save operation: do not save if the following condition is met
        # 1. has already been saved before (num_saved_tokens > 0)
        # 2. number of unsaved tokens is not reached the chunk boundary
        chunk_boundary = cdiv(tracker.num_saved_tokens + 1, block_size) * block_size if discard_partial_chunks else 0
        num_tokens_to_save = (input_token_len // block_size * block_size) if discard_partial_chunks else input_token_len

        skip_save = skip_save or num_tokens_to_save < chunk_boundary
        if skip_save and load_spec is None:
            return None

        if not skip_save:
            tracker.num_saved_tokens = num_tokens_to_save

        token_ids = None
        if tracker.token_ids:
            token_ids = tracker.token_ids

        if load_spec is not None and load_spec.can_load:
            logger.debug(
                "Scheduled to load %d tokens for request %s",
                load_spec.kvpool_cached_tokens,
                tracker.req_id,
            )
        else:
            load_spec = None
        logger.debug("request:%s, meta save spec:%s, meta load spec:%s", tracker.req_id, not skip_save, load_spec)
        return ReqMeta(
            req_id=tracker.req_id,
            token_len_chunk=num_tokens_to_save,
            block_ids_by_group=tracker.allocated_block_ids_by_group,
            can_save=not skip_save,
            load_spec=load_spec,
            block_hashes=block_hashes,
            is_last_chunk=is_last_chunk,
            token_ids=token_ids,
            original_block_size=original_block_size,
            num_prompt_tokens=tracker.num_prompt_tokens,
            kv_cache_group_ids=list(range(len(tracker.allocated_block_ids_by_group))),
            kv_cache_families_by_group=kv_cache_group_families,
        )


class AscendConnectorMetadata(KVConnectorMetadata):
    def __init__(self, unfinished_request_ids, preempted_req_ids):
        self.requests = []
        self.unfinished_request_ids = unfinished_request_ids
        self.preempted_req_ids = preempted_req_ids

    def add_request(self, req_meta: ReqMeta) -> None:
        """Add a request to the metadata."""
        self.requests.append(req_meta)


@dataclass(init=False)
class LayerMultiBlockReqMeta:
    req_id: str
    keys: list[LayerPoolKey]
    starts: list[int]
    ends: list[int]
    block_ids_by_group: list[list[int]]
    layer_id: int
    is_last_chunk: bool | None = True
    current_event: torch.npu.Event | None = None

    def __init__(
        self,
        req_id: str,
        keys: list[LayerPoolKey],
        starts: list[int],
        ends: list[int],
        block_ids_by_group: list[list[int]] | None = None,
        layer_id: int = 0,
        is_last_chunk: bool | None = True,
        current_event: torch.npu.Event | None = None,
        block_ids: list[int] | list[list[int]] | None = None,
    ) -> None:
        self.req_id = req_id
        self.keys = keys
        self.starts = starts
        self.ends = ends
        if block_ids_by_group is None:
            block_ids_by_group = normalize_block_ids_by_group(block_ids or [])
        self.block_ids_by_group = block_ids_by_group
        self.layer_id = layer_id
        self.is_last_chunk = is_last_chunk
        self.current_event = current_event

    @property
    def block_ids(self) -> list[int]:
        return self.block_ids_by_group[0] if self.block_ids_by_group else []

    @block_ids.setter
    def block_ids(self, block_ids: list[int] | list[list[int]]) -> None:
        self.block_ids_by_group = normalize_block_ids_by_group(block_ids)
