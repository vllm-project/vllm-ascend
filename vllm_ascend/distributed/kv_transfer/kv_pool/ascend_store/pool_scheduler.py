import importlib
import math
from typing import Any, cast

import vllm.envs as envs
import zmq
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.logger import logger
from vllm.utils.math_utils import cdiv
from vllm.utils.network_utils import make_zmq_socket
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    MambaSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request
from vllm.v1.serial_utils import MsgpackEncoder

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend import (
    backend_map,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import (
    AscendConnectorMetadata,
    AscendStoreKVConnectorWorkerMetadata,
    KeyMetadata,
    LoadSpec,
    PoolKey,
    ReqMeta,
    RequestTracker,
    get_cache_family_granularity,
    infer_group_cache_families,
    normalize_block_ids_by_group,
)


class KVPoolScheduler:
    def __init__(
        self,
        vllm_config: "VllmConfig",
        use_layerwise,
        kv_cache_config: KVCacheConfig | None = None,
        page_size_bytes: int = 0,
    ):
        if isinstance(kv_cache_config, int):
            page_size_bytes = kv_cache_config
            kv_cache_config = None
        self.vllm_config = vllm_config
        self.use_layerwise = use_layerwise
        self.kv_cache_config = kv_cache_config
        hf_text_config = getattr(vllm_config.model_config, "hf_text_config", None)
        hf_config = getattr(vllm_config.model_config, "hf_config", hf_text_config)
        self.hf_config = hf_text_config or hf_config
        self.compress_ratios = getattr(hf_text_config, "compress_ratios", None)
        if self.compress_ratios is None:
            self.compress_ratios = getattr(hf_config, "compress_ratios", None)
        self.use_compress = self.compress_ratios is not None
        self.use_hybrid = self._uses_hybrid_kv_cache(vllm_config, kv_cache_config)
        self.kv_cache_group_ids = (
            list(range(len(kv_cache_config.kv_cache_groups)))
            if kv_cache_config is not None and self.use_hybrid
            else [0]
        )
        self.kv_cache_group_families = self._infer_group_families()
        self.need_truncate = self.use_compress
        self.num_swa_blocks = self._infer_swa_blocks()
        if kv_cache_config is not None:
            for kv_cache_group in kv_cache_config.kv_cache_groups:
                kv_cache_spec = kv_cache_group.kv_cache_spec
                if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
                    kv_cache_spec = next(iter(kv_cache_spec.kv_cache_specs.values()))
                if isinstance(kv_cache_spec, MambaSpec) and getattr(kv_cache_spec, "mamba_cache_mode", None) != "align":
                    raise NotImplementedError(
                        "AscendStore hybrid linear-attention support currently requires mamba_cache_mode='align'."
                    )
        if self.use_layerwise and len(self.kv_cache_group_ids) > 1:
            raise NotImplementedError("AscendStore layerwise mode does not yet support hybrid KV cache groups.")
        self.kv_role = vllm_config.kv_transfer_config.kv_role
        self.consumer_is_to_load = vllm_config.kv_transfer_config.kv_connector_extra_config.get(
            "consumer_is_to_load", False
        )
        self.consumer_is_to_put = vllm_config.kv_transfer_config.kv_connector_extra_config.get(
            "consumer_is_to_put", False
        )
        self.load_async = vllm_config.kv_transfer_config.kv_connector_extra_config.get("load_async", False)
        self.save_decode_cache = vllm_config.kv_transfer_config.kv_connector_extra_config.get("save_decode_cache", True)
        # request_id -> (vllm cached tokes, kvpool cached tokens)
        self.load_specs: dict[str, LoadSpec] = {}
        self.pcp_size = getattr(vllm_config.parallel_config, "prefill_context_parallel_size", 1)
        self.dcp_size = getattr(vllm_config.parallel_config, "decode_context_parallel_size", 1)

        self.mamba_group_ids = self._infer_mamba_groups()
        self.num_speculative_blocks = (
            vllm_config.speculative_config.num_speculative_tokens if vllm_config.speculative_config else 0
        )
        self.original_block_size = self._infer_group_block_sizes(vllm_config, kv_cache_config)
        cp_scale = self.pcp_size * self.dcp_size
        self.grouped_block_size = [block_size * cp_scale for block_size in self.original_block_size]
        requested_hash_block_size = vllm_config.cache_config.hash_block_size
        if not isinstance(requested_hash_block_size, int):
            requested_hash_block_size = None
        self.hash_block_size = (
            requested_hash_block_size if requested_hash_block_size is not None else min(self.original_block_size)
        ) * cp_scale
        for group_block_size in self.grouped_block_size:
            assert group_block_size % self.hash_block_size == 0, "block_size must be divisible by hash_block_size"
        self._block_size = self.grouped_block_size[0]
        self.lcm_block_size = math.lcm(*self.grouped_block_size)
        self.cache_transfer_granularity = self._infer_cache_transfer_granularity()
        # request_id -> full_token_ids
        self._request_trackers: dict[str, RequestTracker] = {}
        self._preempted_req_ids: set[str] = set()
        # Whether to discard partial chunks
        self._discard_partial_chunks = vllm_config.kv_transfer_config.get_from_extra_config(
            "discard_partial_chunks", True
        )
        if self.use_layerwise:
            self._discard_partial_chunks = vllm_config.kv_transfer_config.get_from_extra_config(
                "discard_partial_chunks", False
            )
        self._unfinished_requests: dict[str, tuple[Request, list[list[int]]]] = {}
        self._unfinished_request_ids: set[str] = set()
        self._loading_req_ids: set[str] = set()
        self._delayed_free_req_ids: set[str] = set()

        self._block_pool: BlockPool | None = None
        self.sending_event_id = 0
        # {event_id, flattened block_ids}
        self.sending_blocks: dict[int, list[int]] = {}
        # {event_id, completed_woke_count}
        self.sending_events: dict[int, int] = {}
        self._expected_worker_count = vllm_config.parallel_config.world_size

        self.page_size_bytes = page_size_bytes
        logger.info("KV pool page_size_bytes: %d", page_size_bytes)
        backend_name = vllm_config.kv_transfer_config.kv_connector_extra_config.get("backend", "mooncake")
        self.backend_name = backend_name.lower()
        self.use_gva_layerwise = self.use_layerwise and self.backend_name == "memcache"
        backend = backend_map.get(self.backend_name)
        if backend is None:
            raise ValueError(f"Unsupported KV pool backend: {backend_name}")
        backend_path = backend.get("path")
        backend_class_name = backend.get("name")
        assert backend_path is not None and backend_class_name is not None
        backend_module = importlib.import_module(backend_path)
        backend_class = getattr(backend_module, backend_class_name)
        self.store_scheduler = backend_class.create_scheduler_client(vllm_config.parallel_config)

        model_config = vllm_config.model_config
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.pp_size = vllm_config.parallel_config.pipeline_parallel_size
        self.pp_rank = (vllm_config.parallel_config.rank // self.tp_size) % self.pp_size
        self.use_mla = False
        if hasattr(model_config, "use_mla") and isinstance(model_config.use_mla, bool) and model_config.use_mla:
            self.use_mla = True
        if self.use_mla:
            self.num_kv_head = 1
        else:
            self.num_kv_head = model_config.get_total_num_kv_heads()
        if self.num_kv_head < self.tp_size:
            self.put_step = self.tp_size // self.num_kv_head
        else:
            self.put_step = 1
        self.num_layers = vllm_config.model_config.get_num_layers(vllm_config.parallel_config)
        self.model_name = model_config.model.split("/")[-1]

        # Keep this in sync with pool_worker.py because it affects GVA allocation size.
        num_layer_keys = self.num_layers if self.use_gva_layerwise else 1
        keys_per_block_hash = self.pcp_size * self.dcp_size * (self.tp_size // self.put_step) * num_layer_keys
        self.keys_per_block_hash = keys_per_block_hash
        self.client: LookupKeyClient | None = None

    def _get_or_create_request_tracker(self, req_id: str) -> RequestTracker:
        tracker = self._request_trackers.get(req_id)
        if tracker is None:
            tracker = RequestTracker(
                req_id=req_id,
                token_len=0,
                allocated_block_ids=[],
            )
            self._request_trackers[req_id] = tracker
        return tracker

    def _generate_keys_and_alloc(
        self,
        block_hashes,
        request_tracker: RequestTracker,
        has_last_block=False,
    ) -> None:
        keys_to_alloc, last_block_key = self.generate_keys(
            block_hashes,
            req_id=request_tracker.req_id,
            has_last_block=has_last_block,
        )
        alloc_size = self.page_size_bytes * self.keys_per_block_hash

        last_block_gva = request_tracker.last_block_gva
        num_new_block_keys = len(keys_to_alloc)
        if last_block_key and last_block_gva is None:
            keys_to_alloc.append(last_block_key)
        if keys_to_alloc:
            new_gvas = self.store_scheduler.batch_alloc(keys_to_alloc, [alloc_size] * len(keys_to_alloc))
            if any(gva <= 0 for gva in new_gvas):
                raise ValueError(f"Request {request_tracker.req_id}: batch_alloc failed, gvas={new_gvas}")

            request_tracker.block_gvas.extend(new_gvas[:num_new_block_keys])
            request_tracker.block_keys.extend(keys_to_alloc[:num_new_block_keys])
            if last_block_key is not None and len(new_gvas) > num_new_block_keys:
                request_tracker.last_block_key = last_block_key
                request_tracker.last_block_gva = new_gvas[-1]

    def _ensure_tracker_gvas_cover_blocks(
        self,
        request_tracker: RequestTracker,
        block_hashes,
    ) -> None:
        """Ensure layerwise KV pool GVA exists for all requested full blocks.

        Layerwise transfer uses batch_copy with explicit GVA addresses instead
        of the normal key-based put/get path. Existing block keys reuse their
        stored GVA; missing keys are allocated here so later layer load/save
        tasks can build complete transfer arrays.
        """
        block_keys, _ = self.generate_keys(block_hashes)
        if not block_keys:
            request_tracker.block_keys = []
            request_tracker.block_gvas = []
            request_tracker.gva_block_offset = 0
            return

        key_infos = self.store_scheduler.batch_get_key_info(block_keys)
        block_gvas = [0] * len(block_keys)
        missing_keys = []
        missing_indices = []
        for index, key_info in enumerate(key_infos):
            sizes = key_info.size()
            if sizes and sizes > 0:
                block_gvas[index] = key_info.gva_list()[0]
            else:
                missing_keys.append(block_keys[index])
                missing_indices.append(index)

        if missing_keys:
            alloc_size = self.page_size_bytes * self.keys_per_block_hash
            new_gvas = self.store_scheduler.batch_alloc(missing_keys, [alloc_size] * len(missing_keys))
            if any(gva <= 0 for gva in new_gvas):
                raise ValueError(f"Request {request_tracker.req_id}: batch_alloc failed, gvas={new_gvas}")
            for index, gva in zip(missing_indices, new_gvas):
                block_gvas[index] = gva

        request_tracker.block_keys = block_keys
        request_tracker.block_gvas = block_gvas
        request_tracker.gva_block_offset = 0

    def generate_keys(self, block_hashes, req_id="", has_last_block=False):
        block_keys = []
        for block_hash in block_hashes:
            key = f"{self.model_name}@{block_hash.hex()}"
            block_keys.append(key)

        last_block_key = None
        if has_last_block:
            last_block_key = f"{self.model_name}@{req_id}_lastblock"

        return block_keys, last_block_key

    def _generate_store_query_keys(
        self,
        block_hashes,
        include_layers: bool = False,
        kv_cache_group_id: int = 0,
    ) -> list[list[str]]:
        head_or_tp_ranks = self.tp_size // self.put_step
        cache_family = self._get_group_family(self.kv_cache_group_families, kv_cache_group_id)
        keys_by_block = []
        for block_hash in block_hashes:
            block_keys: list[str] = []
            chunk_hash = block_hash if isinstance(block_hash, str) else block_hash.hex()
            pp_ranks = [self.pp_rank] if include_layers else range(self.pp_size)
            for pcp_rank in range(self.pcp_size):
                for dcp_rank in range(self.dcp_size):
                    for head_or_tp_rank in range(head_or_tp_ranks):
                        for pp_rank in pp_ranks:
                            pool_key = PoolKey(
                                KeyMetadata(
                                    self.model_name,
                                    head_or_tp_rank,
                                    pcp_rank,
                                    dcp_rank,
                                    pp_rank,
                                    kv_cache_group_id=kv_cache_group_id,
                                    cache_family=cache_family,
                                ),
                                chunk_hash,
                            )
                            if include_layers:
                                block_keys.extend(
                                    layer_key.to_string() for layer_key in pool_key.split_layers(self.num_layers)
                                )
                            else:
                                block_keys.append(pool_key.to_string())
            keys_by_block.append(block_keys)
        return keys_by_block

    def _get_store_lookup_hit_tokens(
        self,
        request: "Request",
        token_len: int,
        num_computed_tokens: int,
        include_layers: bool = False,
    ) -> int:
        num_blocks = token_len // self._block_size
        # In layerwise mode, always query from block 0 because the remote
        # pool stores per-layer data that may not match local prefix cache.
        query_start_block = 0 if self.use_layerwise else min(num_computed_tokens // self._block_size, num_blocks)
        block_hashes_to_query = request.block_hashes[query_start_block:num_blocks]
        if not block_hashes_to_query:
            return 0

        query_keys_by_block = self._generate_store_query_keys(
            block_hashes_to_query,
            include_layers=include_layers,
        )
        query_keys = [key for block_keys in query_keys_by_block for key in block_keys]
        exists_states = self.store_scheduler.batch_is_exist(query_keys)
        if len(exists_states) != len(query_keys):
            raise RuntimeError(
                "KV pool exists check returned unexpected number of "
                f"states for request {request.request_id}: "
                f"expected={len(query_keys)}, actual={len(exists_states)}"
            )

        num_queried_hit_blocks = 0
        offset = 0
        for block_keys in query_keys_by_block:
            block_states = exists_states[offset : offset + len(block_keys)]
            offset += len(block_keys)
            if all(exists == 1 for exists in block_states):
                num_queried_hit_blocks += 1
                continue
            if any(exists == 0 for exists in block_states):
                break
            raise RuntimeError(f"KV pool exists check failed for request {request.request_id}: states={exists_states}")

        num_hit_blocks = query_start_block + num_queried_hit_blocks
        return num_hit_blocks * self._block_size

    def _get_layerwise_gva_hit_tokens(
        self,
        request: "Request",
        token_len: int,
        num_computed_tokens: int,
    ) -> int:
        num_blocks = token_len // self._block_size
        num_queried_hit_blocks = 0
        block_hashes_to_check = request.block_hashes[:num_blocks]
        keys_to_check = [f"{self.model_name}@{bh.hex()}" for bh in block_hashes_to_check]
        # In layerwise mode, always query from block 0 because the remote
        # pool stores per-layer data that may not match local prefix cache.
        query_start_block = 0 if self.use_layerwise else min(num_computed_tokens // self._block_size, num_blocks)
        keys_to_query = keys_to_check[query_start_block:]
        if not keys_to_query:
            return 0
        tracker = self._get_or_create_request_tracker(request.request_id)
        cached_gvas = []
        key_infos = self.store_scheduler.batch_get_key_info(keys_to_query)
        for key_info in key_infos:
            sizes = key_info.size()
            if sizes and sizes > 0:
                cached_gvas.append(key_info.gva_list()[0])
                num_queried_hit_blocks += 1
            else:
                break
        num_hit_blocks = query_start_block + num_queried_hit_blocks
        tracker.block_keys = keys_to_check[query_start_block:num_hit_blocks]
        tracker.block_gvas = cached_gvas[:num_queried_hit_blocks]
        tracker.gva_block_offset = query_start_block
        return num_hit_blocks * self._block_size

    def _infer_group_families(self) -> list[str]:
        kv_cache_groups = self.kv_cache_config.kv_cache_groups if self.kv_cache_config is not None else None
        return infer_group_cache_families(kv_cache_groups, self.compress_ratios, self.hf_config)

    def _infer_group_block_sizes(
        self,
        vllm_config: "VllmConfig",
        kv_cache_config: KVCacheConfig | None,
    ) -> list[int]:
        if kv_cache_config is None or not self.use_hybrid:
            return [vllm_config.cache_config.block_size]

        block_sizes: list[int] = []
        for kv_cache_group in kv_cache_config.kv_cache_groups:
            kv_cache_spec = kv_cache_group.kv_cache_spec
            if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
                kv_cache_spec = next(iter(kv_cache_spec.kv_cache_specs.values()))
            block_sizes.append(kv_cache_spec.block_size)
        return block_sizes

    def _get_group_block_size(self, group_id: int) -> int:
        if group_id >= len(self.grouped_block_size):
            return self.grouped_block_size[0]
        return self.grouped_block_size[group_id]

    def _get_group_family(self, families: list[str], group_id: int) -> str:
        if group_id >= len(families):
            return "default"
        return families[group_id]

    def _infer_cache_transfer_granularity(self) -> int:
        granularities = [self.lcm_block_size]
        for group_id in self.kv_cache_group_ids:
            granularities.append(
                get_cache_family_granularity(
                    self._get_group_block_size(group_id),
                    self._get_group_family(self.kv_cache_group_families, group_id),
                )
            )
        return math.lcm(*granularities)

    def _floor_to_cache_transfer_granularity(self, token_len: int) -> int:
        return token_len // self.cache_transfer_granularity * self.cache_transfer_granularity

    @staticmethod
    def _uses_hybrid_kv_cache(vllm_config: "VllmConfig", kv_cache_config: KVCacheConfig | None) -> bool:
        if kv_cache_config is None:
            return False
        if getattr(vllm_config.scheduler_config, "disable_hybrid_kv_cache_manager", False):
            return False
        return len(kv_cache_config.kv_cache_groups) > 1 and any(
            not isinstance(group.kv_cache_spec, FullAttentionSpec) for group in kv_cache_config.kv_cache_groups
        )

    def _infer_mamba_groups(self):
        if self.kv_cache_config is None or not self.use_hybrid:
            return []
        mamba_group_ids: list[int] = []
        for group_id, kv_cache_group in enumerate(self.kv_cache_config.kv_cache_groups):
            kv_cache_spec = kv_cache_group.kv_cache_spec
            if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
                kv_cache_spec = next(iter(kv_cache_spec.kv_cache_specs.values()))
            if isinstance(kv_cache_spec, MambaSpec):
                mamba_group_ids.append(group_id)
        return mamba_group_ids

    def _infer_swa_blocks(self) -> list[int]:
        if self.kv_cache_config is None:
            return []

        num_swa_blocks: list[int] = []
        for group in self.kv_cache_config.kv_cache_groups:
            kv_cache_spec = group.kv_cache_spec
            if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
                group_specs = []
                for layer_name in group.layer_names:
                    layer_spec = kv_cache_spec.kv_cache_specs[layer_name]
                    if layer_spec not in group_specs:
                        group_specs.append(layer_spec)
            else:
                group_specs = [kv_cache_spec]

            first_spec = group_specs[0]
            if isinstance(first_spec, SlidingWindowSpec):
                num_swa_blocks.append(cdiv(first_spec.sliding_window, first_spec.block_size) + 1)
            else:
                num_swa_blocks.append(0)
            if any(isinstance(spec, MambaSpec) for spec in group_specs):
                self.need_truncate = True
        return num_swa_blocks

    def get_sw_clipped_blocks(
        self,
        block_ids: tuple[list[int], ...] | list[list[int]],
    ) -> tuple[list[int], ...] | list[list[int]]:
        if len(block_ids) == 0 or not self.use_hybrid:
            return block_ids
        assert len(block_ids) == len(self.num_swa_blocks), "Number of KV cache groups must match"
        clipped = [
            blocks[-self.num_swa_blocks[group_id] :] if self.num_swa_blocks[group_id] > 0 else blocks
            for group_id, blocks in enumerate(block_ids)
        ]
        return tuple(clipped) if isinstance(block_ids, tuple) else clipped

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        """
        Check for external KV cache hit.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            the number of tokens that can be loaded from the
            external KV cache beyond what is already computed.
        """
        if self.kv_role == "kv_consumer" and not self.consumer_is_to_load:
            return 0, False

        if self._discard_partial_chunks:
            token_len = self._floor_to_cache_transfer_granularity(len(request.prompt_token_ids))
        else:
            token_len = len(request.prompt_token_ids)

        if token_len < self.cache_transfer_granularity:
            return 0, False

        if self.use_gva_layerwise:
            num_external_hit_tokens = self._get_layerwise_gva_hit_tokens(request, token_len, num_computed_tokens)
        elif self.use_layerwise:
            num_external_hit_tokens = self._get_store_lookup_hit_tokens(
                request, token_len, num_computed_tokens, include_layers=True
            )
        else:
            if self.client is None:
                self.client = LookupKeyClient(self.vllm_config)
            num_external_hit_tokens = self.client.lookup(
                token_len,
                request.block_hashes,
                self.kv_cache_group_ids,
            )

        if num_external_hit_tokens == 0:
            return 0, False

        if num_external_hit_tokens == request.num_tokens:
            num_external_hit_tokens -= 1

        if num_external_hit_tokens < num_computed_tokens:
            need_to_allocate = 0
        else:
            need_to_allocate = num_external_hit_tokens - num_computed_tokens

        logger.info(
            "Reqid: %s, Total tokens %d, kvpool hit tokens: %d, need to load: %d",
            request.request_id,
            request.num_tokens,
            num_external_hit_tokens,
            need_to_allocate,
        )

        # In layerwise mode, even when vLLM has local cached tokens, we still
        # need to load KV cache from the pool because layerwise transfer loads
        # per-layer data that may not be in HBM.
        force_layerwise_load = self.use_layerwise and num_external_hit_tokens > 0
        if need_to_allocate <= 0 and not force_layerwise_load:
            return 0, False

        self.load_specs[request.request_id] = LoadSpec(
            vllm_cached_tokens=num_computed_tokens,
            kvpool_cached_tokens=num_external_hit_tokens,
            can_load=force_layerwise_load,
        )
        logger.info(
            "KV pool load spec created req=%s vllm_cached=%d kvpool_cached=%d "
            "need_to_allocate=%d load_async=%s use_layerwise=%s",
            request.request_id,
            num_computed_tokens,
            num_external_hit_tokens,
            need_to_allocate,
            self.load_async,
            self.use_layerwise,
        )

        return need_to_allocate, self.load_async and not self.use_layerwise

    def update_state_after_alloc(self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int):
        """
        Update KVConnector state after temporary buffer alloc.

        For SharedStorageConnector, update _request_needs_load
        if the CacheManager this allocated blocks for us.
        """
        local_block_ids: list[list[int]] = [[] for _ in self.kv_cache_group_ids]
        if num_external_tokens > 0:
            local_block_ids = normalize_block_ids_by_group(blocks.get_block_ids())

        self._unfinished_requests[request.request_id] = (request, local_block_ids)
        self._unfinished_request_ids.add(request.request_id)
        if request.request_id not in self.load_specs:
            # No KV tokens from external KV cache, return
            logger.debug(
                "KV pool update_state_after_alloc req=%s has no load spec; num_external_tokens=%d",
                request.request_id,
                num_external_tokens,
            )
            return

        if num_external_tokens == 0:
            # No need to load anything
            self.load_specs[request.request_id].can_load = (
                self.use_layerwise and self.load_specs[request.request_id].kvpool_cached_tokens > 0
            )
            logger.debug(
                "KV pool load spec updated req=%s because num_external_tokens=0 "
                "can_load=%s vllm_cached=%d kvpool_cached=%d",
                request.request_id,
                self.load_specs[request.request_id].can_load,
                self.load_specs[request.request_id].vllm_cached_tokens,
                self.load_specs[request.request_id].kvpool_cached_tokens,
            )
            return

        assert (
            num_external_tokens > 0
            and num_external_tokens
            == self.load_specs[request.request_id].kvpool_cached_tokens
            - self.load_specs[request.request_id].vllm_cached_tokens
        ), (
            f"Mismatch in number of tokens: {num_external_tokens} vs "
            f"{self.load_specs[request.request_id].kvpool_cached_tokens} - "
            f"{self.load_specs[request.request_id].vllm_cached_tokens}"
            f" for request {request.request_id}"
        )

        self.load_specs[request.request_id].can_load = True
        if self.load_async and not self.use_layerwise:
            self._loading_req_ids.add(request.request_id)
        logger.debug(
            "KV pool load spec enabled req=%s num_external_tokens=%d vllm_cached=%d kvpool_cached=%d groups=%s",
            request.request_id,
            num_external_tokens,
            self.load_specs[request.request_id].vllm_cached_tokens,
            self.load_specs[request.request_id].kvpool_cached_tokens,
            [len(blocks) for blocks in local_block_ids],
        )

    def _get_last_chunk_tokens_num(self, prompt_token_ids: list[int]) -> int:
        if self._discard_partial_chunks:
            return self._floor_to_cache_transfer_granularity(len(prompt_token_ids))
        return len(prompt_token_ids)

    def _allocate_gva_if_needed(
        self,
        request_tracker: RequestTracker,
        block_hashes,
        num_blocks: int,
        has_last_block: bool,
    ) -> None:
        if not self.use_gva_layerwise:
            return
        self._ensure_tracker_gvas_cover_blocks(
            request_tracker,
            block_hashes[:num_blocks],
        )
        if has_last_block:
            self._generate_keys_and_alloc(
                [],
                request_tracker=request_tracker,
                has_last_block=True,
            )

    def _build_req_meta(
        self,
        request_tracker: RequestTracker,
        block_hashes,
        load_spec: LoadSpec | None,
        prompt_token_ids: list[int],
        skip_save: bool | None,
    ):
        last_chunk_tokens_num = self._get_last_chunk_tokens_num(prompt_token_ids)
        return ReqMeta.from_request_tracker(
            request_tracker,
            self.cache_transfer_granularity,
            load_spec=load_spec,
            skip_save=skip_save,
            block_hashes=block_hashes,
            is_last_chunk=request_tracker.token_len >= last_chunk_tokens_num,
            discard_partial_chunks=self._discard_partial_chunks,
            original_block_size=self.original_block_size,
            kv_cache_group_families=self.kv_cache_group_families,
        )

    def _process_new_request(
        self,
        request: Request,
        scheduler_output: SchedulerOutput,
        force_skip_save: bool,
    ) -> ReqMeta | None:
        load_spec = self.load_specs.pop(request.req_id, None)
        num_tokens_to_compute = request.num_computed_tokens + scheduler_output.num_scheduled_tokens[request.req_id]
        request_tuple = self._unfinished_requests.get(request.req_id)
        if request_tuple is None:
            raise ValueError(
                f"Request {request.req_id} is not in _unfinished_requests, but it is scheduled as a new request"
            )
        request_real = request_tuple[0]
        block_ids_by_group = normalize_block_ids_by_group(request.block_ids)
        previous_tracker = self._request_trackers.get(request.req_id)
        request_tracker = RequestTracker(
            req_id=request.req_id,
            token_len=num_tokens_to_compute,
            allocated_block_ids_by_group=block_ids_by_group,
            num_saved_tokens=0,
            token_ids=request.prompt_token_ids[:num_tokens_to_compute].copy(),
            block_keys=(previous_tracker.block_keys.copy() if previous_tracker else []),
            block_gvas=(previous_tracker.block_gvas.copy() if previous_tracker else []),
            gva_block_offset=(previous_tracker.gva_block_offset if previous_tracker else 0),
            mamba_group_ids=self.mamba_group_ids,
            num_speculative_blocks=self.num_speculative_blocks,
            block_sizes=self.grouped_block_size,
        )
        self._request_trackers[request.req_id] = request_tracker
        num_blocks = num_tokens_to_compute // self._block_size
        has_last_block = num_tokens_to_compute % self._block_size != 0
        self._allocate_gva_if_needed(
            request_tracker,
            request_real.block_hashes,
            num_blocks,
            has_last_block,
        )
        return self._build_req_meta(
            request_tracker,
            request_real.block_hashes,
            load_spec,
            request.prompt_token_ids,
            force_skip_save,
        )

    def _process_preempted_cached_request(
        self,
        new_block_ids,
        req_id: str,
        i: int,
        cached_reqs,
        scheduler_output: SchedulerOutput,
        force_skip_save: bool,
    ) -> ReqMeta | None:
        new_block_ids_by_group = normalize_block_ids_by_group(new_block_ids)
        self._preempted_req_ids.discard(req_id)
        load_spec = self.load_specs.pop(req_id, None)
        request_tuple = self._unfinished_requests.get(req_id)
        if request_tuple is None:
            raise ValueError(
                f"Request {req_id} is not in _unfinished_requests, but it is scheduled as a preempted cached request"
            )
        request_real = request_tuple[0]
        num_tokens_to_compute = request_real.num_computed_tokens + scheduler_output.num_scheduled_tokens[req_id]
        previous_tracker = self._request_trackers.get(req_id)
        request_tracker = RequestTracker(
            req_id=req_id,
            token_len=num_tokens_to_compute,
            allocated_block_ids_by_group=new_block_ids_by_group,
            num_saved_tokens=0,
            token_ids=request_real.prompt_token_ids[:num_tokens_to_compute].copy(),
            block_keys=(previous_tracker.block_keys.copy() if previous_tracker else []),
            block_gvas=(previous_tracker.block_gvas.copy() if previous_tracker else []),
            gva_block_offset=(previous_tracker.gva_block_offset if previous_tracker else 0),
            mamba_group_ids=self.mamba_group_ids,
            num_speculative_blocks=self.num_speculative_blocks,
            block_sizes=self.grouped_block_size,
        )
        self._request_trackers[req_id] = request_tracker
        num_blocks = len(new_block_ids_by_group[0])
        has_last_block = num_tokens_to_compute % self._block_size != 0
        self._allocate_gva_if_needed(
            request_tracker,
            request_real.block_hashes,
            num_blocks,
            has_last_block,
        )
        return self._build_req_meta(
            request_tracker,
            request_real.block_hashes,
            load_spec,
            request_real.prompt_token_ids,
            force_skip_save,
        )

    def _process_running_cached_request(
        self,
        new_block_ids,
        req_id: str,
        i: int,
        cached_reqs,
        scheduler_output: SchedulerOutput,
        force_skip_save: bool,
    ) -> ReqMeta | None:
        if not self.save_decode_cache:
            return None
        request_tracker = self._request_trackers.get(req_id)
        if request_tracker is None:
            raise ValueError(f"Request {req_id} is not in _request_trackers, but it is scheduled to be cached")
        num_new_tokens = scheduler_output.num_scheduled_tokens[req_id]
        req_tuple = self._unfinished_requests.get(req_id)
        if not req_tuple:
            raise ValueError(f"Request {req_id} is not in _unfinished_requests, but it is scheduled to be cached")
        request = req_tuple[0]
        num_current_tokens = request_tracker.token_len
        new_token_ids = request.all_token_ids[num_current_tokens : num_current_tokens + num_new_tokens]
        if request_tracker.token_ids is not None and new_token_ids:
            request_tracker.token_ids.extend(new_token_ids)
        request_tracker.token_len += num_new_tokens

        prev_token_count = request_tracker.token_len - num_new_tokens
        prev_hash_count = prev_token_count // self._block_size
        current_hash_count = request_tracker.token_len // self._block_size
        new_hash_count = current_hash_count - prev_hash_count
        has_last_block = request_tracker.token_len % self._block_size != 0 or current_hash_count > len(
            request.block_hashes
        )
        if self.use_gva_layerwise and (new_hash_count > 0 or has_last_block):
            self._ensure_tracker_gvas_cover_blocks(
                request_tracker,
                request.block_hashes[:current_hash_count],
            )
            if has_last_block:
                self._generate_keys_and_alloc(
                    [],
                    request_tracker=request_tracker,
                    has_last_block=True,
                )
        if new_block_ids is not None:
            request_tracker.update(new_block_ids, request.num_computed_tokens)
        load_spec = None
        return self._build_req_meta(
            request_tracker,
            request.block_hashes,
            load_spec,
            request.prompt_token_ids,
            force_skip_save,
        )

    def _process_async_load_request(
        self,
        request_id: str,
        request: Request,
        block_ids: list[list[int]],
    ) -> ReqMeta | None:
        load_spec = self.load_specs.pop(request_id, None)
        if not load_spec:
            return None
        num_tokens_to_compute = load_spec.kvpool_cached_tokens
        if (num_tokens_to_compute % self._block_size != 0) and (
            num_tokens_to_compute == len(request.prompt_token_ids) - 1
        ):
            num_tokens_to_compute = num_tokens_to_compute + 1
        previous_tracker = self._request_trackers.get(request_id)
        request_tracker = RequestTracker(
            req_id=request_id,
            token_len=num_tokens_to_compute,
            allocated_block_ids_by_group=block_ids,
            num_saved_tokens=0,
            block_keys=(previous_tracker.block_keys.copy() if previous_tracker else []),
            block_gvas=(previous_tracker.block_gvas.copy() if previous_tracker else []),
            gva_block_offset=(previous_tracker.gva_block_offset if previous_tracker else 0),
            mamba_group_ids=self.mamba_group_ids,
            num_speculative_blocks=self.num_speculative_blocks,
            block_sizes=self.grouped_block_size,
        )
        self._request_trackers[request_id] = request_tracker
        num_blocks = num_tokens_to_compute // self._block_size
        has_last_block = num_tokens_to_compute % self._block_size != 0
        self._allocate_gva_if_needed(
            request_tracker,
            request.block_hashes,
            num_blocks,
            has_last_block,
        )
        return ReqMeta.from_request_tracker(
            request_tracker,
            self.cache_transfer_granularity,
            load_spec=load_spec,
            skip_save=None,
            block_hashes=request.block_hashes,
            discard_partial_chunks=self._discard_partial_chunks,
            original_block_size=self.original_block_size,
            kv_cache_group_families=self.kv_cache_group_families,
        )

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        """Attach the connector metadata to the request object.

        This function should NOT modify other fields in the scheduler_output
        except the `kv_connector_metadata` field.
        Also, calling this function will reset the state of the connector.
        """
        force_skip_save = self.kv_role == "kv_consumer" and not self.consumer_is_to_put

        for finished_req_id in scheduler_output.finished_req_ids:
            self._request_trackers.pop(finished_req_id, None)
            self._unfinished_requests.pop(finished_req_id, None)
            self._unfinished_request_ids.discard(finished_req_id)
            self._preempted_req_ids.discard(finished_req_id)
            self._loading_req_ids.discard(finished_req_id)

        for req_id in scheduler_output.preempted_req_ids:
            self._preempted_req_ids.update(scheduler_output.preempted_req_ids)
            self._request_trackers.pop(req_id, None)
            self._unfinished_requests.pop(req_id, None)
            self._loading_req_ids.discard(req_id)
            self._delayed_free_req_ids.discard(req_id)

        meta = AscendConnectorMetadata(
            self._unfinished_request_ids,
            scheduler_output.preempted_req_ids,
            self._loading_req_ids.copy(),
            self._delayed_free_req_ids.copy(),
        )

        for request in scheduler_output.scheduled_new_reqs:
            req_meta = self._process_new_request(request, scheduler_output, force_skip_save)
            if req_meta is not None:
                self.touch_sending_mamba_blocks(req_meta)
                meta.add_request(req_meta)

        cached_reqs = scheduler_output.scheduled_cached_reqs
        if not force_skip_save:
            for i, req_id in enumerate(cached_reqs.req_ids):
                new_block_ids = cached_reqs.new_block_ids[i]
                if not new_block_ids:
                    continue
                if req_id in self._preempted_req_ids:
                    req_meta = self._process_preempted_cached_request(
                        new_block_ids,
                        req_id,
                        i,
                        cached_reqs,
                        scheduler_output,
                        force_skip_save,
                    )
                else:
                    req_meta = self._process_running_cached_request(
                        new_block_ids,
                        req_id,
                        i,
                        cached_reqs,
                        scheduler_output,
                        force_skip_save,
                    )
                if req_meta is not None:
                    self.touch_sending_mamba_blocks(req_meta)
                    meta.add_request(req_meta)

        request_ids = [req.req_id for req in scheduler_output.scheduled_new_reqs]
        for request_id, (request, block_ids) in self._unfinished_requests.items():
            if request_id not in request_ids and request_id not in cached_reqs.req_ids:
                req_meta = self._process_async_load_request(request_id, request, block_ids)
                if req_meta is not None:
                    self.touch_sending_mamba_blocks(req_meta)
                    meta.add_request(req_meta)

        return meta

    def get_sending_event_id(self):
        """
        get a unique event id for a kv store request
        """
        using_id = self.sending_event_id
        # todo: reset sending_event_id, in case infinitely increasing
        self.sending_event_id += 1
        return using_id

    def touch_sending_mamba_blocks(self, req_meta: ReqMeta):
        """
        keep the reference of all non-null mamba blocks that will send to external kv store
        """
        if not self.use_hybrid or len(self.mamba_group_ids) == 0 or not req_meta.can_save:
            return
        using_event_id = self.get_sending_event_id()
        req_meta.event_id = using_event_id
        current_step_sending: list[int] = []
        for group_id in self.mamba_group_ids:
            group_block_ids = req_meta.block_ids_by_group[group_id]
            current_step_sending.extend([block_id for block_id in group_block_ids if block_id > 0])
        logger.debug("event: %s touch blocks: %s", using_event_id, current_step_sending)
        assert self._block_pool is not None
        self._block_pool.touch([self._block_pool.blocks[block_id] for block_id in current_step_sending])
        self.sending_events[using_event_id] = 0
        self.sending_blocks[using_event_id] = current_step_sending

    def update_connector_output(self, connector_output: KVConnectorOutput):
        """
        hand the connector_output, free non-null mamba blocks and so on.
        """
        meta = connector_output.kv_connector_worker_meta
        if not isinstance(meta, AscendStoreKVConnectorWorkerMetadata) or self._block_pool is None:
            return

        for event_id, count in meta.completed_events.items():
            logger.debug("event %s update with %s", event_id, count)
            total = self.sending_events.get(event_id, -1)
            if total == -1:
                logger.warning("worker reports an invalid event: %s, count %s", event_id, count)
                continue
            total = total + count
            if total >= self._expected_worker_count:
                to_free_block_ids = self.sending_blocks.pop(event_id, [])
                self.sending_events.pop(event_id, None)
                if to_free_block_ids:
                    logger.debug("event %s free blocks: %s", event_id, to_free_block_ids)
                    self._block_pool.free_blocks([self._block_pool.blocks[block_id] for block_id in to_free_block_ids])
            else:
                self.sending_events[event_id] = total

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Once a request is finished, determine whether request blocks
        should be freed now or will be sent asynchronously and freed later.
        """
        if self.kv_role == "kv_consumer" and not self.consumer_is_to_put:
            self._delayed_free_req_ids.discard(request.request_id)
            return False, None
        if self.use_layerwise:
            self._delayed_free_req_ids.discard(request.request_id)
            return False, None
        tracker = self._request_trackers.get(request.request_id)
        if tracker is None or tracker.num_saved_tokens <= 0:
            self._delayed_free_req_ids.discard(request.request_id)
            return False, None
        delay_free_blocks = len(block_ids) > 0
        if delay_free_blocks:
            self._delayed_free_req_ids.add(request.request_id)
            logger.debug("Delaying free of %d blocks for request %s", len(block_ids), request.request_id)
        else:
            self._delayed_free_req_ids.discard(request.request_id)
        return delay_free_blocks, None

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        """HMA path for hybrid KV cache groups."""
        if self.kv_role == "kv_consumer" and not self.consumer_is_to_put:
            self._delayed_free_req_ids.discard(request.request_id)
            return False, None
        tracker = self._request_trackers.get(request.request_id)
        if tracker is not None and tracker.num_saved_tokens <= 0:
            self._delayed_free_req_ids.discard(request.request_id)
            return False, None
        block_ids = cast(tuple[list[int], ...], self.get_sw_clipped_blocks(block_ids))
        valid_group_block_ids = [group_block_ids for group_block_ids in block_ids if group_block_ids]
        delay_free_blocks = bool(valid_group_block_ids)
        if delay_free_blocks:
            self._delayed_free_req_ids.add(request.request_id)
            logger.debug(
                "Delaying free of %d KV cache groups for request %s",
                len(valid_group_block_ids),
                request.request_id,
            )
        else:
            self._delayed_free_req_ids.discard(request.request_id)
        return delay_free_blocks, None

    def bind_gpu_block_pool(self, gpu_block_pool: "BlockPool") -> None:
        self._block_pool = gpu_block_pool

    def update_finished_sending(self, finished_sending: set[str] | None) -> None:
        if finished_sending:
            self._delayed_free_req_ids.difference_update(finished_sending)

    def update_finished_recving(self, finished_recving: set[str] | None) -> None:
        if finished_recving:
            self._loading_req_ids.difference_update(finished_recving)


class LookupKeyClient:
    def __init__(self, vllm_config: "VllmConfig"):
        self.encoder = MsgpackEncoder()
        self.ctx = zmq.Context()  # type: ignore[attr-defined]
        socket_path = get_zmq_rpc_path_lookup(vllm_config)
        self.socket = make_zmq_socket(
            self.ctx,
            socket_path,
            zmq.REQ,  # type: ignore[attr-defined]
            bind=False,
        )

    def lookup(
        self,
        token_len: int,
        block_hashes: list[BlockHash],
        kv_cache_group_ids: list[int] | None = None,
    ) -> int:
        kv_cache_group_ids = kv_cache_group_ids or [0]
        hash_strs = [h.hex() for h in block_hashes]
        hash_frames = self.encoder.encode(hash_strs)
        kv_group_frames = self.encoder.encode(kv_cache_group_ids)
        token_len_bytes = token_len.to_bytes(4, byteorder="big")
        all_frames = [token_len_bytes] + list(kv_group_frames) + list(hash_frames)
        self.socket.send_multipart(all_frames, copy=False)
        resp = self.socket.recv()
        result = int.from_bytes(resp, "big")
        return result

    def close(self):
        self.socket.close(linger=0)


def get_zmq_rpc_path_lookup(vllm_config: "VllmConfig") -> str:
    dp_rank = vllm_config.parallel_config.data_parallel_rank
    base_url = envs.VLLM_RPC_BASE_PATH
    # Default to 0 if not configured
    rpc_port = 0
    if vllm_config is not None:
        extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config
        if "lookup_rpc_port" in extra_config:
            rpc_port = extra_config["lookup_rpc_port"]
        elif "mooncake_rpc_port" in extra_config:
            rpc_port = extra_config["mooncake_rpc_port"]
            logger.warning(
                "It is recommended to use the lookup_rpc_port, as the mooncake_rpc_port will be removed in the future."
            )
    logger.debug("Base URL: %s, RPC Port: %s", base_url, rpc_port)
    return f"ipc://{base_url}/lookup_rpc_port_{rpc_port}_dp_rank{dp_rank}"
