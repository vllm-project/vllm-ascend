import importlib
from typing import Any

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.logger import logger
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import Request
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend import (
    backend_map,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import (
    AscendConnectorMetadata,
    KeyMetadata,
    LoadSpec,
    PoolKey,
    ReqMeta,
    RequestTracker,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise_config import (
    get_layerwise_config,
)


class KVPoolScheduler:
    def __init__(self, vllm_config: "VllmConfig", use_layerwise, page_size_bytes: int):
        self.use_layerwise = use_layerwise
        self.kv_role = vllm_config.kv_transfer_config.kv_role
        self.consumer_is_to_load = vllm_config.kv_transfer_config.kv_connector_extra_config.get(
            "consumer_is_to_load", False
        )
        self.consumer_is_to_put = vllm_config.kv_transfer_config.kv_connector_extra_config.get(
            "consumer_is_to_put", False
        )
        self.load_async = vllm_config.kv_transfer_config.kv_connector_extra_config.get("load_async", False)
        self.save_decode_cache = vllm_config.kv_transfer_config.kv_connector_extra_config.get(
            "save_decode_cache", True)
        # request_id -> (vllm cached tokes, kvpool cached tokens)
        self.load_specs: dict[str, LoadSpec] = {}
        self.pcp_size = getattr(vllm_config.parallel_config, "prefill_context_parallel_size", 1)
        self.dcp_size = getattr(vllm_config.parallel_config, "decode_context_parallel_size", 1)

        self.original_block_size = vllm_config.cache_config.block_size
        self._block_size = vllm_config.cache_config.block_size
        if self.pcp_size > 1:
            self._block_size *= self.pcp_size
        if self.dcp_size > 1:
            self._block_size *= self.dcp_size
        # request_id -> full_token_ids
        self._request_trackers: dict[str, RequestTracker] = {}
        self._preempted_req_ids: set[str] = set()
        self._unfinished_requests: dict[str, tuple[Request, list[int]]] = {}
        self._unfinished_request_ids: set[str] = set()
        self._loading_req_ids: set[str] = set()
        self._delayed_free_req_ids: set[str] = set()

        self.page_size_bytes = page_size_bytes
        logger.info("KV pool page_size_bytes: %d", page_size_bytes)
        backend_name = vllm_config.kv_transfer_config.kv_connector_extra_config.get(
            "backend", "mooncake")
        self.backend_name = backend_name.lower()
        backend = backend_map.get(self.backend_name)
        if backend is None:
            raise ValueError(f"Unsupported KV pool backend: {backend_name}")
        backend_path = backend.get("path")
        backend_class_name = backend.get("name")
        assert backend_path is not None and backend_class_name is not None
        backend_module = importlib.import_module(backend_path)
        backend_class = getattr(backend_module, backend_class_name)
        self.store_scheduler = backend_class.create_scheduler_client(
            vllm_config.parallel_config)

        model_config = vllm_config.model_config
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.pp_size = vllm_config.parallel_config.pipeline_parallel_size
        self.pp_rank = (
            vllm_config.parallel_config.rank // self.tp_size
        ) % self.pp_size
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
        self.model_name = model_config.model.split('/')[-1]

        # Keep this in sync with pool_worker.py because it affects allocation size.
        if self.use_layerwise:
            layerwise_config = get_layerwise_config(self.num_layers)
            num_layer_keys = self.num_layers
            self.layerwise_offload = layerwise_config.has_layer_reuse
        else:
            num_layer_keys = 1
            self.layerwise_offload = False

        keys_per_block_hash = (
            self.pcp_size * self.dcp_size
            * (self.tp_size // self.put_step)
            * num_layer_keys
        )
        self.keys_per_block_hash = keys_per_block_hash
        self.prefill_offload = self.layerwise_offload
        # Whether to discard partial chunks
        self._discard_partial_chunks = (
            vllm_config.kv_transfer_config.get_from_extra_config(
                "discard_partial_chunks", not self.prefill_offload))

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
            new_gvas = self.store_scheduler.batch_alloc(
                keys_to_alloc, [alloc_size] * len(keys_to_alloc))
            if any(gva <= 0 for gva in new_gvas):
                raise ValueError(
                    f"Request {request_tracker.req_id}: batch_alloc failed, "
                    f"gvas={new_gvas}")

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
            new_gvas = self.store_scheduler.batch_alloc(
                missing_keys, [alloc_size] * len(missing_keys))
            if any(gva <= 0 for gva in new_gvas):
                raise ValueError(
                    f"Request {request_tracker.req_id}: batch_alloc failed, "
                    f"gvas={new_gvas}")
            for index, gva in zip(missing_indices, new_gvas):
                block_gvas[index] = gva

        request_tracker.block_keys = block_keys
        request_tracker.block_gvas = block_gvas
        request_tracker.gva_block_offset = 0

    def generate_keys(self, block_hashes, req_id='', has_last_block=False):
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
    ) -> list[list[str]]:
        head_or_tp_ranks = self.tp_size // self.put_step
        keys_by_block = []
        for block_hash in block_hashes:
            block_keys = []
            chunk_hash = block_hash if isinstance(block_hash, str) else block_hash.hex()
            pp_ranks = range(1) if include_layers else range(self.pp_size)
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
                                ),
                                chunk_hash,
                            )
                            if include_layers:
                                block_keys.extend(
                                    layer_key.to_string()
                                    for layer_key in pool_key.split_layers(
                                        self.num_layers)
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
        query_start_block = (
            0 if self.layerwise_offload
            else min(num_computed_tokens // self._block_size, num_blocks)
        )
        block_hashes_to_query = request.block_hashes[
            query_start_block:num_blocks]
        if not block_hashes_to_query:
            return 0

        query_keys_by_block = self._generate_store_query_keys(
            block_hashes_to_query, include_layers=include_layers)
        query_keys = [
            key
            for block_keys in query_keys_by_block
            for key in block_keys
        ]
        exists_states = self.store_scheduler.batch_is_exist(query_keys)
        if len(exists_states) != len(query_keys):
            raise RuntimeError(
                "KV pool exists check returned unexpected number of "
                f"states for request {request.request_id}: "
                f"expected={len(query_keys)}, actual={len(exists_states)}")

        num_queried_hit_blocks = 0
        offset = 0
        for block_keys in query_keys_by_block:
            block_states = exists_states[offset:offset + len(block_keys)]
            offset += len(block_keys)
            if all(exists == 1 for exists in block_states):
                num_queried_hit_blocks += 1
                continue
            if any(exists == 0 for exists in block_states):
                break
            raise RuntimeError(
                "KV pool exists check failed for request "
                f"{request.request_id}: states={exists_states}")

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
        keys_to_check = [
            f"{self.model_name}@{bh.hex()}" for bh in block_hashes_to_check
        ]
        query_start_block = (
            0 if self.layerwise_offload
            else min(num_computed_tokens // self._block_size, num_blocks)
        )
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
            token_len = len(request.prompt_token_ids) // self._block_size * self._block_size
        else:
            token_len = len(request.prompt_token_ids)

        if token_len < self._block_size:
            return 0, False

        if self.use_layerwise and self.backend_name == "memcache":
            num_external_hit_tokens = self._get_layerwise_gva_hit_tokens(
                request, token_len, num_computed_tokens)
        else:
            num_external_hit_tokens = self._get_store_lookup_hit_tokens(
                request,
                token_len,
                num_computed_tokens,
                include_layers=self.use_layerwise,
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

        # With layer reuse, HBM prefix cache only keeps independent layers
        # usable. Reused layers still need to be restored from the KV pool even
        # when vLLM reports the same number of local cached tokens.
        force_layerwise_load = (
            self.layerwise_offload
            and num_external_hit_tokens > 0
        )
        if need_to_allocate <= 0 and not force_layerwise_load:
            return 0, False

        self.load_specs[request.request_id] = LoadSpec(
            vllm_cached_tokens=num_computed_tokens,
            kvpool_cached_tokens=num_external_hit_tokens,
            can_load=force_layerwise_load,
        )

        return need_to_allocate, self.load_async and not self.use_layerwise

    def update_state_after_alloc(self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int):
        """
        Update KVConnector state after temporary buffer alloc.

        For SharedStorageConnector, update _request_needs_load
        if the CacheManager this allocated blocks for us.
        """
        local_block_ids = []
        if num_external_tokens > 0:
            local_block_ids = blocks.get_block_ids()[0]

        self._unfinished_requests[request.request_id] = (request, local_block_ids)
        self._unfinished_request_ids.add(request.request_id)
        if request.request_id not in self.load_specs:
            # No KV tokens from external KV cache, return
            return

        if num_external_tokens == 0:
            # No need to load anything
            self.load_specs[request.request_id].can_load = (
                self.layerwise_offload
                and self.load_specs[request.request_id].kvpool_cached_tokens > 0
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

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        """Attach the connector metadata to the request object.

        This function should NOT modify other fields in the scheduler_output
        except the `kv_connector_metadata` field.
        Also, calling this function will reset the state of the connector.

        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
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
            # Right now, we only load KV for new requests
            load_spec = self.load_specs.pop(request.req_id, None)
            num_tokens_to_compute = request.num_computed_tokens + scheduler_output.num_scheduled_tokens[request.req_id]
            request_tuple = self._unfinished_requests.get(request.req_id)
            if request_tuple is None:
                raise ValueError(
                    f"Request {request.req_id} is not in _unfinished_requests, "
                    "but it is scheduled as a new request"
                )
            request_real = request_tuple[0]  # type: ignore[index]
            if not isinstance(request.block_ids[0], list):
                unfolded_block_ids = request.block_ids.copy()
            else:
                unfolded_block_ids = request.block_ids[0].copy()
            previous_tracker = self._request_trackers.get(request.req_id)
            request_tracker = RequestTracker(
                req_id=request.req_id,
                token_len=num_tokens_to_compute,
                allocated_block_ids=unfolded_block_ids,
                num_saved_tokens=0,
                token_ids=request.prompt_token_ids[:num_tokens_to_compute].copy(),
                block_keys=(previous_tracker.block_keys.copy() if previous_tracker else []),
                block_gvas=(previous_tracker.block_gvas.copy() if previous_tracker else []),
                gva_block_offset=(previous_tracker.gva_block_offset if previous_tracker else 0),
            )
            self._request_trackers[request.req_id] = request_tracker
            last_chunk_tokens_num = (
                (len(request.prompt_token_ids) // self._block_size * self._block_size)
                if self._discard_partial_chunks
                else len(request.prompt_token_ids)
            )

            num_blocks = num_tokens_to_compute // self._block_size
            has_last_block = num_tokens_to_compute % self._block_size != 0

            if self.use_layerwise:
                self._ensure_tracker_gvas_cover_blocks(
                    request_tracker,
                    request_real.block_hashes[:num_blocks],
                )
                if has_last_block:
                    self._generate_keys_and_alloc(
                        [],
                        request_tracker=request_tracker,
                        has_last_block=True,
                    )

            req_meta = ReqMeta.from_request_tracker(
                request_tracker,
                self._block_size,
                load_spec=load_spec,
                skip_save=force_skip_save,
                block_hashes=request_real.block_hashes,
                is_last_chunk=request_tracker.token_len >= last_chunk_tokens_num,
                discard_partial_chunks=self._discard_partial_chunks,
                original_block_size=self.original_block_size,
            )
            if req_meta is not None:
                meta.add_request(req_meta)

        cached_reqs = scheduler_output.scheduled_cached_reqs
        if not force_skip_save:
            for i, req_id in enumerate(cached_reqs.req_ids):
                # resumed request
                new_block_ids = cached_reqs.new_block_ids[i]
                # TODO 调试的时候，添加decode，为了验证精度
                if not new_block_ids and not self.prefill_offload:
                    continue
                if req_id in self._preempted_req_ids:
                    if isinstance(new_block_ids, tuple):
                        new_block_ids = new_block_ids[0].copy()
                    else:
                        new_block_ids = new_block_ids.copy()
                    self._preempted_req_ids.discard(req_id)
                    load_spec = self.load_specs.pop(req_id, None)
                    if self.prefill_offload:
                        load_spec = LoadSpec(
                            vllm_cached_tokens=0,
                            kvpool_cached_tokens=cached_reqs.num_computed_tokens[i],
                            can_load=True,
                        )
                    request_tuple = self._unfinished_requests.get(req_id)
                    if request_tuple is None:
                        raise ValueError(
                            f"Request {req_id} is not in _unfinished_requests, "
                            "but it is scheduled as a preempted cached request"
                        )
                    request_real = request_tuple[0]  # type: ignore[index]
                    num_tokens_to_compute = (
                        request_real.num_computed_tokens + scheduler_output.num_scheduled_tokens[req_id]
                    )
                    previous_tracker = self._request_trackers.get(req_id)
                    request_tracker = RequestTracker(
                        req_id=req_id,
                        token_len=num_tokens_to_compute,
                        allocated_block_ids=new_block_ids,
                        num_saved_tokens=0,
                        token_ids=request_real.prompt_token_ids[:num_tokens_to_compute].copy(),
                        block_keys=(previous_tracker.block_keys.copy() if previous_tracker else []),
                        block_gvas=(previous_tracker.block_gvas.copy() if previous_tracker else []),
                        gva_block_offset=(previous_tracker.gva_block_offset if previous_tracker else 0),
                    )
                    self._request_trackers[req_id] = request_tracker

                    num_blocks = len(new_block_ids)
                    has_last_block = num_tokens_to_compute % self._block_size != 0
                    if self.use_layerwise:
                        self._ensure_tracker_gvas_cover_blocks(
                            request_tracker,
                            request_real.block_hashes[:num_blocks],
                        )
                        if has_last_block:
                            self._generate_keys_and_alloc(
                                [],
                                request_tracker=request_tracker,
                                has_last_block=True,
                            )

                    last_chunk_tokens_num = (
                        (len(request_real.prompt_token_ids) // self._block_size * self._block_size)
                        if self._discard_partial_chunks
                        else len(request_real.prompt_token_ids)
                    )
                    req_meta = ReqMeta.from_request_tracker(
                        request_tracker,
                        self._block_size,
                        load_spec=load_spec,
                        skip_save=force_skip_save,
                        block_hashes=request_real.block_hashes,
                        is_last_chunk=request_tracker.token_len >= last_chunk_tokens_num,
                        discard_partial_chunks=self._discard_partial_chunks,
                        original_block_size=self.original_block_size,
                    )

                # decode/chunked request
                else:
                    if not self.save_decode_cache and not self.prefill_offload:
                        continue
                    request_tracker = self._request_trackers.get(req_id)
                    if request_tracker is None:
                        raise ValueError(
                            f"Request {req_id} is not in _request_trackers, "
                            "but it is scheduled to be cached"
                        )
                    num_new_tokens = scheduler_output.num_scheduled_tokens[req_id]
                    req_tuple = self._unfinished_requests.get(req_id)
                    if req_tuple:
                        request = req_tuple[0]
                        num_current_tokens = request_tracker.token_len
                        new_token_ids = request.all_token_ids[num_current_tokens : num_current_tokens + num_new_tokens]
                        if request_tracker.token_ids is not None and new_token_ids:
                            request_tracker.token_ids.extend(new_token_ids)
                        request_tracker.token_len += num_new_tokens
                    else:
                        raise ValueError(
                            f"Request {req_id} is not in _unfinished_requests, but it is scheduled to be cached"
                        )
                    prev_token_count = request_tracker.token_len - num_new_tokens
                    prev_hash_count = prev_token_count // self._block_size
                    current_hash_count = request_tracker.token_len // self._block_size
                    new_hash_count = current_hash_count - prev_hash_count
                    has_last_block = (
                        request_tracker.token_len % self._block_size != 0
                        or current_hash_count > len(request.block_hashes)
                    )
                    if self.use_layerwise and (new_hash_count > 0 or has_last_block):
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
                        request_tracker.update(new_block_ids)
                    last_chunk_tokens_num = (
                        (len(request.prompt_token_ids) // self._block_size * self._block_size)
                        if self._discard_partial_chunks
                        else len(request.prompt_token_ids)
                    )
                    load_spec = None
                    if self.prefill_offload:
                        load_spec = LoadSpec(
                            vllm_cached_tokens=cached_reqs.num_computed_tokens[i],
                            kvpool_cached_tokens=cached_reqs.num_computed_tokens[i],
                            can_load=True,
                        )
                    req_meta = ReqMeta.from_request_tracker(
                        request_tracker,
                        self._block_size,
                        load_spec=load_spec,
                        skip_save=force_skip_save,
                        block_hashes=request.block_hashes,
                        is_last_chunk=request_tracker.token_len >= last_chunk_tokens_num,
                        discard_partial_chunks=self._discard_partial_chunks,
                        original_block_size=self.original_block_size,
                    )
                if req_meta is not None:
                    meta.add_request(req_meta)
        request_ids = [req.req_id for req in scheduler_output.scheduled_new_reqs]
        for request_id, (request, block_ids) in self._unfinished_requests.items():
            if request_id not in request_ids and request_id not in cached_reqs.req_ids:
                load_spec = self.load_specs.pop(request_id, None)
                if not load_spec:
                    continue
                num_tokens_to_compute = load_spec.kvpool_cached_tokens
                if (num_tokens_to_compute % self._block_size != 0) and (
                    num_tokens_to_compute == len(request.prompt_token_ids) - 1
                ):
                    num_tokens_to_compute = num_tokens_to_compute + 1
                previous_tracker = self._request_trackers.get(request_id)
                request_tracker = RequestTracker(
                    req_id=request_id,
                    token_len=num_tokens_to_compute,
                    allocated_block_ids=block_ids,
                    num_saved_tokens=0,
                    block_keys=(previous_tracker.block_keys.copy() if previous_tracker else []),
                    block_gvas=(previous_tracker.block_gvas.copy() if previous_tracker else []),
                    gva_block_offset=(previous_tracker.gva_block_offset if previous_tracker else 0),
                )

                self._request_trackers[request_id] = request_tracker

                num_blocks = num_tokens_to_compute // self._block_size
                has_last_block = num_tokens_to_compute % self._block_size != 0
                if self.use_layerwise:
                    self._ensure_tracker_gvas_cover_blocks(
                        request_tracker,
                        request.block_hashes[:num_blocks],
                    )
                    if has_last_block:
                        self._generate_keys_and_alloc(
                            [],
                            request_tracker=request_tracker,
                            has_last_block=True,
                        )

                req_meta = ReqMeta.from_request_tracker(
                    request_tracker,
                    self._block_size,
                    load_spec=load_spec,
                    skip_save=None,
                    block_hashes=request.block_hashes,
                    discard_partial_chunks=self._discard_partial_chunks,
                )
                if req_meta is not None:
                    meta.add_request(req_meta)
        return meta

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

    def update_finished_sending(self, finished_sending: set[str] | None) -> None:
        if finished_sending:
            self._delayed_free_req_ids.difference_update(finished_sending)

    def update_finished_recving(self, finished_recving: set[str] | None) -> None:
        if finished_recving:
            self._loading_req_ids.difference_update(finished_recving)
