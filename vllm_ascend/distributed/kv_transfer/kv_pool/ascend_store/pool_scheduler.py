import re
from typing import Any

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.logger import logger
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import Request
from memcache_hybrid import DistributedObjectStore  # type: ignore

import vllm_ascend.envs as ascend_envs
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import (
    AscendConnectorMetadata,
    LoadSpec,
    ReqMeta,
    RequestTracker,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.key_lru_cache import (
    KeyLRUCache,
)


def _parse_dram_size(value: str) -> int:
    if not value or value == "0":
        return 0
    try:
        return int(value)
    except ValueError:
        pass
    cleaned = value.strip().lower()
    unit_multipliers = {"gb": 1024**3, "mb": 1024**2, "kb": 1024, "b": 1}
    match = re.match(r"^\s*([\d.]+)\s*(gb|mb|kb|b)?\s*$", cleaned)
    if not match:
        raise ValueError(f"Invalid DRAM size format: '{value}'")
    number = float(match.group(1))
    unit = match.group(2) or "b"
    return int(number * unit_multipliers[unit])


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
        self.prefill_offload = True
        # Whether to discard partial chunks
        self._discard_partial_chunks = (
            vllm_config.kv_transfer_config.get_from_extra_config(
                "discard_partial_chunks", not self.prefill_offload))
        self._unfinished_requests: dict[str, tuple[Request, list[int]]] = {}
        self._unfinished_request_ids: set[str] = set()

        self.page_size_bytes = page_size_bytes
        logger.info(f"==============> page_size_bytes {page_size_bytes}")
        self.store_scheduler = DistributedObjectStore()
        self.store_scheduler.init(device_id=0, init_bm=False)

        model_config = vllm_config.model_config
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.pp_size = vllm_config.parallel_config.pipeline_parallel_size
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

        dram_size_str = ascend_envs.VLLM_ASCEND_KV_POOL_DRAM_SIZE
        dram_size_bytes = _parse_dram_size(dram_size_str)
        keys_per_block_hash = (
            self.pcp_size * self.dcp_size
            * (self.tp_size // self.put_step)
            * self.num_layers
        )
        memory_per_block_hash = keys_per_block_hash * self.page_size_bytes
        lru_capacity = dram_size_bytes // memory_per_block_hash
        logger.info(
            "KV pool LRU capacity calculated from DRAM size %s: %d "
            "(keys_per_block_hash=%d, memory_per_block_hash=%d)",
            dram_size_str, lru_capacity,
            keys_per_block_hash, memory_per_block_hash,
        )
        self.key_lru_cache = KeyLRUCache(lru_capacity, self.store_scheduler)
        self._req_last_block_gvas: dict[str, dict[str, int | None]] = {}

    def _generate_keys_and_alloc(self, block_hashes, req_id='', has_last_block=False) -> tuple[list[list[str]], list[list[str]], dict[str, int | None]]:
        block_keys_by_layer, last_block_keys_by_layer, block_hash_groups = self.generate_keys(block_hashes, req_id=req_id, has_last_block=has_last_block)
        need_alloc_last_block = has_last_block and req_id not in self._req_last_block_gvas

        if need_alloc_last_block:
            all_keys = [key for layer_keys in block_keys_by_layer for key in layer_keys]
            all_keys.extend([key for layer_keys in last_block_keys_by_layer for key in layer_keys])
            gvas = self.key_lru_cache.batch_get_and_alloc(
                all_keys, self.page_size_bytes, block_hash_groups)
            key_gva_mapping: dict[str, Any] = dict(zip(all_keys, gvas))
            last_block_keys_flat = [key for layer_keys in last_block_keys_by_layer for key in layer_keys]
            self._req_last_block_gvas[req_id] = {
                k: key_gva_mapping[k] for k in last_block_keys_flat if k in key_gva_mapping
            }
        else:
            chunk_keys = [key for layer_keys in block_keys_by_layer for key in layer_keys]
            chunk_block_hash_groups = {
                k: v for k, v in block_hash_groups.items()
                if not k.endswith(b'_lastblock')
            }
            gvas = self.key_lru_cache.batch_get_and_alloc(
                chunk_keys, self.page_size_bytes,
                chunk_block_hash_groups if chunk_block_hash_groups else None)
            # TODO here should verify the gvas is not None
            # TODO if gvas is None, should release the keys and retry
            key_gva_mapping: dict[str, Any] = dict(zip(chunk_keys, gvas))
            if has_last_block and req_id in self._req_last_block_gvas:
                key_gva_mapping.update(self._req_last_block_gvas[req_id])

        return block_keys_by_layer, last_block_keys_by_layer, key_gva_mapping

    def generate_keys(self, chunk_hashes, req_id='', has_last_block=False):
        block_hash_groups: dict[bytes, list[str]] = {}

        def _build_layer_keys(layer_id: int) -> tuple[list[str], list[str]]:
            chunk_keys = []
            for chunk_hash in chunk_hashes:
                if chunk_hash not in block_hash_groups:
                    block_hash_groups[chunk_hash] = []
                keys_for_hash = [
                    f"{self.model_name}@pcp{pcp_rank}@dcp{dcp_rank}"
                    f"@head_or_tp_rank:{head_or_tp_rank}@{chunk_hash.hex()}@{layer_id}"
                    for pcp_rank in range(self.pcp_size)
                    for dcp_rank in range(self.dcp_size)
                    for head_or_tp_rank in range(self.tp_size // self.put_step)
                ]
                chunk_keys.extend(keys_for_hash)
                block_hash_groups[chunk_hash].extend(keys_for_hash)

            last_block_keys = []
            if has_last_block:
                last_block_hash = f"{req_id}_lastblock".encode()
                if last_block_hash not in block_hash_groups:
                    block_hash_groups[last_block_hash] = []
                last_block_keys = [
                    f"{self.model_name}@pcp{pcp_rank}@dcp{dcp_rank}"
                    f"@head_or_tp_rank:{head_or_tp_rank}@{req_id}_lastblock@{layer_id}"
                    for pcp_rank in range(self.pcp_size)
                    for dcp_rank in range(self.dcp_size)
                    for head_or_tp_rank in range(self.tp_size // self.put_step)
                ]
                block_hash_groups[last_block_hash].extend(last_block_keys)
            return chunk_keys, last_block_keys
        results = [_build_layer_keys(layer_id) for layer_id in range(self.num_layers)]
        block_keys_by_layer = [r[0] for r in results]
        last_block_keys_by_layer = [r[1] for r in results]
        return block_keys_by_layer, last_block_keys_by_layer, block_hash_groups

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

        num_blocks = token_len // self._block_size
        num_hit_blocks = 0
        for bh in request.block_hashes[:num_blocks]:
            if self.key_lru_cache.has_block(bh):
                num_hit_blocks += 1
            else:
                break
        num_external_hit_tokens = num_hit_blocks * self._block_size

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

        if need_to_allocate <= 0:
            return 0, False

        self.load_specs[request.request_id] = LoadSpec(
            vllm_cached_tokens=num_computed_tokens,
            kvpool_cached_tokens=num_external_hit_tokens,
            can_load=False,
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
            self.load_specs[request.request_id].can_load = False
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
            self._req_last_block_gvas.pop(finished_req_id, None)

        for req_id in scheduler_output.preempted_req_ids:
            self._preempted_req_ids.update(scheduler_output.preempted_req_ids)
            self._request_trackers.pop(req_id, None)
            self._unfinished_requests.pop(req_id, None)

        meta = AscendConnectorMetadata(self._unfinished_request_ids, scheduler_output.preempted_req_ids)

        for request in scheduler_output.scheduled_new_reqs:
            # Right now, we only load KV for new requests
            load_spec = self.load_specs.pop(request.req_id, None)
            num_tokens_to_compute = request.num_computed_tokens + scheduler_output.num_scheduled_tokens[request.req_id]
            request_tuple = self._unfinished_requests.get(request.req_id)
            request_real = request_tuple[0]  # type: ignore[index]
            if not isinstance(request.block_ids[0], list):
                unfolded_block_ids = request.block_ids.copy()
            else:
                unfolded_block_ids = request.block_ids[0].copy()
            request_tracker = RequestTracker(
                req_id=request.req_id,
                token_len=num_tokens_to_compute,
                allocated_block_ids=unfolded_block_ids,
                num_saved_tokens=0,
                token_ids=request.prompt_token_ids[:num_tokens_to_compute].copy(),
            )
            self._request_trackers[request.req_id] = request_tracker
            last_chunk_tokens_num = (
                (len(request.prompt_token_ids) // self._block_size * self._block_size)
                if self._discard_partial_chunks
                else len(request.prompt_token_ids)
            )

            num_blocks = len(unfolded_block_ids)
            has_last_block = num_tokens_to_compute % self._block_size != 0

            block_keys_by_layer, last_block_keys_by_layer, key_gva_mapping = self._generate_keys_and_alloc(
                request_real.block_hashes[:num_blocks], req_id=request.req_id, has_last_block=has_last_block)
            request_tracker.key_gva_mapping = key_gva_mapping
            request_tracker.block_keys_by_layer = block_keys_by_layer
            request_tracker.last_block_keys_by_layer = last_block_keys_by_layer

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
                req_meta.key_gva_mapping = key_gva_mapping
                req_meta.block_keys_by_layer = block_keys_by_layer
                req_meta.last_block_keys_by_layer = last_block_keys_by_layer
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
                    request_real = request_tuple[0]  # type: ignore[index]
                    num_tokens_to_compute = (
                        request_real.num_computed_tokens + scheduler_output.num_scheduled_tokens[req_id]
                    )
                    request_tracker = RequestTracker(
                        req_id=req_id,
                        token_len=num_tokens_to_compute,
                        allocated_block_ids=new_block_ids,
                        num_saved_tokens=0,
                        token_ids=request_real.prompt_token_ids[:num_tokens_to_compute].copy(),
                    )
                    self._request_trackers[req_id] = request_tracker

                    num_blocks = len(new_block_ids)
                    has_last_block = num_tokens_to_compute % self._block_size != 0
                    block_keys_by_layer, last_block_keys_by_layer, key_gva_mapping = self._generate_keys_and_alloc(
                        request_real.block_hashes[:num_blocks], req_id=req_id, has_last_block=has_last_block)

                    request_tracker.block_keys_by_layer = block_keys_by_layer
                    request_tracker.last_block_keys_by_layer = last_block_keys_by_layer

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
                    if req_meta is not None:
                        req_meta.key_gva_mapping = key_gva_mapping
                        req_meta.block_keys_by_layer = block_keys_by_layer
                        req_meta.last_block_keys_by_layer = last_block_keys_by_layer

                # decode/chunked request
                else:
                    request_tracker = self._request_trackers[req_id]
                    num_new_tokens = scheduler_output.num_scheduled_tokens[req_id]
                    req_tuple = self._unfinished_requests.get(req_id)
                    if req_tuple:
                        request = req_tuple[0]
                        num_current_tokens = request_tracker.token_len
                        new_token_ids = request.all_token_ids[num_current_tokens : num_current_tokens + num_new_tokens]
                        request_tracker.token_len += len(new_token_ids)
                    else:
                        raise ValueError(
                            f"Request {req_id} is not in _unfinished_requests, but it is scheduled to be cached"
                        )
                    prev_token_count = request_tracker.token_len - num_new_tokens
                    prev_hash_count = prev_token_count // self._block_size
                    current_hash_count = request_tracker.token_len // self._block_size
                    new_hash_count = current_hash_count - prev_hash_count
                    if new_hash_count > 0:
                        new_block_hashes = request.block_hashes[prev_hash_count : current_hash_count]
                        block_keys_by_layer, _, key_gva_mapping = self._generate_keys_and_alloc(new_block_hashes)
                        request_tracker.key_gva_mapping.update(key_gva_mapping)

                        if request_tracker.block_keys_by_layer is None:
                            request_tracker.block_keys_by_layer = block_keys_by_layer
                        else:
                            for layer_id, layer_keys in enumerate(block_keys_by_layer):
                                request_tracker.block_keys_by_layer[layer_id].extend(layer_keys)

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
                            vllm_cached_tokens=0,
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
                    req_meta.key_gva_mapping = request_tracker.key_gva_mapping
                    req_meta.block_keys_by_layer = request_tracker.block_keys_by_layer
                    req_meta.last_block_keys_by_layer = request_tracker.last_block_keys_by_layer
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
                request_tracker = RequestTracker(
                    req_id=request_id,
                    token_len=num_tokens_to_compute,
                    allocated_block_ids=block_ids,
                    num_saved_tokens=0,
                )

                self._request_trackers[request_id] = request_tracker

                num_blocks = num_tokens_to_compute // self._block_size
                has_last_block = num_tokens_to_compute % self._block_size != 0
                block_hashes_for_keys = request.block_hashes[:num_blocks]
                block_keys_by_layer, last_block_keys_by_layer, key_gva_mapping = self._generate_keys_and_alloc(
                    block_hashes_for_keys, req_id=request_id, has_last_block=has_last_block)
                request_tracker.key_gva_mapping = key_gva_mapping
                request_tracker.block_keys_by_layer = block_keys_by_layer
                request_tracker.last_block_keys_by_layer = last_block_keys_by_layer

                req_meta = ReqMeta.from_request_tracker(
                    request_tracker,
                    self._block_size,
                    load_spec=load_spec,
                    skip_save=None,
                    block_hashes=request.block_hashes,
                    discard_partial_chunks=self._discard_partial_chunks,
                )
                if req_meta is not None:
                    req_meta.key_gva_mapping = key_gva_mapping
                    req_meta.block_keys_by_layer = block_keys_by_layer
                    req_meta.last_block_keys_by_layer = last_block_keys_by_layer
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
            return False, None
        tracker = self._request_trackers.get(request.request_id)
        if tracker is not None and tracker.num_saved_tokens <= 0:
            return False, None
        delay_free_blocks = len(block_ids) > 0
        # if self.key_lru_cache is not None and request.block_hashes:
        #     removed_keys = self.key_lru_cache.remove_blocks(
        #         request.block_hashes[:len(block_ids)])
        #     if removed_keys and self.store_scheduler is not None:
        #         self.store_scheduler.remove_batch(removed_keys)
        if delay_free_blocks:
            logger.debug("Delaying free of %d blocks for request %s", len(block_ids), request.request_id)
        return delay_free_blocks, None


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
