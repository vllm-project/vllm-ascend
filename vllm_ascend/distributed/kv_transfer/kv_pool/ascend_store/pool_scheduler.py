import math
from typing import Any, cast

import vllm.envs as envs
import zmq
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.logger import logger
from vllm.utils.math_utils import cdiv
from vllm.utils.network_utils import make_zmq_socket
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
from vllm.v1.request import Request
from vllm.v1.serial_utils import MsgpackEncoder

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import (
    AscendConnectorMetadata,
    LoadSpec,
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
    ):
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
        self.client = LookupKeyClient(vllm_config)
        # request_id -> (vllm cached tokes, kvpool cached tokens)
        self.load_specs: dict[str, LoadSpec] = {}
        self.pcp_size = getattr(vllm_config.parallel_config, "prefill_context_parallel_size", 1)
        self.dcp_size = getattr(vllm_config.parallel_config, "decode_context_parallel_size", 1)

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
        self._unfinished_requests: dict[str, tuple[Request, list[list[int]]]] = {}
        self._unfinished_request_ids: set[str] = set()

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

        num_external_hit_tokens = self.client.lookup(
            token_len,
            request.block_hashes,
            self.kv_cache_group_ids,
        )

        if num_external_hit_tokens == request.num_tokens:
            num_external_hit_tokens -= 1

        if num_external_hit_tokens < num_computed_tokens:
            need_to_allocate = 0
        else:
            need_to_allocate = num_external_hit_tokens - num_computed_tokens

        logger.debug(
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
            self.load_specs[request.request_id].can_load = False
            logger.debug(
                "KV pool load spec disabled req=%s because num_external_tokens=0 vllm_cached=%d kvpool_cached=%d",
                request.request_id,
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
        logger.debug(
            "KV pool load spec enabled req=%s num_external_tokens=%d vllm_cached=%d kvpool_cached=%d groups=%s",
            request.request_id,
            num_external_tokens,
            self.load_specs[request.request_id].vllm_cached_tokens,
            self.load_specs[request.request_id].kvpool_cached_tokens,
            [len(blocks) for blocks in local_block_ids],
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

        for req_id in scheduler_output.preempted_req_ids:
            self._preempted_req_ids.update(scheduler_output.preempted_req_ids)
            self._request_trackers.pop(req_id, None)
            self._unfinished_requests.pop(req_id, None)

        meta = AscendConnectorMetadata(self._unfinished_request_ids, scheduler_output.preempted_req_ids)

        for request in scheduler_output.scheduled_new_reqs:
            # Right now, we only load KV for new requests
            load_spec = self.load_specs.pop(request.req_id, None)
            if load_spec is not None:
                logger.debug(
                    "KV pool build meta attaches load spec req=%s can_load=%s "
                    "vllm_cached=%d kvpool_cached=%d scheduled_tokens=%d "
                    "num_computed=%d",
                    request.req_id,
                    load_spec.can_load,
                    load_spec.vllm_cached_tokens,
                    load_spec.kvpool_cached_tokens,
                    scheduler_output.num_scheduled_tokens[request.req_id],
                    request.num_computed_tokens,
                )
            num_tokens_to_compute = request.num_computed_tokens + scheduler_output.num_scheduled_tokens[request.req_id]
            request_tuple = self._unfinished_requests.get(request.req_id)
            request_real = request_tuple[0]  # type: ignore[index]
            request_tracker = RequestTracker(
                req_id=request.req_id,
                token_len=num_tokens_to_compute,
                allocated_block_ids_by_group=normalize_block_ids_by_group(request.block_ids),
                num_saved_tokens=0,
                token_ids=request.prompt_token_ids[:num_tokens_to_compute].copy(),
            )
            self._request_trackers[request.req_id] = request_tracker
            last_chunk_tokens_num = (
                self._floor_to_cache_transfer_granularity(len(request.prompt_token_ids))
                if self._discard_partial_chunks
                else len(request.prompt_token_ids)
            )

            req_meta = ReqMeta.from_request_tracker(
                request_tracker,
                self.cache_transfer_granularity,
                load_spec=load_spec,
                skip_save=force_skip_save,
                block_hashes=request_real.block_hashes,
                is_last_chunk=request_tracker.token_len >= last_chunk_tokens_num,
                discard_partial_chunks=self._discard_partial_chunks,
                original_block_size=self.original_block_size,
                kv_cache_group_families=self.kv_cache_group_families,
            )
            if req_meta is not None:
                meta.add_request(req_meta)

        cached_reqs = scheduler_output.scheduled_cached_reqs
        if not force_skip_save:
            for i, req_id in enumerate(cached_reqs.req_ids):
                # resumed request
                new_block_ids = cached_reqs.new_block_ids[i]
                if not new_block_ids:
                    continue
                if req_id in self._preempted_req_ids:
                    self._preempted_req_ids.discard(req_id)
                    load_spec = self.load_specs.pop(req_id, None)
                    request_tuple = self._unfinished_requests.get(req_id)
                    request_real = request_tuple[0]  # type: ignore[index]
                    num_tokens_to_compute = (
                        request_real.num_computed_tokens + scheduler_output.num_scheduled_tokens[req_id]
                    )
                    request_tracker = RequestTracker(
                        req_id=req_id,
                        token_len=num_tokens_to_compute,
                        allocated_block_ids_by_group=normalize_block_ids_by_group(new_block_ids),
                        num_saved_tokens=0,
                        token_ids=request_real.prompt_token_ids[:num_tokens_to_compute].copy(),
                    )
                    self._request_trackers[req_id] = request_tracker
                    last_chunk_tokens_num = (
                        self._floor_to_cache_transfer_granularity(len(request_real.prompt_token_ids))
                        if self._discard_partial_chunks
                        else len(request_real.prompt_token_ids)
                    )
                    req_meta = ReqMeta.from_request_tracker(
                        request_tracker,
                        self.cache_transfer_granularity,
                        load_spec=load_spec,
                        skip_save=force_skip_save,
                        block_hashes=request_real.block_hashes,
                        is_last_chunk=request_tracker.token_len >= last_chunk_tokens_num,
                        discard_partial_chunks=self._discard_partial_chunks,
                        original_block_size=self.original_block_size,
                        kv_cache_group_families=self.kv_cache_group_families,
                    )

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
                    num_computed_token = cached_reqs.num_computed_tokens[i]
                    if num_computed_token >= len(request.prompt_token_ids):
                        continue
                    request_tracker.update(new_block_ids)

                    last_chunk_tokens_num = (
                        self._floor_to_cache_transfer_granularity(len(request.prompt_token_ids))
                        if self._discard_partial_chunks
                        else len(request.prompt_token_ids)
                    )
                    req_meta = ReqMeta.from_request_tracker(
                        request_tracker,
                        self.cache_transfer_granularity,
                        load_spec=None,
                        skip_save=force_skip_save,
                        block_hashes=request.block_hashes,
                        is_last_chunk=request_tracker.token_len >= last_chunk_tokens_num,
                        discard_partial_chunks=self._discard_partial_chunks,
                        original_block_size=self.original_block_size,
                        kv_cache_group_families=self.kv_cache_group_families,
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
                if (num_tokens_to_compute % self.cache_transfer_granularity != 0) and (
                    num_tokens_to_compute == len(request.prompt_token_ids) - 1
                ):
                    num_tokens_to_compute = num_tokens_to_compute + 1
                request_tracker = RequestTracker(
                    req_id=request_id,
                    token_len=num_tokens_to_compute,
                    allocated_block_ids_by_group=block_ids,
                    num_saved_tokens=0,
                )

                self._request_trackers[request_id] = request_tracker
                req_meta = ReqMeta.from_request_tracker(
                    request_tracker,
                    self.cache_transfer_granularity,
                    load_spec=load_spec,
                    skip_save=None,
                    block_hashes=request.block_hashes,
                    discard_partial_chunks=self._discard_partial_chunks,
                    kv_cache_group_families=self.kv_cache_group_families,
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
            return False, None
        tracker = self._request_trackers.get(request.request_id)
        if tracker is not None and tracker.num_saved_tokens <= 0:
            return False, None
        delay_free_blocks = len(block_ids) > 0
        if delay_free_blocks:
            if logger.isEnabledFor(10):
                logger.debug("Delaying free of %d blocks for request %s", len(block_ids), request.request_id)
        return delay_free_blocks, None

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        """HMA path for hybrid KV cache groups."""
        if self.kv_role == "kv_consumer" and not self.consumer_is_to_put:
            return False, None
        tracker = self._request_trackers.get(request.request_id)
        if tracker is not None and tracker.num_saved_tokens <= 0:
            return False, None
        block_ids = cast(tuple[list[int], ...], self.get_sw_clipped_blocks(block_ids))
        valid_group_block_ids = [group_block_ids for group_block_ids in block_ids if group_block_ids]
        delay_free_blocks = bool(valid_group_block_ids)
        if delay_free_blocks:
            logger.debug(
                "Delaying free of %d KV cache groups for request %s",
                len(valid_group_block_ids),
                request.request_id,
            )
        return delay_free_blocks, None


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
