import math
from abc import ABC, abstractmethod
from typing import Any

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.logger import logger
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    MambaSpec,
    RSWASpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
    get_mamba_prefill_checkpoint_position,
)
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend import (
    resolve_backend,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.coordinator import AscendStoreCoordinator
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.hisparse.layout import has_hisparse_cache_layout
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.in_flight_save_blocks import InFlightSaveBlockTracker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.lookup_key import LookupKeyClient
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    AscendConnectorMetadata,
    AscendStoreKVConnectorWorkerMetadata,
    BoundaryPut,
    LoadSpec,
    ReqMeta,
    RequestTracker,
    StoreLookupResult,
    infer_cache_transfer_granularity,
    infer_cacheable_group_ids,
    infer_group_block_sizes,
    infer_group_cache_families,
    infer_tp_mismatch_info,
    normalize_block_ids_by_group,
    uses_hybrid_kv_cache,
)


def _new_req_prefill_tokens(request: NewRequestData) -> list[int]:
    """Return the complete range re-prefilled by a scheduled-new request."""
    if request.prefill_token_ids is not None:
        return request.prefill_token_ids
    assert request.prompt_token_ids is not None
    return request.prompt_token_ids


class BaseKVPoolScheduler(ABC):
    """Shared request tracking, metadata, and cache lifecycle for pool schedulers."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        use_layerwise: bool,
        kv_cache_config: KVCacheConfig | None = None,
    ):
        self.vllm_config = vllm_config
        self.use_layerwise = use_layerwise
        self.kv_cache_config = kv_cache_config
        extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config
        self.backend_name, backend_class = resolve_backend(extra_config)
        self.layerwise_offload: bool = False
        hf_text_config = getattr(vllm_config.model_config, "hf_text_config", None)
        hf_config = getattr(vllm_config.model_config, "hf_config", hf_text_config)
        self.hf_config = hf_text_config or hf_config
        self.compress_ratios = getattr(hf_text_config, "compress_ratios", None)
        if self.compress_ratios is None:
            self.compress_ratios = getattr(hf_config, "compress_ratios", None)
        kv_cache_groups = kv_cache_config.kv_cache_groups if kv_cache_config is not None else None
        self.use_hybrid = self._use_hybrid(vllm_config, kv_cache_groups)
        self.kv_cache_group_ids = (
            list(range(len(kv_cache_config.kv_cache_groups)))
            if kv_cache_config is not None and self.use_hybrid
            else [0]
        )
        self.kv_cache_group_families = infer_group_cache_families(kv_cache_groups, self.compress_ratios, self.hf_config)
        self.num_mamba_speculative_blocks_by_group = self._infer_mamba_groups()
        self.kv_role = vllm_config.kv_transfer_config.kv_role
        self.enable_lookup = extra_config.get("enable_lookup", self.kv_role != "kv_consumer")
        self.load_async = vllm_config.kv_transfer_config.kv_connector_extra_config.get("load_async", False)
        kv_event_config = vllm_config.kv_events_config
        self.enable_kv_events = bool(kv_event_config and kv_event_config.enable_kv_cache_events)
        self.retention_interval = vllm_config.cache_config.prefix_cache_retention_interval
        self.save_decode_cache = vllm_config.kv_transfer_config.kv_connector_extra_config.get(
            "save_decode_cache", False
        )
        self.can_put = self.kv_role in ("kv_producer", "kv_both") or self.save_decode_cache
        # request_id -> (vllm cached tokes, kvpool cached tokens)
        self.load_specs: dict[str, LoadSpec] = {}
        self.pcp_size = getattr(vllm_config.parallel_config, "prefill_context_parallel_size", 1)
        self.dcp_size = getattr(vllm_config.parallel_config, "decode_context_parallel_size", 1)

        speculative_config = getattr(vllm_config, "speculative_config", None)
        use_eagle_fn = getattr(speculative_config, "use_eagle", None)
        self.use_eagle = use_eagle_fn() is True if callable(use_eagle_fn) else False
        use_eagle_block_drop_fn = getattr(speculative_config, "use_eagle_block_drop", None)
        self.use_eagle_block_drop = use_eagle_block_drop_fn() is True if callable(use_eagle_block_drop_fn) else False
        self.original_block_size = infer_group_block_sizes(vllm_config.cache_config.block_size, kv_cache_groups)
        self.cacheable_group_ids = infer_cacheable_group_ids(kv_cache_groups)
        self.swa_group_ids = {
            group_id
            for group_id in self.cacheable_group_ids
            if kv_cache_groups is not None
            and (
                isinstance(kv_cache_groups[group_id].kv_cache_spec, (SlidingWindowSpec, RSWASpec))
                or (
                    isinstance(kv_cache_groups[group_id].kv_cache_spec, UniformTypeKVCacheSpecs)
                    and any(
                        isinstance(spec, (SlidingWindowSpec, RSWASpec))
                        for spec in kv_cache_groups[group_id].kv_cache_spec.kv_cache_specs.values()
                    )
                )
            )
        }
        self.grouped_block_size = [block_size * self.dcp_size for block_size in self.original_block_size]
        assert kv_cache_config is not None, "AscendStore requires Core's resolved KV cache configuration"
        self.hash_block_size = kv_cache_config.hash_block_size
        for group_id in self.cacheable_group_ids:
            group_block_size = self.grouped_block_size[group_id]
            assert group_block_size % self.hash_block_size == 0, "block_size must be divisible by hash_block_size"
        self._block_size = self.grouped_block_size[0]
        self.lcm_block_size = math.lcm(*(self.grouped_block_size[i] for i in self.cacheable_group_ids))
        # Layerwise and HiSparse external transfers use complete pages.
        self.enable_partial_hash_hits = (
            not use_layerwise
            and not has_hisparse_cache_layout(kv_cache_config)
            and kv_cache_config.cache_hit_alignment_tokens < self.lcm_block_size
        )
        self.cache_transfer_granularity = infer_cache_transfer_granularity(
            self.grouped_block_size, self.lcm_block_size, self.cacheable_group_ids
        )
        self.cache_coordinator = self._build_cache_coordinator()
        # request_id -> full_token_ids
        self._request_trackers: dict[str, RequestTracker] = {}
        self._finished_partial_tail_metas: dict[str, ReqMeta] = {}
        self._preempted_req_ids: set[str] = set()
        # Whether to discard partial chunks
        self._discard_partial_chunks = vllm_config.kv_transfer_config.get_from_extra_config(
            "discard_partial_chunks", True
        )
        self._unfinished_requests: dict[str, tuple[Request, list[list[int]]]] = {}
        self._async_loading_req_ids: set[str] = set()

        self._block_pool: BlockPool | None = None
        self._kv_cache_manager: KVCacheManager | None = None

        use_mla = getattr(vllm_config.model_config, "use_mla", False)
        tp_mismatch_info = infer_tp_mismatch_info(
            self.kv_role,
            vllm_config.kv_transfer_config.kv_connector_extra_config,
            getattr(vllm_config.parallel_config, "tensor_parallel_size", 1),
            vllm_config.model_config.get_total_num_kv_heads(),
            use_mla if isinstance(use_mla, bool) else False,
            use_hybrid=self.use_hybrid,
        )
        self.tp_mismatch = tp_mismatch_info.enabled

        if backend_class is None:
            raise ValueError(f"Unsupported KV pool backend: {self.backend_name}")
        self.store_scheduler = backend_class.create_scheduler_client(vllm_config.parallel_config)

        model_config = vllm_config.model_config
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.pp_size = vllm_config.parallel_config.pipeline_parallel_size
        self.pp_rank = (vllm_config.parallel_config.rank // self.tp_size) % self.pp_size
        # Global layer offset for layerwise pool keys under PP (matches the
        # pool worker's pp_layer_offset).
        self.pp_layer_offset = 0
        try:
            start, _ = vllm_config.model_config.get_layers_start_end_indices(vllm_config.parallel_config)
            self.pp_layer_offset = start
        except AttributeError:
            self.pp_layer_offset = 0
        except Exception:
            self.pp_layer_offset = 0
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

        self.client: LookupKeyClient | None = None

    def _use_hybrid(self, vllm_config: VllmConfig, kv_cache_groups: list[Any] | None) -> bool:
        return uses_hybrid_kv_cache(vllm_config.scheduler_config, kv_cache_groups)

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

    def _build_cache_coordinator(self) -> AscendStoreCoordinator | None:
        """Build the hybrid cache-hit/mask coordinator (mirrors the worker)."""
        if self.kv_cache_config is None or not self.use_hybrid:
            return None
        return AscendStoreCoordinator(
            self.kv_cache_config.kv_cache_groups,
            scheduler_block_size=self.cache_transfer_granularity,
            hash_block_size=self.hash_block_size,
            group_block_sizes=self.grouped_block_size,
            group_cache_families=self.kv_cache_group_families,
            use_eagle=self.use_eagle,
            retention_interval=self.retention_interval,
            enable_partial_hash_hits=self.enable_partial_hash_hits,
        )

    def _floor_to_cache_transfer_granularity(self, token_len: int) -> int:
        return token_len // self.cache_transfer_granularity * self.cache_transfer_granularity

    def _infer_mamba_groups(self) -> dict[int, int]:
        """Validate Mamba cache modes and return hybrid group scratch counts."""
        if self.kv_cache_config is None:
            return {}

        num_mamba_speculative_blocks_by_group: dict[int, int] = {}
        for group_id, kv_cache_group in enumerate(self.kv_cache_config.kv_cache_groups):
            kv_cache_spec = kv_cache_group.kv_cache_spec
            if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
                kv_cache_spec = next(iter(kv_cache_spec.kv_cache_specs.values()))
            if not isinstance(kv_cache_spec, MambaSpec):
                continue
            if getattr(kv_cache_spec, "mamba_cache_mode", None) != "align":
                raise NotImplementedError(
                    "AscendStore hybrid linear-attention support currently requires mamba_cache_mode='align'."
                )
            if self.use_hybrid:
                num_mamba_speculative_blocks_by_group[group_id] = kv_cache_spec.num_speculative_blocks
        return num_mamba_speculative_blocks_by_group

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
        # A waiting request can be looked up again before it is admitted.
        # Never retain a load plan from an earlier hit across a miss/early return.
        self.load_specs.pop(request.request_id, None)
        if not self.enable_lookup:
            return 0, False

        # Core passes the candidate load start and owns local-tail arbitration:
        # it drops a local sub-block tail only if the remote hit strictly
        # extends it.

        lookup_result = self._lookup_external_hit_tokens(request, num_computed_tokens)
        if lookup_result is None or not lookup_result.hit_length:
            return 0, False
        num_external_hit_tokens = lookup_result.hit_length

        store_skip_tokens = num_external_hit_tokens
        if self.use_eagle and not self.enable_partial_hash_hits:
            # Keep the draft model's recomputation zone intact: the
            # generation-point hidden states must be freshly computed, and the
            # local prefix-cache path already drops its trailing block
            # (drop_eagle_block). Only trim the external hit when it reaches
            # into the prompt's final granularity block, so that (local +
            # external) never covers the last block whose KV the engine will
            # rewrite during MTP draft/verify steps. Partial hits that stop on
            # an interior block boundary carry a valid mamba state snapshot
            # at that boundary and can be loaded as-is.
            # The final granularity block starts at the lcm-aligned boundary
            # containing the last token; num_tokens - lcm_block_size equals
            # that boundary only for lcm-aligned prompts and over-trims
            # unaligned prompts whose hit stops exactly on the boundary.
            final_block_start = (request.num_tokens - 1) // self.lcm_block_size * self.lcm_block_size
            hit_reaches_final_block = num_external_hit_tokens > final_block_start
            if hit_reaches_final_block:
                num_external_hit_tokens = max(
                    num_computed_tokens,
                    num_external_hit_tokens - self.lcm_block_size,
                )
        if num_external_hit_tokens == request.num_tokens:
            num_external_hit_tokens -= 1
        if len(self.cacheable_group_ids) != len(self.original_block_size):
            # Private compressor state is rebuilt locally. Recompute the last
            # whole token page rather than resume inside a compressed block.
            num_external_hit_tokens = self._floor_to_cache_transfer_granularity(num_external_hit_tokens)

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

        process_hit_without_alloc = self._process_hit_without_alloc(store_skip_tokens)
        if need_to_allocate <= 0 and not process_hit_without_alloc:
            return 0, False

        self.load_specs[request.request_id] = LoadSpec(
            vllm_cached_tokens=num_computed_tokens,
            kvpool_cached_tokens=num_external_hit_tokens,
            can_load=process_hit_without_alloc,
            kvpool_store_skip_tokens=store_skip_tokens,
            tail_key_boundaries=lookup_result.tail_key_boundaries,
        )
        logger.debug(
            "KV pool load spec created req=%s vllm_cached=%d kvpool_cached=%d "
            "need_to_allocate=%d load_async=%s use_layerwise=%s",
            request.request_id,
            num_computed_tokens,
            num_external_hit_tokens,
            need_to_allocate,
            self.load_async,
            self.use_layerwise,
        )

        return need_to_allocate, self.load_async

    def _process_hit_without_alloc(self, store_skip_tokens: int) -> bool:
        """Whether to process an external hit without allocating new blocks."""
        return False

    def get_running_target_token_len(self, request: Request, num_new_tokens: int) -> int:
        """Choose the token length to record for a running request."""
        return min(request.num_computed_tokens + num_new_tokens, len(request.all_token_ids))

    def _running_load_spec(self, request_tracker: RequestTracker, num_current_tokens: int) -> LoadSpec | None:
        """Describe a load required by the current running step."""
        return None

    @abstractmethod
    def _lookup_external_hit_tokens(self, request: "Request", num_computed_tokens: int) -> StoreLookupResult | None:
        """Return the external cache hit using the selected transfer mode."""

    def register_finished_partial_tail(
        self,
        request: Request,
        block_ids: tuple[list[int], ...],
        partial_tail_offloads: list[tuple[int, int, int]],
    ) -> bool:
        """Decline finish-time tails unless the transfer mode supports them."""
        return False

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
        if request.request_id not in self.load_specs:
            # No KV tokens from external KV cache, return
            logger.debug(
                "KV pool update_state_after_alloc req=%s has no load spec; num_external_tokens=%d",
                request.request_id,
                num_external_tokens,
            )
            return

        if num_external_tokens == 0:
            # Preserve the hit policy chosen before allocation. Layerwise may
            # still need the hit metadata even when no new blocks were allocated.
            load_spec = self.load_specs[request.request_id]
            logger.debug(
                "KV pool load spec retained req=%s because num_external_tokens=0 "
                "can_load=%s vllm_cached=%d kvpool_cached=%d",
                request.request_id,
                load_spec.can_load,
                load_spec.vllm_cached_tokens,
                load_spec.kvpool_cached_tokens,
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
        if self.load_async:
            self._async_loading_req_ids.add(request.request_id)
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
            save_partial_block=self.layerwise_offload,
            hash_block_size=self.hash_block_size,
            enable_partial_hash_hits=self.enable_partial_hash_hits,
        )

    def _process_new_request(
        self,
        request: Request,
        scheduler_output: SchedulerOutput,
        force_skip_save: bool,
    ) -> ReqMeta | None:
        # Model Runner V2 resumes through scheduled_new_reqs with a full block
        # table. Clear the marker so the next RUNNING delta extends this table
        # instead of rebuilding the tracker from the delta and resetting save
        # progress.
        self._preempted_req_ids.discard(request.req_id)
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
        prefill_tokens = _new_req_prefill_tokens(request)
        request_tracker = RequestTracker(
            req_id=request.req_id,
            token_len=num_tokens_to_compute,
            allocated_block_ids_by_group=block_ids_by_group,
            num_saved_tokens=0,
            token_ids=(prefill_tokens[:num_tokens_to_compute].copy() if self.enable_kv_events else None),
            num_prompt_tokens=len(request.prompt_token_ids),
            prefill_end_tokens=len(prefill_tokens),
            block_gvas=(previous_tracker.block_gvas.copy() if previous_tracker else []),
            gva_block_offset=(previous_tracker.gva_block_offset if previous_tracker else 0),
            num_mamba_speculative_blocks_by_group=self.num_mamba_speculative_blocks_by_group,
            block_sizes=self.grouped_block_size,
        )
        self._request_trackers[request.req_id] = request_tracker
        return self._build_req_meta(
            request_tracker,
            request_real.block_hashes,
            load_spec,
            prefill_tokens,
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
            token_ids=(request_real.all_token_ids[:num_tokens_to_compute].copy() if self.enable_kv_events else None),
            num_prompt_tokens=len(request_real.prompt_token_ids),
            prefill_end_tokens=len(request_real.all_token_ids),
            block_gvas=(previous_tracker.block_gvas.copy() if previous_tracker else []),
            gva_block_offset=(previous_tracker.gva_block_offset if previous_tracker else 0),
            num_mamba_speculative_blocks_by_group=self.num_mamba_speculative_blocks_by_group,
            block_sizes=self.grouped_block_size,
        )
        self._request_trackers[req_id] = request_tracker
        return self._build_req_meta(
            request_tracker,
            request_real.block_hashes,
            load_spec,
            list(request_real.all_token_ids),
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
        # Reused buffers must save every step; otherwise only explicit
        # save_decode_cache keeps Decode increments.
        req_tuple = self._unfinished_requests.get(req_id)
        num_computed_tokens = cached_reqs.num_computed_tokens[i]
        request_tracker = self._request_trackers.get(req_id)
        if request_tracker is None:
            raise ValueError(f"Request {req_id} is not in _request_trackers, but it is scheduled to be cached")
        prefill_end_tokens = request_tracker.prefill_end_tokens or request_tracker.num_prompt_tokens or 0
        is_decoding = num_computed_tokens >= prefill_end_tokens
        if is_decoding and not self.save_decode_cache and not self.layerwise_offload:
            return None
        num_new_tokens = scheduler_output.num_scheduled_tokens[req_id]
        if req_tuple:
            request = req_tuple[0]
            # The scheduler rolls this back when speculative tokens are rejected.
            num_current_tokens = request.num_computed_tokens
            target_token_len = self.get_running_target_token_len(request, num_new_tokens)
            new_token_ids = request.all_token_ids[num_current_tokens:target_token_len]
            if request_tracker.token_ids is not None and new_token_ids:
                request_tracker.token_ids.extend(new_token_ids)
            request_tracker.token_len = target_token_len
        else:
            raise ValueError(f"Request {req_id} is not in _unfinished_requests, but it is scheduled to be cached")
        if new_block_ids is not None:
            request_tracker.update(new_block_ids, request.num_computed_tokens)
        self._sync_swa_block_ids(request_tracker)
        load_spec = self._running_load_spec(request_tracker, num_current_tokens)
        # A consumer skips Prefill writes, including KV received over PD.
        # Its first Decode save may restore missing Prefill ancestors in the pool.
        skip_save = force_skip_save and not (is_decoding and self.save_decode_cache)
        meta = self._build_req_meta(
            request_tracker,
            request.block_hashes,
            load_spec,
            request.prompt_token_ids,
            skip_save,
        )
        if meta is not None:
            meta.computed_prefix_tokens = num_current_tokens
        return meta

    def _sync_swa_block_ids(self, tracker: RequestTracker) -> None:
        """Use the current request table rather than retained SWA block IDs."""
        if not self.swa_group_ids:
            return
        assert self._kv_cache_manager is not None
        current_ids = self._kv_cache_manager.get_block_ids(tracker.req_id)
        for group_id in self.swa_group_ids:
            if group_id < len(current_ids) and group_id < len(tracker.allocated_block_ids_by_group):
                tracker.allocated_block_ids_by_group[group_id][:] = current_ids[group_id]

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
            num_prompt_tokens=len(request.prompt_token_ids),
            prefill_end_tokens=len(request.prompt_token_ids),
            block_gvas=(previous_tracker.block_gvas.copy() if previous_tracker else []),
            gva_block_offset=(previous_tracker.gva_block_offset if previous_tracker else 0),
            num_mamba_speculative_blocks_by_group=self.num_mamba_speculative_blocks_by_group,
            block_sizes=self.grouped_block_size,
        )
        self._request_trackers[request_id] = request_tracker
        return ReqMeta.from_request_tracker(
            request_tracker,
            self.cache_transfer_granularity,
            load_spec=load_spec,
            skip_save=None,
            block_hashes=request.block_hashes,
            discard_partial_chunks=self._discard_partial_chunks,
            original_block_size=self.original_block_size,
            kv_cache_group_families=self.kv_cache_group_families,
            hash_block_size=self.hash_block_size,
        )

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        """Attach the connector metadata to the request object.

        This function should NOT modify other fields in the scheduler_output
        except the `kv_connector_metadata` field.
        Also, calling this function will reset the state of the connector.
        """
        force_skip_save = self.kv_role == "kv_consumer"

        cancelled_load_req_ids = set()
        for finished_req_id in scheduler_output.finished_req_ids:
            load_spec = self.load_specs.pop(finished_req_id, None)
            if load_spec is not None and load_spec.can_load and finished_req_id in self._async_loading_req_ids:
                # Core is waiting for receive completion once it hands us the
                # destination pages. A spec still present here was never sent.
                cancelled_load_req_ids.add(finished_req_id)
            self._request_trackers.pop(finished_req_id, None)
            self._unfinished_requests.pop(finished_req_id, None)
            self._preempted_req_ids.discard(finished_req_id)

        for req_id in scheduler_output.preempted_req_ids:
            self._preempted_req_ids.update(scheduler_output.preempted_req_ids)
            self._request_trackers.pop(req_id, None)
            self._unfinished_requests.pop(req_id, None)
            self._async_loading_req_ids.discard(req_id)

        meta = AscendConnectorMetadata(
            scheduler_output.preempted_req_ids,
            self._async_loading_req_ids.copy(),
            cancelled_load_req_ids=cancelled_load_req_ids,
        )

        for request in scheduler_output.scheduled_new_reqs:
            req_meta = self._process_new_request(request, scheduler_output, force_skip_save)
            if req_meta is not None:
                meta.add_request(req_meta)

        cached_reqs = scheduler_output.scheduled_cached_reqs
        if self.can_put:
            for i, req_id in enumerate(cached_reqs.req_ids):
                new_block_ids = cached_reqs.new_block_ids[i]
                # A decode block is normally allocated before the step that
                # fills it. Decode offload must therefore inspect boundary
                # steps even when this tick allocated no new block.
                if (
                    not new_block_ids
                    and not self.tp_mismatch
                    and not self.layerwise_offload
                    and not self.save_decode_cache
                    and not self.enable_partial_hash_hits
                ):
                    continue
                if req_id in self._preempted_req_ids:
                    if not new_block_ids:
                        continue
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
                    meta.add_request(req_meta)

        if self._unfinished_requests:
            # Index unfinished IDs instead of copying a potentially large decode
            # batch. Keep the original dictionary order when processing them.
            unscheduled_request_ids = set(self._unfinished_requests)
            unscheduled_request_ids.difference_update(req.req_id for req in scheduler_output.scheduled_new_reqs)
            if len(unscheduled_request_ids) == 1:
                # A single membership check needs no index of cached IDs.
                request_id = next(iter(unscheduled_request_ids))
                if request_id in cached_reqs.req_ids:
                    unscheduled_request_ids.clear()
            elif unscheduled_request_ids:
                if len(unscheduled_request_ids) >= len(cached_reqs.req_ids):
                    unscheduled_request_ids.difference_update(cached_reqs.req_ids)
                else:
                    # Sparse lookups can finish once all unfinished IDs match.
                    unscheduled_request_ids.difference_update(unscheduled_request_ids.intersection(cached_reqs.req_ids))
            for request_id, (request, block_ids) in self._unfinished_requests.items():
                if request_id in unscheduled_request_ids:
                    req_meta = self._process_async_load_request(request_id, request, block_ids)
                    if req_meta is not None:
                        meta.add_request(req_meta)

        # Layerwise transfer uses the request tracker's block tables. Only
        # bulk transfer can consume and pin Core's standalone checkpoints.
        if self.use_layerwise:
            return meta

        state = getattr(scheduler_output, "kv_connector_block_state", None)
        if state is not None and state.boundary_state_offloads and self.can_put:
            self._handle_boundary_state_offloads(state.boundary_state_offloads, meta)
        self._apply_current_save_block_ids(meta, scheduler_output)
        # Finish-time sources were pinned before request cleanup and do not
        # require a current block table.
        for req_meta in self._finished_partial_tail_metas.values():
            meta.add_request(req_meta)
        self._finished_partial_tail_metas.clear()
        for req_meta in meta.requests:
            self.retain_blocks_for_save(req_meta)
        return meta

    def _handle_boundary_state_offloads(
        self, offloads: dict[str, list[tuple[int, int, int]]], meta: AscendConnectorMetadata
    ) -> None:
        """Attach Core's exact checkpoint sources, creating boundary-only jobs as needed."""
        coord = self.cache_coordinator
        assert coord is not None
        save_metas = {req.req_id: req for req in meta.requests if req.can_save}
        for req_id, entries in offloads.items():
            tracker = self._request_trackers.get(req_id)
            request = self._unfinished_requests.get(req_id)
            if tracker is None or request is None or not self.can_put:
                continue
            accepted = [
                BoundaryPut(group, block, boundary)
                for group, block, boundary in entries
                if group in coord.mamba_group_ids
                and block > 0
                and boundary <= (tracker.prefill_end_tokens or tracker.num_prompt_tokens or 0)
            ]
            if not accepted:
                continue
            req_meta = save_metas.get(req_id)
            if req_meta is None:
                req_meta = ReqMeta(
                    req_id=req_id,
                    token_len_chunk=0,
                    can_save=True,
                    block_hashes=list(request[0].block_hashes),
                    num_prompt_tokens=tracker.num_prompt_tokens,
                    completed_token_len=tracker.token_len,
                )
                meta.add_request(req_meta)
                save_metas[req_id] = req_meta
            req_meta.boundary_puts = accepted

    def _apply_current_save_block_ids(self, meta: AscendConnectorMetadata, output: SchedulerOutput) -> None:
        """Replace tracker mirrors with Core's current tables for save jobs."""
        save_metas = [req for req in meta.requests if req.can_save]
        if not save_metas:
            return
        state = output.kv_connector_block_state
        assert state is not None, "Current block tables are required for AscendStore jobs"
        for req_meta in save_metas:
            block_ids = state.get_block_ids(req_meta.req_id)
            assert block_ids is not None, f"Missing Core save block table: {req_meta.req_id}"
            req_meta.block_ids_by_group = [list(group) for group in block_ids]

    def retain_blocks_for_save(self, req_meta: ReqMeta) -> None:
        """Retain in-flight Mamba blocks according to the transfer mode. It is only used in Blockwise."""
        return None

    def has_pending_push_work(self) -> bool:
        return False

    def update_connector_output(self, connector_output: KVConnectorOutput):
        """Update requests whose external loads have finished."""
        if connector_output.finished_recving:
            self._async_loading_req_ids.difference_update(connector_output.finished_recving)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """Allow the scheduler to free blocks after synchronous saving."""
        return False, None

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        """Allow the scheduler to free all groups after synchronous saving."""
        return False, None

    def bind_gpu_block_pool(self, gpu_block_pool: "BlockPool") -> None:
        self._block_pool = gpu_block_pool

    def bind_kv_cache_manager(self, kv_cache_manager: KVCacheManager) -> None:
        self._kv_cache_manager = kv_cache_manager

    def reset_store(self) -> bool:
        """Drain the send queue and ``remove_all`` when no in-flight blocks are held."""
        stale_ids = bool(self._async_loading_req_ids)
        # True when any block besides the null block is still held.
        if self._block_pool is None:
            blocks_still_referenced = True
        else:
            used = self._block_pool.num_gpu_blocks - self._block_pool.get_num_free_blocks()
            blocks_still_referenced = used != 1
        if stale_ids and blocks_still_referenced:
            logger.error(
                "AscendStore reset_store refused; blocks still referenced loading_req_ids=%s",
                self._async_loading_req_ids,
            )
            return False
        if stale_ids:
            self._async_loading_req_ids.clear()

        if self.client is None:
            self.client = LookupKeyClient(self.vllm_config)
        try:
            ok = self.client.reset()
            if ok:
                logger.info("AscendStore reset via remove_all succeeded.")
            else:
                logger.warning("AscendStore reset returned NACK from worker.")
            return ok
        except Exception as e:
            logger.error("AscendStore reset_store RPC failed: %s", e)
            return False


class KVPoolScheduler(BaseKVPoolScheduler):
    """Scheduler for whole-block KV pool transfers."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        use_layerwise: bool,
        kv_cache_config: KVCacheConfig | None = None,
    ) -> None:
        super().__init__(vllm_config, use_layerwise, kv_cache_config)
        self._failed_external_load_req_ids: set[str] = set()
        self._in_flight_save_blocks = InFlightSaveBlockTracker(vllm_config.parallel_config.world_size)

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        if connector_output.failed_recving and has_hisparse_cache_layout(self.kv_cache_config):
            for request_id in self._async_loading_req_ids & connector_output.failed_recving:
                request_and_blocks = self._unfinished_requests.get(request_id)
                if request_and_blocks is not None and not request_and_blocks[0].is_finished():
                    self._failed_external_load_req_ids.add(request_id)
        super().update_connector_output(connector_output)
        meta = connector_output.kv_connector_worker_meta
        if isinstance(meta, AscendStoreKVConnectorWorkerMetadata) and self._block_pool is not None:
            self._in_flight_save_blocks.complete(meta.completed_events, self._block_pool)

    def request_finished_all_groups(
        self, request: Request, block_ids: tuple[list[int], ...]
    ) -> tuple[bool, dict[str, Any] | None]:
        self._failed_external_load_req_ids.discard(request.request_id)
        return super().request_finished_all_groups(request, block_ids)

    def has_pending_push_work(self) -> bool:
        return self._in_flight_save_blocks.has_pending_work()

    def reset_store(self) -> bool:
        tracker = self._in_flight_save_blocks
        if tracker.has_tracked_blocks():
            logger.error(
                "AscendStore reset_store refused; in-flight saves still track sending_events=%s sending_blocks=%s",
                list(tracker.sending_events),
                list(tracker.sending_blocks),
            )
            return False
        return super().reset_store()

    def retain_blocks_for_save(self, req_meta: ReqMeta) -> None:
        """Keep Mamba source blocks alive until bulk save completes."""
        if req_meta.can_save and req_meta.event_id is None and self.cache_coordinator is not None:
            if tail := self._partial_tail_non_mamba_puts(req_meta):
                req_meta.boundary_puts = [*tail, *(req_meta.boundary_puts or [])]
        self._in_flight_save_blocks.retain(
            req_meta,
            self._block_pool,
        )

    def _partial_tail_non_mamba_puts(self, req_meta: ReqMeta) -> list[BoundaryPut]:
        """Locate the non-Mamba blocks a partial-tail save publishes.

        A later request resumes at the prompt's Mamba checkpoint (``boundary``)
        only if both pieces of KV are stored:

        - Mamba groups: the state at ``boundary``, handed off by Core as a
          boundary put of its own, so Mamba groups are not listed here.
        - Other groups: the KV from the last LCM-aligned normal save up to
          ``boundary`` plus the group's EAGLE proof margin (``proof_end``).
          Each block is keyed by the hash at its end, clipped to ``proof_end``.

        Groups whose proof is not computed yet are omitted. For example, with
        LCM 16, blocks of 4 tokens, and a checkpoint at 44, the full-attention
        blocks are 8-10, keyed at 36, 40 and 44.

        A Mamba state handed off at any other position, i.e. a shared-prefix
        junction (``--enable-mamba-shared-prefix-checkpoint``), adds no puts here.
        """
        coord = self.cache_coordinator
        assert coord is not None
        mamba_tails = [
            put.num_tokens
            for put in req_meta.boundary_puts or []
            if put.group_id in coord.mamba_group_ids and put.num_tokens % self.grouped_block_size[put.group_id]
        ]
        # The tail is due on the save that first covers the prompt, or when Mamba
        # hands off its checkpoint state.
        if not mamba_tails and not req_meta.publish_partial_tail:
            return []
        prompt_tokens = req_meta.num_prompt_tokens
        completed = req_meta.completed_token_len
        assert prompt_tokens is not None and completed is not None
        hash_block_size = coord.hash_block_size
        # Core drops the Mamba checkpoint whenever EAGLE block drop is on,
        # whichever groups carry the eagle flag.
        boundary = get_mamba_prefill_checkpoint_position(prompt_tokens, hash_block_size, coord.use_eagle)
        # Any other sub-block Mamba hand-off is a shared-prefix junction. Its
        # attention KV, EAGLE proof included, ends on an attention block end no
        # later than this checkpoint's proof, so normal and prompt-tail saves
        # store it.
        if not req_meta.publish_partial_tail and boundary not in mamba_tails:
            return []
        num_hashes = len(req_meta.block_hashes)
        if boundary == 0 or boundary // hash_block_size > num_hashes:
            return []
        start = boundary // coord.lcm_block_size * coord.lcm_block_size
        puts: list[BoundaryPut] = []
        # Ascend keeps original group IDs; exclude non-cacheable groups here
        # rather than renumbering the block tables as Mooncake Store does.
        for group_id in self.cacheable_group_ids:
            if group_id in coord.mamba_group_ids:
                continue
            proof_end = boundary + coord.eagle_proof_margin_by_group.get(group_id, 0)
            if proof_end > completed or proof_end // hash_block_size > num_hashes:
                continue
            block_size = self.grouped_block_size[group_id]
            group_blocks = req_meta.block_ids_by_group[group_id]
            for block_idx in range(start // block_size, cdiv(proof_end, block_size)):
                if block_idx >= len(group_blocks):
                    break
                block_id = group_blocks[block_idx]
                if block_id == NULL_BLOCK_ID:
                    continue
                num_tokens = min((block_idx + 1) * block_size, proof_end)
                puts.append(BoundaryPut(group_id, block_id, num_tokens))
        return puts

    def register_finished_partial_tail(
        self,
        request: Request,
        block_ids: tuple[list[int], ...],
        partial_tail_offloads: list[tuple[int, int, int]],
    ) -> bool:
        """Queue and pin a finish-time tail for the next connector step."""
        if not self.can_put or not partial_tail_offloads:
            return False
        tracker = self._request_trackers.get(request.request_id)
        if tracker is None or not any(block_ids):
            return False
        boundaries = {boundary for _, _, boundary in partial_tail_offloads}
        if len(boundaries) != 1:
            raise ValueError("Partial-tail offloads for one request must share a boundary")
        boundary_tokens = next(iter(boundaries))
        assert tracker.prefill_end_tokens is not None
        if boundary_tokens > tracker.prefill_end_tokens:
            return False

        coord = self.cache_coordinator
        assert coord is not None
        mamba_puts: list[BoundaryPut] = []
        for group_id, block_id, boundary in partial_tail_offloads:
            # Ascend retains Core's group IDs instead of remapping Store groups.
            if group_id not in coord.mamba_group_ids:
                return False
            if block_id == NULL_BLOCK_ID:
                return False
            mamba_puts.append(BoundaryPut(group_id, block_id, boundary))
        req_meta = ReqMeta(
            req_id=request.request_id,
            token_len_chunk=0,
            can_save=True,
            block_ids_by_group=[list(group) for group in block_ids],
            block_hashes=list(request.block_hashes),
            num_prompt_tokens=request.num_prompt_tokens,
            completed_token_len=request.num_computed_tokens,
            boundary_puts=mamba_puts,
        )
        req_meta.boundary_puts = [*self._partial_tail_non_mamba_puts(req_meta), *mamba_puts]
        assert request.request_id not in self._finished_partial_tail_metas
        # This job has no normal save range. Pin only its exact boundary sources.
        self._in_flight_save_blocks.retain(req_meta, self._block_pool)
        self._finished_partial_tail_metas[request.request_id] = req_meta
        # The store job owns exact block refs, so request cleanup need not wait.
        return False

    def _lookup_external_hit_tokens(self, request: Request, num_computed_tokens: int) -> StoreLookupResult | None:
        # Core may reschedule the request with zero computed tokens after an
        # import failure. A key can still be visible to lookup while GET fails.
        if request.request_id in self._failed_external_load_req_ids:
            return None
        prompt_token_len = len(request.prompt_token_ids)
        if self.enable_partial_hash_hits:
            # Preserve the prompt end so the worker can reserve the final token for recomputation.
            token_len = prompt_token_len
        elif len(self.cacheable_group_ids) != len(self.original_block_size):
            # Private compressor state is rebuilt from a whole token page.
            token_len = self._floor_to_cache_transfer_granularity(max(prompt_token_len - 1, 0))
        else:
            token_len = (
                self._floor_to_cache_transfer_granularity(prompt_token_len)
                if self._discard_partial_chunks
                else prompt_token_len
            )
        if token_len < self.cache_transfer_granularity or num_computed_tokens >= token_len:
            return None
        if self.client is None or self.client.closed:
            self.client = LookupKeyClient(self.vllm_config)
        return self.client.lookup(
            token_len,
            request.block_hashes,
            self.kv_cache_group_ids,
            hbm_hit_tokens=num_computed_tokens,
        )
