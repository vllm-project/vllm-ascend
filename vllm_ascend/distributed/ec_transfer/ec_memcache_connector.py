"""ECMemcacheConnector backed by memcache.

Integrate with vLLM through the ec_connector framework without modifying vLLM:
    --ec-transfer-config '{
        "ec_connector": "ECMemcacheConnector",
        "ec_connector_module_path":
            "vllm_ascend.distributed.ec_transfer.ec_memcache_connector",
        "ec_role": "ec_both",
        "ec_connector_extra_config": {
            "recv_buffer_tokens": 8192,
            "similarity_cache": {
                "enabled": true,
                "matcher": "phash_ssim",
                "max_hamming": 5,
                "ssim_threshold": 0.99
            }
        }
    }'

Cache hit rules, adapted from MultiLevelEncoderCacheManager:
  - key = mm_hash (request.mm_features[i].identifier); a hit skips ViT;
  - when similarity_cache is enabled, an L1 miss may reuse an existing
    embedding recalled by pHash and verified against the SSIM threshold;
  - only a miss runs ViT, then stores the result in memcache under the image's
    mm_hash.

Memcache manages eviction; this connector does not implement an eviction policy.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorBase,
    ECConnectorMetadata,
    ECConnectorRole,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.common import (
    _get_encoder_cache_hidden_dim,
)
from vllm.distributed.parallel_state import (
    get_pcp_group,
    get_tensor_model_parallel_rank,
)
from vllm.logger import logger

from vllm_ascend.distributed.ec_transfer.perceptual_ssim_cache_matcher import (
    PhashSSIMCacheMatcher,
    SimilarityCacheConfig,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.memcache_backend import (
    MemcacheBackend,
    MmcDirect,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request


MIN_RECV_BUFFER_TOKENS = 2048
DEFAULT_RECV_BUFFER_TOKENS = 8192


def _get_recv_buffer_tokens(vllm_config: "VllmConfig") -> int:
    ec_config = vllm_config.ec_transfer_config
    if ec_config is None:
        raise ValueError("ECMemcacheConnector requires ec_transfer_config")

    recv_buffer_tokens = ec_config.get_from_extra_config("recv_buffer_tokens", DEFAULT_RECV_BUFFER_TOKENS)
    if isinstance(recv_buffer_tokens, bool) or not isinstance(recv_buffer_tokens, int):
        raise ValueError(f"ec_connector_extra_config.recv_buffer_tokens must be an integer, got {recv_buffer_tokens!r}")

    max_num_batched_tokens = vllm_config.scheduler_config.max_num_batched_tokens
    if not MIN_RECV_BUFFER_TOKENS <= recv_buffer_tokens <= max_num_batched_tokens:
        raise ValueError(
            "ec_connector_extra_config.recv_buffer_tokens must be between "
            f"{MIN_RECV_BUFFER_TOKENS} and scheduler_config."
            f"max_num_batched_tokens ({max_num_batched_tokens}), "
            f"got {recv_buffer_tokens}"
        )
    return recv_buffer_tokens


@dataclass
class ECMemcacheConnectorMetadata(ECConnectorMetadata):
    mm_hashes_need_loads: list[str] = field(default_factory=list)
    mm_hashes_need_saves: set[str] = field(default_factory=set)
    # Only similarity hits need an alternate lookup key. Exact hits omit an
    # entry and load from Memcache with the request mm_hash itself.
    mm_hash_to_store_key: dict[str, str] = field(default_factory=dict)


@dataclass
class _GetRequestParam:
    src_idx: int
    key: str
    size: int
    num_tokens: int


class ECMemcacheConnector(ECConnectorBase):
    """Encoder cache connector backed by memcache."""

    def __init__(self, vllm_config: "VllmConfig", role: ECConnectorRole) -> None:
        super().__init__(vllm_config=vllm_config, role=role)
        model_config = vllm_config.model_config

        self._model_id: str = model_config.model
        self._similarity_config = SimilarityCacheConfig.from_vllm_config(vllm_config)
        self._cache_matcher: PhashSSIMCacheMatcher | None = None

        if role == ECConnectorRole.SCHEDULER:
            # Similarity matching persists and reloads resized tensors. Select
            # the scheduler's final backend capability before the process's
            # first Memcache initialization; a metadata-only native client
            # cannot be upgraded by constructing a second backend object.
            if self._similarity_config.enabled:
                self._backend = MemcacheBackend.create_scheduler_data_client(vllm_config.parallel_config)
                hf_config = getattr(model_config, "hf_config", None)
                vision_config = getattr(hf_config, "vision_config", None)
                merge_size = int(getattr(vision_config, "spatial_merge_size", 2) or 2)
                self._cache_matcher = PhashSSIMCacheMatcher(
                    self._backend,
                    self._similarity_config,
                    merge_size=merge_size,
                )
            else:
                self._backend = MemcacheBackend.create_scheduler_client(vllm_config.parallel_config)
            # Exact hits contain only the current request's original mm_hash.
            self._mm_hash_hits: set[str] = set()
            # Similarity hits map the current request's original mm_hash to
            # the similar candidate original mm_hash whose embedding will be loaded.
            self._similarity_mm_hash_hits: dict[str, str] = {}
            self._mm_hash_hit_count = 0
            self._similarity_hit_count = 0
            self._miss_count = 0
            # for worker
            self._mm_hashes_need_loads: set[str] = set()
            self._mm_hash_to_store_key: dict[str, str] = {}
            self._mm_hashes_need_saves: set[str] = set()
        elif role == ECConnectorRole.WORKER:
            # Data-plane client for reading and writing embeddings.
            self._backend = MemcacheBackend(vllm_config.parallel_config)
            self._hidden_dim = _get_encoder_cache_hidden_dim(vllm_config)
            self._dtype = model_config.dtype
            self._elem_size = torch.empty(0, dtype=self._dtype).element_size()
            self._recv_buffer_tokens = _get_recv_buffer_tokens(vllm_config)
            self._recv_buffer = torch.empty(
                self._recv_buffer_tokens,
                self._hidden_dim,
                dtype=self._dtype,
                device="npu",
            )
            self._recv_buffer_reuse_event = None
            self._recv_buffer_registered = False
            self._backend.register_buffer(
                [self._recv_buffer.data_ptr()],
                [self._recv_buffer.nbytes],
            )
            self._recv_buffer_registered = True
            logger.info(
                "Registered EC receive buffer: tokens=%d hidden_dim=%d dtype=%s bytes=%d",
                self._recv_buffer_tokens,
                self._hidden_dim,
                self._dtype,
                self._recv_buffer.nbytes,
            )
            self._save_rank = get_tensor_model_parallel_rank() == 0 and get_pcp_group().rank_in_group == 0
            self._put_executor = (
                ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix="ec-put",
                )
                if self._save_rank
                else None
            )
        else:
            raise ValueError(f"Unknown ECConnectorRole: {role}")

        logger.info(
            "ECMemcacheConnector init: role=%s model=%s similarity_cache_enabled=%s matcher=%s",
            role,
            self._model_id,
            self._similarity_config.enabled,
            self._similarity_config.matcher,
        )

    # ==============================
    # Scheduler-side methods
    # ==============================

    def ensure_cache_available(self, request: "Request", num_computed_tokens: int) -> bool:
        similarity_enabled = self._similarity_config.enabled
        missing_features = self._ensure_exact_cache_available(
            request,
            collect_missing=similarity_enabled,
        )
        if similarity_enabled:
            self._ensure_similarity_cache_available(missing_features)
        return True

    def has_cache_item(self, identifier: str) -> bool:
        if identifier in self._mm_hash_hits:
            return True
        # Fall back to a direct L1 lookup for paths not covered by
        # ensure_cache_available, such as continued chunk scheduling.
        if self._backend.exists([identifier]) == [1]:
            self._mm_hash_hits.add(identifier)
            self._mm_hash_hit_count += 1
            logger.debug("Current mm_hash=%s hits in memcache", identifier)
            return True
        return False

    def update_state_after_alloc(self, request: "Request", index: int) -> None:
        feature = request.mm_features[index]
        current_image_mm_hash = feature.identifier

        # If there is an exact hit for current_image_mm_hash, it will certainly be loaded.
        if current_image_mm_hash in self._mm_hash_hits:
            self._mm_hashes_need_loads.add(current_image_mm_hash)
            return

        similarity_enabled = self._similarity_config.enabled
        if not similarity_enabled:
            self._miss_count += 1
            self._mm_hashes_need_saves.add(current_image_mm_hash)

        # If there is a similarity hit for current_image_mm_hash in current batch, it will certainly be loaded.
        similar_mm_hash = self._similarity_mm_hash_hits.get(current_image_mm_hash)
        if similar_mm_hash is not None:
            self._mm_hashes_need_loads.add(similar_mm_hash)
            return

        # If there is no similarity hit for current_image_mm_hash in current batch, start the similarity lookup process.
        cache_matcher = self._cache_matcher
        if cache_matcher is not None:
            try:
                similar_mm_hash = cache_matcher.get_or_update_similarity_match(current_image_mm_hash, feature.data)
                if similar_mm_hash is not None:
                    self._similarity_hit_count += 1
                    self._mm_hashes_need_loads.add(similar_mm_hash)
                    logger.debug("EC similarity match success: mm_hash=%s, similar_mm_hash=%s", current_image_mm_hash, similar_mm_hash)
                    return
            except Exception:
                logger.exception("EC similarity match failed: mm_hash=%s", current_image_mm_hash)
        self._miss_count += 1
        self._mm_hashes_need_saves.add(current_image_mm_hash)


    @property
    def hit_rate(self) -> float:
        """Return the cumulative exact and similarity cache hit rate."""
        hits = self._mm_hash_hit_count + self._similarity_hit_count
        total = hits + self._miss_count
        if total == 0:
            return 0.0
        return hits / total

    def build_connector_meta(self, scheduler_output: "SchedulerOutput") -> ECMemcacheConnectorMetadata:
        meta = ECMemcacheConnectorMetadata(
            mm_hashes_need_loads=self._mm_hashes_need_loads,
            mm_hashes_need_saves=self._mm_hashes_need_saves,
            mm_hash_to_store_key=self._mm_hash_to_store_key,
        )
        if meta.mm_hashes_need_loads or meta.mm_hashes_need_saves:
            logger.debug(
                "EC meta: %d loads, %d saves this step | "
                "EC meta loads: %r, EC meta saves: %r this step | "
                "exact_hits=%d similarity_hits=%d misses=%d hit_rate=%.2f%%",
                len(meta.mm_hashes_need_loads),
                len(meta.mm_hashes_need_saves),
                meta.mm_hashes_need_loads,
                meta.mm_hashes_need_saves,
                self._mm_hash_hit_count,
                self._similarity_hit_count,
                self._miss_count,
                self.hit_rate * 100,
            )
        # ECMemcacheConnectorMetadata will be passed to worker, so states must be cleared before next batch coming
        # and the statistics fields keep remaining
        self._mm_hashes_need_loads = []
        self._mm_hash_to_store_key = {}
        self._mm_hashes_need_saves = set()
        self._mm_hash_hits.clear()
        self._similarity_mm_hash_hits.clear()
        if self._cache_matcher is not None:
            self._cache_matcher.clear_step_state()
        return meta

    # ==============================
    # Scheduler-side memcache helpers
    # ==============================

    def _ensure_exact_cache_available(
        self,
        request: "Request",
        *,
        collect_missing: bool,
    ) -> list[Any]:
        seen_mm_hashes_in_current_request: list[str] = []
        feature_by_mm_hash: dict[str, Any] = {}
        for feature in request.mm_features:
            current_image_mm_hash = feature.identifier
            if (
                current_image_mm_hash in self._mm_hash_hits
                or current_image_mm_hash in self._similarity_mm_hash_hits
                or current_image_mm_hash in seen_mm_hashes_in_current_request
            ):
                continue
            seen_mm_hashes_in_current_request.append(current_image_mm_hash)
            if collect_missing:
                feature_by_mm_hash[current_image_mm_hash] = feature

        exists_res = (
            self._backend.exists(seen_mm_hashes_in_current_request) if seen_mm_hashes_in_current_request else []
        )
        if seen_mm_hashes_in_current_request:
            if not exists_res:
                logger.debug("EC memcache exists failed: keys=%r", seen_mm_hashes_in_current_request)
                # If no exists_res returned, fill exists_res with zeros, which means all keys are not exist
                exists_res = [0] * len(seen_mm_hashes_in_current_request)
            elif len(exists_res) != len(seen_mm_hashes_in_current_request):
                logger.debug(
                    "EC memcache exists returned %d results for %d keys: keys=%r",
                    len(exists_res),
                    len(seen_mm_hashes_in_current_request),
                    seen_mm_hashes_in_current_request,
                )
                exists_res = [0] * len(seen_mm_hashes_in_current_request)

        missing_features: list[Any] = []
        for current_image_mm_hash, existed in zip(
            seen_mm_hashes_in_current_request,
            exists_res,
        ):
            if existed == 1:
                self._mm_hash_hits.add(current_image_mm_hash)
                self._mm_hash_hit_count += 1
                logger.debug("Current mm_hash=%s is available in memcache", current_image_mm_hash)
            elif collect_missing:
                missing_features.append(feature_by_mm_hash[current_image_mm_hash])
        return missing_features

    def _ensure_similarity_cache_available(
        self,
        missing_features: list[Any],
    ) -> None:
        matcher = self._cache_matcher
        if matcher is None or not missing_features:
            return
        mm_hashes = [feature.identifier for feature in missing_features]
        try:
            matcher.batch_ensure_similarity_cache_available(mm_hashes, missing_features)
        except Exception:
            logger.exception("EC similarity batch prewarm failed: mm_hashes=%r", mm_hashes)
            return


    # ==============================
    # Worker-side methods
    # ==============================

    def start_load_caches(self, encoder_cache: dict[str, torch.Tensor], **kwargs) -> None:
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, ECMemcacheConnectorMetadata)
        loading_hashes_in_current_step = [
            mm_hash for mm_hash in metadata.mm_hashes_need_loads if mm_hash not in encoder_cache
        ]
        if not loading_hashes_in_current_step:
            return

        store_keys = [
            metadata.mm_hash_to_store_key.get(mm_hash, mm_hash) for mm_hash in loading_hashes_in_current_step
        ]
        embeddings = self._ec_get_batch(store_keys)
        if not embeddings:
            return
        for mm_hash, store_key, embedding in zip(loading_hashes_in_current_step, store_keys, embeddings):
            if embedding is None:
                logger.warning(
                    "EC LOAD miss: request_key=%s store_key=%s (may be evicted by memcache?)",
                    mm_hash,
                    store_key,
                )
                continue
            encoder_cache[mm_hash] = embedding
            logger.debug(
                "EC LOAD: request_key=%s store_key=%s hit_kind=%s embedding shape=%r",
                mm_hash,
                store_key,
                "exact" if mm_hash == store_key else self._similarity_config.matcher,
                embedding.shape,
            )

    def save_caches(self, encoder_cache: dict[str, torch.Tensor], mm_hash: str, **kwargs) -> None:
        if not self._save_rank:
            return
        if mm_hash not in encoder_cache:
            return
        embedding = encoder_cache[mm_hash]
        self._ec_put(mm_hash, embedding)

    # ==============================
    # Worker-side memcache helpers
    # ==============================

    def _wait_recv_buffer_reusable(self) -> None:
        event = getattr(self, "_recv_buffer_reuse_event", None)
        if event is None:
            return
        event.synchronize()
        self._recv_buffer_reuse_event = None

    def _ec_get_batch(self, keys: list[str]) -> list[torch.Tensor | None] | None:
        if not keys:
            return None
        embeddings: list[torch.Tensor | None] = [None] * len(keys)
        key_infos = self._backend.batch_get_key_info(keys)
        if not key_infos:
            logger.warning("EC memcache batch_get_key_info failed: keys=%s", keys)
            return None
        if len(key_infos) != len(keys):
            logger.warning(
                "EC memcache batch_get_key_info returned %d results for %d keys",
                len(key_infos),
                len(keys),
            )
            return None

        request_params = self._build_get_request_params(keys, key_infos)
        regular_requests: list[_GetRequestParam] = []
        oversized_requests: list[_GetRequestParam] = []
        for req in request_params:
            if req.num_tokens > self._recv_buffer_tokens:
                oversized_requests.append(req)
            else:
                regular_requests.append(req)

        # First, use the persistent, user-sized buffer for ordinary requests.
        for regular_batch in self._build_capacity_batches(regular_requests, self._recv_buffer_tokens):
            self._load_regular_requests(regular_batch, embeddings)

        # Second, Oversized requests are deferred. so one temporary registration can be
        # shared by every oversized value in this scheduler batch.
        self._load_oversized_requests(oversized_requests, embeddings)
        return embeddings

    def _build_get_request_params(self, keys: list[str], key_infos: list[Any]) -> list[_GetRequestParam]:
        bytes_per_token = self._elem_size * self._hidden_dim
        request_params: list[_GetRequestParam] = []
        for i, (key, key_info) in enumerate(zip(keys, key_infos)):
            nbytes = int(key_info.size())
            if nbytes == 0:
                continue
            if nbytes % bytes_per_token != 0:
                logger.warning(
                    "EC memcache value size is not aligned with the encoder "
                    "hidden dimension: key=%s nbytes=%d bytes_per_token=%d",
                    key,
                    nbytes,
                    bytes_per_token,
                )
                continue
            num_tokens = nbytes // bytes_per_token
            request_params.append(
                _GetRequestParam(
                    i,
                    key,
                    nbytes,
                    num_tokens,
                )
            )
        return request_params

    def _load_regular_requests(
        self,
        request_params: list[_GetRequestParam],
        embeddings: list[torch.Tensor | None],
    ) -> None:
        self._wait_recv_buffer_reusable()
        recv_buffer = self._recv_buffer
        if recv_buffer is None:
            raise RuntimeError("EC receive buffer is unavailable")

        self._load_batch_into_buffer(
            request_params,
            embeddings,
            recv_buffer,
            # The persistent buffer remains registered. Defer the wait until
            # the next reuse so this call does not block on copy-out.
            should_sync_copy_out_event_immediately=False,
        )

    def _load_batch_into_buffer(
        self,
        request_params: list[_GetRequestParam],
        embeddings: list[torch.Tensor | None],
        recv_buffer: torch.Tensor,
        *,
        should_sync_copy_out_event_immediately: bool,
    ) -> None:
        bytes_per_token = self._elem_size * self._hidden_dim
        base_addr = recv_buffer.data_ptr()
        offsets: list[int] = []
        offset_tokens = 0
        for req in request_params:
            offsets.append(offset_tokens)
            offset_tokens += req.num_tokens

        group_res = self._backend.batch_get_into_buffers(
            [req.key for req in request_params],
            [base_addr + offset * bytes_per_token for offset in offsets],
            [req.size for req in request_params],
            MmcDirect.COPY_G2L.value,
        )
        logger.debug(
            "EC BATCH GET: keys length=%d keys content=%r direction=%d",
            len(request_params),
            [req.key for req in request_params],
            MmcDirect.COPY_G2L.value,
        )
        if group_res is None:
            for req in request_params:
                logger.warning("EC memcache get failed: key=%s res=None", req.key)
            return
        if len(group_res) != len(request_params):
            logger.warning(
                "EC memcache batch_get_into_buffers returned %d results for %d keys",
                len(group_res),
                len(request_params),
            )

        copy_out_event_enqueued = False
        try:
            for result_idx, (req, offset) in enumerate(zip(request_params, offsets)):
                if result_idx >= len(group_res):
                    logger.warning("EC memcache get returned no result for key=%s", req.key)
                    continue
                code = group_res[result_idx]
                if code != 0:
                    logger.warning("EC memcache get failed: key=%s res=%s", req.key, code)
                    continue
                staging_view = recv_buffer[offset : offset + req.num_tokens]
                output = torch.empty(
                    req.num_tokens,
                    self._hidden_dim,
                    dtype=self._dtype,
                    device="npu",
                )
                output.copy_(staging_view, non_blocking=True)
                copy_out_event_enqueued = True
                embeddings[req.src_idx] = output
                logger.debug(
                    "EC memcache get success: key=%s output shape=%r",
                    req.key,
                    output.shape,
                )
        finally:
            if copy_out_event_enqueued:
                # This event tracks output.copy_(staging_view), which must
                # finish before the staging buffer is overwritten or released.
                copy_done_event = torch.npu.Event()
                copy_done_event.record(torch.npu.current_stream())
                # The param should_sync_copy_out_event_immediately is working with self._wait_recv_buffer_reusable()
                # method.
                # 1. For regular request, should_sync_copy_out_event_immediately is false, we use a global buffer
                # to receive embeddings, and the copy out event can wait util the next batch begins.
                # 2. For oversized request, should_sync_copy_out_event_immediately is true, we use a temporary local
                # buffer to receive embeddings every batch, so the copy out event should be synchronized within
                # the same batch.
                if should_sync_copy_out_event_immediately:
                    copy_done_event.synchronize()
                else:
                    self._recv_buffer_reuse_event = copy_done_event

    @staticmethod
    def _build_capacity_batches(
        request_params: list[_GetRequestParam], capacity_tokens: int
    ) -> list[list[_GetRequestParam]]:
        """Greedily group requests without changing their relative order."""
        batches: list[list[_GetRequestParam]] = []
        current_batch: list[_GetRequestParam] = []
        current_batch_tokens = 0
        for req in request_params:
            if req.num_tokens > capacity_tokens:
                raise ValueError(
                    "Request exceeds receive buffer capacity: "
                    f"key={req.key} request_tokens={req.num_tokens} "
                    f"capacity_tokens={capacity_tokens}"
                )
            if current_batch_tokens + req.num_tokens > capacity_tokens:
                batches.append(current_batch)
                current_batch = []
                current_batch_tokens = 0
            current_batch.append(req)
            current_batch_tokens += req.num_tokens
        if current_batch:
            batches.append(current_batch)
        return batches

    def _load_oversized_requests(
        self,
        request_params: list[_GetRequestParam],
        embeddings: list[torch.Tensor | None],
    ) -> None:
        if not request_params:
            return
        self._wait_recv_buffer_reusable()
        temporary_buffer_tokens = max(req.num_tokens for req in request_params)
        logger.warning(
            "EC memcache values exceed the configured persistent receive "
            "buffer; using one temporary buffer for this batch: "
            "num_requests=%d temporary_buffer_tokens=%d "
            "persistent_capacity_tokens=%d",
            len(request_params),
            temporary_buffer_tokens,
            self._recv_buffer_tokens,
        )
        temporary_buffer = torch.empty(
            temporary_buffer_tokens,
            self._hidden_dim,
            dtype=self._dtype,
            device="npu",
        )
        temporary_buffer_ptr = temporary_buffer.data_ptr()
        registered = False
        try:
            # recv_buffer_tokens limits persistent NPU memory only. Oversized
            # values share this per-call staging buffer, which is registered
            # once and released after all second-pass batches finish.
            self._backend.register_buffer([temporary_buffer_ptr], [temporary_buffer.nbytes])
            registered = True
            for batch in self._build_capacity_batches(request_params, temporary_buffer_tokens):
                self._load_batch_into_buffer(
                    batch,
                    embeddings,
                    temporary_buffer,
                    # The next second-pass batch overwrites this temporary
                    # buffer, and the buffer is unregistered after the loop.
                    # Therefore each copy-out must complete before returning.
                    should_sync_copy_out_event_immediately=True,
                )
        finally:
            if registered:
                self._backend.unregister_buffer([temporary_buffer_ptr], [temporary_buffer.nbytes])

    def _ec_put(self, mm_hash: str, embedding: torch.Tensor) -> None:
        executor = self._put_executor
        if executor is None:
            raise RuntimeError("EC PUT executor is unavailable on this rank")

        t = embedding.contiguous()
        ready_event = torch.npu.Event()
        ready_event.record(torch.npu.current_stream())
        executor.submit(
            self._put_async,
            mm_hash,
            t,
            ready_event,
        )

    def _put_async(
        self,
        mm_hash: str,
        t: torch.Tensor,
        ready_event,
    ) -> None:
        try:
            torch.npu.set_device(t.device)
            ready_event.synchronize()

            res = self._backend.put_from(
                mm_hash,
                t.data_ptr(),
                t.nbytes,
            )
            if res != 0:
                return
        except Exception:
            logger.exception("EC PUT failed: mm_hash=%s", mm_hash)

    def shutdown(self) -> None:
        matcher = getattr(self, "_cache_matcher", None)
        if matcher is not None:
            matcher.shutdown()

        executor = getattr(self, "_put_executor", None)
        if executor is not None:
            executor.shutdown(wait=True)

        self._wait_recv_buffer_reusable()
        recv_buffer = getattr(self, "_recv_buffer", None)
        if recv_buffer is None:
            return
        if self._recv_buffer_registered:
            self._backend.unregister_buffer([recv_buffer.data_ptr()], [recv_buffer.nbytes])
            self._recv_buffer_registered = False
        self._recv_buffer = None
