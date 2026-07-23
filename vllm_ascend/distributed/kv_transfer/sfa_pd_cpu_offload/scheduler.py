# mypy: ignore-errors
# SPDX-License-Identifier: Apache-2.0
"""Scheduler side of the PD-disaggregated SFA connector.

D (``kv_consumer``): retain the normal vLLM main/indexer block ids allocated
for the request. Main ids address KVOffloadDecodeManager's CPU pool and indexer
ids address rank-local HBM. The metaserver rendezvous carries only contact info
and ``do_remote_decode``; D resolves its destinations by request id.

P (``kv_producer``): build metadata for layer-wise READ_READY notifications.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import TYPE_CHECKING, Any

import httpx
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.logger import logger
from vllm.utils.math_utils import round_down
from vllm.utils.network_utils import get_ip
from vllm.v1.kv_cache_interface import KVCacheConfig

from vllm_ascend import envs
from vllm_ascend.distributed.kv_transfer.sfa_pd_cpu_offload.protocol import (
    BATCH_KV_TRANSFER_PARAMS,
    SfaPDConsumerMetadata,
    SfaPDProducerMetadata,
    get_external_request_id,
    infer_sfa_component_group_ids,
)

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request

METASERVER_MAX_RETRIES = 3
METASERVER_RETRY_DELAY_SECONDS = 1.0


class _SendReqInfo:
    def __init__(
        self,
        local_block_ids: list[list[int]],
        local_transferred_tokens: int,
        local_computed_tokens: int,
        request: Request,
    ) -> None:
        self.local_block_ids = local_block_ids
        self.local_transferred_tokens = local_transferred_tokens
        self.local_computed_tokens = local_computed_tokens
        self.request = request

    def extend_local_block_ids(self, new_block_ids: list[list[int]]) -> None:
        for i, new_block_id in enumerate(new_block_ids):
            self.local_block_ids[i].extend(new_block_id)

    def update_computed_tokens(self, computed_tokens: int) -> None:
        self.local_computed_tokens = computed_tokens

    def update_transferred_tokens(self, transferred_tokens: int) -> None:
        self.local_transferred_tokens = transferred_tokens


class SFAPDProducerScheduler:
    """P-side scheduler for SFA PD (pull mode).

    D's metaserver rendezvous carries ``do_remote_decode=True`` plus D's ZMQ
    endpoint. P tracks its own local block ids and emits per-step metadata for
    the pull-mode sending thread; D looks up its destination blocks by req_id.
    """

    def __init__(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig, engine_id: str):
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        self.engine_id = engine_id
        self.block_size = [group_spec.kv_cache_spec.block_size for group_spec in kv_cache_config.kv_cache_groups]
        self._reqs_need_send_layerwise: dict[str, _SendReqInfo] = {}

    @staticmethod
    def _normalize_block_ids(block_ids: Any) -> list[list[int]]:
        if block_ids is None:
            return []
        if block_ids and isinstance(block_ids[0], int):
            return [list(block_ids)]
        if isinstance(block_ids, tuple):
            return [list(group) for group in block_ids]
        return [list(group) for group in block_ids]

    def get_num_new_matched_tokens(self, request: Request, num_computed_tokens: int) -> tuple[int, bool]:
        return 0, False

    def update_state_after_alloc(
        self,
        request: Request,
        blocks: KVCacheBlocks,
        num_external_tokens: int,
    ) -> None:
        params = request.kv_transfer_params
        if params is None or not params.get("do_remote_decode"):
            return

        batch_params = params.get(BATCH_KV_TRANSFER_PARAMS)
        if batch_params is not None:
            external_request_id = get_external_request_id(request.request_id)
            params = batch_params.get(external_request_id)
            if params is None:
                raise RuntimeError(f"SFA PD batch metadata does not contain request {external_request_id!r}")
            # Each vLLM child request must retain only its own D-side endpoint
            # and cache state after a batched OpenAI completion is expanded.
            request.kv_transfer_params = params

        local_block_ids = self._normalize_block_ids(blocks.get_block_ids())
        remote_cache_tokens = params["remote_cached_tokens"]
        send_req_info = _SendReqInfo(
            local_block_ids=local_block_ids,
            local_transferred_tokens=remote_cache_tokens,
            local_computed_tokens=0,
            request=request,
        )
        self._reqs_need_send_layerwise[request.request_id] = send_req_info

        if envs.VLLM_ASCEND_SFA_DEBUG:
            logger.info(
                "SFAPD P register remote-decode req %s: local_block_ids=%s, "
                "remote_host=%s, remote_port=%s, remote_tp_size=%s, "
                "remote_cached_tokens=%s",
                request.request_id,
                local_block_ids,
                params.get("remote_host"),
                params.get("remote_port"),
                params.get("remote_tp_size"),
                remote_cache_tokens,
            )

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        meta = SfaPDProducerMetadata()
        cached_reqs = scheduler_output.scheduled_cached_reqs
        new_reqs = scheduler_output.scheduled_new_reqs
        scheduled_spec_decode_tokens = scheduler_output.scheduled_spec_decode_tokens

        for req_id, new_blocks in zip(cached_reqs.req_ids, cached_reqs.new_block_ids):
            if req_id in self._reqs_need_send_layerwise and new_blocks is not None:
                normalized = self._normalize_block_ids(new_blocks)
                self._reqs_need_send_layerwise[req_id].extend_local_block_ids(normalized)
                if envs.VLLM_ASCEND_SFA_DEBUG:
                    logger.info(
                        "SFAPD P extend remote-decode req %s: new_blocks=%s",
                        req_id,
                        normalized,
                    )

        computed_tokens = dict(
            list(zip(cached_reqs.req_ids, cached_reqs.num_computed_tokens))
            + [(req.req_id, req.num_computed_tokens) for req in new_reqs]
        )
        min_block_size = min(self.block_size)
        for req_id, scheduled_tokens in scheduler_output.num_scheduled_tokens.items():
            send_req_info = self._reqs_need_send_layerwise.get(req_id)
            if send_req_info is None:
                continue

            send_req_info.update_transferred_tokens(round_down(send_req_info.local_computed_tokens, min_block_size))
            spec_decode_tokens = (
                len(scheduled_spec_decode_tokens[req_id]) if req_id in scheduled_spec_decode_tokens else 0
            )
            send_req_info.update_computed_tokens(computed_tokens.get(req_id, 0) + scheduled_tokens - spec_decode_tokens)
            request = send_req_info.request
            assert request.kv_transfer_params is not None
            chunk_finish = send_req_info.local_computed_tokens >= len(request.all_token_ids)
            meta.add_new_req(
                request_id=req_id,
                local_block_ids=send_req_info.local_block_ids,
                kv_transfer_params=request.kv_transfer_params,
                token_ids=[],
                chunk_finish=chunk_finish,
                remote_cache_tokens=request.kv_transfer_params.get("remote_cached_tokens"),
                prompt_len=len(request.all_token_ids),
                local_computed_tokens=send_req_info.local_computed_tokens,
                local_transed_tokens=send_req_info.local_transferred_tokens,
            )
            if envs.VLLM_ASCEND_SFA_DEBUG:
                logger.info(
                    "SFAPD P add transfer task req %s: local_block_ids=%s, "
                    "local_transed_tokens=%s, local_computed_tokens=%s, "
                    "remote_cache_tokens=%s, prompt_len=%s, chunk_finish=%s, "
                    "remote_host=%s, remote_port=%s",
                    req_id,
                    send_req_info.local_block_ids,
                    send_req_info.local_transferred_tokens,
                    send_req_info.local_computed_tokens,
                    request.kv_transfer_params.get("remote_cached_tokens"),
                    len(request.all_token_ids),
                    chunk_finish,
                    request.kv_transfer_params.get("remote_host"),
                    request.kv_transfer_params.get("remote_port"),
                )
            if chunk_finish:
                self._reqs_need_send_layerwise.pop(req_id)
        return meta

    def request_finished(
        self,
        request: Request,
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        return self.request_finished_all_groups(request, (block_ids,))

    def request_finished_all_groups(
        self,
        request: Request,
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        # The final chunk normally removes this tracker in
        # build_connector_meta. Cancellation, preemption, and failures can
        # finish earlier, so clean it up here as well.
        self._reqs_need_send_layerwise.pop(request.request_id, None)
        return False, None


class SFAPDCpuOffloadScheduler:
    def __init__(
        self,
        vllm_config: VllmConfig,
        use_layerwise: bool,
        kv_cache_config: KVCacheConfig | None,
    ):
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        self.use_layerwise = use_layerwise
        self.engine_id = vllm_config.kv_transfer_config.engine_id

        if kv_cache_config is None:
            raise ValueError("SFAPDCpuOffloadScheduler requires KVCacheConfig")
        self.block_size = [group_spec.kv_cache_spec.block_size for group_spec in kv_cache_config.kv_cache_groups]
        self.main_group_idx, self.indexer_group_idx = infer_sfa_component_group_ids(kv_cache_config)

        self.side_channel_host = get_ip()
        self.side_channel_port = (
            vllm_config.kv_transfer_config.kv_port
            + vllm_config.parallel_config.data_parallel_rank * vllm_config.parallel_config.tensor_parallel_size
        )

        # Decode offload reuses vLLM's normal block ids.  Main block ids index
        # KVOffloadDecodeManager's CPU tensors; no connector-private block pool
        # is allocated.
        self._request_trackers: dict[str, tuple[list[int], list[int]]] = {}
        # req_ids awaiting their first build_connector_meta seed (so the worker
        # can build request_map for get_finished even while async-waiting KV).
        self._reqs_need_recv: set[str] = set()
        self.executor = ThreadPoolExecutor(32)
        self._metaserver_futures = {}
        self._metaserver_retry_timers = {}
        self._cancelled_metaserver_requests: set[str] = set()
        self._metaserver_lock = threading.Lock()
        self._shutdown_event = threading.Event()

    # ------------------------------------------------------------------
    # D side (kv_consumer)
    # ------------------------------------------------------------------
    def get_num_new_matched_tokens(self, request: Request, num_computed_tokens: int) -> tuple[int, bool]:
        # Pull the entire prompt KV from the remote P node into D's CPU pool
        # (main MLA) / HBM (indexer). Async relative to engine execution.
        params = request.kv_transfer_params
        if params is not None and params.get("do_remote_prefill"):
            assert num_computed_tokens % min(self.block_size) == 0
            count = max(len(request.prompt_token_ids) - num_computed_tokens, 0)
            return count, count > 0
        return 0, False

    def update_state_after_alloc(
        self,
        request: Request,
        blocks: KVCacheBlocks,
        num_external_tokens: int,
    ):
        params = request.kv_transfer_params
        if params is None or not params.get("do_remote_prefill"):
            return

        block_ids_by_group = SFAPDProducerScheduler._normalize_block_ids(blocks.get_block_ids())
        required_group = max(self.main_group_idx, self.indexer_group_idx)
        if len(block_ids_by_group) <= required_group:
            raise RuntimeError(
                "SFAPD D allocation did not provide all SFA KV cache groups: "
                f"required={required_group + 1}, got={len(block_ids_by_group)}"
            )
        main_block_ids = list(block_ids_by_group[self.main_group_idx])
        indexer_block_ids = list(block_ids_by_group[self.indexer_group_idx])
        self._request_trackers[request.request_id] = (main_block_ids, indexer_block_ids)
        self._reqs_need_recv.add(request.request_id)

        # Notify P via the metaserver rendezvous that D is ready to pull this
        # request. D does NOT send its block ids to P — D keeps them (passed to
        # the D worker via connector_meta) and looks
        # them up by req_id when P's READ_READY arrives. Only contact info +
        # the do_remote_decode "go" flag go to P. (Sending block ids to P was a
        # push-model leftover; in pull mode P only needs P's own source blocks.)
        kv_transfer_params = dict(
            request_id=get_external_request_id(request.request_id),
            do_remote_prefill=False,
            do_remote_decode=True,
            remote_engine_id=self.engine_id,
            remote_host=self.side_channel_host,
            remote_port=self.side_channel_port,
            remote_tp_size=self.vllm_config.parallel_config.tensor_parallel_size,
            remote_pcp_size=self.vllm_config.parallel_config.prefill_context_parallel_size,
            remote_dcp_size=self.vllm_config.parallel_config.decode_context_parallel_size,
            remote_cached_tokens=request.num_computed_tokens,
        )
        # Allocation is complete once the rendezvous request is submitted.
        # Keep the vLLM remote-prefill state independent of the legacy proxy's
        # HTTP result; old proxies return 500 for extra prompt-list children
        # even though the first child has already dispatched the whole batch.
        params["do_remote_prefill"] = False
        metaserver = params.get("metaserver")
        if metaserver is not None and not params.get("do_virtual", False):
            with self._metaserver_lock:
                self._cancelled_metaserver_requests.discard(request.request_id)
            self._submit_metaserver_request(
                request_id=request.request_id,
                url=metaserver,
                message=kv_transfer_params,
            )
        if envs.VLLM_ASCEND_SFA_DEBUG:
            logger.info(
                "SFAPDCpuOffload D advertised req %s: indexer_hbm_ids=%s, "
                "main_cpu_ids=%s, remote_host=%s, remote_port=%s, metaserver=%s",
                request.request_id,
                indexer_block_ids,
                main_block_ids,
                self.side_channel_host,
                self.side_channel_port,
                metaserver,
            )

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:

        meta = SfaPDConsumerMetadata()
        for req_id in list(self._reqs_need_recv):
            tracker = self._request_trackers.get(req_id)
            if tracker is None:
                continue
            main_block_ids, indexer_block_ids = tracker
            meta.add_request(req_id, main_block_ids, indexer_block_ids)
        self._reqs_need_recv.clear()
        return meta

    def request_finished(self, request: Request, block_ids: list[int]) -> tuple[bool, dict[str, Any] | None]:
        return self.request_finished_all_groups(request, (block_ids,))

    def request_finished_all_groups(
        self,
        request: Request,
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        # vLLM owns the block lifecycle; the connector only drops its lookup.
        self._request_trackers.pop(request.request_id, None)
        self._reqs_need_recv.discard(request.request_id)
        with self._metaserver_lock:
            self._cancelled_metaserver_requests.add(request.request_id)
            future = self._metaserver_futures.pop(request.request_id, None)
            timer = self._metaserver_retry_timers.pop(request.request_id, None)
        if future is not None:
            future.cancel()
        if timer is not None:
            timer.cancel()
        return False, None

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _access_metaserver(self, url: str, message: dict[str, Any]):
        with httpx.Client(
            limits=httpx.Limits(max_connections=100000),
            timeout=None,
        ) as client:
            retry = 0
            while retry < METASERVER_MAX_RETRIES:
                retry += 1
                try:
                    response = client.post(url, json=message)
                    if response.is_error:
                        logger.warning(
                            "Metaserver returned HTTP %d for request %s; "
                            "treating it as delivered for legacy-proxy compatibility",
                            response.status_code,
                            message.get("request_id"),
                        )
                    return
                except httpx.RequestError as error:
                    logger.error(
                        "Metaserver transport failed: url=%s, retry=%d, error=%s: %s",
                        url,
                        retry,
                        type(error).__name__,
                        error,
                    )
                    if retry == METASERVER_MAX_RETRIES:
                        raise

    def _submit_metaserver_request(
        self,
        *,
        request_id: str,
        url: str,
        message: dict[str, Any],
    ) -> None:
        if self._shutdown_event.is_set():
            return
        future = None
        with self._metaserver_lock:
            if self._shutdown_event.is_set() or request_id in self._cancelled_metaserver_requests:
                return
            self._metaserver_retry_timers.pop(request_id, None)
            if request_id not in self._metaserver_futures:
                future = self.executor.submit(
                    self._access_metaserver,
                    url=url,
                    message=message,
                )
                self._metaserver_futures[request_id] = future
        if future is not None:
            future.add_done_callback(
                partial(
                    self._on_metaserver_done,
                    request_id=request_id,
                    url=url,
                    message=message,
                )
            )

    def _on_metaserver_done(
        self,
        future,
        *,
        request_id: str,
        url: str,
        message: dict[str, Any],
    ):
        with self._metaserver_lock:
            self._metaserver_futures.pop(request_id, None)
        if future.cancelled() or self._shutdown_event.is_set():
            return
        with self._metaserver_lock:
            if request_id in self._cancelled_metaserver_requests:
                return
        error = future.exception()
        if error is not None:
            logger.error(
                "Access metaserver failed for request %s; retrying in %.1f seconds: %s",
                request_id,
                METASERVER_RETRY_DELAY_SECONDS,
                error,
            )
            timer = threading.Timer(
                METASERVER_RETRY_DELAY_SECONDS,
                self._submit_metaserver_request,
                kwargs={
                    "request_id": request_id,
                    "url": url,
                    "message": message,
                },
            )
            timer.daemon = True
            with self._metaserver_lock:
                if self._shutdown_event.is_set():
                    return
                self._metaserver_retry_timers[request_id] = timer
            timer.start()
            return

    def shutdown(self) -> None:
        self._shutdown_event.set()
        with self._metaserver_lock:
            futures = list(self._metaserver_futures.values())
            timers = list(self._metaserver_retry_timers.values())
            self._metaserver_futures.clear()
            self._metaserver_retry_timers.clear()
        for future in futures:
            future.cancel()
        for timer in timers:
            timer.cancel()
        self.executor.shutdown(wait=False, cancel_futures=True)
