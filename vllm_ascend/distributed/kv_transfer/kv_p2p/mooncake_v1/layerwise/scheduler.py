# SPDX-License-Identifier: Apache-2.0
"""MooncakeLayerwiseConnectorScheduler (moved out of the entry module).

Pure code movement: the class keeps every method verbatim and imports the
vllm/vllm_ascend modules directly, never the entry module.
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.logger import logger
from vllm.utils.math_utils import round_down
from vllm.utils.network_utils import get_ip
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.base.scheduler import (
    MooncakeConnectorSchedulerBase,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.common.block_selection import (
    has_attn_mamba_hybrid_cache,
    hybrid_prefill_token_count,
    iter_kv_cache_specs,
    trim_hybrid_remote_block_ids,
    truncate_request_for_hybrid_prefill,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.common.endpoints import layerwise_scheduler_base_port
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.common.protocol import get_external_request_id
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.layerwise.metaserver import MetaserverClient
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.layerwise.protocol import (
    MooncakeLayerwiseConnectorMetadata,
    SendReqInfo,
)

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request


class MooncakeLayerwiseConnectorScheduler(MooncakeConnectorSchedulerBase):
    """Implementation of Scheduler side methods"""

    def __init__(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig, engine_id: str):
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        self.block_size = [group_spec.kv_cache_spec.block_size for group_spec in kv_cache_config.kv_cache_groups]
        self.engine_id = engine_id
        logger.info("Initializing Mooncake Scheduler %s", engine_id)

        self.side_channel_host = get_ip()

        # disable prefill context parallel on decoder nodes
        if vllm_config.kv_transfer_config.is_kv_consumer:
            assert vllm_config.parallel_config.prefill_context_parallel_size == 1, (
                "Prefill context parallel is not support on decoder nodes"
            )

        # Handshake base port
        self.side_channel_port = layerwise_scheduler_base_port(
            kv_port=vllm_config.kv_transfer_config.kv_port,
            dp_rank=vllm_config.parallel_config.data_parallel_rank,
            tp_size=vllm_config.parallel_config.tensor_parallel_size,
        )

        # Requests that need to start recv.
        # New requests are added by update_state_after_alloc in
        # the scheduler. Used to make metadata passed to Worker.
        self._reqs_need_recv: dict[str, tuple[Request, list[int], list[list[int]]]] = {}
        self._reqs_need_send_layerwise: dict[str, SendReqInfo] = {}
        self.need_truncate = self._has_attn_mamba_hybrid_cache(kv_cache_config)
        self.executor = ThreadPoolExecutor(32)
        self.metaserver_client = MetaserverClient(vllm_config)
        self._shutdown_complete = False

    def shutdown(self) -> None:
        """Idempotent best-effort shutdown: close the metaserver client, then stop the executor."""
        if getattr(self, "_shutdown_complete", False):
            return
        self._shutdown_complete = True
        # Close first: queued tasks and later retries fail immediately instead
        # of starting new posts. An in-flight post is not interrupted by close;
        # the finite read timeout bounds it instead, so wait=True cannot block
        # forever. cancel_futures drops the queued requests.
        try:
            self.metaserver_client.close()
        except Exception as error:
            logger.warning("Failed to close layerwise metaserver client. error=%s. ", error)
        try:
            self.executor.shutdown(wait=True, cancel_futures=True)
        except Exception as error:
            logger.warning("Failed to shutdown layerwise scheduler executor. error=%s. ", error)
        logger.info("MooncakeLayerwiseConnectorScheduler shutdown complete. engine_id=%s", self.engine_id)

    @staticmethod
    def _iter_kv_cache_specs(kv_cache_config: KVCacheConfig):
        return iter_kv_cache_specs(kv_cache_config)

    @classmethod
    def _has_attn_mamba_hybrid_cache(cls, kv_cache_config: KVCacheConfig) -> bool:
        return has_attn_mamba_hybrid_cache(kv_cache_config)

    def _hybrid_prefill_token_count(self, num_prompt_tokens: int) -> int:
        return hybrid_prefill_token_count(num_prompt_tokens, need_truncate=self.need_truncate)

    def _truncate_request_for_hybrid_prefill(self, request: "Request") -> None:
        truncate_request_for_hybrid_prefill(request, need_truncate=self.need_truncate)

    def _trim_hybrid_remote_block_ids(self, block_ids: tuple[list[int], ...], prompt_len: int) -> tuple[list[int], ...]:
        return trim_hybrid_remote_block_ids(
            block_ids,
            prompt_len,
            need_truncate=self.need_truncate,
            block_sizes=self.block_size,
        )

    def get_num_new_matched_tokens(self, request: "Request", num_computed_tokens: int) -> tuple[int, bool]:
        """
        For remote prefill, pull all prompt blocks from remote
        asynchronously relative to engine execution.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request
        Returns:
            * the number of tokens that can be loaded from the
              external KV cache beyond what is already computed.
            * true if the external KV cache tokens will be loaded
              asynchronously (between scheduler steps).
        """

        params = request.kv_transfer_params
        logger.debug(
            "MooncakeLayerwiseConnector get_num_new_matched_tokens: num_computed_tokens=%s, kv_transfer_params=%s",
            num_computed_tokens,
            params,
        )

        if params is not None and params.get("do_remote_prefill"):
            # Remote prefill: get all prompt blocks from remote.
            assert num_computed_tokens % min(self.block_size) == 0
            count = max(self._hybrid_prefill_token_count(len(request.prompt_token_ids)) - num_computed_tokens, 0)
            return count, count > 0

        if params is not None and params.get("do_remote_decode"):
            self._truncate_request_for_hybrid_prefill(request)

        # No remote prefill for this request.
        return 0, False

    def update_state_after_alloc(self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int):
        params = request.kv_transfer_params
        logger.debug(
            "MooncakeLayerwiseConnector update_state_after_alloc: num_external_tokens=%s, kv_transfer_params=%s",
            num_external_tokens,
            params,
        )

        if params is not None and params.get("do_remote_prefill"):
            do_virtual = params.get("do_virtual", False)
            local_block_ids = (blocks.get_block_ids()) if num_external_tokens > 0 else []
            remote_block_ids = self._trim_hybrid_remote_block_ids(local_block_ids, len(request.prompt_token_ids))
            remote_cached_tokens = request.num_computed_tokens
            # Get unhashed blocks to pull from remote.
            logger.debug(
                "MooncakeLayerwiseConnector update_state_after_alloc: add %s to need recv queue", request.request_id
            )
            self._reqs_need_recv[request.request_id] = (
                request,
                [],  # request._all_token_ids,
                local_block_ids,
            )

            params["do_remote_prefill"] = False

            logger.info("Send request: %s to proxy metaserver: %s", request.request_id, params.get("metaserver", None))
            # All parameters here should appear in the returned dict of
            # request_finished in the scheduler side except "request_id".
            # change the format of request_id if vllm-version >= 0.14.0
            external_req_id = get_external_request_id(request.request_id)
            kv_transfer_params = dict(
                token_ids=[],
                request_id=external_req_id,
                do_remote_prefill=False,
                do_remote_decode=True,
                remote_block_ids=remote_block_ids,
                remote_block_size=self.block_size,
                remote_engine_id=self.engine_id,
                remote_host=self.side_channel_host,
                remote_port=self.side_channel_port,
                remote_tp_size=self.vllm_config.parallel_config.tensor_parallel_size,
                remote_pcp_size=self.vllm_config.parallel_config.prefill_context_parallel_size,
                remote_dcp_size=self.vllm_config.parallel_config.decode_context_parallel_size,
                remote_cached_tokens=remote_cached_tokens,
            )
            if not do_virtual:
                future = self.executor.submit(
                    self._access_metaserver, url=params.get("metaserver", None), message=kv_transfer_params
                )

                def handle_exception(future):
                    if future.exception():
                        logger.error("Access metaserver fail. error=%s. ", future.exception())

                future.add_done_callback(handle_exception)

        # Layerwise prefiller add request need send
        if params is not None and params.get("do_remote_decode"):
            local_block_ids = list(blocks.get_block_ids())
            logger.debug(
                "MooncakeLayerwiseConnector update_state_after_alloc: add %s to need send queue", request.request_id
            )
            remote_cache_tokens = params["remote_cached_tokens"]
            local_transferred_tokens = remote_cache_tokens
            local_computed_tokens = 0
            self._reqs_need_send_layerwise[request.request_id] = SendReqInfo(
                local_block_ids=local_block_ids,
                local_transferred_tokens=local_transferred_tokens,
                local_computed_tokens=local_computed_tokens,
                request=request,
            )

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        meta = MooncakeLayerwiseConnectorMetadata()

        if self.vllm_config.kv_transfer_config.is_kv_consumer:
            # Loop through scheduled reqs and convert to ReqMeta.
            for req_id, (req, token_ids, block_ids) in self._reqs_need_recv.items():
                assert req.kv_transfer_params is not None
                # For the case where there are no remote blocks to pull
                # (block_ids is empty), we don't need to schedule
                # an async read on the worker side.
                meta.add_new_req(
                    request_id=req_id,
                    local_block_ids=block_ids,
                    kv_transfer_params=req.kv_transfer_params,
                    token_ids=token_ids,
                )

            # Clear the list once workers start the transfers
            self._reqs_need_recv.clear()
        else:
            cached_reqs = scheduler_output.scheduled_cached_reqs
            new_reqs = scheduler_output.scheduled_new_reqs
            scheduled_spec_decode_tokens = scheduler_output.scheduled_spec_decode_tokens
            for req_id, new_blocks in zip(cached_reqs.req_ids, cached_reqs.new_block_ids):
                if req_id in self._reqs_need_send_layerwise and new_blocks is not None:
                    self._reqs_need_send_layerwise[req_id].extend_local_block_ids(new_blocks)
            computed_tokens = dict(
                list(zip(cached_reqs.req_ids, cached_reqs.num_computed_tokens))
                + [(x.req_id, x.num_computed_tokens) for x in new_reqs]
            )
            for req_id, scheduled_tokens in scheduler_output.num_scheduled_tokens.items():
                if req_id in self._reqs_need_send_layerwise:
                    send_req_info = self._reqs_need_send_layerwise[req_id]
                    # update local transferred tokens
                    send_req_info.update_transferred_tokens(
                        round_down(send_req_info.local_computed_tokens, min(self.block_size))
                    )
                    # update local computed tokens, not transfer spec decode tokens
                    spec_decode_tokens = (
                        len(scheduled_spec_decode_tokens[req_id]) if (req_id in scheduled_spec_decode_tokens) else 0
                    )
                    send_req_info.update_computed_tokens(
                        computed_tokens.get(req_id, 0) + scheduled_tokens - spec_decode_tokens
                    )

                    def add_transfer_task(req_id, send_req_info: SendReqInfo, chunk_finish=False):
                        (
                            local_block_ids,
                            local_transed_tokens,
                            local_computed_tokens,
                            request,
                        ) = send_req_info.unpack()
                        meta.add_new_req(
                            request_id=req_id,
                            local_block_ids=local_block_ids,
                            kv_transfer_params=request.kv_transfer_params,
                            token_ids=[],
                            chunk_finish=chunk_finish,
                            remote_cache_tokens=request.kv_transfer_params.get("remote_cached_tokens"),
                            prompt_len=len(request.all_token_ids),
                            local_computed_tokens=local_computed_tokens,
                            local_transed_tokens=local_transed_tokens,
                        )
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(
                                "MooncakeLayerwiseConnector build_connector_meta: req_id=%r prompt_len=%s "
                                "local_computed_tokens=%r local_transed_tokens=%r remote_cache_tokens=%s "
                                "chunk_finish=%r local_block_ids=%r remote_block_ids=%s",
                                req_id,
                                len(request.all_token_ids),
                                local_computed_tokens,
                                local_transed_tokens,
                                request.kv_transfer_params.get("remote_cached_tokens"),
                                chunk_finish,
                                local_block_ids,
                                request.kv_transfer_params.get("remote_block_ids"),
                            )

                    # whether chunk finish
                    chunk_finish = send_req_info.local_computed_tokens >= len(send_req_info.request.all_token_ids)

                    add_transfer_task(req_id, send_req_info, chunk_finish=chunk_finish)
                    if chunk_finish:
                        self._reqs_need_send_layerwise.pop(req_id)
        return meta

    def _access_metaserver(self, url, message):
        self.metaserver_client.access(url, message)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Once a request is finished, determine whether request blocks
        should be freed now or will be sent asynchronously and freed later.
        """
        # layer_wise push, not need delay_free_blocks
        return False, None

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Once a request is finished, determine whether request blocks
        should be freed now or will be sent asynchronously and freed later.
        """
        # layer_wise push, not need delay_free_blocks
        return False, None
