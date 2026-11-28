# SPDX-License-Identifier: Apache-2.0
"""MooncakeConnectorScheduler (moved out of the entry module).

Pure code movement: the class keeps every method verbatim and imports the
vllm/vllm_ascend modules directly, never the entry module.
"""

import math
import time
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.utils import BlockIds
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorHandshakeMetadata,
    KVConnectorMetadata,
)
from vllm.logger import logger
from vllm.utils.network_utils import get_ip
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.request import RequestStatus

from vllm_ascend.ascend_config import (
    get_ascend_config,
    init_ascend_config,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.base.scheduler import (
    MooncakeConnectorSchedulerBase,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.common.block_selection import (
    GroupTransferInfo,
    build_group_transfer_info,
    group_unique_specs,
    state_prefill_token_count,
    swa_transfer_block_ids,
    transfer_block_ids,
    truncate_request_for_prefill,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.common.endpoints import pull_base_port
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.pull.metadata import MooncakeConnectorMetadata

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request


class MooncakeConnectorScheduler(MooncakeConnectorSchedulerBase):
    """Implementation of Scheduler side methods"""

    def __init__(self, vllm_config: VllmConfig, engine_id: str, kv_cache_config: KVCacheConfig):
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        init_ascend_config(vllm_config)
        self.ascend_config = get_ascend_config()
        self.block_size = vllm_config.cache_config.block_size
        self.engine_id = engine_id
        self.local_ip = get_ip()
        logger.info("Initializing Mooncake Scheduler %s", engine_id)

        self.side_channel_host = get_ip()
        self.pcp_size = vllm_config.parallel_config.prefill_context_parallel_size
        self.dcp_size = vllm_config.parallel_config.decode_context_parallel_size
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.max_device_id = (
            vllm_config.parallel_config.tensor_parallel_size
            * vllm_config.parallel_config.data_parallel_size
            * self.pcp_size
            * vllm_config.parallel_config.pipeline_parallel_size
        )

        # Use the serving index, which survives dense-engine DP rank reset.
        # Keep scheduler metadata and worker handshake ports consistent.
        self.side_channel_port = pull_base_port(
            kv_port=vllm_config.kv_transfer_config.kv_port,
            dp_index=vllm_config.parallel_config.data_parallel_index,
            tp_size=vllm_config.parallel_config.tensor_parallel_size,
            pp_size=vllm_config.parallel_config.pipeline_parallel_size,
            pcp_size=self.pcp_size,
        )
        # Requests that need to start recv.
        # New requests are added by update_state_after_alloc in
        # the scheduler. Used to make metadata passed to Worker.
        self._reqs_need_recv: dict[str, tuple[Request, BlockIds, BlockIds, int]] = {}
        self._reqs_need_send: dict[str, float] = {}
        self._reqs_in_batch: set[str] = set()

        # master-slave meta information for cross-nodes
        self.multi_nodes_meta_mapping: dict[str, dict[str, Any]] = {}
        self.kv_cache_groups = kv_cache_config.kv_cache_groups
        self.use_compress = self._model_uses_compress()
        self.group_transfer_info = [self._get_group_transfer_info(group) for group in kv_cache_config.kv_cache_groups]
        self.need_truncate = self.use_compress or any(info.is_state_group for info in self.group_transfer_info)

    def _model_uses_compress(self) -> bool:
        hf_config = getattr(self.vllm_config.model_config, "hf_config", None)
        compress_ratios = getattr(hf_config, "compress_ratios", None)
        return isinstance(compress_ratios, (list, tuple, dict))

    def _get_group_transfer_info(self, group: Any) -> GroupTransferInfo:
        return build_group_transfer_info(group, default_block_size=self.block_size)

    def _get_group_unique_specs(self, group: Any) -> list[Any]:
        return group_unique_specs(group)

    def _get_transfer_block_ids(self, block_ids: BlockIds, prompt_len: int) -> BlockIds:
        """Return blocks that contain prompt KV, dropping MTP extra blocks.

        In aligned Mamba mode, normalize each state group to the single block
        containing the final prompt state. This prevents the receiver from
        inferring a block index from a speculative *token* count. Non-aligned
        state groups keep their existing behavior. SWA tail clipping is handled
        as a separate step after this.
        """
        return transfer_block_ids(
            block_ids,
            prompt_len,
            group_transfer_info=self.group_transfer_info,
            mamba_cache_mode=getattr(self.vllm_config.cache_config, "mamba_cache_mode", None),
            dcp_size=self.dcp_size,
        )

    def _get_swa_transfer_block_ids(self, block_ids: BlockIds) -> BlockIds:
        """Clip SWA groups to their window tail and drop placeholder block 0."""
        return swa_transfer_block_ids(block_ids, group_transfer_info=self.group_transfer_info)

    def _state_prefill_token_count(self, num_prompt_tokens: int) -> int:
        """D-side only. Returns N-1 for Mamba models since the decoder
        always recomputes the last token and must start from h(N-1)."""
        return state_prefill_token_count(num_prompt_tokens, need_truncate=self.need_truncate)

    def _truncate_request_for_prefill(self, request: "Request") -> None:
        """P-side only: drop the last prompt token so the prefiller computes
        h(N-1) instead of h(N). The decoder recomputes the last token to
        derive h(N) correctly.

        Guarded by ``_p_side_truncated`` to avoid repeated truncation if the
        request is preempted and rescheduled."""
        truncate_request_for_prefill(
            request,
            hash_block_size=self.kv_cache_config.hash_block_size,
            num_hash_lookahead_tokens=self.vllm_config.num_hash_lookahead_tokens,
        )

    def on_new_request(self, request: "Request") -> None:
        # Truncate before local prefix-cache lookup so its hit count stays
        # consistent with the prompt length used by the scheduler.
        params = request.kv_transfer_params
        if params is not None and params.get("do_remote_decode") and self.need_truncate:
            self._truncate_request_for_prefill(request)

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
            "MooncakeConnector get_num_new_matched_tokens: num_computed_tokens=%s, kv_transfer_params=%s",
            num_computed_tokens,
            params,
        )

        if params is not None and params.get("do_remote_prefill"):
            # Remote prefill: get all prompt blocks from remote.
            token_ids = request.prompt_token_ids or []
            actual = self._state_prefill_token_count(len(token_ids))
            params["num_computed_tokens"] = num_computed_tokens
            count = max(actual - num_computed_tokens, 0)
            if count > 0:
                return count, True

        # No remote prefill for this request.
        return 0, False

    def update_state_after_alloc(self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int):
        params = request.kv_transfer_params
        logger.debug(
            "MooncakeConnector update_state_after_alloc: num_external_tokens=%s, kv_transfer_params=%s",
            num_external_tokens,
            params,
        )

        if params is not None and (params.get("do_remote_prefill", False) or params.get("do_remote_decode", False)):
            self._reqs_in_batch.add(request.request_id)
        if params is not None and params.get("do_remote_prefill"):
            if params.get("remote_block_ids"):
                if all(p in params for p in ("remote_engine_id", "remote_host", "remote_port", "remote_request_id")):
                    local_block_ids = blocks.get_unhashed_block_ids_all_groups() if num_external_tokens > 0 else []
                    local_full_block_ids = blocks.get_block_ids() if num_external_tokens > 0 else tuple()
                    # Get unhashed blocks to pull from remote.
                    self._reqs_need_recv[request.request_id] = (
                        request,
                        local_block_ids,
                        local_full_block_ids,
                        num_external_tokens,
                    )
                else:
                    logger.warning("Got invalid KVTransferParams. params=%s. ", params)
            else:
                assert num_external_tokens == 0
            # Only trigger 1 KV transfer per request.
            params["do_remote_prefill"] = False

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        meta = MooncakeConnectorMetadata()

        # Loop through scheduled reqs and convert to ReqMeta.
        for req_id, (req, block_ids, full_block_ids, num_external_tokens) in self._reqs_need_recv.items():
            assert req.kv_transfer_params is not None
            # For the case where there are no remote blocks to pull
            # (block_ids is empty), we don't need to schedule
            # an async read on the worker side.
            meta.add_new_req(
                request_id=req_id,
                local_block_ids=block_ids,
                local_full_block_ids=full_block_ids,
                num_external_tokens=num_external_tokens,
                kv_transfer_params=req.kv_transfer_params,
            )

        # Clear the list once workers start the transfers
        self._reqs_need_recv.clear()
        meta.requests_to_send = self._reqs_need_send
        self._reqs_need_send = {}
        meta.reqs_in_batch = self._reqs_in_batch
        self._reqs_in_batch = set()

        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: BlockIds,
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Once a request is finished, determine whether request blocks
        should be freed now or will be sent asynchronously and freed later.
        """

        params = request.kv_transfer_params
        logger.debug(
            "MooncakeConnector request_finished, request_status=%s, kv_transfer_params=%s", request.status, params
        )

        if params is None:
            return False, None

        # A remote-prefill request can be rejected before scheduler admission
        # (for example, when prompt + max_tokens exceeds max_model_len). In
        # that case update_state_after_alloc() never gets a chance to schedule
        # the receive, so explicitly enqueue an empty receive. The worker skips
        # the data transfer for empty block IDs but still sends the completion
        # signal to the P node, allowing it to release the stranded KV blocks.
        if params.get("do_remote_prefill") or request.request_id in self._reqs_need_recv:
            empty_block_ids: BlockIds = tuple([] for _ in self.kv_cache_groups)
            self._reqs_need_recv[request.request_id] = (
                request,
                empty_block_ids,
                empty_block_ids,
                0,
            )
            params["do_remote_prefill"] = False
            return False, None

        if not params.get("do_remote_decode") or request.status != RequestStatus.FINISHED_LENGTH_CAPPED:
            return False, None

        num_prompt_blocks = math.ceil(len(request.prompt_token_ids) / self.block_size)
        computed_block_ids = self._get_transfer_block_ids(block_ids, len(request.prompt_token_ids))
        computed_block_ids = self._get_swa_transfer_block_ids(computed_block_ids)
        computed_block_lens = [len(block_id_list) for block_id_list in computed_block_ids]
        delay_free_blocks = sum(computed_block_lens) > 0
        if delay_free_blocks:
            logger.info("Delaying free of %d blocks for request %s", sum(computed_block_lens), request.request_id)
            self._reqs_need_send[request.request_id] = time.time()

        return delay_free_blocks, dict(
            do_remote_prefill=True,
            do_remote_decode=False,
            remote_block_ids=computed_block_ids,
            remote_engine_id=self.engine_id,
            remote_request_id=request.request_id,
            remote_host=self.side_channel_host,
            remote_port=self.side_channel_port,
            remote_pcp_size=self.pcp_size,
            remote_dcp_size=self.dcp_size,
            remote_ptp_size=self.tp_size,
            last_token_id=request.output_token_ids[-1],
            remote_multi_nodes_meta_mapping=self.multi_nodes_meta_mapping,
            num_prompt_blocks=num_prompt_blocks,
            remote_block_size=self.block_size,
        )

    def _port_offset_from_handshake_metadata(
        self,
        rank_metadata: KVConnectorHandshakeMetadata,
        metadata_key: int | tuple[int, ...],
    ) -> int:
        kv_port = self.vllm_config.kv_transfer_config.kv_port
        handshake_port = getattr(rank_metadata, "handshake_port", 0)
        if handshake_port > 0:
            return handshake_port - kv_port
        if isinstance(metadata_key, int):
            return metadata_key
        raise ValueError(f"Mooncake handshake metadata missing handshake_port for worker key {metadata_key}")

    def set_xfer_handshake_metadata_from_workers(
        self,
        metadata: Mapping[int | tuple[int, ...], KVConnectorHandshakeMetadata],
    ) -> None:
        """Build host mapping for one DP group that may span multiple nodes."""
        if not metadata:
            return

        updated_mapping: dict[str, dict[str, Any]] = {}
        kv_port = self.vllm_config.kv_transfer_config.kv_port
        for metadata_key, rank_metadata in metadata.items():
            port_offset = self._port_offset_from_handshake_metadata(rank_metadata, metadata_key)
            updated_mapping[str(port_offset)] = {
                "host": rank_metadata.local_ip,
                "engine_id": rank_metadata.engine_id,
                "handshake_port": kv_port + port_offset,
            }

        self.multi_nodes_meta_mapping.update(updated_mapping)
        logger.info(
            "MooncakeConnector set_xfer_handshake_metadata: worker_count=%d, updated=%s, multi_nodes_meta_mapping=%s",
            len(metadata),
            updated_mapping,
            self.multi_nodes_meta_mapping,
        )

    def set_xfer_handshake_metadata(
        self, metadata: Mapping[int | tuple[int, ...], KVConnectorHandshakeMetadata]
    ) -> None:
        """Legacy int-keyed entry point (port offset keys)."""
        self.set_xfer_handshake_metadata_from_workers(metadata)
