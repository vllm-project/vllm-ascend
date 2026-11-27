# mypy: ignore-errors
# SPDX-License-Identifier: Apache-2.0
"""Data-plane threads of the layerwise mooncake V1 connector.

The classes used to live inside mooncake_layerwise_connector. They are moved
verbatim so the entry module can stay a compatibility re-export layer. The
entry module keeps importing these names, so class identity and the existing
test patches on the entry module stay valid for the code that remains there.
"""

import math
import queue
import threading
import time
from collections.abc import Callable

import msgspec
import numpy as np
import numpy.typing as npt
import torch
import zmq
from mooncake.engine import TransferEngine  # type: ignore
from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_world_group
from vllm.logger import logger
from vllm.utils.network_utils import get_ip, make_zmq_path
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec, MambaSpec

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.common.constants import (
    THREAD_STOP_POLL_SECONDS,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.common.endpoints import (
    layerwise_handshake_port,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.common.protocol import (
    DONE_SENDING_MSG,
    FAILED_SENDING_MSG,
    GET_META_MSG,
    LayerMetadata,
    LayerwiseAgentMetadata,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.common.transport import zmq_ctx
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.layerwise.protocol import (
    ReqMeta,
    SendTask,
    TransferMeta,
)
from vllm_ascend.utils import npu_stream_switch

MooncakeAgentMetadata = LayerwiseAgentMetadata


class KVCacheSendingLayerThread(threading.Thread):
    def __init__(
        self,
        engine: TransferEngine,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        kv_cache_specs: list[KVCacheSpec],
        attn_resharding_group_idx: set,
        total_layers: int,
        ready_event: threading.Event,
        tp_size: int,
        tp_rank: int,
        pd_head_ratio: int,
        num_head_replica: int,
        layer_metadata: dict[str, LayerMetadata],
        use_mla: bool,
        use_attn_mamba_hybrid: bool,
        k_buffer: torch.Tensor,
        v_buffer: torch.Tensor,
        enable_kv_quant: bool,
        enable_c8_quant: bool,
        resharding_stream: torch.npu.Stream,
        callback_func: Callable[..., None] = lambda x: None,
        reuse_completion_callback: Callable[[int, str | None], None] | None = None,
    ):
        super().__init__(daemon=True, name="KVCacheSendingLayerThread")
        self.engine = engine
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        self.kv_cache_specs = kv_cache_specs
        self.attn_resharding_group_idx = attn_resharding_group_idx
        self.mamba_cache_mode = self.vllm_config.cache_config.mamba_cache_mode
        self.num_speculative_tokens = (
            self.vllm_config.speculative_config.num_speculative_tokens
            if self.vllm_config.speculative_config is not None
            else 0
        )
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.pd_head_ratio = pd_head_ratio
        self.num_head_replica = num_head_replica
        self.layer_metadata = layer_metadata
        self.total_layers = total_layers
        self.use_mla = use_mla
        self.use_attn_mamba_hybrid = use_attn_mamba_hybrid
        self.resharding_stream = resharding_stream
        self.current_layer = -1

        send_queue_size = 0
        if self.pd_head_ratio != 1:
            if self.use_attn_mamba_hybrid:
                send_queue_size = len(self.kv_cache_specs)
            else:
                send_queue_size = 1
        self.send_queue = queue.Queue[SendTask](maxsize=send_queue_size)
        self.failed_reqs: set[str] = set()
        self.k_buffer = k_buffer
        self.v_buffer = v_buffer
        self.enable_kv_quant = enable_kv_quant
        self.enable_c8_quant = enable_c8_quant
        self.ready_event = ready_event
        self.callback_func = callback_func
        self.reuse_completion_callback = reuse_completion_callback
        # Set by stop() to let the send loop exit; see run().
        self._stop_event = threading.Event()
        # First fatal (unrecoverable) error raised by run(); read by the worker.
        self.fatal_error: BaseException | None = None

    def stop(self) -> None:
        """Signal the send loop to exit; idempotent."""
        self._stop_event.set()

    def run(self):
        local_rank = get_world_group().local_rank
        device = torch.device(f"npu:{local_rank}")
        torch.npu.set_device(device)
        self.ready_event.set()
        try:
            while not self._stop_event.is_set():
                try:
                    send_task = self.send_queue.get(timeout=THREAD_STOP_POLL_SECONDS)
                except queue.Empty:
                    continue
                self._handle_request(send_task)
        except Exception as error:
            self.fatal_error = error
            logger.exception("Mooncake KVCacheSendingLayerThread terminated unexpectedly. error=%s. ", error)
            raise

    def _handle_request(self, send_task: SendTask):
        error: str | None = None
        try:
            error = self._transfer_kv_cache(send_task)
        except Exception as e:
            error = str(e)
            self.failed_reqs.update(send_task.send_request)
        finally:
            if error is not None:
                logger.error(
                    "Failed to transfer KV cache. layer_idx=%s, error=%s. Check transfer engine and memory state.",
                    send_task.layer_idx,
                    error,
                )
            self.failed_reqs.update(send_task.failed_requests)
            if send_task.layer_idx == (self.total_layers - 1):
                completion_requests = {
                    **send_task.send_request,
                    **send_task.failed_requests,
                }
                if completion_requests:
                    primary_group_idx = self.layer_metadata[send_task.layer_name].tensor_group_idx[0]
                    for req_id, req_meta in completion_requests.items():
                        if not req_meta.chunk_finish:
                            continue
                        transfer_succeeded = req_id not in self.failed_reqs
                        self.callback_func(
                            req_id,
                            req_meta,
                            primary_group_idx,
                            trans_flag=transfer_succeeded,
                        )
                        if not transfer_succeeded:
                            self.failed_reqs.discard(req_id)
            if self.reuse_completion_callback is not None:
                self.reuse_completion_callback(send_task.layer_idx, error)
            self.send_queue.task_done()

    def get_transfer_meta(
        self,
        send_task: SendTask,
        req_id: str,
        req_meta: ReqMeta,
        layer_name: str,
        layer_group_idx: int,
    ):
        src_list: list[int] = []
        dst_list: list[int] = []
        length_list: list[int] = []

        layer_kv_cache_spec = self.kv_cache_specs[layer_group_idx]
        remote_block_ids = req_meta.remote_block_ids[layer_group_idx]
        remote_layer_metadata = req_meta.remote_layer_metadata[layer_name]
        local_layer_metadata = self.layer_metadata[layer_name]
        local_block_ids = req_meta.local_block_ids[layer_group_idx]

        if isinstance(layer_kv_cache_spec, MambaSpec):
            # only support one block transfer for mamba
            if self.mamba_cache_mode == "align":
                local_transfer_idx = len(local_block_ids) - self.num_speculative_tokens - 1
            else:
                local_transfer_idx = 0
            remote_transfer_idx = len(remote_block_ids) - self.num_speculative_tokens - 1
            local_conv_addr, local_ssm_addr = local_layer_metadata.kv_caches_base_addr
            remote_conv_addr, remote_ssm_addr = remote_layer_metadata.kv_caches_base_addr
            local_conv_len, local_ssm_len = local_layer_metadata.block_len
            tp_ratio = self.tp_size // req_meta.remote_tp_size
            if tp_ratio == 1:
                src_list.extend(
                    [
                        local_conv_addr + local_block_ids[local_transfer_idx] * local_conv_len,
                        local_ssm_addr + local_block_ids[local_transfer_idx] * local_ssm_len,
                    ]
                )
                dst_list.extend(
                    [
                        remote_conv_addr + remote_block_ids[remote_transfer_idx] * local_conv_len,
                        remote_ssm_addr + remote_block_ids[remote_transfer_idx] * local_ssm_len,
                    ]
                )
                length_list.extend([local_conv_len, local_ssm_len])
            else:
                conv_shape, ssm_shape = layer_kv_cache_spec.shapes
                conv_dtype, ssm_dtype = layer_kv_cache_spec.dtypes
                remote_conv_len, remote_ssm_len = remote_layer_metadata.block_len
                # conv
                linear_key_head_dim = self.vllm_config.model_config.hf_text_config.linear_key_head_dim
                linear_num_key_heads = self.vllm_config.model_config.hf_text_config.linear_num_key_heads
                linear_value_head_dim = self.vllm_config.model_config.hf_text_config.linear_value_head_dim
                linear_num_value_heads = self.vllm_config.model_config.hf_text_config.linear_num_value_heads
                local_num_key_heads = linear_num_key_heads // self.tp_size
                local_num_value_heads = linear_num_value_heads // self.tp_size
                local_conv_offsets = [
                    0,
                    local_num_key_heads * linear_key_head_dim,
                    local_num_key_heads * 2 * linear_key_head_dim,
                ]
                local_conv_sizes = [
                    local_num_key_heads * linear_key_head_dim,
                    local_num_key_heads * linear_key_head_dim,
                    local_num_value_heads * linear_value_head_dim,
                ]
                for i in range(conv_shape[0]):
                    for local_conv_offset, local_conv_size in zip(local_conv_offsets, local_conv_sizes):
                        local_addr_offset = (i * conv_shape[1] + local_conv_offset) * get_dtype_size(conv_dtype)
                        remote_addr_offset = (
                            (i * conv_shape[1] + local_conv_offset) * tp_ratio
                            + (self.tp_rank % tp_ratio) * local_conv_size
                        ) * get_dtype_size(conv_dtype)
                        src_list.append(
                            local_conv_addr + local_block_ids[local_transfer_idx] * local_conv_len + local_addr_offset
                        )
                        dst_list.append(
                            remote_conv_addr
                            + remote_block_ids[remote_transfer_idx] * remote_conv_len
                            + remote_addr_offset
                        )
                        length_list.append(local_conv_size * get_dtype_size(conv_dtype))
                # ssm
                remote_addr_offset = (self.tp_rank % tp_ratio) * math.prod(ssm_shape) * get_dtype_size(ssm_dtype)
                src_list.append(local_ssm_addr + local_block_ids[local_transfer_idx] * local_ssm_len)
                dst_list.append(
                    remote_ssm_addr + remote_block_ids[remote_transfer_idx] * remote_ssm_len + remote_addr_offset
                )
                length_list.append(local_ssm_len)
        else:
            use_staging_buffer = layer_name == send_task.layer_name and (
                send_task.k_cache is not None or send_task.k_quant_cache is not None
            )
            if self.pd_head_ratio == 1 or not use_staging_buffer:
                layer_local_kv_base_addr = local_layer_metadata.kv_caches_base_addr
                layer_remote_kv_base_addr = remote_layer_metadata.kv_caches_base_addr
                block_lens = local_layer_metadata.block_len
                grouped_remote_block_ids, grouped_local_block_ids = group_contiguous_ids(
                    remote_block_ids, local_block_ids
                )
                # kv cache quantization scenario
                if (
                    use_staging_buffer
                    and (self.enable_kv_quant or self.enable_c8_quant)
                    and send_task.k_quant_cache is not None
                ):
                    assert len(block_lens) == 2, "Quantization block length must be 2!"
                    if self.enable_kv_quant:
                        quant_block_lens = [block_lens[0] // 2, block_lens[1]]
                    else:
                        quant_block_lens = [block_lens[0] // 2, block_lens[1] // 2]
                    layer_local_quant_kv_addr = [self.k_buffer.data_ptr(), self.v_buffer.data_ptr()]
                    rearrange_block_ids = send_task.group_rearrange_block_ids[layer_group_idx]
                    # eg:[5,6,7,9] -> {5:0, 6:1, 7:2, 9:3}
                    rearrange_block_dict = {
                        value: index
                        for index, value in enumerate(rearrange_block_ids)  # type:ignore
                    }
                    for block_len, src_layer_base_addr, dst_layer_base_addr in zip(
                        quant_block_lens, layer_local_quant_kv_addr, layer_remote_kv_base_addr
                    ):
                        for group_remote_block_id, group_local_block_id in zip(
                            grouped_remote_block_ids, grouped_local_block_ids
                        ):
                            src = src_layer_base_addr + rearrange_block_dict[group_local_block_id[0]] * block_len
                            dst = dst_layer_base_addr + group_remote_block_id[0] * block_len
                            length = len(group_local_block_id) * block_len
                            src_list.append(src)
                            dst_list.append(dst)
                            length_list.append(length)
                else:
                    for k, (src_layer_base_addr, dst_layer_base_addr) in enumerate(
                        zip(layer_local_kv_base_addr, layer_remote_kv_base_addr)
                    ):
                        block_len = block_lens[k]
                        for group_remote_block_id, group_local_block_id in zip(
                            grouped_remote_block_ids, grouped_local_block_ids
                        ):
                            src = src_layer_base_addr + group_local_block_id[0] * block_len
                            dst = dst_layer_base_addr + group_remote_block_id[0] * block_len
                            length = len(group_local_block_id) * block_len
                            src_list.append(src)
                            dst_list.append(dst)
                            length_list.append(length)
            else:
                rearrange_block_ids = send_task.group_rearrange_block_ids[layer_group_idx]
                rearrange_block_dict = {
                    value: index
                    for index, value in enumerate(rearrange_block_ids)  # type:ignore
                }
                layer_local_kv_base_addr = [self.k_buffer.data_ptr(), self.v_buffer.data_ptr()]
                layer_remote_kv_base_addr = remote_layer_metadata.kv_caches_base_addr
                block_lens = local_layer_metadata.block_len
                remote_block_lens = remote_layer_metadata.block_len
                assert len(layer_remote_kv_base_addr) == 2, (
                    "Layer kv_cache resharding only supports two kv cache tensors."
                )
                src_list, dst_list, length_list = [], [], []
                for k, (src_layer_base_addr, dst_layer_base_addr) in enumerate(
                    zip(layer_local_kv_base_addr, layer_remote_kv_base_addr)
                ):
                    block_len = block_lens[k]
                    if self.enable_c8_quant:
                        block_len = block_len // 2
                    remote_block_len = remote_block_lens[k]
                    for remote_block_id, local_block_id in zip(remote_block_ids, local_block_ids):
                        src = src_layer_base_addr + rearrange_block_dict[local_block_id] * block_len
                        dst = (
                            dst_layer_base_addr
                            + remote_block_id * remote_block_len
                            + block_len * ((self.tp_rank // self.num_head_replica) % self.pd_head_ratio)
                        )
                        src_list.append(src)
                        dst_list.append(dst)
                        length_list.append(block_len)
        return (src_list, dst_list, length_list)

    def _transfer_kv_cache(self, send_task: SendTask) -> str | None:
        key = send_task.k_cache
        value = send_task.v_cache
        if self.pd_head_ratio > 1 and key is not None and value is not None:
            with npu_stream_switch(self.resharding_stream):
                key = key.view(-1, key.shape[-1])  # type:ignore
                value = value.view(-1, key.shape[-1])  # type:ignore
                self.k_buffer[: key.shape[0]].copy_(key)  # [:4, 128] ->
                self.v_buffer[: value.shape[0]].copy_(value)
        if send_task.k_quant_cache is not None:
            with npu_stream_switch(self.resharding_stream):
                key_quant = send_task.k_quant_cache
                key_quant = key_quant.view(-1, key_quant.shape[-1])  # type:ignore
                self.k_buffer[: key_quant.shape[0]].copy_(key_quant)
                value_quant = send_task.v_quant_cache
                value_quant = value_quant.view(-1, value_quant.shape[-1])  # type:ignore
                self.v_buffer[: value_quant.shape[0]].copy_(value_quant)

        # Merge transmission tasks of the same session
        session_meta: dict[str, TransferMeta] = {}
        layer_names = send_task.layer_names or [send_task.layer_name]
        layer_group_idx = self.layer_metadata[send_task.layer_name].tensor_group_idx[0]
        for req_id, req_meta in send_task.send_request.items():
            session_id = f"{req_meta.remote_host}:{req_meta.remote_te_rpc_port}"
            if session_id not in session_meta:
                session_meta[session_id] = TransferMeta(src=[], dst=[], length=[], req_ids=[])

            for layer_name in layer_names:
                src_list, dst_list, length_list = self.get_transfer_meta(
                    send_task,
                    req_id,
                    req_meta,
                    layer_name,
                    layer_group_idx,
                )
                session_meta[session_id].src.extend(src_list)
                session_meta[session_id].dst.extend(dst_list)
                session_meta[session_id].length.extend(length_list)
            session_meta[session_id].req_ids.append(req_id)

        if send_task.k_quant_cache is not None:
            self.resharding_stream.synchronize()
        elif send_task.k_cache is None or self.pd_head_ratio == 1:
            """
            Note: Due to a bug in ADXL, calling current_event.synchronize() may occasionally hang.
            This issue will be fixed in CANN version 8.5.rc1.
            You can manually build the master branch of the project at https://gitcode.com/cann/hixl
            to resolve this issue before the 8.5.RC1 release.
            """
            send_task.wait_event.synchronize()  # type:ignore
        elif self.pd_head_ratio > 1:
            self.resharding_stream.synchronize()

        transfer_errors: list[str] = []
        for session_id, transfer_meta in session_meta.items():
            if len(transfer_meta.src) > 0:
                req_start_time = time.perf_counter()
                ret = self.engine.batch_transfer_sync_write(
                    session_id, transfer_meta.src, transfer_meta.dst, transfer_meta.length
                )
                if ret < 0:
                    self.failed_reqs.update(transfer_meta.req_ids)
                    transfer_errors.append(
                        "Mooncake transfer failed for send requests. "
                        f"req_ids={transfer_meta.req_ids}, destination={session_id}, ret={ret}"
                    )
                else:
                    req_end_time = time.perf_counter()
                    total_transfer_size = sum(transfer_meta.length) / 1024
                    req_transfer_elapsed = (req_end_time - req_start_time) * 1000
                    logger.debug(
                        "Layer%d KV cache transfer task %dKB to remote_session_id [%s] took %.3f ms.",
                        send_task.layer_idx,
                        total_transfer_size,
                        session_id,
                        req_transfer_elapsed,
                    )
        return "; ".join(transfer_errors) if transfer_errors else None


class KVCacheRecvingLayerThread(threading.Thread):
    def __init__(
        self,
        tp_rank: int,
        side_channel_port: int,
        tp_size: int,
        pd_head_ratio: int,
        local_engine_id: str,
        metadata: MooncakeAgentMetadata,
        ready_event: threading.Event,
    ):
        super().__init__(daemon=True, name="KVCacheRecvingLayerThread")
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.pd_head_ratio = pd_head_ratio
        self.local_engine_id = local_engine_id
        self.side_channel_host = get_ip()
        self.side_channel_port = side_channel_port
        self.lock = threading.Lock()
        self.done_requests = set[str]()
        self.failed_requests = set[str]()
        self.task_tracker = dict[str, int]()
        self.ready_event = ready_event
        self.metadata = metadata
        # Set by stop() to let the recv loop exit; see run().
        self._stop_event = threading.Event()
        # First fatal (unrecoverable) error raised by run(); read by the worker.
        self.fatal_error: BaseException | None = None

    def get_and_clear_done_requests(self) -> set[str]:
        """
        Get and clear the requests that have been completed.
        Returns:
            A set of request IDs that have been completed.
        """
        with self.lock:
            finished_requests = self.done_requests
            self.done_requests = set()
        return finished_requests

    def get_and_clear_failed_requests(self) -> set[str]:
        """
        Get and clear the requests that have failed.
        Returns:
            A set of request IDs that have failed.
        """
        with self.lock:
            failed_requests = self.failed_requests
            self.failed_requests = set()
        return failed_requests

    def update_failed_task(self, req_id: str) -> None:
        """
        Handle a failed task by adding it to the failed_requests set and removing it from the task tracker.
        Args:
            req_id: The ID of the request that has failed.
        """
        with self.lock:
            if req_id not in self.task_tracker:
                self.task_tracker[req_id] = set()
            self.task_tracker.pop(req_id, None)
            self.failed_requests.add(req_id)

    def update_done_task(self, req_id, trans_count, side_channel_path):
        """
        Handle a completed task by adding it to the done_requests set and removing it from the task tracker.
        Args:
            req_id: The ID of the request that has completed.
        """
        with self.lock:
            if req_id not in self.task_tracker:
                self.task_tracker[req_id] = set()
            self.task_tracker[req_id].add(side_channel_path)
            if len(self.task_tracker[req_id]) == trans_count:
                self.task_tracker.pop(req_id)
                self.done_requests.add(req_id)

    def stop(self) -> None:
        """Signal the recv loop to exit; idempotent."""
        self._stop_event.set()

    def run(self):
        """Run the thread to handle KV cache transfer requests."""
        try:
            handshake_port = layerwise_handshake_port(base_port=self.side_channel_port, tp_rank=self.tp_rank)
            path = make_zmq_path("tcp", self.side_channel_host, handshake_port)
            logger.info("KVCacheRecvingLayerThread listening on %s, tp_rank=%d", path, self.tp_rank)
            encoder = msgspec.msgpack.Encoder()
            encoded_data = encoder.encode(self.metadata)
            with zmq_ctx(zmq.ROUTER, path) as sock:  # type: ignore
                self.ready_event.set()
                decoder = msgspec.msgpack.Decoder(type=tuple)
                while not self._stop_event.is_set():
                    try:
                        if sock.poll(int(THREAD_STOP_POLL_SECONDS * 1000)) == 0:
                            continue
                        frames = sock.recv_multipart()
                    except Exception as error:
                        # Only the transport layer is fatal: the ROUTER socket can
                        # no longer be used, so exit instead of spinning on it.
                        self.fatal_error = error
                        logger.exception(
                            "Mooncake KVCacheRecvingLayerThread transport failure. tp_rank=%d. error=%s. ",
                            self.tp_rank,
                            error,
                        )
                        return
                    response = None
                    try:
                        if len(frames) < 2:
                            logger.error(
                                "Invalid message format. expected>=2 frames, got %d. frames=%s", len(frames), frames
                            )
                            continue

                        identity = frames[0]
                        payload = [f for f in frames[1:] if f != b""]
                        if len(payload) != 1:
                            logger.error("Invalid payload count. expected=1, got %d. frames=%s", len(payload), frames)
                            continue

                        msg = decoder.decode(payload[0])
                        if msg[0] == GET_META_MSG:
                            logger.info("Got GET META INFO for request %s", msg[0])
                            response = (identity, b"", encoded_data)
                        elif msg[0] == DONE_SENDING_MSG:
                            logger.debug("Got DONE_RECVING_MSG for request %s", msg[1])
                            request_id = msg[1]
                            trans_count = msg[2]
                            side_channel_path = msg[3]
                            self.update_done_task(request_id, trans_count, side_channel_path)
                            response = (identity, b"", b"ACK")
                        elif msg[0] == FAILED_SENDING_MSG:
                            request_id = msg[1]
                            logger.error("Got FAILED_SENDING_MSG for request. request_id=%s. ", msg[1])
                            self.update_failed_task(request_id)
                            response = (identity, b"", b"ACK")
                        else:
                            logger.error(
                                "Unexpected message type: %s. expected GET_META_MSG or DONE_RECVING_MSG. msg=%s",
                                msg[0] if msg else "empty",
                                msg,
                            )
                    except Exception as e:
                        # Decode/format/protocol errors are recoverable: the
                        # next message on the same socket can still be handled.
                        logger.error(
                            "Failed to decode message. type=%s, error=%s. context=decoding payload",
                            type(e).__name__,
                            e,
                        )
                    if response is None:
                        continue
                    try:
                        sock.send_multipart(response)
                    except Exception as error:
                        # The response could not be written to the transport: the
                        # peer can no longer be answered, so stop the thread.
                        self.fatal_error = error
                        logger.exception(
                            "Mooncake KVCacheRecvingLayerThread failed to send response. tp_rank=%d. error=%s. ",
                            self.tp_rank,
                            error,
                        )
                        return
        except Exception as error:
            # Socket setup/teardown failures are fatal for this thread; record
            # and exit quietly so the worker reports it during shutdown.
            self.fatal_error = error
            logger.exception("Mooncake KVCacheRecvingLayerThread terminated unexpectedly. error=%s. ", error)


def group_contiguous_ids(
    src: list[int], dst: list[int] | None = None
) -> tuple[list[npt.NDArray[np.int64]], list[npt.NDArray[np.int64]]]:
    """Vectorised NumPy implementation."""
    if dst is None:
        dst = []
    if not dst:
        src_only_indices: npt.NDArray[np.int64] = np.array(src, dtype=np.int64)

        if src_only_indices.size == 0:
            return [], []

        brk = np.where(np.diff(src_only_indices) != 1)[0] + 1
        src_groups = np.split(src_only_indices, brk)
        src_groups = [g.tolist() for g in src_groups]

        return src_groups, []

    else:
        src_indices: npt.NDArray[np.int64] = np.array(src, dtype=np.int64)
        dst_indices: npt.NDArray[np.int64] = np.array(dst, dtype=np.int64)

        if src_indices.size == 0:
            return [], []

        brk = np.where((np.diff(src_indices) != 1) | (np.diff(dst_indices) != 1))[0] + 1
        src_groups = np.split(src_indices, brk)
        dst_groups = np.split(dst_indices, brk)

        src_groups = [g.tolist() for g in src_groups]
        dst_groups = [g.tolist() for g in dst_groups]

        return src_groups, dst_groups


# Back-compat alias for the historical name. This layerwise helper is not the
# same as ``mooncake_v1.common.transport.group_concurrent_contiguous``: it takes
# an optional destination list, groups on id adjacency only (no block stride or
# block_len), and returns plain lists, so it is deliberately kept local.
group_concurrent_contiguous = group_contiguous_ids
