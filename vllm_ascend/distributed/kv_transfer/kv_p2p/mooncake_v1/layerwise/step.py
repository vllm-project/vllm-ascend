# SPDX-License-Identifier: Apache-2.0
"""Layerwise step controller for the Mooncake layerwise connector.

The controller owns the per-forward layerwise step state and the step
arithmetic that used to live inside MooncakeLayerwiseConnectorWorker. It is
constructed with explicit dependencies and never holds or imports the worker
module. prepare() returns a pure LayerwiseTransferPlan; apply() reproduces
the legacy metadata write-back verbatim so this extraction stays zero-behaviour
until a later change consumes the plan directly.
"""

import copy
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch_npu
from vllm.logger import logger
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec, SlidingWindowSpec

from vllm_ascend.distributed.kv_transfer.utils.utils import kv_alltoall_and_rearrange
from vllm_ascend.utils import npu_stream_switch


@dataclass
class LayerwiseSendTaskUpdate:
    group_rearrange_block_ids: list[list[int]]
    group_num_blocks: list[int]
    group_num_tokens: list[int]
    group_block_table: list[Any]
    group_block_len_tensor: list[Any]
    group_seq_start_tensor: list[Any]


@dataclass
class LayerwiseTransferPlan:
    producer: bool
    requests: dict[str, Any]
    aligned_remote_block_ids: list[tuple[Any, dict[int, list[int]]]]
    send_task_update: LayerwiseSendTaskUpdate | None = None


class LayerwiseStepController:
    def __init__(
        self,
        *,
        vllm_config: Any,
        kv_cache_specs: Any,
        num_kv_cache_groups: int,
        block_size: Any,
        kernel_block_size_scale: Any,
        attn_resharding_group_idx: Any,
        total_layers: int,
        index_to_name: Any,
        layer_metadata: Any,
        sparse_sfa_c8_layers: Any,
        pd_head_ratio: Any,
        enable_c8_quant: bool,
        enable_kv_quant: bool,
        k_buffer: Any,
        v_buffer: Any,
        kv_send_layer_thread: Any,
        resharding_stream: Any,
        current_layer: int,
        cache_write_events: Any,
        layer_storage_slots: Any,
        storage_send_done_events: Any,
        pending_reuse_layers: Any,
        storage_send_errors: Any,
        align_remote_block_ids: Callable[..., Any],
        get_kv_split_metadata: Callable[..., Any],
        get_kv_split_metadata_for_mamba: Callable[..., Any],
        get_kernel_block_ids: Callable[..., Any],
        update_decoder_info: Callable[..., Any],
        get_nz_cache: Callable[..., Any],
        mark_layer_reuse_pending: Callable[..., Any],
        complete_layer_reuse: Callable[..., Any],
        send_task_factory: Callable[..., Any],
    ) -> None:
        self.vllm_config = vllm_config
        self.kv_cache_specs = kv_cache_specs
        self.num_kv_cache_groups = num_kv_cache_groups
        self.block_size = block_size
        self.kernel_block_size_scale = kernel_block_size_scale
        self.attn_resharding_group_idx = attn_resharding_group_idx
        self.total_layers = total_layers
        self.index_to_name = index_to_name
        self.layer_metadata = layer_metadata
        self.sparse_sfa_c8_layers = sparse_sfa_c8_layers
        self.pd_head_ratio = pd_head_ratio
        self.enable_c8_quant = enable_c8_quant
        self.enable_kv_quant = enable_kv_quant
        self.k_buffer = k_buffer
        self.v_buffer = v_buffer
        self.kv_send_layer_thread = kv_send_layer_thread
        self.resharding_stream = resharding_stream
        self.current_layer = current_layer
        self._cache_write_events = cache_write_events
        self.layer_storage_slots = layer_storage_slots
        self.storage_send_done_events = storage_send_done_events
        self._pending_reuse_layers = pending_reuse_layers
        self._storage_send_errors = storage_send_errors
        self._align_remote_block_ids = align_remote_block_ids
        self._get_kv_split_metadata = get_kv_split_metadata
        self._get_kv_split_metadata_for_mamba = get_kv_split_metadata_for_mamba
        self._get_kernel_block_ids = get_kernel_block_ids
        self.update_decoder_info = update_decoder_info
        self.get_nz_cache = get_nz_cache
        self._mark_layer_reuse_pending = mark_layer_reuse_pending
        self._complete_layer_reuse = complete_layer_reuse
        self.send_task_factory = send_task_factory

    def prepare(self, metadata: Any) -> LayerwiseTransferPlan:
        """Compute the layerwise step plan without mutating the input metadata."""
        self.current_layer = 0
        self._cache_write_events = [None] * self.total_layers
        producer = bool(self.vllm_config.kv_transfer_config.is_kv_producer)
        request_plan: dict[str, Any] = {}
        aligned_ops: list[tuple[Any, dict[int, list[int]]]] = []
        send_task_update = None
        if producer:
            update_metadata = {}
            for req_idx, (req_id, req_meta) in enumerate(metadata.requests.items()):
                transfer_mappings: dict[tuple[str, int], dict[str, Any]] = {}
                aligned_req_meta = copy.deepcopy(req_meta)
                self._align_remote_block_ids(aligned_req_meta)
                changed_remote_block_ids = {
                    i: aligned_req_meta.remote_block_ids[i]
                    for i in range(self.num_kv_cache_groups)
                    if aligned_req_meta.remote_block_ids[i] != req_meta.remote_block_ids[i]
                }
                aligned_ops.append((req_meta, changed_remote_block_ids))
                for i, kv_cache_spec in enumerate(self.kv_cache_specs):
                    if isinstance(kv_cache_spec, MambaSpec):
                        single_group_transfer_mappings = self._get_kv_split_metadata_for_mamba(
                            aligned_req_meta, req_idx, req_id, i
                        )
                    else:
                        single_group_transfer_mappings = self._get_kv_split_metadata(
                            aligned_req_meta, req_idx, req_id, i
                        )
                    for (host, port), block_dict in single_group_transfer_mappings.items():
                        if (host, port) not in transfer_mappings:
                            transfer_mappings[(host, port)] = {
                                "local_block_ids": [[] for _ in range(self.num_kv_cache_groups)],
                                "remote_block_ids": [[] for _ in range(self.num_kv_cache_groups)],
                                "trans_count": [0 for _ in range(self.num_kv_cache_groups)],
                            }
                        transfer_mappings[(host, port)]["local_block_ids"][i].extend(
                            single_group_transfer_mappings[(host, port)]["local_block_ids"]
                        )
                        transfer_mappings[(host, port)]["remote_block_ids"][i].extend(
                            single_group_transfer_mappings[(host, port)]["remote_block_ids"]
                        )
                        transfer_mappings[(host, port)]["trans_count"][i] = single_group_transfer_mappings[
                            (host, port)
                        ]["trans_count"]
                assert len(transfer_mappings) <= 1, f"Not support add mutil transfer task for req_id:{req_id}"
                update_req_meta = copy.deepcopy(aligned_req_meta)
                for (host, port), block_dict in transfer_mappings.items():
                    update_req_meta.remote_host = host
                    update_req_meta.remote_port = port
                    update_req_meta.local_block_ids = self._get_kernel_block_ids(block_dict["local_block_ids"])
                    update_req_meta.remote_block_ids = self._get_kernel_block_ids(block_dict["remote_block_ids"])
                    update_req_meta.trans_count = block_dict["trans_count"]
                    update_metadata[req_id] = update_req_meta
            request_plan = update_metadata

            if self.attn_resharding_group_idx:
                group_rearrange_block_ids = [[] for _ in range(self.num_kv_cache_groups)]
                group_num_blocks = [0 for _ in range(self.num_kv_cache_groups)]
                group_num_tokens = [0 for _ in range(self.num_kv_cache_groups)]
                group_block_table = [None for _ in range(self.num_kv_cache_groups)]
                group_block_len_tensor = [None for _ in range(self.num_kv_cache_groups)]
                group_seq_start_tensor = [None for _ in range(self.num_kv_cache_groups)]
                device = self.k_buffer.device  # type: ignore
                for i in self.attn_resharding_group_idx:
                    group_rearrange_block_ids[i].extend(
                        sorted(
                            {
                                block_id
                                for req_id in request_plan
                                for block_id in request_plan[req_id].local_block_ids[i]
                            }
                        )
                    )
                    flat_block_ids = group_rearrange_block_ids[i]
                    block_ids_tensor = torch.tensor(flat_block_ids, dtype=torch.int32, device=device)
                    group_num_blocks[i] = len(flat_block_ids)
                    group_num_tokens[i] = group_num_blocks[i] * (self.block_size[i] // self.kernel_block_size_scale[i])
                    group_block_table[i] = block_ids_tensor.view(1, -1)
                    group_block_len_tensor[i] = torch.tensor([group_num_tokens[i]], dtype=torch.int32, device=device)
                    group_seq_start_tensor[i] = torch.tensor([0], dtype=torch.int32, device=device)
                send_task_update = LayerwiseSendTaskUpdate(
                    group_rearrange_block_ids=group_rearrange_block_ids,
                    group_num_blocks=group_num_blocks,
                    group_num_tokens=group_num_tokens,
                    group_block_table=group_block_table,
                    group_block_len_tensor=group_block_len_tensor,
                    group_seq_start_tensor=group_seq_start_tensor,
                )
        return LayerwiseTransferPlan(
            producer=producer,
            requests=request_plan,
            aligned_remote_block_ids=aligned_ops,
            send_task_update=send_task_update,
        )

    def apply(self, plan: LayerwiseTransferPlan, metadata: Any) -> None:
        """Reproduce the legacy metadata write-back for a prepared plan."""
        if not plan.producer:
            return
        for req_meta, changed_remote_block_ids in plan.aligned_remote_block_ids:
            for group_idx, block_ids in changed_remote_block_ids.items():
                req_meta.remote_block_ids[group_idx] = block_ids
        metadata.requests = {}
        for req_id, req_meta in plan.requests.items():
            metadata.requests[req_id] = req_meta

        update = plan.send_task_update
        if update is not None and self.attn_resharding_group_idx:
            send_task = metadata.send_task
            send_task.group_rearrange_block_ids = [[] for _ in range(self.num_kv_cache_groups)]
            send_task.group_num_blocks = [0 for _ in range(self.num_kv_cache_groups)]
            send_task.group_num_tokens = [0 for _ in range(self.num_kv_cache_groups)]
            send_task.group_block_table = [None for _ in range(self.num_kv_cache_groups)]
            send_task.group_block_len_tensor = [None for _ in range(self.num_kv_cache_groups)]
            send_task.group_seq_start_tensor = [None for _ in range(self.num_kv_cache_groups)]
            for i in self.attn_resharding_group_idx:
                send_task.group_rearrange_block_ids[i].extend(update.group_rearrange_block_ids[i])
                send_task.group_num_blocks[i] = update.group_num_blocks[i]
                send_task.group_num_tokens[i] = update.group_num_tokens[i]
                send_task.group_block_table[i] = update.group_block_table[i]
                send_task.group_block_len_tensor[i] = update.group_block_len_tensor[i]
                send_task.group_seq_start_tensor[i] = update.group_seq_start_tensor[i]

    def on_written(self, layer_name: str) -> None:
        if self.current_layer >= self.total_layers:
            return
        layer_names = self.index_to_name[self.current_layer]
        if not layer_names:
            raise RuntimeError(f"Mooncake layerwise metadata is missing physical layer {self.current_layer}")
        if layer_name and layer_name not in layer_names:
            raise RuntimeError(
                f"Mooncake layer hook {layer_name!r} does not belong to physical layer {self.current_layer}: "
                f"{layer_names}"
            )
        cache_write_event = torch.npu.Event()
        cache_write_event.record()
        self._cache_write_events[self.current_layer] = cache_write_event

    def save_layer(
        self,
        layer_name: str,
        kv_layer: list[Any],
        attn_metadata: Any,
        connector_metadata: Any,
        **kwargs,
    ) -> None:
        """MooncakeLayerwiseConnector does not save explicitly."""
        if self.vllm_config.kv_transfer_config.is_kv_producer and connector_metadata.requests.keys():
            if self.current_layer >= self.total_layers:
                self.current_layer += 1
                return
            layer_names = self.index_to_name[self.current_layer]
            if not layer_names:
                raise RuntimeError(f"Mooncake layerwise metadata is missing physical layer {self.current_layer}")
            # Use the scatter-time event recorded by on_kv_cache_written.
            # The fallback preserves correctness for attention paths that do
            # not call the early hook, but gives up transfer/compute overlap.
            if layer_name == "":
                layer_name = layer_names[0]
            elif layer_name not in layer_names:
                raise RuntimeError(
                    f"Mooncake layer hook {layer_name!r} does not belong to physical layer {self.current_layer}: "
                    f"{layer_names}"
                )
            cache_write_event = self._cache_write_events[self.current_layer]
            self._cache_write_events[self.current_layer] = None
            if cache_write_event is None:
                cache_write_event = torch.npu.Event()
                cache_write_event.record()
            send_task = connector_metadata.send_task
            layer_group_idx = self.layer_metadata[layer_name].tensor_group_idx[0]
            keys = None
            values = None
            quant_keys = None
            quant_values = None
            if layer_name not in self.sparse_sfa_c8_layers and (
                (
                    self.pd_head_ratio != 1
                    and (isinstance(self.kv_cache_specs[layer_group_idx], (FullAttentionSpec, SlidingWindowSpec)))
                    and send_task.group_num_blocks[layer_group_idx] > 0
                )
                or (self.enable_c8_quant and self.current_layer in self.vllm_config.quant_config.c8_quant_layers)
                or (self.enable_kv_quant and self.current_layer in self.vllm_config.quant_config.kvcache_quant_layers)
            ):
                assert self.resharding_stream is not None
                with npu_stream_switch(self.resharding_stream):
                    cache_write_event.wait()
                    device = self.k_buffer.device  # type: ignore
                    # Initialize buffers
                    keys = torch.empty(
                        (send_task.group_num_tokens[layer_group_idx], *kv_layer[0].size()[-2:]),
                        dtype=kv_layer[0].dtype,
                        device=device,
                    )
                    values = torch.empty(
                        (send_task.group_num_tokens[layer_group_idx], *kv_layer[1].size()[-2:]),
                        dtype=kv_layer[1].dtype,
                        device=device,
                    )

                    # Load cache data into buffers
                    torch_npu.npu_gather_pa_kv_cache(
                        kv_layer[0],
                        kv_layer[1],
                        send_task.group_block_table[layer_group_idx],
                        send_task.group_block_len_tensor[layer_group_idx],
                        seq_offset=send_task.group_seq_start_tensor[layer_group_idx],
                        key=keys,
                        value=values,
                    )
                    if self.pd_head_ratio != 1:
                        # sort kv caches for each block
                        keys = (
                            keys.view(
                                send_task.group_num_blocks[layer_group_idx], self.pd_head_ratio, -1, *keys.shape[1:]
                            )
                            .transpose(0, 1)
                            .reshape_as(keys)
                        )
                        values = (
                            values.view(
                                send_task.group_num_blocks[layer_group_idx], self.pd_head_ratio, -1, *values.shape[1:]
                            )
                            .transpose(0, 1)
                            .reshape_as(values)
                        )
                        # reshard kv cache
                        keys = keys.reshape(-1, *kv_layer[0].shape[2:])
                        values = values.reshape(-1, *kv_layer[1].shape[2:])

                        (keys, values) = kv_alltoall_and_rearrange(self.pd_head_ratio, keys, values)
                    if self.enable_c8_quant:
                        layer = self.vllm_config.compilation_config.static_forward_context[layer_name]
                        quant_keys = torch.clamp(
                            torch.round(keys * layer._c8_k_inv_scale + layer._c8_k_offset),
                            -128,
                            127,
                        ).to(torch.int8)
                        quant_values = torch.clamp(
                            torch.round(values * layer._c8_v_inv_scale + layer._c8_v_offset),
                            -128,
                            127,
                        ).to(torch.int8)
                        quant_keys = self.get_nz_cache(quant_keys, layer_group_idx)
                        quant_values = self.get_nz_cache(quant_values, layer_group_idx)
                    if (
                        self.enable_kv_quant
                        and self.current_layer in self.vllm_config.quant_config.kvcache_quant_layers
                    ):
                        layer = self.vllm_config.compilation_config.static_forward_context[layer_name]
                        keys = torch.ops.vllm.quantize(
                            keys, layer.fak_descale, layer.fak_descale_reciprocal, layer.fak_offset
                        )
                        quant_keys = self.get_nz_cache(keys, layer_group_idx)
                        quant_values = self.get_nz_cache(values, layer_group_idx)

            assert self.kv_send_layer_thread is not None
            layer_send_task = self.send_task_factory(
                wait_event=cache_write_event,
                k_cache=keys,
                v_cache=values,
                k_quant_cache=quant_keys,
                v_quant_cache=quant_values,
                layer_idx=self.current_layer,
                layer_name=layer_name,
                layer_names=layer_names,
                group_rearrange_block_ids=send_task.group_rearrange_block_ids,
            )
            for req_id, req_meta in connector_metadata.requests.items():
                if len(req_meta.local_block_ids) <= layer_group_idx or not req_meta.local_block_ids[layer_group_idx]:
                    continue
                try:
                    req_meta_update = self.update_decoder_info(req_id, req_meta)
                except Exception as e:
                    logger.warning(
                        "MooncakeLayerwiseConnector transfer fail. req_id=%s, layer_idx=%s, error=%s. ",
                        req_id,
                        self.current_layer,
                        e,
                    )
                    layer_send_task.failed_requests[req_id] = req_meta
                    continue
                logger.debug("Add request %s to kv send layer thread. req_meta_update=%r", req_id, req_meta_update)
                layer_send_task.send_request[req_id] = req_meta_update

            needs_reuse_gate = any(
                len(req_meta.local_block_ids) > layer_group_idx and bool(req_meta.local_block_ids[layer_group_idx])
                for req_meta in connector_metadata.requests.values()
            )
            if needs_reuse_gate:
                # This must happen before AscendStore publishes StoreSaveDone.
                # Otherwise its receiver may consume the previous set state
                # and start H2D before this transfer is registered as pending.
                self._mark_layer_reuse_pending(self.current_layer)
            try:
                self.kv_send_layer_thread.send_queue.put(layer_send_task)
            except Exception as error:
                if needs_reuse_gate:
                    self._complete_layer_reuse(self.current_layer, str(error))
                raise
            self.current_layer += 1

    def init_reuse_state(self) -> None:
        slot_by_addr: dict[int, int] = {}
        layer_storage_slots: dict[int, tuple[int, ...]] = {}
        for layer_idx, layer_names in self.index_to_name.items():
            slots: list[int] = []
            for layer_name in layer_names:
                for addr in self.layer_metadata[layer_name].kv_caches_base_addr:
                    slot_id = slot_by_addr.setdefault(addr, len(slot_by_addr))
                    if slot_id not in slots:
                        slots.append(slot_id)
            layer_storage_slots[layer_idx] = tuple(slots)
        self.layer_storage_slots = layer_storage_slots
        self.storage_send_done_events = []
        for _ in range(len(slot_by_addr)):
            event = threading.Event()
            event.set()
            self.storage_send_done_events.append(event)


__all__ = [
    "LayerwiseSendTaskUpdate",
    "LayerwiseStepController",
    "LayerwiseTransferPlan",
]
