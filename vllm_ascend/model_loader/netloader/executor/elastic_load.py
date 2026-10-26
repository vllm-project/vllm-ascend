#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import math
import time
from typing import Any

import torch
import torch_npu
from torch.nn import Module
from vllm.logger import logger

from vllm_ascend.utils import ACL_FORMAT_FRACTAL_ND, ACL_FORMAT_FRACTAL_NZ

from .netloader_pg import destroy_stateless_process_group, stateless_init_process_group

_NETLOADER_TRANSFER_ITEMS_ATTR = "_netloader_processed_layout_transfer_items"


def _is_transferable_tensor(tensor: torch.Tensor) -> bool:
    return not tensor.is_meta and tensor.numel() > 0 and tensor.device.type == "npu"


def _iter_tensors_in_value(prefix: str, value: Any, visited_object_ids: set[int], scan_objects: bool = False):
    if isinstance(value, torch.Tensor):
        yield prefix, value
        return

    if isinstance(value, (Module, str, bytes)) or callable(value):
        return

    if isinstance(value, (list, tuple, dict)):
        value_id = id(value)
        if value_id in visited_object_ids:
            return
        visited_object_ids.add(value_id)

    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _iter_tensors_in_value(f"{prefix}.{index}", item, visited_object_ids, scan_objects)
        return

    if isinstance(value, dict):
        for key, item in value.items():
            yield from _iter_tensors_in_value(f"{prefix}.{key}", item, visited_object_ids, scan_objects)
        return

    if not scan_objects:
        return

    value_id = id(value)
    if value_id in visited_object_ids:
        return
    if not hasattr(value, "__dict__"):
        return
    visited_object_ids.add(value_id)
    for attr_name, attr_value in vars(value).items():
        if attr_name.startswith("_"):
            continue
        yield from _iter_tensors_in_value(f"{prefix}.{attr_name}", attr_value, visited_object_ids, scan_objects)


def _try_collect_transferable_tensor(
    name: str,
    tensor: torch.Tensor,
    seen_data_ptrs: set[int],
    collected_tensors: list[tuple[str, torch.Tensor]],
) -> None:
    if not _is_transferable_tensor(tensor):
        return

    data_ptr = tensor.data_ptr()
    if data_ptr in seen_data_ptrs:
        return

    seen_data_ptrs.add(data_ptr)
    collected_tensors.append((name, tensor))


def _collect_processed_layout_tensors(model: Module) -> list[tuple[str, torch.Tensor]]:
    """Collect live inference tensors for int8_cache=no processed-weight transfer."""
    seen_data_ptrs: set[int] = set()
    visited_object_ids: set[int] = set()
    collected_tensors: list[tuple[str, torch.Tensor]] = []

    for name, tensor in model.named_parameters():
        _try_collect_transferable_tensor(name, tensor, seen_data_ptrs, collected_tensors)

    for name, tensor in model.named_buffers():
        _try_collect_transferable_tensor(name, tensor, seen_data_ptrs, collected_tensors)

    # Ascend post-load paths may store derived runtime tensors as module attributes,
    # e.g. aclnn_input_scale_reciprocal and MLA/SFA W_UV / W_UK_T.
    for module_prefix, module in model.named_modules():
        for attr_name, attr_value in vars(module).items():
            if attr_name.startswith("_") or isinstance(attr_value, Module):
                continue

            # Attention impl objects store derived tensors (e.g. W_UV/W_UK_T) on plain Python
            # attributes; only "impl" needs deep scan. Other attrs are direct tensors,
            # list/dict containers, or nn.Module children already covered above. Keep
            # one visited set for the entire model because every attention impl can
            # reference the same large vllm_config object graph.
            scan_objects = attr_name == "impl"
            for tensor_name, tensor in _iter_tensors_in_value(
                attr_name,
                attr_value,
                visited_object_ids,
                scan_objects,
            ):
                full_name = f"{module_prefix}.{tensor_name}" if module_prefix else tensor_name
                _try_collect_transferable_tensor(full_name, tensor, seen_data_ptrs, collected_tensors)

    return collected_tensors


def _iter_raw_send_transfer_params(model):
    for name, param in model.named_parameters():
        if "aclnn_input_scale" in name:
            continue
        yield name, param


def _iter_raw_recv_transfer_params(model):
    for name, param in model.named_parameters():
        if len(param.shape) == 0:
            continue
        yield name, param


def register_processed_layout_transfer_items(model: Module) -> list[tuple[str, torch.Tensor]]:
    """Register processed-layout transfer items after weights are finalized."""
    cached_items = get_cached_processed_layout_transfer_items(model)
    if cached_items is not None:
        return cached_items
    return _collect_processed_layout_tensors(model)


def cache_processed_layout_transfer_manifest(model: Module) -> int:
    """Cache processed-layout transfer items on the model after process_weights."""
    transfer_items = _collect_processed_layout_tensors(model)
    setattr(model, _NETLOADER_TRANSFER_ITEMS_ATTR, transfer_items)
    return len(transfer_items)


def get_cached_processed_layout_transfer_items(model: Module) -> list[tuple[str, torch.Tensor]] | None:
    items = getattr(model, _NETLOADER_TRANSFER_ITEMS_ATTR, None)
    if items is None:
        return None
    return items


def build_transfer_shape_manifest(
    transfer_items: list[tuple[str, torch.Tensor]],
) -> dict[str, tuple[int, ...]]:
    """Build a name-to-shape manifest for processed-layout transfer."""
    return {name: tuple(tensor.shape) for name, tensor in transfer_items}


def reshape_tensor_to_manifest_shape(
    name: str,
    tensor: torch.Tensor,
    manifest_shape: tuple[int, ...] | None,
) -> bool:
    """View a received tensor into the seed/manifest shape when numel matches."""
    if manifest_shape is None or tuple(tensor.shape) == manifest_shape:
        return True

    if tensor.numel() != math.prod(manifest_shape):
        logger.error(
            "Weight shape mismatch for %s, local shape %s cannot view as manifest shape %s",
            name,
            tuple(tensor.shape),
            manifest_shape,
        )
        return False

    local_shape = tuple(tensor.shape)
    try:
        tensor.data = tensor.data.view(manifest_shape)
    except Exception as e:
        logger.error(
            "Failed to reshape netloader tensor %s from %s to manifest shape %s: %s",
            name,
            local_shape,
            manifest_shape,
            e,
        )
        return False

    logger.debug(
        "Reshaped netloader tensor %s from %s to %s after recv",
        name,
        local_shape,
        manifest_shape,
    )
    return True


def reshape_transfer_items_to_manifest(
    transfer_items: list[tuple[str, torch.Tensor]],
    transfer_shape_manifest: dict[str, tuple[int, ...]] | None,
) -> bool:
    """Reshape received tensors to match the server transfer manifest."""
    if not transfer_shape_manifest:
        return True

    for name, tensor in transfer_items:
        manifest_shape = transfer_shape_manifest.get(name)
        if manifest_shape is None:
            logger.error("Missing manifest shape for transfer tensor %s", name)
            return False
        if not reshape_tensor_to_manifest_shape(name, tensor, manifest_shape):
            return False
    return True


def validate_transfer_items_numel_against_manifest(
    transfer_items: list[tuple[str, torch.Tensor]],
    transfer_shape_manifest: dict[str, tuple[int, ...]] | None,
) -> bool:
    """Fail-fast before HCCL: local tensors must match seed manifest numel.

    Shape/view may differ; mismatched numel would size the recv buffer wrong and
    can hang or corrupt during transfer. Post-recv reshape still handles view align.
    """
    if not transfer_shape_manifest:
        return True

    for name, tensor in transfer_items:
        manifest_shape = transfer_shape_manifest.get(name)
        if manifest_shape is None:
            logger.error("Missing manifest shape for transfer tensor %s before HCCL", name)
            return False
        manifest_numel = math.prod(manifest_shape)
        if tensor.numel() != manifest_numel:
            logger.error(
                "Weight numel mismatch before HCCL for %s: local shape %s (numel=%s) vs manifest shape %s (numel=%s)",
                name,
                tuple(tensor.shape),
                tensor.numel(),
                manifest_shape,
                manifest_numel,
            )
            return False
    return True


def _get_send_transfer_items(
    model,
    send_processed_weights: bool,
    registered_transfer_items: list[tuple[str, torch.Tensor]] | None = None,
) -> list[tuple[str, torch.Tensor]]:
    if send_processed_weights:
        if registered_transfer_items is None:
            raise RuntimeError(
                "Processed-layout P2P send requires registered transfer items. "
                "Call ElasticServer.register_transfer_manifest() after process_weights."
            )
        return registered_transfer_items
    return list(_iter_raw_send_transfer_params(model))


def _get_recv_transfer_items(model, transfer_processed_layout: bool) -> list[tuple[str, torch.Tensor]]:
    if transfer_processed_layout:
        cached_items = get_cached_processed_layout_transfer_items(model)
        if cached_items is not None:
            return cached_items
        return _collect_processed_layout_tensors(model)
    return list(_iter_raw_recv_transfer_params(model))


def _get_npu_format_int(tensor: torch.Tensor) -> int | None:
    if tensor.device.type != "npu":
        return None
    try:
        return int(torch_npu.get_npu_format(tensor))
    except Exception:
        return None


def _is_fractal_nz_tensor(tensor: torch.Tensor) -> bool:
    return _get_npu_format_int(tensor) == ACL_FORMAT_FRACTAL_NZ


def _is_packed_int32_nz_tensor(tensor: torch.Tensor) -> bool:
    """W4A8 packs int8 NZ as int32 via ``view``; TransData cannot cast that int32 NZ."""
    return tensor.dtype == torch.int32 and tensor.numel() > 0 and tensor.shape[-1] > 0 and _is_fractal_nz_tensor(tensor)


def _cast_packed_int32_via_int8(tensor: torch.Tensor, acl_format: int) -> torch.Tensor:
    """Export/import W4A8 packed int32 through the underlying int8 layout.

    ``process_weights`` does ``maybe_trans_nz(int8)`` then ``view(int32)``. The
    physical tiling stays int8-NZ, so HCCL-facing TransData must run on int8.

    Casting a dtype-view (int32 storage interpreted as int8) can raise
    ``Cannot resize storage without base format`` on ND→NZ. Materialize a real
    int8 tensor first, and avoid ``contiguous()`` after viewing NZ as int32.
    """
    int8_view = tensor.view(torch.int8)
    # Break the int32↔int8 view chain before TransData.
    int8_tensor = int8_view.contiguous().clone()
    casted_int8 = torch_npu.npu_format_cast(int8_tensor, acl_format)

    # Match process_weights pack semantics: view(int32) only on NZ.
    packed = casted_int8.view(torch.int32)
    if acl_format == ACL_FORMAT_FRACTAL_ND and not packed.is_contiguous():
        packed = packed.contiguous()
    return packed


def _cast_tensor_to_fractal_nd(tensor: torch.Tensor) -> torch.Tensor:
    """Cast an NPU tensor to FRACTAL_ND for HCCL transfer."""
    contiguous = tensor if tensor.is_contiguous() else tensor.contiguous()
    if not _is_fractal_nz_tensor(contiguous):
        return contiguous
    # Packed W4A8 weights keep int8-NZ tiling under an int32 view.
    if _is_packed_int32_nz_tensor(contiguous):
        return _cast_packed_int32_via_int8(contiguous, ACL_FORMAT_FRACTAL_ND)
    return torch_npu.npu_format_cast(contiguous, ACL_FORMAT_FRACTAL_ND)


def _cast_tensor_to_fractal_nz(tensor: torch.Tensor) -> torch.Tensor:
    """Cast an NPU tensor back to FRACTAL_NZ after HCCL transfer."""
    contiguous = tensor if tensor.is_contiguous() else tensor.contiguous()
    if _is_fractal_nz_tensor(contiguous):
        return contiguous
    # Recv buffer for packed W4A8 is int32 ND; restore NZ via underlying int8 view.
    if contiguous.dtype == torch.int32 and contiguous.numel() > 0 and contiguous.shape[-1] > 0:
        return _cast_packed_int32_via_int8(contiguous, ACL_FORMAT_FRACTAL_NZ)
    return torch_npu.npu_format_cast(contiguous, ACL_FORMAT_FRACTAL_NZ)


def _hccl_transfer_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Prepare a processed-layout tensor for HCCL: contiguous FRACTAL_ND."""
    return _cast_tensor_to_fractal_nd(tensor)


def synchronize_npu(device_type: str | None = None) -> None:
    """Synchronize NPU when the active device type is NPU."""
    if device_type is not None and device_type != "npu":
        return
    torch_npu.npu.synchronize()


def _prepare_hccl_recv_buffer(tensor: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Processed-layout recv: FRACTAL_ND buffer + whether to restore FRACTAL_NZ."""
    if _is_fractal_nz_tensor(tensor):
        # Allocate fresh ND storage. Do not TransData the local NZ weight (W4A8
        # packed int32 NZ is not a valid TransData source; content is overwritten).
        nd_buffer = torch.empty(tensor.shape, dtype=tensor.dtype, device=tensor.device)
        return nd_buffer, True

    if not tensor.is_contiguous():
        tensor.data = tensor.contiguous()
    return tensor, False


def _finalize_hccl_recv_buffer(
    tensor: torch.Tensor,
    recv_buffer: torch.Tensor,
    restore_fractal_nz: bool,
) -> None:
    """Write HCCL recv results back to the target tensor, restoring FRACTAL_NZ if needed."""
    if not restore_fractal_nz:
        return
    tensor.data = _cast_tensor_to_fractal_nz(recv_buffer)


class P2PLoad:
    """
    Class for receiving model parameters in a distributed manner using HCCL backend.
    """

    def __init__(
        self,
        world_name: str,
        source_ip: str,
        source_port: int,
        group_name: str = "netloader",
        transfer_processed_layout: bool = False,
        transfer_shape_manifest: dict[str, tuple[int, ...]] | None = None,
    ):
        """
        Initializes the P2PLoad instance.

        Parameters:
        - world_name: The name of the distributed group.
        - source_ip: The IP address of the source node.
        - source_port: The port number for the source node.
        - group_name: Name of the HCCL process group.
        - transfer_processed_layout: Whether to receive processed inference tensors.
        - transfer_shape_manifest: Seed-side tensor shapes keyed by transfer name.
        """
        self.world_name = world_name
        self.source_ip = source_ip
        self.source_port = source_port
        self.group_name = group_name
        self.transfer_processed_layout = transfer_processed_layout
        self.transfer_shape_manifest = transfer_shape_manifest

    def load(self, model):
        """
        Loads the model parameters using HCCL backend.

        Parameters:
        - model: The model whose parameters are to be loaded.

        Returns:
        - The model if loading is successful, otherwise None.
        """
        receiver_pg = None
        loaded_model = None
        transfer_count = 0
        recv_addr = f"{self.source_ip}:{self.source_port}"
        try:
            start_recv = time.perf_counter()
            transfer_items = _get_recv_transfer_items(model, self.transfer_processed_layout)
            transfer_count = len(transfer_items)

            if self.transfer_processed_layout and not validate_transfer_items_numel_against_manifest(
                transfer_items,
                self.transfer_shape_manifest,
            ):
                logger.error(
                    "[netloader_p2p] abort recv before HCCL due to numel/manifest mismatch "
                    "transfer=%s group=%s addr=%s",
                    transfer_count,
                    self.group_name,
                    recv_addr,
                )
                return None

            receiver_pg = stateless_init_process_group(
                host=self.world_name.split(":")[0],
                port=self.source_port,
                rank=0,
                world_size=2,
                group_name=self.group_name,
            )

            model_device = next(model.parameters()).device
            device_index = model_device.index if model_device.index is not None else torch.npu.current_device()

            def _recv_all() -> None:
                for index, (name, tensor) in enumerate(transfer_items):
                    try:
                        if self.transfer_processed_layout:
                            recv_buffer, restore_fractal_nz = _prepare_hccl_recv_buffer(tensor)
                            receiver_pg.recv([recv_buffer], 1, 0).wait()
                            _finalize_hccl_recv_buffer(tensor, recv_buffer, restore_fractal_nz)
                        else:
                            if not tensor.is_contiguous():
                                tensor.data = tensor.contiguous()
                            receiver_pg.recv([tensor], 1, 0).wait()
                    except Exception as exc:
                        logger.error(
                            "[netloader_p2p] recv failed at index=%s name=%s shape=%s: %s",
                            index,
                            name,
                            tuple(tensor.shape),
                            exc,
                        )
                        raise

            if self.transfer_processed_layout:
                _recv_all()
                torch.distributed.barrier(group=receiver_pg, device_ids=[device_index])
                synchronize_npu()
            else:
                trans_stream = torch_npu.npu.Stream()
                with torch_npu.npu.stream(trans_stream):
                    _recv_all()
                    torch.distributed.barrier(group=receiver_pg, device_ids=[device_index])
                    torch_npu.npu.synchronize(trans_stream)

            if self.transfer_processed_layout:
                if not reshape_transfer_items_to_manifest(
                    transfer_items,
                    self.transfer_shape_manifest,
                ):
                    logger.error(
                        "[netloader_p2p] failed to reshape received tensors to seed manifest "
                        "transfer=%s group=%s addr=%s",
                        transfer_count,
                        self.group_name,
                        recv_addr,
                    )
                    return None

            logger.info(
                "[netloader_p2p] HCCL recv time: %s, transfer=%s group=%s processed_layout=%s addr=%s",
                time.perf_counter() - start_recv,
                transfer_count,
                self.group_name,
                self.transfer_processed_layout,
                recv_addr,
            )
            loaded_model = model
        except Exception as e:
            logger.error(
                "[netloader_p2p] recv failed transfer=%s group=%s processed_layout=%s addr=%s: %s",
                transfer_count if transfer_count else "unknown",
                self.group_name,
                self.transfer_processed_layout,
                recv_addr,
                e,
            )
        finally:
            if receiver_pg:
                destroy_stateless_process_group(receiver_pg)
        return loaded_model


class P2PSend:
    """
    Class for sending model parameters in a distributed manner using HCCL backend.
    """

    def __init__(
        self,
        listen_ip: str,
        listen_port: int,
        comm_name: str,
        group_name: str = "netloader",
        send_processed_weights: bool = False,
    ):
        """
        Initializes the P2PSend instance.

        Parameters:
        - listen_ip: The IP address to listen on.
        - listen_port: The port number to listen on.
        - comm_name: The name of the communication group.
        - group_name: Name of the HCCL process group.
        - send_processed_weights: Whether to send already processed model parameters.
        """
        self.listen_ip = listen_ip
        self.listen_port = listen_port
        self.comm_name = comm_name
        self.group_name = group_name
        self.send_processed_weights = send_processed_weights

    def send(
        self,
        model,
        int8_params: dict,
        registered_transfer_items: list[tuple[str, torch.Tensor]] | None = None,
    ):
        """
        Sends the model parameters using HCCL backend.

        Parameters:
        - model: The model whose parameters are to be sent.
        - int8_params: Dictionary of parameters that are in int8 format.
        - registered_transfer_items: Cached processed-layout transfer items registered on the server.
        """
        model_device = next(model.parameters()).device
        torch.npu.set_device(model_device)
        sender_pg = None
        transfer_count = 0
        send_addr = f"{self.listen_ip}:{self.listen_port}"
        try:
            start_send = time.perf_counter()
            sender_pg = stateless_init_process_group(
                host=self.comm_name.split(":")[0],
                port=self.listen_port,
                rank=1,
                world_size=2,
                group_name=self.group_name,
            )

            transfer_items = _get_send_transfer_items(
                model,
                self.send_processed_weights,
                registered_transfer_items,
            )
            transfer_count = len(transfer_items)
            device_index = model_device.index if model_device.index is not None else torch.npu.current_device()

            def _send_all() -> None:
                for index, (name, tensor_ref) in enumerate(transfer_items):
                    try:
                        if not self.send_processed_weights and name in int8_params:
                            payload = int8_params[name].to(model_device)
                        elif self.send_processed_weights:
                            payload = _hccl_transfer_tensor(tensor_ref)
                        else:
                            payload = tensor_ref if tensor_ref.is_contiguous() else tensor_ref.contiguous()
                        sender_pg.send([payload], 0, 0).wait()
                    except Exception as exc:
                        logger.error(
                            "[netloader_p2p] send failed at index=%s name=%s shape=%s: %s",
                            index,
                            name,
                            tuple(tensor_ref.shape),
                            exc,
                        )
                        raise

            if self.send_processed_weights:
                _send_all()
                torch.distributed.barrier(group=sender_pg, device_ids=[device_index])
                synchronize_npu()
            else:
                trans_stream = torch_npu.npu.Stream()
                with torch_npu.npu.stream(trans_stream):
                    _send_all()
                    torch.distributed.barrier(group=sender_pg, device_ids=[device_index])
                    torch_npu.npu.synchronize(trans_stream)

            logger.info(
                "[netloader_p2p] HCCL send time: %s, transfer=%s group=%s processed_layout=%s addr=%s",
                time.perf_counter() - start_send,
                transfer_count,
                self.group_name,
                self.send_processed_weights,
                send_addr,
            )
        except Exception as e:
            logger.error(
                "[netloader_p2p] send failed transfer=%s group=%s processed_layout=%s addr=%s: %s",
                transfer_count if transfer_count else "unknown",
                self.group_name,
                self.send_processed_weights,
                send_addr,
                e,
            )
            raise
        finally:
            if sender_pg:
                destroy_stateless_process_group(sender_pg)
