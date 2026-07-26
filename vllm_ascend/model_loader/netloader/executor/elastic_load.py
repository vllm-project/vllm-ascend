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

import time
from typing import Any

import torch
import torch_npu
from torch.nn import Module
from vllm.logger import logger

from vllm_ascend.utils import ACL_FORMAT_FRACTAL_ND, ACL_FORMAT_FRACTAL_NZ

from .netloader_pg import destroy_stateless_process_group, stateless_init_process_group

_NETLOADER_TRANSFER_ITEMS_ATTR = "_netloader_processed_layout_transfer_items"
_NETLOADER_TRANSFER_SHAPES_ATTR = "_netloader_processed_layout_transfer_shapes"


def _is_tensor_on_transfer_device(tensor: torch.Tensor) -> bool:
    return tensor.device.type == "npu"


def _is_transferable_tensor(tensor: torch.Tensor) -> bool:
    return not tensor.is_meta and tensor.numel() > 0 and _is_tensor_on_transfer_device(tensor)


def _iter_tensors_in_value(prefix: str, value: Any, visited_object_ids: set[int], scan_objects: bool = False):
    if isinstance(value, torch.Tensor):
        yield prefix, value
        return

    if isinstance(value, (Module, str, bytes)) or callable(value):
        return

    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _iter_tensors_in_value(f"{prefix}.{index}", item, visited_object_ids, scan_objects)
        return

    if isinstance(value, dict):
        for key, item in value.items():
            yield from _iter_tensors_in_value(f"{prefix}.{key}", item, visited_object_ids, scan_objects)
        return

    if not scan_objects or not hasattr(value, "__dict__"):
        return

    value_id = id(value)
    if value_id in visited_object_ids:
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

            scan_objects = attr_name == "impl"
            for tensor_name, tensor in _iter_tensors_in_value(attr_name, attr_value, set(), scan_objects):
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
    """Cache processed-layout transfer manifest on the model after process_weights."""
    transfer_items = _collect_processed_layout_tensors(model)
    setattr(model, _NETLOADER_TRANSFER_ITEMS_ATTR, transfer_items)
    setattr(model, _NETLOADER_TRANSFER_SHAPES_ATTR, build_transfer_shape_manifest(transfer_items))
    return len(transfer_items)


def get_cached_processed_layout_transfer_items(model: Module) -> list[tuple[str, torch.Tensor]] | None:
    items = getattr(model, _NETLOADER_TRANSFER_ITEMS_ATTR, None)
    if items is None:
        return None
    return items


def get_cached_processed_layout_transfer_shapes(model: Module) -> dict[str, tuple[int, ...]] | None:
    shapes = getattr(model, _NETLOADER_TRANSFER_SHAPES_ATTR, None)
    if shapes is None:
        return None
    return shapes


def build_transfer_shape_manifest(
    transfer_items: list[tuple[str, torch.Tensor]],
) -> dict[str, tuple[int, ...]]:
    """Build a name-to-shape manifest for processed-layout transfer."""
    return {name: tuple(tensor.shape) for name, tensor in transfer_items}


def _numel_from_shape(shape: tuple[int, ...]) -> int:
    numel = 1
    for dim in shape:
        numel *= dim
    return numel


def reshape_tensor_to_manifest_shape(
    name: str,
    tensor: torch.Tensor,
    manifest_shape: tuple[int, ...] | None,
) -> bool:
    """View a received tensor into the seed/manifest shape when numel matches."""
    if manifest_shape is None or tuple(tensor.shape) == manifest_shape:
        return True

    if tensor.numel() != _numel_from_shape(manifest_shape):
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


def _get_send_transfer_items(
    model,
    send_processed_weights: bool,
    registered_transfer_items: list[tuple[str, torch.Tensor]] | None = None,
) -> list[tuple[str, torch.Tensor]]:
    if send_processed_weights:
        if registered_transfer_items is not None:
            return registered_transfer_items
        return _collect_processed_layout_tensors(model)
    return list(_iter_raw_send_transfer_params(model))


def _get_recv_transfer_items(model, transfer_processed_layout: bool) -> list[tuple[str, torch.Tensor]]:
    if transfer_processed_layout:
        cached_items = get_cached_processed_layout_transfer_items(model)
        if cached_items is not None:
            return cached_items
        return _collect_processed_layout_tensors(model)
    return list(_iter_raw_recv_transfer_params(model))


def _log_transfer_plan(role: str, transfer_count: int, group_name: str, processed_layout: bool, addr: str) -> None:
    logger.info(
        "[netloader_p2p] %s transfer=%s group=%s processed_layout=%s addr=%s",
        role,
        transfer_count,
        group_name,
        processed_layout,
        addr,
    )


_TRANSFER_DEBUG_HEAD = 5
_TRANSFER_DEBUG_TAIL = 3
_TRANSFER_DEBUG_PROGRESS_INTERVAL = 100


def _get_npu_format_int(tensor: torch.Tensor) -> int | None:
    if tensor.device.type != "npu":
        return None
    try:
        return int(torch_npu.get_npu_format(tensor))
    except Exception:
        return None


def _safe_get_npu_format(tensor: torch.Tensor) -> str:
    npu_format = _get_npu_format_int(tensor)
    if npu_format is None:
        if tensor.device.type != "npu":
            return "n/a"
        return "error:unknown"
    return str(npu_format)


def _is_fractal_nz_tensor(tensor: torch.Tensor) -> bool:
    return _get_npu_format_int(tensor) == ACL_FORMAT_FRACTAL_NZ


def _is_packed_int32_nz_tensor(tensor: torch.Tensor) -> bool:
    """W4A8 packs int8 NZ as int32 via ``view``; TransData cannot cast that int32 NZ."""
    return (
        tensor.dtype == torch.int32
        and tensor.numel() > 0
        and tensor.shape[-1] > 0
        and _is_fractal_nz_tensor(tensor)
    )


def _cast_packed_int32_via_int8(tensor: torch.Tensor, acl_format: int) -> torch.Tensor:
    """Export/import W4A8 packed int32 through the underlying int8 layout.

    ``process_weights`` does ``maybe_trans_nz(int8)`` then ``view(int32)``. The
    physical tiling stays int8-NZ, so HCCL-facing TransData must run on int8.

    Casting a dtype-view (int32 storage interpreted as int8) can raise
    ``Cannot resize storage without base format`` on ND→NZ. Materialize a real
    int8 tensor first, and avoid ``contiguous()`` after viewing NZ as int32.
    """
    src_format = _safe_get_npu_format(tensor)
    int8_view = tensor.view(torch.int8)
    # Break the int32↔int8 view chain before TransData.
    int8_tensor = int8_view.contiguous().clone()
    logger.info(
        "[netloader_p2p][debug] packed_int32_via_int8 begin src_dtype=%s src_shape=%s "
        "src_npu_format=%s src_data_ptr=%s int8_shape=%s int8_data_ptr=%s target_format=%s",
        tensor.dtype,
        tuple(tensor.shape),
        src_format,
        tensor.data_ptr(),
        tuple(int8_tensor.shape),
        int8_tensor.data_ptr(),
        acl_format,
    )
    try:
        casted_int8 = torch_npu.npu_format_cast(int8_tensor, acl_format)
    except Exception as exc:
        logger.error(
            "[netloader_p2p][debug] packed_int32_via_int8 format_cast failed "
            "src_shape=%s src_npu_format=%s int8_shape=%s target_format=%s error=%s",
            tuple(tensor.shape),
            src_format,
            tuple(int8_tensor.shape),
            acl_format,
            exc,
        )
        raise

    # Match process_weights pack semantics: view(int32) only on NZ.
    packed = casted_int8.view(torch.int32)
    if acl_format == ACL_FORMAT_FRACTAL_ND and not packed.is_contiguous():
        packed = packed.contiguous()
    logger.info(
        "[netloader_p2p][debug] packed_int32_via_int8 end out_shape=%s out_dtype=%s "
        "out_npu_format=%s out_contiguous=%s out_data_ptr=%s",
        tuple(packed.shape),
        packed.dtype,
        _safe_get_npu_format(packed),
        packed.is_contiguous(),
        packed.data_ptr(),
    )
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
    """Prepare a tensor for HCCL: contiguous FRACTAL_ND layout."""
    return _cast_tensor_to_fractal_nd(tensor)


def _format_transfer_tensor_debug(name: str, tensor: torch.Tensor, index: int | None = None) -> str:
    index_prefix = f"index={index}, " if index is not None else ""
    storage_nbytes = tensor.untyped_storage().nbytes() if tensor.numel() > 0 else 0
    return (
        f"{index_prefix}name={name}, shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
        f"numel={tensor.numel()}, contiguous={tensor.is_contiguous()}, "
        f"npu_format={_safe_get_npu_format(tensor)}, storage_nbytes={storage_nbytes}, "
        f"data_ptr={tensor.data_ptr()}, device={tensor.device}"
    )


_NETLOADER_DEBUG_PRIORITY_KEYWORDS = ("w13_weight", "w2_weight", "experts.", "aclnn_input_scale")


def _safe_current_npu_stream() -> str:
    try:
        return str(torch_npu.npu.current_stream())
    except Exception as exc:
        return f"error:{exc}"


def log_netloader_debug_checkpoint(phase: str, **fields: object) -> None:
    detail = " ".join(f"{key}={value}" for key, value in fields.items())
    logger.info("[netloader][debug] checkpoint phase=%s %s", phase, detail)


def synchronize_npu_with_debug(phase: str, device_type: str | None = None) -> float:
    """Synchronize NPU and emit timing logs for netloader debug."""
    if device_type is not None and device_type != "npu":
        return 0.0

    start = time.perf_counter()
    log_netloader_debug_checkpoint(
        phase,
        event="npu.synchronize_begin",
        stream=_safe_current_npu_stream(),
    )
    try:
        torch_npu.npu.synchronize()
    except Exception as exc:
        logger.error(
            "[netloader][debug] npu.synchronize failed phase=%s stream=%s error=%s",
            phase,
            _safe_current_npu_stream(),
            exc,
        )
        raise
    elapsed = time.perf_counter() - start
    log_netloader_debug_checkpoint(
        phase,
        event="npu.synchronize_end",
        elapsed=f"{elapsed:.6f}s",
        stream=_safe_current_npu_stream(),
    )
    return elapsed


def log_transfer_manifest_debug_sample(model: Module, phase: str, max_samples: int = 8) -> None:
    """Log a small sample of processed-layout tensors to correlate layout with failures."""
    try:
        items = get_cached_processed_layout_transfer_items(model)
        if items is None:
            items = _collect_processed_layout_tensors(model)
    except Exception as exc:
        logger.warning("[netloader][debug] manifest sample skipped phase=%s error=%s", phase, exc)
        return
    if not items:
        log_netloader_debug_checkpoint(phase, event="manifest_sample", total=0, showing=0)
        return

    priority_items = [
        item for item in items if any(keyword in item[0] for keyword in _NETLOADER_DEBUG_PRIORITY_KEYWORDS)
    ]
    other_items = [item for item in items if item not in priority_items]
    sample_items = (priority_items + other_items)[:max_samples]
    log_netloader_debug_checkpoint(
        phase,
        event="manifest_sample",
        total=len(items),
        showing=len(sample_items),
    )
    for index, (name, tensor) in enumerate(sample_items):
        logger.info(
            "[netloader][debug] manifest sample phase=%s index=%s %s",
            phase,
            index,
            _format_transfer_tensor_debug(name, tensor, index),
        )


def _log_transfer_items_debug(role: str, transfer_items: list[tuple[str, torch.Tensor]], group_name: str) -> None:
    names = [name for name, _ in transfer_items]
    tail_names = names[-_TRANSFER_DEBUG_TAIL :] if len(names) > _TRANSFER_DEBUG_TAIL else []
    logger.info(
        "[netloader_p2p][debug] %s transfer_items count=%s group=%s head=%s tail=%s",
        role,
        len(names),
        group_name,
        names[:_TRANSFER_DEBUG_HEAD],
        tail_names,
    )
    for index, (name, tensor) in enumerate(transfer_items):
        if index < _TRANSFER_DEBUG_HEAD or index >= len(transfer_items) - _TRANSFER_DEBUG_TAIL:
            logger.info(
                "[netloader_p2p][debug] %s transfer[%s] %s",
                role,
                index,
                _format_transfer_tensor_debug(name, tensor),
            )


def _should_log_transfer_progress(index: int, transfer_count: int) -> bool:
    if index < _TRANSFER_DEBUG_HEAD:
        return True
    if index >= transfer_count - _TRANSFER_DEBUG_TAIL:
        return True
    return (index + 1) % _TRANSFER_DEBUG_PROGRESS_INTERVAL == 0


def _log_transfer_order_diff(
    label: str,
    local_names: list[str],
    server_names: list[str],
    group_name: str,
) -> None:
    if local_names == server_names:
        logger.info(
            "[netloader_p2p][debug] %s transfer order matches server manifest group=%s count=%s",
            label,
            group_name,
            len(local_names),
        )
        return

    first_mismatch_index = next(
        (index for index, pair in enumerate(zip(local_names, server_names, strict=False)) if pair[0] != pair[1]),
        min(len(local_names), len(server_names)),
    )
    logger.warning(
        "[netloader_p2p][debug] %s transfer order mismatch group=%s local_count=%s server_count=%s "
        "first_index=%s local=%s server=%s",
        label,
        group_name,
        len(local_names),
        len(server_names),
        first_mismatch_index,
        local_names[first_mismatch_index] if first_mismatch_index < len(local_names) else None,
        server_names[first_mismatch_index] if first_mismatch_index < len(server_names) else None,
    )


def _prepare_hccl_recv_buffer(tensor: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Return an FRACTAL_ND recv buffer and whether to restore FRACTAL_NZ after recv."""
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
    logger.info(
        "[netloader_p2p][debug] finalize_recv restore_nz begin target=%s recv_buffer=%s",
        _format_transfer_tensor_debug("target", tensor),
        _format_transfer_tensor_debug("recv_buffer", recv_buffer),
    )
    try:
        restored = _cast_tensor_to_fractal_nz(recv_buffer)
    except Exception as exc:
        logger.error(
            "[netloader_p2p][debug] finalize_recv restore_nz failed target=%s recv_buffer=%s error=%s",
            _format_transfer_tensor_debug("target", tensor),
            _format_transfer_tensor_debug("recv_buffer", recv_buffer),
            exc,
        )
        raise
    tensor.data = restored
    logger.info(
        "[netloader_p2p][debug] finalize_recv restore_nz end target=%s",
        _format_transfer_tensor_debug("target", tensor),
    )


def _ensure_hccl_recv_buffer(tensor: torch.Tensor) -> torch.Tensor:
    """Make tensor storage contiguous in-place for HCCL recv when needed."""
    recv_buffer, _ = _prepare_hccl_recv_buffer(tensor)
    return recv_buffer


def _barrier_after_p2p_transfer(process_group) -> None:
    """Handshake both netloader P2P ranks after the transfer loop.

    Do not pass ``device_ids=[model_device.index]``: the stateless netloader PG
    only has ranks {0, 1}, and global TP device indices trigger
    ``Global rank X is not part of group``.
    """
    torch.distributed.barrier(group=process_group)


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
        model_device = next(model.parameters()).device
        logger.info(
            "Start init_process_group, name: %s, addr: %s:%s", self.world_name, self.source_ip, self.source_port
        )
        receiver_pg = None
        loaded_model = None
        try:
            start_init_process_group = time.perf_counter()
            receiver_pg = stateless_init_process_group(
                host=self.world_name.split(":")[0],
                port=self.source_port,
                rank=0,
                world_size=2,
                group_name=self.group_name,
            )
            init_process_group_time = time.perf_counter() - start_init_process_group
            logger.info(
                "Finish init_process_group, name: %s, addr: %s:%s", self.world_name, self.source_ip, self.source_port
            )
            logger.info(
                "[netloader_p2p] init_process_group time: %s, group=%s, addr=%s",
                init_process_group_time,
                self.group_name,
                f"{self.source_ip}:{self.source_port}",
            )

            start_get_transfer_items = time.perf_counter()
            transfer_items = _get_recv_transfer_items(model, self.transfer_processed_layout)
            get_transfer_items_time = time.perf_counter() - start_get_transfer_items
            transfer_count = len(transfer_items)
            recv_addr = f"{self.source_ip}:{self.source_port}"
            _log_transfer_plan("recv", transfer_count, self.group_name, self.transfer_processed_layout, recv_addr)
            _log_transfer_items_debug("recv_before", transfer_items, self.group_name)
            if self.transfer_shape_manifest:
                server_names = list(self.transfer_shape_manifest.keys())
                _log_transfer_order_diff(
                    "recv_before",
                    [name for name, _ in transfer_items],
                    server_names,
                    self.group_name,
                )
            logger.info(
                "Start recv, name: %s, addr: %s:%s",
                self.world_name,
                self.source_ip,
                self.source_port,
            )
            logger.info("Model device: %s", model_device)

            start_hccl_recv = time.perf_counter()
            if self.transfer_processed_layout:
                log_netloader_debug_checkpoint(
                    "p2p_recv_hccl_loop_begin",
                    role="recv",
                    transfer_count=transfer_count,
                    group=self.group_name,
                    addr=recv_addr,
                    stream=_safe_current_npu_stream(),
                )
                for index, (name, tensor) in enumerate(transfer_items):
                    if _should_log_transfer_progress(index, transfer_count):
                        logger.info(
                            "[netloader_p2p][debug] recv begin %s",
                            _format_transfer_tensor_debug(name, tensor, index),
                        )
                    try:
                        recv_buffer, restore_fractal_nz = _prepare_hccl_recv_buffer(tensor)
                        receiver_pg.recv([recv_buffer], 1, 0).wait()
                        _finalize_hccl_recv_buffer(tensor, recv_buffer, restore_fractal_nz)
                    except Exception as exc:
                        logger.error(
                            "[netloader_p2p][debug] recv failed at %s: %s",
                            _format_transfer_tensor_debug(name, tensor, index),
                            exc,
                        )
                        raise
                log_netloader_debug_checkpoint(
                    "p2p_recv_hccl_loop_end",
                    role="recv",
                    transfer_count=transfer_count,
                    group=self.group_name,
                    addr=recv_addr,
                    stream=_safe_current_npu_stream(),
                )
                log_netloader_debug_checkpoint(
                    "p2p_recv_barrier_begin",
                    role="recv",
                    group=self.group_name,
                    addr=recv_addr,
                )
                _barrier_after_p2p_transfer(receiver_pg)
                log_netloader_debug_checkpoint(
                    "p2p_recv_barrier_end",
                    role="recv",
                    group=self.group_name,
                    addr=recv_addr,
                )
                synchronize_npu_with_debug("p2p_recv_post_hccl")
            else:
                trans_stream = torch_npu.npu.Stream()
                recv_buffer_stats = {"contiguous_replaced": 0, "contiguous_reused": 0}
                with torch_npu.npu.stream(trans_stream):
                    for index, (name, tensor) in enumerate(transfer_items):
                        if _should_log_transfer_progress(index, transfer_count):
                            logger.info(
                                "[netloader_p2p][debug] recv begin %s",
                                _format_transfer_tensor_debug(name, tensor, index),
                            )
                        was_contiguous = tensor.is_contiguous()
                        try:
                            recv_buffer, restore_fractal_nz = _prepare_hccl_recv_buffer(tensor)
                        except Exception as exc:
                            logger.error(
                                "[netloader_p2p][debug] recv buffer prepare failed at %s: %s",
                                _format_transfer_tensor_debug(name, tensor, index),
                                exc,
                            )
                            raise
                        if was_contiguous and not restore_fractal_nz:
                            recv_buffer_stats["contiguous_reused"] += 1
                        else:
                            recv_buffer_stats["contiguous_replaced"] += 1
                            if not restore_fractal_nz:
                                logger.info(
                                    "[netloader_p2p][debug] recv replaced non-contiguous storage at %s",
                                    _format_transfer_tensor_debug(name, tensor, index),
                                )
                        try:
                            receiver_pg.recv([recv_buffer], 1, 0).wait()
                            _finalize_hccl_recv_buffer(tensor, recv_buffer, restore_fractal_nz)
                        except Exception as exc:
                            logger.error(
                                "[netloader_p2p][debug] recv failed at %s buffer=%s: %s",
                                _format_transfer_tensor_debug(name, tensor, index),
                                _format_transfer_tensor_debug(name, recv_buffer, index),
                                exc,
                            )
                            raise
                    _barrier_after_p2p_transfer(receiver_pg)
                    torch_npu.npu.synchronize(trans_stream)
                logger.info(
                    "[netloader_p2p][debug] recv contiguous stats group=%s replaced=%s reused=%s",
                    self.group_name,
                    recv_buffer_stats["contiguous_replaced"],
                    recv_buffer_stats["contiguous_reused"],
                )
            hccl_recv_time = time.perf_counter() - start_hccl_recv

            post_recv_reshape_time = 0.0
            if self.transfer_processed_layout:
                start_post_recv_reshape = time.perf_counter()
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
                post_recv_reshape_time = time.perf_counter() - start_post_recv_reshape

            logger.info(
                "[netloader_p2p] HCCL recv time: %s, get_transfer_items time: %s, post-recv reshape time: %s, "
                "transfer=%s group=%s processed_layout=%s addr=%s",
                hccl_recv_time,
                get_transfer_items_time,
                post_recv_reshape_time,
                transfer_count,
                self.group_name,
                self.transfer_processed_layout,
                recv_addr,
            )
            logger.info(
                "[netloader_p2p] recv done transfer=%s group=%s processed_layout=%s addr=%s",
                transfer_count,
                self.group_name,
                self.transfer_processed_layout,
                recv_addr,
            )
            logger.info("Finish recv, name: %s, addr: %s:%s", self.world_name, self.source_ip, self.source_port)
            loaded_model = model
        except Exception as e:
            logger.error(
                "[netloader_p2p] recv failed transfer=%s group=%s processed_layout=%s addr=%s:%s: %s",
                transfer_count if "transfer_count" in locals() else "unknown",
                self.group_name,
                self.transfer_processed_layout,
                self.source_ip,
                self.source_port,
                e,
            )
            logger.error("Failed to recv model: %s", e)
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
        logger.info("Start init_process_group, name: %s, addr: %s:%s", self.comm_name, self.listen_ip, self.listen_port)
        sender_pg = None
        transfer_count = 0
        send_addr = f"{self.listen_ip}:{self.listen_port}"
        try:
            sender_pg = stateless_init_process_group(
                host=self.comm_name.split(":")[0],
                port=self.listen_port,
                rank=1,
                world_size=2,
                group_name=self.group_name,
            )
            logger.info(
                "Finish init_process_group, name: %s, addr: %s:%s", self.comm_name, self.listen_ip, self.listen_port
            )

            if self.send_processed_weights and registered_transfer_items is None:
                raise RuntimeError(
                    "Processed-layout P2P send requires registered transfer items on the server. "
                    "Call ElasticServer.register_transfer_manifest() after process_weights."
                )

            transfer_items = _get_send_transfer_items(
                model,
                self.send_processed_weights,
                registered_transfer_items,
            )
            transfer_count = len(transfer_items)
            use_int8_backup = not self.send_processed_weights
            _log_transfer_plan("send", transfer_count, self.group_name, self.send_processed_weights, send_addr)
            _log_transfer_items_debug("send", transfer_items, self.group_name)
            if self.send_processed_weights:
                logger.info(
                    "[netloader_p2p] send uses registered manifest transfer=%s group=%s",
                    transfer_count,
                    self.group_name,
                )
            logger.info(
                "Start send, name: %s, addr: %s:%s",
                self.comm_name,
                self.listen_ip,
                self.listen_port,
            )
            logger.info("Model device: %s", model_device)

            if self.send_processed_weights:
                log_netloader_debug_checkpoint(
                    "p2p_send_hccl_loop_begin",
                    role="send",
                    transfer_count=transfer_count,
                    group=self.group_name,
                    addr=f"{self.listen_ip}:{self.listen_port}",
                    stream=_safe_current_npu_stream(),
                )
                for index, (name, tensor_ref) in enumerate(transfer_items):
                    try:
                        send_tensor = _hccl_transfer_tensor(tensor_ref)
                        if _should_log_transfer_progress(index, transfer_count):
                            logger.info(
                                "[netloader_p2p][debug] send begin index=%s name=%s src_contiguous=%s "
                                "payload_contiguous=%s payload_npu_format=%s %s",
                                index,
                                name,
                                tensor_ref.is_contiguous(),
                                send_tensor.is_contiguous(),
                                _safe_get_npu_format(send_tensor),
                                _format_transfer_tensor_debug(name, send_tensor),
                            )
                        sender_pg.send([send_tensor], 0, 0).wait()
                    except Exception as exc:
                        logger.error(
                            "[netloader_p2p][debug] send failed at index=%s name=%s tensor=%s: %s",
                            index,
                            name,
                            _format_transfer_tensor_debug(name, tensor_ref, index),
                            exc,
                        )
                        raise
                log_netloader_debug_checkpoint(
                    "p2p_send_hccl_loop_end",
                    role="send",
                    transfer_count=transfer_count,
                    group=self.group_name,
                    addr=f"{self.listen_ip}:{self.listen_port}",
                    stream=_safe_current_npu_stream(),
                )
                log_netloader_debug_checkpoint(
                    "p2p_send_barrier_begin",
                    role="send",
                    group=self.group_name,
                    addr=f"{self.listen_ip}:{self.listen_port}",
                )
                _barrier_after_p2p_transfer(sender_pg)
                log_netloader_debug_checkpoint(
                    "p2p_send_barrier_end",
                    role="send",
                    group=self.group_name,
                    addr=f"{self.listen_ip}:{self.listen_port}",
                )
                synchronize_npu_with_debug("p2p_send_post_hccl")
            else:
                trans_stream = torch_npu.npu.Stream()
                with torch_npu.npu.stream(trans_stream):
                    for index, (name, tensor_ref) in enumerate(transfer_items):
                        try:
                            if use_int8_backup and name in int8_params:
                                payload = int8_params[name].to(model_device)
                                if _should_log_transfer_progress(index, transfer_count):
                                    logger.info(
                                        "[netloader_p2p][debug] send begin index=%s name=%s source=int8_backup shape=%s",
                                        index,
                                        name,
                                        tuple(payload.shape),
                                    )
                                sender_pg.send([payload], 0, 0).wait()
                            else:
                                send_tensor = _hccl_transfer_tensor(tensor_ref)
                                if _should_log_transfer_progress(index, transfer_count):
                                    logger.info(
                                        "[netloader_p2p][debug] send begin index=%s name=%s src_contiguous=%s "
                                        "payload_contiguous=%s payload_npu_format=%s %s",
                                        index,
                                        name,
                                        tensor_ref.is_contiguous(),
                                        send_tensor.is_contiguous(),
                                        _safe_get_npu_format(send_tensor),
                                        _format_transfer_tensor_debug(name, send_tensor),
                                    )
                                sender_pg.send([send_tensor], 0, 0).wait()
                        except Exception as exc:
                            logger.error(
                                "[netloader_p2p][debug] send failed at index=%s name=%s tensor=%s: %s",
                                index,
                                name,
                                _format_transfer_tensor_debug(name, tensor_ref, index),
                                exc,
                            )
                            raise
                    _barrier_after_p2p_transfer(sender_pg)
                torch_npu.npu.synchronize(trans_stream)
            logger.info(
                "[netloader_p2p] send done transfer=%s group=%s processed_layout=%s addr=%s",
                transfer_count,
                self.group_name,
                self.send_processed_weights,
                send_addr,
            )
            logger.info("Finish send, name: %s, addr: %s:%s", self.comm_name, self.listen_ip, self.listen_port)
        except Exception as e:
            logger.error(
                "[netloader_p2p] send failed transfer=%s group=%s processed_layout=%s addr=%s comm_name=%s: %s",
                transfer_count if transfer_count else "unknown",
                self.group_name,
                self.send_processed_weights,
                send_addr,
                self.comm_name,
                e,
            )
            raise
        finally:
            if sender_pg:
                destroy_stateless_process_group(sender_pg)
