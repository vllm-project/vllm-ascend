# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch

ACL_MEMCPY_HOST_TO_DEVICE = 1


class TorchExpertTransfer:
    """Batch native H2D copies on the caller's stream, fencing before publication."""

    @staticmethod
    def copy_experts(adapter, host_store, destination, loads, device) -> None:
        if not loads:
            return
        is_npu = torch.device(device).type == "npu"
        if is_npu:
            # Keep ACL optional for disabled offload and CPU-only tests.
            import acl  # type: ignore[import-not-found]
            import torch_npu

            if set(host_store.prepared) != set(destination):
                raise RuntimeError("incomplete prepared expert transfer bundle")
            # Validate the entire batch before touching any resident storage.
            copies = [
                TorchExpertTransfer._host_copy_parameters(destination[name], prepared, expert_id, slot, torch_npu)
                for expert_id, slot in loads
                for name, prepared in host_store.prepared.items()
            ]
            stream = torch.npu.current_stream(device)
            try:
                for dst_ptr, src_ptr, size in copies:
                    error = acl.rt.memcpy_async(
                        dst_ptr, size, src_ptr, size, ACL_MEMCPY_HOST_TO_DEVICE, stream.npu_stream
                    )
                    if error:
                        raise RuntimeError(f"asynchronous expert host copy failed: ACL error {error}")
            finally:
                # Also drain partially submitted batches on failure. Host bytes
                # remain owned until completion; callers publish maps afterwards.
                stream.synchronize()
            return

        # CPU contract tests use the exact scheme without NPU native storage.
        for expert_id, slot in loads:
            if type(slot) is not int or any(not 0 <= slot < tensor.shape[0] for tensor in destination.values()):
                raise ValueError("expert destination slot is out of range")
            bundle = adapter.materialize(host_store, (expert_id,), device)
            if set(bundle) != set(destination):
                raise RuntimeError("incomplete expert transfer bundle")
            for name, tensor in bundle.items():
                destination[name][slot].copy_(tensor[0])

    @staticmethod
    def _host_copy_parameters(destination, prepared, expert_id, slot, torch_npu):
        if (
            destination.device != prepared.device
            or destination.dtype != prepared.dtype
            or tuple(destination.shape[1:]) != prepared.shape
            or tuple(destination.stride()) != prepared.stride
            or destination.storage_offset() != 0
            or torch_npu.get_npu_format(destination) != prepared.npu_format
        ):
            raise ValueError("incompatible expert physical layouts")
        if type(slot) is not int or not 0 <= slot < destination.shape[0]:
            raise ValueError("expert destination slot is out of range")
        if type(expert_id) is not int or not 0 <= expert_id < prepared.data.shape[0]:
            raise ValueError("logical expert is out of range")
        size = prepared.expert_bytes
        storage = destination.untyped_storage()
        if (
            size <= 0
            or int(torch_npu.get_storage_size(destination)) * destination.element_size() != size * destination.shape[0]
            or storage.nbytes() < size * destination.shape[0]
            or prepared.data.device.type != "cpu"
            or prepared.data.dtype != torch.uint8
            or prepared.data.shape[1:] != (size,)
            or not prepared.data.is_contiguous()
            or not prepared.data.is_pinned()
        ):
            raise ValueError("expert physical storage does not contain uniform pinned bounded blocks")
        return storage.data_ptr() + slot * size, prepared.data.data_ptr() + expert_id * size, size

    @staticmethod
    def _physical_copy_parameters(destination, source, slot, torch_npu):
        """Copy one complete physical expert block without interpreting NZ as ND.

        Both tensors come from the same adapter. Its leading expert dimension is
        an unpadded batch axis; NZ padding belongs to each expert's trailing matrix.
        Scales retain their original transposed strides and packed byte layout.
        """
        if (
            len(source.shape) not in (3, 4)
            or len(destination.shape) != len(source.shape)
            or destination.device != source.device
            or destination.dtype != source.dtype
            or source.shape[0] != 1
            or destination.shape[1:] != source.shape[1:]
            or destination.stride() != source.stride()
            or destination.storage_offset() != 0
            or source.storage_offset() != 0
            or torch_npu.get_npu_format(destination) != torch_npu.get_npu_format(source)
        ):
            raise ValueError("incompatible expert physical layouts")
        if type(slot) is not int or not 0 <= slot < destination.shape[0]:
            raise ValueError("expert destination slot is out of range")
        size = int(torch_npu.get_storage_size(source)) * source.element_size()
        total = int(torch_npu.get_storage_size(destination)) * destination.element_size()
        dst_storage = destination.untyped_storage()
        src_storage = source.untyped_storage()
        if (
            size <= 0
            or total != size * destination.shape[0]
            or total > dst_storage.nbytes()
            or size > src_storage.nbytes()
        ):
            raise ValueError("expert physical storage does not contain uniform bounded blocks")
        return dst_storage.data_ptr() + slot * size, src_storage.data_ptr(), size
