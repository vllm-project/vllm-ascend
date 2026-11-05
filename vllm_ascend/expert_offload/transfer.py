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

            copies = [
                TorchExpertTransfer._host_copy_parameters(destination[name], prepared, expert_id, slot)
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
            bundle = adapter.materialize(host_store, (expert_id,), device)
            for name, tensor in bundle.items():
                destination[name][slot].copy_(tensor[0])

    @staticmethod
    def _host_copy_parameters(destination, prepared, expert_id, slot):
        source, row = prepared.source_chunk(expert_id)
        size = prepared.expert_bytes
        storage = destination.untyped_storage()
        return storage.data_ptr() + slot * size, source.data_ptr() + row * size, size

    @staticmethod
    def _physical_copy_parameters(destination, source, slot, torch_npu):
        """Copy one complete physical expert block without interpreting NZ as ND.

        Both tensors come from the same adapter. Its leading expert dimension is
        an unpadded batch axis; NZ padding belongs to each expert's trailing matrix.
        Scales retain their original transposed strides and packed byte layout.
        """
        size = int(torch_npu.get_storage_size(source)) * source.element_size()
        dst_storage = destination.untyped_storage()
        src_storage = source.untyped_storage()
        return dst_storage.data_ptr() + slot * size, src_storage.data_ptr(), size
