# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch

ACL_MEMCPY_HOST_TO_DEVICE = 1


class TorchExpertTransfer:
    """Batch native H2D copies on the caller's stream, fencing before publication."""

    @staticmethod
    def copy_experts(adapter, host_store, destination_weights, expert_loads, device) -> None:
        if not expert_loads:
            return
        is_npu = torch.device(device).type == "npu"
        if is_npu:
            # Keep ACL optional for disabled offload and CPU-only tests.
            import acl  # type: ignore[import-not-found]

            copy_parameters = [
                TorchExpertTransfer._host_copy_parameters(destination_weights[name], prepared, expert_id, slot)
                for expert_id, slot in expert_loads
                for name, prepared in host_store.prepared.items()
            ]
            stream = torch.npu.current_stream(device)
            try:
                for dst_ptr, src_ptr, expert_bytes in copy_parameters:
                    error = acl.rt.memcpy_async(
                        dst_ptr, expert_bytes, src_ptr, expert_bytes, ACL_MEMCPY_HOST_TO_DEVICE, stream.npu_stream
                    )
                    if error:
                        raise RuntimeError(f"asynchronous expert host copy failed: ACL error {error}")
            finally:
                # Also drain partially submitted batches on failure. Host bytes
                # remain owned until completion; callers publish maps afterwards.
                stream.synchronize()
            return

        # CPU contract tests use the exact scheme without NPU native storage.
        for expert_id, slot in expert_loads:
            expert_weights = adapter.load_experts(host_store, (expert_id,), device)
            for name, tensor in expert_weights.items():
                destination_weights[name][slot].copy_(tensor[0])

    @staticmethod
    def _host_copy_parameters(destination_tensor, prepared, expert_id, slot):
        source_chunk, row = prepared.source_chunk(expert_id)
        expert_bytes = prepared.expert_bytes
        storage = destination_tensor.untyped_storage()
        return storage.data_ptr() + slot * expert_bytes, source_chunk.data_ptr() + row * expert_bytes, expert_bytes
