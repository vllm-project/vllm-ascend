# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch

ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_DEVICE = 3


class TorchExpertTransfer:
    """Batch native expert copies on the caller's stream, fencing before publication."""

    @staticmethod
    def copy_experts(
        adapter,
        host_store,
        destination_weights,
        expert_loads,
        device,
        *,
        resident_weights=None,
        resident_loads=(),
        stream=None,
    ) -> None:
        if not expert_loads and not resident_loads:
            return
        if resident_loads and resident_weights is None:
            raise ValueError("resident weights are required for device-to-device expert copies")
        is_npu = torch.device(device).type == "npu"
        if is_npu:
            # Keep ACL optional for disabled offload and CPU-only tests.
            import acl  # type: ignore[import-not-found]

            copy_parameters = TorchExpertTransfer._copy_parameters(
                host_store, destination_weights, expert_loads, resident_weights, resident_loads
            )
            synchronize = stream is None
            stream = stream if stream is not None else torch.npu.current_stream(device)
            try:
                for dst_ptr, src_ptr, expert_bytes, direction in copy_parameters:
                    error = acl.rt.memcpy_async(
                        dst_ptr, expert_bytes, src_ptr, expert_bytes, direction, stream.npu_stream
                    )
                    if error:
                        kind = "host" if direction == ACL_MEMCPY_HOST_TO_DEVICE else "device"
                        raise RuntimeError(f"asynchronous expert {kind} copy failed: ACL error {error}")
            except BaseException:
                # A failed partial batch must be drained before the pool can be reused.
                stream.synchronize()
                raise
            if synchronize:
                stream.synchronize()
            return

        # CPU contract tests use the exact scheme without NPU native storage.
        for expert_id, slot in expert_loads:
            expert_weights = adapter.load_experts(host_store, (expert_id,), device)
            for name, tensor in expert_weights.items():
                destination_weights[name][slot].copy_(tensor[0])
        if resident_loads:
            for source_slot, destination_slot in resident_loads:
                for name, tensor in resident_weights.items():
                    destination_weights[name][destination_slot].copy_(tensor[source_slot])

    @staticmethod
    def _copy_parameters(host_store, destination_weights, expert_loads, resident_weights, resident_loads):
        parameters = []
        for name, prepared in host_store.prepared.items():
            destination = destination_weights[name].untyped_storage().data_ptr()
            expert_bytes = prepared.expert_bytes
            copies = []
            for expert_id, slot in expert_loads:
                chunk, row = prepared.source_chunk(expert_id)
                copies.append((slot, chunk.data_ptr() + row * expert_bytes, ACL_MEMCPY_HOST_TO_DEVICE, chunk))
            if resident_loads:
                resident = resident_weights[name]
                source = resident.untyped_storage().data_ptr()
                copies.extend(
                    (dst, source + src * expert_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, resident)
                    for src, dst in resident_loads
                )
            previous_source = None
            for slot, source, direction, allocation in copies:
                destination_ptr = destination + slot * expert_bytes
                # Merge only adjacent slots within one source allocation and tensor.
                if (
                    allocation is previous_source
                    and parameters[-1][0] + parameters[-1][2] == destination_ptr
                    and parameters[-1][1] + parameters[-1][2] == source
                    and parameters[-1][3] == direction
                ):
                    dst, src, size, _ = parameters[-1]
                    parameters[-1] = (dst, src, size + expert_bytes, direction)
                else:
                    parameters.append((destination_ptr, source, expert_bytes, direction))
                previous_source = allocation
        return parameters
