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
        synchronize=True,
    ) -> None:
        if not expert_loads and not resident_loads:
            return
        if resident_loads and resident_weights is None:
            raise ValueError("resident weights are required for device-to-device expert copies")
        is_npu = torch.device(device).type == "npu"
        if is_npu:
            # Keep ACL optional for disabled offload and CPU-only tests.
            import acl  # type: ignore[import-not-found]

            if resident_loads:
                # Prefill copies most experts into a full, expert-ID-ordered
                # pool. Adjacent slots in the same source allocation can use a
                # single ACL submission while retaining their physical bytes.
                copy_parameters = TorchExpertTransfer._prefill_copy_parameters(
                    host_store, destination_weights, expert_loads, resident_weights, resident_loads
                )
            else:
                copy_parameters = [
                    (
                        *TorchExpertTransfer._host_copy_parameters(
                            destination_weights[name], prepared, expert_id, slot
                        ),
                        ACL_MEMCPY_HOST_TO_DEVICE,
                    )
                    for expert_id, slot in expert_loads
                    for name, prepared in host_store.prepared.items()
                ]
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
    def _host_copy_parameters(destination_tensor, prepared, expert_id, slot):
        source_chunk, row = prepared.source_chunk(expert_id)
        expert_bytes = prepared.expert_bytes
        storage = destination_tensor.untyped_storage()
        return storage.data_ptr() + slot * expert_bytes, source_chunk.data_ptr() + row * expert_bytes, expert_bytes

    @staticmethod
    def _prefill_copy_parameters(host_store, destination_weights, expert_loads, resident_weights, resident_loads):
        parameters = []

        def append_runs(copies, direction):
            current = None
            source_allocation = None
            for dst, src, size, source in copies:
                if (
                    current is not None
                    and source is source_allocation
                    and current[0] + current[2] == dst
                    and current[1] + current[2] == src
                ):
                    current = (current[0], current[1], current[2] + size, direction)
                else:
                    if current is not None:
                        parameters.append(current)
                    current = (dst, src, size, direction)
                    source_allocation = source
            if current is not None:
                parameters.append(current)

        def host_copies(prepared, destination):
            expert_bytes = prepared.expert_bytes
            for expert_id, slot in expert_loads:
                chunk, row = prepared.source_chunk(expert_id)
                yield destination + slot * expert_bytes, chunk.data_ptr() + row * expert_bytes, expert_bytes, chunk

        for name, prepared in host_store.prepared.items():
            destination = destination_weights[name].untyped_storage().data_ptr()
            expert_bytes = prepared.expert_bytes
            append_runs(host_copies(prepared, destination), ACL_MEMCPY_HOST_TO_DEVICE)
            resident = resident_weights[name]
            source = resident.untyped_storage().data_ptr()
            append_runs(
                (
                    (destination + dst * expert_bytes, source + src * expert_bytes, expert_bytes, resident)
                    for src, dst in resident_loads
                ),
                ACL_MEMCPY_DEVICE_TO_DEVICE,
            )
        return parameters
