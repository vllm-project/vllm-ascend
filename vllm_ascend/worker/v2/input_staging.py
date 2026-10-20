# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import numpy as np
import torch

_MAX_STAGING_BANKS = 2


class BatchInputStaging:
    """Reuse pinned inputs on one execution stream without waiting on the host.

    Only single-stream, non-PP/non-PCP input preparation uses this staging.
    Device consumers and subsequent writes are ordered on the owning stream.
    Two host banks permit another submission while the previous H2D is pending;
    if both are busy, the caller uses its ordinary allocation/copy path.
    CPU batch metadata retains its own storage, separate from these H2D banks.
    """

    def __init__(self, max_num_reqs: int, device: torch.device):
        self.max_num_reqs = max_num_reqs
        self.device = device
        self.stream = None
        self.banks: list[tuple[torch.Tensor, torch.Tensor, torch.npu.Event]] = []

    def copy(
        self,
        idx_mapping: np.ndarray,
        cu_num_logits: np.ndarray,
        query_start_loc: np.ndarray,
        query_out: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        stream = torch.npu.current_stream()
        if self.stream is not None and self.stream != stream:
            return None
        num_reqs = len(idx_mapping)
        if (
            num_reqs > self.max_num_reqs
            or len(cu_num_logits) > self.max_num_reqs + 1
            or len(query_start_loc) > self.max_num_reqs + 2
            or idx_mapping.dtype != np.intp
            or any(array.dtype != np.int32 for array in (cu_num_logits, query_start_loc))
        ):
            return None

        bank = next((bank for bank in self.banks if bank[2].query()), None)
        if bank is None:
            if len(self.banks) == _MAX_STAGING_BANKS:
                return None
            self.stream = stream
            mapping_bytes = self.max_num_reqs * np.dtype(np.intp).itemsize
            device_bytes = mapping_bytes + (self.max_num_reqs + 1) * 4
            host_bytes = device_bytes + (self.max_num_reqs + 2) * 4
            host = torch.zeros(host_bytes, dtype=torch.uint8, pin_memory=self.device.type != "cpu")
            device = torch.empty(device_bytes, dtype=torch.uint8, device=self.device)
            bank = (host, device, torch.npu.Event())
            self.banks.append(bank)
        host, device, event = bank
        logits_offset = self.max_num_reqs * np.dtype(np.intp).itemsize
        query_offset = logits_offset + (self.max_num_reqs + 1) * 4
        host_np = host.numpy()
        host_np[:logits_offset].view(np.intp)[:num_reqs] = idx_mapping
        host_np[logits_offset:query_offset].view(np.int32)[: len(cu_num_logits)] = cu_num_logits
        host_np[query_offset:].view(np.int32)[: len(query_start_loc)] = query_start_loc
        # Retain the captured query buffer's address. Mapping/logit boundaries
        # share one H2D submission; query boundaries use their existing buffer.
        device.copy_(host[:query_offset], non_blocking=True)
        query_out.copy_(host[query_offset:].view(torch.int32)[: len(query_start_loc)], non_blocking=True)
        event.record(stream)
        mapping_dtype = torch.int64 if np.dtype(np.intp).itemsize == 8 else torch.int32
        return (
            device[:logits_offset].view(mapping_dtype)[:num_reqs],
            device[logits_offset:].view(torch.int32)[: len(cu_num_logits)],
        )
