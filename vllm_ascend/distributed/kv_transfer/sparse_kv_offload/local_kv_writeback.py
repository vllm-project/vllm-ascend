# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch_npu


class LocalKVWriteback:
    """Persistent fresh KV, with a fork/join writeback stream on the owner.

    Staging is per layer. D2H descriptors may be shared across layers only
    because their construction AND consumption are ordered on the same stream.
    Every forward must join the stream before these buffers can be reused.
    """

    def __init__(self, num_layers, max_tokens, k_width, v_width, device, dtype, is_writer):
        self.k = [torch.empty((max_tokens, k_width), device=device, dtype=dtype) for _ in range(num_layers)]
        self.v = [torch.empty((max_tokens, v_width), device=device, dtype=dtype) for _ in range(num_layers)]
        self.slots = [torch.empty(max_tokens, device=device, dtype=torch.int64) for _ in range(num_layers)]
        self.stream = torch_npu.npu.Stream(device=device) if is_writer else None
        self.ready = [torch_npu.npu.Event() for _ in range(num_layers)] if is_writer else []

    def stage(self, layer_id, k, v, slots):
        count = slots.numel()
        if count > self.k[layer_id].shape[0]:
            raise ValueError("Fresh KV rows exceed the configured staging capacity")
        k_rows = k.reshape(-1, self.k[layer_id].shape[1])
        v_rows = v.reshape(-1, self.v[layer_id].shape[1])
        if k_rows.shape[0] != count or v_rows.shape[0] != count:
            raise ValueError("Fresh K/V and slot_mapping row counts must match")
        staged_k = self.k[layer_id][:count]
        staged_v = self.v[layer_id][:count]
        staged_slots = self.slots[layer_id][:count]
        staged_k.copy_(k_rows)
        staged_v.copy_(v_rows)
        staged_slots.copy_(slots.reshape(-1))
        return staged_k, staged_v, staged_slots

    def submit(self, layer_id, writeback, **kwargs):
        if self.stream is None:
            return
        self.ready[layer_id].record(torch_npu.npu.current_stream())
        self.stream.wait_event(self.ready[layer_id])
        with torch_npu.npu.stream(self.stream):
            writeback(**kwargs)

    def finish_forward(self):
        if self.stream is not None:
            # Join INSIDE the captured forward, after its last attention.
            # This orders the next forward's staging, history loads and its
            # TP visibility broadcast after every outstanding writeback.
            torch_npu.npu.current_stream().wait_stream(self.stream)
