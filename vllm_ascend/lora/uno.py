# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Pack the fixed UNO Tree adapter once, retaining FP32 shrink arithmetic."""

import torch


def uno_lora_weight_key(lora_a_stacked, lora_b_stacked):
    return tuple(weight.data_ptr() for weight in (*lora_a_stacked, *lora_b_stacked))


class UnoPackedLoRA:
    """One GEMM per shrink/expand, with disjoint rank blocks for packed slices.

    The adapter owns its only LoRA slot. Refresh after every adapter load and
    preserve destination addresses so already captured graphs see new weights.
    """

    def __init__(self, lora_a_stacked, lora_b_stacked, output_slices):
        self.output_slices = tuple(output_slices)
        self.key = uno_lora_weight_key(lora_a_stacked, lora_b_stacked)
        self._source_a = tuple(lora_a_stacked)
        self._source_b = tuple(lora_b_stacked)
        aa, bb = self._weights()
        self.a = torch.empty((sum(a.shape[0] for a in aa), aa[0].shape[1]), device=aa[0].device, dtype=torch.float32)
        self.b = torch.empty((sum(self.output_slices), self.a.shape[0]), device=bb[0].device, dtype=bb[0].dtype)
        self.refresh()

    def _weights(self):
        aa, bb = [], []
        if not self._source_a or len(self._source_a) != len(self._source_b):
            raise ValueError("UNO packed LoRA requires paired nonempty A/B slices.")
        if len(self.output_slices) != len(self._source_a):
            raise ValueError("UNO packed LoRA output slices do not match the adapter.")
        for a, b, size in zip(self._source_a, self._source_b, self.output_slices):
            if a.ndim != 4 or b.ndim != 4 or a.shape[:2] != (1, 1) or b.shape[:2] != (1, 1):
                raise ValueError("UNO packed LoRA requires a single adapter slot.")
            a, b = a[0, 0], b[0, 0]
            if b.shape != (size, a.shape[0]) or (aa and a.shape[1] != aa[0].shape[1]):
                raise ValueError("UNO packed LoRA has incompatible projection shapes.")
            aa.append(a)
            bb.append(b)
        return aa, bb

    @torch.no_grad()
    def refresh(self):
        aa, bb = self._weights()
        self.a.copy_(torch.cat(aa, dim=0).float())
        self.b.zero_()
        out_start = rank_start = 0
        for a, b in zip(aa, bb):
            out_end, rank_end = out_start + b.shape[0], rank_start + a.shape[0]
            self.b[out_start:out_end, rank_start:rank_end].copy_(b)
            out_start, rank_start = out_end, rank_end

    def apply(self, y, x, scale, row_mask):
        shrunk = torch.mm(x.float(), self.a.t())
        if scale != 1.0:
            shrunk *= scale
        if row_mask is not None:
            shrunk *= row_mask[: x.shape[0]]
        y.add_(torch.mm(shrunk.to(y.dtype), self.b.to(y.dtype).t()))


def prepare_uno_packed_lora(model):
    """Called only for UNO Tree, after the fixed adapter has been activated."""
    prepared = 0
    for module in model.modules():
        aa, bb = getattr(module, "lora_a_stacked", None), getattr(module, "lora_b_stacked", None)
        if not isinstance(aa, tuple) or not isinstance(bb, tuple) or not aa:
            continue
        wrapper = module.punica_wrapper
        key = uno_lora_weight_key(aa, bb)
        cached = wrapper._uno_packed_lora.get(key)
        if cached is None:
            cached = UnoPackedLoRA(aa, bb, module.output_slices)
            wrapper._uno_packed_lora[key] = cached
        else:
            cached.refresh()
        prepared += 1
    if not prepared:
        raise ValueError("UNO Tree found no fixed LoRA linear projections to prepare.")
    return prepared
