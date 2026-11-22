# SPDX-License-Identifier: Apache-2.0
"""Refresh the private recurrent-state pages of compatible GDN groups in one launch."""

from typing import TYPE_CHECKING

import torch
from vllm.triton_utils import tl, triton

if TYPE_CHECKING:
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

    from vllm_ascend.ops.gdn_attn_builder import AscendGDNHostMetadataBuilder


@triton.jit
def _refresh_gdn_tree_state_page_indices_kernel(pointers, columns, SLOTS: tl.constexpr, BLOCK: tl.constexpr):
    # Each program owns both outputs of one group, including their partial tail.
    row = pointers + 3 * tl.program_id(0)
    src = tl.load(row).to(tl.pointer_type(tl.int32))
    dst = tl.load(row + 1).to(tl.pointer_type(tl.int32))
    init = tl.load(row + 2).to(tl.pointer_type(tl.int32))
    i = tl.arange(0, BLOCK)
    parent = tl.load(columns + i, mask=i < SLOTS, other=0).to(tl.int32)
    tl.store(dst + i, tl.load(src + i, mask=i < SLOTS, other=0), mask=i < SLOTS)
    tl.store(init + i, tl.load(src + parent, mask=i < SLOTS, other=0), mask=i < SLOTS)


class GDNTreePagePointers:
    """Model-owned address table. Page values and logical parents change every step."""

    def __init__(self):
        self.signature = ()
        self.pointers = None
        self.buffers = ()

    def refresh(self, columns, rows):
        signature = tuple((src.data_ptr(), dst.data_ptr(), init.data_ptr()) for src, dst, init in rows)
        if self.signature != signature:
            outputs = [ptr for _, dst, init in signature for ptr in (dst, init)]
            if len(set(outputs)) != len(outputs) or set(outputs).intersection(src for src, _, _ in signature):
                raise ValueError("GDN groups require private output page tables")
            host = torch.tensor(signature, device="cpu", dtype=torch.int64, pin_memory=True)
            self.pointers = host.to(columns.device, non_blocking=True)
            self.buffers = tuple(rows)  # Keep the allocations referenced by the device pointer table alive.
            self.signature = signature
        slots = columns.numel()
        _refresh_gdn_tree_state_page_indices_kernel[(len(rows),)](
            self.pointers, columns, slots, triton.next_power_of_2(slots), multibuffer=False, unit_flag=False
        )


class GDNTreePageBatch:
    """One complete verification row; finish page writes before graph replay."""

    # Only the 32-node tree plus root has been validated for this fast path.
    SLOTS = 33

    def __init__(self, pointers: GDNTreePagePointers):
        self.pointers = pointers
        self.rows = []
        self.finalizers = []

    def add(self, columns, src, builder: "AscendGDNHostMetadataBuilder", metadata: "GDNAttentionMetadata") -> bool:
        dst = metadata.spec_state_indices_tensor
        init = metadata.spec_decode_metadata.init_state_indices
        if (
            src.device.type != "npu"
            or not (src.device == columns.device == dst.device == init.device)
            or not (src.dtype == dst.dtype == init.dtype == torch.int32)
            or columns.dtype not in (torch.int64, torch.int32)
            or not (columns.shape == dst.shape == init.shape == (1, self.SLOTS))
            or src.ndim != 2
            or src.shape[0] != 1
            or src.shape[1] < self.SLOTS
            or not (src.stride(1) == columns.stride(1) == dst.stride(1) == init.stride(1) == 1)
        ):
            return False
        self.rows.append((src, dst, init))
        self.finalizers.append((builder, metadata))
        return True

    def finish(self, columns):
        if self.rows:
            self.pointers.refresh(columns, self.rows)
            # Convolution finalization can read the newly written page indices.
            for builder, metadata in self.finalizers:
                builder._finalize_metadata(metadata)
            self.rows.clear()
            self.finalizers.clear()
