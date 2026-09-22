# SPDX-License-Identifier: Apache-2.0

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _copy_mla_kv_kernel(
    source,
    destination,
    source_slots,
    destination_slots,
    SOURCE_PAGE: tl.constexpr,
    DESTINATION_PAGE: tl.constexpr,
    SOURCE_PAGE_STRIDE: tl.constexpr,
    DESTINATION_PAGE_STRIDE: tl.constexpr,
    SOURCE_ROW_STRIDE: tl.constexpr,
    DESTINATION_ROW_STRIDE: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    src = tl.load(source_slots + token)
    dst = tl.load(destination_slots + token)
    columns = tl.arange(0, BLOCK)
    valid = (src >= 0) & (dst >= 0) & (columns < WIDTH)
    src_offset = src // SOURCE_PAGE * SOURCE_PAGE_STRIDE + src % SOURCE_PAGE * SOURCE_ROW_STRIDE
    dst_offset = dst // DESTINATION_PAGE * DESTINATION_PAGE_STRIDE + dst % DESTINATION_PAGE * DESTINATION_ROW_STRIDE
    values = tl.load(source + src_offset + columns, valid, other=0)
    tl.store(destination + dst_offset + columns, values, valid)


def copy_mla_kv(
    source: torch.Tensor,
    destination: torch.Tensor,
    source_slots: torch.Tensor,
    destination_slots: torch.Tensor,
) -> None:
    """Copy rank-owned current KV into paged history without materializing rows.

    The source contains every live current token; negative destination slots
    identify tokens owned by another DCP rank or padding. Both caches may have
    gaps between pages (the BLHNC layout) and use different page sizes.
    """
    if source_slots.numel() == 0:
        return
    _copy_mla_kv_kernel[(source_slots.numel(),)](
        source,
        destination,
        source_slots,
        destination_slots,
        SOURCE_PAGE=source.shape[1],
        DESTINATION_PAGE=destination.shape[1],
        SOURCE_PAGE_STRIDE=source.stride(0),
        DESTINATION_PAGE_STRIDE=destination.stride(0),
        SOURCE_ROW_STRIDE=source.stride(1),
        DESTINATION_ROW_STRIDE=destination.stride(1),
        WIDTH=source.shape[2],
        BLOCK=triton.next_power_of_2(source.shape[2]),
        multibuffer=False,
    )
