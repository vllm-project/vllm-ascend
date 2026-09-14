# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""CPU-only geometry checks for DFlash's contiguous hybrid cache layout."""

DFLASH_FIA_KERNEL_BLOCK_SIZE = 128
DFLASH_FIA_MAX_KERNEL_BLOCKS = 1 << 16
DFLASH_FIA_MAX_PLANE_ELEMENTS = 1 << 32


def get_dflash_aligned_block_size(
    *,
    conv_page_bytes: int,
    ssm_page_bytes: int,
    common_page_size_bytes: int,
    key_row_bytes: int,
    value_row_bytes: int,
    kernel_block_size: int = DFLASH_FIA_KERNEL_BLOCK_SIZE,
) -> int:
    """Return the token block size that preserves shared block ownership.

    The Ascend hybrid views consist of contiguous planes, not strided pages:
    Mamba uses [all conv, all SSM, padding] and attention uses [padding,
    all K, all V]. With page bytes C + 2*S and K/V block bytes S, those
    views become [all conv, all K/SSM, all V]. Different block IDs then
    access disjoint regions, including across cache groups.
    """
    if conv_page_bytes < 0 or min(ssm_page_bytes, key_row_bytes, value_row_bytes, kernel_block_size) <= 0:
        raise ValueError("DFlash cache layout requires nonnegative conv bytes and positive state/attention dimensions.")
    if common_page_size_bytes != conv_page_bytes + 2 * ssm_page_bytes:
        raise ValueError("DFlash contiguous cache layout requires common page bytes = conv bytes + 2 * SSM bytes.")
    if key_row_bytes != value_row_bytes or ssm_page_bytes % key_row_bytes:
        raise ValueError("DFlash K and V must have equal row bytes that divide one SSM state page.")
    block_size = ssm_page_bytes // key_row_bytes
    if block_size % kernel_block_size:
        raise ValueError("DFlash aligned attention block size must be a multiple of the attention kernel block size.")
    return block_size


def get_dflash_fia_safe_num_blocks(
    *,
    storage_block_size: int,
    key_row_bytes: int,
    value_row_bytes: int,
    key_element_bytes: int,
    value_element_bytes: int,
    kernel_block_size: int = DFLASH_FIA_KERNEL_BLOCK_SIZE,
) -> int:
    """Bound a physical pool before the reproduced FIA address boundary.

    The probe's first failing 128-token block is 65536, which also starts
    at element 2**32 for its cache geometry. Both limits are retained
    conservatively; this does not identify CANN's internal index width.
    The returned count includes block zero, reserved by the scheduler.
    """
    if min(storage_block_size, key_row_bytes, value_row_bytes, key_element_bytes, value_element_bytes) <= 0:
        raise ValueError("DFlash FIA cache dimensions and element sizes must be positive.")
    if kernel_block_size != DFLASH_FIA_KERNEL_BLOCK_SIZE or storage_block_size % kernel_block_size:
        raise ValueError("DFlash FIA address guard requires 128-token kernel blocks and aligned storage blocks.")
    if key_row_bytes % key_element_bytes or value_row_bytes % value_element_bytes:
        raise ValueError("DFlash FIA cache row bytes must contain whole elements.")

    blocks_per_physical = storage_block_size // kernel_block_size
    key_elements_per_page = storage_block_size * (key_row_bytes // key_element_bytes)
    value_elements_per_page = storage_block_size * (value_row_bytes // value_element_bytes)
    return min(
        DFLASH_FIA_MAX_KERNEL_BLOCKS // blocks_per_physical,
        DFLASH_FIA_MAX_PLANE_ELEMENTS // key_elements_per_page,
        DFLASH_FIA_MAX_PLANE_ELEMENTS // value_elements_per_page,
    )
