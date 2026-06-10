# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# cache_config.block_size:
#   [mla, swa, c4 state, c128 state], [page_size_padded_t1, page_size_padded_t2]
DSV4_BLOCK_SIZES = {
    128: [[128, 128, 8, 32], [16640, 131072]],
    64: [[64, 64, 4, 16], [8320, 65536]],
    32: [[32, 32, 2, 8], [4160, 32768]],
}
