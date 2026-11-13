// SPDX-License-Identifier: Apache-2.0
#ifndef ATTN_RES_FWD_UB_LAYOUT_H
#define ATTN_RES_FWD_UB_LAYOUT_H

#include <cstdint>

namespace AttnResFwd {
// Shift the buffers after row_ by half a 512-byte address period on A2/A3.
constexpr uint32_t COOPERATIVE_ROW_PADDING_BYTES = 256;
constexpr uint32_t COOPERATIVE_UB_ADDRESS_PERIOD_BYTES = 512;

// A fused prefix already has a dedicated BF16 buffer for the whole token.
#if defined(__CCE_AICORE__)
__aicore__ inline
#else
constexpr
#endif
    uint32_t LegacyResidentRows(uint32_t rows, bool fusedPrefix) {
  return rows - (fusedPrefix && rows != 0 ? 1U : 0U);
}
}  // namespace AttnResFwd

#endif
