// SPDX-License-Identifier: Apache-2.0
#ifndef ATTN_RES_FWD_UB_LAYOUT_H
#define ATTN_RES_FWD_UB_LAYOUT_H

#include <cstdint>

namespace AttnResFwd {
// Shift the buffers after row_ by half a 512-byte address period on A2/A3.
constexpr uint32_t COOPERATIVE_ROW_PADDING_BYTES = 256;
constexpr uint32_t COOPERATIVE_UB_ADDRESS_PERIOD_BYTES = 512;
}  // namespace AttnResFwd

#endif
