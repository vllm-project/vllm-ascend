// SPDX-License-Identifier: Apache-2.0
#ifndef ATTN_RES_FWD_LEGACY_PLAN_H
#define ATTN_RES_FWD_LEGACY_PLAN_H

#include <cstdint>
#include "../op_kernel/attn_res_fwd_ub_layout.h"

namespace AttnResFwd {

// Retain the allowance for TPipe bookkeeping and temporary alignment.
constexpr uint64_t LEGACY_UB_RESERVE_BYTES = 32U * 1024U;

struct LegacyUbConfig {
  bool resident{false};
  bool fusedPrefix{false};
  bool needBackward{false};
  bool regBase{false};
  bool cacheOutputNorm{false};
};

inline uint64_t LegacyUbBytes(uint32_t hidden, uint32_t rows, const LegacyUbConfig& config) {
  const uint64_t b16Row = (static_cast<uint64_t>(hidden) + 15U) / 16U * 32U;
  const uint64_t fp32Row = (static_cast<uint64_t>(hidden) + 7U) / 8U * 32U;
  const uint64_t meta = (static_cast<uint64_t>(rows) + 7U) / 8U * 8U;
  const uint64_t scores = config.regBase && rows > 128U ? (meta + 63U) / 64U * 64U : meta;
  const uint64_t scratch = !config.regBase && meta < 64U ? 64U : meta;
  // Resident: one input and one output queue; Reload: two inputs and one output.
  const uint64_t residentRows = LegacyResidentRows(rows, config.fusedPrefix);
  uint64_t bytes = (config.resident ? 2U + residentRows : 3U) * b16Row;
  bytes += 3U * fp32Row;  // scoreWeight, vRow and outFp32.
  bytes += (scores + scratch + meta * 8U + 8U) * sizeof(float);
  if (config.needBackward) {
    bytes += (8U + meta) * sizeof(float);  // invQue and probsQue.
  }
  if (config.fusedPrefix) {
    bytes += b16Row;
  }
  if (config.cacheOutputNorm) {
    bytes += b16Row;
  }
  return bytes + LEGACY_UB_RESERVE_BYTES;
}

inline uint32_t LegacyMaxResidentRows(uint32_t hidden, uint64_t available, LegacyUbConfig config) {
  if (hidden == 0 || available <= LEGACY_UB_RESERVE_BYTES) {
    return 0;
  }
  config.resident = true;
  const uint64_t rowBytes = (static_cast<uint64_t>(hidden) + 15U) / 16U * 32U;
  uint64_t low = 0;
  uint64_t high = available / rowBytes;
  if (high > UINT32_MAX) {
    high = UINT32_MAX;
  }
  while (low < high) {
    const uint64_t middle = low + (high - low + 1U) / 2U;
    if (LegacyUbBytes(hidden, static_cast<uint32_t>(middle), config) <= available) {
      low = middle;
    } else {
      high = middle - 1U;
    }
  }
  return static_cast<uint32_t>(low);
}

inline uint32_t LegacyResidentRowLimit(uint32_t hidden, uint64_t available, const LegacyUbConfig& config,
                                       bool fusedAdd) {
  const uint32_t capacity = LegacyMaxResidentRows(hidden, available, config);
  if (!config.regBase || capacity == 0) {
    return capacity;
  }
  // A5 Resident copies can outweigh Reload's pipelined second read. Retain
  // its previous resource-based admission bound separately from UB safety.
  const uint64_t admission = static_cast<uint64_t>(hidden) * (4U * sizeof(float) + 2U * sizeof(uint16_t)) +
                             LEGACY_UB_RESERVE_BYTES +
                             (fusedAdd ? static_cast<uint64_t>(hidden) * sizeof(uint16_t) : 0U);
  if (admission >= available) {
    return 0;
  }
  const uint64_t rowBytes = static_cast<uint64_t>(hidden) * sizeof(uint16_t) + sizeof(float);
  const uint64_t limit = (available - admission) / rowBytes;
  return limit < capacity ? static_cast<uint32_t>(limit) : capacity;
}

}  // namespace AttnResFwd

#endif
