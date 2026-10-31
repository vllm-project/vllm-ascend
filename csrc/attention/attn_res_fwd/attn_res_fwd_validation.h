// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cmath>
#include <cstdint>
#include <limits>

namespace AttnResFwd {
constexpr uint32_t TILING_HIDDEN_ALIGNMENT = 64;
constexpr uint32_t TILING_FLOAT_BLOCK_ELEMS = 8;

// Buffer lengths are uint32_t bytes; reserve alignment headroom before casts.
constexpr uint32_t MAX_TILING_HIDDEN =
    std::numeric_limits<uint32_t>::max() / sizeof(float) / TILING_HIDDEN_ALIGNMENT * TILING_HIDDEN_ALIGNMENT;
constexpr uint32_t MAX_TILING_ROWS = std::numeric_limits<uint32_t>::max() / (TILING_FLOAT_BLOCK_ELEMS * sizeof(float)) /
                                     TILING_FLOAT_BLOCK_ELEMS * TILING_FLOAT_BLOCK_ELEMS;

inline bool AreShapeFieldsRepresentable(int64_t tokens, int64_t capacity, int64_t hidden) {
  const uint64_t maxField = std::numeric_limits<uint32_t>::max();
  return tokens >= 0 && static_cast<uint64_t>(tokens) <= maxField && capacity >= 0 &&
         static_cast<uint64_t>(capacity) <= maxField && hidden >= 0 &&
         static_cast<uint64_t>(hidden) <= MAX_TILING_HIDDEN;
}

inline bool AreCandidateRowsRepresentable(int64_t valid) {
  return valid >= 0 && static_cast<uint64_t>(valid) < MAX_TILING_ROWS;
}

inline bool IsValidBankExtent(int64_t tokens, int64_t capacity, int64_t hidden, int64_t valid, int64_t stride,
                              int64_t writeIndex) {
  if (!AreShapeFieldsRepresentable(tokens, capacity, hidden) || !AreCandidateRowsRepresentable(valid) ||
      valid > capacity || stride < 0 || writeIndex < -1 || writeIndex >= capacity ||
      writeIndex > std::numeric_limits<int32_t>::max() || (writeIndex >= 0 && writeIndex < valid)) {
    return false;
  }
  const uint64_t minimumStride = static_cast<uint64_t>(capacity) * static_cast<uint64_t>(hidden);
  const uint64_t maxElements = std::numeric_limits<uint64_t>::max() / sizeof(uint16_t);
  return static_cast<uint64_t>(stride) >= minimumStride &&
         (tokens == 0 || static_cast<uint64_t>(stride) <= maxElements / static_cast<uint64_t>(tokens));
}

inline bool IsValidNormEpsilon(double epsilon, bool allowZero = false) {
  return std::isfinite(epsilon) && epsilon >= 0.0 && (allowZero || epsilon > 0.0) &&
         epsilon <= std::numeric_limits<float>::max() && (epsilon == 0.0 || static_cast<float>(epsilon) > 0.0f);
}
}  // namespace AttnResFwd
