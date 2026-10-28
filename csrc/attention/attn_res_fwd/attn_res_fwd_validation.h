// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cmath>
#include <limits>

namespace AttnResFwd {
inline bool IsValidNormEpsilon(double epsilon, bool allowZero = false) {
  return std::isfinite(epsilon) && epsilon >= 0.0 && (allowZero || epsilon > 0.0) &&
         epsilon <= std::numeric_limits<float>::max() && (epsilon == 0.0 || static_cast<float>(epsilon) > 0.0f);
}
}  // namespace AttnResFwd
