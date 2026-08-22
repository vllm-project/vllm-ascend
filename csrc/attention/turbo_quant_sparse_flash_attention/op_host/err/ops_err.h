/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

// The aligned CANN tiling source expects logging macros from a newer CANN
// header. Keep the fallbacks operator-local instead of extending the shared API.
#include "../../../../common/include/err/ops_err.h"

#include <string>

namespace tq_sfa_error_compat {

inline const char* SafeString(const char* value) { return value == nullptr ? "" : value; }

inline void LogInvalidWithExpected(const char* opName, const char* kind, const char* param, const char* actual,
                                   const std::string& expected) {
  OP_LOGE(SafeString(opName), "Invalid %s for %s, actual: %s, expected: %s.", SafeString(kind), SafeString(param),
          SafeString(actual), expected.c_str());
}

inline void LogInvalidWithReason(const char* opName, const char* kind, const char* param, const char* actual,
                                 const std::string& reason) {
  OP_LOGE(SafeString(opName), "Invalid %s for %s, actual: %s, reason: %s.", SafeString(kind), SafeString(param),
          SafeString(actual), reason.c_str());
}

}  // namespace tq_sfa_error_compat

#ifndef OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON
  #define OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opname, param, actual, reason) \
    ::tq_sfa_error_compat::LogInvalidWithReason(opname, "dtype", param, actual, reason)
#endif

#ifndef OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON
  #define OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opname, param, actual, reason) \
    ::tq_sfa_error_compat::LogInvalidWithReason(opname, "shape", param, actual, reason)
#endif

#ifndef OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON
  #define OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(opname, param, actual, reason) \
    ::tq_sfa_error_compat::LogInvalidWithReason(opname, "shape dim", param, actual, reason)
#endif

#ifndef OP_LOGE_FOR_INVALID_VALUE
  #define OP_LOGE_FOR_INVALID_VALUE(opname, param, actual, expected) \
    ::tq_sfa_error_compat::LogInvalidWithExpected(opname, "value", param, actual, expected)
#endif

#ifndef OP_LOGE_FOR_INVALID_VALUE_WITH_REASON
  #define OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opname, param, actual, reason) \
    ::tq_sfa_error_compat::LogInvalidWithReason(opname, "value", param, actual, reason)
#endif
