/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mc2_log.h
 * \brief
 */

#ifndef MC2_LOG_H
#define MC2_LOG_H

#include <string>
#include <vector>

#include "mc2_common_log.h"

#include "mc2_tiling_struct.h"
#include "tiling/tiling_api.h"
// Not used by mega_moe — commented out to remove 3rd dependency:
// #include "quant_batch_matmul_v3/op_kernel/arch35/quant_batch_matmul_v3_tiling_data.h"
// #include "mat_mul_v3/op_kernel/arch35/mat_mul_tiling_data.h"

template <typename T>
std::string ConcatString(const T &arg) {
  std::ostringstream oss;
  oss << arg;
  return oss.str();
}

template <typename T, typename... Ts>
std::string ConcatString(T arg, Ts... arg_left) {
  std::ostringstream oss;
  oss << arg;
  oss << ConcatString(arg_left...);
  return oss.str();
}

/**
 * @brief 工具函数：将合法列表转为字符串动态生成
 * @param list: 有效值列表
 */
template <typename T>
std::string VectorToString(const std::vector<T> &list)
{
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < list.size(); i++) {
        if (i > 0) {
          oss << ",";
        }
        oss << list[i];
    }
    oss << "]";
    return oss.str();
}

// Not used by mega_moe — commented out to remove 3rd dependency:
// namespace Mc2Log {
// void PrintMMV3TilingData(const std::string &opName, Mc2Tiling::MC2MatmulV3TilingData &tiling);
// void PrintRCSTilingData(const std::string &opName, Mc2Tiling::RCSTiling& rcsTiling);
// void PrintTileL2TilingData(const std::string &opName, Mc2Tiling::TileL2Tiling& tileL2Tiling);
// void PrintMc2MsgData(const std::string &opName, Mc2Tiling::Mc2Msg& msg);
// void PrintTCubeTilingData(const std::string &opName, ::TCubeTiling &tiling);
// void PrintTCubeTilingWindowParam(const std::string &opName, DequantBmm::Mc2SlidingWindowParams &tiling);
// void PrintTCubeTilingL2cache(const std::string &opName, DequantBmm::Mc2L2cacheTileParams &tiling);
// void PrintTCubeTilingParams(const std::string &opName, DequantBmm::Mc2QuantBatchMatmulV3DataParams &tiling);
// void PrintMMV3TilingData(const std::string &opName, Mc2MatMulV3TilingData &tiling);
// }  // namespace Mc2Log

inline const char *ConvertStringToCstr(const std::string &str) { return str.c_str(); }

#if !defined(__ANDROID__) && !defined(ANDROID)
#define CUBE_INNER_ERR_REPORT(op_name, err_msg, ...)              \
  do {                                                            \
    D_OP_LOGE(op_name, err_msg, ##__VA_ARGS__);                   \
    REPORT_INNER_ERR_MSG("E69999", "op[%s], " err_msg,            \
                         ConvertStringToCstr(Ops::Base::GetOpInfo(op_name)), \
                         ##__VA_ARGS__);                          \
  } while (0)

#define VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op_name, err_msg)                  \
  do {                                                                         \
    OP_LOGE_WITHOUT_REPORT(op_name, "%s", ConvertStringToCstr(err_msg));                  \
    std::string errorStr = "E89999";                                           \
    REPORT_INNER_ERR_MSG(errorStr.c_str(), "%s",                               \
                         ConcatString("op[", op_name, "],", err_msg).c_str()); \
  } while (0)

#define OPS_REPORT_VECTOR_INNER_ERR(opName, ...) OPS_LOG_E(opName, ##__VA_ARGS__)
#define OPS_REPORT_CUBE_INNER_ERR(opName, ...) OPS_LOG_E(opName, ##__VA_ARGS__)
namespace optiling {
#define VECTOR_INNER_ERR_REPORT_TILING(op_name, err_msg, ...)     \
  do {                                                            \
    OP_LOGE_WITHOUT_REPORT(op_name, err_msg, ##__VA_ARGS__);      \
    REPORT_INNER_ERR_MSG("E89999", "op[%s], " err_msg,            \
                         ConvertStringToCstr(Ops::Base::GetOpInfo(op_name)), \
                         ##__VA_ARGS__);                          \
  } while (0)
#define OP_TILING_CHECK(cond, log_func, expr) \
  do {                                        \
    if (cond) {                               \
      log_func;                               \
      expr;                                   \
    }                                         \
  } while (0)
}  // namespace optiling

#else
#define CUBE_INNER_ERR_REPORT(op_name, err_msg, ...)
#define VECTOR_INFER_SHAPE_INNER_ERR_REPORT(opName, err_msg)
#define OPS_REPORT_VECTOR_INNER_ERR(opName, ...)
#define OPS_REPORT_CUBE_INNER_ERR(opName, ...)
namespace optiling {
#define VECTOR_INNER_ERR_REPORT_TILING(opName, err_msg, ...)
#define OP_TILING_CHECK(cond, log_func, expr)
}  // namespace optiling
#endif
#endif