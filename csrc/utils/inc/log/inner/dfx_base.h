/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dfx_base.h
 * \brief 外部模块不应直接引用本头文件
 */

#pragma once

#include <string>
#include <cstdint>
#include <sstream>
#include <unistd.h>
#include <sys/syscall.h>
#include <securec.h>
#include <base/alog_pub.h>
#include <base/err_msg.h>
#include <exe_graph/runtime/tiling_context.h>
#include <exe_graph/runtime/tiling_parse_context.h>
#include <exe_graph/runtime/infer_shape_context.h>
#include <exe_graph/runtime/infer_datatype_context.h>

namespace ops {
namespace utils {

class LogBase {
public:
    static constexpr const int MAX_LOG_LEN = 16000;
    static constexpr const int MSG_HDR_LEN = 200;

    static inline uint64_t GetTid()
    {
        return static_cast<uint64_t>(syscall(__NR_gettid));
    }

    static inline const char *GetStr(const std::string &str)
    {
        return str.c_str();
    }

    static inline const char *GetStr(const char *str)
    {
        return str;
    }

    static inline const std::string &GetOpInfo(const std::string &str)
    {
        return str;
    }

    static inline const char *GetOpInfo(const char *str)
    {
        return str;
    }

    static inline std::string GetOpInfo(const gert::TilingContext *context)
    {
        return GetOpInfoFromContext(context);
    }

    static inline std::string GetOpInfo(const gert::TilingParseContext *context)
    {
        return GetOpInfoFromContext(context);
    }

    static inline std::string GetOpInfo(const gert::InferShapeContext *context)
    {
        return GetOpInfoFromContext(context);
    }

    static inline std::string GetOpInfo(const gert::InferDataTypeContext *context)
    {
        return GetOpInfoFromContext(context);
    }

private:
    template <class T> static inline std::string GetOpInfoFromContext(T context)
    {
        if (context == nullptr) {
            return "nil:nil";
        }
        std::string opInfo = context->GetNodeType() != nullptr ? context->GetNodeType() : "nil";
        opInfo += ":";
        opInfo += context->GetNodeName() != nullptr ? context->GetNodeName() : "nil";
        return opInfo;
    }
};

} // namespace utils

template <typename T>
std::string Shape2String(const T& shape) {
  std::ostringstream oss;
  oss << "[";
  if (shape.GetDimNum() > 0) {
    for (size_t i = 0; i < shape.GetDimNum() - 1; ++i) {
      oss << shape.GetDim(i) << ", ";
    }
    oss << shape.GetDim(shape.GetDimNum() - 1);
  }
  oss << "]";
  return oss.str();
}
} // namespace ops

// 使用本宏前需预定义标识子模块名称的 OPS_UTILS_LOG_SUB_MOD_NAME
// 如: #define OPS_UTILS_LOG_SUB_MOD_NAME "OP_TILING" 或通过 CMake 传递预定义宏
#define OPS_LOG_STUB(MOD_ID, LOG_LEVEL, OPS_DESC, FMT, ...)                                                     \
 do {                                                                                                           \
        if (AlogCheckDebugLevel(static_cast<int>(MOD_ID), (LOG_LEVEL)) == 1) {                                  \
            AlogRecord(static_cast<int>(MOD_ID), DLOG_TYPE_DEBUG, (LOG_LEVEL),                                  \
                "[%s:%d][%s]%s[%s][%lu] OpName:[%s] " #FMT,                                                     \
                 __FILE__, __LINE__, (OPS_UTILS_LOG_SUB_MOD_NAME),                                              \
                (OPS_UTILS_LOG_PACKAGE_TYPE), __FUNCTION__, ops::utils::LogBase::GetTid(),                      \
                ops::utils::LogBase::GetStr(ops::utils::LogBase::GetOpInfo(OPS_DESC)), ##__VA_ARGS__);          \
        }                                                                                                       \
    }while (0)

#define OPS_LOG_STUB_IF(COND, LOG_FUNC, EXPR)                                                                          \
    static_assert(std::is_same<bool, std::decay<decltype(COND)>::type>::value, "condition should be bool");            \
    do {                                                                                                               \
        if (__builtin_expect((COND), 0)) {                                                                             \
            LOG_FUNC;                                                                                                  \
            EXPR;                                                                                                      \
        }                                                                                                              \
    } while (0)

#define OPS_INNER_ERR_STUB(ERR_CODE_STR, OPS_DESC, FMT, ...)                                                           \
    do {                                                                                                               \
        OPS_LOG_STUB(OP, DLOG_ERROR, OPS_DESC, FMT, ##__VA_ARGS__);                                                    \
        REPORT_INNER_ERR_MSG(ERR_CODE_STR, FMT, ##__VA_ARGS__);                                                          \
    } while (0)

#define OPS_CALL_ERR_STUB(ERR_CODE_STR, OPS_DESC, FMT, ...)                                                            \
    do {                                                                                                               \
        OPS_LOG_STUB(OP, DLOG_ERROR, OPS_DESC, FMT, ##__VA_ARGS__);                                                    \
        REPORT_INNER_ERR_MSG(ERR_CODE_STR, FMT, ##__VA_ARGS__);                                                           \
    } while (0)

#define OPS_LOG_STUB_D(OPS_DESC, FMT, ...) OPS_LOG_STUB(OP, DLOG_DEBUG, OPS_DESC, FMT, ##__VA_ARGS__)
#define OPS_LOG_STUB_I(OPS_DESC, FMT, ...) OPS_LOG_STUB(OP, DLOG_INFO, OPS_DESC, FMT, ##__VA_ARGS__)
#define OPS_LOG_STUB_W(OPS_DESC, FMT, ...) OPS_LOG_STUB(OP, DLOG_WARN, OPS_DESC, FMT, ##__VA_ARGS__)
#define OPS_LOG_STUB_E(OPS_DESC, FMT, ...) OPS_LOG_STUB(OP, DLOG_ERROR, OPS_DESC, FMT, ##__VA_ARGS__)
#define OPS_LOG_STUB_EVENT(OPS_DESC, FMT, ...) OPS_LOG_STUB(OP, DLOG_EVENT, OPS_DESC, FMT, ##__VA_ARGS__)

#define OPS_LOG_STUB_FULL(LEVEL, OPS_DESC, FMT, ...)                                                                   \
    do {                                                                                                               \
        if (0 == AlogCheckDebugLevel(OP, (LEVEL))) {                                                                         \
            break;                                                                                                     \
        }                                                                                                              \
        char msgbufxyz[ops::utils::LogBase::MAX_LOG_LEN];                                                              \
        size_t msgmaxlen = (MSG_LENGTH - ops::utils::LogBase::MSG_HDR_LEN);                                            \
        int rettmp = snprintf_s(msgbufxyz, sizeof(msgbufxyz), sizeof(msgbufxyz) - 1, FMT, ##__VA_ARGS__);              \
        if (rettmp == -1) {                                                                                            \
            msgbufxyz[sizeof(msgbufxyz) - 1] = '\0';                                                                   \
        }                                                                                                              \
        size_t msglength = std::strlen(msgbufxyz);                                                                     \
        if (msglength < msgmaxlen) {                                                                                   \
            OPS_LOG_STUB(OP, (LEVEL), (OPS_DESC), "%s", msgbufxyz);                                                    \
            break;                                                                                                     \
        }                                                                                                              \
        char *msgchunkbegin = msgbufxyz;                                                                               \
        char *msgchunkend = nullptr;                                                                                   \
        while (msgchunkbegin < msgbufxyz + msglength) {                                                                \
            if (msgchunkbegin[0] == '\n') {                                                                            \
                OPS_LOG_STUB(OP, (LEVEL), (OPS_DESC), "");                                                             \
                msgchunkbegin += 1;                                                                                    \
                continue;                                                                                              \
            }                                                                                                          \
            msgchunkend = std::strchr(msgchunkbegin, '\n');                                                            \
            if (msgchunkend == nullptr) {                                                                              \
                msgchunkend = msgchunkbegin + std::strlen(msgchunkbegin);                                              \
            }                                                                                                          \
            while (msgchunkend > msgchunkbegin) {                                                                      \
                std::string msgchunk(msgchunkbegin,                                                                    \
                                     std::min(msgmaxlen, static_cast<size_t>(msgchunkend - msgchunkbegin)));           \
                OPS_LOG_STUB(OP, (LEVEL), (OPS_DESC), "%s", msgchunk.c_str());                                         \
                msgchunkbegin += msgchunk.size();                                                                      \
            }                                                                                                          \
            msgchunkbegin += 1;                                                                                        \
        }                                                                                                              \
    } while (0)
