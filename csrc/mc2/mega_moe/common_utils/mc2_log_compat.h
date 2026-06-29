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
 * \file mc2_log_compat.h
 * \brief Forward compatibility: provides OP_LOGE_LIBOPAPI_REPORT and OP_LOGE_FOR_*
 *        macros (EZ0008-EZ0034) when using an older log.h that only defines EZ0001-EZ0007.
 *        Includes log/log.h internally; include this file as the sole log header.
 */

#ifndef MC2_LOG_COMPAT_H
#define MC2_LOG_COMPAT_H

#include <string>
#include <vector>
#include <cinttypes>
#include "log/log.h"

// ============================================================================
// OP_LOGE_LIBOPAPI_REPORT fallback
// New log.h defines this macro; old log.h does not.
// For opapi build (OP_LOG_LIBOPAPI_ONLY): use opdev's DlogRecordInner directly
//   to avoid type conflicts with old log.h's GetOpInfo template.
// For host build: use old log.h's D_OP_LOGE via Ops::Base::GetOpInfo.
// ============================================================================
#ifndef OP_LOGE_LIBOPAPI_REPORT
#ifdef OP_LOG_LIBOPAPI_ONLY
#define OP_LOGE_LIBOPAPI_REPORT(opName, fmt, ...) \
    do { \
        if (CheckLogLevelInner(OP_ID, OP_LOG_ERROR) == 1) { \
            DlogRecordInner(OP_ID, OP_LOG_ERROR, \
                "[%s:%d][%s][%s][%lu] OpName:[%s] " fmt, \
                __FILE__, __LINE__, OPAPI_SUBMOD_NAME, __FUNCTION__, \
                op::OpLog::GetTid(), opName, \
                ##__VA_ARGS__); \
        } \
    } while (0)
#else
#define OP_LOGE_LIBOPAPI_REPORT(opName, fmt, ...) \
    D_OP_LOGE(Ops::Base::GetOpInfo(opName), fmt, ##__VA_ARGS__)
#endif /* OP_LOG_LIBOPAPI_ONLY */
#endif /* OP_LOGE_LIBOPAPI_REPORT */

// ============================================================================
// For opapi build (OP_LOG_LIBOPAPI_ONLY): old log.h unconditionally defines
// OP_LOGE, D_OP_LOGE, OpLogErrSub, etc. with signatures that differ from
// opdev/op_log.h's versions. The opapi source code expects opdev's signatures
// (e.g. OP_LOGE(errno, ...) where first arg is an error code, not opName).
// Undefine old log.h's versions and provide opdev-compatible replacements.
// ============================================================================
#ifdef OP_LOG_LIBOPAPI_ONLY
#undef OpLogSub
#undef OpLogErrSub
#undef D_OP_LOGI
#undef D_OP_LOGW
#undef D_OP_LOGE
#undef D_OP_LOGD
#undef OP_LOGI
#undef OP_LOGW
#undef OP_LOGE_WITHOUT_REPORT
#undef OP_LOGE
#undef OP_LOGD
#undef OP_CHECK_IF
#undef OP_CHECK_NULL_WITH_CONTEXT

// Restore opdev/op_log.h signatures after old log.h shadowed them.
// DOplogSub, CheckLogLevelInner, DlogRecordInner, GetOpName, GetFileName,
// and op::OpLog::GetTid() are from opdev/op_log.h and NOT redefined by old
// log.h (different names), so they remain available.
#define OpLogSub(moduleId, level, op_info, fmt, ...) \
    DOplogSub(static_cast<int32_t>(moduleId), OPAPI_SUBMOD_NAME, level, "[%s][%lu] %s" fmt, \
        __FUNCTION__, op::OpLog::GetTid(), op_info, ##__VA_ARGS__)

#define OpLogErrSub(moduleId, level, op_info, errno, fmt, ...) \
    DOplogSub(static_cast<int32_t>(moduleId), OPAPI_SUBMOD_NAME, level, "[%s][%lu] errno[%d] %s" fmt, \
        __FUNCTION__, op::OpLog::GetTid(), errno, op_info, ##__VA_ARGS__)

#define OpDfxLogSub(moduleId, level, file, line, func, op_info, fmt, ...) \
    DDfxlogSub(static_cast<int32_t>(moduleId), OPAPI_SUBMOD_NAME, level, GetFileName(file), line, \
        "[%s][%lu] %s" fmt, func, op::OpLog::GetTid(), op_info, ##__VA_ARGS__)

#define D_OP_LOGI(opname, fmt, ...) \
    OpLogSub(OP_ID, OP_LOG_INFO, opname, fmt, ##__VA_ARGS__)
#define D_OP_LOGW(opname, fmt, ...) \
    OpLogSub(OP_ID, OP_LOG_WARN, opname, fmt, ##__VA_ARGS__)
#define D_OP_LOGE(opname, errno, fmt, ...) \
    OpLogErrSub(OP_ID, OP_LOG_ERROR, opname, errno, fmt, ##__VA_ARGS__)
#define D_OP_LOGD(opname, fmt, ...) \
    OpLogSub(OP_ID, OP_LOG_DEBUG, opname, fmt, ##__VA_ARGS__)
#define D_OP_EVENT(opname, fmt, ...) \
    OpLogSub(OP_ID, OP_LOG_EVENT, opname, fmt, ##__VA_ARGS__)

#define OP_LOGI(...) D_OP_LOGI(GetOpName().c_str(), __VA_ARGS__)
#define OP_LOGW(...) D_OP_LOGW(GetOpName().c_str(), __VA_ARGS__)
#define OP_LOGE_WITHOUT_REPORT(errno, ...) \
    D_OP_LOGE(GetOpName().c_str(), errno, __VA_ARGS__)
#define OP_LOGE(errno, ...) \
    do { \
        OP_LOGE_WITHOUT_REPORT(errno, __VA_ARGS__); \
        REPORT_ERROR_MESSAGE(errno, __VA_ARGS__); \
    } while (false)
#define OP_LOGD(...) D_OP_LOGD(GetOpName().c_str(), __VA_ARGS__)
#define OP_EVENT(...) D_OP_EVENT(GetOpName().c_str(), __VA_ARGS__)
#endif /* OP_LOG_LIBOPAPI_ONLY */

// ============================================================================
// OP_LOGE_FOR_* macros (EZ0008-EZ0034)
// New log.h defines these; old log.h does not.
// ============================================================================
#ifndef OP_LOGE_FOR_INVALID_SHAPE

// EZ0008: Parameter shape error (single parameter, with correct shape)
#define OP_LOGE_FOR_INVALID_SHAPE(opName, paramName, incorrectShape, correctShape)                             \
    do {                                                                                                       \
        std::string _safe_opName_(opName);                                                                     \
        std::string _safe_paramName_(paramName);                                                               \
        std::string _safe_incorrectShape_(incorrectShape);                                                     \
        std::string _safe_correctShape_(correctShape);                                                         \
        OP_LOGE_LIBOPAPI_REPORT(_safe_opName_.c_str(),                                                          \
            "Parameter %s of operator %s has incorrect shape %s. It should be %s.",                            \
            _safe_paramName_.c_str(), _safe_opName_.c_str(), _safe_incorrectShape_.c_str(),                    \
            _safe_correctShape_.c_str());                                                                      \
        REPORT_INNER_ERR_MSG("EZ0008",                                                                         \
            "Parameter %s of operator %s has incorrect shape %s. It should be %s.",                            \
            _safe_paramName_.c_str(), _safe_opName_.c_str(), _safe_incorrectShape_.c_str(),                    \
            _safe_correctShape_.c_str());                                                                      \
    } while (0)

// EZ0009: Parameter shape error (single parameter, with reason)
#define OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, paramName, incorrectShape, reason)                            \
    do {                                                                                                            \
        std::string _safe_opName_(opName);                                                                          \
        std::string _safe_paramName_(paramName);                                                                    \
        std::string _safe_incorrectShape_(incorrectShape);                                                          \
        std::string _safe_reason_(reason);                                                                          \
        OP_LOGE_LIBOPAPI_REPORT(_safe_opName_.c_str(),                                                               \
            "Parameter %s of operator %s has incorrect shape %s. Reason: %s.",                                      \
            _safe_paramName_.c_str(), _safe_opName_.c_str(), _safe_incorrectShape_.c_str(), _safe_reason_.c_str()); \
        REPORT_INNER_ERR_MSG("EZ0009",                                                                              \
            "Parameter %s of operator %s has incorrect shape %s. Reason: %s.",                                      \
            _safe_paramName_.c_str(), _safe_opName_.c_str(), _safe_incorrectShape_.c_str(), _safe_reason_.c_str()); \
    } while (0)

// EZ0010: Parameters shape error (multiple parameters, with reason)
#define OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(opName, paramNames, incorrectShapes, reason)                           \
    do {                                                                                                              \
        std::string _safe_opName_(opName);                                                                            \
        std::string _safe_paramNames_(paramNames);                                                                    \
        std::string _safe_incorrectShapes_(incorrectShapes);                                                          \
        std::string _safe_reason_(reason);                                                                            \
        OP_LOGE_LIBOPAPI_REPORT(_safe_opName_.c_str(),                                                                 \
            "Parameters %s of operator %s have incorrect shapes %s. Reason: %s.",                                     \
            _safe_paramNames_.c_str(), _safe_opName_.c_str(), _safe_incorrectShapes_.c_str(), _safe_reason_.c_str()); \
        REPORT_INNER_ERR_MSG("EZ0010",                                                                                \
            "Parameters %s of operator %s have incorrect shapes %s. Reason: %s.",                                     \
            _safe_paramNames_.c_str(), _safe_opName_.c_str(), _safe_incorrectShapes_.c_str(), _safe_reason_.c_str()); \
    } while (0)

// EZ0011: Parameter shape dim error (single parameter, with correct dim)
#define OP_LOGE_FOR_INVALID_SHAPEDIM(opName, paramName, incorrectDim, correctDim)                                     \
    do {                                                                                                              \
        std::string _safe_opName_(opName);                                                                            \
        std::string _safe_paramName_(paramName);                                                                      \
        std::string _safe_incorrectDim_(incorrectDim);                                                                \
        std::string _safe_correctDim_(correctDim);                                                                    \
        OP_LOGE_LIBOPAPI_REPORT(_safe_opName_.c_str(),                                                                 \
            "Parameter %s of operator %s has incorrect shape dim %s. It should be %s.",                               \
            _safe_paramName_.c_str(), _safe_opName_.c_str(), _safe_incorrectDim_.c_str(), _safe_correctDim_.c_str()); \
        REPORT_INNER_ERR_MSG("EZ0011",                                                                                \
            "Parameter %s of operator %s has incorrect shape dim %s. It should be %s.",                               \
            _safe_paramName_.c_str(), _safe_opName_.c_str(), _safe_incorrectDim_.c_str(), _safe_correctDim_.c_str()); \
    } while (0)

// EZ0012: Parameter shape dim error (single parameter, with reason)
#define OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(opName, paramName, incorrectDim, reason)                         \
    do {                                                                                                          \
        std::string _safe_opName_(opName);                                                                        \
        std::string _safe_paramName_(paramName);                                                                  \
        std::string _safe_incorrectDim_(incorrectDim);                                                            \
        std::string _safe_reason_(reason);                                                                        \
        OP_LOGE_LIBOPAPI_REPORT(_safe_opName_.c_str(),                                                             \
            "Parameter %s of operator %s has incorrect shape dim %s. Reason: %s.",                                \
            _safe_paramName_.c_str(), _safe_opName_.c_str(), _safe_incorrectDim_.c_str(), _safe_reason_.c_str()); \
        REPORT_INNER_ERR_MSG("EZ0012",                                                                            \
            "Parameter %s of operator %s has incorrect shape dim %s. Reason: %s.",                                \
            _safe_paramName_.c_str(), _safe_opName_.c_str(), _safe_incorrectDim_.c_str(), _safe_reason_.c_str()); \
    } while (0)

// EZ0013: Parameters shape dim error (multiple parameters, with reason)
#define OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(opName, paramNames, incorrectDims, reason)                        \
    do {                                                                                                            \
        std::string _safe_opName_(opName);                                                                          \
        std::string _safe_paramNames_(paramNames);                                                                  \
        std::string _safe_incorrectDims_(incorrectDims);                                                            \
        std::string _safe_reason_(reason);                                                                          \
        OP_LOGE_LIBOPAPI_REPORT(_safe_opName_.c_str(),                                                               \
            "Parameters %s of operator %s have incorrect shape dims %s. Reason: %s.",                               \
            _safe_paramNames_.c_str(), _safe_opName_.c_str(), _safe_incorrectDims_.c_str(), _safe_reason_.c_str()); \
        REPORT_INNER_ERR_MSG("EZ0013",                                                                              \
            "Parameters %s of operator %s have incorrect shape dims %s. Reason: %s.",                               \
            _safe_paramNames_.c_str(), _safe_opName_.c_str(), _safe_incorrectDims_.c_str(), _safe_reason_.c_str()); \
    } while (0)

// EZ0014: Parameter shape size error (single parameter, with correct size)
#define OP_LOGE_FOR_INVALID_SHAPESIZE(opName, paramName, incorrectSize, correctSize)                         \
    do {                                                                                                     \
        std::string _safe_opName_(opName);                                                                   \
        std::string _safe_paramName_(paramName);                                                             \
        std::string _safe_incorrectSize_(incorrectSize);                                                     \
        std::string _safe_correctSize_(correctSize);                                                         \
        OP_LOGE_LIBOPAPI_REPORT(_safe_opName_.c_str(),                                                        \
            "Parameter %s of operator %s has incorrect shape size %s. It should be %s.",                     \
            _safe_paramName_.c_str(), _safe_opName_.c_str(), _safe_incorrectSize_.c_str(),                   \
            _safe_correctSize_.c_str());                                                                     \
        REPORT_INNER_ERR_MSG("EZ0014",                                                                       \
            "Parameter %s of operator %s has incorrect shape size %s. It should be %s.",                     \
            _safe_paramName_.c_str(), _safe_opName_.c_str(), _safe_incorrectSize_.c_str(),                   \
            _safe_correctSize_.c_str());                                                                     \
    } while (0)

// EZ0015: Parameter shape size error (single parameter, with reason)
#define OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(opName, paramName, incorrectSize, reason)                        \
    do {                                                                                                           \
        std::string _safe_opName_(opName);                                                                         \
        std::string _safe_paramName_(paramName);                                                                   \
        std::string _safe_incorrectSize_(incorrectSize);                                                           \
        std::string _safe_reason_(reason);                                                                         \
        OP_LOGE_LIBOPAPI_REPORT(_safe_opName_.c_str(),                                                              \
            "Parameter %s of operator %s has incorrect shape size %s. Reason: %s.",                                \
            _safe_paramName_.c_str(), _safe_opName_.c_str(), _safe_incorrectSize_.c_str(), _safe_reason_.c_str()); \
        REPORT_INNER_ERR_MSG("EZ0015",                                                                              \
            "Parameter %s of operator %s has incorrect shape size %s. Reason: %s.",                                \
            _safe_paramName_.c_str(), _safe_opName_.c_str(), _safe_incorrectSize_.c_str(), _safe_reason_.c_str()); \
    } while (0)

// EZ0016: Parameters shape size error (multiple parameters, with reason)
#define OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(opName, paramNames, incorrectSizes, reason)                       \
    do {                                                                                                             \
        std::string _safe_opName_(opName);                                                                           \
        std::string _safe_paramNames_(paramNames);                                                                   \
        std::string _safe_incorrectSizes_(incorrectSizes);                                                           \
        std::string _safe_reason_(reason);                                                                           \
        OP_LOGE_LIBOPAPI_REPORT(_safe_opName_.c_str(),                                                                \
            "Parameters %s of operator %s have incorrect shape sizes %s. Reason: %s.",                               \
            _safe_paramNames_.c_str(), _safe_opName_.c_str(), _safe_incorrectSizes_.c_str(), _safe_reason_.c_str()); \
        REPORT_INNER_ERR_MSG("EZ0016",                                                                               \
            "Parameters %s of operator %s have incorrect shape sizes %s. Reason: %s.",                               \
            _safe_paramNames_.c_str(), _safe_opName_.c_str(), _safe_incorrectSizes_.c_str(), _safe_reason_.c_str()); \
    } while (0)

// EZ0017: Parameter format error (single parameter, with correct format)
#define OP_LOGE_FOR_INVALID_FORMAT(opName, paramName, incorrectFormat, correctFormat)                            \
    do {                                                                                                         \
        std::string _safe_opName_(opName);                                                                       \
        std::string _safe_paramName_(paramName);                                                                 \
        std::string _safe_incorrectFormat_(incorrectFormat);                                                     \
        std::string _safe_correctFormat_(correctFormat);                                                         \
        OP_LOGE_LIBOPAPI_REPORT(_safe_opName_.c_str(),                                                            \
            "Parameter %s of operator %s has incorrect format %s. It should be %s.",                             \
            _safe_paramName_.c_str(), _safe_opName_.c_str(), _safe_incorrectFormat_.c_str(),                     \
            _safe_correctFormat_.c_str());                                                                       \
        REPORT_INNER_ERR_MSG("EZ0017",                                                                           \
            "Parameter %s of operator %s has incorrect format %s. It should be %s.",                             \
            _safe_paramName_.c_str(), _safe_opName_.c_str(), _safe_incorrectFormat_.c_str(),                     \
            _safe_correctFormat_.c_str());                                                                       \
    } while (0)

// EZ0018: Parameters format error (multiple parameters, with reason)
#define OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(opName, paramNames, incorrectFormats, reason)                          \
    do {                                                                                                               \
        std::string _safe_opName_(opName);                                                                             \
        std::string _safe_paramNames_(paramNames);                                                                     \
        std::string _safe_incorrectFormats_(incorrectFormats);                                                         \
        std::string _safe_reason_(reason);                                                                             \
        OP_LOGE_LIBOPAPI_REPORT(_safe_opName_.c_str(),                                                                  \
            "Parameters %s of operator %s have incorrect formats %s. Reason: %s.",                                     \
            _safe_paramNames_.c_str(), _safe_opName_.c_str(), _safe_incorrectFormats_.c_str(), _safe_reason_.c_str()); \
        REPORT_INNER_ERR_MSG("EZ0018",                                                                                 \
            "Parameters %s of operator %s have incorrect formats %s. Reason: %s.",                                     \
            _safe_paramNames_.c_str(), _safe_opName_.c_str(), _safe_incorrectFormats_.c_str(), _safe_reason_.c_str()); \
    } while (0)

// EZ0019: Parameter dtype error (single parameter, with correct dtype)
#define OP_LOGE_FOR_INVALID_DTYPE(opName, paramName, incorrectDtype, correctDtype)                             \
    do {                                                                                                       \
        std::string _safe_opName_(opName);                                                                     \
        std::string _safe_paramName_(paramName);                                                               \
        std::string _safe_incorrectDtype_(incorrectDtype);                                                     \
        std::string _safe_correctDtype_(correctDtype);                                                         \
        OP_LOGE_LIBOPAPI_REPORT(_safe_opName_.c_str(),                                                          \
            "Parameter %s of operator %s has incorrect dtype %s. It should be %s.",                            \
            _safe_paramName_.c_str(), _safe_opName_.c_str(), _safe_incorrectDtype_.c_str(),                    \
            _safe_correctDtype_.c_str());                                                                      \
        REPORT_INNER_ERR_MSG("EZ0019",                                                                         \
            "Parameter %s of operator %s has incorrect dtype %s. It should be %s.",                            \
            _safe_paramName_.c_str(), _safe_opName_.c_str(), _safe_incorrectDtype_.c_str(),                    \
            _safe_correctDtype_.c_str());                                                                      \
    } while (0)

// EZ0020: Parameter dtype error (single parameter, with reason)
#define OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName, paramName, incorrectDtype, reason)                            \
    do {                                                                                                            \
        std::string _safe_opName_(opName);                                                                          \
        std::string _safe_paramName_(paramName);                                                                    \
        std::string _safe_incorrectDtype_(incorrectDtype);                                                          \
        std::string _safe_reason_(reason);                                                                          \
        OP_LOGE_LIBOPAPI_REPORT(_safe_opName_.c_str(),                                                               \
            "Parameter %s of operator %s has incorrect dtype %s. Reason: %s.",                                      \
            _safe_paramName_.c_str(), _safe_opName_.c_str(), _safe_incorrectDtype_.c_str(), _safe_reason_.c_str()); \
        REPORT_INNER_ERR_MSG("EZ0020",                                                                              \
            "Parameter %s of operator %s has incorrect dtype %s. Reason: %s.",                                      \
            _safe_paramName_.c_str(), _safe_opName_.c_str(), _safe_incorrectDtype_.c_str(), _safe_reason_.c_str()); \
    } while (0)

// EZ0021: Parameters dtype error (multiple parameters, with reason)
#define OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(opName, paramNames, incorrectDtypes, reason)                           \
    do {                                                                                                              \
        std::string _safe_opName_(opName);                                                                            \
        std::string _safe_paramNames_(paramNames);                                                                    \
        std::string _safe_incorrectDtypes_(incorrectDtypes);                                                          \
        std::string _safe_reason_(reason);                                                                            \
        OP_LOGE_LIBOPAPI_REPORT(_safe_opName_.c_str(),                                                                 \
            "Parameters %s of operator %s have incorrect dtypes %s. Reason: %s.",                                     \
            _safe_paramNames_.c_str(), _safe_opName_.c_str(), _safe_incorrectDtypes_.c_str(), _safe_reason_.c_str()); \
        REPORT_INNER_ERR_MSG("EZ0021",                                                                                \
            "Parameters %s of operator %s have incorrect dtypes %s. Reason: %s.",                                     \
            _safe_paramNames_.c_str(), _safe_opName_.c_str(), _safe_incorrectDtypes_.c_str(), _safe_reason_.c_str()); \
    } while (0)

// EZ0022: Parameter tensor num error (single parameter, with correct num)
#define OP_LOGE_FOR_INVALID_TENSORNUM(opName, paramName, incorrectNum, correctNum)                                \
    do {                                                                                                          \
        std::string _safe_opName_(opName);                                                                        \
        std::string _safe_paramName_(paramName);                                                                  \
        std::string _safe_correctNum_(correctNum);                                                                \
        OP_LOGE_LIBOPAPI_REPORT(_safe_opName_.c_str(),                                                             \
            "Parameter %s of operator %s has invalid tensor num %ld. It should be %s.",                           \
            _safe_paramName_.c_str(), _safe_opName_.c_str(), static_cast<int64_t>(incorrectNum),                  \
            _safe_correctNum_.c_str());                                                                           \
        REPORT_INNER_ERR_MSG("EZ0022",                                                                            \
            "Parameter %s of operator %s has invalid tensor num %ld. It should be %s.",                           \
            _safe_paramName_.c_str(), _safe_opName_.c_str(), static_cast<int64_t>(incorrectNum),                  \
            _safe_correctNum_.c_str());                                                                           \
    } while (0)

// EZ0023: Parameters tensor num error (multiple parameters, with reason)
#define OP_LOGE_FOR_INVALID_TENSORNUMS_WITH_REASON(opName, paramNames, incorrectNums, reason)                       \
    do {                                                                                                            \
        std::string _safe_opName_(opName);                                                                          \
        std::string _safe_paramNames_(paramNames);                                                                  \
        std::string _safe_incorrectNums_(incorrectNums);                                                            \
        std::string _safe_reason_(reason);                                                                          \
        OP_LOGE_LIBOPAPI_REPORT(_safe_opName_.c_str(),                                                               \
            "Parameters %s of operator %s have invalid tensor nums %s. Reason: %s.",                                \
            _safe_paramNames_.c_str(), _safe_opName_.c_str(), _safe_incorrectNums_.c_str(), _safe_reason_.c_str()); \
        REPORT_INNER_ERR_MSG("EZ0023",                                                                              \
            "Parameters %s of operator %s have invalid tensor nums %s. Reason: %s.",                                \
            _safe_paramNames_.c_str(), _safe_opName_.c_str(), _safe_incorrectNums_.c_str(), _safe_reason_.c_str()); \
    } while (0)

// EZ0024: Parameter value error (single parameter, with correct value)
#define OP_LOGE_FOR_INVALID_VALUE(opName, paramName, incorrectValue, correctValue)                             \
    do {                                                                                                       \
        std::string _safe_opName_(opName);                                                                     \
        std::string _safe_paramName_(paramName);                                                               \
        std::string _safe_incorrectValue_(incorrectValue);                                                     \
        std::string _safe_correctValue_(correctValue);                                                         \
        OP_LOGE_LIBOPAPI_REPORT(_safe_opName_.c_str(),                                                          \
            "Parameter %s of operator %s has incorrect value %s. It should be %s.",                            \
            _safe_paramName_.c_str(), _safe_opName_.c_str(), _safe_incorrectValue_.c_str(),                    \
            _safe_correctValue_.c_str());                                                                      \
        REPORT_INNER_ERR_MSG("EZ0024",                                                                         \
            "Parameter %s of operator %s has incorrect value %s. It should be %s.",                            \
            _safe_paramName_.c_str(), _safe_opName_.c_str(), _safe_incorrectValue_.c_str(),                    \
            _safe_correctValue_.c_str());                                                                      \
    } while (0)

// EZ0025: Parameter list size error (single parameter, with correct size)
#define OP_LOGE_FOR_INVALID_LISTSIZE(opName, paramName, incorrectSize, correctSize)                          \
    do {                                                                                                     \
        std::string _safe_opName_(opName);                                                                   \
        std::string _safe_paramName_(paramName);                                                             \
        std::string _safe_incorrectSize_(incorrectSize);                                                     \
        std::string _safe_correctSize_(correctSize);                                                         \
        OP_LOGE_LIBOPAPI_REPORT(_safe_opName_.c_str(),                                                        \
            "Parameter %s of operator %s has incorrect element nums %s. It should be %s.",                   \
            _safe_paramName_.c_str(), _safe_opName_.c_str(), _safe_incorrectSize_.c_str(),                   \
            _safe_correctSize_.c_str());                                                                     \
        REPORT_INNER_ERR_MSG("EZ0025",                                                                       \
            "Parameter %s of operator %s has incorrect element nums %s. It should be %s.",                   \
            _safe_paramName_.c_str(), _safe_opName_.c_str(), _safe_incorrectSize_.c_str(),                   \
            _safe_correctSize_.c_str());                                                                     \
    } while (0)

// EZ0026: Parameter value error (single parameter, with reason)
#define OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, paramName, incorrectValue, reason)                            \
    do {                                                                                                            \
        std::string _safe_opName_(opName);                                                                          \
        std::string _safe_paramName_(paramName);                                                                    \
        std::string _safe_incorrectValue_(incorrectValue);                                                          \
        std::string _safe_reason_(reason);                                                                          \
        OP_LOGE_LIBOPAPI_REPORT(_safe_opName_.c_str(),                                                               \
            "Parameter %s of operator %s has incorrect value %s. Reason: %s.",                                      \
            _safe_paramName_.c_str(), _safe_opName_.c_str(), _safe_incorrectValue_.c_str(), _safe_reason_.c_str()); \
        REPORT_INNER_ERR_MSG("EZ0026",                                                                              \
            "Parameter %s of operator %s has incorrect value %s. Reason: %s.",                                      \
            _safe_paramName_.c_str(), _safe_opName_.c_str(), _safe_incorrectValue_.c_str(), _safe_reason_.c_str()); \
    } while (0)

// EZ0027: Parameters value error (multiple parameters, with reason)
#define OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(opName, paramNames, incorrectValues, reason)                           \
    do {                                                                                                              \
        std::string _safe_opName_(opName);                                                                            \
        std::string _safe_paramNames_(paramNames);                                                                    \
        std::string _safe_incorrectValues_(incorrectValues);                                                          \
        std::string _safe_reason_(reason);                                                                            \
        OP_LOGE_LIBOPAPI_REPORT(_safe_opName_.c_str(),                                                                 \
            "Parameters %s of operator %s have incorrect values %s. Reason: %s.",                                     \
            _safe_paramNames_.c_str(), _safe_opName_.c_str(), _safe_incorrectValues_.c_str(), _safe_reason_.c_str()); \
        REPORT_INNER_ERR_MSG("EZ0027",                                                                                \
            "Parameters %s of operator %s have incorrect values %s. Reason: %s.",                                     \
            _safe_paramNames_.c_str(), _safe_opName_.c_str(), _safe_incorrectValues_.c_str(), _safe_reason_.c_str()); \
    } while (0)

// EZ0028: Parameter stride error (single parameter, with correct stride)
#define OP_LOGE_FOR_INVALID_STRIDE(opName, paramName, incorrectStride, correctStride)                            \
    do {                                                                                                         \
        std::string _safe_opName_(opName);                                                                       \
        std::string _safe_paramName_(paramName);                                                                 \
        std::string _safe_incorrectStride_(incorrectStride);                                                     \
        std::string _safe_correctStride_(correctStride);                                                         \
        OP_LOGE_LIBOPAPI_REPORT(_safe_opName_.c_str(),                                                            \
            "Parameter %s of operator %s has incorrect stride %s, it should be %s.",                             \
            _safe_paramName_.c_str(), _safe_opName_.c_str(), _safe_incorrectStride_.c_str(),                     \
            _safe_correctStride_.c_str());                                                                       \
        REPORT_INNER_ERR_MSG("EZ0028",                                                                           \
            "Parameter %s of operator %s has incorrect stride %s, it should be %s.",                             \
            _safe_paramName_.c_str(), _safe_opName_.c_str(), _safe_incorrectStride_.c_str(),                     \
            _safe_correctStride_.c_str());                                                                       \
    } while (0)

// EZ0029: File operation path error
#define OP_LOGE_FOR_FILE_PATH(opName, filePath, reason)                                              \
    do {                                                                                             \
        std::string _safe_opName_(opName);                                                           \
        std::string _safe_filePath_(filePath);                                                       \
        std::string _safe_reason_(reason);                                                           \
        OP_LOGE_LIBOPAPI_REPORT(_safe_opName_.c_str(),                                                \
            "Input path %s is invalid. Reason: %s.", _safe_filePath_.c_str(), _safe_reason_.c_str()); \
        REPORT_INNER_ERR_MSG("EZ0029",                                                               \
            "Input path %s is invalid. Reason: %s.", _safe_filePath_.c_str(), _safe_reason_.c_str()); \
    } while (0)

// EZ0030: File operation open error
#define OP_LOGE_FOR_FILE_OPEN(opName, filePath, reason)                                             \
    do {                                                                                            \
        std::string _safe_opName_(opName);                                                          \
        std::string _safe_filePath_(filePath);                                                      \
        std::string _safe_reason_(reason);                                                          \
        OP_LOGE_LIBOPAPI_REPORT(_safe_opName_.c_str(),                                               \
            "Failed to open file %s. Reason: %s.", _safe_filePath_.c_str(), _safe_reason_.c_str()); \
        REPORT_INNER_ERR_MSG("EZ0030",                                                              \
            "Failed to open file %s. Reason: %s.", _safe_filePath_.c_str(), _safe_reason_.c_str()); \
    } while (0)

// EZ0031: File operation parse error
#define OP_LOGE_FOR_FILE_PARSE(opName, fileName, reason)                                            \
    do {                                                                                            \
        std::string _safe_opName_(opName);                                                          \
        std::string _safe_fileName_(fileName);                                                      \
        std::string _safe_reason_(reason);                                                          \
        OP_LOGE_LIBOPAPI_REPORT(_safe_opName_.c_str(),                                               \
            "Failed to parse file %s. Reason: %s.", _safe_fileName_.c_str(), _safe_reason_.c_str()); \
        REPORT_INNER_ERR_MSG("EZ0031",                                                              \
            "Failed to parse file %s. Reason: %s.", _safe_fileName_.c_str(), _safe_reason_.c_str()); \
    } while (0)

// EZ0032: Config error (single item, with expect)
#define OP_LOGE_FOR_INVALID_CONFIG(opName, configFile, item, value, expect)                               \
    do {                                                                                                  \
        std::string _safe_opName_(opName);                                                                \
        std::string _safe_configFile_(configFile);                                                        \
        std::string _safe_item_(item);                                                                    \
        std::string _safe_value_(value);                                                                  \
        std::string _safe_expect_(expect);                                                                \
        OP_LOGE_LIBOPAPI_REPORT(_safe_opName_.c_str(),                                                     \
            "Value %s of configuration item %s in configuration file %s is invalid, it should be %s.",    \
            _safe_value_.c_str(), _safe_item_.c_str(), _safe_configFile_.c_str(), _safe_expect_.c_str()); \
        REPORT_INNER_ERR_MSG("EZ0032",                                                                    \
            "Value %s of configuration item %s in configuration file %s is invalid, it should be %s.",    \
            _safe_value_.c_str(), _safe_item_.c_str(), _safe_configFile_.c_str(), _safe_expect_.c_str()); \
    } while (0)

// EZ0033: Config error (single item, with reason)
#define OP_LOGE_FOR_INVALID_CONFIG_WITH_REASON(opName, configFile, item, value, reason)                   \
    do {                                                                                                  \
        std::string _safe_opName_(opName);                                                                \
        std::string _safe_configFile_(configFile);                                                        \
        std::string _safe_item_(item);                                                                    \
        std::string _safe_value_(value);                                                                  \
        std::string _safe_reason_(reason);                                                                \
        OP_LOGE_LIBOPAPI_REPORT(_safe_opName_.c_str(),                                                     \
            "Value %s of configuration item %s in configuration file %s is invalid. Reason: %s.",         \
            _safe_value_.c_str(), _safe_item_.c_str(), _safe_configFile_.c_str(), _safe_reason_.c_str()); \
        REPORT_INNER_ERR_MSG("EZ0033",                                                                    \
            "Value %s of configuration item %s in configuration file %s is invalid. Reason: %s.",         \
            _safe_value_.c_str(), _safe_item_.c_str(), _safe_configFile_.c_str(), _safe_reason_.c_str()); \
    } while (0)

// EZ0034: Config error (multiple items, with reason)
#define OP_LOGE_FOR_INVALID_CONFIGS_WITH_REASON(opName, configFile, items, values, reason)                  \
    do {                                                                                                    \
        std::string _safe_opName_(opName);                                                                  \
        std::string _safe_configFile_(configFile);                                                          \
        std::string _safe_items_(items);                                                                    \
        std::string _safe_values_(values);                                                                  \
        std::string _safe_reason_(reason);                                                                  \
        OP_LOGE_LIBOPAPI_REPORT(_safe_opName_.c_str(),                                                       \
            "Values %s of configuration items %s in configuration file %s are invalid. Reason: %s.",        \
            _safe_values_.c_str(), _safe_items_.c_str(), _safe_configFile_.c_str(), _safe_reason_.c_str()); \
        REPORT_INNER_ERR_MSG("EZ0034",                                                                      \
            "Values %s of configuration items %s in configuration file %s are invalid. Reason: %s.",        \
            _safe_values_.c_str(), _safe_items_.c_str(), _safe_configFile_.c_str(), _safe_reason_.c_str()); \
    } while (0)

#endif /* OP_LOGE_FOR_INVALID_SHAPE */

#endif /* MC2_LOG_COMPAT_H */
