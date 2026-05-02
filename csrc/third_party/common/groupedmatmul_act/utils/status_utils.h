/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file status_utils.h
 * \brief
 */

#ifndef UTILS_STATUS_UTILS_H
#define UTILS_STATUS_UTILS_H

namespace Act {
namespace Gemm {
enum class Status {
    success,
    batchErrorExcceedsLimit,
    mnkErrorExceedsLimit,
    mkErrorMatrixExceedsLimit,
    kmErrorMatrixExceedsLimit,
    knErrorMatrixExceedsLimit,
    nkErrorMatrixExceedsLimit,
    bf16BiasErrorInvalidDataType,
    tileShapeErrorExceedsLimit,
    l1L0ErrorExceedsLimit,
    l1L0ErrorNotAlign,
    l1MnL0MnErrorNotSame,
    l1kErrorSmallerL0k,
    l1kErrorL0kNotAlign,
    l1L0ErrorNotAlignInt8
};
constexpr const char* GetStatusString(Status status)
{
    switch (status) {
        case Status::success:
            return "Success.\n";
        case Status::batchErrorExcceedsLimit:
            return "[ERROR] batch cannot greater than int32 max val 2147483647.\n";
        case Status::mnkErrorExceedsLimit:
            return "[ERROR] mnk cannot greater than int32 max val 2147483647.\n";
        case Status::mkErrorMatrixExceedsLimit:
            return "[ERROR] mk matrix k cannot greater than 65535.\n";
        case Status::kmErrorMatrixExceedsLimit:
            return "[ERROR] km matrix m cannot greater than 65535.\n";
        case Status::knErrorMatrixExceedsLimit:
            return "[ERROR] kn matrix n cannot greater than 65535.\n";
        case Status::nkErrorMatrixExceedsLimit:
            return "[ERROR] nk matrix k cannot greater than 65535.\n";
        case Status::bf16BiasErrorInvalidDataType:
            return "[ERROR] bfloat16_t type is not supported for bias.\n";
        case Status::tileShapeErrorExceedsLimit:
            return "[ERROR] L1 or L0 tile shape error, mnk config exceed limit.\n";
        case Status::l1L0ErrorExceedsLimit:
            return "[ERROR] L1/L0 tile shape m can't greater than 128 and n can't grater than 256.\n";
        case Status::l1L0ErrorNotAlign:
            return "[ERROR] L1/L0 tile shape m/n/k should be an integral multiple of 16.\n";
        case Status::l1MnL0MnErrorNotSame:
            return "[ERROR] L1 tile shape m/n should be same with L0 tile shape m/n.\n";
        case Status::l1kErrorSmallerL0k:
            return "[ERROR] L1 tile shape k can't less than L0 tile shape k.\n";
        case Status::l1kErrorL0kNotAlign:
            return "[ERROR] L1 tile shape k should be an integral multiple of L0 tile shape k.\n";
        case Status::l1L0ErrorNotAlignInt8:
            return "[ERROR] L0 tile shape m/n should be an integral multiple of 32 for quant.\n";
        default:
            return "[ERROR] unknown error.\n";
    }
}

#define CHECK_AND_RETURN(status)                                                                                       \
    do {                                                                                                               \
        Status ret = status;                                                                                           \
        if (ret != Status::success) {                                                                                  \
            return ret;                                                                                                \
        }                                                                                                              \
    } while (0)

#define ACT_CHECK(status)                                                                                              \
    do {                                                                                                               \
        Status ret = status;                                                                                           \
        if (ret != Status::success) {                                                                                  \
            std::cerr << "Got act error: " << GetStatusString(ret) << std::endl;                                       \
            return;                                                                                                    \
        }                                                                                                              \
    } while (0)

} // namespace Gemm
} // namespace Act
#endif