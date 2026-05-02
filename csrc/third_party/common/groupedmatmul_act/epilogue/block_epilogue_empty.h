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
 * \file block_epilogue_empty.h
 * \brief
 */

#ifndef EPILOGUE_BLOCK_EPILOGUE_EMPTY_H
#define EPILOGUE_BLOCK_EPILOGUE_EMPTY_H
#include "kernel_operator.h"
#include "fusion/default_fusion_op.h"
#include "../utils/common_utils.h"
#include "../utils/device_utils.h"
#include "../utils/status_utils.h"

namespace Act {
namespace Gemm {
namespace Block {

class BlockEpilogueEmpty {
public:
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t>;

    struct Arguments {
        Arguments() = default;
    };

    struct Params {
        Params() = default;
    };

    __aicore__ inline BlockEpilogueEmpty() {}

    __aicore__ inline void Run()
    {
        return;
    }

    __aicore__ inline void operator()(Arguments const& params)
    {
        Run();
    }

    __host_aicore__ static Params InitParams(Arguments const& args, GM_ADDR workspaceGm)
    {
        Params params = {};
        return params;
    }

    __host_aicore__ static size_t GetWorkspaceSize(int64_t blockNum, int64_t l1M, int64_t l1N)
    {
        return 0;
    }

    __host_aicore__ static Status CanImplement(Arguments const& args)
    {
        return Status::success;
    }

    __aicore__ inline void operator()(BlockShape const& blockShape, BlockCoord const& blockCoord,
                                      int64_t dstStartOffset = 0, int64_t srcStartOffset = 0)
    {
        return;
    }
};
} // namespace Block
} // namespace Gemm
} // namespace Act
#endif
