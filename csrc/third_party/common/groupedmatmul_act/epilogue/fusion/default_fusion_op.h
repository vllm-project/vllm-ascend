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
 * \file default_fusion_op.h
 * \brief
 */

#ifndef EPILOGUE_FUSION_DEFAULT_FUSION_OP_H
#define EPILOGUE_FUSION_DEFAULT_FUSION_OP_H
#include "kernel_operator.h"
#include "../../utils/common_utils.h"
#include "../../utils/device_utils.h"

namespace Act {
namespace Gemm {
namespace Block {
template <typename DataTypeOut_, typename DataTypeIn_>
class DefaultFusion {
public:
    using DataTypeOut = DataTypeOut_;
    using DataTypeIn = DataTypeIn_;
    __aicore__ inline DefaultFusion(){};

    struct Arguments {};

    struct Params {};

    __aicore__ inline void Init(Params const& params, int64_t calcM, int64_t calcN, int64_t n) {}
    __aicore__ inline void InitBuffers() {}
    __aicore__ inline int64_t GetUbSizeOneM(int64_t ubCalcN)
    {
        return 0;
    }
    __aicore__ inline void Run(AscendC::LocalTensor<DataTypeOut>& dstLocal, AscendC::LocalTensor<DataTypeIn>& srcLocal,
                               int64_t curAivM, int64_t curAivN, int64_t mIdx, int64_t nIdx)
    {
        dstLocal = srcLocal;
        return;
    }

    __aicore__ inline void operator()(AscendC::LocalTensor<DataTypeOut>& dstLocal,
                                      AscendC::LocalTensor<DataTypeIn>& srcLocal, int64_t curAivM, int64_t curAivN,
                                      int64_t mIdx, int64_t nIdx)
    {
        Run(dstLocal, srcLocal, curAivM, curAivN, mIdx, nIdx);
    }
};
} // namespace Block
} // namespace Gemm
} // namespace Act
#endif // EPILOGUE_FUSION_DEFAULT_FUSION_OP_H
