/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file tiling_base.h
 * \brief
 */

#ifndef MC2_TILING_BASE_H
#define MC2_TILING_BASE_H

namespace Mc2Tiling {
struct AiCoreParams {
    uint64_t ubSize;
    uint64_t blockDim;
    uint64_t aicNum;
    uint64_t l1Size;
    uint64_t l0aSize;
    uint64_t l0bSize;
    uint64_t l0cSize;
};

class TilingBaseClass {
public:
    bool DoTiling(int64_t m, int64_t cols, int64_t topK, int64_t expertCapacity, int64_t expertNum, int64_t activeNum,
                  int64_t dropPadMode, int64_t expertTokensCountOrCumsumFlag, bool expertTokensBeforeCapacityFlag,
                  int64_t inuptXDtypeSize, int64_t quantMode, int64_t scaleDim0, int64_t aivCoreNum,
                  int64_t ubSizePlatForm)
    {
        bool ret = GetShapeAttrsInfo(m, cols, topK, expertCapacity, expertNum, activeNum, dropPadMode,
                                     expertTokensCountOrCumsumFlag, expertTokensBeforeCapacityFlag, inuptXDtypeSize,
                                     quantMode, scaleDim0);
        if (!ret) {
            return ret;
        }
        ret = GetPlatformInfo(aivCoreNum, ubSizePlatForm);
        if (!ret) {
            return ret;
        }
        ret = DoOpTiling();
        if (!ret) {
            return ret;
        }
        ret = GetWorkspaceSize();
        if (!ret) {
            return ret;
        }
        ret = PostTiling();
        if (!ret) {
            return ret;
        }
        tilingKey_ = GetTilingKey();
        return true;
    }

    virtual bool GetPlatformInfo(int64_t aivCoreNum, int64_t ubSizePlatForm) = 0;
    virtual bool GetShapeAttrsInfo(int64_t m, int64_t cols, int64_t topK, int64_t expertCapacity, int64_t expertNum,
                                   int64_t activeNum, int64_t dropPadMode, int64_t expertTokensCountOrCumsumFlag,
                                   bool expertTokensBeforeCapacityFlag, int64_t inuptXDtypeSize, int64_t quantMode,
                                   int64_t scaleDim0) = 0;

    virtual bool DoOpTiling() = 0;
    virtual bool GetWorkspaceSize() = 0;
    virtual bool PostTiling()
    {
        return true;
    }
    virtual uint64_t GetTilingKey() const = 0;
    uint32_t blockDim_{0};
    uint64_t workspaceSize_{0};
    uint64_t tilingKey_{0};
    AiCoreParams aicoreParams_{0, 0, 0, 0, 0, 0, 0};
};

} // namespace Mc2Tiling

#endif // MC2_TILING_BASE_H