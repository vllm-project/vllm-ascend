/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

#include "kernel_operator.h"
#include "add_custom_tiling.h"

class KernelAdd {
public:
    __aicore__ inline KernelAdd() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                const __gm__ AddTilingData* tiling)
    {
        this->tiling = tiling;
        uint32_t blockIdx = AscendC::GetBlockIdx();
        if (blockIdx < tiling->blockNum - 1) {
            total = tiling->numPerCore;
            tileNum = total / TILE_LENGTH;
            tailTileElementNum = TILE_LENGTH;
        } else {
            total = tiling->tailNumLastCore;
            tileNum = (total + TILE_LENGTH - 1) / TILE_LENGTH;
            tailTileElementNum = total - TILE_LENGTH * (tileNum - 1);
        }

        xGm.SetGlobalBuffer((__gm__ half *)x + blockIdx * tiling->numPerCore, total);
        yGm.SetGlobalBuffer((__gm__ half *)y + blockIdx * tiling->numPerCore, total);
        zGm.SetGlobalBuffer((__gm__ half *)z + blockIdx * tiling->numPerCore, total);

        pipe.InitBuffer(inQueueX, DOUBLE_BUFFER, TILE_LENGTH * sizeof(half));
        pipe.InitBuffer(inQueueY, DOUBLE_BUFFER, TILE_LENGTH * sizeof(half));
        pipe.InitBuffer(outQueueZ, DOUBLE_BUFFER, TILE_LENGTH * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < tileNum; i++) {
            uint32_t count = (i == tileNum - 1) ? tailTileElementNum : TILE_LENGTH;
            CopyIn(i, count);
            Compute(count);
            CopyOut(i, count);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress, uint32_t count)
    {
        AscendC::LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
        AscendC::LocalTensor<half> yLocal = inQueueY.AllocTensor<half>();
        AscendC::DataCopy(xLocal, xGm[progress * TILE_LENGTH], count);
        AscendC::DataCopy(yLocal, yGm[progress * TILE_LENGTH], count);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }
    __aicore__ inline void Compute(uint32_t count)
    {
        AscendC::LocalTensor<half> xLocal = inQueueX.DeQue<half>();
        AscendC::LocalTensor<half> yLocal = inQueueY.DeQue<half>();
        AscendC::LocalTensor<half> zLocal = outQueueZ.AllocTensor<half>();
        AscendC::Add(zLocal, xLocal, yLocal, count);
        outQueueZ.EnQue<half>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress, uint32_t count)
    {
        AscendC::LocalTensor<half> zLocal = outQueueZ.DeQue<half>();
        AscendC::DataCopy(zGm[progress * TILE_LENGTH], zLocal, count);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    const __gm__ AddTilingData* tiling;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX, inQueueY;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueZ;
    AscendC::GlobalTensor<half> xGm;
    AscendC::GlobalTensor<half> yGm;
    AscendC::GlobalTensor<half> zGm;

    uint32_t total;
    uint32_t tileNum;
    uint32_t tailTileElementNum;
};

extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR tiling)
{
    KernelAdd op;
    op.Init(x, y, z, (__gm__ AddTilingData*)tiling);
    op.Process();
}
