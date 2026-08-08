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
 * \file mla_custom_matmul_policy_d192.h
 * \brief
 */

#ifndef MLA_CUSTOM_MATMUL_POLICY_D192_H
#define MLA_CUSTOM_MATMUL_POLICY_D192_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"
#include "mla_custom_matmul_policy_common.h"

namespace AscendC {
namespace Impl {
namespace Detail {
template <typename IMPL, class INPUT_TYPE, const auto &MM_CFG> 
class MlaCopyCubeInMM1HeadDim192 {
    MATMUL_USE_MODULE(Context);
    MATMUL_USE_MODULE_ON(MatmulTensorInfo, INPUT_TYPE::TAG);
    MATMUL_USE_MODULE(MatmulShapeInfo);
    MATMUL_USE_MODULE(MatmulUserDefineInfo);
    MATMUL_USE_MODULE_ON(CubeInBuffer, INPUT_TYPE::TAG);
    MATMUL_USE_MODULE_ON(CopyCubeInParams, INPUT_TYPE::TAG);
    MATMUL_USE_MODULE_ON(DataCopyUtils, INPUT_TYPE::TAG);

    using TransT = typename INPUT_TYPE::TRANS_T;
    using SrcT = typename INPUT_TYPE::T;

public:
    using InputType = INPUT_TYPE;
    inline __aicore__ MlaCopyCubeInMM1HeadDim192() = default;
    inline __aicore__ ~MlaCopyCubeInMM1HeadDim192() = default;    
    static constexpr uint32_t ALIGN_BYTE_NUM_16 = 16; // 搬运时16对齐

    __aicore__ inline void Init() {}

    __aicore__ inline void SetInput(const LocalTensor<SrcT> &localMatrix, bool isTranspose) {}

    __aicore__ inline void SetInput(const GlobalTensor<SrcT> &globalMatrix, bool isTranspose)
    {
        MATMUL_MODULE(MatmulTensorInfo)->template SetGlobalTensor<false>(globalMatrix, isTranspose);
        srcGlobalAddr_ = globalMatrix.address_;
    }

    // L1可以容纳整个A矩阵，全局alloc一次，free一次，K方向全载
    __aicore__ inline LocalTensor<TransT> LoadFullDataA(int32_t curRow, int32_t curCol, int32_t tileHeight,
        int32_t tileWidth, uint8_t l1Index, int32_t posL1,
        int32_t baseBlockSize)
    {
        MATMUL_MODULE(Context)->l1IndexA = l1Index;
        cacheBaseBlockNum = MATMUL_MODULE(CopyCubeInParams)->GetTotalCol();
        int32_t totalBlockNum = MATMUL_MODULE(CopyCubeInParams)->GetTotalCol() *
            MATMUL_MODULE(CopyCubeInParams)->GetTotalRow();
        if (posL1 < totalBlockNum - 1) {
            this->offsetA[posL1 + 1] = this->offsetA[posL1] + AlignUp(tileHeight, ALIGN_BYTE_NUM_16);
        }

        // single块内的base块复用，需要在搬运之前判断是否命中cache
        if (posL1 < l1Global[l1Index].cacheSize) {
            MATMUL_MODULE(Context)->CurrentTensorA.SetAddr(l1Global[l1Index].srcAddr);
            return MATMUL_MODULE(Context)->CurrentTensorA[this->offsetA[posL1]];
        }

        if (MATMUL_MODULE(Context)->needAllocA) {
            MATMUL_MODULE(Context)->CurrentTensorA = MATMUL_MODULE(CubeInBuffer)->AllocTensor(posL1);
        } else {
            LocalTensor<TransT> aL1Tensor;
            aL1Tensor.SetAddr(l1Global[l1Index].srcAddr);
            MATMUL_MODULE(Context)->CurrentTensorA = aL1Tensor[this->offsetA[posL1]];
        }

        // k方向一次搬运两份
        MATMUL_MODULE(DataCopyUtils)->CopyTileToCube(MATMUL_MODULE(Context)->CurrentTensorA, curRow, curCol,
            tileHeight, singleWidth_);

        if (posL1 == 0) {
            l1Global[l1Index].srcAddr = MATMUL_MODULE(Context)->CurrentTensorA.address_;
            l1Global[l1Index].bufferSize = MATMUL_MODULE(Context)->CurrentTensorA.GetSize();
        }

        // 一次搬运两份，缓存个数加2
        l1Global[l1Index].cacheSize += cacheBaseBlockNum;

        l1Global[l1Index].localQue.EnQue(MATMUL_MODULE(Context)->CurrentTensorA);
        l1Global[l1Index].localQue.DeQue<TransT>();

        return MATMUL_MODULE(Context)->CurrentTensorA;
    }

    // L1可以容纳整个B矩阵，全局alloc一次，free一次
    __aicore__ inline LocalTensor<TransT> LoadFullDataB(int32_t curRow, int32_t curCol, int32_t tileHeight,
        int32_t tileWidth, uint8_t l1Index, int32_t posL1,
        int32_t baseBlockSize)
    {
        MATMUL_MODULE(Context)->l1IndexB = l1Index;
        MATMUL_MODULE(Context)->needAllocB = MATMUL_MODULE(Context)->isFirstBaseBlockA &&
            MATMUL_MODULE(Context)->isFirstBaseBlockB;
        MATMUL_MODULE(Context)->needFreeB = MATMUL_MODULE(Context)->isLastBaseBlockA &&
            MATMUL_MODULE(Context)->isLastBaseBlockB;
        
        int32_t totalBlockNum = MATMUL_MODULE(CopyCubeInParams)->GetTotalCol() *
            MATMUL_MODULE(CopyCubeInParams)->GetTotalRow();
        if (posL1 < totalBlockNum - 1) {
            this->offsetB[posL1 + 1] = this->offsetB[posL1] + AlignUp(tileHeight, ALIGN_BYTE_NUM_16) *
                AlignUp(tileWidth, ALIGN_BYTE_NUM_16);
        }
        
        // single块内的base块复用，需要在搬运之前判断是否命中cache
        if (posL1 < l1Global[l1Index].cacheSize) {
            MATMUL_MODULE(Context)->CurrentTensorB.SetAddr(l1Global[l1Index].srcAddr);
            return MATMUL_MODULE(Context)->CurrentTensorB[this->offsetB[posL1]];
        }

        if (MATMUL_MODULE(Context)->needAllocB) {
            MATMUL_MODULE(Context)->CurrentTensorB = MATMUL_MODULE(CubeInBuffer)->AllocTensor(posL1);
        } else {
            LocalTensor<TransT> bL1Tensor;
            bL1Tensor.SetAddr(l1Global[l1Index].srcAddr);
            MATMUL_MODULE(Context)->CurrentTensorB = bL1Tensor[this->offsetB[posL1]];
        }

        // k方向一次搬运两份
        cacheBaseBlockNum = NUM_2;
        MATMUL_MODULE(DataCopyUtils)->CopyTileToCube(MATMUL_MODULE(Context)->CurrentTensorB, curRow, curCol,
            singleHeight_, tileWidth);

        if (posL1 == 0) {
            l1Global[l1Index].srcAddr = MATMUL_MODULE(Context)->CurrentTensorB.address_;
            l1Global[l1Index].bufferSize = MATMUL_MODULE(Context)->CurrentTensorB.GetSize();
        }

        // 一次搬运两份，缓存个数加2
        l1Global[l1Index].cacheSize += cacheBaseBlockNum;

        l1Global[l1Index].localQue.EnQue(MATMUL_MODULE(Context)->CurrentTensorB);
        l1Global[l1Index].localQue.DeQue<TransT>();

        return MATMUL_MODULE(Context)->CurrentTensorB;
    }

    __aicore__ inline LocalTensor<TransT> LoadData(int32_t curRow, int32_t curCol, int32_t tileHeight,
        int32_t tileWidth)
    {
        LocalTensor<TransT> CurrentTensor;
        uint8_t l1Index = -1;
        int32_t posL1 = 0;
        int32_t baseBlockSize = MATMUL_MODULE(CopyCubeInParams)->GetBufferSize();
        bool isFirstBaseBlock = curRow == 0 && curCol == 0;
        bool isLastBaseBlock = (curRow == MATMUL_MODULE(CopyCubeInParams)->GetTotalRow() - 1) &&
            (curCol == MATMUL_MODULE(CopyCubeInParams)->GetTotalCol() - 1);
        MATMUL_MODULE(Context)->mmIdx = static_cast<int32_t>(MATMUL_MODULE(MatmulUserDefineInfo)->GetSelfDefineData());
        singleHeight_ = MATMUL_MODULE(CopyCubeInParams)->GetSingleHeight();
        singleWidth_ = MATMUL_MODULE(CopyCubeInParams)->GetSingleWidth();

        if constexpr (INPUT_TYPE::TAG == InputTypeTag::A) {
            MATMUL_MODULE(Context)->baseBlockSizeA = baseBlockSize;
            MATMUL_MODULE(Context)->isFirstBaseBlockA = isFirstBaseBlock;
            MATMUL_MODULE(Context)->isLastBaseBlockA = isLastBaseBlock;
            MATMUL_MODULE(Context)->needAllocA = MATMUL_MODULE(Context)->isFirstBaseBlockA &&
                MATMUL_MODULE(Context)->isFirstBaseBlockB;
            posL1 = curRow * MATMUL_MODULE(CopyCubeInParams)->GetTotalCol() + curCol;
            l1Index = Q_VEC1_INDEX;
            return LoadFullDataA(curRow, curCol, tileHeight, tileWidth, l1Index, posL1, baseBlockSize);
        }

        if constexpr (INPUT_TYPE::TAG == InputTypeTag::B) {
            MATMUL_MODULE(Context)->baseBlockSizeB = baseBlockSize;
            MATMUL_MODULE(Context)->isFirstBaseBlockB = isFirstBaseBlock;
            MATMUL_MODULE(Context)->isLastBaseBlockB = isLastBaseBlock;
            posL1 = curCol * MATMUL_MODULE(CopyCubeInParams)->GetTotalRow() + curRow;
            l1Index = K_V_INDEX;
            MATMUL_MODULE(Context)->needFreeA = MATMUL_MODULE(Context)->isLastBaseBlockA &&
                MATMUL_MODULE(Context)->isLastBaseBlockB;
            return LoadFullDataB(curRow, curCol, tileHeight, tileWidth, l1Index, posL1, baseBlockSize);
        }
        return CurrentTensor;
    }

    __aicore__ inline void ClearLoadData(const LocalTensor<TransT> &tensor = LocalTensor<TransT>{},
        int32_t curRow = 0, int32_t curCol = 0)
    {
        MATMUL_MODULE(CubeInBuffer)->FreeTensor();
    }

    __aicore__ inline void Reset() {}
    __aicore__ inline void Destroy()
    {
        MATMUL_MODULE(CubeInBuffer)->Destroy();
    }

private:
    int32_t baseBlockSize{0};
    __gm__ SrcT *srcGlobalAddr_{nullptr};
    int64_t singleHeight_{0};
    int64_t singleWidth_{0};
    int64_t offsetA[8] = {0};
    int64_t offsetB[16] = {0};
    int64_t cacheBaseBlockNum = 0;
};

template <const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
class MlaMatmulPolicyMM1HeadDim192 : public MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>
{
public:
    using Context = MlaContext<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CopyCubeInA = MlaCopyCubeInMM1HeadDim192<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CopyCubeInB = MlaCopyCubeInMM1HeadDim192<IMPL, MatmulInputBType<B_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CubeInBufferA = MlaCubeInBuffer<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CubeInBufferB = MlaCubeInBuffer<IMPL, MatmulInputBType<B_TYPE, typename A_TYPE::T>, MM_CFG>;
};
}
}
}

#endif // MLA_CUSTOM_MATMUL_POLICY_D192_H