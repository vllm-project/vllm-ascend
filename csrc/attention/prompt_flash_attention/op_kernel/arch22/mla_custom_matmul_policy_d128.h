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
 * \file mla_custom_matmul_policy_d128.h
 * \brief
 */

#ifndef MLA_CUSTOM_MATMUL_POLICY_D128_H
#define MLA_CUSTOM_MATMUL_POLICY_D128_H

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
class MlaCopyCubeInMM2HeadDim128 {
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
    inline __aicore__ MlaCopyCubeInMM2HeadDim128() = default;
    inline __aicore__ ~MlaCopyCubeInMM2HeadDim128() = default;

    __aicore__ inline void Init() {}
    __aicore__ inline void SetInput(const LocalTensor<SrcT> &localMatrix, bool isTranspose) {}
    __aicore__ inline void SetInput(const GlobalTensor<SrcT> &globalMatrix, bool isTranspose)
    {
        MATMUL_MODULE(MatmulTensorInfo)->template SetGlobalTensor<false>(globalMatrix, isTranspose);
        srcGlobalAddr_ = globalMatrix.address_;
    }

    // L1无法容纳整个A矩阵，需要分多次alloc、free
    __aicore__ inline LocalTensor<TransT> LoadPartDataA(int32_t curRow, int32_t curCol, int32_t tileHeight,
        int32_t tileWidth, uint8_t l1Index, int32_t posL1,
        int32_t baseBlockSize)
    {
        MATMUL_MODULE(Context)->l1IndexA = l1Index;
        int32_t baseBlockNum = l1Global[l1Index].bufferSize / baseBlockSize;

        posL1 = posL1 % baseBlockNum;
        // 当搬运base块到达分配的L1上限时，需要重新alloc，并把缓存数清零
        if (posL1 == 0) {
            MATMUL_MODULE(Context)->needAllocA = true;
            l1Global[l1Index].cacheSize = 0;
        }

        // 当搬运base块到达分配的L1上限时,需要freeA
        MATMUL_MODULE(Context)->isBufferFullA = false;
        if (posL1 == baseBlockNum - 1) {
            MATMUL_MODULE(Context)->isBufferFullA = true;
        }

        // single块内的base块复用，需要在搬运之前判断是否命中cache
        if (posL1 < l1Global[l1Index].cacheSize) {
            MATMUL_MODULE(Context)->CurrentTensorA.SetAddr(l1Global[l1Index].srcAddr);
            return MATMUL_MODULE(Context)->CurrentTensorA[posL1 * baseBlockSize];
        }

        if (MATMUL_MODULE(Context)->needAllocA) {
            MATMUL_MODULE(Context)->CurrentTensorA = MATMUL_MODULE(CubeInBuffer)->AllocTensor(posL1);
        } else {
            LocalTensor<TransT> aL1Tensor;
            aL1Tensor.SetAddr(l1Global[l1Index].srcAddr);
            MATMUL_MODULE(Context)->CurrentTensorA = aL1Tensor[posL1 * baseBlockSize];
        }

        MATMUL_MODULE(DataCopyUtils)->CopyTileToCube(MATMUL_MODULE(Context)->CurrentTensorA, curRow, curCol,
            tileHeight, tileWidth);
        if (posL1 == 0) {
            l1Global[l1Index].srcAddr = MATMUL_MODULE(Context)->CurrentTensorA.address_;
            l1Global[l1Index].bufferSize = MATMUL_MODULE(Context)->CurrentTensorA.GetSize();
        }
        l1Global[l1Index].cacheSize++;
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

        // single块内的base块复用，需要在搬运之前判断是否命中cache
        if (posL1 < l1Global[l1Index].cacheSize) {
            MATMUL_MODULE(Context)->CurrentTensorB.SetAddr(l1Global[l1Index].srcAddr);
            return MATMUL_MODULE(Context)->CurrentTensorB[posL1 * baseBlockSize];
        }

        if (MATMUL_MODULE(Context)->needAllocB) {
            MATMUL_MODULE(Context)->CurrentTensorB = MATMUL_MODULE(CubeInBuffer)->AllocTensor(posL1);
        } else {
            LocalTensor<TransT> bL1Tensor;
            bL1Tensor.SetAddr(l1Global[l1Index].srcAddr);
            MATMUL_MODULE(Context)->CurrentTensorB = bL1Tensor[posL1 * baseBlockSize];
        }
        MATMUL_MODULE(DataCopyUtils)->CopyTileToCube(MATMUL_MODULE(Context)->CurrentTensorB, curRow, curCol,
            tileHeight, tileWidth);
        if (posL1 == 0) {
            l1Global[l1Index].srcAddr = MATMUL_MODULE(Context)->CurrentTensorB.address_;
            l1Global[l1Index].bufferSize = MATMUL_MODULE(Context)->CurrentTensorB.GetSize();
        }

        l1Global[l1Index].cacheSize++;

        l1Global[l1Index].localQue.EnQue(MATMUL_MODULE(Context)->CurrentTensorB);
        l1Global[l1Index].localQue.DeQue<TransT>();

        return MATMUL_MODULE(Context)->CurrentTensorB;
    }

    __aicore__ inline LocalTensor<TransT> LoadData(int32_t curRow, int32_t curCol,
        int32_t tileHeight, int32_t tileWidth)
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
        orgWidth_ = MATMUL_MODULE(CopyCubeInParams)->GetOrgWidth();
        if constexpr (INPUT_TYPE::TAG == InputTypeTag::A) {
            MATMUL_MODULE(Context)->baseBlockSizeA = baseBlockSize;
            MATMUL_MODULE(Context)->isFirstBaseBlockA = isFirstBaseBlock;
            MATMUL_MODULE(Context)->isLastBaseBlockA = isLastBaseBlock;
            MATMUL_MODULE(Context)->needAllocA = MATMUL_MODULE(Context)->isFirstBaseBlockA &&
                MATMUL_MODULE(Context)->isFirstBaseBlockB;
            posL1 = curRow * MATMUL_MODULE(CopyCubeInParams)->GetTotalCol() + curCol;
            l1Index = Q_VEC1_INDEX;
            return LoadPartDataA(curRow, curCol, tileHeight, tileWidth, l1Index, posL1, baseBlockSize);
        }

        if constexpr (INPUT_TYPE::TAG == InputTypeTag::B) {
            MATMUL_MODULE(Context)->baseBlockSizeB = baseBlockSize;
            MATMUL_MODULE(Context)->isFirstBaseBlockB = isFirstBaseBlock;
            MATMUL_MODULE(Context)->isLastBaseBlockB = isLastBaseBlock;
            posL1 = curCol * MATMUL_MODULE(CopyCubeInParams)->GetTotalRow() + curRow;
            l1Index = K_V_INDEX;
            MATMUL_MODULE(Context)->needFreeA = MATMUL_MODULE(Context)->isLastBaseBlockA &&
                MATMUL_MODULE(Context)->isLastBaseBlockB;
            if (MATMUL_MODULE(Context)->isBufferFullA) {
                MATMUL_MODULE(Context)->needFreeA = true;
            }
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
    int64_t orgWidth_{0};
    int64_t offsetA[8] = {0};
    int64_t offsetB[16] = {0};
    int64_t cacheBaseBlockNum = 0;
};

template <typename IMPL, class INPUT_TYPE, const auto &MM_CFG>
class FaCubeInBufferMM2HeadDim128 {
    MATMUL_USE_MODULE(Context);

    using SrcT = typename INPUT_TYPE::TRANS_T;

public:
    __aicore__ inline FaCubeInBufferMM2HeadDim128() {}

    __aicore__ inline void Init(int32_t baseBlockSize, int32_t cacheSize, int32_t reduceAxisCnt) {
        baseBlockSize_ = baseBlockSize;
    }

    __aicore__ inline void Destroy()
    {
        if constexpr (INPUT_TYPE::TAG == InputTypeTag::A) {
            MATMUL_MODULE(Context)->needAllocA = true;
            MATMUL_MODULE(Context)->needFreeA = true;
            MATMUL_MODULE(Context)->isFirstBaseBlockA = true;
            MATMUL_MODULE(Context)->isLastBaseBlockA = false;
            MATMUL_MODULE(Context)->isBufferFullA = false;
            l1Global[MATMUL_MODULE(Context)->l1IndexA].cacheSize = 0;
            l1Global[MATMUL_MODULE(Context)->l1IndexA].localQue.FreeAllEvent();
        }

        if constexpr (INPUT_TYPE::TAG == InputTypeTag::B) {
            MATMUL_MODULE(Context)->needAllocB = true;
            MATMUL_MODULE(Context)->needFreeB = true;
            MATMUL_MODULE(Context)->isFirstBaseBlockB = true;
            MATMUL_MODULE(Context)->isLastBaseBlockB = false;
            l1Global[MATMUL_MODULE(Context)->l1IndexB].cacheSize = 0;
            l1Global[MATMUL_MODULE(Context)->l1IndexB].localQue.FreeAllEvent();
        }
    }

    __aicore__ inline LocalTensor<SrcT> AllocTensor(int32_t bufferPos = -1)
    {
        if constexpr (INPUT_TYPE::TAG == InputTypeTag::A) {
            LocalTensor<SrcT> aL1Buffer = l1Global[MATMUL_MODULE(Context)->l1IndexA].localQue.template AllocTensor<SrcT>();
            return aL1Buffer[bufferPos * MATMUL_MODULE(Context)->baseBlockSizeA];
        }

        if constexpr (INPUT_TYPE::TAG == InputTypeTag::B) {
            LocalTensor<SrcT> bL1Buffer = l1Global[MATMUL_MODULE(Context)->l1IndexB].localQue.template AllocTensor<SrcT>();
            return bL1Buffer[bufferPos * MATMUL_MODULE(Context)->baseBlockSizeB];
        }
    }

    __aicore__ inline void FreeTensor(int32_t bufferPos = -1, const LocalTensor<SrcT> &tensor = LocalTensor<SrcT>{})
    {
        if constexpr (INPUT_TYPE::TAG == InputTypeTag::A) {
            if (MATMUL_MODULE(Context)->needFreeA) {
                l1Global[MATMUL_MODULE(Context)->l1IndexA].localQue.FreeTensor(MATMUL_MODULE(Context)->CurrentTensorA);
            }
        }

        if constexpr (INPUT_TYPE::TAG == InputTypeTag::B) {
            if (MATMUL_MODULE(Context)->needFreeB) {
                l1Global[MATMUL_MODULE(Context)->l1IndexB].localQue.FreeTensor(MATMUL_MODULE(Context)->CurrentTensorB);
            }
        }
    }

    __aicore__ inline void EnQue(LocalTensor<SrcT> &tensor) {}
    __aicore__ inline void DeQue() {}
    __aicore__ inline void Reset() {}

private:
    int32_t baseBlockSize_{0};
};

template <const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
class MlaMatmulPolicyMM2HeadDim128 : public MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>
{
public:
    using Context = MlaContext<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CopyCubeInA = MlaCopyCubeInMM2HeadDim128<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CopyCubeInB = MlaCopyCubeInMM2HeadDim128<IMPL, MatmulInputBType<B_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CubeInBufferA = MlaCubeInBuffer<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CubeInBufferB = MlaCubeInBuffer<IMPL, MatmulInputBType<B_TYPE, typename A_TYPE::T>, MM_CFG>;
};
}
}
}

#endif // MLA_CUSTOM_MATMUL_POLICY_D128_H