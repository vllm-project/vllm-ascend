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
* \file mla_custom_matmul_policy_common.h
* \brief
*/

#ifndef MLA_CUSTOM_MATMUL_POLICY_COMMON_H
#define MLA_CUSTOM_MATMUL_POLICY_COMMON_H

namespace AscendC {

constexpr uint32_t Q_VEC1_BUFFER_SIZE = 128 * 512 * 2;
constexpr uint32_t K_V_BUFFER_SIZE = 192 * 1024 * 2;
constexpr uint32_t WIDTH_512 = 512;
constexpr uint32_t Q_VEC1_INDEX = 0;
constexpr uint32_t K_V_INDEX = 1;
constexpr uint32_t MM1_INDEX = 1;
constexpr uint32_t MM2_INDEX = 2;
constexpr uint32_t L1_BUF_NUM = 2;
constexpr uint32_t NUM_2 = 2;
constexpr uint32_t NUM_4 = 4;
constexpr uint32_t BLOCK_CEILDIV_NUM = 15;

// 定义L1 buffer全局管理结构体
struct GlobalL1Array {
    __aicore__ inline GlobalL1Array() {};
    TQue<TPosition::A1, 1> localQue;
    TBuffAddr srcAddr;
    int32_t cacheSize{0}; // 表示已缓存base块的个数
    int32_t bufferSize{0}; // base块元素个数
};

__BLOCK_LOCAL__ __inline__ GlobalL1Array *l1Global;

__aicore__ inline void InitL1Buffer(TPipe *tPipe, GlobalL1Array *l1Global)
{
    if (g_coreType == AIC) {
        tPipe->InitBuffer(l1Global[Q_VEC1_INDEX].localQue, 1, Q_VEC1_BUFFER_SIZE);
        tPipe->InitBuffer(l1Global[K_V_INDEX].localQue, 1, K_V_BUFFER_SIZE);
    }
}

namespace Impl {
namespace Detail {
template <typename IMPL, class INPUT_TYPE, const auto &MM_CFG>
class MlaCubeInBuffer {
    MATMUL_USE_MODULE(Context);
    using SrcT = typename INPUT_TYPE::TRANS_T;
public:
    __aicore__ inline MlaCubeInBuffer() {}
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

    __aicore__ inline void FreeTensor(int32_t bufferPos = -1, const LocalTensor<SrcT>& tensor = LocalTensor<SrcT>{})
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

    __aicore__ inline void EnQue(LocalTensor<SrcT>& tensor) {}

    __aicore__ inline void DeQue() {}

    __aicore__ inline void Reset() {}

private:
    int32_t baseBlockSize_{0};
};

template <typename IMPL, class INPUT_TYPE, const auto &MM_CFG>
class MlaContext {
    using SrcT = typename INPUT_TYPE::TRANS_T;

public:
    __aicore__ inline MlaContext() {}
    __aicore__ inline void Init() {}
    int32_t mmIdx{0};
    int32_t l1IndexA = -1;
    LocalTensor<SrcT> CurrentTensorA;
    bool needAllocA{true};
    bool needFreeA{true};
    bool isFirstBaseBlockA{true};
    bool isLastBaseBlockA{false};
    bool isBufferFullA{false};
    int32_t baseBlockSizeA{0};

    int32_t l1IndexB = -1;
    LocalTensor<SrcT> CurrentTensorB;
    bool needAllocB{true};
    bool needFreeB{true};
    bool isFirstBaseBlockB{true};
    bool isLastBaseBlockB{false};
    int32_t baseBlockSizeB{0};
};
}
}
}

#endif // MLA_CUSTOM_MATMUL_POLICY_COMMON_H