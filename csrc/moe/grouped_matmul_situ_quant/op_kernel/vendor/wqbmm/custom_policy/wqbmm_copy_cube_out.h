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
 * \file wqbmm_copy_cube_out.h
 * \brief
 */
#ifndef WQBMM_COPY_CUBE_OUT_H
#define WQBMM_COPY_CUBE_OUT_H
#include "lib/matmul_intf.h"

namespace WeightQuantBatchMatmulV2::Arch35 {
template <typename IMPL, class A_TYPE, class B_TYPE, class C_TYPE, const auto &MM_CFG,
          typename = std::enable_if_t<!AscendC::Impl::Detail::MatmulFeatureTrait<MM_CFG>::IsNeedUB()>>
class WQBmmCustomCopyCubeOut {
    using DstT = typename C_TYPE::T;
    using SrcT = typename AscendC::GetMmDstType<typename A_TYPE::T>::Type;
    using FixpipeAdaptor = AscendC::Impl::Detail::FixpipeParamsUtil<
        A_TYPE, C_TYPE, MM_CFG, AscendC::Impl::Detail::MatmulFeatureTrait<MM_CFG>::GetFixpipeParamsType()>;

    MATMUL_USE_MODULE(MatmulShapeTiling);
    MATMUL_USE_MODULE(MatmulSubBlockInfo);

public:
    __aicore__ inline WQBmmCustomCopyCubeOut() = default;

    template <bool enSequentialWrite = false, typename ScheduleContext = int>
    __aicore__ inline void Copy(const GlobalTensor<DstT> &gm, const LocalTensor<SrcT> &co1Local, int32_t curRow,
                                int32_t curCol, int32_t baseHeight, int32_t baseWidth, int32_t baseBlockHeight,
                                int32_t baseBlockWidth, const ScheduleContext &context = 0)
    {
        if constexpr (ToMatmulConfig(MM_CFG).intraBlockPartSum) {
            if (!MATMUL_MODULE(MatmulSubBlockInfo)->GetFakeMsg()) {
                CopyOutImpl<enSequentialWrite, const GlobalTensor<DstT>, true>(
                    gm, co1Local, curRow, curCol, baseHeight, baseWidth, baseBlockHeight, baseBlockWidth);
                return;
            }
        }
        CopyOutImpl<enSequentialWrite, const GlobalTensor<DstT>, false>(gm, co1Local, curRow, curCol, baseHeight,
                                                                        baseWidth, baseBlockHeight, baseBlockWidth);
    }

    template <bool enSequentialWrite = false, typename ScheduleContext = int>
    __aicore__ inline void Copy(const LocalTensor<DstT> &co2Local, const LocalTensor<SrcT> &co1Local, int32_t curRow,
                                int32_t curCol, int32_t baseHeight, int32_t baseWidth, int32_t baseBlockHeight,
                                int32_t baseBlockWidth, const ScheduleContext &context = 0)
    {
        baseHeight = (baseHeight + 1) / 2 * 2;  // 2含义：指令要求m必须是偶数，向上对齐到偶数
        CopyOutImpl<enSequentialWrite, const LocalTensor<DstT>>(co2Local, co1Local, curRow, curCol, baseHeight,
                                                                baseWidth, baseBlockHeight, baseBlockWidth);
    }

private:
    template <bool enSequentialWrite, class T, bool IS_INTRA_BLOCK = false>
    __aicore__ inline void CopyOutImpl(const T &dst, const LocalTensor<SrcT> &co1Local, int32_t curRow, int32_t curCol,
                                       int32_t baseHeight, int32_t baseWidth, int32_t baseBlockHeight,
                                       int32_t baseBlockWidth)
    {
        // 2含义：指令要求m必须是偶数，向上对齐到偶数
        ASCENDC_ASSERT((baseHeight % 2 == 0),
                       { KERNEL_LOG(KERNEL_ERROR, "If split M when copy cube out, baseHeight must be even"); });
        if constexpr (C_TYPE::format == CubeFormat::ND || C_TYPE::format == CubeFormat::ND_ALIGN) {
            CopyOutNZ2ND<enSequentialWrite, T, IS_INTRA_BLOCK>(dst, co1Local, curRow, curCol, baseHeight, baseWidth,
                                                               baseBlockHeight, baseBlockWidth);
        } else {
            ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "CopyOut: unsupport Matmul format type."); });
        }
    }

    template <bool enSequentialWrite, class T, bool IS_INTRA_BLOCK = false>
    __aicore__ inline void CopyOutNZ2ND(const T &dst, const LocalTensor<SrcT> &co1Local, int32_t curRow, int32_t curCol,
                                        int32_t baseHeight, int32_t baseWidth, int32_t baseBlockHeight,
                                        int32_t baseBlockWidth)
    {
        int64_t stride = baseWidth;
        int64_t dstOffset = 0;
        if constexpr (C_TYPE::format == CubeFormat::ND_ALIGN) {
            constexpr int64_t blockCount = AscendC::VECTOR_REG_WIDTH / sizeof(SrcT);
            stride = (stride + blockCount - 1) / blockCount * blockCount;
        }

        FixpipeAdaptor fixpipe(baseWidth, baseHeight, baseBlockWidth, baseBlockHeight,
                               MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseM(), stride);
        fixpipe.SetMcgShfMode(AscendC::McgShfMode::DUAL_DST_SPLIT_M);
        fixpipe.SetCastMode();
        fixpipe.template FixpipeOut<T>(dst, co1Local);
    }
};

}  // namespace WeightQuantBatchMatmulV2::Arch35
#endif  // WQBMM_COPY_CUBE_OUT_H
