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
 * \file qbmm_copy_cube_out.h
 * \brief
 */
#ifndef MC2_QBMM_COPY_CUBE_OUT_H
#define MC2_QBMM_COPY_CUBE_OUT_H
#include "lib/matmul_intf.h"

namespace AscendC {

template <typename IMPL, class A_TYPE, class B_TYPE, class C_TYPE, const auto &MM_CFG, McgShfMode FIXPIPE_MODE>
class QBmmCustomCopyCubeOut {
    using DstT = typename C_TYPE::T;
    using SrcT = typename GetMmDstType<typename A_TYPE::T>::Type;
    using FixpipeAdaptor = AscendC::Impl::Detail::FixpipeParamsUtil<
        A_TYPE, C_TYPE, MM_CFG, AscendC::Impl::Detail::MatmulFeatureTrait<MM_CFG>::GetFixpipeParamsType()>;

    MATMUL_USE_MODULE(Context);
    MATMUL_USE_MODULE(MatmulQuantProcessor);
    MATMUL_USE_MODULE(MatmulShapeInfo);
    MATMUL_USE_MODULE(MatmulShapeTiling);
    MATMUL_USE_MODULE(MatmulUserDefineInfo);
    MATMUL_USE_MODULE(MatmulSubBlockInfo);

public:
    __aicore__ inline QBmmCustomCopyCubeOut() = default;

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
        if constexpr (FIXPIPE_MODE == McgShfMode::DUAL_DST_SPLIT_M) {
            baseHeight = (baseHeight + 1) / 2 * 2; // 指令要求m必须是偶数，向上对齐到2的倍数
        }
        CopyOutImpl<enSequentialWrite>(co2Local, co1Local, curRow, curCol, baseHeight, baseWidth, baseBlockHeight,
                                       baseBlockWidth);
    }

private:
    template <bool enSequentialWrite, class T, bool IS_INTRA_BLOCK = false>
    __aicore__ inline void CopyOutImpl(const T &dst, const LocalTensor<SrcT> &co1Local, int32_t curRow, int32_t curCol,
                                       int32_t baseHeight, int32_t baseWidth, int32_t baseBlockHeight,
                                       int32_t baseBlockWidth)
    {
        if constexpr (FIXPIPE_MODE == McgShfMode::DUAL_DST_SPLIT_M && PhyPosIsUB(C_TYPE::pos)) {
            ASCENDC_ASSERT((baseHeight % 2 == 0), // 校验tileHeight是否2对齐
                           { KERNEL_LOG(KERNEL_ERROR, "If split M when copy cube out, baseHeight must be even."); });
        }
        if constexpr (C_TYPE::format == CubeFormat::ND || C_TYPE::format == CubeFormat::ND_ALIGN) {
            CopyOutNZ2ND<enSequentialWrite, T, IS_INTRA_BLOCK>(dst, co1Local, curRow, curCol, baseHeight, baseWidth,
                                                               baseBlockHeight, baseBlockWidth);
        } else {
            ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Copy: unsupport Matmul format type."); });
        }
    }

    template <bool enSequentialWrite, class T, bool IS_INTRA_BLOCK = false>
    __aicore__ inline void CopyOutNZ2ND(const T &dst, const LocalTensor<SrcT> &co1Local, int32_t curRow, int32_t curCol,
                                        int32_t baseHeight, int32_t baseWidth, int32_t baseBlockHeight,
                                        int32_t baseBlockWidth)
    {
        auto stride = baseWidth;
        int64_t dstOffset = 0;
        if constexpr (!enSequentialWrite) {
            stride = GetOrgWidth<IS_INTRA_BLOCK>();
            if constexpr (!IsBasic(MM_CFG)) {
                dstOffset = GetDstOffset(curRow, curCol, baseHeight, stride);
            }
        }
        if constexpr (FIXPIPE_MODE == McgShfMode::DUAL_DST_SPLIT_N && PhyPosIsUB(C_TYPE::pos)) {
            stride = stride >> 1;
        }

        FixpipeAdaptor fixpipe(baseWidth, baseHeight, baseBlockWidth, baseBlockHeight,
                               MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseM(), stride);
        SetFixpipeParams(fixpipe);
        CopyTensor(dst[dstOffset], co1Local, fixpipe, curCol, baseWidth);
    }

    __aicore__ inline int64_t GetDstOffset(int32_t curRow, int32_t curCol, int32_t baseHeight, int32_t stride)
    {
        int64_t dstOffset = 0;
        if constexpr (AscendC::Impl::Detail::MatmulFeatureTrait<MM_CFG>::IsSupportL0CToUB() &&
                      PhyPosIsUB(C_TYPE::pos) &&
                      ((A_TYPE::ibShare && B_TYPE::ibShare) || FIXPIPE_MODE == McgShfMode::DUAL_DST_SPLIT_M)) {
            dstOffset = (static_cast<int64_t>(static_cast<int64_t>(
                             curRow * MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseM() * stride)) >>
                         1) +
                        static_cast<int64_t>(curCol * MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseN());
        } else if constexpr (AscendC::Impl::Detail::MatmulFeatureTrait<MM_CFG>::IsSupportL0CToUB() &&
                             PhyPosIsUB(C_TYPE::pos) && FIXPIPE_MODE == McgShfMode::DUAL_DST_SPLIT_N) {
            dstOffset =
                static_cast<int64_t>(curRow * MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseM() * stride) +
                static_cast<int64_t>(curCol * MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseN() * baseHeight);
            dstOffset = dstOffset >> 1;
        } else {
            dstOffset =
                static_cast<int64_t>(curRow * MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseM() * stride) +
                static_cast<int64_t>(curCol * MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseN());
        }
        return dstOffset;
    }

    __aicore__ inline void SetFixpipeParams(FixpipeAdaptor &fixpipe)
    {
        if constexpr (PhyPosIsUB(C_TYPE::pos) &&
                      AscendC::Impl::Detail::MatmulFeatureTrait<MM_CFG>::IsSupportL0CToUB()) {
            fixpipe.SetSubBlockId(MATMUL_MODULE(MatmulSubBlockInfo)->GetSubBlockIdx());
            if constexpr (A_TYPE::ibShare && B_TYPE::ibShare) {
                fixpipe.SetMcgShfMode(McgShfMode::DUAL_DST_SPLIT_M);
            } else {
                fixpipe.SetMcgShfMode(FIXPIPE_MODE);
            }
        }
    }

    template <class T>
    __aicore__ inline void CopyTensor(const T &dst, const LocalTensor<SrcT> &co1Local, FixpipeAdaptor &fixpipe,
                                      const int32_t curN = 0, const int32_t baseUseN = 0)
    {
        if (MATMUL_MODULE(MatmulQuantProcessor)->IsQuantSenario()) {
            fixpipe.SetQuantMode(MATMUL_MODULE(MatmulQuantProcessor)->GetMatmulQuantMode());
            if (MATMUL_MODULE(MatmulQuantProcessor)->IsPerChannelSenario()) {
                LocalTensor<uint64_t> quantTensor;
                MATMUL_MODULE(MatmulQuantProcessor)->CopyQuantTensor(quantTensor, curN, baseUseN);
                fixpipe.template FixpipeOut<T>(dst, co1Local, quantTensor);
                MATMUL_MODULE(MatmulQuantProcessor)->FreeQuantTensor(quantTensor);
            } else {
                fixpipe.SetQuantScalar(MATMUL_MODULE(MatmulQuantProcessor)->GetQuantScalarValue());
                fixpipe.template FixpipeOut<T>(dst, co1Local);
            }
        } else {
            fixpipe.SetCastMode();
            fixpipe.template FixpipeOut<T>(dst, co1Local);
        }
    }

    template <bool IS_INTRA_BLOCK = false>
    __aicore__ inline uint32_t GetOrgWidth()
    {
        uint32_t dimN = GetOrgN<IS_INTRA_BLOCK>();
        if (GetOrgKc<IS_INTRA_BLOCK>() != 0) {
            dimN = GetOrgKc<IS_INTRA_BLOCK>();
        }
        constexpr uint32_t blockCount = ONE_BLK_SIZE / sizeof(DstT);
        if constexpr (C_TYPE::format == CubeFormat::ND_ALIGN) {
            dimN = Ceil(dimN, blockCount) * blockCount;
        }
        return dimN;
    }

    template <bool IS_INTRA_BLOCK = false>
    __aicore__ inline uint32_t GetOrgKc()
    {
        if constexpr ((C_TYPE::layout == LayoutMode::SBNGD) || (C_TYPE::layout == LayoutMode::BSNGD)) {
            return 0;
        } else {
            return MATMUL_MODULE(MatmulShapeInfo)->template GetOrgKc<IS_INTRA_BLOCK>();
        }
    }

    template <bool IS_INTRA_BLOCK = false>
    __aicore__ inline uint32_t GetOrgM()
    {
        if constexpr (C_TYPE::layout == LayoutMode::SBNGD) {
            return MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetCLayoutInfoB() *
                   MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetCLayoutInfoS1();
        } else if constexpr (C_TYPE::layout == LayoutMode::BSNGD) {
            return MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetCLayoutInfoS1();
        } else if constexpr (ToMatmulConfig(MM_CFG).isEnableChannelSplit && A_TYPE::format == CubeFormat::ND &&
                             C_TYPE::format == CubeFormat::NZ) {
            return Ceil(MATMUL_MODULE(MatmulShapeInfo)->template GetOrgM<IS_INTRA_BLOCK>(), BLOCK_CUBE) * BLOCK_CUBE;
        } else {
            return MATMUL_MODULE(MatmulShapeInfo)->template GetOrgM<IS_INTRA_BLOCK>();
        }
    }

    template <bool IS_INTRA_BLOCK = false>
    __aicore__ inline uint32_t GetOrgN()
    {
        if constexpr (C_TYPE::layout == LayoutMode::SBNGD) {
            return MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetCLayoutInfoG() *
                   MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetCLayoutInfoS2() *
                   MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetCLayoutInfoN() *
                   MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetCLayoutInfoB();
        } else if constexpr (C_TYPE::layout == LayoutMode::BSNGD) {
            return MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetCLayoutInfoG() *
                   MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetCLayoutInfoS2() *
                   MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetCLayoutInfoN();
        } else {
            return MATMUL_MODULE(MatmulShapeInfo)->template GetOrgN<IS_INTRA_BLOCK>();
        }
    }
};

} // namespace AscendC
#endif // QBMM_COPY_CUBE_OUT_H
