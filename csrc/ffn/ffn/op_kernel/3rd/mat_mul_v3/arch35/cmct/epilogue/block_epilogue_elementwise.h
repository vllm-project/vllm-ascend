/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file block_epilogue_elementwise.h
 * \brief
 */

#pragma once
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "../utils/common_utils.h"
#include "../utils/device_utils.h"
#include "fusion/default_fusion_op.h"
#include "fusion/fusion_add.h"
#include "fusion/fusion_mul.h"
#include "fusion/fusion_gelu.h"
#include "../utils/status_utils.h"

namespace Cmct {
namespace Gemm {
namespace Block {

// FusionOp 是否把 N 压缩为一半（swiglu single：每 32 列 -> 16 列）
template <typename T, typename = void>
struct FusionHalfWidthDetect : std::false_type {};

template <typename T>
struct FusionHalfWidthDetect<T, std::void_t<decltype(T::HALF_WIDTH_OUT)>>
    : std::bool_constant<T::HALF_WIDTH_OUT> {};

template <typename L0TileShape_, typename DataTypeOut_, typename DataTypeIn_, typename FusionOp_>
class BlockEpilogueElementwise {
public:
    using FusionArguments = typename FusionOp_::Arguments;
    using FusionParams = typename FusionOp_::Params;

    __aicore__ inline BlockEpilogueElementwise() {}

    struct Arguments {
        GM_ADDR outGmAddr{nullptr};
        FusionArguments fusionArgs{};
    };

    struct Params {
        GM_ADDR outGmAddr{nullptr};
        FusionParams fusionParams{};
    };

    using DataTypeOut = DataTypeOut_;
    using DataTypeIn = DataTypeIn_;
    using FusionOp = FusionOp_;
    static constexpr bool kHalfWidth = FusionHalfWidthDetect<FusionOp>::value;
    static constexpr uint16_t ZERO_FLAG = 0;
    static constexpr int64_t l0M = GetIntegralConstant<MNK_M, L0TileShape_>();
    static constexpr int64_t l0N = GetIntegralConstant<MNK_N, L0TileShape_>();
    // shape
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;

    // GM ADDR
    AscendC::LocalTensor<DataTypeIn> cLocal_{AscendC::TPosition::VECIN, 0, AscendC::TOTAL_UB_SIZE / sizeof(DataTypeIn)};
    AscendC::LocalTensor<DataTypeIn> cLocalTmp_{AscendC::TPosition::VECIN, 0,
                                                AscendC::TOTAL_UB_SIZE / sizeof(DataTypeIn)};
    AscendC::GlobalTensor<DataTypeOut> outputGlobal_;
    // vector核一次最多计算多少个元素
    int64_t stageSize_ = 0;
    // attribute
    FusionOp fusionOp_;
    ProblemShape problemShape_;

    __aicore__ inline void Init(Params const& params, int64_t l1M, int64_t l1N, ProblemShape& problemShape)
    {
        outputGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ DataTypeOut*>(params.outGmAddr));
        problemShape_ = problemShape;
        int64_t l1NAlign = AlignBlock<DataTypeOut>(l1N);
        int64_t ubOffset = l1M * l1NAlign;
        fusionOp_.Init(params.fusionParams, cLocal_, l1M, l1NAlign, ubOffset, stageSize_);
        cLocalTmp_ = cLocal_[ubOffset].template ReinterpretCast<DataTypeIn>();
    }

    __aicore__ inline AscendC::LocalTensor<DataTypeOut> DoFusionAndCast(int64_t stageOffset, int64_t offset,
                                                                        int64_t blockShapeM, int64_t blockShapeN,
                                                                        int64_t N, int64_t stageSize)
    {
        // Do fusionOp{add, mul, gelu} in ub:  (cLocal_[stageOffset], x3 or None) -> cLocal_
        if constexpr (FusionOp::kRegBaseAct && !AscendC::IsSameType<DataTypeOut, DataTypeIn>::value) {
            // RegBase 激活：VF 寄存器内完成激活 + Cast RINT，窄类型就地写回 cLocal_[stageOffset]
            // （半宽覆盖、读先于写；无 UB 暂存往返与逐指令 PipeBarrier）。outputLocal 形参传
            // 原位 fp32 视图，fusion 内部 ReinterpretCast 为输出类型。
            AscendC::LocalTensor<DataTypeIn> stageLocal = cLocal_[stageOffset];
            fusionOp_(stageLocal, stageLocal, offset, blockShapeM, blockShapeN, N, stageSize);
            return cLocal_[stageOffset].template ReinterpretCast<DataTypeOut>();
        }
        fusionOp_(cLocal_[stageOffset], cLocalTmp_, offset, blockShapeM, blockShapeN, N, stageSize);
        AscendC::LocalTensor<DataTypeOut> outputLocal = cLocal_[stageOffset].template ReinterpretCast<DataTypeOut>();
        if constexpr (AscendC::IsSameType<DataTypeOut, DataTypeIn>::value) {
            outputLocal = cLocalTmp_;
        } else {
            int64_t castSize = kHalfWidth ? stageSize / 2 : stageSize;
            Cast(outputLocal, cLocalTmp_, AscendC::RoundMode::CAST_RINT, castSize);
            AscendC::PipeBarrier<PIPE_V>();
        }
        return outputLocal;
    }

    __aicore__ inline void Run(BlockShape const& blockShape, int64_t dstOffset, int64_t flagId = 5)
    {
        // 默认1-2不再基于splitM区分, aiv 0~1分别搬运blockShapeM/2
        int64_t blockShapeM = Get<0>(blockShape);
        int64_t halfBlockShapeM = Cmct::Gemm::CeilDiv(blockShapeM, AscendC::GetTaskRation());
        blockShapeM = ((static_cast<uint64_t>(blockShapeM) & 1UL) > 0UL) ?
                          (halfBlockShapeM - AscendC::GetSubBlockIdx()) :
                          halfBlockShapeM;
        if (blockShapeM <= 0) {
            // M 为奇数的 tail tile：subBlockIdx=1 分到 0 行，直接通知 AIC 并返回
            AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(flagId);
            return;
        }
        int64_t blockShapeN = Get<1>(blockShape);
        int64_t N = Get<MNK_N>(problemShape_);
        // swiglu single：mmad N=2H，输出 N=H
        int64_t outN = kHalfWidth ? N / 2 : N;
        int64_t outWidth = kHalfWidth ? blockShapeN / 2 : blockShapeN;
        int64_t blockShapeNAlign = AlignBlock<DataTypeOut>(blockShapeN); // 对齐16
        int64_t inputSize = blockShapeM * blockShapeNAlign;

        // 一次计算最多取Min(baseM/2 * baseN, stageSize_)
        int64_t stageSize = AscendC::Std::min(stageSize_, inputSize) / blockShapeNAlign * blockShapeNAlign;
        ASCENDC_ASSERT(stageSize > 0, {
            KERNEL_LOG(KERNEL_EORROR, "stageSize size limit %ld, %ld, %ld!", stageSize_, blockShapeM, blockShapeN);
        });
        int64_t loop = 0;
        int64_t stageOffset = 0;
        while (stageOffset < inputSize) {
            int64_t offset = dstOffset + loop * stageSize / blockShapeNAlign * outN;
            // Aiv1需要多偏移aiv0所处理的数据
            offset += AscendC::GetSubBlockIdx() * halfBlockShapeM * outN;
            stageSize = AscendC::Std::min(stageSize, inputSize - stageOffset);
            AscendC::LocalTensor<DataTypeOut> outputLocal = DoFusionAndCast(stageOffset, offset, blockShapeM,
                                                                            blockShapeN, N, stageSize);

            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ZERO_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ZERO_FLAG);
            // copy result from ub to gm
            // DataCopyPad 的 UB 源端 stride 以 32B 块为单位且 blockLen 自动 32B
            // 对齐：blockLen=outWidth*sizeof 未对齐时（如 136×2B=272B→288B），
            // 行间天然按对齐后的宽度（144×2B）连续排布，srcStride=0 即正确。
            // swiglu single 非 32 对齐尾块：mmad/UB 帧宽已补齐（Align(原始,32)），
            // 计算宽度=补齐半宽，但只写有效半宽（outN - 列偏移）。
            int64_t writeWidth = outWidth;
            if (kHalfWidth) {
                int64_t valid = outN - (offset % outN);
                if (valid < writeWidth) {
                    writeWidth = valid;
                }
            }
            AscendC::DataCopyExtParams copyParams{static_cast<uint16_t>(stageSize / blockShapeNAlign),
                                                  static_cast<uint32_t>(writeWidth * sizeof(DataTypeOut)),
                                                  0,
                                                  static_cast<uint32_t>((outN - writeWidth) * sizeof(DataTypeOut)), 0};
            AscendC::DataCopyPad<DataTypeOut>(outputGlobal_[offset], outputLocal, copyParams);
            stageOffset += stageSize;
            loop++;
        }
        AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(flagId);
    }

    // GetTensor from ub from current AIV
    __aicore__ inline auto GetTensor() { return cLocal_; }

    __aicore__ inline void operator()(BlockShape const& blockShape, int64_t dstOffset = 0, int64_t flagId = 5)
    {
        Run(blockShape, dstOffset, flagId);
        return;
    }

    // static init
    __host_aicore__ static Params InitParams(Arguments const& args, GM_ADDR x3Gm)
    {
        FusionParams fusionParams = FusionOp::InitParams(args.fusionArgs, x3Gm);
        Params params = {args.outGmAddr, fusionParams};
        return params;
    }

    __host_aicore__ static size_t GetWorkspaceSize(int64_t blockNum, int64_t l1M, int64_t l1N)
    {
        // only quant kernel need workspace
        return 0;
    }

    __host_aicore__ static Status CanImplement(Arguments const& args)
    {
        if (l0M * l0N * sizeof(DataTypeIn_) > AscendC::TOTAL_UB_SIZE) {
            return Status::l1L0ErrorExceedsLimit;
        }
        return Status::success;
    }
};

} // namespace Block
} // namespace Gemm
} // namespace Cmct
#endif // EPILOGUE_BLOCK_EPILOGUE_H
