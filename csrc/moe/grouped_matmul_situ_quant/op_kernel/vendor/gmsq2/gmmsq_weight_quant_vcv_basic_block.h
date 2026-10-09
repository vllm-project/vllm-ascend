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
 * \file gmmsq_weight_quant_vcv_basic_block.h
 * \brief GMMSQ MxA8W4 VCV BasicBlock header file
 */
#ifndef GMMSQ_WEIGHT_QUANT_VCV_BASIC_BLOCK_H
#define GMMSQ_WEIGHT_QUANT_VCV_BASIC_BLOCK_H

#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#include "basic_block_config.h"
#include "basic_block_vf_mx.h"
#include "../wqbmm/weight_quant_tool.h"
#include "gmmsq_weight_quant_cube_compute.h"
#include "gmmsq_weight_quant_vec_compute.h"
#include "gmmsq_weight_quant_cube_compute_tools.h"

using AscendC::GetSubBlockIdx;
using AscendC::LocalTensor;
using AscendC::TBuf;
using AscendC::TPipe;
using AscendC::TPosition;
using namespace WeightQuantBatchMatmulV2::Arch35;

namespace GMMSQWeightQuant {

#define GMMSQ_WQ_VCV_BASIC_BLOCK_TEMPLATE_PARAM                                                                        \
    template <typename xType, typename wType, typename weightScaleType, typename xScaleType, typename yType,           \
              typename yScaleType, const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>

#define GMMSQ_WQ_VCV_BASIC_BLOCK_CLASS                                                                                 \
    GMMSQWeightQuantVcvBasicBlock<xType, wType, weightScaleType, xScaleType, yType, yScaleType, wqmmConfig, vecConfig>

GMMSQ_WQ_VCV_BASIC_BLOCK_TEMPLATE_PARAM
class GMMSQWeightQuantVcvBasicBlock {
public:
    __aicore__ inline GMMSQWeightQuantVcvBasicBlock() = default;
    __aicore__ inline void Init(uint64_t antiQuantGroupSize, __gm__ yType *y, __gm__ yScaleType *yScale,
                            float beta, float invBeta, float linearBeta, float invLinearBeta);
    __aicore__ inline void UpdateGlobalAddr(__gm__ xType *x, __gm__ wType *weight, __gm__ weightScaleType *weightScale,
                                            __gm__ xScaleType *xScale, __gm__ yType *y, __gm__ yScaleType *yScale,
                                            const bool weightL2Cacheable);
    __aicore__ inline void ComputeBasicBlock(const BasicBlockOffsetParam &curOffsetParam,
                                             const BasicBlockOffsetParam &previousOffsetParam);
    __aicore__ inline void End(const BasicBlockOffsetParam &curOffsetParam);
    // sv3_cont（M 条件化双角色拆分）：true = sv3 双角色拆分（sub0 供数专核 / sub1
    // epilogue 专核，供数单核串搬两半片，fixpipe 单投 sub1，握手走单窗口）；false =
    // 原始交错路径（两 AIV 各供一半 K 片 + 各做半量 epilogue，fixpipe dualDstCtl=1
    // 对半分投，握手走双窗口）。由控制器按组表推导的 M 阈值设定，整个 launch 恒定。
    __aicore__ inline void SetDualRoleMode(const bool dualRoleMode)
    {
        dualRoleMode_ = dualRoleMode;
    }

protected:
    __aicore__ inline void IterateNzNkWithAiv(const BasicBlockOffsetParam &curOffsetParam,
                                              const BasicBlockOffsetParam &previousOffsetParam);
    __aicore__ inline void IterateNzNkWithKAic(const BasicBlockOffsetParam &curOffsetParam,
                                               const BasicBlockOffsetParam &previousOffsetParam);
    __aicore__ inline void VecComputeNzNkWithStartLimit(uint64_t &kMte2Offset, uint64_t kMte2Limit,
                                                        const BasicBlockOffsetParam &curOffsetParam);

    template <const pipe_t pipe>
    __aicore__ inline void SetAivToAic(uint64_t syncFlag)
    {
#ifndef __CCE_KT_TEST__
        CrossCoreSetFlag<SYNC_MODE4, pipe>(syncFlag);
#endif
    };

    template <const pipe_t pipe>
    __aicore__ inline void WaitAivToAic(uint64_t syncFlag)
    {
#ifndef __CCE_KT_TEST__
        CrossCoreWaitFlag<SYNC_MODE4, pipe>(syncFlag + FLAG_ID_MAX);
        CrossCoreWaitFlag<SYNC_MODE4, pipe>(syncFlag);
#endif
    };

    template <const pipe_t pipe>
    __aicore__ inline void SetAicToAiv(uint64_t syncFlag)
    {
#ifndef __CCE_KT_TEST__
        CrossCoreSetFlag<SYNC_MODE4, pipe>(syncFlag + FLAG_ID_MAX);
        CrossCoreSetFlag<SYNC_MODE4, pipe>(syncFlag);
#endif
    };

    template <const pipe_t pipe>
    __aicore__ inline void WaitAicToAiv(uint64_t syncFlag)
    {
#ifndef __CCE_KT_TEST__
        CrossCoreWaitFlag<SYNC_MODE4, pipe>(syncFlag);
#endif
    };

    GMMSQSituVecCompute<xType, wType, yType, yScaleType, wqmmConfig, vecConfig> vecCompute_;
    GMMSQWeightQuantCubeCompute<xType, weightScaleType, xScaleType, float, wqmmConfig> cubeCompute_;

    uint64_t cvLoopIdx_ = 0;
    uint64_t weightL1DbOffset_ = 0;
    // sv3（AIV 双角色拆分）：epilogue 专核（sub 1）的 basic block 计数。供数核不再
    // 走 K 环计数以外的路径，其 cvLoopIdx_ 与 AIC 严格同拍；epilogue 核没有 K 环，
    // 用独立计数器判定「是否存在上一块输出」。仅在双角色拆分模式下使用。
    uint64_t epiLoopIdx_ = 0;
    // sv3_cont：双角色拆分 / 原始交错的模式开关（控制器按 M 阈值设定，launch 内恒定；
    // 默认 true 保持 sv3 行为）
    bool dualRoleMode_ = true;

    LocalTensor<xType> weightL1_;
    LocalTensor<float> yF32Buffer_;
};

GMMSQ_WQ_VCV_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_WQ_VCV_BASIC_BLOCK_CLASS::Init(uint64_t antiQuantGroupSize, __gm__ yType *y,
                                                            __gm__ yScaleType *yScale, float beta, float invBeta,
                                                            float linearBeta, float invLinearBeta)
{
    weightL1_ = LocalTensor<xType>(TPosition::TSCM, 0, L1_SIZE_BYTE / sizeof(xType));

    static constexpr uint64_t MXA8W4_WEIGHT_SIZE = 256 * 256;
    weightL1DbOffset_ = L1_SIZE * GetKBUnit<xType>() - MXA8W4_WEIGHT_SIZE;

    uint64_t l1RemainSize = L1_SIZE_BYTE - MXA8W4_WEIGHT_SIZE * DOUBLE_BUFFER_NUM;
    uint64_t l1StartSize = MXA8W4_WEIGHT_SIZE;

    constexpr uint64_t ubOffset = GetGmmsqSituBufferInfo<vecConfig>().weightLowbitTotalSize;
    constexpr uint64_t highBitSize = GetGmmsqSituBufferInfo<vecConfig>().weightHighBitTotalSize;
    yF32Buffer_ = LocalTensor<float>(TPosition::LCM, ubOffset, highBitSize);

    if ASCEND_IS_AIC {
        cubeCompute_.MxA8W4Init(l1RemainSize, l1StartSize);
        SetAicToAiv<PIPE_MTE1>(SYNC_AIC_AIV_FLAG);
        SetAicToAiv<PIPE_MTE1>(SYNC_AIC_AIV_FLAG);
    } else {
        vecCompute_.Init(y, yScale, beta, invBeta, linearBeta, invLinearBeta);
    }
    cvLoopIdx_ = 0;
}

GMMSQ_WQ_VCV_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_WQ_VCV_BASIC_BLOCK_CLASS::UpdateGlobalAddr(__gm__ xType *x, __gm__ wType *weight,
                                                                        __gm__ weightScaleType *weightScale,
                                                                        __gm__ xScaleType *xScale, __gm__ yType *y,
                                                                        __gm__ yScaleType *yScale,
                                                                        const bool weightL2Cacheable)
{
    if ASCEND_IS_AIC {
        cubeCompute_.UpdateGlobalAddr(x, reinterpret_cast<__gm__ float *>(y), weightScale, xScale);
    } else {
        vecCompute_.UpdateGlobalAddr(weight, weightL2Cacheable);
    }
}

GMMSQ_WQ_VCV_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void
GMMSQ_WQ_VCV_BASIC_BLOCK_CLASS::ComputeBasicBlock(const BasicBlockOffsetParam &curOffsetParam,
                                                  const BasicBlockOffsetParam &previousOffsetParam)
{
    if ASCEND_IS_AIV {
        IterateNzNkWithAiv(curOffsetParam, previousOffsetParam);
    } else {
        IterateNzNkWithKAic(curOffsetParam, previousOffsetParam);
    }
}

GMMSQ_WQ_VCV_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void
GMMSQ_WQ_VCV_BASIC_BLOCK_CLASS::IterateNzNkWithAiv(const BasicBlockOffsetParam &curOffsetParam,
                                                   const BasicBlockOffsetParam &previousOffsetParam)
{
    // sv3（AIV 双角色拆分）：同一 AIV 核内「权重供数（MTE2 读 FP4→反量化→MTE3 写 L1）」
    // 与「SiTU 激活 + MXQuant epilogue（输出侧）」两角色按 GetSubBlockIdx 拆开——
    //   sub 0 = 供数专核：本块全部 K 片连续供数，全程不碰输出路径；
    //   sub 1 = epilogue 专核：上一块输出的 SiTU/MXQuant 全量处理，全程不碰供数路径。
    // 消除原「同核串行切换角色」造成的输出 drain 与权重供数争同核流水/UB 端口的结构性
    // 串行化；每输出的累加/量化次序逐行不变，仅改「谁在哪个核上做」。
    // sv3_cont：dualRoleMode_=false（小 M）时回退原始交错路径（round_1 n2 基座逐字节
    // 恢复）：每 AIV 先供首 K 块的各自半片 → 跨块 epilogue 前 MTE2 预发射下一片 →
    // FIX/VF 握手 + 前块半量 SituEpilogue → 余下 K 片供数。两模式全程只按 launch 级
    // 恒定开关二选一，无运行中切换。
    if (dualRoleMode_) {
        if (GetSubBlockIdx() == 0) {
            uint64_t kMte2Offset = 0;
            VecComputeNzNkWithStartLimit(kMte2Offset, curOffsetParam.kSize, curOffsetParam);
        } else {
            if (epiLoopIdx_ > 0) {
                // FIX 语义与原实现一致：上一块 epilogue 的 relay 读取与输出 MTE3 均已排空
                // （SituEpilogue 尾部 MTE3_V Set/Wait），AIC 可向本核 relay 区 fixpipe 下一块
                // F32；随后等 VF（fixpipe 完成）再消费。AIC 侧配对本核窗口（flagId+FLAG_ID_MAX）。
                SetAivToAic<PIPE_MTE3>(SYNC_AIV_MTE3_AIC_FIX_FLAG);
                WaitAicToAiv<PIPE_V>(SYNC_AIC_FIX_AIV_VF_FLAG);

                vecCompute_.SituEpilogue(yF32Buffer_, previousOffsetParam, true);
            }
            epiLoopIdx_++;
        }
    } else {
        uint64_t kMte2Offset = 0;
        uint64_t curCvLoopIdx = cvLoopIdx_;

        VecComputeNzNkWithStartLimit(kMte2Offset, Min(curOffsetParam.kSize, DOUBLE_BUFFER_NUM * curOffsetParam.kbL1Size),
                                     curOffsetParam);

        if (curCvLoopIdx > 0) {
            // n2（路线2·跨块 epilogue 插入点）：进入旧块 SiTU epilogue 前，仅把下一片 FP4
            // 「发起」到独立低位环槽（不等待完成、不做 FP8 展宽——展宽会写高位区，可能覆盖
            // F32 relay）。旧块 epilogue 的 Vector 工作与输出 MTE3 期间该片 MTE2 在途，
            // 消除原节拍「epilogue 期间供数完全停摆」的空泡；epilogue 结束后由
            // VecComputeNzNkWithStartLimit 首次迭代消费该片。本插入点只覆盖跨块 epilogue，
            // 常规 K 环的 issue/consume 配对保持原样。
            if (kMte2Offset < curOffsetParam.kSize) {
                vecCompute_.IssueNextSliceCopy(kMte2Offset, curOffsetParam.kSize, curOffsetParam);
            }

            SetAivToAic<PIPE_MTE3>(SYNC_AIV_MTE3_AIC_FIX_FLAG);
            WaitAicToAiv<PIPE_V>(SYNC_AIC_FIX_AIV_VF_FLAG);

            vecCompute_.SituEpilogue(yF32Buffer_, previousOffsetParam, false);
        }

        VecComputeNzNkWithStartLimit(kMte2Offset, curOffsetParam.kSize, curOffsetParam);
    }
}

GMMSQ_WQ_VCV_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void
GMMSQ_WQ_VCV_BASIC_BLOCK_CLASS::VecComputeNzNkWithStartLimit(uint64_t &kMte2Offset, uint64_t kMte2Limit,
                                                             const BasicBlockOffsetParam &curOffsetParam)
{
    // sv3：供数专核（sub 0）独跑本函数。原实现按 GetSubBlockIdx 把每个 K 块（kbL1Size 行）
    // 对半分给两颗 AIV；现改由本核按 [0,kbL1/2)、[kbL1/2,kbL1) 两半片顺序供数：
    //   - UB 低位环槽尺寸（16KiB）、FP8 展宽、L1 写地址（同一 parity buffer 内
    //     nL1AlignSize*kL1Offset 偏移）与拷贝公式逐字节保持原样；
    //   - L1-free（SYNC_AIC_AIV_FLAG）每 K 块等一次（原每核每块各等一次、共同对应同一个
    //     AIC「整 buffer 排空」事件，语义不变）；
    //   - ready（SYNC_AIV_AIC_FLAG）每 K 块发一次（两半片写完后），AIC 侧配对改为单
    //     本核窗口等待——AIC 收到时两半片必然全部就位，与原「双核各发一次、AIC 双窗口
    //     各收一次后再 LaunchMatmul」的屏障语义等价。
    // sv3_cont：dualRoleMode_=false 时为原始实现（round_1 n2 基座逐字节恢复）：每 AIV
    // 只供本核半片（kL1Offset = GetSubBlockIdx()*kbL1/2），含 n2 预发射消费配对。
    uint64_t kMte2BaseSize = curOffsetParam.kbL1Size / DOUBLE_BUFFER_NUM;
    uint64_t nL1AlignSize = CeilAlign(curOffsetParam.nL1Size, static_cast<uint64_t>(BLOCK_CUBE));

    if (dualRoleMode_) {
        for (; kMte2Offset < kMte2Limit; kMte2Offset += curOffsetParam.kbL1Size, cvLoopIdx_++) {
            // 半片 0：kL1Offset = 0（原 sub 0 半区）
            uint64_t mte2RealK0 = kMte2Offset >= kMte2Limit ? 0 :
                                  kMte2Offset + kMte2BaseSize >= kMte2Limit ? kMte2Limit - kMte2Offset :
                                                                              kMte2BaseSize;
            vecCompute_.WaitVToMTE2();
            vecCompute_.CopyGmToUb(mte2RealK0, kMte2Offset, 0, curOffsetParam);

            WaitAicToAiv<PIPE_MTE3>(SYNC_AIC_AIV_FLAG);

            vecCompute_.WeightAntiQuantComputeNzNk(
                mte2RealK0, kMte2Offset, weightL1_[(cvLoopIdx_ & 1) * weightL1DbOffset_], curOffsetParam);
            vecCompute_.SetVToMTE2();

            // 半片 1：kL1Offset = kbL1/2（原 sub 1 半区），同一 L1 buffer，free 已由上一次等待覆盖
            uint64_t kL1Offset1 = kMte2BaseSize;
            uint64_t mte2RealK1 = kMte2Offset + kL1Offset1 >= kMte2Limit ? 0 :
                                  kMte2Offset + kL1Offset1 + kMte2BaseSize >= kMte2Limit ?
                                                                         kMte2Limit - kMte2Offset - kL1Offset1 :
                                                                         kMte2BaseSize;
            vecCompute_.WaitVToMTE2();
            vecCompute_.CopyGmToUb(mte2RealK1, kMte2Offset, kL1Offset1, curOffsetParam);

            vecCompute_.WeightAntiQuantComputeNzNk(
                mte2RealK1, kMte2Offset, weightL1_[(cvLoopIdx_ & 1) * weightL1DbOffset_ + nL1AlignSize * kL1Offset1],
                curOffsetParam);
            vecCompute_.SetVToMTE2();

            SetAivToAic<PIPE_MTE3>(SYNC_AIV_AIC_FLAG);
        }
    } else {
        for (; kMte2Offset < kMte2Limit; kMte2Offset += curOffsetParam.kbL1Size, cvLoopIdx_++) {
            uint64_t kL1Offset = GetSubBlockIdx() * kMte2BaseSize;
            uint64_t mte2RealK = kMte2Offset + kL1Offset >= kMte2Limit ? 0 :
                                 kMte2Offset + kL1Offset + kMte2BaseSize >= kMte2Limit ?
                                                                         kMte2Limit - kMte2Offset - kL1Offset :
                                                                         kMte2BaseSize;
            if (vecCompute_.HasPreIssuedSlice()) {
                // n2：该片已在 epilogue 前发起（槽位 WaitVToMTE2 与拷贝均已执行），此处仅等待
                // MTE2 排空，不重复发起，保证搬运字节量与原实现一致
                vecCompute_.ConsumePreIssuedSlice();
            } else {
                vecCompute_.WaitVToMTE2();
                vecCompute_.CopyGmToUb(mte2RealK, kMte2Offset, kL1Offset, curOffsetParam);
            }

            WaitAicToAiv<PIPE_MTE3>(SYNC_AIC_AIV_FLAG);

            vecCompute_.WeightAntiQuantComputeNzNk(
                mte2RealK, kMte2Offset, weightL1_[(cvLoopIdx_ & 1) * weightL1DbOffset_ + nL1AlignSize * kL1Offset],
                curOffsetParam);
            SetAivToAic<PIPE_MTE3>(SYNC_AIV_AIC_FLAG);
            vecCompute_.SetVToMTE2();
        }
    }
}

GMMSQ_WQ_VCV_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void
GMMSQ_WQ_VCV_BASIC_BLOCK_CLASS::IterateNzNkWithKAic(const BasicBlockOffsetParam &curOffsetParam,
                                                    const BasicBlockOffsetParam &previousOffsetParam)
{
    if (cvLoopIdx_ > 0) {
        // sv3：FIX 只等 epilogue 专核（sub 1）窗口 = flagId + FLAG_ID_MAX（intra-block
        // 硬件 flag = flagId + subBlockId*16，AIC 原值访问即两 AIV 窗口）。供数专核不再
        // 参与 FIX/VF 握手，双窗口等待会死等其空窗口。
        // sv3_cont：原始模式两 AIV 各在自己窗口发 FIX（sub0→f，sub1→f+16），AIC 双窗口
        // 等待（f+16 先、f 后，与原实现次序一致）。
        if (dualRoleMode_) {
            CrossCoreWaitFlag<SYNC_MODE4, PIPE_FIX>(SYNC_AIV_MTE3_AIC_FIX_FLAG + FLAG_ID_MAX);
        } else {
            WaitAivToAic<PIPE_FIX>(SYNC_AIV_MTE3_AIC_FIX_FLAG);
        }
        cubeCompute_.GetTensorC(yF32Buffer_, previousOffsetParam, dualRoleMode_);
        SetAicToAiv<PIPE_FIX>(SYNC_AIC_FIX_AIV_VF_FLAG);
    }

    cubeCompute_.SetWeightMTE1ToMTE2(curOffsetParam);
    uint64_t kMutilLoadL1Size = MX_SCALE_K_L1_SIZE;
    for (uint64_t kbL1Offset = 0; kbL1Offset < curOffsetParam.kSize;
         kbL1Offset += curOffsetParam.kbL1Size, cvLoopIdx_++) {
        uint64_t kbL1RealSize = (kbL1Offset + curOffsetParam.kbL1Size) >= curOffsetParam.kSize ?
                                    curOffsetParam.kSize - kbL1Offset :
                                    curOffsetParam.kbL1Size;
        if (kbL1Offset % MX_SCALE_K_L1_SIZE == 0) {
            kMutilLoadL1Size = (kbL1Offset + MX_SCALE_K_L1_SIZE) > curOffsetParam.kSize ?
                                   CeilAlign(curOffsetParam.kSize - kbL1Offset, K_ALIGNMENT64) :
                                   MX_SCALE_K_L1_SIZE;
        }
        cubeCompute_.WaitScaleMTE1ToMTE2(kbL1Offset);
        cubeCompute_.CopyMxScaleGmToL1(curOffsetParam, kbL1Offset);
        cubeCompute_.WaitMTE1ToMTE2(kbL1Offset, curOffsetParam);
        cubeCompute_.CopyAGmToL1(curOffsetParam, kbL1Offset);
        cubeCompute_.WaitMTE1ToMTE2(kbL1Offset, curOffsetParam, kbL1RealSize);
        cubeCompute_.PadL1IfNotAlign(weightL1_, weightL1DbOffset_, cvLoopIdx_, kbL1RealSize, curOffsetParam.nL1Size);
        cubeCompute_.SetMTE1ToMTE2(kbL1Offset, curOffsetParam, kbL1RealSize);
        // sv3：ready 只等供数专核（sub 0）窗口 = flagId 原值。供数核每 K 块在两半片
        // 全部写入 L1 后发一次，LaunchMatmul 屏障语义与原双窗口版本等价。
        // sv3_cont：原始模式两 AIV 各自半片写完后各发一次（sub0→f，sub1→f+16），AIC
        // 双窗口等待（f+16 先、f 后，与原实现次序一致）。
        if (dualRoleMode_) {
            CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIV_AIC_FLAG);
        } else {
            WaitAivToAic<PIPE_MTE1>(SYNC_AIV_AIC_FLAG);
        }
        cubeCompute_.LaunchMatmul(weightL1_[(cvLoopIdx_ & 1) * weightL1DbOffset_], kbL1Offset, kbL1RealSize,
                                  kMutilLoadL1Size, curOffsetParam);
        cubeCompute_.SetMTE1ToMTE2(kbL1Offset, curOffsetParam);
        cubeCompute_.SetScaleMTE1ToMTE2(kbL1Offset, curOffsetParam);
        SetAicToAiv<PIPE_MTE1>(SYNC_AIC_AIV_FLAG);
    }
}

GMMSQ_WQ_VCV_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_WQ_VCV_BASIC_BLOCK_CLASS::End(const BasicBlockOffsetParam &previousOffsetParam)
{
    if ASCEND_IS_AIC {
        if (cvLoopIdx_ > 0) {
            // sv3：FIX 只等 epilogue 专核（sub 1）窗口，理由同 IterateNzNkWithKAic
            if (dualRoleMode_) {
                CrossCoreWaitFlag<SYNC_MODE4, PIPE_FIX>(SYNC_AIV_MTE3_AIC_FIX_FLAG + FLAG_ID_MAX);
            } else {
                WaitAivToAic<PIPE_FIX>(SYNC_AIV_MTE3_AIC_FIX_FLAG);
            }
            cubeCompute_.GetTensorC(yF32Buffer_, previousOffsetParam, dualRoleMode_);
            SetAicToAiv<PIPE_FIX>(SYNC_AIC_FIX_AIV_VF_FLAG);
        }
        cubeCompute_.EndSync();
    } else {
        if (dualRoleMode_) {
            if (GetSubBlockIdx() != 0) {
                // epilogue 专核：末块输出处理；不参与 L1-free 排空（供数路径不归本核）
                if (epiLoopIdx_ > 0) {
                    SetAivToAic<PIPE_MTE3>(SYNC_AIV_MTE3_AIC_FIX_FLAG);
                    WaitAicToAiv<PIPE_V>(SYNC_AIC_FIX_AIV_VF_FLAG);

                    vecCompute_.SituEpilogue(yF32Buffer_, previousOffsetParam, true);
                }
            } else {
                // 供数专核：排空本核窗口的 L1-free 旗标（Init ×2 + 每 K 块 ×1 的收支平衡，
                // 与原实现逐条一致）；不参与 FIX/VF 握手
                WaitAicToAiv<PIPE_MTE3>(SYNC_AIC_AIV_FLAG);
                WaitAicToAiv<PIPE_MTE3>(SYNC_AIC_AIV_FLAG);
            }
        } else {
            // 原始模式（round_1 n2 基座逐字节恢复）：两 AIV 各自末块半量 epilogue +
            // 各自排空本核窗口的 L1-free 旗标（Init ×2 + 每 K 块 ×1，每窗口收支平衡）
            if (cvLoopIdx_ > 0) {
                SetAivToAic<PIPE_MTE3>(SYNC_AIV_MTE3_AIC_FIX_FLAG);
                WaitAicToAiv<PIPE_V>(SYNC_AIC_FIX_AIV_VF_FLAG);

                vecCompute_.SituEpilogue(yF32Buffer_, previousOffsetParam, false);
            }
            WaitAicToAiv<PIPE_MTE3>(SYNC_AIC_AIV_FLAG);
            WaitAicToAiv<PIPE_MTE3>(SYNC_AIC_AIV_FLAG);
        }
        vecCompute_.End();
    }
}

} // namespace GMMSQWeightQuant

#endif // GMMSQ_WEIGHT_QUANT_VCV_BASIC_BLOCK_H
