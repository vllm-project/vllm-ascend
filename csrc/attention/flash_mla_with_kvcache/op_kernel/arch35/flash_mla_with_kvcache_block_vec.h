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
 * \file flash_mla_with_kvcache_block_vec.h
 * \brief arch35 flash_mla_with_kvcache MLA vector；
 *        静态 tensor UbLayout + Mutex/cross-core 显式同步 + FA 协同清零
 *        （FD 业务 UB 从 120 KiB 起，避开 Cube/Vector 共用的 MM1/MM2 区）
 */
#ifndef FLASH_MLA_WITH_KVCACHE_BLOCK_VEC_H_
#define FLASH_MLA_WITH_KVCACHE_BLOCK_VEC_H_

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include <limits>

#include "../utils/attenmask_gs1.h"
#include "../../../a5_mla_common/op_kernel/arch35/flash_attention_score_common_regbase_arch35.h"
#include "adv_api/activation/softmax.h"
#include "../../../a5_mla_common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz.h"
#include "../../../a5_mla_common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz_dn.h"
#include "../../../a5_mla_common/op_kernel/arch35/vf/vf_flashupdate_new.h"
#include "../../../a5_mla_common/op_kernel/arch35/vf/vf_div_cast_arch35.h"
#include "../../../a5_mla_common/op_kernel/arch35/vf/vf_flash_decode_arch35.h"
#include "../../../a5_mla_common/op_kernel/init_output.h"
#include "../../../a5_mla_common/op_kernel/const_def.h"
#include "flash_mla_with_kvcache_public_define_arch35.h"
#include "../../../a5_mla_common/op_kernel/vector_common.h"
#include "memory_copy_arch35.h"
#include "../utils/flash_mla_with_kvcache_type.h"

using namespace AscendC;
using namespace FaVectorApi;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;
using namespace AttentionCommon;

namespace FlashAttnKernel {
template <bool first, bool last>
__simd_vf__ inline void MlaStreamUpdate(__ubuf__ float *dst, __ubuf__ float *cur, __ubuf__ float *exp,
                                        __ubuf__ float *sum, uint16_t rows)
{
    using namespace AscendC;
    using namespace AscendC::Reg;
    MaskReg mask = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> e0, s0, x00, x01, x02, x03, o00, o01, o02, o03;
    for (uint16_t i = 0; i < rows; ++i) {
        if constexpr (!first) {
            LoadAlign<float, LoadDist::DIST_BRC_B32>(e0, exp + i);
        }
        if constexpr (last) {
            LoadAlign<float, LoadDist::DIST_BRC_B32>(s0, sum + i);
        }
        LoadAlign(x00, cur + (i) * 256U + 0U);
        if constexpr (!first) {
            LoadAlign(o00, dst + (i) * 512U + 0U);
        }
        LoadAlign(x01, cur + (i) * 256U + 64U);
        if constexpr (!first) {
            LoadAlign(o01, dst + (i) * 512U + 64U);
        }
        LoadAlign(x02, cur + (i) * 256U + 128U);
        if constexpr (!first) {
            LoadAlign(o02, dst + (i) * 512U + 128U);
        }
        LoadAlign(x03, cur + (i) * 256U + 192U);
        if constexpr (!first) {
            LoadAlign(o03, dst + (i) * 512U + 192U);
        }
        if constexpr (!first) {
            Mul(o00, o00, e0, mask);
        }
        if constexpr (!first) {
            Mul(o01, o01, e0, mask);
        }
        if constexpr (!first) {
            Mul(o02, o02, e0, mask);
        }
        if constexpr (!first) {
            Mul(o03, o03, e0, mask);
        }
        if constexpr (!first) {
            Add(x00, o00, x00, mask);
        }
        if constexpr (!first) {
            Add(x01, o01, x01, mask);
        }
        if constexpr (!first) {
            Add(x02, o02, x02, mask);
        }
        if constexpr (!first) {
            Add(x03, o03, x03, mask);
        }
        if constexpr (last) {
            Div(x00, x00, s0, mask);
        }
        if constexpr (last) {
            Div(x01, x01, s0, mask);
        }
        if constexpr (last) {
            Div(x02, x02, s0, mask);
        }
        if constexpr (last) {
            Div(x03, x03, s0, mask);
        }
        StoreAlign<float, StoreDist::DIST_NORM_B32>(dst + (i) * 512U + 0U, x00, mask);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(dst + (i) * 512U + 64U, x01, mask);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(dst + (i) * 512U + 128U, x02, mask);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(dst + (i) * 512U + 192U, x03, mask);
    }
}

template <typename FA_T>
class FlashMlaWithKvcacheNoQuantMlaBlockVec {
public:
    using INPUT_T = typename FA_T::inputType;
    using T = typename FA_T::mmType;
    using OUTPUT_T = typename FA_T::outputType;
    static constexpr FLASH_MLA_WITH_KVCACHE_LAYOUT LAYOUT_T = FA_T::qLayout;
    static constexpr FLASH_MLA_WITH_KVCACHE_LAYOUT outLayout = FA_T::attnOutLayout;
    static constexpr bool hasAtten = FA_T::hasMask;
    /* =================编译期常量的基本块信息================= */
    static constexpr uint32_t mBaseSize = (uint32_t)FA_T::mBaseSize;
    static constexpr uint32_t s2BaseSize = (uint32_t)FA_T::s2BaseSize;
    static constexpr uint32_t dVBaseSize = (uint32_t)FA_T::dVBaseSize;
    static constexpr uint32_t vec1HalfS1BaseSize = mBaseSize >> 1;
    static constexpr uint32_t vec1Srcstride = (mBaseSize >> 1) + 1; // 解bank冲突，需要加1行
    static constexpr uint32_t dVTemplateAlign64 = BaseApi::Align64Func((uint16_t)FA_T::dVBaseSize);

    static constexpr uint32_t DB = 2;
    static constexpr uint32_t PRELOAD_N = 2; // C1 C1 C2 C2
    static constexpr bool HAS_MASK = hasAtten;
    static constexpr bool FLASH_DECODE = FA_T::flashDecode;
    static constexpr bool HAS_DROP = false;                             // 不支持drop mask
    static constexpr PseTypeEnum PSE_MODE = PseTypeEnum::PSE_NONE_TYPE; // 不支持PSE

    static constexpr ActualSeqLensMode Q_MODE = GetQActSeqMode<LAYOUT_T>();
    static constexpr ActualSeqLensMode KV_MODE = GetKvActSeqMode<LAYOUT_T, (FA_T::kvLayoutType > 0)>();
    using SeqLensToolType = FlashMlaSeqLensTool<Q_MODE, KV_MODE>;
    static constexpr MaskFormat MASK_LAYOUT =
        (LAYOUT_T == FLASH_MLA_WITH_KVCACHE_LAYOUT::BSH || LAYOUT_T == FLASH_MLA_WITH_KVCACHE_LAYOUT::TND) ?
            MaskFormat::SG :
            MaskFormat::GS;

    using pseShiftType = INPUT_T;

    static constexpr T BOOL_ATTEN_MASK_SCALAR_VALUE = -1000000000000.0; // 用于mask为bool类型
    uint32_t negativeIntScalar = *((uint32_t *)&BOOL_ATTEN_MASK_SCALAR_VALUE);

    using attenMaskGmType = typename std::conditional<hasAtten, GlobalTensor<uint8_t>, int8_t>::type;
    using flashdecodeGmType = typename std::conditional<FLASH_DECODE, GlobalTensor<float>, int8_t>::type;
    using ConstInfoNoQuant = ConstInfo_t<FlashMlaWithKvcacheKernelType::NO_QUANT>;
    using OUT_T = OUTPUT_T;

    // ================== 核间同步 ID（与 cube 侧逐字节一致）==================
    static constexpr uint64_t CROSS_CORE_SYNC_MODE = 4U;
    static constexpr uint32_t CROSSCORE_BMM1_0 = 0U;
    static constexpr uint32_t CROSSCORE_BMM2_0 = 2U;
    static constexpr uint32_t CROSSCORE_BMM2_1 = 3U;
    static constexpr uint32_t CROSSCORE_L1P_0 = 5U;
    static constexpr uint32_t CROSSCORE_L1P_1 = 6U;
    static constexpr uint32_t CROSSCORE_L1P_2 = 7U;
    // AIV0_AIV1_OFFSET 使用全局宏（attention/a5_mla_common/op_kernel/buffer.h:30，=16）

    // ================== AIV 核内 Mutex ID：0–5 用于输出，6/7 用于 mask 槽 ==================
    static constexpr uint32_t UB_OUT_VEC2_RES_EVENT0 = 0U; // vec2Res（96K@120K）V/MTE3
    static constexpr uint32_t EVENT_ID0 = 1U;              // InitOutput atten-out pop 同步（PIPE_V/PIPE_MTE3）
    static constexpr uint32_t UB_OUT_VEC1_RES_EVENT0 = 2U; // stage1/vec1Res（12.25K@216K）V/MTE3
    static constexpr uint32_t EVENT_ID1 = 3U;              // InitOutput LSE pop 同步（PIPE_V/PIPE_MTE3）
    static constexpr uint32_t UB_OUT_LSE_OUT_EVENT0 = 4U;  // LSE 1.5 KiB 输出（V/MTE3）
    static constexpr uint32_t UB_BRDCST_SUM_EVENT = 5U;    // sumBrdcst 2 KiB（V/MTE3）
    // mask 槽 2×6 KiB 的 per-slot 双 pipe 生产-消费握手：
    // MTE2(加载) ↔ V(软max消费) 严格交替，避免跨 S2 轮次槽复用读到上一轮掩码（旧深度1队列的进出队语义）
    static constexpr uint32_t UB_IN_MASK_EVENT0 = 6U;
    static constexpr uint32_t UB_IN_MASK_EVENT1 = 7U;
    static constexpr uint32_t UB_BRDCST_MAX_EVENT = UB_BRDCST_SUM_EVENT; // maxBrdcst 2K（V/MTE3）

    // ================== 静态布局常量（M96/CV_RATIO=2；UB 合计 246.5 KiB ≤ 248 KiB）==================
    static constexpr uint32_t BUFFER_SIZE_BYTE_32K = 32768;
    static constexpr uint32_t UB_MM2_RES_BUFCNT = 2U;
    static constexpr uint32_t UB_MM2_RES_BUF_BYTES = mBaseSize / CV_RATIO * 256U * sizeof(T);
    static constexpr uint32_t UB_MM1_RES_BUFCNT = 1U;
    static constexpr uint32_t UB_MM1_RES_BUF_BYTES = mBaseSize / CV_RATIO * 128U * sizeof(T);
    static constexpr uint32_t UB_VEC2_RES_BUF_BYTES = mBaseSize / CV_RATIO * dVTemplateAlign64 * sizeof(T); // 96K
    static constexpr uint32_t UB_VEC1_RES_BUF_BYTES = (mBaseSize / CV_RATIO + 1) * 128U * sizeof(INPUT_T);  // 12.25K
    static constexpr uint32_t UB_MASK_BUFCNT = DB;
    static constexpr uint32_t UB_MASK_BUF_BYTES = 6144U; // attenMaskInQue[2] 各 6144B
    static constexpr uint32_t UB_SOFTMAX_BUFCNT = PRELOAD_N + 1;
    static constexpr uint32_t UB_SOFTMAX_BUF_BYTES = 256U; // SOFTMAX VF max/sum/exp 按 256B 对齐
    static constexpr uint32_t UB_LSE_OUT_BUF_BYTES = mBaseSize / CV_RATIO * sizeof(float) * 8; // 1.5K（48×4×8）
    static constexpr uint32_t UB_BRDCST_BUF_BYTES = 2048U;                                     // max/sumBrdcst
    static constexpr uint32_t UB_TMP_BUF_BYTES = 512U;                                         // commonTBuf 512B

    // L1 P（A1）：vec1 写 P 的目标 = cube 的 L1 KVP 区（A1 108K 起、3×133K、每个任务 loop%3 槽）；
    // P 在槽内偏移为 s2BaseSize*dVBaseSize 个元素，即 112*512*2 = 112 KiB；
    // P 的 96*112*2 = 21 KiB 覆盖 rope 的 14 KiB 与预留的 7 KiB。
    static constexpr uint32_t L1_P_BUFCNT = 3U;
    static constexpr uint32_t L1_P_BUF_BYTES = s2BaseSize * 608 * sizeof(INPUT_T);   // 133K
    static constexpr uint32_t L1_Q_PREFIX_BYTES = mBaseSize * 576 * sizeof(INPUT_T); // 108K（cube Q 区，跳过）

    // gm
    GlobalTensor<OUTPUT_T> attentionOutGm;
    GlobalTensor<float> softmaxLseGm;
    // seq-lens INT32；q 侧 ACCUM 带首零头（cu_seqlens_q [b+1]），ACTLEN_T=uint32_t；
    // parser 所有权在 kernel 侧 FlashMlaSeqLensTool，本 block 只读引用
    SeqLensToolType &seqLensTool_;

    attenMaskGmType attenMaskGmInt;

    flashdecodeGmType accumOutGm;
    flashdecodeGmType softmaxFDSumGm;
    flashdecodeGmType softmaxFDMaxGm;

    // 输出 GM tensor + 拷贝器（flash_attn 风格成员变量，编译期模板化）
    static constexpr GmFormat OUT_FORMAT = GetQueryGmFormat<outLayout>();
    using FaGmTensorOut = FaGmTensor<OUTPUT_T, OUT_FORMAT, uint32_t,
                                     (outLayout == FLASH_MLA_WITH_KVCACHE_LAYOUT::TND ||
                                      outLayout == FLASH_MLA_WITH_KVCACHE_LAYOUT::NTD)>;
    FaGmTensorOut outGmTensor_;
    CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<LAYOUT_T>()> copyAttenOutUbToGm_;

    // ub/l1 静态 tensor（按 InitBuffers 中的 UbLayout 和 L1 常量建立）
    LocalTensor<uint8_t> ubMm2ResBuffers_;
    LocalTensor<uint8_t> ubMm1ResBuffers_;
    LocalTensor<uint8_t> ubVec2Res_;
    LocalTensor<uint8_t> ubVec1ResBuffers_;
    LocalTensor<uint8_t> ubMaskBuffers_;
    LocalTensor<float> softmaxSumBuf_;
    LocalTensor<float> softmaxMaxBuf_;
    LocalTensor<float> softmaxExpBuf_;
    LocalTensor<uint8_t> ubLseOutBuf_;
    LocalTensor<uint8_t> ubSumBrdcstBuf_;
    LocalTensor<uint8_t> ubMaxBrdcstBuf_;
    LocalTensor<uint8_t> vec1ApiTmpBuf_;
    LocalTensor<uint8_t> l1PBuffers_;

    const ConstInfoNoQuant &constInfo;
    T negativeFloatScalar = *((const T *)&NEGATIVE_MIN_VALUE_FP32);
    int64_t bmm2SubBlockOffset = 0;
    int64_t vec2SubBlockOffset = 0;

    // ==================== Functions ======================
    __aicore__ inline FlashMlaWithKvcacheNoQuantMlaBlockVec(ConstInfoNoQuant &constInfo, SeqLensToolType &seqLensTool)
        : constInfo(constInfo),
          seqLensTool_(seqLensTool){};

    __aicore__ inline void InitVecBlock(__gm__ uint8_t *attenMask, __gm__ uint8_t *softmaxLse,
                                        __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace)
    {
        InitVecInput(attenMask, softmaxLse, attentionOut, workspace);
    }

    __aicore__ inline void InitVecInput(__gm__ uint8_t *attenMask, __gm__ uint8_t *softmaxLse,
                                        __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace)
    {
        this->attentionOutGm.SetGlobalBuffer((__gm__ OUTPUT_T *)attentionOut);

        if (unlikely(constInfo.isSoftmaxLseEnable)) {
            softmaxLseGm.SetGlobalBuffer((__gm__ float *)softmaxLse);
        }

        if constexpr (hasAtten) {
            attenMaskGmInt.SetGlobalBuffer((__gm__ uint8_t *)attenMask);
        }

        if constexpr (FLASH_DECODE) {
            accumOutGm.SetGlobalBuffer((__gm__ float *)workspace);
            softmaxFDSumGm.SetGlobalBuffer((__gm__ float *)workspace + constInfo.accumOutSize);
            softmaxFDMaxGm.SetGlobalBuffer((__gm__ float *)workspace + constInfo.accumOutSize +
                                           constInfo.logSumExpSize);
        }
        InitAttenOutBuffer();
    }

    __aicore__ inline void InitAttenOutBuffer()
    {
        outGmTensor_.gmTensor = attentionOutGm;
        if constexpr (GmLayoutParams<OUT_FORMAT>::CATEGORY == FormatCategory::GM_Q_OUT_BNGSD) {
            outGmTensor_.offsetCalculator.Init(constInfo.bSize, constInfo.n2Size, constInfo.gSize, constInfo.s1Size,
                                               constInfo.dSizeV, seqLensTool_.qActSeqLensParser);
        } else {
            outGmTensor_.offsetCalculator.Init(constInfo.n2Size, constInfo.gSize, constInfo.dSizeV,
                                               seqLensTool_.qActSeqLensParser);
        }
    }

    __aicore__ inline void ProcessVec1(FlashMlaWithKvcacheRunInfoX runInfo)
    {
        if (unlikely(runInfo.loop == 0U)) {
            AscendC::ICachePreLoad(7);
        }
        uint32_t mm1ResUbBufId = runInfo.loop % UB_MM1_RES_BUFCNT;
        uint32_t pL1BufId = runInfo.loop % L1_P_BUFCNT;
        uint32_t c1v1CrossCoreSyncIdx = CROSSCORE_BMM1_0 + mm1ResUbBufId;
        uint32_t v1c2CrossCoreSyncIdx = CROSSCORE_L1P_0 + pL1BufId;
        LocalTensor<INPUT_T> pL1Tensor = l1PBuffers_[pL1BufId * L1_P_BUF_BYTES].template ReinterpretCast<INPUT_T>();
        auto mm1ResUbTensor = ubMm1ResBuffers_[mm1ResUbBufId * UB_MM1_RES_BUF_BYTES].template ReinterpretCast<T>();

        // 首轮 softmax 折叠：统一走 Update VF 族（ProcessVec1Vf updateFlag=true），由
        // ResetSoftmaxBuffer 把 mloop 槽 sum/max 置 0/-inf 替代旧 isFirstS2Loop 区分
        // （消除首次 softmax VF 分支）
        if (unlikely(runInfo.isFirstS2Loop)) {
            ResetSoftmaxBuffer(runInfo.mloop % UB_SOFTMAX_BUFCNT);
            AscendC::PipeBarrier<PIPE_V>();
        }

        // Prepare the current mask while QK is running; consume it only after QK is ready.
        bool useMask = false;
        if constexpr (hasAtten) {
            if (runInfo.actVecMSize != 0) {
                auto maskUb = ubMaskBuffers_[(runInfo.loop % DB) * UB_MASK_BUF_BYTES];
                useMask = AttenMaskCopyIn(maskUb, 0, runInfo.actVecMSize, runInfo);
            }
        }
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(c1v1CrossCoreSyncIdx);
        if (unlikely(runInfo.actVecMSize == 0)) {
            CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(c1v1CrossCoreSyncIdx); // 反堵 c1v1：AIC 可覆写本槽
            CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(v1c2CrossCoreSyncIdx); // 反堵 v1c2：本 AIV 无 P 行
            return;
        }

        ProcessVec1Nd(pL1Tensor, mm1ResUbTensor, runInfo, useMask);

        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(c1v1CrossCoreSyncIdx);    // C1 收到后可启动 FIXPIPE 写
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(v1c2CrossCoreSyncIdx); // P 已写 L1，C2 可读
        Vec1PostProcess(runInfo);
    }

    // 初始化借用 UB：attentionOut 使用 [0,32) KiB，LSE 使用 [32,64) KiB。
    // 仅在 kernel Init 阶段执行；SyncAll 后才在 Process 中发布 BMM1/BMM2 可用标志。
    // EVENT_ID0/1 保护各 pop-buffer 的 V/MTE3 搬运，初始化与 FA 业务阶段分离。
    __aicore__ inline void ClearOutput(bool needInit)
    {
        if (needInit) {
            uint32_t vecCoreNum = 2 * constInfo.coreNum;
            int64_t tSize = constInfo.bSize * constInfo.s1Size;
            if constexpr (LAYOUT_T == FLASH_MLA_WITH_KVCACHE_LAYOUT::TND ||
                          LAYOUT_T == FLASH_MLA_WITH_KVCACHE_LAYOUT::NTD ||
                          LAYOUT_T == FLASH_MLA_WITH_KVCACHE_LAYOUT::NTD_TND) {
                tSize = seqLensTool_.qActSeqLensParser.GetTSize();
            }
            int64_t totalOutputSize = tSize * constInfo.n2Size * constInfo.gSize * constInfo.dSizeV;

            static constexpr uint32_t UB_ATTEN_POP_BUF_ELE = BUFFER_SIZE_BYTE_32K / sizeof(OUT_T);
            AttentionCommon::InitOutput<OUT_T, EVENT_ID0, 0U, UB_ATTEN_POP_BUF_ELE, true>(
                attentionOutGm, static_cast<uint64_t>(totalOutputSize), vecCoreNum, static_cast<OUT_T>(0));

            if (unlikely(constInfo.isSoftmaxLseEnable)) {
                int64_t lseTotalSize = tSize * constInfo.n2Size * constInfo.gSize;
                static constexpr uint32_t UB_LSE_POP_BUF_ELE = BUFFER_SIZE_BYTE_32K / sizeof(float);
                AttentionCommon::InitOutput<float, EVENT_ID1, BUFFER_SIZE_BYTE_32K, UB_LSE_POP_BUF_ELE, true>(
                    softmaxLseGm, static_cast<uint64_t>(lseTotalSize), vecCoreNum,
                    3e+99); // 3e+99: set the value of invalid batch to inf
            }
            SyncAll();
        }
    }

    __aicore__ inline void SoftmaxDataCopyOut(FlashMlaWithKvcacheRunInfoX runInfo, LocalTensor<float> &sumUb,
                                              LocalTensor<float> &maxUb)
    {
        if constexpr (FLASH_DECODE) {
            if (runInfo.isS2SplitCore) {
                ComputeLogSumExpAndCopyToGm(runInfo, sumUb, maxUb);
            }
        }

        if constexpr (FLASH_DECODE) {
            if (!runInfo.isS2SplitCore && constInfo.isSoftmaxLseEnable) {
                SoftmaxLseCopyOut(sumUb, maxUb, runInfo);
            }
        } else {
            if (unlikely(constInfo.isSoftmaxLseEnable)) {
                SoftmaxLseCopyOut(sumUb, maxUb, runInfo);
            }
        }
    }

    __aicore__ inline void SoftmaxLseCopyOut(LocalTensor<float> &softmaxSumTmp, LocalTensor<float> &softmaxMaxTmp,
                                             FlashMlaWithKvcacheRunInfoX &runInfo)
    {
        if (unlikely(runInfo.actVecMSize == 0)) {
            return;
        }

        Mutex::Lock<PIPE_V>(UB_OUT_LSE_OUT_EVENT0);
        uint32_t vecMIdx = runInfo.gS1Idx + runInfo.vecMbaseIdx;
        LocalTensor<float> lseUb = ubLseOutBuf_.template ReinterpretCast<float>();
        ComputeLseOutputVF(lseUb, softmaxSumTmp, softmaxMaxTmp, runInfo.actVecMSize);
        Mutex::Unlock<PIPE_V>(UB_OUT_LSE_OUT_EVENT0);
        Mutex::Lock<PIPE_MTE3>(UB_OUT_LSE_OUT_EVENT0);

        // flash_mla_with_kvcache LSE 契约为 (N,T) 头主序：runInfo 的 gS1 区间是 [gIdx][s1Idx] token 主序排布，
        // 用 TND→(N,T) 转置写出；bN2Offset = n2Idx * gSize * t1Size = head 块 GM 起址。
        if constexpr (LAYOUT_T == FLASH_MLA_WITH_KVCACHE_LAYOUT::NTD) {
            uint32_t prefixBS1 = seqLensTool_.qActSeqLensParser.GetTBase(runInfo.bIdx);
            uint32_t s1Size = seqLensTool_.qActSeqLensParser.GetActualSeqLength(runInfo.bIdx);
            uint64_t bN2Offset = runInfo.n2Idx * constInfo.n2Size * constInfo.gSize * constInfo.t1Size;
            DataCopySoftmaxLseNTDArch35<T, ConstInfoNoQuant>(softmaxLseGm, lseUb, bN2Offset, vecMIdx,
                                                             runInfo.actVecMSize, constInfo, s1Size);
        } else if constexpr (LAYOUT_T == FLASH_MLA_WITH_KVCACHE_LAYOUT::TND) {
            uint32_t prefixBS1 = seqLensTool_.qActSeqLensParser.GetTBase(runInfo.bIdx);
            uint64_t bN2Offset = runInfo.n2Idx * constInfo.n2Size * constInfo.gSize * constInfo.t1Size;
            DataCopySoftmaxLseTNDtoNTArch35<T, ConstInfoNoQuant>(softmaxLseGm, lseUb, bN2Offset, vecMIdx,
                                                                 runInfo.actVecMSize, prefixBS1, constInfo);
        } else if constexpr (LAYOUT_T == FLASH_MLA_WITH_KVCACHE_LAYOUT::BSH) {
            uint64_t bN2Offset = runInfo.bIdx * constInfo.n2Size * constInfo.gSize * constInfo.s1Size +
                                 runInfo.n2Idx * constInfo.gSize * constInfo.s1Size;
            uint64_t qActSeqLens = seqLensTool_.qActSeqLensParser.GetActualSeqLength(runInfo.bIdx);
            DataCopySoftmaxLseBSNDArch35<T, ConstInfoNoQuant>(softmaxLseGm, lseUb, bN2Offset, vecMIdx,
                                                              runInfo.actVecMSize, constInfo, 0);
        } else { // BNSD
            uint64_t bN2Offset = runInfo.bIdx * constInfo.n2Size * constInfo.gSize * constInfo.s1Size +
                                 runInfo.n2Idx * constInfo.gSize * constInfo.s1Size;
            uint64_t qActSeqLens = seqLensTool_.qActSeqLensParser.GetActualSeqLength(runInfo.bIdx);
            DataCopySoftmaxLseBNSDArch35<T, ConstInfoNoQuant>(softmaxLseGm, lseUb, bN2Offset, vecMIdx,
                                                              runInfo.actVecMSize, constInfo, qActSeqLens, 0);
        }

        Mutex::Unlock<PIPE_MTE3>(UB_OUT_LSE_OUT_EVENT0);
    }

    template <bool APPLY_MASK>
    __aicore__ inline void ProcessSoftmax(LocalTensor<INPUT_T> &stage1CastTensor, LocalTensor<T> &mmRes,
                                          LocalTensor<uint8_t> &attenMaskUb, const FlashMlaWithKvcacheRunInfoX &runInfo)
    {
        LocalTensor<pseShiftType> pseUb;
        LocalTensor<uint8_t> dropMaskUb;
        float slopes = 0.0f;
        float posShift = 0.0f;
        uint32_t pseStride = 0;
        float descaleQK = 1.0;
        float deSCaleKValue = 1.0;

        LocalTensor<float> sumUb =
            softmaxSumBuf_[(runInfo.mloop % UB_SOFTMAX_BUFCNT) * (UB_SOFTMAX_BUF_BYTES / sizeof(float))];
        LocalTensor<float> maxUb =
            softmaxMaxBuf_[(runInfo.mloop % UB_SOFTMAX_BUFCNT) * (UB_SOFTMAX_BUF_BYTES / sizeof(float))];
        LocalTensor<float> expUb =
            softmaxExpBuf_[(runInfo.loop % UB_SOFTMAX_BUFCNT) * (UB_SOFTMAX_BUF_BYTES / sizeof(float))];
        LocalTensor<T> pScaleUb;
        LocalTensor<T> queryScaleUb;
        LocalTensor<uint8_t> apiTmpBuffer;

        apiTmpBuffer = this->vec1ApiTmpBuf_;
        // 统一走 Update VF 族（updateFlag=true）：首轮 sum/max 由 ProcessVec1 的
        // ResetSoftmaxBuffer 置 0/-inf，消除 isFirstS2Loop 分支
        if (likely(runInfo.actSingleLoopS2Size == 128)) {
            ProcessVec1Vf<T, INPUT_T, pseShiftType, true, mBaseSize, 128U, EQ_128, APPLY_MASK, PSE_MODE, HAS_DROP,
                          false, false>(stage1CastTensor, nullptr, sumUb, maxUb, mmRes, expUb, sumUb, maxUb,
                                        attenMaskUb, pseUb, dropMaskUb, apiTmpBuffer, pScaleUb, runInfo.actVecMSize,
                                        runInfo.actSingleLoopS2Size, pseStride, slopes, posShift,
                                        constInfo.scaleValue, // constInfo.scaleValue 已是 T float类型
                                        descaleQK, negativeFloatScalar, 0.0F, queryScaleUb, deSCaleKValue);
        } else if (runInfo.actSingleLoopS2Size <= 64) {
            ProcessVec1Vf<T, INPUT_T, pseShiftType, true, mBaseSize, 128U, GT_0_AND_LTE_64, APPLY_MASK, PSE_MODE,
                          HAS_DROP, false, false>(
                stage1CastTensor, nullptr, sumUb, maxUb, mmRes, expUb, sumUb, maxUb, attenMaskUb, pseUb, dropMaskUb,
                apiTmpBuffer, pScaleUb, runInfo.actVecMSize, runInfo.actSingleLoopS2Size, pseStride, slopes, posShift,
                constInfo.scaleValue, descaleQK, negativeFloatScalar, 0.0F, queryScaleUb, deSCaleKValue);
        } else if (runInfo.actSingleLoopS2Size < 128) {
            ProcessVec1Vf<T, INPUT_T, pseShiftType, true, mBaseSize, 128U, GT_64_AND_LTE_128, APPLY_MASK, PSE_MODE,
                          HAS_DROP, false, false>(
                stage1CastTensor, nullptr, sumUb, maxUb, mmRes, expUb, sumUb, maxUb, attenMaskUb, pseUb, dropMaskUb,
                apiTmpBuffer, pScaleUb, runInfo.actVecMSize, runInfo.actSingleLoopS2Size, pseStride, slopes, posShift,
                constInfo.scaleValue, descaleQK, negativeFloatScalar, 0.0F, queryScaleUb, deSCaleKValue);
        } else {
            if constexpr (s2BaseSize == 256) {
                ProcessVec1Vf<T, INPUT_T, pseShiftType, true, mBaseSize, 128U, GT_128_AND_LTE_256, APPLY_MASK, PSE_MODE,
                              HAS_DROP>(stage1CastTensor, nullptr, sumUb, maxUb, mmRes, expUb, sumUb, maxUb,
                                        attenMaskUb, pseUb, dropMaskUb, apiTmpBuffer, expUb, runInfo.actVecMSize,
                                        runInfo.actSingleLoopS2Size, pseStride, slopes, posShift, constInfo.scaleValue,
                                        descaleQK, negativeFloatScalar, 0.0F);
            }
        }
    }

    __aicore__ inline void ProcessVec1Nd(LocalTensor<INPUT_T> &pL1Tensor, LocalTensor<T> &mm1ResUbTensor,
                                         FlashMlaWithKvcacheRunInfoX runInfo, bool useMask)
    {
        LocalTensor<uint8_t> attenMaskUb;
        const uint32_t maskBufId = runInfo.loop % DB;
        if constexpr (hasAtten) {
            attenMaskUb = ubMaskBuffers_[maskBufId * UB_MASK_BUF_BYTES];
            // Mask preparation was issued before the QK wait. Fully visible tiles do not use this slot.
            if (useMask) {
                Mutex::Lock<PIPE_V>(UB_IN_MASK_EVENT0 + maskBufId);
            }
        }

        Mutex::Lock<PIPE_V>(UB_OUT_VEC1_RES_EVENT0);
        LocalTensor<INPUT_T> stage1CastTensor = ubVec1ResBuffers_.template ReinterpretCast<INPUT_T>();
        if constexpr (hasAtten) {
            if (useMask) {
                ProcessSoftmax<true>(stage1CastTensor, mm1ResUbTensor, attenMaskUb, runInfo);
            } else {
                ProcessSoftmax<false>(stage1CastTensor, mm1ResUbTensor, attenMaskUb, runInfo);
            }
        } else {
            ProcessSoftmax<false>(stage1CastTensor, mm1ResUbTensor, attenMaskUb, runInfo);
        }
        Mutex::Unlock<PIPE_V>(UB_OUT_VEC1_RES_EVENT0);
        if constexpr (hasAtten) {
            // 软max 已消费完本槽 mask，Unlock 释放供下一轮 MTE2 覆写
            if (useMask) {
                Mutex::Unlock<PIPE_V>(UB_IN_MASK_EVENT0 + maskBufId);
            }
        }

        // ===================DataCopy to L1 ====================
        Mutex::Lock<PIPE_MTE3>(UB_OUT_VEC1_RES_EVENT0);
        LocalTensor<INPUT_T> mm2AL1Tensor = pL1Tensor;

        if (likely(runInfo.actVecMSize != 0)) {
            DataCopy(mm2AL1Tensor[s2BaseSize * dVBaseSize +
                                  runInfo.vecMbaseIdx * (AttentionCommon::BYTE_BLOCK / sizeof(INPUT_T))],
                     stage1CastTensor,
                     {s2BaseSize / 16, (uint16_t)runInfo.actVecMSize, (uint16_t)(vec1Srcstride - runInfo.actVecMSize),
                      (uint16_t)(mBaseSize - runInfo.actVecMSize)});
        }
        Mutex::Unlock<PIPE_MTE3>(UB_OUT_VEC1_RES_EVENT0);
    }

    __aicore__ inline void Vec1PostProcess(FlashMlaWithKvcacheRunInfoX runInfo)
    {
        LocalTensor<float> sumUb =
            softmaxSumBuf_[(runInfo.mloop % UB_SOFTMAX_BUFCNT) * (UB_SOFTMAX_BUF_BYTES / sizeof(float))];
        LocalTensor<float> maxUb =
            softmaxMaxBuf_[(runInfo.mloop % UB_SOFTMAX_BUFCNT) * (UB_SOFTMAX_BUF_BYTES / sizeof(float))];
        LocalTensor<float> expUb =
            softmaxExpBuf_[(runInfo.loop % UB_SOFTMAX_BUFCNT) * (UB_SOFTMAX_BUF_BYTES / sizeof(float))];

        // 折叠软最大值：首轮时 mloop 槽 sum/max 已由 ProcessVec1 的 ResetSoftmaxBuffer
        // 置 0/-inf，Update 恒执行（消除 isFirstS2Loop 区分）
        UpdateExpSumAndExpMax<T>(sumUb, maxUb, expUb, sumUb, maxUb, vec1ApiTmpBuf_, runInfo.actVecMSize);

        if (unlikely(runInfo.isLastS2Loop)) {
            SoftmaxDataCopyOut(runInfo, sumUb, maxUb);
        }
    }

    __aicore__ inline void ProcessVec2(FlashMlaWithKvcacheRunInfoX runInfo)
    {
        auto dst = ubVec2Res_.template ReinterpretCast<T>();
        auto exp = softmaxExpBuf_[(runInfo.loop % UB_SOFTMAX_BUFCNT) * (UB_SOFTMAX_BUF_BYTES / sizeof(float))];
        auto sum = softmaxSumBuf_[(runInfo.mloop % UB_SOFTMAX_BUFCNT) * (UB_SOFTMAX_BUF_BYTES / sizeof(float))];
        Mutex::Lock<PIPE_V>(UB_OUT_VEC2_RES_EVENT0);
        for (uint32_t n = 0; n < 2; ++n) {
            auto cur = ubMm2ResBuffers_[(n & 1U) * UB_MM2_RES_BUF_BYTES].template ReinterpretCast<T>();
            CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CROSSCORE_BMM2_0 + (n & 1U));
            if (runInfo.actVecMSize != 0) {
                if (runInfo.isFirstS2Loop) {
                    if (runInfo.isLastS2Loop) {
                        MlaStreamUpdate<true, true>((__ubuf__ float *)dst.GetPhyAddr() + n * 256U,
                                                    (__ubuf__ float *)cur.GetPhyAddr(),
                                                    (__ubuf__ float *)exp.GetPhyAddr(),
                                                    (__ubuf__ float *)sum.GetPhyAddr(), runInfo.actVecMSize);
                    } else {
                        MlaStreamUpdate<true, false>((__ubuf__ float *)dst.GetPhyAddr() + n * 256U,
                                                     (__ubuf__ float *)cur.GetPhyAddr(),
                                                     (__ubuf__ float *)exp.GetPhyAddr(),
                                                     (__ubuf__ float *)sum.GetPhyAddr(), runInfo.actVecMSize);
                    }
                } else {
                    if (runInfo.isLastS2Loop) {
                        MlaStreamUpdate<false, true>((__ubuf__ float *)dst.GetPhyAddr() + n * 256U,
                                                     (__ubuf__ float *)cur.GetPhyAddr(),
                                                     (__ubuf__ float *)exp.GetPhyAddr(),
                                                     (__ubuf__ float *)sum.GetPhyAddr(), runInfo.actVecMSize);
                    } else {
                        MlaStreamUpdate<false, false>((__ubuf__ float *)dst.GetPhyAddr() + n * 256U,
                                                      (__ubuf__ float *)cur.GetPhyAddr(),
                                                      (__ubuf__ float *)exp.GetPhyAddr(),
                                                      (__ubuf__ float *)sum.GetPhyAddr(), runInfo.actVecMSize);
                    }
                }
            }
            CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CROSSCORE_BMM2_0 + (n & 1U));
        }
        Mutex::Unlock<PIPE_V>(UB_OUT_VEC2_RES_EVENT0);
        if (runInfo.isLastS2Loop && runInfo.actVecMSize != 0) {
            CopyOutAttentionOut(runInfo, dst, 0, runInfo.actVecMSize);
        }
    }

    __aicore__ inline void CopyOutAttentionOut(FlashMlaWithKvcacheRunInfoX runInfo, LocalTensor<T> &vec2ResUb,
                                               uint32_t mStartVec, uint32_t mDealSize)
    {
        if constexpr (FLASH_DECODE) {
            if (runInfo.isS2SplitCore) {
                Mutex::Lock<PIPE_MTE3>(UB_OUT_VEC2_RES_EVENT0);
                Bmm2ResForFDCopyOut(runInfo, vec2ResUb, mStartVec, mDealSize);
                Mutex::Unlock<PIPE_MTE3>(UB_OUT_VEC2_RES_EVENT0);
            } else {
                Bmm2ResCastAndCopyOut(runInfo, vec2ResUb, mStartVec, mDealSize);
            }
        } else {
            Bmm2ResCastAndCopyOut(runInfo, vec2ResUb, mStartVec, mDealSize);
        }
    }

    __aicore__ inline void Bmm2ResCastAndCopyOut(FlashMlaWithKvcacheRunInfoX &runInfo, LocalTensor<T> &vec2ResUb,
                                                 uint32_t mStartVec, uint32_t mDealSize)
    {
        LocalTensor<OUTPUT_T> attenOut;
        attenOut.SetAddr(vec2ResUb.address_);

        int64_t dSizeAligned64 = static_cast<int64_t>(FA_T::dVBaseSize);

        Mutex::Lock<PIPE_V>(UB_OUT_VEC2_RES_EVENT0);
        RowInvalid(vec2ResUb, mStartVec, mDealSize, runInfo, dSizeAligned64);
        Cast(attenOut, vec2ResUb, RoundMode::CAST_ROUND, mDealSize * dSizeAligned64);
        Mutex::Unlock<PIPE_V>(UB_OUT_VEC2_RES_EVENT0);
        Mutex::Lock<PIPE_MTE3>(UB_OUT_VEC2_RES_EVENT0);
        Bmm2DataCopyOutTrans(runInfo, attenOut, mStartVec, mDealSize);
        Mutex::Unlock<PIPE_MTE3>(UB_OUT_VEC2_RES_EVENT0);
    }

    __aicore__ inline bool CalcBlockNeedRowInvalid(FlashMlaWithKvcacheRunInfoX &runInfo, int64_t s1FirstValidToken,
                                                   int64_t s1LastValidToken)
    {
        int32_t vecMStartIdx = runInfo.gS1Idx + runInfo.vecMbaseIdx;
        int32_t vecMEndIdx = vecMStartIdx + runInfo.actVecMSize - 1;
        int32_t s1StartTdx;
        int32_t s1EndTdx;
        bool ret = false;
        if constexpr (LAYOUT_T == FLASH_MLA_WITH_KVCACHE_LAYOUT::BSH ||
                      LAYOUT_T == FLASH_MLA_WITH_KVCACHE_LAYOUT::TND) {
            // S1G layout
            s1StartTdx = vecMStartIdx / constInfo.gSize;
            s1EndTdx = vecMEndIdx / constInfo.gSize;
            ret = (s1StartTdx < s1FirstValidToken) || (s1EndTdx > s1LastValidToken);
        } else {
            // GS1 layout
            s1StartTdx = vecMStartIdx % runInfo.actS1Size;
            s1EndTdx = vecMEndIdx % runInfo.actS1Size;
            int32_t gStartIdx = vecMStartIdx / runInfo.actS1Size;
            int32_t gEndIdx = vecMEndIdx / runInfo.actS1Size;
            if (gStartIdx != gEndIdx) { // 跨多个G
                ret = (s1FirstValidToken > 0) || (s1LastValidToken < (runInfo.actS1Size - 1));
            } else { // 只跨1个G
                ret = (s1StartTdx < s1FirstValidToken) || (s1EndTdx > s1LastValidToken);
            }
        }
        return ret;
    }

    template <typename VEC2_RES_T>
    __aicore__ inline void RowInvalid(LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t mStartVec, int64_t mDealSize,
                                      FlashMlaWithKvcacheRunInfoX &runInfo, int64_t dSizeAligned64)
    {
        if constexpr (hasAtten) {
            int64_t s1FirstValidToken =
                AttentionCommon::Min(AttentionCommon::Max(-runInfo.nextTokensLeftUp, 0), runInfo.actS1Size);
            int64_t s1LastValidToken = AttentionCommon::Min(
                AttentionCommon::Max(runInfo.preTokensLeftUp + runInfo.actS2Size, 0), runInfo.actS1Size);
            s1LastValidToken = AttentionCommon::Max(s1LastValidToken - 1, 0);
            bool hasValidRow = (s1FirstValidToken > 0) || (s1LastValidToken < runInfo.actS1Size);
            bool batchNeedRowInvalid = constInfo.isRowInvalidOpen || // 手动开启行无效
                                       ((constInfo.sparseMode != SparseMode::LEFT_UP_CAUSAL) &&
                                        hasValidRow); // sparse = 0 or 3 or 4，preTokens or nextTokens负数
            if (!batchNeedRowInvalid) {
                return;
            }
            bool blockNeedRowInvalid = CalcBlockNeedRowInvalid(runInfo, s1FirstValidToken, s1LastValidToken);
            blockNeedRowInvalid = blockNeedRowInvalid || constInfo.isRowInvalidOpen;
            if (blockNeedRowInvalid) {
                LocalTensor<float> maxTensor =
                    softmaxMaxBuf_[(runInfo.mloop % UB_SOFTMAX_BUFCNT) * (UB_SOFTMAX_BUF_BYTES / sizeof(float)) +
                                   mStartVec];
                RowInvalidUpdateVF<float>(vec2ResUb, maxTensor, mDealSize, constInfo.dSizeV,
                                          static_cast<uint32_t>(dSizeAligned64));
            }
        }
    }

    __aicore__ inline void Bmm2DataCopyOutTrans(const FlashMlaWithKvcacheRunInfoX &info,
                                                LocalTensor<OUTPUT_T> &attenOutUb, uint32_t vecMIdx,
                                                uint32_t dealRowCount)
    {
        GmCoordGs1Merge gmCoord{.bIdx = info.bIdx,
                                .n2Idx = info.n2Idx,
                                .gS1Idx = (info.gS1Idx + info.vecMbaseIdx + vecMIdx),
                                .dIdx = 0,
                                .gS1DealSize = dealRowCount,
                                .dDealSize = (uint32_t)constInfo.dSizeV};
        FaUbTensor<OUTPUT_T, false> ubTensor{
            .tensor = attenOutUb, .rowCount = dealRowCount, .colCount = (uint32_t)(dVTemplateAlign64)};
        CopyAttentionOut(ubTensor, gmCoord);
    }

    __aicore__ inline void CopyAttentionOut(FaUbTensor<OUTPUT_T, false> &ubTensor, GmCoordGs1Merge &gmCoord)
    {
        copyAttenOutUbToGm_(outGmTensor_, ubTensor, gmCoord);
    }

    __aicore__ inline void BroadCastAndCopyOut(const FlashMlaWithKvcacheRunInfoX &runInfo, LocalTensor<float> &sumUb,
                                               LocalTensor<float> &maxUb, int64_t gmOffset, int64_t calculateSize)
    {
        // Copy sum to gm
        Mutex::Lock<PIPE_V>(UB_BRDCST_SUM_EVENT);
        LocalTensor<float> sumOutTensor = ubSumBrdcstBuf_.template ReinterpretCast<float>();
        FaVectorApi::BroadcastMaxSum(sumOutTensor, sumUb, runInfo.actVecMSize);
        Mutex::Unlock<PIPE_V>(UB_BRDCST_SUM_EVENT);
        Mutex::Lock<PIPE_MTE3>(UB_BRDCST_SUM_EVENT);
        DataCopy(softmaxFDSumGm[gmOffset], sumOutTensor, calculateSize);
        Mutex::Unlock<PIPE_MTE3>(UB_BRDCST_SUM_EVENT);

        // Copy max to gm
        Mutex::Lock<PIPE_V>(UB_BRDCST_MAX_EVENT);
        LocalTensor<float> maxOutTensor = ubMaxBrdcstBuf_.template ReinterpretCast<float>();
        FaVectorApi::BroadcastMaxSum(maxOutTensor, maxUb, runInfo.actVecMSize);
        Mutex::Unlock<PIPE_V>(UB_BRDCST_MAX_EVENT);
        Mutex::Lock<PIPE_MTE3>(UB_BRDCST_MAX_EVENT);
        DataCopy(softmaxFDMaxGm[gmOffset], maxOutTensor, calculateSize);
        Mutex::Unlock<PIPE_MTE3>(UB_BRDCST_MAX_EVENT);
    }

    __aicore__ inline void ComputeLogSumExpAndCopyToGm(const FlashMlaWithKvcacheRunInfoX &runInfo,
                                                       LocalTensor<float> &sumUb, LocalTensor<float> &maxUb)
    {
        if (unlikely(runInfo.actVecMSize == 0)) {
            return;
        }
        int64_t calculateSize = runInfo.actVecMSize * fp32BaseSize;
        int64_t gmOffset = runInfo.faTmpOutWsPos * mBaseSize * fp32BaseSize + runInfo.vecMbaseIdx * fp32BaseSize;
        // Copy sum to gm
        BroadCastAndCopyOut(runInfo, sumUb, maxUb, gmOffset, calculateSize);
    }

    __aicore__ inline void Bmm2ResForFDCopyOut(const FlashMlaWithKvcacheRunInfoX &runInfo, LocalTensor<T> &vec2ResUb,
                                               uint32_t mStartVec, uint32_t mDealSize)
    {
        int64_t dSizeAligned64 = (int64_t)FA_T::dVBaseSize;
        uint64_t gmOffset =
            runInfo.faTmpOutWsPos * mBaseSize * constInfo.dSizeV + (runInfo.vecMbaseIdx + mStartVec) * constInfo.dSizeV;
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = mDealSize;
        dataCopyParams.blockLen = constInfo.dSizeV * sizeof(T);
        dataCopyParams.srcStride = (dSizeAligned64 - constInfo.dSizeV) / (BaseApi::FA_BYTE_BLOCK / sizeof(T));
        dataCopyParams.dstStride = 0;
        DataCopyPad(accumOutGm[gmOffset], vec2ResUb, dataCopyParams);
    }

    __aicore__ inline void InitBuffers()
    {
        /*--------------------------------------------L1--------------------------------------------*/
        // l1P 三缓冲：A1 108K 起（跳过 cube Q 区），槽内 P 从 112 KiB 偏移开始，覆盖 rope 与预留区
        l1PBuffers_ = LocalTensor<uint8_t>(TPosition::A1, L1_Q_PREFIX_BYTES, L1_P_BUFCNT * L1_P_BUF_BYTES);

        /*--------------------------------------------UB--------------------------------------------*/
        struct UbLayout {
            uint8_t mm2ResBuffers[UB_MM2_RES_BUFCNT]
                                 [UB_MM2_RES_BUF_BYTES]; // 2*48K=96K @0，CV通信BUF（与 cube 同一偏移）
            uint8_t mm1ResBuffers[UB_MM1_RES_BUFCNT][UB_MM1_RES_BUF_BYTES]; // 1*24K=24K @96K，CV通信BUF
            uint8_t vec2Res[UB_VEC2_RES_BUF_BYTES];                         // 96K @120K，vec2/输出BUF（单槽）
            uint8_t vec1Res[UB_VEC1_RES_BUF_BYTES];                 // 12.25K @216K，stage1/softmax结果（单槽）
            uint8_t maskBuffers[UB_MASK_BUFCNT][UB_MASK_BUF_BYTES]; // 2*6144B=12 KiB，MASK 输入双槽
            uint8_t softmaxSumBuf[UB_SOFTMAX_BUFCNT][UB_SOFTMAX_BUF_BYTES]; // 3*256B，sum 常驻缓冲
            uint8_t softmaxMaxBuf[UB_SOFTMAX_BUFCNT][UB_SOFTMAX_BUF_BYTES]; // 3*256B，max 常驻缓冲
            uint8_t softmaxExpBuf[UB_SOFTMAX_BUFCNT][UB_SOFTMAX_BUF_BYTES]; // 3*256B，exp 常驻缓冲
            uint8_t lseOutBuf[UB_LSE_OUT_BUF_BYTES];                        // 1.5K，LSE输出
            uint8_t sumBrdcstBuf[UB_BRDCST_BUF_BYTES];                      // 2K，FD sum/max staging
            // FD max/sum 共用 2K staging，并使用同一个 V/MTE3 Mutex。
            uint8_t tmpBuf[UB_TMP_BUF_BYTES]; // 0.5K，softmax 中间结果缓存
        };
        static_assert(sizeof(UbLayout) <= 248 * 1024, "UB buffer too large");
        ubMm2ResBuffers_ = LocalTensor<uint8_t>(TPosition::VECIN, OFFSET_OF_MEMBER(UbLayout, mm2ResBuffers),
                                                SIZE_OF_MEMBER(UbLayout, mm2ResBuffers));
        ubMm1ResBuffers_ = LocalTensor<uint8_t>(TPosition::VECIN, OFFSET_OF_MEMBER(UbLayout, mm1ResBuffers),
                                                SIZE_OF_MEMBER(UbLayout, mm1ResBuffers));
        ubVec2Res_ = LocalTensor<uint8_t>(TPosition::VECIN, OFFSET_OF_MEMBER(UbLayout, vec2Res),
                                          SIZE_OF_MEMBER(UbLayout, vec2Res));
        ubVec1ResBuffers_ = LocalTensor<uint8_t>(TPosition::VECIN, OFFSET_OF_MEMBER(UbLayout, vec1Res),
                                                 SIZE_OF_MEMBER(UbLayout, vec1Res));
        ubMaskBuffers_ = LocalTensor<uint8_t>(TPosition::VECIN, OFFSET_OF_MEMBER(UbLayout, maskBuffers),
                                              SIZE_OF_MEMBER(UbLayout, maskBuffers));
        softmaxSumBuf_ = LocalTensor<uint8_t>(TPosition::VECIN, OFFSET_OF_MEMBER(UbLayout, softmaxSumBuf),
                                              SIZE_OF_MEMBER(UbLayout, softmaxSumBuf))
                             .template ReinterpretCast<float>();
        softmaxMaxBuf_ = LocalTensor<uint8_t>(TPosition::VECIN, OFFSET_OF_MEMBER(UbLayout, softmaxMaxBuf),
                                              SIZE_OF_MEMBER(UbLayout, softmaxMaxBuf))
                             .template ReinterpretCast<float>();
        softmaxExpBuf_ = LocalTensor<uint8_t>(TPosition::VECIN, OFFSET_OF_MEMBER(UbLayout, softmaxExpBuf),
                                              SIZE_OF_MEMBER(UbLayout, softmaxExpBuf))
                             .template ReinterpretCast<float>();
        ubLseOutBuf_ = LocalTensor<uint8_t>(TPosition::VECIN, OFFSET_OF_MEMBER(UbLayout, lseOutBuf),
                                            SIZE_OF_MEMBER(UbLayout, lseOutBuf));
        ubSumBrdcstBuf_ = LocalTensor<uint8_t>(TPosition::VECIN, OFFSET_OF_MEMBER(UbLayout, sumBrdcstBuf),
                                               SIZE_OF_MEMBER(UbLayout, sumBrdcstBuf));
        ubMaxBrdcstBuf_ = ubSumBrdcstBuf_;
        vec1ApiTmpBuf_ = LocalTensor<uint8_t>(TPosition::VECIN, OFFSET_OF_MEMBER(UbLayout, tmpBuf),
                                              SIZE_OF_MEMBER(UbLayout, tmpBuf));
    }

    __aicore__ inline void ResetSoftmaxBuffer(uint32_t slotIdx)
    {
        constexpr uint32_t softmaxBufElementCount = UB_SOFTMAX_BUF_BYTES / sizeof(float);
        LocalTensor<float> sumUb = softmaxSumBuf_[slotIdx * softmaxBufElementCount];
        LocalTensor<float> maxUb = softmaxMaxBuf_[slotIdx * softmaxBufElementCount];
        Duplicate<float>(sumUb, static_cast<float>(0), softmaxBufElementCount);
        Duplicate<float>(maxUb, static_cast<float>(-std::numeric_limits<float>::infinity()), softmaxBufElementCount);
    }

    __aicore__ inline void InitCrossCoreSync()
    {
        // AIV 预置 BMM1_0、BMM2_0/1 三个可用标志：AIC 首个 bmm1/bmm2 Wait 立即通过；
        // v1c2 三旗标随任务 ping-pong 无需预置（首轮 bmm2 的 Wait 有 AIV 首个 P 写保证）
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CROSSCORE_BMM2_0);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CROSSCORE_BMM2_1);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CROSSCORE_BMM1_0);
    }

    __aicore__ inline void UnInitCrossCoreSync() {}

    __aicore__ inline void AllocEventID()
    {
        // 核内同步使用上方定义的静态 Mutex ID，无动态事件可分配
    }

    __aicore__ inline void FreeEventID() {}

    // Return false for a fully visible tile without touching mask buffers or their synchronization.
    __aicore__ inline bool AttenMaskCopyIn(LocalTensor<uint8_t> attenMaskUb, uint32_t vecMIdx, uint32_t mDealSize,
                                           FlashMlaWithKvcacheRunInfoX &runInfo)
    {
        MaskInfo maskInfo;
        maskInfo.gs1StartIdx = runInfo.gS1Idx + runInfo.vecMbaseIdx + vecMIdx;
        maskInfo.gs1dealNum = mDealSize;
        maskInfo.s1Size = runInfo.actS1Size;
        maskInfo.gSize = constInfo.gSize;
        maskInfo.s2StartIdx = runInfo.s2Idx;
        maskInfo.s2dealNum = runInfo.actSingleLoopS2Size;
        maskInfo.s2Size = runInfo.actS2Size;
        maskInfo.nBaseSize = 128U;
        maskInfo.preToken = constInfo.preTokens;
        maskInfo.nextToken = constInfo.nextTokens;
        maskInfo.sparseMode = static_cast<SparseMode>(constInfo.sparseMode);
        maskInfo.batchIdx = (constInfo.attenMaskBatch == 1) ? 0 : runInfo.bIdx;
        maskInfo.attenMaskBatchStride = constInfo.attenMaskS1Size * constInfo.attenMaskS2Size;
        maskInfo.attenMaskS1Stride = constInfo.attenMaskS2Size;
        maskInfo.attenMaskDstStride = (128U - AttentionCommon::Align(maskInfo.s2dealNum, 32U)) / 32;
        maskInfo.maskValue = negativeIntScalar;
        maskInfo.s1LeftPaddingSize = runInfo.qPaddingBeginOffset;
        maskInfo.s2LeftPaddingSize = runInfo.kvPaddingBeginOffset;
        maskInfo.maskFormat = MASK_LAYOUT;
        maskInfo.attenMaskType = MASK_BOOL; // compatible with int8/uint8

        bool IsSkipMask = IsSkipAttentionmask(maskInfo);
        bool IsSkipMaskForPre = IsSkipAttentionmaskForPre(maskInfo);
        // 锁全部内置于本函数（各槽 PIPE_V/MTE2 握手在拷贝内完成）
        const uint32_t maskBufId = runInfo.loop % DB;
        if (IsSkipMask && IsSkipMaskForPre) {
            // Fully visible tiles use the unmasked VF and do not access the mask slots.
            return false;
        }

        if (!IsSkipMask) {
            AttentionmaskCopyIn<uint8_t, MASK_LAYOUT, true, 128U>(attenMaskUb, attenMaskGmInt, maskInfo, false,
                                                                  UB_IN_MASK_EVENT0 + maskBufId);
        } else {
            Mutex::Lock<PIPE_V>(UB_IN_MASK_EVENT0 + maskBufId);
            Duplicate(attenMaskUb, static_cast<uint8_t>(0U), maskInfo.gs1dealNum * 128U);
            Mutex::Unlock<PIPE_V>(UB_IN_MASK_EVENT0 + maskBufId);
        }

        if (!IsSkipMaskForPre) {
            const uint32_t preBufId = maskBufId ^ 1U;
            LocalTensor<uint8_t> attenMaskUbPre = ubMaskBuffers_[preBufId * UB_MASK_BUF_BYTES];
            AttentionmaskCopyIn<uint8_t, MASK_LAYOUT, true, 128U>(attenMaskUbPre, attenMaskGmInt, maskInfo, true,
                                                                  UB_IN_MASK_EVENT0 + preBufId);
            Mutex::Lock<PIPE_V>(UB_IN_MASK_EVENT0 + preBufId);
            MergeMask(attenMaskUb, attenMaskUbPre, maskInfo.gs1dealNum, 128U);
            Mutex::Unlock<PIPE_V>(UB_IN_MASK_EVENT0 + preBufId);
        }
        return true;
    }
};

template <typename FA_T>
class FlashMlaWithKvcacheNoQuantMlaBlockVecDummy {
public:
    using INPUT_T = typename FA_T::inputType;
    using OUTPUT_T = typename FA_T::outputType;
    static constexpr bool HAS_MASK = FA_T::hasMask;
    static constexpr bool FLASH_DECODE = FA_T::flashDecode;
    using OUT_T = OUTPUT_T;
    using ConstInfoNoQuant = ConstInfo_t<FlashMlaWithKvcacheKernelType::NO_QUANT>;
    template <typename FlashMlaSeqLensToolT>
    __aicore__ inline FlashMlaWithKvcacheNoQuantMlaBlockVecDummy(ConstInfoNoQuant &constInfo,
                                                                 FlashMlaSeqLensToolT &seqLensTool){};
};

} // namespace FlashAttnKernel

#endif // FLASH_MLA_WITH_KVCACHE_BLOCK_VEC_H_
