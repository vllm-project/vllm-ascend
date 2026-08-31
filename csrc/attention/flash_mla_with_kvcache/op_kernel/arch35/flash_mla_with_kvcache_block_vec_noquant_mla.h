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
 * \file flash_mla_with_kvcache_block_vec_noquant_mla.h
 * \brief arch35 flash_mla_with_kvcache 非量化 MLA vector（由 fia_block_vec_noquant_mla.h 复制改名）；
 *        静态 tensor UbLayout + Mutex/cross-core 显式同步 + FA 协同清零
 *        （镜像 flash_attn_block_vec_nd.h；FD 阶段 buffer 复用见表 1 注 4/注 5）
 */
#ifndef FLASH_MLA_WITH_KVCACHE_BLOCK_VEC_NOQUANT_MLA_H_
#define FLASH_MLA_WITH_KVCACHE_BLOCK_VEC_NOQUANT_MLA_H_

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include <limits>

#include "../utils/attenmask_gs1.h"
#if __has_include("../../../common/op_kernel/arch35/flash_attention_score_common_regbase_arch35.h")
#include "../../../common/op_kernel/arch35/flash_attention_score_common_regbase_arch35.h"
#else
#include "../../common/arch35/flash_attention_score_common_regbase_arch35.h"
#endif
#include "adv_api/activation/softmax.h"
#if __has_include("../../../common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz.h")
#include "../../../common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz.h"
#include "../../../common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz_dn.h"
#include "../../../common/op_kernel/arch35/vf/vf_flashupdate_new.h"
#include "../../../common/op_kernel/arch35/vf/vf_div_cast_arch35.h"
#include "../../../common/op_kernel/arch35/vf/vf_flash_decode_arch35.h"
#include "../../../common/op_kernel/init_output.h"
#else
#include "../../common/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz.h"
#include "../../common/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz_dn.h"
#include "../../common/arch35/vf/vf_flashupdate_new.h"
#include "../../common/arch35/vf/vf_div_cast_arch35.h"
#include "../../common/arch35/vf/vf_flash_decode_arch35.h"
#include "../../common/init_output.h"
#endif
#include "flash_mla_with_kvcache_public_define_arch35.h"
#if __has_include("../../../common/op_kernel/vector_common.h")
#include "../../../common/op_kernel/vector_common.h"
#else
#include "../../common/vector_common.h"
#endif
#include "memory_copy_arch35.h"
#include "flash_mla_with_kvcache_type.h"

using namespace AscendC;
using namespace FaVectorApi;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;
using namespace AttentionCommon;

namespace FlashAttnKernel {
template <typename FA_T>
class FlashMlaWithKvcacheNoQuantMlaBlockVec {
public:
    using INPUT_T = typename FA_T::inputType;
    using T = typename FA_T::mmType;
    using OUTPUT_T = typename FA_T::outputType;
    static constexpr LayOutTypeEnum layout = FA_T::qLayout;
    static constexpr LayOutTypeEnum outLayout = FA_T::attnOutLayout;
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

    static constexpr ActualSeqLensMode Q_MODE = GetQActSeqMode<layout>();
    static constexpr ActualSeqLensMode KV_MODE = GetKvActSeqMode<layout, (FA_T::kvLayoutType > 0)>();
    using SeqLensToolType = FlashMlaSeqLensTool<Q_MODE, KV_MODE>;
    static constexpr MaskFormat MASK_LAYOUT =
        (layout == LayOutTypeEnum::LAYOUT_BSH || layout == LayOutTypeEnum::LAYOUT_TND) ? MaskFormat::SG :
                                                                                         MaskFormat::GS;

    using pseShiftType = INPUT_T;

    static constexpr T BOOL_ATTEN_MASK_SCALAR_VALUE = -1000000000000.0; // 用于mask为bool类型
    uint32_t negativeIntScalar = *((uint32_t *)&BOOL_ATTEN_MASK_SCALAR_VALUE);

    using attenMaskGmType = typename std::conditional<hasAtten, GlobalTensor<uint8_t>, int8_t>::type;
    using flashdecodeGmType = typename std::conditional<FLASH_DECODE, GlobalTensor<float>, int8_t>::type;
    using ConstInfoNoQuant = ConstInfo_t<FlashMlaWithKvcacheKernelType::NO_QUANT>;
    using OUT_T = OUTPUT_T;

    // ================== 核间同步 ID（与 cube 侧逐字节一致，参照 flash_attn_block_vec_nd.h:74-82）==================
    static constexpr uint64_t CROSS_CORE_SYNC_MODE = 4U;
    static constexpr uint32_t CROSSCORE_BMM1_0 = 0U;
    static constexpr uint32_t CROSSCORE_BMM1_1 = 1U;
    static constexpr uint32_t CROSSCORE_BMM2_0 = 2U;
    static constexpr uint32_t CROSSCORE_BMM2_1 = 3U;
    static constexpr uint32_t CROSSCORE_L1P_0 = 5U;
    static constexpr uint32_t CROSSCORE_L1P_1 = 6U;
    static constexpr uint32_t CROSSCORE_L1P_2 = 7U;
    // AIV0_AIV1_OFFSET 使用全局宏（attention/common/op_kernel/buffer.h:30，=16）

    // ================== 核内 Mutex ID（AIV 核内，表 2 + 注 10 枚举；mask 槽由 common 自同步、6/7 预留同 FA
    // 排布）==================
    static constexpr uint32_t UB_OUT_VEC2_RES_EVENT0 = 0U; // vec2Res（64K@160K）V/MTE3
    static constexpr uint32_t EVENT_ID0 = 1U;              // InitOutput atten-out pop 同步（PIPE_V/PIPE_MTE3）
    static constexpr uint32_t UB_OUT_VEC1_RES_EVENT0 = 2U; // stage1/vec1Res（8.25K@224K）V/MTE3
    static constexpr uint32_t EVENT_ID1 = 3U;              // InitOutput LSE pop 同步（PIPE_V/PIPE_MTE3）
    static constexpr uint32_t UB_OUT_LSE_OUT_EVENT0 = 4U;  // LSE 1K 输出（V/MTE3）
    static constexpr uint32_t UB_BRDCST_SUM_EVENT = 5U;    // sumBrdcst 1K（V/MTE3）
    // mask 槽 2×4K 的 per-slot 双 pipe 生产-消费握手（FA 同构 flash_attn_block_vec_nd.h:92-93,728-746）：
    // MTE2(加载) ↔ V(软max消费) 严格交替，避免跨 S2 轮次槽复用读到上一轮掩码（旧深度1队列的进出队语义）
    static constexpr uint32_t UB_IN_MASK_EVENT0 = 6U;
    static constexpr uint32_t UB_IN_MASK_EVENT1 = 7U;
    static constexpr uint32_t UB_BRDCST_MAX_EVENT = 8U; // maxBrdcst 1K（V/MTE3）

    // ================== 静态布局常量（表 1，数值 = 现状容量；UB 合计 ≈245.25K ≤ 248K）==================
    static constexpr uint32_t BUFFER_SIZE_BYTE_32K = 32768;
    static constexpr uint32_t UB_MM2_RES_BUFCNT = 2U;
    static constexpr uint32_t UB_MM2_RES_BUF_BYTES = mBaseSize / CV_RATIO * dVBaseSize * sizeof(T);
    static constexpr uint32_t UB_MM1_RES_BUFCNT = 2U;
    static constexpr uint32_t UB_MM1_RES_BUF_BYTES = mBaseSize / CV_RATIO * s2BaseSize * sizeof(T);
    static constexpr uint32_t UB_VEC2_RES_BUF_BYTES = mBaseSize / CV_RATIO * dVTemplateAlign64 * sizeof(T); // 64K
    static constexpr uint32_t UB_VEC1_RES_BUF_BYTES =
        (mBaseSize / CV_RATIO + 1) * s2BaseSize * sizeof(INPUT_T); // 8.25K
    static constexpr uint32_t UB_MASK_BUFCNT = DB;
    static constexpr uint32_t UB_MASK_BUF_BYTES = 4096U; // attenMaskInQue[2] 各 4096B
    static constexpr uint32_t UB_SOFTMAX_BUFCNT = PRELOAD_N + 1;
    static constexpr uint32_t UB_SOFTMAX_BUF_BYTES = 256U; // SOFTMAX VF max/sum/exp 按 256B 对齐
    static constexpr uint32_t UB_LSE_OUT_BUF_BYTES = mBaseSize / CV_RATIO * sizeof(float) * 8; // 1K（32×4×8）
    static constexpr uint32_t UB_BRDCST_BUF_BYTES = 1024U;                                     // max/sumBrdcst
    static constexpr uint32_t UB_TMP_BUF_BYTES = 512U;                                         // commonTBuf 512B

    // L1 P（A1）：vec1 写 P 的目标 = cube 的 L1 KVP 区（A1 72K 起、3×144K、每个任务 loop%3 槽）；
    // P 在槽内 rope 段偏移 = s2BaseSize*dVBaseSize（元素，与 cube CopyKeyAndRopeTile 的 dSize*dstStride 一致，表 2 注
    // 2）
    static constexpr uint32_t L1_P_BUFCNT = 3U;
    static constexpr uint32_t L1_P_BUF_BYTES = s2BaseSize * 576 * sizeof(INPUT_T);   // 144K
    static constexpr uint32_t L1_Q_PREFIX_BYTES = mBaseSize * 576 * sizeof(INPUT_T); // 72K（cube Q 区，跳过）

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

    // ub/l1 静态 tensor（InitBuffers 内按表 1 布局建立）
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
    }

    __aicore__ inline void ProcessVec1(FlashMlaWithKvcacheRunInfoX runInfo)
    {
        uint32_t mm1ResUbBufId = runInfo.loop % UB_MM1_RES_BUFCNT;
        uint32_t pL1BufId = runInfo.loop % L1_P_BUFCNT;
        uint32_t c1v1CrossCoreSyncIdx = CROSSCORE_BMM1_0 + mm1ResUbBufId;
        uint32_t v1c2CrossCoreSyncIdx = CROSSCORE_L1P_0 + pL1BufId;
        LocalTensor<INPUT_T> pL1Tensor = l1PBuffers_[pL1BufId * L1_P_BUF_BYTES].template ReinterpretCast<INPUT_T>();
        auto mm1ResUbTensor = ubMm1ResBuffers_[mm1ResUbBufId * UB_MM1_RES_BUF_BYTES].template ReinterpretCast<T>();

        // 首轮 softmax 折叠：统一走 Update VF 族（ProcessVec1Vf updateFlag=true），由
        // ResetSoftmaxBuffer 把 mloop 槽 sum/max 置 0/-inf 替代旧 isFirstS2Loop 区分
        // （对齐 flash_attn 5520571a4：消除首次 softmax VF 分支）
        if (unlikely(runInfo.isFirstS2Loop)) {
            ResetSoftmaxBuffer(runInfo.mloop % UB_SOFTMAX_BUFCNT);
            AscendC::PipeBarrier<PIPE_V>();
        }

        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(c1v1CrossCoreSyncIdx);
        if (unlikely(runInfo.actVecMSize == 0)) {
            CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(c1v1CrossCoreSyncIdx); // 反堵 c1v1：AIC 可覆写本槽
            CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(v1c2CrossCoreSyncIdx); // 反堵 v1c2：本 AIV 无 P 行
            return;
        }

        ProcessVec1Nd(pL1Tensor, mm1ResUbTensor, runInfo);

        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(c1v1CrossCoreSyncIdx);    // C1 收到后可启动 FIXPIPE 写
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(v1c2CrossCoreSyncIdx); // P 已写 L1，C2 可读
        Vec1PostProcess(runInfo);
    }

    __aicore__ inline bool IsInitAttentionOutGm()
    {
        // TND、NTD场景且不存在无效行,不需要初始化
        if constexpr (layout == LayOutTypeEnum::LAYOUT_TND || layout == LayOutTypeEnum::LAYOUT_NTD ||
                      layout == LayOutTypeEnum::LAYOUT_NTD_TND) {
            /*
             * tiling中提前算好了是否可能出现无效行, 正常从tiling中提取这个标记位(constInfo.isExistRowInvalid),
             * 对于FD场景, 有可能整体是没有无效行的,
             * 但当前FD处理的这部分s2是无效的。为规避潜在的风险，只要带mask(constInfo.isExistRowInvalid)
             * 就认为可能存在无效行
             */
            bool isExistRowInvalid = FLASH_DECODE ? HAS_MASK : constInfo.isExistRowInvalid;
            if (!isExistRowInvalid) {
                return false;
            }
        }
        return true;
    }

    // FA 协同清零（flash_attn_block_vec_nd.h:298-326 + init_output.h:39-81）：pop-buffer 瞬时别名落在
    // bmm2 CV 槽 [0..64K)（表 1 注 5），安全依赖 c2v2 反向旗标门控；仅在 kernel Init 阶段执行一次，
    // EVENT_ID0/1 为 Mutex ID（PIPE_V/PIPE_MTE3），与后段 vec 核内同步时序不重叠（表 2 注 9）
    __aicore__ inline void ClearOutput()
    {
        if (IsInitAttentionOutGm()) {
            uint32_t vecCoreNum = 2 * constInfo.coreNum;
            int64_t tSize = constInfo.bSize * constInfo.s1Size;
            if constexpr (layout == LayOutTypeEnum::LAYOUT_TND || layout == LayOutTypeEnum::LAYOUT_NTD ||
                          layout == LayOutTypeEnum::LAYOUT_NTD_TND) {
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
        // 用 TND→(N,T) 转置写出（flash_attn 范式）；bN2Offset = n2Idx * gSize * t1Size = head 块 GM 起址。
        if constexpr (layout == LayOutTypeEnum::LAYOUT_NTD) {
            uint32_t prefixBS1 = seqLensTool_.qActSeqLensParser.GetTBase(runInfo.bIdx);
            uint32_t s1Size = seqLensTool_.qActSeqLensParser.GetActualSeqLength(runInfo.bIdx);
            uint64_t bN2Offset = runInfo.n2Idx * constInfo.n2Size * constInfo.gSize * constInfo.t1Size;
            DataCopySoftmaxLseNTDArch35<T, ConstInfoNoQuant>(softmaxLseGm, lseUb, bN2Offset, vecMIdx,
                                                             runInfo.actVecMSize, constInfo, s1Size);
        } else if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
            uint32_t prefixBS1 = seqLensTool_.qActSeqLensParser.GetTBase(runInfo.bIdx);
            uint64_t bN2Offset = runInfo.n2Idx * constInfo.n2Size * constInfo.gSize * constInfo.t1Size;
            DataCopySoftmaxLseTNDtoNTArch35<T, ConstInfoNoQuant>(softmaxLseGm, lseUb, bN2Offset, vecMIdx,
                                                                 runInfo.actVecMSize, prefixBS1, constInfo);
        } else if constexpr (layout == LayOutTypeEnum::LAYOUT_BSH) {
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

    __aicore__ inline void ProcessVec1Nd(LocalTensor<INPUT_T> &pL1Tensor, LocalTensor<T> &mm1ResUbTensor,
                                         FlashMlaWithKvcacheRunInfoX runInfo)
    {
        LocalTensor<pseShiftType> pseUb;
        LocalTensor<uint8_t> dropMaskUb;
        float slopes = 0.0f;
        float posShift = 0.0f;
        uint32_t pseStride = 0;
        float descaleQK = 1.0;
        float deSCaleKValue = 1.0;

        LocalTensor<uint8_t> attenMaskUb;
        LocalTensor<uint8_t> attenMaskUbPre;
        const uint32_t maskBufId = runInfo.loop % DB;
        if constexpr (hasAtten) {
            attenMaskUb = ubMaskBuffers_[maskBufId * UB_MASK_BUF_BYTES];
            // [flash_attn 同款协议] 两槽生产-消费锁全部在 AttenMaskCopyIn 内部，调用点不再套锁
            AttenMaskCopyIn(attenMaskUb, 0, runInfo.actVecMSize, runInfo); // 全量拷贝（含 pre 槽与 MergeMask）
        }

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
        LocalTensor<T> mmRes = mm1ResUbTensor;
        Mutex::Lock<PIPE_V>(UB_OUT_VEC1_RES_EVENT0);
        LocalTensor<INPUT_T> stage1CastTensor = ubVec1ResBuffers_.template ReinterpretCast<INPUT_T>();
        // 统一走 Update VF 族（updateFlag=true）：首轮 sum/max 由 ProcessVec1 的
        // ResetSoftmaxBuffer 置 0/-inf，消除 isFirstS2Loop 分支（对齐 flash_attn 5520571a4）
        if (likely(runInfo.actSingleLoopS2Size == 128)) {
            ProcessVec1Vf<T, INPUT_T, pseShiftType, true, mBaseSize, s2BaseSize, EQ_128, hasAtten, PSE_MODE, HAS_DROP,
                          false, false>(stage1CastTensor, nullptr, sumUb, maxUb, mmRes, expUb, sumUb, maxUb,
                                        attenMaskUb, pseUb, dropMaskUb, apiTmpBuffer, pScaleUb, runInfo.actVecMSize,
                                        runInfo.actSingleLoopS2Size, pseStride, slopes, posShift,
                                        constInfo.scaleValue, // constInfo.scaleValue 已是 T float类型
                                        descaleQK, negativeFloatScalar, 0.0F, queryScaleUb, deSCaleKValue);
        } else if (runInfo.actSingleLoopS2Size <= 64) {
            ProcessVec1Vf<T, INPUT_T, pseShiftType, true, mBaseSize, s2BaseSize, GT_0_AND_LTE_64, hasAtten, PSE_MODE,
                          HAS_DROP, false, false>(
                stage1CastTensor, nullptr, sumUb, maxUb, mmRes, expUb, sumUb, maxUb, attenMaskUb, pseUb, dropMaskUb,
                apiTmpBuffer, pScaleUb, runInfo.actVecMSize, runInfo.actSingleLoopS2Size, pseStride, slopes, posShift,
                constInfo.scaleValue, descaleQK, negativeFloatScalar, 0.0F, queryScaleUb, deSCaleKValue);
        } else if (runInfo.actSingleLoopS2Size < 128) {
            ProcessVec1Vf<T, INPUT_T, pseShiftType, true, mBaseSize, s2BaseSize, GT_64_AND_LTE_128, hasAtten, PSE_MODE,
                          HAS_DROP, false, false>(
                stage1CastTensor, nullptr, sumUb, maxUb, mmRes, expUb, sumUb, maxUb, attenMaskUb, pseUb, dropMaskUb,
                apiTmpBuffer, pScaleUb, runInfo.actVecMSize, runInfo.actSingleLoopS2Size, pseStride, slopes, posShift,
                constInfo.scaleValue, descaleQK, negativeFloatScalar, 0.0F, queryScaleUb, deSCaleKValue);
        } else {
            if constexpr (s2BaseSize == 256) {
                ProcessVec1Vf<T, INPUT_T, pseShiftType, true, mBaseSize, s2BaseSize, GT_128_AND_LTE_256, hasAtten,
                              PSE_MODE, HAS_DROP>(stage1CastTensor, nullptr, sumUb, maxUb, mmRes, expUb, sumUb, maxUb,
                                                  attenMaskUb, pseUb, dropMaskUb, apiTmpBuffer, expUb,
                                                  runInfo.actVecMSize, runInfo.actSingleLoopS2Size, pseStride, slopes,
                                                  posShift, constInfo.scaleValue, descaleQK, negativeFloatScalar, 0.0F);
            }
        }
        Mutex::Unlock<PIPE_V>(UB_OUT_VEC1_RES_EVENT0);

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

        if constexpr (hasAtten) {
            // per-slot 消费端释放（V 管道到达此处即两槽读完成，允许下一轮 MTE2 覆写）
            // [flash_attn 同款协议] 与 AttenMaskCopyIn 内部生产端 MTE2 锁配对，缺此块会死锁
            Mutex::Lock<PIPE_V>(UB_IN_MASK_EVENT0 + (maskBufId ^ 1U));
            Mutex::Unlock<PIPE_V>(UB_IN_MASK_EVENT0 + (maskBufId ^ 1U));
            Mutex::Lock<PIPE_V>(UB_IN_MASK_EVENT0 + maskBufId);
            Mutex::Unlock<PIPE_V>(UB_IN_MASK_EVENT0 + maskBufId);
        }
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
        // 置 0/-inf，Update 恒执行（对齐 flash_attn 5520571a4：消除 isFirstS2Loop 区分）
        UpdateExpSumAndExpMax<T>(sumUb, maxUb, expUb, sumUb, maxUb, vec1ApiTmpBuf_, runInfo.actVecMSize);

        if (unlikely(runInfo.isLastS2Loop)) {
            SoftmaxDataCopyOut(runInfo, sumUb, maxUb);
        }
    }

    __aicore__ inline void ProcessVec2(FlashMlaWithKvcacheRunInfoX runInfo)
    {
        uint32_t mm2ResUbBufId = runInfo.loop % UB_MM2_RES_BUFCNT;
        uint32_t c2v2CrossCoreSyncIdx = CROSSCORE_BMM2_0 + mm2ResUbBufId;
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(c2v2CrossCoreSyncIdx);
        ProcessVec2OnUb(mm2ResUbBufId, runInfo);
        return;
    }

    __aicore__ inline void ProcessVec2OnUb(uint32_t mm2ResUbBufId, FlashMlaWithKvcacheRunInfoX runInfo)
    {
        uint32_t c2v2CrossCoreSyncIdx = CROSSCORE_BMM2_0 + mm2ResUbBufId;
        if (unlikely(runInfo.actVecMSize == 0)) {
            CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(c2v2CrossCoreSyncIdx); // 反堵 c2v2：本 AIV 无行
            return;
        }

        int64_t vec2CalcSize = runInfo.actVecMSize * dVTemplateAlign64;
        LocalTensor<T> vec2ResUb = ubVec2Res_.template ReinterpretCast<T>();
        LocalTensor<T> mmRes = ubMm2ResBuffers_[mm2ResUbBufId * UB_MM2_RES_BUF_BYTES].template ReinterpretCast<T>();
        Mutex::Lock<PIPE_V>(UB_OUT_VEC2_RES_EVENT0); // 等待上一轮 MTE3 拷贝完成（旧 WaitFlag<MTE3_V> 语义）
        if (unlikely(runInfo.isFirstS2Loop)) {
            DataCopy(vec2ResUb, mmRes, vec2CalcSize);
        } else {
            LocalTensor<float> expUb =
                softmaxExpBuf_[(runInfo.loop % UB_SOFTMAX_BUFCNT) * (UB_SOFTMAX_BUF_BYTES / sizeof(float))];
            LocalTensor<T> pScaleUb;

            if (likely(!runInfo.isLastS2Loop)) {
                FlashUpdateNew<T, INPUT_T, OUTPUT_T, dVTemplateAlign64, false, false>(
                    vec2ResUb, mmRes, vec2ResUb, expUb, pScaleUb, runInfo.actVecMSize, dVTemplateAlign64, 1.0, 1.0);
            } else {
                LocalTensor<float> sumUb =
                    softmaxSumBuf_[(runInfo.mloop % UB_SOFTMAX_BUFCNT) * (UB_SOFTMAX_BUF_BYTES / sizeof(float))];
                FlashUpdateLastNew<T, INPUT_T, OUTPUT_T, dVTemplateAlign64, false, false>(
                    vec2ResUb, mmRes, vec2ResUb, expUb, pScaleUb, sumUb, runInfo.actVecMSize, dVTemplateAlign64, 1.0,
                    1.0);
            }
        }
        Mutex::Unlock<PIPE_V>(UB_OUT_VEC2_RES_EVENT0);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(c2v2CrossCoreSyncIdx); // mmRes 之后不能使用，C2 可覆写
        if (unlikely(runInfo.isLastS2Loop)) {
            if (unlikely(runInfo.isFirstS2Loop)) {
                Mutex::Lock<PIPE_V>(UB_OUT_VEC2_RES_EVENT0);
                LocalTensor<float> sumUb =
                    softmaxSumBuf_[(runInfo.mloop % UB_SOFTMAX_BUFCNT) * (UB_SOFTMAX_BUF_BYTES / sizeof(float))];
                LastDivNew<T, INPUT_T, OUTPUT_T, dVTemplateAlign64, false>(
                    vec2ResUb, vec2ResUb, sumUb, runInfo.actVecMSize, (uint16_t)dVTemplateAlign64, 0.0F);
                Mutex::Unlock<PIPE_V>(UB_OUT_VEC2_RES_EVENT0);
            }
            CopyOutAttentionOut(runInfo, vec2ResUb, 0, runInfo.actVecMSize);
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
        if constexpr (layout == LayOutTypeEnum::LAYOUT_BSH || layout == LayOutTypeEnum::LAYOUT_TND) {
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
        GmCoord gmCoord{.bIdx = info.bIdx,
                        .n2Idx = info.n2Idx,
                        .gS1Idx = (info.gS1Idx + info.vecMbaseIdx + vecMIdx),
                        .dIdx = 0,
                        .gS1DealSize = dealRowCount,
                        .dDealSize = (uint32_t)constInfo.dSizeV};
        FaUbTensor<OUTPUT_T, false> ubTensor{
            .tensor = attenOutUb, .rowCount = dealRowCount, .colCount = (uint32_t)(dVTemplateAlign64)};
        CopyAttentionOut(ubTensor, gmCoord);
    }

    __aicore__ inline void CopyAttentionOut(FaUbTensor<OUTPUT_T, false> &ubTensor, GmCoord &gmCoord)
    {
        if (constInfo.outputLayout == FLASH_MLA_WITH_KVCACHE_LAYOUT::BSH) {
            constexpr GmFormat OUT_FORMAT = GmFormat::BSNGD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT, uint32_t> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm;
            outGmTensor.offsetCalculator.Init(constInfo.bSize, constInfo.n2Size, constInfo.gSize, constInfo.s1Size,
                                              constInfo.dSizeV, seqLensTool_.actualSeqLengthsGmQ,
                                              constInfo.actualSeqLenSize);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<layout>()> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        } else if (constInfo.outputLayout == FLASH_MLA_WITH_KVCACHE_LAYOUT::BNSD) {
            constexpr GmFormat OUT_FORMAT = GmFormat::BNGSD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT, uint32_t> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm;
            outGmTensor.offsetCalculator.Init(constInfo.bSize, constInfo.n2Size, constInfo.gSize, constInfo.s1Size,
                                              constInfo.dSizeV, seqLensTool_.actualSeqLengthsGmQ,
                                              constInfo.actualSeqLenSize);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<layout>()> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        } else if (constInfo.outputLayout == FLASH_MLA_WITH_KVCACHE_LAYOUT::TND) {
            constexpr GmFormat OUT_FORMAT = GmFormat::TNGD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT, uint32_t, true> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm;
            outGmTensor.offsetCalculator.Init(constInfo.n2Size, constInfo.gSize, constInfo.dSizeV,
                                              seqLensTool_.actualSeqLengthsGmQ, constInfo.actualSeqLenSize);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<layout>()> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        } else if (constInfo.outputLayout == FLASH_MLA_WITH_KVCACHE_LAYOUT::NTD) {
            constexpr GmFormat OUT_FORMAT = GmFormat::NGTD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT, uint32_t, true> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm;
            outGmTensor.offsetCalculator.Init(constInfo.n2Size, constInfo.gSize, constInfo.dSizeV,
                                              seqLensTool_.actualSeqLengthsGmQ, constInfo.actualSeqLenSize);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<layout>()> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        } else if (constInfo.outputLayout == FLASH_MLA_WITH_KVCACHE_LAYOUT::NBSD) {
            constexpr GmFormat OUT_FORMAT = GmFormat::NGBSD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT, uint32_t> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm;
            outGmTensor.offsetCalculator.Init(constInfo.bSize, constInfo.n2Size, constInfo.gSize, constInfo.s1Size,
                                              constInfo.dSizeV, seqLensTool_.actualSeqLengthsGmQ,
                                              constInfo.actualSeqLenSize);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<layout>()> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        }
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
        // l1P 三缓冲：A1 72K 起（跳过 cube Q 区），槽内 P 在 rope 段（表 2 注 2）
        l1PBuffers_ = LocalTensor<uint8_t>(TPosition::A1, L1_Q_PREFIX_BYTES, L1_P_BUFCNT * L1_P_BUF_BYTES);

        /*--------------------------------------------UB--------------------------------------------*/
        struct UbLayout {
            uint8_t mm2ResBuffers[UB_MM2_RES_BUFCNT]
                                 [UB_MM2_RES_BUF_BYTES]; // 2*64K=128K @0，CV通信BUF（与 cube 同一偏移）
            uint8_t mm1ResBuffers[UB_MM1_RES_BUFCNT][UB_MM1_RES_BUF_BYTES]; // 2*16K=32K @128K，CV通信BUF
            uint8_t vec2Res[UB_VEC2_RES_BUF_BYTES];                         // 64K @160K，vec2/输出BUF（单槽）
            uint8_t vec1Res[UB_VEC1_RES_BUF_BYTES];                 // 8.25K @224K，stage1/softmax结果（单槽）
            uint8_t maskBuffers[UB_MASK_BUFCNT][UB_MASK_BUF_BYTES]; // 2*4K=8K，输入BUF: MASK拷入
            uint8_t softmaxSumBuf[UB_SOFTMAX_BUFCNT][UB_SOFTMAX_BUF_BYTES]; // 2*256B，sum常驻BUF
            uint8_t softmaxMaxBuf[UB_SOFTMAX_BUFCNT][UB_SOFTMAX_BUF_BYTES]; // 2*256B，max常驻BUF
            uint8_t softmaxExpBuf[UB_SOFTMAX_BUFCNT][UB_SOFTMAX_BUF_BYTES]; // 2*256B，exp常驻BUF
            uint8_t lseOutBuf[UB_LSE_OUT_BUF_BYTES];                        // 1K，LSE输出
            uint8_t sumBrdcstBuf[UB_BRDCST_BUF_BYTES];                      // 1K，FD sum [32,8]
            uint8_t maxBrdcstBuf[UB_BRDCST_BUF_BYTES];                      // 1K，FD max [32,8]
            uint8_t tmpBuf[UB_TMP_BUF_BYTES];                               // 0.5K，softmax 中间结果缓存
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
        ubMaxBrdcstBuf_ = LocalTensor<uint8_t>(TPosition::VECIN, OFFSET_OF_MEMBER(UbLayout, maxBrdcstBuf),
                                               SIZE_OF_MEMBER(UbLayout, maxBrdcstBuf));
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
        // AIV 预置 c1v1+c2v2 四旗标（表 2 注 8）：AIC 首个 bmm1/bmm2 Wait 立即通过；
        // v1c2 三旗标随任务 ping-pong 无需预置（首轮 bmm2 的 Wait 有 AIV 首个 P 写保证）
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CROSSCORE_BMM2_0);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CROSSCORE_BMM2_1);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CROSSCORE_BMM1_0);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CROSSCORE_BMM1_1);
    }

    __aicore__ inline void UnInitCrossCoreSync() {}

    __aicore__ inline void AllocEventID()
    {
        // 核内同步已全部静态化为 Mutex（表 2），无动态事件可分配
    }

    __aicore__ inline void FreeEventID() {}

    __aicore__ inline void AttenMaskCopyIn(LocalTensor<uint8_t> attenMaskUb, uint32_t vecMIdx, uint32_t mDealSize,
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
        maskInfo.nBaseSize = s2BaseSize;
        maskInfo.preToken = constInfo.preTokens;
        maskInfo.nextToken = constInfo.nextTokens;
        maskInfo.sparseMode = static_cast<SparseMode>(constInfo.sparseMode);
        maskInfo.batchIdx = (constInfo.attenMaskBatch == 1) ? 0 : runInfo.bIdx;
        maskInfo.attenMaskBatchStride = constInfo.attenMaskS1Size * constInfo.attenMaskS2Size;
        maskInfo.attenMaskS1Stride = constInfo.attenMaskS2Size;
        maskInfo.attenMaskDstStride = (s2BaseSize - AttentionCommon::Align(maskInfo.s2dealNum, 32U)) / 32;
        maskInfo.maskValue = negativeIntScalar;
        maskInfo.s1LeftPaddingSize = runInfo.qPaddingBeginOffset;
        maskInfo.s2LeftPaddingSize = runInfo.kvPaddingBeginOffset;
        maskInfo.maskFormat = MASK_LAYOUT;
        maskInfo.attenMaskType = MASK_BOOL; // compatible with int8/uint8

        bool IsSkipMask = IsSkipAttentionmask(maskInfo);
        bool IsSkipMaskForPre = IsSkipAttentionmaskForPre(maskInfo);
        // [flash_attn 同款协议] 锁全部内置于本函数（各槽 PIPE_V/MTE2 握手在拷贝内完成）
        const uint32_t maskBufId = runInfo.loop % DB;
        if (IsSkipMask && IsSkipMaskForPre) {
            Mutex::Lock<PIPE_V>(UB_IN_MASK_EVENT0 + maskBufId);
            Duplicate(attenMaskUb, static_cast<uint8_t>(0U), maskInfo.gs1dealNum * s2BaseSize);
            Mutex::Unlock<PIPE_V>(UB_IN_MASK_EVENT0 + maskBufId);
            return;
        }

        if (!IsSkipMask) {
            AttentionmaskCopyIn<uint8_t, MASK_LAYOUT, true, s2BaseSize>(attenMaskUb, attenMaskGmInt, maskInfo, false,
                                                                        UB_IN_MASK_EVENT0 + maskBufId);
        } else {
            Mutex::Lock<PIPE_V>(UB_IN_MASK_EVENT0 + maskBufId);
            Duplicate(attenMaskUb, static_cast<uint8_t>(0U), maskInfo.gs1dealNum * s2BaseSize);
            Mutex::Unlock<PIPE_V>(UB_IN_MASK_EVENT0 + maskBufId);
        }

        if (!IsSkipMaskForPre) {
            const uint32_t preBufId = maskBufId ^ 1U;
            LocalTensor<uint8_t> attenMaskUbPre = ubMaskBuffers_[preBufId * UB_MASK_BUF_BYTES];
            AttentionmaskCopyIn<uint8_t, MASK_LAYOUT, true, s2BaseSize>(attenMaskUbPre, attenMaskGmInt, maskInfo, true,
                                                                        UB_IN_MASK_EVENT0 + preBufId);
            Mutex::Lock<PIPE_V>(UB_IN_MASK_EVENT0 + preBufId);
            MergeMask(attenMaskUb, attenMaskUbPre, maskInfo.gs1dealNum, s2BaseSize);
            Mutex::Unlock<PIPE_V>(UB_IN_MASK_EVENT0 + preBufId);
        }
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

#endif // FLASH_MLA_WITH_KVCACHE_BLOCK_VEC_NOQUANT_MLA_H_
