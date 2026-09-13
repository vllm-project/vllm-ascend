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
 * \file sparse_flash_mla_csa_block_vector_arch35.h
 * \brief
 */
#ifndef SPARSE_FLASH_MLA_CSA_BLOCK_VECTOR_ARCH35_H
#define SPARSE_FLASH_MLA_CSA_BLOCK_VECTOR_ARCH35_H

#include "util_regbase.h"
#include "sparse_flash_mla_common_arch35.h"
#include "kernel_operator_list_tensor_intf.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"

using AscendC::Reg::StoreDist;

#include "common/flash_decode.h"

#if __has_include("../../common/op_kernel/arch35/vf/vf_flash_decode_arch35.h")
#include "../../common/op_kernel/arch35/vf/vf_flash_decode_arch35.h"
#else
#include "../common/arch35/vf/vf_flash_decode_arch35.h"
#endif

#if __has_include("../../common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz_sfa.h")
#include "../../common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz_sfa.h"
#else
#include "../../common/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz_sfa.h"
#endif

#if __has_include("../../common/op_kernel/arch35/vf/vf_flashupdate_new.h")
#include "../../common/op_kernel/arch35/vf/vf_flashupdate_new.h"
#else
#include "../../common/arch35/vf/vf_flashupdate_new.h"
#endif

#if __has_include("../../common/op_kernel/buffers_policy.h")
#include "../../common/op_kernel/buffers_policy.h"
#else
#include "../common/buffers_policy.h"
#endif
#if __has_include("../../common/op_kernel/attn_buffer_manager.h")
#include "../../common/op_kernel/attn_buffer_manager.h"
#else
#include "../common/attn_buffer_manager.h"
#endif
#if __has_include("../../common/op_kernel/attn_buffer.h")
#include "../../common/op_kernel/attn_buffer.h"
#else
#include "../common/attn_buffer.h"
#endif
#if __has_include("../../common/op_kernel/init_output.h")
#include "../../common/op_kernel/init_output.h"
#else
#include "../common/init_output.h"
#endif

using namespace AscendC;
using namespace FaVectorApi;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;
using namespace matmul;
using namespace fa_base_matmul;
using AttentionCommon::FdRunInfo;

namespace SMLAKernel {

// 统一窗口公式
struct PhyAddrValidInfo {
    static constexpr int64_t BIAS_UNBOUND = 0x7FFFFFFF; // INT32_MAX
    int64_t oriLeftBias = BIAS_UNBOUND;
    int64_t oriRightBias = BIAS_UNBOUND;
    int32_t oriS2Act = 0;
    bool oriTopkMode = false; // oriMaskMode==0: 走topkLength语义(保持原行为)
    bool cmpTopkMode = true;  // cmpMaskMode==0: 走topkLength语义(保持原行为)
    int64_t cmpBase = 0;      // restoredSize - actualS1Size + 1
};

TEMPLATES_DEF
class CSABlockVec {
public:
    // BUFFER的字节数
    static constexpr uint32_t BUFFER_SIZE_BYTE_32B = 32;
    /* =================编译期常量的基本块信息================= */
    static constexpr uint32_t s1BaseSize = 64;
    static constexpr uint32_t s2BaseSize = 128;
    static constexpr uint32_t vec1Srcstride = (s1BaseSize >> 1) + 1;
    static constexpr uint32_t dVTemplateType = 512;
    static constexpr uint32_t dTemplateAlign64 = Align64Func(dVTemplateType);
    static constexpr float R0 = 1.0f;
    static constexpr uint32_t initOutputEventId =
        INNERCORE_INITOUT_MTE3_V; // attenOut和lse，刷无效行会用到剩余ub，需要加同步
    // Sparse KV搬入/拷出块大小：每次搬入8行，每16行做一次拷出
    static constexpr int64_t KV_COPYIN_UNIT = 8;   // 每次搬入8行
    static constexpr int64_t KV_PROCESS_UNIT = 16; // 每次拷出16行

    // ==================== Functions ======================
    __aicore__ inline CSABlockVec(){};
    __aicore__ inline void InitVecBlock(__gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *cuSeqlensOriKv,
                                        __gm__ uint8_t *cuSeqlensCmpKv, __gm__ uint8_t *seqUsedOriKV,
                                        __gm__ uint8_t *seqUsedCmpKV, __gm__ uint8_t *cmpResidualKV)
    {
        if ASCEND_IS_AIV {
            if (cuSeqlensQ != nullptr) {
                cuSeqlensQGm.SetGlobalBuffer((__gm__ int32_t *)cuSeqlensQ);
            }
            if (cuSeqlensOriKv != nullptr) {
                cuSeqlensOriKvGm.SetGlobalBuffer((__gm__ int32_t *)cuSeqlensOriKv);
            }
            if (cuSeqlensCmpKv != nullptr) {
                cuSeqlensCmpKvGm.SetGlobalBuffer((__gm__ int32_t *)cuSeqlensCmpKv);
            }
            if (seqUsedOriKV != nullptr) {
                actualSeqLengthsKVGm.SetGlobalBuffer((__gm__ int32_t *)seqUsedOriKV);
            }
            if (seqUsedCmpKV != nullptr) {
                actualSeqLengthsCmpKVGm.SetGlobalBuffer((__gm__ int32_t *)seqUsedCmpKV);
            }
            if (cmpResidualKV != nullptr) {
                cmpResidualKVGm.SetGlobalBuffer((__gm__ int32_t *)cmpResidualKV);
            }
            this->GetExtremeValue(this->negativeFloatScalar);
        }
    }

    // 初始化LocalTensor
    __aicore__ inline void InitLocalBuffer(ConstInfo &constInfo, uint32_t ubBaseAddr);
    // 初始化attentionOutGM
    __aicore__ inline void CleanOutput(__gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse, ConstInfo &constInfo);
    __aicore__ inline void InitGlobalBuffer(__gm__ uint8_t *oriKV, __gm__ uint8_t *cmpKV,
                                            __gm__ uint8_t *oriSparseIndices, __gm__ uint8_t *cmpSparseIndices,
                                            __gm__ uint8_t *oriBlockTable, __gm__ uint8_t *cmpBlockTable,
                                            __gm__ uint8_t *sequsedQ, __gm__ uint8_t *sinks,
                                            __gm__ uint8_t *sequsedOriKv, __gm__ uint8_t *sequsedCmpKv,
                                            __gm__ uint8_t *cmpResidualKv);
    __aicore__ inline void InitOutputSingleCore(ConstInfo &constInfo);
    __aicore__ inline void ProcessVec0(Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
                                       const RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void ProcessVec1(StaticBuffer<Q_T> &outputBuf, StaticBuffer<T> &bmm1ResBuf, RunInfo &runInfo,
                                       ConstInfo &constInfo);
    __aicore__ inline void InitS2SplitStaging(Buffer<BufferType::GM, SyncType::NO_SYNC> &fdStaging)
    {
        fdStagingBase = fdStaging.template GetTensor<uint8_t>().GetPhyAddr(0);
        stagingOutGm = fdStaging.template GetTensor<float>();
    }
    __aicore__ inline void InitS2SplitStaging(Buffer<BufferType::GM, SyncType::NO_SYNC> &intraCoreCombine,
                                              Buffer<BufferType::GM, SyncType::NO_SYNC> &crossCoreCombine)
    {
        intraCoreCombineBase = intraCoreCombine.template GetTensor<uint8_t>().GetPhyAddr(0);
        intraCoreCombineGm = intraCoreCombine.template GetTensor<float>();
        crossCoreCombineBase = crossCoreCombine.template GetTensor<uint8_t>().GetPhyAddr(0);
        crossCoreCombineGm = crossCoreCombine.template GetTensor<float>();
        fdStagingBase = crossCoreCombineBase;
        stagingOutGm = crossCoreCombineGm;
    }
    __aicore__ inline void InitFDBuffers(FdRunInfo &fdRunInfo);
    __aicore__ inline void ProcessFlashDecode(FdRunInfo &fdRunInfo, ConstInfo &constInfo);
    __aicore__ inline void ProcessVec2(StaticBuffer<T> &bmm2ResBuf, RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void GetKVPhyAddr(uint32_t hasLoad, uint32_t bN2StartIdx, uint32_t bN2EndIdx,
                                        uint32_t gS1StartIdx, uint32_t nextGs1Idx, bool hasActualSeqQlen,
                                        bool hasCuSeqlensQ, bool hasActualSeqOriKvlen, bool hasCuSeqlensOriKv,
                                        GlobalTensor<int32_t> actualSeqOriKvlenGm,
                                        GlobalTensor<int32_t> cuSeqlensOriKvGm, GlobalTensor<int32_t> oriTopkLengthGm,
                                        bool hasActualSeqCmpKvlen, bool hasCuSeqlensCmpKv,
                                        GlobalTensor<int32_t> actualSeqCmpKvlenGm,
                                        GlobalTensor<int32_t> cuSeqlensCmpKvGm, GlobalTensor<int32_t> cmpTopkLengthGm,
                                        GlobalTensor<int32_t> cmpResidualKvGm, GlobalTensor<int32_t> actualSeqQlenGm,
                                        GlobalTensor<int32_t> cuSeqlensQGm, __gm__ uint8_t *workspace,
                                        ConstInfo &constInfo);
    __aicore__ inline void FreeEvent(ConstInfo &constInfo);

private:
    template <bool UPDATE>
    __aicore__ inline void ComputeVec1Softmax(LocalTensor<Q_T> &stage1CastTensor, LocalTensor<T> &mmRes,
                                              LocalTensor<float> &sumUb, LocalTensor<float> &maxUb,
                                              LocalTensor<T> &apiTmpBuffer, RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void InitVec1SoftmaxFromSinks(LocalTensor<float> &sumUb, LocalTensor<float> &maxUb,
                                                    RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void CopyVec1ResultToL1(StaticBuffer<Q_T> &outputBuf, LocalTensor<Q_T> &stage1CastTensor,
                                              RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void StageCrossCoreVec1Lse(LocalTensor<float> &maxUb, LocalTensor<float> &sumUb, RunInfo &runInfo,
                                                 ConstInfo &constInfo);
    __aicore__ inline void StageBatchConsistencyVec1Lse(LocalTensor<float> &maxUb, LocalTensor<float> &sumUb,
                                                        RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void StageLegacyVec1Lse(LocalTensor<float> &maxUb, LocalTensor<float> &sumUb, RunInfo &runInfo,
                                              ConstInfo &constInfo);
    __aicore__ inline void CopyOutVec1Lse(LocalTensor<float> &maxUb, LocalTensor<float> &sumUb, RunInfo &runInfo,
                                          ConstInfo &constInfo);

    __aicore__ inline uint32_t GetStagingSlotNum(bool isInner = false) const
    {
        if constexpr (IS_BATCH_CONSISTENCY) {
            if (isInner) {
                if constexpr (IS_SPLIT_G) {
                    return GetBlockNum();
                }
                return GetBlockNum() << 1U;
            }
            if constexpr (IS_SPLIT_G) {
                return BATCH_CONSISTENCY_MAX_REDUCE_BLOCK_NUM * (GetBlockNum() >> 1U);
            }
            return BATCH_CONSISTENCY_MAX_REDUCE_BLOCK_NUM * GetBlockNum();
        }
        if constexpr (IS_SPLIT_G) {
            return AttentionCommon::FD_MAX_S2_SPLIT_NUM * (GetBlockNum() >> 1U);
        } else {
            return AttentionCommon::FD_MAX_S2_SPLIT_NUM * GetBlockNum();
        }
    }

    __aicore__ inline uint32_t GetIntraCoreWorkspaceIdx(const RunInfo &runInfo, const ConstInfo &constInfo) const
    {
        uint32_t coreIdx;
        if constexpr (IS_SPLIT_G) {
            coreIdx = static_cast<uint32_t>(constInfo.aivIdx >> 2U);
        } else {
            coreIdx = static_cast<uint32_t>(constInfo.aivIdx >> 1U);
        }
        return (coreIdx << 1U) + runInfo.multiCoreIdxMod2;
    }

    __aicore__ inline uint32_t GetCrossCoreWorkspaceIdx(const RunInfo &runInfo) const
    {
        return static_cast<uint32_t>(runInfo.firstFdDataWorkspaceIdx + runInfo.s2SplitIdx);
    }

    __aicore__ inline int64_t GetFaStagingMOffset(const RunInfo &runInfo, const ConstInfo &constInfo) const
    {
        int64_t stagingMOffset = (constInfo.subBlockIdx == 1) ? static_cast<int64_t>(runInfo.firstHalfMRealSize) : 0L;
        if constexpr (IS_SPLIT_G) {
            stagingMOffset += static_cast<int64_t>(runInfo.goIdx);
        }
        return stagingMOffset;
    }

    __aicore__ inline void ProcessSparseKv(Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
                                           const RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void CalSparseCalSize(const RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline int64_t GetkeyOffset(int64_t s2Idx, const RunInfo &runInfo, ConstInfo &constInfo);
    template <bool IS_FULL = false>
    __aicore__ inline void GetRealCmpS2Idx(int64_t *tokenData, int64_t s2IdxInBase, const RunInfo &runInfo,
                                           ConstInfo &constInfo);
    template <bool IS_FULL = false>
    __aicore__ inline void CopyInKvSparse(LocalTensor<KV_T> kvInUb, int64_t startRow, int64_t *tokenData,
                                          const RunInfo &runInfo, ConstInfo &constInfo);
    template <bool IS_FULL = false>
    __aicore__ inline void CopyIn8Block(LocalTensor<KV_T> kvInUb, int64_t startRow, int64_t &s2, const RunInfo &runInfo,
                                        ConstInfo &constInfo);
    __aicore__ inline void CopyToOutUb(LocalTensor<Q_T> kvNzUb, LocalTensor<KV_T> srcTensor, int64_t dealRow,
                                       ConstInfo &constInfo);
    __aicore__ inline void CopyOutKvUb2Gm(Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
                                          LocalTensor<Q_T> kvOutUb, int64_t dealRow, int64_t s2StartIdx,
                                          const RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void CopyInSingleKv(LocalTensor<KV_T> kvInUb, int64_t startRow, int64_t keyOffset,
                                          ConstInfo &constInfo);
    template <bool IS_FULL = false>
    __aicore__ inline void GetRealS2Addr(int64_t *tokenData, int64_t s2IdxInBase, const RunInfo &runInfo,
                                         ConstInfo &constInfo);
    __aicore__ inline void GetKVPhyAddrForKvType(
        uint32_t bN2StartIdx, uint32_t bN2EndIdx, uint32_t gS1StartIdx, uint32_t nextGs1Idx, bool hasActualSeqQlen,
        bool hasCuSeqlensQ, bool hasActualSeqKvlen, bool hasCuSeqlensKv, GlobalTensor<int32_t> actualSeqQlenGm,
        GlobalTensor<int32_t> cuSeqlensQGm, GlobalTensor<int32_t> actualSeqKvlenGm, GlobalTensor<int32_t> cuSeqlensKvGm,
        GlobalTensor<int32_t> topkLengthGm, GlobalTensor<int32_t> cmpResidualKvGm, ConstInfo &constInfo,
        GlobalTensor<int32_t> &blockTableGm, GlobalTensor<int32_t> &sparseIndicesGm, GlobalTensor<uint32_t> &phyAddrGm,
        uint32_t kvStride, uint32_t blockSize, uint32_t maxBlockNumPerBatch, uint32_t sparseBlockCount,
        uint32_t alignedSparseBlockCount, bool isOriKv);
    __aicore__ inline int32_t GetSeqLen(int32_t bIdx, bool hasActualSeq, bool hasCuSeqlens,
                                        GlobalTensor<int32_t> &actualSeqGm, GlobalTensor<int32_t> &cuSeqlensGm,
                                        int64_t defaultSize);
    __aicore__ inline PhyAddrValidInfo CalcPhyAddrValidInfo(bool isOriKv, int32_t actualS1Size, int32_t actualOriS2Size,
                                                            int32_t restoredSize, ConstInfo &constInfo);
    __aicore__ inline int32_t CalcCurValidS2(uint32_t bIdx, int32_t s1Idx, int32_t actualS1Size, bool isOriKv,
                                             GlobalTensor<int32_t> &cuSeqlensQGm, GlobalTensor<int32_t> &topkLengthGm,
                                             ConstInfo &constInfo, int32_t sparseBlockCount,
                                             const PhyAddrValidInfo &validInfo);
    __aicore__ inline void CopyPhyAddrToGm(LocalTensor<uint32_t> kvPhyAddrUb, int64_t bS1Idx, int64_t s1Idx,
                                           int64_t validS2, int64_t alignNum, GlobalTensor<uint32_t> &phyAddrGm,
                                           uint32_t alignedSparseBlockCount);
    __aicore__ inline void CopyPaTableToUb(LocalTensor<int32_t> blkTableUb, int64_t bIdx,
                                           GlobalTensor<int32_t> &blockTableGm, uint32_t maxBlockNumPerBatch);
    __aicore__ inline void CopySparseIdxToUb(LocalTensor<int32_t> sparseIdxUb, int64_t bS1Idx, int64_t s1Idx,
                                             int64_t validS2, GlobalTensor<int32_t> &sparseIndicesGm,
                                             uint32_t sparseBlockCount);
    /* VEC2_RES_T 表示bmm2ResUb当前的类型，VEC2_RES_T = Q_T那么不需要做Cast。另外，无效行场景当前默认需要做Cast */
    template <typename VEC2_RES_T>
    __aicore__ inline void Bmm2DataCopyOut(RunInfo &runInfo, ConstInfo &constInfo, LocalTensor<VEC2_RES_T> &vec2ResUb,
                                           int64_t vec2S1Idx, int64_t vec2CalcSize = 0);
    template <typename VEC2_RES_T>
    __aicore__ inline void CopyOutAttentionOut(RunInfo &runInfo, ConstInfo &constInfo,
                                               LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t vec2S1Idx,
                                               int64_t vec2CalcSize);
    __aicore__ inline void SoftmaxInitBuffer(uint32_t &ubAddr);
    __aicore__ inline void GetExtremeValue(T &negativeScalar);
    __aicore__ inline void InitSinksBuffer(ConstInfo &constInfo);
    __aicore__ inline void ReduceIntraBlockAndStage(RunInfo &runInfo, ConstInfo &constInfo, LocalTensor<T> &vec2ResUb,
                                                    LocalTensor<T> &partialTmpUb);

    GlobalTensor<OUTPUT_T> attentionOutGm;
    GlobalTensor<T> softmaxLseGm;
    GlobalTensor<KV_T> oriKVGm;
    GlobalTensor<KV_T> cmpKVGm;
    GlobalTensor<KV_T> keyGm;
    GlobalTensor<int32_t> cuSeqlensKvGm;
    GlobalTensor<int32_t> oriSparseIndicesGm;
    GlobalTensor<int32_t> cmpSparseIndicesGm;
    GlobalTensor<int32_t> sparseIndicesGm;
    GlobalTensor<int32_t> oriBlockTableGm;
    GlobalTensor<int32_t> cmpBlockTableGm;
    GlobalTensor<int32_t> blockTableGm;
    GlobalTensor<T> sinksGm;
    GlobalTensor<int32_t> cuSeqlensQGm;
    GlobalTensor<int32_t> cuSeqlensOriKvGm;
    GlobalTensor<int32_t> cuSeqlensCmpKvGm;
    GlobalTensor<int32_t> actualSeqLengthsKVGm;
    GlobalTensor<int32_t> actualSeqLengthsCmpKVGm;
    GlobalTensor<int32_t> cmpResidualKVGm;
    GlobalTensor<uint32_t> oriKvPhyAddrGm;
    GlobalTensor<uint32_t> cmpKvPhyAddrGm;

    StaticBuffer<T> commonUb;
    StaticBuffer<T> sinksUb;
    StaticBuffer<Q_T> stage1OutBufs[2];
    StaticBuffer<T> stage2OutBufs;
    StaticBuffer<Q_T> stage0OutBufs[2];
    StaticBuffer<float> softmaxMaxBufs[2];
    StaticBuffer<float> softmaxSumBufs[2];
    StaticBuffer<float> softmaxFinalMaxBufs[2];
    StaticBuffer<float> softmaxFinalSumBufs[2];
    StaticBuffer<T> softmaxExpBufs[2];
    StaticBuffer<float> batchReduceTmpUb;
    StaticBuffer<float> outLseUbs[2];
    TBuf<> vselrIndexesBuf[2];
    AttentionCommon::FdBuffers<StaticBuffer<uint8_t>> fdBuffers;
    uint32_t pingPongV0 = 0;
    __gm__ uint8_t *fdStagingBase = nullptr;
    GlobalTensor<float> stagingOutGm;
    __gm__ uint8_t *intraCoreCombineBase = nullptr;
    GlobalTensor<float> intraCoreCombineGm;
    __gm__ uint8_t *crossCoreCombineBase = nullptr;
    GlobalTensor<float> crossCoreCombineGm;

    T negativeFloatScalar;
    bool isSinks = false;
    uint32_t maxBlockNumPerBatch;
    uint32_t blockSize;
    int64_t sparseCalSize;
    int64_t sparseS2Start;
    int64_t sparseS2End;
};

TEMPLATES_DEF_NO_DEFAULT
template <bool IS_FULL>
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::GetRealCmpS2Idx(int64_t *tokenData, int64_t s2IdxInBase,
                                                                   const RunInfo &runInfo, ConstInfo &constInfo)
{
    int64_t curSparseS2End = this->sparseS2End;
    int64_t sparseBlockCount = 0;
    int64_t curS2LoopCnt = runInfo.s2LoopCount;
    if constexpr (TEMPLATE_MODE == SMLATemplateMode::CSA_TEMPLATE_MODE) {
        sparseBlockCount = constInfo.cmpSparseBlockCount;
        curS2LoopCnt -= runInfo.oriKvLoopEndIdx;
    } else if constexpr (TEMPLATE_MODE == SMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE) {
        sparseBlockCount = constInfo.oriSparseBlockCount;
    } else if constexpr (TEMPLATE_MODE == SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        if (runInfo.isCmp) {
            sparseBlockCount = constInfo.cmpSparseBlockCount;
            curS2LoopCnt -= runInfo.oriKvLoopEndIdx;
        } else {
            sparseBlockCount = constInfo.oriSparseBlockCount;
        }
    }
    uint64_t topkBS1Idx = 0;
    if constexpr (LAYOUT_T == SMLA_LAYOUT::TND) {
        uint64_t actualSeqQPrefixSum = cuSeqlensQGm.GetValue(runInfo.boIdx);
        topkBS1Idx += (actualSeqQPrefixSum + runInfo.s1oIdx) * sparseBlockCount; // T, N2(1), K
    } else {
        topkBS1Idx +=
            runInfo.boIdx * constInfo.s1Size * sparseBlockCount + runInfo.s1oIdx * sparseBlockCount; // B, S1, N2(1), K
    }

    uint64_t topkKIdx = s2IdxInBase + curS2LoopCnt * constInfo.s2BaseSize;
    for (uint64_t i = 0; i < KV_COPYIN_UNIT; ++i) { // 每次处理8个数据块
        uint64_t idx = topkBS1Idx + runInfo.s2StartIdx + topkKIdx + i;
        if constexpr (!IS_FULL) {
            // 尾块：保留边界判断，防止越界读取
            if (likely(s2IdxInBase + i < curSparseS2End)) {
                tokenData[i] = sparseIndicesGm.GetValue(idx);
            } else {
                break;
            }
        } else {
            // 非尾块：8行均有效，直接读取
            tokenData[i] = sparseIndicesGm.GetValue(idx);
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
template <bool IS_FULL>
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::GetRealS2Addr(int64_t *tokenData, int64_t s2IdxInBase,
                                                                 const RunInfo &runInfo, ConstInfo &constInfo)
{
    int64_t curSparseS2End = this->sparseS2End;
    uint32_t alignedSparseBlockCount =
        runInfo.isCmp ? constInfo.alignedCmpSparseBlockCount : constInfo.alignedOriSparseBlockCount;
    int64_t curS2LoopCnt = runInfo.s2LoopCount;
    GlobalTensor<int64_t> phyAddrGm64;
    if (runInfo.isCmp) {
        curS2LoopCnt -= runInfo.oriKvLoopEndIdx;
        phyAddrGm64 = cmpKvPhyAddrGm.template ReinterpretCast<int64_t>();
    } else {
        phyAddrGm64 = oriKvPhyAddrGm.template ReinterpretCast<int64_t>();
    }

    uint64_t topkBS1Idx = 0;
    if constexpr (LAYOUT_T == SMLA_LAYOUT::TND) {
        uint64_t actualSeqQPrefixSum = cuSeqlensQGm.GetValue(runInfo.boIdx);
        topkBS1Idx += (actualSeqQPrefixSum + runInfo.s1oIdx) * alignedSparseBlockCount;
    } else {
        topkBS1Idx +=
            runInfo.boIdx * constInfo.s1Size * alignedSparseBlockCount + runInfo.s1oIdx * alignedSparseBlockCount;
    }
    uint64_t topkKIdx = s2IdxInBase + curS2LoopCnt * constInfo.s2BaseSize;
    for (uint64_t i = 0; i < KV_COPYIN_UNIT; ++i) { // 每次处理8个数据块
        uint64_t idx = topkBS1Idx + runInfo.s2StartIdx + topkKIdx + i;
        if constexpr (!IS_FULL) {
            // 尾块：保留边界判断，防止越界读取
            if (likely(s2IdxInBase + i < curSparseS2End)) {
                tokenData[i] = phyAddrGm64.GetValue(idx);
            } else {
                break;
            }
        } else {
            // 非尾块：8行均有效，直接读取
            tokenData[i] = phyAddrGm64.GetValue(idx);
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline int64_t CSABlockVec<TEMPLATE_ARGS>::GetkeyOffset(int64_t s2Idx, const RunInfo &runInfo,
                                                                   ConstInfo &constInfo)
{
    if (s2Idx < 0) {
        return -1;
    }
    int64_t realkeyOffset = 0;
    if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::PA_BBND) {
        int64_t blkTableIdx = s2Idx / blockSize;
        int64_t blkTableOffset = s2Idx % blockSize;
        int64_t paBlockStride = runInfo.isCmp ? constInfo.cmpKeyStride0 : constInfo.oriKeyStride0;
        realkeyOffset = blockTableGm.GetValue(runInfo.boIdx * maxBlockNumPerBatch + blkTableIdx) * paBlockStride +
                        blkTableOffset * constInfo.dSizeVInput; // BlockNum, BlockSize, N(1), D
    } else if constexpr (LAYOUT_T == SMLA_LAYOUT::BSND) {
        if (runInfo.isCmp) {
            realkeyOffset = runInfo.boIdx * constInfo.n2Size * constInfo.cmpS2Size * constInfo.dSize +
                            runInfo.n2oIdx * constInfo.cmpS2Size * constInfo.dSize + s2Idx * constInfo.dSize; // BSN(1)D
        } else {
            realkeyOffset = runInfo.boIdx * constInfo.n2Size * constInfo.s2Size * constInfo.dSize +
                            runInfo.n2oIdx * constInfo.s2Size * constInfo.dSize + s2Idx * constInfo.dSize; // BSN(1)D
        }
    } else if constexpr (LAYOUT_T == SMLA_LAYOUT::TND) {
        realkeyOffset = (cuSeqlensKvGm.GetValue(runInfo.boIdx) + s2Idx) * constInfo.n2Size * constInfo.dSize +
                        runInfo.n2oIdx * constInfo.dSize; // TN(1)D
    }
    return realkeyOffset;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::CopyInSingleKv(LocalTensor<KV_T> kvInUb, int64_t startRow,
                                                                  int64_t keyOffset, ConstInfo &constInfo)
{
    if (keyOffset < 0) {
        return;
    }
    DataCopyExtParams intriParams;
    intriParams.blockCount = 1;
    intriParams.dstStride = 0;
    intriParams.srcStride = 0;
    intriParams.blockLen = constInfo.dSize * sizeof(KV_T);

    DataCopyPadExtParams<KV_T> padParams;
    padParams.isPad = true;
    padParams.leftPadding = 0;
    padParams.rightPadding =
        (CeilAlign(constInfo.dSize * sizeof(KV_T), BUFFER_SIZE_BYTE_32B) - constInfo.dSize * sizeof(KV_T)) /
        sizeof(KV_T);
    padParams.paddingValue = 0;
    DataCopyPad(kvInUb[startRow * constInfo.dSize], keyGm[keyOffset], intriParams, padParams);
}

TEMPLATES_DEF_NO_DEFAULT
template <bool IS_FULL>
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::CopyInKvSparse(LocalTensor<KV_T> kvInUb, int64_t startRow,
                                                                  int64_t *tokenData, const RunInfo &runInfo,
                                                                  ConstInfo &constInfo)
{
    for (uint32_t i = 0; i < 8; i += 2) { // 遍历8个元素的数组/缓冲区，每次处理2个元素
        int64_t keyOffset0;
        int64_t keyOffset1;
        if constexpr (IS_VEC_S2PHYADDR) {
            keyOffset0 = tokenData[i];
            keyOffset1 = tokenData[i + 1];
        } else {
            keyOffset0 = GetkeyOffset(tokenData[i], runInfo, constInfo);
            keyOffset1 = GetkeyOffset(tokenData[i + 1], runInfo, constInfo);
        }
        if constexpr (!IS_FULL) {
            // 尾块：提前返回判断
            if (unlikely(keyOffset0 < 0 && keyOffset1 < 0)) {
                return;
            }
        }
        int64_t combineBytes = constInfo.dSizeVInput * sizeof(KV_T);
        int64_t keySrcStride =
            (keyOffset0 > keyOffset1 ? (keyOffset0 - keyOffset1) : (keyOffset1 - keyOffset0)) * sizeof(KV_T) -
            combineBytes;
        if (unlikely(keySrcStride >= INT32_MAX || keySrcStride < 0) || constInfo.sparseBlockSize > 1) {
            // stride溢出、stride为负数、s2超长等异常场景，还原成2条搬运指令
            CopyInSingleKv(kvInUb, startRow, keyOffset0, constInfo);
            CopyInSingleKv(kvInUb, startRow + 1, keyOffset1, constInfo);
        } else {
            DataCopyExtParams intriParams;
            if constexpr (!IS_FULL) {
                // 尾块：根据实际有效条目数设置blockCount,且此处仅有可能存在keyOffset1为-1的情况
                intriParams.blockCount = 1 + (keyOffset1 >= 0);
            } else {
                // 非尾块：两条均有效，blockCount恒为2
                intriParams.blockCount = 2;
            }
            intriParams.blockLen = combineBytes;
            intriParams.dstStride = 0;
            intriParams.srcStride = keySrcStride;
            DataCopyPadExtParams<KV_T> padParams;
            padParams.isPad = true;
            padParams.leftPadding = 0;
            padParams.rightPadding = (CeilAlign(combineBytes, BUFFER_SIZE_BYTE_32B) - combineBytes) / sizeof(KV_T);
            padParams.paddingValue = 0;

            int64_t keyOffset;
            if constexpr (!IS_FULL) {
                keyOffset = (keyOffset1 > -1 && keyOffset1 < keyOffset0) ? keyOffset1 : keyOffset0;
            } else {
                // 非尾块：两条均有效，取较小地址作为起始
                keyOffset = keyOffset0 < keyOffset1 ? keyOffset0 : keyOffset1;
            }
            DataCopyPad(kvInUb[startRow * constInfo.dSize], keyGm[keyOffset], intriParams, padParams);
        }
        startRow += 2; // 每次迭代处理2个输入元素
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::CopyToOutUb(LocalTensor<Q_T> kvOutUb, LocalTensor<KV_T> srcTensor,
                                                               int64_t dealRow, ConstInfo &constInfo)
{
    LocalTensor<Q_T> kvNdUb = srcTensor.template ReinterpretCast<Q_T>();
    DataCopy(kvOutUb, kvNdUb, dealRow * constInfo.dSize);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::CopyOutKvUb2Gm(
    Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm, LocalTensor<Q_T> kvOutUb, int64_t dealRow,
    int64_t s2StartIdx, const RunInfo &runInfo, ConstInfo &constInfo)
{
    GlobalTensor<Q_T> v0ResGmTensor = v0ResGm.template GetTensor<Q_T>();
    DataCopy(v0ResGmTensor[s2StartIdx * constInfo.dSize], kvOutUb, dealRow * constInfo.dSize);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::CalSparseCalSize(const RunInfo &runInfo, ConstInfo &constInfo)
{
    if constexpr (IS_SPLIT_G) {
        uint32_t aicIdx = constInfo.aivIdx >> 1U;
        uint32_t v0S2SizeFirstCore = CeilDiv(runInfo.s2RealSize, 2);
        uint32_t v0S2SizeSecondCore = runInfo.s2RealSize - v0S2SizeFirstCore;
        int32_t vecCnt = (aicIdx % 2U == 0) ?
                             (GetSubBlockIdx() == 0 ? 0 : 1) :
                             (GetSubBlockIdx() == 0 ? 2 : 3); // 2，3：根据核心索引和子块索引设置处理参数
        if (aicIdx % 2 == 0) { // 2：根据aicIdx的奇偶性来区分不同的处理逻辑
            if (GetSubBlockIdx() == 0) {
                sparseCalSize = CeilDiv(v0S2SizeFirstCore, 2); // 2：处理大小为v0S2SizeFirstCore的一半
                sparseS2Start = 0;
            } else {
                sparseCalSize = v0S2SizeFirstCore - CeilDiv(v0S2SizeFirstCore, 2); // 2：处理剩余部分
                sparseS2Start = CeilDiv(v0S2SizeFirstCore, 2); // 2：起始位置为v0S2SizeFirstCore的一半
            }
        } else {
            if (GetSubBlockIdx() == 0) {
                sparseCalSize = CeilDiv(v0S2SizeSecondCore, 2); // 2：处理大小为v0S2SizeSecondCore的一半
                sparseS2Start = v0S2SizeFirstCore;
            } else {
                sparseCalSize = v0S2SizeSecondCore - CeilDiv(v0S2SizeSecondCore, 2); // 2：处理剩余部分
                sparseS2Start =
                    v0S2SizeFirstCore +
                    CeilDiv(v0S2SizeSecondCore, 2); // 2：起始位置为v0S2SizeFirstCore加上v0S2SizeSecondCore的一半
            }
        }
        sparseS2End = sparseS2Start + sparseCalSize;
    } else {
        uint32_t v0S2SizeFirstCore = CeilDiv(runInfo.s2RealSize, 2); // 2：平均分配给两个子块
        sparseCalSize = GetSubBlockIdx() == 0 ? v0S2SizeFirstCore : runInfo.s2RealSize - v0S2SizeFirstCore;
        sparseS2Start = GetSubBlockIdx() == 0 ? 0 : v0S2SizeFirstCore;
        sparseS2End = sparseS2Start + sparseCalSize;
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::ProcessVec0(
    Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm, const RunInfo &runInfo, ConstInfo &constInfo)
{
    if constexpr (TEMPLATE_MODE == SMLATemplateMode::CSA_TEMPLATE_MODE) {
        if (runInfo.s2LoopCount < runInfo.oriKvLoopEndIdx) {
            return;
        }
        keyGm = cmpKVGm;
        cuSeqlensKvGm = cuSeqlensCmpKvGm;
        sparseIndicesGm = cmpSparseIndicesGm;
        if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::PA_BBND) {
            blockTableGm = cmpBlockTableGm;
            blockSize = constInfo.cmpBlockSize;
            maxBlockNumPerBatch = constInfo.cmpMaxBlockNumPerBatch;
        }
        CalSparseCalSize(runInfo, constInfo);
        ProcessSparseKv(v0ResGm, runInfo, constInfo);
        v0ResGm.SetCrossCore();
    } else if constexpr (TEMPLATE_MODE == SMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                         TEMPLATE_MODE == SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        if (!runInfo.isCmp) {
            keyGm = oriKVGm;
            cuSeqlensKvGm = cuSeqlensOriKvGm;
            sparseIndicesGm = oriSparseIndicesGm;
            if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::PA_BBND) {
                blockTableGm = oriBlockTableGm;
                blockSize = constInfo.oriBlockSize;
                maxBlockNumPerBatch = constInfo.oriMaxBlockNumPerBatch;
            }
        } else {
            keyGm = cmpKVGm;
            cuSeqlensKvGm = cuSeqlensCmpKvGm;
            sparseIndicesGm = cmpSparseIndicesGm;
            if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::PA_BBND) {
                blockTableGm = cmpBlockTableGm;
                blockSize = constInfo.cmpBlockSize;
                maxBlockNumPerBatch = constInfo.cmpMaxBlockNumPerBatch;
            }
        }
        CalSparseCalSize(runInfo, constInfo);
        ProcessSparseKv(v0ResGm, runInfo, constInfo);
        v0ResGm.SetCrossCore();
    }
}

TEMPLATES_DEF_NO_DEFAULT
template <bool IS_FULL>
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::CopyIn8Block(LocalTensor<KV_T> kvInUb, int64_t startRow, int64_t &s2,
                                                                const RunInfo &runInfo, ConstInfo &constInfo)
{
    // tokenData元素为-1表示无效token，尾块场景下GetReal*仅填充有效区间，其余保持-1
    int64_t tokenData[KV_COPYIN_UNIT] = {-1, -1, -1, -1, -1, -1, -1, -1};
    if constexpr (IS_VEC_S2PHYADDR) {
        GetRealS2Addr<IS_FULL>(tokenData, s2, runInfo, constInfo);
    } else {
        GetRealCmpS2Idx<IS_FULL>(tokenData, s2, runInfo, constInfo);
    }
    s2 += KV_COPYIN_UNIT;
    CopyInKvSparse<IS_FULL>(kvInUb, startRow, tokenData, runInfo, constInfo);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::ProcessSparseKv(
    Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm, const RunInfo &runInfo, ConstInfo &constInfo)
{
    if constexpr (TEMPLATE_MODE == SMLATemplateMode::CSA_TEMPLATE_MODE ||
                  TEMPLATE_MODE == SMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                  TEMPLATE_MODE == SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        int64_t curSparseCalSize = this->sparseCalSize;
        if (curSparseCalSize == 0) {
            return;
        }

        // 前置计算：16行拷出循环数、尾块行数、尾块内8行搬入次数及剩余行数
        int64_t process16LoopCnt = curSparseCalSize / KV_PROCESS_UNIT;
        int64_t tail16Rows = curSparseCalSize % KV_PROCESS_UNIT;
        int64_t tail8FullCnt = tail16Rows / KV_COPYIN_UNIT;
        int64_t tail8Remain = tail16Rows % KV_COPYIN_UNIT;
        int64_t s2 = sparseS2Start;

        // 阶段1：完整16行块（非尾块，8行均有效，无需判断）
        for (int64_t i = 0; i < process16LoopCnt; i++) {
            int64_t s2StartIdx = sparseS2Start + i * KV_PROCESS_UNIT;
            // 1、copy kv in, gm -> ub
            LocalTensor<Q_T> stage0OutUb = this->stage0OutBufs[pingPongV0].tensor;
            WaitFlag<HardEvent::MTE3_MTE2>(INNERCORE_STAGE0OUT_MTE3_MTE2(pingPongV0));
            CopyIn8Block<true>(stage0OutUb, 0, s2, runInfo, constInfo);
            CopyIn8Block<true>(stage0OutUb, KV_COPYIN_UNIT, s2, runInfo, constInfo);
            // 2、copy kv out, ub -> l1
            SetFlag<HardEvent::MTE2_MTE3>(INNERCORE_STAGE0OUT_MTE2_MTE3(pingPongV0));
            WaitFlag<HardEvent::MTE2_MTE3>(INNERCORE_STAGE0OUT_MTE2_MTE3(pingPongV0));
            CopyOutKvUb2Gm(v0ResGm, stage0OutUb, KV_PROCESS_UNIT, s2StartIdx, runInfo, constInfo);
            SetFlag<HardEvent::MTE3_MTE2>(INNERCORE_STAGE0OUT_MTE3_MTE2(pingPongV0));
            pingPongV0 ^= 1;
        }

        // 阶段2：尾块（不足16行，保留判断逻辑）
        if (tail16Rows > 0) {
            int64_t s2StartIdx = sparseS2Start + process16LoopCnt * KV_PROCESS_UNIT;
            int64_t dealRow = 0;
            // 1、copy kv in, gm -> ub
            LocalTensor<Q_T> stage0OutUb = this->stage0OutBufs[pingPongV0].tensor;
            WaitFlag<HardEvent::MTE3_MTE2>(INNERCORE_STAGE0OUT_MTE3_MTE2(pingPongV0));
            // 尾块内满8行搬入（8行均有效，无需判断）
            for (int64_t j = 0; j < tail8FullCnt; j++) {
                CopyIn8Block<true>(stage0OutUb, dealRow, s2, runInfo, constInfo);
                dealRow += KV_COPYIN_UNIT;
            }
            // 尾块内不足8行（需判断边界，与当前逻辑一致）
            if (tail8Remain > 0) {
                CopyIn8Block<false>(stage0OutUb, dealRow, s2, runInfo, constInfo);
                dealRow += tail8Remain;
            }
            // 2、copy kv out, ub -> l1
            SetFlag<HardEvent::MTE2_MTE3>(INNERCORE_STAGE0OUT_MTE2_MTE3(pingPongV0));
            WaitFlag<HardEvent::MTE2_MTE3>(INNERCORE_STAGE0OUT_MTE2_MTE3(pingPongV0));
            CopyOutKvUb2Gm(v0ResGm, stage0OutUb, tail16Rows, s2StartIdx, runInfo, constInfo);
            SetFlag<HardEvent::MTE3_MTE2>(INNERCORE_STAGE0OUT_MTE3_MTE2(pingPongV0));
            pingPongV0 ^= 1;
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
template <bool UPDATE>
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::ComputeVec1Softmax(LocalTensor<Q_T> &stage1CastTensor,
                                                                      LocalTensor<T> &mmRes, LocalTensor<float> &sumUb,
                                                                      LocalTensor<float> &maxUb,
                                                                      LocalTensor<T> &apiTmpBuffer, RunInfo &runInfo,
                                                                      ConstInfo &constInfo)
{
    if (likely(runInfo.s2RealSize == 128 && runInfo.s2RealSizeUpdate == 128)) {
        ProcessVec1Vf<T, Q_T, UPDATE, s1BaseSize, s2BaseSize, FaVectorApi::OriginNRange::EQ_128_SFA>(
            stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, vselrIndexesBuf, runInfo.halfMRealSize,
            runInfo.s2RealSizeUpdate, static_cast<T>(constInfo.softmaxScale), negativeFloatScalar);
    } else if (runInfo.s2RealSize <= 64) {
        ProcessVec1Vf<T, Q_T, UPDATE, s1BaseSize, s2BaseSize, FaVectorApi::OriginNRange::GT_0_AND_LTE_64_SFA>(
            stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, vselrIndexesBuf, runInfo.halfMRealSize,
            runInfo.s2RealSizeUpdate, static_cast<T>(constInfo.softmaxScale), negativeFloatScalar);
    } else if (runInfo.s2RealSize < 128 || runInfo.s2RealSizeUpdate < 128) {
        ProcessVec1Vf<T, Q_T, UPDATE, s1BaseSize, s2BaseSize, FaVectorApi::OriginNRange::GT_64_AND_LTE_128_SFA>(
            stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, vselrIndexesBuf, runInfo.halfMRealSize,
            runInfo.s2RealSizeUpdate, static_cast<T>(constInfo.softmaxScale), negativeFloatScalar);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::InitVec1SoftmaxFromSinks(LocalTensor<float> &sumUb,
                                                                            LocalTensor<float> &maxUb, RunInfo &runInfo,
                                                                            ConstInfo &constInfo)
{
    bool includeSink = (!runInfo.isCrossCoreSplit) || runInfo.isFirstS2SplitCore;
    if constexpr (IS_BATCH_CONSISTENCY) {
        includeSink = includeSink && (runInfo.reduceBlockId == 0);
    }
    if (!includeSink) {
        Duplicate(maxUb, this->negativeFloatScalar, runInfo.halfMRealSize);
        Duplicate(sumUb, static_cast<T>(0), runInfo.halfMRealSize);
        return;
    }
    int64_t sinksOffset = 0;
    if constexpr (!IS_SPLIT_G) {
        sinksOffset = GetBlockIdx() % 2 == 0 ? 0 : runInfo.firstHalfMRealSize; // 2：判断块索引的奇偶性
    } else {
        sinksOffset = runInfo.goIdx;
        if (constInfo.subBlockIdx == 1) {
            sinksOffset += runInfo.firstHalfMRealSize;
        }
    }
    LocalTensor<T> sinksUb = this->sinksUb.tensor;
    InitSoftmaxFromSinks<T>(sumUb, maxUb, sinksUb, sinksOffset, R0, runInfo.halfMRealSize);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::CopyVec1ResultToL1(StaticBuffer<Q_T> &outputBuf,
                                                                      LocalTensor<Q_T> &stage1CastTensor,
                                                                      RunInfo &runInfo, ConstInfo &constInfo)
{
    int64_t stage1Offset = runInfo.taskIdMod2;
    SetFlag<HardEvent::V_MTE3>(INNERCORE_STAGE1(stage1Offset));
    WaitFlag<HardEvent::V_MTE3>(INNERCORE_STAGE1(stage1Offset));
    LocalTensor<Q_T> mm2AL1Tensor = outputBuf.tensor;
    if (likely(runInfo.halfMRealSize != 0)) {
        DataCopy(mm2AL1Tensor[constInfo.subBlockIdx * (BLOCK_BYTE / sizeof(Q_T)) *
                              (runInfo.mRealSize - runInfo.halfMRealSize)],
                 stage1CastTensor,
                 {s2BaseSize / 16, static_cast<uint16_t>(runInfo.halfMRealSize),
                  static_cast<uint16_t>(vec1Srcstride - runInfo.halfMRealSize),
                  static_cast<uint16_t>(Align16Func(runInfo.mRealSize) - runInfo.halfMRealSize)});
    }
    SetFlag<HardEvent::MTE3_V>(INNERCORE_STAGE1(stage1Offset));
    CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(CROSSCORE_L1P(outputBuf.idx));
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::StageCrossCoreVec1Lse(LocalTensor<float> &maxUb,
                                                                         LocalTensor<float> &sumUb, RunInfo &runInfo,
                                                                         ConstInfo &constInfo)
{
    AttentionCommon::S2SplitFdStagingLayout stagingLayout = {
        constInfo.gSize, dTemplateAlign64, GetStagingSlotNum(false), AttentionCommon::FD_BROADCAST_ELEMS_PER_ROW,
        AttentionCommon::FD_REDUCE_CHUNK_ROWS};
    LocalTensor<float> tmpUb = this->batchReduceTmpUb.tensor;
    AttentionCommon::StageVec1Lse(stagingLayout, crossCoreCombineBase, GetCrossCoreWorkspaceIdx(runInfo),
                                  GetFaStagingMOffset(runInfo, constInfo), runInfo.halfMRealSize, maxUb, sumUb, tmpUb,
                                  INNERCORE_STAGE2, INNERCORE_STAGE_FD_MTE3_V);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::StageBatchConsistencyVec1Lse(LocalTensor<float> &maxUb,
                                                                                LocalTensor<float> &sumUb,
                                                                                RunInfo &runInfo, ConstInfo &constInfo)
{
    if (!runInfo.isLastBase) {
        return;
    }
    if (runInfo.halfMRealSize > 0) {
        LocalTensor<float> finalMaxUb = this->softmaxFinalMaxBufs[runInfo.taskIdMod2].tensor;
        LocalTensor<float> finalSumUb = this->softmaxFinalSumBufs[runInfo.taskIdMod2].tensor;
        uint64_t snapshotElems = Align8Func(runInfo.halfMRealSize);
        DataCopy(finalMaxUb, maxUb, snapshotElems);
        DataCopy(finalSumUb, sumUb, snapshotElems);
    }
    if (runInfo.isCrossCoreSplit && !runInfo.isFirstS2SplitCore) {
        StageCrossCoreVec1Lse(maxUb, sumUb, runInfo, constInfo);
    } else if (runInfo.isFirstS2SplitCore && runInfo.reduceBlockId == 0 && runInfo.s2LoopCount < runInfo.s2LoopLimit) {
        AttentionCommon::S2SplitFdStagingLayout stagingLayout = {
            constInfo.gSize, dTemplateAlign64, GetStagingSlotNum(true), AttentionCommon::FD_BROADCAST_ELEMS_PER_ROW,
            AttentionCommon::FD_REDUCE_CHUNK_ROWS};
        LocalTensor<float> tmpUb = this->batchReduceTmpUb.tensor;
        AttentionCommon::StageVec1Lse(stagingLayout, intraCoreCombineBase, GetIntraCoreWorkspaceIdx(runInfo, constInfo),
                                      GetFaStagingMOffset(runInfo, constInfo), runInfo.halfMRealSize, maxUb, sumUb,
                                      tmpUb, INNERCORE_STAGE2, INNERCORE_STAGE_FD_MTE3_V);
        SetFlag<HardEvent::MTE3_MTE2>(INNERCORE_INTRALSE_MTE3_MTE2(runInfo.multiCoreIdxMod2));
    } else if (runInfo.isCrossCoreSplit && runInfo.isFirstS2SplitCore && runInfo.reduceBlockId == 0) {
        StageCrossCoreVec1Lse(maxUb, sumUb, runInfo, constInfo);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::StageLegacyVec1Lse(LocalTensor<float> &maxUb,
                                                                      LocalTensor<float> &sumUb, RunInfo &runInfo,
                                                                      ConstInfo &constInfo)
{
    if (!runInfo.isCrossCoreSplit || runInfo.halfMRealSize <= 0 || runInfo.s2LoopCount != runInfo.s2LoopLimit) {
        return;
    }
    AttentionCommon::S2SplitFdStagingLayout stagingLayout = {constInfo.gSize, dTemplateAlign64, GetStagingSlotNum(),
                                                             AttentionCommon::FD_BROADCAST_ELEMS_PER_ROW,
                                                             AttentionCommon::FD_REDUCE_CHUNK_ROWS};
    LocalTensor<float> tmpUb = this->stage2OutBufs.tensor.template ReinterpretCast<float>();
    AttentionCommon::StageVec1Lse(stagingLayout, fdStagingBase, GetCrossCoreWorkspaceIdx(runInfo),
                                  GetFaStagingMOffset(runInfo, constInfo), static_cast<uint32_t>(runInfo.halfMRealSize),
                                  maxUb, sumUb, tmpUb, INNERCORE_STAGE2, INNERCORE_STAGE_FD_MTE3_V);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::CopyOutVec1Lse(LocalTensor<float> &maxUb, LocalTensor<float> &sumUb,
                                                                  RunInfo &runInfo, ConstInfo &constInfo)
{
    bool copyOutLse =
        constInfo.returnSoftmaxLse && runInfo.halfMRealSize > 0 && runInfo.s2LoopCount == runInfo.s2LoopLimit;
    if constexpr (IS_BATCH_CONSISTENCY) {
        copyOutLse = copyOutLse && !runInfo.isCrossCoreSplit && !runInfo.needReduce;
    }
    if (!copyOutLse) {
        return;
    }
    LocalTensor<float> outLse = this->outLseUbs[runInfo.multiCoreIdxMod2].tensor;
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = 1;
    dataCopyParams.blockLen = sizeof(float) * runInfo.halfMRealSize;
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;
    WaitFlag<HardEvent::MTE3_V>(INNERCORE_LSE_MTE3_V);
    ComputeLse<float>(outLse, sumUb, maxUb, runInfo.halfMRealSize);
    SetFlag<HardEvent::V_MTE3>(INNERCORE_LSE_V_MTE3);
    WaitFlag<HardEvent::V_MTE3>(INNERCORE_LSE_V_MTE3);
    DataCopyPad(this->softmaxLseGm[runInfo.softmaxLseOffset], outLse, dataCopyParams);
    SetFlag<HardEvent::MTE3_V>(INNERCORE_LSE_MTE3_V);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::ProcessVec1(StaticBuffer<Q_T> &outputBuf,
                                                               StaticBuffer<T> &bmm1ResBuf, RunInfo &runInfo,
                                                               ConstInfo &constInfo)
{
    CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CROSSCORE_BMM1(bmm1ResBuf.idx));

    LocalTensor<float> sumUb = this->softmaxSumBufs[runInfo.multiCoreIdxMod2].tensor;
    LocalTensor<float> maxUb = this->softmaxMaxBufs[runInfo.multiCoreIdxMod2].tensor;
    LocalTensor<float> expUb = this->softmaxExpBufs[runInfo.taskIdMod2].tensor;
    int64_t stage1Offset = runInfo.taskIdMod2;
    WaitFlag<HardEvent::MTE3_V>(INNERCORE_STAGE1(stage1Offset));
    LocalTensor<Q_T> stage1CastTensor = this->stage1OutBufs[stage1Offset].tensor;

    LocalTensor<T> apiTmpBuffer = this->commonUb.tensor;
    LocalTensor<T> mmRes = bmm1ResBuf.tensor;

    runInfo.s2RealSizeUpdate = runInfo.s2RealSize;

    bool isFirstSoftmaxBase = runInfo.s2LoopCount == 0;
    if constexpr (IS_BATCH_CONSISTENCY) {
        isFirstSoftmaxBase = runInfo.isFirstBase;
    }
    // loopCount = 0 但传入sinks时走update分支，maxUb通过sinks初始化，sumUb初始化为1.0
    if (isFirstSoftmaxBase && !isSinks) {
        ComputeVec1Softmax<false>(stage1CastTensor, mmRes, sumUb, maxUb, apiTmpBuffer, runInfo, constInfo);
    } else {
        if (isFirstSoftmaxBase && isSinks) {
            InitVec1SoftmaxFromSinks(sumUb, maxUb, runInfo, constInfo);
        }
        ComputeVec1Softmax<true>(stage1CastTensor, mmRes, sumUb, maxUb, apiTmpBuffer, runInfo, constInfo);
    }
    CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CROSSCORE_BMM1(bmm1ResBuf.idx));
    CopyVec1ResultToL1(outputBuf, stage1CastTensor, runInfo, constInfo);
    if (!isFirstSoftmaxBase || isSinks) {
        SFAUpdateExpSumAndExpMax<T>(sumUb, maxUb, expUb, sumUb, maxUb, apiTmpBuffer, runInfo.halfMRealSize);
    }
    if constexpr (IS_BATCH_CONSISTENCY) {
        StageBatchConsistencyVec1Lse(maxUb, sumUb, runInfo, constInfo);
    } else {
        StageLegacyVec1Lse(maxUb, sumUb, runInfo, constInfo);
    }
    CopyOutVec1Lse(maxUb, sumUb, runInfo, constInfo);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::ReduceIntraBlockAndStage(RunInfo &runInfo, ConstInfo &constInfo,
                                                                            LocalTensor<T> &vec2ResUb,
                                                                            LocalTensor<T> &partialTmpUb)
{
    AttentionCommon::S2SplitFdStagingLayout intraLayout = {constInfo.gSize, dTemplateAlign64, GetStagingSlotNum(true),
                                                           AttentionCommon::FD_BROADCAST_ELEMS_PER_ROW,
                                                           AttentionCommon::FD_REDUCE_CHUNK_ROWS};
    AttentionCommon::S2SplitFdStagingLayout crossLayout = {constInfo.gSize, dTemplateAlign64, GetStagingSlotNum(false),
                                                           AttentionCommon::FD_BROADCAST_ELEMS_PER_ROW,
                                                           AttentionCommon::FD_REDUCE_CHUNK_ROWS};
    uint32_t intraWorkspaceIdx = GetIntraCoreWorkspaceIdx(runInfo, constInfo);
    uint32_t crossWorkspaceIdx =
        static_cast<uint32_t>(runInfo.firstFdDataWorkspaceIdx + runInfo.s2SplitIdx - runInfo.reduceBlockId);
    int64_t stagingMOffset = GetFaStagingMOffset(runInfo, constInfo);
    LocalTensor<float> tmpUb = this->batchReduceTmpUb.tensor;
    LocalTensor<float> blockMaxUb = tmpUb;
    LocalTensor<float> blockSumUb = tmpUb[256];
    LocalTensor<float> lseBroadcastUb = tmpUb[512];
    LocalTensor<float> sumBroadcastUb = tmpUb[640];
    LocalTensor<float> maxUb = this->softmaxFinalMaxBufs[runInfo.taskIdMod2].tensor;
    LocalTensor<float> sumUb = this->softmaxFinalSumBufs[runInfo.taskIdMod2].tensor;
    bool copyOutMergedLse =
        constInfo.returnSoftmaxLse && !runInfo.isCrossCoreSplit && runInfo.s2LoopCount == runInfo.s2LoopLimit;

    WaitFlag<HardEvent::MTE3_MTE2>(INNERCORE_INTRALSE_MTE3_MTE2(runInfo.multiCoreIdxMod2));
    WaitFlag<HardEvent::MTE3_MTE2>(INNERCORE_INTRAATTN_MTE3_MTE2(runInfo.multiCoreIdxMod2));
    LocalTensor<T> sinkUb;
    int64_t startRow = 0;
    while (startRow < runInfo.vec2MRealSize) {
        int64_t dealRowCount = intraLayout.chunkRows;
        if (startRow + dealRowCount > runInfo.vec2MRealSize) {
            dealRowCount = runInfo.vec2MRealSize - startRow;
        }
        LocalTensor<T> chunkCurrent = vec2ResUb[startRow * dTemplateAlign64];
        LocalTensor<float> chunkMaxUb = maxUb[startRow];
        LocalTensor<float> chunkSumUb = sumUb[startRow];
        if (copyOutMergedLse) {
            WaitFlag<HardEvent::MTE3_V>(INNERCORE_LSE_MTE3_V);
        }
        AttentionCommon::MergeStagedAndCurrentChunk<T, dTemplateAlign64>(
            intraLayout, intraCoreCombineBase, intraWorkspaceIdx, stagingMOffset + startRow, dealRowCount,
            static_cast<int64_t>(constInfo.dSizeV), chunkMaxUb, chunkSumUb, chunkCurrent, blockMaxUb, blockSumUb,
            partialTmpUb, lseBroadcastUb, sumBroadcastUb, sinkUb, INNERCORE_REDUCE_MAXSUM_V_MTE2,
            INNERCORE_INTRAPARTIALO_V_MTE2, INNERCORE_REDUCE_MTE2_V);

        AttentionCommon::StageBroadcastMaxSum(intraLayout, intraCoreCombineBase, intraWorkspaceIdx,
                                              stagingMOffset + startRow, dealRowCount, lseBroadcastUb, sumBroadcastUb,
                                              INNERCORE_STAGE2, INNERCORE_STAGE_FD_MTE3_V);
        if (copyOutMergedLse) {
            DataCopyExtParams lseParams;
            lseParams.blockCount = static_cast<uint16_t>(dealRowCount);
            lseParams.blockLen = sizeof(float);
            lseParams.srcStride = 0;
            lseParams.dstStride = 0;
            SetFlag<HardEvent::V_MTE3>(INNERCORE_LSE_V_MTE3);
            WaitFlag<HardEvent::V_MTE3>(INNERCORE_LSE_V_MTE3);
            DataCopyPad(this->softmaxLseGm[runInfo.softmaxLseOffset + startRow], lseBroadcastUb, lseParams);
            SetFlag<HardEvent::MTE3_V>(INNERCORE_LSE_MTE3_V);
        }
        if (runInfo.isCrossCoreSplit && runInfo.s2LoopCount == runInfo.s2LoopLimit) {
            AttentionCommon::StageBroadcastMaxSum(crossLayout, crossCoreCombineBase, crossWorkspaceIdx,
                                                  stagingMOffset + startRow, dealRowCount, lseBroadcastUb,
                                                  sumBroadcastUb, INNERCORE_STAGE2, INNERCORE_STAGE_FD_MTE3_V);
        }
        startRow += intraLayout.chunkRows;
    }

    AttentionCommon::StageVec2PartialOAndWait<T>(intraLayout, intraCoreCombineGm, intraWorkspaceIdx, stagingMOffset,
                                                 runInfo.vec2MRealSize, static_cast<uint32_t>(constInfo.dSizeV),
                                                 vec2ResUb, INNERCORE_STAGE2, INNERCORE_STAGE_FD_MTE3_V);
    if (runInfo.isCrossCoreSplit && runInfo.s2LoopCount == runInfo.s2LoopLimit) {
        AttentionCommon::StageVec2PartialOAndWait<T>(crossLayout, crossCoreCombineGm, crossWorkspaceIdx, stagingMOffset,
                                                     runInfo.vec2MRealSize, static_cast<uint32_t>(constInfo.dSizeV),
                                                     vec2ResUb, INNERCORE_STAGE2, INNERCORE_STAGE_FD_MTE3_V);
    }
    if (runInfo.s2LoopCount < runInfo.s2LoopLimit) {
        SetFlag<HardEvent::MTE3_MTE2>(INNERCORE_INTRALSE_MTE3_MTE2(runInfo.multiCoreIdxMod2));
        SetFlag<HardEvent::MTE3_MTE2>(INNERCORE_INTRAATTN_MTE3_MTE2(runInfo.multiCoreIdxMod2));
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::ProcessVec2(StaticBuffer<T> &bmm2ResBuf, RunInfo &runInfo,
                                                               ConstInfo &constInfo)
{
    CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CROSSCORE_BMM2);
    if (unlikely(runInfo.vec2MBaseSize == 0)) {
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CROSSCORE_BMM2);
        return;
    }

    runInfo.vec2MRealSize = runInfo.vec2MBaseSize;
    int64_t vec2CalcSize = runInfo.vec2MRealSize * dTemplateAlign64;
    LocalTensor<T> vec2ResUb = this->stage2OutBufs.tensor;
    LocalTensor<T> mmRes = bmm2ResBuf.tensor;
    WaitFlag<HardEvent::MTE3_V>(INNERCORE_STAGE2);
    bool needIntraBlockReduce = false;
    if constexpr (IS_BATCH_CONSISTENCY) {
        needIntraBlockReduce = runInfo.isLastBase && runInfo.isFirstS2SplitCore && runInfo.reduceBlockId > 0;
        if (needIntraBlockReduce) {
            WaitFlag<HardEvent::V_MTE2>(INNERCORE_INTRAPARTIALO_V_MTE2);
            WaitFlag<HardEvent::V_MTE2>(INNERCORE_REDUCE_MAXSUM_V_MTE2);
        }
    }
    bool isFirstVec2Base = runInfo.s2LoopCount == 0;
    if constexpr (IS_BATCH_CONSISTENCY) {
        isFirstVec2Base = runInfo.isFirstBase;
    }
    if (unlikely(isFirstVec2Base)) {
        DataCopy(vec2ResUb, mmRes, vec2CalcSize);
    } else {
        if (runInfo.s2RealSizeUpdate > 0) {
            LocalTensor<T> expUb = softmaxExpBufs[runInfo.taskIdMod2].tensor;
            bool isLastVec2Base = (runInfo.s2LoopCount == runInfo.s2LoopLimit);
            if constexpr (IS_BATCH_CONSISTENCY) {
                isLastVec2Base = runInfo.isLastBase;
            }
            if (isLastVec2Base) {
                LocalTensor<float> sumUb;
                if constexpr (IS_BATCH_CONSISTENCY) {
                    sumUb = this->softmaxFinalSumBufs[runInfo.taskIdMod2].tensor;
                } else {
                    sumUb = this->softmaxSumBufs[runInfo.multiCoreIdxMod2].tensor;
                }
                FlashUpdateLastNew<T, Q_T, OUTPUT_T, dTemplateAlign64, false, false>(
                    vec2ResUb, mmRes, vec2ResUb, expUb, expUb, sumUb, runInfo.vec2MRealSize, dTemplateAlign64, 1.0,
                    1.0);
            } else {
                FlashUpdateNew<T, Q_T, OUTPUT_T, dTemplateAlign64, false, false>(
                    vec2ResUb, mmRes, vec2ResUb, expUb, expUb, runInfo.vec2MRealSize, dTemplateAlign64, 1.0, 1.0);
            }
        } else {
            bool isLastVec2Base = runInfo.s2LoopCount >= runInfo.s2LoopLimit;
            if constexpr (IS_BATCH_CONSISTENCY) {
                isLastVec2Base = runInfo.isLastBase;
            }
            if (isLastVec2Base) {
                LocalTensor<float> sumUb;
                if constexpr (IS_BATCH_CONSISTENCY) {
                    sumUb = this->softmaxFinalSumBufs[runInfo.taskIdMod2].tensor;
                } else {
                    sumUb = this->softmaxSumBufs[runInfo.multiCoreIdxMod2].tensor;
                }
                LastDivNew<T, Q_T, OUTPUT_T, dTemplateAlign64, false>(vec2ResUb, vec2ResUb, sumUb,
                                                                      runInfo.vec2MRealSize, dTemplateAlign64, 1.0);
            }
        }
    }

    if constexpr (IS_BATCH_CONSISTENCY) {
        if (runInfo.isLastBase) {
            if (unlikely(isFirstVec2Base)) {
                LocalTensor<float> sumUb = this->softmaxFinalSumBufs[runInfo.taskIdMod2].tensor;
                LastDivNew<T, Q_T, OUTPUT_T, dTemplateAlign64, false>(vec2ResUb, vec2ResUb, sumUb,
                                                                      runInfo.vec2MRealSize, dTemplateAlign64, 1.0);
            }
            if (needIntraBlockReduce) {
                SetFlag<HardEvent::V_MTE2>(INNERCORE_INTRAPARTIALO_V_MTE2);
                SetFlag<HardEvent::V_MTE2>(INNERCORE_REDUCE_MAXSUM_V_MTE2);
                ReduceIntraBlockAndStage(runInfo, constInfo, vec2ResUb, mmRes);
                if (!runInfo.isCrossCoreSplit && runInfo.s2LoopCount == runInfo.s2LoopLimit) {
                    this->CopyOutAttentionOut(runInfo, constInfo, vec2ResUb, 0, vec2CalcSize);
                }
            } else {
                if (runInfo.isFirstS2SplitCore && runInfo.reduceBlockId == 0 &&
                    runInfo.s2LoopCount < runInfo.s2LoopLimit) {
                    AttentionCommon::S2SplitFdStagingLayout stagingLayout = {
                        constInfo.gSize, dTemplateAlign64, GetStagingSlotNum(true),
                        AttentionCommon::FD_BROADCAST_ELEMS_PER_ROW, AttentionCommon::FD_REDUCE_CHUNK_ROWS};
                    int64_t stagingMOffset = GetFaStagingMOffset(runInfo, constInfo);
                    AttentionCommon::StageVec2PartialOAndWait<T>(
                        stagingLayout, intraCoreCombineGm, GetIntraCoreWorkspaceIdx(runInfo, constInfo), stagingMOffset,
                        runInfo.vec2MRealSize, static_cast<uint32_t>(constInfo.dSizeV), vec2ResUb, INNERCORE_STAGE2,
                        INNERCORE_STAGE_FD_MTE3_V);
                    SetFlag<HardEvent::MTE3_MTE2>(INNERCORE_INTRAATTN_MTE3_MTE2(runInfo.multiCoreIdxMod2));
                }
                if (runInfo.isCrossCoreSplit &&
                    (!runInfo.isFirstS2SplitCore || (runInfo.isFirstS2SplitCore && runInfo.reduceBlockId == 0 &&
                                                     runInfo.s2LoopCount == runInfo.s2LoopLimit))) {
                    AttentionCommon::S2SplitFdStagingLayout stagingLayout = {
                        constInfo.gSize, dTemplateAlign64, GetStagingSlotNum(false),
                        AttentionCommon::FD_BROADCAST_ELEMS_PER_ROW, AttentionCommon::FD_REDUCE_CHUNK_ROWS};
                    uint32_t workspaceIdx = GetCrossCoreWorkspaceIdx(runInfo);
                    int64_t stagingMOffset = GetFaStagingMOffset(runInfo, constInfo);
                    AttentionCommon::StageVec2PartialOAndWait<T>(stagingLayout, crossCoreCombineGm, workspaceIdx,
                                                                 stagingMOffset, runInfo.vec2MRealSize,
                                                                 static_cast<uint32_t>(constInfo.dSizeV), vec2ResUb,
                                                                 INNERCORE_STAGE2, INNERCORE_STAGE_FD_MTE3_V);
                } else if (!runInfo.isCrossCoreSplit && runInfo.s2LoopCount == runInfo.s2LoopLimit) {
                    this->CopyOutAttentionOut(runInfo, constInfo, vec2ResUb, 0, vec2CalcSize);
                }
            }
        }
    } else if (runInfo.s2LoopCount == runInfo.s2LoopLimit) {
        if (unlikely(runInfo.s2LoopCount == 0)) {
            LocalTensor<float> sumUb = this->softmaxSumBufs[runInfo.multiCoreIdxMod2].tensor;
            LastDivNew<T, Q_T, OUTPUT_T, dTemplateAlign64, false>(vec2ResUb, vec2ResUb, sumUb, runInfo.vec2MRealSize,
                                                                  dTemplateAlign64, 1.0);
        }
        if (runInfo.isCrossCoreSplit) {
            AttentionCommon::S2SplitFdStagingLayout stagingLayout = {
                constInfo.gSize, dTemplateAlign64, GetStagingSlotNum(), AttentionCommon::FD_BROADCAST_ELEMS_PER_ROW,
                AttentionCommon::FD_REDUCE_CHUNK_ROWS};
            uint32_t workspaceIdx = GetCrossCoreWorkspaceIdx(runInfo);
            int64_t stagingMOffset = GetFaStagingMOffset(runInfo, constInfo);
            AttentionCommon::StageVec2PartialO<T>(
                stagingLayout, stagingOutGm, workspaceIdx, stagingMOffset, static_cast<uint32_t>(runInfo.vec2MRealSize),
                static_cast<uint32_t>(constInfo.dSizeV), vec2ResUb, INNERCORE_STAGE2, INNERCORE_STAGE_FD_MTE3_V);
        } else {
            this->CopyOutAttentionOut(runInfo, constInfo, vec2ResUb, 0, vec2CalcSize);
        }
    }
    CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CROSSCORE_BMM2);
    SetFlag<HardEvent::MTE3_V>(INNERCORE_STAGE2);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::InitFDBuffers(FdRunInfo &fdRunInfo)
{
    FdRunInfo fdBufferInfo = fdRunInfo;
    if (fdBufferInfo.mNum > AttentionCommon::FD_REDUCE_CHUNK_ROWS) {
        fdBufferInfo.mNum = AttentionCommon::FD_REDUCE_CHUNK_ROWS;
    }
    AttentionCommon::InitFDBuffersStatic<T, dTemplateAlign64>(fdBufferInfo, 0, fdBuffers);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::ProcessFlashDecode(FdRunInfo &fdRunInfo, ConstInfo &constInfo)
{
    InitFDBuffers(fdRunInfo);
    int64_t seqOffset = 0;
    if constexpr (LAYOUT_T == SMLA_LAYOUT::TND) {
        seqOffset = this->cuSeqlensQGm.GetValue(fdRunInfo.bn2Idx);
    } else {
        seqOffset = fdRunInfo.bn2Idx * constInfo.s1Size;
    }
    int64_t attentionOutOffset =
        seqOffset * constInfo.n2GDv + fdRunInfo.mIdx * constInfo.n2GDv + fdRunInfo.mStartIdx * constInfo.dSizeV;
    int64_t softmaxLseOffset = 0;
    if constexpr (LAYOUT_T == SMLA_LAYOUT::TND) {
        softmaxLseOffset = (seqOffset + fdRunInfo.mIdx) * constInfo.gSize + fdRunInfo.mStartIdx;
    } else {
        softmaxLseOffset =
            (fdRunInfo.bn2Idx * constInfo.s1Size + fdRunInfo.mIdx) * constInfo.gSize + fdRunInfo.mStartIdx;
    }
    LocalTensor<T> accumulatedO = this->fdBuffers.accumOut.tensor.template ReinterpretCast<T>();
    LocalTensor<float> lseExpUb = this->fdBuffers.lseExp.tensor.template ReinterpretCast<float>();
    LocalTensor<float> blockMaxUb = this->fdBuffers.blockMax.tensor.template ReinterpretCast<float>();
    LocalTensor<float> blockSumUb = this->fdBuffers.blockSum.tensor.template ReinterpretCast<float>();
    LocalTensor<T> partialOFp32 = this->fdBuffers.partialO.tensor.template ReinterpretCast<T>();
    AttentionCommon::S2SplitFdStagingLayout stagingLayout = {constInfo.gSize, dTemplateAlign64, GetStagingSlotNum(),
                                                             AttentionCommon::FD_BROADCAST_ELEMS_PER_ROW,
                                                             AttentionCommon::FD_REDUCE_CHUNK_ROWS};
    int64_t attentionOutRowStride =
        static_cast<int64_t>(constInfo.dSizeV) + static_cast<int64_t>(constInfo.attentionOutStride) / sizeof(OUTPUT_T);
    int64_t startRow = 0;
    while (startRow < fdRunInfo.mNum) {
        int64_t dealRowCount = AttentionCommon::FD_REDUCE_CHUNK_ROWS;
        if (startRow + dealRowCount > fdRunInfo.mNum) {
            dealRowCount = fdRunInfo.mNum - startRow;
        }
        WaitFlag<HardEvent::MTE3_V>(INNERCORE_FD_MTE3_V);
        if constexpr (IS_BATCH_CONSISTENCY) {
            WaitFlag<HardEvent::MTE3_MTE2>(INNERCORE_FD_MTE3_MTE2);
            AttentionCommon::ReducePairwiseWithLse<T, dTemplateAlign64>(
                stagingLayout, fdStagingBase, fdRunInfo.workspaceIdx, fdRunInfo.workspaceNum,
                static_cast<uint32_t>(fdRunInfo.mStartIdx + startRow), dealRowCount,
                static_cast<uint32_t>(constInfo.dSizeV), accumulatedO, lseExpUb, blockMaxUb, blockSumUb, partialOFp32,
                constInfo.returnSoftmaxLse, softmaxLseGm, softmaxLseOffset + startRow, INNERCORE_FD_V_MTE2(0),
                INNERCORE_FD_V_MTE2(1), INNERCORE_FD_MTE2_V, INNERCORE_LSE_V_MTE3, INNERCORE_LSE_MTE3_V);
        } else {
            AttentionCommon::ReduceWithLse<T, dTemplateAlign64>(
                stagingLayout, fdStagingBase, fdRunInfo.workspaceIdx, fdRunInfo.workspaceNum,
                static_cast<uint32_t>(fdRunInfo.mStartIdx + startRow), dealRowCount,
                static_cast<uint32_t>(constInfo.dSizeV), accumulatedO, lseExpUb, blockMaxUb, blockSumUb, partialOFp32,
                constInfo.returnSoftmaxLse, softmaxLseGm, softmaxLseOffset + startRow, INNERCORE_FD_V_MTE2(0),
                INNERCORE_FD_V_MTE2(1), INNERCORE_FD_MTE2_V, INNERCORE_LSE_V_MTE3, INNERCORE_LSE_MTE3_V);
        }
        RunInfo runInfo;
        runInfo.vec2MRealSize = dealRowCount;
        runInfo.attentionOutOffset = attentionOutOffset + startRow * attentionOutRowStride;
        int64_t vec2CalcSize = dealRowCount * dTemplateAlign64;
        this->CopyOutAttentionOut(runInfo, constInfo, accumulatedO, 0, vec2CalcSize);
        if constexpr (IS_BATCH_CONSISTENCY) {
            SetFlag<HardEvent::MTE3_MTE2>(INNERCORE_FD_MTE3_MTE2);
        }
        SetFlag<HardEvent::MTE3_V>(INNERCORE_FD_MTE3_V);
        startRow += dealRowCount;
    }
}

TEMPLATES_DEF_NO_DEFAULT
template <typename VEC2_RES_T>
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::Bmm2DataCopyOut(RunInfo &runInfo, ConstInfo &constInfo,
                                                                   LocalTensor<VEC2_RES_T> &vec2ResUb,
                                                                   int64_t vec2S1Idx, int64_t vec2CalcSize)
{
    LocalTensor<OUTPUT_T> attenOut;
    int64_t dSizeAligned64 = (int64_t)dTemplateAlign64;

    attenOut.SetAddr(vec2ResUb.address_);
    Cast(attenOut, vec2ResUb, RoundMode::CAST_ROUND, vec2CalcSize);
    SetFlag<HardEvent::V_MTE3>(INNERCORE_STAGE2);
    WaitFlag<HardEvent::V_MTE3>(INNERCORE_STAGE2);

    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockLen = constInfo.dSizeV * sizeof(OUTPUT_T);
    dataCopyParams.srcStride = (dSizeAligned64 - constInfo.dSizeV) >> 4; // 以32B为单位偏移，bf16类型即偏移16个数，右移4
    dataCopyParams.dstStride = constInfo.attentionOutStride;
    dataCopyParams.blockCount = runInfo.vec2MRealSize;

    DataCopyPad(this->attentionOutGm[runInfo.attentionOutOffset], attenOut, dataCopyParams);
}

TEMPLATES_DEF_NO_DEFAULT
template <typename VEC2_RES_T>
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::CopyOutAttentionOut(RunInfo &runInfo, ConstInfo &constInfo,
                                                                       LocalTensor<VEC2_RES_T> &vec2ResUb,
                                                                       int64_t vec2S1Idx, int64_t vec2CalcSize)
{
    this->Bmm2DataCopyOut(runInfo, constInfo, vec2ResUb, vec2S1Idx, vec2CalcSize);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::InitOutputSingleCore(ConstInfo &constInfo)
{
    uint32_t coreNum = GetBlockNum();
    uint32_t vecCoreNum = CV_RATIO * coreNum;
    uint64_t totalOutputSize = 0;

    // n2 = 1, n1 = gn2 = gSize
    if constexpr (LAYOUT_T == SMLA_LAYOUT::BSND) {
        totalOutputSize = constInfo.bSize * constInfo.gSize * constInfo.s1Size * constInfo.dSizeV;
    } else if constexpr (LAYOUT_T == SMLA_LAYOUT::TND) {
        totalOutputSize = constInfo.s1Size * constInfo.gSize * constInfo.dSizeV;
    }

    static constexpr uint32_t ATTEN_OUT_POP_BUF_START_ADDR = 184U * 1024U;
    static constexpr uint32_t ATTEN_OUT_POP_BUF_ELE_SIZE = (32U * 1024U) / sizeof(OUTPUT_T);
    if (coreNum != 0 && totalOutputSize > 0) {
        AttentionCommon::InitOutput<OUTPUT_T, initOutputEventId, ATTEN_OUT_POP_BUF_START_ADDR,
                                    ATTEN_OUT_POP_BUF_ELE_SIZE, false>(this->attentionOutGm, totalOutputSize,
                                                                       vecCoreNum, static_cast<OUTPUT_T>(0));
    }
    if (constInfo.returnSoftmaxLse) {
        uint64_t totalReturnSoftmaxSize = 0;
        if constexpr (LAYOUT_T == SMLA_LAYOUT::BSND) {
            totalReturnSoftmaxSize = constInfo.bSize * constInfo.n2Size * constInfo.s1Size * constInfo.gSize;
        } else if constexpr (LAYOUT_T == SMLA_LAYOUT::TND) {
            totalReturnSoftmaxSize = constInfo.n2Size * constInfo.s1Size * constInfo.gSize; // (N2,T1,G)
        }
        static constexpr uint32_t LSE_POP_BUF_START_ADDR = 216U * 1024U;
        static constexpr uint32_t LSE_POP_BUF_ELE_SIZE = (32U * 1024U) / sizeof(float);
        if (coreNum != 0 && totalReturnSoftmaxSize > 0) {
            AttentionCommon::InitOutput<float, initOutputEventId, LSE_POP_BUF_START_ADDR, LSE_POP_BUF_ELE_SIZE, false>(
                this->softmaxLseGm, totalReturnSoftmaxSize, vecCoreNum, static_cast<float>(0));
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::CleanOutput(__gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse,
                                                               ConstInfo &constInfo)
{
    if ASCEND_IS_AIV {
        this->attentionOutGm.SetGlobalBuffer((__gm__ OUTPUT_T *)attentionOut);
        this->softmaxLseGm.SetGlobalBuffer((__gm__ T *)softmaxLse);
        if (constInfo.needInit == 1) {
            InitOutputSingleCore(constInfo);
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::InitGlobalBuffer(
    __gm__ uint8_t *oriKV, __gm__ uint8_t *cmpKV, __gm__ uint8_t *oriSparseIndices, __gm__ uint8_t *cmpSparseIndices,
    __gm__ uint8_t *oriBlockTable, __gm__ uint8_t *cmpBlockTable, __gm__ uint8_t *sequsedQ, __gm__ uint8_t *sinks,
    __gm__ uint8_t *sequsedOriKv, __gm__ uint8_t *sequsedCmpKv, __gm__ uint8_t *cmpResidualKv)
{
    oriKVGm.SetGlobalBuffer((__gm__ KV_T *)(oriKV));
    if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::PA_BBND) {
        oriBlockTableGm.SetGlobalBuffer((__gm__ int32_t *)oriBlockTable);
    }
    if constexpr (TEMPLATE_MODE == SMLATemplateMode::CSA_TEMPLATE_MODE ||
                  TEMPLATE_MODE == SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        cmpKVGm.SetGlobalBuffer((__gm__ KV_T *)cmpKV);
        if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::PA_BBND) {
            cmpBlockTableGm.SetGlobalBuffer((__gm__ int32_t *)cmpBlockTable);
        }
        cmpSparseIndicesGm.SetGlobalBuffer((__gm__ int32_t *)cmpSparseIndices);
    }

    if constexpr (TEMPLATE_MODE == SMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                  TEMPLATE_MODE == SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        oriSparseIndicesGm.SetGlobalBuffer((__gm__ int32_t *)oriSparseIndices);
    }

    if (sinks != nullptr) {
        sinksGm.SetGlobalBuffer((__gm__ T *)sinks);
        this->isSinks = true;
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::SoftmaxInitBuffer(uint32_t &ubAddr)
{
    constexpr uint32_t softmaxBufSize = 256; // VF单次操作256Byte
    constexpr uint32_t softmaxElems = softmaxBufSize / sizeof(float);
    softmaxSumBufs[0] = {LocalTensor<float>(TPosition::VECIN, ubAddr, softmaxElems), 0};
    ubAddr += softmaxBufSize;
    softmaxSumBufs[1] = {LocalTensor<float>(TPosition::VECIN, ubAddr, softmaxElems), 1};
    ubAddr += softmaxBufSize;
    softmaxMaxBufs[0] = {LocalTensor<float>(TPosition::VECIN, ubAddr, softmaxElems), 0};
    ubAddr += softmaxBufSize;
    softmaxMaxBufs[1] = {LocalTensor<float>(TPosition::VECIN, ubAddr, softmaxElems), 1};
    ubAddr += softmaxBufSize;
    if constexpr (IS_BATCH_CONSISTENCY) {
        softmaxFinalSumBufs[0] = {LocalTensor<float>(TPosition::VECIN, ubAddr, softmaxElems), 0};
        ubAddr += softmaxBufSize;
        softmaxFinalSumBufs[1] = {LocalTensor<float>(TPosition::VECIN, ubAddr, softmaxElems), 1};
        ubAddr += softmaxBufSize;
        softmaxFinalMaxBufs[0] = {LocalTensor<float>(TPosition::VECIN, ubAddr, softmaxElems), 0};
        ubAddr += softmaxBufSize;
        softmaxFinalMaxBufs[1] = {LocalTensor<float>(TPosition::VECIN, ubAddr, softmaxElems), 1};
        ubAddr += softmaxBufSize;
        batchReduceTmpUb = {LocalTensor<float>(TPosition::VECIN, ubAddr, 768),
                            0}; // 768：batchReduceTmpUb申请内存大小为768个float
        ubAddr += 768U * sizeof(float);
    }
    softmaxExpBufs[0] = {LocalTensor<T>(TPosition::VECIN, ubAddr, softmaxBufSize / sizeof(T)), 0};
    ubAddr += softmaxBufSize;
    softmaxExpBufs[1] = {LocalTensor<T>(TPosition::VECIN, ubAddr, softmaxBufSize / sizeof(T)), 1};
    ubAddr += softmaxBufSize;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::InitSinksBuffer(ConstInfo &constInfo)
{
    LocalTensor<T> sinksUb = this->sinksUb.tensor;
    const uint32_t maxN = constInfo.gSize; // N最大支持128, sink shape是[N]
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = 1U;
    dataCopyParams.blockLen = maxN * sizeof(T);
    dataCopyParams.srcStride = 0U;
    dataCopyParams.dstStride = 0U;
    DataCopyPadExtParams<T> padParams;
    DataCopyPad(sinksUb, this->sinksGm, dataCopyParams, padParams);
    SetFlag<AscendC::HardEvent::MTE2_V>(INNERCORE_SINKS_SYNC);
    WaitFlag<AscendC::HardEvent::MTE2_V>(INNERCORE_SINKS_SYNC);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::InitLocalBuffer(ConstInfo &constInfo, uint32_t ubBaseAddr)
{
    uint32_t ubAddr = ubBaseAddr;

    SoftmaxInitBuffer(ubAddr);

    commonUb = {LocalTensor<T>(TPosition::VECIN, ubAddr, 512 / sizeof(T)), 0}; // 512 for common ub size
    ubAddr += 512;                                                             // 512 for common ub offset
    sinksUb = {LocalTensor<T>(TPosition::VECIN, ubAddr, 512 / sizeof(T)), 0};  // 512 for sinks ub size
    ubAddr += 512;                                                             // 512 for sinks ub offset
    if (this->isSinks) {
        InitSinksBuffer(constInfo);
    }

    if constexpr (TEMPLATE_MODE == SMLATemplateMode::CSA_TEMPLATE_MODE ||
                  TEMPLATE_MODE == SMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                  TEMPLATE_MODE == SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        stage0OutBufs[0] = {LocalTensor<Q_T>(TPosition::VECIN, ubAddr, dVTemplateType * 16U),
                            0}; // 输出缓冲区处理16个seq
        ubAddr += dVTemplateType * 16U * sizeof(KV_T);
        stage0OutBufs[1] = {LocalTensor<Q_T>(TPosition::VECIN, ubAddr, dVTemplateType * 16U),
                            1}; // 输出缓冲区处理16个seq
        ubAddr += dVTemplateType * 16U * sizeof(KV_T);
    }
    if (constInfo.returnSoftmaxLse) {
        outLseUbs[0] = {LocalTensor<float>(TPosition::VECIN, ubAddr, 256 / sizeof(float)),
                        0}; // outLseBuf[0]内存申请256B
        ubAddr += 256U;
        outLseUbs[1] = {LocalTensor<float>(TPosition::VECIN, ubAddr, 256 / sizeof(float)),
                        1}; // outLseBuf[1]内存申请256B
        ubAddr += 256U;
    }

    stage1OutBufs[0] = {LocalTensor<Q_T>(TPosition::VECIN, ubAddr, vec1Srcstride * s2BaseSize), 0};
    ubAddr += vec1Srcstride * s2BaseSize * sizeof(Q_T);
    stage1OutBufs[1] = {LocalTensor<Q_T>(TPosition::VECIN, ubAddr, vec1Srcstride * s2BaseSize), 1};
    ubAddr += vec1Srcstride * s2BaseSize * sizeof(Q_T);

    stage2OutBufs = {LocalTensor<T>(TPosition::VECIN, ubAddr, (s1BaseSize / CV_RATIO) * dTemplateAlign64), 0};

    // 显式 flag 初始化 (替代 AllocEventID + 初始 SetFlag)
    SetFlag<HardEvent::MTE3_V>(INNERCORE_STAGE2);
    if constexpr (IS_BATCH_CONSISTENCY) {
        SetFlag<HardEvent::V_MTE2>(INNERCORE_INTRAPARTIALO_V_MTE2);
        SetFlag<HardEvent::V_MTE2>(INNERCORE_REDUCE_MAXSUM_V_MTE2);
    }
    if (constInfo.returnSoftmaxLse) {
        SetFlag<HardEvent::MTE3_V>(INNERCORE_LSE_MTE3_V);
    }
    SetFlag<HardEvent::MTE3_MTE2>(INNERCORE_STAGE0OUT_MTE3_MTE2(0));
    SetFlag<HardEvent::MTE3_MTE2>(INNERCORE_STAGE0OUT_MTE3_MTE2(1));
    SetFlag<HardEvent::MTE3_V>(INNERCORE_STAGE1(0));
    SetFlag<HardEvent::MTE3_V>(INNERCORE_STAGE1(1));
    SetFlag<HardEvent::V_MTE2>(INNERCORE_FD_V_MTE2(0));
    SetFlag<HardEvent::V_MTE2>(INNERCORE_FD_V_MTE2(1));
    SetFlag<HardEvent::MTE3_V>(INNERCORE_FD_MTE3_V);
    if constexpr (IS_BATCH_CONSISTENCY) {
        SetFlag<HardEvent::MTE3_MTE2>(INNERCORE_FD_MTE3_MTE2);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::FreeEvent(ConstInfo &constInfo)
{
    if constexpr (IS_BATCH_CONSISTENCY) {
        WaitFlag<HardEvent::V_MTE2>(INNERCORE_INTRAPARTIALO_V_MTE2);
        WaitFlag<HardEvent::V_MTE2>(INNERCORE_REDUCE_MAXSUM_V_MTE2);
    }
    WaitFlag<HardEvent::MTE3_V>(INNERCORE_STAGE2);
    if (constInfo.returnSoftmaxLse) {
        WaitFlag<HardEvent::MTE3_V>(INNERCORE_LSE_MTE3_V);
    }
    WaitFlag<HardEvent::MTE3_MTE2>(INNERCORE_STAGE0OUT_MTE3_MTE2(0));
    WaitFlag<HardEvent::MTE3_MTE2>(INNERCORE_STAGE0OUT_MTE3_MTE2(1));
    WaitFlag<HardEvent::MTE3_V>(INNERCORE_STAGE1(0));
    WaitFlag<HardEvent::MTE3_V>(INNERCORE_STAGE1(1));
    WaitFlag<HardEvent::V_MTE2>(INNERCORE_FD_V_MTE2(0));
    WaitFlag<HardEvent::V_MTE2>(INNERCORE_FD_V_MTE2(1));
    WaitFlag<HardEvent::MTE3_V>(INNERCORE_FD_MTE3_V);
    if constexpr (IS_BATCH_CONSISTENCY) {
        WaitFlag<HardEvent::MTE3_MTE2>(INNERCORE_FD_MTE3_MTE2);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::GetExtremeValue(T &negativeScalar)
{
    uint32_t tmp1 = NEGATIVE_MIN_VAULE_FP32;
    negativeScalar = *((float *)&tmp1);
}

template <typename T>
__simd_vf__ void GetKVPhyAddrVFPaImpl(__ubuf__ uint32_t *kvPhyAddrUb, __ubuf__ int32_t *sparseIdxUb,
                                      __ubuf__ int32_t *blkTableUb, const uint16_t s2Loop, uint32_t s2Tail,
                                      const uint32_t blockSize, const int16_t shiftRightNum,
                                      const uint32_t sparseBlockSize, const uint32_t kvDim, const uint32_t kvStride)
{
    static const uint16_t s2_num_per_loop = 128;
    static const uint16_t s2_num_per_reg = 64;
    static const uint16_t out_offset_per_loop = 256;
    static const uint16_t out_offset_per_reg = 128;
    static const uint32_t invalid_value = 0xFFFFFFFF;
    Reg::MaskReg preg_all_b32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg add_carry_l_1;
    Reg::MaskReg add_carry_h_1;
    Reg::MaskReg add_carry_l_2;
    Reg::MaskReg add_carry_h_2;
    Reg::MaskReg preg_tail_neg_1_b32;
    Reg::MaskReg preg_tail_neg_2_b32;

    Reg::RegTensor<uint32_t> vreg_kv_stride;
    Reg::RegTensor<uint32_t> vreg_sparse_idx_1;
    Reg::RegTensor<uint32_t> vreg_sparse_idx_2;
    Reg::RegTensor<uint32_t> vreg_block_size;
    Reg::RegTensor<uint32_t> vreg_shift_rights_num;
    Reg::RegTensor<uint32_t> vreg_pa_blk_idx_1;
    Reg::RegTensor<uint32_t> vreg_pa_blk_idx_2;
    Reg::RegTensor<uint32_t> vreg_pa_tmp_1;
    Reg::RegTensor<uint32_t> vreg_pa_tmp_2;
    Reg::RegTensor<uint32_t> vreg_pa_offset_1;
    Reg::RegTensor<uint32_t> vreg_pa_offset_2;
    Reg::RegTensor<uint32_t> vreg_phy_offset_1;
    Reg::RegTensor<uint32_t> vreg_phy_offset_2;
    Reg::RegTensor<uint32_t> vreg_phy_blk_idx_1;
    Reg::RegTensor<uint32_t> vreg_phy_blk_idx_2;

    Reg::RegTensor<uint32_t> vreg_blk_id_mul_stride_h_1;
    Reg::RegTensor<uint32_t> vreg_blk_id_mul_stride_tmp_h_1;
    Reg::RegTensor<uint32_t> vreg_blk_id_mul_stride_l_1;
    Reg::RegTensor<uint32_t> vreg_mul_overflow_l_1;
    Reg::RegTensor<uint32_t> vreg_total_offset_l_1;
    Reg::RegTensor<uint32_t> vreg_total_offset_h_1;

    Reg::RegTensor<uint32_t> vreg_blk_id_mul_stride_h_2;
    Reg::RegTensor<uint32_t> vreg_blk_id_mul_stride_tmp_h_2;
    Reg::RegTensor<uint32_t> vreg_blk_id_mul_stride_l_2;
    Reg::RegTensor<uint32_t> vreg_mul_overflow_l_2;
    Reg::RegTensor<uint32_t> vreg_total_offset_l_2;
    Reg::RegTensor<uint32_t> vreg_total_offset_h_2;

    Reg::RegTensor<uint32_t> vreg_zero;
    Reg::Duplicate(vreg_zero, 0);
    Reg::Duplicate(vreg_kv_stride, kvStride);

    for (; s2Loop > 1;) {
        for (uint16_t i = 0; i < s2Loop - 1; i++) {
            Reg::LoadAlign<int32_t, Reg::LoadDist::DIST_NORM>((Reg::RegTensor<int32_t> &)vreg_sparse_idx_1,
                                                              sparseIdxUb + i * s2_num_per_loop);
            Reg::LoadAlign<int32_t, Reg::LoadDist::DIST_NORM>((Reg::RegTensor<int32_t> &)vreg_sparse_idx_2,
                                                              sparseIdxUb + s2_num_per_reg + i * s2_num_per_loop);
            // * sparseBlockSize
            Reg::Muls(vreg_sparse_idx_1, vreg_sparse_idx_1, sparseBlockSize, preg_all_b32);
            Reg::Muls(vreg_sparse_idx_2, vreg_sparse_idx_2, sparseBlockSize, preg_all_b32);
            // 计算右移位数
            // 右移 -> 除blockSize 得到paBlockIdx，vreg_sparse_idx - pa_idx * blocksize -> pa offset
            Reg::ShiftRights(vreg_pa_blk_idx_1, vreg_sparse_idx_1, shiftRightNum, preg_all_b32);
            Reg::ShiftRights(vreg_pa_blk_idx_2, vreg_sparse_idx_2, shiftRightNum, preg_all_b32);

            Reg::Muls(vreg_pa_tmp_1, vreg_pa_blk_idx_1, blockSize, preg_all_b32);
            Reg::Muls(vreg_pa_tmp_2, vreg_pa_blk_idx_2, blockSize, preg_all_b32);
            // offset
            Reg::Sub(vreg_pa_offset_1, vreg_sparse_idx_1, vreg_pa_tmp_1, preg_all_b32);
            Reg::Sub(vreg_pa_offset_2, vreg_sparse_idx_2, vreg_pa_tmp_2, preg_all_b32);
            // 物理页内offset
            Reg::Muls(vreg_phy_offset_1, vreg_pa_offset_1, kvDim, preg_all_b32);
            Reg::Muls(vreg_phy_offset_2, vreg_pa_offset_2, kvDim, preg_all_b32);

            // int32 paBlockId -> 物理id
            DataCopyGather(vreg_phy_blk_idx_1, blkTableUb, vreg_pa_blk_idx_1, preg_all_b32);
            DataCopyGather(vreg_phy_blk_idx_2, blkTableUb, vreg_pa_blk_idx_2, preg_all_b32);

            // 分高低32位计算int64物理地址 -- 乘 stride
            // 低位乘 带进位
            Reg::Mull(vreg_blk_id_mul_stride_l_1, vreg_mul_overflow_l_1, vreg_phy_blk_idx_1, vreg_kv_stride,
                      preg_all_b32);
            Reg::Mull(vreg_blk_id_mul_stride_l_2, vreg_mul_overflow_l_2, vreg_phy_blk_idx_2, vreg_kv_stride,
                      preg_all_b32);

            // 分高低32位计算int64物理地址 -- 加 offset
            Reg::Add(add_carry_l_1, vreg_total_offset_l_1, vreg_blk_id_mul_stride_l_1, vreg_phy_offset_1, preg_all_b32);
            Reg::Add(add_carry_l_2, vreg_total_offset_l_2, vreg_blk_id_mul_stride_l_2, vreg_phy_offset_2, preg_all_b32);

            Reg::AddC(add_carry_h_1, vreg_total_offset_h_1, vreg_mul_overflow_l_1, vreg_zero, add_carry_l_1,
                      preg_all_b32);
            Reg::AddC(add_carry_h_2, vreg_total_offset_h_2, vreg_mul_overflow_l_2, vreg_zero, add_carry_l_2,
                      preg_all_b32);

            // 搬出 由于拆分为了int32类型，元素个数翻倍
            Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_INTLV_B32>(
                kvPhyAddrUb + i * out_offset_per_loop, vreg_total_offset_l_1, vreg_total_offset_h_1, preg_all_b32);
            Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_INTLV_B32>(
                kvPhyAddrUb + out_offset_per_reg + i * out_offset_per_loop, vreg_total_offset_l_2,
                vreg_total_offset_h_2, preg_all_b32);
        }
        break;
    }

    for (uint16_t i = s2Loop - 1; i < s2Loop; i++) {
        Reg::MaskReg preg_tail_1_b32 = Reg::UpdateMask<int32_t>(s2Tail);
        Reg::MaskReg preg_tail_2_b32 = Reg::UpdateMask<int32_t>(s2Tail);
        Reg::Not(preg_tail_neg_1_b32, preg_tail_1_b32, preg_all_b32);
        Reg::Not(preg_tail_neg_2_b32, preg_tail_2_b32, preg_all_b32);

        Reg::LoadAlign<int32_t, Reg::LoadDist::DIST_NORM>((Reg::RegTensor<int32_t> &)vreg_sparse_idx_1,
                                                          sparseIdxUb + i * s2_num_per_loop);
        Reg::LoadAlign<int32_t, Reg::LoadDist::DIST_NORM>((Reg::RegTensor<int32_t> &)vreg_sparse_idx_2,
                                                          sparseIdxUb + s2_num_per_reg + i * s2_num_per_loop);
        // * sparseBlockSize
        Reg::Muls(vreg_sparse_idx_1, vreg_sparse_idx_1, sparseBlockSize, preg_tail_1_b32);
        Reg::Muls(vreg_sparse_idx_2, vreg_sparse_idx_2, sparseBlockSize, preg_tail_2_b32);
        // 计算右移位数
        // 右移 -> 除blockSize 得到paBlockIdx，vreg_sparse_idx - pa_idx * blocksize -> pa offset
        Reg::ShiftRights(vreg_pa_blk_idx_1, vreg_sparse_idx_1, shiftRightNum, preg_tail_1_b32);
        Reg::ShiftRights(vreg_pa_blk_idx_2, vreg_sparse_idx_2, shiftRightNum, preg_tail_2_b32);

        Reg::Muls(vreg_pa_tmp_1, vreg_pa_blk_idx_1, blockSize, preg_tail_1_b32);
        Reg::Muls(vreg_pa_tmp_2, vreg_pa_blk_idx_2, blockSize, preg_tail_2_b32);
        // offset
        Reg::Sub(vreg_pa_offset_1, vreg_sparse_idx_1, vreg_pa_tmp_1, preg_tail_1_b32);
        Reg::Sub(vreg_pa_offset_2, vreg_sparse_idx_2, vreg_pa_tmp_2, preg_tail_2_b32);
        // 物理页内offset
        Reg::Muls(vreg_phy_offset_1, vreg_pa_offset_1, kvDim, preg_tail_1_b32);
        Reg::Muls(vreg_phy_offset_2, vreg_pa_offset_2, kvDim, preg_tail_2_b32);

        // int32 paBlockId -> 物理id
        DataCopyGather(vreg_phy_blk_idx_1, blkTableUb, vreg_pa_blk_idx_1, preg_tail_1_b32);
        DataCopyGather(vreg_phy_blk_idx_2, blkTableUb, vreg_pa_blk_idx_2, preg_tail_2_b32);

        // 分高低32位计算int64物理地址 -- 乘 stride
        // 低位乘 带进位
        Reg::Mull(vreg_blk_id_mul_stride_l_1, vreg_mul_overflow_l_1, vreg_phy_blk_idx_1, vreg_kv_stride,
                  preg_tail_1_b32);
        Reg::Mull(vreg_blk_id_mul_stride_l_2, vreg_mul_overflow_l_2, vreg_phy_blk_idx_2, vreg_kv_stride,
                  preg_tail_2_b32);

        // 分高低32位计算int64物理地址 -- 加 offset
        Reg::Add(add_carry_l_1, vreg_total_offset_l_1, vreg_blk_id_mul_stride_l_1, vreg_phy_offset_1, preg_tail_1_b32);
        Reg::Add(add_carry_l_2, vreg_total_offset_l_2, vreg_blk_id_mul_stride_l_2, vreg_phy_offset_2, preg_tail_2_b32);

        Reg::AddC(add_carry_h_1, vreg_total_offset_h_1, vreg_mul_overflow_l_1, vreg_zero, add_carry_l_1,
                  preg_tail_1_b32);
        Reg::AddC(add_carry_h_2, vreg_total_offset_h_2, vreg_mul_overflow_l_2, vreg_zero, add_carry_l_2,
                  preg_tail_2_b32);

        // 无效值填充-1(0xFFFFFFFF)
        Reg::Duplicate<uint32_t, Reg::MaskMergeMode::MERGING>(vreg_total_offset_l_1, invalid_value,
                                                              preg_tail_neg_1_b32);
        Reg::Duplicate<uint32_t, Reg::MaskMergeMode::MERGING>(vreg_total_offset_h_1, invalid_value,
                                                              preg_tail_neg_1_b32);
        Reg::Duplicate<uint32_t, Reg::MaskMergeMode::MERGING>(vreg_total_offset_l_2, invalid_value,
                                                              preg_tail_neg_2_b32);
        Reg::Duplicate<uint32_t, Reg::MaskMergeMode::MERGING>(vreg_total_offset_h_2, invalid_value,
                                                              preg_tail_neg_2_b32);
        Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_INTLV_B32>(
            kvPhyAddrUb + i * out_offset_per_loop, vreg_total_offset_l_1, vreg_total_offset_h_1, preg_all_b32);
        Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_INTLV_B32>(
            kvPhyAddrUb + out_offset_per_reg + i * out_offset_per_loop, vreg_total_offset_l_2, vreg_total_offset_h_2,
            preg_all_b32);
    }
}

template <typename T>
__aicore__ inline void GetKVPhyAddrVFPa(LocalTensor<uint32_t> kvPhyAddrTensor, LocalTensor<int32_t> sparseIdxTensor,
                                        LocalTensor<int32_t> blkTableTensor, const uint16_t s2Loop,
                                        const uint32_t s2Tail, const uint32_t blockSize, const int16_t shiftRightNum,
                                        const uint32_t sparseBlockSize, const uint32_t kvDim, const uint32_t kvStride)
{
    __ubuf__ uint32_t *kv_phy_addr_ub = (__ubuf__ uint32_t *)(kvPhyAddrTensor.GetPhyAddr());
    __ubuf__ int32_t *sparse_idx_ub = (__ubuf__ int32_t *)(sparseIdxTensor.GetPhyAddr());
    __ubuf__ int32_t *blk_table_ub = (__ubuf__ int32_t *)(blkTableTensor.GetPhyAddr());
    GetKVPhyAddrVFPaImpl<uint32_t>(kv_phy_addr_ub, sparse_idx_ub, blk_table_ub, s2Loop, s2Tail, blockSize,
                                   shiftRightNum, sparseBlockSize, kvDim, kvStride);
}

template <typename T>
__simd_vf__ void GetKVPhyAddrVFTndImpl(__ubuf__ uint32_t *kvPhyAddrUb, __ubuf__ int32_t *sparseIdxUb,
                                       const uint16_t s2Loop, uint32_t s2Tail, const uint32_t sparseBlockSize,
                                       const uint32_t kvDim, const uint32_t kvPrefix)
{
    static const uint16_t s2_num_per_loop = 128;
    static const uint16_t s2_num_per_reg = 64;
    static const uint16_t out_offset_per_loop = 256;
    static const uint16_t out_offset_per_reg = 128;
    static const uint32_t invalid_value = 0xFFFFFFFF;
    Reg::MaskReg preg_all_b32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg preg_tail_neg_1_b32;
    Reg::MaskReg preg_tail_neg_2_b32;

    Reg::RegTensor<uint32_t> vreg_sparse_idx_1;
    Reg::RegTensor<uint32_t> vreg_sparse_idx_2;
    Reg::RegTensor<uint32_t> vreg_kv_prefix;
    Reg::RegTensor<uint32_t> vreg_kv_dim;
    Reg::RegTensor<uint32_t> vreg_sum_1;
    Reg::RegTensor<uint32_t> vreg_sum_2;
    Reg::RegTensor<uint32_t> vreg_mul_overflow_l_1;
    Reg::RegTensor<uint32_t> vreg_mul_overflow_l_2;
    Reg::RegTensor<uint32_t> vreg_total_offset_l_1;
    Reg::RegTensor<uint32_t> vreg_total_offset_h_1;
    Reg::RegTensor<uint32_t> vreg_total_offset_l_2;
    Reg::RegTensor<uint32_t> vreg_total_offset_h_2;

    Reg::Duplicate(vreg_kv_prefix, kvPrefix);
    Reg::Duplicate(vreg_kv_dim, kvDim);

    for (; s2Loop > 1;) {
        for (uint16_t i = 0; i < s2Loop - 1; i++) {
            Reg::LoadAlign<int32_t, Reg::LoadDist::DIST_NORM>((Reg::RegTensor<int32_t> &)vreg_sparse_idx_1,
                                                              sparseIdxUb + i * s2_num_per_loop);
            Reg::LoadAlign<int32_t, Reg::LoadDist::DIST_NORM>((Reg::RegTensor<int32_t> &)vreg_sparse_idx_2,
                                                              sparseIdxUb + s2_num_per_reg + i * s2_num_per_loop);
            // * sparseBlockSize
            Reg::Muls(vreg_sparse_idx_1, vreg_sparse_idx_1, sparseBlockSize, preg_all_b32);
            Reg::Muls(vreg_sparse_idx_2, vreg_sparse_idx_2, sparseBlockSize, preg_all_b32);
            // (kvPrefix + sparseIdx) * kvDim -> int64 物理地址
            Reg::Add(vreg_sum_1, vreg_sparse_idx_1, vreg_kv_prefix, preg_all_b32);
            Reg::Add(vreg_sum_2, vreg_sparse_idx_2, vreg_kv_prefix, preg_all_b32);
            // 带进位乘法
            Reg::Mull(vreg_total_offset_l_1, vreg_total_offset_h_1, vreg_sum_1, vreg_kv_dim, preg_all_b32);
            Reg::Mull(vreg_total_offset_l_2, vreg_total_offset_h_2, vreg_sum_2, vreg_kv_dim, preg_all_b32);
            // 搬出
            Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_INTLV_B32>(
                kvPhyAddrUb + i * out_offset_per_loop, vreg_total_offset_l_1, vreg_total_offset_h_1, preg_all_b32);
            Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_INTLV_B32>(
                kvPhyAddrUb + out_offset_per_reg + i * out_offset_per_loop, vreg_total_offset_l_2,
                vreg_total_offset_h_2, preg_all_b32);
        }
        break;
    }

    for (uint16_t i = s2Loop - 1; i < s2Loop; i++) {
        Reg::MaskReg preg_tail_1_b32 = Reg::UpdateMask<int32_t>(s2Tail);
        Reg::MaskReg preg_tail_2_b32 = Reg::UpdateMask<int32_t>(s2Tail);
        Reg::Not(preg_tail_neg_1_b32, preg_tail_1_b32, preg_all_b32);
        Reg::Not(preg_tail_neg_2_b32, preg_tail_2_b32, preg_all_b32);

        Reg::LoadAlign<int32_t, Reg::LoadDist::DIST_NORM>((Reg::RegTensor<int32_t> &)vreg_sparse_idx_1,
                                                          sparseIdxUb + i * s2_num_per_loop);
        Reg::LoadAlign<int32_t, Reg::LoadDist::DIST_NORM>((Reg::RegTensor<int32_t> &)vreg_sparse_idx_2,
                                                          sparseIdxUb + s2_num_per_reg + i * s2_num_per_loop);
        // * sparseBlockSize
        Reg::Muls(vreg_sparse_idx_1, vreg_sparse_idx_1, sparseBlockSize, preg_tail_1_b32);
        Reg::Muls(vreg_sparse_idx_2, vreg_sparse_idx_2, sparseBlockSize, preg_tail_2_b32);
        // (kvPrefix + sparseIdx) * kvDim -> int64 物理地址
        Reg::Add(vreg_sum_1, vreg_sparse_idx_1, vreg_kv_prefix, preg_tail_1_b32);
        Reg::Add(vreg_sum_2, vreg_sparse_idx_2, vreg_kv_prefix, preg_tail_2_b32);
        // 带进位乘法
        Reg::Mull(vreg_total_offset_l_1, vreg_total_offset_h_1, vreg_sum_1, vreg_kv_dim, preg_tail_1_b32);
        Reg::Mull(vreg_total_offset_l_2, vreg_total_offset_h_2, vreg_sum_2, vreg_kv_dim, preg_tail_2_b32);
        // 无效值填充-1(0xFFFFFFFF)
        Reg::Duplicate<uint32_t, Reg::MaskMergeMode::MERGING>(vreg_total_offset_l_1, invalid_value,
                                                              preg_tail_neg_1_b32);
        Reg::Duplicate<uint32_t, Reg::MaskMergeMode::MERGING>(vreg_total_offset_h_1, invalid_value,
                                                              preg_tail_neg_1_b32);
        Reg::Duplicate<uint32_t, Reg::MaskMergeMode::MERGING>(vreg_total_offset_l_2, invalid_value,
                                                              preg_tail_neg_2_b32);
        Reg::Duplicate<uint32_t, Reg::MaskMergeMode::MERGING>(vreg_total_offset_h_2, invalid_value,
                                                              preg_tail_neg_2_b32);
        Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_INTLV_B32>(
            kvPhyAddrUb + i * out_offset_per_loop, vreg_total_offset_l_1, vreg_total_offset_h_1, preg_all_b32);
        Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_INTLV_B32>(
            kvPhyAddrUb + out_offset_per_reg + i * out_offset_per_loop, vreg_total_offset_l_2, vreg_total_offset_h_2,
            preg_all_b32);
    }
}

template <typename T>
__aicore__ inline void GetKVPhyAddrVFTnd(LocalTensor<uint32_t> kvPhyAddrTensor, LocalTensor<int32_t> sparseIdxTensor,
                                         const uint16_t s2Loop, const uint32_t s2Tail, const uint32_t sparseBlockSize,
                                         const uint32_t kvDim, const uint32_t kvPrefix)
{
    __ubuf__ uint32_t *kv_phy_addr_ub = (__ubuf__ uint32_t *)(kvPhyAddrTensor.GetPhyAddr());
    __ubuf__ int32_t *sparse_idx_ub = (__ubuf__ int32_t *)(sparseIdxTensor.GetPhyAddr());
    GetKVPhyAddrVFTndImpl<uint32_t>(kv_phy_addr_ub, sparse_idx_ub, s2Loop, s2Tail, sparseBlockSize, kvDim, kvPrefix);
}

template <typename T>
__simd_vf__ void GetKVPhyAddrVFBsndImpl(__ubuf__ uint32_t *kvPhyAddrUb, __ubuf__ int32_t *sparseIdxUb,
                                        const uint16_t s2Loop, uint32_t s2Tail, const uint32_t sparseBlockSize,
                                        const uint32_t kvDim, const uint32_t bS2BaseLow, const uint32_t bS2BaseHigh)
{
    static const uint16_t s2_num_per_loop = 128;
    static const uint16_t s2_num_per_reg = 64;
    static const uint16_t out_offset_per_loop = 256;
    static const uint16_t out_offset_per_reg = 128;
    static const uint32_t invalid_value = 0xFFFFFFFF;
    Reg::MaskReg preg_all_b32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg add_carry_l_1;
    Reg::MaskReg add_carry_h_1;
    Reg::MaskReg add_carry_l_2;
    Reg::MaskReg add_carry_h_2;
    Reg::MaskReg preg_tail_neg_1_b32;
    Reg::MaskReg preg_tail_neg_2_b32;

    Reg::RegTensor<uint32_t> vreg_sparse_idx_1;
    Reg::RegTensor<uint32_t> vreg_sparse_idx_2;
    Reg::RegTensor<uint32_t> vreg_kv_dim;
    Reg::RegTensor<uint32_t> vreg_b_s2_base_low;
    Reg::RegTensor<uint32_t> vreg_b_s2_base_high;
    Reg::RegTensor<uint32_t> vreg_s2_offset_l_1;
    Reg::RegTensor<uint32_t> vreg_s2_offset_l_2;
    Reg::RegTensor<uint32_t> vreg_mul_overflow_l_1;
    Reg::RegTensor<uint32_t> vreg_mul_overflow_l_2;
    Reg::RegTensor<uint32_t> vreg_total_offset_l_1;
    Reg::RegTensor<uint32_t> vreg_total_offset_h_1;
    Reg::RegTensor<uint32_t> vreg_total_offset_l_2;
    Reg::RegTensor<uint32_t> vreg_total_offset_h_2;
    Reg::RegTensor<uint32_t> vreg_zero;

    Reg::Duplicate(vreg_zero, 0);
    Reg::Duplicate(vreg_kv_dim, kvDim);
    Reg::Duplicate(vreg_b_s2_base_low, bS2BaseLow);
    Reg::Duplicate(vreg_b_s2_base_high, bS2BaseHigh);

    for (; s2Loop > 1;) {
        for (uint16_t i = 0; i < s2Loop - 1; i++) {
            Reg::LoadAlign<int32_t, Reg::LoadDist::DIST_NORM>((Reg::RegTensor<int32_t> &)vreg_sparse_idx_1,
                                                              sparseIdxUb + i * s2_num_per_loop);
            Reg::LoadAlign<int32_t, Reg::LoadDist::DIST_NORM>((Reg::RegTensor<int32_t> &)vreg_sparse_idx_2,
                                                              sparseIdxUb + s2_num_per_reg + i * s2_num_per_loop);
            // * sparseBlockSize
            Reg::Muls(vreg_sparse_idx_1, vreg_sparse_idx_1, sparseBlockSize, preg_all_b32);
            Reg::Muls(vreg_sparse_idx_2, vreg_sparse_idx_2, sparseBlockSize, preg_all_b32);
            // sparseIdx * kvDim (带进位乘法)
            Reg::Mull(vreg_s2_offset_l_1, vreg_mul_overflow_l_1, vreg_sparse_idx_1, vreg_kv_dim, preg_all_b32);
            Reg::Mull(vreg_s2_offset_l_2, vreg_mul_overflow_l_2, vreg_sparse_idx_2, vreg_kv_dim, preg_all_b32);
            // s2_offset + bS2Base (int64 + int64)
            Reg::Add(add_carry_l_1, vreg_total_offset_l_1, vreg_s2_offset_l_1, vreg_b_s2_base_low, preg_all_b32);
            Reg::Add(add_carry_l_2, vreg_total_offset_l_2, vreg_s2_offset_l_2, vreg_b_s2_base_low, preg_all_b32);
            Reg::AddC(add_carry_h_1, vreg_total_offset_h_1, vreg_mul_overflow_l_1, vreg_b_s2_base_high, add_carry_l_1,
                      preg_all_b32);
            Reg::AddC(add_carry_h_2, vreg_total_offset_h_2, vreg_mul_overflow_l_2, vreg_b_s2_base_high, add_carry_l_2,
                      preg_all_b32);
            // 搬出
            Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_INTLV_B32>(
                kvPhyAddrUb + i * out_offset_per_loop, vreg_total_offset_l_1, vreg_total_offset_h_1, preg_all_b32);
            Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_INTLV_B32>(
                kvPhyAddrUb + out_offset_per_reg + i * out_offset_per_loop, vreg_total_offset_l_2,
                vreg_total_offset_h_2, preg_all_b32);
        }
        break;
    }

    for (uint16_t i = s2Loop - 1; i < s2Loop; i++) {
        Reg::MaskReg preg_tail_1_b32 = Reg::UpdateMask<int32_t>(s2Tail);
        Reg::MaskReg preg_tail_2_b32 = Reg::UpdateMask<int32_t>(s2Tail);
        Reg::Not(preg_tail_neg_1_b32, preg_tail_1_b32, preg_all_b32);
        Reg::Not(preg_tail_neg_2_b32, preg_tail_2_b32, preg_all_b32);

        Reg::LoadAlign<int32_t, Reg::LoadDist::DIST_NORM>((Reg::RegTensor<int32_t> &)vreg_sparse_idx_1,
                                                          sparseIdxUb + i * s2_num_per_loop);
        Reg::LoadAlign<int32_t, Reg::LoadDist::DIST_NORM>((Reg::RegTensor<int32_t> &)vreg_sparse_idx_2,
                                                          sparseIdxUb + s2_num_per_reg + i * s2_num_per_loop);
        // * sparseBlockSize
        Reg::Muls(vreg_sparse_idx_1, vreg_sparse_idx_1, sparseBlockSize, preg_tail_1_b32);
        Reg::Muls(vreg_sparse_idx_2, vreg_sparse_idx_2, sparseBlockSize, preg_tail_2_b32);
        // sparseIdx * kvDim (带进位乘法)
        Reg::Mull(vreg_s2_offset_l_1, vreg_mul_overflow_l_1, vreg_sparse_idx_1, vreg_kv_dim, preg_tail_1_b32);
        Reg::Mull(vreg_s2_offset_l_2, vreg_mul_overflow_l_2, vreg_sparse_idx_2, vreg_kv_dim, preg_tail_2_b32);
        // s2_offset + bS2Base (int64 + int64)
        Reg::Add(add_carry_l_1, vreg_total_offset_l_1, vreg_s2_offset_l_1, vreg_b_s2_base_low, preg_tail_1_b32);
        Reg::Add(add_carry_l_2, vreg_total_offset_l_2, vreg_s2_offset_l_2, vreg_b_s2_base_low, preg_tail_2_b32);
        Reg::AddC(add_carry_h_1, vreg_total_offset_h_1, vreg_mul_overflow_l_1, vreg_b_s2_base_high, add_carry_l_1,
                  preg_tail_1_b32);
        Reg::AddC(add_carry_h_2, vreg_total_offset_h_2, vreg_mul_overflow_l_2, vreg_b_s2_base_high, add_carry_l_2,
                  preg_tail_2_b32);
        // 无效值填充-1(0xFFFFFFFF)
        Reg::Duplicate<uint32_t, Reg::MaskMergeMode::MERGING>(vreg_total_offset_l_1, invalid_value,
                                                              preg_tail_neg_1_b32);
        Reg::Duplicate<uint32_t, Reg::MaskMergeMode::MERGING>(vreg_total_offset_h_1, invalid_value,
                                                              preg_tail_neg_1_b32);
        Reg::Duplicate<uint32_t, Reg::MaskMergeMode::MERGING>(vreg_total_offset_l_2, invalid_value,
                                                              preg_tail_neg_2_b32);
        Reg::Duplicate<uint32_t, Reg::MaskMergeMode::MERGING>(vreg_total_offset_h_2, invalid_value,
                                                              preg_tail_neg_2_b32);
        Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_INTLV_B32>(
            kvPhyAddrUb + i * out_offset_per_loop, vreg_total_offset_l_1, vreg_total_offset_h_1, preg_all_b32);
        Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_INTLV_B32>(
            kvPhyAddrUb + out_offset_per_reg + i * out_offset_per_loop, vreg_total_offset_l_2, vreg_total_offset_h_2,
            preg_all_b32);
    }
}

template <typename T>
__aicore__ inline void GetKVPhyAddrVFBsnd(LocalTensor<uint32_t> kvPhyAddrTensor, LocalTensor<int32_t> sparseIdxTensor,
                                          const uint16_t s2Loop, const uint32_t s2Tail, const uint32_t sparseBlockSize,
                                          const uint32_t kvDim, const uint32_t bS2BaseLow, const uint32_t bS2BaseHigh)
{
    __ubuf__ uint32_t *kv_phy_addr_ub = (__ubuf__ uint32_t *)(kvPhyAddrTensor.GetPhyAddr());
    __ubuf__ int32_t *sparse_idx_ub = (__ubuf__ int32_t *)(sparseIdxTensor.GetPhyAddr());
    GetKVPhyAddrVFBsndImpl<uint32_t>(kv_phy_addr_ub, sparse_idx_ub, s2Loop, s2Tail, sparseBlockSize, kvDim, bS2BaseLow,
                                     bS2BaseHigh);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline int32_t CSABlockVec<TEMPLATE_ARGS>::GetSeqLen(int32_t bIdx, bool hasActualSeq, bool hasCuSeqlens,
                                                                GlobalTensor<int32_t> &actualSeqGm,
                                                                GlobalTensor<int32_t> &cuSeqlensGm, int64_t defaultSize)
{
    if (hasActualSeq) {
        return actualSeqGm.GetValue(bIdx);
    } else if (hasCuSeqlens) {
        return cuSeqlensGm.GetValue(bIdx + 1) - cuSeqlensGm.GetValue(bIdx);
    } else {
        return defaultSize;
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline PhyAddrValidInfo CSABlockVec<TEMPLATE_ARGS>::CalcPhyAddrValidInfo(bool isOriKv, int32_t actualS1Size,
                                                                                    int32_t actualOriS2Size,
                                                                                    int32_t restoredSize,
                                                                                    ConstInfo &constInfo)
{
    // per-batch执行一次,  per-s1循环内不再判断maskmode
    PhyAddrValidInfo validInfo;
    if constexpr (TEMPLATE_MODE == SMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                  TEMPLATE_MODE == SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        validInfo.oriS2Act = actualOriS2Size;
        if (isOriKv) {
            if (constInfo.oriMaskMode == 0U) {
                validInfo.oriTopkMode = true;
            } else if (constInfo.oriMaskMode == 3U) {
                validInfo.oriRightBias = 0;
            } else {
                validInfo.oriLeftBias =
                    (constInfo.oriWinLeft == -1) ? PhyAddrValidInfo::BIAS_UNBOUND : constInfo.oriWinLeft + 1;
                validInfo.oriRightBias =
                    (constInfo.oriWinRight == -1) ? PhyAddrValidInfo::BIAS_UNBOUND : constInfo.oriWinRight;
            }
        }
    }
    if constexpr (TEMPLATE_MODE == SMLATemplateMode::CSA_TEMPLATE_MODE ||
                  TEMPLATE_MODE == SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        if (!isOriKv) {
            validInfo.cmpTopkMode = (constInfo.cmpMaskMode == 0U);
            validInfo.cmpBase = restoredSize - actualS1Size + 1;
        }
    }
    return validInfo;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline int32_t CSABlockVec<TEMPLATE_ARGS>::CalcCurValidS2(uint32_t bIdx, int32_t s1Idx, int32_t actualS1Size,
                                                                     bool isOriKv, GlobalTensor<int32_t> &cuSeqlensQGm,
                                                                     GlobalTensor<int32_t> &topkLengthGm,
                                                                     ConstInfo &constInfo, int32_t sparseBlockCount,
                                                                     const PhyAddrValidInfo &validInfo)
{
    bool topkMode = false;
    bool hasTopk = false;
    if constexpr (TEMPLATE_MODE == SMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                  TEMPLATE_MODE == SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        if (isOriKv) {
            topkMode = validInfo.oriTopkMode;
            hasTopk = constInfo.hasOriTopkLength;
        }
    }
    if constexpr (TEMPLATE_MODE == SMLATemplateMode::CSA_TEMPLATE_MODE ||
                  TEMPLATE_MODE == SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        if (!isOriKv) {
            topkMode = validInfo.cmpTopkMode;
            hasTopk = constInfo.hasCmpTopkLength;
        }
    }
    if (topkMode) {
        uint64_t topkIdx =
            (LAYOUT_T == SMLA_LAYOUT::TND) ? (cuSeqlensQGm.GetValue(bIdx) + s1Idx) : (bIdx * constInfo.s1Size + s1Idx);
        int32_t topkLen = hasTopk ? topkLengthGm.GetValue(topkIdx) : sparseBlockCount;
        return Min(topkLen, sparseBlockCount);
    }

    if constexpr (TEMPLATE_MODE == SMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                  TEMPLATE_MODE == SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        if (isOriKv) {
            int64_t thr = validInfo.oriS2Act - actualS1Size + 1 + s1Idx;
            int64_t leftBound = Max(thr - validInfo.oriLeftBias, 0);
            int64_t rightBound = Min(thr + validInfo.oriRightBias, static_cast<int64_t>(validInfo.oriS2Act));
            return Min(static_cast<int32_t>(Max(0, rightBound - leftBound)), sparseBlockCount);
        }
    }
    if constexpr (TEMPLATE_MODE == SMLATemplateMode::CSA_TEMPLATE_MODE ||
                  TEMPLATE_MODE == SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        int64_t numerator = Max(validInfo.cmpBase + s1Idx, 0);
        return Min(sparseBlockCount, static_cast<int32_t>(numerator / static_cast<int32_t>(constInfo.cmpRatio)));
    }
    return 0;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::CopyPhyAddrToGm(LocalTensor<uint32_t> kvPhyAddrUb, int64_t bS1Idx,
                                                                   int64_t s1Idx, int64_t validS2, int64_t alignNum,
                                                                   GlobalTensor<uint32_t> &phyAddrGm,
                                                                   uint32_t alignedSparseBlockCount)
{
    constexpr int64_t numPerBlock = 32;
    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = 1U;
    dataCopyParams.blockLen = ((validS2 + alignNum - 1) / alignNum * alignNum) * sizeof(int64_t) / numPerBlock;
    dataCopyParams.srcGap = 0U;
    dataCopyParams.dstGap = 0U;
    DataCopy(phyAddrGm[(bS1Idx + s1Idx) * alignedSparseBlockCount * 2], kvPhyAddrUb, dataCopyParams);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::CopyPaTableToUb(LocalTensor<int32_t> blkTableUb, int64_t bIdx,
                                                                   GlobalTensor<int32_t> &blockTableGm,
                                                                   uint32_t maxBlockNumPerBatch)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = 1U;
    dataCopyParams.blockLen = maxBlockNumPerBatch * sizeof(int32_t);
    dataCopyParams.srcStride = 0U;
    dataCopyParams.dstStride = 0U;
    DataCopyPadExtParams<int32_t> padParams;
    DataCopyPad(blkTableUb, blockTableGm[bIdx * maxBlockNumPerBatch], dataCopyParams, padParams);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::CopySparseIdxToUb(LocalTensor<int32_t> sparseIdxUb, int64_t bS1Idx,
                                                                     int64_t s1Idx, int64_t validS2,
                                                                     GlobalTensor<int32_t> &sparseIndicesGm,
                                                                     uint32_t sparseBlockCount)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = 1U;
    dataCopyParams.blockLen = validS2 * sizeof(int32_t);
    dataCopyParams.srcStride = 0U;
    dataCopyParams.dstStride = 0U;
    DataCopyPadExtParams<int32_t> padParams;
    DataCopyPad(sparseIdxUb, sparseIndicesGm[(bS1Idx + s1Idx) * sparseBlockCount], dataCopyParams, padParams);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::GetKVPhyAddrForKvType(
    uint32_t bN2StartIdx, uint32_t bN2EndIdx, uint32_t gS1StartIdx, uint32_t nextGs1Idx, bool hasActualSeqQlen,
    bool hasCuSeqlensQ, bool hasActualSeqKvlen, bool hasCuSeqlensKv, GlobalTensor<int32_t> actualSeqQlenGm,
    GlobalTensor<int32_t> cuSeqlensQGm, GlobalTensor<int32_t> actualSeqKvlenGm, GlobalTensor<int32_t> cuSeqlensKvGm,
    GlobalTensor<int32_t> topkLengthGm, GlobalTensor<int32_t> cmpResidualKvGm, ConstInfo &constInfo,
    GlobalTensor<int32_t> &blockTableGm, GlobalTensor<int32_t> &sparseIndicesGm, GlobalTensor<uint32_t> &phyAddrGm,
    uint32_t kvStride, uint32_t blockSize, uint32_t maxBlockNumPerBatch, uint32_t sparseBlockCount,
    uint32_t alignedSparseBlockCount, bool isOriKv)
{
    static constexpr uint16_t s2NumPerLoop = 128;
    static constexpr uint32_t vecCoreNum = IS_SPLIT_G ? 4 : 2;
    uint32_t vecCoreIdx = IS_SPLIT_G ? constInfo.aivIdx % 4 : constInfo.aivIdx % 2;
    uint32_t phyAddrUb = 0;
    int16_t shiftRightNum = 0;
    LocalTensor<int32_t> blkTableUb;

    if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::PA_BBND) {
        int32_t blkSize = static_cast<int32_t>(blockSize);
        while (blkSize > 1) {
            blkSize >>= 1;
            shiftRightNum++;
        }
        blkTableUb = LocalTensor<int32_t>(TPosition::VECIN, phyAddrUb, maxBlockNumPerBatch);
        phyAddrUb = CeilAlign(phyAddrUb + maxBlockNumPerBatch * sizeof(int32_t), BUFFER_SIZE_BYTE_32B);
    }
    LocalTensor<int32_t> sparseIdxUb(TPosition::VECIN, phyAddrUb, alignedSparseBlockCount);
    phyAddrUb += alignedSparseBlockCount * sizeof(int32_t);
    LocalTensor<uint32_t> kvPhyAddrUb(TPosition::VECIN, phyAddrUb, alignedSparseBlockCount * 2); // 2 for ori/cmp kv

    // 第一遍: 统计totalValidS1
    int64_t totalValidS1 = 0;
    uint32_t tmpGS1Start = gS1StartIdx;
    for (uint32_t bIdx = bN2StartIdx; bIdx < bN2EndIdx; ++bIdx) {
        bool lastBN = (bIdx == bN2EndIdx - 1);
        int32_t actualS1Size =
            GetSeqLen(bIdx, hasActualSeqQlen, hasCuSeqlensQ, actualSeqQlenGm, cuSeqlensQGm, constInfo.s1Size);
        int32_t s1End = actualS1Size;
        if (lastBN && nextGs1Idx != 0) {
            s1End = nextGs1Idx;
        }

        int64_t bS1IdxBase = 0;
        if constexpr (LAYOUT_T == SMLA_LAYOUT::TND) {
            bS1IdxBase = hasCuSeqlensQ ? cuSeqlensQGm.GetValue(bIdx) : constInfo.s1Size * bIdx;
        } else {
            bS1IdxBase = constInfo.s1Size * bIdx;
        }

        int32_t restoredSize = 0;
        if constexpr (TEMPLATE_MODE == SMLATemplateMode::CSA_TEMPLATE_MODE ||
                      TEMPLATE_MODE == SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
            if (!isOriKv && constInfo.cmpMaskMode != 0) {
                int32_t actualKvSize = GetSeqLen(bIdx, hasActualSeqKvlen, hasCuSeqlensKv, actualSeqKvlenGm,
                                                 cuSeqlensKvGm, constInfo.cmpS2Size);
                restoredSize = actualKvSize * static_cast<int32_t>(constInfo.cmpRatio) + cmpResidualKvGm.GetValue(bIdx);
            }
        }
        int32_t actualOriS2Size = 0;
        if constexpr (TEMPLATE_MODE == SMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                      TEMPLATE_MODE == SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
            if (isOriKv && constInfo.oriMaskMode != 0) {
                actualOriS2Size = GetSeqLen(bIdx, hasActualSeqKvlen, hasCuSeqlensKv, actualSeqKvlenGm, cuSeqlensKvGm,
                                            constInfo.s2Size);
            }
        }
        PhyAddrValidInfo validInfo =
            CalcPhyAddrValidInfo(isOriKv, actualS1Size, actualOriS2Size, restoredSize, constInfo);

        for (int32_t s1Idx = tmpGS1Start; s1Idx < s1End; ++s1Idx) {
            int32_t curValidS2 = CalcCurValidS2(bIdx, s1Idx, actualS1Size, isOriKv, cuSeqlensQGm, topkLengthGm,
                                                constInfo, static_cast<int32_t>(sparseBlockCount), validInfo);
            if (curValidS2 > 0) {
                totalValidS1++;
            }
        }
        tmpGS1Start = 0;
    }

    int64_t s1PerVecCore = totalValidS1 / vecCoreNum;
    int64_t s1Tail = totalValidS1 % vecCoreNum;
    int64_t curStart = s1PerVecCore * vecCoreIdx + Min((int64_t)vecCoreIdx, s1Tail);
    int64_t curCount = s1PerVecCore + (vecCoreIdx < (uint32_t)s1Tail ? 1 : 0);

    if (curCount == 0) {
        return;
    }

    // 第二遍: 实际计算
    int64_t validCounter = 0;
    int64_t processedCount = 0;
    tmpGS1Start = gS1StartIdx;
    bool done = false;

    if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::PA_BBND) {
        SetFlag<AscendC::HardEvent::V_MTE2>(INNERCORE_PHYADDR_BLKTABLE_FREE);
    }
    SetFlag<AscendC::HardEvent::V_MTE2>(INNERCORE_PHYADDR_SPARSEIDX_FREE);
    SetFlag<AscendC::HardEvent::MTE3_V>(INNERCORE_PHYADDR_KVADDR_FREE);
    for (uint32_t bIdx = bN2StartIdx; bIdx < bN2EndIdx && !done; ++bIdx) {
        bool lastBN = (bIdx == bN2EndIdx - 1);
        int32_t actualS1Size =
            GetSeqLen(bIdx, hasActualSeqQlen, hasCuSeqlensQ, actualSeqQlenGm, cuSeqlensQGm, constInfo.s1Size);
        int64_t bS1Idx = 0;
        if constexpr (LAYOUT_T == SMLA_LAYOUT::TND) {
            bS1Idx = hasCuSeqlensQ ? cuSeqlensQGm.GetValue(bIdx) : constInfo.s1Size * bIdx;
        } else {
            bS1Idx = constInfo.s1Size * bIdx;
        }

        int32_t s1End = actualS1Size;
        if (lastBN && nextGs1Idx != 0) {
            s1End = nextGs1Idx;
        }

        // per-batch 参数预计算
        uint32_t kvPrefix = 0;
        uint32_t bS2BaseLow = 0;
        uint32_t bS2BaseHigh = 0;
        if constexpr (LAYOUT_T == SMLA_LAYOUT::TND) {
            kvPrefix = static_cast<uint32_t>(cuSeqlensKvGm.GetValue(bIdx));
        } else {
            uint32_t s2Size =
                isOriKv ? static_cast<uint32_t>(constInfo.s2Size) : static_cast<uint32_t>(constInfo.cmpS2Size);
            uint64_t bS2Base = static_cast<uint64_t>(bIdx) * s2Size * static_cast<uint64_t>(constInfo.dSize);
            bS2BaseLow = static_cast<uint32_t>(bS2Base);
            bS2BaseHigh = static_cast<uint32_t>(bS2Base >> 32U);
        }

        int32_t restoredSize = 0;
        if constexpr (TEMPLATE_MODE == SMLATemplateMode::CSA_TEMPLATE_MODE ||
                      TEMPLATE_MODE == SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
            if (!isOriKv && constInfo.cmpMaskMode != 0) {
                int32_t actualKvSize = GetSeqLen(bIdx, hasActualSeqKvlen, hasCuSeqlensKv, actualSeqKvlenGm,
                                                 cuSeqlensKvGm, constInfo.cmpS2Size);
                restoredSize = actualKvSize * static_cast<int32_t>(constInfo.cmpRatio) + cmpResidualKvGm.GetValue(bIdx);
            }
        }
        int32_t actualOriS2Size = 0;
        if constexpr (TEMPLATE_MODE == SMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                      TEMPLATE_MODE == SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
            if (isOriKv && constInfo.oriMaskMode != 0) {
                actualOriS2Size = GetSeqLen(bIdx, hasActualSeqKvlen, hasCuSeqlensKv, actualSeqKvlenGm, cuSeqlensKvGm,
                                            constInfo.s2Size);
            }
        }
        PhyAddrValidInfo validInfo =
            CalcPhyAddrValidInfo(isOriKv, actualS1Size, actualOriS2Size, restoredSize, constInfo);

        if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::PA_BBND) {
            WaitFlag<AscendC::HardEvent::V_MTE2>(INNERCORE_PHYADDR_BLKTABLE_FREE);
            CopyPaTableToUb(blkTableUb, bIdx, blockTableGm, maxBlockNumPerBatch);
            SetFlag<AscendC::HardEvent::MTE2_V>(INNERCORE_PHYADDR_BLKTABLE_READY);
            WaitFlag<AscendC::HardEvent::MTE2_V>(INNERCORE_PHYADDR_BLKTABLE_READY);
        }

        for (int32_t s1Idx = tmpGS1Start; s1Idx < s1End; ++s1Idx) {
            int32_t curValidS2 = CalcCurValidS2(bIdx, s1Idx, actualS1Size, isOriKv, cuSeqlensQGm, topkLengthGm,
                                                constInfo, static_cast<int32_t>(sparseBlockCount), validInfo);
            if (curValidS2 <= 0) {
                continue;
            }

            if (validCounter < curStart || validCounter >= curStart + curCount) {
                validCounter++;
                continue;
            }
            validCounter++;

            uint16_t s2Loop = (curValidS2 + s2NumPerLoop - 1) / s2NumPerLoop;
            int32_t s2Tail = curValidS2 - (s2Loop - 1) * s2NumPerLoop;
            WaitFlag<AscendC::HardEvent::V_MTE2>(INNERCORE_PHYADDR_SPARSEIDX_FREE);
            CopySparseIdxToUb(sparseIdxUb, bS1Idx, s1Idx, curValidS2, sparseIndicesGm, sparseBlockCount);
            SetFlag<AscendC::HardEvent::MTE2_V>(INNERCORE_PHYADDR_SPARSEIDX_READY);

            WaitFlag<AscendC::HardEvent::MTE2_V>(INNERCORE_PHYADDR_SPARSEIDX_READY);
            WaitFlag<AscendC::HardEvent::MTE3_V>(INNERCORE_PHYADDR_KVADDR_FREE);
            if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::PA_BBND) {
                GetKVPhyAddrVFPa<uint32_t>(kvPhyAddrUb, sparseIdxUb, blkTableUb, s2Loop, s2Tail, blockSize,
                                           shiftRightNum, constInfo.sparseBlockSize,
                                           static_cast<uint32_t>(constInfo.dSize), kvStride);
            } else if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::TND) {
                GetKVPhyAddrVFTnd<uint32_t>(kvPhyAddrUb, sparseIdxUb, s2Loop, s2Tail, constInfo.sparseBlockSize,
                                            static_cast<uint32_t>(constInfo.dSize), kvPrefix);
            } else {
                GetKVPhyAddrVFBsnd<uint32_t>(kvPhyAddrUb, sparseIdxUb, s2Loop, s2Tail, constInfo.sparseBlockSize,
                                             static_cast<uint32_t>(constInfo.dSize), bS2BaseLow, bS2BaseHigh);
            }
            SetFlag<AscendC::HardEvent::V_MTE2>(INNERCORE_PHYADDR_SPARSEIDX_FREE);
            SetFlag<AscendC::HardEvent::V_MTE3>(INNERCORE_PHYADDR_KVADDR_READY);
            WaitFlag<AscendC::HardEvent::V_MTE3>(INNERCORE_PHYADDR_KVADDR_READY);
            CopyPhyAddrToGm(kvPhyAddrUb, bS1Idx, s1Idx, curValidS2, s2NumPerLoop, phyAddrGm, alignedSparseBlockCount);
            SetFlag<AscendC::HardEvent::MTE3_V>(INNERCORE_PHYADDR_KVADDR_FREE);

            processedCount++;
            if (processedCount >= curCount) {
                done = true;
                break;
            }
        }
        if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::PA_BBND) {
            SetFlag<AscendC::HardEvent::V_MTE2>(INNERCORE_PHYADDR_BLKTABLE_FREE);
        }
        tmpGS1Start = 0;
    }
    if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::PA_BBND) {
        WaitFlag<AscendC::HardEvent::V_MTE2>(INNERCORE_PHYADDR_BLKTABLE_FREE);
    }
    WaitFlag<AscendC::HardEvent::V_MTE2>(INNERCORE_PHYADDR_SPARSEIDX_FREE);
    WaitFlag<AscendC::HardEvent::MTE3_V>(INNERCORE_PHYADDR_KVADDR_FREE);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::GetKVPhyAddr(
    uint32_t hasLoad, uint32_t bN2StartIdx, uint32_t bN2EndIdx, uint32_t gS1StartIdx, uint32_t nextGs1Idx,
    bool hasActualSeqQlen, bool hasCuSeqlensQ, bool hasActualSeqOriKvlen, bool hasCuSeqlensOriKv,
    GlobalTensor<int32_t> actualSeqOriKvlenGm, GlobalTensor<int32_t> cuSeqlensOriKvGm,
    GlobalTensor<int32_t> oriTopkLengthGm, bool hasActualSeqCmpKvlen, bool hasCuSeqlensCmpKv,
    GlobalTensor<int32_t> actualSeqCmpKvlenGm, GlobalTensor<int32_t> cuSeqlensCmpKvGm,
    GlobalTensor<int32_t> cmpTopkLengthGm, GlobalTensor<int32_t> cmpResidualKvGm, GlobalTensor<int32_t> actualSeqQlenGm,
    GlobalTensor<int32_t> cuSeqlensQGm, __gm__ uint8_t *workspace, ConstInfo &constInfo)
{
    if (hasLoad == 0) {
        return;
    }

    // GM分配: ori在前, cmp在后
    int64_t v0TotalOffset = 0;
    uint32_t v0ResSize = constInfo.s2BaseSize * constInfo.dSize * sizeof(Q_T);
    if constexpr (IS_SPLIT_G) {
        v0TotalOffset = v0ResSize * 3 * (GetBlockNum() >> 1U);
    } else {
        v0TotalOffset = v0ResSize * 3 * GetBlockNum();
    }

    // SMLA特有: 加上s2RealBuf大小
    constexpr uint32_t TRIPLE_BUFFER_NUM = 3;
    constexpr uint32_t S2_REAL_BUF_LEN = 128;
    v0TotalOffset += TRIPLE_BUFFER_NUM * S2_REAL_BUF_LEN * sizeof(int32_t) * GetBlockNum();

    uint32_t totalBS1 = (LAYOUT_T == SMLA_LAYOUT::TND) ? constInfo.s1Size : (constInfo.bSize * constInfo.s1Size);

    uint64_t oriPhyAddrSize = 0;
    if constexpr (TEMPLATE_MODE == SMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                  TEMPLATE_MODE == SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        oriPhyAddrSize = static_cast<uint64_t>(totalBS1) * constInfo.alignedOriSparseBlockCount * sizeof(int64_t);
        this->oriKvPhyAddrGm.SetGlobalBuffer((__gm__ uint32_t *)(workspace + v0TotalOffset));
    }

    if constexpr (TEMPLATE_MODE == SMLATemplateMode::CSA_TEMPLATE_MODE ||
                  TEMPLATE_MODE == SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        uint64_t cmpPhyAddrSize =
            static_cast<uint64_t>(totalBS1) * constInfo.alignedCmpSparseBlockCount * sizeof(int64_t);
        this->cmpKvPhyAddrGm.SetGlobalBuffer((__gm__ uint32_t *)(workspace + v0TotalOffset + oriPhyAddrSize));
    }

    // ori部分 (先计算)
    if constexpr (TEMPLATE_MODE == SMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                  TEMPLATE_MODE == SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        GetKVPhyAddrForKvType(bN2StartIdx, bN2EndIdx, gS1StartIdx, nextGs1Idx, hasActualSeqQlen, hasCuSeqlensQ,
                              hasActualSeqOriKvlen, hasCuSeqlensOriKv, actualSeqQlenGm, cuSeqlensQGm,
                              actualSeqOriKvlenGm, cuSeqlensOriKvGm, oriTopkLengthGm, cmpResidualKvGm, constInfo,
                              oriBlockTableGm, oriSparseIndicesGm, oriKvPhyAddrGm, constInfo.oriKeyStride0,
                              constInfo.oriBlockSize, constInfo.oriMaxBlockNumPerBatch, constInfo.oriSparseBlockCount,
                              constInfo.alignedOriSparseBlockCount, true);
    }

    // cmp部分 (后计算)
    if constexpr (TEMPLATE_MODE == SMLATemplateMode::CSA_TEMPLATE_MODE ||
                  TEMPLATE_MODE == SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        GetKVPhyAddrForKvType(bN2StartIdx, bN2EndIdx, gS1StartIdx, nextGs1Idx, hasActualSeqQlen, hasCuSeqlensQ,
                              hasActualSeqCmpKvlen, hasCuSeqlensCmpKv, actualSeqQlenGm, cuSeqlensQGm,
                              actualSeqCmpKvlenGm, cuSeqlensCmpKvGm, cmpTopkLengthGm, cmpResidualKvGm, constInfo,
                              cmpBlockTableGm, cmpSparseIndicesGm, cmpKvPhyAddrGm, constInfo.cmpKeyStride0,
                              constInfo.cmpBlockSize, constInfo.cmpMaxBlockNumPerBatch, constInfo.cmpSparseBlockCount,
                              constInfo.alignedCmpSparseBlockCount, false);
    }
}

TEMPLATES_DEF
class CSABlockVecDummy {
public:
    __aicore__ inline CSABlockVecDummy(){};
    __aicore__ inline void CleanOutput(__gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse, ConstInfo &constInfo)
    {}
    __aicore__ inline void InitGlobalBuffer(__gm__ uint8_t *oriKV, __gm__ uint8_t *cmpKV,
                                            __gm__ uint8_t *oriSparseIndices, __gm__ uint8_t *cmpSparseIndices,
                                            __gm__ uint8_t *oriBlockTable, __gm__ uint8_t *cmpBlockTable,
                                            __gm__ uint8_t *sequsedQ, __gm__ uint8_t *sinks,
                                            __gm__ uint8_t *sequsedOriKv, __gm__ uint8_t *sequsedCmpKv,
                                            __gm__ uint8_t *cmpResidualKv)
    {}
    __aicore__ inline void InitVecBlock(__gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *cuSeqlensOriKv,
                                        __gm__ uint8_t *cuSeqlensCmpKv, __gm__ uint8_t *seqUsedOriKV,
                                        __gm__ uint8_t *seqUsedCmpKV, __gm__ uint8_t *cmpResidualKV) {};
    __aicore__ inline void InitS2SplitStaging(Buffer<BufferType::GM, SyncType::NO_SYNC> &fdStaging) {}
    __aicore__ inline void InitS2SplitStaging(Buffer<BufferType::GM, SyncType::NO_SYNC> &intraCoreCombine,
                                              Buffer<BufferType::GM, SyncType::NO_SYNC> &crossCoreCombine)
    {}
    __aicore__ inline void InitLocalBuffer(ConstInfo &constInfo, uint32_t ubBaseAddr) {}
    __aicore__ inline void InitFDBuffers(FdRunInfo &fdRunInfo) {}
    __aicore__ inline void ProcessFlashDecode(FdRunInfo &fdRunInfo, ConstInfo &constInfo) {}
    __aicore__ inline void ProcessVec1(StaticBuffer<Q_T> &outputBuf, StaticBuffer<T> &bmm1ResBuf, RunInfo &runInfo,
                                       ConstInfo &constInfo)
    {}
    __aicore__ inline void ProcessVec2(StaticBuffer<T> &bmm2ResBuf, RunInfo &runInfo, ConstInfo &constInfo) {}
    __aicore__ inline void FreeEvent(ConstInfo &constInfo) {}
};
} // namespace SMLAKernel
#endif // SPARSE_FLASH_MLA_CSA_BLOCK_VECTOR_ARCH35_H
