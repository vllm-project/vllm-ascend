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
 * \file mixed_quant_sparse_flash_mla_csa_block_vector.h
 * \brief
 */
#ifndef MIXED_QUANT_SPARSE_FLASH_MLA_CSA_BLOCK_VECTOR_H
#define MIXED_QUANT_SPARSE_FLASH_MLA_CSA_BLOCK_VECTOR_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "../vendor/c6240b268/attention/common/op_kernel/init_output.h"
#include "util_regbase.h"
#include "mixed_quant_sparse_flash_mla_common_arch35.h"
using AscendC::Reg::StoreDist;
#include "../vendor/c6240b268/attention/sparse_flash_mla/op_kernel/arch35/common/flash_decode.h"
#include "../vendor/c6240b268/attention/sparse_flash_mla/op_kernel/arch35/common/get_kv_phy_addr_vf.h"
#include "../vendor/c6240b268/attention/sparse_flash_mla/op_kernel/arch35/common/smla_vector_common_arch35.h"
#include "../vendor/c6240b268/attention/common/op_kernel/arch35/vf/vf_flash_decode_arch35.h"
#include "../vendor/c6240b268/attention/common/op_kernel/buffers_policy.h"
#include "../vendor/c6240b268/attention/common/op_kernel/attn_buffer_manager.h"
#include "../vendor/c6240b268/attention/common/op_kernel/attn_buffer.h"
#include "../vendor/c6240b268/attention/common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz_sfa.h"
#include "../vendor/c6240b268/attention/common/op_kernel/arch35/vf/vf_flashupdate_new.h"

using namespace AscendC;
using namespace FaVectorApi;
using namespace AscendC::Impl::Detail;
using namespace optiling;
using namespace optiling::detail;
using namespace regbaseutil;
using namespace matmul;
using namespace fa_base_matmul;
using AttentionCommon::FdRunInfo;

namespace BaseApi {

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
    static constexpr uint32_t dVTemplateTypeInput = 608; // Dsize
    static constexpr uint32_t v0BufferDSize = 640;
    static constexpr uint32_t dCombineBytes = 576; // rope(64 * 2) + nope(448)
    static constexpr uint32_t scaleBytes = 8;
    static constexpr float R0 = 1.0f;
    static constexpr uint64_t SYNC_SINKS_BUF_FLAG = 6;
    static constexpr uint32_t DATABLOCK_BYTES = 32;
    static constexpr uint32_t uint64Touint8 = sizeof(uint64_t) / sizeof(uint8_t);
    static constexpr uint32_t initOutputEventId = 0U; // attenOut和lse，刷无效行会用到剩余ub，需要加同步
    // Sparse KV搬入/Dequant块大小：每次搬入8行，每16行做一次dequant
    static constexpr int64_t KV_COPYIN_UNIT = 8;   // 每次搬入8行
    static constexpr int64_t KV_DEQUANT_UNIT = 16; // 每次dequant 16行
    bool isSoftmaxLseGmValid = false;              // 标记 softmaxLseGm 是否已有效 SetGlobalBuffer
    // ==================== Functions ======================
    __aicore__ inline CSABlockVec(){};
    __aicore__ inline void InitVecBlock(__gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *cuSeqlensOriKv,
                                        __gm__ uint8_t *cuSeqlensCmpKv, __gm__ uint8_t *sequsedOriKv,
                                        __gm__ uint8_t *sequsedCmpKv, __gm__ uint8_t *cmpResidualKv)
    {
        if ASCEND_IS_AIV {
            if (cuSeqlensQ != nullptr) {
                cuSeqlensQGm.SetGlobalBuffer((__gm__ int32_t *)cuSeqlensQ);
            }
            if (cuSeqlensOriKv != nullptr) {
                cuSeqlensOriKvGm.SetGlobalBuffer((__gm__ int32_t *)cuSeqlensOriKv);
            }
            if constexpr (TEMPLATE_MODE != QSMLATemplateMode::SWA_TEMPLATE_MODE &&
                          TEMPLATE_MODE != QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE) {
                if (cuSeqlensCmpKv != nullptr) {
                    cuSeqlensCmpKvGm.SetGlobalBuffer((__gm__ int32_t *)cuSeqlensCmpKv);
                }
            }
            if (sequsedOriKv != nullptr) {
                actualSeqOriKvGm.SetGlobalBuffer((__gm__ int32_t *)sequsedOriKv);
            }
            if constexpr (TEMPLATE_MODE != QSMLATemplateMode::SWA_TEMPLATE_MODE &&
                          TEMPLATE_MODE != QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE) {
                if (sequsedCmpKv != nullptr) {
                    actualSeqCmpKvGm.SetGlobalBuffer((__gm__ int32_t *)sequsedCmpKv);
                }
                if (cmpResidualKv != nullptr) {
                    cmpResidualKvGm.SetGlobalBuffer((__gm__ int32_t *)cmpResidualKv);
                }
            }
            this->GetExtremeValue(this->negativeFloatScalar);
        }
    }

    // 初始化LocalTensor
    __aicore__ inline void InitLocalBuffer(ConstInfo<HIGH_PERF> &constInfo, uint32_t ubBaseAddr);
    __aicore__ inline void InitFDBuffers(FdRunInfo &fdRunInfo);
    // 初始化attentionOutGM
    __aicore__ inline void CleanOutput(__gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse,
                                       ConstInfo<HIGH_PERF> &constInfo);
    __aicore__ inline void FreeEvent(ConstInfo<HIGH_PERF> &constInfo);
    __aicore__ inline void InitGlobalBuffer(__gm__ uint8_t *oriKV, __gm__ uint8_t *cmpKV,
                                            __gm__ uint8_t *oriSparseIndices, __gm__ uint8_t *cmpSparseIndices,
                                            __gm__ uint8_t *oriBlockTable, __gm__ uint8_t *cmpBlockTable,
                                            __gm__ uint8_t *sequsedQ, __gm__ uint8_t *sinks,
                                            __gm__ uint8_t *sequsedOriKv, __gm__ uint8_t *sequsedCmpKv,
                                            __gm__ uint8_t *cmpResidualKv);
    __aicore__ inline void InitOutputSingleCore(ConstInfo<HIGH_PERF> &constInfo);
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
    using mm2ResPos = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;
    __aicore__ inline void ProcessFlashDecode(FdRunInfo &fdRunInfo, ConstInfo<HIGH_PERF> &constInfo);
    __aicore__ inline void ProcessVec0(Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
                                       const RunInfo<HIGH_PERF> &runInfo, ConstInfo<HIGH_PERF> &constInfo);
    __aicore__ inline void ProcessVec1(StaticBuffer<Q_T> &outputBuf, StaticBuffer<T> &bmm1ResBuf,
                                       RunInfo<HIGH_PERF> &runInfo, ConstInfo<HIGH_PERF> &constInfo);
    __aicore__ inline void ProcessVec2(StaticBuffer<T> &bmm2ResBuf, RunInfo<HIGH_PERF> &runInfo,
                                       ConstInfo<HIGH_PERF> &constInfo);
    __aicore__ inline void GetKVPhyAddr(uint32_t hasLoad, uint32_t bN2StartIdx, uint32_t bN2EndIdx,
                                        uint32_t gS1StartIdx, uint32_t nextGs1Idx, bool hasActualSeqQlen,
                                        bool hasCuSeqlensQ, bool hasActualSeqOriKvlen, bool hasCuSeqlensOriKv,
                                        GlobalTensor<int32_t> &actualSeqOriKvlenGm,
                                        GlobalTensor<int32_t> &cuSeqlensOriKvGm, GlobalTensor<int32_t> &oriTopkLengthGm,
                                        bool hasActualSeqCmpKvlen, bool hasCuSeqlensCmpKv,
                                        GlobalTensor<int32_t> &actualSeqCmpKvlenGm,
                                        GlobalTensor<int32_t> &cuSeqlensCmpKvGm, GlobalTensor<int32_t> &cmpTopkLengthGm,
                                        GlobalTensor<int32_t> &cmpResidualKvGm, GlobalTensor<int32_t> &actualSeqQlenGm,
                                        GlobalTensor<int32_t> &cuSeqlensQGm, __gm__ uint8_t *workspace,
                                        ConstInfo<HIGH_PERF> &constInfo);

private:
    __aicore__ inline uint32_t GetStagingSlotNum(bool isInner = false) const
    {
        if constexpr (!IS_BATCH_CONSISTENCY) {
            if constexpr (IS_SPLIT_G) {
                return AttentionCommon::FD_MAX_S2_SPLIT_NUM * (GetBlockNum() >> 1U);
            } else {
                return AttentionCommon::FD_MAX_S2_SPLIT_NUM * GetBlockNum();
            }
        } else {
            if constexpr (IS_SPLIT_G) {
                if (isInner) { // 核内
                    return GetBlockNum();
                }
                return BATCH_CONSISTENCY_MAX_REDUCE_BLOCK_NUM * (GetBlockNum() >> 1U); // 核间
            } else {
                if (isInner) { // 核内
                    return GetBlockNum() << 1U;
                }
                return BATCH_CONSISTENCY_MAX_REDUCE_BLOCK_NUM * GetBlockNum(); // 核间
            }
        }
    }

    __aicore__ inline uint32_t GetCrossCoreWorkspaceIdx(const RunInfo<HIGH_PERF> &runInfo) const
    {
        uint32_t workspaceIdx = static_cast<uint32_t>(runInfo.firstFdDataWorkspaceIdx + runInfo.s2SplitIdx);
        return workspaceIdx;
    }

    __aicore__ inline uint32_t GetIntraCoreWorkspaceIdx(const RunInfo<HIGH_PERF> &runInfo,
                                                        const ConstInfo<HIGH_PERF> &constInfo) const
    {
        uint32_t coreIdx;
        if constexpr (IS_SPLIT_G) {
            coreIdx = static_cast<uint32_t>(constInfo.aivIdx >> 2U);
        } else {
            coreIdx = static_cast<uint32_t>(constInfo.aivIdx >> 1U);
        }
        return (coreIdx << 1U) + runInfo.multiCoreIdxMod2;
    }

    __aicore__ inline int64_t GetFaStagingMOffset(const RunInfo<HIGH_PERF> &runInfo,
                                                  const ConstInfo<HIGH_PERF> &constInfo) const
    {
        int64_t stagingMOffset = (constInfo.subBlockIdx == 1) ? static_cast<int64_t>(runInfo.firstHalfMRealSize) : 0L;
        if constexpr (IS_SPLIT_G) {
            stagingMOffset += static_cast<int64_t>(runInfo.goIdx);
        }
        return stagingMOffset;
    }

    __aicore__ inline void ProcessSparseKv(Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
                                           const RunInfo<HIGH_PERF> &runInfo, ConstInfo<HIGH_PERF> &constInfo);
    __aicore__ inline void ProcessNotSparseKv(Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
                                              const RunInfo<HIGH_PERF> &runInfo, ConstInfo<HIGH_PERF> &constInfo);
    __aicore__ inline void CalProcessSize(const RunInfo<HIGH_PERF> &runInfo, ConstInfo<HIGH_PERF> &constInfo);
    __aicore__ inline void GetKeyOffset(int64_t s2Idx, int64_t &realKeyOffset, int64_t &realScaleOffset,
                                        const RunInfo<HIGH_PERF> &runInfo, ConstInfo<HIGH_PERF> &constInfo);
    template <bool IS_FULL = false>
    __aicore__ inline void GetRealCmpS2Idx(int64_t *tokenData, int64_t s2IdxInBase, const RunInfo<HIGH_PERF> &runInfo,
                                           ConstInfo<HIGH_PERF> &constInfo);
    template <bool IS_FULL = false>
    __aicore__ inline void GetRealS2Addr(int64_t *tokenData, int64_t s2IdxInBase, const RunInfo<HIGH_PERF> &runInfo,
                                         ConstInfo<HIGH_PERF> &constInfo);
    __aicore__ inline void CopyInKvNotSparse(LocalTensor<KV_T> kvMergUb, int64_t v0Loop, int64_t dealRow,
                                             int64_t s2StartIdx, const RunInfo<HIGH_PERF> &runInfo,
                                             ConstInfo<HIGH_PERF> &constInfo);
    template <bool IS_FULL = false>
    __aicore__ inline void CopyInKvSparse(LocalTensor<KV_T> kvInUb, int64_t startRow, int64_t *tokenData,
                                          const RunInfo<HIGH_PERF> &runInfo, ConstInfo<HIGH_PERF> &constInfo);
    template <bool IS_FULL = false>
    __aicore__ inline void CopyIn8Block(LocalTensor<KV_T> kvInUb, int64_t startRow, int64_t &s2,
                                        const RunInfo<HIGH_PERF> &runInfo, ConstInfo<HIGH_PERF> &constInfo);
    __aicore__ inline void DequantAndCopyOutKv(Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
                                               LocalTensor<KV_T> kvInUb, int64_t dealRow, int64_t s2StartIdx,
                                               const RunInfo<HIGH_PERF> &runInfo, ConstInfo<HIGH_PERF> &constInfo);
    __aicore__ inline void DequantKv(LocalTensor<Q_T> antiKvTensorAsB16, LocalTensor<KV_T> srcTensor, int64_t dealRow,
                                     ConstInfo<HIGH_PERF> &constInfo);
    __aicore__ inline void CopyOutKvUb2Gm(Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
                                          LocalTensor<Q_T> antiKvTensorAsB16, int64_t dealRow, int64_t s2StartIdx,
                                          const RunInfo<HIGH_PERF> &runInfo, ConstInfo<HIGH_PERF> &constInfo);
    __aicore__ inline void CopyInSingleKv(LocalTensor<KV_T> kvInUb, int64_t startRow, int64_t keyOffset,
                                          int64_t scaleOffset);
    __aicore__ inline void CopyInSingleKvNoRope(LocalTensor<KV_T> kvInUb, int64_t startRow, int64_t keyOffset,
                                              int64_t scaleOffset);
    /* VEC2_RES_T 表示bmm2ResUb当前的类型，VEC2_RES_T = Q_T那么不需要做Cast。另外，无效行场景当前默认需要做Cast */
    using VEC2_RES_T = T;
    template <typename VEC2_RES_T>
    __aicore__ inline void Bmm2DataCopyOut(RunInfo<HIGH_PERF> &runInfo, ConstInfo<HIGH_PERF> &constInfo,
                                           LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t vec2S1Idx,
                                           int64_t vec2CalcSize = 0);
    template <typename VEC2_RES_T>
    __aicore__ inline void CopyOutAttentionOut(RunInfo<HIGH_PERF> &runInfo, ConstInfo<HIGH_PERF> &constInfo,
                                               LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t vec2S1Idx,
                                               int64_t vec2CalcSize);
    __aicore__ inline void SoftmaxInitBuffer(uint32_t &ubAddr);
    __aicore__ inline void GetExtremeValue(T &negativeScalar);
    __aicore__ inline void InitSinksBuffer(ConstInfo<HIGH_PERF> &constInfo);
    __aicore__ inline int32_t GetSeqLenForPhyAddr(int32_t bIdx, bool hasActualSeq, bool hasCuSeqlens,
                                                  GlobalTensor<int32_t> &actualSeqGm,
                                                  GlobalTensor<int32_t> &cuSeqlensGm, int64_t defaultSize);
    __aicore__ inline PhyAddrValidInfo CalcPhyAddrValidInfo(bool isOriKv, int32_t actualS1Size, int32_t actualOriS2Size,
                                                            int32_t restoredSize, ConstInfo<HIGH_PERF> &constInfo);
    __aicore__ inline int32_t CalcCurValidS2ForPhyAddr(uint32_t bIdx, int32_t s1Idx, int32_t actualS1Size, bool isOriKv,
                                                       GlobalTensor<int32_t> &cuSeqlensQGm,
                                                       GlobalTensor<int32_t> &topkLengthGm,
                                                       ConstInfo<HIGH_PERF> &constInfo, int32_t sparseBlockCount,
                                                       const PhyAddrValidInfo &validInfo);
    __aicore__ inline void GetKVPhyAddrForKvType(
        uint32_t bN2StartIdx, uint32_t bN2EndIdx, uint32_t gS1StartIdx, uint32_t nextGs1Idx, bool hasActualSeqQlen,
        bool hasCuSeqlensQ, bool hasActualSeqKvlen, bool hasCuSeqlensKv, GlobalTensor<int32_t> &actualSeqQlenGm,
        GlobalTensor<int32_t> &cuSeqlensQGm, GlobalTensor<int32_t> &actualSeqKvlenGm,
        GlobalTensor<int32_t> &cuSeqlensKvGm, GlobalTensor<int32_t> &topkLengthGm,
        GlobalTensor<int32_t> &cmpResidualKvGm, ConstInfo<HIGH_PERF> &constInfo, GlobalTensor<int32_t> &blockTableGm,
        GlobalTensor<int32_t> &sparseIndicesGm, GlobalTensor<uint32_t> &phyAddrGm, uint32_t kvStride,
        uint32_t blockSize, uint32_t maxBlockNumPerBatch, uint32_t sparseBlockCount, uint32_t alignedSparseBlockCount,
        bool isOriKv);
    __aicore__ inline void ReduceIntraBlockAndStage(RunInfo<HIGH_PERF> &runInfo, ConstInfo<HIGH_PERF> &constInfo,
                                                    LocalTensor<T> &vec2ResUb, LocalTensor<T> &partialTmpUb);

    GlobalTensor<OUTPUT_T> attentionOutGm;
    GlobalTensor<float> softmaxLseGm;
    GlobalTensor<KV_T> oriKVGm;
    GlobalTensor<KV_T> cmpKVGm;
    GlobalTensor<KV_T> keyGm;
    GlobalTensor<int32_t> cmpSparseIndicesGm;
    GlobalTensor<int32_t> oriSparseIndicesGm;
    GlobalTensor<int32_t> sparseIndicesGm;
    GlobalTensor<int32_t> oriBlockTableGm;
    GlobalTensor<int32_t> cmpBlockTableGm;
    GlobalTensor<int32_t> blockTableGm;
    GlobalTensor<T> sinksGm;
    GlobalTensor<int32_t> cuSeqlensQGm;
    GlobalTensor<int32_t> cuSeqlensKvGm;
    GlobalTensor<int32_t> cuSeqlensOriKvGm;
    GlobalTensor<int32_t> cuSeqlensCmpKvGm;
    GlobalTensor<int32_t> actualSeqOriKvGm;
    GlobalTensor<int32_t> actualSeqCmpKvGm;
    GlobalTensor<int32_t> cmpResidualKvGm;
    GlobalTensor<uint32_t> oriKvPhyAddrGm;
    GlobalTensor<uint32_t> cmpKvPhyAddrGm;

    StaticBuffer<T> commonUb;
    StaticBuffer<T> sinksUb;
    StaticBuffer<Q_T> stage1OutBufs[2];
    StaticBuffer<T> stage2OutBufs;
    StaticBuffer<float> softmaxMaxBufs[2];
    StaticBuffer<float> softmaxSumBufs[2];
    StaticBuffer<float> softmaxFinalMaxBufs[2];
    StaticBuffer<float> softmaxFinalSumBufs[2];
    StaticBuffer<T> softmaxExpBufs[2];
    StaticBuffer<float> batchReduceTmpUb;
    StaticBuffer<float> dequantScaleUb;
    StaticBuffer<float> outLseUbs[2];
    StaticBuffer<KV_T> stage0InBufs[2];
    StaticBuffer<Q_T> stage0OutBufs[2];
    TBuf<> vselrIndexesBuf[2];
    AttentionCommon::FdBuffers<StaticBuffer<uint8_t>> fdBuffers;
    uint32_t pingPongV0 = 0;

    T negativeFloatScalar;
    bool isSinks = false;
    uint32_t maxBlockNumPerBatch;
    uint32_t blockSize;
    int64_t processSize;
    int64_t processS2Start;
    int64_t processS2End;
    __gm__ uint8_t *fdStagingBase = nullptr;
    GlobalTensor<float> stagingOutGm;
    __gm__ uint8_t *intraCoreCombineBase = nullptr;
    GlobalTensor<float> intraCoreCombineGm;
    __gm__ uint8_t *crossCoreCombineBase = nullptr;
    GlobalTensor<float> crossCoreCombineGm;
};

TEMPLATES_DEF_NO_DEFAULT
template <bool IS_FULL>
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::GetRealCmpS2Idx(int64_t *tokenData, int64_t s2IdxInBase,
                                                                   const RunInfo<HIGH_PERF> &runInfo,
                                                                   ConstInfo<HIGH_PERF> &constInfo)
{
    uint32_t curProcessS2End = this->processS2End;
    int64_t sparseBlockCount = 0;
    int64_t curS2LoopCnt = runInfo.s2LoopCount;
    if constexpr (TEMPLATE_MODE == QSMLATemplateMode::CSA_TEMPLATE_MODE) {
        sparseBlockCount = constInfo.cmpSparseBlockCount;
        curS2LoopCnt -= runInfo.oriKvLoopEndIdx;
    } else if constexpr (TEMPLATE_MODE == QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE) {
        sparseBlockCount = constInfo.oriSparseBlockCount;
    } else if constexpr (TEMPLATE_MODE == QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        if (runInfo.isCmp) {
            sparseBlockCount = constInfo.cmpSparseBlockCount;
            curS2LoopCnt -= runInfo.oriKvLoopEndIdx;
        } else {
            sparseBlockCount = constInfo.oriSparseBlockCount;
        }
    }
    uint64_t topkBS1Idx = 0;
    if constexpr (LAYOUT_T == QSMLA_LAYOUT::TND) {
        uint64_t actualSeqQPrefixSum = cuSeqlensQGm.GetValue(runInfo.boIdx);
        topkBS1Idx += (actualSeqQPrefixSum + runInfo.s1oIdx) * sparseBlockCount;
    } else {
        topkBS1Idx += runInfo.boIdx * constInfo.s1Size * sparseBlockCount + runInfo.s1oIdx * sparseBlockCount;
    }

    uint64_t topkKIdx = s2IdxInBase + curS2LoopCnt * constInfo.s2BaseSize;
    uint64_t idxBase = topkBS1Idx + runInfo.s2StartIdx + topkKIdx;
    for (uint64_t i = 0; i < KV_COPYIN_UNIT; ++i) {
        uint64_t idx = idxBase + i;
        if constexpr (!IS_FULL) {
            // 尾块：保留边界判断，防止越界读取
            if (likely(s2IdxInBase + i < curProcessS2End)) {
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
                                                                 const RunInfo<HIGH_PERF> &runInfo,
                                                                 ConstInfo<HIGH_PERF> &constInfo)
{
    uint32_t curProcessS2End = this->processS2End;
    uint32_t alignedSparseBlockCount = 0;
    int64_t curS2LoopCnt = runInfo.s2LoopCount;
    GlobalTensor<int64_t> phyAddrGm;
    if constexpr (TEMPLATE_MODE == QSMLATemplateMode::CSA_TEMPLATE_MODE) {
        alignedSparseBlockCount = constInfo.alignedCmpSparseBlockCount;
        curS2LoopCnt -= runInfo.oriKvLoopEndIdx;
        phyAddrGm = cmpKvPhyAddrGm.template ReinterpretCast<int64_t>();
    } else if constexpr (TEMPLATE_MODE == QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE) {
        alignedSparseBlockCount = constInfo.alignedOriSparseBlockCount;
        phyAddrGm = oriKvPhyAddrGm.template ReinterpretCast<int64_t>();
    } else if constexpr (TEMPLATE_MODE == QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        if (runInfo.isCmp) {
            alignedSparseBlockCount = constInfo.alignedCmpSparseBlockCount;
            curS2LoopCnt -= runInfo.oriKvLoopEndIdx;
            phyAddrGm = cmpKvPhyAddrGm.template ReinterpretCast<int64_t>();
        } else {
            alignedSparseBlockCount = constInfo.alignedOriSparseBlockCount;
            phyAddrGm = oriKvPhyAddrGm.template ReinterpretCast<int64_t>();
        }
    }
    uint64_t topkBS1Idx = 0;
    if constexpr (LAYOUT_T == QSMLA_LAYOUT::TND) {
        topkBS1Idx = (cuSeqlensQGm.GetValue(runInfo.boIdx) + runInfo.s1oIdx) * alignedSparseBlockCount;
    } else {
        topkBS1Idx = (runInfo.boIdx * constInfo.s1Size + runInfo.s1oIdx) * alignedSparseBlockCount;
    }
    uint64_t topkKIdx = s2IdxInBase + curS2LoopCnt * constInfo.s2BaseSize;
    uint64_t idxBase = topkBS1Idx + runInfo.s2StartIdx + topkKIdx;
    for (uint64_t i = 0; i < KV_COPYIN_UNIT; ++i) {
        uint64_t idx = idxBase + i;
        if constexpr (!IS_FULL) {
            // 尾块：保留边界判断，防止越界读取
            if (likely(s2IdxInBase + i < curProcessS2End)) {
                tokenData[i] = phyAddrGm.GetValue(idx);
            } else {
                break;
            }
        } else {
            // 非尾块：8行均有效，直接读取
            tokenData[i] = phyAddrGm.GetValue(idx);
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::GetKeyOffset(int64_t s2Idx, int64_t &realKeyOffset,
                                                                int64_t &realScaleOffset,
                                                                const RunInfo<HIGH_PERF> &runInfo,
                                                                ConstInfo<HIGH_PERF> &constInfo)
{
    if (s2Idx < 0) {
        return;
    }
    if constexpr (isPa) {
        int64_t blkTableIdx = s2Idx / blockSize;
        int64_t blkTableOffset = s2Idx % blockSize;
        int64_t paBlockStride = runInfo.isCmp ? constInfo.cmpKvStride : constInfo.oriKvStride;
        int32_t physBlockId = blockTableGm.GetValue(runInfo.boIdx * maxBlockNumPerBatch + blkTableIdx);
        if constexpr (QUANT_MODE == SCALE_CONTIGUOUS_MODE::CONTIGUOUS) {
            realKeyOffset = physBlockId * paBlockStride + blkTableOffset * constInfo.dSizeVInput;
        } else {
            const int64_t featureBytes = constInfo.hasRopeInput ? dCombineBytes : 448;
            realKeyOffset = physBlockId * paBlockStride + blkTableOffset * constInfo.n2Size * featureBytes +
                            static_cast<uint64_t>(runInfo.n2oIdx * featureBytes);
            realScaleOffset = physBlockId * paBlockStride +
                              static_cast<int64_t>(blockSize) * constInfo.n2Size * featureBytes +
                              blkTableOffset * scaleBytes;
        }
    } else if constexpr (KV_LAYOUT_T == QSMLA_LAYOUT::TND) {
        int64_t tPrefix =
            runInfo.isCmp ? cuSeqlensCmpKvGm.GetValue(runInfo.boIdx) : cuSeqlensOriKvGm.GetValue(runInfo.boIdx);
        realKeyOffset =
            (tPrefix + s2Idx) * constInfo.n2Size * constInfo.dSizeVInput + runInfo.n2oIdx * constInfo.dSizeVInput;
    } else {
        if (runInfo.isCmp) {
            realKeyOffset = runInfo.boIdx * constInfo.n2Size * constInfo.cmpS2Size * constInfo.dSizeVInput +
                            runInfo.n2oIdx * constInfo.cmpS2Size * constInfo.dSizeVInput +
                            s2Idx * constInfo.dSizeVInput; // BSN(1)D
        } else {
            realKeyOffset = runInfo.boIdx * constInfo.n2Size * constInfo.s2Size * constInfo.dSizeVInput +
                            runInfo.n2oIdx * constInfo.s2Size * constInfo.dSizeVInput +
                            s2Idx * constInfo.dSizeVInput; // BSN(1)D
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::CopyInSingleKv(LocalTensor<KV_T> kvInUb, int64_t startRow,
                                                                  int64_t keyOffset, int64_t scaleOffset)
{
    if (keyOffset < 0) {
        return;
    }
    if constexpr (QUANT_MODE == SCALE_CONTIGUOUS_MODE::CONTIGUOUS) {
        DataCopyExtParams intriParams;
        intriParams.blockCount = 1;
        intriParams.dstStride = 0;
        intriParams.srcStride = 0;
        DataCopyPadExtParams<KV_T> padParams;
        // 当前仅支持COMBINE模式
        uint32_t combineBytes = dVTemplateTypeInput * sizeof(KV_T);
        intriParams.blockLen = combineBytes;
        uint32_t combineDim = combineBytes / sizeof(KV_T);
        uint32_t combineDimAlign = CeilAlign(combineBytes, BUFFER_SIZE_BYTE_32B) / sizeof(KV_T);
        padParams.isPad = true;
        padParams.leftPadding = 0;
        padParams.rightPadding = combineDimAlign - combineDim;
        padParams.paddingValue = 0;
        DataCopyPad(kvInUb[startRow * combineDimAlign], keyGm[keyOffset], intriParams, padParams);
    } else {
        // 当前仅支持COMBINE模式
        uint32_t combineDim = dVTemplateTypeInput / sizeof(KV_T);
        uint32_t combineDimAlign = CeilAlign(dVTemplateTypeInput, BUFFER_SIZE_BYTE_32B) / sizeof(KV_T);
        DataCopyExtParams intriParams;
        intriParams.blockLen = dCombineBytes;
        intriParams.blockCount = 1;
        intriParams.dstStride = 0;
        intriParams.srcStride = 0;

        DataCopyPadExtParams<KV_T> padParams;
        padParams.isPad = true;
        padParams.leftPadding = 0;
        padParams.rightPadding = combineDimAlign - combineDim;
        padParams.paddingValue = 0;
        DataCopyPad(kvInUb[startRow * combineDimAlign], keyGm[keyOffset], intriParams, padParams);

        DataCopyExtParams scaleParams;
        DataCopyPadExtParams<KV_T> scalePadParams;
        scaleParams.blockCount = 1;
        scaleParams.blockLen = scaleBytes;
        scaleParams.srcStride = 0;
        scaleParams.dstStride = 0;
        scalePadParams.isPad = false;
        scalePadParams.leftPadding = 0;
        scalePadParams.rightPadding = 0;
        scalePadParams.paddingValue = 0;
        if (scaleOffset >= 0) {
            DataCopyPad(kvInUb[startRow * combineDimAlign + dCombineBytes], keyGm[scaleOffset], scaleParams,
                        scalePadParams);
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::CopyInSingleKvNoRope(LocalTensor<KV_T> kvInUb, int64_t startRow,
                                                                      int64_t keyOffset, int64_t scaleOffset)
{
    // Normalize each external record to the existing 608-byte UB record on EVERY reuse.
    // Mode 1 UB: [zero RoPE128B | NOPE448B | BF16 scale14B | pad18B].
    // Mode 2 UB: [NOPE448B | zero RoPE128B | E8M0 scale7B | input pad1B | zero pad24B].
    LocalTensor<KV_T> rowUb = kvInUb[startRow * dVTemplateTypeInput];
    auto rowWords = rowUb.template ReinterpretCast<uint16_t>();
    Duplicate(rowWords, static_cast<uint16_t>(0), dVTemplateTypeInput / sizeof(uint16_t));
    // The caller consumed the previous V_MTE2 event before entering this copy phase.
    // This local set/wait orders the NEW vector zeroing before GM->UB writes.
    SetFlag<HardEvent::V_MTE2>(INNERCORE_STAGE0_IN(pingPongV0));
    WaitFlag<HardEvent::V_MTE2>(INNERCORE_STAGE0_IN(pingPongV0));
    if (keyOffset < 0) {
        return;
    }
    DataCopyExtParams params;
    params.blockCount = 1;
    params.srcStride = 0;
    params.dstStride = 0;
    DataCopyPadExtParams<KV_T> padParams;
    padParams.isPad = true;
    padParams.leftPadding = 0;
    padParams.rightPadding = 0;
    padParams.paddingValue = 0;
    if constexpr (QUANT_MODE == SCALE_CONTIGUOUS_MODE::CONTIGUOUS) {
        params.blockLen = 480;
        DataCopyPad(rowUb[128], keyGm[keyOffset], params, padParams);
    } else {
        params.blockLen = 448;
        DataCopyPad(rowUb, keyGm[keyOffset], params, padParams);
        if (scaleOffset >= 0) {
            params.blockLen = scaleBytes;
            padParams.rightPadding = DATABLOCK_BYTES - scaleBytes;
            DataCopyPad(rowUb[dCombineBytes], keyGm[scaleOffset], params, padParams);
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
template <bool IS_FULL>
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::CopyInKvSparse(LocalTensor<KV_T> kvInUb, int64_t startRow,
                                                                  int64_t *tokenData, const RunInfo<HIGH_PERF> &runInfo,
                                                                  ConstInfo<HIGH_PERF> &constInfo)
{
    if (!constInfo.hasRopeInput) {
        for (uint32_t i = 0; i < KV_COPYIN_UNIT; ++i) {
            int64_t keyOffset = -1;
            int64_t scaleOffset = -1;
            if constexpr (QUANT_MODE == SCALE_CONTIGUOUS_MODE::CONTIGUOUS && IS_VEC_S2PHYADDR) {
                keyOffset = tokenData[i];
            } else {
                GetKeyOffset(tokenData[i], keyOffset, scaleOffset, runInfo, constInfo);
            }
            CopyInSingleKvNoRope(kvInUb, startRow + i, keyOffset, scaleOffset);
        }
        return;
    }
    int64_t s2IdLimit = runInfo.s2RealSize;
    s2IdLimit = (runInfo.s2RealSize - runInfo.actualS1Size + runInfo.s1oIdx + 1) / constInfo.cmpRatio;
    for (uint32_t i = 0; i < 8; i += 2) { // 遍历8个元素的数组/缓冲区，每次处理2个元素
        int64_t keyOffset0 = -1;
        int64_t keyOffset1 = -1;
        int64_t scaleOffset0 = -1;
        int64_t scaleOffset1 = -1;
        if constexpr (QUANT_MODE == SCALE_CONTIGUOUS_MODE::CONTIGUOUS && IS_VEC_S2PHYADDR) {
            keyOffset0 = tokenData[i];
            keyOffset1 = tokenData[i + 1];
        } else {
            GetKeyOffset(tokenData[i], keyOffset0, scaleOffset0, runInfo, constInfo);
            GetKeyOffset(tokenData[i + 1], keyOffset1, scaleOffset1, runInfo, constInfo);
        }
        if constexpr (!IS_FULL) {
            // 尾块：提前返回判断
            if (unlikely(keyOffset0 < 0 && keyOffset1 < 0)) {
                return;
            }
        }
        uint32_t combineBytes;
        int64_t keySrcStride;
        int64_t scaleSrcStride;
        if constexpr (QUANT_MODE == SCALE_CONTIGUOUS_MODE::CONTIGUOUS) {
            combineBytes = constInfo.dSizeVInput * sizeof(KV_T);
            keySrcStride =
                (keyOffset0 > keyOffset1 ? (keyOffset0 - keyOffset1) : (keyOffset1 - keyOffset0)) - combineBytes;
        } else {
            keySrcStride =
                (keyOffset0 > keyOffset1 ? (keyOffset0 - keyOffset1) : (keyOffset1 - keyOffset0)) - dCombineBytes;
            scaleSrcStride =
                (scaleOffset0 > scaleOffset1 ? (scaleOffset0 - scaleOffset1) : (scaleOffset1 - scaleOffset0)) -
                scaleBytes;
        }
        if (unlikely(keySrcStride >= INT32_MAX || keySrcStride < 0) || constInfo.sparseBlockSize > 1) {
            // stride溢出、stride为负数、s2超长等异常场景，还原成2条搬运指令
            CopyInSingleKv(kvInUb, startRow, keyOffset0, scaleOffset0);
            CopyInSingleKv(kvInUb, startRow + 1, keyOffset1, scaleOffset1);
        } else {
            DataCopyExtParams intriParams;
            if constexpr (!IS_FULL) {
                // 尾块：根据实际有效条目数设置blockCount,且此处仅有可能存在keyOffset1为-1的情况
                intriParams.blockCount = 1 + (keyOffset1 >= 0);
            } else {
                // 非尾块：两条均有效，blockCount恒为2
                intriParams.blockCount = 2;
            }
            if constexpr (QUANT_MODE == SCALE_CONTIGUOUS_MODE::CONTIGUOUS) {
                intriParams.blockLen = combineBytes;
                intriParams.dstStride = 0;
            } else {
                intriParams.blockLen = dCombineBytes;
                intriParams.dstStride = (dVTemplateTypeInput - dCombineBytes) / DATABLOCK_BYTES;
            }
            intriParams.srcStride = keySrcStride;
            DataCopyPadExtParams<KV_T> padParams;

            int64_t keyOffset;
            if constexpr (!IS_FULL) {
                keyOffset = (keyOffset1 > -1 && keyOffset1 < keyOffset0) ? keyOffset1 : keyOffset0;
            } else {
                // 非尾块：两条均有效，取较小地址作为起始
                keyOffset = keyOffset0 < keyOffset1 ? keyOffset0 : keyOffset1;
            }

            // 当前仅支持COMBINE模式
            uint32_t combineDim;
            uint32_t combineDimAlign;
            if constexpr (QUANT_MODE == SCALE_CONTIGUOUS_MODE::CONTIGUOUS) {
                combineDim = combineBytes / sizeof(KV_T);
                combineDimAlign = CeilAlign(combineBytes, BUFFER_SIZE_BYTE_32B) / sizeof(KV_T);
            } else {
                combineDim = dVTemplateTypeInput / sizeof(KV_T);
                combineDimAlign = CeilAlign(dVTemplateTypeInput, BUFFER_SIZE_BYTE_32B) / sizeof(KV_T);
            }
            padParams.isPad = true;
            padParams.leftPadding = 0;
            padParams.rightPadding = combineDimAlign - combineDim;
            padParams.paddingValue = 0;
            DataCopyPad(kvInUb[startRow * combineDimAlign], keyGm[keyOffset], intriParams, padParams);
            if constexpr (QUANT_MODE == SCALE_CONTIGUOUS_MODE::NONCONTIGUOUS) {
                DataCopyExtParams scaleParams;
                DataCopyPadExtParams<KV_T> scalePadParams;
                scaleParams.blockCount = 1;
                scaleParams.blockLen = scaleBytes;
                scaleParams.srcStride = 0;
                scaleParams.dstStride = 0;
                scalePadParams.isPad = false;
                scalePadParams.leftPadding = 0;
                scalePadParams.rightPadding = 0;
                scalePadParams.paddingValue = 0;

                // feature搬运从较小地址开始，当keyOffset0 > keyOffset1时
                // feature顺序被交换(token1→row0, token0→row1)，scale需同步交换
                // 尾块中keyOffset1可能为-1（仅1条有效），此时不发生交换
                bool swapOrder = (keyOffset0 > -1 && keyOffset1 > -1 && keyOffset0 > keyOffset1);
                if (scaleOffset0 >= 0) {
                    uint32_t dstRow = swapOrder ? (startRow + 1) : startRow;
                    DataCopyPad(kvInUb[dstRow * combineDimAlign + dCombineBytes], keyGm[scaleOffset0], scaleParams,
                                scalePadParams);
                }
                if (scaleOffset1 >= 0) {
                    uint32_t dstRow = swapOrder ? startRow : (startRow + 1);
                    DataCopyPad(kvInUb[dstRow * combineDimAlign + dCombineBytes], keyGm[scaleOffset1], scaleParams,
                                scalePadParams);
                }
            }
        }
        startRow += 2U;
    }
}

// fp8->fp32
static constexpr Reg::CastTrait castTraitFp8_1 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                  Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
// fp8->fp32
static constexpr Reg::CastTrait castTraitFp8_2 = {Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN,
                                                  Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
// fp32->fp16
static constexpr Reg::CastTrait castTraitFp8_3 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                  Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
// fp32->fp16
static constexpr Reg::CastTrait castTraitFp8_4 = {Reg::RegLayout::ONE, Reg::SatMode::NO_SAT,
                                                  Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
template <typename Q_T, typename KV_T>
__simd_vf__ void CastScaleImpl(__ubuf__ float *ubDstAddr, __ubuf__ int8_t *ubSrcAddr, uint32_t dealRowCount)
{
    Reg::RegTensor<fp8_e8m0_t> vScale0;
    Reg::RegTensor<bfloat16_t> vScalebf16Res0;
    Reg::RegTensor<float> vScalefp32Res0;
    __ubuf__ int8_t *ubScaleSrcAddrTemp = ubSrcAddr;
    __ubuf__ float *ubDstAddrTmp = ubDstAddr;
    Reg::MaskReg bf16TypeMaskAll = Reg::CreateMask<bfloat16_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg fp32MaskAll = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    for (uint16_t i = 0; i < static_cast<uint16_t>(dealRowCount); i++) {
        // load scale
        Reg::LoadAlign<int8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_UNPACK4_B8>(
            (Reg::RegTensor<int8_t> &)vScale0, ubScaleSrcAddrTemp, 608);

        Reg::Cast<bfloat16_t, fp8_e8m0_t, castTraitFp8_1>(vScalebf16Res0, vScale0, bf16TypeMaskAll);
        Reg::Cast<float, bfloat16_t, castTraitFp8_1>(vScalefp32Res0, vScalebf16Res0, fp32MaskAll);

        Reg::StoreAlign<float, Reg::PostLiteral::POST_MODE_UPDATE>(ubDstAddrTmp, vScalefp32Res0, 64, bf16TypeMaskAll);
    }
}

template <typename Q_T, typename KV_T>
__aicore__ inline void CastScale(LocalTensor<float> &outputUb, LocalTensor<KV_T> &inputUb, uint32_t dealRowCount)
{
    __ubuf__ float *ubDstAddr = (__ubuf__ float *)(outputUb.GetPhyAddr());
    __ubuf__ int8_t *ubScaleAddr = (__ubuf__ int8_t *)(inputUb[448 + 64 * 2].GetPhyAddr());
    CastScaleImpl<Q_T, KV_T>(ubDstAddr, ubScaleAddr, dealRowCount);
}

template <typename Q_T, typename KV_T>
__simd_vf__ void AntiquantVFImplFp8D448(__ubuf__ Q_T *ubKRopeNzAddr, __ubuf__ int8_t *ubSrcAddr,
                                        __ubuf__ Q_T *ubDstAddr, __ubuf__ Q_T *ubScaleSrcAddr,
                                        __ubuf__ int8_t *ubKRopeAddr, uint32_t dealRowCount)
{
    uint32_t combineDim = 608; // Dsize，32对齐
    Reg::RegTensor<KV_T> vKvData0;
    Reg::RegTensor<KV_T> vKvData1;
    Reg::RegTensor<Q_T> vScale0Bf16;
    Reg::RegTensor<Q_T> vScale1Bf16;
    Reg::RegTensor<half> vKvDataHalf0;
    Reg::RegTensor<half> vKvDataHalf1;
    Reg::RegTensor<float> vCastFp32Res0;
    Reg::RegTensor<float> vCastFp32Res1;
    Reg::RegTensor<Q_T> vMulRes0;
    Reg::RegTensor<Q_T> vMulRes1;
    Reg::RegTensor<Q_T> vCastRes0;
    Reg::RegTensor<Q_T> vCastRes1;
    Reg::RegTensor<Q_T> vCastResPack0;
    Reg::RegTensor<Q_T> vCastResPack1;
    Reg::RegTensor<int8_t> vKvRope;

    Reg::MaskReg kvTypeMaskAll = Reg::CreateMask<KV_T, Reg::MaskPattern::ALL>();
    Reg::MaskReg kvRopeTypeMaskAll = Reg::CreateMask<Q_T, Reg::MaskPattern::ALL>();
    Reg::MaskReg kvRopeTypeMaskHalf = Reg::CreateMask<Q_T, Reg::MaskPattern::H>();
    Reg::MaskReg fp32MaskAll = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::MaskReg fp16MaskAll = Reg::CreateMask<Q_T, Reg::MaskPattern::ALL>();
    uint32_t blockStride = 17; // +1 to solve bank confict
    uint32_t repeatStride = 1;
    const uint32_t nopeDim = 448;
    const uint32_t kvNumPerLoop = 128;
    const uint32_t scaleNumPerLoop = 2;
    const uint32_t tileSize = 64;

    // tilesize is 64, deal 128 b8 kv, deal 2 fp32 scale
    for (uint16_t j = 0; j < (nopeDim / kvNumPerLoop); j++) {
        __ubuf__ int8_t *ubSrcTemp = ubSrcAddr + j * kvNumPerLoop;
        __ubuf__ Q_T *ubScaleSrcAddrTemp = ubScaleSrcAddr + j * scaleNumPerLoop;
        __ubuf__ Q_T *ubDstAddrTmp = ubDstAddr + j * kvNumPerLoop * blockStride;
        for (uint16_t i = 0; i < static_cast<uint16_t>(dealRowCount); i++) {
            // load scale
            Reg::LoadAlign<int8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_UNPACK4_B8>(
                (Reg::RegTensor<int8_t> &)vKvData0, ubSrcTemp, tileSize);
            Reg::LoadAlign<int8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_UNPACK4_B8>(
                (Reg::RegTensor<int8_t> &)vKvData1, ubSrcTemp, combineDim - tileSize);

            Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_BRC_B16>((Reg::RegTensor<bfloat16_t> &)vScale0Bf16,
                                                                    ubScaleSrcAddrTemp + i * (combineDim / 2));
            Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_BRC_B16>((Reg::RegTensor<bfloat16_t> &)vScale1Bf16,
                                                                    ubScaleSrcAddrTemp + 1 + i * (combineDim / 2));

            Reg::Cast<float, KV_T, castTraitFp8_1>(vCastFp32Res0, vKvData0, fp32MaskAll);
            Reg::Cast<float, KV_T, castTraitFp8_1>(vCastFp32Res1, vKvData1, fp32MaskAll);

            Reg::Cast<Q_T, float, castTraitFp8_3>(vCastRes0, vCastFp32Res0, fp16MaskAll);
            Reg::Cast<Q_T, float, castTraitFp8_3>(vCastRes1, vCastFp32Res1, fp16MaskAll);

            Reg::Mul<Q_T, Reg::MaskMergeMode::ZEROING>(vMulRes0, vCastRes0, vScale0Bf16, fp16MaskAll);
            Reg::Mul<Q_T, Reg::MaskMergeMode::ZEROING>(vMulRes1, vCastRes1, vScale1Bf16, fp16MaskAll);

            Reg::DeInterleave(vCastResPack0, vCastResPack1, vMulRes0, vMulRes1);

            Reg::StoreAlign<Q_T, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
                ubDstAddrTmp, vCastResPack0, blockStride, repeatStride, kvRopeTypeMaskAll);
        }
    }

    uint16_t lastLoopOffset = nopeDim / kvNumPerLoop; // 偏移已经处理的循环次数
    __ubuf__ int8_t *ubSrcTemp = ubSrcAddr + lastLoopOffset * kvNumPerLoop;
    __ubuf__ Q_T *ubScaleSrcAddrTemp = ubScaleSrcAddr + lastLoopOffset * scaleNumPerLoop;
    __ubuf__ Q_T *ubDstAddrTmp = ubDstAddr + lastLoopOffset * kvNumPerLoop * blockStride;
    Reg::Duplicate(vCastRes1, 0.0);
    for (uint16_t i = 0; i < static_cast<uint16_t>(dealRowCount); i++) {
        Reg::LoadAlign<int8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_NORM>(
            (Reg::RegTensor<int8_t> &)vKvRope, ubKRopeAddr, combineDim);
        // load scale
        Reg::LoadAlign<int8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_UNPACK4_B8>(
            (Reg::RegTensor<int8_t> &)vKvData0, ubSrcTemp, combineDim);
        Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_BRC_B16>((Reg::RegTensor<bfloat16_t> &)vScale0Bf16,
                                                                ubScaleSrcAddrTemp + i * (combineDim / 2));
        Reg::Cast<float, KV_T, castTraitFp8_1>(vCastFp32Res0, vKvData0, fp32MaskAll);
        Reg::Cast<Q_T, float, castTraitFp8_3>(vCastRes0, vCastFp32Res0, fp16MaskAll);
        Reg::Mul<Q_T, Reg::MaskMergeMode::ZEROING>(vMulRes0, vCastRes0, vScale0Bf16, fp16MaskAll);
        Reg::DeInterleave(vCastResPack0, vCastResPack1, vMulRes0, vCastRes1);

        Reg::StoreAlign<Q_T, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
            ubDstAddrTmp, vCastResPack0, blockStride, repeatStride, kvRopeTypeMaskHalf);
        Reg::StoreAlign<Q_T, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
            ubKRopeNzAddr, (Reg::RegTensor<Q_T> &)vKvRope, blockStride, repeatStride, kvRopeTypeMaskHalf);
    }
}

template <typename Q_T, typename KV_T>
__aicore__ inline void AntiquantVFFp8D448(LocalTensor<Q_T> &kRopeUbNz, LocalTensor<Q_T> &outputUb,
                                          LocalTensor<KV_T> &inputUb, LocalTensor<Q_T> &scaleUb,
                                          LocalTensor<int8_t> &kRopeUb, uint32_t dealRowCount)
{
    const uint32_t nopeDim = 448;
    const uint32_t ropeDim = 64;
    __ubuf__ int8_t *ubSrcAddr = (__ubuf__ int8_t *)(inputUb[64 * sizeof(Q_T)].GetPhyAddr());
    // sizeof(bf16) / sizeof(fp8) = 2
    __ubuf__ Q_T *ubScaleAddr = (__ubuf__ Q_T *)(scaleUb[ropeDim + nopeDim / 2].GetPhyAddr());
    __ubuf__ int8_t *ubKRopeAddr = (__ubuf__ int8_t *)(kRopeUb.GetPhyAddr());
    __ubuf__ Q_T *ubDstAddr = (__ubuf__ Q_T *)(outputUb.GetPhyAddr());
    __ubuf__ Q_T *ubKRopeNzAddr = (__ubuf__ Q_T *)(kRopeUbNz.GetPhyAddr());

    AntiquantVFImplFp8D448<Q_T, KV_T>(ubKRopeNzAddr, ubSrcAddr, ubDstAddr, ubScaleAddr, ubKRopeAddr, dealRowCount);
}

template <typename Q_T, typename KV_T>
__simd_vf__ void AntiquantVFImplFp8D448_FloatScale(__ubuf__ Q_T *ubKRopeNzAddr, __ubuf__ int8_t *ubSrcAddr,
                                                   __ubuf__ Q_T *ubDstAddr, __ubuf__ float *ubScaleSrcAddr,
                                                   __ubuf__ int8_t *ubKRopeAddr, uint32_t dealRowCount)
{
    uint32_t combineDim = 608;
    Reg::RegTensor<KV_T> vKvData0;
    Reg::RegTensor<KV_T> vKvData1;
    Reg::RegTensor<float> vScale0;
    Reg::RegTensor<float> vScale1;
    Reg::RegTensor<float> vCastFp32Res0;
    Reg::RegTensor<float> vCastFp32Res1;
    Reg::RegTensor<float> vMulRes0;
    Reg::RegTensor<float> vMulRes1;
    Reg::RegTensor<Q_T> vCastRes0;
    Reg::RegTensor<Q_T> vCastRes1;
    Reg::RegTensor<Q_T> vCastResPack0;
    Reg::RegTensor<Q_T> vCastResPack1;
    Reg::RegTensor<int8_t> vKvRope;

    Reg::MaskReg kvTypeMaskAll = Reg::CreateMask<KV_T, Reg::MaskPattern::ALL>();
    Reg::MaskReg kvRopeTypeMaskAll = Reg::CreateMask<Q_T, Reg::MaskPattern::ALL>();
    Reg::MaskReg kvRopeTypeMaskHalf = Reg::CreateMask<Q_T, Reg::MaskPattern::H>();
    Reg::MaskReg fp32MaskAll = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    uint32_t blockStride = 17;
    uint32_t repeatStride = 1;
    const uint32_t nopeDim = 448;
    const uint32_t kvNumPerLoop = 128;
    const uint32_t scaleNumPerLoop = 2;
    const uint32_t tileSize = 64;

    for (uint16_t j = 0; j < (nopeDim / kvNumPerLoop); j++) {
        __ubuf__ int8_t *ubSrcTemp = ubSrcAddr + j * kvNumPerLoop;
        __ubuf__ float *ubScaleSrcAddrTemp = ubScaleSrcAddr + j * scaleNumPerLoop;
        __ubuf__ Q_T *ubDstAddrTmp = ubDstAddr + j * kvNumPerLoop * blockStride;
        for (uint16_t i = 0; i < static_cast<uint16_t>(dealRowCount); i++) {
            Reg::LoadAlign<int8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_UNPACK4_B8>(
                (Reg::RegTensor<int8_t> &)vKvData0, ubSrcTemp, tileSize);
            Reg::LoadAlign<int8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_UNPACK4_B8>(
                (Reg::RegTensor<int8_t> &)vKvData1, ubSrcTemp, combineDim - tileSize);

            Reg::LoadAlign<float, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_BRC_B32>(
                (Reg::RegTensor<float> &)vScale0, ubScaleSrcAddrTemp, 1);
            Reg::LoadAlign<float, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_BRC_B32>(
                (Reg::RegTensor<float> &)vScale1, ubScaleSrcAddrTemp, tileSize - 1);

            Reg::Cast<float, KV_T, castTraitFp8_1>(vCastFp32Res0, vKvData0, fp32MaskAll);
            Reg::Cast<float, KV_T, castTraitFp8_1>(vCastFp32Res1, vKvData1, fp32MaskAll);

            Reg::Mul<float, Reg::MaskMergeMode::ZEROING>(vMulRes0, vCastFp32Res0, vScale0, fp32MaskAll);
            Reg::Mul<float, Reg::MaskMergeMode::ZEROING>(vMulRes1, vCastFp32Res1, vScale1, fp32MaskAll);

            Reg::Cast<Q_T, float, castTraitFp8_3>(vCastRes0, vMulRes0, fp32MaskAll);
            Reg::Cast<Q_T, float, castTraitFp8_3>(vCastRes1, vMulRes1, fp32MaskAll);

            Reg::DeInterleave(vCastResPack0, vCastResPack1, vCastRes0, vCastRes1);

            Reg::StoreAlign<Q_T, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
                ubDstAddrTmp, vCastResPack0, blockStride, repeatStride, kvRopeTypeMaskAll);
        }
    }

    uint16_t lastLoopOffset = nopeDim / kvNumPerLoop;
    __ubuf__ int8_t *ubSrcTemp = ubSrcAddr + lastLoopOffset * kvNumPerLoop;
    __ubuf__ float *ubScaleSrcAddrTemp = ubScaleSrcAddr + lastLoopOffset * scaleNumPerLoop;
    __ubuf__ Q_T *ubDstAddrTmp = ubDstAddr + lastLoopOffset * kvNumPerLoop * blockStride;
    Reg::Duplicate(vCastRes1, 0.0);
    for (uint16_t i = 0; i < static_cast<uint16_t>(dealRowCount); i++) {
        Reg::LoadAlign<int8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_NORM>(
            (Reg::RegTensor<int8_t> &)vKvRope, ubKRopeAddr, combineDim);
        Reg::LoadAlign<int8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_UNPACK4_B8>(
            (Reg::RegTensor<int8_t> &)vKvData0, ubSrcTemp, combineDim);
        Reg::LoadAlign<float, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_BRC_B32>(
            (Reg::RegTensor<float> &)vScale0, ubScaleSrcAddrTemp, tileSize);
        Reg::Cast<float, KV_T, castTraitFp8_1>(vCastFp32Res0, vKvData0, fp32MaskAll);
        Reg::Mul<float, Reg::MaskMergeMode::ZEROING>(vMulRes0, vCastFp32Res0, vScale0, fp32MaskAll);
        Reg::Cast<Q_T, float, castTraitFp8_3>(vCastRes0, vMulRes0, fp32MaskAll);
        Reg::DeInterleave(vCastResPack0, vCastResPack1, vCastRes0, vCastRes1);

        Reg::StoreAlign<Q_T, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
            ubDstAddrTmp, vCastResPack0, blockStride, repeatStride, kvRopeTypeMaskHalf);
        Reg::StoreAlign<Q_T, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
            ubKRopeNzAddr, (Reg::RegTensor<Q_T> &)vKvRope, blockStride, repeatStride, kvRopeTypeMaskHalf);
    }
}

template <typename Q_T, typename KV_T>
__aicore__ inline void AntiquantVFFp8D448_FloatScale(LocalTensor<Q_T> &kRopeUbNz, LocalTensor<Q_T> &outputUb,
                                                     LocalTensor<KV_T> &inputUb, LocalTensor<float> &scaleUb,
                                                     LocalTensor<int8_t> &kRopeUb, uint32_t dealRowCount)
{
    __ubuf__ int8_t *ubSrcAddr = (__ubuf__ int8_t *)(inputUb.GetPhyAddr());
    __ubuf__ int8_t *ubKRopeAddr = (__ubuf__ int8_t *)(kRopeUb.GetPhyAddr());
    __ubuf__ Q_T *ubDstAddr = (__ubuf__ Q_T *)(outputUb.GetPhyAddr());
    __ubuf__ Q_T *ubKRopeNzAddr = (__ubuf__ Q_T *)(kRopeUbNz.GetPhyAddr());
    __ubuf__ float *ubScaleAddr = (__ubuf__ float *)(scaleUb.GetPhyAddr());

    AntiquantVFImplFp8D448_FloatScale<Q_T, KV_T>(ubKRopeNzAddr, ubSrcAddr, ubDstAddr, ubScaleAddr, ubKRopeAddr,
                                                 dealRowCount);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::DequantKv(LocalTensor<Q_T> antiKvTensorAsB16,
                                                             LocalTensor<KV_T> srcTensor, int64_t dealRow,
                                                             ConstInfo<HIGH_PERF> &constInfo)
{
    LocalTensor<int8_t> kRopeUb;
    LocalTensor<Q_T> kRopeUbNz = antiKvTensorAsB16[constInfo.dSizeNope * (16 + 1)];
    if constexpr (QUANT_MODE == SCALE_CONTIGUOUS_MODE::CONTIGUOUS) {
        kRopeUb = srcTensor.template ReinterpretCast<int8_t>();
        LocalTensor<Q_T> scaleUb = srcTensor.template ReinterpretCast<bfloat16_t>();
        AntiquantVFFp8D448<Q_T, KV_T>(kRopeUbNz, antiKvTensorAsB16, srcTensor, scaleUb, kRopeUb, dealRow);
    } else {
        LocalTensor<float> floatScale = dequantScaleUb.tensor;
        kRopeUb = srcTensor[448].template ReinterpretCast<int8_t>();
        CastScale<Q_T, KV_T>(floatScale, srcTensor, dealRow);
        AntiquantVFFp8D448_FloatScale<Q_T, KV_T>(kRopeUbNz, antiKvTensorAsB16, srcTensor, floatScale, kRopeUb, dealRow);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::CopyOutKvUb2Gm(
    Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm, LocalTensor<Q_T> antiKvTensorAsB16,
    int64_t dealRow, int64_t s2StartIdx, const RunInfo<HIGH_PERF> &runInfo, ConstInfo<HIGH_PERF> &constInfo)
{
    GlobalTensor<Q_T> v0ResGmTensor = v0ResGm.template GetTensor<Q_T>();
    uint64_t blockElementNum = 16;
    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = (constInfo.dSizeNope + constInfo.dSizeRope) / blockElementNum;
    dataCopyParams.blockLen = dealRow;
    dataCopyParams.srcGap = blockElementNum + 1 - dealRow;
    dataCopyParams.dstGap = Align16Func(runInfo.s2RealSize) - dealRow;
    DataCopy(v0ResGmTensor[s2StartIdx * blockElementNum], antiKvTensorAsB16, dataCopyParams);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::ProcessNotSparseKv(
    Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm, const RunInfo<HIGH_PERF> &runInfo,
    ConstInfo<HIGH_PERF> &constInfo)
{
    if (processSize == 0) {
        return;
    }
    constexpr int64_t v0ProcessBase = 16;
    int64_t v0ProcessSize = v0ProcessBase;
    int64_t loopTimes = (processSize + v0ProcessBase - 1) / v0ProcessBase;
    for (uint32_t i = 0; i < loopTimes; i++) {
        int64_t s2StartIdx = processS2Start + i * v0ProcessBase;
        if (i == loopTimes - 1) {
            v0ProcessSize = processSize - i * v0ProcessBase;
        }
        // 1、copy kv in, gm ->ub
        WaitFlag<HardEvent::V_MTE2>(INNERCORE_STAGE0_IN(pingPongV0));
        LocalTensor<KV_T> kvInUb = stage0InBufs[pingPongV0].tensor;
        CopyInKvNotSparse(kvInUb, i, v0ProcessSize, s2StartIdx, runInfo, constInfo);
        SetFlag<HardEvent::MTE2_V>(INNERCORE_STAGE0_IN(pingPongV0));
        WaitFlag<HardEvent::MTE2_V>(INNERCORE_STAGE0_IN(pingPongV0));

        // 2、dequant by vf
        WaitFlag<HardEvent::MTE3_V>(INNERCORE_STAGE0_OUT(pingPongV0));
        LocalTensor<Q_T> kvDequantOutUb = stage0OutBufs[pingPongV0].tensor;
        DequantKv(kvDequantOutUb, kvInUb, v0ProcessSize, constInfo);
        SetFlag<HardEvent::V_MTE2>(INNERCORE_STAGE0_IN(pingPongV0));

        // 3、copy kv out, ub -> l1
        SetFlag<HardEvent::V_MTE3>(INNERCORE_STAGE0_OUT(pingPongV0));
        WaitFlag<HardEvent::V_MTE3>(INNERCORE_STAGE0_OUT(pingPongV0));
        CopyOutKvUb2Gm(v0ResGm, kvDequantOutUb, v0ProcessSize, s2StartIdx, runInfo, constInfo);
        SetFlag<HardEvent::MTE3_V>(INNERCORE_STAGE0_OUT(pingPongV0));
        pingPongV0 ^= 1;
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::CopyInKvNotSparse(LocalTensor<KV_T> kvMergUb, int64_t v0Loop,
                                                                     int64_t dealRow, int64_t s2StartOffset,
                                                                     const RunInfo<HIGH_PERF> &runInfo,
                                                                     ConstInfo<HIGH_PERF> &constInfo)
{
    int64_t s2LoopCount = (runInfo.s2LoopCount >= runInfo.oriKvLoopEndIdx) ?
                              (runInfo.s2LoopCount - runInfo.oriKvLoopEndIdx) :
                              runInfo.s2LoopCount;
    int64_t s2Idx = s2StartOffset + s2LoopCount * constInfo.s2BaseSize + runInfo.s2StartIdx;
    if (!constInfo.hasRopeInput) {
        for (int64_t row = 0; row < dealRow; ++row) {
            int64_t keyOffset = -1;
            int64_t scaleOffset = -1;
            GetKeyOffset(s2Idx + row, keyOffset, scaleOffset, runInfo, constInfo);
            CopyInSingleKvNoRope(kvMergUb, row, keyOffset, scaleOffset);
        }
        return;
    }
    uint32_t combineBytes;
    uint32_t combineDim;
    uint32_t combineDimAlign;
    if constexpr (QUANT_MODE == SCALE_CONTIGUOUS_MODE::CONTIGUOUS) {
        combineBytes = constInfo.dSizeVInput * sizeof(KV_T);
        combineDim = combineBytes / sizeof(KV_T);
        combineDimAlign = CeilAlign(combineBytes, BUFFER_SIZE_BYTE_32B) / sizeof(KV_T);
    } else {
        combineDim = dVTemplateTypeInput / sizeof(KV_T);
        combineDimAlign = CeilAlign(dVTemplateTypeInput, BUFFER_SIZE_BYTE_32B) / sizeof(KV_T);
    }
    DataCopyExtParams intriParams;
    intriParams.blockCount = dealRow;
    if constexpr (QUANT_MODE == SCALE_CONTIGUOUS_MODE::CONTIGUOUS) {
        intriParams.blockLen = combineBytes;
        intriParams.dstStride = 0;
    } else {
        intriParams.blockLen = dCombineBytes;
        intriParams.dstStride = (dVTemplateTypeInput - dCombineBytes) / DATABLOCK_BYTES;
    }
    intriParams.srcStride = 0;
    DataCopyPadExtParams<KV_T> padParams;
    padParams.isPad = true;
    padParams.leftPadding = 0;
    padParams.rightPadding = combineDimAlign - combineDim;
    padParams.paddingValue = 0;
    if constexpr (isPa) {
        uint64_t mqsmlaBlockTableBaseOffset = runInfo.boIdx * maxBlockNumPerBatch;
        uint64_t mqsmlaDstOffset = 0;
        uint32_t mqsmlaCopyFinishElmenCnt = 0;
        uint32_t mqsmlaCurSequence = s2Idx;
        int64_t mqsmlaPaBlockStride = runInfo.isCmp ? constInfo.cmpKvStride : constInfo.oriKvStride;
        while (mqsmlaCopyFinishElmenCnt < dealRow) {
            uint64_t blockIdOffset = mqsmlaCurSequence / blockSize;
            uint64_t remainElmenCnt = mqsmlaCurSequence % blockSize;
            uint64_t idInBlockTable = blockTableGm.GetValue(mqsmlaBlockTableBaseOffset + blockIdOffset);
            uint32_t copyElmenCnt = blockSize - remainElmenCnt;
            if (copyElmenCnt + mqsmlaCopyFinishElmenCnt > dealRow) {
                copyElmenCnt = dealRow - mqsmlaCopyFinishElmenCnt;
            }
            if constexpr (QUANT_MODE == SCALE_CONTIGUOUS_MODE::CONTIGUOUS) {
                uint64_t srcOffset = idInBlockTable * mqsmlaPaBlockStride +
                                     remainElmenCnt * constInfo.n2Size * combineBytes +
                                     (uint64_t)(runInfo.n2oIdx * combineBytes); // BlockNum, BlockSize, N, D
                intriParams.blockCount = copyElmenCnt;                          // base s2 size
                DataCopyPad(kvMergUb[mqsmlaDstOffset * combineDimAlign], keyGm[srcOffset], intriParams, padParams);
            } else {
                uint64_t keyOffset = idInBlockTable * mqsmlaPaBlockStride +
                                     remainElmenCnt * constInfo.n2Size * dCombineBytes +
                                     (uint64_t)(runInfo.n2oIdx * dCombineBytes); // BlockNum, BlockSize, N, D
                uint64_t scaleOffset = idInBlockTable * mqsmlaPaBlockStride +
                                       blockSize * constInfo.n2Size * dCombineBytes + remainElmenCnt * scaleBytes;
                intriParams.blockCount = copyElmenCnt; // base s2 size
                DataCopyPad(kvMergUb[mqsmlaDstOffset * combineDimAlign], keyGm[keyOffset], intriParams, padParams);
                // 获取scale部分并将scale写入ub对应位置
                DataCopyExtParams scaleParams;
                DataCopyPadExtParams<KV_T> scalePadParams;
                scaleParams.blockCount = copyElmenCnt;
                scaleParams.blockLen = scaleBytes;
                scaleParams.srcStride = 0;
                scaleParams.dstStride = dCombineBytes / DATABLOCK_BYTES;
                scalePadParams.isPad = true;
                scalePadParams.leftPadding = 0;
                scalePadParams.rightPadding = DATABLOCK_BYTES - scaleBytes;
                scalePadParams.paddingValue = 0;
                DataCopyPad(kvMergUb[mqsmlaDstOffset * combineDimAlign + dCombineBytes], keyGm[scaleOffset],
                            scaleParams, scalePadParams);
            }
            mqsmlaDstOffset += copyElmenCnt;
            mqsmlaCopyFinishElmenCnt += copyElmenCnt;
            mqsmlaCurSequence += copyElmenCnt;
        }
    } else if constexpr (KV_LAYOUT_T == QSMLA_LAYOUT::TND) {
        int64_t tPrefix =
            runInfo.isCmp ? cuSeqlensCmpKvGm.GetValue(runInfo.boIdx) : cuSeqlensOriKvGm.GetValue(runInfo.boIdx);
        DataCopyPad(kvMergUb, keyGm[(tPrefix + s2Idx) * constInfo.n2Size * combineDim + runInfo.n2oIdx * combineDim],
                    intriParams, padParams);
    } else {
        int64_t realKeyOffset;
        if (runInfo.isCmp) {
            realKeyOffset = runInfo.boIdx * constInfo.cmpS2Size * constInfo.n2Size * constInfo.dSizeVInput +
                            runInfo.n2oIdx * constInfo.dSizeVInput + s2Idx * constInfo.dSizeVInput;
        } else {
            realKeyOffset = runInfo.boIdx * constInfo.s2Size * constInfo.n2Size * constInfo.dSizeVInput +
                            runInfo.n2oIdx * constInfo.dSizeVInput + s2Idx * constInfo.dSizeVInput;
        }
        DataCopyPad(kvMergUb, keyGm[realKeyOffset], intriParams, padParams);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::CalProcessSize(const RunInfo<HIGH_PERF> &runInfo,
                                                                  ConstInfo<HIGH_PERF> &constInfo)
{
    if constexpr (TEMPLATE_MODE == QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        sparseIndicesGm = runInfo.isCmp ? cmpSparseIndicesGm : oriSparseIndicesGm;
    } else if constexpr (TEMPLATE_MODE == QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE) {
        sparseIndicesGm = oriSparseIndicesGm;
    } else {
        sparseIndicesGm = cmpSparseIndicesGm;
    }
    if constexpr (IS_SPLIT_G) {
        uint32_t mqsmlaAicIdx = constInfo.aivIdx >> 1U;
        uint32_t mqsmlaV0S2SizeFirstCore = CeilDiv(runInfo.s2RealSize, 2);
        uint32_t mqsmlaV0S2SizeSecondCore = runInfo.s2RealSize - mqsmlaV0S2SizeFirstCore;
        int32_t vecCnt = (mqsmlaAicIdx % 2U == 0) ? (GetSubBlockIdx() == 0 ? 0 : 1) : (GetSubBlockIdx() == 0 ? 2 : 3);
        if (mqsmlaAicIdx % 2U == 0) {
            if (GetSubBlockIdx() == 0) {
                processSize = CeilDiv(mqsmlaV0S2SizeFirstCore, 2U);
                processS2Start = 0;
            } else {
                processSize = mqsmlaV0S2SizeFirstCore - CeilDiv(mqsmlaV0S2SizeFirstCore, 2U);
                processS2Start = CeilDiv(mqsmlaV0S2SizeFirstCore, 2U);
            }
        } else {
            if (GetSubBlockIdx() == 0) {
                processSize = CeilDiv(mqsmlaV0S2SizeSecondCore, 2U);
                processS2Start = mqsmlaV0S2SizeFirstCore;
            } else {
                processSize = mqsmlaV0S2SizeSecondCore - CeilDiv(mqsmlaV0S2SizeSecondCore, 2U);
                processS2Start = mqsmlaV0S2SizeFirstCore + CeilDiv(mqsmlaV0S2SizeSecondCore, 2U);
            }
        }
        processS2End = processS2Start + processSize;
    } else {
        int64_t mqsmlaS2PerVecLoop = 2LL;
        int64_t mqsmlaVecNum = 2LL;
        int64_t mqsmlaS2Loops = CeilDiv(CeilDiv(runInfo.s2RealSize, mqsmlaVecNum), mqsmlaS2PerVecLoop);
        processS2Start = GetSubBlockIdx() == 0 ? 0 : Min(mqsmlaS2Loops * mqsmlaS2PerVecLoop, runInfo.s2RealSize);
        processS2End =
            GetSubBlockIdx() == 0 ? Min(mqsmlaS2Loops * mqsmlaS2PerVecLoop, runInfo.s2RealSize) : runInfo.s2RealSize;
        processSize = processS2End - processS2Start;
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::ProcessVec0(
    Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm, const RunInfo<HIGH_PERF> &runInfo,
    ConstInfo<HIGH_PERF> &constInfo)
{
    bool mqsmlaIsCmp = runInfo.s2LoopCount >= runInfo.oriKvLoopEndIdx;
    if (mqsmlaIsCmp) {
        keyGm = cmpKVGm;
        if constexpr (isPa) {
            blockTableGm = cmpBlockTableGm;
            blockSize = constInfo.cmpBlockSize;
            maxBlockNumPerBatch = constInfo.cmpMaxBlockNumPerBatch;
        }
    } else {
        keyGm = oriKVGm;
        if constexpr (isPa) {
            blockTableGm = oriBlockTableGm;
            blockSize = constInfo.oriBlockSize;
            maxBlockNumPerBatch = constInfo.oriMaxBlockNumPerBatch;
        }
    }

    CalProcessSize(runInfo, constInfo);
    if constexpr (TEMPLATE_MODE == QSMLATemplateMode::CSA_TEMPLATE_MODE) {
        if (mqsmlaIsCmp) {
            ProcessSparseKv(v0ResGm, runInfo, constInfo);
        } else {
            ProcessNotSparseKv(v0ResGm, runInfo, constInfo);
        }
    } else if constexpr (TEMPLATE_MODE == QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                         TEMPLATE_MODE == QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        ProcessSparseKv(v0ResGm, runInfo, constInfo);
    } else {
        ProcessNotSparseKv(v0ResGm, runInfo, constInfo);
    }
    v0ResGm.SetCrossCore();
}

TEMPLATES_DEF_NO_DEFAULT
template <bool IS_FULL>
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::CopyIn8Block(LocalTensor<KV_T> kvInUb, int64_t startRow, int64_t &s2,
                                                                const RunInfo<HIGH_PERF> &runInfo,
                                                                ConstInfo<HIGH_PERF> &constInfo)
{
    // tokenData元素为-1表示无效token，尾块场景下GetReal*仅填充有效区间，其余保持-1
    int64_t tokenData[KV_COPYIN_UNIT] = {-1, -1, -1, -1, -1, -1, -1, -1};
    if constexpr (QUANT_MODE == SCALE_CONTIGUOUS_MODE::CONTIGUOUS && IS_VEC_S2PHYADDR) {
        GetRealS2Addr<IS_FULL>(tokenData, s2, runInfo, constInfo);
    } else {
        GetRealCmpS2Idx<IS_FULL>(tokenData, s2, runInfo, constInfo);
    }
    s2 += KV_COPYIN_UNIT;
    CopyInKvSparse<IS_FULL>(kvInUb, startRow, tokenData, runInfo, constInfo);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::DequantAndCopyOutKv(
    Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm, LocalTensor<KV_T> kvInUb, int64_t dealRow,
    int64_t s2StartIdx, const RunInfo<HIGH_PERF> &runInfo, ConstInfo<HIGH_PERF> &constInfo)
{
    SetFlag<HardEvent::MTE2_V>(INNERCORE_STAGE0_IN(pingPongV0));
    WaitFlag<HardEvent::MTE2_V>(INNERCORE_STAGE0_IN(pingPongV0));
    // 2、dequant by vf
    WaitFlag<HardEvent::MTE3_V>(INNERCORE_STAGE0_OUT(pingPongV0));
    LocalTensor<Q_T> kvDequantOutUb = stage0OutBufs[pingPongV0].tensor;
    DequantKv(kvDequantOutUb, kvInUb, dealRow, constInfo);
    SetFlag<HardEvent::V_MTE2>(INNERCORE_STAGE0_IN(pingPongV0));
    // 3、copy kv out, ub -> l1
    SetFlag<HardEvent::V_MTE3>(INNERCORE_STAGE0_OUT(pingPongV0));
    WaitFlag<HardEvent::V_MTE3>(INNERCORE_STAGE0_OUT(pingPongV0));
    CopyOutKvUb2Gm(v0ResGm, kvDequantOutUb, dealRow, s2StartIdx, runInfo, constInfo);
    SetFlag<HardEvent::MTE3_V>(INNERCORE_STAGE0_OUT(pingPongV0));
    pingPongV0 ^= 1;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::ProcessSparseKv(
    Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm, const RunInfo<HIGH_PERF> &runInfo,
    ConstInfo<HIGH_PERF> &constInfo)
{
    int64_t curProcessSize = this->processSize;
    if (curProcessSize == 0) {
        return;
    }

    // 前置计算：16行dequant循环数、尾块行数、尾块内8行搬入次数及剩余行数
    int64_t dequant16LoopCnt = curProcessSize / KV_DEQUANT_UNIT;
    int64_t tail16Rows = curProcessSize % KV_DEQUANT_UNIT;
    int64_t tail8FullCnt = tail16Rows / KV_COPYIN_UNIT;
    int64_t tail8Remain = tail16Rows % KV_COPYIN_UNIT;
    int64_t s2 = processS2Start;

    // 阶段1：完整16行块（非尾块，8行均有效，无需判断）
    for (int64_t i = 0; i < dequant16LoopCnt; i++) {
        int64_t s2StartIdx = processS2Start + i * KV_DEQUANT_UNIT;
        // 1、copy kv in, gm ->ub
        WaitFlag<HardEvent::V_MTE2>(INNERCORE_STAGE0_IN(pingPongV0));
        LocalTensor<KV_T> kvInUb = stage0InBufs[pingPongV0].tensor;
        CopyIn8Block<true>(kvInUb, 0, s2, runInfo, constInfo);
        CopyIn8Block<true>(kvInUb, KV_COPYIN_UNIT, s2, runInfo, constInfo);
        // 2、dequant 3、copy kv out, ub -> l1
        DequantAndCopyOutKv(v0ResGm, kvInUb, KV_DEQUANT_UNIT, s2StartIdx, runInfo, constInfo);
    }

    // 阶段2：尾块（不足16行，保留判断逻辑）
    if (tail16Rows > 0) {
        int64_t s2StartIdx = processS2Start + dequant16LoopCnt * KV_DEQUANT_UNIT;
        int64_t dealRow = 0;
        // 1、copy kv in, gm ->ub
        WaitFlag<HardEvent::V_MTE2>(INNERCORE_STAGE0_IN(pingPongV0));
        LocalTensor<KV_T> kvInUb = stage0InBufs[pingPongV0].tensor;
        // 尾块内满8行搬入（8行均有效，无需判断）
        for (int64_t j = 0; j < tail8FullCnt; j++) {
            CopyIn8Block<true>(kvInUb, dealRow, s2, runInfo, constInfo);
            dealRow += KV_COPYIN_UNIT;
        }
        // 尾块内不足8行（需判断边界，与当前逻辑一致）
        if (tail8Remain > 0) {
            CopyIn8Block<false>(kvInUb, dealRow, s2, runInfo, constInfo);
            dealRow += tail8Remain;
        }
        // 2、dequant 3、copy kv out, ub -> l1
        DequantAndCopyOutKv(v0ResGm, kvInUb, tail16Rows, s2StartIdx, runInfo, constInfo);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::ProcessVec1(StaticBuffer<Q_T> &outputBuf,
                                                               StaticBuffer<T> &bmm1ResBuf, RunInfo<HIGH_PERF> &runInfo,
                                                               ConstInfo<HIGH_PERF> &constInfo)
{
    CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CROSSCORE_BMM1(bmm1ResBuf.idx));

    LocalTensor<float> mqsmlaSumUb = this->softmaxSumBufs[runInfo.multiCoreIdxMod2].tensor;
    LocalTensor<float> mqsmlaMaxUb = this->softmaxMaxBufs[runInfo.multiCoreIdxMod2].tensor;
    LocalTensor<float> mqsmlaExpUb = this->softmaxExpBufs[runInfo.taskIdMod2].tensor;
    int64_t mqsmlaStage1Offset = runInfo.taskIdMod2;
    WaitFlag<HardEvent::MTE3_V>(INNERCORE_STAGE1(mqsmlaStage1Offset));
    LocalTensor<Q_T> mqsmlaStage1CastTensor = this->stage1OutBufs[mqsmlaStage1Offset].tensor;

    LocalTensor<T> mqsmlaApiTmpBuffer = this->commonUb.tensor;
    LocalTensor<T> mqsmlaMmRes = bmm1ResBuf.tensor;

    bool isFirstSoftmaxBase = runInfo.s2LoopCount == 0;
    if constexpr (IS_BATCH_CONSISTENCY) {
        isFirstSoftmaxBase = runInfo.isFirstBase;
    }
    // The first base block of every reduce block starts a new online-softmax state.
    // With sinks, use update mode after initializing max/sum from the sink state.
    if (isFirstSoftmaxBase && !isSinks) {
        if (likely(runInfo.s2RealSize == 128)) { // s2RealSize等于128分档, VF内常量化减少if判断
            ProcessVec1Vf<T, Q_T, false, s1BaseSize, s2BaseSize, FaVectorApi::OriginNRange::EQ_128_SFA>(
                mqsmlaStage1CastTensor, mqsmlaMmRes, mqsmlaSumUb, mqsmlaMaxUb, mqsmlaMaxUb, mqsmlaApiTmpBuffer,
                vselrIndexesBuf, runInfo.halfMRealSize, runInfo.s2RealSize, static_cast<T>(constInfo.softmaxScale),
                negativeFloatScalar);
        } else if (runInfo.s2RealSize <= 64) { // s2RealSize小于等于64分档, VF内常量化减少if判断
            ProcessVec1Vf<T, Q_T, false, s1BaseSize, s2BaseSize, FaVectorApi::OriginNRange::GT_0_AND_LTE_64_SFA>(
                mqsmlaStage1CastTensor, mqsmlaMmRes, mqsmlaSumUb, mqsmlaMaxUb, mqsmlaMaxUb, mqsmlaApiTmpBuffer,
                vselrIndexesBuf, runInfo.halfMRealSize, runInfo.s2RealSize, static_cast<T>(constInfo.softmaxScale),
                negativeFloatScalar);
        } else if (runInfo.s2RealSize < 128) { // s2RealSize小于128分档, VF内常量化减少if判断
            ProcessVec1Vf<T, Q_T, false, s1BaseSize, s2BaseSize, FaVectorApi::OriginNRange::GT_64_AND_LTE_128_SFA>(
                mqsmlaStage1CastTensor, mqsmlaMmRes, mqsmlaSumUb, mqsmlaMaxUb, mqsmlaMaxUb, mqsmlaApiTmpBuffer,
                vselrIndexesBuf, runInfo.halfMRealSize, runInfo.s2RealSize, static_cast<T>(constInfo.softmaxScale),
                negativeFloatScalar);
        }
    } else {
        if (isFirstSoftmaxBase && isSinks) {
            bool includeSink = (!runInfo.isCrossCoreSplit) || runInfo.isFirstS2SplitCore;
            if constexpr (IS_BATCH_CONSISTENCY) {
                includeSink = includeSink && runInfo.reduceBlockId == 0;
            }
            if (includeSink) {
                // s1切1,vec0: 0 ~ halfMRealSize - 1, vec1: gSize - halfMRealSize ~ gSize
                int64_t sinksOffset = 0;
                if constexpr (!IS_SPLIT_G) {
                    sinksOffset = GetBlockIdx() % 2U == 0 ? 0 : runInfo.firstHalfMRealSize;
                } else {
                    sinksOffset = runInfo.goIdx;
                    if (constInfo.subBlockIdx == 1) {
                        sinksOffset += runInfo.firstHalfMRealSize;
                    }
                }
                LocalTensor<T> sinksUb = this->sinksUb.tensor;
                InitSoftmaxFromSinks<T>(mqsmlaSumUb, mqsmlaMaxUb, sinksUb, sinksOffset, R0, runInfo.halfMRealSize);
            } else {
                // Later reduce blocks and non-first FD splits start without the sink state.
                Duplicate(mqsmlaMaxUb, this->negativeFloatScalar, runInfo.halfMRealSize);
                Duplicate(mqsmlaSumUb, static_cast<T>(0), runInfo.halfMRealSize);
            }
        }
        if (likely(runInfo.s2RealSize == 128)) { // s2RealSize等于128分档, VF内常量化减少if判断
            ProcessVec1Vf<T, Q_T, true, s1BaseSize, s2BaseSize, FaVectorApi::OriginNRange::EQ_128_SFA>(
                mqsmlaStage1CastTensor, mqsmlaMmRes, mqsmlaSumUb, mqsmlaMaxUb, mqsmlaMaxUb, mqsmlaApiTmpBuffer,
                vselrIndexesBuf, runInfo.halfMRealSize, runInfo.s2RealSize, static_cast<T>(constInfo.softmaxScale),
                negativeFloatScalar);
        } else if (runInfo.s2RealSize <= 64) { // s2RealSize小于等于64分档, VF内常量化减少if判断
            ProcessVec1Vf<T, Q_T, true, s1BaseSize, s2BaseSize, FaVectorApi::OriginNRange::GT_0_AND_LTE_64_SFA>(
                mqsmlaStage1CastTensor, mqsmlaMmRes, mqsmlaSumUb, mqsmlaMaxUb, mqsmlaMaxUb, mqsmlaApiTmpBuffer,
                vselrIndexesBuf, runInfo.halfMRealSize, runInfo.s2RealSize, static_cast<T>(constInfo.softmaxScale),
                negativeFloatScalar);
        } else if (runInfo.s2RealSize < 128) { // s2RealSize小于128分档, VF内常量化减少if判断
            ProcessVec1Vf<T, Q_T, true, s1BaseSize, s2BaseSize, FaVectorApi::OriginNRange::GT_64_AND_LTE_128_SFA>(
                mqsmlaStage1CastTensor, mqsmlaMmRes, mqsmlaSumUb, mqsmlaMaxUb, mqsmlaMaxUb, mqsmlaApiTmpBuffer,
                vselrIndexesBuf, runInfo.halfMRealSize, runInfo.s2RealSize, static_cast<T>(constInfo.softmaxScale),
                negativeFloatScalar);
        }
    }
    CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CROSSCORE_BMM1(bmm1ResBuf.idx));

    // ===================DataCopy to L1 ====================
    SetFlag<HardEvent::V_MTE3>(INNERCORE_STAGE1(mqsmlaStage1Offset));
    WaitFlag<HardEvent::V_MTE3>(INNERCORE_STAGE1(mqsmlaStage1Offset));

    LocalTensor<Q_T> mm2AL1Tensor = outputBuf.tensor;
    if (likely(runInfo.halfMRealSize != 0)) {
        DataCopy(mm2AL1Tensor[constInfo.subBlockIdx * (BLOCK_BYTE / sizeof(Q_T)) *
                              (runInfo.mRealSize - runInfo.halfMRealSize)],
                 mqsmlaStage1CastTensor,
                 {s2BaseSize / 16, (uint16_t)runInfo.halfMRealSize, (uint16_t)(vec1Srcstride - runInfo.halfMRealSize),
                  (uint16_t)(Align16Func(runInfo.mRealSize) - runInfo.halfMRealSize)});
    }

    SetFlag<HardEvent::MTE3_V>(INNERCORE_STAGE1(mqsmlaStage1Offset));

    CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(CROSSCORE_L1P(outputBuf.idx));
    // ======================================================
    if (!isFirstSoftmaxBase || isSinks) {
        SFAUpdateExpSumAndExpMax<T>(mqsmlaSumUb, mqsmlaMaxUb, mqsmlaExpUb, mqsmlaSumUb, mqsmlaMaxUb, mqsmlaApiTmpBuffer,
                                    runInfo.halfMRealSize);
    }

    if constexpr (IS_BATCH_CONSISTENCY) {
        if (runInfo.isLastBase) {
            if (runInfo.halfMRealSize > 0) {
                LocalTensor<float> finalMaxUb = this->softmaxFinalMaxBufs[runInfo.taskIdMod2].tensor;
                LocalTensor<float> finalSumUb = this->softmaxFinalSumBufs[runInfo.taskIdMod2].tensor;
                // FP32 UB-to-UB DataCopy must cover complete 32-byte blocks.
                uint64_t snapshotElems = Align8Func(runInfo.halfMRealSize);
                DataCopy(finalMaxUb, mqsmlaMaxUb, snapshotElems);
                DataCopy(finalSumUb, mqsmlaSumUb, snapshotElems);
            }
            if (runInfo.isCrossCoreSplit && !runInfo.isFirstS2SplitCore) {
                AttentionCommon::S2SplitFdStagingLayout stagingLayout = {
                    constInfo.gSize, dTemplateAlign64, GetStagingSlotNum(false),
                    AttentionCommon::FD_BROADCAST_ELEMS_PER_ROW, AttentionCommon::FD_REDUCE_CHUNK_ROWS};
                uint32_t workspaceIdx = GetCrossCoreWorkspaceIdx(runInfo);
                int64_t stagingMOffset = GetFaStagingMOffset(runInfo, constInfo);
                LocalTensor<float> tmpUb = this->batchReduceTmpUb.tensor;
                AttentionCommon::StageVec1Lse(stagingLayout, crossCoreCombineBase, workspaceIdx, stagingMOffset,
                                              runInfo.halfMRealSize, mqsmlaMaxUb, mqsmlaSumUb, tmpUb, INNERCORE_STAGE2,
                                              INNERCORE_STAGE_FD_MTE3_V);
            } else if (runInfo.isFirstS2SplitCore && runInfo.reduceBlockId == 0 &&
                       runInfo.s2LoopCount < runInfo.s2LoopLimit) {
                AttentionCommon::S2SplitFdStagingLayout stagingLayout = {
                    constInfo.gSize, dTemplateAlign64, GetStagingSlotNum(true),
                    AttentionCommon::FD_BROADCAST_ELEMS_PER_ROW, AttentionCommon::FD_REDUCE_CHUNK_ROWS};
                int64_t stagingMOffset = GetFaStagingMOffset(runInfo, constInfo);
                LocalTensor<float> tmpUb = this->batchReduceTmpUb.tensor;
                AttentionCommon::StageVec1Lse(stagingLayout, intraCoreCombineBase,
                                              GetIntraCoreWorkspaceIdx(runInfo, constInfo), stagingMOffset,
                                              runInfo.halfMRealSize, mqsmlaMaxUb, mqsmlaSumUb, tmpUb, INNERCORE_STAGE2,
                                              INNERCORE_STAGE_FD_MTE3_V);
                SetFlag<HardEvent::MTE3_MTE2>(INNERCORE_INTRALSE_MTE3_MTE2(runInfo.multiCoreIdxMod2));
            } else if (runInfo.isCrossCoreSplit && runInfo.isFirstS2SplitCore && runInfo.reduceBlockId == 0) {
                AttentionCommon::S2SplitFdStagingLayout stagingLayout = {
                    constInfo.gSize, dTemplateAlign64, GetStagingSlotNum(false),
                    AttentionCommon::FD_BROADCAST_ELEMS_PER_ROW, AttentionCommon::FD_REDUCE_CHUNK_ROWS};
                uint32_t workspaceIdx = GetCrossCoreWorkspaceIdx(runInfo);
                int64_t stagingMOffset = GetFaStagingMOffset(runInfo, constInfo);
                LocalTensor<float> tmpUb = this->batchReduceTmpUb.tensor;
                AttentionCommon::StageVec1Lse(stagingLayout, crossCoreCombineBase, workspaceIdx, stagingMOffset,
                                              runInfo.halfMRealSize, mqsmlaMaxUb, mqsmlaSumUb, tmpUb, INNERCORE_STAGE2,
                                              INNERCORE_STAGE_FD_MTE3_V);
            }
        }
    } else {
        // 常规FD
        if (runInfo.isLastBase && runInfo.isCrossCoreSplit) {
            AttentionCommon::S2SplitFdStagingLayout stagingLayout = {
                constInfo.gSize, dTemplateAlign64, GetStagingSlotNum(), AttentionCommon::FD_BROADCAST_ELEMS_PER_ROW,
                AttentionCommon::FD_REDUCE_CHUNK_ROWS};
            uint32_t workspaceIdx = GetCrossCoreWorkspaceIdx(runInfo);
            int64_t stagingMOffset = GetFaStagingMOffset(runInfo, constInfo);
            LocalTensor<float> tmpUb = this->dequantScaleUb.tensor;
            AttentionCommon::StageVec1Lse(stagingLayout, fdStagingBase, workspaceIdx, stagingMOffset,
                                          runInfo.halfMRealSize, mqsmlaMaxUb, mqsmlaSumUb, tmpUb, INNERCORE_STAGE2,
                                          INNERCORE_STAGE_FD_MTE3_V);
        }
    }
    if constexpr (!HIGH_PERF) {
        bool copyOutLse = constInfo.isSoftmaxLseEnable && this->isSoftmaxLseGmValid && runInfo.halfMRealSize > 0 &&
                          runInfo.s2LoopCount == runInfo.s2LoopLimit;
        if constexpr (IS_BATCH_CONSISTENCY) {
            copyOutLse = copyOutLse && !runInfo.isCrossCoreSplit && !runInfo.needReduce;
        }
        if (copyOutLse) {
            LocalTensor<float> outLse = this->outLseUbs[runInfo.multiCoreIdxMod2].tensor;
            DataCopyExtParams lseParams;
            lseParams.blockCount = 1;
            lseParams.blockLen = static_cast<uint32_t>(runInfo.halfMRealSize * sizeof(float));
            lseParams.srcStride = 0;
            lseParams.dstStride = 0;
            WaitFlag<HardEvent::MTE3_V>(INNERCORE_LSE_MTE3_V);
            ComputeLse<float>(outLse, mqsmlaSumUb, mqsmlaMaxUb, static_cast<uint32_t>(runInfo.halfMRealSize));
            SetFlag<HardEvent::V_MTE3>(INNERCORE_LSE_V_MTE3);
            WaitFlag<HardEvent::V_MTE3>(INNERCORE_LSE_V_MTE3);
            DataCopyPad(this->softmaxLseGm[runInfo.softmaxLseOffset], outLse, lseParams);
            SetFlag<HardEvent::MTE3_V>(INNERCORE_LSE_MTE3_V);
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::ReduceIntraBlockAndStage(RunInfo<HIGH_PERF> &runInfo,
                                                                            ConstInfo<HIGH_PERF> &constInfo,
                                                                            LocalTensor<T> &vec2ResUb,
                                                                            LocalTensor<T> &partialTmpUb)
{
    AttentionCommon::S2SplitFdStagingLayout mqsmlaIntraLayout = {
        constInfo.gSize, dTemplateAlign64, GetStagingSlotNum(true), AttentionCommon::FD_BROADCAST_ELEMS_PER_ROW,
        AttentionCommon::FD_REDUCE_CHUNK_ROWS};
    AttentionCommon::S2SplitFdStagingLayout mqsmlaCrossLayout = {
        constInfo.gSize, dTemplateAlign64, GetStagingSlotNum(false), AttentionCommon::FD_BROADCAST_ELEMS_PER_ROW,
        AttentionCommon::FD_REDUCE_CHUNK_ROWS};
    uint32_t intraWorkspaceIdx = GetIntraCoreWorkspaceIdx(runInfo, constInfo);
    // s2SplitIdx advances across all reduce blocks handled by this core. Move back
    // to the first slot of the current intra-core reduce group before staging it.
    uint32_t crossWorkspaceIdx =
        static_cast<uint32_t>(runInfo.firstFdDataWorkspaceIdx + runInfo.s2SplitIdx - runInfo.reduceBlockId);
    int64_t stagingMOffset = GetFaStagingMOffset(runInfo, constInfo);
    LocalTensor<float> tmpUb = this->batchReduceTmpUb.tensor;
    LocalTensor<float> blockMaxUb = tmpUb;
    LocalTensor<float> blockSumUb = tmpUb[256];
    LocalTensor<float> lseBroadcastUb = tmpUb[512];
    LocalTensor<float> sumBroadcastUb = tmpUb[640];
    LocalTensor<float> maxUb = this->softmaxMaxBufs[runInfo.multiCoreIdxMod2].tensor;
    LocalTensor<float> sumUb = this->softmaxSumBufs[runInfo.multiCoreIdxMod2].tensor;
    if constexpr (IS_BATCH_CONSISTENCY) {
        maxUb = this->softmaxFinalMaxBufs[runInfo.taskIdMod2].tensor;
        sumUb = this->softmaxFinalSumBufs[runInfo.taskIdMod2].tensor;
    }
    bool copyOutMergedLse = false;
    if constexpr (!HIGH_PERF) {
        copyOutMergedLse = constInfo.isSoftmaxLseEnable && this->isSoftmaxLseGmValid && !runInfo.isCrossCoreSplit &&
                           runInfo.s2LoopCount == runInfo.s2LoopLimit;
    }
    WaitFlag<HardEvent::MTE3_MTE2>(INNERCORE_INTRALSE_MTE3_MTE2(runInfo.multiCoreIdxMod2));
    WaitFlag<HardEvent::MTE3_MTE2>(INNERCORE_INTRAATTN_MTE3_MTE2(runInfo.multiCoreIdxMod2));
    LocalTensor<T> sinkUb;
    int64_t startRow = 0;
    while (startRow < runInfo.vec2MRealSize) {
        int64_t dealRowCount = mqsmlaIntraLayout.chunkRows;
        if (startRow + dealRowCount > runInfo.vec2MRealSize) {
            dealRowCount = runInfo.vec2MRealSize - startRow;
        }
        LocalTensor<T> chunkCurrent = vec2ResUb[startRow * dTemplateAlign64];
        LocalTensor<float> chunkMaxUb = maxUb[startRow];
        LocalTensor<float> chunkSumUb = sumUb[startRow];
        if constexpr (!HIGH_PERF) {
            if (copyOutMergedLse) {
                WaitFlag<HardEvent::MTE3_V>(INNERCORE_LSE_MTE3_V);
            }
        }
        AttentionCommon::MergeStagedAndCurrentChunk<T, dTemplateAlign64>(
            mqsmlaIntraLayout, intraCoreCombineBase, intraWorkspaceIdx, stagingMOffset + startRow, dealRowCount,
            static_cast<int64_t>(constInfo.dSizeV), chunkMaxUb, chunkSumUb, chunkCurrent, blockMaxUb, blockSumUb,
            partialTmpUb, lseBroadcastUb, sumBroadcastUb, sinkUb, INNERCORE_REDUCE_MAXSUM_V_MTE2,
            INNERCORE_INTRAPARTIALO_V_MTE2, INNERCORE_REDUCE_MTE2_V);

        AttentionCommon::StageBroadcastMaxSum(mqsmlaIntraLayout, intraCoreCombineBase, intraWorkspaceIdx,
                                              stagingMOffset + startRow, dealRowCount, lseBroadcastUb, sumBroadcastUb,
                                              INNERCORE_STAGE2, INNERCORE_STAGE_FD_MTE3_V);
        if constexpr (!HIGH_PERF) {
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
        }
        if (runInfo.isCrossCoreSplit && runInfo.s2LoopCount == runInfo.s2LoopLimit) {
            AttentionCommon::StageBroadcastMaxSum(mqsmlaCrossLayout, crossCoreCombineBase, crossWorkspaceIdx,
                                                  stagingMOffset + startRow, dealRowCount, lseBroadcastUb,
                                                  sumBroadcastUb, INNERCORE_STAGE2, INNERCORE_STAGE_FD_MTE3_V);
        }
        startRow += mqsmlaIntraLayout.chunkRows;
    }

    AttentionCommon::StageVec2PartialOAndWait<T>(
        mqsmlaIntraLayout, intraCoreCombineGm, intraWorkspaceIdx, stagingMOffset, runInfo.vec2MRealSize,
        static_cast<uint32_t>(constInfo.dSizeV), vec2ResUb, INNERCORE_STAGE2, INNERCORE_STAGE_FD_MTE3_V);
    if (runInfo.isCrossCoreSplit && runInfo.s2LoopCount == runInfo.s2LoopLimit) {
        AttentionCommon::StageVec2PartialOAndWait<T>(
            mqsmlaCrossLayout, crossCoreCombineGm, crossWorkspaceIdx, stagingMOffset, runInfo.vec2MRealSize,
            static_cast<uint32_t>(constInfo.dSizeV), vec2ResUb, INNERCORE_STAGE2, INNERCORE_STAGE_FD_MTE3_V);
    }
    if (runInfo.s2LoopCount < runInfo.s2LoopLimit) {
        SetFlag<HardEvent::MTE3_MTE2>(INNERCORE_INTRALSE_MTE3_MTE2(runInfo.multiCoreIdxMod2));
        SetFlag<HardEvent::MTE3_MTE2>(INNERCORE_INTRAATTN_MTE3_MTE2(runInfo.multiCoreIdxMod2));
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::ProcessVec2(StaticBuffer<T> &bmm2ResBuf, RunInfo<HIGH_PERF> &runInfo,
                                                               ConstInfo<HIGH_PERF> &constInfo)
{
    CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CROSSCORE_BMM2);
    if (unlikely(runInfo.vec2MBaseSize == 0)) {
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CROSSCORE_BMM2);
        return;
    }

    runInfo.vec2S1RealSize = runInfo.vec2S1BaseSize;
    runInfo.vec2MRealSize = runInfo.vec2MBaseSize;
    int64_t mqsmlaVec2CalcSize = runInfo.vec2MRealSize * dTemplateAlign64;
    LocalTensor<T> mqsmlaVec2ResUb = this->stage2OutBufs.tensor;
    LocalTensor<T> mqsmlaMmRes = bmm2ResBuf.tensor;
    WaitFlag<HardEvent::MTE3_V>(INNERCORE_STAGE2);
    bool mqsmlaNeedIntraBlockReduce = false;
    if constexpr (IS_BATCH_CONSISTENCY) {
        mqsmlaNeedIntraBlockReduce = runInfo.isLastBase && runInfo.isFirstS2SplitCore && runInfo.reduceBlockId > 0;
        if (mqsmlaNeedIntraBlockReduce) {
            WaitFlag<HardEvent::V_MTE2>(INNERCORE_INTRAPARTIALO_V_MTE2);
            WaitFlag<HardEvent::V_MTE2>(INNERCORE_REDUCE_MAXSUM_V_MTE2);
        }
    }
    bool mqsmlaIsFirstVec2Base = runInfo.s2LoopCount == 0;
    if constexpr (IS_BATCH_CONSISTENCY) {
        mqsmlaIsFirstVec2Base = runInfo.isFirstBase;
    }
    if (unlikely(mqsmlaIsFirstVec2Base)) {
        DataCopy(mqsmlaVec2ResUb, mqsmlaMmRes, mqsmlaVec2CalcSize);
    } else {
        LocalTensor<T> expUb = softmaxExpBufs[runInfo.taskIdMod2].tensor;
        if constexpr (IS_BATCH_CONSISTENCY) {
            if (runInfo.isLastBase) {
                LocalTensor<float> mqsmlaSumUb = this->softmaxFinalSumBufs[runInfo.taskIdMod2].tensor;
                FlashUpdateLastNew<T, Q_T, OUTPUT_T, dTemplateAlign64, false, false>(
                    mqsmlaVec2ResUb, mqsmlaMmRes, mqsmlaVec2ResUb, expUb, expUb, mqsmlaSumUb, runInfo.vec2MRealSize,
                    dTemplateAlign64, 1.0, 1.0);
            } else {
                FlashUpdateNew<T, Q_T, OUTPUT_T, dTemplateAlign64, false, false>(
                    mqsmlaVec2ResUb, mqsmlaMmRes, mqsmlaVec2ResUb, expUb, expUb, runInfo.vec2MRealSize,
                    dTemplateAlign64, 1.0, 1.0);
            }
        } else {
            if (runInfo.s2LoopCount < runInfo.s2LoopLimit) {
                FlashUpdateNew<T, Q_T, OUTPUT_T, dTemplateAlign64, false, false>(
                    mqsmlaVec2ResUb, mqsmlaMmRes, mqsmlaVec2ResUb, expUb, expUb, runInfo.vec2MRealSize,
                    dTemplateAlign64, 1.0, 1.0);
            } else {
                LocalTensor<float> mqsmlaSumUb = this->softmaxSumBufs[runInfo.multiCoreIdxMod2].tensor;
                FlashUpdateLastNew<T, Q_T, OUTPUT_T, dTemplateAlign64, false, false>(
                    mqsmlaVec2ResUb, mqsmlaMmRes, mqsmlaVec2ResUb, expUb, expUb, mqsmlaSumUb, runInfo.vec2MRealSize,
                    dTemplateAlign64, 1.0, 1.0);
            }
        }
    }

    if constexpr (IS_BATCH_CONSISTENCY) {
        if (runInfo.isLastBase) {
            if (unlikely(mqsmlaIsFirstVec2Base)) {
                LocalTensor<float> mqsmlaSumUb = this->softmaxFinalSumBufs[runInfo.taskIdMod2].tensor;
                LastDivNew<T, Q_T, OUTPUT_T, dTemplateAlign64, false>(mqsmlaVec2ResUb, mqsmlaVec2ResUb, mqsmlaSumUb,
                                                                      runInfo.vec2MRealSize, dTemplateAlign64, 1.0);
            }
            if (mqsmlaNeedIntraBlockReduce) {
                SetFlag<HardEvent::V_MTE2>(INNERCORE_INTRAPARTIALO_V_MTE2);
                SetFlag<HardEvent::V_MTE2>(INNERCORE_REDUCE_MAXSUM_V_MTE2);
                ReduceIntraBlockAndStage(runInfo, constInfo, mqsmlaVec2ResUb, mqsmlaMmRes);
                if (!runInfo.isCrossCoreSplit && runInfo.s2LoopCount == runInfo.s2LoopLimit) {
                    this->CopyOutAttentionOut(runInfo, constInfo, mqsmlaVec2ResUb, 0, mqsmlaVec2CalcSize);
                }
            } else {
                if (runInfo.isFirstS2SplitCore && runInfo.reduceBlockId == 0 &&
                    runInfo.s2LoopCount < runInfo.s2LoopLimit) {
                    AttentionCommon::S2SplitFdStagingLayout stagingLayout = {
                        constInfo.gSize, dTemplateAlign64, GetStagingSlotNum(true),
                        AttentionCommon::FD_BROADCAST_ELEMS_PER_ROW, AttentionCommon::FD_REDUCE_CHUNK_ROWS};
                    int64_t mqsmlaStagingMOffset = GetFaStagingMOffset(runInfo, constInfo);
                    AttentionCommon::StageVec2PartialOAndWait<T>(
                        stagingLayout, intraCoreCombineGm, GetIntraCoreWorkspaceIdx(runInfo, constInfo),
                        mqsmlaStagingMOffset, runInfo.vec2MRealSize, static_cast<uint32_t>(constInfo.dSizeV),
                        mqsmlaVec2ResUb, INNERCORE_STAGE2, INNERCORE_STAGE_FD_MTE3_V);
                    SetFlag<HardEvent::MTE3_MTE2>(INNERCORE_INTRAATTN_MTE3_MTE2(runInfo.multiCoreIdxMod2));
                }
                if (runInfo.isCrossCoreSplit &&
                    (!runInfo.isFirstS2SplitCore || (runInfo.isFirstS2SplitCore && runInfo.reduceBlockId == 0 &&
                                                     runInfo.s2LoopCount == runInfo.s2LoopLimit))) {
                    AttentionCommon::S2SplitFdStagingLayout stagingLayout = {
                        constInfo.gSize, dTemplateAlign64, GetStagingSlotNum(false),
                        AttentionCommon::FD_BROADCAST_ELEMS_PER_ROW, AttentionCommon::FD_REDUCE_CHUNK_ROWS};
                    uint32_t mqsmlaWorkspaceIdx = GetCrossCoreWorkspaceIdx(runInfo);
                    int64_t mqsmlaStagingMOffset = GetFaStagingMOffset(runInfo, constInfo);
                    AttentionCommon::StageVec2PartialOAndWait<T>(
                        stagingLayout, crossCoreCombineGm, mqsmlaWorkspaceIdx, mqsmlaStagingMOffset,
                        runInfo.vec2MRealSize, static_cast<uint32_t>(constInfo.dSizeV), mqsmlaVec2ResUb,
                        INNERCORE_STAGE2, INNERCORE_STAGE_FD_MTE3_V);
                } else if (!runInfo.isCrossCoreSplit && runInfo.s2LoopCount == runInfo.s2LoopLimit) {
                    this->CopyOutAttentionOut(runInfo, constInfo, mqsmlaVec2ResUb, 0, mqsmlaVec2CalcSize);
                }
            }
        }
    } else {
        if (runInfo.s2LoopCount == runInfo.s2LoopLimit) {
            if (unlikely(runInfo.s2LoopCount == 0)) {
                LocalTensor<float> mqsmlaSumUb = this->softmaxSumBufs[runInfo.multiCoreIdxMod2].tensor;
                LastDivNew<T, Q_T, OUTPUT_T, dTemplateAlign64, false>(mqsmlaVec2ResUb, mqsmlaVec2ResUb, mqsmlaSumUb,
                                                                      runInfo.vec2MRealSize, dTemplateAlign64, 1.0);
            }
            if (runInfo.isCrossCoreSplit) {
                AttentionCommon::S2SplitFdStagingLayout stagingLayout = {
                    constInfo.gSize, dTemplateAlign64, GetStagingSlotNum(), AttentionCommon::FD_BROADCAST_ELEMS_PER_ROW,
                    AttentionCommon::FD_REDUCE_CHUNK_ROWS};
                uint32_t mqsmlaWorkspaceIdx = GetCrossCoreWorkspaceIdx(runInfo);
                int64_t mqsmlaStagingMOffset = GetFaStagingMOffset(runInfo, constInfo);
                AttentionCommon::StageVec2PartialOAndWait<T>(stagingLayout, stagingOutGm, mqsmlaWorkspaceIdx,
                                                             mqsmlaStagingMOffset, runInfo.vec2MRealSize,
                                                             static_cast<uint32_t>(constInfo.dSizeV), mqsmlaVec2ResUb,
                                                             INNERCORE_STAGE2, INNERCORE_STAGE_FD_MTE3_V);
            } else {
                this->CopyOutAttentionOut(runInfo, constInfo, mqsmlaVec2ResUb, 0, mqsmlaVec2CalcSize);
            }
        }
    }
    CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CROSSCORE_BMM2);
    SetFlag<HardEvent::MTE3_V>(INNERCORE_STAGE2);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::ProcessFlashDecode(FdRunInfo &fdRunInfo,
                                                                      ConstInfo<HIGH_PERF> &constInfo)
{
    InitFDBuffers(fdRunInfo);
    int64_t seqOffset = 0;
    if constexpr (LAYOUT_T == QSMLA_LAYOUT::TND) {
        seqOffset = this->cuSeqlensQGm.GetValue(fdRunInfo.bn2Idx);
    } else {
        seqOffset = fdRunInfo.bn2Idx * constInfo.s1Size;
    }
    int64_t attentionOutOffset =
        seqOffset * constInfo.n2GDv + fdRunInfo.mIdx * constInfo.n2GDv + fdRunInfo.mStartIdx * constInfo.dSizeV;
    int64_t softmaxLseOffset = 0;
    if constexpr (LAYOUT_T == QSMLA_LAYOUT::TND) {
        softmaxLseOffset = (seqOffset + fdRunInfo.mIdx) * constInfo.gSize + fdRunInfo.mStartIdx;
    } else {
        softmaxLseOffset =
            (fdRunInfo.bn2Idx * constInfo.s1Size + fdRunInfo.mIdx) * constInfo.gSize + fdRunInfo.mStartIdx;
    }
    LocalTensor<T> mqsmlaAccumulatedO = this->fdBuffers.accumOut.tensor.template ReinterpretCast<T>();
    LocalTensor<float> mqsmlaLseExpUb = this->fdBuffers.lseExp.tensor.template ReinterpretCast<float>();
    LocalTensor<float> mqsmlaBlockMaxUb = this->fdBuffers.blockMax.tensor.template ReinterpretCast<float>();
    LocalTensor<float> mqsmlaBlockSumUb = this->fdBuffers.blockSum.tensor.template ReinterpretCast<float>();
    LocalTensor<T> mqsmlaPartialOFp32 = this->fdBuffers.partialO.tensor.template ReinterpretCast<T>();

    AttentionCommon::S2SplitFdStagingLayout stagingLayout = {constInfo.gSize, dTemplateAlign64, GetStagingSlotNum(),
                                                             AttentionCommon::FD_BROADCAST_ELEMS_PER_ROW,
                                                             AttentionCommon::FD_REDUCE_CHUNK_ROWS};
    int64_t attentionOutRowStride =
        static_cast<int64_t>(constInfo.dSizeV) + static_cast<int64_t>(constInfo.attentionOutStride) / sizeof(OUTPUT_T);
    int64_t startRow = 0;
    bool softmaxLseFlag = false;
    if constexpr (!HIGH_PERF) {
        softmaxLseFlag = constInfo.isSoftmaxLseEnable;
    }
    while (startRow < fdRunInfo.mNum) {
        int64_t mqsmlaDealRowCount = AttentionCommon::FD_REDUCE_CHUNK_ROWS;
        if (startRow + mqsmlaDealRowCount > fdRunInfo.mNum) {
            mqsmlaDealRowCount = fdRunInfo.mNum - startRow;
        }
        WaitFlag<HardEvent::MTE3_V>(INNERCORE_FD_MTE3_V);
        if constexpr (IS_BATCH_CONSISTENCY) {
            WaitFlag<HardEvent::MTE3_MTE2>(INNERCORE_FD_MTE3_MTE2);
            AttentionCommon::ReducePairwiseWithLse<T, dTemplateAlign64>(
                stagingLayout, fdStagingBase, fdRunInfo.workspaceIdx, fdRunInfo.workspaceNum,
                static_cast<int64_t>(fdRunInfo.mStartIdx) + startRow, mqsmlaDealRowCount,
                static_cast<int64_t>(constInfo.dSizeV), mqsmlaAccumulatedO, mqsmlaLseExpUb, mqsmlaBlockMaxUb,
                mqsmlaBlockSumUb, mqsmlaPartialOFp32, softmaxLseFlag, softmaxLseGm, softmaxLseOffset + startRow,
                INNERCORE_FD_V_MTE2(0), INNERCORE_FD_V_MTE2(1), INNERCORE_FD_MTE2_V, INNERCORE_LSE_V_MTE3,
                INNERCORE_LSE_MTE3_V);
        } else {
            AttentionCommon::ReduceWithLse<T, dTemplateAlign64>(
                stagingLayout, fdStagingBase, fdRunInfo.workspaceIdx, fdRunInfo.workspaceNum,
                static_cast<int64_t>(fdRunInfo.mStartIdx) + startRow, mqsmlaDealRowCount,
                static_cast<int64_t>(constInfo.dSizeV), mqsmlaAccumulatedO, mqsmlaLseExpUb, mqsmlaBlockMaxUb,
                mqsmlaBlockSumUb, mqsmlaPartialOFp32, softmaxLseFlag, softmaxLseGm, softmaxLseOffset + startRow,
                INNERCORE_FD_V_MTE2(0), INNERCORE_FD_V_MTE2(1), INNERCORE_FD_MTE2_V, INNERCORE_LSE_V_MTE3,
                INNERCORE_LSE_MTE3_V);
        }

        RunInfo<HIGH_PERF> runInfo;
        runInfo.vec2MRealSize = mqsmlaDealRowCount;
        runInfo.attentionOutOffset = attentionOutOffset + startRow * attentionOutRowStride;
        int64_t mqsmlaVec2CalcSize = mqsmlaDealRowCount * dTemplateAlign64;
        this->CopyOutAttentionOut(runInfo, constInfo, mqsmlaAccumulatedO, 0, mqsmlaVec2CalcSize);
        if constexpr (IS_BATCH_CONSISTENCY) {
            SetFlag<HardEvent::MTE3_MTE2>(INNERCORE_FD_MTE3_MTE2);
        }
        SetFlag<HardEvent::MTE3_V>(INNERCORE_FD_MTE3_V);
        startRow += mqsmlaDealRowCount;
    }
}

TEMPLATES_DEF_NO_DEFAULT
template <typename VEC2_RES_T>
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::Bmm2DataCopyOut(RunInfo<HIGH_PERF> &runInfo,
                                                                   ConstInfo<HIGH_PERF> &constInfo,
                                                                   LocalTensor<VEC2_RES_T> &vec2ResUb,
                                                                   int64_t vec2S1Idx, int64_t vec2CalcSize)
{
    LocalTensor<OUTPUT_T> mqsmlaAttenOut;
    int64_t mqsmlaDSizeAligned64 = (int64_t)dTemplateAlign64;

    mqsmlaAttenOut.SetAddr(vec2ResUb.address_);
    Cast(mqsmlaAttenOut, vec2ResUb, RoundMode::CAST_ROUND, vec2CalcSize);
    SetFlag<HardEvent::V_MTE3>(INNERCORE_STAGE2);
    WaitFlag<HardEvent::V_MTE3>(INNERCORE_STAGE2);
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockLen = constInfo.dSizeV * sizeof(OUTPUT_T);
    dataCopyParams.srcStride =
        (mqsmlaDSizeAligned64 - constInfo.dSizeV) >> 4; // 以32B为单位偏移，bf16类型即偏移16个数，右移4
    dataCopyParams.dstStride = constInfo.attentionOutStride;
    dataCopyParams.blockCount = runInfo.vec2MRealSize;

    int64_t outputGmOffset = runInfo.attentionOutOffset;
    if (!constInfo.hasRopeInput) {
        // All internal offsets/FD buffers use virtual 512-wide rows. Convert once at GM.
        outputGmOffset = runInfo.attentionOutOffset / constInfo.dSizeV * constInfo.qInputD;
        dataCopyParams.blockLen = constInfo.qInputD * sizeof(OUTPUT_T);
        dataCopyParams.srcStride = (mqsmlaDSizeAligned64 - constInfo.qInputD) >> 4;
        dataCopyParams.dstStride = constInfo.attentionOutStride / constInfo.dSizeV * constInfo.qInputD;
    }
    DataCopyPad(this->attentionOutGm[outputGmOffset], mqsmlaAttenOut, dataCopyParams);
}

TEMPLATES_DEF_NO_DEFAULT
template <typename VEC2_RES_T>
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::CopyOutAttentionOut(RunInfo<HIGH_PERF> &runInfo,
                                                                       ConstInfo<HIGH_PERF> &constInfo,
                                                                       LocalTensor<VEC2_RES_T> &vec2ResUb,
                                                                       int64_t vec2S1Idx, int64_t vec2CalcSize)
{
    this->Bmm2DataCopyOut(runInfo, constInfo, vec2ResUb, vec2S1Idx, vec2CalcSize);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::InitOutputSingleCore(ConstInfo<HIGH_PERF> &constInfo)
{
    uint32_t coreNum = GetBlockNum();
    uint32_t vecCoreNum = CV_RATIO * coreNum;
    uint64_t totalOutputSize = 0;

    // n2 = 1, n1 = gn2 = gSize
    if (LAYOUT_T == QSMLA_LAYOUT::BSND) {
        totalOutputSize = constInfo.bSize * constInfo.gSize * constInfo.s1Size * constInfo.qInputD;
    } else if (LAYOUT_T == QSMLA_LAYOUT::TND) {
        totalOutputSize = constInfo.s1Size * constInfo.gSize * constInfo.qInputD;
    }

    static constexpr uint32_t ATTEN_OUT_POP_BUF_START_ADDR = 184U * 1024U;
    static constexpr uint32_t ATTEN_OUT_POP_BUF_ELE_SIZE = (32U * 1024U) / sizeof(OUTPUT_T);
    if (coreNum != 0 && totalOutputSize > 0) {
        AttentionCommon::InitOutput<OUTPUT_T, initOutputEventId, ATTEN_OUT_POP_BUF_START_ADDR,
                                    ATTEN_OUT_POP_BUF_ELE_SIZE, false>(this->attentionOutGm, totalOutputSize,
                                                                       vecCoreNum, static_cast<OUTPUT_T>(0));
    }
    if constexpr (!HIGH_PERF) {
        if (constInfo.isSoftmaxLseEnable && this->isSoftmaxLseGmValid) {
            uint64_t totalLseSize = 0;
            if constexpr (LAYOUT_T == QSMLA_LAYOUT::BSND) {
                totalLseSize = constInfo.bSize * constInfo.n2Size * constInfo.gSize * constInfo.s1Size;
            } else if constexpr (LAYOUT_T == QSMLA_LAYOUT::TND) {
                totalLseSize = constInfo.n2Size * constInfo.s1Size * constInfo.gSize;
            }
            static constexpr uint32_t LSE_POP_BUF_START_ADDR = 216U * 1024U;
            static constexpr uint32_t LSE_POP_BUF_ELE_SIZE = (32U * 1024U) / sizeof(float);
            if (coreNum != 0 && totalLseSize > 0) {
                AttentionCommon::InitOutput<float, initOutputEventId, LSE_POP_BUF_START_ADDR, LSE_POP_BUF_ELE_SIZE,
                                            false>(this->softmaxLseGm, totalLseSize, vecCoreNum, static_cast<float>(0));
            }
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::CleanOutput(__gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse,
                                                               ConstInfo<HIGH_PERF> &constInfo)
{
    if ASCEND_IS_AIV {
        this->attentionOutGm.SetGlobalBuffer((__gm__ OUTPUT_T *)attentionOut);
        if constexpr (!HIGH_PERF) {
            if (constInfo.isSoftmaxLseEnable && softmaxLse != nullptr) {
                this->softmaxLseGm.SetGlobalBuffer((__gm__ float *)softmaxLse);
                this->isSoftmaxLseGmValid = true;
            }
        }
        if (constInfo.needInit == 1) {
            InitOutputSingleCore(constInfo);
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline int32_t CSABlockVec<TEMPLATE_ARGS>::GetSeqLenForPhyAddr(int32_t bIdx, bool hasActualSeq,
                                                                          bool hasCuSeqlens,
                                                                          GlobalTensor<int32_t> &actualSeqGm,
                                                                          GlobalTensor<int32_t> &cuSeqlensGm,
                                                                          int64_t defaultSize)
{
    if (hasActualSeq) {
        return actualSeqGm.GetValue(bIdx);
    }
    if (hasCuSeqlens) {
        return cuSeqlensGm.GetValue(bIdx + 1) - cuSeqlensGm.GetValue(bIdx);
    }
    return defaultSize;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline PhyAddrValidInfo CSABlockVec<TEMPLATE_ARGS>::CalcPhyAddrValidInfo(bool isOriKv, int32_t actualS1Size,
                                                                                    int32_t actualOriS2Size,
                                                                                    int32_t restoredSize,
                                                                                    ConstInfo<HIGH_PERF> &constInfo)
{
    // per-batch执行一次,  per-s1循环内不再判断maskmode
    PhyAddrValidInfo validInfo;
    if constexpr (TEMPLATE_MODE == QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                  TEMPLATE_MODE == QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        validInfo.oriS2Act = actualOriS2Size;
        if constexpr (HIGH_PERF) {
            validInfo.oriLeftBias =
                (constInfo.oriWinLeft == -1) ? PhyAddrValidInfo::BIAS_UNBOUND : constInfo.oriWinLeft + 1;
            validInfo.oriRightBias =
                (constInfo.oriWinRight == -1) ? PhyAddrValidInfo::BIAS_UNBOUND : constInfo.oriWinRight;
        } else {
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
    }
    if constexpr (TEMPLATE_MODE == QSMLATemplateMode::CSA_TEMPLATE_MODE ||
                  TEMPLATE_MODE == QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        if (!isOriKv) {
            validInfo.cmpTopkMode = (constInfo.cmpMaskMode == 0U);
            validInfo.cmpBase = restoredSize - actualS1Size + 1;
        }
    }
    return validInfo;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline int32_t CSABlockVec<TEMPLATE_ARGS>::CalcCurValidS2ForPhyAddr(
    uint32_t bIdx, int32_t s1Idx, int32_t actualS1Size, bool isOriKv, GlobalTensor<int32_t> &cuSeqlensQGm,
    GlobalTensor<int32_t> &topkLengthGm, ConstInfo<HIGH_PERF> &constInfo, int32_t sparseBlockCount,
    const PhyAddrValidInfo &validInfo)
{
    bool topkMode = false;
    bool hasTopk = false;
    if constexpr (!HIGH_PERF) {
        if constexpr (TEMPLATE_MODE == QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                      TEMPLATE_MODE == QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
            if (isOriKv) {
                topkMode = validInfo.oriTopkMode;
                hasTopk = constInfo.hasOriTopkLength;
            }
        }
        if constexpr (TEMPLATE_MODE == QSMLATemplateMode::CSA_TEMPLATE_MODE ||
                      TEMPLATE_MODE == QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
            if (!isOriKv) {
                topkMode = validInfo.cmpTopkMode;
                hasTopk = constInfo.hasCmpTopkLength;
            }
        }
    }
    if (topkMode) {
        uint64_t topkIdx =
            (LAYOUT_T == QSMLA_LAYOUT::TND) ? (cuSeqlensQGm.GetValue(bIdx) + s1Idx) : (bIdx * constInfo.s1Size + s1Idx);
        int32_t topkLen = hasTopk ? topkLengthGm.GetValue(topkIdx) : sparseBlockCount;
        return Min(topkLen, sparseBlockCount);
    }

    if constexpr (TEMPLATE_MODE == QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                  TEMPLATE_MODE == QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        if (isOriKv) {
            int64_t thr = validInfo.oriS2Act - actualS1Size + 1 + s1Idx;
            int64_t leftBound = Max(thr - validInfo.oriLeftBias, 0);
            int64_t rightBound = Min(thr + validInfo.oriRightBias, static_cast<int64_t>(validInfo.oriS2Act));
            return Min(static_cast<int32_t>(Max(0, rightBound - leftBound)), sparseBlockCount);
        }
    }
    if constexpr (TEMPLATE_MODE == QSMLATemplateMode::CSA_TEMPLATE_MODE ||
                  TEMPLATE_MODE == QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        int64_t numerator = Max(validInfo.cmpBase + s1Idx, 0);
        return Min(sparseBlockCount, static_cast<int32_t>(numerator / static_cast<int32_t>(constInfo.cmpRatio)));
    }
    return 0;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::GetKVPhyAddrForKvType(
    uint32_t bN2StartIdx, uint32_t bN2EndIdx, uint32_t gS1StartIdx, uint32_t nextGs1Idx, bool hasActualSeqQlen,
    bool hasCuSeqlensQ, bool hasActualSeqKvlen, bool hasCuSeqlensKv, GlobalTensor<int32_t> &actualSeqQlenGm,
    GlobalTensor<int32_t> &cuSeqlensQGm, GlobalTensor<int32_t> &actualSeqKvlenGm, GlobalTensor<int32_t> &cuSeqlensKvGm,
    GlobalTensor<int32_t> &topkLengthGm, GlobalTensor<int32_t> &cmpResidualKvGm, ConstInfo<HIGH_PERF> &constInfo,
    GlobalTensor<int32_t> &blockTableGm, GlobalTensor<int32_t> &sparseIndicesGm, GlobalTensor<uint32_t> &phyAddrGm,
    uint32_t kvStride, uint32_t blockSize, uint32_t maxBlockNumPerBatch, uint32_t sparseBlockCount,
    uint32_t alignedSparseBlockCount, bool isOriKv)
{
    constexpr uint16_t s2NumPerLoop = 128;
    constexpr uint32_t vecCoreNum = IS_SPLIT_G ? 4 : 2;
    uint32_t vecCoreIdx = IS_SPLIT_G ? constInfo.aivIdx % 4 : constInfo.aivIdx % 2;
    int16_t shiftRightNum = 0;
    for (int32_t size = static_cast<int32_t>(blockSize); size > 1; size >>= 1) {
        ++shiftRightNum;
    }

    uint32_t phyAddrUb = 0;
    LocalTensor<int32_t> blkTableUb(TPosition::VECIN, phyAddrUb, maxBlockNumPerBatch);
    phyAddrUb = CeilAlign(phyAddrUb + maxBlockNumPerBatch * sizeof(int32_t), BUFFER_SIZE_BYTE_32B);
    LocalTensor<int32_t> sparseIdxUb(TPosition::VECIN, phyAddrUb, alignedSparseBlockCount);
    phyAddrUb += alignedSparseBlockCount * sizeof(int32_t);
    LocalTensor<uint32_t> kvPhyAddrUb(TPosition::VECIN, phyAddrUb,
                                      alignedSparseBlockCount * 2); // 2：每个稀疏块需要存储两个物理地址

    int64_t totalValidS1 = 0;
    uint32_t tmpGS1Start = gS1StartIdx;
    for (uint32_t bIdx = bN2StartIdx; bIdx < bN2EndIdx; ++bIdx) {
        bool lastBN = bIdx == bN2EndIdx - 1;
        int32_t actualS1Size =
            GetSeqLenForPhyAddr(bIdx, hasActualSeqQlen, hasCuSeqlensQ, actualSeqQlenGm, cuSeqlensQGm, constInfo.s1Size);
        int32_t s1End = (lastBN && nextGs1Idx != 0) ? nextGs1Idx : actualS1Size;
        int32_t restoredSize = 0;
        if constexpr (TEMPLATE_MODE == QSMLATemplateMode::CSA_TEMPLATE_MODE ||
                      TEMPLATE_MODE == QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
            if (!isOriKv && constInfo.cmpMaskMode != 0) {
                int32_t actualKvSize = GetSeqLenForPhyAddr(bIdx, hasActualSeqKvlen, hasCuSeqlensKv, actualSeqKvlenGm,
                                                           cuSeqlensKvGm, constInfo.cmpS2Size);
                restoredSize = actualKvSize * static_cast<int32_t>(constInfo.cmpRatio) + cmpResidualKvGm.GetValue(bIdx);
            }
        }
        int32_t actualOriS2Size = 0;
        if constexpr (TEMPLATE_MODE == QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                      TEMPLATE_MODE == QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
            if (isOriKv && constInfo.oriMaskMode != 0) {
                actualOriS2Size = GetSeqLenForPhyAddr(bIdx, hasActualSeqKvlen, hasCuSeqlensKv, actualSeqKvlenGm,
                                                      cuSeqlensKvGm, constInfo.s2Size);
            }
        }

        PhyAddrValidInfo validInfo =
            CalcPhyAddrValidInfo(isOriKv, actualS1Size, actualOriS2Size, restoredSize, constInfo);
        for (int32_t s1Idx = tmpGS1Start; s1Idx < s1End; ++s1Idx) {
            int32_t validS2 = CalcCurValidS2ForPhyAddr(bIdx, s1Idx, actualS1Size, isOriKv, cuSeqlensQGm, topkLengthGm,
                                                       constInfo, sparseBlockCount, validInfo);
            totalValidS1 += validS2 > 0;
        }
        tmpGS1Start = 0;
    }
    int64_t rowsPerCore = totalValidS1 / vecCoreNum;
    int64_t tailRows = totalValidS1 % vecCoreNum;
    int64_t rowStart = rowsPerCore * vecCoreIdx + Min(static_cast<int64_t>(vecCoreIdx), tailRows);
    int64_t rowCount = rowsPerCore + (vecCoreIdx < static_cast<uint32_t>(tailRows) ? 1 : 0);
    if (rowCount == 0) {
        return;
    }

    int64_t validCounter = 0;
    int64_t processedCount = 0;
    tmpGS1Start = gS1StartIdx;
    bool done = false;
    SetFlag<HardEvent::V_MTE2>(INNERCORE_PHYADDR_BLKTABLE_FREE);
    SetFlag<HardEvent::V_MTE2>(INNERCORE_PHYADDR_SPARSEIDX_FREE);
    SetFlag<HardEvent::MTE3_V>(INNERCORE_PHYADDR_KVADDR_FREE);
    for (uint32_t bIdx = bN2StartIdx; bIdx < bN2EndIdx && !done; ++bIdx) {
        bool lastBN = bIdx == bN2EndIdx - 1;
        int32_t actualS1Size =
            GetSeqLenForPhyAddr(bIdx, hasActualSeqQlen, hasCuSeqlensQ, actualSeqQlenGm, cuSeqlensQGm, constInfo.s1Size);
        int64_t bS1Idx = (LAYOUT_T == QSMLA_LAYOUT::TND) ?
                             (hasCuSeqlensQ ? cuSeqlensQGm.GetValue(bIdx) : constInfo.s1Size * bIdx) :
                             constInfo.s1Size * bIdx;
        int32_t s1End = (lastBN && nextGs1Idx != 0) ? nextGs1Idx : actualS1Size;
        int32_t restoredSize = 0;
        if constexpr (TEMPLATE_MODE == QSMLATemplateMode::CSA_TEMPLATE_MODE ||
                      TEMPLATE_MODE == QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
            if (!isOriKv && constInfo.cmpMaskMode != 0) {
                int32_t actualKvSize = GetSeqLenForPhyAddr(bIdx, hasActualSeqKvlen, hasCuSeqlensKv, actualSeqKvlenGm,
                                                           cuSeqlensKvGm, constInfo.cmpS2Size);
                restoredSize = actualKvSize * static_cast<int32_t>(constInfo.cmpRatio) + cmpResidualKvGm.GetValue(bIdx);
            }
        }
        int32_t actualOriS2Size = 0;
        if constexpr (TEMPLATE_MODE == QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                      TEMPLATE_MODE == QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
            if (isOriKv && constInfo.oriMaskMode != 0) {
                actualOriS2Size = GetSeqLenForPhyAddr(bIdx, hasActualSeqKvlen, hasCuSeqlensKv, actualSeqKvlenGm,
                                                      cuSeqlensKvGm, constInfo.s2Size);
            }
        }

        PhyAddrValidInfo validInfo =
            CalcPhyAddrValidInfo(isOriKv, actualS1Size, actualOriS2Size, restoredSize, constInfo);
        WaitFlag<HardEvent::V_MTE2>(INNERCORE_PHYADDR_BLKTABLE_FREE);
        AttentionCommon::CopyPaTableToUb(blkTableUb, bIdx, blockTableGm, maxBlockNumPerBatch);
        SetFlag<HardEvent::MTE2_V>(INNERCORE_PHYADDR_BLKTABLE_READY);
        WaitFlag<HardEvent::MTE2_V>(INNERCORE_PHYADDR_BLKTABLE_READY);
        for (int32_t s1Idx = tmpGS1Start; s1Idx < s1End; ++s1Idx) {
            int32_t validS2 = CalcCurValidS2ForPhyAddr(bIdx, s1Idx, actualS1Size, isOriKv, cuSeqlensQGm, topkLengthGm,
                                                       constInfo, sparseBlockCount, validInfo);
            if (validS2 <= 0) {
                continue;
            }
            if (validCounter < rowStart || validCounter >= rowStart + rowCount) {
                ++validCounter;
                continue;
            }
            ++validCounter;
            uint16_t s2Loop = (validS2 + s2NumPerLoop - 1) / s2NumPerLoop;
            int32_t s2Tail = validS2 - (s2Loop - 1) * s2NumPerLoop;
            WaitFlag<HardEvent::V_MTE2>(INNERCORE_PHYADDR_SPARSEIDX_FREE);
            AttentionCommon::CopySparseIdxToUb(sparseIdxUb, bS1Idx, s1Idx, validS2, sparseIndicesGm, sparseBlockCount);
            SetFlag<HardEvent::MTE2_V>(INNERCORE_PHYADDR_SPARSEIDX_READY);
            WaitFlag<HardEvent::MTE2_V>(INNERCORE_PHYADDR_SPARSEIDX_READY);
            WaitFlag<HardEvent::MTE3_V>(INNERCORE_PHYADDR_KVADDR_FREE);
            AttentionCommon::GetKVPhyAddrVFPa<uint32_t>(kvPhyAddrUb, sparseIdxUb, blkTableUb, s2Loop, s2Tail, blockSize,
                                                        shiftRightNum, constInfo.sparseBlockSize, constInfo.dSizeVInput,
                                                        kvStride);
            SetFlag<HardEvent::V_MTE2>(INNERCORE_PHYADDR_SPARSEIDX_FREE);
            SetFlag<HardEvent::V_MTE3>(INNERCORE_PHYADDR_KVADDR_READY);
            WaitFlag<HardEvent::V_MTE3>(INNERCORE_PHYADDR_KVADDR_READY);
            AttentionCommon::CopyPhyAddrToGm(kvPhyAddrUb, bS1Idx, s1Idx, validS2, s2NumPerLoop, phyAddrGm,
                                             alignedSparseBlockCount);
            SetFlag<HardEvent::MTE3_V>(INNERCORE_PHYADDR_KVADDR_FREE);
            if (++processedCount >= rowCount) {
                done = true;
                break;
            }
        }
        SetFlag<HardEvent::V_MTE2>(INNERCORE_PHYADDR_BLKTABLE_FREE);
        tmpGS1Start = 0;
    }
    WaitFlag<HardEvent::V_MTE2>(INNERCORE_PHYADDR_BLKTABLE_FREE);
    WaitFlag<HardEvent::V_MTE2>(INNERCORE_PHYADDR_SPARSEIDX_FREE);
    WaitFlag<HardEvent::MTE3_V>(INNERCORE_PHYADDR_KVADDR_FREE);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::GetKVPhyAddr(
    uint32_t hasLoad, uint32_t bN2StartIdx, uint32_t bN2EndIdx, uint32_t gS1StartIdx, uint32_t nextGs1Idx,
    bool hasActualSeqQlen, bool hasCuSeqlensQ, bool hasActualSeqOriKvlen, bool hasCuSeqlensOriKv,
    GlobalTensor<int32_t> &actualSeqOriKvlenGm, GlobalTensor<int32_t> &cuSeqlensOriKvGm,
    GlobalTensor<int32_t> &oriTopkLengthGm, bool hasActualSeqCmpKvlen, bool hasCuSeqlensCmpKv,
    GlobalTensor<int32_t> &actualSeqCmpKvlenGm, GlobalTensor<int32_t> &cuSeqlensCmpKvGm,
    GlobalTensor<int32_t> &cmpTopkLengthGm, GlobalTensor<int32_t> &cmpResidualKvGm,
    GlobalTensor<int32_t> &actualSeqQlenGm, GlobalTensor<int32_t> &cuSeqlensQGm, __gm__ uint8_t *workspace,
    ConstInfo<HIGH_PERF> &constInfo)
{
    if (hasLoad == 0) {
        return;
    }
    uint64_t v0ResSize = static_cast<uint64_t>(constInfo.s2BaseSize) * constInfo.dSize * sizeof(Q_T);
    uint64_t v0RegionSize = v0ResSize * 3 * (IS_SPLIT_G ? (GetBlockNum() >> 1U) : GetBlockNum());
    uint64_t totalBS1 =
        (LAYOUT_T == QSMLA_LAYOUT::TND) ? constInfo.s1Size : static_cast<uint64_t>(constInfo.bSize) * constInfo.s1Size;
    uint64_t oriPhyAddrSize = 0;
    if constexpr (TEMPLATE_MODE == QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                  TEMPLATE_MODE == QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        oriPhyAddrSize = totalBS1 * constInfo.alignedOriSparseBlockCount * sizeof(int64_t);
        oriKvPhyAddrGm.SetGlobalBuffer((__gm__ uint32_t *)(workspace + v0RegionSize));
        GetKVPhyAddrForKvType(bN2StartIdx, bN2EndIdx, gS1StartIdx, nextGs1Idx, hasActualSeqQlen, hasCuSeqlensQ,
                              hasActualSeqOriKvlen, hasCuSeqlensOriKv, actualSeqQlenGm, cuSeqlensQGm,
                              actualSeqOriKvlenGm, cuSeqlensOriKvGm, oriTopkLengthGm, cmpResidualKvGm, constInfo,
                              oriBlockTableGm, oriSparseIndicesGm, oriKvPhyAddrGm, constInfo.oriKvStride,
                              constInfo.oriBlockSize, constInfo.oriMaxBlockNumPerBatch, constInfo.oriSparseBlockCount,
                              constInfo.alignedOriSparseBlockCount, true);
    }
    if constexpr (TEMPLATE_MODE == QSMLATemplateMode::CSA_TEMPLATE_MODE ||
                  TEMPLATE_MODE == QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        cmpKvPhyAddrGm.SetGlobalBuffer((__gm__ uint32_t *)(workspace + v0RegionSize + oriPhyAddrSize));
        GetKVPhyAddrForKvType(bN2StartIdx, bN2EndIdx, gS1StartIdx, nextGs1Idx, hasActualSeqQlen, hasCuSeqlensQ,
                              hasActualSeqCmpKvlen, hasCuSeqlensCmpKv, actualSeqQlenGm, cuSeqlensQGm,
                              actualSeqCmpKvlenGm, cuSeqlensCmpKvGm, cmpTopkLengthGm, cmpResidualKvGm, constInfo,
                              cmpBlockTableGm, cmpSparseIndicesGm, cmpKvPhyAddrGm, constInfo.cmpKvStride,
                              constInfo.cmpBlockSize, constInfo.cmpMaxBlockNumPerBatch, constInfo.cmpSparseBlockCount,
                              constInfo.alignedCmpSparseBlockCount, false);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::InitGlobalBuffer(
    __gm__ uint8_t *oriKV, __gm__ uint8_t *cmpKV, __gm__ uint8_t *oriSparseIndices, __gm__ uint8_t *cmpSparseIndices,
    __gm__ uint8_t *oriBlockTable, __gm__ uint8_t *cmpBlockTable, __gm__ uint8_t *sequsedQ, __gm__ uint8_t *sinks,
    __gm__ uint8_t *sequsedOriKv, __gm__ uint8_t *sequsedCmpKv, __gm__ uint8_t *cmpResidualKv)
{
    oriKVGm.SetGlobalBuffer((__gm__ KV_T *)(oriKV));
    if (oriBlockTable != nullptr) {
        oriBlockTableGm.SetGlobalBuffer((__gm__ int32_t *)oriBlockTable);
    }

    if constexpr (TEMPLATE_MODE != QSMLATemplateMode::SWA_TEMPLATE_MODE &&
                  TEMPLATE_MODE != QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE) {
        cmpKVGm.SetGlobalBuffer((__gm__ KV_T *)cmpKV);
        if (cmpBlockTable != nullptr) {
            cmpBlockTableGm.SetGlobalBuffer((__gm__ int32_t *)cmpBlockTable);
        }
    }

    if constexpr (TEMPLATE_MODE == QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                  TEMPLATE_MODE == QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        oriSparseIndicesGm.SetGlobalBuffer((__gm__ int32_t *)oriSparseIndices);
    }

    if constexpr (TEMPLATE_MODE == QSMLATemplateMode::CSA_TEMPLATE_MODE ||
                  TEMPLATE_MODE == QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        cmpSparseIndicesGm.SetGlobalBuffer((__gm__ int32_t *)cmpSparseIndices);
    }

    if (sinks != nullptr) {
        sinksGm.SetGlobalBuffer((__gm__ T *)sinks);
        this->isSinks = true;
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::SoftmaxInitBuffer(uint32_t &ubBaseAddr)
{
    constexpr uint32_t softmaxBufSize = 256; // VF单次操作256Byte
    constexpr uint32_t softmaxElems = softmaxBufSize / sizeof(float);
    softmaxSumBufs[0] = {LocalTensor<float>(TPosition::VECIN, ubBaseAddr, softmaxElems), 0};
    ubBaseAddr += softmaxBufSize;
    softmaxSumBufs[1] = {LocalTensor<float>(TPosition::VECIN, ubBaseAddr, softmaxElems), 1};
    ubBaseAddr += softmaxBufSize;
    softmaxMaxBufs[0] = {LocalTensor<float>(TPosition::VECIN, ubBaseAddr, softmaxElems), 0};
    ubBaseAddr += softmaxBufSize;
    softmaxMaxBufs[1] = {LocalTensor<float>(TPosition::VECIN, ubBaseAddr, softmaxElems), 1};
    ubBaseAddr += softmaxBufSize;
    if constexpr (IS_BATCH_CONSISTENCY) {
        softmaxFinalSumBufs[0] = {LocalTensor<float>(TPosition::VECIN, ubBaseAddr, softmaxElems), 0};
        ubBaseAddr += softmaxBufSize;
        softmaxFinalSumBufs[1] = {LocalTensor<float>(TPosition::VECIN, ubBaseAddr, softmaxElems), 1};
        ubBaseAddr += softmaxBufSize;
        softmaxFinalMaxBufs[0] = {LocalTensor<float>(TPosition::VECIN, ubBaseAddr, softmaxElems), 0};
        ubBaseAddr += softmaxBufSize;
        softmaxFinalMaxBufs[1] = {LocalTensor<float>(TPosition::VECIN, ubBaseAddr, softmaxElems), 1};
        ubBaseAddr += softmaxBufSize;
        batchReduceTmpUb = {LocalTensor<float>(TPosition::VECIN, ubBaseAddr, 768), 0};
        ubBaseAddr += 768U * sizeof(float);
    }
    softmaxExpBufs[0] = {LocalTensor<T>(TPosition::VECIN, ubBaseAddr, softmaxBufSize / sizeof(T)), 0};
    ubBaseAddr += softmaxBufSize;
    softmaxExpBufs[1] = {LocalTensor<T>(TPosition::VECIN, ubBaseAddr, softmaxBufSize / sizeof(T)), 1};
    ubBaseAddr += softmaxBufSize;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::InitSinksBuffer(ConstInfo<HIGH_PERF> &constInfo)
{
    LocalTensor<T> mqsmlaSinksUb = this->sinksUb.tensor;
    const uint32_t mqsmlaMaxN = constInfo.gSize; // N最大支持128, sink shape是[N]
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = 1U;
    dataCopyParams.blockLen = mqsmlaMaxN * sizeof(T);
    dataCopyParams.srcStride = 0U;
    dataCopyParams.dstStride = 0U;
    DataCopyPadExtParams<T> padParams;
    DataCopyPad(mqsmlaSinksUb, this->sinksGm, dataCopyParams, padParams);
    SetFlag<AscendC::HardEvent::MTE2_V>(INNERCORE_SINKS_SYNC);
    WaitFlag<AscendC::HardEvent::MTE2_V>(INNERCORE_SINKS_SYNC);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::InitLocalBuffer(ConstInfo<HIGH_PERF> &constInfo, uint32_t ubBaseAddr)
{
    uint32_t ubAddr = ubBaseAddr;

    // {64, 16, 2}：每个向量操作单元处理的元素个数为64，每个处理块包含的向量数量为16，需要处理两个不同的数据流或阶段
    dequantScaleUb = {LocalTensor<float>(TPosition::VECIN, ubAddr, 64 * 16 * 2), 0};
    ubAddr += 64 * 16 * 2 * sizeof(float); // {64, 16, 2}：同上

    SoftmaxInitBuffer(ubAddr);

    commonUb = {LocalTensor<T>(TPosition::VECIN, ubAddr, 512 / sizeof(T)), 0}; // 512：缓冲区大小，单位是字节
    ubAddr += 512;                                                             // 512：同上
    sinksUb = {LocalTensor<T>(TPosition::VECIN, ubAddr, 512 / sizeof(T)), 0};  // 512：同上
    ubAddr += 512;                                                             // 512：同上
    if constexpr (!HIGH_PERF) {
        if (constInfo.isSoftmaxLseEnable) {
            outLseUbs[0] = {LocalTensor<float>(TPosition::VECIN, ubAddr, 256 / sizeof(float)),
                            0}; // 256：缓冲区大小，单位是字节
            ubAddr += 256U;
            outLseUbs[1] = {LocalTensor<float>(TPosition::VECIN, ubAddr, 256 / sizeof(float)), 1}; // 256：同上
            ubAddr += 256U;
        }
    }

    stage0InBufs[0] = {LocalTensor<KV_T>(TPosition::VECIN, ubAddr, v0BufferDSize * 16),
                       0};                       // 16：每个v0缓冲区块中存储的向量数量
    ubAddr += v0BufferDSize * 16 * sizeof(KV_T); // 16：同上
    stage0InBufs[1] = {LocalTensor<KV_T>(TPosition::VECIN, ubAddr, v0BufferDSize * 16), 1}; // 16：同上
    ubAddr += v0BufferDSize * 16 * sizeof(KV_T);                                            // 16：同上
    stage0OutBufs[0] = {LocalTensor<Q_T>(TPosition::VECIN, ubAddr, v0BufferDSize * (16U + 1)), 0};
    ubAddr += v0BufferDSize * (16U + 1) * sizeof(Q_T);
    stage0OutBufs[1] = {LocalTensor<Q_T>(TPosition::VECIN, ubAddr, v0BufferDSize * (16U + 1)), 1};
    ubAddr += v0BufferDSize * (16U + 1) * sizeof(Q_T);

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
    if constexpr (!HIGH_PERF) {
        if (constInfo.isSoftmaxLseEnable) {
            SetFlag<HardEvent::MTE3_V>(INNERCORE_LSE_MTE3_V);
        }
    }
    SetFlag<HardEvent::V_MTE2>(INNERCORE_STAGE0_IN(0));
    SetFlag<HardEvent::V_MTE2>(INNERCORE_STAGE0_IN(1));
    SetFlag<HardEvent::MTE3_V>(INNERCORE_STAGE0_OUT(0));
    SetFlag<HardEvent::MTE3_V>(INNERCORE_STAGE0_OUT(1));
    SetFlag<HardEvent::MTE3_V>(INNERCORE_STAGE1(0));
    SetFlag<HardEvent::MTE3_V>(INNERCORE_STAGE1(1));
    SetFlag<HardEvent::V_MTE2>(INNERCORE_FD_V_MTE2(0));
    SetFlag<HardEvent::V_MTE2>(INNERCORE_FD_V_MTE2(1));
    SetFlag<HardEvent::MTE3_V>(INNERCORE_FD_MTE3_V);
    if constexpr (IS_BATCH_CONSISTENCY) {
        SetFlag<HardEvent::MTE3_MTE2>(INNERCORE_FD_MTE3_MTE2);
    }

    if (this->isSinks) {
        InitSinksBuffer(constInfo);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::FreeEvent(ConstInfo<HIGH_PERF> &constInfo)
{
    if constexpr (IS_BATCH_CONSISTENCY) {
        WaitFlag<HardEvent::V_MTE2>(INNERCORE_INTRAPARTIALO_V_MTE2);
        WaitFlag<HardEvent::V_MTE2>(INNERCORE_REDUCE_MAXSUM_V_MTE2);
    }
    WaitFlag<HardEvent::MTE3_V>(INNERCORE_STAGE2);
    if constexpr (!HIGH_PERF) {
        if (constInfo.isSoftmaxLseEnable) {
            WaitFlag<HardEvent::MTE3_V>(INNERCORE_LSE_MTE3_V);
        }
    }
    WaitFlag<HardEvent::V_MTE2>(INNERCORE_STAGE0_IN(0));
    WaitFlag<HardEvent::V_MTE2>(INNERCORE_STAGE0_IN(1));
    WaitFlag<HardEvent::MTE3_V>(INNERCORE_STAGE0_OUT(0));
    WaitFlag<HardEvent::MTE3_V>(INNERCORE_STAGE0_OUT(1));
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
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::InitFDBuffers(FdRunInfo &fdRunInfo)
{
    FdRunInfo mqsmlaFdBufferInfo = fdRunInfo;
    if (mqsmlaFdBufferInfo.mNum > AttentionCommon::FD_REDUCE_CHUNK_ROWS) {
        mqsmlaFdBufferInfo.mNum = AttentionCommon::FD_REDUCE_CHUNK_ROWS;
    }
    AttentionCommon::InitFDBuffersStatic<T, dTemplateAlign64>(mqsmlaFdBufferInfo, 0, fdBuffers);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::GetExtremeValue(T &negativeScalar)
{
    uint32_t mqsmlaTmp1 = NEGATIVE_MIN_VALUE_FP32;
    negativeScalar = *((float *)&mqsmlaTmp1);
}

TEMPLATES_DEF
class CSABlockVecDummy {
public:
    __aicore__ inline CSABlockVecDummy(){};
    __aicore__ inline void CleanOutput(__gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse,
                                       ConstInfo<HIGH_PERF> &constInfo)
    {}
    __aicore__ inline void InitGlobalBuffer(__gm__ uint8_t *oriKV, __gm__ uint8_t *cmpKV,
                                            __gm__ uint8_t *oriSparseIndices, __gm__ uint8_t *cmpSparseIndices,
                                            __gm__ uint8_t *oriBlockTable, __gm__ uint8_t *cmpBlockTable,
                                            __gm__ uint8_t *sequsedQ, __gm__ uint8_t *sinks,
                                            __gm__ uint8_t *sequsedOriKv, __gm__ uint8_t *sequsedCmpKv,
                                            __gm__ uint8_t *cmpResidualKv)
    {}
    __aicore__ inline void InitVecBlock(__gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *cuSeqlensOriKv,
                                        __gm__ uint8_t *cuSeqlensCmpKv, __gm__ uint8_t *sequsedOriKv,
                                        __gm__ uint8_t *sequsedCmpKv, __gm__ uint8_t *cmpResidualKv) {};
    __aicore__ inline void InitLocalBuffer(ConstInfo<HIGH_PERF> &constInfo, uint32_t ubBaseAddr) {}
    __aicore__ inline void InitFDBuffers(FdRunInfo &fdRunInfo) {}
    __aicore__ inline void ProcessVec1(StaticBuffer<Q_T> &outputBuf, StaticBuffer<T> &bmm1ResBuf,
                                       RunInfo<HIGH_PERF> &runInfo, ConstInfo<HIGH_PERF> &constInfo)
    {}

    __aicore__ inline void ProcessVec2(StaticBuffer<T> &bmm2ResBuf, RunInfo<HIGH_PERF> &runInfo,
                                       ConstInfo<HIGH_PERF> &constInfo)
    {}
    __aicore__ inline void FreeEvent(ConstInfo<HIGH_PERF> &constInfo) {}
    __aicore__ inline void InitS2SplitStaging(Buffer<BufferType::GM, SyncType::NO_SYNC> &fdStaging) {}
    __aicore__ inline void InitS2SplitStaging(Buffer<BufferType::GM, SyncType::NO_SYNC> &intraCoreCombine,
                                              Buffer<BufferType::GM, SyncType::NO_SYNC> &crossCoreCombine)
    {}
    __aicore__ inline void ProcessFlashDecode(FdRunInfo &fdRunInfo, ConstInfo<HIGH_PERF> &constInfo) {}
};
} // namespace BaseApi
#endif // MIXED_QUANT_SPARSE_FLASH_MLA_CSA_BLOCK_VECTOR_H
