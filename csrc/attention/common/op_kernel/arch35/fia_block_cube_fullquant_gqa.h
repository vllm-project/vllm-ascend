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
 * \file fia_block_cube_fullquant_gqa.h
 * \brief
 */
#ifndef FIA_BLOCK_CUBE_FULLQUANT_GQA_H_
#define FIA_BLOCK_CUBE_FULLQUANT_GQA_H_
#include "../offset_calculator.h"
#include "../matmul.h"
#include "../FixpipeOut.h"
#include "../memory_copy_arch35.h"

#include "infer_flash_attention_comm.h"
#include "flash_attention_score_common_regbase.h"
#include "kernel_operator_list_tensor_intf.h"
using namespace AscendC;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;
using namespace fa_base_matmul;
using namespace AttentionCommon;

namespace BaseApi {

template <typename INPUT_T, typename T, LayOutTypeEnum layout = LayOutTypeEnum::None,
          S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
          S2TemplateType s2TemplateType = S2TemplateType::Aligned128,
          DTemplateType dTemplateType = DTemplateType::Aligned128,
          DTemplateType dVTemplateType = DTemplateType::Aligned128, bool hasRope = false, uint8_t KvLayoutType = 0,
          bool enableKVPrefix = false, bool useDn = false, bool bmm2Write2Ub = true, bool splitD = false>
class FAFullQuantGqaBlockCube {
public:
    /* ============确定Key的L1类型============= */
    template <bool hasKRope>
    struct KVL1BuffSel {
        using Type = std::conditional_t<hasKRope, BuffersPolicyDB<BufferType::L1>, BuffersPolicy4buff<BufferType::L1>>;
    };

    /* ============确定L0A的类型============= */
    struct L0ABuffSel {
        using Type = BuffersPolicyDB<BufferType::L0A>;
    };
    /* ============确定L0B的类型============= */
    template <uint32_t s2BaseSize, uint32_t dBaseSize>
    struct L0BBuffSel {
        using Type = std::conditional_t<(s2BaseSize == 256 && dBaseSize > 128),
                                        BuffersPolicySingleBuffer<BufferType::L0B>, BuffersPolicyDB<BufferType::L0B>>;
    };

    /* ============确定L0C的类型============= */
    template <uint32_t mBaseSize, uint32_t s2BaseSize, uint32_t dVBaseSize>
    struct L0CBuffSel {
        using Type = std::conditional_t<(mBaseSize * s2BaseSize * FLOAT_BYTES <= (L0C_SIZE * KB_TO_BYTES) / NUM_4 &&
                                         mBaseSize * dVBaseSize * FLOAT_BYTES <= (L0C_SIZE * KB_TO_BYTES) / NUM_4),
                                        BuffersPolicy4buff<BufferType::L0C>, BuffersPolicyDB<BufferType::L0C>>;
    };

    static constexpr uint32_t mBaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr uint32_t dBaseSize = (uint32_t)dTemplateType;
    static constexpr uint32_t dVBaseSize = (uint32_t)dVTemplateType;
    static constexpr uint32_t l1BaseD = 128;
    static constexpr uint32_t s2SplitSize = 256U;
    static constexpr uint32_t MXFP_GROUP_SIZE = 32U;
    static constexpr uint32_t MXFP_DIVISOR_SIZE = 64U;
    static constexpr uint32_t MXFP_MULTI_BASE_SIZE = 2U;
    static constexpr LayOutTypeEnum LAYOUT = layout;
    static constexpr bool PAGE_ATTENTION = (KvLayoutType > 0);
    static constexpr bool HAS_ROPE = hasRope;
    static constexpr bool BMM2_TOUB = bmm2Write2Ub;
    static constexpr bool USE_DN = useDn;
    static constexpr bool SPLITD = splitD;
    static constexpr uint8_t KV_LAYOUT = 4; // 4: K与K_Scale在同一物理内存中交叉排列
    static constexpr FixpipeConfig BMM2_FIXPIPE_CONFIG = {CO2Layout::ROW_MAJOR, BMM2_TOUB};
    static constexpr GmFormat Q_FORMAT = GetQueryGmFormat<layout>();
    static constexpr GmFormat KV_FORMAT = GetKVGmFormat<layout, KV_LAYOUT, PAGE_ATTENTION>();
    static constexpr GmFormat Q_SCALE_FORMAT = GetQueryScaleGmFormat<layout, USE_DN>();
    static constexpr GmFormat K_SCALE_FORMAT = GetKeyScaleGmFormat<layout, KvLayoutType, PAGE_ATTENTION>();
    static constexpr GmFormat V_SCALE_FORMAT = GetValueScaleGmFormat<layout, KvLayoutType, PAGE_ATTENTION>();

    using ROPE_T = bfloat16_t;
    using Q_T = INPUT_T;
    using KV_T = INPUT_T;
    using SCALE_T = fp8_e8m0_t;
    using MM_T = T;
    using mm2ResPos = typename std::conditional<BMM2_TOUB, Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>,
                                                Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_FORWARD>>::type;

    using MM1_DBUF_T = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;
    using MM2_ABUF_POLICY_T = BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD>;
    using MM2_ABUF_T = Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD>;
    using MM2_DBUF_T = mm2ResPos;

    /* ============确定key scale的类型============= */
    template <uint8_t kvLayoutType>
    struct KeyScaleGmToL1Sel {
        using Type = std::conditional_t<(kvLayoutType == 3), // 3: PA_NZ
                                        CopyKeyScaleGmToL1<SCALE_T, K_SCALE_FORMAT, L1Format::NZ, ScaleTrans::NO_TRANS>,
                                        CopyKeyScaleGmToL1<SCALE_T, K_SCALE_FORMAT, L1Format::NZ, ScaleTrans::DN2NZ>>;
    };

    /* ============确定value scale的类型============= */
    template <uint8_t kvLayoutType>
    struct ValueScaleGmToL1Sel {
        using Type =
            std::conditional_t<(kvLayoutType == 3), // 3: PA_NZ
                               CopyValueScaleGmToL1<SCALE_T, V_SCALE_FORMAT, L1Format::NZ, ScaleTrans::NO_TRANS>,
                               CopyValueScaleGmToL1<SCALE_T, V_SCALE_FORMAT, L1Format::NZ, ScaleTrans::ND2NZ>>;
    };

    using L1KvType = typename KVL1BuffSel<HAS_ROPE>::Type;
    using L0AType = typename L0ABuffSel::Type;
    using L0BType = typename L0BBuffSel<s2BaseSize, dBaseSize>::Type;
    using L0CType = typename L0CBuffSel<mBaseSize, s2BaseSize, dVBaseSize>::Type;
    using KeyScaleGmToL1Type = typename KeyScaleGmToL1Sel<KvLayoutType>::Type;
    using ValueScaleGmToL1Type = typename ValueScaleGmToL1Sel<KvLayoutType>::Type;

    using ConstInfoX = ConstInfo_t<FiaKernelType::FULL_QUANT>;
    TPipe *tPipe = nullptr;
    /* =====================GM变量(with layout)==================== */
    FaGmTensor<Q_T, Q_FORMAT> queryGm;
    FaGmTensor<KV_T, KV_FORMAT> keyGm;
    FaGmTensor<KV_T, KV_FORMAT> valueGm;
    FaGmTensor<SCALE_T, Q_SCALE_FORMAT> queryScaleGm;
    FaGmTensor<SCALE_T, K_SCALE_FORMAT> keyScaleGm;
    FaGmTensor<SCALE_T, V_SCALE_FORMAT> valueScaleGm;
    FaGmTensor<ROPE_T, Q_FORMAT> queryRopeGm;
    FaGmTensor<ROPE_T, KV_FORMAT> keyRopeGm;
    GlobalTensor<int32_t> blockTableGm;
    GlobalTensor<uint64_t> actualSeqLengthsGmQ;
    GlobalTensor<uint64_t> actualSeqLengthsGmKv;

    CopyQueryGmToL1<Q_T, Q_FORMAT> copyQueryGmToL1;
    CopyKvGmToL1<KV_T, KV_FORMAT> copyKvGmToL1;
    CopyQueryScaleGmToL1<SCALE_T, Q_SCALE_FORMAT> copyQueryScaleGmToL1;
    KeyScaleGmToL1Type copyKeyScaleGmToL1;
    ValueScaleGmToL1Type copyValueScaleGmToL1;
    CopyQueryGmToL1<ROPE_T, Q_FORMAT> copyQueryRopeGmToL1;
    CopyKvGmToL1<ROPE_T, KV_FORMAT> copyKeyRopeGmToL1;

    /* =====================LocalBuffer变量====================*/
    BufferManager<BufferType::L1> *l1BufferManagerPtr;
    BufferManager<BufferType::L0A> l0aBufferManager;
    BufferManager<BufferType::L0B> l0bBufferManager;
    BufferManager<BufferType::L0C> l0cBufferManager;

    BuffersPolicyDB<BufferType::L1> l1QBuffers;
    L1KvType l1KVBuffers;
    BuffersPolicyDB<BufferType::L1> l1VBuffers;

    L0AType mmL0ABuffers;
    L0BType mmL0BBuffers;
    L0CType mmL0CBuffers;

    __gm__ uint8_t *keyPtr = nullptr;
    __gm__ uint8_t *valuePtr = nullptr;

    const ConstInfoX &constInfo;

    /*============================================================================== */
    __aicore__ inline FAFullQuantGqaBlockCube(ConstInfoX &constInfo) : constInfo(constInfo){};

    __aicore__ inline void InitCubeBlock(TPipe *pipe, BufferManager<BufferType::L1> *l1BuffMgr, __gm__ uint8_t *query,
                                         __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *blockTable,
                                         __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope,
                                         __gm__ uint8_t *actualSeqQlenAddr, __gm__ uint8_t *actualSeqKvlenAddr)
    {
        tPipe = pipe;
        l1BufferManagerPtr = l1BuffMgr;
        InitCubeInput(query, key, value, blockTable, queryRope, keyRope, actualSeqQlenAddr, actualSeqKvlenAddr);
    }

    __aicore__ inline void InitBuffers()
    {
        constexpr uint32_t mm1QSize = mBaseSize * dBaseSize * sizeof(INPUT_T);
        constexpr uint32_t mm1QScaleSize = 0;
        constexpr uint32_t mmKVSize = dBaseSize * s2BaseSize * sizeof(INPUT_T);
        constexpr uint32_t mmKVScaleSize = 0;

        if constexpr (HAS_ROPE) {
            constexpr uint32_t qRopeSize = 0; // 64: rope d is 64
            l1QBuffers.Init((*l1BufferManagerPtr), mm1QSize + mm1QScaleSize + qRopeSize);
            constexpr uint32_t kRopeSize = 0; // 64: rope d is 64
            l1KVBuffers.Init((*l1BufferManagerPtr), mmKVSize + mmKVScaleSize + kRopeSize);
            l1VBuffers.Init((*l1BufferManagerPtr), mmKVSize + mmKVScaleSize);
        } else {
            l1QBuffers.Init((*l1BufferManagerPtr), mm1QSize + mm1QScaleSize);
            l1KVBuffers.Init((*l1BufferManagerPtr), mmKVSize + mmKVScaleSize);
        }

        // L0A B C 当前写死，能否通过基础api获取
        l0aBufferManager.Init(tPipe, 65536);  // 64 * 1024
        l0bBufferManager.Init(tPipe, 65536);  // 64 * 1024
        l0cBufferManager.Init(tPipe, 262144); // 256 * 1024
        // L0A B C当前写死，要改成通过计算获取
        mmL0ABuffers.Init(l0aBufferManager, 32 * 1024);
        mmL0BBuffers.Init(l0bBufferManager, 32 * 1024);

        if constexpr (mBaseSize * s2BaseSize * FLOAT_BYTES <= (L0C_SIZE * KB_TO_BYTES) / NUM_4 &&
                      mBaseSize * dVBaseSize * FLOAT_BYTES <= (L0C_SIZE * KB_TO_BYTES) / NUM_4) {
            mmL0CBuffers.Init(l0cBufferManager, (L0C_SIZE / NUM_4) * KB_TO_BYTES);
        } else {
            mmL0CBuffers.Init(l0cBufferManager, (L0C_SIZE / NUM_2) * KB_TO_BYTES);
        }
    }

    __aicore__ inline void InitCubeInput(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                         __gm__ uint8_t *blockTable, __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope,
                                         __gm__ uint8_t *actualSeqQlenAddr, __gm__ uint8_t *actualSeqKvlenAddr)
    {
        if (constInfo.actualSeqLenSize != 0) {
            actualSeqLengthsGmQ.SetGlobalBuffer((__gm__ uint64_t *)actualSeqQlenAddr, constInfo.actualSeqLenSize);
        }
        if (constInfo.actualSeqLenKVSize != 0) {
            actualSeqLengthsGmKv.SetGlobalBuffer((__gm__ uint64_t *)actualSeqKvlenAddr, constInfo.actualSeqLenKVSize);
        }
        if constexpr (PAGE_ATTENTION) {
            blockTableGm.SetGlobalBuffer((__gm__ int32_t *)blockTable);
        }

        InitQBuffer(constInfo.bSize, constInfo.realN2Size, constInfo.realGSize, constInfo.s1Size, constInfo.dSize,
                    actualSeqLengthsGmQ, constInfo.actualSeqLenSize, queryGm, query);

        keyPtr = key;
        valuePtr = value;
        if (constInfo.isKvContinuous) {
            ListTensorDesc keyListTensorDesc((__gm__ void *)(this->keyPtr));
            __gm__ uint8_t *key_ = (__gm__ uint8_t *)keyListTensorDesc.GetDataPtr<__gm__ uint8_t>(0);
            ListTensorDesc valueListTensorDesc((__gm__ void *)(this->valuePtr));
            __gm__ uint8_t *value_ = (__gm__ uint8_t *)valueListTensorDesc.GetDataPtr<__gm__ uint8_t>(0);

            InitKVBuffer(constInfo.bSize, constInfo.s2Size, actualSeqLengthsGmKv, constInfo.actualSeqLenKVSize,
                         constInfo.n2Size, constInfo.blockSize, constInfo.dSize, keyGm, key_,
                         constInfo.keyStrides.bnStride, constInfo.keyStrides.n2Stride);
            InitKVBuffer(constInfo.bSize, constInfo.s2Size, actualSeqLengthsGmKv, constInfo.actualSeqLenKVSize,
                         constInfo.n2Size, constInfo.blockSize, constInfo.dSizeV, valueGm, value_,
                         constInfo.valueStrides.bnStride, constInfo.valueStrides.n2Stride);
        }

        if constexpr (HAS_ROPE) {
            InitQRopeBuffer(constInfo.bSize, constInfo.realN2Size, constInfo.realGSize, constInfo.s1Size,
                        constInfo.dSizeRope, actualSeqLengthsGmQ, constInfo.actualSeqLenSize, queryRopeGm, queryRope);
            InitKRopeBuffer(constInfo.bSize, constInfo.s2Size, actualSeqLengthsGmKv, constInfo.actualSeqLenKVSize,
                         constInfo.n2Size, constInfo.blockSize, constInfo.dSizeRope, keyRopeGm, keyRope);
        }
    }

    __aicore__ inline void InitQBuffer(uint32_t batchSize, uint32_t n2Size, uint32_t gSize, uint32_t qSeqSize,
                                       uint32_t headDim, GlobalTensor<uint64_t> actualSeqLenGmQ,
                                       uint32_t actualLenQDims, FaGmTensor<Q_T, Q_FORMAT> &qGmTensor,
                                       __gm__ uint8_t *gm)
    {
        qGmTensor.gmTensor.SetGlobalBuffer((__gm__ Q_T *)gm);
        if constexpr (GmLayoutParams<Q_FORMAT>::CATEGORY == FormatCategory::GM_Q_OUT_BNGSD) {
            qGmTensor.offsetCalculator.Init(batchSize, n2Size, gSize, qSeqSize, headDim, actualSeqLenGmQ,
                                            actualLenQDims, 0, 0);
        } else if constexpr (GmLayoutParams<Q_FORMAT>::CATEGORY == FormatCategory::GM_Q_OUT_TND) {
            qGmTensor.offsetCalculator.Init(n2Size, gSize, headDim, actualSeqLenGmQ, actualLenQDims);
        }
    }

    __aicore__ inline void InitQRopeBuffer(uint32_t batchSize, uint32_t n2Size, uint32_t gSize, uint32_t qSeqSize,
                                           uint32_t headDim, GlobalTensor<uint64_t> actualSeqLenGmQ,
                                           uint32_t actualLenQDims, FaGmTensor<ROPE_T, Q_FORMAT> &qRopeGmTensor,
                                           __gm__ uint8_t *gm)
    {
        qRopeGmTensor.gmTensor.SetGlobalBuffer((__gm__ ROPE_T *)gm);
        if constexpr (GmLayoutParams<Q_FORMAT>::CATEGORY == FormatCategory::GM_Q_OUT_BNGSD) {
            qRopeGmTensor.offsetCalculator.Init(batchSize, n2Size, gSize, qSeqSize, headDim, actualSeqLenGmQ,
                                                actualLenQDims, 0, 0);
        } else if constexpr (GmLayoutParams<Q_FORMAT>::CATEGORY == FormatCategory::GM_Q_OUT_TND) {
            qRopeGmTensor.offsetCalculator.Init(n2Size, gSize, headDim, actualSeqLenGmQ, actualLenQDims);
        }
    }

    __aicore__ inline void InitKVBuffer(uint32_t batchSize, uint32_t kvSeqSize, GlobalTensor<uint64_t> actualSeqLenGmKv,
                                        uint32_t actualLenDims, uint32_t n2Size, uint32_t kvCacheBlockSize,
                                        uint32_t headDim, FaGmTensor<KV_T, KV_FORMAT> &kvGmTensor, __gm__ uint8_t *gm,
                                        uint64_t bnStrides, uint64_t n2Strides)
    {
        kvGmTensor.gmTensor.SetGlobalBuffer((__gm__ KV_T *)gm);
        if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_PA_BNBD) {
            kvGmTensor.offsetCalculator.Init(n2Size, kvCacheBlockSize, headDim, blockTableGm,
                                             constInfo.maxBlockNumPerBatch);
        } else if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_PA_NZ) {
            uint32_t d0 = 32 / sizeof(KV_T);
            uint32_t d1 = headDim / d0;
            kvGmTensor.offsetCalculator.Init(n2Size, kvCacheBlockSize, d1, d0, blockTableGm,
                                             constInfo.maxBlockNumPerBatch);
        } else if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_BNSD) {
            kvGmTensor.offsetCalculator.Init(batchSize, n2Size, kvSeqSize, headDim, actualSeqLenGmKv, actualLenDims,
                                             false, 0);
        } else if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_TND) {
            kvGmTensor.offsetCalculator.Init(n2Size, headDim, actualSeqLenGmKv, actualLenDims);
        } else if (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_BnNBsD_KS) {
            kvGmTensor.offsetCalculator.Init(n2Size, kvCacheBlockSize, headDim, blockTableGm,
                                             constInfo.maxBlockNumPerBatch, bnStrides, n2Strides);
        }
    }

    __aicore__ inline void InitKRopeBuffer(uint32_t batchSize, uint32_t kvSeqSize,
                                           GlobalTensor<uint64_t> actualSeqLenGmKv, uint32_t actualLenDims,
                                           uint32_t n2Size, uint32_t kvCacheBlockSize, uint32_t headDim,
                                           FaGmTensor<ROPE_T, KV_FORMAT> &kRopeGmTensor, __gm__ uint8_t *gm)
    {
        kRopeGmTensor.gmTensor.SetGlobalBuffer((__gm__ ROPE_T *)gm);
        if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_PA_BNBD) {
            kRopeGmTensor.offsetCalculator.Init(n2Size, kvCacheBlockSize, headDim, blockTableGm,
                                                constInfo.maxBlockNumPerBatch);
        } else if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_PA_NZ) {
            uint32_t d0 = 32 / sizeof(ROPE_T);
            uint32_t d1 = headDim / d0;
            kRopeGmTensor.offsetCalculator.Init(n2Size, kvCacheBlockSize, d1, d0, blockTableGm,
                                                constInfo.maxBlockNumPerBatch);
        } else if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_BNSD) {
            kRopeGmTensor.offsetCalculator.Init(batchSize, n2Size, kvSeqSize, headDim, actualSeqLenGmKv, actualLenDims,
                                                false, 0);
        } else if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_TND) {
            kRopeGmTensor.offsetCalculator.Init(n2Size, headDim, actualSeqLenGmKv, actualLenDims);
        }
    }

    __aicore__ inline void AllocEventID()
    {
        // InitBuffers阶段已完成eventId申请和SetFlag，这里为空实现
    }

    __aicore__ inline void FreeEventID()
    {
        l1QBuffers.Uninit((*l1BufferManagerPtr));
        l1KVBuffers.Uninit((*l1BufferManagerPtr));
        mmL0ABuffers.Uninit(l0aBufferManager);
        mmL0BBuffers.Uninit(l0bBufferManager);
        mmL0CBuffers.Uninit(l0cBufferManager);
    }

    // copy query with full s1g
    __aicore__ inline void CopyQuerySlice(const LocalTensor<Q_T> &dstTensor, uint32_t dOffset, uint32_t dRealSize,
                                          RunInfoX &runInfo)
    {
        constexpr uint32_t blockNumDtype = 32 / sizeof(Q_T);
        uint32_t nopeDealSize = dRealSize;
        uint32_t dstStride = (runInfo.actMSize + 31) >> 5 << 5;
        FaL1Tensor<Q_T, L1Format::NZ> l1Tensor {
            .tensor = dstTensor,
            .rowCount = dstStride
        };

        GmCoord gmCoord {
            .bIdx = runInfo.bIdx,
            .n2Idx = runInfo.realN2Idx,
            .gS1Idx = runInfo.gS1Idx,
            .dIdx = dOffset,
            .gS1DealSize = runInfo.actMSize,
            .dDealSize = nopeDealSize
        };
        copyQueryGmToL1(l1Tensor, queryGm, gmCoord);

        if constexpr (HAS_ROPE) {
            uint32_t ropeDealSize = constInfo.dSizeRope;
            uint32_t dstStrideRope = (runInfo.actMSize + 15) >> 4 << 4;
            uint32_t offset = AttentionCommon::Align(nopeDealSize, blockNumDtype) *
                              AttentionCommon::Align(runInfo.actMSize, blockNumDtype);
            FaL1Tensor<ROPE_T, L1Format::NZ> l1Tensor {
                .tensor = (dstTensor[offset]).template ReinterpretCast<ROPE_T>(),
                .rowCount = dstStrideRope
            };

            GmCoord gmCoord = {
                .bIdx = runInfo.bIdx,
                .n2Idx = runInfo.realN2Idx,
                .gS1Idx = runInfo.gS1Idx,
                .dIdx = 0,
                .gS1DealSize = runInfo.actMSize,
                .dDealSize = ropeDealSize
            };
            copyQueryRopeGmToL1(l1Tensor, queryRopeGm, gmCoord);
        }
    }

    __aicore__ inline void CopyQueryTile(const LocalTensor<Q_T> &dstTensor, RunInfoX &runInfo)
    {
        CopyQuerySlice(dstTensor, 0, constInfo.dSize, runInfo);
    }

    // copy query scale with full s1g
    __aicore__ inline void CopyQueryScaleSlice(const LocalTensor<SCALE_T> &dstTensor, uint32_t dOffset,
                                               uint32_t dRealSize, RunInfoX &runInfo)
    {
        uint32_t dstStride = (runInfo.actMSize + 31) >> 5 << 5;
        FaL1Tensor<SCALE_T, L1Format::NZ> l1Tensor {
            .tensor = dstTensor,
            .rowCount = dstStride
        };

        GmCoord gmCoord {
            .bIdx = runInfo.bIdx,
            .n2Idx = runInfo.realN2Idx,
            .gS1Idx = runInfo.gS1Idx,
            .dIdx = dOffset,
            .gS1DealSize = runInfo.actMSize,
            .dDealSize = dRealSize
        };
        copyQueryScaleGmToL1(l1Tensor, queryScaleGm, gmCoord);
    }

    __aicore__ inline void CopyQueryScaleTile(const LocalTensor<SCALE_T> &dstTensor, RunInfoX &runInfo)
    {
        CopyQueryScaleSlice(dstTensor, 0, constInfo.dSize / MXFP_GROUP_SIZE, runInfo);
    }

    // copy key with full s2
    __aicore__ inline void CopyKeySlice(const LocalTensor<KV_T> &dstTensor, uint32_t s2Offset, uint32_t s2RealSize,
                                        uint32_t dOffset, uint32_t dRealSize, RunInfoX &runInfo)
    {
        constexpr uint32_t blockNumDtype = 32 / sizeof(KV_T);
        uint32_t dstStride = (s2RealSize + 31) >> 5 << 5;

        uint32_t nopeDealSize = dRealSize;
        FaL1Tensor<KV_T, L1Format::NZ> l1Tensor {
            .tensor = dstTensor,
            .rowCount = dstStride
        };

        GmKvCoord gmCoord {
            .bIdx = constInfo.isKvContinuous ? runInfo.bIdx : 0,
            .n2Idx = runInfo.n2Idx,
            .s2Idx = s2Offset,
            .dIdx = dOffset,
            .s2DealSize = s2RealSize,
            .dDealSize = nopeDealSize
        };

        copyKvGmToL1(l1Tensor, keyGm, gmCoord);

        if constexpr (HAS_ROPE) {
            uint32_t ropeDealSize = constInfo.dSizeRope;
            uint32_t dstStrideRope = (s2RealSize + 15) >> 4 << 4;
            uint32_t offset = AttentionCommon::Align(nopeDealSize, blockNumDtype) *
                              AttentionCommon::Align(s2RealSize, blockNumDtype);
            FaL1Tensor<ROPE_T, L1Format::NZ> l1Tensor =  {
                .tensor = (dstTensor[offset]).template ReinterpretCast<ROPE_T>(),
                .rowCount = dstStrideRope
            };

            GmKvCoord gmCoord = {
                .bIdx = constInfo.isKvContinuous ? runInfo.bIdx : 0,
                .n2Idx = runInfo.n2Idx,
                .s2Idx = s2Offset,
                .dIdx = 0,
                .s2DealSize = s2RealSize,
                .dDealSize = ropeDealSize
            };
            copyKeyRopeGmToL1(l1Tensor, keyRopeGm, gmCoord);
        }
    }

    // 全量拷贝
    __aicore__ inline void CopyKeyTile(const LocalTensor<KV_T> &dstTensor, RunInfoX &runInfo, uint32_t s2RealSize)
    {
        uint32_t s2Offset = runInfo.s2Idx;
        CopyKeySlice(dstTensor, s2Offset, s2RealSize, 0, constInfo.dSize, runInfo);
    }

    // copy key scale with full s2
    __aicore__ inline void CopyKeyScaleSlice(const LocalTensor<SCALE_T> &dstTensor, uint32_t s2Offset,
                                             uint32_t s2RealSize, uint32_t dOffset, uint32_t dRealSize,
                                             RunInfoX &runInfo)
    {
        uint32_t dstStride = (s2RealSize + 31) >> 5 << 5;
        FaL1Tensor<SCALE_T, L1Format::NZ> l1Tensor {
            .tensor = dstTensor,
            .rowCount = dstStride
        };

        GmKvCoord gmCoord {
            .bIdx = runInfo.bIdx,
            .n2Idx = runInfo.n2Idx,
            .s2Idx = s2Offset,
            .dIdx = dOffset,
            .s2DealSize = s2RealSize,
            .dDealSize = dRealSize
        };
        copyKeyScaleGmToL1(l1Tensor, keyScaleGm, gmCoord);
    }

    // 全量拷贝
    __aicore__ inline void CopyKeyScaleTile(const LocalTensor<SCALE_T> &dstTensor, RunInfoX &runInfo,
                                            uint32_t s2RealSize)
    {
        uint32_t s2Offset = runInfo.s2Idx;
        CopyKeyScaleSlice(dstTensor, s2Offset, s2RealSize, 0, constInfo.dSize / MXFP_GROUP_SIZE, runInfo);
    }

    // copy key with full s2
    __aicore__ inline void CopyValueSlice(const LocalTensor<KV_T> &dstTensor, uint32_t s2Offset, uint32_t s2RealSize,
                                          uint32_t dOffset, uint32_t dRealSize, RunInfoX &runInfo)
    {
        uint32_t dstStride = (s2RealSize + 31) >> 5 << 5;
        FaL1Tensor<KV_T, L1Format::NZ> l1Tensor {
            .tensor = dstTensor,
            .rowCount = dstStride
        };

        GmKvCoord gmCoord {
            .bIdx = constInfo.isKvContinuous ? runInfo.bIdx : 0,
            .n2Idx = runInfo.n2Idx,
            .s2Idx = s2Offset,
            .dIdx = dOffset,
            .s2DealSize = s2RealSize,
            .dDealSize = dRealSize
        };
        copyKvGmToL1(l1Tensor, valueGm, gmCoord);
    }

    __aicore__ inline void CopyValueTile(const LocalTensor<KV_T> &dstTensor, RunInfoX &runInfo)
    {
        CopyValueSlice(dstTensor, runInfo.s2Idx, runInfo.actSingleLoopS2Size, 0, constInfo.dSizeV, runInfo);
    }

    // copy key with full s2
    __aicore__ inline void CopyValueScaleSlice(const LocalTensor<SCALE_T> &dstTensor, uint32_t s2Offset,
                                               uint32_t s2RealSize, uint32_t dOffset, uint32_t dRealSize,
                                               RunInfoX &runInfo)
    {
        FaL1Tensor<SCALE_T, L1Format::NZ> l1Tensor {
            .tensor = dstTensor,
            .rowCount = s2RealSize
        };

        GmKvCoord gmCoord {
            .bIdx = runInfo.bIdx,
            .n2Idx = runInfo.n2Idx,
            .s2Idx = s2Offset,
            .dIdx = dOffset,
            .s2DealSize = s2RealSize,
            .dDealSize = dRealSize
        };
        copyValueScaleGmToL1(l1Tensor, valueScaleGm, gmCoord);
    }

    __aicore__ inline void CopyValueScaleTile(const LocalTensor<SCALE_T> &dstTensor, RunInfoX &runInfo)
    {
        CopyValueScaleSlice(dstTensor, runInfo.s2Idx, runInfo.actSingleLoopS2Size / MXFP_DIVISOR_SIZE, 0,
                            constInfo.dSizeV * MXFP_MULTI_BASE_SIZE, runInfo);
    }

    __aicore__ inline void UpdateKey(uint32_t bIdx)
    {
        ListTensorDesc keyListTensorDesc((__gm__ void *)(this->keyPtr));
        __gm__ uint8_t *key_ = (__gm__ uint8_t *)keyListTensorDesc.GetDataPtr<__gm__ uint8_t>(bIdx);

        uint64_t s2Size = SeqLenFromTensorList<LAYOUT>(this->keyPtr, bIdx);
        keyGm.gmTensor.SetGlobalBuffer((__gm__ KV_T *)key_);
        keyGm.offsetCalculator.Init(0, constInfo.n2Size, s2Size, constInfo.dSize, actualSeqLengthsGmKv,
                                    constInfo.actualSeqLenKVSize);
    }

    __aicore__ inline void UpdateValue(uint32_t bIdx)
    {
        ListTensorDesc valueListTensorDesc((__gm__ void *)(this->valuePtr));
        __gm__ uint8_t *value_ = (__gm__ uint8_t *)valueListTensorDesc.GetDataPtr<__gm__ uint8_t>(bIdx);
        uint64_t s2Size = SeqLenFromTensorList<LAYOUT>(valuePtr, bIdx);
        valueGm.gmTensor.SetGlobalBuffer((__gm__ KV_T *)value_);
        valueGm.offsetCalculator.Init(0, constInfo.n2Size, s2Size, constInfo.dSize, actualSeqLengthsGmKv,
                                      constInfo.actualSeqLenKVSize);
    }

    __aicore__ inline void IterateBmm1(MM1_DBUF_T &outputBuf, RunInfoX &runInfo)
    {
        if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_BNSD) {
            if (runInfo.isChangeBatch) {
                UpdateKey(runInfo.bIdx);
            }
        }

        if constexpr (USE_DN) {
            IterateBmm1Dn(outputBuf, runInfo);
        } else {
            IterateBmm1Nd(outputBuf, runInfo);
        }
    }

    /* 针对S1Base=128, S2Base = 256, D = 128场景, L1全载/L0全载, 左矩阵驻留. GS1<=80, S=S1*S2 */
    __aicore__ inline void IterateBmm1Nd(MM1_DBUF_T &outputBuf, RunInfoX &runInfo)
    {
        uint32_t s2CalcSize = runInfo.actSingleLoopS2Size;
        Buffer<BufferType::L1> mm1A;
        uint32_t qScaleOffset = ((runInfo.actMSize + 31) >> 5 << 5) * constInfo.dSize; // QScale在mm1A的偏移量（单位：元素）
        if constexpr (HAS_ROPE) {
            qScaleOffset += ((runInfo.actMSize + 31) >> 5 << 5) * constInfo.dSizeRope * sizeof(ROPE_T);
        }
        if (unlikely(runInfo.isFirstS2Loop)) {
            mm1A = l1QBuffers.Get();
            mm1A.Wait<HardEvent::MTE1_MTE2>();
            LocalTensor<Q_T> mm1ATensor = mm1A.GetTensor<Q_T>();
            CopyQueryTile(mm1ATensor, runInfo);

            LocalTensor<SCALE_T> mm1AScaleTensor = mm1A.GetTensor<SCALE_T>(qScaleOffset);
            CopyQueryScaleTile(mm1AScaleTensor, runInfo);
            mm1A.Set<HardEvent::MTE2_MTE1>();
        } else {
            mm1A = l1QBuffers.GetPre();
            mm1A.Set<HardEvent::MTE2_MTE1>();
        }

        Buffer<BufferType::L1> mm1B = l1KVBuffers.Get();
        mm1B.Wait<HardEvent::MTE1_MTE2>();
        LocalTensor<KV_T> mm1BTensor = mm1B.GetTensor<KV_T>();
        CopyKeyTile(mm1BTensor, runInfo, s2CalcSize);

        uint32_t kScaleOffset = ((s2CalcSize + 31) >> 5 << 5) * constInfo.dSize; // KScale在mm1B的偏移量（单位：元素）
        if constexpr (HAS_ROPE) {
            kScaleOffset += ((s2CalcSize + 31) >> 5 << 5) * constInfo.dSizeRope * sizeof(ROPE_T);
        }
        LocalTensor<SCALE_T> mm1BScaleTensor = mm1B.GetTensor<SCALE_T>(kScaleOffset);
        CopyKeyScaleTile(mm1BScaleTensor, runInfo, s2CalcSize);
        mm1B.Set<HardEvent::MTE2_MTE1>();
        mm1A.Wait<HardEvent::MTE2_MTE1>();
        mm1B.Wait<HardEvent::MTE2_MTE1>();

        Buffer<BufferType::L0C> mm1ResL0C = mmL0CBuffers.Get();
        mm1ResL0C.Wait<HardEvent::FIX_M>();
        MMParam param = MakeMMParam((uint32_t)runInfo.actMSize, (uint32_t)s2CalcSize,
                                    (uint32_t)constInfo.dSize, false, true);
        MatmulFull<Q_T, KV_T, T, 128, 256, dBaseSize, ABLayout::MK, ABLayout::KN, L0AType, L0BType,
                   SCALE_T, SCALE_T, mx_fp8_e4m3_t, mx_fp8_e4m3_t>(
            mm1A.GetTensor<INPUT_T>(), mm1B.GetTensor<INPUT_T>(),
            mmL0ABuffers, mmL0BBuffers,
            mm1ResL0C.GetTensor<T>(),
            param,
            mm1A.GetTensor<SCALE_T>(qScaleOffset), mm1B.GetTensor<SCALE_T>(kScaleOffset));

        if constexpr (HAS_ROPE) {
            uint32_t qRopeOffset = ((runInfo.actMSize + 31) >> 5 << 5) * constInfo.dSize / sizeof(ROPE_T);
            uint32_t kRopeOffset = ((s2CalcSize + 31) >> 5 << 5) * constInfo.dSize / sizeof(ROPE_T);
            MMParam ropeParam = MakeMMParam((uint32_t)runInfo.actMSize, (uint32_t)s2CalcSize,
                                            (uint32_t)constInfo.dSizeRope, false, true, true, false);
            MatmulFull<ROPE_T, ROPE_T, T, 128, 256, 64, ABLayout::MK, ABLayout::KN>(
                mm1A.GetTensor<ROPE_T>(qRopeOffset), mm1B.GetTensor<ROPE_T>(kRopeOffset),
                mmL0ABuffers, mmL0BBuffers,
                mm1ResL0C.GetTensor<T>(),
                ropeParam);
        }
        if (unlikely(runInfo.isLastS2Loop)) {
            mm1A.Set<HardEvent::MTE1_MTE2>();
        }
        mm1B.Set<HardEvent::MTE1_MTE2>();   // 释放L1B
        mm1ResL0C.Set<HardEvent::M_FIX>();  // 通知
        mm1ResL0C.Wait<HardEvent::M_FIX>(); // 等待L0C

        outputBuf.WaitCrossCore();

        FixpipeMm1(outputBuf.template GetTensor<T>(), mm1ResL0C.GetTensor<T>(), runInfo, s2CalcSize);

        mm1ResL0C.Set<HardEvent::FIX_M>();
        outputBuf.SetCrossCore();
    }

    __aicore__ inline void FixpipeMm1(const LocalTensor<T> &dstTensor, const LocalTensor<T> &l0C, RunInfoX &runInfo,
                                      uint32_t s2RealSize)
    {
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        // L0C上的bmm1结果矩阵N方向的size大小, 使能NZ2ND, nSize*sizeof(T) 必须是32B的倍数
        fixpipeParams.nSize = (s2RealSize + 7) >> 3 << 3;
        // 有效数据不足16行，只需输出部分行即可;L0C上的bmm1结果矩阵M方向的size大小必须是偶数
        fixpipeParams.mSize = (runInfo.actMSize + 1) >> 1 << 1;
        // L0C上matmul结果相邻连续数据片断间隔（前面一个数据块的头与后面数据块的头的间隔），单位为16 *sizeof(T)
        // 源NZ矩阵中相邻Z排布的起始地址偏移
        fixpipeParams.srcStride = ((fixpipeParams.mSize + 15) / 16) * 16;
        fixpipeParams.dstStride = s2SplitSize; // mmResUb上两行之间的间隔，单位：element
        fixpipeParams.dualDstCtl = 1;          // 双目标模式，按M维度拆分， M / 2 * N写入每个UB，M必须为2的倍数
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;

        Fixpipe<T, T, PFA_CFG_ROW_MAJOR_UB>(dstTensor, l0C, fixpipeParams);
    }

    __aicore__ inline void FixpipeMm1Dn(const LocalTensor<T> &dstTensor, const LocalTensor<T> &l0C, RunInfoX &runInfo,
                                        uint32_t s2RealSize)
    {
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        // L0C上的bmm1结果矩阵N方向的size大小, 使能NZ2ND, nSize*sizeof(T) 必须是32B的倍数
        fixpipeParams.nSize = (runInfo.actMSize + 31) >> 5 << 5;
        // 有效数据不足16行，只需输出部分行即可;L0C上的bmm1结果矩阵M方向的size大小必须是偶数
        fixpipeParams.mSize = (s2RealSize + 7) >> 3 << 3;
        // L0C上matmul结果相邻连续数据片断间隔（前面一个数据块的头与后面数据块的头的间隔），单位为16 *sizeof(T)
        // 源NZ矩阵中相邻Z排布的起始地址偏移
        fixpipeParams.srcStride = ((fixpipeParams.mSize + 15) / 16) * 16;
        fixpipeParams.dstStride = 64;  // mmResUb上两行之间的间隔，单位：element
        fixpipeParams.dualDstCtl = 2;  // 双目标模式，按M维度拆分， M * N / 2写入每个UB，M必须为2的倍数
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;

        Fixpipe<T, T, PFA_CFG_ROW_MAJOR_UB>(dstTensor, l0C, fixpipeParams);
    }

    /* 针对S1Base=128, S2Base = 256, D = 128场景, L1全载/L0全载, K * Q^T, Q矩阵驻留. GS1>80, S=S1*S2 */
    __aicore__ inline void IterateBmm1Dn(MM1_DBUF_T &outputBuf, RunInfoX &runInfo)
    {
        uint32_t s2CalcSize = runInfo.actSingleLoopS2Size;
        Buffer<BufferType::L1> mm1B;
        if (unlikely(runInfo.isFirstS2Loop)) {
            mm1B = l1QBuffers.Get();
            mm1B.Wait<HardEvent::MTE1_MTE2>();
            LocalTensor<Q_T> mm1BTensor = mm1B.GetTensor<Q_T>();
            CopyQueryTile(mm1BTensor, runInfo);
            mm1B.Set<HardEvent::MTE2_MTE1>();
        } else {
            mm1B = l1QBuffers.GetPre();
            mm1B.Set<HardEvent::MTE2_MTE1>();
        }

        Buffer<BufferType::L1> mm1A = l1KVBuffers.Get();
        mm1A.Wait<HardEvent::MTE1_MTE2>();
        LocalTensor<KV_T> mm1ATensor = mm1A.GetTensor<KV_T>();
        CopyKeyTile(mm1ATensor, runInfo, s2CalcSize);

        mm1A.Set<HardEvent::MTE2_MTE1>();
        mm1A.Wait<HardEvent::MTE2_MTE1>();
        mm1B.Wait<HardEvent::MTE2_MTE1>();

        Buffer<BufferType::L0C> mm1ResL0C = mmL0CBuffers.Get();
        mm1ResL0C.Wait<HardEvent::FIX_M>();
        MMParam param = MakeMMParam((uint32_t)s2CalcSize, (uint32_t)runInfo.actMSize,
                                    (uint32_t)constInfo.dSize, false, true);
        MatmulFull<Q_T, KV_T, T, 256, 128, dBaseSize, ABLayout::MK, ABLayout::KN, L0AType, L0BType>(
            mm1A.GetTensor<INPUT_T>(), mm1B.GetTensor<INPUT_T>(),
            mmL0ABuffers, mmL0BBuffers,
            mm1ResL0C.GetTensor<T>(),
            param);

        if constexpr (HAS_ROPE) {
            uint32_t qRopeOffset = ((runInfo.actMSize + 31) >> 5 << 5) * constInfo.dSize / sizeof(ROPE_T);
            uint32_t kRopeOffset = ((s2CalcSize + 31) >> 5 << 5) * constInfo.dSize / sizeof(ROPE_T);
            MMParam ropeParam = MakeMMParam((uint32_t)s2CalcSize, (uint32_t)runInfo.actMSize,
                                            (uint32_t)constInfo.dSizeRope, false, true, true, false);
            MatmulFull<ROPE_T, ROPE_T, T, 128, 256, 64, ABLayout::MK, ABLayout::KN>(
                mm1A.GetTensor<ROPE_T>(kRopeOffset), mm1B.GetTensor<ROPE_T>(qRopeOffset),
                mmL0ABuffers, mmL0BBuffers,
                mm1ResL0C.GetTensor<T>(),
                ropeParam);
        }
        if (unlikely(runInfo.isLastS2Loop)) {
            mm1B.Set<HardEvent::MTE1_MTE2>();
        }
        mm1A.Set<HardEvent::MTE1_MTE2>();   // 释放L1B
        mm1ResL0C.Set<HardEvent::M_FIX>();  // 通知
        mm1ResL0C.Wait<HardEvent::M_FIX>(); // 等待L0C

        outputBuf.WaitCrossCore();

        FixpipeMm1Dn(outputBuf.template GetTensor<T>(), mm1ResL0C.GetTensor<T>(), runInfo, s2CalcSize);

        mm1ResL0C.Set<HardEvent::FIX_M>();
        outputBuf.SetCrossCore();
    }

    __aicore__ inline void IterateBmm2(mm2ResPos &outputBuf, MM2_ABUF_POLICY_T &inputBuf, RunInfoX &runInfo)
    {
        if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_BNSD) {
            if (runInfo.isChangeBatch) {
                UpdateValue(runInfo.bIdx);
            }
        }

        if constexpr (USE_DN) {
            IterateBmm2Dn(outputBuf, inputBuf, runInfo);
        }
    }

    template <typename DST_TENSOR_T>
    __aicore__ inline void FixpipeMm2(const DST_TENSOR_T &dstTensor, const LocalTensor<T> &l0C, RunInfoX &runInfo)
    {
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams; // L0C→UB;FixpipeParamsM300:L0C→UB
        if constexpr (BMM2_TOUB) {
            fixpipeParams.nSize = (constInfo.dSizeV + 7) >> 3 << 3;
        } else {
            fixpipeParams.nSize = constInfo.dSizeV;
        }

        if constexpr (!USE_DN) {
            fixpipeParams.mSize = (runInfo.actMSize + 1) >> 1 << 1;
            fixpipeParams.srcStride = ((fixpipeParams.mSize + 15) / 16) * 16;
        } else {
            fixpipeParams.mSize = mBaseSize;
            fixpipeParams.srcStride = ((mBaseSize + 15) / 16) * 16;
        }

        if constexpr (BMM2_TOUB || SPLITD) {
            fixpipeParams.dstStride = ((uint32_t)dVTemplateType + 15) >> 4 << 4;
        } else {
            fixpipeParams.dstStride = (uint32_t)constInfo.dSizeV;
        }
        fixpipeParams.dualDstCtl = 1;
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;
        Fixpipe<T, T, BMM2_FIXPIPE_CONFIG>(dstTensor, l0C, fixpipeParams);
    }

    __aicore__ inline void IterateBmm2Dn(mm2ResPos &outputBuf, MM2_ABUF_POLICY_T &inputBuf, RunInfoX &runInfo)
    {
        MM2_ABUF_T mm2A = inputBuf.Get();
        mm2A.WaitCrossCore();

        Buffer<BufferType::L0C> mm2ResL0C = mmL0CBuffers.Get();
        mm2ResL0C.Wait<HardEvent::FIX_M>();

        uint32_t realK = runInfo.actSingleLoopS2Size;
        Buffer<BufferType::L1> mm2B;
        mm2B = l1KVBuffers.Get();
        mm2B.Wait<HardEvent::MTE1_MTE2>();
        LocalTensor<KV_T> mm2BTensor = mm2B.GetTensor<KV_T>();
        CopyValueSlice(mm2BTensor, runInfo.s2Idx, realK, 0, constInfo.dSizeV, runInfo);
        mm2B.Set<HardEvent::MTE2_MTE1>();
        mm2B.Wait<HardEvent::MTE2_MTE1>();
        MMParam param = MakeMMParam((uint32_t)mBaseSize, (uint32_t)constInfo.dSizeV,
                                    realK, USE_DN, false, true, true);
        MatmulFull<INPUT_T, KV_T, T, 128, 128, 256,
                   ABLayout::MK, ABLayout::KN, L0AType, L0BType>(
            mm2A.GetTensor<INPUT_T>(),
            mm2BTensor, mmL0ABuffers, mmL0BBuffers,
            mm2ResL0C.GetTensor<T>(), param);
        mm2B.Set<HardEvent::MTE1_MTE2>();
        mm2ResL0C.Set<HardEvent::M_FIX>();
        mm2ResL0C.Wait<HardEvent::M_FIX>();

        if constexpr (BMM2_TOUB) {
            outputBuf.WaitCrossCore();
        }
        FixpipeMm2(outputBuf.template GetTensor<T>(), mm2ResL0C.GetTensor<T>(), runInfo);
        mm2ResL0C.Set<HardEvent::FIX_M>();

        outputBuf.SetCrossCore();
    }
}; // FAFullQuantGqaBlockCube

template <typename INPUT_T, typename T, LayOutTypeEnum layout = LayOutTypeEnum::None,
          S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
          S2TemplateType s2TemplateType = S2TemplateType::Aligned128,
          DTemplateType dTemplateType = DTemplateType::Aligned128,
          DTemplateType dVTemplateType = DTemplateType::Aligned128, bool hasRope = false, uint8_t KvLayoutType = 0,
          bool enableKVPrefix = false, bool useDn = false, bool bmm2Write2Ub = true, bool splitD = false>
class FAFullQuantGqaBlockCubeDummy {
public:
    static constexpr uint32_t mBaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr uint32_t dBaseSize = (uint32_t)dTemplateType;
    static constexpr uint32_t dVBaseSize = (uint32_t)dVTemplateType;
    static constexpr LayOutTypeEnum LAYOUT = layout;
    static constexpr bool PAGE_ATTENTION = (KvLayoutType > 0);
    static constexpr bool HAS_ROPE = hasRope;
    static constexpr bool HAS_PREFIX = enableKVPrefix;
    static constexpr bool BMM2_TOUB = bmm2Write2Ub;
    static constexpr bool USE_DN = useDn;

    using ROPE_T = bfloat16_t;
    using Q_T = INPUT_T;
    using KV_T = INPUT_T;
    using MM_T = T;
    using mm2ResPos = typename std::conditional<bmm2Write2Ub, Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>,
                                                Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_FORWARD>>::type;

    using MM1_DBUF_T = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;
    using MM2_ABUF_POLICY_T = BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD>;
    using MM2_ABUF_T = Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD>;

    using ConstInfoX = ConstInfo_t<FiaKernelType::FULL_QUANT>;
    __aicore__ inline FAFullQuantGqaBlockCubeDummy(ConstInfoX &constInfo) {};
};
} // namespace BaseApi

#endif // FIA_BLOCK_CUBE_FULLQUANT_GQA_H_