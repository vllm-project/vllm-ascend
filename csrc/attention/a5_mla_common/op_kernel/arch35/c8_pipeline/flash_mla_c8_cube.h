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
 * \file flash_mla_c8_cube.h
 * \brief MLA 全量化重构模板 Cube Block
 */
#ifndef FIA_BLOCK_CUBE_FULLQUANT_MLA_H_
#define FIA_BLOCK_CUBE_FULLQUANT_MLA_H_
#include "../../offset_calculator.h"
#include "../../matmul.h"
#include "../../FixpipeOut.h"

#include "flash_mla_c8_memory.h"
#include "../infer_flash_attention_comm_arch35.h"
#include "../flash_attention_score_common_regbase_arch35.h"

#include "kernel_operator_list_tensor_intf.h"
using namespace AscendC;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;
using namespace fa_base_matmul;
using namespace AttentionCommon;

namespace BaseApi {

/* ============确定Query/Key的L1类型============= */
template <typename INPUT_T, uint32_t dBaseSize>
struct QL1BuffSelMla {
    using Type =
        std::conditional_t<std::is_same_v<INPUT_T, float> ||
                               (!(std::is_same_v<INPUT_T, fp8_e4m3fn_t> || std::is_same_v<INPUT_T, fp8_e5m2_t> ||
                                  std::is_same_v<INPUT_T, int8_t> || std::is_same_v<INPUT_T, hifloat8_t>) &&
                                dBaseSize > 256),
                           BuffersPolicySingleBuffer<BufferType::L1>, BuffersPolicyDB<BufferType::L1>>;
};

template <typename INPUT_T, uint32_t s2BaseSize, uint32_t dBaseSize>
struct KVL1BuffSelMla {
    constexpr static bool isFP8DType = std::is_same_v<INPUT_T, fp8_e4m3fn_t> || std::is_same_v<INPUT_T, fp8_e5m2_t> ||
                                       std::is_same_v<INPUT_T, hifloat8_t>;
    constexpr static bool isINT8DType = std::is_same_v<INPUT_T, int8_t>;
    using Type = std::conditional_t<
        ((isFP8DType || isINT8DType) && s2BaseSize == 128 && dBaseSize == 576), BuffersPolicy4buff<BufferType::L1>,
        std::conditional_t<(!(isFP8DType || isINT8DType) && s2BaseSize == 256 && dBaseSize > 128),
                           BuffersPolicySingleBuffer<BufferType::L1>, BuffersPolicyDB<BufferType::L1>>>;
};

template <typename INPUT_T>
struct L0ABuffSelMla {
    using Type = std::conditional_t<std::is_same_v<INPUT_T, float>, BuffersPolicySingleBuffer<BufferType::L0A>,
                                    BuffersPolicyDB<BufferType::L0A>>;
};

template <typename INPUT_T, uint32_t s2BaseSize, uint32_t dBaseSize>
struct L0BBuffSelMla {
    using Type =
        std::conditional_t<std::is_same_v<INPUT_T, float> ||
                               (s2BaseSize == 256 && dBaseSize > 128 &&
                                !(std::is_same_v<INPUT_T, fp8_e4m3fn_t> || std::is_same_v<INPUT_T, fp8_e5m2_t> ||
                                  std::is_same_v<INPUT_T, int8_t> || std::is_same_v<INPUT_T, hifloat8_t>)),
                           BuffersPolicySingleBuffer<BufferType::L0B>, BuffersPolicyDB<BufferType::L0B>>;
};

template <typename INPUT_T, uint32_t s1BaseSize, uint32_t s2BaseSize, uint32_t dVBaseSize>
struct L0CBuffSelMla {
    using Type = std::conditional_t<(s1BaseSize * s2BaseSize * FLOAT_BYTES <= (L0C_SIZE * KB_TO_BYTES) / NUM_4 &&
                                     s1BaseSize * dVBaseSize * FLOAT_BYTES <= (L0C_SIZE * KB_TO_BYTES) / NUM_4),
                                    BuffersPolicy4buff<BufferType::L0C>, BuffersPolicyDB<BufferType::L0C>>;
};

template <typename INPUT_T, typename T, FlashMlaC8Layout layout = FlashMlaC8Layout::None,
          S1TemplateType s1TemplateType = S1TemplateType::Aligned64,
          S2TemplateType s2TemplateType = S2TemplateType::Aligned128,
          DTemplateType dTemplateType = DTemplateType::Aligned576,
          DTemplateType dVTemplateType = DTemplateType::Aligned512, bool hasRope = false, uint8_t KvLayoutType = 0,
          bool enableKVPrefix = false, bool useDn = false, bool bmm2Write2Ub = true, bool splitD = false>
class FAFullQuantMlaBlockCube {
public:
    static constexpr uint32_t mBaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr uint32_t dBaseSize = (uint32_t)dTemplateType;
    static constexpr uint32_t dVBaseSize = (uint32_t)dVTemplateType;
    static constexpr FlashMlaC8Layout LAYOUT = layout;
    static constexpr bool PAGE_ATTENTION = (KvLayoutType > 0);
    static constexpr bool HAS_ROPE = hasRope;
    static constexpr bool BMM2_TOUB = bmm2Write2Ub;
    static constexpr bool USE_DN = useDn;
    static constexpr bool SPLITD = splitD;
    static constexpr bool RESIDENT_Q = IsSameType<INPUT_T, fp8_e4m3fn_t>::value &&
        mBaseSize == 64 && s2BaseSize == 128 && dBaseSize == 576 && dVBaseSize == 512 && HAS_ROPE;


    static constexpr bool isFp8 = IsSameType<INPUT_T, fp8_e5m2_t>::value || IsSameType<INPUT_T, fp8_e4m3fn_t>::value ||
                                  IsSameType<INPUT_T, hifloat8_t>::value;
    static constexpr bool isInt8 = IsSameType<INPUT_T, int8_t>::value;
    static constexpr TPosition bmm2OutPos =
        GetC2Position(dVTemplateType,
                      UbOutCondition<INPUT_T>(IsSameType<INPUT_T, float>::value, PseTypeEnum::PSE_NONE_TYPE, false,
                                              false, hasRope, mBaseSize == 64),
                      (s2BaseSize == 256 && mBaseSize == 64), true);
    static constexpr FixpipeConfig BMM2_FIXPIPE_CONFIG = {CO2Layout::ROW_MAJOR, BMM2_TOUB};

    static constexpr GmFormat Q_FORMAT = GetQueryGmFormat<layout>();
    static constexpr GmFormat KV_FORMAT = GetKVGmFormat<layout, KvLayoutType, PAGE_ATTENTION>();

    using ROPE_T = bfloat16_t;
    using Q_T = INPUT_T;
    using KV_T = INPUT_T;
    using MM_T = T;
    using MLA_FULLQUANT_MM2_T = std::conditional_t<isInt8, int32_t, T>;
    using mm2ResPos = typename std::conditional<BMM2_TOUB, Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>,
                                                Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_FORWARD>>::type;

    using MM1_DBUF_T = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;
    using MM2_ABUF_POLICY_T = BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD>;
    using MM2_ABUF_T = Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD>;
    using MM2_DBUF_T = mm2ResPos;

    using L1KvType = typename KVL1BuffSelMla<INPUT_T, s2BaseSize, dBaseSize>::Type;
    using L0AType = typename L0ABuffSelMla<INPUT_T>::Type;
    using L0BType = typename L0BBuffSelMla<INPUT_T, s2BaseSize, dBaseSize>::Type;
    using L0CType = typename L0CBuffSelMla<INPUT_T, mBaseSize, s2BaseSize, dVBaseSize>::Type;

    using ConstInfoX = ConstInfo_t<FiaKernelType::FULL_QUANT>;
    TPipe *tPipe = nullptr;

    /* =====================GM变量(with layout)==================== */
    FaGmTensor<Q_T, Q_FORMAT, uint32_t, true> queryGm;
    FaGmTensor<KV_T, KV_FORMAT, uint32_t> keyGm;
    FaGmTensor<KV_T, KV_FORMAT, uint32_t> valueGm;
    FaGmTensor<ROPE_T, Q_FORMAT, uint32_t, true> queryRopeGm;
    FaGmTensor<ROPE_T, KV_FORMAT, uint32_t> keyRopeGm;
    GlobalTensor<int32_t> blockTableGm;
    GlobalTensor<uint32_t> actualSeqLengthsGmQ;
    GlobalTensor<uint32_t> actualSeqLengthsGmKv;
    GlobalTensor<float> deScaleQGm;
    GlobalTensor<float> deScaleKGm;
    GlobalTensor<float> deScaleVGm;

    CopyQueryGmToL1<Q_T, Q_FORMAT> copyQueryGmToL1;
    CopyKvGmToL1<KV_T, KV_FORMAT> copyKvGmToL1;
    CopyQueryGmToL1<ROPE_T, Q_FORMAT> copyQueryRopeGmToL1;
    CopyKvGmToL1<ROPE_T, KV_FORMAT> copyKeyRopeGmToL1;

    /* =====================LocalBuffer变量==================== */
    BufferManager<BufferType::L1> *l1BufferManagerPtr;
    BufferManager<BufferType::L0A> l0aBufferManager;
    BufferManager<BufferType::L0B> l0bBufferManager;
    BufferManager<BufferType::L0C> l0cBufferManager;

    typename QL1BuffSelMla<INPUT_T, dBaseSize>::Type l1QBuffers;
    L1KvType l1KBuffers;

    L0AType mmL0ABuffers;
    BuffersPolicySingleBuffer<BufferType::L0A> residentQL0A;
    L0BType mmL0BBuffers;
    L0CType mmL0CBuffers;

    __gm__ uint8_t *keyPtr = nullptr;
    __gm__ uint8_t *valuePtr = nullptr;

    const ConstInfoX &constInfo;

    __aicore__ inline FAFullQuantMlaBlockCube(ConstInfoX &constInfo)
        : constInfo(constInfo){};

    __aicore__ inline void InitCubeBlock(TPipe *pipe, BufferManager<BufferType::L1> *l1BuffMgr, __gm__ uint8_t *query,
                                         __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *blockTable,
                                         __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope,
                                         __gm__ uint8_t *actualSeqQlenAddr, __gm__ uint8_t *actualSeqKvlenAddr,
                                         __gm__ uint8_t *dequantScaleQuery, __gm__ uint8_t *dequantScaleKey,
                                         __gm__ uint8_t *dequantScaleValue)
    {
        tPipe = pipe;
        l1BufferManagerPtr = l1BuffMgr;
        InitCubeInput(query, key, value, blockTable, queryRope, keyRope, actualSeqQlenAddr, actualSeqKvlenAddr,
                      dequantScaleQuery, dequantScaleKey, dequantScaleValue);
    }

    __aicore__ inline void InitBuffers()
    {
        constexpr uint32_t dRopeBaseSize = dBaseSize - dVBaseSize;
        constexpr uint32_t mm1QSize = mBaseSize * dVBaseSize * sizeof(INPUT_T);
        constexpr uint32_t mm1QRopeSize = mBaseSize * dRopeBaseSize * sizeof(bfloat16_t);
        constexpr uint32_t mm1KSize = dVBaseSize * s2BaseSize * sizeof(INPUT_T);
        constexpr uint32_t mm1KRopeSize = dRopeBaseSize * s2BaseSize * sizeof(bfloat16_t);

        l1QBuffers.Init((*l1BufferManagerPtr), mm1QSize + mm1QRopeSize);
        l1KBuffers.Init((*l1BufferManagerPtr), mm1KSize + mm1KRopeSize);

        l0aBufferManager.Init(tPipe, 65536);  // 64 * 1024
        l0bBufferManager.Init(tPipe, 65536);  // 64 * 1024
        l0cBufferManager.Init(tPipe, 262144); // 256 * 1024
        if constexpr (RESIDENT_Q) {
            // Q[0:256] [0,16KiB), Q[256:512] [16,32KiB), QR [32,40KiB).
            // PV uses independent ping/pong P slots [40,48KiB), [48,56KiB).
            residentQL0A.Init(l0aBufferManager, 40 * 1024);
            mmL0ABuffers.Init(l0aBufferManager, 8 * 1024);
        } else {
            mmL0ABuffers.Init(l0aBufferManager, 32 * 1024);
        }
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
                                         __gm__ uint8_t *actualSeqQlenAddr, __gm__ uint8_t *actualSeqKvlenAddr,
                                         __gm__ uint8_t *dequantScaleQuery, __gm__ uint8_t *dequantScaleKey,
                                         __gm__ uint8_t *dequantScaleValue)
    {
        if (constInfo.actualSeqLenSize != 0) {
            actualSeqLengthsGmQ.SetGlobalBuffer((__gm__ uint32_t *)actualSeqQlenAddr, constInfo.actualSeqLenSize);
        }
        if (constInfo.actualSeqLenKVSize != 0) {
            actualSeqLengthsGmKv.SetGlobalBuffer((__gm__ uint32_t *)actualSeqKvlenAddr, constInfo.actualSeqLenKVSize);
        }
        if constexpr (PAGE_ATTENTION) {
            blockTableGm.SetGlobalBuffer((__gm__ int32_t *)blockTable);
        }

        InitQBuffer(constInfo.bSize, constInfo.realN2Size, constInfo.realGSize, constInfo.s1Size, constInfo.dSizeV,
                    actualSeqLengthsGmQ, constInfo.actualSeqLenSize, queryGm, query);
        if constexpr (HAS_ROPE) {
            InitQRopeBuffer(constInfo.bSize, constInfo.realN2Size, constInfo.realGSize, constInfo.s1Size,
                            constInfo.dSizeRope, actualSeqLengthsGmQ, constInfo.actualSeqLenSize, queryRopeGm,
                            queryRope);
        }

        keyPtr = key;
        valuePtr = value;
        __gm__ uint8_t *key_ = key;
        __gm__ uint8_t *value_ = value;
        InitKVBuffer(constInfo.bSize, constInfo.s2Size, actualSeqLengthsGmKv, constInfo.actualSeqLenKVSize,
                     constInfo.n2Size, constInfo.blockSize, constInfo.dSizeV, keyGm, key_,
                     constInfo.keyStrides.bnStride, constInfo.keyStrides.n2Stride);
        InitKVBuffer(constInfo.bSize, constInfo.s2Size, actualSeqLengthsGmKv, constInfo.actualSeqLenKVSize,
                     constInfo.n2Size, constInfo.blockSize, constInfo.dSizeV, valueGm, value_,
                     constInfo.valueStrides.bnStride, constInfo.valueStrides.n2Stride);
        if constexpr (HAS_ROPE) {
            InitKRopeBuffer(constInfo.bSize, constInfo.s2Size, actualSeqLengthsGmKv, constInfo.actualSeqLenKVSize,
                            constInfo.n2Size, constInfo.blockSize, constInfo.dSizeRope, keyRopeGm, keyRope,
                            constInfo.kRopeStrides.bnStride, constInfo.kRopeStrides.n2Stride);
        }

        // MLA 全量化 dequantScale: Q per-token, KV per-tensor
        if (dequantScaleQuery != nullptr) {
            deScaleQGm.SetGlobalBuffer((__gm__ float *)dequantScaleQuery);
        }
        if (dequantScaleKey != nullptr) {
            deScaleKGm.SetGlobalBuffer((__gm__ float *)dequantScaleKey);
        }
        if (dequantScaleValue != nullptr) {
            deScaleVGm.SetGlobalBuffer((__gm__ float *)dequantScaleValue);
        }
    }

    __aicore__ inline void InitQBuffer(uint32_t batchSize, uint32_t n2Size, uint32_t gSize, uint32_t qSeqSize,
                                       uint32_t headDim, GlobalTensor<uint32_t> actualSeqLenGmQ,
                                       uint32_t actualLenQDims, FaGmTensor<Q_T, Q_FORMAT, uint32_t, true> &qGmTensor,
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
                                           uint32_t headDim, GlobalTensor<uint32_t> actualSeqLenGmQ,
                                           uint32_t actualLenQDims, FaGmTensor<ROPE_T, Q_FORMAT, uint32_t, true> &qRopeGmTensor,
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

    __aicore__ inline void InitKVBuffer(uint32_t batchSize, uint32_t kvSeqSize, GlobalTensor<uint32_t> actualSeqLenGmKv,
                                        uint32_t actualLenDims, uint32_t n2Size, uint32_t kvCacheBlockSize,
                                        uint32_t headDim, FaGmTensor<KV_T, KV_FORMAT, uint32_t> &kvGmTensor, __gm__ uint8_t *gm,
                                        uint64_t bnStrides, uint64_t n2Strides)
    {
        kvGmTensor.gmTensor.SetGlobalBuffer((__gm__ KV_T *)gm);
        if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_PA_BNBD) {
            kvGmTensor.offsetCalculator.Init(n2Size, kvCacheBlockSize, headDim, blockTableGm,
                                             constInfo.maxBlockNumPerBatch, bnStrides, n2Strides);
        } else if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_PA_NZ) {
            constexpr uint32_t d0 = 32 / sizeof(KV_T);
            uint32_t d1 = headDim / d0;
            kvGmTensor.offsetCalculator.Init(n2Size, kvCacheBlockSize, d1, d0, blockTableGm,
                                             constInfo.maxBlockNumPerBatch, bnStrides, n2Strides);
        } else if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_BNSD) {
            kvGmTensor.offsetCalculator.Init(batchSize, n2Size, kvSeqSize, headDim, actualSeqLenGmKv, actualLenDims,
                                             false, 0);
        } else if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_TND) {
            kvGmTensor.offsetCalculator.Init(n2Size, headDim, actualSeqLenGmKv, actualLenDims);
        }
    }

    __aicore__ inline void InitKRopeBuffer(uint32_t batchSize, uint32_t kvSeqSize,
                                           GlobalTensor<uint32_t> actualSeqLenGmKv, uint32_t actualLenDims,
                                           uint32_t n2Size, uint32_t kvCacheBlockSize, uint32_t headDim,
                                           FaGmTensor<ROPE_T, KV_FORMAT, uint32_t> &kRopeGmTensor, __gm__ uint8_t *gm,
                                           uint64_t bnStrides, uint64_t n2Strides)
    {
        kRopeGmTensor.gmTensor.SetGlobalBuffer((__gm__ ROPE_T *)gm);
        if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_PA_BNBD) {
            kRopeGmTensor.offsetCalculator.Init(n2Size, kvCacheBlockSize, headDim, blockTableGm,
                                                constInfo.maxBlockNumPerBatch, bnStrides, n2Strides);
        } else if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_PA_NZ) {
            constexpr uint32_t d0 = 32 / sizeof(ROPE_T);
            uint32_t d1 = headDim / d0;
            kRopeGmTensor.offsetCalculator.Init(n2Size, kvCacheBlockSize, d1, d0, blockTableGm,
                                                constInfo.maxBlockNumPerBatch, bnStrides, n2Strides);
        } else if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_BNSD) {
            kRopeGmTensor.offsetCalculator.Init(batchSize, n2Size, kvSeqSize, headDim, actualSeqLenGmKv, actualLenDims,
                                                false, 0);
        } else if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_TND) {
            kRopeGmTensor.offsetCalculator.Init(n2Size, headDim, actualSeqLenGmKv, actualLenDims);
        }
    }

    __aicore__ inline void SetQueryLengths(__gm__ uint8_t *cu, __gm__ uint8_t *used)
    {

        ActualSeqLensParser<ActualSeqLensMode::ACCUM, uint32_t, true> parser;
        parser.Init(cu, constInfo.actualSeqLenSize, used, used ? constInfo.bSize : 0);
        queryGm.offsetCalculator.Init(constInfo.realN2Size, constInfo.realGSize, constInfo.dSizeV, parser);
        queryRopeGm.offsetCalculator.Init(constInfo.realN2Size, constInfo.realGSize, constInfo.dSizeRope, parser);
    }

    __aicore__ inline void AllocEventID()
    {
        // InitBuffers阶段已完成eventId申请和SetFlag，这里为空实现
    }

    __aicore__ inline void FreeEventID()
    {
        l1QBuffers.Uninit((*l1BufferManagerPtr));
        l1KBuffers.Uninit((*l1BufferManagerPtr));
        if constexpr (RESIDENT_Q) {
            residentQL0A.Uninit(l0aBufferManager);
        }
        mmL0ABuffers.Uninit(l0aBufferManager);
        mmL0BBuffers.Uninit(l0bBufferManager);
        mmL0CBuffers.Uninit(l0cBufferManager);
    }

    // copy query nope part (dSizeV)
    __aicore__ inline void CopyQueryTile(const LocalTensor<Q_T> &dstTensor, RunInfoX &runInfo)
    {
        constexpr uint32_t blockNumDtype = 32 / sizeof(Q_T);
        uint32_t nopeDealSize = constInfo.dSizeV;
        uint32_t dstStride = (runInfo.actMSize + 31) >> 5 << 5;
        FaL1Tensor<Q_T, L1Format::NZ> l1Tensor{.tensor = dstTensor, .rowCount = dstStride};

        GmCoordGs1Merge gmCoord{.bIdx = runInfo.bIdx,
                                .n2Idx = runInfo.realN2Idx,
                                .gS1Idx = runInfo.gS1Idx,
                                .dIdx = 0,
                                .gS1DealSize = runInfo.actMSize,
                                .dDealSize = nopeDealSize};
        copyQueryGmToL1(l1Tensor, queryGm, gmCoord);

        if constexpr (HAS_ROPE) {
            uint32_t ropeDealSize = constInfo.dSizeRope;
            uint32_t dstStrideRope = (runInfo.actMSize + 15) >> 4 << 4;
            uint32_t offsetQRopeByElement = AttentionCommon::Align(nopeDealSize, blockNumDtype) *
                                            AttentionCommon::Align(runInfo.actMSize, blockNumDtype);
            FaL1Tensor<ROPE_T, L1Format::NZ> l1RopeTensor{
                .tensor = (dstTensor[offsetQRopeByElement]).template ReinterpretCast<ROPE_T>(),
                .rowCount = dstStrideRope};

            GmCoordGs1Merge gmCoordRope{.bIdx = runInfo.bIdx,
                                        .n2Idx = runInfo.realN2Idx,
                                        .gS1Idx = runInfo.gS1Idx,
                                        .dIdx = 0,
                                        .gS1DealSize = runInfo.actMSize,
                                        .dDealSize = ropeDealSize};
            copyQueryRopeGmToL1(l1RopeTensor, queryRopeGm, gmCoordRope);
        }
    }

    // copy key nope part + rope part
    __aicore__ inline void CopyKeyTile(const LocalTensor<KV_T> &dstTensor, RunInfoX &runInfo, uint32_t s2RealSize)
    {
        constexpr uint32_t blockNumDtype = 32 / sizeof(KV_T);
        uint32_t dstStride = (s2RealSize + 31) >> 5 << 5;
        uint32_t nopeDealSize = constInfo.dSizeV;
        FaL1Tensor<KV_T, L1Format::NZ> l1Tensor{.tensor = dstTensor, .rowCount = dstStride};

        GmKvCoord gmCoord{.bIdx = runInfo.bIdx,
                          .n2Idx = runInfo.n2Idx,
                          .s2Idx = runInfo.s2Idx,
                          .dIdx = 0,
                          .s2DealSize = s2RealSize,
                          .dDealSize = nopeDealSize};
        copyKvGmToL1(l1Tensor, keyGm, gmCoord);

        if constexpr (HAS_ROPE) {
            uint32_t ropeDealSize = constInfo.dSizeRope;
            uint32_t dstStrideRope = (s2RealSize + 15) >> 4 << 4;
            uint32_t offsetKRopeByElement =
                AttentionCommon::Align(nopeDealSize, blockNumDtype) * AttentionCommon::Align(s2RealSize, blockNumDtype);
            FaL1Tensor<ROPE_T, L1Format::NZ> l1RopeTensor{
                .tensor = (dstTensor[offsetKRopeByElement]).template ReinterpretCast<ROPE_T>(),
                .rowCount = dstStrideRope};

            GmKvCoord gmCoordRope{.bIdx = runInfo.bIdx,
                                  .n2Idx = runInfo.n2Idx,
                                  .s2Idx = runInfo.s2Idx,
                                  .dIdx = 0,
                                  .s2DealSize = s2RealSize,
                                  .dDealSize = ropeDealSize};
            copyKeyRopeGmToL1(l1RopeTensor, keyRopeGm, gmCoordRope);
        }
    }

    // MLA 中 V = K_nope, bmm2 复用 K 的 L1 buffer, 无需独立搬运 V

    __aicore__ inline void IterateBmm1(MM1_DBUF_T &outputBuf, RunInfoX &runInfo)
    {
        // MLA 全量化 bmm1: Q_nope @ K_nope^T + Q_rope @ K_rope^T
        // Q (含 rope) 首次全载 L1，后续 s2 循环复用
        Buffer<BufferType::L1> mm1A;
        uint32_t dTypeRatio = sizeof(bfloat16_t) / sizeof(INPUT_T);
        uint32_t dstNzC0StrideQNope = (runInfo.actMSize + 31) >> 5 << 5;
        uint32_t offsetQRopeByElement = dstNzC0StrideQNope * constInfo.dSizeV / dTypeRatio;

        if (unlikely(runInfo.isFirstS2Loop)) {
            mm1A = l1QBuffers.Get();
            mm1A.Wait<HardEvent::MTE1_MTE2>();
            LocalTensor<Q_T> mm1ATensor = mm1A.GetTensor<Q_T>();
            CopyQueryTile(mm1ATensor, runInfo);
            mm1A.Set<HardEvent::MTE2_MTE1>();
        } else {
            mm1A = l1QBuffers.GetPre();
            mm1A.Set<HardEvent::MTE2_MTE1>();
        }

        // 加载当前轮的 K (含 rope) 到 L1
        Buffer<BufferType::L1> mm1B = l1KBuffers.Get();
        mm1B.Wait<HardEvent::MTE1_MTE2>();
        LocalTensor<KV_T> mm1BTensor = mm1B.GetTensor<KV_T>();
        uint32_t s2CurSize = runInfo.actSingleLoopS2Size;
        uint32_t dstNzC0StrideKNope = (s2CurSize + 31) >> 5 << 5;
        uint32_t offsetKRopeByElement = dstNzC0StrideKNope * constInfo.dSizeV / dTypeRatio;
        CopyKeyTile(mm1BTensor, runInfo, s2CurSize);
        mm1B.Set<HardEvent::MTE2_MTE1>();

        mm1A.Wait<HardEvent::MTE2_MTE1>();
        mm1B.Wait<HardEvent::MTE2_MTE1>();

        Buffer<BufferType::L0C> mm1ResL0C = mmL0CBuffers.Get();
        mm1ResL0C.Wait<HardEvent::FIX_M>();

        if constexpr (RESIDENT_Q) {
            ResidentQueryQK(mm1A.GetTensor<INPUT_T>(), mm1B.GetTensor<INPUT_T>(),
                mm1A.GetTensor<bfloat16_t>(offsetQRopeByElement),
                mm1B.GetTensor<bfloat16_t>(offsetKRopeByElement),
                mm1ResL0C.GetTensor<T>(), runInfo, s2CurSize);
        } else {
        // Nope MatMul: Q_nope @ K_nope^T
        MMParam param =
            MakeMMParam((uint32_t)runInfo.actMSize, (uint32_t)s2CurSize, (uint32_t)constInfo.dSizeV, false, true);
        MatmulK<INPUT_T, INPUT_T, T, 64, 128, 256, ABLayout::MK, ABLayout::KN>(
            mm1A.GetTensor<INPUT_T>(), mm1B.GetTensor<INPUT_T>(), mmL0ABuffers, mmL0BBuffers, mm1ResL0C.GetTensor<T>(),
            param);

        // Rope MatMul: Q_rope @ K_rope^T (累加 Nope L0C)
        if constexpr (HAS_ROPE) {
            MMParam paramRope = MakeMMParam((uint32_t)runInfo.actMSize, (uint32_t)s2CurSize,
                                            (uint32_t)constInfo.dSizeRope, false, true, true, false);
            MatmulFull<bfloat16_t, bfloat16_t, T, 64, 128, 64, ABLayout::MK, ABLayout::KN>(
                mm1A.GetTensor<bfloat16_t>(offsetQRopeByElement), mm1B.GetTensor<bfloat16_t>(offsetKRopeByElement),
                mmL0ABuffers, mmL0BBuffers, mm1ResL0C.GetTensor<T>(), paramRope);
        }

        }

        if (unlikely(runInfo.isLastS2Loop)) {
            mm1A.Set<HardEvent::MTE1_MTE2>();
        }
        mm1ResL0C.Set<HardEvent::M_FIX>();
        mm1ResL0C.Wait<HardEvent::M_FIX>();

        outputBuf.WaitCrossCore();

        FixpipeMm1(outputBuf.template GetTensor<T>(), mm1ResL0C.GetTensor<T>(), runInfo, s2CurSize);

        mm1ResL0C.Set<HardEvent::FIX_M>();
        outputBuf.SetCrossCore();
    }

    __aicore__ inline void ResidentQueryQK(
        const LocalTensor<INPUT_T> &qL1, const LocalTensor<INPUT_T> &kL1,
        const LocalTensor<bfloat16_t> &qrL1, const LocalTensor<bfloat16_t> &krL1,
        const LocalTensor<T> &cL0, RunInfoX &runInfo, uint32_t n)
    {
        // Fixed regions preserve each original K256 load's compact M layout,
        // including M tails. No consumer ever interprets the gap between them.
        auto qBuffer = residentQL0A.Get();
        auto q0 = qBuffer.template GetTensor<INPUT_T>();
        auto q1 = qBuffer.template GetTensor<INPUT_T>(16 * 1024);
        auto qr = qBuffer.template GetTensor<bfloat16_t>(32 * 1024 / sizeof(bfloat16_t));
        auto param = MakeMMParam(static_cast<uint32_t>(runInfo.actMSize), n, 512, false, true);
        auto ropeParam = MakeMMParam(static_cast<uint32_t>(runInfo.actMSize), n, 64,
                                    false, true, true, false);
        const uint32_t qL1Stride = ((param.singleM + 31) / 32) * 32 * 256;
        const uint32_t kL1Stride = ((n + 31) / 32) * 32 * 256;
        if (unlikely(runInfo.isFirstS2Loop)) {
            // Previous M task's final QK must stop reading Q before overwrite.
            // The initial free token is supplied by Buffer::Init.
            qBuffer.template Wait<HardEvent::M_MTE1>();
            LoadDataToL0A(q0, qL1, param, 0, 256, param.singleM);
            LoadDataToL0A(q1, qL1, param, qL1Stride, 256, param.singleM);
            LoadDataToL0A(qr, qrL1, ropeParam, 0, 64, param.singleM);
            qBuffer.template Set<HardEvent::MTE1_M>();
            qBuffer.template Wait<HardEvent::MTE1_M>();
        }

        for (uint32_t k = 0; k < 2; ++k) {
            auto bBuffer = mmL0BBuffers.Get();
            bBuffer.template Wait<HardEvent::M_MTE1>();
            auto bL0 = bBuffer.template GetTensor<INPUT_T>();
            LoadDataToL0B(bL0, kL1, param, k * kL1Stride, 256, GetLoadN(param), 1);
            bBuffer.template Set<HardEvent::MTE1_M>();
            bBuffer.template Wait<HardEvent::MTE1_M>();
            MmadParams mm;
            mm.m = param.singleM == 1 ? 16 : param.singleM;
            mm.n = n;
            mm.k = 256;
            mm.cmatrixInitVal = k == 0;
            mm.cmatrixSource = false;
            if (k == 0) {
                Mmad(cL0, q0, bL0, mm);
            } else {
                Mmad(cL0, q1, bL0, mm);
            }
            bBuffer.template Set<HardEvent::M_MTE1>();
        }
        auto bBuffer = mmL0BBuffers.Get();
        bBuffer.template Wait<HardEvent::M_MTE1>();
        auto bL0 = bBuffer.template GetTensor<bfloat16_t>();
        LoadDataToL0B(bL0, krL1, ropeParam, 0, 64, GetLoadN(ropeParam), 1);
        bBuffer.template Set<HardEvent::MTE1_M>();
        bBuffer.template Wait<HardEvent::MTE1_M>();
        MmadParams mm;
        mm.m = param.singleM == 1 ? 16 : param.singleM;
        mm.n = n;
        mm.k = 64;
        mm.cmatrixInitVal = false;
        mm.cmatrixSource = false;
        Mmad(cL0, qr, bL0, mm);
        bBuffer.template Set<HardEvent::M_MTE1>();
        if (unlikely(runInfo.isLastS2Loop)) {
            // M-pipe completion covers both latent MMADs and component MMAD.
            // Later PV uses only P ping/pong; it cannot read or overwrite Q.
            qBuffer.template Set<HardEvent::M_MTE1>();
        }
    }

    __aicore__ inline void FixpipeMm1(const LocalTensor<T> &dstTensor, const LocalTensor<T> &l0C, RunInfoX &runInfo,
                                      uint32_t s2RealSize)
    {
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        fixpipeParams.nSize = (s2RealSize + 7) >> 3 << 3;
        fixpipeParams.mSize = (runInfo.actMSize + 1) >> 1 << 1;
        fixpipeParams.srcStride = ((runInfo.actMSize + 15) / 16) * 16;
        fixpipeParams.dstStride = s2BaseSize;
        fixpipeParams.dualDstCtl = 1;
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;

        Fixpipe<T, T, PFA_CFG_ROW_MAJOR_UB>(dstTensor, l0C, fixpipeParams);
    }

    __aicore__ inline void IterateBmm2(mm2ResPos &outputBuf, MM2_ABUF_POLICY_T &inputBuf, RunInfoX &runInfo)
    {
        // MLA 全量化 bmm2: P @ V_nope, V 复用 K 的 nope 部分(MLA 中 V=K_nope)
        MM2_ABUF_T mm2A = inputBuf.Get();
        mm2A.WaitCrossCore();

        if constexpr (BMM2_TOUB) {
            outputBuf.WaitCrossCore();
        }

        Buffer<BufferType::L1> mm2B = l1KBuffers.GetReused();
        Buffer<BufferType::L0C> mm2ResL0C = mmL0CBuffers.Get();
        mm2ResL0C.Wait<HardEvent::FIX_M>();

        MMParam param = MakeMMParam((uint32_t)mBaseSize, (uint32_t)constInfo.dSizeV,
                                    (uint32_t)runInfo.actSingleLoopS2Size, false, false);
        MatmulN<INPUT_T, INPUT_T, MLA_FULLQUANT_MM2_T, 128, 256, 128, ABLayout::MK, ABLayout::KN>(
            mm2A.GetTensor<INPUT_T>(), mm2B.GetTensor<INPUT_T>(), mmL0ABuffers, mmL0BBuffers,
            mm2ResL0C.GetTensor<MLA_FULLQUANT_MM2_T>(), param);
        // KV's last reader is PV LoadData, not the earlier QK LoadData.
        // This lifetime is independent of whether Q stays resident in L0A.
        mm2B.Set<HardEvent::MTE1_MTE2>();

        mm2ResL0C.Set<HardEvent::M_FIX>();
        mm2ResL0C.Wait<HardEvent::M_FIX>();

        FixpipeMm2(outputBuf.template GetTensor<MLA_FULLQUANT_MM2_T>(), mm2ResL0C.GetTensor<MLA_FULLQUANT_MM2_T>(),
                   runInfo);
        mm2ResL0C.Set<HardEvent::FIX_M>();
        outputBuf.SetCrossCore();
    }

    template <typename DST_TENSOR_T>
    __aicore__ inline void FixpipeMm2(const DST_TENSOR_T &dstTensor, const LocalTensor<MLA_FULLQUANT_MM2_T> &l0C,
                                      RunInfoX &runInfo)
    {
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        if constexpr (BMM2_TOUB) {
            fixpipeParams.nSize = ((uint32_t)constInfo.dSizeV + 7) >> 3 << 3;
        } else {
            fixpipeParams.nSize = constInfo.dSizeV;
        }
        fixpipeParams.mSize = mBaseSize;
        fixpipeParams.srcStride = ((mBaseSize + 15) / 16) * 16;
        if constexpr (BMM2_TOUB) {
            fixpipeParams.dstStride = ((uint32_t)dVBaseSize + 15) >> 4 << 4;
        } else {
            fixpipeParams.dstStride = (uint32_t)constInfo.dSizeV;
        }
        fixpipeParams.dualDstCtl = 1;
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;
        Fixpipe<MLA_FULLQUANT_MM2_T, MLA_FULLQUANT_MM2_T, BMM2_FIXPIPE_CONFIG>(dstTensor, l0C, fixpipeParams);
    }
};

// Dummy 类用于非 MLA 场景占位
template <typename INPUT_T, typename T, FlashMlaC8Layout layout = FlashMlaC8Layout::None,
          S1TemplateType s1TemplateType = S1TemplateType::Aligned64,
          S2TemplateType s2TemplateType = S2TemplateType::Aligned128,
          DTemplateType dTemplateType = DTemplateType::Aligned576,
          DTemplateType dVTemplateType = DTemplateType::Aligned512, bool hasRope = false, uint8_t KvLayoutType = 0,
          bool enableKVPrefix = false, bool useDn = false, bool bmm2Write2Ub = true, bool splitD = false>
class FAFullQuantMlaBlockCubeDummy {
public:
    __aicore__ inline void SetQueryLengths(__gm__ uint8_t *, __gm__ uint8_t *) {}

    static constexpr uint32_t mBaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr uint32_t dBaseSize = (uint32_t)dTemplateType;
    static constexpr uint32_t dVBaseSize = (uint32_t)dVTemplateType;
    static constexpr FlashMlaC8Layout LAYOUT = layout;
    static constexpr bool PAGE_ATTENTION = (KvLayoutType > 0);
    static constexpr bool HAS_ROPE = hasRope;
    static constexpr bool BMM2_TOUB = bmm2Write2Ub;
    static constexpr bool USE_DN = useDn;
    static constexpr bool SPLITD = splitD;
    static constexpr bool RESIDENT_Q = IsSameType<INPUT_T, fp8_e4m3fn_t>::value &&
        mBaseSize == 64 && s2BaseSize == 128 && dBaseSize == 576 && dVBaseSize == 512 && HAS_ROPE;

    static constexpr bool isFp8 =
        FAFullQuantMlaBlockCube<INPUT_T, T, layout, s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType,
                                hasRope, KvLayoutType, enableKVPrefix, useDn, bmm2Write2Ub, splitD>::isFp8;
    static constexpr bool isInt8 =
        FAFullQuantMlaBlockCube<INPUT_T, T, layout, s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType,
                                hasRope, KvLayoutType, enableKVPrefix, useDn, bmm2Write2Ub, splitD>::isInt8;
    static constexpr bool useDnDummy = false;
    static constexpr bool useNz = false;
    static constexpr TPosition bmm2OutPos =
        FAFullQuantMlaBlockCube<INPUT_T, T, layout, s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType,
                                hasRope, KvLayoutType, enableKVPrefix, useDn, bmm2Write2Ub, splitD>::bmm2OutPos;
    static constexpr bool bmm2Write2UbDummy = bmm2Write2Ub;
    static constexpr bool splitDDummy = splitD;
    using ROPE_T = bfloat16_t;
    using Q_T = INPUT_T;
    using KV_T = INPUT_T;
    using MM_T = T;
    using mm2ResPos = typename std::conditional<BMM2_TOUB, Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>,
                                                Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_FORWARD>>::type;
    using MM1_DBUF_T = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;
    using MM2_ABUF_POLICY_T = BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD>;
    using MM2_ABUF_T = Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD>;
    using ConstInfoX = ConstInfo_t<FiaKernelType::FULL_QUANT>;
    __aicore__ inline FAFullQuantMlaBlockCubeDummy(ConstInfoX &constInfo)
        : constInfo(constInfo){};
    __aicore__ inline void InitCubeBlock(TPipe *, BufferManager<BufferType::L1> *, __gm__ uint8_t *, __gm__ uint8_t *,
                                         __gm__ uint8_t *, __gm__ uint8_t *, __gm__ uint8_t *, __gm__ uint8_t *,
                                         __gm__ uint8_t *, __gm__ uint8_t *, __gm__ uint8_t *, __gm__ uint8_t *,
                                         __gm__ uint8_t *)
    {}
    __aicore__ inline void InitCubeInput(__gm__ uint8_t *, __gm__ uint8_t *, __gm__ uint8_t *, __gm__ uint8_t *,
                                         __gm__ uint8_t *, __gm__ uint8_t *, __gm__ uint8_t *, __gm__ uint8_t *,
                                         __gm__ uint8_t *, __gm__ uint8_t *, __gm__ uint8_t *)
    {}
    __aicore__ inline void InitBuffers() {}
    __aicore__ inline void InitDequantParams(__gm__ uint8_t *, __gm__ uint8_t *, __gm__ uint8_t *) {}
    __aicore__ inline void IterateBmm1(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &, RunInfoX &) {}
    __aicore__ inline void IterateBmm2(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &,
                                       BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &,
                                       RunInfoX &)
    {}
    __aicore__ inline void IterateBmm2(Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_FORWARD> &,
                                       BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &,
                                       RunInfoX &)
    {}
    __aicore__ inline void AllocEventID() {}
    __aicore__ inline void FreeEventID() {}
    const ConstInfoX &constInfo;
};

} // namespace BaseApi
#endif // FIA_BLOCK_CUBE_FULLQUANT_MLA_H_
