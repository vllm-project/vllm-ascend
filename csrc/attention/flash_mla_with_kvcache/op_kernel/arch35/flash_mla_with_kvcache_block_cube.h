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
 * \file flash_mla_with_kvcache_block_cube.h
 * \brief arch35 flash_mla_with_kvcache block cube MLA；
 *        静态 tensor buffer + Mutex/cross-core 显式同步。
 *        L0C：bmm2 单次 Mmad 足迹 96*256*sizeof(float) = 96 KiB，槽保持 2×128K 静态 + 2 Mutex（不得拆 4×64K）。
 */
#ifndef FLASH_MLA_WITH_KVCACHE_BLOCK_CUBE_H_
#define FLASH_MLA_WITH_KVCACHE_BLOCK_CUBE_H_
#include "../../../a5_mla_common/op_kernel/const_def.h"
#include "kernel_operator_list_tensor_intf.h"
#include "../utils/flash_mla_with_kvcache_type.h"
using namespace AscendC;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;
using namespace AttentionCommon;

namespace FlashAttnKernel {

template <typename FA_T>
class FlashMlaWithKvcacheNoQuantMlaBlockCube {
public:
    using INPUT_T = typename FA_T::inputType;
    using T = typename FA_T::mmType;
    static constexpr uint32_t mBaseSize = (uint32_t)FA_T::mBaseSize;
    static constexpr uint32_t s2BaseSize = (uint32_t)FA_T::s2BaseSize;
    static constexpr uint32_t dBaseSize = (uint32_t)FA_T::dBaseSize;
    static constexpr uint32_t dVBaseSize = (uint32_t)FA_T::dVBaseSize;
    static constexpr FLASH_MLA_WITH_KVCACHE_LAYOUT LAYOUT = FA_T::qLayout;
    static constexpr bool PAGE_ATTENTION = FA_T::pageAttention;
    static constexpr ActualSeqLensMode Q_MODE = GetQActSeqMode<LAYOUT>();
    static constexpr ActualSeqLensMode KV_MODE = GetKvActSeqMode<LAYOUT, PAGE_ATTENTION>();
    using SeqLensToolType = FlashMlaSeqLensTool<Q_MODE, KV_MODE>;

    static constexpr FixpipeConfig FIXPIPE_ROW_MAJOR_UB = {CO2Layout::ROW_MAJOR, true};
    static constexpr GmFormat Q_FORMAT = GetQueryGmFormat<FA_T::qLayout>();
    static constexpr GmFormat KV_FORMAT = GetKVGmFormat<FA_T::qLayout, FA_T::kvLayoutType, PAGE_ATTENTION>();

    using Q_T = INPUT_T;
    using KV_T = INPUT_T;
    using MM_T = T;

    using ConstInfoX = ConstInfo_t<FlashMlaWithKvcacheKernelType::NO_QUANT>;

    // ================== 核间同步 ID（与 vec 侧逐字节一致）==================
    static constexpr uint64_t CROSS_CORE_SYNC_MODE = 4U;
    static constexpr uint32_t CROSSCORE_BMM1_0 = 0U;
    static constexpr uint32_t CROSSCORE_BMM2_0 = 2U;
    static constexpr uint32_t CROSSCORE_BMM2_1 = 3U;
    static constexpr uint32_t CROSSCORE_L1P_0 = 5U;
    static constexpr uint32_t CROSSCORE_L1P_1 = 6U;
    static constexpr uint32_t CROSSCORE_L1P_2 = 7U;
    // 两个 AIV 的核间同步旗标相差 16。
    static constexpr uint32_t AIV_SYNC_OFFSET = 16U;

    // ================== 核内 Mutex ID（AIC 核内 0-26 自洽）==================
    static constexpr uint32_t L0C_BUFFER_ID0 = 0U;
    static constexpr uint32_t L0C_BUFFER_ID1 = 1U;
    static constexpr uint32_t Q_L1_BUFFER_ID0 = 2U; // Q 单槽，容量 mBaseSize*576*sizeof(Q_T)
    static constexpr uint32_t KV_L1_BUFFER_ID0 = 4U;
    static constexpr uint32_t KV_L1_BUFFER_ID1 = 5U;
    static constexpr uint32_t KV_L1_BUFFER_ID2 = 6U;
    static constexpr uint32_t L0A_BUFFER_ID0 = 7U;
    static constexpr uint32_t L0B_BUFFER_ID0 = 9U;
    static constexpr uint32_t L0B_BUFFER_ID1 = 10U;

    // ================== 静态布局常量（M96、S2=112、输入元素 2 字节）==================
    // UB CV 区（VECIN，与 vec 侧同一偏移别名）：
    //   mm2Res 2×48K @0、mm1Res 1×24K @96K
    static constexpr uint32_t UB_MM2_RES_BUFCNT = 2U;
    static constexpr uint32_t UB_MM2_RES_BUF_BYTES = mBaseSize / CV_RATIO * 256U * sizeof(MM_T);
    static constexpr uint32_t UB_MM1_RES_BUFCNT = 1U;
    static constexpr uint32_t UB_MM1_RES_BUF_BYTES = mBaseSize / CV_RATIO * 128U * sizeof(MM_T);
    // L1（A1）：Q 1×108K @0（576宽），KV/P 3×133K @108K（mm12RightSize；k==v，P 复用）
    static constexpr uint32_t L1_Q_BUFCNT = 1U;
    static constexpr uint32_t L1_Q_BUF_BYTES = mBaseSize * 576 * sizeof(Q_T);
    static constexpr uint32_t L1_KVP_BUFCNT = 3U;
    static constexpr uint32_t L1_KVP_BUF_BYTES = s2BaseSize * 608 * sizeof(INPUT_T);
    // L0C（CO1）：2×128 KiB，每槽容纳 96×256 FP32 结果
    static constexpr uint32_t L0C_BUFCNT = 2U;
    static constexpr uint32_t L0C_BUF_BYTES = 128U * 1024U;
    // L0A：首个 96 宽 Q 分块常驻 18K @0；临时 Q128 使用 24K @18K，Q112/P 使用 21K @42K。
    // L0B：2×32K 双缓冲，独立游标跨 mm1/mm2 轮转。
    static constexpr uint32_t BUFFER_SIZE_BYTE_64K = 65536;
    static constexpr uint32_t BUFFER_SIZE_BYTE_32K = 32768;

    // buffer位置+用途+Buffers；使用时命名: 用途+buffer位置+Tensor
    LocalTensor<uint8_t> ubMm1ResBuffers_;
    LocalTensor<uint8_t> ubMm2ResBuffers_;
    LocalTensor<uint8_t> l1QBuffers_;
    LocalTensor<uint8_t> l1KvpBuffers_; // K/V/P 共用 ring（vec 写 P 的目标偏移注释见 vec 块）
    LocalTensor<uint8_t> l0CBuffers_;
    uint32_t l0cBufId_ = 0U;
    LocalTensor<uint8_t> l0ABuffers_;
    LocalTensor<uint8_t> l0BBuffers_;
    uint32_t l0bBufId_ = 0U;
    uint32_t l0aBufId_ = 0U;
    bool lastMatmulWasPv_ = false;

    /* =====================GM变量(with layout)==================== */
    // q/k_cache 为 576 宽单张量（nope 512 + rope 64 合并），无独立 queryRope/keyRope 张量；
    // query 与 key 的 GM stride 布局按 576 宽初始化（偏移切分见 CopyQueryAndRopeTile/CopyKeyAndRopeTile）。
    // seq-lens（cu_seqlens_q/cache_seqlens）为 INT32，ACTLEN_T=uint32_t。
    // WITH_ZERO_HEAD 仅对 TND（cu_seqlens 首零头）成立；BSND/BNSD 走 GM_Q_OUT_BNGSD 3-参实现
    // （offset_calculator_v2.h，无 WITH_ZERO_HEAD 形参），与 IS_TND 同构。
    FaGmTensor<Q_T, Q_FORMAT, uint32_t, LAYOUT == FLASH_MLA_WITH_KVCACHE_LAYOUT::TND> queryGm;
    FaGmTensor<KV_T, KV_FORMAT> kCacheGm;
    GlobalTensor<int32_t> blockTableGm;
    // seq-lens GM 张量与解析器所有权在 kernel 侧 FlashMlaSeqLensTool，本 block 只读引用
    SeqLensToolType &seqLensTool_;

    CopyQueryGmToL1<Q_T, Q_FORMAT> copyQueryGmToL1;
    CopyKvGmToL1<KV_T, KV_FORMAT> copyKvGmToL1;

    const ConstInfoX &constInfo;

    /*============================================================================== */
    __aicore__ inline FlashMlaWithKvcacheNoQuantMlaBlockCube(ConstInfoX &constInfo, SeqLensToolType &seqLensTool)
        : constInfo(constInfo),
          seqLensTool_(seqLensTool){};

    __aicore__ inline void InitCubeBlock(__gm__ uint8_t *query, __gm__ uint8_t *kCache, __gm__ uint8_t *blockTable)
    {
        InitCubeInput(query, kCache, blockTable);
    }

    __aicore__ inline void InitBuffers()
    {
        static_assert(mBaseSize == 96 && s2BaseSize == 112, "mBaseSize != 96 or s2BaseSize != 112");
        /*--------------------------------------------UB--------------------------------------------*/
        // CV 区与 vec 侧同一偏移别名（mm2@0..96K、mm1@96K..120K）
        uint32_t addrUb = 0;
        ubMm2ResBuffers_ = LocalTensor<uint8_t>(TPosition::VECIN, addrUb, UB_MM2_RES_BUFCNT * UB_MM2_RES_BUF_BYTES);
        addrUb = UB_MM2_RES_BUFCNT * UB_MM2_RES_BUF_BYTES;
        ubMm1ResBuffers_ = LocalTensor<uint8_t>(TPosition::VECIN, addrUb, UB_MM1_RES_BUFCNT * UB_MM1_RES_BUF_BYTES);

        /*--------------------------------------------L1--------------------------------------------*/
        struct L1Layout {
            uint8_t qBuffers[L1_Q_BUFCNT][L1_Q_BUF_BYTES];
            uint8_t kvpBuffers[L1_KVP_BUFCNT][L1_KVP_BUF_BYTES];
        };
        static_assert(sizeof(L1Layout) <= 512 * 1024, "L1 buffer too large");
        l1QBuffers_ = LocalTensor<uint8_t>(TPosition::A1, 0U, SIZE_OF_MEMBER(L1Layout, qBuffers));
        // KV/P 区从 A1 108K 起；vec 写 P 的目标偏移必须与本布局逐字节一致：
        //   l1KvpBuffers_[loop%3 * L1_KVP_BUF_BYTES] + s2BaseSize*dVBaseSize（rope 段）
        l1KvpBuffers_ =
            LocalTensor<uint8_t>(TPosition::A1, L1_Q_BUFCNT * L1_Q_BUF_BYTES, SIZE_OF_MEMBER(L1Layout, kvpBuffers));

        /*--------------------------------------------L0A/B--------------------------------------------*/
        static_assert(mBaseSize * s2BaseSize * sizeof(Q_T) <= BUFFER_SIZE_BYTE_32K, "L0A tile exceeds 32K");
        l0ABuffers_ = LocalTensor<uint8_t>(TPosition::A2, 0U, BUFFER_SIZE_BYTE_64K);
        l0BBuffers_ = LocalTensor<uint8_t>(TPosition::B2, 0U, BUFFER_SIZE_BYTE_64K);
        l0bBufId_ = 0U;
        lastMatmulWasPv_ = false;

        /*--------------------------------------------L0C--------------------------------------------*/
        // 2×128K 静态 + M/FIX Mutex 对（bmm2 单次 96*256*4 = 96 KiB，不得拆成 64 KiB 槽）
        l0CBuffers_ = LocalTensor<uint8_t>(TPosition::CO1, 0U, L0C_BUFCNT * L0C_BUF_BYTES);
    }

    __aicore__ inline void InitCubeInput(__gm__ uint8_t *query, __gm__ uint8_t *kCache, __gm__ uint8_t *blockTable)
    {
        if constexpr (PAGE_ATTENTION) {
            blockTableGm.SetGlobalBuffer((__gm__ int32_t *)blockTable);
        }

        // query 576 宽单张量（nope 512 + rope 64 合并）——GM stride 按 576 初始化，
        // 供 CopyQueryAndRopeTile 的 rope 段按 dIdx=512 偏移定位
        InitQBuffer(constInfo.bSize, constInfo.n2Size, constInfo.gSize, constInfo.s1Size,
                    constInfo.dSize + constInfo.dSizeRope, queryGm, query);

        // k_cache 为普通 ND 张量（非 TensorList）：直接取裸指针，
        // 不能用 ListTensorDesc 解码（会把 k_cache 数据首部误解析为 list 描述符 → GM 越界）

        // k_cache 576 宽单张量（nope 512 + rope 64 合并）——key 与 value 同源（k==v）：
        // bmm2 的 V 段 = L1 上 nope 区（512 宽，mm1 后仍留在 L1），无独立 value GM 张量/读取
        // 仅 PA 实例化（host 强制 PA 路由），非 PA 分支已删
        InitKVBuffer(constInfo.n2Size, constInfo.blockSize, constInfo.dSize + constInfo.dSizeRope, kCacheGm, kCache,
                     constInfo.keyStrides.bnStride, constInfo.keyStrides.n2Stride);

        // decode 场景 K/V 数据量远超 L2 容量，关闭 L2 Cache 避免无意义的缓存填充/驱逐开销
        // （由 tiling 侧 gSize*s1Size <= 128 下发开关，见 flash_mla_with_kvcache_tiling.cpp SetFATilingData）
        if (constInfo.l2CacheOffFlag) {
#ifndef ASCENDC_OOM
            kCacheGm.gmTensor.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
#endif
        }
    }

    __aicore__ inline void InitQBuffer(
        uint32_t batchSize, uint32_t n2Size, uint32_t gSize, uint32_t qSeqSize, uint32_t headDim,
        FaGmTensor<Q_T, Q_FORMAT, uint32_t, LAYOUT == FLASH_MLA_WITH_KVCACHE_LAYOUT::TND> &qGmTensor,
        __gm__ uint8_t *gm)
    {
        qGmTensor.gmTensor.SetGlobalBuffer((__gm__ Q_T *)gm);
        if constexpr (GmLayoutParams<Q_FORMAT>::CATEGORY == FormatCategory::GM_Q_OUT_BNGSD) {
            qGmTensor.offsetCalculator.Init(batchSize, n2Size, gSize, qSeqSize, headDim,
                                            seqLensTool_.qActSeqLensParser);
        } else if constexpr (GmLayoutParams<Q_FORMAT>::CATEGORY == FormatCategory::GM_Q_OUT_TND) {
            qGmTensor.offsetCalculator.Init(n2Size, gSize, headDim, seqLensTool_.qActSeqLensParser);
        }
    }

    __aicore__ inline void InitKVBuffer(uint32_t n2Size, uint32_t kvCacheBlockSize, uint32_t headDim,
                                        FaGmTensor<KV_T, KV_FORMAT> &kvGmTensor, __gm__ uint8_t *gm, uint64_t bnStride,
                                        uint64_t n2Stride)
    {
        kvGmTensor.gmTensor.SetGlobalBuffer((__gm__ KV_T *)gm);

        if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_PA_BNBD) {
            kvGmTensor.offsetCalculator.Init(n2Size, kvCacheBlockSize, headDim, blockTableGm,
                                             constInfo.maxBlockNumPerBatch, bnStride, n2Stride);
        } else if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_PA_NZ) {
            constexpr uint32_t d0 = 32 / sizeof(KV_T);
            uint32_t d1 = headDim / d0;
            kvGmTensor.offsetCalculator.Init(n2Size, kvCacheBlockSize, d1, d0, blockTableGm,
                                             constInfo.maxBlockNumPerBatch, bnStride, n2Stride);
        }
    }

    __aicore__ inline void InitCrossCoreSync()
    {
        // AIC 侧无需预置（AIV 的 InitCrossCoreSync 预置 BMM1 单槽和 BMM2 双槽的三个可用标志）
    }

    __aicore__ inline void UnInitCrossCoreSync()
    {
        // 收尾消耗 6 个 c1v1/c2v2 标志（3 个槽 × 2 个 AIV），防悬空旗标
        // 污染下一 kernel 的首次 Wait
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_BMM1_0);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_BMM1_0 + AIV_SYNC_OFFSET);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_BMM2_0);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_BMM2_0 + AIV_SYNC_OFFSET);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_BMM2_1);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_BMM2_1 + AIV_SYNC_OFFSET);
    }

    // copy query（q 为 576 宽单张量，GM 上 nope+rope 合并存储，
    // 一次拷满 576 宽连续块，避免两次 strided 拷贝（nope 512 + rope 64）在 GM 行内留 gap 降低 MTE2 效率；
    // L1 NZ 布局中 rope 段自然落在 [dSize*dstStride) 偏移处，与双段拷贝一致）
    __aicore__ inline void CopyQueryAndRopeTile(const LocalTensor<Q_T> &dstTensor, FlashMlaWithKvcacheRunInfoX &runInfo)
    {
        uint32_t dstStride = (runInfo.actMSize + 15) >> 4 << 4;
        FaL1Tensor<Q_T, L1Format::NZ> l1Tensor{.tensor = dstTensor, .rowCount = dstStride};

        GmCoordGs1Merge gmCoord{.bIdx = runInfo.bIdx,
                                .n2Idx = runInfo.n2Idx,
                                .gS1Idx = runInfo.gS1Idx,
                                .dIdx = 0,
                                .gS1DealSize = runInfo.actMSize,
                                .dDealSize = (uint32_t)(constInfo.dSize + constInfo.dSizeRope)};
        copyQueryGmToL1(l1Tensor, queryGm, gmCoord);
    }

    // copy key（k_cache 为 576 宽单张量，GM 上 nope+rope 合并存储，
    // 一次拷满 576 宽连续块，避免两次 strided 拷贝在 GM 行内留 gap；L1 NZ 布局 rope 段自然落在
    // [dSize*dstStride) 偏移处；key and value 是同一份数据（k==v），bmm2 的 V 段 = L1 nope 区（512 宽），
    // 无独立 value GM 读取）
    __aicore__ inline void CopyKeyAndRopeTile(const LocalTensor<KV_T> &dstTensor, FlashMlaWithKvcacheRunInfoX &runInfo)
    {
        uint32_t dstStride = (runInfo.actSingleLoopS2Size + 15) >> 4 << 4;
        FaL1Tensor<KV_T, L1Format::NZ> l1Tensor{.tensor = dstTensor, .rowCount = dstStride};

        GmKvCoord gmCoord{.bIdx = runInfo.bIdx,
                          .n2Idx = runInfo.n2Idx,
                          .s2Idx = runInfo.s2Idx,
                          .dIdx = 0,
                          .s2DealSize = runInfo.actSingleLoopS2Size,
                          .dDealSize = (uint32_t)(constInfo.dSize + constInfo.dSizeRope)};
        copyKvGmToL1(l1Tensor, kCacheGm, gmCoord);
    }

    // 提前一轮发起 KV 搬运（只发 MTE2、不等落地），让 MTE2 与 MAC 流水重叠
    __aicore__ inline void IterateBmm1Load(FlashMlaWithKvcacheRunInfoX &runInfo)
    {
        uint32_t kvpL1BufId = runInfo.loop % L1_KVP_BUFCNT;
        LocalTensor<KV_T> kvpL1Tensor = l1KvpBuffers_[kvpL1BufId * L1_KVP_BUF_BYTES].template ReinterpretCast<KV_T>();
        Mutex::Lock<PIPE_MTE2>(KV_L1_BUFFER_ID0 + kvpL1BufId);
        CopyKeyAndRopeTile(kvpL1Tensor, runInfo);
        Mutex::Unlock<PIPE_MTE2>(KV_L1_BUFFER_ID0 + kvpL1BufId);
    }

    // 仅用于 kernel 启动首任务（loop==0）：Q 的 MTE2 拷贝提前与 KV 同拍发出，压缩启动期。
    // 中途换 Q 块时不可提前（Q 单槽 L1，前块 MTE1 锁未释时会写冲突），仍走 IterateBmm1 原路径。
    __aicore__ inline void IterateQPreload(FlashMlaWithKvcacheRunInfoX &runInfo)
    {
        LocalTensor<Q_T> qL1Tensor = l1QBuffers_.template ReinterpretCast<Q_T>();
        Mutex::Lock<PIPE_MTE2>(Q_L1_BUFFER_ID0);
        CopyQueryAndRopeTile(qL1Tensor, runInfo);
        Mutex::Unlock<PIPE_MTE2>(Q_L1_BUFFER_ID0);
    }

    __aicore__ inline void FixpipeMm1(const LocalTensor<T> &dstTensor, const LocalTensor<T> &l0C,
                                      FlashMlaWithKvcacheRunInfoX &runInfo)
    {
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        // L0C上的bmm1结果矩阵N方向的size大小, 使能NZ2ND, nSize*sizeof(T) 必须是32B的倍数
        fixpipeParams.nSize = (runInfo.actSingleLoopS2Size + 7) >> 3 << 3;
        // 有效数据不足16行，只需输出部分行即可;L0C上的bmm1结果矩阵M方向的size大小必须是偶数
        fixpipeParams.mSize = (runInfo.actMSize + 1) >> 1 << 1;
        // L0C上matmul结果相邻连续数据片断间隔（前面一个数据块的头与后面数据块的头的间隔），单位为16 *sizeof(T)
        // 源NZ矩阵中相邻Z排布的起始地址偏移
        fixpipeParams.srcStride = (fixpipeParams.mSize + 15) >> 4 << 4;
        fixpipeParams.dstStride = 128U; // mmResUb上两行之间的间隔，单位：element
        fixpipeParams.dualDstCtl = 1; // 双目标模式，按M维度拆分， M / 2 * N写入每个UB，M必须为2的倍数
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;

        Fixpipe<T, T, FIXPIPE_ROW_MAJOR_UB>(dstTensor, l0C, fixpipeParams);
    }

    // ================== 特化 LoadData (310 分支 LoadData2DParamsV2, 常量折叠) ==================
    // mm1 L0A: MK, isLeftTranspose=false, realM=0; kSplit=tileK, mSplit=actM
    __aicore__ inline void Mm1LoadL0A(LocalTensor<Q_T> &l0a, const LocalTensor<Q_T> &l1, uint64_t offset,
                                      uint32_t tileK, uint32_t actM)
    {
        LoadData2DParamsV2 p;
        p.mStartPosition = 0;
        p.kStartPosition = 0;
        p.ifTranspose = false;
        p.mStep = ((actM + 15) >> 4 << 4) >> 4;
        p.kStep = ((tileK + 15) >> 4 << 4) >> 4;
        p.srcStride = p.mStep;
        p.dstStride = p.mStep;
        LoadData(l0a, l1[offset], p);
    }
    // mm1 L0B: KN, isRightTranspose=true → ifTranspose=false; kSplit=tileK, nSplit=actS2
    __aicore__ inline void Mm1LoadL0B(LocalTensor<KV_T> &l0b, const LocalTensor<KV_T> &l1, uint64_t offset,
                                      uint32_t tileK, uint32_t actS2)
    {
        LoadData2DParamsV2 p;
        p.mStartPosition = 0;
        p.kStartPosition = 0;
        p.ifTranspose = false;
        p.mStep = ((actS2 + 15) >> 4 << 4) >> 4;
        p.kStep = ((tileK + 15) >> 4 << 4) >> 4;
        p.srcStride = p.mStep;
        p.dstStride = p.mStep;
        LoadData(l0b, l1[offset], p);
    }
    // mm2 L0A: MK, isLeftTranspose=false, realM≠0 → mStep 用 realM; kSplit=actS2, mSplit=64
    __aicore__ inline void Mm2LoadL0A(LocalTensor<KV_T> &l0a, const LocalTensor<KV_T> &l1, uint32_t actS2,
                                      uint32_t realM)
    {
        LoadData2DParamsV2 p;
        p.mStartPosition = 0;
        p.kStartPosition = 0;
        p.ifTranspose = false;
        p.mStep = ((mBaseSize + 15) >> 4 << 4) >> 4;
        p.kStep = ((actS2 + 15) >> 4 << 4) >> 4;
        p.srcStride = p.mStep;
        p.mStep = ((realM + 15) >> 4 << 4) >> 4;
        p.dstStride = p.mStep;
        LoadData(l0a, l1[0], p);
    }
    // mm2 L0B: KN, isRightTranspose=false → ifTranspose=true; kSplit=actS2, nSplit=128(常量)
    __aicore__ inline void Mm2LoadL0B(LocalTensor<KV_T> &l0b, const LocalTensor<KV_T> &l1, uint64_t offset,
                                      uint32_t actS2)
    {
        constexpr uint32_t nSplit = 128;
        LoadData2DParamsV2 p;
        p.mStartPosition = 0;
        p.kStartPosition = 0;
        p.ifTranspose = true;
        p.mStep = ((actS2 + 15) >> 4 << 4) >> 4;
        p.kStep = ((nSplit + 15) >> 4 << 4) >> 4;
        p.srcStride = p.mStep;
        p.dstStride = (nSplit + 15) >> 4;
        LoadData(l0b, l1[offset], p);
    }

    // ================== 特化 MatmulK（mm1: Q·K^T, K=576 固定, 96+112+128+112+128, M=actM, N=actS2）==================
    // 首个 Q96 分块常驻，其余分块交替使用 Q112/P 与 Q128 临时槽。
    // A/B 共用 B 槽 Mutex；Q112/P 最后一次 QK 读取位于倒数第二条 Mmad。
    __aicore__ inline void Mm1MatmulK(const LocalTensor<Q_T> &qL1, const LocalTensor<KV_T> &kvL1,
                                      const LocalTensor<MM_T> &c, uint32_t actM, uint32_t actS2, bool loadQueryCache)
    {
        const uint32_t am = (actM + 15U) / 16U * 16U;
        const uint32_t an = (actS2 + 15U) / 16U * 16U;
        uint32_t koff = 0U;
#pragma unroll
        for (uint32_t k = 0; k < 5U; ++k) {
            const uint32_t tk = k == 0U ? 96U : ((k & 1U) ? 112U : 128U);
            const uint32_t ao = k == 0U ? 0U : ((k & 1U) ? 42U * 1024U : 18U * 1024U);
            const uint32_t bid = L0B_BUFFER_ID0 + l0bBufId_;
            auto a = l0ABuffers_[ao].template ReinterpretCast<Q_T>();
            auto b = l0BBuffers_[l0bBufId_ * BUFFER_SIZE_BYTE_32K].template ReinterpretCast<KV_T>();
            Mutex::Lock<PIPE_MTE1>(bid);
            if (k != 0U || loadQueryCache) {
                Mm1LoadL0A(a, qL1, koff * am, tk, actM);
            }
            Mm1LoadL0B(b, kvL1, koff * an, tk, actS2);
            Mutex::Unlock<PIPE_MTE1>(bid);
            Mutex::Lock<PIPE_M>(bid);
            MmadParams mp;
            mp.m = actM == 1U ? 16U : actM;
            mp.n = actS2;
            mp.k = tk;
            mp.cmatrixInitVal = k == 0U;
            mp.cmatrixSource = false;
            mp.unitFlag = 0;
            Mmad(c, a, b, mp);
#if (__CCE_AICORE__ != 310) && (!(defined __DAV_310R6__))
            // 与公共 MatmulKPP 一致：小 M×N 时在释放 L0 槽前同步 M 管线。
            if ((mp.m / 16U) * (mp.n / 16U) < 10U) {
                AscendC::PipeBarrier<PIPE_M>();
            }
#endif
            Mutex::Unlock<PIPE_M>(bid);
            l0bBufId_ ^= 1U;
            koff += tk;
        }
        lastMatmulWasPv_ = false;
    }

    // qPreloaded：本任务 Q 已由启动路径（loop==0 IterateQPreload）提前发起 MTE2，跳过重复拷贝
    __aicore__ inline void IterateBmm1(FlashMlaWithKvcacheRunInfoX &runInfo, bool qPreloaded = false)
    {
        uint32_t mm1ResUbBufId = runInfo.loop % UB_MM1_RES_BUFCNT;
        uint32_t kvpL1BufId = runInfo.loop % L1_KVP_BUFCNT;
        uint32_t c1v1CrossCoreSyncIdx = CROSSCORE_BMM1_0 + mm1ResUbBufId;
        LocalTensor<MM_T> mm1ResUbTensor =
            ubMm1ResBuffers_[mm1ResUbBufId * UB_MM1_RES_BUF_BYTES].template ReinterpretCast<MM_T>();

        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(c1v1CrossCoreSyncIdx);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(c1v1CrossCoreSyncIdx + AIV_SYNC_OFFSET);

        LocalTensor<Q_T> qL1Tensor = l1QBuffers_.template ReinterpretCast<Q_T>();
        if (unlikely(runInfo.isFirstS2Loop)) {
            if (!qPreloaded) {
                Mutex::Lock<PIPE_MTE2>(Q_L1_BUFFER_ID0);
                CopyQueryAndRopeTile(qL1Tensor, runInfo);
                Mutex::Unlock<PIPE_MTE2>(Q_L1_BUFFER_ID0);
            }
            Mutex::Lock<PIPE_MTE1>(Q_L1_BUFFER_ID0);
        }

        LocalTensor<KV_T> kvpL1Tensor = l1KvpBuffers_[kvpL1BufId * L1_KVP_BUF_BYTES].template ReinterpretCast<KV_T>();
        Mutex::Lock<PIPE_MTE1>(KV_L1_BUFFER_ID0 + kvpL1BufId);
        {
            Mutex::Lock<PIPE_M>(L0C_BUFFER_ID0 + l0cBufId_);
            LocalTensor<MM_T> l0CSubTensor = l0CBuffers_[l0cBufId_ * L0C_BUF_BYTES].template ReinterpretCast<MM_T>();

            Mm1MatmulK(qL1Tensor, kvpL1Tensor, l0CSubTensor, (uint32_t)runInfo.actMSize,
                       (uint32_t)runInfo.actSingleLoopS2Size, runInfo.isFirstS2Loop);

            Mutex::Unlock<PIPE_M>(L0C_BUFFER_ID0 + l0cBufId_);
            Mutex::Lock<PIPE_FIX>(L0C_BUFFER_ID0 + l0cBufId_);
            FixpipeMm1(mm1ResUbTensor, l0CSubTensor, runInfo);
            Mutex::Unlock<PIPE_FIX>(L0C_BUFFER_ID0 + l0cBufId_);
            l0cBufId_ = (l0cBufId_ + 1) % L0C_BUFCNT;
        }

        if (unlikely(runInfo.isLastS2Loop)) {
            Mutex::Unlock<PIPE_MTE1>(Q_L1_BUFFER_ID0);
        }

        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(c1v1CrossCoreSyncIdx);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(c1v1CrossCoreSyncIdx + AIV_SYNC_OFFSET);
    }

    template <typename DST_TENSOR_T>
    __aicore__ inline void FixpipeMm2PartialN(const DST_TENSOR_T &dstTensor, const LocalTensor<T> &l0C, uint32_t realN,
                                              FlashMlaWithKvcacheRunInfoX &runInfo)
    {
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams; // L0C→UB;FixpipeParamsM300:L0C→UB
        fixpipeParams.nSize = (realN + 7) >> 3 << 3;
        fixpipeParams.mSize = (runInfo.actMSize + 1) >> 1 << 1;
        fixpipeParams.srcStride = (fixpipeParams.mSize + 15) >> 4 << 4;
        fixpipeParams.dstStride = 256U;
        fixpipeParams.dualDstCtl = 1;
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;
        Fixpipe<T, T, FIXPIPE_ROW_MAJOR_UB>(dstTensor, l0C, fixpipeParams);
    }

    __aicore__ inline void IterateBmm2(FlashMlaWithKvcacheRunInfoX &runInfo)
    {
        uint32_t kvslot = runInfo.loop % L1_KVP_BUFCNT;
        auto kv = l1KvpBuffers_[kvslot * L1_KVP_BUF_BYTES].template ReinterpretCast<KV_T>();
        auto a = l0ABuffers_[42U * 1024U].template ReinterpretCast<KV_T>();
        uint32_t am = (runInfo.actMSize + 15U) / 16U * 16U;
        uint32_t an = (runInfo.actSingleLoopS2Size + 15U) / 16U * 16U;
        uint32_t kl = (runInfo.actSingleLoopS2Size + 63U) / 64U;
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(CROSSCORE_L1P_0 + kvslot);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(CROSSCORE_L1P_0 + kvslot + AIV_SYNC_OFFSET);
        if (lastMatmulWasPv_) {
            l0bBufId_ ^= 1U;
        }
        for (uint32_t n = 0; n < 2U; ++n) {
            uint32_t cid = L0C_BUFFER_ID0 + l0cBufId_;
            uint32_t cflag = CROSSCORE_BMM2_0 + n;
            auto c = l0CBuffers_[l0cBufId_ * L0C_BUF_BYTES].template ReinterpretCast<MM_T>();
            auto ub = ubMm2ResBuffers_[n * UB_MM2_RES_BUF_BYTES].template ReinterpretCast<MM_T>();
            Mutex::Lock<PIPE_M>(cid);
            for (uint32_t k = 0; k < kl; ++k) {
                uint32_t tk = k + 1U < kl ? 64U : runInfo.actSingleLoopS2Size - k * 64U;
                uint32_t bid = L0B_BUFFER_ID0 + l0bBufId_;
                auto b = l0BBuffers_[l0bBufId_ * BUFFER_SIZE_BYTE_32K].template ReinterpretCast<KV_T>();
                Mutex::Lock<PIPE_MTE1>(bid);
                if (n == 0U && k == 0U) {
                    Mm2LoadL0A(a, kv[s2BaseSize * dVBaseSize], runInfo.actSingleLoopS2Size, runInfo.actMSize);
                }
                LoadData2DParamsV2 lp;
                lp.mStartPosition = 0;
                lp.kStartPosition = 0;
                lp.ifTranspose = true;
                lp.mStep = (tk + 15U) / 16U;
                lp.kStep = 16U;
                lp.srcStride = an / 16U;
                lp.dstStride = 16U;
                LoadData(b, kv[n * 256U * an + k * 64U * 16U], lp);
                Mutex::Unlock<PIPE_MTE1>(bid);
                Mutex::Lock<PIPE_M>(bid);
                MmadParams mp;
                mp.m = runInfo.actMSize == 1U ? 16U : runInfo.actMSize;
                mp.n = 256U;
                mp.k = tk;
                mp.cmatrixInitVal = k == 0U;
                mp.cmatrixSource = false;
                mp.unitFlag = 0;
                Mmad(c, a[k * 64U * am], b, mp);
#if (__CCE_AICORE__ != 310) && (!(defined __DAV_310R6__))
                // 与公共 MatmulKPP 一致：小 M×N 时在释放 L0 槽前同步 M 管线。
                if ((mp.m / 16U) * (mp.n / 16U) < 10U) {
                    AscendC::PipeBarrier<PIPE_M>();
                }
#endif
                Mutex::Unlock<PIPE_M>(bid);
                l0bBufId_ ^= 1U;
            }
            Mutex::Unlock<PIPE_M>(cid);
            CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(cflag);
            CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(cflag + AIV_SYNC_OFFSET);
            Mutex::Lock<PIPE_FIX>(cid);
            FixpipeMm2PartialN(ub, c, 256U, runInfo);
            Mutex::Unlock<PIPE_FIX>(cid);
            CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(cflag);
            CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(cflag + AIV_SYNC_OFFSET);
            l0cBufId_ ^= 1U;
        }
        lastMatmulWasPv_ = true;
        Mutex::Unlock<PIPE_MTE1>(KV_L1_BUFFER_ID0 + kvslot);
    }

}; // FlashMlaWithKvcacheNoQuantMlaBlockCube

template <typename FA_T>
class FlashMlaWithKvcacheNoQuantMlaBlockCubeDummy {
public:
    using INPUT_T = typename FA_T::inputType;
    using T = typename FA_T::mmType;
    static constexpr uint32_t mBaseSize = (uint32_t)FA_T::mBaseSize;
    static constexpr uint32_t s2BaseSize = (uint32_t)FA_T::s2BaseSize;
    static constexpr uint32_t dBaseSize = (uint32_t)FA_T::dBaseSize;
    static constexpr uint32_t dVBaseSize = (uint32_t)FA_T::dVBaseSize;
    static constexpr FLASH_MLA_WITH_KVCACHE_LAYOUT LAYOUT = FA_T::qLayout;
    static constexpr bool PAGE_ATTENTION = FA_T::pageAttention;

    using Q_T = INPUT_T;
    using KV_T = INPUT_T;
    using MM_T = T;

    using ConstInfoX = ConstInfo_t<FlashMlaWithKvcacheKernelType::NO_QUANT>;
    template <typename FlashMlaSeqLensToolT>
    __aicore__ inline FlashMlaWithKvcacheNoQuantMlaBlockCubeDummy(ConstInfoX &constInfo,
                                                                  FlashMlaSeqLensToolT &seqLensTool){};
};

} // namespace FlashAttnKernel

#endif // FLASH_MLA_WITH_KVCACHE_BLOCK_CUBE_H_
