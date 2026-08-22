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
 * \file fia_block_cube_nonquant_gqa.h
 * \brief
 */
#ifndef FIA_BLOCK_CUBE_NONQUANT_GQA_H
#define FIA_BLOCK_CUBE_NONQUANT_GQA_H

#include "../array.h"
#include "../axis.h"
#include "../fia_public_define.h"
#include "../vector_common.h"
#include "../memory_copy.h"
#include "kernel_common.h"

/**
 * 【Q2V2KP3 Tiling方案】
 * 1. Q常驻，V全载；
 * 2. 为Q分配2块L1 buffer，每块96KB(256*192); V分配2块L1 bufer，每块64KB(256*128)；K和P共享3块L1 buffer, 每块64KB(192*128 or 128*256)
 * ● 条件：align(Dn, 16) + align(Dr, 16) <= 192 & align(Dn, 16) <= 128
 * ● L1 Buffer
 *     ○ Q：512*(Dn+Dr) <= 512*192 = 192K
 *     ○ V：512*Dn <= 512*128 = 128K
 *     ○ PK：64K*3=192K；三buffer循环复用
 *         ■ P：128*256 = 64K
 *         ■ K：若 Dn+Dr <= 128，则s2切256；否则s2切128；
 * ● L1切分
 *     ○ mm1：Q*K
 *         ■ A.m=512，A.k全载
 *         ■ B.k全载；若k <= 128，B.n=256，否则B.n=128；
 *     ○ mm2：P*V
 *         ■ A.m=128，A.k=256
 *         ■ B.n=128，B.k全载
 * ● L0A/B/C Buffer
 *     ○ 两个MM使用相同的Buffer策略
 *     ○ L0A：32K*2=64K
 *     ○ L0B：32K*2=64K
 *     ○ L0C：64K*2=128K
 *         ■ L0C使能UnitFlag
 * ● L0切分：
 *     ○ mm1
 *         ■ m=128 n=128 k=align(ceil(k, ceil(k, 128)), 16)
 *     ○ mm2
 *         ■ m=128 n=128 k=128
 *
 * 【mm1, mm2计算过程】
 * # L1空间
 * __CBuffer__ L1Q[2] = {L1_0_0, L1_0_1}    # Q: 2*96K
 * __CBuffer__ L1V[2] = {L1_0_2, L1_0_3}    # V: 2*64K
 * __CBuffer__ L1KP[3] = {L1_1_0, L1_1_1, L1_1_2}   # PK: 3*64K
 *
 * # MTE2到MTE1的同步ID
 * __EVENT_ID__ mte21QIds[2] = {EVENTID_0, EVENTID_1}
 * __EVENT_ID__ mte21VIds[2] = {EVENTID_2, EVENTID_3}
 * __EVENT_ID__ mte21KPIds[3] = {EVENTID_4, EVENTID_5, EVENTID_6}
 *
 * # MTE1到MTE2的同步ID
 * __EVENT_ID__ mte12QIds[2] = {EVENTID_0, EVENTID_1}
 * __EVENT_ID__ mte12VIds[2] = {EVENTID_2, EVENTID_3}
 * __EVENT_ID__ mte12KPIds[3] = {EVENTID_4, EVENTID_5, EVENTID_6}
 *
 * # L0C空间
 * __L0CBuffer__ L0C[2] = {L0C_1, L0C_2}    # 2*64K
 *
 * # M到FixP的同步ID
 * __EVENT_ID__ mfixpIds[2] = {EVENTID_0, EVENTID_1}
 *
 * # FixP到M的同步ID
 * __EVENT_ID__ fixpmIds[2] = {EVENTID_0, EVENTID_1}
 *
 * iQ, iV, iKP = -1, -1, -1
 *
 * # 非MTE2相关的同步比较简单，伪码中省略
 *
 * for i in B*S1*G*S2.o:
 *     mm1() {
 *         for i.1n in n.l1: # n轴切分：512=128*4 OR 512=256*2
 *             iKP++
 *             wait(mte1->mte2, mte12KPIds[iKP%3])
 *
 *             copyB(dst=L1KP[iKP%3], src=K, size=(n.inner, Dn+Dr))
 *
 *             set(mte2->mte1, mte21KPIds[iKP%3])
 *             wait(mte2->mte1, mte21KPIds[iKP%3])
 *
 *             for i.1m in m.l1: # m轴切分：512=256*2
 *                 if i.1n == 0 && isChanged(b*n2):
 *                     iQ++
 *                     wait(mte1->mte2, mte12QIds[iQ%2])
 *
 *                     copyA(dst=L1Q[iQ%2], src=Q, size=(256, Dn+Dr))
 *
 *                     set(mte2->mte1, mte21QIds[iQ%2])
 *                     wait(mte2->mte1, mte21QIds[iQ%2])
 *
 *                     ka = iQ
 *                 else:
 *                     ka = iQ - (m.l1.ext-i.1m-1)
 *
 *                 for i.0m in m.l0:
 *                     for i.0n in n.l0:
 *                         for i.0k in k.l0:
 *                             loadA(src=L1Q[ka], size=(128, k.inner))
 *                             loadB(src=L1KP[iKP%3], size=(128, k.inner))
 *                             mmad()
 *                         fixp()
 *
 *                 if isLastLoop(i.1n) and willChanged(b*n2):
 *                     set(mte1->mte2, mte12QIds[ka%2])
 *
 *             set(mte1->mte2, mte12KPIds[iKP%3])
 *      }
 *
 *      if i > 1:
 *         mm2() {
 *             for i.1m in m.l1: # m轴切分：512=128*4
 *                 for i.1k in k.l1: # k轴切分：512=256*2
 *                     iKP++
 *                     wait(mte1->mte2, mte12KPIds[iKP%3])
 *
 *                     copyA(dst=L1KP[iKP%3], src=P, size=(128, 256))
 *
 *                     set(mte2->mte1, mte21KPIds[iKP%3])
 *                     wait(mte2->mte1, mte21KPIds[iKP%3])
 *
 *                     if i.1m == 0:
 *                         iV++
 *                         wait(mte1->mte2, mte12VIds[iV%2])
 *
 *                         copyB(dst=L1V[iV%2], src=V, size=(256, Dn))
 *
 *                         set(mte2->mte1, mte21VIds[iV%2])
 *                         wait(mte2->mte1, mte21VIds[iV%2])
 *
 *                         kb=iV
 *                     else:
 *                         kb=iV-(k.l1.ext-i.1k-1)
 *
 *                     for i.0k in k.l0:
 *                         loadA(src=L1KP[iKP%3], size=(128, 128))
 *                         loadB(src=L1V[kb%2], size=(128, 128))
 *                         mmad()
 *
 *                     if isLastLoop(i.1m):
 *                         set(mte1->mte2, mte12VIds[kb%2])
 *
 *                     set(mte1->mte2, mte12KPIds[iKP%3])
 *                 fixp()
 *          }
 *
 */

#define DEBUG_MATMUL_GQA 0
#define DEBUG_DISABLE_C1 0
#define DEBUG_DISABLE_C2 0

/**
 *
 * @tparam FIAT FIA算子模板参数，参见FIAType
 */
template <typename FIAT, typename Config=void>
class FiaBlockCubeNonQuantGqa {
    struct DefaultCfg {
        static constexpr bool ENABLE_UNIFLAG = false;
        static constexpr uint32_t PRELOAD_NUM = 2;    // same as fia_kernel_nonquant.h PRELOAD_NUM
        static constexpr bool S2_BASICSIZE_IS_1024 = false;
    };
    using T = float;
    using CFG = typename AscendC::Conditional<IsSameType<Config, void>::value, DefaultCfg, Config>::type;
    using Q_T = typename FIAT::queryType;
    using KV_T = typename FIAT::kvType;

    static constexpr bool PAGE_ATTENTION = FIAT::pageAttention;
    static constexpr FIA_LAYOUT PA_LAYOUT = FIAT::kvLayout;

    static constexpr bool ANTIQUANT = !IsSameType<Q_T, KV_T>::value;
    static constexpr bool QUANT = (IsSameType<Q_T, KV_T>::value && IsSameType<KV_T, int8_t>::value);
    using MM_OUT_T = typename AscendC::Conditional<(ANTIQUANT || QUANT), int32_t, T>::type;

    static constexpr FIA_LAYOUT LAYOUT_T = FIAT::layout;
    static constexpr FIA_LAYOUT KV_LAYOUT_T = FIAT::kvLayout;

    static constexpr GmFormat Q_FORMAT = GetQueryGmFormat<LAYOUT_T>();
    static constexpr GmFormat KV_FORMAT = GetKVFormat<KV_LAYOUT_T, PAGE_ATTENTION>();

    struct BufSnapshot {
        uint64_t signature = (uint64_t)-1;
        uint32_t firstBufId;
        uint32_t bufCnt = 0;
    };

    static_assert(IsSameType<Q_T, KV_T>::value, "Only support Q dtype as same as KV dtype");
    static_assert((! PAGE_ATTENTION) || ((PA_LAYOUT == FIA_LAYOUT::BSH) || (PA_LAYOUT == FIA_LAYOUT::BNSD) || (PA_LAYOUT == FIA_LAYOUT::NZ)), "If enable PA, layout only support BSH/BNBD/NZ");

public:
    __aicore__ inline void InitParams(const ConstInfo &constInfo);
    __aicore__ inline void Init(
        __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pseShift,
        __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengths,
        __gm__ uint8_t *deqScale1, __gm__ uint8_t *quantScale1, __gm__ uint8_t *deqScale2, __gm__ uint8_t *quantScale2,
        __gm__ uint8_t *quantOffset2, __gm__ uint8_t *antiquantScale, __gm__ uint8_t *antiquantOffset,
        __gm__ uint8_t *blockTable, __gm__ uint8_t *queryPaddingSize, __gm__ uint8_t *kvPaddingSize,
        __gm__ uint8_t *keyAntiquantScale, __gm__ uint8_t *keyAntiquantOffset, __gm__ uint8_t *valueAntiquantScale,
        __gm__ uint8_t *valueAntiquantOffset, __gm__ uint8_t *keySharedPrefix, __gm__ uint8_t *valueSharedPrefix,
        __gm__ uint8_t *actualSharedPrefixLen, __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope,
        __gm__ uint8_t *keyRopeAntiquantScale, __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse);
    __aicore__ inline void InitMm1GlobalTensor(const GlobalTensor<MM_OUT_T>& mm1ResGm);
    __aicore__ inline void InitMm2GlobalTensor(const GlobalTensor<KV_T>& vec1ResGm, const GlobalTensor<MM_OUT_T>& mm2ResGm);

    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();

    template <CubeFormat OutFormat=CubeFormat::ND, CubeFormat BFormat=CubeFormat::ND>
    __aicore__ inline void ComputeMm1(const RunInfo &info);
    template <CubeFormat OutFormat=CubeFormat::ND, CubeFormat AFormat=CubeFormat::ND>
    __aicore__ inline void ComputeMm2(const RunInfo &info);

private:
    /**
     * @pre alignment必须是2的幂，且s>=0
     */
    template <typename T1>
    static __aicore__ inline constexpr T1 AlignToPowerOfTwo(T1 s, T1 alignment)
    {
        if constexpr (IsSameType<T1, uint64_t>::value || IsSameType<T1, uint32_t>::value || IsSameType<T1, uint16_t>::value || IsSameType<T1, uint8_t>::value) {
            return (s + alignment-1) & (~(alignment-1));
        } else {
            return uint64_t(s + alignment-1) & (~uint64_t(alignment-1));
        }
    }

    template <typename T1>
    static __aicore__ inline constexpr size_t BlockAlign(size_t s)
    {
        if constexpr (IsSameType<T1, int4b_t>::value) {
            return AlignToPowerOfTwo(s, 64);
        } else {
            return AlignToPowerOfTwo(s, ONE_BLK_SIZE / sizeof(T1));
        }
    }

    template <typename T1>
    __aicore__ inline constexpr uint32_t GetC0Num()
    {
        if constexpr (IsSameType<T1, int4b_t>::value) {
            return 64;
        } else {
            return ONE_BLK_SIZE / sizeof(T1);
        }
    }

    __aicore__ inline void InitKeyGm(uint32_t bIdx);
    __aicore__ inline void InitValueGm(uint32_t bIdx);
    __aicore__ inline void UpdateKey(uint32_t bIdx);
    __aicore__ inline void UpdateValue(uint32_t bIdx);

    __aicore__ inline void CopyQToL1(uint32_t dstBufId, const RunInfo &info, uint32_t subMStart, uint32_t subMSize);
    __aicore__ inline void CopyKToL1(uint32_t dstBufId, const RunInfo &info, uint32_t subNStart, uint32_t subNSize);
    template <CubeFormat Format>
    __aicore__ inline void CopyPToL1(uint32_t dstBufId, const RunInfo &info, uint32_t gmStride,
                                         uint32_t subMStart, uint32_t subMSize, uint32_t subKStart, uint32_t subKSize);
    __aicore__ inline void CopyVToL1(uint32_t dstBufId, const RunInfo &info, uint32_t subKStart, uint32_t subKSize);

    template <uint32_t M_L1_SIZE, bool FOECE=false>
    __aicore__ inline void ResetLoad3DConfig();

    template <uint32_t M_L1_SPLIT_Size, typename T1>
    __aicore__ inline void LoadAToL0(uint32_t dstBufId, const LocalTensor<T1> &l1Tensor, uint32_t mL1Size,
                                         uint32_t subMStart, uint32_t subMSize, uint32_t subKStart, uint32_t subKSize);
    __aicore__ inline void LoadBTransposeToL0(uint32_t dstBufId, const LocalTensor<KV_T> &l1Tensor, uint32_t nL1Size,
                                         uint32_t subNStart, uint32_t subNSize, uint32_t subKStart, uint32_t subKSize);
    __aicore__ inline void LoadBToL0(uint32_t dstBufId, const LocalTensor<KV_T> &l1Tensor, uint32_t kL1Size,
                                         uint32_t subKStart, uint32_t subKSize, uint32_t subNStart, uint32_t subNSize);
    template <CubeFormat GMFormat>
    __aicore__ inline void FixpipeCToGM(GlobalTensor<MM_OUT_T> &mmResGm, uint32_t cL0BufId,
                                        uint32_t dstStride, uint32_t subMStart, uint32_t subMSize,
                                        uint32_t subNStart, uint32_t subNSize);

private:
    // =================================L1 Buffer=================================
    static constexpr uint32_t L1_Q_SIZE = CFG::S2_BASICSIZE_IS_1024 ? 128 * 192 : 256 * 192;    // 64KB or 96KB
    static constexpr uint32_t L1_V_SIZE = 256 * 128;    // 64KB
    static constexpr uint32_t L1_KP_SIZE = 128 * 256;   // 64KB. K: 128 * 192 or 128 * 256; P: 128 * 256

    static constexpr uint32_t L1_Q_BUFCNT = 2;
    static constexpr uint32_t L1_V_BUFCNT = CFG::S2_BASICSIZE_IS_1024 ? 4 : 2;
    static constexpr uint32_t L1_KP_BUFCNT = CFG::S2_BASICSIZE_IS_1024 ? 2 : 3;

    Array<LocalTensor<Q_T>, L1_Q_BUFCNT, L1_Q_SIZE> qL1Tensor;
    Array<LocalTensor<KV_T>, L1_V_BUFCNT, L1_V_SIZE> vL1Tensor;
    Array<LocalTensor<KV_T>, L1_KP_BUFCNT, L1_KP_SIZE> kpL1Tensor;

    BufSnapshot qL1Snapshot;
    BufSnapshot vL1Snapshot;

    uint32_t qL1BufId = 0;
    uint32_t vL1BufId = 0;
    uint32_t kpL1BufId = 0;

    TBuf<TPosition::A1> L1_Q;
    TBuf<TPosition::A1> L1_V;
    TBuf<TPosition::A1> L1_KP;

    // =================================L0 Buffer=================================
    static constexpr uint32_t L0A_PP_SIZE = 128 * 128;  // 32KB
    static constexpr uint32_t L0B_PP_SIZE = 128 * 128;  // 32KB
    static constexpr uint32_t L0C_PP_SIZE = 128 * 128;  // 64KB

    static constexpr uint32_t L0AB_BUFCNT = 2;
    static constexpr uint32_t L0C_BUFCNT = 2;

    Array<LocalTensor<Q_T>, L0AB_BUFCNT, L0A_PP_SIZE> aL0Tensor;
    Array<LocalTensor<KV_T>, L0AB_BUFCNT, L0B_PP_SIZE> bL0Tensor;
    Array<LocalTensor<T>, L0C_BUFCNT, L0C_PP_SIZE> cL0Tensor;

    uint32_t abL0BufId = 0;
    uint32_t cL0BufId = 0;

    TBuf<TPosition::A2> L0_A;
    TBuf<TPosition::B2> L0_B;
    TBuf<TPosition::CO1> L0_C;

    // =================================Event&Buffer ID===========================
    // mte2 <> mte1
    static constexpr uint32_t Q_EVENT0 = EVENT_ID0;
    static constexpr uint32_t V_EVENT0 = Q_EVENT0 + L1_Q_BUFCNT;
    static constexpr uint32_t KP_EVENT0 = V_EVENT0 + L1_V_BUFCNT;

    // m <> mte1
    static constexpr uint32_t L0AB_EVENT0 = EVENT_ID3;
    static constexpr uint32_t L0AB_EVENT1 = EVENT_ID4;
    static constexpr uint32_t L0_READY_EVENT = EVENT_ID7;

    // fix <> m
    static constexpr uint32_t L0C_EVENT0 = EVENT_ID3;
    static constexpr uint32_t L0C_EVENT1 = EVENT_ID4;

    // =================================params=================================
    FaGmTensor<Q_T, Q_FORMAT> queryGmTensor;
    FaGmTensor<Q_T, Q_FORMAT> queryRopeGmTensor;
    CopyQueryGmToL1<Q_T, Q_FORMAT> copyQueryGmToL1;

    FaGmTensor<KV_T, KV_FORMAT> keyGmTensor;
    FaGmTensor<KV_T, KV_FORMAT> keyRopeGmTensor;
    FaGmTensor<KV_T, KV_FORMAT> valueGmTensor;
    CopyKvGmToL1<KV_T, KV_FORMAT> copyKvGmToL1;

    ConstInfo constInfo{};

    // key和value的TensorList原始地址
    __gm__ uint8_t *keyPtr = nullptr;
    __gm__ uint8_t *valuePtr = nullptr;

    // mm1
    GlobalTensor<Q_T> queryGm;
    GlobalTensor<KV_T> keyGm;
    GlobalTensor<Q_T> qRopeGm;
    GlobalTensor<KV_T> kRopeGm;
    Array<GlobalTensor<MM_OUT_T>, CFG::PRELOAD_NUM> mm1ResGm;

    // mm2
    Array<GlobalTensor<KV_T>, CFG::PRELOAD_NUM> vec1ResGm;
    GlobalTensor<KV_T> valueGm;
    Array<GlobalTensor<MM_OUT_T>, CFG::PRELOAD_NUM> mm2ResGm;

    __gm__ uint8_t *actualSequenceLengthsQ = nullptr;
    GlobalTensor<uint64_t> actualSeqLengthsGmQ;
    GlobalTensor<uint64_t> actualSeqLengthsGm;
    __gm__ uint8_t *actualSeqLengths = nullptr;

    // page attention
    GlobalTensor<int32_t> blockTableGm;

    // Load L1->L0 params
    static constexpr IsResetLoad3dConfig LOAD3DV2_CONFIG = {false, false};
    uint32_t load3DL1SizeCfg = 0;
    LoadData3DParamsV2<Q_T> loadData3DParams;
    LoadData2DParams mm1LoadDataBTransposeToL0Params;
    LoadData2DParams mm2LoadDataBToL0Params;

public:
    // =================================debug=================================
#if DEBUG_MATMUL_GQA
    uint64_t qMemSize = 0;
    uint64_t kMemSize = 0;
    uint64_t pMemSize = 0;
    uint64_t vMemSize = 0;
#endif
};

template <typename FIAT, typename Config=void>
class FiaBlockCubeNonQuantGqaDummy {
    using T = float;
    using Q_T = typename FIAT::queryType;
    using KV_T = typename FIAT::kvType;

    static constexpr bool ANTIQUANT = !IsSameType<Q_T, KV_T>::value;
    static constexpr bool QUANT = (IsSameType<Q_T, KV_T>::value && IsSameType<KV_T, int8_t>::value);
    using MM_OUT_T = typename AscendC::Conditional<(ANTIQUANT || QUANT), int32_t, T>::type;

public:
    __aicore__ inline void InitParams(const ConstInfo &constInfo);
    __aicore__ inline void Init(
        __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pseShift,
        __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengths,
        __gm__ uint8_t *deqScale1, __gm__ uint8_t *quantScale1, __gm__ uint8_t *deqScale2, __gm__ uint8_t *quantScale2,
        __gm__ uint8_t *quantOffset2, __gm__ uint8_t *antiquantScale, __gm__ uint8_t *antiquantOffset,
        __gm__ uint8_t *blockTable, __gm__ uint8_t *queryPaddingSize, __gm__ uint8_t *kvPaddingSize,
        __gm__ uint8_t *keyAntiquantScale, __gm__ uint8_t *keyAntiquantOffset, __gm__ uint8_t *valueAntiquantScale,
        __gm__ uint8_t *valueAntiquantOffset, __gm__ uint8_t *keySharedPrefix, __gm__ uint8_t *valueSharedPrefix,
        __gm__ uint8_t *actualSharedPrefixLen, __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope,
        __gm__ uint8_t *keyRopeAntiquantScale, __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse);
    __aicore__ inline void InitMm1GlobalTensor(const GlobalTensor<MM_OUT_T>& mm1ResGm);
    __aicore__ inline void InitMm2GlobalTensor(const GlobalTensor<KV_T>& vec1ResGm, const GlobalTensor<MM_OUT_T>& mm2ResGm);

    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();

    template <CubeFormat OutFormat=CubeFormat::ND, CubeFormat BFormat=CubeFormat::ND>
        __aicore__ inline void ComputeMm1(const RunInfo &info);
    template <CubeFormat OutFormat=CubeFormat::ND, CubeFormat AFormat=CubeFormat::ND>
        __aicore__ inline void ComputeMm2(const RunInfo &info);
};

template <typename FIAT, typename Config>
__aicore__ inline void FiaBlockCubeNonQuantGqa<FIAT, Config>::InitParams(const ConstInfo &constInfo)
{
    this->constInfo = constInfo;

    loadData3DParams.l1W = GetC0Num<Q_T>(); // Win=M0
    loadData3DParams.padList[0] = 0;
    loadData3DParams.padList[1] = 0;
    loadData3DParams.padList[2] = 0;
    loadData3DParams.padList[3] = 255; // 尾部数据不影响滑窗的结果

    loadData3DParams.mStartPt = 0;
    loadData3DParams.kStartPt = 0;
    loadData3DParams.strideW = 1;
    loadData3DParams.strideH = 1;
    loadData3DParams.filterW = 1;
    loadData3DParams.filterSizeW = 0;
    loadData3DParams.filterH = 1;
    loadData3DParams.filterSizeH = 0;
    loadData3DParams.dilationFilterW = 1;
    loadData3DParams.dilationFilterH = 1;
    loadData3DParams.enTranspose = 0;
    loadData3DParams.fMatrixCtrl = 0;

    mm1LoadDataBTransposeToL0Params.startIndex = 0;
    mm1LoadDataBTransposeToL0Params.srcStride = 1;
    mm1LoadDataBTransposeToL0Params.dstGap = 0;
    mm1LoadDataBTransposeToL0Params.ifTranspose = false;

    mm2LoadDataBToL0Params.startIndex = 0;
    mm2LoadDataBToL0Params.repeatTimes = constInfo.headDim / GetC0Num<Q_T>(); // 列（N轴）方向上的分形数
    mm2LoadDataBToL0Params.dstGap = 0;
    mm2LoadDataBToL0Params.ifTranspose = true; // 将小Z格式转置为小N
}

template <typename FIAT, typename Config>
__aicore__ inline void FiaBlockCubeNonQuantGqa<FIAT, Config>::InitKeyGm(uint32_t bIdx)
{
    ListTensorDesc keyListTensorDesc((__gm__ void *)keyPtr);
    __gm__ uint8_t *key_ = (__gm__ uint8_t *)keyListTensorDesc.GetDataPtr<__gm__ uint8_t>(bIdx);

    keyGm.SetGlobalBuffer((__gm__ KV_T *)key_);
    if (constInfo.l2CacheOffFlag) {
        // 关闭K、V的L2 Cache
#ifndef ASCENDC_OOM
        keyGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
#endif
    }
}

template <typename FIAT, typename Config>
__aicore__ inline void FiaBlockCubeNonQuantGqa<FIAT, Config>::InitValueGm(uint32_t bIdx)
{
    ListTensorDesc valueListTensorDesc((__gm__ void *)valuePtr);
    __gm__ uint8_t *value_ = (__gm__ uint8_t *)valueListTensorDesc.GetDataPtr<__gm__ uint8_t>(bIdx);

    valueGm.SetGlobalBuffer((__gm__ KV_T *)value_);
    if (constInfo.l2CacheOffFlag) {
        // 关闭K、V的L2 Cache
#ifndef ASCENDC_OOM
        valueGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
#endif
    }
}

template <typename FIAT, typename Config>
__aicore__ inline void FiaBlockCubeNonQuantGqa<FIAT, Config>::UpdateKey(uint32_t bIdx)
{
    static_assert(GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_BNSD,
                  "when only KV TensorList, support update KeyGm");
    InitKeyGm(bIdx);
    uint64_t s2Size = fa_base_kernel::SeqLenFromTensorList<KV_LAYOUT_T>(keyPtr, bIdx);
    keyGmTensor.gmTensor = keyGm;
    keyGmTensor.offsetCalculator.Init(0, constInfo.kvHeadNum, s2Size, constInfo.headDim, actualSeqLengthsGm, constInfo.actualLenDims);
}

template <typename FIAT, typename Config>
__aicore__ inline void FiaBlockCubeNonQuantGqa<FIAT, Config>::UpdateValue(uint32_t bIdx)
{
    static_assert(GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_BNSD,
                  "when only KV TensorList, support update ValueGm");
    InitValueGm(bIdx);
    uint64_t s2Size = fa_base_kernel::SeqLenFromTensorList<KV_LAYOUT_T>(valuePtr, bIdx);
    valueGmTensor.gmTensor = valueGm;
    valueGmTensor.offsetCalculator.Init(0, constInfo.kvHeadNum, s2Size, constInfo.headDim, actualSeqLengthsGm, constInfo.actualLenDims);
}

template <typename FIAT, typename Config>
__aicore__ inline void FiaBlockCubeNonQuantGqa<FIAT, Config>::Init(
            __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pseShift,
            __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengths,
            __gm__ uint8_t *deqScale1, __gm__ uint8_t *quantScale1, __gm__ uint8_t *deqScale2, __gm__ uint8_t *quantScale2,
            __gm__ uint8_t *quantOffset2, __gm__ uint8_t *antiquantScale, __gm__ uint8_t *antiquantOffset,
            __gm__ uint8_t *blockTable, __gm__ uint8_t *queryPaddingSize, __gm__ uint8_t *kvPaddingSize,
            __gm__ uint8_t *keyAntiquantScale, __gm__ uint8_t *keyAntiquantOffset, __gm__ uint8_t *valueAntiquantScale,
            __gm__ uint8_t *valueAntiquantOffset, __gm__ uint8_t *keySharedPrefix, __gm__ uint8_t *valueSharedPrefix,
            __gm__ uint8_t *actualSharedPrefixLen, __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope,
            __gm__ uint8_t *keyRopeAntiquantScale, __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse)
{
    // 先初始化基础参数
    if (constInfo.actualLenQDims != 0) {
        actualSeqLengthsGmQ.SetGlobalBuffer((__gm__ uint64_t *)actualSeqLengthsQ, constInfo.actualLenQDims);
        this->actualSequenceLengthsQ = actualSeqLengthsQ;
    }
    if (constInfo.actualLenDims != 0) {
        actualSeqLengthsGm.SetGlobalBuffer((__gm__ uint64_t *)actualSeqLengths, constInfo.actualLenDims);
        this->actualSeqLengths = actualSeqLengths;
    }
    if constexpr (PAGE_ATTENTION) {
        blockTableGm.SetGlobalBuffer((__gm__ int32_t *)blockTable);
    }

    // 再初始化复杂参数
    uint32_t qkTensorD = constInfo.ropeSplitMode ? constInfo.headDim : (constInfo.headDim + constInfo.headDimRope);
    queryGm.SetGlobalBuffer((__gm__ Q_T *)query);
    {
        queryGmTensor.gmTensor = queryGm;
        if constexpr (GmLayoutParams<Q_FORMAT>::CATEGORY == FormatCategory::GM_Q_OUT_BNGSD) {
            queryGmTensor.offsetCalculator.Init(constInfo.batchSize, constInfo.kvHeadNum, constInfo.gSize,
                                                constInfo.qSeqSize, qkTensorD, actualSeqLengthsGmQ,
                                                constInfo.actualLenQDims);
        } else if constexpr (GmLayoutParams<Q_FORMAT>::CATEGORY == FormatCategory::GM_Q_OUT_TND) {
            queryGmTensor.offsetCalculator.Init(constInfo.kvHeadNum, constInfo.gSize, qkTensorD, actualSeqLengthsGmQ,
                                                constInfo.actualLenQDims);
        }
    }

    if (constInfo.ropeSplitMode) {
        // query rope
        qRopeGm.SetGlobalBuffer((__gm__ Q_T *)queryRope);
        queryRopeGmTensor.gmTensor = qRopeGm;
        if constexpr (GmLayoutParams<Q_FORMAT>::CATEGORY == FormatCategory::GM_Q_OUT_BNGSD) {
            queryRopeGmTensor.offsetCalculator.Init(constInfo.batchSize, constInfo.kvHeadNum, constInfo.gSize,
                                                    constInfo.qSeqSize, constInfo.headDimRope, actualSeqLengthsGmQ,
                                                    constInfo.actualLenQDims);
        } else if constexpr (GmLayoutParams<Q_FORMAT>::CATEGORY == FormatCategory::GM_Q_OUT_TND) {
            queryRopeGmTensor.offsetCalculator.Init(constInfo.kvHeadNum, constInfo.gSize, constInfo.headDimRope,
                                                    actualSeqLengthsGmQ, constInfo.actualLenQDims);
        }
        // key rope
        kRopeGm.SetGlobalBuffer((__gm__ KV_T *)keyRope);
        keyRopeGmTensor.gmTensor = kRopeGm;
        if constexpr (PAGE_ATTENTION) {
            if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_PA_BNBD) {
                keyRopeGmTensor.offsetCalculator.Init(constInfo.kvHeadNum, constInfo.kvCacheBlockSize,
                                                      constInfo.headDimRope, blockTableGm,
                                                      constInfo.maxBlockNumPerBatch);
            } else if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_PA_NZ) {
                uint32_t d0 = 32 / sizeof(KV_T);
                uint32_t d1 = constInfo.headDimRope / d0;
                keyRopeGmTensor.offsetCalculator.Init(constInfo.kvHeadNum, constInfo.kvCacheBlockSize, d1, d0,
                                                      blockTableGm, constInfo.maxBlockNumPerBatch);
            }
        } else {
            if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_BNSD) {
                keyRopeGmTensor.offsetCalculator.Init(constInfo.batchSize, constInfo.kvHeadNum, constInfo.kvSeqSize,
                                                      constInfo.headDimRope, actualSeqLengthsGm, constInfo.actualLenDims);
            } else if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_TND) {
                keyRopeGmTensor.offsetCalculator.Init(constInfo.kvHeadNum, constInfo.headDimRope, actualSeqLengthsGm,
                                                      constInfo.actualLenDims);
            }
        }
    }

    keyPtr = key;
    valuePtr = value;
    // batch连续时,只需要初始化一次;不连续时,需要在使用时根据batchIdx初始化
    if (constInfo.batchContinuous) {
        InitKeyGm(0);
        {
            keyGmTensor.gmTensor = keyGm;
            if constexpr (PAGE_ATTENTION) {
                if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_PA_BNBD) {
                    keyGmTensor.offsetCalculator.Init(constInfo.kvHeadNum, constInfo.kvCacheBlockSize, qkTensorD,
                                                      blockTableGm, constInfo.maxBlockNumPerBatch);
                } else if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_PA_NZ) {
                    uint32_t d0 = 32 / sizeof(KV_T);
                    uint32_t d1 = qkTensorD / d0;
                    keyGmTensor.offsetCalculator.Init(constInfo.kvHeadNum, constInfo.kvCacheBlockSize, d1,
                                                      d0, blockTableGm, constInfo.maxBlockNumPerBatch);
                }
            } else {
                if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_BNSD) {
                    keyGmTensor.offsetCalculator.Init(constInfo.batchSize, constInfo.kvHeadNum, constInfo.kvSeqSize,
                                                      qkTensorD, actualSeqLengthsGm, constInfo.actualLenDims);
                } else if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_TND) {
                    keyGmTensor.offsetCalculator.Init(constInfo.kvHeadNum, qkTensorD,
                                                      actualSeqLengthsGm, constInfo.actualLenDims);
                }
            }
        }

        InitValueGm(0);
        {
            valueGmTensor.gmTensor = valueGm;
            if constexpr (PAGE_ATTENTION) {
                if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_PA_BNBD) {
                    valueGmTensor.offsetCalculator.Init(constInfo.kvHeadNum, constInfo.kvCacheBlockSize,
                                                        constInfo.headDim, blockTableGm, constInfo.maxBlockNumPerBatch);
                } else if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_PA_NZ) {
                    uint32_t d0 = 32 / sizeof(KV_T);
                    uint32_t d1 = constInfo.headDim / d0;
                    valueGmTensor.offsetCalculator.Init(constInfo.kvHeadNum, constInfo.kvCacheBlockSize, d1,
                                                        d0, blockTableGm, constInfo.maxBlockNumPerBatch);
                }
            } else {
                if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_BNSD) {
                    valueGmTensor.offsetCalculator.Init(constInfo.batchSize, constInfo.kvHeadNum, constInfo.kvSeqSize,
                                                        constInfo.headDim, actualSeqLengthsGm, constInfo.actualLenDims);
                } else if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_TND) {
                    valueGmTensor.offsetCalculator.Init(constInfo.kvHeadNum, constInfo.headDim,
                                                        actualSeqLengthsGm, constInfo.actualLenDims);
                }
            }
        }
    }
}

template <typename FIAT, typename Config>
__aicore__ inline void FiaBlockCubeNonQuantGqa<FIAT, Config>::InitMm1GlobalTensor(const GlobalTensor<MM_OUT_T> &mm1ResGm)
{
    // mm1
    this->mm1ResGm.Init(mm1ResGm, constInfo.mmResUbSize);
}

template <typename FIAT, typename Config>
__aicore__ inline void FiaBlockCubeNonQuantGqa<FIAT, Config>::InitMm2GlobalTensor(const GlobalTensor<KV_T> &vec1ResGm, const GlobalTensor<MM_OUT_T> &mm2ResGm)
{
    // mm2
    this->vec1ResGm.Init(vec1ResGm, constInfo.vec1ResUbSize);
    this->mm2ResGm.Init(mm2ResGm, constInfo.bmm2ResUbSize);
}

template <typename FIAT, typename Config>
__aicore__ inline void FiaBlockCubeNonQuantGqa<FIAT, Config>::InitBuffers(TPipe *pipe)
{
    // L1 Q
    pipe->InitBuffer(L1_Q, L1_Q_SIZE * L1_Q_BUFCNT * sizeof(Q_T));
    qL1Tensor.Init(L1_Q.Get<Q_T>());

    // L1 V
    pipe->InitBuffer(L1_V, L1_V_SIZE * L1_V_BUFCNT * sizeof(KV_T));
    vL1Tensor.Init(L1_V.Get<KV_T>());

    // L1 KP
    pipe->InitBuffer(L1_KP, L1_KP_SIZE * L1_KP_BUFCNT * sizeof(KV_T));
    kpL1Tensor.Init(L1_KP.Get<KV_T>());

    // L0A
    pipe->InitBuffer(L0_A, L0A_PP_SIZE * 2 * sizeof(Q_T));
    aL0Tensor.Init(L0_A.Get<Q_T>());

    // L0B
    pipe->InitBuffer(L0_B, L0B_PP_SIZE * 2 * sizeof(KV_T));
    bL0Tensor.Init(L0_B.Get<KV_T>());

    // L0C
    pipe->InitBuffer(L0_C, L0C_PP_SIZE * 2 * sizeof(MM_OUT_T));
    cL0Tensor.Init(L0_C.Get<T>());
}

template <typename FIAT, typename Config>
__aicore__ inline void FiaBlockCubeNonQuantGqa<FIAT, Config>::AllocEventID()
{
    for (uint32_t i = 0; i < L1_KP_BUFCNT; ++i) {
        SetFlag<HardEvent::MTE1_MTE2>(KP_EVENT0 + i);
    }

    for (uint32_t i = 0; i < L1_V_BUFCNT; ++i) {
        SetFlag<HardEvent::MTE1_MTE2>(V_EVENT0 + i);
    }

    SetFlag<HardEvent::M_MTE1>(L0AB_EVENT0);
    SetFlag<HardEvent::M_MTE1>(L0AB_EVENT1);

    SetFlag<HardEvent::FIX_M>(L0C_EVENT0);
    SetFlag<HardEvent::FIX_M>(L0C_EVENT1);
}

template <typename FIAT, typename Config>
__aicore__ inline void FiaBlockCubeNonQuantGqa<FIAT, Config>::FreeEventID()
{
    for (uint32_t i = 0; i < L1_KP_BUFCNT; ++i) {
        WaitFlag<HardEvent::MTE1_MTE2>(KP_EVENT0 + i);
    }

    for (uint32_t i = 0; i < L1_V_BUFCNT; ++i) {
        WaitFlag<HardEvent::MTE1_MTE2>(V_EVENT0 + i);
    }

    WaitFlag<HardEvent::M_MTE1>(L0AB_EVENT0);
    WaitFlag<HardEvent::M_MTE1>(L0AB_EVENT1);

    WaitFlag<HardEvent::FIX_M>(L0C_EVENT0);
    WaitFlag<HardEvent::FIX_M>(L0C_EVENT1);
}

template <typename FIAT, typename Config>
__aicore__ inline void FiaBlockCubeNonQuantGqa<FIAT, Config>::CopyQToL1(
            uint32_t dstBufId, const RunInfo &info, uint32_t subMStart, uint32_t subMSize)
{
    auto qL1Tensor = this->qL1Tensor[dstBufId];
    auto subMSizeAlign = AlignToPowerOfTwo<uint32_t>(subMSize, (uint32_t)BLOCK_CUBE); // 目的矩阵z分形的高固定为16，行为C0_SIZE(字节数)。目的矩阵的“高”将对齐为z分形的整数倍，所以将srcN按16对齐即为Z型矩阵相邻行起始地址之间的偏移
    uint32_t nopeDealSize = constInfo.ropeSplitMode ? constInfo.headDim : (constInfo.headDim + constInfo.headDimRope);

    FaL1Tensor<Q_T, L1Format::NZ> dstTensor {
        .tensor = qL1Tensor,
        .rowCount = subMSizeAlign
    };
    GmCoord gmCoord {
        .bIdx = info.bIdx,
        .n2Idx = info.n2Idx,
        .gS1Idx = info.gS1Idx + subMStart,
        .dIdx = 0,
        .gS1DealSize = subMSize,
        .dDealSize = nopeDealSize
    };
    copyQueryGmToL1(dstTensor, queryGmTensor, gmCoord);

    if (constInfo.ropeSplitMode) {
        dstTensor.tensor = qL1Tensor[subMSizeAlign * nopeDealSize];

        gmCoord.dDealSize = (uint32_t)constInfo.headDimRope;

        copyQueryGmToL1(dstTensor, queryRopeGmTensor, gmCoord);
    }

#if DEBUG_MATMUL_GQA
    this->qMemSize += subMSize * (constInfo.headDim + constInfo.headDimRope);
#endif
}

template <typename FIAT, typename Config>
__aicore__ inline void FiaBlockCubeNonQuantGqa<FIAT, Config>::CopyKToL1(
            uint32_t dstBufId, const RunInfo &info, uint32_t subNStart, uint32_t subNSize)
{
    auto kpL1Tensor = this->kpL1Tensor[dstBufId];
    auto subNSizeAlign = AlignToPowerOfTwo(subNSize, (uint32_t)BLOCK_CUBE);
    uint32_t nopeDealSize = constInfo.ropeSplitMode ? constInfo.headDim : (constInfo.headDim + constInfo.headDimRope);

    FaL1Tensor<KV_T, L1Format::NZ> dstTensor {
        .tensor = kpL1Tensor,
        .rowCount = subNSizeAlign
    };
    GmKvCoord gmCoord {
        .bIdx = constInfo.batchContinuous ? info.bIdx : 0,
        .n2Idx = info.n2Idx,
        .s2Idx = info.s2Idx * constInfo.s2BaseSize + subNStart,
        .dIdx = 0,
        .s2DealSize = subNSize,
        .dDealSize = nopeDealSize
    };
    copyKvGmToL1(dstTensor, keyGmTensor, gmCoord);

    if (constInfo.ropeSplitMode) {
        dstTensor.tensor = kpL1Tensor[subNSizeAlign * nopeDealSize];

        gmCoord.dDealSize = (uint32_t)constInfo.headDimRope;

        copyKvGmToL1(dstTensor, keyRopeGmTensor, gmCoord);
    }

#if DEBUG_MATMUL_GQA
    this->kMemSize += subNSize * (constInfo.headDim + constInfo.headDimRope);
#endif
}

template <typename FIAT, typename Config>
template <CubeFormat Format>
__aicore__ inline void FiaBlockCubeNonQuantGqa<FIAT, Config>::CopyPToL1(
            uint32_t dstBufId, const RunInfo &info, uint32_t gmStride,
            uint32_t subMStart, uint32_t subMSize, uint32_t subKStart, uint32_t subKSize)
{
    auto dstL1 = this->kpL1Tensor[dstBufId];
    if constexpr (Format == CubeFormat::NZ) {
        auto srcGm = this->vec1ResGm[info.loop % CFG::PRELOAD_NUM][gmStride * subKStart + GetC0Num<KV_T>() * subMStart];
        uint32_t blockElementCnt = 32 / sizeof(T);
        DataCopyParams intriParams;
        intriParams.blockCount = subKSize / blockElementCnt;
        intriParams.blockLen = subMSize;
        intriParams.dstStride = 0;
        intriParams.srcStride = gmStride;
        DataCopy(dstL1, srcGm, intriParams);
    } else {
        auto srcGm = this->vec1ResGm[info.loop % CFG::PRELOAD_NUM][gmStride * subMStart + subKStart];
        auto subMSizeAlign = AlignToPowerOfTwo(subMSize, (uint32_t)BLOCK_CUBE);
        CopySingleMatrixNDToNZ(dstL1, srcGm, subMSize, subKSize, gmStride, subMSizeAlign);
    }

#if DEBUG_MATMUL_GQA
    this->pMemSize += subMSize * subKSize;
#endif
}

template <typename FIAT, typename Config>
__aicore__ inline void FiaBlockCubeNonQuantGqa<FIAT, Config>::CopyVToL1(
            uint32_t dstBufId, const RunInfo &info, uint32_t subKStart, uint32_t subKSize)
{
    auto vL1Tensor = this->vL1Tensor[dstBufId];
    auto subKSizeAlign = AlignToPowerOfTwo(subKSize, (uint32_t)BLOCK_CUBE);

    FaL1Tensor<KV_T, L1Format::NZ> dstTensor {
        .tensor = vL1Tensor,
        .rowCount = subKSizeAlign
    };
    GmKvCoord gmCoord {
        .bIdx = constInfo.batchContinuous ? info.bIdx : 0,
        .n2Idx = info.n2Idx,
        .s2Idx = info.s2Idx * constInfo.s2BaseSize + subKStart,
        .dIdx = 0,
        .s2DealSize = subKSize,
        .dDealSize = (uint32_t)constInfo.headDim
    };
    copyKvGmToL1(dstTensor, valueGmTensor, gmCoord);

#if DEBUG_MATMUL_GQA
    this->vMemSize += this->valueDSize * subKSize;
#endif
}

template <typename FIAT, typename Config>
template <uint32_t M_L1_SIZE, bool FORCE>
__aicore__ inline void FiaBlockCubeNonQuantGqa<FIAT, Config>::ResetLoad3DConfig()
{
    if constexpr (! FORCE) {
        if (this->load3DL1SizeCfg == M_L1_SIZE) {
            return;
        }
    }

    this->load3DL1SizeCfg = M_L1_SIZE;
    constexpr uint8_t padList[] = {0, 0, 0, 255};
    SetFmatrix(M_L1_SIZE / BLOCK_CUBE, GetC0Num<Q_T>(), padList, FmatrixMode::FMATRIX_LEFT);
    SetLoadDataPaddingValue<Q_T>(0);
}

template <typename FIAT, typename Config>
template <uint32_t M_L1_SPLIT_Size, typename T1>
__aicore__ inline void FiaBlockCubeNonQuantGqa<FIAT, Config>::LoadAToL0(
            uint32_t dstBufId, const LocalTensor<T1> &l1Tensor, uint32_t mL1Size,
            uint32_t subMStart, uint32_t subMSize, uint32_t subKStart, uint32_t subKSize)
{
    auto srcTensor = l1Tensor[mL1Size * subKStart][GetC0Num<T1>() * subMStart];
    auto dstTensor = this->aL0Tensor[dstBufId];
    loadData3DParams.mExtension = subMSize;
    loadData3DParams.kExtension = subKSize;
    loadData3DParams.channelSize = subKSize;

    if (likely(mL1Size == M_L1_SPLIT_Size)) {
        LoadData<T1, LOAD3DV2_CONFIG>(dstTensor, srcTensor, loadData3DParams);
    } else {
        loadData3DParams.l1H = mL1Size / BLOCK_CUBE;
        LoadData(dstTensor, srcTensor, loadData3DParams);

        ResetLoad3DConfig<M_L1_SPLIT_Size, true>();
    }
}

template <typename FIAT, typename Config>
__aicore__ inline void FiaBlockCubeNonQuantGqa<FIAT, Config>::LoadBTransposeToL0(
            uint32_t dstBufId, const LocalTensor<KV_T> &l1Tensor, uint32_t nL1Size,
            uint32_t subNStart, uint32_t subNSize, uint32_t subKStart, uint32_t subKSize)
{
    auto srcTensor = l1Tensor[nL1Size * subKStart][GetC0Num<KV_T>() * subNStart];
    auto dstTensor = this->bL0Tensor[dstBufId];
    if (nL1Size == subNSize) {
        mm1LoadDataBTransposeToL0Params.repeatTimes = AttentionCommon::CeilDiv(subKSize, (uint32_t)BLOCK_CUBE) * AttentionCommon::CeilDiv(subNSize, GetC0Num<KV_T>());
        LoadData(dstTensor, srcTensor, mm1LoadDataBTransposeToL0Params);
    } else {
        mm1LoadDataBTransposeToL0Params.repeatTimes = AttentionCommon::CeilDiv(subNSize, (uint32_t)BLOCK_CUBE);
        uint32_t kLoops = subKSize / GetC0Num<KV_T>();
        for (uint32_t i = 0; i < kLoops; i++) {
            LoadData(dstTensor[subNSize * i * BLOCK_CUBE], srcTensor[nL1Size * i * GetC0Num<KV_T>()], mm1LoadDataBTransposeToL0Params);
        }
    }
}

template <typename FIAT, typename Config>
__aicore__ inline void FiaBlockCubeNonQuantGqa<FIAT, Config>::LoadBToL0(
            uint32_t dstBufId, const LocalTensor<KV_T> &l1Tensor, uint32_t kL1Size,
            uint32_t subKStart, uint32_t subKSize, uint32_t subNStart, uint32_t subNSize)
{
    mm2LoadDataBToL0Params.srcStride = kL1Size / BLOCK_CUBE;

    auto srcTensor = l1Tensor[kL1Size * subNStart][GetC0Num<KV_T>() * subKStart];
    auto dstTensor = this->bL0Tensor[dstBufId];

    uint32_t kLoops = subKSize / BLOCK_CUBE;
    for (uint32_t i = 0; i < kLoops; i++) {
        LoadData(dstTensor[i * BLOCK_CUBE * subNSize], srcTensor[i * BLOCK_CUBE * GetC0Num<KV_T>()], mm2LoadDataBToL0Params);
    }
}

template <typename FIAT, typename Config>
template <CubeFormat GMFormat>
__aicore__ inline void FiaBlockCubeNonQuantGqa<FIAT, Config>::FixpipeCToGM(
            GlobalTensor<MM_OUT_T> &mmResGm, uint32_t cL0BufId,
            uint32_t dstStride, uint32_t subMStart, uint32_t subMSize,
            uint32_t subNStart, uint32_t subNSize)
{
    FixpipeParamsV220 fixParams;
    fixParams.nSize = subNSize;
    fixParams.mSize = subMSize;
    fixParams.srcStride = AlignToPowerOfTwo(subMSize, (uint32_t)BLOCK_CUBE);
    fixParams.ndNum = 1;
    if constexpr (CFG::ENABLE_UNIFLAG) {
        fixParams.unitFlag = 3;
    }

    auto cL0Tensor = this->cL0Tensor[cL0BufId];
    if constexpr (GMFormat == CubeFormat::NZ) {
        ASCENDC_ASSERT(subNSize % BLOCK_CUBE == 0, { KERNEL_LOG(KERNEL_ERROR, "subNSize must be divisible by 16, current value is %u", subNSize); });
        fixParams.dstStride = dstStride * BLOCK_CUBE * sizeof(MM_OUT_T) / ONE_BLK_SIZE;
        auto dstTensor = mmResGm[dstStride * subNStart][BLOCK_CUBE * subMStart];
        Fixpipe<MM_OUT_T, T, CFG_NZ>(dstTensor, cL0Tensor, fixParams);
    } else {
        fixParams.dstStride = dstStride;
        auto dstTensor = mmResGm[dstStride * subMStart + subNStart];
        Fixpipe<MM_OUT_T, T, CFG_ROW_MAJOR>(dstTensor, cL0Tensor, fixParams);
    }
}

template <typename FIAT, typename Config>
template <CubeFormat OutFormat, CubeFormat BFormat>
__aicore__ inline void FiaBlockCubeNonQuantGqa<FIAT, Config>::ComputeMm1(const RunInfo &info)
{
    if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_BNSD) {
        if (info.isChangeBatch) {
            UpdateKey(info.bIdx);
        }
    }
#if !DEBUG_DISABLE_C1
    Axis m(info.actMBaseSize);
    Axis n(info.actualSingleProcessSInnerSize);
    Axis k(constInfo.headDim + constInfo.headDimRope);
    auto mmResGm = this->mm1ResGm[info.loop % CFG::PRELOAD_NUM];

    constexpr uint32_t M_SPLIT_SIZE = CFG::S2_BASICSIZE_IS_1024 ? 128 : 256;
    const uint32_t N_SPLIT_SIZE = (k.sizeAct <= 128) ? 256 : 128;

    constexpr uint32_t M_BASE = 128;
    constexpr uint32_t K_BASE = 128;
    constexpr uint32_t N_BASE = 128;

    ResetLoad3DConfig<M_SPLIT_SIZE>();

    auto mSlices = m.Split(M_SPLIT_SIZE);
    auto kL0Slices = k.Split(K_BASE);

    bool canFullLoadQ = (mSlices.size() <= L1_Q_BUFCNT);
    uint64_t qCoord = ((uint64_t)info.bIdx << 48) | ((uint64_t)info.n2Idx << 32) | ((uint64_t)info.gS1Idx);
    bool reuseQBuf = canFullLoadQ && (qL1Snapshot.signature == qCoord);
    if (!reuseQBuf) {
        qL1Snapshot.bufCnt = 0;
        qL1Snapshot.firstBufId = this->qL1BufId;
        qL1Snapshot.signature = qCoord;
    }

    for (auto& nL1 : n.Split(N_SPLIT_SIZE)) {
        WaitFlag<HardEvent::MTE1_MTE2>(KP_EVENT0 + this->kpL1BufId);
        CopyKToL1(this->kpL1BufId, info, nL1.start, nL1.sizeAct);

        SetFlag<HardEvent::MTE2_MTE1>(KP_EVENT0 + this->kpL1BufId);
        WaitFlag<HardEvent::MTE2_MTE1>(KP_EVENT0 + this->kpL1BufId);

        int32_t mL1Id = -1;
        for (auto& mL1 : mSlices) {
            ++mL1Id;

            if (unlikely(mL1Id >= qL1Snapshot.bufCnt)) {
                reuseQBuf = false;
            }

            uint32_t qBufId;
            if (unlikely(!reuseQBuf)) {
                qBufId = this->qL1BufId;
                // 在需要搬入Q前才去Set MTE1->MTE2事件，而不是在L0算完后就去Set，是考虑到Q_L1 buf的生命周期可能跨越多轮MM1计算，
                // 如果前一次MM1计算还未完成，还在复用Q_L1 buf，后一次MM1计算就开始搬运，就会覆盖了前一次计算的数据
                SetFlag<HardEvent::MTE1_MTE2>(Q_EVENT0 + this->qL1BufId);
                WaitFlag<HardEvent::MTE1_MTE2>(Q_EVENT0 + qBufId);
                CopyQToL1(qBufId, info, mL1.start, mL1.sizeAct);

                SetFlag<HardEvent::MTE2_MTE1>(Q_EVENT0 + qBufId);
                WaitFlag<HardEvent::MTE2_MTE1>(Q_EVENT0 + qBufId);
                qL1Snapshot.bufCnt += static_cast<uint32_t>(canFullLoadQ); // 不能全载Q时，不缓存到快照中
            } else {
                qBufId = (qL1Snapshot.firstBufId + mL1Id) % L1_Q_BUFCNT;
            }

            auto nL0Slices = nL1.Split(N_BASE);
            for (auto& mL0 : mL1.Split(M_BASE)) {
                for (auto& nL0 : nL0Slices) {
                    WaitFlag<HardEvent::FIX_M>(L0C_EVENT0 + this->cL0BufId);

                    for (auto& kL0 : kL0Slices) {
                        WaitFlag<HardEvent::M_MTE1>(L0AB_EVENT0 + this->abL0BufId);

                        LoadAToL0<M_SPLIT_SIZE>(this->abL0BufId, qL1Tensor[qBufId], mL1.AlignedSize(), mL0.start, mL0.AlignedSize(), kL0.start, kL0.AlignedSize());
                        LoadBTransposeToL0(this->abL0BufId, kpL1Tensor[this->kpL1BufId], nL1.AlignedSize(), nL0.start, nL0.AlignedSize(), kL0.start, kL0.AlignedSize());
                        
                        SetFlag<HardEvent::MTE1_M>(L0_READY_EVENT);
                        WaitFlag<HardEvent::MTE1_M>(L0_READY_EVENT);

                        bool isLastK = kL0.IsTailOf(k);
                        MmadParams mmadParams;
                        mmadParams.m = mL0.AlignedSize();
                        mmadParams.n = nL0.sizeAct;
                        mmadParams.k = kL0.sizeAct;
                        mmadParams.cmatrixInitVal = (kL0.start == 0);
                        mmadParams.cmatrixSource = false;
                        if constexpr (CFG::ENABLE_UNIFLAG) {
                            mmadParams.unitFlag = isLastK ? 3 : 2;
                        }
                        Mmad(cL0Tensor[this->cL0BufId], aL0Tensor[this->abL0BufId], bL0Tensor[this->abL0BufId], mmadParams);
                        if (likely(! isLastK)) {
                            AscendC::PipeBarrier<PIPE_M>();
                        }
                        SetFlag<HardEvent::M_MTE1>(L0AB_EVENT0 + this->abL0BufId);
                        this->abL0BufId = (this->abL0BufId + 1) % L0AB_BUFCNT;
                    }
                    if constexpr (! CFG::ENABLE_UNIFLAG) {
                        SetFlag<HardEvent::M_FIX>(L0C_EVENT0 + this->cL0BufId);
                        WaitFlag<HardEvent::M_FIX>(L0C_EVENT0 + this->cL0BufId);
                    }
                    FixpipeCToGM<OutFormat>(mmResGm, this->cL0BufId, (OutFormat == CubeFormat::NZ) ? m.sizeAct : info.actualSingleProcessSInnerSizeAlign, /* 输出ND时，mm1ResGm两行之间间隔元素个数按32对齐 */
                                            mL1.start + mL0.start, mL0.sizeAct, nL1.start + nL0.start,
                                            (OutFormat == CubeFormat::NZ) ? nL0.AlignedSize() : nL0.sizeAct);
                    SetFlag<HardEvent::FIX_M>(L0C_EVENT0 + this->cL0BufId);
                    this->cL0BufId = (this->cL0BufId + 1) % L0C_BUFCNT;
                }
            }
            if (unlikely(!reuseQBuf)) {
                this->qL1BufId = (this->qL1BufId + 1) % L1_Q_BUFCNT;
                reuseQBuf = canFullLoadQ;
            }
        }
        SetFlag<HardEvent::MTE1_MTE2>(KP_EVENT0 + this->kpL1BufId);
        ++this->kpL1BufId;
        if (this->kpL1BufId >= L1_KP_BUFCNT) {
            this->kpL1BufId = 0;
        }
    }
#endif
    CrossCoreSetFlag<ConstInfo::FIA_SYNC_MODE2, PIPE_FIX>(constInfo.syncC1V1);
}

template <typename FIAT, typename Config>
template <CubeFormat OutFormat, CubeFormat AFormat>
__aicore__ inline void FiaBlockCubeNonQuantGqa<FIAT, Config>::ComputeMm2(const RunInfo &info)
{
    if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_BNSD) {
        if (info.isChangeBatch) {
            UpdateValue(info.bIdx);
        }
    }

    CrossCoreWaitFlag(constInfo.syncV1C2);
#if !DEBUG_DISABLE_C2
    constexpr uint32_t M_SPLIT_SIZE = 128;
    constexpr uint32_t K_SPLIT_SIZE = 256;
    constexpr uint32_t K_BASE = 128;

    Axis m(info.actMBaseSize);
    Axis n(constInfo.headDim);
    Axis k(info.actualSingleProcessSInnerSize);
    auto mmResGm = this->mm2ResGm[info.loop % CFG::PRELOAD_NUM];

    ResetLoad3DConfig<M_SPLIT_SIZE>();

    auto KL1Slices = k.Split(K_SPLIT_SIZE);

    bool canFullLoadV = (KL1Slices.size() <= L1_V_BUFCNT);
    uint64_t vCoord = ((uint64_t)info.bIdx << 48) | ((uint64_t)info.n2Idx << 32) | ((uint64_t)info.s2Idx);
    bool reuseVBuf = false;
    vL1Snapshot.bufCnt = 0;
    vL1Snapshot.firstBufId = this->vL1BufId;
    vL1Snapshot.signature = vCoord;

    for (auto& mL1 : m.Split(M_SPLIT_SIZE)) {
        WaitFlag<HardEvent::FIX_M>(L0C_EVENT0 + this->cL0BufId);

        int32_t kL1Id = -1;
        for (auto& kL1 : KL1Slices) {
            ++kL1Id;
            WaitFlag<HardEvent::MTE1_MTE2>(KP_EVENT0 + this->kpL1BufId);
            CopyPToL1<AFormat>(this->kpL1BufId, info, (AFormat == CubeFormat::NZ) ? m.AlignedSize() : info.actualSingleProcessSInnerSizeAlign, /* P为ND时，每行元素个数会按32对齐 */
                                             mL1.start, mL1.sizeAct, kL1.start, kL1.sizeAct);
            
            SetFlag<HardEvent::MTE2_MTE1>(KP_EVENT0 + this->kpL1BufId);
            WaitFlag<HardEvent::MTE2_MTE1>(KP_EVENT0 + this->kpL1BufId);

            if (unlikely(kL1Id >= vL1Snapshot.bufCnt)) {
                reuseVBuf = false;
            }

            uint32_t vBufId;
            if (unlikely(!reuseVBuf)) {
                vBufId = this->vL1BufId;
                WaitFlag<HardEvent::MTE1_MTE2>(V_EVENT0 + vBufId);
                CopyVToL1(vBufId, info, kL1.start, kL1.sizeAct);

                SetFlag<HardEvent::MTE2_MTE1>(V_EVENT0 + vBufId);
                WaitFlag<HardEvent::MTE2_MTE1>(V_EVENT0 + vBufId);
                vL1Snapshot.bufCnt += static_cast<uint32_t>(canFullLoadV); // 不能全载V时，不缓存到快照中
            } else {
                vBufId = (vL1Snapshot.firstBufId + kL1Id) % L1_V_BUFCNT;
            }

            for (auto& kL0 : kL1.Split(K_BASE)) {
                WaitFlag<HardEvent::M_MTE1>(L0AB_EVENT0 + this->abL0BufId);

                LoadAToL0<M_SPLIT_SIZE>(this->abL0BufId, kpL1Tensor[this->kpL1BufId], mL1.AlignedSize(), 0, mL1.AlignedSize(), kL0.start, kL0.AlignedSize());
                LoadBToL0(this->abL0BufId, vL1Tensor[vBufId], kL1.AlignedSize(), kL0.start, kL0.AlignedSize(), 0, n.AlignedSize());
                SetFlag<HardEvent::MTE1_M>(L0_READY_EVENT);
                WaitFlag<HardEvent::MTE1_M>(L0_READY_EVENT);

                bool isLastK = ((k.sizeAct - kL1.start - kL0.start) <= K_BASE);
                MmadParams mmadParams;
                mmadParams.m = mL1.AlignedSize();
                mmadParams.n = n.sizeAct;
                mmadParams.k = kL0.sizeAct;
                mmadParams.cmatrixInitVal = (kL1.start == 0) && (kL0.start == 0);
                mmadParams.cmatrixSource = false;
                if constexpr (CFG::ENABLE_UNIFLAG) {
                    mmadParams.unitFlag = isLastK ? 3 : 2;
                }
                Mmad(cL0Tensor[this->cL0BufId], aL0Tensor[this->abL0BufId], bL0Tensor[this->abL0BufId], mmadParams);
                if (likely(! isLastK)) {
                    AscendC::PipeBarrier<PIPE_M>();
                }
                SetFlag<HardEvent::M_MTE1>(L0AB_EVENT0 + this->abL0BufId);
                this->abL0BufId = (this->abL0BufId + 1) % L0AB_BUFCNT;
            }
            SetFlag<HardEvent::MTE1_MTE2>(KP_EVENT0 + this->kpL1BufId);
            ++this->kpL1BufId;
            if (this->kpL1BufId >= L1_KP_BUFCNT) {
                this->kpL1BufId = 0;
            }

            if (unlikely(!canFullLoadV || mL1.IsTailOf(m))) {
                // 当可以复用V_L1 buf时，它的生命周期跨越整个mL1的迭代，在mL1迭代完成释放
                SetFlag<HardEvent::MTE1_MTE2>(V_EVENT0 + vBufId);
            }

            if (unlikely(!reuseVBuf)) {
                this->vL1BufId = (this->vL1BufId + 1) % L1_V_BUFCNT;
                reuseVBuf = canFullLoadV;
            }
        }
        if constexpr (! CFG::ENABLE_UNIFLAG) {
            SetFlag<HardEvent::M_FIX>(L0C_EVENT0 + this->cL0BufId);
            WaitFlag<HardEvent::M_FIX>(L0C_EVENT0 + this->cL0BufId);
        }
        FixpipeCToGM<OutFormat>(mmResGm, this->cL0BufId, (OutFormat == CubeFormat::NZ) ? m.sizeAct : constInfo.headDimAlign, /* 输出ND时，mm2ResGm两行之间间隔元素个数按32对齐 */
                                mL1.start, mL1.sizeAct, 0, (OutFormat == CubeFormat::NZ) ? n.AlignedSize() : n.sizeAct);
        SetFlag<HardEvent::FIX_M>(L0C_EVENT0 + this->cL0BufId);
        this->cL0BufId = (this->cL0BufId + 1) % L0C_BUFCNT;
    }
#endif
    CrossCoreSetFlag<ConstInfo::FIA_SYNC_MODE2, PIPE_FIX>(constInfo.syncC2V2);
}
#endif // FIA_BLOCK_CUBE_NONQUANT_GQA_H
