/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file service_matmul.h
 * \brief
 */

#ifndef SERVICE_MATMUL_H
#define SERVICE_MATMUL_H

#include "mla_prolog_comm.h"

namespace MlaProlog {

constexpr uint8_t UNIT_FLAG_DISABLE = 0;  // 0: disable: 不配置unitFlag
constexpr uint8_t UNIT_FLAG_CHECK = 0b10; // 2: enable: 将mmadParams.unitFlag设置为 0b10
constexpr uint8_t UNIT_FLAG_SET = 0b11;   // 3: enable: 在k的最后一轮循环，会将mmadParams.unitFlag设置为 0b11


/**
 * @brief Struct to encapsulate all local tensor objects for matrix multiplication
 * @tparam T Data type for A, B, and L0 tensors
 * @tparam O_L0C Data type for C L0 tensor
 */
template <typename T, typename O_L0C>
struct mmLocalTensors {
    LocalTensor<T> aL1Tensor;
    LocalTensor<T> bL1Tensor;
    LocalTensor<T> aL0Tensor;
    LocalTensor<T> bL0Tensor;
    LocalTensor<O_L0C> cL0Tensor;

    /**
     * @brief Initialize all local tensors with buffer addresses from bufParam
     * @param bufParam Buffer parameter object containing all buffer addresses
     */
    __aicore__ inline void Init(const MMBufParams &bufParam)
    {
        aL1Tensor.SetAddr(bufParam.aL1BufAddr);
        bL1Tensor.SetAddr(bufParam.bL1BufAddr);
        aL0Tensor.SetAddr(bufParam.aL0BufAddr);
        bL0Tensor.SetAddr(bufParam.bL0BufAddr);
        cL0Tensor.SetAddr(bufParam.cL0BufAddr);
    }
};

template <typename SrcT>
__aicore__ inline constexpr uint32_t GetC0Num()
{
    if (sizeof(SrcT) == sizeof(float)) {
        return 8;
    } else if (sizeof(SrcT) == sizeof(int8_t)) {
        return 32;
    }
    return 16;
}

template <typename T>
__aicore__ inline void CopyNDGmToL1(LocalTensor<T> &l1Tensor, const GlobalTensor<T> &gmSrcTensor, uint32_t srcN,
                                    uint32_t srcD, uint32_t srcDstride)
{
    Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = 1;
    nd2nzPara.nValue = srcN; // 行数
    nd2nzPara.dValue = srcD;
    nd2nzPara.srcDValue = srcDstride;
    nd2nzPara.dstNzC0Stride = CeilDivT(srcN, BLOCK_CUBE_SIZE) * BLOCK_CUBE_SIZE; // 对齐到16 单位 block
    nd2nzPara.dstNzNStride = 1;
    nd2nzPara.srcNdMatrixStride = 0;
    nd2nzPara.dstNzMatrixStride = 0;
    DataCopy(l1Tensor, gmSrcTensor, nd2nzPara);
}

template <typename T>
__aicore__ inline void CopyNZGmToL1(LocalTensor<T> &l1Tensor, const GlobalTensor<T> &gmSrcTensor, uint32_t srcN,
                                    uint32_t srcD, uint32_t srcNstride)
{
    DataCopyParams param;
    param.blockCount = CeilDivT(srcD, GetC0Num<T>());
    param.blockLen = srcN;                 // 单位为32B srcN*16/16
    param.srcStride = (srcNstride - srcN); // 单位为32B (srcNstride - srcN)*16/16
    param.dstStride = 0;
    DataCopy(l1Tensor, gmSrcTensor, param);
}

template <typename T, bool preload = false>
__aicore__ inline void LoadL1A(const GlobalTensor<T> &tensorAGm, const uint32_t mInput, const uint32_t kL1StepSize,
                               const uint32_t kInput, MMBufParams &bufParam)
{
    uint32_t bufIter = bufParam.aL1BufIter;
    if constexpr (preload) {
        bufIter++;
    }
    LocalTensor<T> aL1Tensor;
    aL1Tensor.SetAddr(bufParam.aL1BufAddr);
    uint32_t aL1Offset = L1_A_SIZE / sizeof(T);

    WaitFlag<HardEvent::MTE1_MTE2>(A_EVENT0 + (bufIter & 1u));
    LocalTensor<T> aL1 = aL1Tensor[(bufIter & 1u) * aL1Offset];
    CopyNDGmToL1(aL1, tensorAGm, mInput, kL1StepSize, kInput);
    SetFlag<HardEvent::MTE2_MTE1>(A_EVENT0 + (bufIter & 1u));
}

template <typename T, DataFormat loadFormat = DataFormat::NZ, bool preload = false>
__aicore__ inline void LoadL1B(const GlobalTensor<T> &tensorBGm, const uint32_t nL1Size, const uint32_t nInput,
                               const uint32_t kL1StepSize, const uint32_t kInput, MMBufParams &bufParam)
{
    uint32_t bufIter = bufParam.bL1BufIter;
    if constexpr (preload) {
        bufIter++;
    }
    LocalTensor<T> bL1Tensor;
    bL1Tensor.SetAddr(bufParam.bL1BufAddr);
    uint32_t bL1Offset = L1_B_SIZE / sizeof(T);

    WaitFlag<HardEvent::MTE1_MTE2>(B_EVENT0 + (bufIter & 1u));
    LocalTensor<T> bL1 = bL1Tensor[(bufIter & 1u) * bL1Offset];
    if constexpr (loadFormat == DataFormat::NZ) {
        CopyNZGmToL1(bL1, tensorBGm, kL1StepSize, nL1Size, kInput);
    } else {
        CopyNDGmToL1(bL1, tensorBGm, kInput, nL1Size, nInput);
    }
    SetFlag<HardEvent::MTE2_MTE1>(B_EVENT0 + (bufIter & 1u));
}

template <typename T, typename S, bool preload = false, bool scaleSrcPadFlag = false>
__aicore__ inline void LoadL1AAndScale(const GlobalTensor<T> &tensorAGm, const GlobalTensor<S> &tensorAScaleGm,
                                       const uint32_t mInput, const uint32_t kL1StepSize, const uint32_t kInput,
                                       const uint32_t nInput, const uint64_t scaleAInL1BOffset, MMBufParams &bufParam)
{
    uint32_t bufIter = bufParam.aL1BufIter;
    if constexpr (preload) {
        bufIter++;
    }
    LocalTensor<T> aL1Tensor;
    aL1Tensor.SetAddr(bufParam.aL1BufAddr);
    uint32_t aL1Offset = L1_A_SIZE / sizeof(T);

    LocalTensor<bfloat16_t> bL1Tensor;
    bL1Tensor.SetAddr(bufParam.bL1BufAddr);
    uint32_t srcDValue = nInput / FP8_TWO;
    if constexpr (scaleSrcPadFlag) {
        srcDValue = Align(nInput / FP8_TWO, BLOCK_CUBE_SIZE);
    }
    GlobalTensor<bfloat16_t> tensorScaleGmCast;
    tensorScaleGmCast.SetGlobalBuffer(((__gm__ bfloat16_t *)(tensorAScaleGm.GetPhyAddr())));

    WaitFlag<HardEvent::MTE1_MTE2>(A_EVENT0 + (bufIter & 1u));
    LocalTensor<T> aL1 = aL1Tensor[(bufIter & 1u) * aL1Offset];
    CopyNDGmToL1(aL1, tensorAGm, mInput, kL1StepSize, kInput);

    Dn2NzParams dn2Nzparam;
    dn2Nzparam.dnNum = 1;
    dn2Nzparam.nValue = nInput / FP8_TWO; // 单个DN矩阵的列数N, 单位为元素个数
    dn2Nzparam.dValue = mInput;           // 单个DN矩阵的实际行数D, 单位为元素个数
    dn2Nzparam.srcDnMatrixStride = 0;     // 相邻DN矩阵起始地址之间的偏移, 单位为元素个数
    dn2Nzparam.srcDValue = srcDValue;     // 同一个DN矩阵中相邻行起始地址之间的偏移, 单位为元素个数
    dn2Nzparam.dstNzC0Stride =
        nInput / FP8_TWO;        // 来自DN矩阵中同一列的相邻两个block在NZ矩阵中起始地址间的偏移，单位为block(32B)个数
    dn2Nzparam.dstNzNStride = 1; // 转换为NZ矩阵后，DN中相邻两行在NZ矩阵中起始地址之间的偏移，单位为元素个数
    dn2Nzparam.dstNzMatrixStride = dn2Nzparam.nValue; // 两个NZ矩阵，起始地址之间的偏移，单位为block个数
    DataCopy(bL1Tensor[scaleAInL1BOffset / FP8_TWO], tensorScaleGmCast, dn2Nzparam);

    SetFlag<HardEvent::MTE2_MTE1>(A_EVENT0 + (bufIter & 1u));
}

template <typename T, typename S, DataFormat loadFormat = DataFormat::NZ, bool preload = false>
__aicore__ inline void LoadL1BAndScale(const GlobalTensor<T> &tensorBGm, const GlobalTensor<S> &tensorBScaleGm,
                                       const uint32_t nL1Size, const uint32_t kL1StepSize, const uint32_t kInput,
                                       const uint32_t nInput, const uint64_t scaleBInL1BOffset, MMBufParams &bufParam)
{
    uint32_t bufIter = bufParam.bL1BufIter;
    if constexpr (preload) {
        bufIter++;
    }
    LocalTensor<T> bL1Tensor;
    bL1Tensor.SetAddr(bufParam.bL1BufAddr);
    uint32_t bL1Offset = L1_B_SIZE / sizeof(T);

    LocalTensor<bfloat16_t> bL1ScaleTensor;
    bL1ScaleTensor.SetAddr(bufParam.bL1BufAddr);
    GlobalTensor<bfloat16_t> tensorScaleGmCast;
    tensorScaleGmCast.SetGlobalBuffer(((__gm__ bfloat16_t *)(tensorBScaleGm.GetPhyAddr())));

    WaitFlag<HardEvent::MTE1_MTE2>(B_EVENT0 + (bufIter & 1u));

    LocalTensor<T> bL1 = bL1Tensor[(bufIter & 1u) * bL1Offset];
    if constexpr (loadFormat == DataFormat::NZ) {
        CopyNZGmToL1(bL1, tensorBGm, kL1StepSize, nL1Size, kInput);
    } else {
        CopyNDGmToL1(bL1, tensorBGm, kInput, nL1Size, nL1Size);
    }

    Dn2NzParams dn2Nzparam;
    dn2Nzparam.dnNum = 1;
    dn2Nzparam.nValue = nInput / FP8_TWO;    // 单个DN矩阵的列数N, 单位为元素个数
    dn2Nzparam.dValue = nL1Size;             // 单个DN矩阵的实际行数D, 单位为元素个数
    dn2Nzparam.srcDnMatrixStride = 0;        // 相邻DN矩阵起始地址之间的偏移, 单位为元素个数
    dn2Nzparam.srcDValue = nInput / FP8_TWO; // 同一个DN矩阵中相邻行起始地址之间的偏移, 单位为元素个数
    dn2Nzparam.dstNzC0Stride =
        dn2Nzparam.srcDValue;    // 来自DN矩阵中同一列的相邻两个block在NZ矩阵中起始地址间的偏移，单位为block(32B)个数
    dn2Nzparam.dstNzNStride = 1; // 转换为NZ矩阵后，DN中相邻两行在NZ矩阵中起始地址之间的偏移，单位为元素个数
    dn2Nzparam.dstNzMatrixStride = dn2Nzparam.dstNzC0Stride; // 两个NZ矩阵，起始地址之间的偏移，单位为block个数
    DataCopy(bL1ScaleTensor[scaleBInL1BOffset / FP8_TWO], tensorScaleGmCast, dn2Nzparam);

    SetFlag<HardEvent::MTE2_MTE1>(B_EVENT0 + (bufIter & 1u));
}

/**
 * @brief 使用3D Pro模式将数据从L1加载到L0，可选择是否进行转置操作
 * @tparam T 张量的数据类型
 * @tparam enTranspose 是否启用转置操作（用于B矩阵）
 * @param l0Tensor L0缓冲区中的目标张量
 * @param l1Tensor L1缓冲区中的源张量
 * @param mSize 行维度（对应A矩阵的m，B矩阵的k）
 * @param kSize 列维度（对应A矩阵的k，B矩阵的n）
 * @param l1StepSize L1缓冲区中的步长大小，用于B矩阵转置
 * @param kl1Size L1缓冲区中的大小，用于B矩阵转置
 */
template <typename T, bool enTranspose = false>
__aicore__ inline void LoadDataL1ToL0(const LocalTensor<T> &l0Tensor, const LocalTensor<T> &l1Tensor,
                                      const uint32_t mSize, const uint32_t kSize, const uint32_t l1StepSize = 0)
{
    if constexpr (enTranspose) { // B
        LoadData2DParamsV2 loadData2DV2;
        loadData2DV2.mStartPosition = 0;
        loadData2DV2.kStartPosition = 0;
        loadData2DV2.mStep = ((mSize + ROUND_UP_UNIT) >> SHIFTS_UNIT << SHIFTS_UNIT) / BLOCK_CUBE_SIZE;
        loadData2DV2.kStep = ((kSize + ROUND_UP_UNIT) >> SHIFTS_UNIT << SHIFTS_UNIT) / BLOCK_CUBE_SIZE;
        loadData2DV2.srcStride = ((l1StepSize + ROUND_UP_UNIT) >> SHIFTS_UNIT << SHIFTS_UNIT) / BLOCK_CUBE_SIZE;
        loadData2DV2.dstStride = ((kSize + ROUND_UP_UNIT) >> SHIFTS_UNIT << SHIFTS_UNIT) / BLOCK_CUBE_SIZE;
        loadData2DV2.ifTranspose = true;
        LoadData<T>(l0Tensor, l1Tensor, loadData2DV2);
    } else { // A
        LoadData3DParamsV2<T> loadDataAParams;
        loadDataAParams.l1W = 1;
        loadDataAParams.l1H = mSize;
        loadDataAParams.channelSize = kSize;
        loadDataAParams.kExtension = kSize;
        loadDataAParams.mExtension = mSize;
        loadDataAParams.kStartPt = 0;
        loadDataAParams.mStartPt = 0;
        loadDataAParams.strideW = 1;
        loadDataAParams.strideH = 1;
        loadDataAParams.filterW = 1;
        loadDataAParams.filterH = 1;
        loadDataAParams.dilationFilterW = 1;
        loadDataAParams.dilationFilterH = 1;
        loadDataAParams.enTranspose = false;
        loadDataAParams.enSmallK = false;
        loadDataAParams.padValue = 0;
        loadDataAParams.filterSizeW = 0;
        loadDataAParams.filterSizeH = 0;
        loadDataAParams.fMatrixCtrl = false;
        uint16_t dstStride = CeilDivT(mSize, BLOCK_CUBE_SIZE);
#if defined(ASC_DEVKIT_VERSION_NUM) && (ASC_DEVKIT_VERSION_NUM >= 90000000)
        SetLoadDataRepeatWithStride({0, 1, 0, dstStride}); // >= 9.0.0 release 新 API
        LoadDataWithStride<T>(l0Tensor, l1Tensor, loadDataAParams);
#else
        SetLoadDataRepeat({0, 1, 0, dstStride}); // < 9.0.0 (beta.2) 旧 API
        LoadData<T>(l0Tensor, l1Tensor, loadDataAParams);
#endif
    }
}

template <typename T>
__aicore__ inline void LoadDataL1ToL0Mxfp8(const LocalTensor<T> &l0Tensor, const LocalTensor<T> &l1Tensor,
                                           const LocalTensor<T> &l1AMxTensor, const uint32_t mSize,
                                           const uint32_t kSize, const uint32_t kL1Size = 0)
{
    // LoadData and A Scale
    LocalTensor<mx_fp8_e4m3_t> dstTensor = l0Tensor.template ReinterpretCast<mx_fp8_e4m3_t>();
    LocalTensor<fp8_e8m0_t> l1MxTensor = l1AMxTensor.template ReinterpretCast<fp8_e8m0_t>();

    LoadData2DParamsV2 loadDataA2DParams;
    loadDataA2DParams.mStartPosition = 0;
    loadDataA2DParams.kStartPosition = 0;
    loadDataA2DParams.ifTranspose = false;
    loadDataA2DParams.mStep = CeilDivT(mSize, BLOCK_CUBE_SIZE); // m轴分型大小为16
    loadDataA2DParams.kStep = CeilDivT(kSize, K_STEP_SIZE_32);  // k轴分型大小为32B
    // 配合ub->L1使用256 * 32 / 256
    // 64搬运
    loadDataA2DParams.srcStride = loadDataA2DParams.mStep; // m轴全搬，无pad
    loadDataA2DParams.dstStride = loadDataA2DParams.mStep; // 搬运前后不转置，m轴不变

    LoadData2DMxParams loadAScaleParam;
    loadAScaleParam.xStartPosition = 0;
    loadAScaleParam.yStartPosition = 0;
    loadAScaleParam.xStep = CeilDivT(mSize, BLOCK_CUBE_SIZE);
    loadAScaleParam.yStep = CeilDivT(kSize, K_STEP_SIZE_32 * FP8_TWO);       // ksize对应baseK
    loadAScaleParam.srcStride = CeilDivT(kL1Size, K_STEP_SIZE_32 * FP8_TWO); // kL1Size对应baseK*stepK
    loadAScaleParam.dstStride = loadAScaleParam.yStep;
    LoadData(dstTensor, l1Tensor, l1MxTensor, loadDataA2DParams, loadAScaleParam);
}

template <typename T>
__aicore__ inline void LoadDataL1ToL0B(const LocalTensor<T> &bL0Tensor, const LocalTensor<T> &bL1Tensor,
                                       const LocalTensor<T> &bscaleL1, const uint32_t kSize, const uint32_t nSize,
                                       const uint32_t kL1StepSize = 0, const uint32_t kL1Size = 0)
{
    LoadData2DParamsV2 loadData2DV2;
    loadData2DV2.mStartPosition = 0;
    loadData2DV2.kStartPosition = 0;
    loadData2DV2.mStep = CeilDivT(kSize, BLOCK_CUBE_SIZE);
    loadData2DV2.kStep = CeilDivT(nSize, K_STEP_SIZE_32);
    loadData2DV2.srcStride = CeilDivT(kL1StepSize, BLOCK_CUBE_SIZE); // k轴切分，stride为完整k的长度
    loadData2DV2.dstStride =
        CeilDivT(nSize, UNIT_SIZE / K_STEP_SIZE_32); // n轴切分，B矩阵搬运后转置，dst中M轴变成src中的K轴
    loadData2DV2.ifTranspose = true;                 // 搬运后转置

    if (std::is_same<T, FP8E4M3>::value) {
        LocalTensor<mx_fp8_e4m3_t> dstTensor = bL0Tensor.template ReinterpretCast<mx_fp8_e4m3_t>();
        LocalTensor<FP8E4M3> srcTensor = bL1Tensor.template ReinterpretCast<FP8E4M3>();
        LocalTensor<fp8_e8m0_t> l1MxTensor = bscaleL1.template ReinterpretCast<fp8_e8m0_t>();
        LoadData2DMxParams loadDataMxParams;
        loadDataMxParams.xStartPosition = 0;
        loadDataMxParams.yStartPosition = 0;
        loadDataMxParams.xStep = CeilDivT(nSize, BLOCK_CUBE_SIZE);
        loadDataMxParams.yStep = CeilDivT(kSize, K_STEP_SIZE_32 * FP8_TWO);       // ksize对应baseK
        loadDataMxParams.srcStride = CeilDivT(kL1Size, K_STEP_SIZE_32 * FP8_TWO); // kL1Size对应baseK*stepK
        loadDataMxParams.dstStride = loadDataMxParams.yStep;
        LoadData(dstTensor, srcTensor, l1MxTensor, loadData2DV2, loadDataMxParams);
    } else {
        LoadData<T>(bL0Tensor, bL1Tensor, loadData2DV2);
    }
}

template <typename T, typename O, typename S>
__aicore__ inline void MatmulL0(MMBufParams &bufParam, const LocalTensor<T> &aL1, const LocalTensor<T> &bL1,
                                const LocalTensor<T> &aL0Tensor, const LocalTensor<T> &bL0Tensor,
                                const LocalTensor<O> &cL0Tensor, const MmadParams &mmadParams,
                                const uint32_t kL1StepSize, const uint32_t kL1Size = 0,
                                const LocalTensor<T> &aScaleL1 = {}, const LocalTensor<T> &bscaleL1 = {})
{
    WaitFlag<HardEvent::M_MTE1>(L0A_EVENT0 + (bufParam.aL0BufIter & 1u));
    LocalTensor<T> aL0 = aL0Tensor[(bufParam.aL0BufIter & 1u) * (L0A_PP_SIZE / sizeof(T))];
    if constexpr (std::is_same<T, FP8E4M3>::value && std::is_same<S, fp8_e8m0_t>::value) {
        LoadDataL1ToL0Mxfp8<T>(aL0, aL1, aScaleL1, mmadParams.m, mmadParams.k, kL1Size);
    } else {
        LoadDataL1ToL0<T>(aL0, aL1, mmadParams.m, mmadParams.k);
    }
    SetFlag<HardEvent::MTE1_M>(L0A_EVENT0 + (bufParam.aL0BufIter & 1u));
    WaitFlag<HardEvent::MTE1_M>(L0A_EVENT0 + (bufParam.aL0BufIter & 1u));
    WaitFlag<HardEvent::M_MTE1>(L0B_EVENT0 + (bufParam.bL0BufIter & 1u));

    LocalTensor<T> bL0 = bL0Tensor[(bufParam.bL0BufIter & 1u) * (L0B_PP_SIZE / sizeof(T))];
    if constexpr (std::is_same<T, int8_t>::value || std::is_same<T, HIF8>::value ||
                  (std::is_same<T, FP8E4M3>::value && std::is_same<S, float>::value)) {
        LoadDataL1ToL0B(bL0, bL1, bscaleL1, mmadParams.k, mmadParams.n, kL1StepSize);
    } else if constexpr (std::is_same<T, FP8E4M3>::value && std::is_same<S, fp8_e8m0_t>::value) {
        LoadDataL1ToL0B(bL0, bL1, bscaleL1, mmadParams.k, mmadParams.n, kL1StepSize, kL1Size);
    } else {
        LoadDataL1ToL0<T, true>(bL0, bL1, mmadParams.k, mmadParams.n, kL1StepSize);
    }

    SetFlag<HardEvent::MTE1_M>(L0B_EVENT0 + (bufParam.bL0BufIter & 1u));
    WaitFlag<HardEvent::MTE1_M>(L0B_EVENT0 + (bufParam.bL0BufIter & 1u));
    if constexpr (std::is_same<T, FP8E4M3>::value && std::is_same<S, fp8_e8m0_t>::value) {
        LocalTensor<mx_fp8_e4m3_t> aL0Tmp = aL0.template ReinterpretCast<mx_fp8_e4m3_t>();
        LocalTensor<mx_fp8_e4m3_t> bL0Tmp = bL0.template ReinterpretCast<mx_fp8_e4m3_t>();
        Mmad(cL0Tensor, aL0Tmp, bL0Tmp, mmadParams);
    } else {
        Mmad(cL0Tensor, aL0, bL0, mmadParams);
    }
    PipeBarrier<PIPE_M>();
    SetFlag<HardEvent::M_MTE1>(L0B_EVENT0 + (bufParam.bL0BufIter & 1u));
    bufParam.bL0BufIter++;
    SetFlag<HardEvent::M_MTE1>(L0A_EVENT0 + (bufParam.aL0BufIter & 1u));
    bufParam.aL0BufIter++;
}

template <typename T, typename O, typename O_L0C, bool enUnitFlag = false>
__aicore__ inline void GetTensorC(const GlobalTensor<O> &tensorCGm, const LocalTensor<O_L0C> &cL0, const uint32_t mSize,
                                  const uint32_t nSize, const uint32_t srcStride, const uint32_t dstStride,
                                  MMBufParams &bufParam)
{
    FixpipeParamsV220 fixParams;
    fixParams.nSize = nSize;         // 实现切片大小
    fixParams.mSize = mSize;         // msdIterNum * gSize; // 有效数据不足16行，只需要输出部分行即可
    fixParams.srcStride = srcStride; // ((fixParams.mSize + 15) / 16) * 16
    fixParams.dstStride = dstStride;
    fixParams.ndNum = 1;
    if constexpr (enUnitFlag) {
        fixParams.unitFlag = UNIT_FLAG_SET;
    } else {
        fixParams.unitFlag = UNIT_FLAG_DISABLE;
    }
    if constexpr (std::is_same<O_L0C, float>::value && std::is_same<O, bfloat16_t>::value) {
        fixParams.quantPre = QuantMode_t::F322BF16;
    }
    if constexpr (!enUnitFlag) {
        SetFlag<HardEvent::M_FIX>(L0C_EVENT0 + (bufParam.cL0BufIter & 1u));
        WaitFlag<HardEvent::M_FIX>(L0C_EVENT0 + (bufParam.cL0BufIter & 1u));
    }
    Fixpipe(tensorCGm, cL0, fixParams);
}

/**
 * @brief 将AB矩阵从GM加载到L1缓存以进行矩阵乘法运算
 * @tparam T 张量的数据类型
 * @tparam hasL1ALoaded A矩阵是否已经加载到L1中
 * @param tensorAGm GM中的A矩阵
 * @param tensorBGm GM中的B矩阵
 * @param kL1 当前K L1循环迭代次数
 * @param kL1Loops K L1循环的总次数
 * @param para 包含维度和步长信息的矩阵乘法参数
 * @param nL1Offset L1缓冲区中的N维度偏移
 * @param nL1Size L1缓冲区中的N维度大小
 * @param kOffesetUnit K维度寻址的偏移单位
 * @param bufParam L1缓冲区管理的参数对象
 */
template <typename T, bool hasL1ALoaded, DataFormat bLoadFormat = DataFormat::NZ>
__aicore__ inline void LoadL1AB(const GlobalTensor<T> &tensorAGm, const GlobalTensor<T> &tensorBGm, uint32_t kL1,
                                uint32_t kL1Loops, const MMParams &para, uint32_t nL1Offset, uint32_t nL1Size,
                                uint32_t kOffesetUnit, MMBufParams &bufParam)
{
    if (kL1 == 0) {
        if constexpr (!hasL1ALoaded) {
            LoadL1A(tensorAGm[kL1 * para.kL1StepSize], para.m, para.kL1StepSize, para.orgKa, bufParam);
        }
        if constexpr (bLoadFormat == DataFormat::NZ) {
            LoadL1B(tensorBGm[para.k * nL1Offset + kL1 * kOffesetUnit], nL1Size, para.n, para.kL1StepSize, para.k,
                    bufParam);
        } else {
            LoadL1B<T, DataFormat::ND, false>(tensorBGm[kL1 * para.kL1StepSize * para.n + nL1Offset], nL1Size, para.n,
                                              para.kL1StepSize, para.kL1StepSize, bufParam);
        }
    }

    if (kL1 + 1 < kL1Loops) {
        if constexpr (!hasL1ALoaded) {
            LoadL1A<T, true>(tensorAGm[(kL1 + 1) * para.kL1StepSize], para.m, para.kL1StepSize, para.orgKa, bufParam);
        }
        if constexpr (bLoadFormat == DataFormat::NZ) {
            LoadL1B<T, DataFormat::NZ, true>(tensorBGm[para.k * nL1Offset + (kL1 + 1) * kOffesetUnit], nL1Size, para.n,
                                             para.kL1StepSize, para.k, bufParam);
        } else {
            LoadL1B<T, DataFormat::ND, true>(tensorBGm[(kL1 + 1) * para.kL1StepSize * para.n + nL1Offset], nL1Size,
                                             para.n, para.kL1StepSize, para.kL1StepSize, bufParam);
        }
    }
}

/**
 * @brief 将AB矩阵和用于伪量化的Scale矩阵从GM加载到L1缓存以进行矩阵乘法运算
 * @tparam T 张量的数据类型
 * @tparam S Scale矩阵的数据类型
 * @tparam hasL1ALoaded A矩阵是否已经加载到L1中
 * @tparam scaleSrcPadFlag Scale矩阵是否需要对齐搬运，在mm3中AScale为True
 * @param tensorAGm GM中的A矩阵
 * @param tensorBGm GM中的B矩阵
 * @param tensorAScaleGm GM中的AScale矩阵
 * @param tensorBScaleGm GM中的BScale矩阵
 * @param kL1 当前K L1循环迭代次数
 * @param kL1Loops K L1循环的总次数
 * @param para 包含维度和步长信息的矩阵乘法参数
 * @param nL1Offset L1缓冲区中的N维度偏移
 * @param nL1Size L1缓冲区中的N维度大小
 * @param kOffesetUnit K维度寻址的偏移单位
 * @param bufParam L1缓冲区管理的参数对象
 */
template <typename T, typename S, bool hasL1ALoaded, bool scaleSrcPadFlag = false>
__aicore__ inline void LoadL1ABAndScale(const GlobalTensor<T> &tensorAGm, const GlobalTensor<T> &tensorBGm,
                                        const GlobalTensor<S> &tensorAScaleGm, const GlobalTensor<S> &tensorBScaleGm,
                                        uint32_t kL1, uint32_t kL1Loops, const MMParams &para, uint32_t nL1Offset,
                                        uint32_t nL1Size, uint32_t kOffesetUnit, MMBufParams &bufParam)
{
    uint64_t offsetL1B = L1_B_SIZE / 2 / sizeof(T); // // 2表示scale起始地址固定从L1B上ping的64k开始
    if (kL1 == 0) {
        if constexpr (!hasL1ALoaded) {
            LoadL1AAndScale<T, S, hasL1ALoaded, scaleSrcPadFlag>(tensorAGm, tensorAScaleGm, para.m, para.kL1StepSize,
                                                                 para.k, para.kScale, offsetL1B, bufParam);
        }
        uint64_t scaleOffsetL1B = offsetL1B + para.kScale * Align(para.m, BLOCK_CUBE_SIZE);
        LoadL1BAndScale(tensorBGm[para.k * nL1Offset], tensorBScaleGm[para.kScale * nL1Offset], nL1Size,
                        para.kL1StepSize, para.k, para.kScale, scaleOffsetL1B, bufParam);
    }

    if (kL1 + 1 < kL1Loops) {
        if constexpr (!hasL1ALoaded) {
            LoadL1A<T, true>(tensorAGm[(kL1 + 1) * para.kL1StepSize], para.m, para.kL1StepSize, para.k, bufParam);
        }
        LoadL1B<T, DataFormat::NZ, true>(tensorBGm[para.k * nL1Offset + (kL1 + 1) * kOffesetUnit], nL1Size, para.n,
                                         para.kL1StepSize, para.k, bufParam);
    }
}

/**
 * @brief B矩阵数据加载到L1缓冲区的辅助函数
 * @tparam T B张量的数据类型
 * @tparam isContinuousCopy 是否使用连续复制模式
 * @param bL1 B矩阵L1张量
 * @param tensorBGm B矩阵GM张量
 * @param kL1StepSize L1中K维度的步长大小
 * @param subNL1SplitSize L1中N维度的分割大小
 * @param para MMParams参数
 * @param bufParam 缓冲区管理参数
 * @param nL1 当前N L1迭代
 * @param kL1 当前K L1迭代
 * @param kOffesetUnit K维度偏移单位
 */
template <typename T, bool isContinuousCopy>
__aicore__ inline void LoadL1BGroupCompute(LocalTensor<T> &bL1, const GlobalTensor<T> &tensorBGm,
                                           uint32_t subNL1SplitSize, const MMParams &para, int64_t nL1, uint32_t kL1,
                                           uint32_t kOffesetUnit)
{
    int64_t tensorBGmOffset = para.k * nL1 * para.baseN + kL1 * kOffesetUnit;
    auto tensorBGmForL1 = tensorBGm[tensorBGmOffset];
    if constexpr (isContinuousCopy) {
        CopyNZGmToL1(bL1, tensorBGmForL1, para.kL1StepSize, subNL1SplitSize, para.k);
    } else {
        // 每次搬运两块K*64拼接成K*128
        CopyNZGmToL1(bL1, tensorBGmForL1, para.kL1StepSize, subNL1SplitSize >> 1, para.k);
        LocalTensor<T> b2L1 = bL1[(para.kL1StepSize * subNL1SplitSize) >> 1];
        auto tensorB2GmForL1 = tensorBGmForL1[para.k * DIM_HEAD_SIZE_QCQR];
        CopyNZGmToL1(b2L1, tensorB2GmForL1, para.kL1StepSize, subNL1SplitSize >> 1, para.k);
    }
}

/**
 * @brief 执行矩阵乘法的K L0循环计算
 * @tparam T A和B矩阵的数据类型
 * @tparam O_L0C L0C缓存中的数据类型
 * @tparam enUnitFlag 是否启用unitFlag功能
 * @param cL0 L0缓存中的C矩阵张量
 * @param aL1 L1缓存中的A矩阵张量
 * @param bL1 L1缓存中的B矩阵张量
 * @param localTensors 矩阵乘法所需的本地张量对象
 * @param bufParam L1缓存管理的缓冲区参数对象
 * @param kL1 当前K L1循环迭代次数
 * @param kL1Loops K L1循环的总次数
 * @param stepK K L0步数
 * @param aOffsetUnit A矩阵L0寻址的偏移单位
 * @param bOffsetUnit B矩阵L0寻址的偏移单位
 * @param mmadParams 矩阵乘法参数
 * @param aOffset A矩阵计算的起始偏移
 * @param bOffset B矩阵计算的起始偏移
 */
template <typename T, typename O, typename S, typename O_L0C, bool enUnitFlag>
__aicore__ inline void MatmulL1(const LocalTensor<O_L0C> &cL0, const LocalTensor<T> &aL1, const LocalTensor<T> &bL1,
                                const mmLocalTensors<T, O_L0C> &localTensors, MMBufParams &bufParam,
                                const MMParams &para, uint32_t kL1, uint32_t kL1Loops, MmadParams &mmadParams,
                                int64_t &aOffset)
{
    uint32_t mSize = Align(para.m, BLOCK_CUBE_SIZE);
    uint64_t weightSizeL1B = L1_B_SIZE / 2 / sizeof(T); // 2表示scale起始地址固定从L1B上ping的64k开始
    uint64_t scaleASize = mSize * para.kScale;          // BS*224
    uint32_t bOffset = 0;
    uint32_t aOffsetUnit = mSize * para.baseK;
    uint32_t bOffsetUnit = GetC0Num<T>() * para.baseK;

    uint32_t aScaleOffset = kL1 * para.kScale / kL1Loops * BLOCK_CUBE_SIZE;
    uint32_t bScaleOffset = kL1 * para.kScale / kL1Loops * BLOCK_CUBE_SIZE;
    uint32_t aScaleOffsetUnit = para.baseK / BYTE_BLOCK * BLOCK_CUBE_SIZE;
    uint32_t bScaleOffsetUnit = para.baseK / BYTE_BLOCK * BLOCK_CUBE_SIZE;
    const LocalTensor<T> scaleALocalTensor = localTensors.bL1Tensor[weightSizeL1B];
    const LocalTensor<T> scaleBLocalTensor = scaleALocalTensor[scaleASize];

    for (int64_t kL0Loops = 0; kL0Loops < para.stepK; kL0Loops++) {
        mmadParams.cmatrixInitVal = ((kL1 == 0) && (kL0Loops == 0));
        if constexpr (enUnitFlag) {
            mmadParams.unitFlag =
                (kL1 == kL1Loops - 1) && (kL0Loops == para.stepK - 1) ? UNIT_FLAG_SET : UNIT_FLAG_CHECK;
        }
        MatmulL0<T, O_L0C, S>(bufParam, aL1[aOffset], bL1[bOffset], localTensors.aL0Tensor, localTensors.bL0Tensor, cL0,
                              mmadParams, para.kL1StepSize, para.k, scaleALocalTensor[aScaleOffset],
                              scaleBLocalTensor[bScaleOffset]);
        aOffset += aOffsetUnit; // 16(BS)*256=4096
        bOffset += bOffsetUnit; // 32(Get<T>)*256=8192
        if constexpr (std::is_same<T, FP8E4M3>::value && std::is_same<S, fp8_e8m0_t>::value) {
            aScaleOffset += aScaleOffsetUnit; // 16(BS)*baseK/32=16(BS)*8=128
            bScaleOffset += bScaleOffsetUnit; // baseK/32*baseN=8*64
        }
    }
}

/**
 * @brief MatmulSplitK 通过切分K轴，进行L1的数据管理和矩阵乘运算; 用于mmCq, mmCKvKr和mmQcQr。
 * @param tensorAGm A矩阵在GM的位置
 * @param tensorBGm B矩阵在GM的位置
 * @param tensorCGm C矩阵在GM的位置
 * @param para 表示matmul形状信息的结构体参数
 * @param bufParam 管理L1 buffer地址和同步计数的结构体参数
 * @param nL1Offset 本次计算中，L1 buffer内B矩阵在n轴上的偏移
 * @param nL1Size 本次计算中，L1 buffer内B矩阵在n轴上的计算量
 * @param tensorAScaleGm AScale矩阵在GM的位置
 * @param tensorBScaleGm BScale矩阵在GM的位置
 */
template <typename T, typename O, typename S, bool hasL1ALoaded = false, bool scaleSrcPadFlag = false,
          bool enUnitFlag = false, DataFormat bLoadFormat = DataFormat::NZ>
__aicore__ inline void
MatmulSplitK(const GlobalTensor<O> &tensorCGm, const GlobalTensor<T> &tensorAGm, const GlobalTensor<T> &tensorBGm,
             const MMParams &para, MMBufParams &bufParam, const uint32_t nL1Offset, const uint32_t nL1Size,
             const GlobalTensor<S> &tensorAScaleGm = {}, const GlobalTensor<S> &tensorBScaleGm = {})
{
    using O_L0C = typename std::conditional<std::is_same<T, int8_t>::value, int32_t, float>::type;

    // 全局L1管理
    uint32_t kOffesetUnit = para.kL1StepSize * GetC0Num<T>();
    uint32_t mSize = Align(para.m, BLOCK_CUBE_SIZE);
    uint32_t kL1Loops = CeilDivT(para.k, para.kL1StepSize);

    mmLocalTensors<T, O_L0C> localTensors;
    localTensors.Init(bufParam);

    LocalTensor<T> aL1, bL1;

    if constexpr (hasL1ALoaded) {
        aL1 = localTensors.aL1Tensor[(bufParam.aL1BufIter & 1u) * L1_A_SIZE / sizeof(T)];
    }

    WaitFlag<HardEvent::FIX_M>(L0C_EVENT0 + (bufParam.cL0BufIter & 1u));
    LocalTensor<O_L0C> cL0 = localTensors.cL0Tensor[(bufParam.cL0BufIter & 1u) * (L0C_PP_SIZE / sizeof(O_L0C))];

    MmadParams mmadParams = MmadParams(mSize, nL1Size, para.baseK, UNIT_FLAG_DISABLE, false, true);

    int64_t aOffset = 0;
    if constexpr (std::is_same<T, FP8E4M3>::value && std::is_same<S, fp8_e8m0_t>::value) {
        WaitFlag<HardEvent::MTE1_MTE2>(SCALE_EVENT);
    }

    for (uint32_t kL1 = 0; kL1 < kL1Loops; kL1++) {
        // Load data from global memory to L1 cache
        if constexpr (std::is_same<T, FP8E4M3>::value && std::is_same<S, fp8_e8m0_t>::value) {
            LoadL1ABAndScale<T, S, hasL1ALoaded, scaleSrcPadFlag>(tensorAGm, tensorBGm, tensorAScaleGm, tensorBScaleGm,
                                                                  kL1, kL1Loops, para, nL1Offset, nL1Size, kOffesetUnit,
                                                                  bufParam);
        } else {
            LoadL1AB<T, hasL1ALoaded, bLoadFormat>(tensorAGm, tensorBGm, kL1, kL1Loops, para, nL1Offset, nL1Size,
                                                   kOffesetUnit, bufParam);
        }

        WaitFlag<HardEvent::MTE2_MTE1>(B_EVENT0 + (bufParam.bL1BufIter & 1u));
        bL1 = localTensors.bL1Tensor[(bufParam.bL1BufIter & 1u) * L1_B_SIZE / sizeof(T)];
        if constexpr (!hasL1ALoaded) {
            WaitFlag<HardEvent::MTE2_MTE1>(A_EVENT0 + (bufParam.aL1BufIter & 1u));
            aL1 = localTensors.aL1Tensor[(bufParam.aL1BufIter & 1u) * L1_A_SIZE / sizeof(T)];
            aOffset = 0;
        }

        // Perform core K L0 loops computation
        MatmulL1<T, O, S, O_L0C, enUnitFlag>(cL0, aL1, bL1, localTensors, bufParam, para, kL1, kL1Loops, mmadParams,
                                             aOffset);

        SetFlag<HardEvent::MTE1_MTE2>(B_EVENT0 + (bufParam.bL1BufIter & 1u));
        bufParam.bL1BufIter++;
        if constexpr (!hasL1ALoaded) {
            SetFlag<HardEvent::MTE1_MTE2>(A_EVENT0 + (bufParam.aL1BufIter & 1u));
            bufParam.aL1BufIter++;
        }
    }
    if constexpr (std::is_same<T, FP8E4M3>::value && std::is_same<S, fp8_e8m0_t>::value) {
        SetFlag<HardEvent::MTE1_MTE2>(SCALE_EVENT);
    }
    GetTensorC<T, O, O_L0C, enUnitFlag>(tensorCGm[nL1Offset], cL0, para.m, nL1Size, mSize, para.orgKc, bufParam);
    SetFlag<HardEvent::FIX_M>(L0C_EVENT0 + (bufParam.cL0BufIter & 1u));
    bufParam.cL0BufIter++;
}

/**
 * @brief MatmulFullLoad A，B矩阵能够全载L1时的矩阵运算，用于mmQn;
 * @param tensorAGm A矩阵在GM的位置
 * @param tensorBGm B矩阵在GM的位置
 * @param tensorCGm C矩阵在GM的位置
 * @param para 表示matmul形状信息的结构体参数
 * @param bufParam 管理L1 buffer地址和同步计数的结构体参数
 */
template <typename T, typename O, typename S, bool hasL1BLoaded = false, bool enUnitFlag = false>
__aicore__ inline void MatmulFullLoad(const GlobalTensor<O> &tensorCGm, const GlobalTensor<T> &tensorAGm,
                                      const GlobalTensor<T> &tensorBGm, const MMParams &para, MMBufParams &bufParam)
{
    using O_L0C = typename std::conditional<std::is_same<T, int8_t>::value, int32_t, float>::type;
    uint32_t mSize = Align(para.m, BLOCK_CUBE_SIZE);
    constexpr uint32_t aL1Offset = L1_A_SIZE / sizeof(T);
    constexpr uint32_t bL1Offset = L1_B_SIZE / sizeof(T);

    mmLocalTensors<T, O_L0C> localTensors;
    localTensors.Init(bufParam);

    LoadL1A(tensorAGm, para.m, para.k, para.orgKa, bufParam);
    WaitFlag<HardEvent::MTE2_MTE1>(A_EVENT0 + (bufParam.aL1BufIter & 1u));
    LocalTensor<T> aL1 = localTensors.aL1Tensor[(bufParam.aL1BufIter & 1u) * aL1Offset];

    if constexpr (!hasL1BLoaded) {
        LoadL1B<T, DataFormat::ND, false>(tensorBGm, para.n, para.n, para.k, para.k, bufParam);
        WaitFlag<HardEvent::MTE2_MTE1>(B_EVENT0 + (bufParam.bL1BufIter & 1u));
    }
    LocalTensor<T> bL1 = localTensors.bL1Tensor[(bufParam.bL1BufIter & 1u) * bL1Offset];

    uint32_t nSplitSize = 128;
    uint32_t nSplitSizeAct = nSplitSize;
    uint32_t nloops = CeilDivT(para.n, nSplitSize);

    MmadParams mmadParams = MmadParams(mSize, nSplitSizeAct, para.k, UNIT_FLAG_DISABLE, false, true);

    for (uint32_t n = 0; n < nloops; n++) {
        if (n == nloops - 1) {
            nSplitSizeAct = para.n - (nloops - 1) * nSplitSize;
            mmadParams.n = nSplitSizeAct;
        }
        // Perform matrix multiplication computation for current N-dimension split
        if constexpr (enUnitFlag) {
            mmadParams.unitFlag = UNIT_FLAG_SET;
        }
        WaitFlag<HardEvent::FIX_M>(L0C_EVENT0 + (bufParam.cL0BufIter & 1u));
        LocalTensor<O_L0C> cL0 = localTensors.cL0Tensor[(bufParam.cL0BufIter & 1u) * (L0C_PP_SIZE / sizeof(O_L0C))];
        MatmulL0<T, O_L0C, S>(bufParam, aL1, bL1[para.k * nSplitSize * n], localTensors.aL0Tensor,
                              localTensors.bL0Tensor, cL0, mmadParams, para.kL1StepSize);
        GetTensorC<T, O, O_L0C, enUnitFlag>(tensorCGm[n * nSplitSize], cL0, para.m, nSplitSizeAct, mSize, para.orgKc,
                                            bufParam);
        SetFlag<HardEvent::FIX_M>(L0C_EVENT0 + (bufParam.cL0BufIter & 1u));
        bufParam.cL0BufIter++;
    }

    SetFlag<HardEvent::MTE1_MTE2>(B_EVENT0 + (bufParam.bL1BufIter & 1u));
    bufParam.bL1BufIter++;
    SetFlag<HardEvent::MTE1_MTE2>(A_EVENT0 + (bufParam.aL1BufIter & 1u));
    bufParam.aL1BufIter++;
}

/**
 * @brief MatmulGroupComputeAFullLoad
 * A矩阵能够全载，通过切分K轴，进行L1的数据管理和矩阵乘运算，仅适用于算力分组场景下的mmQc和mmQr;
 * @param tensorAGm A矩阵在GM的位置
 * @param tensorBGm B矩阵在GM的位置
 * @param tensorCGm C矩阵在GM的位置
 * @param para 表示matmul形状信息的结构体参数
 * @param bufParam 管理L1 buffer地址和同步计数的结构体参数
 */
template <typename T, typename O, typename S, bool isContinuousCopy = true, bool enUnitFlag = false>
__aicore__ inline void MatmulGroupComputeAFullLoad(const GlobalTensor<O> &tensorCGm, const GlobalTensor<T> &tensorAGm,
                                                   const GlobalTensor<T> &tensorBGm, const MMParams &para,
                                                   MMBufParams &bufParam)
{
    using O_L0C = typename std::conditional<std::is_same<T, int8_t>::value, int32_t, float>::type;
    // 全局L1管理
    uint32_t kOffesetUnit = para.kL1StepSize * GetC0Num<T>();
    uint32_t mSize = Align(para.m, BLOCK_CUBE_SIZE);
    uint32_t kL1Loops = CeilDivT(para.k, para.kL1StepSize);
    uint32_t nL1loops = CeilDivT(para.n, para.baseN);

    mmLocalTensors<T, O_L0C> localTensors;
    localTensors.Init(bufParam);

    WaitFlag<HardEvent::MTE1_MTE2>(A_EVENT0 + (bufParam.aL1BufIter & 1u));
    LocalTensor<T> aL1 = localTensors.aL1Tensor[(bufParam.aL1BufIter & 1u) * (L1_A_SIZE / sizeof(T))];
    CopyNDGmToL1(aL1, tensorAGm, para.m, para.k, para.k);
    SetFlag<HardEvent::MTE2_MTE1>(A_EVENT0 + (bufParam.aL1BufIter & 1u));
    WaitFlag<HardEvent::MTE2_MTE1>(A_EVENT0 + (bufParam.aL1BufIter & 1u));

    uint32_t subNL1SplitSize = para.baseN;

    MmadParams mmadParams = MmadParams(mSize, subNL1SplitSize, para.baseK, UNIT_FLAG_DISABLE, false, true);

    for (int64_t nL1 = 0; nL1 < nL1loops; nL1++) {
        if (nL1 == nL1loops - 1) {
            subNL1SplitSize = para.n - (nL1loops - 1) * para.baseN;
            mmadParams.n = subNL1SplitSize;
        }

        WaitFlag<HardEvent::FIX_M>(L0C_EVENT0 + (bufParam.cL0BufIter & 1u));
        LocalTensor<O_L0C> cL0 = localTensors.cL0Tensor[(bufParam.cL0BufIter & 1u) * (L0C_PP_SIZE / sizeof(O_L0C))];

        int64_t aOffset = 0;
        for (uint32_t kL1 = 0; kL1 < kL1Loops; kL1++) {
            WaitFlag<HardEvent::MTE1_MTE2>(B_EVENT0 + (bufParam.bL1BufIter & 1u));
            LocalTensor<T> bL1 = localTensors.bL1Tensor[(bufParam.bL1BufIter & 1u) * (L1_B_SIZE / sizeof(T))];
            LoadL1BGroupCompute<T, isContinuousCopy>(bL1, tensorBGm, subNL1SplitSize, para, nL1, kL1, kOffesetUnit);
            SetFlag<HardEvent::MTE2_MTE1>(B_EVENT0 + (bufParam.bL1BufIter & 1u));
            WaitFlag<HardEvent::MTE2_MTE1>(B_EVENT0 + (bufParam.bL1BufIter & 1u));

            // Perform core K L0 loops computation using shared function
            MatmulL1<T, O, S, O_L0C, enUnitFlag>(cL0, aL1, bL1, localTensors, bufParam, para, kL1, kL1Loops, mmadParams,
                                                 aOffset);
            SetFlag<HardEvent::MTE1_MTE2>(B_EVENT0 + (bufParam.bL1BufIter & 1u));
            bufParam.bL1BufIter++;
        }
        GetTensorC<T, O, O_L0C, enUnitFlag>(tensorCGm[nL1 * para.baseN], cL0, para.m, subNL1SplitSize, mSize,
                                            para.orgKc, bufParam);
        SetFlag<HardEvent::FIX_M>(L0C_EVENT0 + (bufParam.cL0BufIter & 1u));
        bufParam.cL0BufIter++;
    }
    SetFlag<HardEvent::MTE1_MTE2>(A_EVENT0 + (bufParam.aL1BufIter & 1u));
    bufParam.aL1BufIter++;
}

/**
 * @brief K-outer: 加载B矩阵到L1B（支持双缓冲slot选择）
 *        bScaleL1Offset: B-scale在slot内的bf16偏移 (需放在A-scale之后避免冲突)
 */
template <typename T, typename S, DataFormat loadFormat = DataFormat::NZ>
__aicore__ inline void LoadL1BOuter(const GlobalTensor<T> &tensorBGm, const GlobalTensor<S> &tensorBScaleGm,
                                    uint32_t nL1Size, uint32_t kL1StepSize, uint32_t kInput, uint32_t kScale,
                                    uint32_t bEventIdx, uint32_t bScaleL1Offset, MMBufParams &bufParam)
{
    LocalTensor<T> bL1Tensor;
    bL1Tensor.SetAddr(bufParam.bL1BufAddr);
    uint32_t bL1SlotSize = L1_B_SIZE / sizeof(T);

    WaitFlag<HardEvent::MTE1_MTE2>(B_EVENT0 + bEventIdx);
    LocalTensor<T> bL1 = bL1Tensor[bEventIdx * bL1SlotSize];

    if constexpr (std::is_same<T, FP8E4M3>::value && std::is_same<S, fp8_e8m0_t>::value) {
        CopyNZGmToL1(bL1, tensorBGm, kL1StepSize, nL1Size, kInput);

        LocalTensor<bfloat16_t> bL1ScaleTensor = bL1.template ReinterpretCast<bfloat16_t>();
        GlobalTensor<bfloat16_t> tensorScaleGmCast;
        tensorScaleGmCast.SetGlobalBuffer(((__gm__ bfloat16_t *)(tensorBScaleGm.GetPhyAddr())));
        Dn2NzParams dn2Nzparam;
        dn2Nzparam.dnNum = 1;
        uint32_t nValueB = CeilDivT(kScale, static_cast<uint32_t>(FP8_TWO));
        dn2Nzparam.nValue = nValueB;
        dn2Nzparam.dValue = nL1Size;
        dn2Nzparam.srcDnMatrixStride = 0;
        dn2Nzparam.srcDValue = nValueB;     // B-scale无padding, 保持原值
        dn2Nzparam.dstNzC0Stride = nValueB; // = nValue, NZ输出布局不变
        dn2Nzparam.dstNzNStride = 1;
        dn2Nzparam.dstNzMatrixStride = nValueB;
        DataCopy(bL1ScaleTensor[bScaleL1Offset], tensorScaleGmCast, dn2Nzparam);
    } else if constexpr (loadFormat == DataFormat::NZ) {
        CopyNZGmToL1(bL1, tensorBGm, kL1StepSize, nL1Size, kInput);
    } else {
        CopyNDGmToL1(bL1, tensorBGm, kInput, nL1Size, nL1Size);
    }
    SetFlag<HardEvent::MTE2_MTE1>(B_EVENT0 + bEventIdx);
}


/**
 * @brief K-outer: 主Matmul函数。A数据每kL1加载到L1A，A-scale复用原LoadL1AAndScale放L1B一次加载全量。
 *        B每nb双缓冲加载，B-scale紧随B数据之后。FixPipe使用AtomicAdd累加到GM。
 */
template <typename T, typename O, typename S>
__aicore__ inline void MatmulSplitMKOuter(const GlobalTensor<O> &tensorCGm, const GlobalTensor<T> &tensorAGm,
                                          const GlobalTensor<T> &tensorBGm, const MMParams &para, MMBufParams &bufParam,
                                          const GlobalTensor<S> &tensorAScaleGm, const GlobalTensor<S> &tensorBScaleGm)
{
    using O_L0C = typename std::conditional<std::is_same<T, int8_t>::value, int32_t, float>::type;

    uint32_t nBlocks = CeilDivT(para.n, para.baseN);
    uint32_t kL1Loops = CeilDivT(para.k, para.kL1StepSize);
    uint32_t kOffesetUnit = para.kL1StepSize * GetC0Num<T>();
    uint32_t mSizeAligned = Align(para.m, BLOCK_CUBE_SIZE);
    uint32_t aOffsetUnit = mSizeAligned * para.baseK;
    uint32_t bOffsetUnit = GetC0Num<T>() * para.baseK;
    uint32_t bL1SlotSize = L1_B_SIZE / sizeof(T);
    uint32_t scaleOffUnit = para.baseK / static_cast<uint32_t>(BYTE_BLOCK) * BLOCK_CUBE_SIZE;

    mmLocalTensors<T, O_L0C> localTensors;
    localTensors.Init(bufParam);

    // ──── 一次加载全量A-scale到L1B, 复用原LoadL1AAndScale的DN2NZ参数 ────
    LocalTensor<T> aScaleL1Base;
    uint32_t bScaleBf16Off = 0;
    uint32_t bScaleL1BaseOff = 0;
    if constexpr (std::is_same<T, FP8E4M3>::value && std::is_same<S, fp8_e8m0_t>::value) {
        uint64_t scaleAOff = L1_B_SIZE / 2 / sizeof(T);
        LocalTensor<bfloat16_t> bL1Bf16;
        bL1Bf16.SetAddr(bufParam.bL1BufAddr);
        GlobalTensor<bfloat16_t> scaleGmCast;
        scaleGmCast.SetGlobalBuffer(((__gm__ bfloat16_t *)(tensorAScaleGm.GetPhyAddr())));
        Dn2NzParams dn2nz;
        dn2nz.dnNum = 1;
        uint32_t nValue = CeilDivT(para.kScale, static_cast<uint32_t>(FP8_TWO));
        dn2nz.nValue = nValue;
        dn2nz.dValue = para.m;
        dn2nz.srcDnMatrixStride = 0;
        dn2nz.srcDValue = nValue;
        dn2nz.dstNzC0Stride = nValue;
        dn2nz.dstNzNStride = 1;
        dn2nz.dstNzMatrixStride = nValue;
        DataCopy(bL1Bf16[scaleAOff / FP8_TWO], scaleGmCast, dn2nz);
        aScaleL1Base = localTensors.bL1Tensor[scaleAOff];

        uint32_t aScaleTotalSize = mSizeAligned * para.kScale;
        bScaleL1BaseOff = static_cast<uint32_t>(scaleAOff) + aScaleTotalSize;
        bScaleBf16Off = bScaleL1BaseOff / FP8_TWO;
    }

    for (uint32_t kL1 = 0; kL1 < kL1Loops; ++kL1) {
        // ── 加载A数据到L1A slot0 ──
        LocalTensor<T> aL1;
        WaitFlag<HardEvent::MTE1_MTE2>(A_EVENT0);
        aL1 = localTensors.aL1Tensor;
        CopyNDGmToL1(aL1, tensorAGm[kL1 * para.kL1StepSize], para.m, para.kL1StepSize, para.k);
        SetFlag<HardEvent::MTE2_MTE1>(A_EVENT0);
        WaitFlag<HardEvent::MTE2_MTE1>(A_EVENT0);

        // A-scale起始偏移: 原MatmulL1公式
        uint32_t kL1ScaleOff = kL1 * para.kScale / kL1Loops * BLOCK_CUBE_SIZE;

        // ── 预加载 B[0] ──
        uint32_t subN0 = (nBlocks == 1) ? para.n : para.baseN;
        int64_t bGmOff0 = static_cast<int64_t>(kL1) * kOffesetUnit;
        if constexpr (std::is_same<T, FP8E4M3>::value && std::is_same<S, fp8_e8m0_t>::value) {
            int64_t bScaleOff0 = static_cast<int64_t>(0); // 全量K-step从0开始
            LoadL1BOuter<T, S>(tensorBGm[bGmOff0], tensorBScaleGm[bScaleOff0], subN0, para.kL1StepSize, para.k,
                               para.kScale, 0, bScaleBf16Off, bufParam);
        } else {
            LoadL1BOuter<T, S>(tensorBGm[bGmOff0], tensorBScaleGm, subN0, para.kL1StepSize, para.k, 0, 0, bScaleBf16Off,
                               bufParam);
        }

        // ── N分块: B双缓冲ping-pong ──
        for (uint32_t nb = 0; nb < nBlocks; ++nb) {
            uint32_t subN = (nb == nBlocks - 1) ? (para.n - nb * para.baseN) : para.baseN;
            uint32_t curSlot = nb & 1u;

            WaitFlag<HardEvent::MTE2_MTE1>(B_EVENT0 + curSlot);
            LocalTensor<T> bL1 = localTensors.bL1Tensor[curSlot * bL1SlotSize];
            LocalTensor<T> bScaleL1;
            if constexpr (std::is_same<T, FP8E4M3>::value && std::is_same<S, fp8_e8m0_t>::value) {
                bScaleL1 = localTensors.bL1Tensor[curSlot * bL1SlotSize + bScaleL1BaseOff];
            }

            // 预取 B[nb+1]
            if (nb + 1 < nBlocks) {
                uint32_t nextSlot = (nb + 1) & 1u;
                uint32_t subNx = ((nb + 1) == nBlocks - 1) ? (para.n - (nb + 1) * para.baseN) : para.baseN;
                int64_t bOffNext =
                    static_cast<int64_t>(para.k) * (nb + 1) * para.baseN + static_cast<int64_t>(kL1) * kOffesetUnit;
                if constexpr (std::is_same<T, FP8E4M3>::value && std::is_same<S, fp8_e8m0_t>::value) {
                    int64_t bScOffNext = static_cast<int64_t>((nb + 1) * para.baseN * para.kScale);
                    LoadL1BOuter<T, S>(tensorBGm[bOffNext], tensorBScaleGm[bScOffNext], subNx, para.kL1StepSize, para.k,
                                       para.kScale, nextSlot, bScaleBf16Off, bufParam);
                } else {
                    LoadL1BOuter<T, S>(tensorBGm[bOffNext], tensorBScaleGm, subNx, para.kL1StepSize, para.k, 0,
                                       nextSlot, bScaleBf16Off, bufParam);
                }
            }

            // ── L0C ──
            WaitFlag<HardEvent::FIX_M>(L0C_EVENT0 + (bufParam.cL0BufIter & 1u));
            LocalTensor<O_L0C> cL0 = localTensors.cL0Tensor[(bufParam.cL0BufIter & 1u) * (L0C_PP_SIZE / sizeof(O_L0C))];

            // ── stepK L0循环 ──
            MmadParams mmadParams = MmadParams(mSizeAligned, subN, para.baseK, UNIT_FLAG_DISABLE, false, true);
            int64_t aOffset = 0;
            uint32_t bOffset = 0;
            uint32_t aScaleOff = kL1ScaleOff;
            uint32_t bScaleOff = kL1ScaleOff;
            for (int64_t sk = 0; sk < para.stepK; sk++) {
                mmadParams.cmatrixInitVal = (sk == 0);
                MatmulL0<T, O_L0C, S>(bufParam, aL1[aOffset], bL1[bOffset], localTensors.aL0Tensor,
                                      localTensors.bL0Tensor, cL0, mmadParams, para.kL1StepSize, para.k,
                                      aScaleL1Base[aScaleOff], bScaleL1[bScaleOff]);
                aOffset += aOffsetUnit;
                bOffset += bOffsetUnit;
                aScaleOff += scaleOffUnit;
                bScaleOff += scaleOffUnit;
            }

            // ── FixPipe AtomicAdd ──
            SetFlag<HardEvent::M_FIX>(L0C_EVENT0 + (bufParam.cL0BufIter & 1u));
            WaitFlag<HardEvent::M_FIX>(L0C_EVENT0 + (bufParam.cL0BufIter & 1u));
            FixpipeParamsV220 fixParams;
            fixParams.nSize = subN;
            fixParams.mSize = para.m;
            fixParams.srcStride = mSizeAligned;
            fixParams.dstStride = para.orgKc;
            fixParams.ndNum = 1;
            fixParams.unitFlag = UNIT_FLAG_DISABLE;
            SetAtomicAdd<float>();
            Fixpipe(tensorCGm[nb * para.baseN], cL0, fixParams);
            SetAtomicNone();

            SetFlag<HardEvent::FIX_M>(L0C_EVENT0 + (bufParam.cL0BufIter & 1u));
            bufParam.cL0BufIter++;

            SetFlag<HardEvent::MTE1_MTE2>(B_EVENT0 + curSlot);
        }
        SetFlag<HardEvent::MTE1_MTE2>(A_EVENT0);
    }
}

} // namespace MlaProlog


#endif
