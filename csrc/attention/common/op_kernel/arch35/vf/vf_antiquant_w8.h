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
 * \file vf_antiquant_w8.h
 * \brief
 */
#ifndef VF_ANTIQUANT_W8
#define VF_ANTIQUANT_W8

#include "kernel_tensor.h"

namespace FaVectorApi {
// w8转Q_T
static constexpr MicroAPI::CastTrait castTrait = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                  MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
// fp32->Q_T
static constexpr MicroAPI::CastTrait castTrait0 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                   MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
// fp16 -> bf16                                                      
static constexpr MicroAPI::CastTrait castTrait1 = {MicroAPI::RegLayout::UNKNOWN, MicroAPI::SatMode::UNKNOWN,
                                                      MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
// fp8->fp32
static constexpr MicroAPI::CastTrait castTraitFp8_1 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                       MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
// fp8->fp32
static constexpr MicroAPI::CastTrait castTraitFp8_2 = {MicroAPI::RegLayout::ONE, MicroAPI::SatMode::UNKNOWN,
                                                       MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
// fp32->fp16
static constexpr MicroAPI::CastTrait castTraitFp8_3 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                       MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
// fp32->fp16
static constexpr MicroAPI::CastTrait castTraitFp8_4 = {MicroAPI::RegLayout::ONE, MicroAPI::SatMode::NO_SAT,
                                                       MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

template <typename Q_T, typename KV_T, typename ANTIQ_PARAMS_T, uint32_t baseSize, bool hasOffset = false>
__simd_vf__ void AntiquantVFImplW8Nz(__ubuf__ uint8_t* ubSrcAddr, __ubuf__ Q_T* ubDstAddr, __ubuf__ Q_T* ubOffsetAddr, 
                                    __ubuf__ Q_T* ubScaleAddr, uint32_t dealRowCount)
{
  MicroAPI::RegTensor<KV_T> vKvData;
  MicroAPI::RegTensor<Q_T> vOffset;
  MicroAPI::RegTensor<Q_T> vScale;
  MicroAPI::RegTensor<Q_T> vRes;
  MicroAPI::RegTensor<half> vCastFp16Res;

  MicroAPI::MaskReg kvTypeMaskAll = MicroAPI::CreateMask<KV_T, MicroAPI::MaskPattern::ALL>();
  MicroAPI::MaskReg qTypeMaskAll = MicroAPI::CreateMask<Q_T, MicroAPI::MaskPattern::ALL>(); // Q_T 所有元素（共128个）

  // UB总共dealRowCount行 * baseSize列，每次处理8行 * 16列 = 128个元素
  uint32_t rowBaseSize = 8; // 8行
  uint32_t colBaseSize = 16; // 16列
  uint32_t dealBaseNum = 128; // 128个元素
  uint32_t colDstStride = dealRowCount * colBaseSize;
  uint32_t colSrcStride = (dealRowCount * colBaseSize + 31) / 32 * 32; // 32B对齐
  const uint16_t rowLoopCnt = static_cast<uint16_t>((dealRowCount + rowBaseSize - 1) / rowBaseSize); // 8行对齐
  const uint16_t colLoopCnt = static_cast<uint16_t>(baseSize / colBaseSize);

  for (uint16_t colLoopIdx = 0; colLoopIdx < colLoopCnt; colLoopIdx++) {
    if constexpr (hasOffset) {
      MicroAPI::LoadAlign<Q_T, MicroAPI::LoadDist::DIST_BLK>(vOffset, ubOffsetAddr + colBaseSize * colLoopIdx);
    }
    MicroAPI::LoadAlign<Q_T, MicroAPI::LoadDist::DIST_BLK>(vScale, ubScaleAddr + colBaseSize * colLoopIdx);

    // #pragma unroll(4)
    for (uint16_t rowLoop = 0; rowLoop < rowLoopCnt; rowLoop++) {
      uint16_t rowLoopIdx = rowLoopCnt - 1 - rowLoop;
      __ubuf__ Q_T* ubDstAddrTmp = ubDstAddr + dealBaseNum * rowLoopIdx + colDstStride * colLoopIdx;
      __ubuf__ uint8_t* ubSrcTemp = ubSrcAddr + colSrcStride * colLoopIdx + dealBaseNum * rowLoopIdx;;
      MicroAPI::LoadAlign<uint8_t, MicroAPI::LoadDist::DIST_UNPACK_B8>(
          (MicroAPI::RegTensor<uint8_t>&)vKvData, ubSrcTemp);
      if constexpr (std::is_same<Q_T, bfloat16_t>::value) {
        MicroAPI::Cast<half, KV_T, castTrait>(vCastFp16Res, vKvData, kvTypeMaskAll);
        MicroAPI::Cast<Q_T, half, castTrait1>(vRes, vCastFp16Res, kvTypeMaskAll);
      } else {
        MicroAPI::Cast<Q_T, KV_T, castTrait>(vRes, vKvData, kvTypeMaskAll);
      }
      if constexpr (hasOffset) {
        MicroAPI::Add<Q_T, MicroAPI::MaskMergeMode::ZEROING>(vRes, vRes, vOffset, qTypeMaskAll);
      }
      MicroAPI::Mul<Q_T, MicroAPI::MaskMergeMode::ZEROING>(vRes, vRes, vScale, qTypeMaskAll);
      MicroAPI::StoreAlign<Q_T, MicroAPI::StoreDist::DIST_NORM_B16>(ubDstAddrTmp, vRes, qTypeMaskAll);
    }
  }
}

template <typename Q_T, typename KV_T, typename ANTIQ_PARAMS_T, uint32_t baseSize, bool hasOffset = false>
__aicore__ inline void AntiquantVFW8Nz(LocalTensor<KV_T>& antiqInUb, LocalTensor<Q_T>& antiqResUb,
                                           LocalTensor<ANTIQ_PARAMS_T>& antiqOffsetUb,
                                           LocalTensor<ANTIQ_PARAMS_T>& antiqScaleUb,
                                           uint32_t dealRowCount) {
  static_assert(baseSize % 16 == 0);
  static_assert(IsSameType<KV_T, int8_t>::value || IsSameType<KV_T, hifloat8_t>::value,
                "antiquant w8, KV_T must be int8_t or hifloat8_t");
  __ubuf__ uint8_t* ubSrcAddr = (__ubuf__ uint8_t*)(antiqInUb.GetPhyAddr());
  __ubuf__ Q_T* ubDstAddr = (__ubuf__ Q_T*)(antiqResUb.GetPhyAddr());
  __ubuf__ Q_T* ubOffsetAddr = (__ubuf__ Q_T*)antiqOffsetUb.GetPhyAddr();
  __ubuf__ Q_T* ubScaleAddr = (__ubuf__ Q_T*)antiqScaleUb.GetPhyAddr();

  AntiquantVFImplW8Nz<Q_T, KV_T, ANTIQ_PARAMS_T, baseSize, hasOffset>(
    ubSrcAddr, ubDstAddr, ubOffsetAddr, ubScaleAddr, dealRowCount);
}

template <typename Q_T, typename KV_T, uint32_t baseSize>
__simd_vf__ void AntiquantVFImplFp8Nz(__ubuf__ uint8_t* ubSrcAddr, __ubuf__ Q_T* ubDstAddr, __ubuf__ Q_T* ubScaleAddr, 
                                      uint32_t dealRowCount)
{
  MicroAPI::RegTensor<KV_T> vKvData;
  MicroAPI::RegTensor<float> vCastFp32Res0;
  MicroAPI::RegTensor<float> vCastFp32Res1;
  MicroAPI::RegTensor<Q_T> vCastRes0;
  MicroAPI::RegTensor<Q_T> vCastRes1;
  MicroAPI::RegTensor<Q_T> vScale;
  MicroAPI::RegTensor<Q_T> vRes;

  MicroAPI::MaskReg kvTypeMaskAll = MicroAPI::CreateMask<KV_T, MicroAPI::MaskPattern::ALL>();
  MicroAPI::MaskReg qTypeMaskAll = MicroAPI::CreateMask<Q_T, MicroAPI::MaskPattern::ALL>();

  // UB总共dealRowCount行 * baseSize列，每次处理8行 * 16列 = 128个元素
  uint32_t rowBaseSize = 8; // 8行
  uint32_t colBaseSize = 16; // 16列
  uint32_t dealBaseNum = 128; // 128个元素
  uint32_t colDstStride = colBaseSize * dealRowCount;
  uint32_t colSrcStride = (dealRowCount * colBaseSize + 31) / 32 * 32; // 32B对齐
  const uint16_t colLoopCnt = static_cast<uint16_t>(baseSize / colBaseSize);
  const uint16_t rowLoopCnt = static_cast<uint16_t>((dealRowCount + rowBaseSize - 1) / rowBaseSize);

  for (uint16_t colLoopIdx = 0; colLoopIdx < colLoopCnt; colLoopIdx++) {
    // 加载 scale
    MicroAPI::LoadAlign<Q_T, MicroAPI::LoadDist::DIST_BLK>(vScale, ubScaleAddr + colLoopIdx * colBaseSize);

    for (uint16_t rowLoop = 0; rowLoop < rowLoopCnt; rowLoop++) {
      uint16_t rowLoopIdx = rowLoopCnt - 1 - rowLoop;
      __ubuf__ Q_T* ubDstAddrTmp = ubDstAddr + colDstStride * colLoopIdx + dealBaseNum * rowLoopIdx;
      __ubuf__ uint8_t* ubSrcTemp = ubSrcAddr + colSrcStride * colLoopIdx + dealBaseNum * rowLoopIdx;;
      MicroAPI::LoadAlign<uint8_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(
          (MicroAPI::RegTensor<uint8_t>&)vKvData, ubSrcTemp);

      // cast操作, Fp8->Fp32
      MicroAPI::Cast<float, KV_T, castTraitFp8_1>(vCastFp32Res0, vKvData, kvTypeMaskAll);
      MicroAPI::Cast<float, KV_T, castTraitFp8_2>(vCastFp32Res1, vKvData, kvTypeMaskAll);
      // cast操作, Fp32->Fp16/Bf16
      MicroAPI::Cast<Q_T, float, castTraitFp8_3>(vCastRes0, vCastFp32Res0, kvTypeMaskAll);
      MicroAPI::Cast<Q_T, float, castTraitFp8_4>(vCastRes1, vCastFp32Res1, kvTypeMaskAll);
      MicroAPI::Or<uint16_t, MicroAPI::MaskMergeMode::ZEROING>((MicroAPI::RegTensor<uint16_t>&)vCastRes0,
                                                              (MicroAPI::RegTensor<uint16_t>&)vCastRes0,
                                                              (MicroAPI::RegTensor<uint16_t>&)vCastRes1, kvTypeMaskAll);
      MicroAPI::Mul<Q_T, MicroAPI::MaskMergeMode::ZEROING>(vRes, vCastRes0, vScale, qTypeMaskAll);
      // 将输出结果copy到UB
      MicroAPI::StoreAlign<Q_T, MicroAPI::StoreDist::DIST_NORM_B16>(ubDstAddrTmp, vRes, qTypeMaskAll);
    }
  }
}

template <typename Q_T, typename KV_T, bool hasOffset>
__simd_callee__ static inline void W8NzQuantAndStore(MicroAPI::RegTensor<KV_T>& vKvData,
                                                      MicroAPI::RegTensor<half>& vCastFp16Res,
                                                      MicroAPI::RegTensor<Q_T>& vRes,
                                                      __ubuf__ uint8_t* ubSrc,
                                                      MicroAPI::RegTensor<Q_T>& vOffset,
                                                      MicroAPI::RegTensor<Q_T>& vScale,
                                                      __ubuf__ Q_T* ubDst,
                                                      MicroAPI::MaskReg& kvTypeMaskAll,
                                                      MicroAPI::MaskReg& qTypeMaskAll)
{
  MicroAPI::LoadAlign<uint8_t, MicroAPI::LoadDist::DIST_UNPACK_B8>(
      (MicroAPI::RegTensor<uint8_t>&)vKvData, ubSrc);
  if constexpr (std::is_same<Q_T, bfloat16_t>::value) {
    MicroAPI::Cast<half, KV_T, castTrait>(vCastFp16Res, vKvData, kvTypeMaskAll);
    MicroAPI::Cast<Q_T, half, castTrait1>(vRes, vCastFp16Res, kvTypeMaskAll);
  } else {
    MicroAPI::Cast<Q_T, KV_T, castTrait>(vRes, vKvData, kvTypeMaskAll);
  }
  if constexpr (hasOffset) {
    MicroAPI::Add<Q_T, MicroAPI::MaskMergeMode::ZEROING>(vRes, vOffset, vRes, qTypeMaskAll);
  }
  MicroAPI::Mul<Q_T, MicroAPI::MaskMergeMode::ZEROING>(vRes, vRes, vScale, qTypeMaskAll);
  MicroAPI::StoreAlign<Q_T, MicroAPI::StoreDist::DIST_NORM_B16>(ubDst, vRes, qTypeMaskAll);
}

template <typename Q_T, typename KV_T, typename ANTIQ_PARAMS_T, uint32_t baseSize, bool hasOffset = false>
__simd_vf__ void AntiquantVFImplW8PerTokenNz(__ubuf__ uint8_t* ubSrcAddr, __ubuf__ Q_T* ubDstAddr,
                                             __ubuf__ Q_T* ubOffsetAddr, __ubuf__ Q_T* ubScaleAddr,
                                             uint32_t dealRowCount)
{
  MicroAPI::RegTensor<KV_T> vKvData;
  MicroAPI::RegTensor<Q_T> vOffsetFirst;
  MicroAPI::RegTensor<Q_T> vOffsetBack;
  MicroAPI::RegTensor<Q_T> vScaleFirst;
  MicroAPI::RegTensor<Q_T> vScaleBack;
  MicroAPI::RegTensor<Q_T> vRes;
  MicroAPI::RegTensor<half> vCastFp16Res;

  MicroAPI::MaskReg kvTypeMaskAll = MicroAPI::CreateMask<KV_T, MicroAPI::MaskPattern::ALL>();
  MicroAPI::MaskReg qTypeMaskAll = MicroAPI::CreateMask<Q_T, MicroAPI::MaskPattern::ALL>(); // Q_T 所有元素（共128个）

  // UB总共dealRowCount行 * baseSize列，每次处理8行 * 16列 = 128个元素
  const uint32_t rowBaseSize = 8; // 8行
  const uint32_t colBaseSize = 16; // 16列
  const uint32_t dealBaseNum = 128; // 128个元素
  const uint32_t doubleRowBaseSize = 16; // 每16行交替，防止bank冲突

  const uint32_t rowStride = doubleRowBaseSize * colBaseSize; 
  const uint32_t colDstStride = dealRowCount * colBaseSize;
  const uint32_t colSrcStride = (dealRowCount * colBaseSize + 31) >> 5U << 5U; // 32B对齐
  const uint16_t rowLoopCnt = static_cast<uint16_t>((dealRowCount + doubleRowBaseSize - 1) / doubleRowBaseSize); // 16行对齐
  const uint16_t colLoopCnt = static_cast<uint16_t>(baseSize / colBaseSize);

  for (uint16_t rowLoop = 0; rowLoop < rowLoopCnt; rowLoop++) {
    uint16_t rowLoopIdx = rowLoopCnt - 1 - rowLoop;
    __ubuf__ Q_T* ubOffsetAddrTmp = ubOffsetAddr + doubleRowBaseSize * rowLoopIdx;
    __ubuf__ Q_T* ubScaleAddrTmp = ubScaleAddr + doubleRowBaseSize * rowLoopIdx;

    if constexpr (hasOffset) {
      MicroAPI::LoadAlign<Q_T, MicroAPI::LoadDist::DIST_E2B_B16>(vOffsetFirst, ubOffsetAddrTmp);
      MicroAPI::LoadAlign<Q_T, MicroAPI::LoadDist::DIST_E2B_B16>(vOffsetBack, rowBaseSize + ubOffsetAddrTmp);
    }
    MicroAPI::LoadAlign<Q_T, MicroAPI::LoadDist::DIST_E2B_B16>(vScaleFirst, ubScaleAddrTmp);
    MicroAPI::LoadAlign<Q_T, MicroAPI::LoadDist::DIST_E2B_B16>(vScaleBack, rowBaseSize + ubScaleAddrTmp);
    for (uint16_t colLoopIdx = 0; colLoopIdx < colLoopCnt; colLoopIdx++) {
      __ubuf__ uint8_t* ubSrcTemp = ubSrcAddr + rowStride * rowLoopIdx + colSrcStride * colLoopIdx;
      __ubuf__ Q_T* ubDstAddrTmp = ubDstAddr + rowStride * rowLoopIdx + colDstStride * colLoopIdx;

      // 后半组
      W8NzQuantAndStore<Q_T, KV_T, hasOffset>(vKvData, vCastFp16Res, vRes, ubSrcTemp + dealBaseNum,
                                               vOffsetBack, vScaleBack, ubDstAddrTmp + dealBaseNum,
                                               kvTypeMaskAll, qTypeMaskAll);

      // 前半组
      W8NzQuantAndStore<Q_T, KV_T, hasOffset>(vKvData, vCastFp16Res, vRes, ubSrcTemp,
                                               vOffsetFirst, vScaleFirst, ubDstAddrTmp,
                                               kvTypeMaskAll, qTypeMaskAll);
    }
  }
}

template <typename Q_T, typename KV_T, typename ANTIQ_PARAMS_T, uint32_t baseSize, bool hasOffset = false>
__aicore__ inline void AntiquantVFW8PerTokenNz(LocalTensor<KV_T>& antiqInUb, LocalTensor<Q_T>& antiqResUb,
                                               LocalTensor<ANTIQ_PARAMS_T>& antiqOffsetUb,
                                               LocalTensor<ANTIQ_PARAMS_T>& antiqScaleUb,
                                               uint32_t dealRowCount) {
  static_assert(baseSize % 16 == 0);
  static_assert(IsSameType<KV_T, int8_t>::value, "antiquant w4 PerToken, KV_T must be int4_t");
  __ubuf__ uint8_t* ubSrcAddr = (__ubuf__ uint8_t*)(antiqInUb.GetPhyAddr());
  __ubuf__ Q_T* ubDstAddr = (__ubuf__ Q_T*)(antiqResUb.GetPhyAddr());
  __ubuf__ Q_T* ubOffsetAddr = (__ubuf__ Q_T*)antiqOffsetUb.GetPhyAddr();
  __ubuf__ Q_T* ubScaleAddr = (__ubuf__ Q_T*)antiqScaleUb.GetPhyAddr();
  AntiquantVFImplW8PerTokenNz<Q_T, KV_T, ANTIQ_PARAMS_T, baseSize, hasOffset>(
    ubSrcAddr, ubDstAddr, ubOffsetAddr, ubScaleAddr, dealRowCount);
}

template <typename Q_T, typename KV_T, uint32_t baseSize>
__aicore__ inline void AntiquantVFFp8Nz(LocalTensor<KV_T>& antiqInUb, LocalTensor<Q_T>& antiqResUb,
                                          LocalTensor<Q_T>& antiqScaleFp16Ub, uint32_t dealRowCount) {
  static_assert(baseSize % 16 == 0);
  static_assert(IsSameType<KV_T, fp8_e4m3fn_t>::value || IsSameType<KV_T, fp8_e5m2_t>::value,
                "antiquant w8, KV_T must be fp8_e4m3fn_t or fp8_e5m2_t");
  __ubuf__ uint8_t* ubSrcAddr = (__ubuf__ uint8_t*)(antiqInUb.GetPhyAddr());
  __ubuf__ Q_T* ubDstAddr = (__ubuf__ Q_T*)(antiqResUb.GetPhyAddr());
  __ubuf__ Q_T* ubScaleAddr = (__ubuf__ Q_T*)antiqScaleFp16Ub.GetPhyAddr();

  AntiquantVFImplFp8Nz<Q_T, KV_T, baseSize>(ubSrcAddr, ubDstAddr, ubScaleAddr, dealRowCount);
}

template <typename Q_T, typename KV_T, typename ANTIQ_PARAMS_T, uint32_t baseSize, bool hasOffset = false>
__simd_vf__ void AntiquantVFImplW8D64(__ubuf__ uint8_t* ubSrcAddr, __ubuf__ Q_T* ubDstAddr, __ubuf__ Q_T* ubDstAddr_, 
                                      __ubuf__ Q_T* ubOffsetAddr, __ubuf__ Q_T* ubScaleAddr, uint32_t dealRowCount)
{
  MicroAPI::RegTensor<KV_T> vKvData;
  MicroAPI::RegTensor<Q_T> vOffset;
  MicroAPI::RegTensor<Q_T> vScale;
  MicroAPI::RegTensor<Q_T> vRes;
  MicroAPI::RegTensor<half> vCastFp16Res;

  MicroAPI::MaskReg kvTypeMaskAll = MicroAPI::CreateMask<KV_T, MicroAPI::MaskPattern::ALL>();
  MicroAPI::MaskReg qTypeMaskLower64 = MicroAPI::CreateMask<Q_T, MicroAPI::MaskPattern::VL64>();
  MicroAPI::MaskReg qTypeMaskLower128 = MicroAPI::CreateMask<Q_T, MicroAPI::MaskPattern::VL128>();
  MicroAPI::MaskReg qTypeMaskAll = MicroAPI::CreateMask<Q_T, MicroAPI::MaskPattern::ALL>(); // Q_T 所有元素（共128个）
  MicroAPI::MaskReg qTypeMaskHigher64;
  MicroAPI::Xor(qTypeMaskHigher64, qTypeMaskLower64, qTypeMaskAll, qTypeMaskAll); // qTypeMaskAll与qTypeMaskLower64异或得到qTypeMaskHigher64

  uint32_t blockStride = 1 + dealRowCount;
  uint32_t repeatStride = 2;
  uint16_t loopCnt = static_cast<uint16_t>((dealRowCount + 1) / 2); // +1是为了兼容处理奇数行

  __ubuf__ uint8_t* ubSrcTemp = ubSrcAddr;
  if constexpr (hasOffset) {
    MicroAPI::LoadAlign<Q_T, MicroAPI::LoadDist::DIST_NORM>(vOffset, ubOffsetAddr);
  }
  MicroAPI::LoadAlign<Q_T, MicroAPI::LoadDist::DIST_NORM>(vScale, ubScaleAddr);

  // 对D64优化，相邻2行合并计算；
  for (uint16_t i = 0; i < loopCnt; i++) {
    MicroAPI::LoadAlign<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_UNPACK_B8>(
        (MicroAPI::RegTensor<uint8_t>&)vKvData, ubSrcTemp, baseSize * 2);

    if constexpr (std::is_same<Q_T, bfloat16_t>::value) {
      MicroAPI::Cast<half, KV_T, castTrait>(vCastFp16Res, vKvData, kvTypeMaskAll);
      MicroAPI::Cast<Q_T, half, castTrait1>(vRes, vCastFp16Res, kvTypeMaskAll);
    } else {
      MicroAPI::Cast<Q_T, KV_T, castTrait>(vRes, vKvData, kvTypeMaskAll);
    }

    if constexpr (hasOffset) {
      MicroAPI::Add<Q_T, MicroAPI::MaskMergeMode::ZEROING>(vRes, vRes, vOffset, qTypeMaskLower128);
    }
    MicroAPI::Mul<Q_T, MicroAPI::MaskMergeMode::ZEROING>(vRes, vRes, vScale, qTypeMaskLower128);

    MicroAPI::StoreAlign<Q_T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
        ubDstAddr, vRes, blockStride, repeatStride, qTypeMaskLower64);

    MicroAPI::StoreAlign<Q_T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
        ubDstAddr_, vRes, blockStride, repeatStride, qTypeMaskHigher64);
  }
}

template <typename Q_T, typename KV_T, typename ANTIQ_PARAMS_T,
  uint32_t baseSize, bool hasOffset = false>
__aicore__ inline void AntiquantVFW8D64(LocalTensor<KV_T>& antiqInUb, LocalTensor<Q_T>& antiqResUb,
                                            LocalTensor<ANTIQ_PARAMS_T>& antiqOffsetUb, LocalTensor<ANTIQ_PARAMS_T>& antiqScaleUb,
                                            uint32_t dealRowCount) {
  static_assert(baseSize == 64);
  static_assert(IsSameType<KV_T, int8_t>::value || IsSameType<KV_T, hifloat8_t>::value,
                "antiquant w8, KV_T must be int8_t or hifloat8_t");
  __ubuf__ uint8_t* ubSrcAddr = (__ubuf__ uint8_t*)(antiqInUb.GetPhyAddr());
  __ubuf__ Q_T* ubDstAddr = (__ubuf__ Q_T*)(antiqResUb.GetPhyAddr());
  __ubuf__ Q_T* ubDstAddr_ = ubDstAddr + 16 - (dealRowCount + 1) * 32 * 4 / 2;
  __ubuf__ Q_T* ubOffsetAddr = (__ubuf__ Q_T*)antiqOffsetUb.GetPhyAddr();
  __ubuf__ Q_T* ubScaleAddr = (__ubuf__ Q_T*)antiqScaleUb.GetPhyAddr();

  AntiquantVFImplW8D64<Q_T, KV_T, ANTIQ_PARAMS_T, baseSize, hasOffset>(
    ubSrcAddr, ubDstAddr, ubDstAddr_, ubOffsetAddr, ubScaleAddr, dealRowCount);
}

template <typename Q_T, typename ANTIQ_PARAMS_T>
__simd_callee__ static inline void LoadCastDupLowHighOr(
    MicroAPI::RegTensor<ANTIQ_PARAMS_T>& vParam, MicroAPI::UnalignRegForLoad& uReg,
    __ubuf__ ANTIQ_PARAMS_T*& ubAddr, MicroAPI::RegTensor<Q_T>& vParamFp16,
    MicroAPI::RegTensor<Q_T>& vParamFp16Low, MicroAPI::RegTensor<Q_T>& vParamFp16High,
    MicroAPI::MaskReg& maskOne, MicroAPI::MaskReg& qTypeMaskLower64,
    MicroAPI::MaskReg& qTypeMaskHigher64, MicroAPI::MaskReg& kvTypeMaskAll)
{
  MicroAPI::LoadUnAlign<ANTIQ_PARAMS_T>(vParam, uReg, ubAddr, 1); // 1表示ub自动往后偏移1个float
  MicroAPI::Cast<Q_T, ANTIQ_PARAMS_T, castTrait0>(vParamFp16, vParam, maskOne);
  MicroAPI::Duplicate<uint16_t, MicroAPI::HighLowPart::LOWEST, MicroAPI::MaskMergeMode::ZEROING>
    ((MicroAPI::RegTensor<uint16_t>&)vParamFp16Low, (MicroAPI::RegTensor<uint16_t>&)vParamFp16, qTypeMaskLower64);

  MicroAPI::LoadUnAlign<ANTIQ_PARAMS_T>(vParam, uReg, ubAddr, 1); // 1表示ub自动往后偏移1个float
  MicroAPI::Cast<Q_T, ANTIQ_PARAMS_T, castTrait0>(vParamFp16, vParam, maskOne);
  MicroAPI::Duplicate<uint16_t, MicroAPI::HighLowPart::LOWEST, MicroAPI::MaskMergeMode::ZEROING>
    ((MicroAPI::RegTensor<uint16_t>&)vParamFp16High, (MicroAPI::RegTensor<uint16_t>&)vParamFp16, qTypeMaskHigher64);
  MicroAPI::Or<uint16_t, MicroAPI::MaskMergeMode::ZEROING>((MicroAPI::RegTensor<uint16_t>&)vParamFp16,
                (MicroAPI::RegTensor<uint16_t>&)vParamFp16Low,
                (MicroAPI::RegTensor<uint16_t>&)vParamFp16High, kvTypeMaskAll);
}

template <typename Q_T, typename KV_T>
__simd_callee__ static inline void CastW8KvToRes(MicroAPI::RegTensor<Q_T>& vRes,
                                                  MicroAPI::RegTensor<KV_T>& vKvData,
                                                  MicroAPI::RegTensor<half>& vCastFp16Res,
                                                  MicroAPI::MaskReg& kvTypeMaskAll)
{
  if constexpr (std::is_same<Q_T, bfloat16_t>::value) {
    MicroAPI::Cast<half, KV_T, castTrait>(vCastFp16Res, vKvData, kvTypeMaskAll);
    MicroAPI::Cast<Q_T, half, castTrait1>(vRes, vCastFp16Res, kvTypeMaskAll);
  } else {
    MicroAPI::Cast<Q_T, KV_T, castTrait>(vRes, vKvData, kvTypeMaskAll);
  }
}

template <typename Q_T, typename KV_T, typename ANTIQ_PARAMS_T, uint32_t baseSize, bool hasOffset = false>
__simd_vf__ void AntiquantVFImplW8PerTokenD64(__ubuf__ uint8_t* ubSrcAddr, __ubuf__ Q_T* ubDstAddr, 
                                              __ubuf__ Q_T *ubDstAddr_, __ubuf__ ANTIQ_PARAMS_T* ubOffsetAddr, 
                                              __ubuf__ ANTIQ_PARAMS_T* ubScaleAddr, uint32_t dealRowCount)
{
  MicroAPI::RegTensor<KV_T> vKvData;
  MicroAPI::RegTensor<ANTIQ_PARAMS_T> vOffset;
  MicroAPI::RegTensor<ANTIQ_PARAMS_T> vScale;
  MicroAPI::RegTensor<Q_T> vOffsetFp16;
  MicroAPI::RegTensor<Q_T> vOffsetFp16Low;
  MicroAPI::RegTensor<Q_T> vOffsetFp16High;
  MicroAPI::RegTensor<Q_T> vScaleFp16;
  MicroAPI::RegTensor<Q_T> vScaleFp16High;
  MicroAPI::RegTensor<Q_T> vScaleFp16Low;
  MicroAPI::RegTensor<Q_T> vRes;
  MicroAPI::RegTensor<half> vCastFp16Res;
  MicroAPI::MaskReg maskOne = MicroAPI::CreateMask<ANTIQ_PARAMS_T, MicroAPI::MaskPattern::VL1>();
  MicroAPI::MaskReg kvTypeMaskAll = MicroAPI::CreateMask<KV_T, MicroAPI::MaskPattern::ALL>();
  MicroAPI::MaskReg qTypeMaskLower128 = MicroAPI::CreateMask<Q_T, MicroAPI::MaskPattern::VL128>();
  MicroAPI::MaskReg qTypeMaskLower64 = MicroAPI::CreateMask<Q_T, MicroAPI::MaskPattern::VL64>();
  MicroAPI::MaskReg qTypeMaskAll = MicroAPI::CreateMask<Q_T, MicroAPI::MaskPattern::ALL>(); // Q_T 所有元素（共128个）
  MicroAPI::MaskReg qTypeMaskHigher64;
  MicroAPI::Xor(qTypeMaskHigher64, qTypeMaskLower64, qTypeMaskAll, qTypeMaskAll); // qTypeMaskAll与qTypeMaskLower64异或得到qTypeMaskHigher64

  MicroAPI::UnalignRegForLoad u0;
  MicroAPI::UnalignRegForLoad u1;

  uint32_t blockStride = 1 + dealRowCount;
  uint32_t repeatStride = 2;

  MicroAPI::LoadUnAlignPre(u0, ubScaleAddr);
  if constexpr (hasOffset) {
    MicroAPI::LoadUnAlignPre(u1, ubOffsetAddr);
  }
  uint16_t loopCnt = static_cast<uint16_t>((dealRowCount + 1) / 2); // +1是为了兼容处理奇数行
  // 对D64优化，相邻2行合并计算；+1兼容奇数行场景
  for (uint16_t i = 0; i < loopCnt; i++) {
    MicroAPI::LoadAlign<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_UNPACK_B8>(
        (MicroAPI::RegTensor<uint8_t>&)vKvData, ubSrcAddr, baseSize * 2);

    CastW8KvToRes<Q_T, KV_T>(vRes, vKvData, vCastFp16Res, kvTypeMaskAll);

    LoadCastDupLowHighOr<Q_T, ANTIQ_PARAMS_T>(vScale, u0, ubScaleAddr, vScaleFp16, vScaleFp16Low,
                                               vScaleFp16High, maskOne, qTypeMaskLower64,
                                               qTypeMaskHigher64, kvTypeMaskAll);
    if constexpr (hasOffset) {
      LoadCastDupLowHighOr<Q_T, ANTIQ_PARAMS_T>(vOffset, u1, ubOffsetAddr, vOffsetFp16, vOffsetFp16Low,
                                                 vOffsetFp16High, maskOne, qTypeMaskLower64,
                                                 qTypeMaskHigher64, kvTypeMaskAll);
      MicroAPI::Add<Q_T, MicroAPI::MaskMergeMode::ZEROING>(vRes, vRes, vOffsetFp16, qTypeMaskLower128);
    }

    MicroAPI::Mul<Q_T, MicroAPI::MaskMergeMode::ZEROING>(vRes, vRes, vScaleFp16, qTypeMaskLower128);

    MicroAPI::StoreAlign<Q_T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
        ubDstAddr, vRes, blockStride, repeatStride, qTypeMaskLower64);

    MicroAPI::StoreAlign<Q_T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
        ubDstAddr_, vRes, blockStride, repeatStride, qTypeMaskHigher64);
  }
}

template <typename Q_T, typename KV_T, typename ANTIQ_PARAMS_T,
  uint32_t baseSize, bool hasOffset = false>
__aicore__ inline void AntiquantVFW8PerTokenD64(LocalTensor<KV_T>& antiqInUb, LocalTensor<Q_T>& antiqResUb,
                                                    LocalTensor<ANTIQ_PARAMS_T>& antiqOffsetUb, LocalTensor<ANTIQ_PARAMS_T>& antiqScaleUb,
                                                    uint32_t dealRowCount) {
  static_assert(baseSize == 64);
  static_assert(IsSameType<KV_T, int8_t>::value, "antiquant perToken w8, KV_T must be int8_t");
  __ubuf__ uint8_t* ubSrcAddr = (__ubuf__ uint8_t*)(antiqInUb.GetPhyAddr());
  __ubuf__ Q_T* ubDstAddr = (__ubuf__ Q_T*)(antiqResUb.GetPhyAddr());
  __ubuf__ Q_T *ubDstAddr_ = ubDstAddr + 16 - (dealRowCount + 1) * 32 * 4 / 2;
  __ubuf__ ANTIQ_PARAMS_T* ubOffsetAddr = (__ubuf__ ANTIQ_PARAMS_T*)antiqOffsetUb.GetPhyAddr();
  __ubuf__ ANTIQ_PARAMS_T* ubScaleAddr = (__ubuf__ ANTIQ_PARAMS_T*)antiqScaleUb.GetPhyAddr();

  AntiquantVFImplW8PerTokenD64<Q_T, KV_T, ANTIQ_PARAMS_T, baseSize, hasOffset>(
    ubSrcAddr, ubDstAddr, ubDstAddr_, ubOffsetAddr, ubScaleAddr, dealRowCount);
}

template <typename Q_T, typename KV_T, typename ANTIQ_PARAMS_T, uint32_t baseSize, bool hasOffset = false>
__simd_vf__ void AntiquantVFImplW8Norm(__ubuf__ uint8_t* ubSrcAddr, __ubuf__ Q_T* ubDstAddr, __ubuf__ Q_T* ubOffsetAddr, 
                                      __ubuf__ Q_T* ubScaleAddr, uint32_t dealRowCount)
{
  MicroAPI::RegTensor<KV_T> vKvData;
  MicroAPI::RegTensor<Q_T> vOffset;
  MicroAPI::RegTensor<Q_T> vScale;
  MicroAPI::RegTensor<Q_T> vRes;
  MicroAPI::RegTensor<half> vCastFp16Res;

  MicroAPI::MaskReg kvTypeMaskAll = MicroAPI::CreateMask<KV_T, MicroAPI::MaskPattern::ALL>();
  MicroAPI::MaskReg qTypeMaskAll = MicroAPI::CreateMask<Q_T, MicroAPI::MaskPattern::ALL>(); // Q_T 所有元素（共128个）

  uint32_t blockStride = dealRowCount + 1;
  uint32_t repeatStride = 1;
  const uint16_t loops = static_cast<uint16_t>(baseSize / 128);

  for (uint16_t j = 0; j < loops; j++) {
    __ubuf__ Q_T* ubDstAddrTmp = ubDstAddr + blockStride * 128 * j;
    __ubuf__ uint8_t* ubSrcTemp = ubSrcAddr + j * 128;

    if constexpr (hasOffset) {
      MicroAPI::LoadAlign<Q_T, MicroAPI::LoadDist::DIST_NORM>(vOffset, ubOffsetAddr + j * 128);
    }
    MicroAPI::LoadAlign<Q_T, MicroAPI::LoadDist::DIST_NORM>(vScale, ubScaleAddr + j * 128);

    // #pragma unroll(4)
    for (uint16_t i = 0; i < static_cast<uint16_t>(dealRowCount); i++) {
      MicroAPI::LoadAlign<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_UNPACK_B8>(
          (MicroAPI::RegTensor<uint8_t>&)vKvData, ubSrcTemp, baseSize);

      if constexpr (std::is_same<Q_T, bfloat16_t>::value) {
        MicroAPI::Cast<half, KV_T, castTrait>(vCastFp16Res, vKvData, kvTypeMaskAll);
        MicroAPI::Cast<Q_T, half, castTrait1>(vRes, vCastFp16Res, kvTypeMaskAll);
      } else {
        MicroAPI::Cast<Q_T, KV_T, castTrait>(vRes, vKvData, kvTypeMaskAll);
      }

      if constexpr (hasOffset) {
        MicroAPI::Add<Q_T, MicroAPI::MaskMergeMode::ZEROING>(vRes, vRes, vOffset, qTypeMaskAll);
      }

      MicroAPI::Mul<Q_T, MicroAPI::MaskMergeMode::ZEROING>(vRes, vScale, vRes, qTypeMaskAll);

      MicroAPI::StoreAlign<Q_T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
          ubDstAddrTmp, vRes, blockStride, repeatStride, qTypeMaskAll);
    }
  }
}

template <typename Q_T, typename KV_T, typename ANTIQ_PARAMS_T,
  uint32_t baseSize, bool hasOffset = false>
__aicore__ inline void AntiquantVFW8Norm(LocalTensor<KV_T>& antiqInUb, LocalTensor<Q_T>& antiqResUb,
                                             LocalTensor<ANTIQ_PARAMS_T>& antiqOffsetUb, LocalTensor<ANTIQ_PARAMS_T>& antiqScaleUb,
                                             uint32_t dealRowCount) {
  static_assert(baseSize % 128 == 0);
  static_assert(IsSameType<KV_T, int8_t>::value || IsSameType<KV_T, hifloat8_t>::value,
                "antiquant w8, KV_T must be int8_t or hifloat8_t");
  __ubuf__ uint8_t* ubSrcAddr = (__ubuf__ uint8_t*)(antiqInUb.GetPhyAddr());
  __ubuf__ Q_T* ubDstAddr = (__ubuf__ Q_T*)(antiqResUb.GetPhyAddr());
  __ubuf__ Q_T* ubOffsetAddr = (__ubuf__ Q_T*)antiqOffsetUb.GetPhyAddr();
  __ubuf__ Q_T* ubScaleAddr = (__ubuf__ Q_T*)antiqScaleUb.GetPhyAddr();
  
  AntiquantVFImplW8Norm<Q_T, KV_T, ANTIQ_PARAMS_T, baseSize, hasOffset>(
    ubSrcAddr, ubDstAddr, ubOffsetAddr, ubScaleAddr, dealRowCount);
}

template <typename Q_T, typename KV_T, typename ANTIQ_PARAMS_T, uint32_t baseSize, bool hasOffset = false>
__simd_vf__ void AntiquantVFImplW8PerTokenD128(__ubuf__ uint8_t* ubSrcAddr, __ubuf__ Q_T* ubDstAddr,
                                          __ubuf__ ANTIQ_PARAMS_T* ubOffsetAddr, __ubuf__ ANTIQ_PARAMS_T* ubScaleAddr, 
                                          uint32_t dealRowCount)
{
  MicroAPI::RegTensor<KV_T> vKvData;
  MicroAPI::RegTensor<ANTIQ_PARAMS_T> vOffset;
  MicroAPI::RegTensor<ANTIQ_PARAMS_T> vScale;
  MicroAPI::RegTensor<Q_T> vOffsetFp16;
  MicroAPI::RegTensor<Q_T> vScaleFp16;

  MicroAPI::RegTensor<Q_T> vRes;
  MicroAPI::RegTensor<half> vCastFp16Res;

  MicroAPI::MaskReg maskOne = MicroAPI::CreateMask<ANTIQ_PARAMS_T, MicroAPI::MaskPattern::VL1>();
  MicroAPI::MaskReg kvTypeMaskAll = MicroAPI::CreateMask<KV_T, MicroAPI::MaskPattern::ALL>();
  MicroAPI::MaskReg qTypeMaskAll = MicroAPI::CreateMask<Q_T, MicroAPI::MaskPattern::ALL>(); // Q_T 所有元素（共128个）

  uint32_t blockStride = dealRowCount + 1;
  uint32_t repeatStride = 1;
  const uint16_t loops = baseSize / 128;
  MicroAPI::UnalignRegForLoad u0;
  MicroAPI::UnalignRegForLoad u1;
  for (uint16_t j = 0; j < static_cast<uint16_t>(loops); j++) {
    __ubuf__ Q_T* ubDstAddrTmp = ubDstAddr + blockStride * 128 * j;
    __ubuf__ uint8_t* ubSrcTemp = ubSrcAddr + j * 128;
    __ubuf__ ANTIQ_PARAMS_T* ubOffsetAddrTemp = ubOffsetAddr;
    __ubuf__ ANTIQ_PARAMS_T* ubScaleAddrTemp = ubScaleAddr;

    MicroAPI::LoadUnAlignPre(u0, ubScaleAddrTemp);
    if constexpr (hasOffset) {
      MicroAPI::LoadUnAlignPre(u1, ubOffsetAddrTemp);
    }
    for (uint16_t i = 0; i < static_cast<uint16_t>(dealRowCount); i++) {
      MicroAPI::LoadAlign<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_UNPACK_B8>(
          (MicroAPI::RegTensor<uint8_t>&)vKvData, ubSrcTemp, baseSize);

      CastW8KvToRes<Q_T, KV_T>(vRes, vKvData, vCastFp16Res, kvTypeMaskAll);
      MicroAPI::LoadUnAlign<ANTIQ_PARAMS_T>(vScale, u0, ubScaleAddrTemp, 1); // 1表示ub自动往后偏移1个float
      MicroAPI::Cast<Q_T, ANTIQ_PARAMS_T, castTrait0>(vScaleFp16, vScale, maskOne);
      MicroAPI::Duplicate<uint16_t, MicroAPI::HighLowPart::LOWEST, MicroAPI::MaskMergeMode::ZEROING>
        ((MicroAPI::RegTensor<uint16_t>&)vScaleFp16, (MicroAPI::RegTensor<uint16_t>&)vScaleFp16, qTypeMaskAll);
      if constexpr (hasOffset) {
        MicroAPI::LoadUnAlign<ANTIQ_PARAMS_T>(vOffset, u1, ubOffsetAddrTemp, 1); // 1表示ub自动往后偏移1个float
        MicroAPI::Cast<Q_T, ANTIQ_PARAMS_T, castTrait0>(vOffsetFp16, vOffset, maskOne);
        MicroAPI::Duplicate<uint16_t, MicroAPI::HighLowPart::LOWEST, MicroAPI::MaskMergeMode::ZEROING>
          ((MicroAPI::RegTensor<uint16_t>&)vOffsetFp16, (MicroAPI::RegTensor<uint16_t>&)vOffsetFp16, qTypeMaskAll);
        MicroAPI::Add<Q_T, MicroAPI::MaskMergeMode::ZEROING>(vRes, vOffsetFp16, vRes, qTypeMaskAll);
      }

      MicroAPI::Mul<Q_T, MicroAPI::MaskMergeMode::ZEROING>(vRes, vScaleFp16, vRes, qTypeMaskAll);

      MicroAPI::StoreAlign<Q_T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
          ubDstAddrTmp, vRes, blockStride, repeatStride, qTypeMaskAll);
    }
  }
}

template <typename Q_T, typename KV_T, typename ANTIQ_PARAMS_T,
  uint32_t baseSize, bool hasOffset = false>
__aicore__ inline void AntiquantVFW8PerTokenD128(LocalTensor<KV_T>& antiqInUb, LocalTensor<Q_T>& antiqResUb,
                                             LocalTensor<ANTIQ_PARAMS_T>& antiqOffsetUb, LocalTensor<ANTIQ_PARAMS_T>& antiqScaleUb,
                                             uint32_t dealRowCount) {
  static_assert(baseSize == 128);
  static_assert(IsSameType<KV_T, int8_t>::value, "antiquant perToken w8, KV_T must be int8_t");
  __ubuf__ uint8_t* ubSrcAddr = (__ubuf__ uint8_t*)(antiqInUb.GetPhyAddr());
  __ubuf__ Q_T* ubDstAddr = (__ubuf__ Q_T*)(antiqResUb.GetPhyAddr());
  __ubuf__ ANTIQ_PARAMS_T* ubOffsetAddr = (__ubuf__ ANTIQ_PARAMS_T*)antiqOffsetUb.GetPhyAddr();
  __ubuf__ ANTIQ_PARAMS_T* ubScaleAddr = (__ubuf__ ANTIQ_PARAMS_T*)antiqScaleUb.GetPhyAddr();

  AntiquantVFImplW8PerTokenD128<Q_T, KV_T, ANTIQ_PARAMS_T, baseSize, hasOffset>(
    ubSrcAddr, ubDstAddr, ubOffsetAddr, ubScaleAddr, dealRowCount);
}

template <typename Q_T, typename ANTIQ_PARAMS_T, bool hasOffset>
__simd_callee__ static inline void LoadPerTokenAntiqParams(MicroAPI::RegTensor<ANTIQ_PARAMS_T>& vScale,
                                                           MicroAPI::UnalignRegForLoad& u0,
                                                           __ubuf__ ANTIQ_PARAMS_T*& ubScaleAddr,
                                                           MicroAPI::RegTensor<Q_T>& vScaleFp16,
                                                           MicroAPI::MaskReg& maskOne,
                                                           MicroAPI::MaskReg& qMaskAll,
                                                           MicroAPI::RegTensor<ANTIQ_PARAMS_T>& vOffset,
                                                           MicroAPI::UnalignRegForLoad& u1,
                                                           __ubuf__ ANTIQ_PARAMS_T*& ubOffsetAddr,
                                                           MicroAPI::RegTensor<Q_T>& vOffsetFp16)
{
  MicroAPI::LoadUnAlign<ANTIQ_PARAMS_T>(vScale, u0, ubScaleAddr, 1); // 1表示ub自动往后偏移1个float
  MicroAPI::Cast<Q_T, ANTIQ_PARAMS_T, castTrait0>(vScaleFp16, vScale, maskOne);
  MicroAPI::Duplicate<uint16_t, MicroAPI::HighLowPart::LOWEST, MicroAPI::MaskMergeMode::ZEROING>
    ((MicroAPI::RegTensor<uint16_t>&)vScaleFp16, (MicroAPI::RegTensor<uint16_t>&)vScaleFp16, qMaskAll);
  if constexpr (hasOffset) {
    MicroAPI::LoadUnAlign<ANTIQ_PARAMS_T>(vOffset, u1, ubOffsetAddr, 1); // 1表示ub自动往后偏移1个float
    MicroAPI::Cast<Q_T, ANTIQ_PARAMS_T, castTrait0>(vOffsetFp16, vOffset, maskOne);
    MicroAPI::Duplicate<uint16_t, MicroAPI::HighLowPart::LOWEST, MicroAPI::MaskMergeMode::ZEROING>
      ((MicroAPI::RegTensor<uint16_t>&)vOffsetFp16, (MicroAPI::RegTensor<uint16_t>&)vOffsetFp16, qMaskAll);
  }
}

template <typename Q_T, typename KV_T>
__simd_callee__ static inline void CastW8KvToResPair(MicroAPI::RegTensor<Q_T>& vRes,
                                                      MicroAPI::RegTensor<Q_T>& vRes1,
                                                      MicroAPI::RegTensor<KV_T>& vKvData,
                                                      MicroAPI::RegTensor<KV_T>& vKvData1,
                                                      MicroAPI::RegTensor<half>& vCastFp16Res,
                                                      MicroAPI::RegTensor<half>& vCastFp16Res1,
                                                      MicroAPI::MaskReg& kvMaskAll)
{
  if constexpr (std::is_same<Q_T, bfloat16_t>::value) {
    MicroAPI::Cast<half, KV_T, castTrait>(vCastFp16Res, vKvData, kvMaskAll);
    MicroAPI::Cast<half, KV_T, castTrait>(vCastFp16Res1, vKvData1, kvMaskAll);
    MicroAPI::Cast<Q_T, half, castTrait1>(vRes, vCastFp16Res, kvMaskAll);
    MicroAPI::Cast<Q_T, half, castTrait1>(vRes1, vCastFp16Res1, kvMaskAll);
  } else {
    MicroAPI::Cast<Q_T, KV_T, castTrait>(vRes, vKvData, kvMaskAll);
    MicroAPI::Cast<Q_T, KV_T, castTrait>(vRes1, vKvData1, kvMaskAll);
  }
}

template <typename Q_T, typename KV_T, typename ANTIQ_PARAMS_T, uint32_t baseSize, bool hasOffset = false>
__simd_vf__ void AntiquantVFImplW8PerTokenD256(__ubuf__ uint8_t* ubSrcAddr, __ubuf__ uint8_t* ubSrcAddr1, 
                                              __ubuf__ Q_T* ubDstAddr, __ubuf__ ANTIQ_PARAMS_T* ubOffsetAddr, 
                                              __ubuf__ ANTIQ_PARAMS_T* ubScaleAddr, uint32_t dealRowCount)
{
  MicroAPI::RegTensor<KV_T> vKvData;
  MicroAPI::RegTensor<KV_T> vKvData1;
  MicroAPI::RegTensor<ANTIQ_PARAMS_T> vOffset;
  MicroAPI::RegTensor<ANTIQ_PARAMS_T> vScale;
  MicroAPI::RegTensor<Q_T> vScaleFp16;
  MicroAPI::RegTensor<Q_T> vOffsetFp16;

  MicroAPI::RegTensor<Q_T> vRes;
  MicroAPI::RegTensor<Q_T> vRes1;
  MicroAPI::RegTensor<half> vCastFp16Res;
  MicroAPI::RegTensor<half> vCastFp16Res1;

  MicroAPI::MaskReg maskOne = MicroAPI::CreateMask<ANTIQ_PARAMS_T, MicroAPI::MaskPattern::VL1>();
  MicroAPI::MaskReg kvMaskAll = MicroAPI::CreateMask<KV_T, MicroAPI::MaskPattern::ALL>();
  MicroAPI::MaskReg qMaskAll = MicroAPI::CreateMask<Q_T, MicroAPI::MaskPattern::ALL>(); // Q_T 所有元素（共128个）

  uint32_t blockStride = 1 + dealRowCount;
  uint32_t repeatStride = 1;
  MicroAPI::UnalignRegForLoad u1;
  MicroAPI::UnalignRegForLoad u0;
  MicroAPI::LoadUnAlignPre(u0, ubScaleAddr);
  if constexpr (hasOffset) {
    MicroAPI::LoadUnAlignPre(u1, ubOffsetAddr);
  }
  __ubuf__ Q_T* ubDstAddr1 = ubDstAddr + blockStride * 128;
  for (uint16_t j = 0; j < static_cast<uint16_t>(dealRowCount); j++) {
    // 读入每行的伪量化参数
    LoadPerTokenAntiqParams<Q_T, ANTIQ_PARAMS_T, hasOffset>(vScale, u0, ubScaleAddr, vScaleFp16, maskOne,
      qMaskAll, vOffset, u1, ubOffsetAddr, vOffsetFp16);
    MicroAPI::LoadAlign<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_UNPACK_B8>(
      (MicroAPI::RegTensor<uint8_t>&)vKvData, ubSrcAddr, 256); // d=256，自动往后偏移256个数
    MicroAPI::LoadAlign<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_UNPACK_B8>(
      (MicroAPI::RegTensor<uint8_t>&)vKvData1, ubSrcAddr1, 256); // d=256
    CastW8KvToResPair<Q_T, KV_T>(vRes, vRes1, vKvData, vKvData1, vCastFp16Res, vCastFp16Res1, kvMaskAll);
    if constexpr (hasOffset) {
      MicroAPI::Add<Q_T, MicroAPI::MaskMergeMode::ZEROING>(vRes, vRes, vOffsetFp16, qMaskAll);
      MicroAPI::Add<Q_T, MicroAPI::MaskMergeMode::ZEROING>(vRes1, vRes1, vOffsetFp16, qMaskAll);
    }

    MicroAPI::Mul<Q_T, MicroAPI::MaskMergeMode::ZEROING>(vRes, vRes, vScaleFp16, qMaskAll);
    MicroAPI::Mul<Q_T, MicroAPI::MaskMergeMode::ZEROING>(vRes1, vRes1, vScaleFp16, qMaskAll);
    MicroAPI::StoreAlign<Q_T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
        ubDstAddr, vRes, blockStride, repeatStride, qMaskAll);
    MicroAPI::StoreAlign<Q_T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
        ubDstAddr1, vRes1, blockStride, repeatStride, qMaskAll);
  }
}

template <typename Q_T, typename KV_T, typename ANTIQ_PARAMS_T,
  uint32_t baseSize, bool hasOffset = false>
__aicore__ inline void AntiquantVFW8PerTokenD256(LocalTensor<KV_T>& antiqInUb, LocalTensor<Q_T>& antiqResUb,
                                             LocalTensor<ANTIQ_PARAMS_T>& antiqOffsetUb, LocalTensor<ANTIQ_PARAMS_T>& antiqScaleUb,
                                             uint32_t dealRowCount) {
  static_assert(baseSize == 256);
  static_assert(IsSameType<KV_T, int8_t>::value, "antiquant perToken w8, KV_T must be int8_t");
  __ubuf__ uint8_t* ubSrcAddr = (__ubuf__ uint8_t*)(antiqInUb.GetPhyAddr());
  __ubuf__ uint8_t* ubSrcAddr1 = ubSrcAddr + 128;
  __ubuf__ Q_T* ubDstAddr = (__ubuf__ Q_T*)(antiqResUb.GetPhyAddr());
  __ubuf__ ANTIQ_PARAMS_T* ubOffsetAddr = (__ubuf__ ANTIQ_PARAMS_T*)antiqOffsetUb.GetPhyAddr();
  __ubuf__ ANTIQ_PARAMS_T* ubScaleAddr = (__ubuf__ ANTIQ_PARAMS_T*)antiqScaleUb.GetPhyAddr();

  AntiquantVFImplW8PerTokenD256<Q_T, KV_T, ANTIQ_PARAMS_T, baseSize, hasOffset>(
    ubSrcAddr, ubSrcAddr1, ubDstAddr, ubOffsetAddr, ubScaleAddr, dealRowCount);
}

template <typename KV_T>
__simd_callee__ static inline void LoadAlignW8Quad(
    MicroAPI::RegTensor<KV_T>& vKvData, MicroAPI::RegTensor<KV_T>& vKvData1,
    MicroAPI::RegTensor<KV_T>& vKvData2, MicroAPI::RegTensor<KV_T>& vKvData3,
    __ubuf__ uint8_t*& ubSrcAddr, __ubuf__ uint8_t*& ubSrcAddr1,
    __ubuf__ uint8_t*& ubSrcAddr2, __ubuf__ uint8_t*& ubSrcAddr3)
{
  MicroAPI::LoadAlign<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_UNPACK_B8>(
      (MicroAPI::RegTensor<uint8_t>&)vKvData, ubSrcAddr, 512); // d=512 每次往后偏移512
  MicroAPI::LoadAlign<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_UNPACK_B8>(
      (MicroAPI::RegTensor<uint8_t>&)vKvData1, ubSrcAddr1, 512); // d=512
  MicroAPI::LoadAlign<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_UNPACK_B8>(
      (MicroAPI::RegTensor<uint8_t>&)vKvData2, ubSrcAddr2, 512); // d=512
  MicroAPI::LoadAlign<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_UNPACK_B8>(
      (MicroAPI::RegTensor<uint8_t>&)vKvData3, ubSrcAddr3, 512); // d=512
}

template <typename Q_T, typename KV_T>
__simd_callee__ static inline void CastW8KvToResQuad(
    MicroAPI::RegTensor<Q_T>& vRes, MicroAPI::RegTensor<Q_T>& vRes1,
    MicroAPI::RegTensor<Q_T>& vRes2, MicroAPI::RegTensor<Q_T>& vRes3,
    MicroAPI::RegTensor<KV_T>& vKvData, MicroAPI::RegTensor<KV_T>& vKvData1,
    MicroAPI::RegTensor<KV_T>& vKvData2, MicroAPI::RegTensor<KV_T>& vKvData3,
    MicroAPI::RegTensor<half>& vCastFp16Res, MicroAPI::RegTensor<half>& vCastFp16Res1,
    MicroAPI::RegTensor<half>& vCastFp16Res2, MicroAPI::RegTensor<half>& vCastFp16Res3,
    MicroAPI::MaskReg& kvMaskAll)
{
  if constexpr (std::is_same<Q_T, bfloat16_t>::value) {
    MicroAPI::Cast<half, KV_T, castTrait>(vCastFp16Res, vKvData, kvMaskAll);
    MicroAPI::Cast<half, KV_T, castTrait>(vCastFp16Res1, vKvData1, kvMaskAll);
    MicroAPI::Cast<half, KV_T, castTrait>(vCastFp16Res2, vKvData2, kvMaskAll);
    MicroAPI::Cast<half, KV_T, castTrait>(vCastFp16Res3, vKvData3, kvMaskAll);
    MicroAPI::Cast<Q_T, half, castTrait1>(vRes, vCastFp16Res, kvMaskAll);
    MicroAPI::Cast<Q_T, half, castTrait1>(vRes1, vCastFp16Res1, kvMaskAll);
    MicroAPI::Cast<Q_T, half, castTrait1>(vRes2, vCastFp16Res2, kvMaskAll);
    MicroAPI::Cast<Q_T, half, castTrait1>(vRes3, vCastFp16Res3, kvMaskAll);
  } else {
    MicroAPI::Cast<Q_T, KV_T, castTrait>(vRes, vKvData, kvMaskAll);
    MicroAPI::Cast<Q_T, KV_T, castTrait>(vRes1, vKvData1, kvMaskAll);
    MicroAPI::Cast<Q_T, KV_T, castTrait>(vRes2, vKvData2, kvMaskAll);
    MicroAPI::Cast<Q_T, KV_T, castTrait>(vRes3, vKvData3, kvMaskAll);
  }
}

template <typename Q_T, bool hasOffset>
__simd_callee__ static inline void AddMulStoreFp8Quad(
    MicroAPI::RegTensor<Q_T>& vRes, MicroAPI::RegTensor<Q_T>& vRes1,
    MicroAPI::RegTensor<Q_T>& vRes2, MicroAPI::RegTensor<Q_T>& vRes3,
    MicroAPI::RegTensor<Q_T>& vOffsetFp16, MicroAPI::RegTensor<Q_T>& vScaleFp16,
    __ubuf__ Q_T*& ubDstAddr, __ubuf__ Q_T*& ubDstAddr1,
    __ubuf__ Q_T*& ubDstAddr2, __ubuf__ Q_T*& ubDstAddr3,
    uint32_t blockStride, uint32_t repeatStride,
    MicroAPI::MaskReg& qMaskAll)
{
  if constexpr (hasOffset) {
    MicroAPI::Add<Q_T, MicroAPI::MaskMergeMode::ZEROING>(vRes, vRes, vOffsetFp16, qMaskAll);
    MicroAPI::Add<Q_T, MicroAPI::MaskMergeMode::ZEROING>(vRes1, vRes1, vOffsetFp16, qMaskAll);
    MicroAPI::Add<Q_T, MicroAPI::MaskMergeMode::ZEROING>(vRes2, vRes2, vOffsetFp16, qMaskAll);
    MicroAPI::Add<Q_T, MicroAPI::MaskMergeMode::ZEROING>(vRes3, vRes3, vOffsetFp16, qMaskAll);
  }
  MicroAPI::Mul<Q_T, MicroAPI::MaskMergeMode::ZEROING>(vRes, vScaleFp16, vRes, qMaskAll);
  MicroAPI::Mul<Q_T, MicroAPI::MaskMergeMode::ZEROING>(vRes1, vScaleFp16, vRes1, qMaskAll);
  MicroAPI::Mul<Q_T, MicroAPI::MaskMergeMode::ZEROING>(vRes2, vRes2, vScaleFp16, qMaskAll);
  MicroAPI::Mul<Q_T, MicroAPI::MaskMergeMode::ZEROING>(vRes3, vRes3, vScaleFp16, qMaskAll);
  MicroAPI::StoreAlign<Q_T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
      ubDstAddr, vRes, blockStride, repeatStride, qMaskAll);
  MicroAPI::StoreAlign<Q_T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
      ubDstAddr1, vRes1, blockStride, repeatStride, qMaskAll);
  MicroAPI::StoreAlign<Q_T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
      ubDstAddr2, vRes2, blockStride, repeatStride, qMaskAll);
  MicroAPI::StoreAlign<Q_T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
      ubDstAddr3, vRes3, blockStride, repeatStride, qMaskAll);
}

template <typename Q_T, typename KV_T, typename ANTIQ_PARAMS_T, uint32_t baseSize, bool hasOffset = false>
__simd_vf__ void AntiquantVFImplW8PerTokenD512(__ubuf__ uint8_t* ubSrcAddr, __ubuf__ uint8_t* ubSrcAddr1, 
                                              __ubuf__ uint8_t* ubSrcAddr2, __ubuf__ uint8_t* ubSrcAddr3, 
                                              __ubuf__ Q_T* ubDstAddr, __ubuf__ ANTIQ_PARAMS_T* ubOffsetAddr, 
                                              __ubuf__ ANTIQ_PARAMS_T* ubScaleAddr, uint32_t dealRowCount)
{
  MicroAPI::RegTensor<KV_T> vKvData;
  MicroAPI::RegTensor<KV_T> vKvData1;
  MicroAPI::RegTensor<KV_T> vKvData2;
  MicroAPI::RegTensor<KV_T> vKvData3;
  MicroAPI::RegTensor<ANTIQ_PARAMS_T> vScale;
  MicroAPI::RegTensor<ANTIQ_PARAMS_T> vOffset;
  MicroAPI::RegTensor<Q_T> vOffsetFp16;
  MicroAPI::RegTensor<Q_T> vScaleFp16;

  MicroAPI::RegTensor<Q_T> vRes;
  MicroAPI::RegTensor<Q_T> vRes1;
  MicroAPI::RegTensor<Q_T> vRes2;
  MicroAPI::RegTensor<Q_T> vRes3;
  MicroAPI::RegTensor<half> vCastFp16Res;
  MicroAPI::RegTensor<half> vCastFp16Res1;
  MicroAPI::RegTensor<half> vCastFp16Res2;
  MicroAPI::RegTensor<half> vCastFp16Res3;

  MicroAPI::MaskReg kvMaskAll = MicroAPI::CreateMask<KV_T, MicroAPI::MaskPattern::ALL>();
  MicroAPI::MaskReg maskOne = MicroAPI::CreateMask<ANTIQ_PARAMS_T, MicroAPI::MaskPattern::VL1>();
  MicroAPI::MaskReg qMaskAll = MicroAPI::CreateMask<Q_T, MicroAPI::MaskPattern::ALL>(); // Q_T 所有元素（共128个）

  uint32_t blockStride = 1 + dealRowCount;
  uint32_t repeatStride = 1;
  MicroAPI::UnalignRegForLoad u0;
  MicroAPI::UnalignRegForLoad u1;
  MicroAPI::LoadUnAlignPre(u0, ubScaleAddr);
  if constexpr (hasOffset) {
    MicroAPI::LoadUnAlignPre(u1, ubOffsetAddr);
  }
  __ubuf__ Q_T* ubDstAddr1 = ubDstAddr + blockStride * 128;
  __ubuf__ Q_T* ubDstAddr2 = ubDstAddr + blockStride * 128 * 2;
  __ubuf__ Q_T* ubDstAddr3 = ubDstAddr + blockStride * 128 * 3;
  for (uint16_t j = 0; j < static_cast<uint16_t>(dealRowCount); j++) {
    // 读入每行的伪量化参数
    LoadPerTokenAntiqParams<Q_T, ANTIQ_PARAMS_T, hasOffset>(vScale, u0, ubScaleAddr, vScaleFp16, maskOne,
      qMaskAll, vOffset, u1, ubOffsetAddr, vOffsetFp16);
    LoadAlignW8Quad<KV_T>(vKvData, vKvData1, vKvData2, vKvData3,
                          ubSrcAddr, ubSrcAddr1, ubSrcAddr2, ubSrcAddr3);
    CastW8KvToResQuad<Q_T, KV_T>(vRes, vRes1, vRes2, vRes3, vKvData, vKvData1, vKvData2, vKvData3,
                                  vCastFp16Res, vCastFp16Res1, vCastFp16Res2, vCastFp16Res3, kvMaskAll);
    AddMulStoreFp8Quad<Q_T, hasOffset>(vRes, vRes1, vRes2, vRes3, vOffsetFp16, vScaleFp16,
                                        ubDstAddr, ubDstAddr1, ubDstAddr2, ubDstAddr3,
                                        blockStride, repeatStride, qMaskAll);
  }
}

template <typename Q_T, typename KV_T, typename ANTIQ_PARAMS_T,
  uint32_t baseSize, bool hasOffset = false>
__aicore__ inline void AntiquantVFW8PerTokenD512(LocalTensor<KV_T>& antiqInUb, LocalTensor<Q_T>& antiqResUb,
                                             LocalTensor<ANTIQ_PARAMS_T>& antiqOffsetUb, LocalTensor<ANTIQ_PARAMS_T>& antiqScaleUb,
                                             uint32_t dealRowCount) {
  static_assert(baseSize == 512);
  static_assert(IsSameType<KV_T, int8_t>::value, "antiquant perToken w8, KV_T must be int8_t");
  __ubuf__ uint8_t* ubSrcAddr = (__ubuf__ uint8_t*)(antiqInUb.GetPhyAddr());
  __ubuf__ uint8_t* ubSrcAddr1 = ubSrcAddr + 128;
  __ubuf__ uint8_t* ubSrcAddr2 = ubSrcAddr + 128 * 2;
  __ubuf__ uint8_t* ubSrcAddr3 = ubSrcAddr + 128 * 3;
  __ubuf__ Q_T* ubDstAddr = (__ubuf__ Q_T*)(antiqResUb.GetPhyAddr());
  __ubuf__ ANTIQ_PARAMS_T* ubOffsetAddr = (__ubuf__ ANTIQ_PARAMS_T*)antiqOffsetUb.GetPhyAddr();
  __ubuf__ ANTIQ_PARAMS_T* ubScaleAddr = (__ubuf__ ANTIQ_PARAMS_T*)antiqScaleUb.GetPhyAddr();

  AntiquantVFImplW8PerTokenD512<Q_T, KV_T, ANTIQ_PARAMS_T, baseSize, hasOffset>(
    ubSrcAddr, ubSrcAddr1, ubSrcAddr2, ubSrcAddr3, ubDstAddr, ubOffsetAddr, ubScaleAddr, dealRowCount);
}

template <typename Q_T, typename ANTIQ_PARAMS_T, uint32_t baseSize, bool hasOffset = false, bool isPerToken = false, bool isKvCacheNz = false>
__aicore__ inline void AntiquantVFImpl(LocalTensor<int8_t>& antiqInUb, LocalTensor<Q_T>& antiqResUb,
                                       LocalTensor<ANTIQ_PARAMS_T>& antiqOffsetUb, LocalTensor<ANTIQ_PARAMS_T>& antiqScaleUb,
                                       uint32_t dealRowCount, uint32_t headDim, uint32_t copyTotalS) {
  if constexpr (isKvCacheNz) {
    if constexpr (!isPerToken) {
      AntiquantVFW8Nz<Q_T, int8_t, ANTIQ_PARAMS_T, baseSize, hasOffset>
        (antiqInUb, antiqResUb, antiqOffsetUb, antiqScaleUb, dealRowCount);
    } else {
      AntiquantVFW8PerTokenNz<Q_T, int8_t, ANTIQ_PARAMS_T, baseSize, hasOffset>
        (antiqInUb, antiqResUb, antiqOffsetUb, antiqScaleUb, dealRowCount);
    }
  } else {
    if constexpr (isPerToken) {
      if constexpr (baseSize == 64) {
        AntiquantVFW8PerTokenD64<Q_T, int8_t, ANTIQ_PARAMS_T, baseSize, hasOffset>
          (antiqInUb, antiqResUb, antiqOffsetUb, antiqScaleUb, dealRowCount);
      } else if constexpr (baseSize == 128) {
        AntiquantVFW8PerTokenD128<Q_T, int8_t, ANTIQ_PARAMS_T, baseSize, hasOffset>
          (antiqInUb, antiqResUb, antiqOffsetUb, antiqScaleUb, dealRowCount);
      } else if constexpr (baseSize == 256) {
        AntiquantVFW8PerTokenD256<Q_T, int8_t, ANTIQ_PARAMS_T, baseSize, hasOffset>
          (antiqInUb, antiqResUb, antiqOffsetUb, antiqScaleUb, dealRowCount);
      } else {
        AntiquantVFW8PerTokenD512<Q_T, int8_t, ANTIQ_PARAMS_T, baseSize, hasOffset>
          (antiqInUb, antiqResUb, antiqOffsetUb, antiqScaleUb, dealRowCount);
      }
    } else {
      if constexpr (baseSize == 64) {
        AntiquantVFW8D64<Q_T, int8_t, ANTIQ_PARAMS_T, baseSize, hasOffset>
          (antiqInUb, antiqResUb, antiqOffsetUb, antiqScaleUb, dealRowCount);
      } else {
        AntiquantVFW8Norm<Q_T, int8_t, ANTIQ_PARAMS_T, baseSize, hasOffset>
          (antiqInUb, antiqResUb, antiqOffsetUb, antiqScaleUb, dealRowCount);
      }
    }
  }
}

template <typename Q_T, typename ANTIQ_PARAMS_T, uint32_t baseSize, bool hasOffset = false, bool isPerToken = false, bool isKvCacheNz = false>
__aicore__ inline void AntiquantVFImpl(LocalTensor<hifloat8_t>& antiqInUb, LocalTensor<Q_T>& antiqResUb,
                                       LocalTensor<Q_T>& antiqOffsetUb, LocalTensor<Q_T>& antiqScaleUb,
                                       uint32_t dealRowCount, uint32_t headDim, uint32_t copyTotalS) {
  if constexpr (isKvCacheNz) {
    AntiquantVFW8Nz<Q_T, hifloat8_t, ANTIQ_PARAMS_T, baseSize, hasOffset>
      (antiqInUb, antiqResUb, antiqOffsetUb, antiqScaleUb, dealRowCount);
  } else {
    if constexpr (baseSize == 64) {
      AntiquantVFW8D64<Q_T, hifloat8_t, ANTIQ_PARAMS_T, baseSize, hasOffset>(antiqInUb, antiqResUb, antiqOffsetUb, antiqScaleUb,
                                                              dealRowCount);
    } else {
      AntiquantVFW8Norm<Q_T, hifloat8_t, ANTIQ_PARAMS_T, baseSize, hasOffset>(antiqInUb, antiqResUb, antiqOffsetUb, antiqScaleUb,
                                                            dealRowCount);
    }
  }
}

template <typename Q_T, typename KV_T, uint32_t baseSize>
__simd_vf__ void AntiquantVFImplFp8D64(__ubuf__ uint8_t* ubSrcAddr, __ubuf__ Q_T* ubDstAddr, __ubuf__ Q_T* ubDstAddrEven, 
                                      __ubuf__ Q_T *ubDstAddr_, __ubuf__ Q_T* ubScalerSrcAddr, uint32_t dealRowCount)
{
  MicroAPI::RegTensor<KV_T> vKvData;
  MicroAPI::RegTensor<float> vCastFp32Res0;
  MicroAPI::RegTensor<float> vCastFp32Res1;
  MicroAPI::RegTensor<Q_T> vCastRes0;
  MicroAPI::RegTensor<Q_T> vCastRes1;
  MicroAPI::RegTensor<Q_T> vScale;
  MicroAPI::RegTensor<Q_T> vMulRes;

  MicroAPI::MaskReg kvTypeMaskAll = MicroAPI::CreateMask<KV_T, MicroAPI::MaskPattern::ALL>();
  MicroAPI::MaskReg qTypeMaskLower64 = MicroAPI::CreateMask<Q_T, MicroAPI::MaskPattern::VL64>();
  MicroAPI::MaskReg qTypeMaskLower128 = MicroAPI::CreateMask<Q_T, MicroAPI::MaskPattern::VL128>();
  MicroAPI::MaskReg qTypeMaskAll = MicroAPI::CreateMask<Q_T, MicroAPI::MaskPattern::ALL>();
  MicroAPI::MaskReg qTypeMaskHigher64;
  // NZ:
  uint32_t blockStride = dealRowCount + 1;
  uint32_t repeatStride = 2;
  uint16_t loopCnt = static_cast<uint16_t>((dealRowCount + 1) / 2); // +1是为了兼容处理奇数行

  __ubuf__ uint8_t* ubSrcTemp = ubSrcAddr;

  // 加载 scale
  MicroAPI::LoadAlign<Q_T, MicroAPI::LoadDist::DIST_NORM>(vScale, ubScalerSrcAddr);
  MicroAPI::Xor(qTypeMaskHigher64, qTypeMaskLower64, qTypeMaskAll, qTypeMaskAll); // qTypeMaskAll与qTypeMaskLower64异或得到qTypeMaskHigher64

  // D=64时相邻2行合并做伪量化计算，减小循环次数；额外+1是为了处理奇数行时场景
  for (uint16_t i = 0; i < loopCnt; i++) {
    MicroAPI::LoadAlign<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_UNPACK_B16>(
        (MicroAPI::RegTensor<uint8_t>&)vKvData, ubSrcTemp, baseSize * 2);

    // cast操作, Fp8->Fp32
    MicroAPI::Cast<float, KV_T, castTraitFp8_1>(vCastFp32Res0, vKvData, kvTypeMaskAll);
    MicroAPI::Cast<float, KV_T, castTraitFp8_2>(vCastFp32Res1, vKvData, kvTypeMaskAll);

    // cast操作, Fp32->Fp16/Bf16
    MicroAPI::Cast<Q_T, float, castTraitFp8_3>(vCastRes0, vCastFp32Res0, kvTypeMaskAll);
    MicroAPI::Cast<Q_T, float, castTraitFp8_4>(vCastRes1, vCastFp32Res1, kvTypeMaskAll);

    MicroAPI::Or<uint16_t, MicroAPI::MaskMergeMode::ZEROING>((MicroAPI::RegTensor<uint16_t>&)vCastRes0,
                                                            (MicroAPI::RegTensor<uint16_t>&)vCastRes0,
                                                            (MicroAPI::RegTensor<uint16_t>&)vCastRes1, kvTypeMaskAll);

    MicroAPI::Mul<Q_T, MicroAPI::MaskMergeMode::ZEROING>(vMulRes, vCastRes0, vScale, qTypeMaskLower128);

    // 将输出结果copy到UB
    MicroAPI::StoreAlign<Q_T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
        ubDstAddr, vMulRes, blockStride, repeatStride, qTypeMaskLower64);

    MicroAPI::StoreAlign<Q_T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
        ubDstAddr_, vMulRes, blockStride, repeatStride, qTypeMaskHigher64);
  }
}

template <typename Q_T, typename KV_T, uint32_t baseSize>
__aicore__ inline void AntiquantVFFp8D64(LocalTensor<KV_T>& antiqInUb, LocalTensor<Q_T>& antiqResUb,
                                          LocalTensor<Q_T>& antiqScaleFp16Ub, uint32_t dealRowCount) {
  static_assert(baseSize == 64);
  static_assert(IsSameType<KV_T, fp8_e4m3fn_t>::value || IsSameType<KV_T, fp8_e5m2_t>::value,
                "antiquant w8, KV_T must be fp8_e4m3fn_t or fp8_e5m2_t");
  __ubuf__ uint8_t* ubSrcAddr = (__ubuf__ uint8_t*)(antiqInUb.GetPhyAddr());
  __ubuf__ Q_T* ubDstAddr = (__ubuf__ Q_T*)(antiqResUb.GetPhyAddr());
  __ubuf__ Q_T* ubDstAddrEven = (__ubuf__ Q_T*)(antiqResUb.GetPhyAddr());
  __ubuf__ Q_T *ubDstAddr_ = ubDstAddr + 16 - (dealRowCount + 1) * 32 * 4 / 2;
  __ubuf__ Q_T* ubScalerSrcAddr = (__ubuf__ Q_T*)antiqScaleFp16Ub.GetPhyAddr();

  AntiquantVFImplFp8D64<Q_T, KV_T, baseSize>(ubSrcAddr, ubDstAddr, ubDstAddrEven, ubDstAddr_, ubScalerSrcAddr, dealRowCount);
}

template <typename Q_T, typename KV_T, uint32_t baseSize>
__simd_vf__ void AntiquantVFImplFp8Norm(__ubuf__ uint8_t* ubSrcAddr, __ubuf__ Q_T* ubDstAddr, 
                                      __ubuf__ Q_T* ubScalerSrcAddr, uint32_t dealRowCount)
{
  MicroAPI::RegTensor<KV_T> vKvData;
  MicroAPI::RegTensor<float> vCastFp32Res0;
  MicroAPI::RegTensor<float> vCastFp32Res1;
  MicroAPI::RegTensor<Q_T> vCastRes0;
  MicroAPI::RegTensor<Q_T> vCastRes1;
  MicroAPI::RegTensor<Q_T> vScale;
  MicroAPI::RegTensor<Q_T> vMulRes;

  MicroAPI::MaskReg kvTypeMaskAll = MicroAPI::CreateMask<KV_T, MicroAPI::MaskPattern::ALL>();
  MicroAPI::MaskReg qTypeMaskAll = MicroAPI::CreateMask<Q_T, MicroAPI::MaskPattern::ALL>();

  // 目前不支持D泛化, 仅支持D128对齐场景

  // NZ:
  uint32_t blockStride = dealRowCount + 1;
  uint32_t repeatStride = 1;
  const uint16_t loops = static_cast<uint16_t>(baseSize / 128);
  for (uint16_t j = 0; j < loops; j++) {
    __ubuf__ Q_T* ubDstAddrOdd = ubDstAddr + blockStride * 128 * j;
    __ubuf__ uint8_t* ubSrcTemp = ubSrcAddr + j * 128;
    // 加载 scale
    MicroAPI::LoadAlign<Q_T, MicroAPI::LoadDist::DIST_NORM>(vScale, ubScalerSrcAddr + j * 128);

    for (uint16_t i = 0; i < static_cast<uint16_t>(dealRowCount); i++) {  // 共处理dealRowCount * 128个Fp8元素
      MicroAPI::LoadAlign<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_UNPACK_B16>(
          (MicroAPI::RegTensor<uint8_t>&)vKvData, ubSrcTemp, baseSize);

      // cast操作, Fp8->Fp32
      MicroAPI::Cast<float, KV_T, castTraitFp8_1>(vCastFp32Res0, vKvData, kvTypeMaskAll);
      MicroAPI::Cast<float, KV_T, castTraitFp8_2>(vCastFp32Res1, vKvData, kvTypeMaskAll);

      // cast操作, Fp32->Fp16/Bf16
      MicroAPI::Cast<Q_T, float, castTraitFp8_3>(vCastRes0, vCastFp32Res0, kvTypeMaskAll);
      MicroAPI::Cast<Q_T, float, castTraitFp8_4>(vCastRes1, vCastFp32Res1, kvTypeMaskAll);

      MicroAPI::Or<uint16_t, MicroAPI::MaskMergeMode::ZEROING>((MicroAPI::RegTensor<uint16_t>&)vCastRes0,
                                                              (MicroAPI::RegTensor<uint16_t>&)vCastRes0,
                                                              (MicroAPI::RegTensor<uint16_t>&)vCastRes1, kvTypeMaskAll);

      MicroAPI::Mul<Q_T, MicroAPI::MaskMergeMode::ZEROING>(vMulRes, vCastRes0, vScale, qTypeMaskAll);

      // 将输出结果copy到UB
      MicroAPI::StoreAlign<Q_T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
          ubDstAddrOdd, vMulRes, blockStride, repeatStride, qTypeMaskAll);
    }
  }
}

template <typename Q_T, typename KV_T, uint32_t baseSize>
__aicore__ inline void AntiquantVFFp8Norm(LocalTensor<KV_T>& antiqInUb, LocalTensor<Q_T>& antiqResUb,
                                          LocalTensor<Q_T>& antiqScaleFp16Ub, uint32_t dealRowCount) {
  static_assert(baseSize % 128 == 0);
  static_assert(IsSameType<KV_T, fp8_e4m3fn_t>::value || IsSameType<KV_T, fp8_e5m2_t>::value,
                "antiquant w8, KV_T must be fp8_e4m3fn_t or fp8_e5m2_t");
  __ubuf__ uint8_t* ubSrcAddr = (__ubuf__ uint8_t*)(antiqInUb.GetPhyAddr());
  __ubuf__ Q_T* ubDstAddr = (__ubuf__ Q_T*)(antiqResUb.GetPhyAddr());
  __ubuf__ Q_T* ubScalerSrcAddr = (__ubuf__ Q_T*)antiqScaleFp16Ub.GetPhyAddr();

  AntiquantVFImplFp8Norm<Q_T, KV_T, baseSize>(ubSrcAddr, ubDstAddr, ubScalerSrcAddr, dealRowCount);
}

template <typename Q_T, typename ANTIQ_PARAMS_T, uint32_t baseSize, bool hasOffset = false, bool isPerToken = false, bool isKvCacheNz = false>
__aicore__ inline void AntiquantVFImpl(LocalTensor<fp8_e5m2_t>& antiqInUb, LocalTensor<Q_T>& antiqResUb,
                                       LocalTensor<ANTIQ_PARAMS_T>& antiqOffsetUb, LocalTensor<ANTIQ_PARAMS_T>& antiqScaleUb,
                                       uint32_t dealRowCount, uint32_t headDim, uint32_t copyTotalS) {
  if constexpr (isKvCacheNz) {
    AntiquantVFFp8Nz<Q_T, fp8_e5m2_t, baseSize>(antiqInUb, antiqResUb, antiqScaleUb, dealRowCount);
  } else {
    if constexpr (baseSize == 64) {
      AntiquantVFFp8D64<Q_T, fp8_e5m2_t, baseSize>(antiqInUb, antiqResUb, antiqScaleUb, dealRowCount);
    } else {
      AntiquantVFFp8Norm<Q_T, fp8_e5m2_t, baseSize>(antiqInUb, antiqResUb, antiqScaleUb, dealRowCount);
    }
  }
}

template <typename Q_T, typename KV_T>
__simd_callee__ static inline void Fp8CastToRes(MicroAPI::RegTensor<Q_T>& vRes,
                                                MicroAPI::RegTensor<KV_T>& vKvData,
                                                MicroAPI::RegTensor<float>& vCastFp32Res0,
                                                MicroAPI::RegTensor<float>& vCastFp32Res1,
                                                MicroAPI::RegTensor<Q_T>& vCastRes0,
                                                MicroAPI::RegTensor<Q_T>& vCastRes1,
                                                MicroAPI::MaskReg& kvMaskAll)
{
  MicroAPI::Cast<float, KV_T, castTraitFp8_1>(vCastFp32Res0, vKvData, kvMaskAll);
  MicroAPI::Cast<float, KV_T, castTraitFp8_2>(vCastFp32Res1, vKvData, kvMaskAll);
  MicroAPI::Cast<Q_T, float, castTraitFp8_3>(vCastRes0, vCastFp32Res0, kvMaskAll);
  MicroAPI::Cast<Q_T, float, castTraitFp8_4>(vCastRes1, vCastFp32Res1, kvMaskAll);
  MicroAPI::Or<uint16_t, MicroAPI::MaskMergeMode::ZEROING>((MicroAPI::RegTensor<uint16_t>&)vRes,
                                                              (MicroAPI::RegTensor<uint16_t>&)vCastRes0,
                                                              (MicroAPI::RegTensor<uint16_t>&)vCastRes1, kvMaskAll);
}

template <typename Q_T, typename KV_T, typename ANTIQ_PARAMS_T, uint32_t baseSize, bool hasOffset = false>
__simd_vf__ void AntiquantVFImplFp8PerTokenD64(__ubuf__ uint8_t* ubSrcAddr, __ubuf__ Q_T* ubDstAddr, 
                                              __ubuf__ Q_T *ubDstAddr_, __ubuf__ ANTIQ_PARAMS_T* ubOffsetAddr, 
                                              __ubuf__ ANTIQ_PARAMS_T* ubScaleAddr, uint32_t dealRowCount)
{
  MicroAPI::RegTensor<KV_T> vKvData;
  MicroAPI::RegTensor<ANTIQ_PARAMS_T> vOffset;
  MicroAPI::RegTensor<ANTIQ_PARAMS_T> vScale;
  MicroAPI::RegTensor<float> vCastFp32Res0;
  MicroAPI::RegTensor<float> vCastFp32Res1;
  MicroAPI::RegTensor<Q_T> vCastRes0;
  MicroAPI::RegTensor<Q_T> vCastRes1;
  MicroAPI::RegTensor<Q_T> vOffsetFp16;
  MicroAPI::RegTensor<Q_T> vOffsetFp16High;
  MicroAPI::RegTensor<Q_T> vOffsetFp16Low;
  MicroAPI::RegTensor<Q_T> vScaleFp16;
  MicroAPI::RegTensor<Q_T> vScaleFp16High;
  MicroAPI::RegTensor<Q_T> vScaleFp16Low;
  MicroAPI::MaskReg maskOne = MicroAPI::CreateMask<ANTIQ_PARAMS_T, MicroAPI::MaskPattern::VL1>();
  MicroAPI::MaskReg kvTypeMaskAll = MicroAPI::CreateMask<KV_T, MicroAPI::MaskPattern::ALL>();
  MicroAPI::MaskReg qTypeMaskLower64 = MicroAPI::CreateMask<Q_T, MicroAPI::MaskPattern::VL64>();
  MicroAPI::MaskReg qTypeMaskLower128 = MicroAPI::CreateMask<Q_T, MicroAPI::MaskPattern::VL128>();
  MicroAPI::MaskReg qTypeMaskAll = MicroAPI::CreateMask<Q_T, MicroAPI::MaskPattern::ALL>(); // Q_T 所有元素（共128个）
  MicroAPI::MaskReg qTypeMaskHigher64;
  MicroAPI::Xor(qTypeMaskHigher64, qTypeMaskLower64, qTypeMaskAll, qTypeMaskAll); // 异或得到higher64

  MicroAPI::UnalignRegForLoad u0;
  MicroAPI::UnalignRegForLoad u1;

  uint32_t blockStride = dealRowCount + 1;
  uint32_t repeatStride = 2;

  MicroAPI::LoadUnAlignPre(u0, ubScaleAddr);
  if constexpr (hasOffset) {
    MicroAPI::LoadUnAlignPre(u1, ubOffsetAddr);
  }
  uint16_t loopCnt = static_cast<uint16_t>((dealRowCount + 1) / 2); // +1是为了兼容处理奇数行
  // 对D64优化，相邻2行合并计算；+1兼容奇数行场景
  for (uint16_t i = 0; i < loopCnt; i++) {
    // POST_MODE_UPDATE 表示 UB 地址在搬入后要自动更新
    MicroAPI::LoadAlign<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_UNPACK_B16>(
        (MicroAPI::RegTensor<uint8_t>&)vKvData, ubSrcAddr, baseSize * 2);
    Fp8CastToRes<Q_T, KV_T>(vCastRes0, vKvData, vCastFp32Res0, vCastFp32Res1, vCastRes0, vCastRes1, kvTypeMaskAll);
    LoadCastDupLowHighOr<Q_T, ANTIQ_PARAMS_T>(vScale, u0, ubScaleAddr, vScaleFp16, vScaleFp16Low, vScaleFp16High,
                                                maskOne, qTypeMaskLower64, qTypeMaskHigher64, kvTypeMaskAll);
    if constexpr (hasOffset) {
      LoadCastDupLowHighOr<Q_T, ANTIQ_PARAMS_T>(vOffset, u1, ubOffsetAddr, vOffsetFp16, vOffsetFp16Low,
                                                 vOffsetFp16High, maskOne, qTypeMaskLower64,
                                                 qTypeMaskHigher64, kvTypeMaskAll);
      MicroAPI::Add<Q_T, MicroAPI::MaskMergeMode::ZEROING>(vCastRes0, vCastRes0, vOffsetFp16, qTypeMaskLower128);
    }

    MicroAPI::Mul<Q_T, MicroAPI::MaskMergeMode::ZEROING>(vCastRes0, vCastRes0, vScaleFp16, qTypeMaskLower128);

    MicroAPI::StoreAlign<Q_T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
        ubDstAddr, vCastRes0, blockStride, repeatStride, qTypeMaskLower64);
    MicroAPI::StoreAlign<Q_T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
        ubDstAddr_, vCastRes0, blockStride, repeatStride, qTypeMaskHigher64);
  }
}

template <typename Q_T, typename KV_T, typename ANTIQ_PARAMS_T,
  uint32_t baseSize, bool hasOffset = false>
__aicore__ inline void AntiquantVFFp8PerTokenD64(LocalTensor<KV_T>& antiqInUb, LocalTensor<Q_T>& antiqResUb,
                                                    LocalTensor<ANTIQ_PARAMS_T>& antiqOffsetUb, LocalTensor<ANTIQ_PARAMS_T>& antiqScaleUb,
                                                    uint32_t dealRowCount) {
  ASCENDC_ASSERT((baseSize == 64),
                 {KERNEL_LOG(KERNEL_ERROR, "baseSize is %d, which must be 64.", baseSize); });
  ASCENDC_ASSERT((IsSameType<KV_T, fp8_e4m3fn_t>::value),
                 {KERNEL_LOG(KERNEL_ERROR, "Antiquant fp8 PerToken, KV_T must be fp8_e4m3."); });  
  __ubuf__ uint8_t* ubSrcAddr = (__ubuf__ uint8_t*)(antiqInUb.GetPhyAddr());
  __ubuf__ Q_T* ubDstAddr = (__ubuf__ Q_T*)(antiqResUb.GetPhyAddr());
  __ubuf__ Q_T *ubDstAddr_ = ubDstAddr + 16 - (dealRowCount + 1) * 32 * 4 / 2;  
  __ubuf__ ANTIQ_PARAMS_T* ubOffsetAddr = (__ubuf__ ANTIQ_PARAMS_T*)antiqOffsetUb.GetPhyAddr();
  __ubuf__ ANTIQ_PARAMS_T* ubScaleAddr = (__ubuf__ ANTIQ_PARAMS_T*)antiqScaleUb.GetPhyAddr();

  AntiquantVFImplFp8PerTokenD64<Q_T, KV_T, ANTIQ_PARAMS_T, baseSize, hasOffset>(
    ubSrcAddr, ubDstAddr, ubDstAddr_, ubOffsetAddr, ubScaleAddr, dealRowCount);
}

template <typename Q_T, typename KV_T, typename ANTIQ_PARAMS_T, uint32_t baseSize, bool hasOffset = false>
__simd_vf__ void AntiquantVFImplFp8PerTokenD128(__ubuf__ uint8_t* ubSrcAddr, __ubuf__ Q_T* ubDstAddr,
                                          __ubuf__ ANTIQ_PARAMS_T* ubOffsetAddr, __ubuf__ ANTIQ_PARAMS_T* ubScaleAddr, 
                                          uint32_t dealRowCount)
{
  MicroAPI::RegTensor<KV_T> vKvData;
  MicroAPI::RegTensor<ANTIQ_PARAMS_T> vOffset;
  MicroAPI::RegTensor<ANTIQ_PARAMS_T> vScale;
  MicroAPI::RegTensor<float> vCastFp32Res0;
  MicroAPI::RegTensor<float> vCastFp32Res1;
  MicroAPI::RegTensor<Q_T> vCastRes0;
  MicroAPI::RegTensor<Q_T> vCastRes1;
  MicroAPI::RegTensor<Q_T> vOffsetFp16;
  MicroAPI::RegTensor<Q_T> vScaleFp16;
  MicroAPI::RegTensor<Q_T> vRes;
  MicroAPI::RegTensor<half> vCastFp16Res;

  MicroAPI::MaskReg maskOne = MicroAPI::CreateMask<ANTIQ_PARAMS_T, MicroAPI::MaskPattern::VL1>();
  MicroAPI::MaskReg kvTypeMaskAll = MicroAPI::CreateMask<KV_T, MicroAPI::MaskPattern::ALL>();
  MicroAPI::MaskReg qTypeMaskAll = MicroAPI::CreateMask<Q_T, MicroAPI::MaskPattern::ALL>(); // Q_T 所有元素（共128个）

  uint32_t blockStride = dealRowCount + 1;
  uint32_t repeatStride = 1;
  const uint16_t loops = baseSize / 128;
  MicroAPI::UnalignRegForLoad u0;
  MicroAPI::UnalignRegForLoad u1;
  for (uint16_t j = 0; j < static_cast<uint16_t>(loops); j++) {
    __ubuf__ Q_T* ubDstAddrTmp = ubDstAddr + blockStride * 128 * j;
    __ubuf__ uint8_t* ubSrcTemp = ubSrcAddr + j * 128;
    __ubuf__ ANTIQ_PARAMS_T* ubScaleAddrTemp = ubScaleAddr;
    __ubuf__ ANTIQ_PARAMS_T* ubOffsetAddrTemp = ubOffsetAddr;

    MicroAPI::LoadUnAlignPre(u0, ubScaleAddrTemp);
    if constexpr (hasOffset) {
      MicroAPI::LoadUnAlignPre(u1, ubOffsetAddrTemp);
    }
    for (uint16_t i = 0; i < static_cast<uint16_t>(dealRowCount); i++) {
      MicroAPI::LoadAlign<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_UNPACK_B16>(
          (MicroAPI::RegTensor<uint8_t>&)vKvData, ubSrcTemp, baseSize);

      Fp8CastToRes<Q_T, KV_T>(vRes, vKvData, vCastFp32Res0, vCastFp32Res1, vCastRes0, vCastRes1, kvTypeMaskAll);
      MicroAPI::LoadUnAlign<ANTIQ_PARAMS_T>(vScale, u0, ubScaleAddrTemp, 1); // 1表示ub自动往后偏移1个float
      MicroAPI::Cast<Q_T, ANTIQ_PARAMS_T, castTrait0>(vScaleFp16, vScale, maskOne);
      MicroAPI::Duplicate<uint16_t, MicroAPI::HighLowPart::LOWEST, MicroAPI::MaskMergeMode::ZEROING>
        ((MicroAPI::RegTensor<uint16_t>&)vScaleFp16, (MicroAPI::RegTensor<uint16_t>&)vScaleFp16, qTypeMaskAll);

      if constexpr (hasOffset) {
        MicroAPI::LoadUnAlign<ANTIQ_PARAMS_T>(vOffset, u1, ubOffsetAddrTemp, 1); // 1表示ub自动往后偏移1个float
        MicroAPI::Cast<Q_T, ANTIQ_PARAMS_T, castTrait0>(vOffsetFp16, vOffset, maskOne);
        MicroAPI::Duplicate<uint16_t, MicroAPI::HighLowPart::LOWEST, MicroAPI::MaskMergeMode::ZEROING>
          ((MicroAPI::RegTensor<uint16_t>&)vOffsetFp16, (MicroAPI::RegTensor<uint16_t>&)vOffsetFp16, qTypeMaskAll);
        MicroAPI::Add<Q_T, MicroAPI::MaskMergeMode::ZEROING>(vRes, vRes, vOffsetFp16, qTypeMaskAll);
      }

      MicroAPI::Mul<Q_T, MicroAPI::MaskMergeMode::ZEROING>(vRes, vRes, vScaleFp16, qTypeMaskAll);

      MicroAPI::StoreAlign<Q_T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
          ubDstAddrTmp, vRes, blockStride, repeatStride, qTypeMaskAll);
    }
  }
}

template <typename Q_T, typename KV_T, typename ANTIQ_PARAMS_T,
  uint32_t baseSize, bool hasOffset = false>
__aicore__ inline void AntiquantVFFp8PerTokenD128(LocalTensor<KV_T>& antiqInUb, LocalTensor<Q_T>& antiqResUb,
                                             LocalTensor<ANTIQ_PARAMS_T>& antiqOffsetUb, LocalTensor<ANTIQ_PARAMS_T>& antiqScaleUb,
                                             uint32_t dealRowCount) {
  ASCENDC_ASSERT((baseSize == 128),
                 {KERNEL_LOG(KERNEL_ERROR, "baseSize is %d, which must be 128.", baseSize); });
  ASCENDC_ASSERT((IsSameType<KV_T, fp8_e4m3fn_t>::value),
                 {KERNEL_LOG(KERNEL_ERROR, "Antiquant fp8 PerToken, KV_T must be fp8_e4m3."); });  
  __ubuf__ uint8_t* ubSrcAddr = (__ubuf__ uint8_t*)(antiqInUb.GetPhyAddr());
  __ubuf__ Q_T* ubDstAddr = (__ubuf__ Q_T*)(antiqResUb.GetPhyAddr());
  __ubuf__ ANTIQ_PARAMS_T* ubOffsetAddr = (__ubuf__ ANTIQ_PARAMS_T*)antiqOffsetUb.GetPhyAddr();
  __ubuf__ ANTIQ_PARAMS_T* ubScaleAddr = (__ubuf__ ANTIQ_PARAMS_T*)antiqScaleUb.GetPhyAddr();

  AntiquantVFImplFp8PerTokenD128<Q_T, KV_T, ANTIQ_PARAMS_T, baseSize, hasOffset>(
    ubSrcAddr, ubDstAddr, ubOffsetAddr, ubScaleAddr, dealRowCount);
}

template <typename Q_T, typename KV_T, typename ANTIQ_PARAMS_T, uint32_t baseSize, bool hasOffset = false>
__simd_vf__ void AntiquantVFImplFp8PerTokenD256(__ubuf__ uint8_t* ubSrcAddr, __ubuf__ uint8_t* ubSrcAddr1, 
                                              __ubuf__ Q_T* ubDstAddr, __ubuf__ ANTIQ_PARAMS_T* ubOffsetAddr, 
                                              __ubuf__ ANTIQ_PARAMS_T* ubScaleAddr, uint32_t dealRowCount)
{
  MicroAPI::RegTensor<KV_T> vKvData;
  MicroAPI::RegTensor<KV_T> vKvData1;
  MicroAPI::RegTensor<ANTIQ_PARAMS_T> vOffset;
  MicroAPI::RegTensor<ANTIQ_PARAMS_T> vScale;
  MicroAPI::RegTensor<Q_T> vOffsetFp16;
  MicroAPI::RegTensor<Q_T> vScaleFp16;

  MicroAPI::RegTensor<Q_T> vRes;
  MicroAPI::RegTensor<Q_T> vRes1;
  MicroAPI::RegTensor<float> vCastFp32Res0;
  MicroAPI::RegTensor<float> vCastFp32Res1;
  MicroAPI::RegTensor<Q_T> vCastRes0;
  MicroAPI::RegTensor<Q_T> vCastRes1;

  MicroAPI::MaskReg maskOne = MicroAPI::CreateMask<ANTIQ_PARAMS_T, MicroAPI::MaskPattern::VL1>();
  MicroAPI::MaskReg kvMaskAll = MicroAPI::CreateMask<KV_T, MicroAPI::MaskPattern::ALL>();
  MicroAPI::MaskReg qMaskAll = MicroAPI::CreateMask<Q_T, MicroAPI::MaskPattern::ALL>(); // Q_T 所有元素（共128个）

  uint32_t blockStride = dealRowCount + 1;
  uint32_t repeatStride = 1;
  MicroAPI::UnalignRegForLoad u0;
  MicroAPI::UnalignRegForLoad u1;
  MicroAPI::LoadUnAlignPre(u0, ubScaleAddr);
  if constexpr (hasOffset) {
    MicroAPI::LoadUnAlignPre(u1, ubOffsetAddr);
  }
  __ubuf__ Q_T* ubDstAddr1 = ubDstAddr + blockStride * 128;

  for (uint16_t j = 0; j < static_cast<uint16_t>(dealRowCount); j++) {
    // 读入每行的伪量化参数
    MicroAPI::LoadUnAlign<ANTIQ_PARAMS_T>(vScale, u0, ubScaleAddr, 1); // 1表示ub自动往后偏移1个float
    MicroAPI::Cast<Q_T, ANTIQ_PARAMS_T, castTrait0>(vScaleFp16, vScale, maskOne);
    MicroAPI::Duplicate<uint16_t, MicroAPI::HighLowPart::LOWEST, MicroAPI::MaskMergeMode::ZEROING>
      ((MicroAPI::RegTensor<uint16_t>&)vScaleFp16, (MicroAPI::RegTensor<uint16_t>&)vScaleFp16, qMaskAll);
    if constexpr (hasOffset) {
      MicroAPI::LoadUnAlign<ANTIQ_PARAMS_T>(vOffset, u1, ubOffsetAddr, 1); // 1表示ub自动往后偏移1个float
      MicroAPI::Cast<Q_T, ANTIQ_PARAMS_T, castTrait0>(vOffsetFp16, vOffset, maskOne);
      MicroAPI::Duplicate<uint16_t, MicroAPI::HighLowPart::LOWEST, MicroAPI::MaskMergeMode::ZEROING>
        ((MicroAPI::RegTensor<uint16_t>&)vOffsetFp16, (MicroAPI::RegTensor<uint16_t>&)vOffsetFp16, qMaskAll);
    }

    MicroAPI::LoadAlign<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_UNPACK_B16>(
      (MicroAPI::RegTensor<uint8_t>&)vKvData, ubSrcAddr, 256); // d=256，自动往后偏移256个数
    MicroAPI::LoadAlign<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_UNPACK_B16>(
      (MicroAPI::RegTensor<uint8_t>&)vKvData1, ubSrcAddr1, 256); // d=256

    Fp8CastToRes<Q_T, KV_T>(vRes, vKvData, vCastFp32Res0, vCastFp32Res1, vCastRes0, vCastRes1, kvMaskAll);
    Fp8CastToRes<Q_T, KV_T>(vRes1, vKvData1, vCastFp32Res0, vCastFp32Res1, vCastRes0, vCastRes1, kvMaskAll);

    MicroAPI::Mul<Q_T, MicroAPI::MaskMergeMode::ZEROING>(vRes, vRes, vScaleFp16, qMaskAll);
    MicroAPI::Mul<Q_T, MicroAPI::MaskMergeMode::ZEROING>(vRes1, vRes1, vScaleFp16, qMaskAll);

    MicroAPI::StoreAlign<Q_T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
        ubDstAddr, vRes, blockStride, repeatStride, qMaskAll);
    MicroAPI::StoreAlign<Q_T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
        ubDstAddr1, vRes1, blockStride, repeatStride, qMaskAll);
  }
}

template <typename Q_T, typename KV_T, typename ANTIQ_PARAMS_T,
  uint32_t baseSize, bool hasOffset = false>
__aicore__ inline void AntiquantVFFp8PerTokenD256(LocalTensor<KV_T>& antiqInUb, LocalTensor<Q_T>& antiqResUb,
                                             LocalTensor<ANTIQ_PARAMS_T>& antiqOffsetUb, LocalTensor<ANTIQ_PARAMS_T>& antiqScaleUb,
                                             uint32_t dealRowCount) {
  ASCENDC_ASSERT((baseSize == 256),
                 {KERNEL_LOG(KERNEL_ERROR, "baseSize is %d, which must be 256.", baseSize); });
  ASCENDC_ASSERT((IsSameType<KV_T, fp8_e4m3fn_t>::value),
                 {KERNEL_LOG(KERNEL_ERROR, "Antiquant fp8 PerToken, KV_T must be fp8_e4m3."); });  
  __ubuf__ uint8_t* ubSrcAddr = (__ubuf__ uint8_t*)(antiqInUb.GetPhyAddr());
  __ubuf__ uint8_t* ubSrcAddr1 = ubSrcAddr + 128;
  __ubuf__ Q_T* ubDstAddr = (__ubuf__ Q_T*)(antiqResUb.GetPhyAddr());
  __ubuf__ ANTIQ_PARAMS_T* ubOffsetAddr = (__ubuf__ ANTIQ_PARAMS_T*)antiqOffsetUb.GetPhyAddr();
  __ubuf__ ANTIQ_PARAMS_T* ubScaleAddr = (__ubuf__ ANTIQ_PARAMS_T*)antiqScaleUb.GetPhyAddr();

  AntiquantVFImplFp8PerTokenD256<Q_T, KV_T, ANTIQ_PARAMS_T, baseSize, hasOffset>(
    ubSrcAddr, ubSrcAddr1, ubDstAddr, ubOffsetAddr, ubScaleAddr, dealRowCount);
}

template <typename KV_T>
__simd_callee__ static inline void LoadAlignFp8Quad(
    MicroAPI::RegTensor<KV_T>& vKvData, MicroAPI::RegTensor<KV_T>& vKvData1,
    MicroAPI::RegTensor<KV_T>& vKvData2, MicroAPI::RegTensor<KV_T>& vKvData3,
    __ubuf__ uint8_t*& ubSrcAddr, __ubuf__ uint8_t*& ubSrcAddr1,
    __ubuf__ uint8_t*& ubSrcAddr2, __ubuf__ uint8_t*& ubSrcAddr3)
{
  MicroAPI::LoadAlign<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_UNPACK_B16>(
      (MicroAPI::RegTensor<uint8_t>&)vKvData, ubSrcAddr, 512); // d=512 每次往后偏移512
  MicroAPI::LoadAlign<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_UNPACK_B16>(
      (MicroAPI::RegTensor<uint8_t>&)vKvData1, ubSrcAddr1, 512); // d=512
  MicroAPI::LoadAlign<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_UNPACK_B16>(
      (MicroAPI::RegTensor<uint8_t>&)vKvData2, ubSrcAddr2, 512); // d=512
  MicroAPI::LoadAlign<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_UNPACK_B16>(
      (MicroAPI::RegTensor<uint8_t>&)vKvData3, ubSrcAddr3, 512); // d=512
}

template <typename Q_T, typename KV_T, typename ANTIQ_PARAMS_T, uint32_t baseSize, bool hasOffset = false>
__simd_vf__ void AntiquantVFImplFp8PerTokenD512(__ubuf__ uint8_t* ubSrcAddr, __ubuf__ uint8_t* ubSrcAddr1, 
                                              __ubuf__ uint8_t* ubSrcAddr2, __ubuf__ uint8_t* ubSrcAddr3, 
                                              __ubuf__ Q_T* ubDstAddr, __ubuf__ ANTIQ_PARAMS_T* ubOffsetAddr, 
                                              __ubuf__ ANTIQ_PARAMS_T* ubScaleAddr, uint32_t dealRowCount)
{
  MicroAPI::RegTensor<KV_T> vKvData;
  MicroAPI::RegTensor<KV_T> vKvData1;
  MicroAPI::RegTensor<KV_T> vKvData2;
  MicroAPI::RegTensor<KV_T> vKvData3;
  MicroAPI::RegTensor<ANTIQ_PARAMS_T> vOffset;
  MicroAPI::RegTensor<ANTIQ_PARAMS_T> vScale;
  MicroAPI::RegTensor<Q_T> vOffsetFp16;
  MicroAPI::RegTensor<Q_T> vScaleFp16;
  MicroAPI::RegTensor<float> vCastFp32Res0;
  MicroAPI::RegTensor<float> vCastFp32Res1;
  MicroAPI::RegTensor<Q_T> vCastRes0;
  MicroAPI::RegTensor<Q_T> vCastRes1;

  MicroAPI::RegTensor<Q_T> vRes;
  MicroAPI::RegTensor<Q_T> vRes1;
  MicroAPI::RegTensor<Q_T> vRes2;
  MicroAPI::RegTensor<Q_T> vRes3;

  MicroAPI::MaskReg maskOne = MicroAPI::CreateMask<ANTIQ_PARAMS_T, MicroAPI::MaskPattern::VL1>();
  MicroAPI::MaskReg kvMaskAll = MicroAPI::CreateMask<KV_T, MicroAPI::MaskPattern::ALL>();
  MicroAPI::MaskReg qMaskAll = MicroAPI::CreateMask<Q_T, MicroAPI::MaskPattern::ALL>(); // Q_T 所有元素（共128个）

  uint32_t repeatStride = 1;
  uint32_t blockStride = dealRowCount + 1;
  MicroAPI::UnalignRegForLoad u0;
  MicroAPI::UnalignRegForLoad u1;
  MicroAPI::LoadUnAlignPre(u0, ubScaleAddr);
  if constexpr (hasOffset) {
    MicroAPI::LoadUnAlignPre(u1, ubOffsetAddr);
  }
  __ubuf__ Q_T* ubDstAddr1 = ubDstAddr + blockStride * 128;
  __ubuf__ Q_T* ubDstAddr2 = ubDstAddr + blockStride * 256; // 128*2
  __ubuf__ Q_T* ubDstAddr3 = ubDstAddr + blockStride * 384; // 128*3

  for (uint16_t j = 0; j < static_cast<uint16_t>(dealRowCount); j++) {
    // 读入每行的伪量化参数
    LoadPerTokenAntiqParams<Q_T, ANTIQ_PARAMS_T, hasOffset>(vScale, u0, ubScaleAddr, vScaleFp16, maskOne,
      qMaskAll, vOffset, u1, ubOffsetAddr, vOffsetFp16);

    LoadAlignFp8Quad<KV_T>(vKvData, vKvData1, vKvData2, vKvData3,
                           ubSrcAddr, ubSrcAddr1, ubSrcAddr2, ubSrcAddr3);

    Fp8CastToRes<Q_T, KV_T>(vRes, vKvData, vCastFp32Res0, vCastFp32Res1, vCastRes0, vCastRes1, kvMaskAll);
    Fp8CastToRes<Q_T, KV_T>(vRes1, vKvData1, vCastFp32Res0, vCastFp32Res1, vCastRes0, vCastRes1, kvMaskAll);
    Fp8CastToRes<Q_T, KV_T>(vRes2, vKvData2, vCastFp32Res0, vCastFp32Res1, vCastRes0, vCastRes1, kvMaskAll);
    Fp8CastToRes<Q_T, KV_T>(vRes3, vKvData3, vCastFp32Res0, vCastFp32Res1, vCastRes0, vCastRes1, kvMaskAll);

    AddMulStoreFp8Quad<Q_T, hasOffset>(vRes, vRes1, vRes2, vRes3, vOffsetFp16, vScaleFp16,
                                        ubDstAddr, ubDstAddr1, ubDstAddr2, ubDstAddr3,
                                        blockStride, repeatStride, qMaskAll);
  }
}

template <typename Q_T, typename KV_T, typename ANTIQ_PARAMS_T,
  uint32_t baseSize, bool hasOffset = false>
__aicore__ inline void AntiquantVFFp8PerTokenD512(LocalTensor<KV_T>& antiqInUb, LocalTensor<Q_T>& antiqResUb,
                                             LocalTensor<ANTIQ_PARAMS_T>& antiqOffsetUb, LocalTensor<ANTIQ_PARAMS_T>& antiqScaleUb,
                                             uint32_t dealRowCount) {
  ASCENDC_ASSERT((baseSize == 512),
                 {KERNEL_LOG(KERNEL_ERROR, "baseSize is %d, which must be 512.", baseSize); });
  ASCENDC_ASSERT((IsSameType<KV_T, fp8_e4m3fn_t>::value),
                 {KERNEL_LOG(KERNEL_ERROR, "Antiquant fp8 PerToken, KV_T must be fp8_e4m3."); });  
  __ubuf__ uint8_t* ubSrcAddr = (__ubuf__ uint8_t*)(antiqInUb.GetPhyAddr());
  __ubuf__ uint8_t* ubSrcAddr1 = ubSrcAddr + 128;
  __ubuf__ uint8_t* ubSrcAddr2 = ubSrcAddr + 256; // 128*2
  __ubuf__ uint8_t* ubSrcAddr3 = ubSrcAddr + 384; // 128*3
  __ubuf__ Q_T* ubDstAddr = (__ubuf__ Q_T*)(antiqResUb.GetPhyAddr());
  __ubuf__ ANTIQ_PARAMS_T* ubOffsetAddr = (__ubuf__ ANTIQ_PARAMS_T*)antiqOffsetUb.GetPhyAddr();
  __ubuf__ ANTIQ_PARAMS_T* ubScaleAddr = (__ubuf__ ANTIQ_PARAMS_T*)antiqScaleUb.GetPhyAddr();

  AntiquantVFImplFp8PerTokenD512<Q_T, KV_T, ANTIQ_PARAMS_T, baseSize, hasOffset>(
    ubSrcAddr, ubSrcAddr1, ubSrcAddr2, ubSrcAddr3, ubDstAddr, ubOffsetAddr, ubScaleAddr, dealRowCount);
}

template <typename Q_T, typename KV_T, bool hasOffset>
__simd_callee__ static inline void Fp8NzQuantAndStore(MicroAPI::RegTensor<KV_T>& vKvData,
                                                      MicroAPI::RegTensor<Q_T>& vRes,
                                                      __ubuf__ uint8_t* ubSrc,
                                                      MicroAPI::RegTensor<Q_T>& vOffset,
                                                      MicroAPI::RegTensor<Q_T>& vScale,
                                                      __ubuf__ Q_T* ubDst,
                                                      MicroAPI::RegTensor<float>& vCastFp32Res0,
                                                      MicroAPI::RegTensor<float>& vCastFp32Res1,
                                                      MicroAPI::RegTensor<Q_T>& vCastRes0,
                                                      MicroAPI::RegTensor<Q_T>& vCastRes1,
                                                      MicroAPI::MaskReg& kvTypeMaskAll,
                                                      MicroAPI::MaskReg& qTypeMaskAll)
{
  MicroAPI::LoadAlign<uint8_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(
      (MicroAPI::RegTensor<uint8_t>&)vKvData, ubSrc);
  MicroAPI::Cast<float, KV_T, castTraitFp8_1>(vCastFp32Res0, vKvData, kvTypeMaskAll);
  MicroAPI::Cast<float, KV_T, castTraitFp8_2>(vCastFp32Res1, vKvData, kvTypeMaskAll);
  MicroAPI::Cast<Q_T, float, castTraitFp8_3>(vCastRes0, vCastFp32Res0, kvTypeMaskAll);
  MicroAPI::Cast<Q_T, float, castTraitFp8_4>(vCastRes1, vCastFp32Res1, kvTypeMaskAll);
  MicroAPI::Or<uint16_t, MicroAPI::MaskMergeMode::ZEROING>((MicroAPI::RegTensor<uint16_t>&)vRes,
                                                              (MicroAPI::RegTensor<uint16_t>&)vCastRes0,
                                                              (MicroAPI::RegTensor<uint16_t>&)vCastRes1, kvTypeMaskAll);
  if constexpr (hasOffset) {
    MicroAPI::Add<Q_T, MicroAPI::MaskMergeMode::ZEROING>(vRes, vRes, vOffset, qTypeMaskAll);
  }
  MicroAPI::Mul<Q_T, MicroAPI::MaskMergeMode::ZEROING>(vRes, vRes, vScale, qTypeMaskAll);
  MicroAPI::StoreAlign<Q_T, MicroAPI::StoreDist::DIST_NORM_B16>(ubDst, vRes, qTypeMaskAll);
}

template <typename Q_T, typename KV_T, typename ANTIQ_PARAMS_T, uint32_t baseSize, bool hasOffset = false>
__simd_vf__ void AntiquantVFImplFp8PerTokenNz(__ubuf__ uint8_t* ubSrcAddr, __ubuf__ Q_T* ubDstAddr,
                                             __ubuf__ Q_T* ubOffsetAddr, __ubuf__ Q_T* ubScaleAddr,
                                             uint32_t dealRowCount)
{
  MicroAPI::RegTensor<KV_T> vKvData;
  MicroAPI::RegTensor<Q_T> vOffsetFirst;
  MicroAPI::RegTensor<Q_T> vOffsetBack;
  MicroAPI::RegTensor<Q_T> vScaleFirst;
  MicroAPI::RegTensor<Q_T> vScaleBack;
  MicroAPI::RegTensor<Q_T> vRes;
  MicroAPI::RegTensor<Q_T> vCastRes0;
  MicroAPI::RegTensor<Q_T> vCastRes1;
  MicroAPI::RegTensor<float> vCastFp32Res0;
  MicroAPI::RegTensor<float> vCastFp32Res1;
  

  MicroAPI::MaskReg kvTypeMaskAll = MicroAPI::CreateMask<KV_T, MicroAPI::MaskPattern::ALL>();
  MicroAPI::MaskReg qTypeMaskAll = MicroAPI::CreateMask<Q_T, MicroAPI::MaskPattern::ALL>(); // Q_T 所有元素（共128个）

  // UB总共dealRowCount行 * baseSize列，每次处理8行 * 16列 = 128个元素
  const uint32_t rowBaseSize = 8; // 8行
  const uint32_t colBaseSize = 16; // 16列
  const uint32_t dealBaseNum = 128; // 128个元素
  const uint32_t doubleRowBaseSize = 16; // 每16行交替，防止bank冲突

  const uint32_t rowStride = doubleRowBaseSize * colBaseSize; // 16 * 16
  const uint32_t colDstStride = dealRowCount * colBaseSize;
  const uint32_t colSrcStride = (dealRowCount * colBaseSize + 31) >> 5U << 5U; // 32B对齐
  const uint16_t colLoopCnt = static_cast<uint16_t>(baseSize / colBaseSize);
  const uint16_t rowLoopCnt = static_cast<uint16_t>((dealRowCount + doubleRowBaseSize - 1) / doubleRowBaseSize); // 16行对齐

  for (uint16_t rowLoop = 0; rowLoop < rowLoopCnt; rowLoop++) {
    uint16_t rowLoopIdx = rowLoopCnt - 1 - rowLoop;
    __ubuf__ Q_T* ubOffsetAddrTmp = ubOffsetAddr + rowLoopIdx * doubleRowBaseSize;
    __ubuf__ Q_T* ubScaleAddrTmp = ubScaleAddr + rowLoopIdx * doubleRowBaseSize;

    if constexpr (hasOffset) {
      MicroAPI::LoadAlign<Q_T, MicroAPI::LoadDist::DIST_E2B_B16>(vOffsetFirst, ubOffsetAddrTmp);
      MicroAPI::LoadAlign<Q_T, MicroAPI::LoadDist::DIST_E2B_B16>(vOffsetBack, ubOffsetAddrTmp + rowBaseSize);
    }
    MicroAPI::LoadAlign<Q_T, MicroAPI::LoadDist::DIST_E2B_B16>(vScaleFirst, ubScaleAddrTmp);
    MicroAPI::LoadAlign<Q_T, MicroAPI::LoadDist::DIST_E2B_B16>(vScaleBack, ubScaleAddrTmp + rowBaseSize);
    for (uint16_t colLoopIdx = 0; colLoopIdx < colLoopCnt; colLoopIdx++) {
      __ubuf__ uint8_t* ubSrcTemp = ubSrcAddr + rowStride * rowLoopIdx + colSrcStride * colLoopIdx;
      __ubuf__ Q_T* ubDstAddrTmp = ubDstAddr + rowStride * rowLoopIdx + colDstStride * colLoopIdx;

      // 后半组
      Fp8NzQuantAndStore<Q_T, KV_T, hasOffset>(vKvData, vRes, ubSrcTemp + dealBaseNum,
                                               vOffsetBack, vScaleBack, ubDstAddrTmp + dealBaseNum,
                                               vCastFp32Res0, vCastFp32Res1, vCastRes0, vCastRes1,
                                               kvTypeMaskAll, qTypeMaskAll);

      // 前半组
      Fp8NzQuantAndStore<Q_T, KV_T, hasOffset>(vKvData, vRes, ubSrcTemp,
                                               vOffsetFirst, vScaleFirst, ubDstAddrTmp,
                                               vCastFp32Res0, vCastFp32Res1, vCastRes0, vCastRes1,
                                               kvTypeMaskAll, qTypeMaskAll);
    }
  }
}

template <typename Q_T, typename KV_T, typename ANTIQ_PARAMS_T, uint32_t baseSize, bool hasOffset = false>
__aicore__ inline void AntiquantVFFp8PerTokenNz(LocalTensor<KV_T>& antiqInUb, LocalTensor<Q_T>& antiqResUb,
                                               LocalTensor<ANTIQ_PARAMS_T>& antiqOffsetUb,
                                               LocalTensor<ANTIQ_PARAMS_T>& antiqScaleUb,
                                               uint32_t dealRowCount) {
  ASCENDC_ASSERT((baseSize % 16 == 0),
                 {KERNEL_LOG(KERNEL_ERROR, "baseSize is %d, which must be 16 aligned.", baseSize); });
  ASCENDC_ASSERT((IsSameType<KV_T, fp8_e4m3fn_t>::value),
                 {KERNEL_LOG(KERNEL_ERROR, "Antiquant fp8 PerToken, KV_T must be fp8_e4m3."); });  
  __ubuf__ uint8_t* ubSrcAddr = (__ubuf__ uint8_t*)(antiqInUb.GetPhyAddr());
  __ubuf__ Q_T* ubDstAddr = (__ubuf__ Q_T*)(antiqResUb.GetPhyAddr());
  __ubuf__ Q_T* ubOffsetAddr = (__ubuf__ Q_T*)antiqOffsetUb.GetPhyAddr();
  __ubuf__ Q_T* ubScaleAddr = (__ubuf__ Q_T*)antiqScaleUb.GetPhyAddr();
  AntiquantVFImplFp8PerTokenNz<Q_T, KV_T, ANTIQ_PARAMS_T, baseSize, hasOffset>(
    ubSrcAddr, ubDstAddr, ubOffsetAddr, ubScaleAddr, dealRowCount);
}

template <typename Q_T, typename ANTIQ_PARAMS_T, uint32_t baseSize, bool hasOffset = false, bool isPerToken = false, bool isKvCacheNz = false>
__aicore__ inline void AntiquantVFImpl(LocalTensor<fp8_e4m3fn_t>& antiqInUb, LocalTensor<Q_T>& antiqResUb,
                                       LocalTensor<ANTIQ_PARAMS_T>& antiqOffsetUb, LocalTensor<ANTIQ_PARAMS_T>& antiqScaleUb,
                                       uint32_t dealRowCount, uint32_t headDim, uint32_t copyTotalS) {
  if constexpr (isPerToken) {
    if constexpr (isKvCacheNz) {
        AntiquantVFFp8PerTokenNz<Q_T, fp8_e4m3fn_t, ANTIQ_PARAMS_T, baseSize, hasOffset>
          (antiqInUb, antiqResUb, antiqOffsetUb, antiqScaleUb, dealRowCount);
    } else {
      if constexpr (baseSize == 64) {
        AntiquantVFFp8PerTokenD64<Q_T, fp8_e4m3fn_t, ANTIQ_PARAMS_T, baseSize, hasOffset>
          (antiqInUb, antiqResUb, antiqOffsetUb, antiqScaleUb, dealRowCount);
      } else if constexpr (baseSize == 128) {
        AntiquantVFFp8PerTokenD128<Q_T, fp8_e4m3fn_t, ANTIQ_PARAMS_T, baseSize, hasOffset>
          (antiqInUb, antiqResUb, antiqOffsetUb, antiqScaleUb, dealRowCount);
      } else if constexpr (baseSize == 256) {
        AntiquantVFFp8PerTokenD256<Q_T, fp8_e4m3fn_t, ANTIQ_PARAMS_T, baseSize, hasOffset>
          (antiqInUb, antiqResUb, antiqOffsetUb, antiqScaleUb, dealRowCount);
      } else if constexpr (baseSize == 512) {
        AntiquantVFFp8PerTokenD512<Q_T, fp8_e4m3fn_t, ANTIQ_PARAMS_T, baseSize, hasOffset>
          (antiqInUb, antiqResUb, antiqOffsetUb, antiqScaleUb, dealRowCount);
      }
    }
  } else {
    if constexpr (isKvCacheNz) {
      AntiquantVFFp8Nz<Q_T, fp8_e4m3fn_t, baseSize>(antiqInUb, antiqResUb, antiqScaleUb, dealRowCount);
    } else {
      if constexpr (baseSize == 64) {
        AntiquantVFFp8D64<Q_T, fp8_e4m3fn_t, baseSize>(antiqInUb, antiqResUb, antiqScaleUb, dealRowCount);
      } else {
        AntiquantVFFp8Norm<Q_T, fp8_e4m3fn_t, baseSize>(antiqInUb, antiqResUb, antiqScaleUb, dealRowCount);
      }
    }
  }
}

template <typename Q_T, typename KV_T, typename ANTIQ_PARAMS_T,
  uint32_t baseSize, bool hasOffset = false, bool isPerToken = false, bool isKvCacheNz = false>
__aicore__ inline void AntiquantVF(LocalTensor<KV_T>& antiqInUb, LocalTensor<Q_T>& antiqResUb,
                                   LocalTensor<ANTIQ_PARAMS_T>& antiqOffsetUb, LocalTensor<ANTIQ_PARAMS_T>& antiqScaleUb,
                                   uint32_t dealRowCount, uint32_t headDim, uint32_t copyTotalS = 0) {
  AntiquantVFImpl<Q_T, ANTIQ_PARAMS_T, baseSize, hasOffset, isPerToken, isKvCacheNz>
    (antiqInUb, antiqResUb, antiqOffsetUb, antiqScaleUb, dealRowCount, headDim, copyTotalS);
}

};  // namespace FaVectorApi

#endif
