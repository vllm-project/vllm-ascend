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
 * \file vf_antiquant_w4.h
 * \brief
 */

#ifndef VF_ANTIQUANT_W4_H
#define VF_ANTIQUANT_W4_H

namespace FaVectorApi {
// w4转Q_T
static constexpr MicroAPI::CastTrait castTraitW4 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                  MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
// fp32转Q_T
static constexpr MicroAPI::CastTrait castTraitW4_0 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                   MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
// bf16转fp16
static constexpr MicroAPI::CastTrait castTraitW4_1 = {MicroAPI::RegLayout::UNKNOWN, MicroAPI::SatMode::SAT,
                                                   MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};
// fp16转bf16
static constexpr MicroAPI::CastTrait castTraitW4_2 = {MicroAPI::RegLayout::UNKNOWN, MicroAPI::SatMode::UNKNOWN,
                                                      MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
// fp32转fp16
static constexpr MicroAPI::CastTrait castTraitFp32 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                      MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};


template <typename Q_T, typename KV_T>
__simd_callee__ inline void CastMulStoreW4GroupNz(MicroAPI::RegTensor<KV_T>& vKvData,
    MicroAPI::RegTensor<half>& vCastFp16Res, MicroAPI::RegTensor<bfloat16_t>& vRes,
    MicroAPI::RegTensor<bfloat16_t>& vScale, __ubuf__ Q_T* ubDstAddr,
    __ubuf__ uint8_t* ubSrcAddr, MicroAPI::MaskReg& kvTypeMaskAll, MicroAPI::MaskReg& qTypeMaskAll)
{
    MicroAPI::LoadAlign<uint8_t, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
        (MicroAPI::RegTensor<uint8_t>&)vKvData, ubSrcAddr);
    MicroAPI::Cast<bfloat16_t, KV_T, castTraitW4>(vRes, vKvData, kvTypeMaskAll);
    MicroAPI::Mul<bfloat16_t, MicroAPI::MaskMergeMode::ZEROING>(vRes, vRes, vScale, qTypeMaskAll);
    if constexpr (std::is_same<Q_T, bfloat16_t>::value) {
        MicroAPI::StoreAlign<Q_T, MicroAPI::StoreDist::DIST_NORM_B16>(ubDstAddr, vRes, qTypeMaskAll);
    } else {
        MicroAPI::Cast<half, bfloat16_t, castTraitW4_1>(vCastFp16Res, vRes, qTypeMaskAll);
        MicroAPI::StoreAlign<Q_T, MicroAPI::StoreDist::DIST_NORM_B16>(ubDstAddr, vCastFp16Res, qTypeMaskAll);
    }
}

template <typename Q_T, typename KV_T, uint32_t baseSize, bool hasOffset = false>
__aicore__ inline void AntiquantVFImplW4PerTokenGroupNz(LocalTensor<KV_T>& antiqInUb, LocalTensor<Q_T>& antiqResUb,
                                           LocalTensor<Q_T>& antiqOffsetUb,
                                           LocalTensor<Q_T>& antiqScaleUb,
                                           uint32_t dealRowCount, uint32_t copyTotalS) {
                                
  __VEC_SCOPE__ {
    __ubuf__ uint8_t* ubSrcAddr = (__ubuf__ uint8_t*)(antiqInUb.GetPhyAddr());
    __ubuf__ Q_T* ubDstAddr = (__ubuf__ Q_T*)(antiqResUb.GetPhyAddr());
    __ubuf__ bfloat16_t* ubOffsetAddr = (__ubuf__ bfloat16_t*)antiqOffsetUb.GetPhyAddr();
    __ubuf__ bfloat16_t* ubScaleAddr = (__ubuf__ bfloat16_t*)antiqScaleUb.GetPhyAddr();

    MicroAPI::RegTensor<KV_T> vKvData;
    MicroAPI::RegTensor<bfloat16_t> vScaleFirst, vScaleBack, vRes;
    MicroAPI::RegTensor<half> vCastFp16Res;

    MicroAPI::MaskReg kvTypeMaskAll = MicroAPI::CreateMask<KV_T, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg qTypeMaskAll = MicroAPI::CreateMask<Q_T, MicroAPI::MaskPattern::ALL>(); // Q_T 所有元素（共128个）

    // UB总共dealRowCount行 * baseSize列，每次处理8行 * 16列 = 128个元素
    const uint32_t rowBaseSize = 8; // 8行
    const uint32_t colBaseSize = 16; // 16列
    const uint32_t dealBaseNum = 128; // 128个元素
    const uint32_t doubleRowBaseSize = 16;

    const uint32_t rowScaleStride = doubleRowBaseSize;
    const uint32_t rowDstStride = doubleRowBaseSize * colBaseSize;
    const uint32_t rowSrcStride = doubleRowBaseSize * colBaseSize / 2;
    const uint32_t colDstStride = dealRowCount * colBaseSize;
    const uint32_t colSrcStride = (dealRowCount * colBaseSize / 2 + 31) / 32 * 32; // 32B对齐
    const uint16_t innerLoopCnt = 2;//D方向，每个group内部的VF基本块循环次数（=2）
    const uint16_t colLoopCnt = static_cast<uint16_t>(baseSize /(colBaseSize*innerLoopCnt));
    const uint16_t rowLoopCnt = static_cast<uint16_t>((dealRowCount + doubleRowBaseSize - 1) / doubleRowBaseSize); // 16行对齐
    const uint32_t colScaleStride = (copyTotalS + 31) / 32 * 32; //伪量化参数32byte对齐

    for (uint16_t rowLoop = 0; rowLoop < rowLoopCnt; rowLoop++) {
      uint16_t rowLoopIdx = rowLoopCnt - 1 - rowLoop;
      for (uint16_t colLoopIdx = 0; colLoopIdx < colLoopCnt; colLoopIdx++) {
        __ubuf__ bfloat16_t* ubScaleAddrTmp = ubScaleAddr + rowScaleStride * rowLoopIdx + colLoopIdx * colScaleStride;
        MicroAPI::LoadAlign<bfloat16_t, MicroAPI::LoadDist::DIST_E2B_B16>(vScaleFirst, ubScaleAddrTmp);
        MicroAPI::LoadAlign<bfloat16_t, MicroAPI::LoadDist::DIST_E2B_B16>(vScaleBack, ubScaleAddrTmp + rowBaseSize);
        for(uint16_t innerLoopIdx = 0; innerLoopIdx < innerLoopCnt; innerLoopIdx++) {
          __ubuf__ uint8_t* ubSrcTemp = ubSrcAddr + rowSrcStride * rowLoopIdx + colLoopIdx * innerLoopCnt * colSrcStride + innerLoopIdx * colSrcStride;
          __ubuf__ Q_T* ubDstAddrTmp = ubDstAddr + rowDstStride * rowLoopIdx + colLoopIdx * innerLoopCnt * colDstStride + innerLoopIdx * colDstStride;
          // 后半组
          CastMulStoreW4GroupNz<Q_T, KV_T>(vKvData, vCastFp16Res, vRes, vScaleBack,
              ubDstAddrTmp + dealBaseNum, ubSrcTemp + 64, kvTypeMaskAll, qTypeMaskAll);
          // 前半组
          CastMulStoreW4GroupNz<Q_T, KV_T>(vKvData, vCastFp16Res, vRes, vScaleFirst,
              ubDstAddrTmp, ubSrcTemp, kvTypeMaskAll, qTypeMaskAll);
        }
      }
    }
  }
}


template <typename Q_T, typename KV_T, typename ANTIQ_PARAMS_T, uint32_t baseSize, bool hasOffset = false>
__simd_vf__ void AntiquantVFImplW4Nz(__ubuf__ uint8_t* ubSrcAddr, __ubuf__ Q_T* ubDstAddr, __ubuf__ Q_T* ubOffsetAddr, 
                                    __ubuf__ Q_T* ubScaleAddr, uint32_t dealRowCount)
{
  MicroAPI::RegTensor<int4x2_t> vKvData;
  MicroAPI::RegTensor<Q_T> vOffset;
  MicroAPI::RegTensor<Q_T> vScale;
  MicroAPI::RegTensor<Q_T> vRes;
  MicroAPI::RegTensor<half> vCastFp16Res;

  MicroAPI::MaskReg kvMaskAll = MicroAPI::CreateMask<KV_T, MicroAPI::MaskPattern::ALL>();
  MicroAPI::MaskReg qTypeMaskAll = MicroAPI::CreateMask<Q_T, MicroAPI::MaskPattern::ALL>();

  // UB总共dealRowCount行 * baseSize列，每次处理8行 * 16列 = 128个元素
  uint32_t rowBaseSize = 8; // 8行
  uint32_t colBaseSize = 16; // 16列
  uint32_t dealBaseNum = 128; // 128个元素
  uint32_t dealBaseSize = 64; // 128 * sizeof(int4) = 64B
  uint32_t colDstStride = dealRowCount * colBaseSize;
  uint32_t colSrcStride = (dealRowCount * 8 + 31) / 32 * 32; // 32B对齐
  const uint16_t colLoopCnt = static_cast<uint16_t>(baseSize / colBaseSize);
  const uint16_t rowLoopCnt = static_cast<uint16_t>((dealRowCount + rowBaseSize - 1) / rowBaseSize);

  for (uint16_t colLoopIdx = 0; colLoopIdx < colLoopCnt; colLoopIdx++) {
    if constexpr (hasOffset) {
      MicroAPI::LoadAlign<Q_T, MicroAPI::LoadDist::DIST_BLK>(vOffset, ubOffsetAddr + colLoopIdx * colBaseSize);
    }
    MicroAPI::LoadAlign<Q_T, MicroAPI::LoadDist::DIST_BLK>(vScale, ubScaleAddr + colLoopIdx * colBaseSize);

    // #pragma unroll(4)
    for (uint16_t rowLoop = 0; rowLoop < rowLoopCnt; rowLoop++) {
      uint16_t rowLoopIdx = rowLoopCnt - 1 - rowLoop;
      __ubuf__ Q_T* ubDstAddrTmp = ubDstAddr + colDstStride * colLoopIdx + dealBaseNum * rowLoopIdx;
      __ubuf__ uint8_t* ubSrcTemp = ubSrcAddr + colSrcStride * colLoopIdx + dealBaseSize * rowLoopIdx;;
      MicroAPI::LoadAlign<uint8_t, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
          (MicroAPI::RegTensor<uint8_t>&)vKvData, ubSrcTemp);

      if constexpr (std::is_same<Q_T, bfloat16_t>::value) {
        MicroAPI::Cast<half, int4x2_t, castTraitW4>(vCastFp16Res, vKvData, kvMaskAll);
        MicroAPI::Cast<Q_T, half, castTraitW4_2>(vRes, vCastFp16Res, kvMaskAll);
      } else {
        MicroAPI::Cast<Q_T, int4x2_t, castTraitW4>(vRes, vKvData, kvMaskAll);
      }

      if constexpr (hasOffset) {
        MicroAPI::Add<Q_T, MicroAPI::MaskMergeMode::ZEROING>(vRes, vRes, vOffset, qTypeMaskAll);
      }
      MicroAPI::Mul<Q_T, MicroAPI::MaskMergeMode::ZEROING>(vRes, vRes, vScale, qTypeMaskAll);
      MicroAPI::StoreAlign<Q_T, MicroAPI::StoreDist::DIST_NORM_B16>(ubDstAddrTmp, vRes, qTypeMaskAll);
    }
  }
}

template <typename Q_T, typename KV_T, bool hasOffset>
__simd_callee__ inline void CastAddMulStoreW4Nz(MicroAPI::RegTensor<int4x2_t>& vKvData,
    MicroAPI::RegTensor<half>& vCastFp16Res, MicroAPI::RegTensor<Q_T>& vRes,
    MicroAPI::RegTensor<Q_T>& vOffset, MicroAPI::RegTensor<Q_T>& vScale,
    __ubuf__ Q_T* ubDstAddr, __ubuf__ uint8_t* ubSrcAddr,
    MicroAPI::MaskReg& kvMaskAll, MicroAPI::MaskReg& qTypeMaskAll)
{
    MicroAPI::LoadAlign<uint8_t, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
        (MicroAPI::RegTensor<uint8_t>&)vKvData, ubSrcAddr);
    if constexpr (std::is_same<Q_T, bfloat16_t>::value) {
        MicroAPI::Cast<half, int4x2_t, castTraitW4>(vCastFp16Res, vKvData, kvMaskAll);
        MicroAPI::Cast<Q_T, half, castTraitW4_2>(vRes, vCastFp16Res, kvMaskAll);
    } else {
        MicroAPI::Cast<Q_T, int4x2_t, castTraitW4>(vRes, vKvData, kvMaskAll);
    }
    if constexpr (hasOffset) {
        MicroAPI::Add<Q_T, MicroAPI::MaskMergeMode::ZEROING>(vRes, vRes, vOffset, qTypeMaskAll);
    }
    MicroAPI::Mul<Q_T, MicroAPI::MaskMergeMode::ZEROING>(vRes, vScale, vRes, qTypeMaskAll);
    MicroAPI::StoreAlign<Q_T, MicroAPI::StoreDist::DIST_NORM_B16>(ubDstAddr, vRes, qTypeMaskAll);
}

template <typename Q_T, typename KV_T, typename ANTIQ_PARAMS_T, uint32_t baseSize, bool hasOffset = false>
__simd_vf__ void AntiquantVFImplW4PerTokenNz(__ubuf__ uint8_t* ubSrcAddr, __ubuf__ Q_T* ubDstAddr,
                                             __ubuf__ Q_T* ubOffsetAddr, __ubuf__ Q_T* ubScaleAddr,
                                             uint32_t dealRowCount) 
{
  MicroAPI::RegTensor<int4x2_t> vKvData;
  MicroAPI::RegTensor<Q_T> vOffsetFirst;
  MicroAPI::RegTensor<Q_T> vOffsetBack;
  MicroAPI::RegTensor<Q_T> vScaleFirst;
  MicroAPI::RegTensor<Q_T> vScaleBack;
  MicroAPI::RegTensor<Q_T> vRes;
  MicroAPI::RegTensor<half> vCastFp16Res;

  MicroAPI::MaskReg kvTypeMaskAll = MicroAPI::CreateMask<int4x2_t, MicroAPI::MaskPattern::ALL>();
  MicroAPI::MaskReg qTypeMaskAll = MicroAPI::CreateMask<Q_T, MicroAPI::MaskPattern::ALL>(); // Q_T 所有元素（共128个）

  // UB总共dealRowCount行 * baseSize列，每次处理8行 * 16列 = 128个元素
  const uint32_t rowBaseSize = 8; // 8行
  const uint32_t colBaseSize = 16; // 16列
  const uint32_t dealBaseNum = 128; // 128个元素
  const uint32_t doubleRowBaseSize = 16; // 每16行交替，防止bank冲突

  const uint32_t rowDstStride = doubleRowBaseSize * colBaseSize;
  const uint32_t rowSrcStride = doubleRowBaseSize * colBaseSize >> 1U;
  const uint32_t colDstStride = dealRowCount * colBaseSize;
  const uint32_t colSrcStride = (((dealRowCount * colBaseSize) >> 1U) + 31) >> 5U << 5U; // 32B对齐
  const uint16_t colLoopCnt = static_cast<uint16_t>(baseSize / colBaseSize);
  const uint16_t rowLoopCnt = static_cast<uint16_t>((dealRowCount + doubleRowBaseSize - 1) / doubleRowBaseSize); // 16行对齐

  for (uint16_t rowLoop = 0; rowLoop < rowLoopCnt; rowLoop++) {
    uint16_t rowLoopIdx = rowLoopCnt - 1 - rowLoop;
    __ubuf__ Q_T* ubOffsetAddrTmp = ubOffsetAddr + doubleRowBaseSize * rowLoopIdx;
    __ubuf__ Q_T* ubScaleAddrTmp = ubScaleAddr + doubleRowBaseSize * rowLoopIdx;

    if constexpr (hasOffset) {
      MicroAPI::LoadAlign<Q_T, MicroAPI::LoadDist::DIST_E2B_B16>(vOffsetFirst, ubOffsetAddrTmp);
      MicroAPI::LoadAlign<Q_T, MicroAPI::LoadDist::DIST_E2B_B16>(vOffsetBack, ubOffsetAddrTmp + rowBaseSize);
    }
    MicroAPI::LoadAlign<Q_T, MicroAPI::LoadDist::DIST_E2B_B16>(vScaleFirst, ubScaleAddrTmp);
    MicroAPI::LoadAlign<Q_T, MicroAPI::LoadDist::DIST_E2B_B16>(vScaleBack, ubScaleAddrTmp + rowBaseSize);

    for (uint16_t colLoopIdx = 0; colLoopIdx < colLoopCnt; colLoopIdx++) {
      __ubuf__ uint8_t* ubSrcTemp = ubSrcAddr + rowSrcStride * rowLoopIdx + colSrcStride * colLoopIdx;
      __ubuf__ Q_T* ubDstAddrTmp = ubDstAddr + rowDstStride * rowLoopIdx + colDstStride * colLoopIdx;

      // 后半组
      CastAddMulStoreW4Nz<Q_T, KV_T, hasOffset>(vKvData, vCastFp16Res, vRes, vOffsetBack, vScaleBack,
          ubDstAddrTmp + dealBaseNum, ubSrcTemp + 64, kvTypeMaskAll, qTypeMaskAll); // dealBaseNum * sizeof(4bit) = 64
      // 前半组
      CastAddMulStoreW4Nz<Q_T, KV_T, hasOffset>(vKvData, vCastFp16Res, vRes, vOffsetFirst, vScaleFirst,
          ubDstAddrTmp, ubSrcTemp, kvTypeMaskAll, qTypeMaskAll);
    }
  }
}

template <typename Q_T, typename KV_T, typename ANTIQ_PARAMS_T, uint32_t baseSize, bool hasOffset = false>
__aicore__ inline void AntiquantVFW4PerTokenNz(LocalTensor<KV_T>& antiqInUb, LocalTensor<Q_T>& antiqResUb,
                                               LocalTensor<ANTIQ_PARAMS_T>& antiqOffsetUb,
                                               LocalTensor<ANTIQ_PARAMS_T>& antiqScaleUb,
                                               uint32_t dealRowCount) {
  static_assert(baseSize % 16 == 0);
  static_assert(IsSameType<KV_T, int4b_t>::value, "antiquant w4 PerToken, KV_T must be int4_t");
  __ubuf__ uint8_t* ubSrcAddr = (__ubuf__ uint8_t*)(antiqInUb.GetPhyAddr());
  __ubuf__ Q_T* ubDstAddr = (__ubuf__ Q_T*)(antiqResUb.GetPhyAddr());
  __ubuf__ Q_T* ubOffsetAddr = (__ubuf__ Q_T*)antiqOffsetUb.GetPhyAddr();
  __ubuf__ Q_T* ubScaleAddr = (__ubuf__ Q_T*)antiqScaleUb.GetPhyAddr();
  AntiquantVFImplW4PerTokenNz<Q_T, KV_T, ANTIQ_PARAMS_T, baseSize, hasOffset>(ubSrcAddr, ubDstAddr, ubOffsetAddr, ubScaleAddr, dealRowCount);
}

template <typename Q_T, typename KV_T, typename ANTIQ_PARAMS_T, uint32_t baseSize, bool hasOffset = false>
__aicore__ inline void AntiquantVFW4Nz(LocalTensor<KV_T>& antiqInUb, LocalTensor<Q_T>& antiqResUb,
                                             LocalTensor<ANTIQ_PARAMS_T>& antiqOffsetUb, LocalTensor<ANTIQ_PARAMS_T>& antiqScaleUb,
                                             uint32_t dealRowCount) {
  static_assert(baseSize % 16 == 0);
  __ubuf__ uint8_t* ubSrcAddr = (__ubuf__ uint8_t*)(antiqInUb.GetPhyAddr());
  __ubuf__ Q_T* ubDstAddr = (__ubuf__ Q_T*)(antiqResUb.GetPhyAddr());
  __ubuf__ Q_T* ubOffsetAddr = (__ubuf__ Q_T*)antiqOffsetUb.GetPhyAddr();
  __ubuf__ Q_T* ubScaleAddr = (__ubuf__ Q_T*)antiqScaleUb.GetPhyAddr();

  AntiquantVFImplW4Nz<Q_T, KV_T, ANTIQ_PARAMS_T, baseSize, hasOffset>(ubSrcAddr, ubDstAddr, ubOffsetAddr, ubScaleAddr, dealRowCount);
}

template <typename Q_T, typename KV_T, typename ANTIQ_PARAMS_T, uint32_t baseSize, bool hasOffset = false>
__simd_vf__ void AntiquantVFImplW4D64(__ubuf__ uint8_t* ubSrcAddr, __ubuf__ Q_T* ubDstAddr, __ubuf__ Q_T* ubDstAddr_, 
                                      __ubuf__ Q_T* ubOffsetAddr, __ubuf__ Q_T* ubScaleAddr, uint32_t dealRowCount)
{
  MicroAPI::RegTensor<int4x2_t> vKvData;
  MicroAPI::RegTensor<Q_T> vOffset;
  MicroAPI::RegTensor<Q_T> vScale;
  MicroAPI::RegTensor<Q_T> vRes;
  MicroAPI::RegTensor<half> vCastFp16Res;

  MicroAPI::MaskReg qTypeMaskAll = MicroAPI::CreateMask<Q_T, MicroAPI::MaskPattern::ALL>();
  MicroAPI::MaskReg qTypeMaskLower64 = MicroAPI::CreateMask<Q_T, MicroAPI::MaskPattern::VL64>();
  MicroAPI::MaskReg qTypeMaskHigher64;
  MicroAPI::Xor(qTypeMaskHigher64, qTypeMaskLower64, qTypeMaskAll, qTypeMaskAll); // qTypeMaskAll与qTypeMaskLower64异或得到qTypeMaskHigher64

  uint32_t blockStride = dealRowCount + 1;
  uint32_t repeatStride = 2;
  uint16_t loopCnt = static_cast<uint16_t>((dealRowCount + 1) / 2); // +1是为了兼容处理奇数行

  __ubuf__ uint8_t* ubSrcTemp = ubSrcAddr;
  if constexpr (hasOffset) {
    MicroAPI::LoadAlign<Q_T, MicroAPI::LoadDist::DIST_NORM>(vOffset, ubOffsetAddr);
  }
  MicroAPI::LoadAlign<Q_T, MicroAPI::LoadDist::DIST_NORM>(vScale, ubScaleAddr);

  // #pragma unroll(4)
  for (uint16_t dealRowIdx = 0; dealRowIdx < loopCnt; dealRowIdx++) {
    MicroAPI::LoadAlign<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
        (MicroAPI::RegTensor<uint8_t>&)vKvData, ubSrcTemp, baseSize);
    if constexpr (std::is_same<Q_T, bfloat16_t>::value) {
      MicroAPI::Cast<half, int4x2_t, castTraitW4>(vCastFp16Res, vKvData, qTypeMaskAll);
      MicroAPI::Cast<Q_T, half, castTraitW4_2>(vRes, vCastFp16Res, qTypeMaskAll);
    } else {
      MicroAPI::Cast<Q_T, int4x2_t, castTraitW4>(vRes, vKvData, qTypeMaskAll);
    }

    if constexpr (hasOffset) {
      MicroAPI::Add<Q_T, MicroAPI::MaskMergeMode::ZEROING>(vRes, vRes, vOffset, qTypeMaskAll);
    }

    MicroAPI::Mul<Q_T, MicroAPI::MaskMergeMode::ZEROING>(vRes, vRes, vScale, qTypeMaskAll);

    // vRes fp16 128个元素
    MicroAPI::StoreAlign<Q_T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
        ubDstAddr, vRes, blockStride, repeatStride, qTypeMaskLower64);

    MicroAPI::StoreAlign<Q_T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
        ubDstAddr_, vRes, blockStride, repeatStride, qTypeMaskHigher64);
  }
}


template <typename Q_T, typename KV_T, typename ANTIQ_PARAMS_T, uint32_t baseSize, bool hasOffset = false>
__aicore__ inline void AntiquantVFW4D64(LocalTensor<KV_T>& antiqInUb, LocalTensor<Q_T>& antiqResUb,
                                            LocalTensor<ANTIQ_PARAMS_T>& antiqOffsetUb, LocalTensor<ANTIQ_PARAMS_T>& antiqScaleUb,
                                            uint32_t dealRowCount) {
  static_assert(baseSize == 64);
  __ubuf__ uint8_t* ubSrcAddr = (__ubuf__ uint8_t*)(antiqInUb.GetPhyAddr());
  __ubuf__ Q_T* ubDstAddr = (__ubuf__ Q_T*)(antiqResUb.GetPhyAddr());
  __ubuf__ Q_T* ubDstAddr_ = ubDstAddr + 16 - (dealRowCount + 1) * 32 * 4 / 2;
  __ubuf__ Q_T* ubOffsetAddr = (__ubuf__ Q_T*)antiqOffsetUb.GetPhyAddr();
  __ubuf__ Q_T* ubScaleAddr = (__ubuf__ Q_T*)antiqScaleUb.GetPhyAddr();
  
  AntiquantVFImplW4D64<Q_T, KV_T, ANTIQ_PARAMS_T, baseSize, hasOffset>(
    ubSrcAddr, ubDstAddr, ubDstAddr_, ubOffsetAddr, ubScaleAddr, dealRowCount);
}

template <typename Q_T, typename KV_T, typename ANTIQ_PARAMS_T, uint32_t baseSize, bool hasOffset = false>
__simd_vf__ void AntiquantVFImplW4PerTokenD64(__ubuf__ uint8_t* ubSrcAddr, __ubuf__ Q_T* ubDstAddr, __ubuf__ Q_T *ubDstAddr_, 
                                              __ubuf__ ANTIQ_PARAMS_T* ubOffsetAddr, __ubuf__ ANTIQ_PARAMS_T* ubScaleAddr, uint32_t dealRowCount)
{
  MicroAPI::RegTensor<int4x2_t> vKvData;
  MicroAPI::RegTensor<ANTIQ_PARAMS_T> vOffset;
  MicroAPI::RegTensor<ANTIQ_PARAMS_T> vScale;
  MicroAPI::RegTensor<Q_T> vOffsetFp16;
  MicroAPI::RegTensor<Q_T> vOffsetFp16High;
  MicroAPI::RegTensor<Q_T> vOffsetFp16Low;
  MicroAPI::RegTensor<Q_T> vScaleFp16;
  MicroAPI::RegTensor<Q_T> vScaleFp16High;
  MicroAPI::RegTensor<Q_T> vScaleFp16Low;
  MicroAPI::RegTensor<Q_T> vRes;
  MicroAPI::RegTensor<half> vCastFp16Res;

  MicroAPI::MaskReg maskOne = MicroAPI::CreateMask<ANTIQ_PARAMS_T, MicroAPI::MaskPattern::VL1>();
  MicroAPI::MaskReg kvMaskAll = MicroAPI::CreateMask<KV_T, MicroAPI::MaskPattern::ALL>();
  MicroAPI::MaskReg qTypeMaskAll = MicroAPI::CreateMask<Q_T, MicroAPI::MaskPattern::ALL>();

  MicroAPI::UnalignRegForLoad u0;
  MicroAPI::UnalignRegForLoad u1;
  uint32_t blockStride = dealRowCount + 1;
  uint32_t repeatStride = 2;
  uint16_t loopCnt = static_cast<uint16_t>((dealRowCount + 1) / 2); // +1是为了兼容处理奇数行

  MicroAPI::MaskReg qTypeMaskLower64 = MicroAPI::CreateMask<Q_T, MicroAPI::MaskPattern::VL64>();
  MicroAPI::MaskReg qTypeMaskHigher64;
  MicroAPI::Xor(qTypeMaskHigher64, qTypeMaskLower64, qTypeMaskAll, qTypeMaskAll); // qTypeMaskAll与qTypeMaskLower64异或得到qTypeMaskHigher64

  MicroAPI::LoadUnAlignPre(u0, ubScaleAddr);
  if constexpr (hasOffset) {
    MicroAPI::LoadUnAlignPre(u1, ubOffsetAddr);
  }
  for (uint16_t dealRowIdx = 0; dealRowIdx < loopCnt; dealRowIdx++) {
    MicroAPI::LoadAlign<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
        (MicroAPI::RegTensor<uint8_t>&)vKvData, ubSrcAddr, baseSize);
    if constexpr (std::is_same<Q_T, bfloat16_t>::value) {
      MicroAPI::Cast<half, int4x2_t, castTraitW4>(vCastFp16Res, vKvData, kvMaskAll);
      MicroAPI::Cast<Q_T, half, castTraitW4_2>(vRes, vCastFp16Res, kvMaskAll);
    } else {
      MicroAPI::Cast<Q_T, int4x2_t, castTraitW4>(vRes, vKvData, kvMaskAll);
    }

    MicroAPI::LoadUnAlign<ANTIQ_PARAMS_T>(vScale, u0, ubScaleAddr, 1); // 1表示ub自动往后偏移1个float
    MicroAPI::Cast<Q_T, ANTIQ_PARAMS_T, castTraitW4_0>(vScaleFp16, vScale, maskOne);
    MicroAPI::Duplicate<uint16_t, MicroAPI::HighLowPart::LOWEST, MicroAPI::MaskMergeMode::ZEROING>
      ((MicroAPI::RegTensor<uint16_t>&)vScaleFp16Low, (MicroAPI::RegTensor<uint16_t>&)vScaleFp16, qTypeMaskLower64);

    MicroAPI::LoadUnAlign<ANTIQ_PARAMS_T>(vScale, u0, ubScaleAddr, 1); // 1表示ub自动往后偏移1个float
    MicroAPI::Cast<Q_T, ANTIQ_PARAMS_T, castTraitW4_0>(vScaleFp16, vScale, maskOne);
    MicroAPI::Duplicate<uint16_t, MicroAPI::HighLowPart::LOWEST, MicroAPI::MaskMergeMode::ZEROING>
      ((MicroAPI::RegTensor<uint16_t>&)vScaleFp16High, (MicroAPI::RegTensor<uint16_t>&)vScaleFp16, qTypeMaskHigher64);
    MicroAPI::Or<uint16_t, MicroAPI::MaskMergeMode::ZEROING>((MicroAPI::RegTensor<uint16_t>&)vScaleFp16,
                                                              (MicroAPI::RegTensor<uint16_t>&)vScaleFp16Low,
                                                              (MicroAPI::RegTensor<uint16_t>&)vScaleFp16High, kvMaskAll);
    if constexpr (hasOffset) {
      MicroAPI::LoadUnAlign<ANTIQ_PARAMS_T>(vOffset, u1, ubOffsetAddr, 1); // 1表示ub自动往后偏移1个float
      MicroAPI::Cast<Q_T, ANTIQ_PARAMS_T, castTraitW4_0>(vOffsetFp16, vOffset, maskOne);
      MicroAPI::Duplicate<uint16_t, MicroAPI::HighLowPart::LOWEST, MicroAPI::MaskMergeMode::ZEROING>
        ((MicroAPI::RegTensor<uint16_t>&)vOffsetFp16Low, (MicroAPI::RegTensor<uint16_t>&)vOffsetFp16, qTypeMaskLower64);

      MicroAPI::LoadUnAlign<ANTIQ_PARAMS_T>(vOffset, u1, ubOffsetAddr, 1); // 1表示ub自动往后偏移1个float
      MicroAPI::Cast<Q_T, ANTIQ_PARAMS_T, castTraitW4_0>(vOffsetFp16, vOffset, maskOne);
      MicroAPI::Duplicate<uint16_t, MicroAPI::HighLowPart::LOWEST, MicroAPI::MaskMergeMode::ZEROING>
        ((MicroAPI::RegTensor<uint16_t>&)vOffsetFp16High, (MicroAPI::RegTensor<uint16_t>&)vOffsetFp16, qTypeMaskHigher64);
      MicroAPI::Or<uint16_t, MicroAPI::MaskMergeMode::ZEROING>((MicroAPI::RegTensor<uint16_t>&)vOffsetFp16,
                  (MicroAPI::RegTensor<uint16_t>&)vOffsetFp16Low,
                  (MicroAPI::RegTensor<uint16_t>&)vOffsetFp16High, kvMaskAll);
      MicroAPI::Add<Q_T, MicroAPI::MaskMergeMode::ZEROING>(vRes, vRes, vOffsetFp16, kvMaskAll);
    }

    MicroAPI::Mul<Q_T, MicroAPI::MaskMergeMode::ZEROING>(vRes, vRes, vScaleFp16, kvMaskAll);

    // vRes fp16 128个元素
    MicroAPI::StoreAlign<Q_T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
        ubDstAddr, vRes, blockStride, repeatStride, qTypeMaskLower64);

    MicroAPI::StoreAlign<Q_T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
        ubDstAddr_, vRes, blockStride, repeatStride, qTypeMaskHigher64);
  }
}

template <typename Q_T, typename KV_T, typename ANTIQ_PARAMS_T, uint32_t baseSize, bool hasOffset = false>
__aicore__ inline void AntiquantVFW4PerTokenD64(LocalTensor<KV_T>& antiqInUb, LocalTensor<Q_T>& antiqResUb,
                                            LocalTensor<ANTIQ_PARAMS_T>& antiqOffsetUb, LocalTensor<ANTIQ_PARAMS_T>& antiqScaleUb,
                                            uint32_t dealRowCount) {
  static_assert(baseSize == 64);
  __ubuf__ uint8_t* ubSrcAddr = (__ubuf__ uint8_t*)(antiqInUb.GetPhyAddr());
  __ubuf__ Q_T* ubDstAddr = (__ubuf__ Q_T*)(antiqResUb.GetPhyAddr());
  __ubuf__ Q_T *ubDstAddr_ = ubDstAddr + 16 - (dealRowCount + 1) * 32 * 4 / 2;
  __ubuf__ ANTIQ_PARAMS_T* ubOffsetAddr = (__ubuf__ ANTIQ_PARAMS_T*)antiqOffsetUb.GetPhyAddr();
  __ubuf__ ANTIQ_PARAMS_T* ubScaleAddr = (__ubuf__ ANTIQ_PARAMS_T*)antiqScaleUb.GetPhyAddr();
  
  AntiquantVFImplW4PerTokenD64<Q_T, KV_T, ANTIQ_PARAMS_T, baseSize, hasOffset>(
    ubSrcAddr, ubDstAddr, ubDstAddr_, ubOffsetAddr, ubScaleAddr, dealRowCount);
}

template <typename Q_T, typename KV_T, typename ANTIQ_PARAMS_T, uint32_t baseSize, bool hasOffset = false>
__simd_vf__ void AntiquantVFImplW4Norm(__ubuf__ uint8_t* ubSrcAddr, __ubuf__ Q_T* ubDstAddr, __ubuf__ Q_T* ubOffsetAddr, 
                                      __ubuf__ Q_T* ubScaleAddr, uint32_t dealRowCount)
{
  MicroAPI::RegTensor<int4x2_t> vKvData;
  MicroAPI::RegTensor<Q_T> vOffset;
  MicroAPI::RegTensor<Q_T> vScale;
  MicroAPI::RegTensor<Q_T> vRes;
  MicroAPI::RegTensor<half> vCastFp16Res;

  MicroAPI::MaskReg kvMaskAll = MicroAPI::CreateMask<KV_T, MicroAPI::MaskPattern::ALL>();
  MicroAPI::MaskReg qTypeMaskAll = MicroAPI::CreateMask<Q_T, MicroAPI::MaskPattern::ALL>();

  uint32_t blockStride = dealRowCount + 1;
  uint32_t repeatStride = 1;
  const uint16_t colLoopCnt = static_cast<uint16_t>(baseSize / 128);

  for (uint16_t colLoopIdx = 0; colLoopIdx < colLoopCnt; colLoopIdx++) {
    __ubuf__ Q_T* ubDstAddrTmp = ubDstAddr + blockStride * 128 * colLoopIdx;
    __ubuf__ uint8_t* ubSrcTemp = ubSrcAddr + colLoopIdx * 64;
    if constexpr (hasOffset) {
      MicroAPI::LoadAlign<Q_T, MicroAPI::LoadDist::DIST_NORM>(vOffset, ubOffsetAddr + colLoopIdx * 128);
    }
    MicroAPI::LoadAlign<Q_T, MicroAPI::LoadDist::DIST_NORM>(vScale, ubScaleAddr + colLoopIdx * 128);

    // #pragma unroll(4)
    for (uint16_t dealRowIdx = 0; dealRowIdx < static_cast<uint16_t>(dealRowCount); dealRowIdx++) {
      MicroAPI::LoadAlign<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
          (MicroAPI::RegTensor<uint8_t>&)vKvData, ubSrcTemp, baseSize / 2);
      if constexpr (std::is_same<Q_T, bfloat16_t>::value) {
        MicroAPI::Cast<half, int4x2_t, castTraitW4>(vCastFp16Res, vKvData, kvMaskAll);
        MicroAPI::Cast<Q_T, half, castTraitW4_2>(vRes, vCastFp16Res, kvMaskAll);
      } else {
        MicroAPI::Cast<Q_T, int4x2_t, castTraitW4>(vRes, vKvData, kvMaskAll);
      }

      if constexpr (hasOffset) {
        MicroAPI::Add<Q_T, MicroAPI::MaskMergeMode::ZEROING>(vRes, vOffset, vRes, qTypeMaskAll);
      }

      MicroAPI::Mul<Q_T, MicroAPI::MaskMergeMode::ZEROING>(vRes, vRes, vScale, qTypeMaskAll);

      MicroAPI::StoreAlign<Q_T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
          ubDstAddrTmp, vRes, blockStride, repeatStride, qTypeMaskAll);
    }
  }
}

template <typename Q_T, typename KV_T, typename ANTIQ_PARAMS_T, uint32_t baseSize, bool hasOffset = false>
__aicore__ inline void AntiquantVFW4Norm(LocalTensor<KV_T>& antiqInUb, LocalTensor<Q_T>& antiqResUb,
                                             LocalTensor<ANTIQ_PARAMS_T>& antiqOffsetUb, LocalTensor<ANTIQ_PARAMS_T>& antiqScaleUb,
                                             uint32_t dealRowCount) {
  static_assert(baseSize % 128 == 0);
  __ubuf__ uint8_t* ubSrcAddr = (__ubuf__ uint8_t*)(antiqInUb.GetPhyAddr());
  __ubuf__ Q_T* ubDstAddr = (__ubuf__ Q_T*)(antiqResUb.GetPhyAddr());
  __ubuf__ Q_T* ubOffsetAddr = (__ubuf__ Q_T*)antiqOffsetUb.GetPhyAddr();
  __ubuf__ Q_T* ubScaleAddr = (__ubuf__ Q_T*)antiqScaleUb.GetPhyAddr();

  AntiquantVFImplW4Norm<Q_T, KV_T, ANTIQ_PARAMS_T, baseSize, hasOffset>(
    ubSrcAddr, ubDstAddr, ubOffsetAddr, ubScaleAddr, dealRowCount);
}

template <typename Q_T, typename ANTIQ_PARAMS_T>
__simd_callee__ inline void LoadParamAndBroadcast(
    MicroAPI::RegTensor<ANTIQ_PARAMS_T>& vParam, MicroAPI::RegTensor<Q_T>& vParamFp16,
    MicroAPI::UnalignRegForLoad& u, __ubuf__ ANTIQ_PARAMS_T*& ubAddr,
    MicroAPI::MaskReg& maskOne, MicroAPI::MaskReg& qTypeMaskAll)
{
    MicroAPI::LoadUnAlign<ANTIQ_PARAMS_T>(vParam, u, ubAddr, 1); // 1表示ub自动往后偏移1个float
    MicroAPI::Cast<Q_T, ANTIQ_PARAMS_T, castTraitW4_0>(vParamFp16, vParam, maskOne);
    MicroAPI::Duplicate<uint16_t, MicroAPI::HighLowPart::LOWEST, MicroAPI::MaskMergeMode::ZEROING>(
        (MicroAPI::RegTensor<uint16_t>&)vParamFp16, (MicroAPI::RegTensor<uint16_t>&)vParamFp16, qTypeMaskAll);
}

template <typename Q_T, typename KV_T, typename ANTIQ_PARAMS_T, uint32_t baseSize, bool hasOffset = false>
__simd_vf__ void AntiquantVFImplW4PerTokenD128(__ubuf__ uint8_t* ubSrcAddr, __ubuf__ Q_T* ubDstAddr, __ubuf__ ANTIQ_PARAMS_T* ubOffsetAddr,
                                              __ubuf__ ANTIQ_PARAMS_T* ubScaleAddr, uint32_t dealRowCount)
{
  MicroAPI::RegTensor<int4x2_t> vKvData;
  MicroAPI::RegTensor<ANTIQ_PARAMS_T> vOffset;
  MicroAPI::RegTensor<ANTIQ_PARAMS_T> vScale;
  MicroAPI::RegTensor<Q_T> vOffsetFp16;
  MicroAPI::RegTensor<Q_T> vScaleFp16;
  MicroAPI::RegTensor<Q_T> vRes;
  MicroAPI::RegTensor<half> vCastFp16Res;
  MicroAPI::MaskReg maskOne = MicroAPI::CreateMask<ANTIQ_PARAMS_T, MicroAPI::MaskPattern::VL1>();
  MicroAPI::MaskReg kvMaskAll = MicroAPI::CreateMask<KV_T, MicroAPI::MaskPattern::ALL>();
  MicroAPI::MaskReg qTypeMaskAll = MicroAPI::CreateMask<Q_T, MicroAPI::MaskPattern::ALL>();

  uint32_t blockStride = dealRowCount + 1;
  uint32_t repeatStride = 1;
  const uint16_t colLoopCnt = static_cast<uint16_t>(baseSize / 128);
  MicroAPI::UnalignRegForLoad u0;
  MicroAPI::UnalignRegForLoad u1;
  for (uint16_t colLoopIdx = 0; colLoopIdx < colLoopCnt; colLoopIdx++) {
    __ubuf__ Q_T* ubDstAddrTmp = ubDstAddr + blockStride * 128 * colLoopIdx;
    __ubuf__ uint8_t* ubSrcTemp = ubSrcAddr + colLoopIdx * 64;
    __ubuf__ ANTIQ_PARAMS_T* ubScaleAddrTemp = ubScaleAddr;
    __ubuf__ ANTIQ_PARAMS_T* ubOffsetAddrTemp = ubOffsetAddr;

    MicroAPI::LoadUnAlignPre(u0, ubScaleAddrTemp);
    if constexpr (hasOffset) {
      MicroAPI::LoadUnAlignPre(u1, ubOffsetAddrTemp);
    }
    for (uint16_t dealRowIdx = 0; dealRowIdx < static_cast<uint16_t>(dealRowCount); dealRowIdx++) {
      MicroAPI::LoadAlign<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
          (MicroAPI::RegTensor<uint8_t>&)vKvData, ubSrcTemp, baseSize / 2);
      if constexpr (std::is_same<Q_T, bfloat16_t>::value) {
        MicroAPI::Cast<half, int4x2_t, castTraitW4>(vCastFp16Res, vKvData, kvMaskAll);
        MicroAPI::Cast<Q_T, half, castTraitW4_2>(vRes, vCastFp16Res, kvMaskAll);
      } else {
        MicroAPI::Cast<Q_T, int4x2_t, castTraitW4>(vRes, vKvData, kvMaskAll);
      }
      LoadParamAndBroadcast<Q_T, ANTIQ_PARAMS_T>(vScale, vScaleFp16, u0, ubScaleAddrTemp, maskOne, qTypeMaskAll);
      if constexpr (hasOffset) {
        LoadParamAndBroadcast<Q_T, ANTIQ_PARAMS_T>(vOffset, vOffsetFp16, u1, ubOffsetAddrTemp, maskOne, qTypeMaskAll);
        MicroAPI::Add<Q_T, MicroAPI::MaskMergeMode::ZEROING>(vRes, vRes, vOffsetFp16, qTypeMaskAll);
      }

      MicroAPI::Mul<Q_T, MicroAPI::MaskMergeMode::ZEROING>(vRes, vRes, vScaleFp16, qTypeMaskAll);

      MicroAPI::StoreAlign<Q_T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
          ubDstAddrTmp, vRes, blockStride, repeatStride, qTypeMaskAll);
    }
  }
}

template <typename Q_T, typename KV_T, typename ANTIQ_PARAMS_T, uint32_t baseSize, bool hasOffset = false>
__aicore__ inline void AntiquantVFW4PerTokenD128(LocalTensor<KV_T>& antiqInUb, LocalTensor<Q_T>& antiqResUb,
                                                     LocalTensor<ANTIQ_PARAMS_T>& antiqOffsetUb, LocalTensor<ANTIQ_PARAMS_T>& antiqScaleUb,
                                                     uint32_t dealRowCount) {
  static_assert(baseSize == 128);
  static_assert(IsSameType<KV_T, int4b_t>::value, "antiquant perToken w4, KV_T must be int4_t");
  __ubuf__ uint8_t* ubSrcAddr = (__ubuf__ uint8_t*)(antiqInUb.GetPhyAddr());
  __ubuf__ Q_T* ubDstAddr = (__ubuf__ Q_T*)(antiqResUb.GetPhyAddr());
  __ubuf__ ANTIQ_PARAMS_T* ubOffsetAddr = (__ubuf__ ANTIQ_PARAMS_T*)antiqOffsetUb.GetPhyAddr();
  __ubuf__ ANTIQ_PARAMS_T* ubScaleAddr = (__ubuf__ ANTIQ_PARAMS_T*)antiqScaleUb.GetPhyAddr();

  AntiquantVFImplW4PerTokenD128<Q_T, KV_T, ANTIQ_PARAMS_T, baseSize, hasOffset>(
    ubSrcAddr, ubDstAddr, ubOffsetAddr, ubScaleAddr, dealRowCount);
}

template <typename Q_T, typename KV_T, bool hasOffset>
__simd_callee__ inline void CastAddMulStoreW4(MicroAPI::RegTensor<int4x2_t>& vKvData,
    MicroAPI::RegTensor<half>& vCastFp16Res, MicroAPI::RegTensor<Q_T>& vRes,
    MicroAPI::RegTensor<Q_T>& vOffsetFp16, MicroAPI::RegTensor<Q_T>& vScaleFp16,
    __ubuf__ Q_T*& ubDstAddr, uint32_t blockStride, uint32_t repeatStride,
    MicroAPI::MaskReg& kvMaskAll, MicroAPI::MaskReg& qMaskAll)
{
    if constexpr (std::is_same<Q_T, bfloat16_t>::value) {
        MicroAPI::Cast<half, int4x2_t, castTraitW4>(vCastFp16Res, vKvData, kvMaskAll);
        MicroAPI::Cast<Q_T, half, castTraitW4_2>(vRes, vCastFp16Res, kvMaskAll);
    } else {
        MicroAPI::Cast<Q_T, int4x2_t, castTraitW4>(vRes, vKvData, kvMaskAll);
    }
    if constexpr (hasOffset) {
        MicroAPI::Add<Q_T, MicroAPI::MaskMergeMode::ZEROING>(vRes, vRes, vOffsetFp16, qMaskAll);
    }
    MicroAPI::Mul<Q_T, MicroAPI::MaskMergeMode::ZEROING>(vRes, vRes, vScaleFp16, qMaskAll);
    MicroAPI::StoreAlign<Q_T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
        ubDstAddr, vRes, blockStride, repeatStride, qMaskAll);
}

template <typename Q_T, typename KV_T, typename ANTIQ_PARAMS_T, uint32_t baseSize, bool hasOffset = false>
__simd_vf__ void AntiquantVFImplW4PerTokenD256(__ubuf__ uint8_t* ubSrcAddr, __ubuf__ uint8_t* ubSrcAddr1,
                                              __ubuf__ Q_T* ubDstAddr, __ubuf__ ANTIQ_PARAMS_T* ubOffsetAddr, 
                                              __ubuf__ ANTIQ_PARAMS_T* ubScaleAddr, uint32_t dealRowCount)
{
  MicroAPI::RegTensor<int4x2_t> vKvData, vKvData1;
  MicroAPI::RegTensor<ANTIQ_PARAMS_T> vOffset, vScale;
  MicroAPI::RegTensor<Q_T> vOffsetFp16, vScaleFp16, vRes, vRes1;
  MicroAPI::RegTensor<half> vCastFp16Res, vCastFp16Res1;
  MicroAPI::MaskReg maskOne = MicroAPI::CreateMask<ANTIQ_PARAMS_T, MicroAPI::MaskPattern::VL1>();
  MicroAPI::MaskReg kvMaskAll = MicroAPI::CreateMask<KV_T, MicroAPI::MaskPattern::ALL>();
  MicroAPI::MaskReg qMaskAll = MicroAPI::CreateMask<Q_T, MicroAPI::MaskPattern::ALL>();

  uint32_t blockStride = 1 + dealRowCount;
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
    LoadParamAndBroadcast<Q_T, ANTIQ_PARAMS_T>(vScale, vScaleFp16, u0, ubScaleAddr, maskOne, qMaskAll);
    if constexpr (hasOffset) {
      LoadParamAndBroadcast<Q_T, ANTIQ_PARAMS_T>(vOffset, vOffsetFp16, u1, ubOffsetAddr, maskOne, qMaskAll);
    }

    MicroAPI::LoadAlign<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
      (MicroAPI::RegTensor<uint8_t>&)vKvData, ubSrcAddr, 128); // d=256，ub是uint8_t的，偏移128
    MicroAPI::LoadAlign<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
      (MicroAPI::RegTensor<uint8_t>&)vKvData1, ubSrcAddr1, 128); // d=256，ub是uint8_t的，偏移128
    CastAddMulStoreW4<Q_T, KV_T, hasOffset>(vKvData, vCastFp16Res, vRes, vOffsetFp16, vScaleFp16,
      ubDstAddr, blockStride, repeatStride, kvMaskAll, qMaskAll);
    CastAddMulStoreW4<Q_T, KV_T, hasOffset>(vKvData1, vCastFp16Res1, vRes1, vOffsetFp16, vScaleFp16,
      ubDstAddr1, blockStride, repeatStride, kvMaskAll, qMaskAll);
  }
}


template <typename Q_T, typename KV_T, typename ANTIQ_PARAMS_T, uint32_t baseSize, bool hasOffset = false>
__aicore__ inline void AntiquantVFW4PerTokenD256(LocalTensor<KV_T>& antiqInUb, LocalTensor<Q_T>& antiqResUb,
                                                  LocalTensor<ANTIQ_PARAMS_T>& antiqOffsetUb, LocalTensor<ANTIQ_PARAMS_T>& antiqScaleUb,
                                                  uint32_t dealRowCount) {
  static_assert(baseSize == 256);
  static_assert(IsSameType<KV_T, int4b_t>::value, "antiquant perToken w4, KV_T must be int4_t");
  __ubuf__ uint8_t* ubSrcAddr = (__ubuf__ uint8_t*)(antiqInUb.GetPhyAddr());
  __ubuf__ uint8_t* ubSrcAddr1 = ubSrcAddr + 64; // ub是int8_t的
  __ubuf__ Q_T* ubDstAddr = (__ubuf__ Q_T*)(antiqResUb.GetPhyAddr());
  __ubuf__ ANTIQ_PARAMS_T* ubOffsetAddr = (__ubuf__ ANTIQ_PARAMS_T*)antiqOffsetUb.GetPhyAddr();
  __ubuf__ ANTIQ_PARAMS_T* ubScaleAddr = (__ubuf__ ANTIQ_PARAMS_T*)antiqScaleUb.GetPhyAddr();

  AntiquantVFImplW4PerTokenD256<Q_T, KV_T, ANTIQ_PARAMS_T, baseSize, hasOffset>(
    ubSrcAddr, ubSrcAddr1, ubDstAddr, ubOffsetAddr, ubScaleAddr, dealRowCount);
}

template <typename Q_T, typename KV_T, typename ANTIQ_PARAMS_T, uint32_t baseSize, bool hasOffset = false>
__simd_vf__ void AntiquantVFImplW4PerTokenD512(__ubuf__ uint8_t* ubSrcAddr, __ubuf__ uint8_t* ubSrcAddr1,
                                               __ubuf__ uint8_t* ubSrcAddr2, __ubuf__ uint8_t* ubSrcAddr3, 
                                               __ubuf__ Q_T* ubDstAddr, __ubuf__ ANTIQ_PARAMS_T* ubOffsetAddr, 
                                               __ubuf__ ANTIQ_PARAMS_T* ubScaleAddr, uint32_t dealRowCount)
{
  MicroAPI::RegTensor<int4x2_t> vKvData, vKvData1, vKvData2, vKvData3;
  MicroAPI::RegTensor<ANTIQ_PARAMS_T> vOffset, vScale;
  MicroAPI::RegTensor<Q_T> vOffsetFp16, vScaleFp16, vRes, vRes1, vRes2, vRes3;
  MicroAPI::RegTensor<half> vCastFp16Res, vCastFp16Res1, vCastFp16Res2, vCastFp16Res3;
  MicroAPI::MaskReg maskOne = MicroAPI::CreateMask<ANTIQ_PARAMS_T, MicroAPI::MaskPattern::VL1>();
  MicroAPI::MaskReg kvMaskAll = MicroAPI::CreateMask<KV_T, MicroAPI::MaskPattern::ALL>();
  MicroAPI::MaskReg qMaskAll = MicroAPI::CreateMask<Q_T, MicroAPI::MaskPattern::ALL>();

  uint32_t blockStride = dealRowCount + 1;
  uint32_t repeatStride = 1;
  MicroAPI::UnalignRegForLoad u1;
  MicroAPI::UnalignRegForLoad u0;
  MicroAPI::LoadUnAlignPre(u0, ubScaleAddr);
  if constexpr (hasOffset) {
    MicroAPI::LoadUnAlignPre(u1, ubOffsetAddr);
  }
  __ubuf__ Q_T* ubDstAddr1 = ubDstAddr + blockStride * 128;
  __ubuf__ Q_T* ubDstAddr2 = ubDstAddr + blockStride * 128 * 2;
  __ubuf__ Q_T* ubDstAddr3 = ubDstAddr + blockStride * 128 * 3;
  for (uint16_t j = 0; j < static_cast<uint16_t>(dealRowCount); j++) {
    // 读入每行的伪量化参数
    LoadParamAndBroadcast<Q_T, ANTIQ_PARAMS_T>(vScale, vScaleFp16, u0, ubScaleAddr, maskOne, qMaskAll);
    if constexpr (hasOffset) {
      LoadParamAndBroadcast<Q_T, ANTIQ_PARAMS_T>(vOffset, vOffsetFp16, u1, ubOffsetAddr, maskOne, qMaskAll);
    }

    MicroAPI::LoadAlign<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
      (MicroAPI::RegTensor<uint8_t>&)vKvData, ubSrcAddr, 256); // d=512，ub是uint8_t的，ub往后偏移256
    MicroAPI::LoadAlign<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
      (MicroAPI::RegTensor<uint8_t>&)vKvData1, ubSrcAddr1, 256); // d=512，ub是uint8_t的，偏移256
    MicroAPI::LoadAlign<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
      (MicroAPI::RegTensor<uint8_t>&)vKvData2, ubSrcAddr2, 256); // d=512，ub是uint8_t的，偏移256
    MicroAPI::LoadAlign<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
      (MicroAPI::RegTensor<uint8_t>&)vKvData3, ubSrcAddr3, 256); // d=512，ub是uint8_t的，偏移256
    CastAddMulStoreW4<Q_T, KV_T, hasOffset>(vKvData, vCastFp16Res, vRes, vOffsetFp16, vScaleFp16,
      ubDstAddr, blockStride, repeatStride, kvMaskAll, qMaskAll);
    CastAddMulStoreW4<Q_T, KV_T, hasOffset>(vKvData1, vCastFp16Res1, vRes1, vOffsetFp16, vScaleFp16,
      ubDstAddr1, blockStride, repeatStride, kvMaskAll, qMaskAll);
    CastAddMulStoreW4<Q_T, KV_T, hasOffset>(vKvData2, vCastFp16Res2, vRes2, vOffsetFp16, vScaleFp16,
      ubDstAddr2, blockStride, repeatStride, kvMaskAll, qMaskAll);
    CastAddMulStoreW4<Q_T, KV_T, hasOffset>(vKvData3, vCastFp16Res3, vRes3, vOffsetFp16, vScaleFp16,
      ubDstAddr3, blockStride, repeatStride, kvMaskAll, qMaskAll);
  }
}

template <typename Q_T, typename KV_T, typename ANTIQ_PARAMS_T, uint32_t baseSize, bool hasOffset = false>
__aicore__ inline void AntiquantVFW4PerTokenD512(LocalTensor<KV_T>& antiqInUb, LocalTensor<Q_T>& antiqResUb,
                                                LocalTensor<ANTIQ_PARAMS_T>& antiqOffsetUb, LocalTensor<ANTIQ_PARAMS_T>& antiqScaleUb,
                                                uint32_t dealRowCount) {
  static_assert(baseSize == 512);
  static_assert(IsSameType<KV_T, int4b_t>::value, "antiquant perToken w4, KV_T must be int4_t");
  __ubuf__ uint8_t* ubSrcAddr = (__ubuf__ uint8_t*)(antiqInUb.GetPhyAddr());
  __ubuf__ uint8_t* ubSrcAddr1 = ubSrcAddr + 64; // ub是int8_t的
  __ubuf__ uint8_t* ubSrcAddr2 = ubSrcAddr + 64 * 2; // ub是int8_t的
  __ubuf__ uint8_t* ubSrcAddr3 = ubSrcAddr + 64 * 3; // ub是int8_t的
  __ubuf__ Q_T* ubDstAddr = (__ubuf__ Q_T*)(antiqResUb.GetPhyAddr());
  __ubuf__ ANTIQ_PARAMS_T* ubOffsetAddr = (__ubuf__ ANTIQ_PARAMS_T*)antiqOffsetUb.GetPhyAddr();
  __ubuf__ ANTIQ_PARAMS_T* ubScaleAddr = (__ubuf__ ANTIQ_PARAMS_T*)antiqScaleUb.GetPhyAddr();

  AntiquantVFImplW4PerTokenD512<Q_T, KV_T, ANTIQ_PARAMS_T, baseSize, hasOffset>(
    ubSrcAddr, ubSrcAddr1, ubSrcAddr2, ubSrcAddr3, ubDstAddr, ubOffsetAddr, ubScaleAddr, dealRowCount);
}

template <typename Q_T, typename KV_T, uint32_t baseSize, bool hasOffset = false>
__simd_vf__ void AntiquantVFImplFP4_d64(__ubuf__ uint8_t *ubSrcAddr, __ubuf__ Q_T *ubDstAddr, __ubuf__ Q_T *ubDstAddr_, 
                                        __ubuf__ Q_T *ubScalerSrcAddr, uint32_t dealRowCount)
{
  MicroAPI::RegTensor<fp4x2_e2m1_t> w_nd_f4;
  MicroAPI::RegTensor<bfloat16_t> w_nd_bf16;
  MicroAPI::RegTensor<half> w_nd_f16;
  MicroAPI::RegTensor<half> w_nd_f16_1;
  MicroAPI::RegTensor<bfloat16_t> v_mul_res;
  MicroAPI::RegTensor<bfloat16_t> scale;
  MicroAPI::RegTensor<half> offset;

  MicroAPI::MaskReg preg_0;
  MicroAPI::MaskReg preg_tmp;
  MicroAPI::MaskReg preg_prev_64;
  MicroAPI::MaskReg preg_next_64;
  preg_0 = MicroAPI::CreateMask<Q_T, MicroAPI::MaskPattern::ALL>();
  preg_prev_64 = MicroAPI::CreateMask<Q_T, MicroAPI::MaskPattern::VL64>();
  preg_tmp = MicroAPI::CreateMask<Q_T, MicroAPI::MaskPattern::ALL>();

  MicroAPI::Xor(preg_next_64, preg_prev_64, preg_tmp, preg_0);
  uint32_t dealElementSize = 128; // 一次寄存器能处理的元素个数
  uint32_t blockStride = dealRowCount + 1;
  uint32_t repeatStride = 2;
  uint16_t loopCnt = static_cast<uint16_t>((dealRowCount + 1) / 2); // 加1是为了处理奇数行
  for (uint16_t i = 0; i < loopCnt; i++) {
    MicroAPI::LoadAlign<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, 
        MicroAPI::LoadDist::DIST_UNPACK4_B8>((MicroAPI::RegTensor<uint8_t>&)w_nd_f4, ubSrcAddr, 64); // 从UB中读取输入数据

    MicroAPI::Cast<bfloat16_t, KV_T, castTraitW4>(w_nd_bf16, (MicroAPI::RegTensor<KV_T>&)w_nd_f4, preg_0);

    // 加载 scale
    MicroAPI::LoadAlign<bfloat16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, 
        MicroAPI::LoadDist::DIST_E2B_B16>(scale, (__ubuf__ bfloat16_t *&)ubScalerSrcAddr, 8);
    // mul操作
    MicroAPI::Mul<bfloat16_t, MicroAPI::MaskMergeMode::ZEROING>(v_mul_res, w_nd_bf16, scale, preg_0);
    if constexpr (std::is_same<Q_T, bfloat16_t>::value) {
      // 将输出结果copy到UB
      MicroAPI::StoreAlign<Q_T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, 
          MicroAPI::PostLiteral::POST_MODE_UPDATE>(ubDstAddr, v_mul_res, blockStride, repeatStride, preg_prev_64);
      MicroAPI::StoreAlign<Q_T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, 
          MicroAPI::PostLiteral::POST_MODE_UPDATE>(ubDstAddr_, v_mul_res, blockStride, repeatStride, preg_next_64);
    } else {
      MicroAPI::Cast<half, bfloat16_t, castTraitW4_1>(w_nd_f16, v_mul_res, preg_0);
      // 将输出结果copy到UB
      MicroAPI::StoreAlign<half, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, 
          MicroAPI::PostLiteral::POST_MODE_UPDATE>(ubDstAddr, w_nd_f16, blockStride, repeatStride, preg_prev_64);
      MicroAPI::StoreAlign<half, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, 
          MicroAPI::PostLiteral::POST_MODE_UPDATE>(ubDstAddr_, w_nd_f16, blockStride, repeatStride, preg_next_64);
    }
  }
}

template <typename Q_T, typename KV_T, uint32_t baseSize, bool hasOffset = false>
__aicore__ inline void AntiquantVFFP4_d64(LocalTensor<KV_T>& antiqInUb, LocalTensor<Q_T>& antiqResUb,
                                         LocalTensor<Q_T>& antiqOffsetUb, LocalTensor<Q_T>& antiqScaleUb,
                                         uint32_t dealRowCount) {
  __ubuf__ uint8_t *ubSrcAddr = (__ubuf__ uint8_t *)(antiqInUb.GetPhyAddr());
  __ubuf__ Q_T *ubDstAddr = (__ubuf__ Q_T *)(antiqResUb.GetPhyAddr());
  // d=64时，为了充分利用寄存器，每次仍读取d=128的数据，前32和后32个数，nd2nz写出时地址不同
  __ubuf__ Q_T *ubDstAddr_ = ubDstAddr + 16 - (dealRowCount + 1) * 32 * 4 / 2;
  __ubuf__ Q_T *ubScalerSrcAddr = (__ubuf__ Q_T *)antiqScaleUb.GetPhyAddr();

  AntiquantVFImplFP4_d64<Q_T, KV_T, baseSize, hasOffset>(ubSrcAddr, ubDstAddr, ubDstAddr_, ubScalerSrcAddr, dealRowCount);
}

template <typename Q_T, typename KV_T, uint32_t baseSize, bool hasOffset = false>
__simd_vf__ void AntiquantVFImplFP4_norm(__ubuf__ uint8_t *ubSrcAddr, __ubuf__ Q_T *ubDstAddr, 
                                        __ubuf__ Q_T *ubScalerSrcAddr, uint32_t dealRowCount)
{
  MicroAPI::RegTensor<fp4x2_e2m1_t> w_nd_f4;
  MicroAPI::RegTensor<bfloat16_t> w_nd_bf16;
  MicroAPI::RegTensor<half> w_nd_f16;
  MicroAPI::RegTensor<bfloat16_t> v_mul_res;
  MicroAPI::RegTensor<bfloat16_t> scale;
  
  MicroAPI::MaskReg preg_0;
  preg_0 = MicroAPI::CreateMask<Q_T, MicroAPI::MaskPattern::ALL>();

  uint32_t blockStride = dealRowCount + 1;
  uint32_t repeatStride = 1;
  const uint16_t colLoopCnt = static_cast<uint16_t>(baseSize / 128);

  for (uint16_t colLoopIdx = 0; colLoopIdx < colLoopCnt; colLoopIdx++) {
    __ubuf__ Q_T* ubDstAddrTmp = ubDstAddr + blockStride * 128 * colLoopIdx;
    __ubuf__ uint8_t* ubSrcTemp = ubSrcAddr + colLoopIdx * 64;
    __ubuf__ Q_T *ubScalerSrcAddrTemp = ubScalerSrcAddr + colLoopIdx * 8; // 每次内层循环开始的ub
    for (uint16_t i = 0; i < static_cast<uint16_t>(dealRowCount); i++) { 
        MicroAPI::LoadAlign<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_UNPACK4_B8>
          ((MicroAPI::RegTensor<uint8_t>&)w_nd_f4, ubSrcTemp, baseSize / 2); // 每次ub往后偏移baseSize / 2

        // cast操作
        MicroAPI::Cast<bfloat16_t, KV_T, castTraitW4>(w_nd_bf16, (MicroAPI::RegTensor<KV_T>&)w_nd_f4, preg_0);
        // 加载 scale
        MicroAPI::LoadAlign<bfloat16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, 
            MicroAPI::LoadDist::DIST_E2B_B16>(scale, (__ubuf__ bfloat16_t *&)ubScalerSrcAddrTemp, 8 * colLoopCnt);
        // mul操作
        MicroAPI::Mul<bfloat16_t, MicroAPI::MaskMergeMode::ZEROING>(v_mul_res, w_nd_bf16, scale, preg_0);
        
        if constexpr (std::is_same<Q_T, bfloat16_t>::value) {
          // 将输出结果copy到UB
          MicroAPI::StoreAlign<Q_T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, 
              MicroAPI::PostLiteral::POST_MODE_UPDATE>(ubDstAddrTmp, v_mul_res, blockStride, repeatStride, preg_0);
        } else {
          MicroAPI::Cast<half, bfloat16_t, castTraitW4_1>(w_nd_f16, v_mul_res, preg_0);
          // 将输出结果copy到UB
          MicroAPI::StoreAlign<half, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, 
              MicroAPI::PostLiteral::POST_MODE_UPDATE>(ubDstAddrTmp, w_nd_f16, blockStride, repeatStride, preg_0);
        }
    }
  }
} 

template <typename Q_T, typename KV_T, uint32_t baseSize, bool hasOffset = false>
__aicore__ inline void AntiquantVFFP4_norm(LocalTensor<KV_T>& antiqInUb, LocalTensor<Q_T>& antiqResUb,
                                         LocalTensor<Q_T>& antiqOffsetUb, LocalTensor<Q_T>& antiqScaleUb,
                                         uint32_t dealRowCount) {
  __ubuf__ uint8_t *ubSrcAddr = (__ubuf__ uint8_t *)(antiqInUb.GetPhyAddr());
  __ubuf__ Q_T *ubDstAddr = (__ubuf__ Q_T *)(antiqResUb.GetPhyAddr());
  __ubuf__ Q_T *ubScalerSrcAddr = (__ubuf__ Q_T *)antiqScaleUb.GetPhyAddr();

  AntiquantVFImplFP4_norm<Q_T, KV_T, baseSize, hasOffset>(ubSrcAddr, ubDstAddr, ubScalerSrcAddr, dealRowCount);                         
}

template <typename Q_T, typename KV_T, uint32_t baseSize, bool hasOffset = false>
__simd_vf__ void AntiquantVFImplFP4_unalign(__ubuf__ uint8_t *ubSrcAddr, __ubuf__ Q_T *ubDstAddr, __ubuf__ Q_T *ubScalerSrcAddr, 
                                            uint32_t dealRowCount, uint32_t headDim)
{
  MicroAPI::RegTensor<fp4x2_e2m1_t> w_nd_f4;
  MicroAPI::RegTensor<bfloat16_t> w_nd_bf16;
  MicroAPI::RegTensor<half> w_nd_f16;
  MicroAPI::RegTensor<bfloat16_t> v_mul_res;
  MicroAPI::RegTensor<bfloat16_t> scale;
  MicroAPI::RegTensor<uint16_t> mask_tmp;
  MicroAPI::RegTensor<uint16_t> prefixSum;
  MicroAPI::UnalignRegForLoad u0;

  MicroAPI::MaskReg preg_0;
  MicroAPI::MaskReg preg_1;
  preg_0 = MicroAPI::CreateMask<Q_T, MicroAPI::MaskPattern::ALL>();
  MicroAPI::Duplicate(mask_tmp, 0x8000); // 后续填充MaskReg为c0000000
  MicroAPI::MaskGenWithRegTensor<uint16_t, 0>(preg_1, mask_tmp); // 0为mask_tmp的偏移
  MicroAPI::Unsqueeze(prefixSum, preg_1);


  uint32_t blockStride = dealRowCount + 1;
  uint32_t repeatStride = 1;
  const uint16_t colLoopCnt = static_cast<uint16_t>(baseSize / 128);
  const uint16_t scaleStride = static_cast<uint16_t>(headDim / 16); // scale 复制了一份，每16个数对应一个scale

  for (uint16_t colLoopIdx = 0; colLoopIdx < colLoopCnt; colLoopIdx++) {
    __ubuf__ Q_T* ubDstAddrTmp = ubDstAddr + blockStride * 128 * colLoopIdx;
    __ubuf__ uint8_t* ubSrcTemp = ubSrcAddr + colLoopIdx * 64;
    __ubuf__ Q_T *ubScalerSrcAddrTemp = ubScalerSrcAddr + colLoopIdx * 8; // 每次内层循环开始的ub
    for (uint16_t i = 0; i < static_cast<uint16_t>(dealRowCount); i++) { 
        MicroAPI::LoadAlign<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_UNPACK4_B8>
          ((MicroAPI::RegTensor<uint8_t>&)w_nd_f4, ubSrcTemp, baseSize / 2); // 每次ub往后偏移baseSize / 2

        // cast操作
        MicroAPI::Cast<bfloat16_t, KV_T, castTraitW4>(w_nd_bf16, (MicroAPI::RegTensor<KV_T>&)w_nd_f4, preg_0);
        // 加载 scale
        MicroAPI::LoadUnAlignPre(u0, (__ubuf__ bfloat16_t *&)ubScalerSrcAddrTemp);
        MicroAPI::LoadUnAlign<bfloat16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            scale, u0, (__ubuf__ bfloat16_t *&)ubScalerSrcAddrTemp, scaleStride);
        MicroAPI::Gather(scale, scale, prefixSum);
        // mul操作
        MicroAPI::Mul<bfloat16_t, MicroAPI::MaskMergeMode::ZEROING>(v_mul_res, scale, w_nd_bf16, preg_0);
        
        if constexpr (std::is_same<Q_T, bfloat16_t>::value) {
          // 将输出结果copy到UB
          MicroAPI::StoreAlign<Q_T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, 
              MicroAPI::PostLiteral::POST_MODE_UPDATE>(ubDstAddrTmp, v_mul_res, blockStride, repeatStride, preg_0);
        } else {
          MicroAPI::Cast<half, bfloat16_t, castTraitW4_1>(w_nd_f16, v_mul_res, preg_0);
          // 将输出结果copy到UB
          MicroAPI::StoreAlign<half, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, 
              MicroAPI::PostLiteral::POST_MODE_UPDATE>(ubDstAddrTmp, w_nd_f16, blockStride, repeatStride, preg_0);
        }
    }
  }
}

template <typename Q_T, typename KV_T, uint32_t baseSize, bool hasOffset = false>
__aicore__ inline void AntiquantVFFP4_unalign(LocalTensor<KV_T>& antiqInUb, LocalTensor<Q_T>& antiqResUb,
                                         LocalTensor<Q_T>& antiqOffsetUb, LocalTensor<Q_T>& antiqScaleUb,
                                         uint32_t dealRowCount, uint32_t headDim) {
  __ubuf__ uint8_t *ubSrcAddr = (__ubuf__ uint8_t *)(antiqInUb.GetPhyAddr());
  __ubuf__ Q_T *ubDstAddr = (__ubuf__ Q_T *)(antiqResUb.GetPhyAddr());
  __ubuf__ Q_T *ubScalerSrcAddr = (__ubuf__ Q_T *)antiqScaleUb.GetPhyAddr();

  AntiquantVFImplFP4_unalign<Q_T, KV_T, baseSize, hasOffset>(ubSrcAddr, ubDstAddr, ubScalerSrcAddr, dealRowCount, headDim);
}

template <typename Q_T, typename ANTIQ_PARAMS_T, uint32_t baseSize, bool hasOffset = false, bool isPerToken = false, bool isKvCacheNz = false>
__aicore__ inline void AntiquantVFImpl(LocalTensor<fp4x2_e2m1_t>& antiqInUb, LocalTensor<Q_T>& antiqResUb,
                                       LocalTensor<Q_T>& antiqOffsetUb, LocalTensor<Q_T>& antiqScaleUb,
                                       uint32_t dealRowCount, uint32_t headDim, uint32_t copyTotalS) {
    if constexpr (isKvCacheNz) {
      AntiquantVFImplW4PerTokenGroupNz<Q_T, fp4x2_e2m1_t, baseSize, false>(antiqInUb, antiqResUb, antiqOffsetUb, antiqScaleUb, dealRowCount, copyTotalS);
    } else {
      if constexpr (baseSize == 64) {
        AntiquantVFFP4_d64<Q_T, fp4x2_e2m1_t, baseSize, hasOffset>(antiqInUb, antiqResUb, antiqOffsetUb, antiqScaleUb, dealRowCount);
      } else {
        if (baseSize != headDim) {
          AntiquantVFFP4_unalign<Q_T, fp4x2_e2m1_t, baseSize, hasOffset>(antiqInUb, antiqResUb, antiqOffsetUb, antiqScaleUb, dealRowCount, headDim);
        } else {
          AntiquantVFFP4_norm<Q_T, fp4x2_e2m1_t, baseSize, hasOffset>(antiqInUb, antiqResUb, antiqOffsetUb, antiqScaleUb, dealRowCount);
        }
      }
  }
}

template <typename Q_T, typename ANTIQ_PARAMS_T, uint32_t baseSize, bool hasOffset = false, bool isPerToken = false, bool isKvCacheNz = false>
__aicore__ inline void AntiquantVFImpl(LocalTensor<fp4x2_e1m2_t>& antiqInUb, LocalTensor<Q_T>& antiqResUb,
                                       LocalTensor<Q_T>& antiqOffsetUb, LocalTensor<Q_T>& antiqScaleUb,
                                        uint32_t dealRowCount, uint32_t headDim, uint32_t copyTotalS) {
    if constexpr (isKvCacheNz) {
      AntiquantVFImplW4PerTokenGroupNz<Q_T, fp4x2_e1m2_t, baseSize, false>(antiqInUb, antiqResUb, antiqOffsetUb, antiqScaleUb, dealRowCount, copyTotalS);
    } else {
      if constexpr (baseSize == 64) {
        AntiquantVFFP4_d64<Q_T, fp4x2_e1m2_t, baseSize, hasOffset>(antiqInUb, antiqResUb, antiqOffsetUb, antiqScaleUb, dealRowCount);
      } else {
        if (baseSize != headDim) {
          AntiquantVFFP4_unalign<Q_T, fp4x2_e1m2_t, baseSize, hasOffset>(antiqInUb, antiqResUb, antiqOffsetUb, antiqScaleUb, dealRowCount, headDim);
        } else {
          AntiquantVFFP4_norm<Q_T, fp4x2_e1m2_t, baseSize, hasOffset>(antiqInUb, antiqResUb, antiqOffsetUb, antiqScaleUb, dealRowCount);
        }
      }
    }
}

template <typename Q_T, typename ANTIQ_PARAMS_T, uint32_t baseSize, bool hasOffset = false, bool isPerToken = false, bool isKvCacheNz = false>
__aicore__ inline void AntiquantVFImpl(LocalTensor<int4b_t>& antiqInUb, LocalTensor<Q_T>& antiqResUb,
                                       LocalTensor<ANTIQ_PARAMS_T>& antiqOffsetUb, LocalTensor<ANTIQ_PARAMS_T>& antiqScaleUb,
                                       uint32_t dealRowCount, uint32_t headDim, uint32_t copyTotalS) {
  if constexpr (isKvCacheNz) {
    if constexpr (!isPerToken) {
      AntiquantVFW4Nz<Q_T, int4b_t, ANTIQ_PARAMS_T, baseSize, hasOffset>
      (antiqInUb, antiqResUb, antiqOffsetUb, antiqScaleUb, dealRowCount);
    } else {
      AntiquantVFW4PerTokenNz<Q_T, int4b_t, ANTIQ_PARAMS_T, baseSize, hasOffset>
      (antiqInUb, antiqResUb, antiqOffsetUb, antiqScaleUb, dealRowCount);
    }
  } else {
    if constexpr (isPerToken) {
      if constexpr (baseSize == 64) {
        AntiquantVFW4PerTokenD64<Q_T, int4b_t, ANTIQ_PARAMS_T, baseSize, hasOffset>
          (antiqInUb, antiqResUb, antiqOffsetUb, antiqScaleUb, dealRowCount);
      } else if constexpr (baseSize == 128) {
        AntiquantVFW4PerTokenD128<Q_T, int4b_t, ANTIQ_PARAMS_T, baseSize, hasOffset>
          (antiqInUb, antiqResUb, antiqOffsetUb, antiqScaleUb, dealRowCount);
      } else if constexpr (baseSize == 256) {
        AntiquantVFW4PerTokenD256<Q_T, int4b_t, ANTIQ_PARAMS_T, baseSize, hasOffset>
          (antiqInUb, antiqResUb, antiqOffsetUb, antiqScaleUb, dealRowCount);
      } else {
        AntiquantVFW4PerTokenD512<Q_T, int4b_t, ANTIQ_PARAMS_T, baseSize, hasOffset>
          (antiqInUb, antiqResUb, antiqOffsetUb, antiqScaleUb, dealRowCount);
      }
    } else {
      if constexpr (baseSize == 64) {
        AntiquantVFW4D64<Q_T, int4b_t, ANTIQ_PARAMS_T, baseSize, hasOffset>(antiqInUb, antiqResUb, antiqOffsetUb, antiqScaleUb, dealRowCount);
      } else {
        AntiquantVFW4Norm<Q_T, int4b_t, ANTIQ_PARAMS_T, baseSize, hasOffset>(antiqInUb, antiqResUb, antiqOffsetUb, antiqScaleUb, dealRowCount);
      }
    }
  }
}

template <typename Q_T, typename ANTIQ_PARAMS_T>
__simd_vf__ void AntiqScaleByVFImpl(__ubuf__ uint8_t *ub_src_addr, __ubuf__ half *ub_dst_addr, __ubuf__ half *ub_dst_addr_, 
                                    uint16_t loop_cnt, uint32_t tailSize)
{
  MicroAPI::RegTensor<uint8_t> w_nd_s8;
  MicroAPI::RegTensor<bfloat16_t> w_nd_bf16;
  MicroAPI::RegTensor<bfloat16_t> w_nd_bf16_1;
  MicroAPI::RegTensor<half> w_nd_f16;
  MicroAPI::RegTensor<half> w_nd_f16_1;
  MicroAPI::RegTensor<half> e8m0_zero, e8m0_nan;
  MicroAPI::RegTensor<int16_t> b16_zero, b16_nan;
  MicroAPI::MaskReg preg_135;
  MicroAPI::MaskReg p_e8m0_zero, p_e8m0_nan;
  preg_135 = MicroAPI::CreateMask<Q_T, MicroAPI::MaskPattern::ALL>();
  int16_t shift_4bit_0 = 7;

  MicroAPI::Duplicate(e8m0_zero, 0x0);
  MicroAPI::Duplicate(e8m0_nan, 0x7f80); // 0xff << 7
  MicroAPI::Duplicate(b16_zero, 0x40);
  MicroAPI::Duplicate(b16_nan, 0x7fff);
  for (uint16_t i = 0; i < static_cast<uint16_t>(loop_cnt); i++) {
    MicroAPI::LoadAlign<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, 
        MicroAPI::LoadDist::DIST_UNPACK_B8>((MicroAPI::RegTensor<uint8_t>&)w_nd_s8, ub_src_addr, 128);
    MicroAPI::ShiftLefts((MicroAPI::RegTensor<uint16_t>&)w_nd_bf16, 
        (MicroAPI::RegTensor<uint16_t>&)w_nd_s8, shift_4bit_0, preg_135);
    MicroAPI::Compare<uint16_t, CMPMODE::NE>(p_e8m0_zero, 
        (MicroAPI::RegTensor<uint16_t>&)w_nd_bf16, (MicroAPI::RegTensor<uint16_t>&)e8m0_zero, preg_135); // 等于0布尔值设置为0
    MicroAPI::Select((MicroAPI::RegTensor<uint16_t>&)w_nd_bf16, 
        (MicroAPI::RegTensor<uint16_t>&)w_nd_bf16, (MicroAPI::RegTensor<uint16_t>&)b16_zero, p_e8m0_zero); // 布尔值为0的时候，取b16_zero
    MicroAPI::Compare<uint16_t, CMPMODE::NE>(p_e8m0_nan, 
        (MicroAPI::RegTensor<uint16_t>&)w_nd_bf16, (MicroAPI::RegTensor<uint16_t>&)e8m0_nan, preg_135);
    MicroAPI::Select((MicroAPI::RegTensor<uint16_t>&)w_nd_bf16, 
        (MicroAPI::RegTensor<uint16_t>&)w_nd_bf16, (MicroAPI::RegTensor<uint16_t>&)b16_nan, p_e8m0_nan);

    MicroAPI::Interleave((MicroAPI::RegTensor<uint16_t>&)w_nd_bf16, 
        (MicroAPI::RegTensor<uint16_t>&)w_nd_bf16_1, (MicroAPI::RegTensor<uint16_t>&)w_nd_bf16, (MicroAPI::RegTensor<uint16_t>&)w_nd_bf16);
    MicroAPI::StoreAlign<bfloat16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, 
        MicroAPI::StoreDist::DIST_NORM_B16>((__ubuf__ bfloat16_t *&)ub_dst_addr, w_nd_bf16, 256, preg_135);
    MicroAPI::StoreAlign<bfloat16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, 
        MicroAPI::StoreDist::DIST_NORM_B16>((__ubuf__ bfloat16_t *&)ub_dst_addr_, w_nd_bf16_1, 256, preg_135);
  }
}

template <typename Q_T, typename ANTIQ_PARAMS_T>
__simd_vf__ void AntiqScaleByPertokenVFImpl(__ubuf__ ANTIQ_PARAMS_T *ubSrcAddr,
                                            __ubuf__ Q_T *ubDstAddr, uint32_t copyTotalS) 
{
  MicroAPI::RegTensor<ANTIQ_PARAMS_T> vAntiqParam;
  MicroAPI::RegTensor<Q_T> vAntiqParamFp16;
  MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<Q_T, MicroAPI::MaskPattern::ALL>();
  uint32_t splitS = 256 / sizeof(ANTIQ_PARAMS_T);
  uint16_t loopCnt = (copyTotalS + splitS - 1) / splitS;
  for (uint16_t i = 0; i < loopCnt; i++) {
    MicroAPI::LoadAlign<ANTIQ_PARAMS_T, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_NORM>(
      vAntiqParam, ubSrcAddr, splitS);
    MicroAPI::Cast<Q_T, ANTIQ_PARAMS_T, castTraitFp32>(vAntiqParamFp16, vAntiqParam, maskAll);
    MicroAPI::StoreAlign<Q_T, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::StoreDist::DIST_PACK_B32>(
      ubDstAddr, vAntiqParamFp16, splitS, maskAll);
  }
}

template <typename Q_T, typename ANTIQ_PARAMS_T>
__aicore__ inline void AntiqScaleByVF(
  LocalTensor<Q_T> scaleInUb, LocalTensor<ANTIQ_PARAMS_T> scaleResUb, uint32_t dealRowCount, uint32_t grpNum) {
  __ubuf__ uint8_t *ub_src_addr = (__ubuf__ uint8_t *)(scaleInUb.GetPhyAddr());
  __ubuf__ half *ub_dst_addr = (__ubuf__ half *)(scaleResUb.GetPhyAddr());
  __ubuf__ half *ub_dst_addr_ = ub_dst_addr + 128;
  uint16_t loop_cnt = dealRowCount * grpNum / 128;
  uint32_t tailSize = dealRowCount * grpNum % 128; // loop_cnt如果不是整数，则有尾块
  if (tailSize != 0) {
    loop_cnt = loop_cnt + 1;
  }

  AntiqScaleByVFImpl<Q_T, ANTIQ_PARAMS_T>(ub_src_addr, ub_dst_addr, ub_dst_addr_, loop_cnt, tailSize);
}

template <typename Q_T, typename ANTIQ_PARAMS_T>
__aicore__ inline void AntiqScaleByPertokenVF(LocalTensor<ANTIQ_PARAMS_T> antiqParamUb, uint32_t copyTotalS) {
    __ubuf__ ANTIQ_PARAMS_T* ubSrcAddr = (__ubuf__ ANTIQ_PARAMS_T*)antiqParamUb.GetPhyAddr();
    __ubuf__ Q_T* ubDstAddr = (__ubuf__ Q_T*)(antiqParamUb.GetPhyAddr());
    AntiqScaleByPertokenVFImpl<Q_T, ANTIQ_PARAMS_T>(ubSrcAddr, ubDstAddr, copyTotalS);
}

template <typename Q_T, typename ANTIQ_PARAMS_T>
__simd_vf__ void AntiqScalePerTokenGroupByVFImpl(__ubuf__ uint8_t *ub_src_addr, __ubuf__ half *ub_dst_addr, __ubuf__ half *ub_dst_addr_, 
                                      uint16_t loop_cnt, uint32_t tailSize)
  {
    MicroAPI::RegTensor<uint8_t> w_nd_s8;
    MicroAPI::RegTensor<bfloat16_t> w_nd_bf16;
    MicroAPI::RegTensor<bfloat16_t> w_nd_bf16_1;
    MicroAPI::RegTensor<half> w_nd_f16;
    MicroAPI::RegTensor<half> w_nd_f16_1;
    MicroAPI::RegTensor<int16_t> b16_zero, b16_nan;
    MicroAPI::RegTensor<half> e8m0_zero, e8m0_nan;
    MicroAPI::MaskReg preg_135;
    MicroAPI::MaskReg p_e8m0_zero, p_e8m0_nan;
    preg_135 = MicroAPI::CreateMask<Q_T, MicroAPI::MaskPattern::ALL>();
    int16_t shift_4bit_0 = 7;

    MicroAPI::Duplicate(b16_zero, 0x40);
    MicroAPI::Duplicate(b16_nan, 0x7fff);
    MicroAPI::Duplicate(e8m0_zero, 0x0);
    MicroAPI::Duplicate(e8m0_nan, 0x7f80); // 0xff << 7
    for (uint16_t i = 0; i < static_cast<uint16_t>(loop_cnt); i++) {
      MicroAPI::LoadAlign<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, 
          MicroAPI::LoadDist::DIST_UNPACK_B8>((MicroAPI::RegTensor<uint8_t>&)w_nd_s8, ub_src_addr, 128);
      MicroAPI::ShiftLefts((MicroAPI::RegTensor<uint16_t>&)w_nd_bf16, 
          (MicroAPI::RegTensor<uint16_t>&)w_nd_s8, shift_4bit_0, preg_135);
      MicroAPI::Compare<uint16_t, CMPMODE::NE>(p_e8m0_zero, 
          (MicroAPI::RegTensor<uint16_t>&)w_nd_bf16, (MicroAPI::RegTensor<uint16_t>&)e8m0_zero, preg_135); // 等于0布尔值设置为0
      MicroAPI::Select((MicroAPI::RegTensor<uint16_t>&)w_nd_bf16, 
          (MicroAPI::RegTensor<uint16_t>&)w_nd_bf16, (MicroAPI::RegTensor<uint16_t>&)b16_zero, p_e8m0_zero); // 布尔值为0的时候，取b16_zero
      MicroAPI::Compare<uint16_t, CMPMODE::NE>(p_e8m0_nan, 
          (MicroAPI::RegTensor<uint16_t>&)w_nd_bf16, (MicroAPI::RegTensor<uint16_t>&)e8m0_nan, preg_135);
      MicroAPI::Select((MicroAPI::RegTensor<uint16_t>&)w_nd_bf16, 
          (MicroAPI::RegTensor<uint16_t>&)w_nd_bf16, (MicroAPI::RegTensor<uint16_t>&)b16_nan, p_e8m0_nan);

      MicroAPI::StoreAlign<bfloat16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, 
          MicroAPI::StoreDist::DIST_NORM_B16>((__ubuf__ bfloat16_t *&)ub_dst_addr, w_nd_bf16, 128, preg_135);
    }
  }

  template <typename Q_T, typename ANTIQ_PARAMS_T>
  __aicore__ inline void AntiqScalePerTokenGroupByVF(
    LocalTensor<Q_T> scaleInUb, LocalTensor<ANTIQ_PARAMS_T> scaleResUb, uint32_t dealRowCount, uint32_t grpNum) {
    __ubuf__ uint8_t *ub_src_addr = (__ubuf__ uint8_t *)(scaleInUb.GetPhyAddr());
    __ubuf__ half *ub_dst_addr = (__ubuf__ half *)(scaleResUb.GetPhyAddr());
    __ubuf__ half *ub_dst_addr_ = ub_dst_addr + 128;
    uint32_t dealRowCountAlign = (dealRowCount + 31) / 32 * 32;
    uint16_t loop_cnt = dealRowCountAlign * grpNum / 128;
    uint32_t tailSize = dealRowCountAlign * grpNum % 128; // loop_cnt如果不是整数，则有尾块
    if (tailSize != 0) {
      loop_cnt = loop_cnt + 1;
    }
    AntiqScalePerTokenGroupByVFImpl<Q_T, ANTIQ_PARAMS_T>(ub_src_addr, ub_dst_addr, ub_dst_addr_, loop_cnt, tailSize);
  }

}; // namespace FaVectorApi

#endif