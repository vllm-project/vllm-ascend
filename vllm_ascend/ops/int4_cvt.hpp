/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under
the terms and conditions of CANN Open Software License Agreement Version 2.0
(the "License"). Please refer to the License for details. You may not use this
file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN "AS
IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the
full text of the License.
*/

/**
 * @file int4_cvt.hpp
 * @brief Type Conversion (TCVT) implementation for packed FP16 -> INT4
 * conversion
 *
 * FILE ORGANIZATION (for easy navigation):
 * =======================================
 *
 * SUPPORTED CONVERSIONS (quick lookup):
 * ====================================
 * FP16:  -> packed S4
 *
 * 1. GenCastCallFp16ToInt4
 *    - Dispatch to the correct vconv_f162s4 variant by RoundMode
 *
 * 2. TCvtHeadFp16ToInt4Packed
 *    - Processes aligned repeat blocks for the main data region
 *
 * 3. TCvtFp16ToInt4Packed
 *    - Handles aligned region and remainder with vector masking
 *
 * 4. TCVT_FP16_TO_INT4_PACKED_IMPL / TCVT_FP16_TO_INT4_PACKED
 *    - High-level entry points computing repeat configuration
 *
 * QUICK FIND: Search for "GenCastCallFp16ToInt4" or "TCVT_FP16_TO_INT4_PACKED".
 */

#ifndef FAST_HADAMARD_INTCVT4_HPP
#define FAST_HADAMARD_INTCVT4_HPP

#include <pto/pto-inst.hpp>

using namespace pto;

namespace fast_hadamard_int4 {
inline namespace TCvtInternel {
constexpr const int SAT_MODE_BIT = 59;
}  // namespace TCvtInternel

template <typename TileDataD, typename TileDataS>
PTO_INTERNAL void GenCastCallFp16ToInt4(__ubuf__ void *dst,
                                        __ubuf__ typename TileDataS::DType *src,
                                        uint8_t repeatNum, RoundMode mode,
                                        uint16_t dstBlockStride,
                                        uint16_t srcBlockStride,
                                        uint16_t dstRepeatStride,
                                        uint16_t srcRepeatStride) {
  switch (static_cast<RoundMode>(mode)) {
    case RoundMode::CAST_RINT:
      vconv_f162s4r(dst, src, repeatNum, dstBlockStride, srcBlockStride,
                    dstRepeatStride, srcRepeatStride);
      break;
    case RoundMode::CAST_ROUND:
      vconv_f162s4a(dst, src, repeatNum, dstBlockStride, srcBlockStride,
                    dstRepeatStride, srcRepeatStride);
      break;
    case RoundMode::CAST_FLOOR:
      vconv_f162s4f(dst, src, repeatNum, dstBlockStride, srcBlockStride,
                    dstRepeatStride, srcRepeatStride);
      break;
    case RoundMode::CAST_CEIL:
      vconv_f162s4c(dst, src, repeatNum, dstBlockStride, srcBlockStride,
                    dstRepeatStride, srcRepeatStride);
      break;
    case RoundMode::CAST_TRUNC:
      vconv_f162s4z(dst, src, repeatNum, dstBlockStride, srcBlockStride,
                    dstRepeatStride, srcRepeatStride);
      break;
    case RoundMode::CAST_NONE:
      vconv_f162s4(dst, src, repeatNum, dstBlockStride, srcBlockStride,
                   dstRepeatStride, srcRepeatStride);
      break;
    default:
      vconv_f162s4z(dst, src, repeatNum, dstBlockStride, srcBlockStride,
                    dstRepeatStride, srcRepeatStride);
      break;
  }
}

template <typename TileDataD, typename TileDataS, unsigned SS, unsigned DS>
PTO_INST void TCvtHeadFp16ToInt4Packed(
    __ubuf__ typename TileDataD::DType *dstPtr,
    __ubuf__ typename TileDataS::DType *srcPtr, RoundMode mode,
    unsigned numRepeatPerLine, unsigned validRow, unsigned srcElementsPerRepeat,
    unsigned dstBytesPerRepeat, unsigned dstRepeatStride,
    unsigned srcRepeatStride) {
  unsigned numLoop = numRepeatPerLine / REPEAT_MAX;
  unsigned remainAfterLoop = numRepeatPerLine % REPEAT_MAX;

  for (uint32_t i = 0; i < validRow; i++) {
    if (numLoop > 0) {
      for (uint32_t j = 0; j < numLoop; j++) {
        GenCastCallFp16ToInt4<TileDataD, TileDataS>(
            (__ubuf__ void *)(dstPtr + i * DS +
                              j * dstBytesPerRepeat * REPEAT_MAX),
            srcPtr + i * SS + j * srcElementsPerRepeat * REPEAT_MAX,
            (uint8_t)REPEAT_MAX, mode, 1, 1, (uint16_t)dstRepeatStride,
            (uint16_t)srcRepeatStride);
      }
    }
    if (remainAfterLoop > 0) {
      GenCastCallFp16ToInt4<TileDataD, TileDataS>(
          (__ubuf__ void *)(dstPtr + i * DS +
                            numLoop * dstBytesPerRepeat * REPEAT_MAX),
          srcPtr + i * SS + numLoop * srcElementsPerRepeat * REPEAT_MAX,
          (uint8_t)remainAfterLoop, mode, 1, 1, (uint16_t)dstRepeatStride,
          (uint16_t)srcRepeatStride);
    }
  }
}

template <typename TileDataD, typename TileDataS, unsigned SS, unsigned DS>
__tf__ AICORE void TCvtFp16ToInt4Packed(
    typename TileDataD::TileDType __out__ dst,
    typename TileDataS::TileDType __in__ src, RoundMode mode,
    unsigned numRepeatPerLine, unsigned numRemainPerLine, unsigned validRow,
    unsigned srcElementsPerRepeat, unsigned dstBytesPerRepeat,
    unsigned dstRepeatStride, unsigned srcRepeatStride) {
  uint64_t originalCtrl = get_ctrl();
  set_ctrl(sbitset0(originalCtrl, SAT_MODE_BIT));

  __ubuf__ typename TileDataD::DType *dstPtr =
      (__ubuf__ typename TileDataD::DType *)__cce_get_tile_ptr(dst);
  __ubuf__ typename TileDataS::DType *srcPtr =
      (__ubuf__ typename TileDataS::DType *)__cce_get_tile_ptr(src);
  constexpr unsigned dstNElemPerBlock =
      BLOCK_BYTE_SIZE / sizeof(typename TileDataD::DType);
  constexpr unsigned srcNElemPerBlock =
      BLOCK_BYTE_SIZE / sizeof(typename TileDataS::DType);

  if (numRepeatPerLine > 0) {
    TCvtHeadFp16ToInt4Packed<TileDataD, TileDataS, SS, DS>(
        dstPtr, srcPtr, mode, numRepeatPerLine, validRow, srcElementsPerRepeat,
        dstBytesPerRepeat, dstRepeatStride, srcRepeatStride);
  }

  dstPtr += numRepeatPerLine * dstBytesPerRepeat;
  srcPtr += numRepeatPerLine * srcElementsPerRepeat;

  if (numRemainPerLine > 0) {
    unsigned numLoop = validRow / REPEAT_MAX;
    unsigned remainAfterLoop = validRow % REPEAT_MAX;
    SetContinuousMask(numRemainPerLine);
    if (numLoop > 0) {
      for (uint32_t j = 0; j < numLoop; j++) {
        GenCastCallFp16ToInt4<TileDataD, TileDataS>(
            (__ubuf__ void *)(dstPtr + j * DS * REPEAT_MAX),
            srcPtr + j * SS * REPEAT_MAX, (uint8_t)REPEAT_MAX, mode, 1, 1,
            (uint16_t)DS / dstNElemPerBlock, (uint16_t)SS / srcNElemPerBlock);
      }
    }
    if (remainAfterLoop > 0) {
      GenCastCallFp16ToInt4<TileDataD, TileDataS>(
          (__ubuf__ void *)(dstPtr + numLoop * DS * REPEAT_MAX),
          srcPtr + numLoop * SS * REPEAT_MAX, (uint8_t)remainAfterLoop, mode, 1,
          1, (uint16_t)DS / dstNElemPerBlock, (uint16_t)SS / srcNElemPerBlock);
    }
    set_vector_mask(-1, -1);
  }

  set_ctrl(originalCtrl);
}

template <typename TileDataD, typename TileDataS>
PTO_INTERNAL void TCVT_FP16_TO_INT4_PACKED_IMPL(TileDataD &dst, TileDataS &src,
                                                RoundMode mode) {
  static_assert(std::is_same<typename TileDataD::DType, int8_t>::value,
                "Packed int4 destination must use int8_t.");
  static_assert(std::is_same<typename TileDataS::DType, half>::value,
                "Packed int4 conversion expects fp16 source.");

  if (dst.GetValidRow() != src.GetValidRow()) {
    return;
  }

  unsigned logicalSrcCol = src.GetValidCol();
  if ((logicalSrcCol & 1U) != 0 || dst.GetValidCol() * 2U != logicalSrcCol) {
    return;
  }

  constexpr unsigned SS = TileDataS::RowStride;
  constexpr unsigned DS = TileDataD::RowStride;
  unsigned srcElementsPerRepeat =
      REPEAT_BYTE / sizeof(typename TileDataS::DType);
  unsigned dstBytesPerRepeat = srcElementsPerRepeat / 2;
  unsigned dstRepeatStride =
      dstBytesPerRepeat / (BLOCK_BYTE_SIZE / sizeof(typename TileDataD::DType));
  unsigned srcRepeatStride =
      srcElementsPerRepeat /
      (BLOCK_BYTE_SIZE / sizeof(typename TileDataS::DType));
  unsigned numRepeatPerLine = logicalSrcCol / srcElementsPerRepeat;
  unsigned numRemainPerLine = logicalSrcCol % srcElementsPerRepeat;
  unsigned validRow = dst.GetValidRow();

  TCvtFp16ToInt4Packed<TileDataD, TileDataS, SS, DS>(
      dst.data(), src.data(), mode, numRepeatPerLine, numRemainPerLine,
      validRow, srcElementsPerRepeat, dstBytesPerRepeat, dstRepeatStride,
      srcRepeatStride);
}

template <typename DstTile, typename SrcTile>
AICORE void TCVT_FP16_TO_INT4_PACKED(DstTile &dst, SrcTile &src,
                                     RoundMode mode) {
  TCVT_FP16_TO_INT4_PACKED_IMPL(dst, src, mode);
}

}  // namespace fast_hadamard_int4

#endif
