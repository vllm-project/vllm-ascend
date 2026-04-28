/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file copy_out_split_n_with_params.h
 * \brief
 */
#ifndef MATMUL_TILE_COPY_OUT_COPY_OUT_SPLIT_N_WITH_PARAMS_H
#define MATMUL_TILE_COPY_OUT_COPY_OUT_SPLIT_N_WITH_PARAMS_H

#include "../tile_copy_policy.h"
#include "../../utils/tensor_utils.h"

namespace Act {
namespace Gemm {
namespace Tile {
/**
 * @struct Copy
 * @brief Define a Copy struct for implementing data copy operations
 *
 * This struct implements data copy in the Ascend910_95 architecture using the CopyOutSplitNWithParams mode
 *
 * @param [in] OutputType: the type of the output tensor
 * @param [in] InputType: the type of the input tensor
 */
template <class OutputType, class InputType>
struct Copy<
    Arch::Ascend910_95, CopyOutSplitNWithParams, void, OutputType, InputType,
    AscendC::Std::enable_if_t<
        PosIsUB<OutputType::pos>() && IsNDOrAlign<OutputType>() &&       // UB ND/ND_ALIGN
        !IsQuantSenario<typename OutputType::T, typename InputType::T>() // no quant
    >
> {
public:
    using DstT = typename OutputType::T;
    using SrcT = typename AscendC::GetMmDstType<typename InputType::T>::Type;

    /**
     * @brief Overloaded operator() for implementing data copy operations
     * @param [in] dst: destination local tensor
     * @param [in] src: source local tensor
     * @param [in] curRow: current row index
     * @param [in] curCol: current column index
     * @param [in] l0CTileH: L0C tile height
     * @param [in] l0CTileW: L0C tile width
     * @param [in] baseM: base M value
     * @param [in] baseN: base N value
     * @param [in] orgM: original M value
     * @param [in] orgN: original N value
     * @param [in] orgKc: original Kc value
     * @param [in] id: blcok id, default is 0
     * @note This functionimplements data copy in the Ascend910_95 architecture using the CopyOutSplitNWithParams mode
     */
    __aicore__ inline void operator()(const AscendC::LocalTensor<DstT>& dst, const AscendC::LocalTensor<SrcT>& src,
        int32_t curRow, int32_t curCol, int32_t l0CTileH, int32_t l0CTileW, int32_t baseM, int32_t baseN, int32_t orgM,
        int32_t orgN, int32_t orgKc, uint8_t id = 0)
    {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3101)
        constexpr uint32_t blockCount = AscendC::ONE_BLK_SIZE / sizeof(DstT);
        uint32_t dimN = orgKc != 0 ? orgKc : orgN;
        if constexpr (OutputType::format == CubeFormat::ND_ALIGN) {
            dimN = AscendC::Ceil(dimN, blockCount) * blockCount;
        }
        auto stride = dimN;
        int64_t dstOffset = static_cast<int64_t>(static_cast<int64_t>(curRow * baseM) * stride) +
                            static_cast<int64_t>(curCol * baseN * l0CTileH);
        dstOffset = dstOffset >> 1; // splitN

        AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> params;
        params.nSize = static_cast<uint16_t>(l0CTileW);
        if constexpr (OutputType::format == CubeFormat::ND_ALIGN) {
            params.nSize = AscendC::Ceil(params.nSize, blockCount) * blockCount;
        }
        params.mSize = static_cast<uint16_t>(l0CTileH);
        params.srcStride = AscendC::CeilAlign(l0CTileH, AscendC::BLOCK_CUBE);
        params.dstStride = stride >> 1; // splitN
        if constexpr (OutputType::format == CubeFormat::ND_ALIGN) {
            params.dstStride = AscendC::Ceil(params.dstStride, blockCount) * blockCount;
        }
        if constexpr (AscendC::IsSameType<DstT, half>::value && AscendC::IsSameType<SrcT, float>::value) {
            params.quantPre = QuantMode_t::F322F16;
        } else if constexpr (AscendC::IsSameType<DstT, bfloat16_t>::value && AscendC::IsSameType<SrcT, float>::value) {
            params.quantPre = QuantMode_t::F322BF16;
        }
        params.dualDstCtl = static_cast<uint8_t>(AscendC::McgShfMode::DUAL_DST_SPLIT_N);
        params.subBlockId = id;
        AscendC::Fixpipe<DstT, SrcT, AscendC::Impl::CFG_ROW_MAJOR_UB>(dst[dstOffset], src, params);
#else
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Only support Ascend910_95"); });
#endif
    }
};

/**
 * @struct Copy
 * @brief Copy struct for Ascend910_95 architecture with specific conditions
 *
 * This struct is enabled when the output is in UB, the format is NZ, and it's not a quant scenario
 *
 * @param [in] OutputType: the type of the output tensor
 * @param [in] InputType: the type of the input tensor
 */
template <class OutputType, class InputType>
struct Copy<
    Arch::Ascend910_95, CopyOutSplitNWithParams, void, OutputType, InputType,
    AscendC::Std::enable_if_t<
        PosIsUB<OutputType::pos>() && IsNz<OutputType>() &&              // UB NZ
        !IsQuantSenario<typename OutputType::T, typename InputType::T>() // no quant
    >
> {
public:
    using DstT = typename OutputType::T;
    using SrcT = typename AscendC::GetMmDstType<typename InputType::T>::Type;

    /**
     * @brief Overloaded operator() for implementing data copy operations
     * @param [in] dst: destination local tensor
     * @param [in] src: source local tensor
     * @param [in] curRow: current row index
     * @param [in] curCol: current column index
     * @param [in] l0CTileH: L0C tile height
     * @param [in] l0CTileW: L0C tile width
     * @param [in] baseM: base M value
     * @param [in] baseN: base N value
     * @param [in] orgM: original M value
     * @param [in] orgN: original N value
     * @param [in] orgKc: original Kc value
     * @param [in] id: optional id, default is 0
     */
    __aicore__ inline void operator()(const AscendC::LocalTensor<DstT>& dst, const AscendC::LocalTensor<SrcT>& src,
        int32_t curRow, int32_t curCol, int32_t l0CTileH, int32_t l0CTileW, int32_t baseM, int32_t baseN,
        int32_t orgM, int32_t orgN, int32_t orgKc, uint8_t id = 0)
    {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3101)
        uint32_t stride = static_cast<uint32_t>(orgM * AscendC::BLOCK_CUBE);
        int64_t dstOffset =
            static_cast<int64_t>(curCol * baseN) * orgM + static_cast<int64_t>(curRow * baseM) * AscendC::BLOCK_CUBE;
        dstOffset = dstOffset >> 1; // splitM or splitN

        AscendC::FixpipeParamsC310<AscendC::CO2Layout::NZ> params;
        int32_t baseBlockWidth = AscendC::Ceil(l0CTileW, AscendC::BLOCK_CUBE);
        params.nSize = static_cast<uint16_t>(baseBlockWidth * AscendC::BLOCK_CUBE);
        params.mSize = static_cast<uint16_t>(l0CTileH);
        params.srcStride = AscendC::CeilAlign(l0CTileH, AscendC::BLOCK_CUBE);
        params.dstStride = stride;
        if constexpr (AscendC::IsSameType<DstT, half>::value && AscendC::IsSameType<SrcT, float>::value) {
            params.quantPre = QuantMode_t::F322F16;
        } else if constexpr (AscendC::IsSameType<DstT, bfloat16_t>::value && AscendC::IsSameType<SrcT, float>::value) {
            params.quantPre = QuantMode_t::F322BF16;
        }
        params.dualDstCtl = static_cast<uint8_t>(AscendC::McgShfMode::DUAL_DST_SPLIT_N);
        params.subBlockId = id;
        AscendC::Fixpipe<DstT, SrcT, AscendC::Impl::CFG_NZ_UB>(dst[dstOffset], src, params);
#else
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Only support Ascend910_95"); });
#endif
    }
};
} // namespace Tile
} // namespace Gemm
} // namespace Act
#endif