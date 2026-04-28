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
 * \file copy_out_to_ub_with_params_v310.h
 * \brief
 */
#ifndef MATMUL_TILE_COPY_OUT_COPY_OUT_TO_UB_WITH_PARAMS_V310_H
#define MATMUL_TILE_COPY_OUT_COPY_OUT_TO_UB_WITH_PARAMS_V310_H

#include "../tile_copy_policy.h"
#include "../../utils/tensor_utils.h"

namespace Act {
namespace Gemm {
namespace Tile {
/**
 * @struct Copy
 * @brief Copy struct for Ascend910_95 architecture with specific parameters
 *
 * This struct is enabled only if the output is in UB, the format is ND or ND_ALIGN,
 * and it is not a quantization scenario
 *
 * @param [in] OutputType: the type of the output tensor
 * @param [in] InputType: the type of the input tensor
 */
template <class OutputType, class InputType>
struct Copy<
    Arch::Ascend910_95, CopyWithParams, void, OutputType, InputType,
    AscendC::Std::enable_if_t<
        PosIsUB<OutputType::pos>() && IsNDOrAlign<OutputType>() &&       // UB ND
        !IsQuantSenario<typename OutputType::T, typename InputType::T>() // no quant
    >
> {
public:
    using DstT = typename OutputType::T;
    using SrcT = typename AscendC::GetMmDstType<typename InputType::T>::Type;

    /**
     * @brief Overloaded operator() for Ascend910_95 architecture
     * @param [in] dst: destination local tensor
     * @param [in] src: source local tensor
     * @param [in] curRow: current row index
     * @param [in] curCol: current column index
     * @param [in] l0CTileHeight: L0C tile height
     * @param [in] l0CTileWidth: L0C tile width
     * @param [in] baseM: base M value
     * @param [in] baseN: base N value
     * @param [in] orgM: original M value
     * @param [in] orgN: original N value
     * @param [in] orgKc: original Kc value
     * @param [in] subBlockId: sub-block id, default is 0
     */
    __aicore__ inline void operator()(const AscendC::LocalTensor<DstT>& dst, const AscendC::LocalTensor<SrcT>& src,
        int32_t curRow, int32_t curCol, int32_t l0CTileHeight, int32_t l0CTileWidth, int32_t baseM, int32_t baseN,
        int32_t orgM, int32_t orgN, int32_t orgKc, uint8_t subBlockId = 0)
    {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3101)
        uint32_t dimN = orgKc != 0 ? orgKc : orgN;
        constexpr uint32_t blockCount = AscendC::ONE_BLK_SIZE / sizeof(DstT);
        if constexpr (OutputType::format == CubeFormat::ND_ALIGN) {
            dimN = AscendC::Ceil(dimN, blockCount) * blockCount;
        }
        int64_t dstOffset =
            static_cast<int64_t>(static_cast<int64_t>(curRow * baseM) * dimN) + static_cast<int64_t>(curCol * baseN);

        AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> params;
        params.nSize = static_cast<uint16_t>(l0CTileWidth);
        if constexpr (OutputType::format == CubeFormat::ND_ALIGN) {
            params.nSize = AscendC::Ceil(params.nSize, blockCount) * blockCount;
            dimN = AscendC::Ceil(dimN, blockCount) * blockCount;
        }
        params.mSize = static_cast<uint16_t>(l0CTileHeight);
        params.srcStride = AscendC::CeilAlign(l0CTileHeight, AscendC::BLOCK_CUBE);
        params.dstStride = dimN;
        if constexpr (AscendC::IsSameType<DstT, half>::value && AscendC::IsSameType<SrcT, float>::value) {
            params.quantPre = QuantMode_t::F322F16;
        } else if constexpr (AscendC::IsSameType<DstT, bfloat16_t>::value && AscendC::IsSameType<SrcT, float>::value) {
            params.quantPre = QuantMode_t::F322BF16;
        }
        params.subBlockId = subBlockId;
        AscendC::Fixpipe<DstT, SrcT, AscendC::Impl::CFG_ROW_MAJOR_UB>(dst[dstOffset], src, params);
#else
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Only support Ascend910_95"); });
#endif
    }
};

/**
 * @struct Copy
 * @brief Copy struct for Ascend910_95 architecture with specific parameters
 *
 * This struct is enabled only if the output is in UB, the format is NZ,
 * and it is not a quantization scenario
 *
 * @param [in] OutputType: the type of the output tensor
 * @param [in] InputType: the type of the input tensor
 */
template <class OutputType, class InputType>
struct Copy<
    Arch::Ascend910_95, CopyWithParams, void, OutputType, InputType,
    AscendC::Std::enable_if_t<
        PosIsUB<OutputType::pos>() && IsNz<OutputType>() &&              // UB NZ
        !IsQuantSenario<typename OutputType::T, typename InputType::T>() // no quant
    >
> {
public:
    using DstT = typename OutputType::T;
    using SrcT = typename AscendC::GetMmDstType<typename InputType::T>::Type;

    /**
     * @brief Overloaded operator() for Ascend910_95 architecture
     * @param [in] dst: destination local tensor
     * @param [in] src: source local tensor
     * @param [in] curRow: current row index
     * @param [in] curCol: current column index
     * @param [in] l0CTileHeight: L0C tile height
     * @param [in] l0CTileWidth: L0C tile width
     * @param [in] baseM: base M value
     * @param [in] baseN: base N value
     * @param [in] orgM: original M value
     * @param [in] orgN: original N value
     * @param [in] orgKc: original Kc value
     * @param [in] id: sub-block id, default is 0
     */
    __aicore__ inline void operator()(const AscendC::LocalTensor<DstT>& dst, const AscendC::LocalTensor<SrcT>& src,
        int32_t curRow, int32_t curCol, int32_t l0CTileHeight, int32_t l0CTileWidth, int32_t baseM, int32_t baseN,
        int32_t orgM, int32_t orgN, int32_t orgKc, uint8_t id = 0)
    {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3101)
        int64_t dstOffset =
            static_cast<int64_t>(curCol * baseN) * orgM + static_cast<int64_t>(curRow * baseM) * AscendC::BLOCK_CUBE;
        uint32_t stride =
            static_cast<uint32_t>((orgM - l0CTileHeight) * AscendC::BLOCK_CUBE * sizeof(DstT) / AscendC::ONE_BLK_SIZE);

        AscendC::FixpipeParamsC310<AscendC::CO2Layout::NZ> params;
        int32_t baseBlockWidth = AscendC::Ceil(l0CTileWidth, AscendC::BLOCK_CUBE);
        params.nSize = static_cast<uint16_t>(baseBlockWidth * AscendC::BLOCK_CUBE);
        stride =
            stride + static_cast<uint32_t>(l0CTileHeight * AscendC::BLOCK_CUBE * sizeof(SrcT) / AscendC::ONE_BLK_SIZE) *
                         sizeof(DstT) / sizeof(SrcT);
        params.mSize = static_cast<uint16_t>(l0CTileHeight);
        params.dstStride = stride;
        params.srcStride = AscendC::CeilAlign(l0CTileHeight, AscendC::BLOCK_CUBE);
        if constexpr (AscendC::IsSameType<DstT, half>::value && AscendC::IsSameType<SrcT, float>::value) {
            params.quantPre = QuantMode_t::F322F16;
        } else if constexpr (AscendC::IsSameType<DstT, bfloat16_t>::value && AscendC::IsSameType<SrcT, float>::value) {
            params.quantPre = QuantMode_t::F322BF16;
        }
        params.subBlockId = id;
        AscendC::Fixpipe<DstT, SrcT, AscendC::Impl::CFG_NZ_UB>(dst[dstOffset], src, params);
#else
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Only support Ascend910_95"); });
#endif
    }
};

/**
 * @struct Copy
 * @brief Copy struct for Ascend910_95 architecture with specific parameters
 *
 * This struct is only valid when the output is in UB, the format is either ND or ND_ALIGN, and
 * the IsQuantSenario condition is satisfied
 *
 * @param [in] OutputType: the type of the output tensor
 * @param [in] InputType: the type of the input tensor
 */
template <class OutputType, class InputType>
struct Copy<
    Arch::Ascend910_95, CopyWithParams, void, OutputType, InputType,
    AscendC::Std::enable_if_t<
        PosIsUB<OutputType::pos>() && IsNDOrAlign<OutputType>() &&      // UB ND/ND_ALIGN
        IsQuantSenario<typename OutputType::T, typename InputType::T>() // quant
    >
> {
public:
    using DstT = typename OutputType::T;
    using SrcT = typename AscendC::GetMmDstType<typename InputType::T>::Type;

    /**
     * @brief Overloaded operator() for performing data copy operations
     * @param [in] dst: destination local tensor
     * @param [in] src: source local tensor
     * @param [in] curRow: current row index
     * @param [in] curCol: current column index
     * @param [in] l0CTileHeight: L0C tile height
     * @param [in] l0CTileWidth: L0C tile width
     * @param [in] baseM: base M value
     * @param [in] baseN: base N value
     * @param [in] orgM: original M value
     * @param [in] orgN: original N value
     * @param [in] orgKc: original Kc value
     * @param [in] quantTens: quantization tensor
     */
    __aicore__ inline void operator()(const AscendC::LocalTensor<DstT>& dst, const AscendC::LocalTensor<SrcT>& src,
        int32_t curRow, int32_t curCol, int32_t l0CTileHeight, int32_t l0CTileWidth, int32_t baseM, int32_t baseN,
        int32_t orgM, int32_t orgN, int32_t orgKc, const AscendC::LocalTensor<uint64_t>& quantTens)
    {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3101)
        CopyOutNZ2ND(dst, src, curRow, curCol, l0CTileHeight, l0CTileWidth, baseM, baseN,
                     orgKc != 0 ? orgKc : orgN, quantTens);
#else
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Only support Ascend910_95"); });
#endif
    }

    /**
     * @brief Overloaded operator() for performing data copy operations
     * @param [in] dst: destination local tensor
     * @param [in] src: source local tensor
     * @param [in] curRow: current row index
     * @param [in] curCol: current column index
     * @param [in] l0CTileHeight: L0C tile height
     * @param [in] l0CTileWidth: L0C tile width
     * @param [in] baseM: base M value
     * @param [in] baseN: base N value
     * @param [in] orgM: original M value
     * @param [in] orgN: original N value
     * @param [in] orgKc: original Kc value
     * @param [in] quantScalar: quantization scalar tensor
     */
    __aicore__ inline void operator()(const AscendC::LocalTensor<DstT>& dst, const AscendC::LocalTensor<SrcT>& src,
        int32_t curRow, int32_t curCol, int32_t l0CTileHeight, int32_t l0CTileWidth, int32_t baseM, int32_t baseN,
        int32_t orgM, int32_t orgN, int32_t orgKc, uint64_t quantScalar)
    {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3101)
        CopyOutNZ2ND(dst, src, curRow, curCol, l0CTileHeight, l0CTileWidth, baseM, baseN,
                     orgKc != 0 ? orgKc : orgN, quantScalar);
#else
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Only support Ascend910_95"); });
#endif
    }

private:
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3101)
    /**
     * @brief Copy data from a local tensor to a global tensor, supporting multiple data types and quantization modes
     * @param [in] T: quantization object type
     * @param [in] dst: destination global tensor
     * @param [in] src: source local tensor
     * @param [in] curRow: current row index
     * @param [in] curCol: current column index
     * @param [in] l0CTileHeight: L0C tile height
     * @param [in] l0CTileWidth: L0C tile width
     * @param [in] baseM: base M value
     * @param [in] baseN: base N value
     * @param [in] orgN: original N value
     * @param [in] quantObject: the quantization object
     */
    template <typename T>
    __aicore__ inline void CopyOutNZ2ND(const AscendC::LocalTensor<DstT>& dst,
                                        const AscendC::LocalTensor<SrcT>& src, int32_t curRow, int32_t curCol,
                                        int32_t l0CTileHeight, int32_t l0CTileWidth, int32_t baseM, int32_t baseN,
                                        int32_t orgN, T quantObject)
    {
        auto stride = orgN;
        constexpr uint32_t blockCount = AscendC::ONE_BLK_SIZE / sizeof(DstT);
        if constexpr (OutputType::format == CubeFormat::ND_ALIGN) {
            stride = AscendC::Ceil(stride, blockCount) * blockCount;
        }
        int64_t dstOffset =
            static_cast<int64_t>(static_cast<int64_t>(curRow * baseM) * stride) + static_cast<int64_t>(curCol * baseN);

        AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> params;
        params.nSize = static_cast<uint16_t>(l0CTileWidth);
        params.mSize = static_cast<uint16_t>(l0CTileHeight);
        params.srcStride = AscendC::CeilAlign(l0CTileHeight, AscendC::BLOCK_CUBE);
        params.dstStride = stride;
        CopyTensor(dst[dstOffset], src, quantObject, params);
    }

    /**
     * @brief Copy data from a local tensor to a global tensor, supporting multiple data types and quantization modes
     * @param [in] dst: destination global tensor
     * @param [in] src: source local tensor
     * @param [in] quantTensor: quantization tensor
     * @param [in] params: the fixpipe parameters
     */
    __aicore__ inline void CopyTensor(const AscendC::LocalTensor<DstT>& dst, const AscendC::LocalTensor<SrcT>& src,
                                      const AscendC::LocalTensor<uint64_t>& quantTensor,
                                      AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR>& params)
    {
        if constexpr (AscendC::IsSameType<SrcT, int32_t>::value &&
                             (AscendC::IsSameType<DstT, int8_t>::value || AscendC::IsSameType<DstT, uint8_t>::value)) {
            params.quantPre = QuantMode_t::VREQ8;
        } else if constexpr (AscendC::IsSameType<SrcT, int32_t>::value && AscendC::IsSameType<DstT, half>::value) {
            params.quantPre = QuantMode_t::VDEQF16;
        } else if constexpr (AscendC::IsSameType<SrcT, float>::value &&
                             (AscendC::IsSameType<DstT, int8_t>::value || AscendC::IsSameType<DstT, uint8_t>::value)) {
            params.quantPre = QuantMode_t::VQF322B8_PRE;
        }
        AscendC::Fixpipe<DstT, SrcT, AscendC::CFG_ROW_MAJOR>(dst, src, quantTensor, params);
    }

    /**
     * @brief Copydata from a local tensor to a global tensor, supporting multiple data types and quantization modes
     * @param [in] dst: destination global tensor
     * @param [in] src: source local tensor
     * @param [in] quantScal: quantization scalar
     * @param [in] params: the fixpipe parameters
     */
    __aicore__ inline static void CopyTensor(const AscendC::LocalTensor<DstT>& dst,
                                             const AscendC::LocalTensor<SrcT>& src, uint64_t quantScal,
                                             AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR>& params)
    {
        if constexpr (AscendC::IsSameType<SrcT, int32_t>::value && AscendC::IsSameType<DstT, half>::value) {
            params.quantPre = QuantMode_t::DEQF16;
        } else if constexpr ((AscendC::IsSameType<DstT, int8_t>::value || AscendC::IsSameType<DstT, uint8_t>::value) &&
                             AscendC::IsSameType<SrcT, int32_t>::value) {
            params.quantPre = QuantMode_t::REQ8;
        } else if constexpr (AscendC::IsSameType<SrcT, float>::value &&
                             (AscendC::IsSameType<DstT, int8_t>::value || AscendC::IsSameType<DstT, uint8_t>::value)) {
            params.quantPre = QuantMode_t::QF322B8_PRE;
        }
        params.deqScalar = quantScal;
        AscendC::Fixpipe<DstT, SrcT, AscendC::CFG_ROW_MAJOR>(dst, src, params);
    }
#endif
};

/**
 * @struct Copy
 * @brief Copy struct for Ascend910_95 architecture with specific parameters
 *
 * This struct is only valid when the output is in UB, the format is either NZ, and
 * the IsQuantSenario condition is satisfied
 *
 * @param [in] OutputType: the type of the output tensor
 * @param [in] InputType: the type of the input tensor
 */
template <class OutputType, class InputType>
struct Copy<
    Arch::Ascend910_95, CopyWithParams, void, OutputType, InputType,
    AscendC::Std::enable_if_t<
        PosIsUB<OutputType::pos>() && IsNz<OutputType>() &&             // UB NZ
        IsQuantSenario<typename OutputType::T, typename InputType::T>() // quant
    >
> {
public:
    using DstT = typename OutputType::T;
    using SrcT = typename AscendC::GetMmDstType<typename InputType::T>::Type;

    /**
     * @brief Overloaded operator() for performing data copy operations
     * @param [in] dst: destination local tensor
     * @param [in] src: source local tensor
     * @param [in] curRow: current row index
     * @param [in] curCol: current column index
     * @param [in] l0CTileHeight: L0C tile height
     * @param [in] l0CTileWidth: L0C tile width
     * @param [in] baseM: base M value
     * @param [in] baseN: base N value
     * @param [in] orgM: original M value
     * @param [in] orgN: original N value
     * @param [in] orgKc: original Kc value
     * @param [in] quantTensor: quantization tensor
     */
    __aicore__ inline void operator()(const AscendC::LocalTensor<DstT>& dst, const AscendC::LocalTensor<SrcT>& src,
        int32_t curRow, int32_t curCol, int32_t l0CTileHeight, int32_t l0CTileWidth, int32_t baseM, int32_t baseN,
        int32_t orgM, int32_t orgN, int32_t orgKc, const AscendC::LocalTensor<uint64_t>& quantTensor)
    {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3101)
        CopyOutNZ2NZ(dst, src, curRow, curCol, l0CTileHeight, l0CTileWidth, baseM, baseN, orgM, quantTensor);
#else
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Only support Ascend910_95"); });
#endif
    }

    /**
     * @brief Overloaded operator() for performing data copy operations
     * @param [in] dst: destination local tensor
     * @param [in] src: source local tensor
     * @param [in] curRow: current row index
     * @param [in] curCol: current column index
     * @param [in] l0CTileHeight: L0C tile height
     * @param [in] l0CTileWidth: L0C tile width
     * @param [in] baseM: base M value
     * @param [in] baseN: base N value
     * @param [in] orgM: original M value
     * @param [in] orgN: original N value
     * @param [in] orgKc: original Kc value
     * @param [in] quantScalar: quantization scalar tensor
     */
    __aicore__ inline void operator()(const AscendC::LocalTensor<DstT>& dst, const AscendC::LocalTensor<SrcT>& src,
        int32_t curRow, int32_t curCol, int32_t l0CTileHeight, int32_t l0CTileWidth, int32_t baseM, int32_t baseN,
        int32_t orgM, int32_t orgN, int32_t orgKc, uint64_t quantScalar)
    {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3101)
        CopyOutNZ2NZ(dst, src, curRow, curCol, l0CTileHeight, l0CTileWidth, baseM, baseN, orgM, quantScalar);
#else
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Only support Ascend910_95"); });
#endif
    }

private:
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3101)
    /**
     * @brief Copy data from a local tensor to a global tensor, supporting both quantized and non-quantized modes
     * @param [in] T: quantization object type
     * @param [in] dst: destination global tensor
     * @param [in] src: source local tensor
     * @param [in] curRow: current row index
     * @param [in] curCol: current column index
     * @param [in] l0CTileHeight: L0C tile height
     * @param [in] l0CTileWidth: L0C tile width
     * @param [in] baseM: base M value
     * @param [in] baseN: base N value
     * @param [in] orgN: original N value
     * @param [in] quantObject: the quantization object
     */
    template <typename T>
    __aicore__ inline static void CopyOutNZ2NZ(const AscendC::LocalTensor<DstT>& dst,
                                               const AscendC::LocalTensor<SrcT>& src,
                                               int32_t curRow, int32_t curCol,
                                               int32_t l0CTileHeight, int32_t l0CTileWidth,
                                               int32_t baseM, int32_t baseN, int32_t orgM, T quantObject)
    {
        uint32_t stride =
            static_cast<uint32_t>((orgM - l0CTileHeight) * AscendC::BLOCK_CUBE * sizeof(DstT) / AscendC::ONE_BLK_SIZE) +
            static_cast<uint32_t>(l0CTileHeight * AscendC::BLOCK_CUBE * sizeof(SrcT) / AscendC::ONE_BLK_SIZE) *
                sizeof(DstT) / sizeof(SrcT);

        int32_t baseBlockWidth = AscendC::Ceil(l0CTileWidth, AscendC::BLOCK_CUBE);
        int64_t dstOffset =
            static_cast<int64_t>(curCol * baseN) * orgM + static_cast<int64_t>(curRow * baseM) * AscendC::BLOCK_CUBE;

        AscendC::FixpipeParamsC310<AscendC::CO2Layout::NZ> params;
        params.nSize = static_cast<uint16_t>(baseBlockWidth * AscendC::BLOCK_CUBE);
        params.mSize = static_cast<uint16_t>(l0CTileHeight);
        params.dstStride = stride;
        params.srcStride = AscendC::CeilAlign(l0CTileHeight, AscendC::BLOCK_CUBE);
        if constexpr (AscendC::IsSameType<DstT, half>::value && AscendC::IsSameType<SrcT, float>::value) {
            params.quantPre = QuantMode_t::F322F16;
        } else if constexpr (AscendC::IsSameType<DstT, bfloat16_t>::value && AscendC::IsSameType<SrcT, float>::value) {
            params.quantPre = QuantMode_t::F322BF16;
        }
        CopyTensor(dst[dstOffset], src, quantObject, params);
    }

    /**
     * @brief Copy data from a local tensor to a global tensor, supporting both quantized and non-quantized modes
     * @param [in] dst: destination global tensor
     * @param [in] src: source local tensor
     * @param [in] quantTensor: quantization tensor
     * @param [in] params: the fixpipe parameters
     */
    __aicore__ inline static void CopyTensor(const AscendC::LocalTensor<DstT>& dst,
                                             const AscendC::LocalTensor<SrcT>& src,
                                             const AscendC::LocalTensor<uint64_t>& quantTensor,
                                             AscendC::FixpipeParamsC310<AscendC::CO2Layout::NZ>& params)
    {
        if constexpr (AscendC::IsSameType<SrcT, int32_t>::value && AscendC::IsSameType<DstT, half>::value) {
            params.quantPre = QuantMode_t::VDEQF16;
        } else if constexpr (AscendC::IsSameType<SrcT, int32_t>::value &&
                             (AscendC::IsSameType<DstT, int8_t>::value || AscendC::IsSameType<DstT, uint8_t>::value)) {
            params.quantPre = QuantMode_t::VREQ8;
        } else if constexpr ((AscendC::IsSameType<DstT, int8_t>::value || AscendC::IsSameType<DstT, uint8_t>::value) &&
                             AscendC::IsSameType<SrcT, float>::value) {
            params.quantPre = QuantMode_t::VQF322B8_PRE;
        }
        AscendC::Fixpipe<DstT, SrcT, AscendC::CFG_NZ>(dst, src, quantTensor, params);
    }

    /**
     * @brief Copy data from a local tensor to a global tensor, supporting both quantized and non-quantized modes
     * @param [in] dst: destination global tensor
     * @param [in] src: source local tensor
     * @param [in] quantScalar: quantization scalar
     * @param [in] params: the fixpipe parameters
     */
    __aicore__ inline static void CopyTensor(const AscendC::LocalTensor<DstT>& dst,
                                             const AscendC::LocalTensor<SrcT>& src, uint64_t quantScal,
                                             AscendC::FixpipeParamsC310<AscendC::CO2Layout::NZ>& params)
    {
        if constexpr (AscendC::IsSameType<SrcT, int32_t>::value && AscendC::IsSameType<DstT, half>::value) {
            params.quantPre = QuantMode_t::DEQF16;
        } else if constexpr ((AscendC::IsSameType<DstT, int8_t>::value || AscendC::IsSameType<DstT, uint8_t>::value) &&
                             AscendC::IsSameType<SrcT, int32_t>::value) {
            params.quantPre = QuantMode_t::REQ8;
        } else if constexpr ((AscendC::IsSameType<DstT, int8_t>::value || AscendC::IsSameType<DstT, uint8_t>::value) &&
                             AscendC::IsSameType<SrcT, float>::value) {
            params.quantPre = QuantMode_t::QF322B8_PRE;
        }
        params.deqScalar = quantScal;
        AscendC::Fixpipe<DstT, SrcT, AscendC::CFG_NZ>(dst, src, params);
    }
#endif
};
} // namespace Tile
} // namespace Gemm
} // namespace Act
#endif
