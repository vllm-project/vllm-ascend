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
 * \file copy_out_to_gm_with_params_v310.h
 * \brief
 */
#ifndef MATMUL_TILE_COPY_OUT_COPY_OUT_TO_GM_WITH_PARAMS_V310_H
#define MATMUL_TILE_COPY_OUT_COPY_OUT_TO_GM_WITH_PARAMS_V310_H

#include "../tile_copy_policy.h"
#include "../../utils/tensor_utils.h"

namespace Act {
namespace Gemm {
namespace Tile {
/**
 * @struct Copy
 * @brief Copy struct for Ascend910_95 architecture with specific parameters
 * @param [in] OutputType: the type of the output tensor
 * @param [in] InputType: the type of the input tensor
 */
template <class OutputType, class InputType>
struct Copy<
    Arch::Ascend910_95, CopyWithParams, void, OutputType, InputType,
    AscendC::Std::enable_if_t<
        PosIsGM<OutputType::pos>() && IsNDOrAlign<OutputType>() &&       // GM ND/ND_ALIGN
        !IsQuantSenario<typename OutputType::T, typename InputType::T>() // no quant
    >
> {
public:
    using DstT = typename OutputType::T;
    using SrcT = typename AscendC::GetMmDstType<typename InputType::T>::Type;

    /**
     * @brief Overloaded operator() for copying data from CO1 to GM
     * @param [in] gm: destination global tensor
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
     */
    __aicore__ inline void operator()(const AscendC::GlobalTensor<DstT>& gm, const AscendC::LocalTensor<SrcT>& src,
                                      int32_t curRow, int32_t curCol, int32_t l0CTileHeight, int32_t l0CTileWidth,
                                      int32_t baseM, int32_t baseN, int32_t orgM, int32_t orgN, int32_t orgKc)
    {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3101)
        uint32_t dimN = orgKc != 0 ? orgKc : orgN;
        constexpr uint32_t blockCount = AscendC::ONE_BLK_SIZE / sizeof(DstT);
        if constexpr (OutputType::format == CubeFormat::ND_ALIGN) {
            dimN = AscendC::CeilAlign(dimN, blockCount);
        }
        int64_t dstOffset =
            static_cast<int64_t>(static_cast<int64_t>(curRow * baseM) * dimN) + static_cast<int64_t>(curCol * baseN);

        AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> params;
        params.nSize = static_cast<uint16_t>(l0CTileWidth);
        if constexpr (OutputType::format == CubeFormat::ND_ALIGN) {
            params.nSize = AscendC::CeilAlign(params.nSize, blockCount);
            dimN = AscendC::CeilAlign(dimN, blockCount);
        }
        params.mSize = static_cast<uint16_t>(l0CTileHeight);
        params.srcStride = AscendC::CeilAlign(l0CTileHeight, AscendC::BLOCK_CUBE);
        params.dstStride = dimN;
        if constexpr (AscendC::IsSameType<DstT, half>::value && AscendC::IsSameType<SrcT, float>::value) {
            params.quantPre = QuantMode_t::F322F16;
        } else if constexpr (AscendC::IsSameType<DstT, bfloat16_t>::value && AscendC::IsSameType<SrcT, float>::value) {
            params.quantPre = QuantMode_t::F322BF16;
        }
        AscendC::Fixpipe<DstT, SrcT, AscendC::CFG_ROW_MAJOR>(gm[dstOffset], src, params);
#else
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Only support Ascend910_95"); });
#endif
    }
};

/**
 * @struct Copy
 * @brief Copy struct for Ascend910_95 architecture with specific parameters
 * @param [in] OutputType: the type of the output tensor
 * @param [in] InputType: the type of the input tensor
 */
template <class OutputType, class InputType>
struct Copy<Arch::Ascend910_95, CopyWithParams, void, OutputType, InputType,
    AscendC::Std::enable_if_t<
        PosIsGM<OutputType::pos>()  && IsNz<OutputType>() &&             // GM NZ
        !IsQuantSenario<typename OutputType::T, typename InputType::T>() // no quant
    >
> {
public:
    using DstT = typename OutputType::T;
    using SrcT = typename AscendC::GetMmDstType<typename InputType::T>::Type;

    /**
     * @brief Overloaded operator() for copying data from CO1 to GM
     * @param [in] gm: destination global tensor
     * @param [in] co1Local: source local tensor
     * @param [in] curRow: current row index
     * @param [in] curCol: current column index
     * @param [in] l0CTileHeight: L0C tile height
     * @param [in] l0CTileWidth: L0C tile width
     * @param [in] baseM: base M value
     * @param [in] baseN: base N value
     * @param [in] orgM: original M value
     * @param [in] orgN: original N value
     * @param [in] orgKc: original Kc value
     */
    __aicore__ inline void operator()(const AscendC::GlobalTensor<DstT>& gm, const AscendC::LocalTensor<SrcT>& co1Local,
                                      int32_t curRow, int32_t curCol, int32_t l0CTileHeight, int32_t l0CTileWidth,
                                      int32_t baseM, int32_t baseN, int32_t orgM, int32_t orgN, int32_t orgKc)
    {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3101)
        int64_t dstOffset =
            static_cast<int64_t>(curCol * baseN) * orgM + static_cast<int64_t>(curRow * baseM) * AscendC::BLOCK_CUBE;
        uint32_t stride =
            static_cast<uint32_t>((orgM - l0CTileHeight) * AscendC::BLOCK_CUBE * sizeof(DstT) / AscendC::ONE_BLK_SIZE);

        AscendC::FixpipeParamsC310<AscendC::CO2Layout::NZ> params;
        int32_t baseBlockWidth = AscendC::Ceil(l0CTileWidth, AscendC::BLOCK_CUBE);
        params.nSize = static_cast<uint16_t>(baseBlockWidth * AscendC::BLOCK_CUBE);
        stride = stride + static_cast<uint32_t>(l0CTileHeight * AscendC::BLOCK_CUBE * sizeof(SrcT) /
            AscendC::ONE_BLK_SIZE) * sizeof(DstT) / sizeof(SrcT);
        params.mSize = static_cast<uint16_t>(l0CTileHeight);
        params.srcStride = AscendC::CeilAlign(l0CTileHeight, AscendC::BLOCK_CUBE);
        params.dstStride = stride;
        if constexpr (AscendC::IsSameType<DstT, half>::value && AscendC::IsSameType<SrcT, float>::value) {
            params.quantPre = QuantMode_t::F322F16;
        } else if constexpr (AscendC::IsSameType<DstT, bfloat16_t>::value && AscendC::IsSameType<SrcT, float>::value) {
            params.quantPre = QuantMode_t::F322BF16;
        }
        AscendC::Fixpipe<DstT, SrcT, AscendC::CFG_NZ>(gm[dstOffset], co1Local, params);
#else
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Only support Ascend910_95"); });
#endif
    }
};

/**
 * @struct Copy
 * @brief Copy struct for Ascend910_95 architecture with specific parameters
 *
 * This struct is only valid when the output is in GM, the format is either ND or ND_ALIGN, and
 * the IsQuantSenario condition is satisfied
 *
 * @param [in] OutputType: the type of the output tensor
 * @param [in] InputType: the type of the input tensor
 */
template <class OutputType, class InputType>
struct Copy<
    Arch::Ascend910_95, CopyWithParams, void, OutputType, InputType,
    AscendC::Std::enable_if_t<
        PosIsGM<OutputType::pos>() && IsNDOrAlign<OutputType>() &&      // GM ND/ND_ALIGN
        IsQuantSenario<typename OutputType::T, typename InputType::T>() // quant
    >
> {
public:
    using DstT = typename OutputType::T;
    using SrcT = typename AscendC::GetMmDstType<typename InputType::T>::Type;

    /**
     * @brief Overloaded operator() for Ascend910_95 architecture
     * @param [in] dst: destination global tensor
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
    __aicore__ inline void operator()(const AscendC::GlobalTensor<DstT>& dst, const AscendC::LocalTensor<SrcT>& src,
                                      int32_t curRow, int32_t curCol, int32_t l0CTileHeight, int32_t l0CTileWidth,
                                      int32_t baseM, int32_t baseN, int32_t orgM, int32_t orgN, int32_t orgKc,
                                      const AscendC::LocalTensor<uint64_t>& quantTensor)
    {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3101)
        CopyOutNZ2ND(dst, src, curRow, curCol, l0CTileHeight, l0CTileWidth, baseM, baseN, orgKc != 0 ? orgKc : orgN,
                     quantTensor);
#else
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Only support Ascend910_95"); });
#endif
    }

    /**
     * @brief Overloaded operator() for Ascend910_95 architecture
     * @param [in] gm: destination global tensor
     * @param [in] co1Local: source local tensor
     * @param [in] curRow: current row index
     * @param [in] curCol: current column index
     * @param [in] l0CTileHeight: L0C tile height
     * @param [in] l0CTileWidth: L0C tile width
     * @param [in] baseM: base M value
     * @param [in] baseN: base N value
     * @param [in] orgM: original M value
     * @param [in] orgN: original N value
     * @param [in] orgKc: original Kc value
     * @param [in] quantScal: quantization scalar tensor
     */
    __aicore__ inline void operator()(const AscendC::GlobalTensor<DstT>& gm, const AscendC::LocalTensor<SrcT>& co1Local,
                                      int32_t curRow, int32_t curCol, int32_t l0CTileHeight, int32_t l0CTileWidth,
                                      int32_t baseM, int32_t baseN, int32_t orgM, int32_t orgN, int32_t orgKc,
                                      uint64_t quantScal)
    {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3101)
        CopyOutNZ2ND(gm, co1Local, curRow, curCol, l0CTileHeight, l0CTileWidth, baseM, baseN, orgKc != 0 ? orgKc : orgN,
                     quantScal);
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
     * @param [in] co1Local: source local tensor
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
    __aicore__ inline void CopyOutNZ2ND(const AscendC::GlobalTensor<DstT>& dst,
                                        const AscendC::LocalTensor<SrcT>& co1Local, int32_t curRow, int32_t curCol,
                                        int32_t l0CTileHeight, int32_t l0CTileWidth, int32_t baseM, int32_t baseN,
                                        int32_t orgN, T quantObject)
    {
        constexpr uint32_t blockCount = AscendC::ONE_BLK_SIZE / sizeof(DstT);
        auto stride = orgN;
        if constexpr (OutputType::format == CubeFormat::ND_ALIGN) {
            stride = AscendC::Ceil(stride, blockCount) * blockCount;
        }
        int64_t dstOffset =
            static_cast<int64_t>(static_cast<int64_t>(curRow * baseM) * stride) + static_cast<int64_t>(curCol * baseN);

        AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> params;
        params.mSize = static_cast<uint16_t>(l0CTileHeight);
        params.nSize = static_cast<uint16_t>(l0CTileWidth);
        params.dstStride = stride;
        params.srcStride = AscendC::CeilAlign(l0CTileHeight, AscendC::BLOCK_CUBE);
        CopyTensor(dst[dstOffset], co1Local, quantObject, params);
    }

    /**
     * @brief Copy a tensor, supporting both quantized and non-quantized modes
     * @param [in] dst: destination global tensor
     * @param [in] co1Local: source local tensor
     * @param [in] quantTensor: quantization tensor
     * @param [in] params: the fixpipe parameters
     */
    __aicore__ inline void CopyTensor(const AscendC::GlobalTensor<DstT>& dst,
                                      const AscendC::LocalTensor<SrcT>& co1Local,
                                      const AscendC::LocalTensor<uint64_t>& quantTensor,
                                      AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR>& params)
    {
        if constexpr (AscendC::IsSameTypeV<SrcT, int32_t> && AscendC::IsSameTypeV<DstT, bfloat16_t>) {
            params.quantPre = QuantMode_t::VQS322BF16_PRE;
        } else if constexpr (AscendC::IsSameTypeV<SrcT, float> && AscendC::IsSameTypeV<DstT, hifloat8_t>) {
            params.quantPre = QuantMode_t::VQF322HIF8_PRE;
        } else if constexpr (AscendC::IsSameTypeV<SrcT, float> &&
                             AscendC::IsTypeOneOfV<DstT, fp8_e4m3fn_t, fp8_e5m2_t>) {
            params.quantPre = QuantMode_t::VQF322FP8_PRE;
        } else if constexpr (AscendC::IsTypeOneOfV<SrcT, fp8_e4m3fn_t, fp8_e5m2_t, hifloat8_t> &&
                             AscendC::IsSameTypeV<DstT, bfloat16_t>) {
            params.quantPre = QuantMode_t::VQF322BF16_PRE;
        } else if constexpr (AscendC::IsTypeOneOfV<SrcT, fp8_e4m3fn_t, fp8_e5m2_t, hifloat8_t> &&
                             AscendC::IsSameTypeV<DstT, half>) {
            params.quantPre = QuantMode_t::VQF322F16_PRE;
        } else if (AscendC::IsTypeOneOfV<SrcT, fp8_e4m3fn_t, fp8_e5m2_t, hifloat8_t> &&
                   AscendC::IsSameTypeV<DstT, float>) {
            params.quantPre = QuantMode_t::VQF322F32_PRE;
        }
        AscendC::Fixpipe<DstT, SrcT, AscendC::CFG_ROW_MAJOR>(dst, co1Local, quantTensor, params);
    }

    /**
     * @brief Copy a tensor, supporting both quantized and non-quantized modes
     * @param [in] dst: destination global tensor
     * @param [in] co1Local: source local tensor
     * @param [in] quantScalar: quantization scalar
     * @param [in] params: the fixpipe parameters
     */
    __aicore__ inline static void CopyTensor(const AscendC::GlobalTensor<DstT>& dst,
                                             const AscendC::LocalTensor<SrcT>& co1Local, uint64_t quantScalar,
                                             AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR>& params)
    {
        if constexpr (AscendC::IsSameTypeV<SrcT, int32_t> && AscendC::IsSameTypeV<DstT, bfloat16_t>) {
            params.quantPre = QuantMode_t::QS322BF16_PRE;
        } else if constexpr (AscendC::IsTypeOneOfV<SrcT, fp8_e4m3fn_t, fp8_e5m2_t, hifloat8_t> &&
                             AscendC::IsSameTypeV<DstT, half>) {
            params.quantPre = QuantMode_t::QF322F16_PRE;
        } else if constexpr (AscendC::IsSameTypeV<SrcT, float> &&
                             AscendC::IsTypeOneOfV<DstT, fp8_e4m3fn_t, fp8_e5m2_t>) {
            params.quantPre = QuantMode_t::QF322FP8_PRE;
        } else if constexpr (AscendC::IsTypeOneOfV<SrcT, fp8_e4m3fn_t, fp8_e5m2_t, hifloat8_t> &&
                             AscendC::IsSameTypeV<DstT, bfloat16_t>) {
            params.quantPre = QuantMode_t::QF322BF16_PRE;
        } else if constexpr (AscendC::IsSameTypeV<SrcT, float> && AscendC::IsSameTypeV<DstT, hifloat8_t>) {
            params.quantPre = QuantMode_t::QF322HIF8_PRE;
        } else if (AscendC::IsTypeOneOfV<SrcT, fp8_e4m3fn_t, fp8_e5m2_t, hifloat8_t> &&
                   AscendC::IsSameTypeV<DstT, float>) {
            params.quantPre = QuantMode_t::QF322F32_PRE;
        }
        params.deqScalar = quantScalar;
        AscendC::Fixpipe<DstT, SrcT, AscendC::CFG_ROW_MAJOR>(dst, co1Local, params);
    }
#endif
};

/**
 * @struct Copy
 * @brief Copy struct for Ascend910_95 architecture with specific parameters
 *
 * This struct is only valid when the output is in GM, the format is NZ, and
 * the IsQuantSenario condition is satisfied
 *
 * @param [in] OutputType: the type of the output tensor
 * @param [in] InputType: the type of the input tensor
 */
template <class OutputType, class InputType>
struct Copy<
    Arch::Ascend910_95, CopyWithParams, void, OutputType, InputType,
    AscendC::Std::enable_if_t<
        PosIsGM<OutputType::pos>() && IsNz<OutputType>() &&             // GM NZ
        IsQuantSenario<typename OutputType::T, typename InputType::T>() // quant
    >
> {
public:
    using DstT = typename OutputType::T;
    using SrcT = typename AscendC::GetMmDstType<typename InputType::T>::Type;

    /**
     * @brief Overloaded operator() for performing data copy operations
     * @param [in] gm: destination global tensor
     * @param [in] co1Local: source local tensor
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
    __aicore__ inline void operator()(const AscendC::GlobalTensor<DstT>& gm, const AscendC::LocalTensor<SrcT>& co1Local,
                                      int32_t curRow, int32_t curCol, int32_t l0CTileHeight, int32_t l0CTileWidth,
                                      int32_t baseM, int32_t baseN, int32_t orgM, int32_t orgN, int32_t orgKc,
                                      const AscendC::LocalTensor<uint64_t>& quantTensor)
    {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3101)
        CopyOutNZ2NZ(gm, co1Local, curRow, curCol, l0CTileHeight, l0CTileWidth, baseM, baseN, orgM, quantTensor);
#else
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Only support Ascend910_95"); });
#endif
    }

    /**
     * @brief Overloaded operator() for performing data copy operations
     * @param [in] gm: destination global tensor
     * @param [in] co1Local: source local tensor
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
    __aicore__ inline void operator()(const AscendC::GlobalTensor<DstT>& gm, const AscendC::LocalTensor<SrcT>& co1Local,
                                      int32_t curRow, int32_t curCol, int32_t l0CTileHeight, int32_t l0CTileWidth,
                                      int32_t baseM, int32_t baseN, int32_t orgM, int32_t orgN, int32_t orgKc,
                                      uint64_t quantScalar)
    {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3101)
        CopyOutNZ2NZ(gm, co1Local, curRow, curCol, l0CTileHeight, l0CTileWidth, baseM, baseN, orgM, quantScalar);
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
     * @param [in] co1Local: source local tensor
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
    __aicore__ inline static void CopyOutNZ2NZ(const AscendC::GlobalTensor<DstT>& dst,
                                               const AscendC::LocalTensor<SrcT>& co1Local,
                                               int32_t curRow, int32_t curCol,
                                               int32_t l0CTileHeight, int32_t l0CTileWidth, int32_t baseM,
                                               int32_t baseN, int32_t orgM, T quantObject)
    {
        uint32_t stride =
            static_cast<uint32_t>((orgM - l0CTileHeight) * AscendC::BLOCK_CUBE * sizeof(DstT) / AscendC::ONE_BLK_SIZE) +
            static_cast<uint32_t>(l0CTileHeight * AscendC::BLOCK_CUBE * sizeof(SrcT) / AscendC::ONE_BLK_SIZE) *
                sizeof(DstT) / sizeof(SrcT);

        int32_t baseBlockWidth = AscendC::Ceil(l0CTileWidth, AscendC::BLOCK_CUBE);
        int64_t dstOffset =
            static_cast<int64_t>(curCol * baseN) * orgM + static_cast<int64_t>(curRow * baseM) * AscendC::BLOCK_CUBE;

        AscendC::FixpipeParamsC310<AscendC::CO2Layout::NZ> params;
        params.mSize = static_cast<uint16_t>(l0CTileHeight);
        params.nSize = static_cast<uint16_t>(baseBlockWidth * AscendC::BLOCK_CUBE);
        params.srcStride = AscendC::CeilAlign(l0CTileHeight, AscendC::BLOCK_CUBE);
        params.dstStride = stride;
        if constexpr (AscendC::IsSameType<DstT, half>::value && AscendC::IsSameType<SrcT, float>::value) {
            params.quantPre = QuantMode_t::F322F16;
        } else if constexpr (AscendC::IsSameType<DstT, bfloat16_t>::value && AscendC::IsSameType<SrcT, float>::value) {
            params.quantPre = QuantMode_t::F322BF16;
        }
        CopyTensor(dst[dstOffset], co1Local, quantObject, params);
    }

    /**
     * @brief Copy data from a local tensor to a global tensor with quantization
     * @param [in] dst: destination global tensor
     * @param [in] co1Local: source local tensor
     * @param [in] quantTensor: quantization tensor
     * @param [in] params: the fixpipe parameters
     */
    __aicore__ inline static void CopyTensor(const AscendC::GlobalTensor<DstT>& dst,
                                             const AscendC::LocalTensor<SrcT>& co1Local,
                                             const AscendC::LocalTensor<uint64_t>& quantTensor,
                                             AscendC::FixpipeParamsC310<AscendC::CO2Layout::NZ>& params)
    {
        if constexpr (AscendC::IsSameType<SrcT, int32_t>::value && AscendC::IsSameType<DstT, half>::value) {
            params.quantPre = QuantMode_t::VDEQF16;
        } else if constexpr (AscendC::IsSameType<SrcT, float>::value &&
                             (AscendC::IsSameType<DstT, int8_t>::value || AscendC::IsSameType<DstT, uint8_t>::value)) {
            params.quantPre = QuantMode_t::VQF322B8_PRE;
        } else if constexpr (AscendC::IsSameType<SrcT, int32_t>::value &&
                             (AscendC::IsSameType<DstT, int8_t>::value || AscendC::IsSameType<DstT, uint8_t>::value)) {
            params.quantPre = QuantMode_t::VREQ8;
        }
        AscendC::Fixpipe<DstT, SrcT, AscendC::CFG_NZ>(dst, co1Local, quantTensor, params);
    }

    /**
     * @brief Copydata from a local tensor to a global tensor with quantization
     * @param [in] dst: destination global tensor
     * @param [in] co1Local: source local tensor
     * @param [in] quantScalar: quantization scalar
     * @param [in] params: the fixpipe parameters
     */
    __aicore__ inline static void CopyTensor(const AscendC::GlobalTensor<DstT>& dst,
                                             const AscendC::LocalTensor<SrcT>& co1Local, uint64_t quantScalar,
                                             AscendC::FixpipeParamsC310<AscendC::CO2Layout::NZ>& params)
    {
        if constexpr (AscendC::IsSameType<SrcT, float>::value &&
                             (AscendC::IsSameType<DstT, int8_t>::value || AscendC::IsSameType<DstT, uint8_t>::value)) {
            params.quantPre = QuantMode_t::QF322B8_PRE;
        } else if constexpr (AscendC::IsSameType<SrcT, int32_t>::value && AscendC::IsSameType<DstT, half>::value) {
            params.quantPre = QuantMode_t::DEQF16;
        } else if constexpr (AscendC::IsSameType<SrcT, int32_t>::value &&
                             (AscendC::IsSameType<DstT, int8_t>::value || AscendC::IsSameType<DstT, uint8_t>::value)) {
            params.quantPre = QuantMode_t::REQ8;
        }
        params.deqScalar = quantScalar;
        AscendC::Fixpipe<DstT, SrcT, AscendC::CFG_NZ>(dst, co1Local, params);
    }
#endif
};
} // namespace Tile
} // namespace Gemm
} // namespace Act
#endif
