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
 * \file dispatch_policy.h
 * \brief
 */
#pragma once

#include "../utils/arch.h"
#include "../utils/common_utils.h"
#include "../utils/integral_constant.h"

namespace Cmct {
namespace Gemm {
/* block schedule policies */
struct KernelNaivePipeline {};     // Basic pipelining without caching or optimization
struct KernelMultiBlock {};        // Multi-block pipelined data transfer
struct KernelMultiBlockOnKAxis {}; // Multi-tile pipelined transfer with K-axis caching
struct KernelMmadPerBaseK {};      // Perform matrix multiplication with baseK granularity
struct KernelL1Input {};           // L1 input pipeline
struct KernelIterBatch {};         // Multi-tile pipelined transfer with batch caching
struct KernelMergeBatch {};        // Multi-tile pipelined transfer with batch caching
struct KernelMmadWithScale {};     // Multi-block with scale
struct KernelMultiBlockStreamK {}; // Multi-tile transfer with K-axis splitting and caching
struct KernelBatchMatMulToMul {};  // BatchMatmul to mul
struct KernelMixWithWeightPrologue {};
struct KernelMatMulToMul {};
struct KernelFlatQuant {};
struct KernelRotateQuant {};
struct KernelMatmulToVector {};

enum class MatMulL0C2Out : std::uint8_t { ON_THE_FLY = 0, ND_FIXPIPE_1_1 = 1, ND_FIXPIPE_1_2 = 2 };

/**
 * @struct MmadCAccOnUb
 * @brief Matrix multiplication multi-block structure, performing accumulation operations on UB
 * @param [in] SingleCoreShape: the shape of a single core, default is AscendC::Shape<_0, _0, _0, _0>
 */
template <class SingleCoreShape = AscendC::Shape<_0, _0, _0, _0>>
struct MmadCAccOnUb {
    using ScheduleType = KernelMmadPerBaseK;
    using SingleShape = SingleCoreShape;
    constexpr static bool ENABLE_INTRINSICS_CHECK = false;
};

/**
 * @struct QuantMatmulMultiBlock
 * @brief Define a template struct QuantMatmulMultiBlock for multi-block matrix multiplication policies
 * @param [in] SingleCoreShape: the shape of a single core, default is AscendC::Shape<_0, _0, _0, _0>
 */
template <class SingleCoreShape = AscendC::Shape<_0, _0, _0, _0>>
struct QuantMatmulMultiBlock {
    using ScheduleType = KernelMmadWithScale;
    using SingleShape = SingleCoreShape;
    constexpr static bool ENABLE_INTRINSICS_CHECK = false;
    constexpr static bool ENABLE_SET_BIAS = false;
    constexpr static MatmulConfigMode CONFIG = MatmulConfigMode::CONFIG_MDL;
};

/**
 * @struct MatmulNaivePipelineWithLayout
 * @brief Structure for a naive matrix multiplication pipeline with layout
 * @param [in] SingleCoreShape: the shape of a single core, default is AscendC::Shape<_0, _0, _0, _0>
 */
template <class SingleCoreShape = AscendC::Shape<_0, _0, _0, _0>>
struct MatmulNaivePipelineWithLayout {
    using ScheduleType = KernelNaivePipeline;
    using SingleShape = SingleCoreShape;

    constexpr static int l0CStages = 1;
    constexpr static bool enalbeL0CUnitFlag = false;
};

/**
 * @struct MatmulWithScale
 * @brief Matrix multiplication with scaleA and scaleB
 * @param [in] SingleCoreShape: the shape of a single core, default is AscendC::Shape<_0, _0, _0, _0>
 */
template <class SingleCoreShape = AscendC::Shape<_0, _0, _0, _0>, uint64_t FULL_LOAD_MODE_ = 0>
struct MatmulWithScale {
    using ScheduleType = KernelMmadWithScale;
    using SingleShape = SingleCoreShape;
    constexpr static uint64_t fullLoadMode = FULL_LOAD_MODE_;
};

/**
 * @struct MatmulMultiBlockWithLayout
 * @brief Matrix multiplication multi-block layout structure, no bias, no quant
 * @param [in] SingleCoreShape: the shape of a single core, default is AscendC::Shape<_0, _0, _0, _0>
 */
template <class SingleCoreShape = AscendC::Shape<_0, _0, _0, _0>>
struct MatmulMultiBlockWithLayout {
    using ScheduleType = KernelMultiBlock;
    using SingleShape = SingleCoreShape;
    constexpr static bool ENABLE_INTRINSICS_CHECK = false;
    constexpr static bool ENABLE_SET_BIAS = false;
    constexpr static MatmulConfigMode CONFIG = MatmulConfigMode::CONFIG_MDL;
};

/**
 * @struct MatmulMultiBlockBiasWithLayout
 * @brief Matrix multiplication multi-block layout structure, support bias, no quant
 * @param [in] SingleCoreShape: the shape of a single core, default is AscendC::Shape<_0, _0, _0, _0>
 */
template <class SingleCoreShape = AscendC::Shape<_0, _0, _0, _0>>
struct MatmulMultiBlockBiasWithLayout {
    using ScheduleType = KernelMultiBlock;
    using SingleShape = SingleCoreShape;
    constexpr static bool ENABLE_INTRINSICS_CHECK = false;
    constexpr static bool ENABLE_SET_BIAS = true;
    constexpr static MatmulConfigMode CONFIG = MatmulConfigMode::CONFIG_MDL;
};

/**
 * @struct MatmulMultiBlockBiasWithLayout
 * @brief Matrix multiplication multi-block structure, no bias, no quant, implemented based on highlevel api
 * @param [in] SingleCoreShape: the shape of a single core, default is AscendC::Shape<_0, _0, _0, _0>
 */
template <class SingleCoreShape = AscendC::Shape<_0, _0, _0, _0>>
struct MatmulMultiBlock {
    using ScheduleType = KernelMultiBlock;
    using SingleShape = SingleCoreShape;
    constexpr static bool ENABLE_INTRINSICS_CHECK = false;
    constexpr static bool ENABLE_SET_BIAS = false;
    constexpr static MatmulConfigMode CONFIG = MatmulConfigMode::CONFIG_MDL;
};

/**
 * @struct MatmulMultiBlockWithOutQue
 * @brief Matrix multiplication multi-block structure, no quant, implemented based on Layout
 * @param [in] SingleCoreShape: the shape of a single core, default is AscendC::Shape<_0, _0, _0, _0>
 * @param [in] FULL_LOAD_MODE: mode of full load, default is 0(no full load)
 * @param [in] ENABLE_RELU: execute relu after mmad , default is false
 * @param [in] INNER_PRECISE_: 0 = high-precision path. Default 1 keeps original precision.
 */
template <class SingleCoreShape = AscendC::Shape<_0, _0, _0, _0>, uint64_t FULL_LOAD_MODE_ = 0,
          uint64_t FUSED_OP_TYPE_ = 0, int8_t INNER_PRECISE_ = 1>
struct MatmulMultiBlockWithOutQue {
    using ScheduleType = KernelMultiBlockOnKAxis;
    using SingleShape = SingleCoreShape;
    constexpr static uint64_t fullLoadMode = FULL_LOAD_MODE_;
    constexpr static bool enableRelu = (FUSED_OP_TYPE_ == OP_TYPE_RELU || FUSED_OP_TYPE_ == OP_TYPE_RELU_QUANT);
    constexpr static bool enableQuant = (FUSED_OP_TYPE_ == OP_TYPE_QUANT || FUSED_OP_TYPE_ == OP_TYPE_RELU_QUANT);
    constexpr static bool enableAdd = (FUSED_OP_TYPE_ == OP_TYPE_ADD);
    constexpr static bool enableMul = (FUSED_OP_TYPE_ == OP_TYPE_MUL);
    constexpr static bool enableGelu = (FUSED_OP_TYPE_ == OP_TYPE_GELU_ERF || FUSED_OP_TYPE_ == OP_TYPE_GELU_TANH);
    constexpr static bool enableSilu = (FUSED_OP_TYPE_ == OP_TYPE_SILU);
    constexpr static bool enableSwigluSingle = (FUSED_OP_TYPE_ == OP_TYPE_SWIGLU_SINGLE);
    constexpr static bool enableHighPrecision = false;
};

template <class SingleCoreShape, uint64_t FULL_LOAD_MODE_, uint64_t FUSED_OP_TYPE_>
struct MatmulMultiBlockWithOutQue<SingleCoreShape, FULL_LOAD_MODE_, FUSED_OP_TYPE_, 0>
    : MatmulMultiBlockWithOutQue<SingleCoreShape, FULL_LOAD_MODE_, FUSED_OP_TYPE_, 1> {
    constexpr static bool enableHighPrecision = true;
};

/**
 * @struct MatmulMultiBlockWithStreamK
 * @brief Matrix multiplication split k axis processing structure, no quant, no bias, implemented base on layout
 * @param [in] FixpOpti: enum, judge if enabling fixp align optimize, default is ON_THE_FLY
 * @param [in] ENABLE_RELU: execute relu after mmad , default is false
 * @param [in] SingleCoreShape: the shape of a single core, default is AscendC::Shape<_0, _0, _0, _0>
 */
template <MatMulL0C2Out FixpOpti = MatMulL0C2Out::ON_THE_FLY, uint64_t FUSED_OP_TYPE_ = 0,
          class SingleCoreShape = AscendC::Shape<_0, _0, _0, _0>>
struct MatmulMultiBlockWithStreamK {
    using ScheduleType = KernelMultiBlockStreamK;
    using SingleShape = SingleCoreShape;
    constexpr static bool enableInputDataLenCheck = false;
    constexpr static bool enableRelu = (FUSED_OP_TYPE_ == OP_TYPE_RELU);
    constexpr static bool enableAdd = (FUSED_OP_TYPE_ == OP_TYPE_ADD);
    constexpr static bool enableMul = (FUSED_OP_TYPE_ == OP_TYPE_MUL);
    constexpr static MatMulL0C2Out fixpOpti_ = FixpOpti;
};

template <class SingleCoreShape = AscendC::Shape<_0, _0, _0, _0>>
struct BatchMatmulToMul {
    using ScheduleType = KernelBatchMatMulToMul;
    using SingleShape = SingleCoreShape;
};

/**
 * @struct MatmulIterBatch
 * @brief Matrix multiplication iteration batch processing structure, no quant, no bias, implemented base on layout
 * @param [in] EnableSync: judge if enabling synchronization between AIV and AIC
 * @param [in] SingleCoreShape: the shape of a single core, default is AscendC::Shape<_0, _0, _0, _0>
 * @param [in] FUSED_OPTYPE_: execute OP_TYPE_ADD, OP_TYPE_MUL, OP_TYPE_RELU after mmad , default is OP_TYPE_EMPTY
 */
template <MatMulL0C2Out EnableSync = MatMulL0C2Out::ON_THE_FLY, class SingleCoreShape = AscendC::Shape<_0, _0, _0, _0>,
          uint64_t FUSED_OPTYPE_ = 0>
struct MatmulIterBatch {
    using ScheduleType = KernelIterBatch;
    using SingleShape = SingleCoreShape;
    constexpr static bool ENABLE_INTRINSICS_CHECK = false;
    constexpr static MatMulL0C2Out enableSync = EnableSync;
    constexpr static bool enableRelu = (FUSED_OPTYPE_ == OP_TYPE_RELU || FUSED_OPTYPE_ == OP_TYPE_RELU_QUANT);
    constexpr static bool enableQuant = (FUSED_OPTYPE_ == OP_TYPE_QUANT || FUSED_OPTYPE_ == OP_TYPE_RELU_QUANT);
};

template <class SingleCoreShape = AscendC::Shape<_0, _0, _0, _0>, uint64_t FUSED_OPTYPE_ = OP_TYPE_EMPTY>
struct MatmulMergeBatch {
    using ScheduleType = KernelMergeBatch;
    using SingleShape = SingleCoreShape;
    constexpr static bool enableInputDataLenCheck = false;
    constexpr static bool enableRelu = (FUSED_OPTYPE_ == OP_TYPE_RELU);
    constexpr static bool enableAdd = (FUSED_OPTYPE_ == OP_TYPE_ADD);
    constexpr static bool enableMul = (FUSED_OPTYPE_ == OP_TYPE_MUL);
};

template <class SingleCoreShape = AscendC::Shape<_0, _0, _0, _0>>
struct MatmulFlatQuant {
    using ScheduleType = KernelFlatQuant;
    using SingleShape = SingleCoreShape;
    constexpr static bool enableInputDataLenCheck = false;
};

template <class SingleCoreShape = AscendC::Shape<_0, _0, _0, _0>>
struct MatmulRotateQuant {
    using ScheduleType = KernelRotateQuant;
    using SingleShape = SingleCoreShape;
    constexpr static bool enableInputDataLenCheck = false;
};

/**
 * @struct MatmulMultiBlockBias
 * @brief Matrix multiplication multi-block structure, support bias, no quant, implemented based on highlevel api
 * @param [in] SingleCoreShape: the shape of a single core, default is AscendC::Shape<_0, _0, _0, _0>
 */
template <class SingleCoreShape = AscendC::Shape<_0, _0, _0, _0>>
struct MatmulMultiBlockBias {
    using ScheduleType = KernelMultiBlock;
    using SingleShape = SingleCoreShape;
    constexpr static bool ENABLE_INTRINSICS_CHECK = false;
    constexpr static bool ENABLE_SET_BIAS = true;
    constexpr static MatmulConfigMode CONFIG = MatmulConfigMode::CONFIG_MDL;
};

/**
 * @struct MatmulMultiBlockOnKAxisWithLayout
 * @brief Matrix multiplication multi-block structure, with K-axis caching, no quant, implemented based on Layout
 * @param [in] SingleCoreShape: the shape of a single core, default is AscendC::Shape<_0, _0, _0, _0>
 */
template <class SingleCoreShape = AscendC::Shape<_0, _0, _0, _0>>
struct MatmulMultiBlockOnKAxisWithLayout {
    using ScheduleType = KernelMultiBlockOnKAxis;
    using SingleShape = SingleCoreShape;
    constexpr static bool ENABLE_INTRINSICS_CHECK = false;
};

/**
 * @struct SparseMatmulMultiBlockOnKAxisWithLayout
 * @brief Sparse matrix multiplication multi-block structure, with K-axis caching, no quant, no bias, implemented based
 * on Layout
 * @param [in] SingleCoreShape: the shape of a single core, default is AscendC::Shape<_0, _0, _0, _0>
 */
template <class SingleCoreShape = AscendC::Shape<_0, _0, _0, _0>>
struct SparseMatmulMultiBlockOnKAxisWithLayout {
    using ScheduleType = KernelMultiBlockOnKAxis;
    using SingleShape = SingleCoreShape;
};

/**
 * @struct MatmulL1Input
 * @brief Matrix multiplication L1 input structure, no bias, implemented based on highlevel api
 * @param [in] SingleCoreShape: the shape of a single core, default is AscendC::Shape<_0, _0, _0, _0>
 */
template <class SingleCoreShape = AscendC::Shape<_0, _0, _0, _0>>
struct MatmulL1Input {
    using ScheduleType = KernelL1Input;
    using SingleShape = SingleCoreShape;
    constexpr static bool ENABLE_INTRINSICS_CHECK = false;
    constexpr static bool ENABLE_SET_BIAS = false;
    constexpr static MatmulConfigMode CONFIG = MatmulConfigMode::CONFIG_MDL;
};

/**
 * @struct MatmulL1InputBias
 * @brief Matrix multiplication L1 input bias structure, implemented based on highlevel api
 * @param [in] SingleCoreShape: the shape of a single core, default is AscendC::Shape<_0, _0, _0, _0>
 */
template <class SingleCoreShape = AscendC::Shape<_0, _0, _0, _0>>
struct MatmulL1InputBias {
    using ScheduleType = KernelL1Input;
    using SingleShape = SingleCoreShape;
    constexpr static bool ENABLE_INTRINSICS_CHECK = false;
    constexpr static bool ENABLE_SET_BIAS = true;
    constexpr static MatmulConfigMode CONFIG = MatmulConfigMode::CONFIG_MDL;
};

/**
 * @struct MatmulL1InputWithLayout
 * @brief Matrix multiplication L1 input structure, no bias, implemented based on layout
 * @param [in] SingleCoreShape: the shape of a single core, default is AscendC::Shape<_0, _0, _0, _0>
 */
template <class SingleCoreShape = AscendC::Shape<_0, _0, _0, _0>>
struct MatmulL1InputWithLayout {
    using ScheduleType = KernelL1Input;
    using SingleShape = SingleCoreShape;
    constexpr static bool ENABLE_INTRINSICS_CHECK = false;
    constexpr static bool ENABLE_SET_BIAS = false;
    constexpr static MatmulConfigMode CONFIG = MatmulConfigMode::CONFIG_MDL;
};

/**
 * @struct MatmulL1InputBiasWithLayout
 * @brief Matrix multiplication L1 input bias structure, implemented based on layout
 * @param [in] SingleCoreShape: the shape of a single core, default is AscendC::Shape<_0, _0, _0, _0>
 */
template <class SingleCoreShape = AscendC::Shape<_0, _0, _0, _0>>
struct MatmulL1InputBiasWithLayout {
    using ScheduleType = KernelL1Input;
    using SingleShape = SingleCoreShape;
    constexpr static bool ENABLE_INTRINSICS_CHECK = false;
    constexpr static bool ENABLE_SET_BIAS = true;
    constexpr static MatmulConfigMode CONFIG = MatmulConfigMode::CONFIG_MDL;
};

/**
 * @struct MatmulL0COutputWithLayout
 * @brief Matrix multiplication L0C output structure, implemented based on layout
 * @param [in] SingleCoreShape: the shape of a single core, default is AscendC::Shape<_0, _0, _0, _0>
 */
template <class SingleCoreShape = AscendC::Shape<_0, _0, _0, _0>>
struct MatmulL0COutputWithLayout {
    using ScheduleType = KernelMultiBlock;
    using SingleShape = SingleCoreShape;
    constexpr static bool ENABLE_INTRINSICS_CHECK = false;
    constexpr static bool ENABLE_SET_BIAS = false;
    constexpr static MatmulConfigMode CONFIG = MatmulConfigMode::CONFIG_MDL;
};

// Antiquant in ub with single communication single computation
struct UbAntiquantWithScSc {
    using ScheduleType = KernelMixWithWeightPrologue;
    using ArchTag = Cmct::Gemm::Arch::DAV3510;
};

template <uint64_t AivNum>
struct MmadAPrefetchBAntiquantScmc {
    constexpr static uint64_t AIV_NUM = AivNum;
};

/**
 * @struct MatmulToMul
 * @brief Matrix multiplication policy for converting a matmul operation to vector (when M=1 or N=1)
 * @param [in] SingleCoreShape: the shape of a single core, default is AscendC::Shape<_0, _0, _0, _0>
 */
template <class SingleCoreShape = AscendC::Shape<_0, _0, _0, _0>>
struct MatmulToMul {
    using ScheduleType = KernelMatMulToMul;
    using SingleShape = SingleCoreShape;
};

/**
 * @struct MatmulToVector
 * @brief Matrix multiplication policy for converting a matmul operation to vector (when BTrans)
 * @param [in] SingleCoreShape: the shape of a single core, default is AscendC::Shape<_0, _0, _0, _0>
 */
template <class SingleCoreShape = AscendC::Shape<_0, _0, _0, _0>>
struct MatmulToVector {
    using ScheduleType = KernelMatmulToVector;
    using SingleShape = SingleCoreShape;
};

} // namespace Gemm
} // namespace Cmct
