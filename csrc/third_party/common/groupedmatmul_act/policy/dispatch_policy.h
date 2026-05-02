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
 * \file dispatch_policy.h
 * \brief
 */
#ifndef MATMUL_POLICY_DISPATCH_POLICY_H
#define MATMUL_POLICY_DISPATCH_POLICY_H

#include "../utils/integral_constant.h"
#include "../utils/arch.h"

namespace Act {
namespace Gemm {
/* block schedule policies */
struct KernelNaivePipeline {};     // Basic pipelining without caching or optimization
struct KernelMultiBlock {};        // Multi-block pipelined data transfer
struct KernelMultiBlockOnKAxis {}; // Multi-tile pipelined transfer with K-axis caching
struct KernelMmadPerBaseK {};      // Perform matrix multiplication with baseK granularity
struct KernelL1Input {};           // L1 input pipeline
struct KernelIterBatch {};         // Multi-tile pipelined transfer with batch caching
struct KernelMmadWithScale {};     // Multi-block with scale
struct KernelMultiBlockStreamK {}; // Multi-tile transfer with K-axis spliting and caching
struct KernelBatchMatMulToMul {};  // BatchMatmul to mul
struct KernelMixWithWeightPrologue {};

enum class MatMulL0C2Out : std::uint8_t {
    ON_THE_FLY = 0,
    ND_FIXPIPE_1_1 = 1,
    ND_FIXPIPE_1_2 = 2
};

/**
 * @struct GMMPerTile
 * @brief Define a template struct GMMPerTile for configuring block matrix multiplication policies
 * @param [in] SingleCoreShape: the shape of a single core, default is AscendC::Shape<_0, _0, _0, _0>
 */
template <class SingleCoreShape = AscendC::Shape<_0, _0, _0, _0>>
struct GMMPerTile {
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
 * @struct QuantMatmulWithTileMultiBlock
 * @brief Define a template struct QuantMatmulWithTileMultiBlock for multi-block matrix multiplication policies
 * @param [in] SingleCoreShape: the shape of a single core, default is AscendC::Shape<_0, _0, _0, _0>
 */
template <class SingleCoreShape = AscendC::Shape<_0, _0, _0, _0>>
struct QuantMatmulWithTileMultiBlock {
    using ScheduleType = KernelMmadWithScale;
    using SingleShape = SingleCoreShape;
    constexpr static bool enableInputDataLenCheck = false;
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
 */
template <class SingleCoreShape = AscendC::Shape<_0, _0, _0, _0>, uint64_t FULL_LOAD_MODE_ = 0,
    bool ENABLE_RELU_ = false>
struct MatmulMultiBlockWithOutQue {
    using ScheduleType = KernelMultiBlockOnKAxis;
    using SingleShape = SingleCoreShape;
    constexpr static uint64_t fullLoadMode = FULL_LOAD_MODE_;
    constexpr static bool enableRelu = ENABLE_RELU_;
};

/**
 * @struct MatmulMultiBlockWithStreamK
 * @brief Matrix multiplication split k axis processing structure, no quant, no bias, implemented base on layout
 * @param [in] fixpOpti: enum, judge if enabling fixp align optimize, default is ON_THE_FLY
 * @param [in] ENABLE_RELU: execute relu after mmad , default is false
 * @param [in] SingleCoreShape: the shape of a single core, default is AscendC::Shape<_0, _0, _0, _0>
 */
template <MatMulL0C2Out fixpOpti = MatMulL0C2Out::ON_THE_FLY, bool ENABLE_RELU = false,
    class SingleCoreShape = AscendC::Shape<_0, _0, _0, _0>>
struct MatmulMultiBlockWithStreamK {
    using ScheduleType = KernelMultiBlockStreamK;
    using SingleShape = SingleCoreShape;
    constexpr static bool enableInputDataLenCheck = false;
    constexpr static bool enableRelu = ENABLE_RELU;
    constexpr static MatMulL0C2Out fixpOpti_ = fixpOpti;
};

template <class SingleCoreShape = AscendC::Shape<_0, _0, _0, _0>>
struct BatchMatmulToMul {
    using ScheduleType = KernelBatchMatMulToMul;
    using SingleShape = SingleCoreShape;
};

/**
 * @struct MatmulIterBatch
 * @brief Matrix multiplication iteration batch processing structure, no quant, no bias, implemented base on layout
 * @param [in] SingleCoreShape: the shape of a single core, default is AscendC::Shape<_0, _0, _0, _0>
 */
template <class SingleCoreShape = AscendC::Shape<_0, _0, _0, _0>>
struct MatmulIterBatch {
    using ScheduleType = KernelIterBatch;
    using SingleShape = SingleCoreShape;
    constexpr static bool ENABLE_INTRINSICS_CHECK = false;
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
 * @brief Sparse matrix multiplication multi-block structure, with K-axis caching, no quant, no bias, implemented based on Layout
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
    using ArchTag = Act::Gemm::Arch::Ascend910_95;
};
} // namespace Gemm
} // namespace Act
#endif
