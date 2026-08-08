/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSED_BLOCK_EPILOGUE_ONLINE_SOFTMAX_LOW_PREC_HPP
#define FUSED_BLOCK_EPILOGUE_ONLINE_SOFTMAX_LOW_PREC_HPP

#include "../../../attn_infra/fused_base_defs.hpp"
#include "../../../attn_infra/arch/fused_cross_core_sync.hpp"
#include "../../../attn_infra/arch/fused_resource.hpp"
#include "../../../attn_infra/epilogue/fused_epilogue_dispatch_policy.hpp"
#include "../../../attn_infra/epilogue/tile_common/fused_epilogue_tile_copy.hpp"
#include "../../../attn_infra/fused_gemm_coord.hpp"
#include "../../../attn_infra/fused_matrix_coord.hpp"
#include "utils/std/algorithm.h"

namespace NpuArch::Epilogue::Block {

template <
    class OutputType_,
    class InputType_,
    class MaskType_,
    class SinkType_,
    class FullType_,
    LseMode LSE_MODE_,
    SinkMode SINK_MODE_,
    MaskMode MASK_MODE_>
class BlockEpilogue<
    EpilogueAtlasA2OnlineSoftmax<LSE_MODE_, SINK_MODE_, MASK_MODE_, half>,
    OutputType_,
    InputType_,
    MaskType_,
    SinkType_,
    FullType_>
{
public:
    using DispatchPolicy = EpilogueAtlasA2OnlineSoftmax<LSE_MODE_, SINK_MODE_, MASK_MODE_, half>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementInput = typename InputType_::Element;
    using ElementMask = typename MaskType_::Element;
    using ElementSink = typename SinkType_::Element;
    using ElementFull = typename FullType_::Element;
    using ElementOutput = typename OutputType_::Element;
    using LayoutOutput = typename OutputType_::Layout;
    using LayoutInput = typename InputType_::Layout;
    using LayoutMask = typename MaskType_::Layout;
    using LayoutFull = typename FullType_::Layout;
    
    static constexpr LseMode LSE_MODE = DispatchPolicy::LSE_MODE;
    static constexpr SinkMode SINK_MODE = DispatchPolicy::SINK_MODE;

    static constexpr uint32_t BLOCK_SIZE_IN_BYTE = 32;
    static constexpr uint32_t REPEAT_SIZE_IN_BYTE = 256;
    static constexpr uint32_t FLOAT_BLOCK_SIZE = 8;
    static constexpr uint32_t FLOAT_VECTOR_SIZE = 64;
    static constexpr uint32_t HALF_VECTOR_SIZE = 128;
    static constexpr uint32_t BLOCK_SIZE = 16;
    static constexpr uint32_t UB_UINT8_VECTOR_SIZE = 1024;
    static constexpr uint32_t UB_UINT8_BLOCK_SIZE = 16384;
    static constexpr uint32_t VECTOR_SIZE = 128;
    static constexpr uint32_t MAX_UB_S_ELEM_NUM = 16384;

    static constexpr uint32_t REDUCE_UB_SIZE = 1024;
    static constexpr uint32_t ROW_OPS_SPEC_MASK_32 = 32;
    static constexpr uint32_t ROW_OPS_SPEC_MASK_8 = 8;
    static constexpr uint32_t ROW_OPS_SPEC_MASK_4 = 4;
    static constexpr uint32_t ROW_OPS_SPEC_MASK_2 = 2;
    static constexpr uint32_t MAX_ROW_NUM_SUB_CORE = 256;
    static constexpr int64_t UB_FLOAT_LINE_SIZE = 64;

    static constexpr uint32_t SPLIT_COL_IDX_2 = 2;
    static constexpr uint32_t SPLIT_COL_IDX_3 = 3;
    __aicore__ inline
    BlockEpilogue() {}

    __aicore__ inline
    void init(Arch::Resource<ArchTag> &resource, float scaleValue_)
    {
        // Allocate UB space
        constexpr uint32_t LS_UB_TENSOR_OFFSET = 0;
        constexpr uint32_t COMPUTE_UB_TENSOR_OFFSET = 2 * UB_UINT8_BLOCK_SIZE;
        constexpr uint32_t LP_UB_TENSOR_OFFSET = 4 * UB_UINT8_BLOCK_SIZE;
        constexpr uint32_t MASK16_UB_TENSOR_OFFSET = 0;

        constexpr uint32_t TV_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE;
        constexpr uint32_t LM_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 8 * UB_UINT8_VECTOR_SIZE;

        constexpr uint32_t HM_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 9 * UB_UINT8_VECTOR_SIZE;
        constexpr uint32_t GM_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 10 * UB_UINT8_VECTOR_SIZE;
        constexpr uint32_t LL_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 11 * UB_UINT8_VECTOR_SIZE;
        constexpr uint32_t GL_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 12 * UB_UINT8_VECTOR_SIZE;
        constexpr uint32_t DM_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 13 * UB_UINT8_VECTOR_SIZE;

        constexpr uint32_t MASK_UB_TENSOR_OFFSET = 11 * UB_UINT8_BLOCK_SIZE;

        scaleValue = static_cast<half>(scaleValue_);
        lsUbTensor = resource.ubBuf.template GetBufferByByte<half>(LS_UB_TENSOR_OFFSET);
        computeUbTensor = resource.ubBuf.template GetBufferByByte<half>(COMPUTE_UB_TENSOR_OFFSET);
        lpUbTensor = resource.ubBuf.template GetBufferByByte<ElementOutput>(LP_UB_TENSOR_OFFSET);
        maskUbTensor = resource.ubBuf.template GetBufferByByte<ElementMask>(MASK_UB_TENSOR_OFFSET);
        maskUbTensor16 = resource.ubBuf.template GetBufferByByte<half>(MASK16_UB_TENSOR_OFFSET);
        lmUbTensor = resource.ubBuf.template GetBufferByByte<half>(LM_UB_TENSOR_OFFSET);
        hmUbTensor = resource.ubBuf.template GetBufferByByte<half>(HM_UB_TENSOR_OFFSET);
        gmUbTensor = resource.ubBuf.template GetBufferByByte<half>(GM_UB_TENSOR_OFFSET);
        dmUbTensor = resource.ubBuf.template GetBufferByByte<half>(DM_UB_TENSOR_OFFSET);
        llUbTensor = resource.ubBuf.template GetBufferByByte<half>(LL_UB_TENSOR_OFFSET);
        tvUbTensor = resource.ubBuf.template GetBufferByByte<half>(TV_UB_TENSOR_OFFSET);
        glUbTensor = resource.ubBuf.template GetBufferByByte<half>(GL_UB_TENSOR_OFFSET);
    }

    __aicore__ inline
    ~BlockEpilogue() {}
    #include "online_softmax_low_prec/fused_block_epilogue_online_softmax_low_prec_row_ops.inc.hpp"
    #include "online_softmax_low_prec/fused_block_epilogue_online_softmax_low_prec_data_copy.inc.hpp"
    #include "online_softmax_low_prec/fused_block_epilogue_online_softmax_low_prec_mask.inc.hpp"
    #include "online_softmax_low_prec/fused_block_epilogue_online_softmax_low_prec_softmax.inc.hpp"
    #include "online_softmax_low_prec/fused_block_epilogue_online_softmax_low_prec_subcore.inc.hpp"
    #include "online_softmax_low_prec/fused_block_epilogue_online_softmax_low_prec_operator.inc.hpp"
private:
    half scaleValue;
    AscendC::LocalTensor<half> lsUbTensor;
    AscendC::LocalTensor<half> computeUbTensor;
    AscendC::LocalTensor<ElementOutput> lpUbTensor;
    AscendC::LocalTensor<ElementMask> maskUbTensor;
    AscendC::LocalTensor<half> maskUbTensor16;
    AscendC::LocalTensor<half> lmUbTensor;
    AscendC::LocalTensor<half> hmUbTensor;
    AscendC::LocalTensor<half> gmUbTensor;
    AscendC::LocalTensor<half> dmUbTensor;
    AscendC::LocalTensor<half> llUbTensor;
    AscendC::LocalTensor<half> tvUbTensor;
    AscendC::LocalTensor<half> glUbTensor;
};
}

#endif  // EPILOGUE_BLOCK_BLOCK_EPILOGUE_ONLINE_SOFTMAX_LOW_PREC_HPP
