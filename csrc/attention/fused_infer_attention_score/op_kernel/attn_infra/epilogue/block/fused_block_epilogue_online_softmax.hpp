/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSED_BLOCK_EPILOGUE_ONLINE_SOFTMAX_HPP
#define FUSED_BLOCK_EPILOGUE_ONLINE_SOFTMAX_HPP

#include <type_traits>
#include <limits>
#include "../../../attn_infra/fused_base_defs.hpp"
#include "../../../attn_infra/arch/fused_resource.hpp"
#include "../../../attn_infra/arch/fused_cross_core_sync.hpp"
#include "../../../attn_infra/epilogue/fused_epilogue_dispatch_policy.hpp"
#include "../../../attn_infra/epilogue/tile_common/fused_epilogue_tile_copy.hpp"
#include "../../../attn_infra/fused_gemm_coord.hpp"
#include "../../../attn_infra/fused_matrix_coord.hpp"
#include "utils/std/algorithm.h"

namespace NpuArch::Epilogue::Block {

struct SinkLoopParam
{
    uint32_t rowOffsetIoGm;
    uint32_t rowNumCurLoop;
    uint32_t qSBlockSize;
    uint32_t rowOffsetThisSubBlock;

    __aicore__ inline
    SinkLoopParam(
        uint32_t rowOffsetIoGm_,
        uint32_t rowNumCurLoop_,
        uint32_t qSeqBlockSize_,
        uint32_t rowOffsetThisSubBlock_
    ) :
        rowOffsetIoGm(rowOffsetIoGm_),
        rowNumCurLoop(rowNumCurLoop_),
        qSBlockSize(qSeqBlockSize_),
        rowOffsetThisSubBlock(rowOffsetThisSubBlock_)
    {}
};

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
    EpilogueAtlasA2OnlineSoftmax<LSE_MODE_, SINK_MODE_, MASK_MODE_, float>,
    OutputType_,
    InputType_,
    MaskType_,
    SinkType_,
    FullType_>
{
public:
    using DispatchPolicy = EpilogueAtlasA2OnlineSoftmax<LSE_MODE_, SINK_MODE_, MASK_MODE_, float>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementInput = typename InputType_::Element;
    using ElementMask = typename MaskType_::Element;
    using ElementSink = typename SinkType_::Element;
    using ElementFull = typename FullType_::Element;
    using ElementOutput = typename OutputType_::Element;

    using LayoutInput = typename InputType_::Layout;
    using LayoutMask = typename MaskType_::Layout;
    using LayoutFull = typename FullType_::Layout;
    using LayoutOutput = typename OutputType_::Layout;

    static constexpr LseMode LSE_MODE = DispatchPolicy::LSE_MODE;
    static constexpr SinkMode SINK_MODE = DispatchPolicy::SINK_MODE;
    static constexpr MaskMode MASK_MODE = DispatchPolicy::MASK_MODE;

    static constexpr uint32_t BLOCK_SIZE_IN_BYTE = 32;
    static constexpr uint32_t REPEAT_SIZE_IN_BYTE = 256;
    static constexpr uint32_t FLOAT_BLOCK_SIZE = 8;
    static constexpr uint32_t FLOAT_VECTOR_SIZE = 64;
    static constexpr uint32_t HALF_VECTOR_SIZE = 128;
    static constexpr uint32_t BLOCK_SIZE = 16;
    static constexpr uint32_t UB_UINT8_VECTOR_SIZE = 1024;
    static constexpr uint32_t UB_UINT8_BLOCK_SIZE = 16384;
    static constexpr uint32_t VECTOR_SIZE = 128;
    static constexpr uint32_t MAX_UB_S_ELEM_NUM = 8192;

    static constexpr uint32_t REDUCE_UB_SIZE = 1024;
    static constexpr uint32_t ROW_OPS_SPEC_MASK_32 = 32;
    static constexpr uint32_t ROW_OPS_SPEC_MASK_4 = 4;
    static constexpr uint32_t MAX_ROW_NUM_SUB_CORE = 256;
    static constexpr int64_t UB_FLOAT_LINE_SIZE = 64;
    static constexpr uint32_t HEAD_NUM_2 = 2;

    static constexpr float NEG_INF = -std::numeric_limits<float>::infinity();

    __aicore__ inline
    BlockEpilogue() {}

    __aicore__ inline
    void init(Arch::Resource<ArchTag> &resource, float scaleValue_)
    {
        // Allocate UB space
        constexpr uint32_t LS_UB_TENSOR_OFFSET = 0;
        constexpr uint32_t LP_UB_TENSOR_OFFSET = 4 * UB_UINT8_BLOCK_SIZE;
        constexpr uint32_t FULL32_UB_TENSOR_OFFSET = 4 * UB_UINT8_BLOCK_SIZE;
        constexpr uint32_t MASK_UB_TENSOR_OFFSET = 4 * UB_UINT8_BLOCK_SIZE;
        constexpr uint32_t MASK32_UB_TENSOR_OFFSET = 4 * UB_UINT8_BLOCK_SIZE;
        constexpr uint32_t MASK_UB_PREMASK_TENSOR_OFFSET = 5 * UB_UINT8_BLOCK_SIZE;
        constexpr uint32_t TV_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE;
        constexpr uint32_t LM_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 8 * UB_UINT8_VECTOR_SIZE;

        constexpr uint32_t HM_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 9 * UB_UINT8_VECTOR_SIZE;
        constexpr uint32_t GM_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 10 * UB_UINT8_VECTOR_SIZE;
        constexpr uint32_t LL_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 11 * UB_UINT8_VECTOR_SIZE;
        constexpr uint32_t GL_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 12 * UB_UINT8_VECTOR_SIZE;
        constexpr uint32_t DM_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 13 * UB_UINT8_VECTOR_SIZE;
        constexpr uint32_t SEL_MASK_UB_TENSOR_OFFSET = LL_UB_TENSOR_OFFSET;

        constexpr uint32_t FULL16_UB_TENSOR_OFFSET = 11 * UB_UINT8_BLOCK_SIZE;
        constexpr uint32_t MASK16_UB_TENSOR_OFFSET = 11 * UB_UINT8_BLOCK_SIZE;
        scaleValue = scaleValue_;
        lsUbTensor = resource.ubBuf.template GetBufferByByte<float>(LS_UB_TENSOR_OFFSET);
        lpUbTensor = resource.ubBuf.template GetBufferByByte<ElementOutput>(LP_UB_TENSOR_OFFSET);
        maskUbTensor = resource.ubBuf.template GetBufferByByte<ElementMask>(MASK_UB_TENSOR_OFFSET);
        maskUbTensorUint8 = resource.ubBuf.template GetBufferByByte<uint8_t>(MASK_UB_TENSOR_OFFSET);
        maskUbTensor16 = resource.ubBuf.template GetBufferByByte<half>(MASK16_UB_TENSOR_OFFSET);
        maskUbTensor32 = resource.ubBuf.template GetBufferByByte<float>(MASK32_UB_TENSOR_OFFSET);
        fullUbTensor16 = resource.ubBuf.template GetBufferByByte<ElementFull>(FULL16_UB_TENSOR_OFFSET);
        fullUbTensor32 = resource.ubBuf.template GetBufferByByte<float>(FULL32_UB_TENSOR_OFFSET);
        lmUbTensor = resource.ubBuf.template GetBufferByByte<float>(LM_UB_TENSOR_OFFSET);
        hmUbTensor = resource.ubBuf.template GetBufferByByte<float>(HM_UB_TENSOR_OFFSET);
        gmUbTensor = resource.ubBuf.template GetBufferByByte<float>(GM_UB_TENSOR_OFFSET);
        dmUbTensor = resource.ubBuf.template GetBufferByByte<float>(DM_UB_TENSOR_OFFSET);
        llUbTensor = resource.ubBuf.template GetBufferByByte<float>(LL_UB_TENSOR_OFFSET);
        selMaskUbTensor = resource.ubBuf.template GetBufferByByte<uint8_t>(SEL_MASK_UB_TENSOR_OFFSET);
        tvUbTensor = resource.ubBuf.template GetBufferByByte<float>(TV_UB_TENSOR_OFFSET);
        glUbTensor = resource.ubBuf.template GetBufferByByte<float>(GL_UB_TENSOR_OFFSET);
        tempMaskTensor = resource.ubBuf.template GetBufferByByte<half>(MASK_UB_PREMASK_TENSOR_OFFSET);
    }

    __aicore__ inline
    ~BlockEpilogue() {}

    template <typename T>
    __aicore__ inline T Min(T a, T b)
    {
        return (a > b) ? b : a;
    }
    #include "online_softmax/fused_block_epilogue_online_softmax_row_ops.inc.hpp"
    #include "online_softmax/fused_block_epilogue_online_softmax_data_copy.inc.hpp"
    #include "online_softmax/fused_block_epilogue_online_softmax_mask.inc.hpp"
    #include "online_softmax/fused_block_epilogue_online_softmax_softmax.inc.hpp"
    #include "online_softmax/fused_block_epilogue_online_softmax_subcore.inc.hpp"
    #include "online_softmax/fused_block_epilogue_online_softmax_sink.inc.hpp"
    #include "online_softmax/fused_block_epilogue_online_softmax_operator.inc.hpp"
private:
    uint32_t kvheadIdx = 0;
    float scaleValue;
    AscendC::LocalTensor<float> lsUbTensor;
    AscendC::LocalTensor<ElementOutput> lpUbTensor;
    AscendC::LocalTensor<ElementMask> maskUbTensor;
    AscendC::LocalTensor<uint8_t> maskUbTensorUint8;
    AscendC::LocalTensor<half> maskUbTensor16;
    AscendC::LocalTensor<float> maskUbTensor32;
    AscendC::LocalTensor<ElementFull> fullUbTensor16;
    AscendC::LocalTensor<float> fullUbTensor32;
    AscendC::LocalTensor<float> lmUbTensor;
    AscendC::LocalTensor<float> hmUbTensor;
    AscendC::LocalTensor<float> gmUbTensor;
    AscendC::LocalTensor<float> dmUbTensor;
    AscendC::LocalTensor<float> llUbTensor;
    AscendC::LocalTensor<float> tvUbTensor;
    AscendC::LocalTensor<float> glUbTensor;
    AscendC::LocalTensor<half> tempMaskTensor;
    AscendC::LocalTensor<uint8_t> selMaskUbTensor;
};
}

#endif  // EPILOGUE_BLOCK_BLOCK_EPILOGUE_ONLINE_SOFTMAX_HPP