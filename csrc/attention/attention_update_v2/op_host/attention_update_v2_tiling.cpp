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
 * \file attention_update_v2_tiling.h
 * \brief
 */

#include <algorithm>
#include <string>
#include "tiling/tiling_api.h"
#include "platform/platform_info.h"
#include "util/shape_util.h"
#include "common/op_host/matmul_common_infershape.h"
#include "attention_update_v2_tiling.h"
#include "tiling_base/tiling_templates_registry.h"
#include "decode_update_tiling.h"

namespace optiling {
constexpr uint64_t DIM_0 = 0;
constexpr uint64_t DIM_1 = 1;
constexpr uint64_t DIM_2 = 2;
constexpr uint64_t LSE_DIM_NUM = 2;  // lse shape: [sp, bsh]
constexpr uint64_t GO_DIM_NUM  = 3;  // go  shape: [sp, bsh, hd]
constexpr uint64_t D_MIN = 8;
constexpr uint64_t D_MAX = 512;
constexpr uint64_t D_DIVIDE_8 = 8;
constexpr uint64_t ATTR_SP_MAX = 16;

constexpr uint64_t INPUT_LSE_INDEX = 0;
constexpr uint64_t INPUT_GO_INDEX  = 1;
constexpr uint64_t ATTR_UPDATE_TYPE_INDEX = 0;

constexpr uint64_t GO_FLAG_FLOAT32  = 1UL;
constexpr uint64_t GO_FLAG_FLOAT16  = 2UL;
constexpr uint64_t GO_FLAG_BFLOAT16 = 3UL;
constexpr uint64_t NUM_2  = 2UL;
constexpr uint64_t NUM_10 = 10UL;
constexpr uint64_t TILINGKEY_INIT_VALUE = 20000UL;

constexpr uint64_t UB_ALIGN_SIZE = 32UL;
constexpr uint64_t MAX_UB_FACTOR = 4UL;
constexpr uint64_t RESERVED_UB_SIZE = 256;
constexpr uint64_t SYS_WORKSPACE_SIZE = static_cast<uint64_t>(16 * 1024 * 1024);

bool AttentionUpdateV2Tiling::IsCapable()
{
    return true;
}

ge::graphStatus AttentionUpdateV2Tiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AttentionUpdateV2Tiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        auto compileInfoPtr = reinterpret_cast<const DecodeUpdateCompileInfo*>(context_->GetCompileInfo());
        OP_TILING_CHECK(
            compileInfoPtr == nullptr, OP_LOGE(context_, "compile info is null"),
            return ge::GRAPH_FAILED);
        totalCoreNum_ = compileInfoPtr->coreNum;
        ubSize_ = compileInfoPtr->ubSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        totalCoreNum_ = static_cast<uint64_t>(ascendcPlatform.GetCoreNumAiv());
        if (totalCoreNum_ == 0UL) {
            OP_LOGE(context_->GetNodeName(), "coreNum is 0");
            return ge::GRAPH_FAILED;
        }
        uint64_t ubSize = 0;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
        if (ubSize == static_cast<uint64_t>(0)) {
            OP_LOGE(context_->GetNodeName(), "ubSize is 0");
            return ge::GRAPH_FAILED;
        }
        ubSize_ = static_cast<uint64_t>(ubSize);
    }

    return ge::GRAPH_SUCCESS;
}

// 检查输入数据类型
ge::graphStatus AttentionUpdateV2Tiling::CheckInputDtype()
{
    if (lseType_ != ge::DataType::DT_FLOAT) {
        OP_LOGE(context_->GetNodeName(),
            "lse dtype must be fp32, but got %d", lseType_);
        return ge::GRAPH_FAILED;
    }
    if (goType_ != ge::DataType::DT_FLOAT && goType_ != ge::DataType::DT_FLOAT16 &&
        goType_ != ge::DataType::DT_BF16) {
        OP_LOGE(context_->GetNodeName(),
            "go dtype must be fp32, fp16 or bf16, but got %d", goType_);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// 检查输入数据维度：lse 为 2D [sp, bsh]，go 为 3D [sp, bsh, hd]
ge::graphStatus AttentionUpdateV2Tiling::CheckInputDim()
{
    OP_CHECK_IF(
        !(d_ >= D_MIN && d_ <= D_MAX && d_ % D_DIVIDE_8 == 0),
        OP_LOGE(context_->GetNodeName(),
            "go hDim must be in [8,512] and divisible by 8, but got %ld", d_),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        !(lseShape_.GetDimNum() == LSE_DIM_NUM),
        OP_LOGE(context_->GetNodeName(),
            "lse must be 2D [sp, bsh], but got %ld dims", lseShape_.GetDimNum()),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        !(goShape_.GetDimNum() == GO_DIM_NUM),
        OP_LOGE(context_->GetNodeName(),
            "go must be 3D [sp, bsh, hd], but got %ld dims", goShape_.GetDimNum()),
        return ge::GRAPH_FAILED);

    uint64_t goSp  = static_cast<uint64_t>(goShape_.GetDim(DIM_0));
    OP_CHECK_IF(
        !(sp_ == goSp),
        OP_LOGE(context_->GetNodeName(),
            "sp of lse (%ld) must equal sp of go (%ld)", sp_, goSp),
        return ge::GRAPH_FAILED);

    uint64_t goBsh = static_cast<uint64_t>(goShape_.GetDim(DIM_1));
    OP_CHECK_IF(
        !(bshSize_ == goBsh),
        OP_LOGE(context_->GetNodeName(),
            "bsh of lse (%ld) must equal bsh of go (%ld)", bshSize_, goBsh),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// 检查输入参数
ge::graphStatus AttentionUpdateV2Tiling::CheckInputParams()
{
    if (CheckInputDtype() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckInputDim() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // 检查参数update_type
    OP_CHECK_IF(
        !(updateType_ == 0 || updateType_ == 1),
        OP_LOGE(context_->GetNodeName(), "Update_type should be 0 or 1,but got %ld", updateType_),
        return ge::GRAPH_FAILED);

    // 检查参数sp
    OP_CHECK_IF(
        !(sp_ >= 1 && sp_ <= ATTR_SP_MAX), OP_LOGE(context_->GetNodeName(), "Sp need in [1,16],but got %ld", sp_),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AttentionUpdateV2Tiling::GetShapeAttrsInfo()
{
    OP_TILING_CHECK(context_ == nullptr, OP_LOGE("AttentionUpdateV2", "context is null"),
                    return ge::GRAPH_FAILED);
    auto attrs = context_->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    // lse: [sp, bsh],  go: [sp, bsh, hd]
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(INPUT_LSE_INDEX));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(INPUT_GO_INDEX));
    lseShape_ = context_->GetInputShape(INPUT_LSE_INDEX)->GetOriginShape();
    goShape_  = context_->GetInputShape(INPUT_GO_INDEX)->GetOriginShape();

    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(INPUT_LSE_INDEX));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(INPUT_GO_INDEX));
    lseType_ = context_->GetInputDesc(INPUT_LSE_INDEX)->GetDataType();
    goType_  = context_->GetInputDesc(INPUT_GO_INDEX)->GetDataType();

    sp_      = static_cast<uint64_t>(lseShape_.GetDim(DIM_0));  // lse[sp, bsh]
    bshSize_ = static_cast<uint64_t>(lseShape_.GetDim(DIM_1));  // lse[sp, bsh]
    d_       = static_cast<uint64_t>(goShape_.GetDim(DIM_2));   // go [sp, bsh, hd]

    const uint64_t* updateTypePtr = attrs->GetAttrPointer<uint64_t>(ATTR_UPDATE_TYPE_INDEX);
    OP_TILING_CHECK(updateTypePtr == nullptr,
                    OP_LOGE("AttentionUpdateV2", "updateTypePtr is null"),
                    return ge::GRAPH_FAILED);
    updateType_ = *updateTypePtr;
    goSize_ = bshSize_ * d_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AttentionUpdateV2Tiling::DoOpTiling()
{
    /*
    * 最大ub使用率：rate=2*sp+max(sp,2)+2*sp
    * 元素类型大小：elem_size=max(sizeof(goType_), sizeof(lseType_))
    * 计算流程：
        1.使用核数：usedCoreNum_=min(ceil(rate*bsh*elem_size/(ubSize-256)),64)
        2.需要核数：needCoreNum_=ceil(rate*bsh*elem_size/(ubSize-256))
        3.切分大小（按ub算）：formerLength_=floor((ub_size-256)/(rate*elem_size))
        4.尾核切分大小：tailLength_=bsh-formerLength_*(usedCoreNum_-1)
        5.尾核个数：tailBlockNum_=1
        6.正常核个数：formerBlockNum_=needCoreNum_-1
        7.内循环正常次数：innerFormerBlockNum_ = d_
        8.内循环尾数：innerTailBlockNum_=formerLength_%d_?1:0
        9.内循环切分大小：innerFormerLength_=floor(formerLength_/d_)*d_
        10.内循环尾数切分大小：innerTailLength_=formerLength_*d_%innerFormerLength_
        11.尾核内循环与正常核内循环原理相同
        12.尾核内循环正常次数：tailInnerFormerBlockNum_ = d_
        13.尾核内循环尾数：tailInnerTailBlockNum_=tailLength_%d_?1:0
        14.尾核内循环切分大小：tailInnerFormerLength_=floor(tailLength_/d_)*d_
        15.尾核内循环尾数切分大小：tailInnerTailLength_=tailLength_*d_%tailInnerFormerLength_
        16.如果数据量特别小，formerBlockNum_=0时，有关正常核的数据全为0
    */
    uint64_t rate = MAX_UB_FACTOR * sp_ + std::max(sp_, static_cast<uint64_t>(NUM_2));
    uint64_t elem_size = std::max(sizeof(goType_), sizeof(lseType_));
    totalLength_ = bshSize_;

    // 核数，计算除法要用double进行中间计算，并转回uint64
    needCoreNum_ = static_cast<uint64_t>(
        ceil(static_cast<double>(rate * bshSize_ * elem_size) / static_cast<double>(ubSize_ - RESERVED_UB_SIZE)));
    tailBlockNum_ = 1UL;

    // 切分长度，并且32B对齐
    uint64_t max_split_size = static_cast<uint64_t>((ubSize_ - RESERVED_UB_SIZE) / (rate * elem_size));
    uint64_t align_factor = UB_ALIGN_SIZE / sizeof(lseType_);
    formerLength_ = (max_split_size / align_factor * align_factor);
    needCoreNum_ = ceil(static_cast<double>(totalLength_) / static_cast<double>(formerLength_));
    formerBlockNum_ = needCoreNum_ - tailBlockNum_;
    tailLength_ = bshSize_ - formerLength_ * formerBlockNum_;
    usedCoreNum_ = std::min(needCoreNum_, static_cast<uint64_t>(totalCoreNum_));

    if (formerBlockNum_ == 0UL) {
        formerLength_ = 0UL;

        innerFormerBlockNum_ = 0UL;
        innerFormerLength_ = 0UL;
        innerTailLength_ = 0UL;
        innerTailBlockNum_ = 0UL;
    }

    // 内循环
    if (formerLength_) {
        innerFormerBlockNum_ = formerLength_ < d_ ? formerLength_ : d_;
        innerFormerLength_ = static_cast<uint64_t>(((formerLength_) / innerFormerBlockNum_) * d_);
        innerFormerBlockNum_ = formerLength_ * d_ / innerFormerLength_;
        innerTailLength_ = formerLength_ * d_ - innerFormerLength_ * innerFormerBlockNum_;
        innerTailBlockNum_ = innerTailLength_ ? 1UL : 0UL;
    }

    if (tailLength_) {
        tailInnerFormerBlockNum_ = tailLength_ < d_ ? tailLength_ : d_;
        tailInnerFormerLength_ = static_cast<uint64_t>(((tailLength_) / tailInnerFormerBlockNum_) * d_);
        tailInnerFormerBlockNum_ = tailLength_ * d_ / tailInnerFormerLength_;
        tailInnerTailLength_ = tailLength_ * d_ - tailInnerFormerLength_ * tailInnerFormerBlockNum_;
        tailInnerTailBlockNum_ = tailInnerTailLength_ ? 1UL : 0UL;
    }

    if (tailInnerFormerBlockNum_ == 0UL) {
        tailInnerFormerLength_ = 0UL;
    }

    if (tailInnerTailBlockNum_ == 0UL) {
        tailInnerTailLength_ = 0UL;
    }

    if (innerFormerBlockNum_ == 0UL) {
        innerFormerLength_ = 0UL;
    }

    if (innerTailBlockNum_ == 0UL) {
        innerTailLength_ = 0UL;
    }
    return ge::GRAPH_SUCCESS;
}

uint64_t AttentionUpdateV2Tiling::GetTilingKey() const
{
    uint64_t go_flag = 0UL;
    uint64_t tiling_key = TILINGKEY_INIT_VALUE;
    if (goType_ == ge::DataType::DT_FLOAT) {
        go_flag = GO_FLAG_FLOAT32;
    } else if (goType_ == ge::DataType::DT_FLOAT16) {
        go_flag = GO_FLAG_FLOAT16;
    } else if (goType_ == ge::DataType::DT_BF16) {
        go_flag = GO_FLAG_BFLOAT16;
    }
    tiling_key += (go_flag * NUM_10 + updateType_);
    return tiling_key;
}
ge::graphStatus AttentionUpdateV2Tiling::GetWorkspaceSize()
{
    workspaceSize_ = SYS_WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AttentionUpdateV2Tiling::PostTiling()
{
    auto workspaces = context_->GetWorkspaceSizes(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = workspaceSize_;

    context_->SetBlockDim(usedCoreNum_);

    // 输入参数
    tilingData_.set_updateType(updateType_);
    tilingData_.set_bshSize(bshSize_);
    tilingData_.set_sp(sp_);
    tilingData_.set_hDim(d_);
    tilingData_.set_goSize(goSize_);

    // 分核数
    tilingData_.set_usedCoreNum(usedCoreNum_);
    tilingData_.set_formerBlockNum(formerBlockNum_);
    tilingData_.set_tailBlockNum(tailBlockNum_);

    // 单核切分大小，splitBshSize
    tilingData_.set_formerLength(formerLength_);
    tilingData_.set_tailLength(tailLength_);

    // 设置内循环循环次数和切分大小
    tilingData_.set_innerFormerBlockNum(innerFormerBlockNum_);
    tilingData_.set_innerTailBlockNum(innerTailBlockNum_);
    tilingData_.set_innerFormerLength(innerFormerLength_);
    tilingData_.set_innerTailLength(innerTailLength_);

    // 设置尾块内循环次数和切分大小
    tilingData_.set_tailInnerFormerBlockNum(tailInnerFormerBlockNum_);
    tilingData_.set_tailInnerTailBlockNum(tailInnerTailBlockNum_);
    tilingData_.set_tailInnerFormerLength(tailInnerFormerLength_);
    tilingData_.set_tailInnerTailLength(tailInnerTailLength_);

    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

void AttentionUpdateV2Tiling::DumpTilingInfo()
{
    std::ostringstream info;
    // 输入参数
    info << "updateType: " << updateType_ << std::endl;
    info << "bshSize: " << bshSize_ << std::endl;
    info << "goSize: " << goSize_ << std::endl;
    info << "sp: " << sp_ << std::endl;
    info << "hDim: " << d_ << std::endl;

    info << "usedCoreNum: " << usedCoreNum_ << std::endl;
    info << "formerBlockNum: " << formerBlockNum_ << std::endl;
    info << "tailBlockNum: " << tailBlockNum_ << std::endl;

    info << "totalLength: " << totalLength_ << std::endl;
    info << "formerLength: " << formerLength_ << std::endl;
    info << "tailLength: " << tailLength_ << std::endl;

    // 内循环信息
    info << "innerFormerBlockNum: " << innerFormerBlockNum_ << std::endl;
    info << "innerTailBlockNum: " << innerTailBlockNum_ << std::endl;
    info << "innerFormerLength: " << innerFormerLength_ << std::endl;
    info << "innerTailLength: " << innerTailLength_ << std::endl;

    info << "tailInnerFormerBlockNum: " << tailInnerFormerBlockNum_ << std::endl;
    info << "tailInnerTailBlockNum: " << tailInnerTailBlockNum_ << std::endl;
    info << "tailInnerFormerLength: " << tailInnerFormerLength_ << std::endl;
    info << "tailInnerTailLength: " << tailInnerTailLength_ << std::endl;

    OP_LOGI(context_->GetNodeName(), "%s", info.str().c_str());
}

REGISTER_OPS_TILING_TEMPLATE(AttentionUpdateV2, AttentionUpdateV2Tiling, 1);
} // namespace optiling
