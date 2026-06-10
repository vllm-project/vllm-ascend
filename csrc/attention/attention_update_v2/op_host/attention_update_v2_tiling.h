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
#ifndef ASCEND_OPS_ATTENTION_UPDATE_V2_TILING_H
#define ASCEND_OPS_ATTENTION_UPDATE_V2_TILING_H
#include <cstdint>
#include <vector>
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling_base/tiling_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(AttentionUpdateV2TilingData)
    TILING_DATA_FIELD_DEF(uint64_t, updateType);    // 可选参数，控制lse_m的输出，为0不输出lse_m，为1输出lse_m
    TILING_DATA_FIELD_DEF(uint64_t, bshSize);       // lse的数据个数，即bsh大小
    TILING_DATA_FIELD_DEF(uint64_t, goSize);        // go的数据个数，即bsh*hDim大小
    TILING_DATA_FIELD_DEF(uint64_t, sp);            // lse和go的tensor个数
    TILING_DATA_FIELD_DEF(uint64_t, hDim);          // go的尾轴d的大小

    TILING_DATA_FIELD_DEF(uint64_t, usedCoreNum);   // 使用核数
    TILING_DATA_FIELD_DEF(uint64_t, formerBlockNum);// 正常核数
    TILING_DATA_FIELD_DEF(uint64_t, tailBlockNum);  // 尾核数

    TILING_DATA_FIELD_DEF(uint64_t, formerLength);  // 正常核切分长度
    TILING_DATA_FIELD_DEF(uint64_t, tailLength);    // 尾核切分长度

    TILING_DATA_FIELD_DEF(uint64_t, innerFormerBlockNum);   // 内循环正常循环次数
    TILING_DATA_FIELD_DEF(uint64_t, innerTailBlockNum);     // 内循环尾循环次数(1)
    TILING_DATA_FIELD_DEF(uint64_t, innerFormerLength);     // 内循环正常循环切分大小
    TILING_DATA_FIELD_DEF(uint64_t, innerTailLength);       // 内循环尾循环切分大小

    TILING_DATA_FIELD_DEF(uint64_t, tailInnerFormerBlockNum);   // 尾核的内循环正常循环次数
    TILING_DATA_FIELD_DEF(uint64_t, tailInnerTailBlockNum);     // 尾核的内循环尾循环次数(1)
    TILING_DATA_FIELD_DEF(uint64_t, tailInnerFormerLength);     // 尾核的内循环正常循环切分大小
    TILING_DATA_FIELD_DEF(uint64_t, tailInnerTailLength);       // 尾核的内循环尾循环切分大小

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AttentionUpdateV2_20010, AttentionUpdateV2TilingData)
REGISTER_TILING_DATA_CLASS(AttentionUpdateV2_20011, AttentionUpdateV2TilingData)
REGISTER_TILING_DATA_CLASS(AttentionUpdateV2_20020, AttentionUpdateV2TilingData)
REGISTER_TILING_DATA_CLASS(AttentionUpdateV2_20021, AttentionUpdateV2TilingData)
REGISTER_TILING_DATA_CLASS(AttentionUpdateV2_20030, AttentionUpdateV2TilingData)
REGISTER_TILING_DATA_CLASS(AttentionUpdateV2_20031, AttentionUpdateV2TilingData)


class AttentionUpdateV2Tiling : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit AttentionUpdateV2Tiling(gert::TilingContext *context) : TilingBaseClass(context){
    }

protected:
    bool IsCapable() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    void DumpTilingInfo() override;

    ge::graphStatus CheckInputParams();
    ge::graphStatus CheckInputDim();
    ge::graphStatus CheckInputDtype();

private:
    // 硬件信息 
    uint64_t ubSize_ = 0;
    uint64_t totalCoreNum_ = 0;
    uint64_t workspaceSize_ = 0;

    // 输入参数信息
    uint64_t sp_ = 0;
    uint64_t d_ = 0;
    uint64_t updateType_ = 0;
    uint64_t bshSize_ = 0;
    uint64_t goSize_ = 0;

    // 分核信息，核个数
    uint64_t needCoreNum_ = 0;
    uint64_t usedCoreNum_ = 0;
    uint64_t formerBlockNum_ = 0;
    uint64_t tailBlockNum_ = 0;

    // 分核信息，核长度
    uint64_t totalLength_ = 0;
    uint64_t formerLength_ = 0;
    uint64_t tailLength_ = 0;

    // 内循环信息
    uint64_t innerFormerBlockNum_ = 0;
    uint64_t innerTailBlockNum_ = 0;
    uint64_t innerFormerLength_ = 0;
    uint64_t innerTailLength_ = 0;

    // 尾核的内循环信息
    uint64_t tailInnerFormerBlockNum_ = 0;
    uint64_t tailInnerTailBlockNum_ = 0;
    uint64_t tailInnerFormerLength_ = 0;
    uint64_t tailInnerTailLength_ = 0;

    gert::Shape goShape_;
    gert::Shape lseShape_;
    ge::DataType goType_;
    ge::DataType lseType_;
    AttentionUpdateV2TilingData tilingData_;

};

}
#endif  // ASCEND_OPS_ATTENTION_UPDATE_V2_TILING_H