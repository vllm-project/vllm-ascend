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
 * \file fia_tiling_base.h
 * \brief
 */

#ifndef FIA_TILING_BASE_H
#define FIA_TILING_BASE_H

#include <sstream>
#include "register/tilingdata_base.h"
#include <exe_graph/runtime/tiling_context.h>
#include <graph/utils/type_utils.h>
#include <tiling/platform/platform_ascendc.h>
#include "err/ops_err.h"
#include "platform/soc_spec.h"

#ifdef ASCENDC_OP_TEST
#define ASCENDC_EXTERN_C extern "C"
#else
#define ASCENDC_EXTERN_C
#endif

namespace optiling {

class TilingInfo {};

enum class ScheduleMode : uint32_t {
    NORMAL_MODE = 0,
    BATCH_MODE = 1,
    SYNC_MODE = 2
};

class FiaTilingBase {
public:
    FiaTilingBase() = default;

    explicit FiaTilingBase(gert::TilingContext *context) : context_(context)
    {
    }

    virtual ~FiaTilingBase() = default;

    // Tiling执行框架
    //     1、GRAPH_SUCCESS: 成功，并且不需要继续执行后续Tiling类的实现
    //     2、GRAPH_FAILED: 失败，中止整个Tiling流程
    //     3、GRAPH_PARAM_INVALID: 本类不支持，需要继续往下执行其他Tiling类的实现
    ge::graphStatus DoTiling(TilingInfo *tilingInfo)
    {
        InitTilingInfo(tilingInfo);
        if (!IsCapable()) {
            return ge::GRAPH_PARAM_INVALID;
        }
        if (DoOpTiling() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    }

protected:
    gert::TilingContext *context_ = nullptr;
    virtual void InitTilingInfo(TilingInfo *tilingInfo) = 0;
    virtual bool IsCapable() = 0;
    virtual ge::graphStatus DoOpTiling() = 0;

    [[nodiscard]] ge::graphStatus SetNumBlocks(uint32_t numBlocks) const
    {
        context_->SetBlockDim(numBlocks);
        return ge::GRAPH_SUCCESS;
    }

    [[nodiscard]] ge::graphStatus SetTilingKey(uint64_t tilingKey) const
    {
        context_->SetTilingKey(tilingKey);
        return ge::GRAPH_SUCCESS;
    }

    [[nodiscard]] ge::graphStatus SetWorkspaceSize(uint64_t workspaceSize) const
    {
        OP_CHECK_IF(context_->GetWorkspaceSizes(1) == nullptr,
            OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "WorkSpaceSize got from ge is nullptr"),
            return ge::GRAPH_FAILED);
        size_t *workSpaces = context_->GetWorkspaceSizes(1);
        workSpaces[0] = workspaceSize;
        return ge::GRAPH_SUCCESS;
    }

    [[nodiscard]] ge::graphStatus SetTilingData(TilingDef &tilingData) const
    {
        OP_CHECK_IF(context_->GetRawTilingData() == nullptr,
            OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "RawTilingData got from GE context is nullptr."),
            return ge::GRAPH_FAILED);

        tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
        context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

        return ge::GRAPH_SUCCESS;
    }

    [[nodiscard]] ge::graphStatus SetScheduleMode(ScheduleMode scheduleMode) const
    {
        return context_->SetScheduleMode(static_cast<uint32_t>(scheduleMode));
    }

    template <typename T>
    [[nodiscard]] std::string GetShapeDebugStr(const T &shape) const
    {
        std::ostringstream ost;
        ost << "[";
        if (shape.GetDimNum() > 0) {
            for (size_t i = 0; i < shape.GetDimNum() - 1; ++i) {
                ost << shape.GetDim(i) << ", ";
            }
            ost << shape.GetDim(shape.GetDimNum() - 1);
        }
        ost << "]";
        return ost.str();
    }

    [[nodiscard]] std::string GetTensorDebugStr(const gert::StorageShape *shape,
                                                const gert::CompileTimeTensorDesc *tensor) const
    {
        if (shape == nullptr || tensor == nullptr) {
            return "nil ";
        }
        std::ostringstream oss;
        oss << "(dtype: " << ge::TypeUtils::DataTypeToSerialString(tensor->GetDataType()) << "),";
        oss << "(shape:" << GetShapeDebugStr(shape->GetStorageShape()) << "),";
        oss << "(ori_shape:" << GetShapeDebugStr(shape->GetOriginShape()) << "),";
        oss << "(format: "
            << ge::TypeUtils::FormatToSerialString(
                   static_cast<ge::Format>(ge::GetPrimaryFormat(tensor->GetStorageFormat())))
            << "),";
        oss << "(ori_format: " << ge::TypeUtils::FormatToSerialString(tensor->GetOriginFormat()) << ") ";
        return oss.str();
    }

    [[nodiscard]] std::string GetTilingContextDebugStr() const
    {
        std::ostringstream oss;
        for (size_t i = 0; i < context_->GetComputeNodeInfo()->GetInputsNum(); ++i) {
            oss << "input" << i << ": ";
            oss << GetTensorDebugStr(context_->GetInputShape(i), context_->GetInputDesc(i));
        }

        for (size_t i = 0; i < context_->GetComputeNodeInfo()->GetOutputsNum(); ++i) {
            oss << "output" << i << ": ";
            oss << GetTensorDebugStr(context_->GetOutputShape(i), context_->GetOutputDesc(i));
        }
        return oss.str();
    }

    [[nodiscard]] std::string GetTilingDataDebugStr() const
    {
        auto rawTilingData = context_->GetRawTilingData();
        auto rawTilingDataSize = rawTilingData->GetDataSize();
        auto data = reinterpret_cast<const int32_t *>(rawTilingData->GetData());
        size_t len = rawTilingDataSize / sizeof(int32_t);
        std::ostringstream oss;
        for (size_t i = 0; i < len; i++) {
            oss << data[i] << ", ";
        }
        return oss.str();
    }

private:
    std::unique_ptr<platform_ascendc::PlatformAscendC> ascendcPlatform_{nullptr};
};

} // namespace optiling
#endif // FIA_TILING_BASE_H