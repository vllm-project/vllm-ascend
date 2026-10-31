/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file tiling_base.h
 * \brief
 */

#pragma once

#include <sstream>
#include <exe_graph/runtime/tiling_context.h>
#include <graph/utils/type_utils.h>
#include "tiling/platform/platform_ascendc.h"
#include "error_log.h"

#ifdef ASCENDC_OP_TEST
#define ASCENDC_EXTERN_C extern "C"
#else
#define ASCENDC_EXTERN_C
#endif

namespace Ops {
namespace Transformer {
namespace OpTiling {

struct AiCoreParams {
    uint64_t ubSize = 0;
    uint64_t blockDim = 0;
    uint64_t aicNum = 0;
    uint64_t l1Size = 0;
    uint64_t l0aSize = 0;
    uint64_t l0bSize = 0;
    uint64_t l0cSize = 0;
};

struct CompileInfoCommon {
    uint32_t aivNum;
    uint32_t aicNum;
    uint64_t ubSize;
    uint64_t l1Size;
    uint64_t l0aSize;
    uint64_t l0bSize;
    uint64_t l0cSize;
    uint64_t l2CacheSize;
    int64_t coreNum;
    int32_t socVersion;
    uint32_t rsvd;
};

struct FlashAttentionScoreGradCompileInfo {
    uint32_t aivNum;
    uint32_t aicNum;
    uint64_t ubSize;
    uint64_t l1Size;
    uint64_t l0aSize;
    uint64_t l0bSize;
    uint64_t l0cSize;
    uint64_t l2CacheSize;
    int64_t coreNum;
    platform_ascendc::SocVersion socVersion;
};

struct FACompileInfoCommon {
    uint32_t aivNum;
    uint32_t aicNum;
    uint64_t ubSize;
    uint64_t l1Size;
    uint64_t l0aSize;
    uint64_t l0bSize;
    uint64_t l0cSize;
    uint64_t l2CacheSize;
    int64_t coreNum;
    int32_t socVersion;
    uint32_t rsvd;
};

class TilingBaseClass {
public:
    explicit TilingBaseClass(gert::TilingContext* context) : context_(context)
    {}

    virtual ~TilingBaseClass() = default;

    // Tiling execution framework
    //     1. GRAPH_SUCCESS: Success, and no need to continue executing subsequent Tiling class implementations
    //     2. GRAPH_FAILED: Failure, abort the entire Tiling process
    //     3. GRAPH_PARAM_INVALID: This class does not support, need to continue executing other Tiling class implementations
    ge::graphStatus DoTiling()
    {
        auto ret = GetShapeAttrsInfo();
        if (ret != ge::GRAPH_SUCCESS) {
            return ret;
        }
        ret = GetPlatformInfo();
        if (ret != ge::GRAPH_SUCCESS) {
            return ret;
        }
        if (!IsCapable()) {
            return ge::GRAPH_PARAM_INVALID;
        }
        ret = DoOpTiling();
        if (ret != ge::GRAPH_SUCCESS) {
            return ret;
        }
        ret = DoLibApiTiling();
        if (ret != ge::GRAPH_SUCCESS) {
            return ret;
        }
        ret = GetWorkspaceSize();
        if (ret != ge::GRAPH_SUCCESS) {
            return ret;
        }
        ret = PostTiling();
        if (ret != ge::GRAPH_SUCCESS) {
            return ret;
        }
        context_->SetTilingKey(GetTilingKey());
        DumpTilingInfo();
        return ge::GRAPH_SUCCESS;
    }

    // Update context
    virtual void Reset(gert::TilingContext* context)
    {
        context_ = context;
    }

protected:
    virtual bool IsCapable() = 0;
    // 1. Get platform information such as CoreNum, UB/L1/L0C resource sizes
    virtual ge::graphStatus GetPlatformInfo() = 0;
    // 2. Get INPUT/OUTPUT/ATTR information
    virtual ge::graphStatus GetShapeAttrsInfo() = 0;
    // 3. Calculate data splitting TilingData
    virtual ge::graphStatus DoOpTiling() = 0;
    // 4. Calculate high-level API TilingData
    virtual ge::graphStatus DoLibApiTiling() = 0;
    // 5. Calculate TilingKey
    [[nodiscard]] virtual uint64_t GetTilingKey() const = 0;
    // 6. Calculate Workspace size
    virtual ge::graphStatus GetWorkspaceSize() = 0;
    // 7. Save Tiling data
    virtual ge::graphStatus PostTiling() = 0;
    // 8. Dump Tiling data
    virtual void DumpTilingInfo()
    {
        int32_t enable = CheckLogLevel(static_cast<int32_t>(OP), DLOG_DEBUG);
        if (enable != 1) {
            return;
        }
        auto buf = (uint32_t*)context_->GetRawTilingData()->GetData();
        auto bufLen = context_->GetRawTilingData()->GetDataSize();
        std::ostringstream oss;
        oss << "Start to dump tiling info. tilingkey:" << context_->GetTilingKey() << ", tiling data size:" << bufLen
            << ", content:";
        for (size_t i = 0; i < bufLen / sizeof(uint32_t); i++) {
            oss << *(buf + i) << ",";
            if (oss.str().length() > 640) { // Split according to 640 to avoid truncation
                OP_LOGD(context_, "%s", oss.str().c_str());
                oss.str("");
            }
        }
        OP_LOGD(context_, "%s", oss.str().c_str());
    }

    static uint32_t CalcTschBlockDim(uint32_t sliceNum, uint32_t aicCoreNum, uint32_t aivCoreNum)
    {
        uint32_t ration;
        if (aicCoreNum == 0 || aivCoreNum == 0 || aicCoreNum > aivCoreNum) {
            return sliceNum;
        }
        ration = aivCoreNum / aicCoreNum;
        return (sliceNum + (ration - 1)) / ration;
    }

    template <typename T>
    [[nodiscard]] std::string GetShapeDebugStr(const T& shape) const
    {
        std::ostringstream oss;
        oss << "[";
        if (shape.GetDimNum() > 0) {
            for (size_t i = 0; i < shape.GetDimNum() - 1; ++i) {
                oss << shape.GetDim(i) << ", ";
            }
            oss << shape.GetDim(shape.GetDimNum() - 1);
        }
        oss << "]";
        return oss.str();
    }

    [[nodiscard]] std::string GetTensorDebugStr(
        const gert::StorageShape* shape, const gert::CompileTimeTensorDesc* tensor)
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

    [[nodiscard]] std::string GetTilingContextDebugStr()
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
        auto data = reinterpret_cast<const int32_t*>(rawTilingData->GetData());
        size_t len = rawTilingDataSize / sizeof(int32_t);
        std::ostringstream oss;
        for (size_t i = 0; i < len; i++) {
            oss << data[i] << ", ";
        }
        return oss.str();
    }

protected:
    gert::TilingContext* context_ = nullptr;
    std::unique_ptr<platform_ascendc::PlatformAscendC> ascendcPlatform_{nullptr};
    uint32_t blockDim_{0};
    uint64_t workspaceSize_{0};
    uint64_t tilingKey_{0};
    AiCoreParams aicoreParams_;
};

} // namespace OpTiling
} // namespace Transformer
} // namespace Ops