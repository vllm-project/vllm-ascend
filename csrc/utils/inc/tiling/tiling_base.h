/**
 * Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
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
#include <tiling/platform/platform_ascendc.h>
#include "log/ops_log.h"

#ifdef ASCENDC_OP_TEST
#define ASCENDC_EXTERN_C extern "C"
#else
#define ASCENDC_EXTERN_C
#endif

namespace optiling {

struct AiCoreParams {
    uint64_t ubSize;
    uint64_t blockDim;
    uint64_t aicNum;
    uint64_t l1Size;
    uint64_t l0aSize;
    uint64_t l0bSize;
    uint64_t l0cSize;
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
};

class TilingBaseClass {
public:
    TilingBaseClass() = default;

    explicit TilingBaseClass(gert::TilingContext *context) : context_(context)
    {
    }

    virtual ~TilingBaseClass() = default;

    // Tiling执行框架
    //     1、GRAPH_SUCCESS: 成功，并且不需要继续执行后续Tiling类的实现
    //     2、GRAPH_FAILED: 失败，中止整个Tiling流程
    //     3、GRAPH_PARAM_INVALID: 本类不支持，需要继续往下执行其他Tiling类的实现
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

    // 更新 context
    virtual void Reset(gert::TilingContext *context)
    {
        context_ = context;
    }

protected:
    virtual bool IsCapable() = 0;
    // 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
    virtual ge::graphStatus GetPlatformInfo() = 0;
    // 2、获取INPUT/OUTPUT/ATTR信息
    virtual ge::graphStatus GetShapeAttrsInfo() = 0;
    // 3、计算数据切分TilingData
    virtual ge::graphStatus DoOpTiling() = 0;
    // 4、计算高阶API的TilingData
    virtual ge::graphStatus DoLibApiTiling() = 0;
    // 5、计算TilingKey
    [[nodiscard]] virtual uint64_t GetTilingKey() const = 0;
    // 6、计算Workspace 大小
    virtual ge::graphStatus GetWorkspaceSize() = 0;
    // 7、保存Tiling数据
    virtual ge::graphStatus PostTiling() = 0;
    // 8、Dump Tiling数据
    virtual void DumpTilingInfo()
    {
        int32_t enable = AlogCheckDebugLevel(static_cast<int32_t>(OP), DLOG_DEBUG);
        if (enable != 1) {
            return;
        }
        auto buf = (uint32_t *)context_->GetRawTilingData()->GetData();
        auto bufLen = context_->GetRawTilingData()->GetDataSize();
        std::ostringstream oss;
        oss << "Start to dump tiling info. tilingkey:" << GetTilingKey() << ", tiling data size:" << bufLen
            << ", content:";
        for (size_t i = 0; i < bufLen / sizeof(uint32_t); i++) {
            oss << *(buf + i) << ",";
            if (oss.str().length() > 640) { // Split according to 640 to avoid truncation
                OPS_LOG_D(context_, "%s", oss.str().c_str());
                oss.str("");
            }
        }
        OPS_LOG_D(context_, "%s", oss.str().c_str());
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

    template <typename T> [[nodiscard]] std::string GetShapeDebugStr(const T &shape) const
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

    [[nodiscard]] std::string GetTensorDebugStr(const gert::StorageShape *shape,
                                                const gert::CompileTimeTensorDesc *tensor)
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
        auto data = reinterpret_cast<const int32_t *>(rawTilingData->GetData());
        size_t len = rawTilingDataSize / sizeof(int32_t);
        std::ostringstream oss;
        for (size_t i = 0; i < len; i++) {
            oss << data[i] << ", ";
        }
        return oss.str();
    }

protected:
    gert::TilingContext *context_ = nullptr;
    std::unique_ptr<platform_ascendc::PlatformAscendC> ascendcPlatform_{nullptr};
    uint32_t blockDim_{0};
    uint64_t workspaceSize_{0};
    uint64_t tilingKey_{0};
    AiCoreParams aicoreParams_{0, 0, 0, 0, 0, 0, 0};
};

} // namespace optiling
