/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file matmul_base_tiling.h
 * \brief
 */
#ifndef __OP_HOST_MATMUL_BASE_TILING_H__
#define __OP_HOST_MATMUL_BASE_TILING_H__

#include "exe_graph/runtime/tiling_context.h"
#include "mc2_log_compat.h"
#include "matmul_tiling_cfg.h"

namespace optiling {
class Mc2MatMulBaseTiling {
public:
    Mc2MatMulBaseTiling(gert::TilingContext *context, Mc2MatMulTilingCfg &cfg) : context_(context), cfg_(cfg) {};

    virtual ~Mc2MatMulBaseTiling() {};

    ge::graphStatus DoTiling()
    {
        auto ret = GetShapeAttrsInfo();
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
        ret = AdjustOpTiling();
        if (ret != ge::GRAPH_SUCCESS) {
            return ret;
        }
        ret = PostTiling();
        if (ret != ge::GRAPH_SUCCESS) {
            return ret;
        }
        DumpTilingInfo();
        return ge::GRAPH_SUCCESS;
    }

protected:
    // 1、获取额外INPUT/OUTPUT/ATTR信息
    virtual ge::graphStatus GetShapeAttrsInfo() = 0;
    // 2、判断是否在策略支持范围
    virtual bool IsCapable() = 0;
    // 3、计算数据切分TilingData
    virtual ge::graphStatus DoOpTiling() = 0;
    // 4、调整tiling，进一步保护措施
    virtual ge::graphStatus AdjustOpTiling() = 0;
    // 5、计算TilingKey
    virtual uint64_t GetTilingKey() const = 0;
    // 6、计算Workspace 大小
    virtual std::vector<size_t> GetWorkspaceSize() const = 0;
    // 7、计算blockDim大小
    virtual uint64_t GetBlockDim() const = 0;
    // 8、计算tiling, 保存结果
    virtual ge::graphStatus PostTiling() = 0;
    // 9、Dump Tiling数据
    virtual void DumpTilingInfo()
    {
        auto buf = (uint32_t *)context_->GetRawTilingData()->GetData();
        auto bufLen = context_->GetRawTilingData()->GetDataSize();
        std::ostringstream oss;
        oss << "Start to dump tiling info. tilingkey:" << GetTilingKey() << ", tiling data size:" << bufLen
            << ", content:";
        for (size_t i = 0; i < bufLen / sizeof(uint32_t); i++) {
            oss << *(buf + i) << ",";
            if (oss.str().length() > 640) { // Split according to 640 to avoid truncation
                OP_LOGI(context_, "%s", oss.str().c_str());
                oss.str("");
            }
        }
        OP_LOGI(context_, "%s", oss.str().c_str());
    }

protected:
    gert::TilingContext *context_ = nullptr;
    Mc2MatMulTilingCfg &cfg_;
};
} // namespace optiling

#endif // __OP_HOST_MATMUL_BASE_TILING_H__
