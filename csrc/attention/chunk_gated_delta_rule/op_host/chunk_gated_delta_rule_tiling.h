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
 * \file chunk_gated_delta_rule_tiling.h
 * \brief
 */
#ifndef __OP_HOST_CHUNK_GATED_DELTA_RULE_TILING_H__
#define __OP_HOST_CHUNK_GATED_DELTA_RULE_TILING_H__

#include <tiling/tiling_api.h>
#include "register/tilingdata_base.h"
#include "tiling_base/tiling_base.h"
#include "error/ops_error.h"
#include "../op_kernel/chunk_gated_delta_rule_tiling_data.h"

namespace optiling {

struct ChunkGatedDeltaRuleCompileInfo {
    uint64_t aivNum{0UL};
    uint64_t aicNum{0UL};
    uint64_t ubSize{0UL};
};

struct ChunkGatedDeltaRuleInfo {
public:
    int64_t usedCoreNum = 0;
    const char *opName = "ChunkGatedDeltaRule";
};

class ChunkGatedDeltaRuleTiling : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit ChunkGatedDeltaRuleTiling(gert::TilingContext *context)
        : Ops::Transformer::OpTiling::TilingBaseClass(context)
    {
        InitCompileInfo();
    };
    ~ChunkGatedDeltaRuleTiling() override = default;

protected:
    bool IsCapable() override
    {
        return true;
    }
    // 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
    ge::graphStatus GetPlatformInfo() override;

    // 2、获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo() override;

    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;

    // 4、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override;

    // 5、计算TilingKey
    uint64_t GetTilingKey() const override;

    // 6、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override;

    // 7、保存Tiling数据
    ge::graphStatus PostTiling() override;

protected:
    void InitCompileInfo();
    void PrintTilingData();

    ge::graphStatus CheckContext();
    ge::graphStatus AnalyzeDtype();
    ge::graphStatus AnalyzeShapes();
    ge::graphStatus CheckDerivedDimConstraints();
    ge::graphStatus GetScale();
    ge::graphStatus GetOptionalInput();
    ge::graphStatus AnalyzeFormat();
    ge::graphStatus DoMatmulTiling();
    ge::graphStatus SetScheduleConfig();
    ge::graphStatus CheckExpectedShapes(const gert::Shape &queryShape, const gert::Shape &keyShape,
                                        const gert::Shape &valueShape, const gert::Shape &betaShape,
                                        const gert::Shape &stateShape, const gert::Shape &actualSeqLengthsShape,
                                        const gert::Shape &outShape, const gert::Shape &finalStateShape,
                                        const gert::Shape *gShape);
    bool CheckDim(const gert::Shape &shape, const size_t dim, const std::string &dimDesc);
    bool CheckFormat(const gert::CompileTimeTensorDesc *desc, const std::string &name);

    ChunkGatedDeltaRuleCompileInfo compileInfo_;
    ChunkGatedDeltaRule::ChunkGatedDeltaRuleTilingData tilingData_;
    ChunkGatedDeltaRuleInfo inputParams_;
};

} // namespace optiling
#endif // __OP_HOST_CHUNK_GATED_DELTA_RULE_TILING_H__
