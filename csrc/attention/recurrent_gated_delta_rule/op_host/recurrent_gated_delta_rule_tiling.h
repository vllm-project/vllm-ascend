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
 * \file recurrent_gated_delta_rule_tiling.h
 * \brief
 */
#ifndef __OP_HOST_RECURRENT_GETED_DELTA_RULE_TILING_H__
#define __OP_HOST_RECURRENT_GETED_DELTA_RULE_TILING_H__
#include <tiling/tiling_api.h>
#include "register/tilingdata_base.h"
#include "tiling_base/tiling_base.h"
#include "../op_kernel/recurrent_gated_delta_rule_tiling_data.h"

namespace optiling {
using namespace RecurrentGatedDeltaRule;

struct RecurrentGatedDeltaRuleCompileInfo {
    uint64_t aivNum{0UL};
    uint64_t ubSize{0UL};
};

struct RecurrentGatedDeltaRuleInfo {
public:
    int64_t usedCoreNum = 0;
    const char *opName = "RecurrentGatedDeltaRule";
};

class RecurrentGatedDeltaRuleTiling : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit RecurrentGatedDeltaRuleTiling(gert::TilingContext *context) : Ops::Transformer::OpTiling::TilingBaseClass(context)
    {
        InitCompileInfo();
    };
    ~RecurrentGatedDeltaRuleTiling() override = default;

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

    //Host tiling rule-chain engine: compose shape/UB steps by ordered rules.
    using HostRuleFn = ge::graphStatus (RecurrentGatedDeltaRuleTiling::*)();
    struct UbCalcContext {
        int64_t ubSize = 0;
        int64_t aNv = 0;
        int64_t aDv = 0;
        int64_t aDk = 0;
        int64_t fixedUbBytes = 0;
        int64_t workingUbBytes = 0;
        int64_t coeff = 0;
    };
    struct BufferProfile {
        uint32_t stateOutBufferNum = 1;
        uint32_t attnOutBufferNum = 1;
        uint32_t vStep = 0;
        uint32_t repeatTime = 0;
        bool valid = false;
    };

    ge::graphStatus CheckContext();
    ge::graphStatus AnalyzeDtype();
    ge::graphStatus AnalyzeShapes();
    ge::graphStatus CalUbSize();
    ge::graphStatus GetScale();
    ge::graphStatus GetOptionalInput();
    ge::graphStatus AnalyzeFormat();
    //Host tiling refactor helpers: split shape validation/fill and UB calculation.
    ge::graphStatus CheckShapeDimAndRelation(const gert::Shape &queryShape, const gert::Shape &keyShape,
                                             const gert::Shape &valueShape, const gert::Shape &betaShape,
                                             const gert::Shape &stateShape, const gert::Shape &cuSeqlensShape,
                                             const gert::Shape &ssmStateShape);
    void FillTilingShapeData(const gert::Shape &queryShape, const gert::Shape &valueShape, const gert::Shape &stateShape,
                             const gert::Shape &cuSeqlensShape);
    ge::graphStatus CheckShapeValueRangeAndRule();
    void UpdateDynamicBlockDimByTaskUnits();
    int64_t CalcFixedUbBytes(int64_t aNv, int64_t aDv, int64_t aDk) const;
    int64_t CalcWorkingUbBytes(int64_t aNv, int64_t aDv, int64_t aDk) const;
    int64_t CalcVStepCoeff(int64_t aDk, uint32_t stateOutBufferNum, uint32_t attnOutBufferNum) const;
    bool EvaluateBufferProfile(int64_t ubSize, int64_t usedUbBytes, int64_t aDk, uint32_t stateOutBufferNum,
                               uint32_t attnOutBufferNum, BufferProfile &profile) const;
    bool IsBetterProfile(const BufferProfile &candidate, const BufferProfile &current) const;
    ge::graphStatus FinalizeVStepFromUb(int64_t ubSize, int64_t usedUbBytes, int64_t coeff);
    ge::graphStatus RuleCheckShapeDimAndRelation();
    ge::graphStatus RuleFillTilingShapeData();
    ge::graphStatus RuleCheckShapeValueRangeAndRule();
    ge::graphStatus RuleUpdateDynamicBlockDimByTaskUnits();
    ge::graphStatus RuleInitUbCalcContext();
    ge::graphStatus RuleCalcFixedUbBytes();
    ge::graphStatus RuleCalcWorkingUbBytes();
    ge::graphStatus RuleCalcVStepCoeff();
    ge::graphStatus RuleFinalizeVStepFromUb();

    bool CheckDimEqual(const gert::Shape a, const int64_t dimA, gert::Shape b, const int64_t dimB, const std::string &nameA,
                       const std::string &nameB, const std::string &dimDesc);
    bool CheckDim(const gert::Shape shape, const size_t dim, const std::string &dimDesc);
    bool CheckFormat(ge::Format format, const std::string &Desc);

    RecurrentGatedDeltaRuleCompileInfo compileInfo_;
    RecurrentGatedDeltaRuleTilingData tilingData_;
    RecurrentGatedDeltaRuleInfo inputParams_;
    UbCalcContext ubCalcCtx_;
    ge::DataType inputDtype_{ge::DT_BF16};
    int64_t inTypeSize_{2}; // sizeof(inType): 2 for fp16/bf16, 4 for fp32
};

} // namespace optiling
#endif // __OP_HOST_RECURRENT_GETED_DELTA_RULE_TILING_H__
