/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file recurrent_gated_delta_rule_v310_tiling.h
 * \brief
 */
#ifndef __OP_HOST_RECURRENT_GATED_DELTA_RULE_V310_TILING_H__
#define __OP_HOST_RECURRENT_GATED_DELTA_RULE_V310_TILING_H__

#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "../tiling_base/tiling_base.h"
#include "../op_kernel/recurrent_gated_delta_rule_v310_tiling_data.h"

namespace optiling {
using namespace RecurrentGatedDeltaRuleV310;

struct RecurrentGatedDeltaRuleV310CompileInfo {
    uint64_t aivNum{0UL};
    uint64_t ubSize{0UL};
};

struct RecurrentGatedDeltaRuleV310Info {
public:
    int64_t usedCoreNum = 0;
    const char *opName = "RecurrentGatedDeltaRuleV310";
};

class RecurrentGatedDeltaRuleV310Tiling : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit RecurrentGatedDeltaRuleV310Tiling(gert::TilingContext *context) : Ops::Transformer::OpTiling::TilingBaseClass(context)
    {
        InitCompileInfo();
    };
    ~RecurrentGatedDeltaRuleV310Tiling() override = default;

protected:
    bool IsCapable() override
    {
        return true;
    }
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

protected:
    void InitCompileInfo();
    void PrintTilingData();

    //Host tiling rule-chain engine: compose shape/UB steps by ordered rules.
    using HostRuleFn = ge::graphStatus (RecurrentGatedDeltaRuleV310Tiling::*)();
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
        uint32_t stateOutBufferNum;
        uint32_t attnOutBufferNum;
        uint32_t vStep;
        uint32_t repeatTime;
        bool valid;

        BufferProfile() : stateOutBufferNum(1), attnOutBufferNum(1), vStep(0), repeatTime(0), valid(false) {}
        BufferProfile(uint32_t state, uint32_t attn, uint32_t v, uint32_t repeat, bool vld)
            : stateOutBufferNum(state), attnOutBufferNum(attn), vStep(v), repeatTime(repeat), valid(vld) {}
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

    RecurrentGatedDeltaRuleV310CompileInfo compileInfo_;
    RecurrentGatedDeltaRuleV310TilingData tilingData_;
    RecurrentGatedDeltaRuleV310Info inputParams_;
    UbCalcContext ubCalcCtx_;
    ge::DataType inputDtype_{ge::DT_FLOAT16};
};

} // namespace optiling
#endif // __OP_HOST_RECURRENT_GATED_DELTA_RULE_V310_TILING_H__
