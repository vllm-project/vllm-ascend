/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

/*!
 * \file recurrent_kda_tiling.h
 * \brief
 */
#ifndef __OP_HOST_RECURRENT_KDA_TILING_H__
#define __OP_HOST_RECURRENT_KDA_TILING_H__
#include <tiling/tiling_api.h>
#include "register/tilingdata_base.h"
#include "tiling_base/tiling_base.h"
#include "err/ops_err.h"
#include "../op_kernel/recurrent_kda_tiling_data.h"
#include "recurrent_kda_tiling_processor.h"

namespace optiling {
using namespace RecurrentKda;

struct RecurrentKdaCompileInfo {
    uint64_t aivNum{0UL};
    uint64_t ubSize{0UL};
};

struct RecurrentKdaInfo {
public:
    int64_t usedCoreNum = 0;
    const char *opName = "RecurrentKda";
};

class RecurrentKdaTiling : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit RecurrentKdaTiling(gert::TilingContext *context) : Ops::Transformer::OpTiling::TilingBaseClass(context)
    {
        InitCompileInfo();
    };
    ~RecurrentKdaTiling() override = default;

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
    RecurrentKdaTilingContext BuildProcessorContext() const;

    ge::graphStatus CheckContext();
    ge::graphStatus AnalyzeDtype();
    ge::graphStatus AnalyzeShapes();
    ge::graphStatus CalUbSize();
    ge::graphStatus GetAttrsInfo();
    ge::graphStatus GetOptionalInput();
    ge::graphStatus AnalyzeFormat();
    bool CheckFormat(ge::Format format, const std::string &Desc);

    RecurrentKdaCompileInfo compileInfo_;
    RecurrentKdaTilingData tilingData_;
    RecurrentKdaInfo inputParams_;
};

} // namespace optiling
#endif // __OP_HOST_RECURRENT_KDA_TILING_H__
