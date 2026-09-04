/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

/*!
 * \file fia_checker.cpp
 * \brief
 */

#include <map>
#include <numeric>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "register/op_def_registry.h"
#include "fia_checker.h"

namespace optiling {
using std::map;
using std::string;
using std::pair;
using namespace ge;
using namespace AscendC;

ge::graphStatus FIAChecker::Init(const FiaTilingInfo &fiaInfo)
{
    if (fiaInfo.inputQType != fiaInfo.inputKvType) {
        enableAntiQuant_ = true;
    } else if (fiaInfo.inputQType == ge::DT_FLOAT16 || fiaInfo.inputQType == ge::DT_BF16) {
        enableNonQuant_ = true;
    } else {
        enableFullQuant_ = true;
    }

    actualSeqLenChecker_ = std::make_unique<ActualSeqLenChecker>(enableNonQuant_, enableFullQuant_, enableAntiQuant_);
    dequantChecker_ = std::make_unique<DequantChecker>(enableNonQuant_, enableFullQuant_, enableAntiQuant_);
    learnableSinkChecker_ = std::make_unique<LearnableSinkChecker>(enableNonQuant_, enableFullQuant_, enableAntiQuant_);
    leftPaddingChecker_ = std::make_unique<LeftPaddingChecker>(enableNonQuant_, enableFullQuant_, enableAntiQuant_);
    maskChecker_ = std::make_unique<MaskChecker>(enableNonQuant_, enableFullQuant_, enableAntiQuant_);
    pagedAttentionChecker_ =
        std::make_unique<PagedAttentionChecker>(enableNonQuant_, enableFullQuant_, enableAntiQuant_);
    postQuantChecker_ = std::make_unique<PostQuantChecker>(enableNonQuant_, enableFullQuant_, enableAntiQuant_);
    pseChecker_ = std::make_unique<PSEChecker>(enableNonQuant_, enableFullQuant_, enableAntiQuant_);
    ropeChecker_ = std::make_unique<RopeChecker>(enableNonQuant_, enableFullQuant_, enableAntiQuant_);
    commonChecker_ = std::make_unique<CommonChecker>(enableNonQuant_, enableFullQuant_, enableAntiQuant_);
    softmaxLSEChecker_ = std::make_unique<SoftmaxLSEChecker>(enableNonQuant_, enableFullQuant_, enableAntiQuant_);
    systemPrefixChecker_ = std::make_unique<SystemPrefixChecker>(enableNonQuant_, enableFullQuant_, enableAntiQuant_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FIAChecker::CheckSinglePara(const FiaTilingInfo &fiaInfo)
{
    if (ge::GRAPH_SUCCESS != commonChecker_->CheckSinglePara(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != maskChecker_->CheckSinglePara(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != actualSeqLenChecker_->CheckSinglePara(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != pagedAttentionChecker_->CheckSinglePara(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != postQuantChecker_->CheckSinglePara(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != ropeChecker_->CheckSinglePara(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != pseChecker_->CheckSinglePara(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != leftPaddingChecker_->CheckSinglePara(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != systemPrefixChecker_->CheckSinglePara(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != softmaxLSEChecker_->CheckSinglePara(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != learnableSinkChecker_->CheckSinglePara(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != dequantChecker_->CheckSinglePara(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FIAChecker::CheckParaExistence(const FiaTilingInfo &fiaInfo)
{
    if (ge::GRAPH_SUCCESS != commonChecker_->CheckParaExistence(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != maskChecker_->CheckParaExistence(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != actualSeqLenChecker_->CheckParaExistence(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != pagedAttentionChecker_->CheckParaExistence(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != postQuantChecker_->CheckParaExistence(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != ropeChecker_->CheckParaExistence(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != pseChecker_->CheckParaExistence(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != leftPaddingChecker_->CheckParaExistence(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != systemPrefixChecker_->CheckParaExistence(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != softmaxLSEChecker_->CheckParaExistence(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != learnableSinkChecker_->CheckParaExistence(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != dequantChecker_->CheckParaExistence(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FIAChecker::CheckCrossFeature(const FiaTilingInfo &fiaInfo)
{
    if (ge::GRAPH_SUCCESS != commonChecker_->CheckCrossFeature(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != maskChecker_->CheckCrossFeature(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != actualSeqLenChecker_->CheckCrossFeature(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != pagedAttentionChecker_->CheckCrossFeature(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != postQuantChecker_->CheckCrossFeature(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != ropeChecker_->CheckCrossFeature(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != pseChecker_->CheckCrossFeature(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != leftPaddingChecker_->CheckCrossFeature(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != systemPrefixChecker_->CheckCrossFeature(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != softmaxLSEChecker_->CheckCrossFeature(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != learnableSinkChecker_->CheckCrossFeature(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != dequantChecker_->CheckCrossFeature(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FIAChecker::CheckMultiParaConsistency(const FiaTilingInfo &fiaInfo)
{
    if (ge::GRAPH_SUCCESS != commonChecker_->CheckMultiParaConsistency(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != maskChecker_->CheckMultiParaConsistency(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != actualSeqLenChecker_->CheckMultiParaConsistency(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != pagedAttentionChecker_->CheckMultiParaConsistency(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != postQuantChecker_->CheckMultiParaConsistency(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != ropeChecker_->CheckMultiParaConsistency(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != pseChecker_->CheckMultiParaConsistency(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != leftPaddingChecker_->CheckMultiParaConsistency(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != systemPrefixChecker_->CheckMultiParaConsistency(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != softmaxLSEChecker_->CheckMultiParaConsistency(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != learnableSinkChecker_->CheckMultiParaConsistency(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != dequantChecker_->CheckMultiParaConsistency(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FIAChecker::Process(const FiaTilingInfo &fiaInfo)
{
    if (fiaInfo.emptyTensorFlag) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckSinglePara(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != CheckParaExistence(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != CheckCrossFeature(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != CheckMultiParaConsistency(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling
