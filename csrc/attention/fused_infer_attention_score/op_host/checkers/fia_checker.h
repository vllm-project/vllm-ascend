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
 * \file fia_checker.h
 * \brief
 */

#ifndef FIA_CHECKER_H
#define FIA_CHECKER_H

#include <map>
#include <numeric>
#include "tiling/tiling_api.h"
#include "base_checker.h"

#include "./actual_seq_len_checker.h"
#include "./base_checker.h"
#include "./dequant_checker.h"
#include "./learnable_sink_checker.h"
#include "./left_padding_checker.h"
#include "./mask_checker.h"
#include "./paged_attention_checker.h"
#include "./post_quant_checker.h"
#include "./pse_checker.h"
#include "./rope_checker.h"
#include "./common_checker.h"
#include "./softmax_lse_checker.h"
#include "./system_prefix_checker.h"

namespace optiling {
class FIAChecker {
public:
    FIAChecker() = default;
    ~FIAChecker() = default;
    ge::graphStatus Init(const FiaTilingInfo &fiaInfo);
    ge::graphStatus Process(const FiaTilingInfo &fiaInfo);

    ge::graphStatus CheckSinglePara(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckParaExistence(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckCrossFeature(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckMultiParaConsistency(const FiaTilingInfo &fiaInfo);

private:
    std::unique_ptr<ActualSeqLenChecker> actualSeqLenChecker_;
    std::unique_ptr<DequantChecker> dequantChecker_;
    std::unique_ptr<LearnableSinkChecker> learnableSinkChecker_;
    std::unique_ptr<LeftPaddingChecker> leftPaddingChecker_;
    std::unique_ptr<MaskChecker> maskChecker_;
    std::unique_ptr<PagedAttentionChecker> pagedAttentionChecker_;
    std::unique_ptr<PostQuantChecker> postQuantChecker_;
    std::unique_ptr<PSEChecker> pseChecker_;
    std::unique_ptr<RopeChecker> ropeChecker_;
    std::unique_ptr<CommonChecker> commonChecker_;
    std::unique_ptr<SoftmaxLSEChecker> softmaxLSEChecker_;
    std::unique_ptr<SystemPrefixChecker> systemPrefixChecker_;

    bool enableNonQuant_ = false;
    bool enableFullQuant_ = false;
    bool enableAntiQuant_ = false;
};

} // namespace optiling
#endif // FIA_CHECKER_H