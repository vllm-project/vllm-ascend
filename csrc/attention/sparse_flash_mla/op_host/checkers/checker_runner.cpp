/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "checker_runner.h"
#include "common_checker_sparse_flash_mla.h"
#include "mask_checker_sparse_flash_mla.h"
#include "metadata_checker_sparse_flash_mla.h"
#include "paged_attention_checker_sparse_flash_mla.h"
#include "seq_len_checker_sparse_flash_mla.h"
#include "sinks_checker_sparse_flash_mla.h"
#include "softmax_lse_checker_sparse_flash_mla.h"
#include "sparse_compression_checker.h"

namespace optiling {
namespace sparse_mla_checker {

void CheckerRunner::Add(std::unique_ptr<BaseChecker> checker)
{
    checkers_.push_back(std::move(checker));
}

ge::graphStatus CheckerRunner::Run(CheckMethod method, const CheckContext &context) const
{
    for (const auto &checker : checkers_) {
        if ((checker.get()->*method)(context) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckerRunner::Process(const CheckContext &context) const
{
    if (Run(&BaseChecker::CheckSinglePara, context) != ge::GRAPH_SUCCESS ||
        Run(&BaseChecker::CheckParaExistence, context) != ge::GRAPH_SUCCESS ||
        Run(&BaseChecker::CheckFeature, context) != ge::GRAPH_SUCCESS ||
        Run(&BaseChecker::CheckMultiPara, context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

void RegisterCommonCheckers(CheckerRunner &runner)
{
    runner.Add(std::make_unique<CommonChecker>());
    runner.Add(std::make_unique<SeqLenChecker>());
    runner.Add(std::make_unique<SparseCompressionChecker>());
    runner.Add(std::make_unique<MaskChecker>());
    runner.Add(std::make_unique<PagedAttentionChecker>());
    runner.Add(std::make_unique<SinksChecker>());
    runner.Add(std::make_unique<MetadataChecker>());
    runner.Add(std::make_unique<SoftmaxLseChecker>());
}

} // namespace sparse_mla_checker
} // namespace optiling
