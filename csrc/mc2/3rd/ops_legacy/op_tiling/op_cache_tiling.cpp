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
 * \file op_cache_tiling.cpp
 * \brief
 */

#include "op_cache_tiling.h"
#ifdef BUILD_OPEN_PROJECT
#include "legacy_common_manager.h"
#include "mc2_log_compat.h"
#endif

namespace optiling {
#ifdef BUILD_OPEN_PROJECT
bool TilingPrepareForOpCache(gert::TilingContext *context)
{
    using FuncType = bool (*)(gert::TilingContext *);
    const char *symbolName = "LegacyTilingPrepareForOpCache";
    static FuncType func = Ops::MC2::LegacyCommonMgr::GetInstance().GetFunc<FuncType>(symbolName);
    if (func == nullptr) {
        OP_LOGW("LegacyCommonMgr", "dest func %s pointer is null.", symbolName);
        return false;
    } else {
        return func(context);
    }
}

bool TilingPrepareForOpCache(gert::TilingParseContext *context)
{
    using FuncType = bool (*)(gert::TilingParseContext *);
    const char *symbolName = "LegacyTilingParsePrepareForOpCache";
    static FuncType func = Ops::MC2::LegacyCommonMgr::GetInstance().GetFunc<FuncType>(symbolName);
    if (func == nullptr) {
        OP_LOGW("LegacyCommonMgr", "dest func %s pointer is null.", symbolName);
        return false;
    } else {
        return func(context);
    }
}

bool GenTiling(const std::string &op_type, const BatchmatmulCompileParas &compile_params,
               BatchmatmulRunParas &run_params, CacheTilingData &tiling, gert::TilingContext *context)
{
    using FuncType = bool (*)(const std::string &, const BatchmatmulCompileParas &, BatchmatmulRunParas &,
                              CacheTilingData &, gert::TilingContext *);
    const char *symbolName = "LegacyGenTbeMatmulTiling";
    static FuncType func = Ops::MC2::LegacyCommonMgr::GetInstance().GetFunc<FuncType>(symbolName);
    if (func == nullptr) {
        OP_LOGW("LegacyCommonMgr", "dest func %s pointer is null.", symbolName);
        return false;
    } else {
        return func(op_type, compile_params, run_params, tiling, context);
    }
}

bool CheckSupportConditionQbmm(QbmmType type, QuantBatchMatmulRunParas &inputParams, uint64_t aicNum,
                               bool supportL0c2Out)
{
    using FuncType = bool (*)(QbmmType, QuantBatchMatmulRunParas &, uint64_t, bool);
    const char *symbolName = "LegacyCheckSupportConditionQbmm";
    static FuncType func = Ops::MC2::LegacyCommonMgr::GetInstance().GetFunc<FuncType>(symbolName);
    if (func == nullptr) {
        OP_LOGW("LegacyCommonMgr", "dest func %s pointer is null.", symbolName);
        return false;
    } else {
        return func(type, inputParams, aicNum, supportL0c2Out);
    }
}

bool GenWqbmmTiling(const std::string &op_type, const WeightQuantBatchMatmulCacheTilingParas &compile_params,
                    WeightQuantBatchMatmulCacheTilingData &cacheTiling)
{
    using FuncType = bool (*)(const std::string &, const WeightQuantBatchMatmulCacheTilingParas &,
                              WeightQuantBatchMatmulCacheTilingData &);
    const char *symbolName = "LegacyGenWqbmmTiling";
    static FuncType func = Ops::MC2::LegacyCommonMgr::GetInstance().GetFunc<FuncType>(symbolName);
    if (func == nullptr) {
        OP_LOGW("LegacyCommonMgr", "dest func %s pointer is null.", symbolName);
        return false;
    } else {
        return func(op_type, compile_params, cacheTiling);
    }
}
#else
bool TilingPrepareForOpCache(gert::TilingContext * /*context*/)
{
    return true;
}

bool TilingPrepareForOpCache(gert::TilingParseContext * /*context*/)
{
    return true;
}

bool GenTiling(const std::string & /*op_type*/, const BatchmatmulCompileParas & /*compile_params*/,
               BatchmatmulRunParas & /*run_params*/, CacheTilingData & /*tiling*/, gert::TilingContext * /*context*/)
{
    return true;
}

bool CheckSupportConditionQbmm(QbmmType /*type*/, QuantBatchMatmulRunParas & /*inputParams*/, uint64_t /*aicNum*/,
                               bool /*supportL0c2Out*/)
{
    return true;
}

bool GenWqbmmTiling(const std::string & /*op_type*/, const WeightQuantBatchMatmulCacheTilingParas & /*compile_params*/,
                    WeightQuantBatchMatmulCacheTilingData & /*cacheTiling*/)
{
    return true;
}
#endif
} // namespace optiling
