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
 * \file matmul_v3_tiling_advanced.cc
 * \brief
 */

#include "matmul_v3_tiling_advanced.h"

#include "register/op_def_registry.h"
#include "common/op_host/math_util.h"
#include "common/op_host/op_tiling/debug_tiling.h"
#include "./matmul_tiling_registry.h"
#include "./matmul_tiling_cfg.h"
#include "matmul_v3_compile_info_advanced.h"
#include "matmul_v3_tiling_strategy.h"

namespace {
using namespace optiling;
using namespace optiling::mc2_matmul_v3_advanced;

constexpr uint64_t ONE_BATCH_DIM = 1UL;
constexpr uint64_t TWO_BATCH_DIM = 2UL;
constexpr uint64_t THREE_BATCH_DIM = 3UL;
constexpr uint64_t FOUR_BATCH_DIM = 4UL;
constexpr size_t HF32_ATTR_NUM = 4UL;
constexpr size_t HF32_ATTR_INDEX = 3UL;
constexpr size_t BIAS_IDX = 2UL;

inline void GetFormat(const gert::TilingContext &context, Mc2MatMulV3Args &args)
{
    ge::Format formatA = static_cast<ge::Format>(ge::GetPrimaryFormat(context.GetInputDesc(0)->GetStorageFormat()));
    ge::Format formatB = static_cast<ge::Format>(ge::GetPrimaryFormat(context.GetInputDesc(1)->GetStorageFormat()));
    ge::Format formatOut = static_cast<ge::Format>(ge::GetPrimaryFormat(context.GetOutputDesc(0)->GetStorageFormat()));
    args.aFormat = (formatA != ge::FORMAT_FRACTAL_NZ) ? ge::FORMAT_ND : formatA;
    args.bFormat = (formatB != ge::FORMAT_FRACTAL_NZ) ? ge::FORMAT_ND : formatB;
    args.outFormat = (formatOut != ge::FORMAT_FRACTAL_NZ) ? ge::FORMAT_ND : formatOut;
}

inline void GetDtype(const gert::TilingContext &context, Mc2MatMulV3Args &args)
{
    args.aType = context.GetInputDesc(0)->GetDataType();
    args.bType = context.GetInputDesc(1)->GetDataType();
    args.cType = context.GetOutputDesc(0)->GetDataType();
    if (args.hasBias) {
        args.biasType = context.GetInputDesc(BIAS_IDX)->GetDataType();
    }
    // op_impl_mode_enum: 0x1: default 0x2: high_performance 0x4: high_precision 0x8: super_performance
    // 0x10: support_of_bound_index 0x20: enable_float_32_execution 0x40: enable_hi_float_32_execution
    if (context.GetAttrs()->GetAttrNum() >= HF32_ATTR_NUM) {
        args.isHf32 = *((context.GetAttrs())->GetAttrPointer<bool>(HF32_ATTR_INDEX));
    }
    args.aDtypeSize = ge::GetSizeByDataType(args.aType);
    args.bDtypeSize = ge::GetSizeByDataType(args.bType);
    OP_LOGD(args.opName, "Mc2MatMulV3 Hf32 flag is: %d", args.isHf32);
}

ge::graphStatus GetInputDims(const gert::Shape &storageShape, const gert::Shape &oriShape, uint64_t dtypeSize,
                             ge::Format format, int64_t (&dims)[TWO_BATCH_DIM])
{
    const size_t dimNum = storageShape.GetDimNum();
    const size_t oriDimNum = oriShape.GetDimNum();
    dims[0] = oriShape[oriDimNum - TWO_BATCH_DIM];
    dims[1] = oriShape[oriDimNum - ONE_BATCH_DIM];
    if (format == ge::FORMAT_ND) {
        if (dimNum < TWO_BATCH_DIM) {
            return ge::GRAPH_FAILED;
        }
    } else {
        if (dimNum < FOUR_BATCH_DIM) {
            return ge::GRAPH_FAILED;
        }
        int64_t storageShape0 = storageShape[dimNum - THREE_BATCH_DIM] * storageShape[dimNum - TWO_BATCH_DIM];
        int64_t storageShape1 = storageShape[dimNum - FOUR_BATCH_DIM] * storageShape[dimNum - ONE_BATCH_DIM];
        if (ops::CeilAlign(dims[0], static_cast<int64_t>(BASIC_BLOCK_SIZE_16)) != storageShape0 ||
            ops::CeilAlign(dims[1], static_cast<int64_t>(BLOCK_BYTE_SIZE / dtypeSize)) != storageShape1) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                "Mc2MatMulV3", "x1 and x2", (std::to_string(dims[0]) + " and " + std::to_string(dims[1])).c_str(),
                "The shape of x1 and x2 must be equal to storage shape after NZ alignment.");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IsValidDtype(const Mc2MatMulV3Args &args)
{
    std::vector<ge::DataType> dtype = {args.aType, args.bType, args.cType};
    if (args.hasBias) {
        dtype.push_back(args.biasType);
    }
    const std::vector<std::vector<ge::DataType>> dtypeSuportList = {
        // x1,              x2,             y,              bias
        {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
        {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT},
        {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT},
        {ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT},
        {ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16} // david supports bias-bf16
    };
    for (auto &supported : dtypeSuportList) {
        if (std::equal(dtype.begin(), dtype.end(), supported.begin())) {
            return ge::GRAPH_SUCCESS;
        }
    }

    if (args.hasBias) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            args.opName, "x1, x2, output and bias", Ops::Base::ToString(args.aType).c_str(),
            "The dtypes of x1, x2, output and bias must be within the supported range.");
        return ge::GRAPH_FAILED;
    } else {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(args.opName, "x1, x2 and output", Ops::Base::ToString(args.aType).c_str(),
                                              "The dtypes of x1, x2 and output must be within the supported range.");
        return ge::GRAPH_FAILED;
    }
}

ge::graphStatus OpSpecificCheck(const gert::TilingContext &context, const Mc2MatMulV3Args &args)
{
    const bool isMatMulV3 = (strcmp(context.GetNodeType(), "Mc2MatMulV3") == 0);
    const bool isBatchMatMulV3 = (strcmp(context.GetNodeType(), "Mc2BatchMatMulV3") == 0);
    if (!isBatchMatMulV3 && !isMatMulV3) {
        // apply no additional checks for ops other than MMV3, BMMV3, for now
        return ge::GRAPH_SUCCESS;
    }

    // check input dim num equals to 2
    if (isMatMulV3) {
        auto isMatrix = [](const gert::Shape &oriShape) { return oriShape.GetDimNum() == TWO_BATCH_DIM; };
        if (!isMatrix(context.GetInputShape(0)->GetOriginShape()) ||
            !isMatrix(context.GetInputShape(1)->GetOriginShape())) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                args.opName, "x1 and x2",
                std::to_string(context.GetInputShape(0)->GetOriginShape().GetDimNum()).c_str(),
                "The shape dim of x1 and x2 must be 2D.");
            return ge::GRAPH_FAILED;
        }
    }

    // check bias
    if (args.hasBias) {
        const gert::Shape &biasShape = context.GetInputShape(BIAS_IDX)->GetOriginShape();
        const gert::Shape &cShape = context.GetOutputShape(0)->GetOriginShape();
        const int64_t biasValue = biasShape[biasShape.GetDimNum() - 1];
        const int64_t nOriValue = cShape[cShape.GetDimNum() - 1];
        if (biasValue != nOriValue) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(args.opName, "bias", std::to_string(biasValue).c_str(),
                                                  "The shape dim of bias must be equal to the n dimension of output.");
            return ge::GRAPH_FAILED;
        }
    }

    // dtype check
    return IsValidDtype(args);
}

ge::graphStatus GetShape(const gert::TilingContext &context, Mc2MatMulV3Args &args)
{
    // get transpose
    args.isATrans = *((context.GetAttrs())->GetAttrPointer<bool>(0));
    args.isBTrans = *((context.GetAttrs())->GetAttrPointer<bool>(1));

    // get (m, k, n)
    int64_t mkDims[TWO_BATCH_DIM];
    int64_t knDims[TWO_BATCH_DIM];
    if ((GetInputDims(context.GetInputShape(0)->GetStorageShape(), context.GetInputShape(0)->GetOriginShape(),
                      args.aDtypeSize, args.aFormat, mkDims) != ge::GRAPH_SUCCESS) ||
        (GetInputDims(context.GetInputShape(1)->GetStorageShape(), context.GetInputShape(1)->GetOriginShape(),
                      args.bDtypeSize, args.bFormat, knDims) != ge::GRAPH_SUCCESS)) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            args.opName, "x1 and x2", std::to_string(context.GetInputShape(0)->GetStorageShape().GetDimNum()).c_str(),
            "The shape dim of x1 and x2 is too small for given format.");
        return ge::GRAPH_FAILED;
    }
    uint64_t kIdxA = args.isATrans ? 0ULL : 1ULL;
    uint64_t kIdxB = args.isBTrans ? 1ULL : 0ULL;
    int64_t k = mkDims[kIdxA];
    if (k != knDims[kIdxB]) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(args.opName, "x1 and x2",
                                               (std::to_string(k) + " and " + std::to_string(knDims[kIdxB])).c_str(),
                                               "The shape of x1 and x2 must be the same at the k dimension.");
        return ge::GRAPH_FAILED;
    }
    uint64_t mIdx = args.isATrans ? 1ULL : 0ULL;
    uint64_t nIdx = args.isBTrans ? 0ULL : 1ULL;
    int64_t m = mkDims[mIdx];
    int64_t n = knDims[nIdx];
    args.mValue = static_cast<uint64_t>(m);
    args.kValue = static_cast<uint64_t>(k);
    args.nValue = static_cast<uint64_t>(n);
    auto isValidDimValue = [](int64_t dim) -> bool { return (dim > 0) && (dim <= INT32_MAX); };
    if (!isValidDimValue(args.mValue) || !isValidDimValue(args.kValue) || !isValidDimValue(args.nValue)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            args.opName, "m, k and n",
            (std::to_string(args.mValue) + ", " + std::to_string(args.kValue) + " and " + std::to_string(args.nValue))
                .c_str(),
            "The value of m, k and n must be positive and not exceed INT32_MAX.");
        return ge::GRAPH_FAILED;
    }
    // check output dim num
    const gert::Shape &cShape = context.GetOutputShape(0)->GetOriginShape();
    const size_t cDimNum = cShape.GetDimNum();
    if (cDimNum < TWO_BATCH_DIM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(args.opName, "output", (std::to_string(cDimNum) + "D").c_str(),
                                                 "The shape dim of output must be at least 2D.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace

namespace optiling {
namespace mc2_matmul_v3_advanced {
ge::graphStatus Mc2MatMulV3Tiling::GetArgs()
{
    GetFormat(*context_, args_);
    GetDtype(*context_, args_);
    if (GetShape(*context_, args_) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return OpSpecificCheck(*context_, args_);
}

ge::graphStatus Mc2MatMulV3Tiling::CheckArgs()
{
    auto attrs = context_->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    size_t idx = 0;
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs->GetAttrPointer<bool>(idx));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(idx));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(idx));
    idx++;
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs->GetAttrPointer<bool>(idx));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(idx));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(idx));
    idx++;
    // 区分MatMul和GemmV2，只有3个输入的为MatMul，并设置bias标志
    if (context_->GetInputDesc(idx) != nullptr && context_->GetInputDesc(idx + 1) == nullptr) {
        args_.hasBias = true;
    }
    if (attrs->GetAttrNum() >= HF32_ATTR_NUM) {
        OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs->GetAttrPointer<int32_t>(HF32_ATTR_INDEX - 1));
        OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs->GetAttrPointer<bool>(HF32_ATTR_INDEX));
    }
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputDesc(0));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputShape(0));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Mc2MatMulV3Tiling::GetShapeAttrsInfo() // 检查输入属性是否支持
{
    args_.opName = context_->GetNodeName();
    OP_TILING_CHECK(args_.opName == nullptr, OP_LOGE_WITH_INVALID_INPUT("matmul", "opName"), return ge::GRAPH_FAILED);
    OP_LOGI(args_.opName, "TilingContext: %s", Ops::Transformer::DebugTilingContext(context_).c_str());
    OP_TILING_CHECK((CheckArgs() != ge::GRAPH_SUCCESS) || (GetArgs() != ge::GRAPH_SUCCESS),
                    OP_LOGE(args_.opName, "invalid context."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Mc2MatMulV3Tiling::DoTiling()
{
    if (GetShapeAttrsInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    Mc2MatMulTilingCfg tilingCfg(false, context_->GetCompileInfo(), reinterpret_cast<void *>(&args_));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, tilingCfg.compileInfo);
    platform_ascendc::SocVersion socVersion =
        reinterpret_cast<const Mc2MatmulV3CompileInfo *>(tilingCfg.compileInfo)->socVersion;
    auto platformInfo = context_->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    NpuArch npuArch = ascendcPlatform.GetCurNpuArch();
    Mc2MMRegisterCfg registerCfg{"Mc2MatMulV3", npuArch, strategy::GetMatMulV3Priorities(npuArch)};
    return Mc2MMTilingRegistry::GetInstance().DoTilingImpl(context_, tilingCfg, registerCfg);
}
} // namespace mc2_matmul_v3_advanced
} // namespace optiling
