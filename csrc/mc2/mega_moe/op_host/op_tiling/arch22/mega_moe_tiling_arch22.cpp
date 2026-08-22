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
 * \file mega_moe_tiling_a2a3.cpp
 * \brief
 */

#include <vector>
#include <map>
#include <algorithm>
#include <cfloat>
#include <cmath>

#include "tiling/tiling_api.h"
#include "register/tilingdata_base.h"
#include "register/op_def_registry.h"
#include "mc2_log.h"
#include "mc2_hcom_topo_info.h"
#include "mc2_tiling_utils.h"
#include "mc2_exception_dump.h"
#include "../../../op_kernel/arch22/mega_moe_tiling_a2a3.h"
#include "../../../op_kernel/arch22/mega_moe_tiling_key.h"
#include "../../../op_kernel/arch22/moe_init_routing_quant_v2/moe_init_routing_quant_v2_tiling.h"

using namespace AscendC;
using namespace ge;
using namespace mc2tiling;

namespace MegaMoeA2A3Tiling {
const char *K_INNER_DEBUG = "MegaMoeA2A3 Tiling Debug";

// 算子属性索引
constexpr uint32_t ATTR_MOE_EXPERT_NUM_INDEX = 0;          // moe 专家数
constexpr uint32_t ATTR_EP_WORLD_SIZE_INDEX = 1;           // EP 并行 world size
constexpr uint32_t ATTR_CCL_BUFFER_SIZE_INDEX = 2;         // HCCL 通信缓冲区大小
constexpr uint32_t ATTR_MAX_RECV_TOKEN_NUM_INDEX = 3;      // 最大接收 token 数（用于预分配 workspace）
constexpr uint32_t ATTR_DISPATCH_QUANT_MODE_INDEX = 4;     // 分发阶段量化模式
constexpr uint32_t ATTR_DISPATCH_QUANT_OUT_TYPE_INDEX = 5; // 分发阶段量化输出数据类型
constexpr uint32_t ATTR_COMBINE_QUANT_MODE_INDEX = 6;      // 合并阶段量化模式
constexpr uint32_t ATTR_COMM_ALG_INDEX = 7;                // 通信算法配置
constexpr uint32_t ATTR_NUM_MAX_TOKENS_PER_RANK_INDEX = 8; // 每个 rank 的最大 token 数(bs数量)
constexpr uint32_t ATTR_ACTIVATION_INDEX = 9;              // 激活函数类型（如 "swiglu"）
constexpr uint32_t ATTR_ACTIVATION_PARAMS_INDEX = 10;      // 激活函数参数列表
constexpr uint32_t ATTR_ACTIVATION_OUT_DTYPE_INDEX = 11;   // 激活函数输出数据类型
constexpr uint32_t ATTR_TRANSPOSE_WEIGHT1_INDEX = 12;      // weight1 是否转置
constexpr uint32_t ATTR_TRANSPOSE_WEIGHT2_INDEX = 13;      // weight2 是否转置
constexpr uint32_t ATTR_WEIGHT1_INTERLEAVE_INDEX = 14;     // weight1 交错模式

constexpr uint32_t ACTIVATION_CODE_SWIGLU = 0;
constexpr uint32_t ACTIVATION_CODE_SITU = 2;
constexpr uint32_t ACTIVATION_CODE_SWIGLU_STEP = 3;
constexpr uint32_t ACTIVATION_CODE_SWIGLU_OAI = 4;
constexpr float DEFAULT_ACTIVATION_CLAMP = std::numeric_limits<float>::max();
constexpr float DEFAULT_ACTIVATION_PARAMS1 = 1.702f;
constexpr float DEFAULT_ACTIVATION_PARAMS2 = 1.0f;
constexpr float SITU_LINEAR_BETA_DISABLED = 0.0f;

// 输入 tensor 索引
constexpr uint32_t CONTEXT_INDEX = 0;        // context，shape = [?]
constexpr uint32_t X_INDEX = 1;              // 输入 token x，shape = [M, K]
constexpr uint32_t TOPK_IDS_INDEX = 2;       // topk 专家索引，shape = [bs, topK]
constexpr uint32_t TOPK_WEIGHTS_INDEX = 3;   // topk 权重，shape = [bs, topK]
constexpr uint32_t WEIGHT1_INDEX = 4;        // FFN 第一层权重（动态输入，每个专家一组）
constexpr uint32_t WEIGHT2_INDEX = 5;        // FFN 第二层权重（动态输入，每个专家一组）
constexpr uint32_t WEIGHT_SCALES1_INDEX = 6; // FFN 第一层权重量化 scale
constexpr uint32_t WEIGHT_SCALES2_INDEX = 7; // FFN 第二层权重量化 scale
constexpr uint32_t BIAS1_INDEX = 8;          // GMM1 的 bias（OPTIONAL）
constexpr uint32_t BIAS2_INDEX = 9;          // GMM2 的 bias（OPTIONAL）
constexpr uint32_t X_ACTIVE_MASK_INDEX = 10; // token 活跃掩码（OPTIONAL）
constexpr uint32_t SCALES_INDEX = 11;        // 额外 scale（OPTIONAL）

// 输出 tensor 索引
constexpr uint32_t OUTPUT_Y_INDEX = 0;                 // 输出 y，shape = [M, K]
constexpr uint32_t OUTPUT_EXPERT_TOKEN_NUMS_INDEX = 1; // 专家 token 数量，shape = [expert_num]

constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16 * 1024 * 1024;
constexpr uint64_t MB_SIZE = 1024UL * 1024UL;
constexpr uint64_t RESERVED_SPACE_SIZE = 10 * 1024 * 1024;

// CCL Buffer 相关常量
constexpr int64_t PEERMEM_DATA_OFFSET = 1024 * 60LL; // （预留）60KB 固定偏移
constexpr int64_t ALIGN_128 = 128LL;
constexpr int64_t ALIGN_512 = 512LL;
// 与 kernel 侧 SYNC_STATE_RESERVED_SIZE 保持一致：
// 前 256KB 内放 per-rank 状态，后 256KB 放 per-block self state。
constexpr int64_t SYNC_STATE_RESERVED_SIZE = 512 * 1024LL;

// 维度范围限制
constexpr int64_t MIN_BS = 1;
constexpr int64_t MAX_BS = 4096;
constexpr int64_t MIN_HIDDEN_SIZE = 1024;
constexpr int64_t MAX_HIDDEN_SIZE = 8192;
constexpr int64_t MIN_INTERMEDIATE_HIDDEN = 512;
constexpr int64_t MIN_TOPK = 1;
constexpr int64_t MAX_TOPK = 16;
constexpr int64_t MIN_EXPERT_PER_RANK = 1;
constexpr int64_t MAX_EXPERT_PER_RANK = 128;
constexpr int64_t HIDDEN_SIZE_ALIGN = 512;

// 属性范围限制
constexpr int64_t MIN_MOE_EXPERT_NUM = 1;
constexpr int64_t MAX_MOE_EXPERT_NUM = 1024;
constexpr int64_t VALID_EP_WORLD_SIZE[] = {2, 4, 8, 16, 32, 64, 128};

constexpr uint32_t TWO_DIMS = 2U;
constexpr uint32_t ONE_DIM = 1U;
constexpr uint32_t THREE_DIMS = 3U;

//
constexpr int64_t DISPATCH_QUANT_MODE_NO_QUANT = 0;
constexpr int64_t DISPATCH_QUANT_MODE_PER_TENSOR = 2;

static int64_t CalcLeastCclBufferSize(int64_t maxRecvTokenNum, int64_t h, int64_t epWorldSize, int64_t expertPerRank,
                                      bool isQuantRouting, bool isW4A8, bool isA3, int64_t bs, int64_t topK)
{
    // ccl buff的承载的数据块 1（winIn）：
    // TPE = epWorldSize × CeilAlign(epWorldSize × MAX_EXPERT_PER_RANK + 1, 128) × 4B
    int64_t offsetTokenPerExpert = epWorldSize * ops::CeilAlign(epWorldSize * MAX_EXPERT_PER_RANK + 1, ALIGN_128) *
                                   static_cast<int64_t>(sizeof(int32_t));

    // ccl buff的承载的数据块 2：
    // ============================== winIn ==============================
    // FFN的左矩阵，非量化则token为 h×2 字节 (bf16)，量化则为 h+512 字节 (int8)
    int64_t offsetAAfterDispatch = 0;
    if (isA3) { // A3打包发送，+32与+512中取大值
        offsetAAfterDispatch = bs * topK * (isQuantRouting ? (h + ALIGN_512) : h * sizeof(int16_t));
    } else { // A2分开发送
        offsetAAfterDispatch = maxRecvTokenNum * (isQuantRouting ? (h + ALIGN_512) : h * sizeof(int16_t));
    }
    // FFN的输出在分发后，接收的空间大小
    int64_t offsetD = bs * topK * h * sizeof(int16_t);
    int64_t winInTensorSize = offsetAAfterDispatch + offsetD;
    // ============================== winOut（仅A2） ==============================
    int64_t winOutTensorSize = 0;
    if (!isA3) {
        int64_t offsetA = bs * topK * (!isQuantRouting ? h * sizeof(int16_t) : (h + ALIGN_512));
        int64_t offsetC = maxRecvTokenNum * h * sizeof(int16_t);
        winOutTensorSize = offsetA + offsetC;
    }
    int64_t offsetTensor = std::max(winInTensorSize, winOutTensorSize);
    // pertokenscale的额外空间
    if (isQuantRouting) {
        offsetTensor += (isA3 ? bs * topK : maxRecvTokenNum) * sizeof(float); // pertokenScale
    }

    // ccl buff的承载的数据块 3（winIn）：
    // 同步flag
    int64_t offsetFlag = epWorldSize * ALIGN_512; // CrossRankSync所用空间
    if (!isA3) {
        // A2: dispatch flag(EP×E×64B) + allgather flag(EP×64B)
        int64_t dispatchFlag = epWorldSize * MAX_EXPERT_PER_RANK * 64;
        int64_t allgatherFlag = epWorldSize * 64;
        offsetFlag += dispatchFlag + allgatherFlag;
    } else {
        // A3 need self state
        offsetFlag = std::max(epWorldSize * ALIGN_512, SYNC_STATE_RESERVED_SIZE);
    }
    return offsetTokenPerExpert + offsetTensor + offsetFlag + RESERVED_SPACE_SIZE;
}

// =========================== Attr 校验分函数 ===========================

static ge::graphStatus CheckMoeExpertNumAttr(const int64_t *ptr)
{
    OP_TILING_CHECK(ptr == nullptr, OP_LOGE(K_INNER_DEBUG, "moeExpertNum is nullptr."), return GRAPH_FAILED);
    OP_TILING_CHECK(*ptr < MIN_MOE_EXPERT_NUM || *ptr > MAX_MOE_EXPERT_NUM,
                    OP_LOGE(K_INNER_DEBUG, "moeExpertNum should be in [%ld, %ld], but got %ld.", MIN_MOE_EXPERT_NUM,
                            MAX_MOE_EXPERT_NUM, *ptr),
                    return GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckEpWorldSizeAttr(const int64_t *ptr)
{
    OP_TILING_CHECK(ptr == nullptr, OP_LOGE(K_INNER_DEBUG, "epWorldSize is nullptr."), return GRAPH_FAILED);
    bool isValidEpWorldSize = std::find(std::begin(VALID_EP_WORLD_SIZE), std::end(VALID_EP_WORLD_SIZE), *ptr) !=
                              std::end(VALID_EP_WORLD_SIZE);
    OP_TILING_CHECK(
        !isValidEpWorldSize,
        OP_LOGE(K_INNER_DEBUG, "epWorldSize should be one of {2, 4, 8, 16, 32, 64, 128}, but got %ld.", *ptr),
        return GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckMaxRecvTokenNumAttr(const int64_t *ptr)
{
    OP_TILING_CHECK(ptr == nullptr, OP_LOGE(K_INNER_DEBUG, "maxRecvTokenNum is nullptr."), return GRAPH_FAILED);
    OP_TILING_CHECK(*ptr < 0, OP_LOGE(K_INNER_DEBUG, "maxRecvTokenNum is invalid, should be >= 0, but got %ld.", *ptr),
                    return GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// 校验 dispatch_quant_mode 和 dispatch_quant_out_dtype，两者都依赖 weight1 的 dataType
static ge::graphStatus CheckDispatchQuantAttrs(gert::TilingContext *context, const int64_t *dispatchQuantModePtr,
                                               const int64_t *dispatchQuantOutDtypePtr)
{
    OP_TILING_CHECK(dispatchQuantModePtr == nullptr, OP_LOGE(K_INNER_DEBUG, "dispatchQuantMode is null."),
                    return GRAPH_FAILED);
    OP_TILING_CHECK(dispatchQuantOutDtypePtr == nullptr, OP_LOGE(K_INNER_DEBUG, "dispatchQuantOutDtype is null."),
                    return GRAPH_FAILED);

    auto w1Desc = context->GetDynamicInputDesc(WEIGHT1_INDEX, 0);
    OP_TILING_CHECK(w1Desc == nullptr, OP_LOGE(K_INNER_DEBUG, "weight1 desc is null."), return GRAPH_FAILED);
    ge::DataType w1DataType = w1Desc->GetDataType();

    // dispatch_quant_mode 校验：与 weight 数据类型对应
    int64_t expectedDispatchQuantMode = 0;
    if (w1DataType == ge::DT_INT4 || w1DataType == ge::DT_INT8) {
        expectedDispatchQuantMode = DISPATCH_QUANT_MODE_PER_TENSOR;
    } else if (w1DataType == ge::DT_BF16 || w1DataType == ge::DT_FLOAT16) {
        expectedDispatchQuantMode = DISPATCH_QUANT_MODE_NO_QUANT;
    }
    OP_TILING_CHECK(*dispatchQuantModePtr != expectedDispatchQuantMode,
                    OP_LOGE(K_INNER_DEBUG,
                            "dispatch_quant_mode is invalid, should be %ld for weight dataType %d, but got %ld.",
                            expectedDispatchQuantMode, static_cast<int>(w1DataType), *dispatchQuantModePtr),
                    return GRAPH_FAILED);

    // dispatch_quant_out_dtype 校验：与 weight 数据类型对应
    if (*dispatchQuantOutDtypePtr != static_cast<int64_t>(ge::DT_UNDEFINED)) {
        int64_t expectedQuantOutType = 0;
        if (w1DataType == ge::DT_BF16) {
            expectedQuantOutType = static_cast<int64_t>(ge::DT_BF16);
        } else if (w1DataType == ge::DT_FLOAT16) {
            expectedQuantOutType = static_cast<int64_t>(ge::DT_FLOAT16);
        } else if (w1DataType == ge::DT_INT4 || w1DataType == ge::DT_INT8) {
            expectedQuantOutType = static_cast<int64_t>(ge::DT_INT8);
        }
        OP_TILING_CHECK(
            *dispatchQuantOutDtypePtr != expectedQuantOutType,
            OP_LOGE(K_INNER_DEBUG,
                    "dispatch_quant_out_type is invalid, should be %ld for weight dataType %d, but got %ld.",
                    expectedQuantOutType, static_cast<int>(w1DataType), *dispatchQuantOutDtypePtr),
            return GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckCombineQuantModeAttr(const int64_t *ptr)
{
    OP_TILING_CHECK(ptr == nullptr, OP_LOGE(K_INNER_DEBUG, "combineQuantMode is null."), return GRAPH_FAILED);
    OP_TILING_CHECK(*ptr != 0,
                    OP_LOGE(K_INNER_DEBUG, "combine_quant_mode is invalid, only support 0, but got %ld.", *ptr),
                    return GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// 校验 num_max_tokens_per_rank，需要从 x shape 获取 bs
static ge::graphStatus CheckNumMaxTokensPerRankAttr(gert::TilingContext *context, const int64_t *ptr)
{
    OP_TILING_CHECK(ptr == nullptr, OP_LOGE(K_INNER_DEBUG, "numMaxTokensPerRankPtr is null."), return GRAPH_FAILED);
    const gert::StorageShape *xStorageShape = context->GetInputShape(X_INDEX);
    OP_TILING_CHECK(xStorageShape == nullptr, OP_LOGE(K_INNER_DEBUG, "x shape is null."), return GRAPH_FAILED);
    int64_t bs = xStorageShape->GetStorageShape().GetDim(0);
    if (*ptr > 0) {
        OP_TILING_CHECK(
            *ptr < bs,
            OP_LOGE(K_INNER_DEBUG, "num_max_tokens_per_rank is invalid, should be >= bs(%ld), but got %ld.", bs, *ptr),
            return GRAPH_FAILED);
        OP_TILING_CHECK(*ptr < MIN_BS || *ptr > MAX_BS,
                        OP_LOGE(K_INNER_DEBUG,
                                "num_max_tokens_per_rank is invalid, should be in [%ld, %ld], but got %ld.", MIN_BS,
                                MAX_BS, *ptr),
                        return GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckActivationAttr(const char *ptr, uint32_t &activationCode)
{
    OP_TILING_CHECK(ptr == nullptr, OP_LOGE(K_INNER_DEBUG, "activation is null."), return GRAPH_FAILED);
    std::string activationStr(ptr);
    if (activationStr == "swiglu") {
        activationCode = ACTIVATION_CODE_SWIGLU;
    } else if (activationStr == "situglu") {
        activationCode = ACTIVATION_CODE_SITU;
    } else if (activationStr == "swiglustep") {
        activationCode = ACTIVATION_CODE_SWIGLU_STEP;
    } else if (activationStr == "swigluoai") {
        activationCode = ACTIVATION_CODE_SWIGLU_OAI;
    } else {
        OP_LOGE(K_INNER_DEBUG,
                "activation is invalid, support 'swiglu', 'swiglustep', 'swigluoai' and 'situglu', but got '%s'.",
                ptr);
        return GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static bool IsValidActivationClamp(float value)
{
    return value >= 0.0f && !std::isnan(value);
}

static bool IsFiniteActivationParam(float value)
{
    return std::isfinite(value);
}

static bool IsValidSituScale(float value)
{
    return std::isfinite(value) && value != 0.0f;
}

static ge::graphStatus CheckActivationParamCount(uint32_t activationCode, size_t actualCount, size_t minCount,
                                                 size_t maxCount)
{
    OP_TILING_CHECK(actualCount < minCount || actualCount > maxCount,
                    OP_LOGE(K_INNER_DEBUG,
                            "activation_params has invalid size %zu for activation code %u, expected [%zu, %zu].",
                            actualCount, activationCode, minCount, maxCount),
                    return GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckActivationParamsAttr(const gert::TypedContinuousVector<float> *params,
                                                 MegaMoeA2A3TilingData &info)
{
    OP_TILING_CHECK(params == nullptr, OP_LOGE(K_INNER_DEBUG, "activation_params is null."), return GRAPH_FAILED);
    size_t paramCount = params->GetSize();
    info.activationClamp = DEFAULT_ACTIVATION_CLAMP;
    info.activationParams1 = DEFAULT_ACTIVATION_PARAMS1;
    info.activationParams2 = DEFAULT_ACTIVATION_PARAMS2;
    if (info.activationCode == ACTIVATION_CODE_SITU) {
        info.activationParams1 = DEFAULT_ACTIVATION_PARAMS2;
        info.activationParams2 = SITU_LINEAR_BETA_DISABLED;
    }
    if (paramCount == 0U) {
        return ge::GRAPH_SUCCESS;
    }

    const float *data = params->GetData();
    OP_TILING_CHECK(data == nullptr, OP_LOGE(K_INNER_DEBUG, "activation_params data is null."), return GRAPH_FAILED);

    if (info.activationCode == ACTIVATION_CODE_SWIGLU || info.activationCode == ACTIVATION_CODE_SWIGLU_STEP) {
        OP_TILING_CHECK(CheckActivationParamCount(info.activationCode, paramCount, 1U, 1U) != ge::GRAPH_SUCCESS,
                        OP_LOGE(K_INNER_DEBUG, "CheckActivationParamCount failed."), return GRAPH_FAILED);
        info.activationClamp = data[0];
    } else if (info.activationCode == ACTIVATION_CODE_SWIGLU_OAI) {
        OP_TILING_CHECK(CheckActivationParamCount(info.activationCode, paramCount, 3U, 3U) != ge::GRAPH_SUCCESS,
                        OP_LOGE(K_INNER_DEBUG, "CheckActivationParamCount failed."), return GRAPH_FAILED);
        info.activationClamp = data[0];
        info.activationParams1 = data[1];
        info.activationParams2 = data[2];
    } else {
        // SITU prototype order is [beta, linear_beta]; activationParams1 carries beta (data[0]) and
        // activationParams2 carries linear_beta (data[1]), matching the order passed by the prototype.
        OP_TILING_CHECK(CheckActivationParamCount(info.activationCode, paramCount, 1U, 2U) != ge::GRAPH_SUCCESS,
                        OP_LOGE(K_INNER_DEBUG, "CheckActivationParamCount failed."), return GRAPH_FAILED);
        info.activationParams1 = data[0];
        info.activationParams2 = paramCount == 2U ? data[1] : SITU_LINEAR_BETA_DISABLED;
    }

    if (info.activationCode != ACTIVATION_CODE_SITU) {
        OP_TILING_CHECK(
            !IsValidActivationClamp(info.activationClamp),
            OP_LOGE(K_INNER_DEBUG, "activation clamp must be >= 0 and not NAN, but got %f.", info.activationClamp),
            return GRAPH_FAILED);
    }
    if (info.activationCode == ACTIVATION_CODE_SWIGLU_OAI) {
        OP_TILING_CHECK(!IsFiniteActivationParam(info.activationParams1),
                        OP_LOGE(K_INNER_DEBUG, "activation alpha must be finite, but got %f.",
                                info.activationParams1),
                        return GRAPH_FAILED);
    }
    if (info.activationCode == ACTIVATION_CODE_SWIGLU_OAI) {
        OP_TILING_CHECK(!IsFiniteActivationParam(info.activationParams2),
                        OP_LOGE(K_INNER_DEBUG, "swigluoai beta must be finite, but got %f.",
                                info.activationParams2),
                        return GRAPH_FAILED);
    }
    if (info.activationCode == ACTIVATION_CODE_SITU) {
        OP_TILING_CHECK(
            !IsValidSituScale(info.activationParams1),
            OP_LOGE(K_INNER_DEBUG, "situglu beta must be finite and non-zero, but got %f.", info.activationParams1),
            return GRAPH_FAILED);
        if (paramCount == 2U) {
            OP_TILING_CHECK(!IsValidSituScale(info.activationParams2),
                            OP_LOGE(K_INNER_DEBUG, "situglu linear_beta must be finite and non-zero, but got %f.",
                                    info.activationParams2),
                            return GRAPH_FAILED);
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckActivationOutDtypeAttr(const int64_t *ptr)
{
    OP_TILING_CHECK(ptr == nullptr, OP_LOGE(K_INNER_DEBUG, "activationOutDtypePtr is null."), return GRAPH_FAILED);
    OP_TILING_CHECK(*ptr != static_cast<int64_t>(ge::DT_UNDEFINED),
                    OP_LOGE(K_INNER_DEBUG, "activation_out_dtype is invalid, should be default value %ld, but got %ld.",
                            static_cast<int64_t>(ge::DT_UNDEFINED), *ptr),
                    return GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckTransposeWeightAttr(const bool *w1, const bool *w2)
{
    OP_TILING_CHECK(w1 == nullptr, OP_LOGE(K_INNER_DEBUG, "transposeWeight1 is null."), return GRAPH_FAILED);
    OP_TILING_CHECK(w2 == nullptr, OP_LOGE(K_INNER_DEBUG, "transposeWeight2 is null."), return GRAPH_FAILED);
    OP_TILING_CHECK(*w1 != *w2,
                    OP_LOGE(K_INNER_DEBUG,
                            "transpose_weight1 and transpose_weight2 must be the same, "
                            "transpose_weight1 = %d, transpose_weight2 = %d.",
                            static_cast<int>(*w1), static_cast<int>(*w2)),
                    return GRAPH_FAILED);
    OP_TILING_CHECK(*w1 == true, OP_LOGE(K_INNER_DEBUG, "Current soc do not support transpose weight now."),
                    return GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// 校验 weight1_interleave，需要从 weight1 tensor 获取 N
static ge::graphStatus CheckWeight1InterleaveAttr(gert::TilingContext *context, const int64_t *ptr)
{
    OP_TILING_CHECK(ptr == nullptr, OP_LOGE(K_INNER_DEBUG, "weight1Interleave is null."), return GRAPH_FAILED);
    OP_TILING_CHECK(*ptr < 0,
                    OP_LOGE(K_INNER_DEBUG, "weight1_interleave is invalid, should be >= 0, but got %ld.", *ptr),
                    return GRAPH_FAILED);
    auto w1Tensor = context->GetDynamicInputTensor(WEIGHT1_INDEX, 0);
    OP_TILING_CHECK(w1Tensor == nullptr, OP_LOGE(K_INNER_DEBUG, "weight1 tensor is null."), return GRAPH_FAILED);
    uint32_t w1TensorDims = w1Tensor->GetOriginShape().GetDimNum();
    uint32_t N = w1Tensor->GetStorageShape().GetDim(w1TensorDims - 1);
    uint32_t maxInterleave = N / 2;
    OP_TILING_CHECK(static_cast<uint32_t>(*ptr) > maxInterleave,
                    OP_LOGE(K_INNER_DEBUG, "weight1_interleave is invalid, should be in [0, %u], but got %ld.",
                            maxInterleave, *ptr),
                    return GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MegaMoeA2A3CheckAttrAndSetTiling(gert::TilingContext *context, MegaMoeA2A3TilingData &info)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(K_INNER_DEBUG, "attrs is null."), return ge::GRAPH_FAILED);

    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_MOE_EXPERT_NUM_INDEX);
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    auto cclBufferSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_CCL_BUFFER_SIZE_INDEX);
    auto maxRecvTokenNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_MAX_RECV_TOKEN_NUM_INDEX);
    auto dispatchQuantModePtr = attrs->GetAttrPointer<int64_t>(ATTR_DISPATCH_QUANT_MODE_INDEX);
    auto dispatchQuantOutDtypePtr = attrs->GetAttrPointer<int64_t>(ATTR_DISPATCH_QUANT_OUT_TYPE_INDEX);
    auto combineQuantModePtr = attrs->GetAttrPointer<int64_t>(ATTR_COMBINE_QUANT_MODE_INDEX);
    auto commAlgPtr = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_COMM_ALG_INDEX));
    auto numMaxTokensPerRankPtr = attrs->GetAttrPointer<int64_t>(ATTR_NUM_MAX_TOKENS_PER_RANK_INDEX);
    auto activationPtr = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_ACTIVATION_INDEX));
    auto activationParamsPtr = attrs->GetListFloat(ATTR_ACTIVATION_PARAMS_INDEX);
    auto activationOutDtypePtr = attrs->GetAttrPointer<int64_t>(ATTR_ACTIVATION_OUT_DTYPE_INDEX);
    auto transposeWeight1Ptr = attrs->GetAttrPointer<bool>(ATTR_TRANSPOSE_WEIGHT1_INDEX);
    auto transposeWeight2Ptr = attrs->GetAttrPointer<bool>(ATTR_TRANSPOSE_WEIGHT2_INDEX);
    auto weight1InterleavePtr = attrs->GetAttrPointer<int64_t>(ATTR_WEIGHT1_INTERLEAVE_INDEX);

    // 1. moe_expert_num
    OP_TILING_CHECK(CheckMoeExpertNumAttr(moeExpertNumPtr) != ge::GRAPH_SUCCESS,
                    OP_LOGE(K_INNER_DEBUG, "CheckMoeExpertNumAttr failed."), return GRAPH_FAILED);
    info.moeExpertNum = static_cast<uint32_t>(*moeExpertNumPtr);

    // 2. ep_world_size
    OP_TILING_CHECK(CheckEpWorldSizeAttr(epWorldSizePtr) != ge::GRAPH_SUCCESS,
                    OP_LOGE(K_INNER_DEBUG, "CheckEpWorldSizeAttr failed."), return GRAPH_FAILED);
    info.epWorldSize = static_cast<uint32_t>(*epWorldSizePtr);

    // 3. ccl_buffer_size
    OP_TILING_CHECK(cclBufferSizePtr == nullptr, OP_LOGE(K_INNER_DEBUG, "ccl_buffer_size is nullptr."),
                    return GRAPH_FAILED);
    info.cclBufferSize = static_cast<int64_t>(*cclBufferSizePtr);

    // 4. max_recv_token_num
    OP_TILING_CHECK(CheckMaxRecvTokenNumAttr(maxRecvTokenNumPtr) != ge::GRAPH_SUCCESS,
                    OP_LOGE(K_INNER_DEBUG, "CheckMaxRecvTokenNumAttr failed."), return GRAPH_FAILED);
    info.maxRecvTokenNum = static_cast<uint64_t>(*maxRecvTokenNumPtr);
    // max_recv_token_num 为 0 时赋值为最大值；
    if (info.maxRecvTokenNum == 0U) {
        info.maxRecvTokenNum =
            static_cast<uint64_t>(info.M) * info.epWorldSize * std::min(info.topK, info.expertPerRank);
        OP_LOGD(K_INNER_DEBUG, "maxRecvTokenNum auto-calculated to %u (bs=%u * ep=%u * min(topK=%u, expertPerRank=%u))",
                info.maxRecvTokenNum, info.M, info.epWorldSize, info.topK, info.expertPerRank);
    }

    // 5-6. dispatch_quant_mode && dispatch_quant_out_dtype
    OP_TILING_CHECK(CheckDispatchQuantAttrs(context, dispatchQuantModePtr, dispatchQuantOutDtypePtr) !=
                        ge::GRAPH_SUCCESS,
                    OP_LOGE(K_INNER_DEBUG, "CheckDispatchQuantAttrs failed."), return GRAPH_FAILED);
    info.dispatchQuantMode = static_cast<uint32_t>(*dispatchQuantModePtr);
    info.dispatchQuantOutDtype = static_cast<uint32_t>(*dispatchQuantOutDtypePtr);

    // 7. combine_quant_mode
    OP_TILING_CHECK(CheckCombineQuantModeAttr(combineQuantModePtr) != ge::GRAPH_SUCCESS,
                    OP_LOGE(K_INNER_DEBUG, "CheckCombineQuantModeAttr failed."), return GRAPH_FAILED);
    info.combineQuantMode = static_cast<uint32_t>(*combineQuantModePtr);

    // 8. num_max_token_per_rank
    OP_TILING_CHECK(CheckNumMaxTokensPerRankAttr(context, numMaxTokensPerRankPtr) != ge::GRAPH_SUCCESS,
                    OP_LOGE(K_INNER_DEBUG, "CheckNumMaxTokensPerRankAttr failed."), return GRAPH_FAILED);
    info.numMaxTokensPerRank = static_cast<uint32_t>(*numMaxTokensPerRankPtr);

    // 9. activation
    OP_TILING_CHECK(CheckActivationAttr(activationPtr, info.activationCode) != ge::GRAPH_SUCCESS,
                    OP_LOGE(K_INNER_DEBUG, "CheckActivationAttr failed."), return GRAPH_FAILED);

    // 10. activation_params
    OP_TILING_CHECK(CheckActivationParamsAttr(activationParamsPtr, info) != ge::GRAPH_SUCCESS,
                    OP_LOGE(K_INNER_DEBUG, "CheckActivationParamsAttr failed."), return GRAPH_FAILED);

    // 11. activation_out_dtype
    OP_TILING_CHECK(CheckActivationOutDtypeAttr(activationOutDtypePtr) != ge::GRAPH_SUCCESS,
                    OP_LOGE(K_INNER_DEBUG, "CheckActivationOutDtypeAttr failed."), return GRAPH_FAILED);
    info.activationOutDtype = activationOutDtypePtr != nullptr ? static_cast<uint32_t>(*activationOutDtypePtr) :
                                                                 static_cast<uint32_t>(ge::DT_UNDEFINED);

    // 12-13. transpose_weight1/2
    OP_TILING_CHECK(CheckTransposeWeightAttr(transposeWeight1Ptr, transposeWeight2Ptr) != ge::GRAPH_SUCCESS,
                    OP_LOGE(K_INNER_DEBUG, "CheckTransposeWeightAttr failed."), return GRAPH_FAILED);
    info.isTransposeW1 = *transposeWeight1Ptr ? 1U : 0U;
    info.isTransposeW2 = *transposeWeight2Ptr ? 1U : 0U;

    // 14. weight1_interleave
    OP_TILING_CHECK(CheckWeight1InterleaveAttr(context, weight1InterleavePtr) != ge::GRAPH_SUCCESS,
                    OP_LOGE(K_INNER_DEBUG, "CheckWeight1InterleaveAttr failed."), return GRAPH_FAILED);
    info.weight1Interleave = static_cast<uint32_t>(*weight1InterleavePtr);

    info.worldSize = info.epWorldSize;

    OP_LOGD(K_INNER_DEBUG, "moeExpertNum=%u", info.moeExpertNum);
    OP_LOGD(K_INNER_DEBUG, "epWorldSize=%u", info.epWorldSize);
    OP_LOGD(K_INNER_DEBUG, "cclBufferSize=%u", info.cclBufferSize);
    OP_LOGD(K_INNER_DEBUG, "maxRecvTokenNum=%u", info.maxRecvTokenNum);
    OP_LOGD(K_INNER_DEBUG, "dispatchQuantMode=%u", info.dispatchQuantMode);
    OP_LOGD(K_INNER_DEBUG, "dispatchQuantOutDtype=%u", info.dispatchQuantOutDtype);
    OP_LOGD(K_INNER_DEBUG, "combineQuantMode=%u", info.combineQuantMode);
    OP_LOGD(K_INNER_DEBUG, "numMaxTokensPerRank=%u", info.numMaxTokensPerRank);
    OP_LOGD(K_INNER_DEBUG, "activationCode=%u", info.activationCode);
    OP_LOGD(K_INNER_DEBUG, "activationClamp=%f", info.activationClamp);
    OP_LOGD(K_INNER_DEBUG, "activationParams1=%f", info.activationParams1);
    OP_LOGD(K_INNER_DEBUG, "activationParams2=%f", info.activationParams2);
    OP_LOGD(K_INNER_DEBUG, "activationOutDtype=%u", info.activationOutDtype);
    OP_LOGD(K_INNER_DEBUG, "transposeWeight1=%u", info.isTransposeW1);
    OP_LOGD(K_INNER_DEBUG, "transposeWeight2=%u", info.isTransposeW2);
    OP_LOGD(K_INNER_DEBUG, "weight1Interleave=%u", info.weight1Interleave);
    OP_LOGD(K_INNER_DEBUG, "worldSize=%u", info.worldSize);

    return ge::GRAPH_SUCCESS;
}

// 获取指定 DynamicInput 的 TensorList 长度
static uint32_t GetDynamicInputTensorListLen(gert::TilingContext *context, uint32_t inputIndex)
{
    uint32_t listLen = 0;
    while (context->GetDynamicInputTensor(inputIndex, ++listLen) != nullptr) {
    }
    return listLen;
}

// 校验 context 输入
static ge::graphStatus CheckContextInput(gert::TilingContext *context, const gert::StorageShape *contextStorageShape)
{
    auto contextDesc = context->GetInputDesc(CONTEXT_INDEX);
    OP_TILING_CHECK(contextDesc == nullptr, OP_LOGE(K_INNER_DEBUG, "context desc is null."), return GRAPH_FAILED);
    OP_TILING_CHECK(contextDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE(K_INNER_DEBUG, "context dataType is invalid, should be INT32, but got %d.",
                            static_cast<int>(contextDesc->GetDataType())),
                    return GRAPH_FAILED);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(contextDesc->GetStorageFormat())) != ge::FORMAT_ND,
                    OP_LOGE(K_INNER_DEBUG, "context format is invalid, should be FORMAT_ND."), return GRAPH_FAILED);
    OP_TILING_CHECK(contextStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
                    OP_LOGE(K_INNER_DEBUG, "contextShape dims must be 1, but current dim num is %zu.",
                            contextStorageShape->GetStorageShape().GetDimNum()),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// 校验 x 输入，返回 bs 和 hiddenSize
static ge::graphStatus CheckXInput(gert::TilingContext *context, const gert::StorageShape *xStorageShape,
                                   int64_t &outBs, int64_t &outHiddenSize)
{
    OP_TILING_CHECK(
        xStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE(K_INNER_DEBUG, "x must be 2-dimension, but got %lu dim.", xStorageShape->GetStorageShape().GetDimNum()),
        return GRAPH_FAILED);

    int64_t bs = xStorageShape->GetStorageShape().GetDim(0);
    int64_t hiddenSize = xStorageShape->GetStorageShape().GetDim(1);

    OP_TILING_CHECK(
        bs < MIN_BS || bs > MAX_BS,
        OP_LOGE(K_INNER_DEBUG, "x's dim0(bs) is invalid, should be in [%ld, %ld], but got %ld.", MIN_BS, MAX_BS, bs),
        return GRAPH_FAILED);
    OP_TILING_CHECK(hiddenSize < MIN_HIDDEN_SIZE || hiddenSize > MAX_HIDDEN_SIZE,
                    OP_LOGE(K_INNER_DEBUG, "x's dim1(hidden_size) is invalid, should be in [%ld, %ld], but got %ld.",
                            MIN_HIDDEN_SIZE, MAX_HIDDEN_SIZE, hiddenSize),
                    return GRAPH_FAILED);
    OP_TILING_CHECK(hiddenSize % HIDDEN_SIZE_ALIGN != 0,
                    OP_LOGE(K_INNER_DEBUG, "x's dim1(hidden_size) should be %ld aligned, but got %ld.",
                            HIDDEN_SIZE_ALIGN, hiddenSize),
                    return GRAPH_FAILED);

    auto xDesc = context->GetInputDesc(X_INDEX);
    OP_TILING_CHECK(xDesc == nullptr, OP_LOGE(K_INNER_DEBUG, "x desc is null."), return GRAPH_FAILED);
    ge::DataType xDataType = xDesc->GetDataType();
    OP_TILING_CHECK(
        xDataType != ge::DT_BF16,
        OP_LOGE(K_INNER_DEBUG, "x dataType is invalid, should be BF16, but got %d.", static_cast<int>(xDataType)),
        return GRAPH_FAILED);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(xDesc->GetStorageFormat())) != ge::FORMAT_ND,
                    OP_LOGE(K_INNER_DEBUG, "x format is invalid, should be FORMAT_ND."), return GRAPH_FAILED);

    outBs = bs;
    outHiddenSize = hiddenSize;
    return ge::GRAPH_SUCCESS;
}

// 校验 topk_ids 输入，返回 topK
static ge::graphStatus CheckTopkIdsInput(gert::TilingContext *context, const gert::StorageShape *topkIdsStorageShape,
                                         int64_t bs, int64_t &outTopK)
{
    OP_TILING_CHECK(topkIdsStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
                    OP_LOGE(K_INNER_DEBUG, "topk_ids must be 2-dimension, but got %lu dim.",
                            topkIdsStorageShape->GetStorageShape().GetDimNum()),
                    return GRAPH_FAILED);

    int64_t topkIdsDim0 = topkIdsStorageShape->GetStorageShape().GetDim(0);
    int64_t topK = topkIdsStorageShape->GetStorageShape().GetDim(1);

    OP_TILING_CHECK(topkIdsDim0 != bs,
                    OP_LOGE(K_INNER_DEBUG,
                            "topk_ids's dim0 not equal to x's dim0, topk_ids's dim0 = %ld, x's dim0 = %ld.",
                            topkIdsDim0, bs),
                    return GRAPH_FAILED);
    OP_TILING_CHECK(topK < MIN_TOPK || topK > MAX_TOPK,
                    OP_LOGE(K_INNER_DEBUG, "topk_ids's dim1(topK) is invalid, should be in [%ld, %ld], but got %ld.",
                            MIN_TOPK, MAX_TOPK, topK),
                    return GRAPH_FAILED);

    auto topkIdsDesc = context->GetInputDesc(TOPK_IDS_INDEX);
    OP_TILING_CHECK(topkIdsDesc == nullptr, OP_LOGE(K_INNER_DEBUG, "topk_ids desc is null."), return GRAPH_FAILED);
    OP_TILING_CHECK(topkIdsDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE(K_INNER_DEBUG, "topk_ids dataType is invalid, should be INT32, but got %d.",
                            static_cast<int>(topkIdsDesc->GetDataType())),
                    return GRAPH_FAILED);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(topkIdsDesc->GetStorageFormat())) != ge::FORMAT_ND,
                    OP_LOGE(K_INNER_DEBUG, "topk_ids format is invalid, should be FORMAT_ND."), return GRAPH_FAILED);

    outTopK = topK;
    return ge::GRAPH_SUCCESS;
}

// 校验 topk_weights 输入
static ge::graphStatus CheckTopkWeightsInput(gert::TilingContext *context,
                                             const gert::StorageShape *topkWeightsStorageShape, int64_t bs,
                                             int64_t topK)
{
    OP_TILING_CHECK(topkWeightsStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
                    OP_LOGE(K_INNER_DEBUG, "topk_weights must be 2-dimension, but got %lu dim.",
                            topkWeightsStorageShape->GetStorageShape().GetDimNum()),
                    return GRAPH_FAILED);

    int64_t topkWeightsDim0 = topkWeightsStorageShape->GetStorageShape().GetDim(0);
    int64_t topkWeightsDim1 = topkWeightsStorageShape->GetStorageShape().GetDim(1);

    OP_TILING_CHECK(topkWeightsDim0 != bs,
                    OP_LOGE(K_INNER_DEBUG,
                            "topk_weights's dim0 not equal to x's dim0, topk_weights's dim0 = %ld, x's dim0 = %ld.",
                            topkWeightsDim0, bs),
                    return GRAPH_FAILED);
    OP_TILING_CHECK(topkWeightsDim1 != topK,
                    OP_LOGE(K_INNER_DEBUG,
                            "topk_weights's dim1 not equal to topk_ids's dim1, "
                            "topk_weights's dim1 = %ld, topk_ids's dim1 = %ld.",
                            topkWeightsDim1, topK),
                    return GRAPH_FAILED);

    auto topkWeightsDesc = context->GetInputDesc(TOPK_WEIGHTS_INDEX);
    OP_TILING_CHECK(topkWeightsDesc == nullptr, OP_LOGE(K_INNER_DEBUG, "topk_weights desc is null."),
                    return GRAPH_FAILED);
    ge::DataType topkWeightsDataType = topkWeightsDesc->GetDataType();
    OP_TILING_CHECK(topkWeightsDataType != ge::DT_FLOAT && topkWeightsDataType != ge::DT_BF16,
                    OP_LOGE(K_INNER_DEBUG, "topk_weights dataType is invalid, should be FLOAT or BF16, but got %d.",
                            static_cast<int>(topkWeightsDataType)),
                    return GRAPH_FAILED);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(topkWeightsDesc->GetStorageFormat())) != ge::FORMAT_ND,
                    OP_LOGE(K_INNER_DEBUG, "topk_weights format is invalid, should be FORMAT_ND."),
                    return GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// 校验 x_active_mask 输入（可选），返回 isActiveMask
static ge::graphStatus CheckXActiveMaskInput(gert::TilingContext *context, int64_t bs, bool &outIsActiveMask)
{
    const gert::StorageShape *xActiveMaskStorageShape = context->GetOptionalInputShape(X_ACTIVE_MASK_INDEX);
    bool isActiveMask = (xActiveMaskStorageShape != nullptr);
    if (isActiveMask) {
        int64_t xActiveMaskDimNums = xActiveMaskStorageShape->GetStorageShape().GetDimNum();
        OP_TILING_CHECK(xActiveMaskDimNums != ONE_DIM,
                        OP_LOGE(K_INNER_DEBUG,
                                "x_active_mask must be 1-dimension, but got %ld dim. "
                                "2-dimension is not supported now.",
                                xActiveMaskDimNums),
                        return GRAPH_FAILED);
        int64_t xActiveMaskDim0 = xActiveMaskStorageShape->GetStorageShape().GetDim(0);
        OP_TILING_CHECK(xActiveMaskDim0 != bs,
                        OP_LOGE(K_INNER_DEBUG,
                                "x_active_mask's dim0 not equal to x's dim0, "
                                "x_active_mask's dim0 = %ld, x's dim0 = %ld.",
                                xActiveMaskDim0, bs),
                        return GRAPH_FAILED);
        auto xActiveMaskDesc = context->GetOptionalInputDesc(X_ACTIVE_MASK_INDEX);
        OP_TILING_CHECK(xActiveMaskDesc == nullptr, OP_LOGE(K_INNER_DEBUG, "x_active_mask desc is null."),
                        return GRAPH_FAILED);
        OP_TILING_CHECK(xActiveMaskDesc->GetDataType() != ge::DT_INT8,
                        OP_LOGE(K_INNER_DEBUG, "x_active_mask dataType is invalid, should be INT8, but got %d.",
                                static_cast<int>(xActiveMaskDesc->GetDataType())),
                        return GRAPH_FAILED);
        OP_TILING_CHECK(
            static_cast<ge::Format>(ge::GetPrimaryFormat(xActiveMaskDesc->GetStorageFormat())) != ge::FORMAT_ND,
            OP_LOGE(K_INNER_DEBUG, "x_active_mask format is invalid, should be FORMAT_ND."), return GRAPH_FAILED);
    }
    outIsActiveMask = isActiveMask;
    return ge::GRAPH_SUCCESS;
}

// 校验 weight1 动态输入，返回 N / expertPerRank / w1DataType / w1Format
static ge::graphStatus CheckWeight1Input(gert::TilingContext *context, int64_t hiddenSize, uint32_t &outN,
                                         uint32_t &outExpertPerRank, ge::DataType &outW1DataType,
                                         ge::Format &outW1Format)
{
    auto w1Tensor = context->GetDynamicInputTensor(WEIGHT1_INDEX, 0);
    OP_TILING_CHECK(w1Tensor == nullptr, OP_LOGE(K_INNER_DEBUG, "weight1 tensor is null."), return GRAPH_FAILED);

    auto w1Desc = context->GetDynamicInputDesc(WEIGHT1_INDEX, 0);
    OP_TILING_CHECK(w1Desc == nullptr, OP_LOGE(K_INNER_DEBUG, "weight1 desc is null."), return GRAPH_FAILED);
    ge::DataType w1DataType = w1Desc->GetDataType();
    OP_TILING_CHECK(w1DataType != ge::DT_BF16 && w1DataType != ge::DT_INT8 && w1DataType != ge::DT_INT4,
                    OP_LOGE(K_INNER_DEBUG, "weight1 dataType is invalid, should be BF16/INT8/INT4, but got %d.",
                            static_cast<int>(w1DataType)),
                    return GRAPH_FAILED);
    ge::Format w1Format = static_cast<ge::Format>(ge::GetPrimaryFormat(w1Desc->GetStorageFormat()));
    OP_TILING_CHECK(((w1Format != ge::FORMAT_ND && w1DataType == ge::DT_BF16) ||
                     (w1Format != ge::FORMAT_FRACTAL_NZ && (w1DataType == ge::DT_INT8 || w1DataType == ge::DT_INT4))),
                    OP_LOGE(K_INNER_DEBUG,
                            "weight1 format is invalid. "
                            "Expected: (FORMAT_ND with BF16) or (FORMAT_FRACTAL_NZ with INT8/INT4). "
                            "Actual: format=%d, dataType=%d",
                            w1Format, w1DataType),
                    return GRAPH_FAILED);

    uint32_t w1TensorDims = w1Tensor->GetOriginShape().GetDimNum();
    uint32_t N = w1Tensor->GetStorageShape().GetDim(w1TensorDims - 1);

    uint32_t expertPerRank = GetDynamicInputTensorListLen(context, WEIGHT1_INDEX);
    for (uint32_t i = 0; i < expertPerRank; i++) {
        auto wTensorI = context->GetDynamicInputTensor(WEIGHT1_INDEX, i);
        OP_TILING_CHECK(wTensorI == nullptr, OP_LOGE(K_INNER_DEBUG, "weight1[%u] tensor is null.", i),
                        return GRAPH_FAILED);
        OP_TILING_CHECK(wTensorI->GetOriginShape().GetDimNum() != TWO_DIMS,
                        OP_LOGE(K_INNER_DEBUG, "weight1[%u] must be 2-dimension, but got %lu dim.", i,
                                wTensorI->GetOriginShape().GetDimNum()),
                        return GRAPH_FAILED);
        int64_t w1IDim0 = wTensorI->GetStorageShape().GetDim(0);
        OP_TILING_CHECK(w1IDim0 != hiddenSize,
                        OP_LOGE(K_INNER_DEBUG,
                                "weight1[%u]'s dim0 not equal to hidden_size, "
                                "weight1[%u]'s dim0 = %ld, hidden_size = %ld.",
                                i, i, w1IDim0, hiddenSize),
                        return GRAPH_FAILED);
    }

    outN = N;
    outExpertPerRank = expertPerRank;
    outW1DataType = w1DataType;
    outW1Format = w1Format;
    return ge::GRAPH_SUCCESS;
}

// 校验 weight2 动态输入，与 weight1 做交叉校验
static ge::graphStatus CheckWeight2Input(gert::TilingContext *context, int64_t hiddenSize, uint32_t N,
                                         uint32_t expertPerRank, ge::DataType w1DataType, ge::Format w1Format,
                                         ge::DataType &outW2DataType)
{
    auto w2Tensor = context->GetDynamicInputTensor(WEIGHT2_INDEX, 0);
    OP_TILING_CHECK(w2Tensor == nullptr, OP_LOGE(K_INNER_DEBUG, "weight2 tensor is null."), return GRAPH_FAILED);

    auto w2Desc = context->GetDynamicInputDesc(WEIGHT2_INDEX, 0);
    OP_TILING_CHECK(w2Desc == nullptr, OP_LOGE(K_INNER_DEBUG, "weight2 desc is null."), return GRAPH_FAILED);
    ge::DataType w2DataType = w2Desc->GetDataType();
    OP_TILING_CHECK(w2DataType != ge::DT_BF16 && w2DataType != ge::DT_INT8 && w2DataType != ge::DT_INT4,
                    OP_LOGE(K_INNER_DEBUG, "weight2 dataType is invalid, should be BF16/INT8/INT4, but got %d.",
                            static_cast<int>(w2DataType)),
                    return GRAPH_FAILED);

    // weight1 和 weight2 数据类型必须一致
    OP_TILING_CHECK(w1DataType != w2DataType,
                    OP_LOGE(K_INNER_DEBUG,
                            "weight1 and weight2 must have the same dataType, "
                            "weight1 dataType = %d, weight2 dataType = %d.",
                            static_cast<int>(w1DataType), static_cast<int>(w2DataType)),
                    return GRAPH_FAILED);
    ge::Format w2Format = static_cast<ge::Format>(ge::GetPrimaryFormat(w2Desc->GetStorageFormat()));
    OP_TILING_CHECK(w2Format != ge::FORMAT_ND && w2Format != ge::FORMAT_FRACTAL_NZ,
                    OP_LOGE(K_INNER_DEBUG, "weight2 format is invalid, should be FORMAT_ND or FORMAT_FRACTAL_NZ."),
                    return GRAPH_FAILED);
    OP_TILING_CHECK(w1Format != w2Format,
                    OP_LOGE(K_INNER_DEBUG,
                            "weight1 and weight2 must have the same format, "
                            "weight1 format = %d, weight2 format = %d.",
                            static_cast<int>(w1Format), static_cast<int>(w2Format)),
                    return GRAPH_FAILED);

    uint32_t w2ExpertPerRank = GetDynamicInputTensorListLen(context, WEIGHT2_INDEX);
    OP_TILING_CHECK(w2ExpertPerRank != expertPerRank,
                    OP_LOGE(K_INNER_DEBUG,
                            "weight2's listLen not equal to weight1's listLen, "
                            "weight2's listLen = %u, weight1's listLen = %u.",
                            w2ExpertPerRank, expertPerRank),
                    return GRAPH_FAILED);

    uint32_t n2 = N / 2;
    OP_TILING_CHECK(n2 % HIDDEN_SIZE_ALIGN != 0,
                    OP_LOGE(K_INNER_DEBUG, "weight2's dim0(intermediate_hidden) should be %ld aligned, but got %ld.",
                            HIDDEN_SIZE_ALIGN, n2),
                    return GRAPH_FAILED);

    OP_TILING_CHECK(n2 < MIN_INTERMEDIATE_HIDDEN,
                    OP_LOGE(K_INNER_DEBUG,
                            "weight2's dim0(intermediate_hidden) is invalid, should be >= %ld, but got %ld.",
                            MIN_INTERMEDIATE_HIDDEN, n2),
                    return GRAPH_FAILED);

    for (uint32_t i = 0; i < expertPerRank; i++) {
        auto wTensorI = context->GetDynamicInputTensor(WEIGHT2_INDEX, i);
        OP_TILING_CHECK(wTensorI == nullptr, OP_LOGE(K_INNER_DEBUG, "weight2[%u] tensor is null.", i),
                        return GRAPH_FAILED);
        OP_TILING_CHECK(wTensorI->GetOriginShape().GetDimNum() != TWO_DIMS,
                        OP_LOGE(K_INNER_DEBUG, "weight2[%u] must be 2-dimension, but got %lu dim.", i,
                                wTensorI->GetOriginShape().GetDimNum()),
                        return GRAPH_FAILED);
        int64_t w2IDim0 = wTensorI->GetStorageShape().GetDim(0);
        int64_t w2IDim1 = wTensorI->GetStorageShape().GetDim(1);
        OP_TILING_CHECK(w2IDim0 != n2,
                        OP_LOGE(K_INNER_DEBUG,
                                "weight2[%u]'s dim0 not equal to intermediate_hidden, weight2[%u]'s dim0 = %ld, "
                                "intermediate_hidden = %u.",
                                i, i, w2IDim0, n2),
                        return GRAPH_FAILED);
        OP_TILING_CHECK(w2IDim1 != hiddenSize,
                        OP_LOGE(K_INNER_DEBUG,
                                "weight2[%u]'s dim1 not equal to hidden_size, "
                                "weight2[%u]'s dim1 = %ld, hidden_size = %ld.",
                                i, i, w2IDim1, hiddenSize),
                        return GRAPH_FAILED);
    }

    outW2DataType = w2DataType;
    return ge::GRAPH_SUCCESS;
}

// 校验 weight_scales1/weight_scales2 输入（可选），quant 权重时必选
static ge::graphStatus CheckWeightScaleInput(gert::TilingContext *context, uint32_t inputIndex, const char *inputName,
                                             uint32_t expertPerRank, int64_t dim1Expected, const char *dim1Name)
{
    auto wScaleTensor = context->GetDynamicInputTensor(inputIndex, 0);
    OP_TILING_CHECK(wScaleTensor == nullptr,
                    OP_LOGE(K_INNER_DEBUG, "%s is required when weight1/weight2 is INT8 or INT4.", inputName),
                    return GRAPH_FAILED);

    auto wScaleDesc = context->GetDynamicInputDesc(inputIndex, 0);
    OP_TILING_CHECK(wScaleDesc == nullptr, OP_LOGE(K_INNER_DEBUG, "%s desc is null.", inputName), return GRAPH_FAILED);
    OP_TILING_CHECK(wScaleDesc->GetDataType() != ge::DT_UINT64,
                    OP_LOGE(K_INNER_DEBUG, "%s dataType is invalid, should be UINT64, but got %d.", inputName,
                            static_cast<int>(wScaleDesc->GetDataType())),
                    return GRAPH_FAILED);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(wScaleDesc->GetStorageFormat())) != ge::FORMAT_ND,
                    OP_LOGE(K_INNER_DEBUG, "%s format is invalid, should be FORMAT_ND.", inputName),
                    return GRAPH_FAILED);

    uint32_t scaleListLen = GetDynamicInputTensorListLen(context, inputIndex);

    OP_TILING_CHECK(scaleListLen != expertPerRank,
                    OP_LOGE(K_INNER_DEBUG,
                            "%s's listLen not equal to weight1's listLen, %s's listLen = %u, weight1's listLen = %u.",
                            inputName, inputName, scaleListLen, expertPerRank),
                    return GRAPH_FAILED);
    // 多个scale tensor，逐个检查
    for (uint32_t i = 0; i < scaleListLen; i++) {
        auto sTensorI = context->GetDynamicInputTensor(inputIndex, i);
        OP_TILING_CHECK(sTensorI == nullptr, OP_LOGE(K_INNER_DEBUG, "%s[%u] tensor is null.", inputName, i),
                        return GRAPH_FAILED);
        OP_TILING_CHECK(sTensorI->GetOriginShape().GetDimNum() != ONE_DIM,
                        OP_LOGE(K_INNER_DEBUG, "%s[%u] must be 1-dimension, but got %lu dim.", inputName, i,
                                sTensorI->GetOriginShape().GetDimNum()),
                        return GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

// 校验 bias1/bias2 输入, 当且仅当 INT4 权重时存在 bias
static ge::graphStatus CheckBiasInput(gert::TilingContext *context, uint32_t inputIndex, const char *inputName,
                                      uint32_t expertPerRank, int64_t dim0Expected, const char *dim0Name)
{
    const gert::StorageShape *biasStorageShape = context->GetDynamicInputShape(inputIndex, 0);
    OP_TILING_CHECK(biasStorageShape == nullptr,
                    OP_LOGE(K_INNER_DEBUG, "%s is required when weight1/weight2 is INT4.", inputName),
                    return GRAPH_FAILED);
    auto biasTensor = context->GetDynamicInputTensor(inputIndex, 0);
    OP_TILING_CHECK(biasTensor == nullptr, OP_LOGE(K_INNER_DEBUG, "%s tensor is null.", inputName),
                    return GRAPH_FAILED);

    auto biasDesc = context->GetDynamicInputDesc(inputIndex, 0);
    OP_TILING_CHECK(biasDesc == nullptr, OP_LOGE(K_INNER_DEBUG, "%s desc is null.", inputName), return GRAPH_FAILED);
    OP_TILING_CHECK(biasDesc->GetDataType() != ge::DT_FLOAT,
                    OP_LOGE(K_INNER_DEBUG, "%s dataType is invalid, should be FP32, but got %d.", inputName,
                            static_cast<int>(biasDesc->GetDataType())),
                    return GRAPH_FAILED);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(biasDesc->GetStorageFormat())) != ge::FORMAT_ND,
                    OP_LOGE(K_INNER_DEBUG, "%s format is invalid, should be FORMAT_ND.", inputName),
                    return GRAPH_FAILED);

    uint32_t biasListLen = GetDynamicInputTensorListLen(context, inputIndex);
    OP_TILING_CHECK(biasListLen != expertPerRank,
                    OP_LOGE(K_INNER_DEBUG,
                            "%s's listLen not equal to weight1's listLen, %s's listLen = %u, weight1's listLen = %u.",
                            inputName, inputName, biasListLen, expertPerRank),
                    return GRAPH_FAILED);

    for (uint32_t i = 0; i < biasListLen; i++) {
        auto bTensorI = context->GetDynamicInputTensor(inputIndex, i);
        OP_TILING_CHECK(bTensorI == nullptr, OP_LOGE(K_INNER_DEBUG, "%s[%u] tensor is null.", inputName, i),
                        return GRAPH_FAILED);
        OP_TILING_CHECK(bTensorI->GetOriginShape().GetDimNum() != ONE_DIM,
                        OP_LOGE(K_INNER_DEBUG, "%s[%u] must be 1-dimension, but got %lu dim.", inputName, i,
                                bTensorI->GetOriginShape().GetDimNum()),
                        return GRAPH_FAILED);
        uint32_t bDim0 = bTensorI->GetStorageShape().GetDim(0);
        OP_TILING_CHECK(bDim0 != static_cast<uint32_t>(dim0Expected),
                        OP_LOGE(K_INNER_DEBUG, "%s[%u]'s dim0 not equal to %s, %s[%u]'s dim0 = %u, %s = %ld.",
                                inputName, i, dim0Name, inputName, i, bDim0, dim0Name, dim0Expected),
                        return GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MegaMoeA2A3CheckShapeAndSetTiling(gert::TilingContext *context, MegaMoeA2A3TilingData &info)
{
    // ==================== 1. 输入张量非空校验 ====================
    const gert::StorageShape *contextStorageShape = context->GetInputShape(CONTEXT_INDEX);
    const gert::StorageShape *xStorageShape = context->GetInputShape(X_INDEX);
    const gert::StorageShape *topkIdsStorageShape = context->GetInputShape(TOPK_IDS_INDEX);
    const gert::StorageShape *topkWeightsStorageShape = context->GetInputShape(TOPK_WEIGHTS_INDEX);
    OP_TILING_CHECK(contextStorageShape == nullptr, OP_LOGE(K_INNER_DEBUG, "context shape is null."),
                    return GRAPH_FAILED);
    OP_TILING_CHECK(xStorageShape == nullptr, OP_LOGE(K_INNER_DEBUG, "x shape is null."), return GRAPH_FAILED);
    OP_TILING_CHECK(topkIdsStorageShape == nullptr, OP_LOGE(K_INNER_DEBUG, "topk_ids shape is null."),
                    return GRAPH_FAILED);
    OP_TILING_CHECK(topkWeightsStorageShape == nullptr, OP_LOGE(K_INNER_DEBUG, "topk_weights shape is null."),
                    return GRAPH_FAILED);

    // ==================== 2. context 输入校验 ====================
    OP_TILING_CHECK(CheckContextInput(context, contextStorageShape) != ge::GRAPH_SUCCESS,
                    OP_LOGE(K_INNER_DEBUG, "CheckContextInput failed."), return GRAPH_FAILED);

    // ==================== 3. x 输入校验 ====================
    int64_t bs = 0;
    int64_t hiddenSize = 0;
    OP_TILING_CHECK(CheckXInput(context, xStorageShape, bs, hiddenSize) != ge::GRAPH_SUCCESS,
                    OP_LOGE(K_INNER_DEBUG, "CheckXInput failed."), return GRAPH_FAILED);

    // ==================== 4. topk_ids 输入校验 ====================
    int64_t topK = 0;
    OP_TILING_CHECK(CheckTopkIdsInput(context, topkIdsStorageShape, bs, topK) != ge::GRAPH_SUCCESS,
                    OP_LOGE(K_INNER_DEBUG, "CheckTopkIdsInput failed."), return GRAPH_FAILED);

    // ==================== 5. topk_weights 输入校验 ====================
    OP_TILING_CHECK(CheckTopkWeightsInput(context, topkWeightsStorageShape, bs, topK) != ge::GRAPH_SUCCESS,
                    OP_LOGE(K_INNER_DEBUG, "CheckTopkWeightsInput failed."), return GRAPH_FAILED);

    // ==================== 6. weight1 输入校验 ====================
    uint32_t N = 0;
    uint32_t expertPerRank = 0;
    ge::DataType w1DataType = ge::DT_UNDEFINED;
    ge::Format w1Format = ge::FORMAT_RESERVED;
    OP_TILING_CHECK(CheckWeight1Input(context, hiddenSize, N, expertPerRank, w1DataType, w1Format) != ge::GRAPH_SUCCESS,
                    OP_LOGE(K_INNER_DEBUG, "CheckWeight1Input failed."), return GRAPH_FAILED);

    info.isQuantRouting = (w1DataType == ge::DT_FLOAT16 || w1DataType == ge::DT_BF16) ? 0U : 1U;
    info.isW4A8 = (w1DataType == ge::DT_INT32 || w1DataType == ge::DT_INT4) ? 1U : 0U;

    // expertPerRank 范围校验
    OP_TILING_CHECK(expertPerRank < MIN_EXPERT_PER_RANK || expertPerRank > MAX_EXPERT_PER_RANK,
                    OP_LOGE(K_INNER_DEBUG, "expertPerRank is invalid, should be in [%ld, %ld], but got %u.",
                            MIN_EXPERT_PER_RANK, MAX_EXPERT_PER_RANK, expertPerRank),
                    return GRAPH_FAILED);

    // ==================== 7. weight2 输入校验 ====================
    ge::DataType w2DataType = ge::DT_UNDEFINED;
    OP_TILING_CHECK(CheckWeight2Input(context, hiddenSize, N, expertPerRank, w1DataType, w1Format, w2DataType) !=
                        ge::GRAPH_SUCCESS,
                    OP_LOGE(K_INNER_DEBUG, "CheckWeight2Input failed."), return GRAPH_FAILED);

    // ==================== 量化权重必须提供 scale 校验 ====================
    // 当 weight1 和 weight2 的类型都为 INT8 或 INT4 时，GMM1/GMM2 需要进行反量化
    // 此时 weight_scales1/weight_scales2 必须存在
    if (w1DataType == ge::DT_INT8 || w1DataType == ge::DT_INT4) {
        // ==================== 8. weight_scales1 输入校验（可选） ====================
        // weight_scales1：可选输入，weight1/weight2 为 INT8 或 INT4 时需要，GMM1 反量化参数
        // 支持两种场景：
        //   1. 传入一个 TensorList，list 包含一个 2 维 tensor，shape 为 (num_experts_per_rank, N)
        //   2. 传入一个 TensorList，list 包含 num_experts_per_rank 个 tensor，每个 tensor 的 shape 为 (N,)
        OP_TILING_CHECK(CheckWeightScaleInput(context, WEIGHT_SCALES1_INDEX, "weight_scales1", expertPerRank, N, "N") !=
                            ge::GRAPH_SUCCESS,
                        OP_LOGE(K_INNER_DEBUG, "CheckWeightScaleInput for weight_scales1 failed."),
                        return GRAPH_FAILED);

        // ==================== 9. weight_scales2 输入校验（可选） ====================
        // weight_scales2：可选输入，weight1/weight2 为 INT8 或 INT4 时需要，GMM2 反量化参数
        // 支持两种场景：
        //   1. 传入一个 TensorList，list 包含一个 2 维 tensor，shape 为 (num_experts_per_rank, hidden_size)
        //   2. 传入一个 TensorList，list 包含 num_experts_per_rank 个 tensor，
        //      每个 tensor 的 shape 为 (hidden_size,)
        OP_TILING_CHECK(CheckWeightScaleInput(context, WEIGHT_SCALES2_INDEX, "weight_scales2", expertPerRank,
                                              hiddenSize, "hidden_size") != ge::GRAPH_SUCCESS,
                        OP_LOGE(K_INNER_DEBUG, "CheckWeightScaleInput for weight_scales2 failed."),
                        return GRAPH_FAILED);
    } else {
        uint32_t weight1ListLen = GetDynamicInputTensorListLen(context, WEIGHT_SCALES1_INDEX);
        uint32_t weight2ListLen = GetDynamicInputTensorListLen(context, WEIGHT_SCALES2_INDEX);
        // 若tensor为空placeholder（shape为0，由CreateEmptyTensor创建），视为未提供
        bool hasWeight1Scale =
            (weight1ListLen > 0U) && (context->GetDynamicInputDesc(WEIGHT_SCALES1_INDEX, 0) != nullptr) &&
            (context->GetDynamicInputShape(WEIGHT_SCALES1_INDEX, 0)->GetStorageShape().GetDim(0) > 0);
        bool hasWeight2Scale =
            (weight2ListLen > 0U) && (context->GetDynamicInputDesc(WEIGHT_SCALES2_INDEX, 0) != nullptr) &&
            (context->GetDynamicInputShape(WEIGHT_SCALES2_INDEX, 0)->GetStorageShape().GetDim(0) > 0);
        OP_TILING_CHECK(hasWeight1Scale || hasWeight2Scale,
                        OP_LOGE(K_INNER_DEBUG,
                                "weight_scale is only supported for INT8/INT4 data type, "
                                "but got w1DataType=%d, got weight1ListLen=%d, weight2ListLen=%d",
                                w1DataType, weight1ListLen, weight2ListLen),
                        return GRAPH_FAILED);
    }

    if (w1DataType == ge::DT_INT4) {
        // ==================== 10. bias1 输入校验 ====================
        // 当 weight1 和 weight2 数据类型都为 INT4 时，bias1 必须存在
        OP_TILING_CHECK(CheckBiasInput(context, BIAS1_INDEX, "bias1", expertPerRank, N, "N") != ge::GRAPH_SUCCESS,
                        OP_LOGE(K_INNER_DEBUG, "CheckBiasInput for bias1 failed."), return GRAPH_FAILED);

        // ==================== 11. bias2 输入校验 ====================
        // 当 weight1 和 weight2 数据类型都为 INT4 时，bias2 必须存在
        OP_TILING_CHECK(CheckBiasInput(context, BIAS2_INDEX, "bias2", expertPerRank, hiddenSize, "hiddenSize") !=
                            ge::GRAPH_SUCCESS,
                        OP_LOGE(K_INNER_DEBUG, "CheckBiasInput for bias2 failed."), return GRAPH_FAILED);
    } else {
        uint32_t bias1ListLen = GetDynamicInputTensorListLen(context, BIAS1_INDEX);
        uint32_t bias2ListLen = GetDynamicInputTensorListLen(context, BIAS2_INDEX);
        // 若tensor为空placeholder（shape为0，由CreateEmptyTensor创建），视为未提供
        bool hasBias1 = (bias1ListLen > 0U) && (context->GetDynamicInputDesc(BIAS1_INDEX, 0) != nullptr) &&
                        (context->GetDynamicInputShape(BIAS1_INDEX, 0)->GetStorageShape().GetDim(0) > 0);
        bool hasBias2 = (bias2ListLen > 0U) && (context->GetDynamicInputDesc(BIAS2_INDEX, 0) != nullptr) &&
                        (context->GetDynamicInputShape(BIAS2_INDEX, 0)->GetStorageShape().GetDim(0) > 0);
        OP_TILING_CHECK(hasBias1 || hasBias2,
                        OP_LOGE(K_INNER_DEBUG,
                                "bias is only supported for INT4 data type, "
                                "but got w1DataType=%d, got bias1ListLen=%d, bias2ListLen=%d",
                                w1DataType, bias1ListLen, bias2ListLen),
                        return GRAPH_FAILED);
    }

    // ==================== 12. x_active_mask 输入校验（可选） ====================
    bool isActiveMask = false;
    OP_TILING_CHECK(CheckXActiveMaskInput(context, bs, isActiveMask) != ge::GRAPH_SUCCESS,
                    OP_LOGE(K_INNER_DEBUG, "CheckXActiveMaskInput failed."), return GRAPH_FAILED);

    // ==================== 13. scales 输入校验（可选，预留） ====================
    const gert::StorageShape *scalesStorageShape = context->GetOptionalInputShape(SCALES_INDEX);
    OP_TILING_CHECK(scalesStorageShape != nullptr,
                    OP_LOGE(K_INNER_DEBUG, "scales is not supported yet, please pass nullptr."), return GRAPH_FAILED);

    // ==================== 14. 设置 tiling data ====================
    info.M = static_cast<uint32_t>(bs);
    info.N = N;
    info.K = static_cast<uint32_t>(hiddenSize);
    info.expertPerRank = expertPerRank;
    info.topK = static_cast<uint32_t>(topK);
    info.listLen = expertPerRank;

    OP_LOGD(K_INNER_DEBUG, "bs=%u", info.M);
    OP_LOGD(K_INNER_DEBUG, "K=%u", info.K);
    OP_LOGD(K_INNER_DEBUG, "N=%u", info.N);
    OP_LOGD(K_INNER_DEBUG, "expertPerRank=%u", info.expertPerRank);
    OP_LOGD(K_INNER_DEBUG, "topK=%u", info.topK);
    OP_LOGD(K_INNER_DEBUG, "listLen=%u", info.listLen);
    OP_LOGD(K_INNER_DEBUG, "isActiveMask=%d", static_cast<int>(isActiveMask));

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MegaMoeA2A3GetPlatformInfoAndSetTiling(gert::TilingContext *context, MegaMoeA2A3TilingData &info)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize = 0U;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    info.aivNum = aivNum;
    info.totalUbSize = ubSize;

    OP_LOGD(K_INNER_DEBUG, "aivNum=%d", info.aivNum);
    OP_LOGD(K_INNER_DEBUG, "ubSize=%lu", info.totalUbSize);

    return ge::GRAPH_SUCCESS;
}

// 表示通信亲和内存布局算法，当前版本仅支持 ""。
static ge::graphStatus MegaMoeA2A3CommAlg(const gert::TilingContext *context)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(K_INNER_DEBUG, "attrs is null."), return ge::GRAPH_FAILED);
    auto commAlg = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_COMM_ALG_INDEX));
    OP_TILING_CHECK(commAlg == nullptr, OP_LOGE(K_INNER_DEBUG, "commAlg is null."), return ge::GRAPH_FAILED);

    if (strlen(commAlg) > 0) {
        OP_LOGE(K_INNER_DEBUG, "Attr commAlg is invalid, current version only supports \"\", but got \"%s\".", commAlg);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MegaMoeA2A3CheckHcclBuffSize(const gert::TilingContext *context, MegaMoeA2A3TilingData &info)
{
    const char *nodeName = K_INNER_DEBUG;
    // info.cclBufferSize是winIn和winOut空间之和（两者相等），这里只通过winIn校验
    int64_t cclBufferSize = info.cclBufferSize / 2;
    OP_LOGD(nodeName, "cclBufferSize = %ld Bytes (%ld MB).", cclBufferSize,
            ops::CeilDiv(cclBufferSize, static_cast<int64_t>(MB_SIZE)));

    int64_t h = info.K;
    int64_t maxRecvTokenNum = info.maxRecvTokenNum;
    int64_t epWorldSize = info.epWorldSize;
    int64_t expertPerRank = info.moeExpertNum / epWorldSize;
    bool isQuantRouting = (info.isQuantRouting != 0U);
    bool isW4A8 = (info.isW4A8 != 0U);

    std::string socVersion = mc2tiling::GetSocVersion(context);
    bool isA3 = (socVersion == "Ascend910_93");

    int64_t leastCclBufferSize =
        CalcLeastCclBufferSize(maxRecvTokenNum, h, epWorldSize, expertPerRank, isQuantRouting, isW4A8, isA3,
                               static_cast<int64_t>(info.M), static_cast<int64_t>(info.topK));

    OP_TILING_CHECK(cclBufferSize < leastCclBufferSize,
                    OP_LOGE(nodeName,
                            "cclBufferSize(%ld Bytes, %ld MB) should be >= leastCclBufferSize(%ld Bytes, %ld MB). "
                            "maxRecvTokenNum=%ld, h=%ld, epWorldSize=%ld, expertPerRank=%ld, "
                            "isQuantRouting=%d, isW4A8=%d, isA3=%d. "
                            "Suggestion: set hccl buffsize to %ld MB.",
                            cclBufferSize, cclBufferSize / MB_SIZE, leastCclBufferSize, leastCclBufferSize / MB_SIZE,
                            maxRecvTokenNum, h, epWorldSize, expertPerRank, isQuantRouting, isW4A8, isA3,
                            (leastCclBufferSize + MB_SIZE) / MB_SIZE),
                    return ge::GRAPH_FAILED);
    OP_LOGD(nodeName, "cclBufferSize is %ld, leastCclBufferSize is %ld", cclBufferSize, leastCclBufferSize);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MegaMoeA2A3CheckOutputTensor(gert::TilingContext *context, const MegaMoeA2A3TilingData &info)
{
    // ==================== 1. y 输出校验 ====================
    const gert::StorageShape *yStorageShape = context->GetOutputShape(OUTPUT_Y_INDEX);
    OP_TILING_CHECK(yStorageShape == nullptr, OP_LOGE(K_INNER_DEBUG, "y shape is null."), return GRAPH_FAILED);

    // 维度数校验：必须为 2 维
    OP_TILING_CHECK(
        yStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE(K_INNER_DEBUG, "y must be 2-dimension, but got %lu dim.", yStorageShape->GetStorageShape().GetDimNum()),
        return GRAPH_FAILED);

    int64_t yDim0 = yStorageShape->GetStorageShape().GetDim(0);
    int64_t yDim1 = yStorageShape->GetStorageShape().GetDim(1);

    // Shape 一致性校验：y 的 shape 应该与 x 的 shape 一致 [M, K]
    OP_TILING_CHECK(yDim0 != static_cast<int64_t>(info.M),
                    OP_LOGE(K_INNER_DEBUG, "y's dim0 not equal to M, y's dim0 = %ld, M = %u.", yDim0, info.M),
                    return GRAPH_FAILED);
    OP_TILING_CHECK(yDim1 != static_cast<int64_t>(info.K),
                    OP_LOGE(K_INNER_DEBUG, "y's dim1 not equal to K, y's dim1 = %ld, K = %u.", yDim1, info.K),
                    return GRAPH_FAILED);

    // 数据类型校验
    auto yDesc = context->GetOutputDesc(OUTPUT_Y_INDEX);
    OP_TILING_CHECK(yDesc == nullptr, OP_LOGE(K_INNER_DEBUG, "y desc is null."), return GRAPH_FAILED);
    ge::DataType yDataType = yDesc->GetDataType();
    OP_TILING_CHECK(yDataType != ge::DT_FLOAT16 && yDataType != ge::DT_BF16,
                    OP_LOGE(K_INNER_DEBUG, "y dataType is invalid, should be FLOAT16 or BF16, but got %d.",
                            static_cast<int>(yDataType)),
                    return GRAPH_FAILED);

    OP_LOGD(K_INNER_DEBUG, "y dim0 = %ld", yDim0);
    OP_LOGD(K_INNER_DEBUG, "y dim1 = %ld", yDim1);
    OP_LOGD(K_INNER_DEBUG, "y dataType = %d", static_cast<int>(yDataType));

    // ==================== 2. expert_token_nums 输出校验 ====================
    const gert::StorageShape *expertTokenNumsStorageShape = context->GetOutputShape(OUTPUT_EXPERT_TOKEN_NUMS_INDEX);
    OP_TILING_CHECK(expertTokenNumsStorageShape == nullptr, OP_LOGE(K_INNER_DEBUG, "expert_token_nums shape is null."),
                    return GRAPH_FAILED);

    // 维度数校验：必须为 1 维
    OP_TILING_CHECK(expertTokenNumsStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
                    OP_LOGE(K_INNER_DEBUG, "expert_token_nums must be 1-dimension, but got %lu dim.",
                            expertTokenNumsStorageShape->GetStorageShape().GetDimNum()),
                    return GRAPH_FAILED);

    int64_t expertTokenNumsDim0 = expertTokenNumsStorageShape->GetStorageShape().GetDim(0);

    // Shape 校验：expert_token_nums 的 dim0 应该等于 expertPerRank（本卡专家数量）
    OP_TILING_CHECK(expertTokenNumsDim0 != static_cast<int64_t>(info.expertPerRank),
                    OP_LOGE(K_INNER_DEBUG,
                            "expert_token_nums's dim0 not equal to expertPerRank, "
                            "expert_token_nums's dim0 = %ld, expertPerRank = %u.",
                            expertTokenNumsDim0, info.expertPerRank),
                    return GRAPH_FAILED);

    // 数据类型校验
    auto expertTokenNumsDesc = context->GetOutputDesc(OUTPUT_EXPERT_TOKEN_NUMS_INDEX);
    OP_TILING_CHECK(expertTokenNumsDesc == nullptr, OP_LOGE(K_INNER_DEBUG, "expert_token_nums desc is null."),
                    return GRAPH_FAILED);
    OP_TILING_CHECK(expertTokenNumsDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE(K_INNER_DEBUG, "expert_token_nums dataType is invalid, should be INT32, but got %d.",
                            static_cast<int>(expertTokenNumsDesc->GetDataType())),
                    return GRAPH_FAILED);

    OP_LOGD(K_INNER_DEBUG, "expert_token_nums dim0 = %ld", expertTokenNumsDim0);
    OP_LOGD(K_INNER_DEBUG, "expert_token_nums dataType = %d", static_cast<int>(expertTokenNumsDesc->GetDataType()));

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MegaMoeA2A3TilingFuncImpl(gert::TilingContext *context)
{
    // 涉及SyncAll，设置batch mode模式，所有核同时启动
    uint32_t batch_mode = 1U;
    auto ret = context->SetScheduleMode(batch_mode);

    // 1. tilingData
    MegaMoeTilingDataQuant *tilingData = context->GetTilingData<MegaMoeTilingDataQuant>();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE(K_INNER_DEBUG, "tilingData is nullptr."), return ge::GRAPH_FAILED);
    OP_LOGI(K_INNER_DEBUG, "MegaMoeA2A3 get tilingData.");
    MegaMoeA2A3TilingData &info = tilingData->common;
    OP_LOGI(K_INNER_DEBUG, "MegaMoeA2A3 get tilingData info.");

    OP_TILING_CHECK(MegaMoeA2A3CommAlg(context) != ge::GRAPH_SUCCESS,
                    VECTOR_INNER_ERR_REPORT_TILING(K_INNER_DEBUG, "MegaMoeA2A3 CheckCommAlg Failed"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(MegaMoeA2A3CheckShapeAndSetTiling(context, info) != ge::GRAPH_SUCCESS,
                    OP_LOGE(K_INNER_DEBUG, "MegaMoeA2A3 CheckShapeAndSetTiling Failed"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(MegaMoeA2A3CheckAttrAndSetTiling(context, info) != ge::GRAPH_SUCCESS,
                    OP_LOGE(K_INNER_DEBUG, "MegaMoeA2A3 CheckAttrAndSetTiling Failed"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(MegaMoeA2A3CheckOutputTensor(context, info) != ge::GRAPH_SUCCESS,
                    OP_LOGE(K_INNER_DEBUG, "MegaMoeA2A3 CheckOutputTensor Failed"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(MegaMoeA2A3GetPlatformInfoAndSetTiling(context, info) != ge::GRAPH_SUCCESS,
                    OP_LOGE(K_INNER_DEBUG, "MegaMoeA2A3 GetPlatformInfoAndSetTiling Failed"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(MegaMoeA2A3CheckHcclBuffSize(context, info) != ge::GRAPH_SUCCESS,
                    OP_LOGE(K_INNER_DEBUG, "MegaMoeA2A3 CheckHcclBuffSize Failed"), return ge::GRAPH_FAILED);

    // 2. set blockDim
    uint32_t blockDim = 1U;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto aicNum = ascendcPlatform.GetCoreNumAic();
    auto aivNum = ascendcPlatform.GetCoreNumAiv();
    blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    context->SetBlockDim(blockDim);

    // 3. set tiling key
    uint32_t quantMode = info.isQuantRouting != 0U ? MEGA_MOE_QUANT_MODE_PER_TENSOR : MEGA_MOE_QUANT_MODE_NO_QUANT;
    uint32_t quantOutType = MEGA_MOE_QUANT_OUT_TYPE_UNDEFINED;
    if (info.dispatchQuantOutDtype == static_cast<int32_t>(ge::DT_INT8)) {
        quantOutType = MEGA_MOE_QUANT_OUT_TYPE_INT8;
    }
    if (quantMode == MEGA_MOE_QUANT_MODE_NO_QUANT) {
        OP_TILING_CHECK(
            quantOutType != MEGA_MOE_QUANT_OUT_TYPE_UNDEFINED,
            OP_LOGE(K_INNER_DEBUG, "dispatch_quant_out_type must be UNDEFINED in non-quant mode, got %u", quantOutType),
            return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK(
            quantOutType != MEGA_MOE_QUANT_OUT_TYPE_INT8,
            OP_LOGE(K_INNER_DEBUG, "dispatch_quant_out_type must be INT8 in quant mode, got %u", quantOutType),
            return ge::GRAPH_FAILED);
    }
    // archCode 在910B时为0, 910_93时为1
    std::string socVersion = mc2tiling::GetSocVersion(context);
    uint32_t archCode = socVersion == "Ascend910B" ? SOC_ASCEND910B : SOC_ASCEND910_93;

    uint64_t tilingKey = GET_TPL_TILING_KEY(static_cast<int64_t>(info.isTransposeW1),
                                            static_cast<int64_t>(info.isTransposeW2), static_cast<int64_t>(quantMode),
                                            static_cast<int64_t>(quantOutType), static_cast<int64_t>(archCode));
    context->SetTilingKey(tilingKey);

    int64_t inuptXDtypeSize = sizeof(int16_t);
    int64_t scaleDim0 = 0;
    int64_t ubSize = 196352;
    int64_t expertCapacity = 0;
    int64_t expertNum = info.expertPerRank * info.worldSize;
    int64_t activeNum = info.M * info.topK;
    int64_t dropPadMode = 0;
    int64_t expertTokensCountOrCumsumFlag = 2;
    bool expertTokensBeforeCapacityFlag = false;
    int64_t quantModeRouting = 1;
    uint64_t initRoutingQuantTilingKey = 0;
    size_t initRoutingWorkspace = 0;

    if (info.isQuantRouting != 0U) {
        MegaMoeTilingDataQuant *quantTilingData = context->GetTilingData<MegaMoeTilingDataQuant>();
        OP_TILING_CHECK(tilingData == nullptr, OP_LOGE(K_INNER_DEBUG, "quantTilingData is nullptr."),
                        return ge::GRAPH_FAILED);
        quantTilingData->common = info;

        MoeInitRoutingQuantV2TilingBase moeInitRoutingQuantV2TilingBase;
        moeInitRoutingQuantV2TilingBase.DoTiling(
            info.M, info.K, info.topK, expertCapacity, expertNum, activeNum, dropPadMode, expertTokensCountOrCumsumFlag,
            expertTokensBeforeCapacityFlag, inuptXDtypeSize, quantModeRouting, scaleDim0, aivNum, ubSize);
        initRoutingQuantTilingKey = moeInitRoutingQuantV2TilingBase.tilingKey_;
        initRoutingWorkspace = moeInitRoutingQuantV2TilingBase.workspaceSize_;
        quantTilingData->moeInitRoutingQuantV2TilingData = moeInitRoutingQuantV2TilingBase.quantTilingData;
        quantTilingData->common.initRoutingQuantTilingKey = initRoutingQuantTilingKey;
    } else {
        MegaMoeTilingDataNonQuant *nonQuantTilingData = context->GetTilingData<MegaMoeTilingDataNonQuant>();
        OP_TILING_CHECK(tilingData == nullptr, OP_LOGE(K_INNER_DEBUG, "quantTilingData is nullptr."),
                        return ge::GRAPH_FAILED);
        nonQuantTilingData->common = info;

        MoeInitRoutingV2TilingBase moeInitRoutingV2TilingBase;
        moeInitRoutingV2TilingBase.DoTiling(info.M, info.K, info.topK, expertCapacity, expertNum, activeNum,
                                            dropPadMode, expertTokensCountOrCumsumFlag, expertTokensBeforeCapacityFlag,
                                            inuptXDtypeSize, quantModeRouting, scaleDim0, aivNum, ubSize);
        initRoutingQuantTilingKey = moeInitRoutingV2TilingBase.tilingKey_;
        initRoutingWorkspace = moeInitRoutingV2TilingBase.workspaceSize_;
        nonQuantTilingData->moeInitRoutingV2TilingData = moeInitRoutingV2TilingBase.moeInitRoutingTilingData;
        nonQuantTilingData->common.initRoutingQuantTilingKey = initRoutingQuantTilingKey;
    }

    OP_LOGD(K_INNER_DEBUG, "initRoutingQuantTilingKey=%lu (isQuantRouting=%u)", initRoutingQuantTilingKey,
            info.isQuantRouting);

    // 4. workspace
    size_t *workSpaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workSpaces == nullptr, OP_LOGE(K_INNER_DEBUG, "workSpaces is nullptr."), return ge::GRAPH_FAILED);

    uint32_t n2 = info.K;
    uint32_t k2 = info.N / 2;
    uint64_t megeMoeWorkspace = 0;
    if (archCode == SOC_ASCEND910_93) {
        megeMoeWorkspace =
            (info.M + 256 - 1) / 256 * 256 * info.topK * sizeof(int32_t) +                 // expandedRowIdx
            info.worldSize * info.worldSize * info.expertPerRank * sizeof(int32_t) +       // cumsum
            info.maxRecvTokenNum * std::max(info.N, n2) * sizeof(int16_t) +                // GMM1&2 Out
            info.maxRecvTokenNum * std::max(info.K, k2) * sizeof(int16_t) +                // GMM1&2 input
            (info.worldSize * (info.expertPerRank + 16 - 1) / 16 * 16 * sizeof(int32_t)) + // SumBeforeRank
            info.worldSize * sizeof(int32_t) * 16;                                         // sync
        if (info.isQuantRouting == 1U) {
            megeMoeWorkspace += info.maxRecvTokenNum * sizeof(float) * 2; // perTokenScale GMM1&2
        }
        if (info.isW4A8) {
            // when W4A8 matrix_A need int8->int4, M->2*M
            megeMoeWorkspace += info.maxRecvTokenNum * std::max(info.N, n2) * sizeof(int16_t); // GMM1&2 Out
        }
    } else if (archCode == SOC_ASCEND910B) {
        uint64_t swigluSize = info.isQuantRouting ? 1UL : 2UL;
        uint64_t paddedExpertNumAligned = (info.worldSize * info.expertPerRank + 1 + 128 - 1) / 128 * 128;
        megeMoeWorkspace = (info.M + 256 - 1) / 256 * 256 * info.topK *
                               sizeof(int32_t) + // expandedRowIdx
                                                 // cumsum + ptrSumBeforeRankForDispatch + ptrSumBeforeRankForCombine
                           paddedExpertNumAligned * (info.worldSize + 2UL) * sizeof(int32_t) +
                           info.maxRecvTokenNum * std::max(info.N, n2) * sizeof(int16_t) + // GMM1&2 Out
                           info.maxRecvTokenNum * std::max(info.K, k2) * swigluSize +      // swiglu out
                           info.worldSize * sizeof(int32_t) * 16UL;                        // sync
        if (info.isQuantRouting) {
            megeMoeWorkspace += info.maxRecvTokenNum * sizeof(float) * 2UL; // perTokenScale GMM1&2
        }
        if (info.isW4A8) {
            megeMoeWorkspace += info.maxRecvTokenNum * std::max(info.N, n2) * sizeof(int16_t); // GMM1&2 Out
        }
    }

    workSpaces[0] = SYSTEM_NEED_WORKSPACE + std::max(megeMoeWorkspace, initRoutingWorkspace);

    OP_LOGI(K_INNER_DEBUG, "Leave MegaMoeA2A3 tiling func.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MegaMoeA2A3TilingFunc(gert::TilingContext *context)
{
    return MegaMoeA2A3TilingFuncImpl(context);
}

struct MegaMoeA2A3CompileInfo {};
ge::graphStatus TilingParseForMegaMoeA2A3(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MegaMoe).Tiling(MegaMoeA2A3TilingFunc).TilingParse<MegaMoeA2A3CompileInfo>(TilingParseForMegaMoeA2A3);

#if RUNTIME_VERSION_NUM >= EXCEPTION_DUMP_SUPPORT_VERSION && METADEF_VERSION_NUM >= EXCEPTION_DUMP_SUPPORT_VERSION
inline void MegaMoeExceptionImplWrapper(aclrtExceptionInfo *args, void *userdata)
{
    Mc2Exception::Mc2ExceptionImpl(args, userdata, "MegaMoe");
}

__attribute__((constructor)) void RegisterMegaMoeExceptionFunc()
{
    int32_t runtimeVersionNum = 0;
    int32_t metadefVersionNum = 0;

    if (aclsysGetVersionNum("runtime", &runtimeVersionNum) != ACL_SUCCESS) {
        OP_LOGW("MegaMoe", "Get runtime version failed when register exception func.");
        return;
    }
    if (aclsysGetVersionNum("metadef", &metadefVersionNum) != ACL_SUCCESS) {
        OP_LOGW("MegaMoe", "Get metadef version failed when register exception func.");
        return;
    }

    if (runtimeVersionNum < EXCEPTION_DUMP_SUPPORT_VERSION || metadefVersionNum < EXCEPTION_DUMP_SUPPORT_VERSION) {
        OP_LOGW("MegaMoe",
                "The runtime(%d) or metadata(%d) version is lower than the version(%d) supporting exception func.",
                runtimeVersionNum, metadefVersionNum, EXCEPTION_DUMP_SUPPORT_VERSION);
        return;
    }

    IMPL_OP(MegaMoe).ExceptionDumpParseFunc(MegaMoeExceptionImplWrapper);
}
#endif
} // namespace MegaMoeA2A3Tiling
