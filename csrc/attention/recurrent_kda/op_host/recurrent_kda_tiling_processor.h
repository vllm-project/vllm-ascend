/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

/*!
 * \file recurrent_kda_tiling_processor.h
 * \brief Tiling processor shared by aclnn tiling and fast kernel launch.
 */

#ifndef RECURRENT_KDA_TILING_PROCESSOR_H
#define RECURRENT_KDA_TILING_PROCESSOR_H

#include "../op_kernel/recurrent_kda_struct.h"
#include "err/ops_err.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "tiling/tiling_api.h"
#include "util/math_util.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

using RecurrentKdaTilingData = RecurrentKda::RecurrentKdaTilingData;

namespace optiling {

static constexpr size_t RKDA_RANK3_QKV_DIM_NUM = 3;
static constexpr size_t RKDA_RANK4_QKV_DIM_NUM = 4;
static constexpr size_t RKDA_RANK2_BETA_DIM_NUM = 2;
static constexpr size_t RKDA_RANK3_BETA_DIM_NUM = 3;
static constexpr size_t RKDA_STATE_DIM_NUM = 4;
static constexpr size_t RKDA_METADATA_RANK1 = 1;
static constexpr size_t RKDA_METADATA_RANK2 = 2;

static constexpr size_t RKDA_DIM_0 = 0;
static constexpr size_t RKDA_DIM_1 = 1;
static constexpr size_t RKDA_DIM_2 = 2;
static constexpr size_t RKDA_DIM_3 = 3;

static constexpr uint32_t RKDA_LAYOUT_BSND = 0;
static constexpr uint32_t RKDA_LAYOUT_TND = 1;
static constexpr size_t RKDA_MAX_MTP = 8;
static constexpr int64_t RKDA_UB_GUARD_BYTES = 2048;
static constexpr size_t RKDA_SYS_WORKSPACE_SIZE = 16U * 1024U * 1024U;

struct RecurrentKdaTilingContext {
    const char *nodeName = "RecurrentKda";
    gert::Shape queryShape;
    gert::Shape keyShape;
    gert::Shape valueShape;
    gert::Shape gateShape;
    gert::Shape betaShape;
    gert::Shape stateShape;
    gert::Shape cuSeqlensShape;
    gert::Shape ssmStateShape;
    gert::Shape aLogShape;
    gert::Shape dtBiasShape;
    gert::Shape acceptedTokensShape;
    float scale = 1.0f;
    float lowerBound = -5.0f;
    uint32_t layout = RKDA_LAYOUT_BSND;
    uint32_t hasCuSeqlens = 0;
    uint32_t hasSsmStateIndices = 0;
    uint32_t hasALog = 0;
    uint32_t hasDtBias = 0;
    uint32_t hasAcceptedTokens = 0;
    uint32_t useQkL2norm = 0;
    uint32_t useGateInKernel = 0;
    uint32_t useBetaSigmoid = 0;
    uint32_t allowNegEigval = 0;
    uint32_t safeGate = 0;
    uint32_t stateVFirst = 0;
    uint32_t outputFinalState = 0;
    uint32_t inplaceFinalState = 1;
    ge::DataType stateDtype = ge::DT_BF16;
    ge::DataType gateDtype = ge::DT_FLOAT;
    ge::DataType betaDtype = ge::DT_FLOAT;
    ge::DataType cuSeqlensDtype = ge::DT_INT64;
    ge::DataType ssmStateIndicesDtype = ge::DT_INT64;
    ge::DataType acceptedTokensDtype = ge::DT_INT64;
    std::array<int64_t, RKDA_STATE_DIM_NUM> stateInStrides = {};
    std::array<int64_t, RKDA_STATE_DIM_NUM> stateOutStrides = {};
    bool hasStateInStrides = false;
    bool hasStateOutStrides = false;
    uint64_t aivNum = 0;
    uint64_t ubSize = 0;
};

class RecurrentKdaTilingProcessor {
public:
    explicit RecurrentKdaTilingProcessor(const RecurrentKdaTilingContext &ctx) : ctx_(ctx) {}

    ge::graphStatus ProcessShapes(RecurrentKdaTilingData &tiling) const
    {
        tiling.vectorCoreNum = static_cast<uint32_t>(ctx_.aivNum);

        struct RuleItem {
            const char *name;
            ge::graphStatus (RecurrentKdaTilingProcessor::*fn)(RecurrentKdaTilingData &) const;
        };
        const std::array<RuleItem, 4> shapeRules = {{
            {"RuleCheckShapeDimAndRelation", &RecurrentKdaTilingProcessor::RuleCheckShapeDimAndRelation},
            {"RuleFillTilingShapeData", &RecurrentKdaTilingProcessor::RuleFillTilingShapeData},
            {"RuleCheckShapeValueRangeAndRule", &RecurrentKdaTilingProcessor::RuleCheckShapeValueRangeAndRule},
            {"RuleUpdateDynamicBlockDimByTaskUnits", &RecurrentKdaTilingProcessor::RuleUpdateDynamicBlockDimByTaskUnits},
        }};
        for (const auto &rule : shapeRules) {
            OP_CHECK_IF((this->*(rule.fn))(tiling) != ge::GRAPH_SUCCESS,
                        OP_LOGE(ctx_.nodeName, "ProcessShapes rule failed: %s", rule.name),
                        return ge::GRAPH_FAILED);
        }
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus ProcessUb(RecurrentKdaTilingData &tiling) const
    {
        struct RuleItem {
            const char *name;
            ge::graphStatus (RecurrentKdaTilingProcessor::*fn)(RecurrentKdaTilingData &, UbCalcContext &) const;
        };
        UbCalcContext ubCalcCtx;
        const std::array<RuleItem, 5> ubRules = {{
            {"RuleInitUbCalcContext", &RecurrentKdaTilingProcessor::RuleInitUbCalcContext},
            {"RuleCalcFixedUbBytes", &RecurrentKdaTilingProcessor::RuleCalcFixedUbBytes},
            {"RuleCalcWorkingUbBytes", &RecurrentKdaTilingProcessor::RuleCalcWorkingUbBytes},
            {"RuleCalcVStepCoeff", &RecurrentKdaTilingProcessor::RuleCalcVStepCoeff},
            {"RuleFinalizeVStepFromUb", &RecurrentKdaTilingProcessor::RuleFinalizeVStepFromUb},
        }};
        for (const auto &rule : ubRules) {
            OP_CHECK_IF((this->*(rule.fn))(tiling, ubCalcCtx) != ge::GRAPH_SUCCESS,
                        OP_LOGE(ctx_.nodeName, "ProcessUb rule failed: %s", rule.name),
                        return ge::GRAPH_FAILED);
        }
        return ge::GRAPH_SUCCESS;
    }

private:
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

    RecurrentKdaTilingContext ctx_;

    std::array<int64_t, RKDA_STATE_DIM_NUM> ResolveStateStrides(bool output) const
    {
        const bool hasStrides = output ? ctx_.hasStateOutStrides : ctx_.hasStateInStrides;
        if (hasStrides) {
            return output ? ctx_.stateOutStrides : ctx_.stateInStrides;
        }
        const auto &shape = ctx_.stateShape;
        std::array<int64_t, RKDA_STATE_DIM_NUM> strides = {};
        strides[RKDA_DIM_3] = 1;
        strides[RKDA_DIM_2] = shape.GetDim(RKDA_DIM_3);
        strides[RKDA_DIM_1] = shape.GetDim(RKDA_DIM_2) * strides[RKDA_DIM_2];
        strides[RKDA_DIM_0] = shape.GetDim(RKDA_DIM_1) * strides[RKDA_DIM_1];
        return strides;
    }

    ge::graphStatus ValidateStateStrides(
        const std::array<int64_t, RKDA_STATE_DIM_NUM> &strides, const char *name) const
    {
        for (size_t i = 0; i < RKDA_STATE_DIM_NUM; ++i) {
            OP_CHECK_IF(strides[i] <= 0,
                        OP_LOGE(ctx_.nodeName, "%s stride[%zu] must be positive, but it is %ld.",
                                name, i, strides[i]),
                        return ge::GRAPH_FAILED);
        }
        const auto &shape = ctx_.stateShape;
        const int64_t denseRow = shape.GetDim(RKDA_DIM_3);
        const int64_t densePlane = shape.GetDim(RKDA_DIM_2) * denseRow;
        OP_CHECK_IF(strides[RKDA_DIM_3] != 1 || strides[RKDA_DIM_2] != denseRow,
                    OP_LOGE(ctx_.nodeName,
                            "%s must keep its inner state matrix dense: stride[3]=1 and stride[2]=%ld, "
                            "but got [%ld, %ld, %ld, %ld].",
                            name, denseRow, strides[RKDA_DIM_0], strides[RKDA_DIM_1],
                            strides[RKDA_DIM_2], strides[RKDA_DIM_3]),
                    return ge::GRAPH_FAILED);
        const int64_t minSlotStride =
            (shape.GetDim(RKDA_DIM_1) - 1) * strides[RKDA_DIM_1] + densePlane;
        OP_CHECK_IF(strides[RKDA_DIM_1] < densePlane ||
                        strides[RKDA_DIM_0] < minSlotStride,
                    OP_LOGE(ctx_.nodeName,
                            "%s outer strides overlap state rows or heads: [%ld, %ld, %ld, %ld].",
                            name, strides[RKDA_DIM_0], strides[RKDA_DIM_1],
                            strides[RKDA_DIM_2], strides[RKDA_DIM_3]),
                    return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus CheckStateStrides() const
    {
        if (ValidateStateStrides(ResolveStateStrides(false), "initial_state") != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        if (ValidateStateStrides(ResolveStateStrides(true), "state output") != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    }

    bool CheckDim(const gert::Shape &shape, const size_t dim, const std::string &dimDesc) const
    {
        if (shape.GetDimNum() != dim) {
            OP_LOGE(ctx_.nodeName, "The number of dimensions of %s should be %zu, but it is %zu",
                    dimDesc.c_str(), dim, shape.GetDimNum());
            return false;
        }
        return true;
    }

    bool CheckDimEqual(const gert::Shape &a, const int64_t dimA, const gert::Shape &b, const int64_t dimB,
                       const std::string &nameA, const std::string &nameB, const std::string &dimDesc) const
    {
        if (a.GetDim(dimA) != b.GetDim(dimB)) {
            OP_LOGE(ctx_.nodeName, "The %s of %s and %s should be the same, but %s is %ld while %s is %ld",
                    dimDesc.c_str(), nameA.c_str(), nameB.c_str(), nameA.c_str(), a.GetDim(dimA), nameB.c_str(),
                    b.GetDim(dimB));
            return false;
        }
        return true;
    }

    int64_t SeqNum(const gert::Shape &queryShape, const gert::Shape &cuSeqlensShape) const
    {
        if (ctx_.hasCuSeqlens) {
            return cuSeqlensShape.GetDim(RKDA_DIM_0) - 1;
        }
        return ctx_.layout == RKDA_LAYOUT_BSND ? queryShape.GetDim(RKDA_DIM_0) : 1;
    }

    ge::graphStatus CheckMetadataShapes(const gert::Shape &queryShape, const gert::Shape &cuSeqlensShape) const
    {
        if (ctx_.hasCuSeqlens) {
            if (!CheckDim(cuSeqlensShape, RKDA_METADATA_RANK1, "cu_seqlens")) {
                return ge::GRAPH_FAILED;
            }
            OP_CHECK_IF(cuSeqlensShape.GetDim(RKDA_DIM_0) < 2,
                        OP_LOGE(ctx_.nodeName, "cu_seqlens must contain at least 2 elements."),
                        return ge::GRAPH_FAILED);
        }
        int64_t totalTokens =
            (ctx_.layout == RKDA_LAYOUT_TND) ? queryShape.GetDim(RKDA_DIM_0) :
            queryShape.GetDim(RKDA_DIM_0) * queryShape.GetDim(RKDA_DIM_1);
        int64_t seqNum = SeqNum(queryShape, cuSeqlensShape);

        if (ctx_.hasSsmStateIndices) {
            size_t rank = ctx_.ssmStateShape.GetDimNum();
            bool packed1d = rank == RKDA_METADATA_RANK1 &&
                            ctx_.ssmStateShape.GetDim(RKDA_DIM_0) >= totalTokens;
            bool speculative2d = rank == RKDA_METADATA_RANK2 &&
                                 ctx_.ssmStateShape.GetDim(RKDA_DIM_0) == seqNum &&
                                 ctx_.ssmStateShape.GetDim(RKDA_DIM_1) > 0;
            OP_CHECK_IF(!packed1d && !speculative2d,
                        OP_LOGE(ctx_.nodeName,
                                "ssm_state_indices must be packed [T] or speculative [seq_num,max_step]."),
                        return ge::GRAPH_FAILED);
        }
        if (ctx_.hasAcceptedTokens) {
            if (!CheckDim(ctx_.acceptedTokensShape, RKDA_METADATA_RANK1, "num_accepted_tokens")) {
                return ge::GRAPH_FAILED;
            }
            OP_CHECK_IF(ctx_.acceptedTokensShape.GetDim(RKDA_DIM_0) != seqNum,
                        OP_LOGE(ctx_.nodeName, "num_accepted_tokens length must equal sequence number."),
                        return ge::GRAPH_FAILED);
        }
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus CheckOptionalGateShapes(int64_t hvNum, int64_t kDim) const
    {
        if (ctx_.useGateInKernel && !ctx_.hasALog) {
            OP_LOGE(ctx_.nodeName, "A_log is required when use_gate_in_kernel is true.");
            return ge::GRAPH_FAILED;
        }
        if (ctx_.hasALog) {
            if (!CheckDim(ctx_.aLogShape, RKDA_METADATA_RANK1, "A_log")) {
                return ge::GRAPH_FAILED;
            }
            OP_CHECK_IF(ctx_.aLogShape.GetDim(RKDA_DIM_0) != hvNum,
                        OP_LOGE(ctx_.nodeName, "A_log shape must be [HV]."),
                        return ge::GRAPH_FAILED);
        }
        if (ctx_.hasDtBias) {
            size_t rank = ctx_.dtBiasShape.GetDimNum();
            bool valid = (rank == 1 && ctx_.dtBiasShape.GetDim(RKDA_DIM_0) == hvNum * kDim) ||
                         (rank == 2 && ctx_.dtBiasShape.GetDim(RKDA_DIM_0) == hvNum &&
                          ctx_.dtBiasShape.GetDim(RKDA_DIM_1) == kDim);
            OP_CHECK_IF(!valid, OP_LOGE(ctx_.nodeName, "dt_bias must be [HV*K] or [HV, K]."),
                        return ge::GRAPH_FAILED);
        }
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus CheckShapeDimAndRelation(const gert::Shape &queryShape, const gert::Shape &keyShape,
                                             const gert::Shape &valueShape, const gert::Shape &gateShape,
                                             const gert::Shape &betaShape, const gert::Shape &stateShape,
                                             const gert::Shape &cuSeqlensShape) const
    {
        int64_t totalTokens = 0;
        int64_t denseSeqLen = 0;
        int64_t hNum = 0;
        int64_t hvNum = 0;
        int64_t kDim = 0;
        int64_t vDim = 0;
        if (ctx_.layout == RKDA_LAYOUT_TND) {
            if (!CheckDim(queryShape, RKDA_RANK3_QKV_DIM_NUM, "query") ||
                !CheckDim(keyShape, RKDA_RANK3_QKV_DIM_NUM, "key") ||
                !CheckDim(valueShape, RKDA_RANK3_QKV_DIM_NUM, "value") ||
                !CheckDim(gateShape, RKDA_RANK3_QKV_DIM_NUM, "gate") ||
                !CheckDim(betaShape, RKDA_RANK2_BETA_DIM_NUM, "beta")) {
                return ge::GRAPH_FAILED;
            }
            totalTokens = queryShape.GetDim(RKDA_DIM_0);
            denseSeqLen = totalTokens;
            hNum = queryShape.GetDim(RKDA_DIM_1);
            kDim = queryShape.GetDim(RKDA_DIM_2);
            hvNum = valueShape.GetDim(RKDA_DIM_1);
            vDim = valueShape.GetDim(RKDA_DIM_2);
            OP_CHECK_IF(valueShape.GetDim(RKDA_DIM_0) != totalTokens ||
                            gateShape.GetDim(RKDA_DIM_0) != totalTokens ||
                            betaShape.GetDim(RKDA_DIM_0) != totalTokens ||
                            gateShape.GetDim(RKDA_DIM_1) != hvNum ||
                            betaShape.GetDim(RKDA_DIM_1) != hvNum ||
                            gateShape.GetDim(RKDA_DIM_2) != kDim,
                        OP_LOGE(ctx_.nodeName,
                                "TND expects q/k [T,H,K], value [T,HV,V], gate [T,HV,K], beta [T,HV]."),
                        return ge::GRAPH_FAILED);
        } else {
            if (!CheckDim(queryShape, RKDA_RANK4_QKV_DIM_NUM, "query") ||
                !CheckDim(keyShape, RKDA_RANK4_QKV_DIM_NUM, "key") ||
                !CheckDim(valueShape, RKDA_RANK4_QKV_DIM_NUM, "value") ||
                !CheckDim(gateShape, RKDA_RANK4_QKV_DIM_NUM, "gate") ||
                !CheckDim(betaShape, RKDA_RANK3_BETA_DIM_NUM, "beta")) {
                return ge::GRAPH_FAILED;
            }
            int64_t batch = queryShape.GetDim(RKDA_DIM_0);
            denseSeqLen = queryShape.GetDim(RKDA_DIM_1);
            totalTokens = batch * denseSeqLen;
            hNum = queryShape.GetDim(RKDA_DIM_2);
            kDim = queryShape.GetDim(RKDA_DIM_3);
            hvNum = valueShape.GetDim(RKDA_DIM_2);
            vDim = valueShape.GetDim(RKDA_DIM_3);
            OP_CHECK_IF(valueShape.GetDim(RKDA_DIM_0) != batch ||
                            valueShape.GetDim(RKDA_DIM_1) != denseSeqLen ||
                            gateShape.GetDim(RKDA_DIM_0) != batch ||
                            gateShape.GetDim(RKDA_DIM_1) != denseSeqLen ||
                            gateShape.GetDim(RKDA_DIM_2) != hvNum ||
                            gateShape.GetDim(RKDA_DIM_3) != kDim ||
                            betaShape.GetDim(RKDA_DIM_0) != batch ||
                            betaShape.GetDim(RKDA_DIM_1) != denseSeqLen ||
                            betaShape.GetDim(RKDA_DIM_2) != hvNum,
                        OP_LOGE(ctx_.nodeName,
                                "BSND expects q/k [B,T,H,K], value [B,T,HV,V], gate [B,T,HV,K], beta [B,T,HV]."),
                        return ge::GRAPH_FAILED);
        }

        if (!CheckDimEqual(queryShape, RKDA_DIM_0, keyShape, RKDA_DIM_0, "query", "key",
                           "leading token dimension") ||
            !CheckDimEqual(queryShape, queryShape.GetDimNum() - 2, keyShape, keyShape.GetDimNum() - 2,
                           "query", "key", "H dimension") ||
            !CheckDimEqual(queryShape, queryShape.GetDimNum() - 1, keyShape, keyShape.GetDimNum() - 1,
                           "query", "key", "K dimension")) {
            return ge::GRAPH_FAILED;
        }

        OP_CHECK_IF(hNum <= 0 || hvNum <= 0 || kDim <= 0 || vDim <= 0 || totalTokens <= 0 || denseSeqLen <= 0,
                    OP_LOGE(ctx_.nodeName, "input shape dimensions must be positive."), return ge::GRAPH_FAILED);
        OP_CHECK_IF(hvNum % hNum != 0,
                    OP_LOGE(ctx_.nodeName, "HV must be an integer multiple of H, but HV is %ld and H is %ld.",
                            hvNum, hNum),
                    return ge::GRAPH_FAILED);
        if (!CheckDim(stateShape, RKDA_STATE_DIM_NUM, "initial_state")) {
            return ge::GRAPH_FAILED;
        }
        int64_t seqNum = SeqNum(queryShape, cuSeqlensShape);
        bool stateTailMatches = ctx_.stateVFirst ?
            (stateShape.GetDim(RKDA_DIM_2) == vDim && stateShape.GetDim(RKDA_DIM_3) == kDim) :
            (stateShape.GetDim(RKDA_DIM_2) == kDim && stateShape.GetDim(RKDA_DIM_3) == vDim);
        OP_CHECK_IF(stateShape.GetDim(RKDA_DIM_0) <= 0 ||
                        (!ctx_.hasSsmStateIndices && stateShape.GetDim(RKDA_DIM_0) != seqNum) ||
                        stateShape.GetDim(RKDA_DIM_1) != hvNum || !stateTailMatches,
                    OP_LOGE(ctx_.nodeName,
                            "state must be [state_capacity, HV, V, K] when state_v_first=true or "
                            "[state_capacity, HV, K, V] otherwise; without ssm_state_indices, "
                            "state_capacity must equal seq_num."),
                    return ge::GRAPH_FAILED);

        if (CheckStateStrides() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }

        if (CheckMetadataShapes(queryShape, cuSeqlensShape) != ge::GRAPH_SUCCESS ||
            CheckOptionalGateShapes(hvNum, kDim) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    }

    void FillTilingShapeData(const gert::Shape &queryShape, const gert::Shape &valueShape,
                             const gert::Shape &stateShape, const gert::Shape &cuSeqlensShape,
                             RecurrentKdaTilingData &tiling) const
    {
        if (ctx_.layout == RKDA_LAYOUT_TND) {
            tiling.t = static_cast<uint32_t>(queryShape.GetDim(RKDA_DIM_0));
            tiling.seqLen = static_cast<uint32_t>(queryShape.GetDim(RKDA_DIM_0));
            tiling.nk = static_cast<uint32_t>(queryShape.GetDim(RKDA_DIM_1));
            tiling.dk = static_cast<uint32_t>(queryShape.GetDim(RKDA_DIM_2));
            tiling.nv = static_cast<uint32_t>(valueShape.GetDim(RKDA_DIM_1));
            tiling.dv = static_cast<uint32_t>(valueShape.GetDim(RKDA_DIM_2));
            tiling.b = static_cast<uint32_t>(SeqNum(queryShape, cuSeqlensShape));
        } else {
            tiling.seqLen = static_cast<uint32_t>(queryShape.GetDim(RKDA_DIM_1));
            tiling.t = static_cast<uint32_t>(queryShape.GetDim(RKDA_DIM_0) * queryShape.GetDim(RKDA_DIM_1));
            tiling.nk = static_cast<uint32_t>(queryShape.GetDim(RKDA_DIM_2));
            tiling.dk = static_cast<uint32_t>(queryShape.GetDim(RKDA_DIM_3));
            tiling.nv = static_cast<uint32_t>(valueShape.GetDim(RKDA_DIM_2));
            tiling.dv = static_cast<uint32_t>(valueShape.GetDim(RKDA_DIM_3));
            tiling.b = static_cast<uint32_t>(SeqNum(queryShape, cuSeqlensShape));
        }
        tiling.sBlockNum = static_cast<uint32_t>(stateShape.GetDim(RKDA_DIM_0));
        tiling.ssmStateStride = (ctx_.hasSsmStateIndices && ctx_.ssmStateShape.GetDimNum() == RKDA_METADATA_RANK2) ?
                                static_cast<uint32_t>(ctx_.ssmStateShape.GetDim(RKDA_DIM_1)) : 0;
        const auto stateInStrides = ResolveStateStrides(false);
        const auto stateOutStrides = ResolveStateStrides(true);
        tiling.stateInStride0 = static_cast<uint64_t>(stateInStrides[RKDA_DIM_0]);
        tiling.stateInStride1 = static_cast<uint64_t>(stateInStrides[RKDA_DIM_1]);
        tiling.stateInStride2 = static_cast<uint64_t>(stateInStrides[RKDA_DIM_2]);
        tiling.stateInStride3 = static_cast<uint64_t>(stateInStrides[RKDA_DIM_3]);
        tiling.stateOutStride0 = static_cast<uint64_t>(stateOutStrides[RKDA_DIM_0]);
        tiling.stateOutStride1 = static_cast<uint64_t>(stateOutStrides[RKDA_DIM_1]);
        tiling.stateOutStride2 = static_cast<uint64_t>(stateOutStrides[RKDA_DIM_2]);
        tiling.stateOutStride3 = static_cast<uint64_t>(stateOutStrides[RKDA_DIM_3]);
        tiling.scale = ctx_.scale;
        tiling.lowerBound = ctx_.lowerBound;
        tiling.layout = ctx_.layout;
        tiling.hasCuSeqlens = ctx_.hasCuSeqlens;
        tiling.hasSsmStateIndices = ctx_.hasSsmStateIndices;
        tiling.hasALog = ctx_.hasALog;
        tiling.hasDtBias = ctx_.hasDtBias;
        tiling.hasAcceptedTokens = ctx_.hasAcceptedTokens;
        tiling.useQkL2norm = ctx_.useQkL2norm;
        tiling.useGateInKernel = ctx_.useGateInKernel;
        tiling.useBetaSigmoid = ctx_.useBetaSigmoid;
        tiling.allowNegEigval = ctx_.allowNegEigval;
        tiling.safeGate = ctx_.safeGate;
        tiling.stateVFirst = ctx_.stateVFirst;
        tiling.outputFinalState = ctx_.outputFinalState;
        tiling.inplaceFinalState = ctx_.inplaceFinalState;
        tiling.gateDtype = ctx_.gateDtype == ge::DT_FLOAT ? 0 : (ctx_.gateDtype == ge::DT_BF16 ? 1 : 2);
        tiling.betaDtype = ctx_.betaDtype == ge::DT_FLOAT ? 0 : (ctx_.betaDtype == ge::DT_BF16 ? 1 : 2);
        tiling.cuSeqlensDtype = ctx_.cuSeqlensDtype == ge::DT_INT32 ? 0 : 1;
        tiling.ssmStateIndicesDtype = ctx_.ssmStateIndicesDtype == ge::DT_INT32 ? 0 : 1;
        tiling.acceptedTokensDtype = ctx_.acceptedTokensDtype == ge::DT_INT32 ? 0 : 1;
    }

    ge::graphStatus CheckShapeValueRangeAndRule(const RecurrentKdaTilingData &tiling) const
    {
        OP_CHECK_IF(tiling.nk > 256 || tiling.nv > 256,
                    OP_LOGE(ctx_.nodeName,
                            "H/HV must be <= 256, but H=%u, HV=%u.", tiling.nk, tiling.nv),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(tiling.dk != 128 || (tiling.dv != 128 && tiling.dv != 256),
                    OP_LOGE(ctx_.nodeName,
                            "K/V currently support only K=128,V=128 or K=128,V=256, but K=%u, V=%u.",
                            tiling.dk, tiling.dv),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(tiling.nv % tiling.nk != 0,
                    OP_LOGE(ctx_.nodeName, "HV must be divisible by H, but HV=%u and H=%u.",
                            tiling.nv, tiling.nk),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(tiling.safeGate && (tiling.lowerBound < -5.0f || tiling.lowerBound >= 0.0f),
                    OP_LOGE(ctx_.nodeName, "lower_bound must be in [-5, 0) when safe_gate is true."),
                    return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    void UpdateDynamicBlockDimByTaskUnits(RecurrentKdaTilingData &tiling) const
    {
        uint64_t taskUnits = static_cast<uint64_t>(tiling.b) * static_cast<uint64_t>(tiling.nv);
        if (taskUnits == 0) {
            taskUnits = 1;
        }
        uint64_t maxCoreNum = (ctx_.aivNum > 0) ? ctx_.aivNum : 1;
        uint64_t selectedCoreNum = (taskUnits < maxCoreNum) ? taskUnits : maxCoreNum;
        tiling.vectorCoreNum = static_cast<uint32_t>(selectedCoreNum);
        OP_LOGD(ctx_.nodeName, "taskUnits: [%llu], selected vectorCoreNum: [%u]",
                static_cast<unsigned long long>(taskUnits), tiling.vectorCoreNum);
    }

    ge::graphStatus RuleCheckShapeDimAndRelation(RecurrentKdaTilingData &tiling) const
    {
        (void)tiling;
        return CheckShapeDimAndRelation(ctx_.queryShape, ctx_.keyShape, ctx_.valueShape, ctx_.gateShape,
                                        ctx_.betaShape, ctx_.stateShape, ctx_.cuSeqlensShape);
    }

    ge::graphStatus RuleFillTilingShapeData(RecurrentKdaTilingData &tiling) const
    {
        FillTilingShapeData(ctx_.queryShape, ctx_.valueShape, ctx_.stateShape, ctx_.cuSeqlensShape, tiling);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus RuleCheckShapeValueRangeAndRule(RecurrentKdaTilingData &tiling) const
    {
        return CheckShapeValueRangeAndRule(tiling);
    }

    ge::graphStatus RuleUpdateDynamicBlockDimByTaskUnits(RecurrentKdaTilingData &tiling) const
    {
        UpdateDynamicBlockDimByTaskUnits(tiling);
        return ge::GRAPH_SUCCESS;
    }

    int64_t CalcFixedUbBytes(int64_t aNv, int64_t aDv, int64_t aDk) const
    {
        int64_t usedUbBytes = 0;
        usedUbBytes += RKDA_MAX_MTP * (4 * aDk + 2 * aDv);   // q/k/v input queues, bf16.
        usedUbBytes += RKDA_MAX_MTP * 4 * aDk;               // gate input queue, fp32.
        usedUbBytes += RKDA_MAX_MTP * 4 * aNv;               // beta input queue, fp32.
        usedUbBytes += 64 + RKDA_UB_GUARD_BYTES;             // scalar buffer and guard.
        return usedUbBytes;
    }

    int64_t CalcWorkingUbBytes(int64_t aNv, int64_t aDv, int64_t aDk) const
    {
        int64_t usedUbBytes = CalcFixedUbBytes(aNv, aDv, aDk);
        usedUbBytes += RKDA_MAX_MTP * (4 * aDv + 12 * aDk + 4 * aNv);
        return usedUbBytes;
    }

    int64_t CalcVStepCoeff(int64_t aDk, uint32_t stateOutBufferNum, uint32_t attnOutBufferNum) const
    {
        int64_t stateDtypeSize = (ctx_.stateDtype == ge::DT_FLOAT) ? 4 : 2;
        int64_t coeff = stateDtypeSize * aDk; // state input queue.
        coeff += static_cast<int64_t>(stateOutBufferNum) * stateDtypeSize * aDk;
        coeff += static_cast<int64_t>(attnOutBufferNum) * 2;
        coeff += 8 * aDk + 8;
        return coeff;
    }

    bool EvaluateBufferProfile(int64_t ubSize, int64_t usedUbBytes, int64_t aDk, uint32_t stateOutBufferNum,
                               uint32_t attnOutBufferNum, const RecurrentKdaTilingData &tiling,
                               BufferProfile &profile) const
    {
        int64_t coeff = CalcVStepCoeff(aDk, stateOutBufferNum, attnOutBufferNum);
        int64_t vStep = (ubSize - usedUbBytes) / coeff / 8 * 8;
        if (vStep < static_cast<int64_t>(RKDA_MAX_MTP)) {
            return false;
        }
        int64_t repeatTime = Ops::Base::CeilDiv(tiling.dv, static_cast<uint32_t>(vStep));
        vStep = Ops::Base::CeilAlign(Ops::Base::CeilDiv(tiling.dv, static_cast<uint32_t>(repeatTime)),
                                     static_cast<uint32_t>(8));
        if (vStep < static_cast<int64_t>(RKDA_MAX_MTP)) {
            return false;
        }
        profile.stateOutBufferNum = stateOutBufferNum;
        profile.attnOutBufferNum = attnOutBufferNum;
        profile.vStep = static_cast<uint32_t>(vStep);
        profile.repeatTime = static_cast<uint32_t>(repeatTime);
        profile.valid = true;
        return true;
    }

    bool IsBetterProfile(const BufferProfile &candidate, const BufferProfile &current) const
    {
        if (!current.valid) {
            return true;
        }
        if (candidate.repeatTime != current.repeatTime) {
            return candidate.repeatTime < current.repeatTime;
        }
        uint32_t candidateDepth = candidate.stateOutBufferNum + candidate.attnOutBufferNum;
        uint32_t currentDepth = current.stateOutBufferNum + current.attnOutBufferNum;
        if (candidateDepth != currentDepth) {
            return candidateDepth > currentDepth;
        }
        return candidate.vStep > current.vStep;
    }

    ge::graphStatus FinalizeVStepFromUb(int64_t ubSize, int64_t usedUbBytes, int64_t coeff,
                                        RecurrentKdaTilingData &tiling, UbCalcContext &ubCalcCtx) const
    {
        (void)coeff;
        int64_t aDk = Ops::Base::CeilAlign(tiling.dk, static_cast<uint32_t>(16));
        BufferProfile selected;
        const std::array<BufferProfile, 3> candidates = {{
            {1, 1, 0, 0, false},
            {1, 2, 0, 0, false},
            {2, 2, 0, 0, false},
        }};
        for (const auto &candidate : candidates) {
            BufferProfile profile;
            if (!EvaluateBufferProfile(ubSize, usedUbBytes, aDk, candidate.stateOutBufferNum,
                                       candidate.attnOutBufferNum, tiling, profile)) {
                continue;
            }
            if (IsBetterProfile(profile, selected)) {
                selected = profile;
            }
        }

        if (!selected.valid) {
            OP_LOGE(ctx_.nodeName, "vStep should be at least %zu, shape is too big", RKDA_MAX_MTP);
            return ge::GRAPH_FAILED;
        }

        int64_t queueCoeff = CalcVStepCoeff(aDk, selected.stateOutBufferNum, selected.attnOutBufferNum) -
                             (8 * aDk + 8);
        int64_t ubRestBytes = ubSize - ubCalcCtx.fixedUbBytes -
                              queueCoeff * static_cast<int64_t>(selected.vStep);
        if (ubRestBytes < 0) {
            OP_LOGE(ctx_.nodeName, "ubRestBytes should be non-negative, but got %ld", ubRestBytes);
            return ge::GRAPH_FAILED;
        }
        tiling.ubCalSize = static_cast<uint32_t>(ctx_.ubSize);
        tiling.vStep = selected.vStep;
        tiling.stateOutBufferNum = selected.stateOutBufferNum;
        tiling.attnOutBufferNum = selected.attnOutBufferNum;
        tiling.ubRestBytes = static_cast<uint32_t>(ubRestBytes);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus RuleInitUbCalcContext(RecurrentKdaTilingData &tiling, UbCalcContext &ubCalcCtx) const
    {
        ubCalcCtx.ubSize = static_cast<int64_t>(ctx_.ubSize);
        ubCalcCtx.aNv = Ops::Base::CeilAlign(tiling.nv, static_cast<uint32_t>(16));
        ubCalcCtx.aDv = Ops::Base::CeilAlign(tiling.dv, static_cast<uint32_t>(16));
        ubCalcCtx.aDk = Ops::Base::CeilAlign(tiling.dk, static_cast<uint32_t>(16));
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus RuleCalcFixedUbBytes(RecurrentKdaTilingData &tiling, UbCalcContext &ubCalcCtx) const
    {
        ubCalcCtx.fixedUbBytes = CalcFixedUbBytes(ubCalcCtx.aNv, ubCalcCtx.aDv, ubCalcCtx.aDk);
        tiling.ubRestBytes = static_cast<uint32_t>(ubCalcCtx.ubSize - ubCalcCtx.fixedUbBytes);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus RuleCalcWorkingUbBytes(RecurrentKdaTilingData &tiling, UbCalcContext &ubCalcCtx) const
    {
        (void)tiling;
        ubCalcCtx.workingUbBytes = CalcWorkingUbBytes(ubCalcCtx.aNv, ubCalcCtx.aDv, ubCalcCtx.aDk);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus RuleCalcVStepCoeff(RecurrentKdaTilingData &tiling, UbCalcContext &ubCalcCtx) const
    {
        (void)tiling;
        ubCalcCtx.coeff = CalcVStepCoeff(ubCalcCtx.aDk, 1, 1);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus RuleFinalizeVStepFromUb(RecurrentKdaTilingData &tiling, UbCalcContext &ubCalcCtx) const
    {
        return FinalizeVStepFromUb(ubCalcCtx.ubSize, ubCalcCtx.workingUbBytes, ubCalcCtx.coeff, tiling, ubCalcCtx);
    }
};

} // namespace optiling

#endif // RECURRENT_KDA_TILING_PROCESSOR_H
