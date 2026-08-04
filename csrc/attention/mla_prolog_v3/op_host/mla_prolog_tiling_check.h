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
 * \file mla_prolog_tiling_check.h
 * \brief
 */

#ifndef MLA_PROLOG_TILING_CHECK_H
#define MLA_PROLOG_TILING_CHECK_H

#include "mla_prolog_tiling.h"

namespace optiling {

constexpr uint32_t MAX_B_SIZE = 65536U;
constexpr uint32_t MAX_S1_SIZE = 65536U;
constexpr uint32_t MAX_T_SIZE = 1024U * 1024U;
constexpr uint32_t HCQ_SIZE = 1536U;
constexpr uint32_t HCKV_SIZE = 512U;
constexpr uint32_t D_SIZE = 128U;
constexpr uint32_t DR_SIZE = 64U;
constexpr uint32_t NKV_SIZE = 1U;
constexpr uint32_t MIN_BLOCK_SIZE = 16U;
constexpr uint32_t MAX_BLOCK_SIZE = 1024U;
constexpr uint32_t ALIGN_BLOCK_SIZE = 16U;
constexpr uint32_t MXFP8_BLOCK_SIZE = 32U;

constexpr int64_t  NZ_H0_SIZE = 16U;

constexpr char TOKEN_X_NAME[] {"tokenX"};
constexpr char WEIGHT_DQ_NAME[] {"weightDq"};
constexpr char WEIGHT_UQ_QR_NAME[] {"weightUqQr"};
constexpr char WEIGHT_UK_NAME[] {"weightUk"};
constexpr char WEIGHT_DKV_KR_NAME[] {"weightDkvKr"};
constexpr char RMSNORM_GAMMA_CQ_NAME[] {"rmsnormGammaCq"};
constexpr char RMSNORM_GAMMA_CKV_NAME[] {"rmsnormGammaCkv"};
constexpr char ROPE_SIN_NAME[] {"ropeSin"};
constexpr char ROPE_COS_NAME[] {"ropeCos"};
constexpr char CACHE_INDEX_NAME[] {"cacheIndex"};
constexpr char KV_CACHE_NAME[] {"kvCache"};
constexpr char KR_CACHE_NAME[] {"krCache"};
constexpr char DEQUANT_SCALE_X_NAME[] {"dequantScaleX"};
constexpr char DEQUANT_SCALE_W_DQ_NAME[] {"dequantScaleWDq"};
constexpr char DEQUANT_SCALE_W_UQ_QR_NAME[] {"dequantScaleWUqQr"};
constexpr char DEQUANT_SCALE_W_DKV_KR_NAME[] {"dequantScaleWDkvKr"};
constexpr char QUANT_SCALE_CKV_NAME[] {"quantScaleCkv"};
constexpr char QUANT_SCALE_CKR_NAME[] {"quantScaleCkr"};
constexpr char SMOOTH_SCALES_CQ_NAME[] {"smoothScalesCq"};
constexpr char ACTUAL_SEQ_LEN_NAME[] {"actualSeqLen"};
constexpr char K_NOPE_CLIP_ALPHA_NAME[] {"kNopeClipAlpha"};
constexpr char QUERY_NAME[] {"query"};
constexpr char QUERY_ROPE_NAME[] {"queryRope"};
constexpr char KV_CACHE_OUT_NAME[] {"kvCacheOut"};
constexpr char KR_CACHE_OUT_NAME[] {"krCacheOut"};
constexpr char DEQUANT_SCALE_Q_NOPE_NAME[] {"dequantScaleQNope"};
constexpr char QUERY_NORM_NAME[] {"queryNorm"};
constexpr char DEQUANT_SCALE_Q_NORM_NAME[] {"dequantScaleQNorm"};

constexpr uint32_t PARAM_MAP_INIT_RESERVE_NUM = 28;  // 预分配所有key的个数，避免使用时动态扩容

struct ParamInfo {
    ParamInfo() = default;
    explicit ParamInfo(const ParamInfo &) = default;
    ParamInfo &operator=(const ParamInfo &) = default;
    explicit ParamInfo(ParamInfo &&) = default;
    ParamInfo &operator=(ParamInfo &&other) = default;
    ~ParamInfo() = default;
    ParamInfo(const gert::CompileTimeTensorDesc *actualDesc, const gert::StorageShape *actualShape) {
        if (actualDesc != nullptr && actualShape != nullptr) {
            isValid = true;
            dtype = actualDesc->GetDataType();
            format = static_cast<ge::Format>(ge::GetPrimaryFormat(actualDesc->GetStorageFormat()));
            auto &&actualStorageShape = actualShape->GetStorageShape();
            dimNum = actualStorageShape.GetDimNum();
            this->shape.reserve(dimNum);
            for (size_t i = 0; i < dimNum; i++) {
                this->shape.emplace_back(actualStorageShape.GetDim(i));
            }
        }
    }
    explicit ParamInfo(const BaseParaInfo &info) : ParamInfo(info.desc, info.shape) {}
    explicit ParamInfo(const std::vector<uint32_t> &expectedShape)
    {
        isValid = true;
        format = ge::FORMAT_ND;
        dimNum = expectedShape.size();
        this->shape.reserve(dimNum);
        for (size_t i = 0; i < dimNum; i++) {
            this->shape.emplace_back(static_cast<int64_t>(expectedShape[i]));
        }
    }

    bool operator == (const ParamInfo &other) const {
        if (!isValid && !other.isValid) {
            return true;
        }
        static const std::set<ge::Format> ndFormats{ge::FORMAT_ND, ge::FORMAT_NCHW};
        if ((ndFormats.find(format) == ndFormats.end() || ndFormats.find(other.format) == ndFormats.end()) &&
            format != other.format) {
            return false;
        }
        return (isValid == other.isValid && dtype == other.dtype &&
            dimNum == other.dimNum && shape == other.shape);
    }
    bool operator != (const ParamInfo &other) const {
        return !(*this == other);
    }

    bool isValid {};
    ge::DataType dtype {ge::DT_MAX};
    ge::Format format {ge::FORMAT_MAX};
    size_t dimNum {};
    std::vector<int64_t> shape;
};

using ParamInfoMap = std::unordered_map<std::string, ParamInfo>;

class MlaPrologTilingCheck {
public:
    MlaPrologTilingCheck(const MlaPrologContext &context, const MlaPrologBaseShapeInfo &baseShapeInfo,
                         const MlaPrologScenarioInfo &scenarioInfo)
        : context_(context), baseShapeInfo_(baseShapeInfo), scenarioInfo_(scenarioInfo) {}
    ge::graphStatus CheckSingleRequiredParam() const;
    ge::graphStatus CheckCacheMode() const;
    ge::graphStatus CheckQuantMode() const;
    ge::graphStatus CheckDims() const;
    ge::graphStatus CheckParamByScenario();
    ge::graphStatus CheckSpecialScenarioParamShape();
    ge::graphStatus CheckCkvkrRepoMode();
    ge::graphStatus CheckCacheIndexDim();
    ge::graphStatus CheckScenarParam();
    ge::graphStatus CheckAttrs() const;

    NpuArch GetCurNpuArch() const;

private:
    bool CheckAttrsNotNull() const;
    bool CheckAttrsRange() const;
    bool CheckCacheModeParamShape() const;
    ge::graphStatus CheckHcqSize() const;
    ge::graphStatus CheckDSize() const;
    ge::graphStatus CheckDtileSize() const;
    // ==================================单参数校验==================================
    bool IsSingleParamValid(const BaseParaInfo &param, const std::string &paramName,
                            const std::set<ge::DataType> &expectedDtype,
                            const std::set<ge::Format> &expectedFormat,
                            const std::set<size_t> &expectedDimNum) const;
    bool CheckTokenX() const;
    bool CheckWDq() const;
    bool CheckWDkvKr() const;
    bool CheckWUqQr() const;
    bool CheckWUk() const;
    bool CheckRmsnormGammaCkv() const;
    bool CheckRmsnormGammaCq() const;
    bool CheckRopeCos() const;
    bool CheckRopeSin() const;
    bool CheckCacheIndex() const;
    bool CheckKvCache() const;
    bool CheckKrCache() const;
    bool CheckActSeqLen() const;
    // ==================================单参数校验==================================

    // =================================全量参数校验=================================
    void GenExpectedParamInfo();
    void FillCommonParamInfo();
    void FillRequiredParamShapeWithDims();
    void FillOptionalOutputParamShapeWithDims();
    void FillOptionalOutputParamShapeWithDimsV2();
 	void FillOptionalOutputParamShapeWithDimsV3();
    void FillScenarioParamInfo();
    void FillQueryNormScaleShape();
    void FillQueryNormDtypes();
    void FillTokenAndQueryShapes();
    void FillWeightAndNormShapes();
    void FillCacheShapes();
    void CheckRepoMode(bool isPertile, ge::graphStatus &isCorrect);
    void CheckQueryQuantMode(bool isPertensor, ge::graphStatus &isCorrect);
    void FillNonQuantParamInfo();
    void FillPartialQuantParamInfo();
    void FillPartialKVQuantParamInfo();
    void FillPartialKVPertileQuantParamInfo();
    void FillFullQuantParamInfo();
    void FillFullKVQuantParamInfo();
    void FillFullKVPertileQuantParamInfo();
    void FillMxfp8FullQuantParamInfo();
    void FillMxfp8FullKVQuantParamInfo();
    void FillMxfp8FullKVPertileParamInfo();
    void FillFP8FullQuantParamInfo();
    void FillFP8FullKVQuantParamInfo();
    void FillHIF8FullQuantParamInfo();
    void FillHIF8FullKVQuantParamInfo();
    void FillFP8FullKVPertileQuantParamInfo();
    void FillHIF8FullKVPertileQuantParamInfo();

    void GenActualParamInfo();
    // =================================全量参数校验=================================

    const MlaPrologContext &context_;
    const MlaPrologBaseShapeInfo &baseShapeInfo_;
    const MlaPrologScenarioInfo &scenarioInfo_;
    ParamInfoMap expectedParamInfo_ = ParamInfoMap(PARAM_MAP_INIT_RESERVE_NUM);
    ParamInfoMap actualParamInfo_ = ParamInfoMap(PARAM_MAP_INIT_RESERVE_NUM);
};

}  // namespace optiling

#endif