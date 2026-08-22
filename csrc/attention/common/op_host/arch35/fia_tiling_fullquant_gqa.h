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
 * \file fia_tiling_fullquant_gqa.h
 * \brief
 */
#ifndef FIA_TILING_FULLQUANT_GQA_H
#define FIA_TILING_FULLQUANT_GQA_H

#include "register/tilingdata_base.h"
#include "exe_graph/runtime/tiling_context.h"
#include "../fia_tiling_base.h"
#include "../fia_tiling_info.h"
#include "tiling/tiling_api.h" // 这个头文件顺序必须在手写的tiling data前
#include "../../op_kernel/arch35/fia_tiling_data_fullquant.h"
#include "../split_core_v2.h"
#include "../../../fused_infer_attention_score/op_kernel/fused_infer_attention_score_template_tiling_key.h"
#include "../../../prompt_flash_attention/op_kernel/arch35/prompt_flash_attention_tiling_regbase.h"

namespace optiling {
constexpr int64_t SPARSE_MODE_INT_MAX = 2147483647;

struct FiaTilingKeyInfo {
    uint64_t inputLayout = 0;
    uint64_t config = 0;
    uint64_t pseMode = 0;
    uint64_t quantMode = 31;
    bool hasAttenMask = false;
    bool hasRope = false;
    uint64_t kvLayoutType = 0;
    bool isFd = false;
    bool emptyTensor = false;
    bool enableKvPrefix = false;
    bool enableS1OutSplit = false;
    bool isReconstructTemp = false;
};

struct FiaPlatFormInfo {
    uint64_t ubSize = 0;
    uint64_t l2Size = 0;
    uint64_t l1Size = 0;
    uint64_t l0cSize = 0;
    uint64_t l0bSize = 0;
    uint64_t l0aSize = 0;
    uint32_t coreNum = 0;
    uint32_t aicNum = 0;
    uint32_t aivNum = 0;
    uint32_t cvRatio = 0;
    uint64_t defaultSysWorkspaceSize = 0;
};

class FiaTilingFullQuantGqaArch35 : public FiaTilingBase {
public:
    explicit FiaTilingFullQuantGqaArch35(gert::TilingContext *context) : FiaTilingBase(context) {}
    ~FiaTilingFullQuantGqaArch35() override = default;

protected:
    void InitTilingInfo(TilingInfo *tilingInfo) override;
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;

private:
    ge::graphStatus SetPlatMemoryInfo();
    void SplitPolicy();
    void ComputeTilingData();
    void GenTilingKey();
    void CalcWorkspaceSize();
    void UpdateTilingKeyConfig();
    void UpdateTilingKeyLayout();
    void UpdateTilingKeyPseMode();
    void UpdateTilingKeyQuantMode();
    void UpdateTilingKeyHasRope();
    void UpdateTilingKeyInfo();
    void SetFATilingData();
    void AdjustSinnerAndSouter();
    void InitImplParam();
    bool IsActualSeqLengthsKVHasZero(const split_core_v2::BaseInfo &baseInfo);
    bool IsExistRowInvalid(const split_core_v2::BaseInfo &baseInfo);
    void GetSafeActToken(split_core_v2::SparseMode mode, int64_t actSeqLensQ, int64_t actSeqLensKv,
                         int64_t &safePreToken, int64_t &safeNextToken);
    void PrintAllTilingData();
    void CalcMaxWorkspaceSize();
    void CalcScheduleMode();
    void CreateSplitInput(split_core_v2::BaseInfo &baseInfo, split_core_v2::SplitParam &splitParam);
    void SetSplitOutput(const split_core_v2::FAMetaData &splitRes);
    void CalcNumBlocks(uint32_t coreNum);
    void FillTiling();
    ge::graphStatus SetTilingData(FusedInferAttentionScoreFullQuantTilingData &tilingData);

    FusedInferAttentionScoreFullQuantTilingData tilingData_;
    FiaTilingKeyInfo tilingKeyInfo_;
    FiaPlatFormInfo platformInfo_;
    uint32_t sOuterFactor_;
    uint32_t sInnerFactor_;
    bool flashDecodeFlag_ = false;
    bool actualSeqLenQFlag_ = false;
    bool actualSeqLenKVFlag_ = false;
    bool actualSharedPrefixLenFlag_ = false;
    std::vector<int64_t> actualSeqLengthsQ_ = {};
    std::vector<int64_t> actualSeqLengthsKV_ = {};
    bool needInit_ = false;
    bool enableS1OutSplit = false;
    uint64_t tilingKey_ = 0;
    uint64_t workspaceSize_ = 0;
    ScheduleMode scheduleMode_ = ScheduleMode::BATCH_MODE;
    int32_t numBlocks_ = 0;
    bool decodeS1GMerge_ = false;

    // Tiling Info
    FiaTilingInfo *fiaInfo_ = nullptr;
};

} // namespace optiling
#endif // FIA_TILING_FULLQUANT_GQA_H