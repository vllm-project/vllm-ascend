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
 * \file fused_infer_attention_score_tiling_impl.h
 * \brief
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_FUSEDINFERATTENTIONSCORE_IMPL_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_FUSEDINFERATTENTIONSCORE_IMPL_H_
#include "register/tilingdata_base.h"
#include "../../../common/op_host/fia_tiling_base.h"
#include "../../../common/op_host/fia_tiling_info.h"
#include "tiling/tiling_api.h"  //这个头文件顺序必须在手写的tiling data前
#include "../../../common/op_kernel/arch35/flash_attention_score_tiling_regbase.h"
#include "../../op_kernel/fused_infer_attention_score_template_tiling_key.h"
#include "../../../prompt_flash_attention/op_kernel/arch35/prompt_flash_attention_tiling_regbase.h"

namespace optiling {

struct FiaTilingKeyInfo {
    uint64_t inputLayout = 0;
    uint64_t config = 0;
    uint64_t pseMode = 0;
    uint64_t quantMode = 31;
    bool hasAttenMask = false;
    bool hasRope = false;
    uint64_t KvLayoutType = 0;
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
    uint64_t defaultSysWorkspaceSize = 0;
};

class FusedInferAttentionScoreTilingImpl : public FiaTilingBase {
public:
    explicit FusedInferAttentionScoreTilingImpl(gert::TilingContext *context) : FiaTilingBase(context) {}
    ~FusedInferAttentionScoreTilingImpl() override = default;
    void InitTilingInfo(TilingInfo *tilingInfo) override {}
    bool IsCapable() override {}
    ge::graphStatus DoOpTiling() override {}
    ge::graphStatus DoOpTiling(gert::TilingContext *context, const FiaTilingInfo &fiaInfo);

protected:
    ge::graphStatus SetPlatMemoryInfo(const gert::TilingContext *context, const FiaTilingInfo &fiaInfo);
    ge::graphStatus SetEmptyTensor(gert::TilingContext *context, const FiaTilingInfo &fiaInfo);
    ge::graphStatus SplitPolicy(gert::TilingContext *context, const FiaTilingInfo &fiaInfo);
    void ApplySinnerSouterOverrides(const FiaTilingInfo &fiaInfo);
    void ApplySplitStrategy(const FiaTilingInfo &fiaInfo);
    void ApplyRowInvalidCheck(const FiaTilingInfo &fiaInfo);
    bool CheckS1OutSplit(const FiaTilingInfo &fiaInfo);
    int64_t CalcRequiredL2Size(const FiaTilingInfo &fiaInfo);
    void SplitOutSeq(const FiaTilingInfo &fiaInfo);
    ge::graphStatus ComputeTilingData(const FiaTilingInfo &fiaInfo);
    ge::graphStatus SetMaskTilingData(const FiaTilingInfo &fiaInfo);
    uint8_t ComputeSparseType(const FiaTilingInfo &fiaInfo) const;
    void SetAlibiStartIdx(const FiaTilingInfo &fiaInfo);
    void SetLayoutType(const FiaTilingInfo &fiaInfo);
    void SetPaLayoutTilingData(const FiaTilingInfo &fiaInfo);
    void SetTransposeLayout(const FiaTilingInfo &fiaInfo);
    void SetAntiQuantInfo(const FiaTilingInfo &fiaInfo);
    void ComputeNeedInit(const FiaTilingInfo &fiaInfo);
    ge::graphStatus GenTilingKey(gert::TilingContext *context, const FiaTilingInfo &fiaInfo);
    ge::graphStatus SetBlockDim(gert::TilingContext *context, const FiaTilingInfo &fiaInfo);
    ge::graphStatus GetWorkspace(gert::TilingContext *context, const FiaTilingInfo &fiaInfo);
    ge::graphStatus SetTilingData(gert::TilingContext *context, const FiaTilingInfo &fiaInfo);
    void UpdateTilingKeyConfig(const FiaTilingInfo &fiaInfo);
    void UpdateTilingKeyLayout(const FiaTilingInfo &fiaInfo);
    void UpdateTilingKeyPseMode(const FiaTilingInfo &fiaInfo);
    void UpdateTilingKeyQuantMode(const FiaTilingInfo &fiaInfo);
    void UpdateTilingKeyHasRope(const FiaTilingInfo &fiaInfo);
    ge::graphStatus UpdateTilingKeyInfo(const FiaTilingInfo &fiaInfo);
    ge::graphStatus SetWorkspaceNormal(const FiaTilingInfo &fiaInfo, int64_t &curWorkspaceSize);
    ge::graphStatus SetWorkspaceAntiQuant(const FiaTilingInfo &fiaInfo, int64_t &workspaceSize_);
    ge::graphStatus SetWorkspacePTQuant(const FiaTilingInfo &fiaInfo, int64_t &curWorkspaceSize);

    bool EnableMTE2BmmPipe(const FiaTilingInfo &fiaInfo, matmul_tiling::MatmulApiTiling &bmm,
                           TCubeTiling &bmmTilingData);
    ge::graphStatus SetFATilingData(const FiaTilingInfo &fiaInfo);
    void SetFATilingDataInputParams(const FiaTilingInfo &fiaInfo);
    void SetFATilingDataInitOutput(const FiaTilingInfo &fiaInfo);

    ge::graphStatus AdjustSinnerAndSouter(gert::TilingContext *context, const FiaTilingInfo &fiaInfo);
    bool CheckPfaMergeLayout(const FiaTilingInfo &fiaInfo) const;
    bool CheckSmallHeadOptimization(const FiaTilingInfo &fiaInfo) const;
    void GetPreNextTokensLeftUp(const FiaTilingInfo &fiaInfo, int64_t actualSeqLength, int64_t actualSeqLengthKV,
                                int64_t &preTokensLeftUp, int64_t &nextTokensLeftUp) const;
    void FixParamWithRowInvalid(const FiaTilingInfo &fiaInfo, int64_t &actualSeqLength, int64_t actualSeqLengthKV,
                                int64_t &preTokensLeftUp, int64_t &nextTokensLeftUp);
    int64_t GetCutBlockNums(int64_t blockSeqLengthKV, int64_t blockSeqLength, int64_t sInner, int64_t sOuter,
                            int64_t token);
    int64_t GetCalcBlockNumsOneHead(const FiaTilingInfo &fiaInfo, int64_t actualSeqLength, int64_t actualSeqLengthKV,
                                    uint32_t sOuterSize, uint32_t sInnerSize);
    int64_t GetSInnerBlockNums(int64_t sInnerIndexStart, int64_t sInnerIndexEnd, int64_t innerBlockNums) const;
    void ComputeSplitNBSeq(const FiaTilingInfo &fiaInfo, const size_t maxCoreNums, uint32_t sOuterSize,
                           uint32_t sInnerSize, double coreWeightTarget, uint32_t &curCore);
    void SplitNBSeq(const FiaTilingInfo &fiaInfo);
    void InitImplParam(const FiaTilingInfo &fiaInfo);
    void InitTilingFlags(const FiaTilingInfo &fiaInfo, const gert::Tensor *actSeqLenQ, uint32_t qDims,
                         const gert::Tensor *actSeqLenKV, uint32_t kvDims,
                         const gert::Tensor *actSharedPrefixLen, uint32_t prefixDims);
    void ComputeLoopParams(const FiaTilingInfo &fiaInfo);
    void ParseActualSeqLengths(const FiaTilingInfo &fiaInfo);
    void SetIsIFA(const FiaTilingInfo &fiaInfo);
    void SetGSMerge(const FiaTilingInfo &fiaInfo);
    bool CheckPFAMergeSupport(const FiaTilingInfo &fiaInfo) const;
    bool CheckActualSeqLenAligned(const FiaTilingInfo &fiaInfo) const;
    bool CheckAntiQuantGSMerge(const FiaTilingInfo &fiaInfo, bool actualSeqLenUnequal) const;
    bool CheckEnableDN(const FiaTilingInfo &fiaInfo) const;
    bool CheckQKVActualSeqLengthsRight(const FiaTilingInfo &fiaInfo);
    bool CheckFlashDecode(const FiaTilingInfo &fiaInfo);
    ge::graphStatus SplitS2(const FiaTilingInfo &fiaInfo);
    void SetDequantBaseSize(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CalcInnerSize(const FiaTilingInfo &fiaInfo, uint32_t seqSize);
    void GetActualSeqLength(const FiaTilingInfo &fiaInfo, int64_t &actualSeqLengths, int64_t &actualSeqLengthsKV, uint32_t bIdx);
    int64_t SumOfArithmeticSeries(int64_t an, int64_t d) const;
    void ComputeDequantSplitNBSeq(const FiaTilingInfo &fiaInfo, std::vector<int64_t> sOuterLoopTimes,
                                  std::vector<int64_t> sInnerLoopTimes, int64_t sInnerLoopTimesPrefix,
                                  double coreWeightTarget, uint32_t &curCore, const size_t tilingElementArrayLen);
    int64_t GetAntiQuantCalcBlockNumsOneHead( const FiaTilingInfo &fiaInfo, int64_t outerBlockNums,
                                              int64_t innerBlockNums, int64_t sInnerLoopTimesPrefix,
                                              int64_t preTokensLeftUp, int64_t nextTokensLeftUp);
    void GetAntiQuantPreNextTokensLeftUp(const FiaTilingInfo &fiaInfo, int64_t actualSeqLength,
                                         int64_t actualSeqLengthKV, int64_t &preTokensLeftUp,
                                         int64_t &nextTokensLeftUp) const;
    void FixAntiQuantParamWithRowInvalid(const FiaTilingInfo &fiaInfo, int64_t &actualSeqLength, 
                                         int64_t actualSeqLengthKV, int64_t &preTokensLeftUp,
                                         int64_t &nextTokensLeftUp);
    void DequantCubeSplitBNSeq(const FiaTilingInfo &fiaInfo);
    int64_t GetActualInnerBlockNums(int64_t sInnerIndexStart, int64_t sInnerIndexEnd, int64_t innerBlockNums) const;
    void SplitDequant(const FiaTilingInfo &fiaInfo);
    ge::graphStatus SetDequantMMTilingData(const gert::TilingContext *context, const FiaTilingInfo &fiaInfo);
    bool CheckTransposeLayout(const FiaTilingInfo &fiaInfo) const;
    void PrintAllTilingData(const FiaTilingInfo &fiaInfo);
    void PrintInputParams(const FiaTilingInfo &fiaInfo);

    PromptFlashAttentionTilingDataV2 pfaTilingData_;
    IncreFlashAttentionTilingDataRegbase ifaTilingData_;
    FlashAttentionScoreSimplifiedTilingData faRunTilingAdapter_;
    FiaTilingKeyInfo tilingKeyInfo_;
    FiaPlatFormInfo platformInfo_;
    uint32_t nLoopTimes_;
    uint32_t gsSize_;
    uint32_t sOuterFactor_;
    uint32_t sInnerFactor_;
    uint32_t sInnerFactorSize_;
    uint32_t blockDim_;
    uint32_t sInnerSizeAlign_ = 0;
    uint32_t headDimAlign_ = 0;
    bool flashDecodeFlag_ = false;
    bool dnFlag_ = false;
    bool gsMergeFlag_ = false;
    bool pfaMergeFlag_ = false;
    bool isQKVActualSeqLengthsRightFlag_ = false;
    bool actualSeqLenQFlag_ = false;
    bool actualSeqLenKVFlag_ = false;
    bool actualSharedPrefixLenFlag_ = false;
    std::vector<int64_t> actualSeqLengthsQ_ = {};
    std::vector<int64_t> actualSeqLengthsKV_ = {};
    bool fromPFA_ = false;
    bool isPFAFlag_ = false;
    bool isIFAFlag_ = false;
    bool needInit_ = false;
    bool enableS1OutSplit = false;
    bool isRowInvalid_ = false;
};

}  // namespace optiling
#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_FUSEDINFERATTENTIONSCORE_IMPL_H_