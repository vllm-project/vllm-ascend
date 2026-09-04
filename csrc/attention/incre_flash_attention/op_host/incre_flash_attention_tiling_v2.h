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
 * \file incre_flash_attention_tiling.h
 * \brief
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_INCREFLASHATTENTIONSCORE_NEW_V2_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_INCREFLASHATTENTIONSCORE_NEW_V2_H_

#include <queue>
#include <map>
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "op_host/data_copy_transpose_tiling.h"
#include "exe_graph/runtime/tiling_context.h"
#include "register/op_def_registry.h"
#include "../../common/op_kernel/arch35/flash_attention_score_tiling_regbase.h"
#include "../op_kernel/arch35/incre_flash_attention_tiling_regbase.h"
#include "../../prompt_flash_attention/op_kernel/arch35/prompt_flash_attention_template_tiling_key_enum.h"
#include "../../common/op_host/fia_tiling_base.h"
#include "incre_flash_attention_tiling_context.h"
#include "incre_flash_attention_tiling_struct.h"
#include "incre_flash_attention_tiling_base.h"

namespace optiling {

constexpr uint32_t PER_TOKEN_GROUP_MODE = 6;
constexpr uint32_t PER_TENSOR_HEAD_MODE = 2;
constexpr uint32_t PER_TOKEN_HEAD_MODE = 3;
constexpr uint32_t LSE_OUTPUT_INDEX = 1;
constexpr uint32_t PER_TOKEN_PA_MODE = 4;
constexpr uint32_t PER_TOKEN_HEAD_PA_MODE = 5;
constexpr uint32_t DIM_BB = 2;
constexpr uint32_t DIM_BNB = 3;
constexpr uint32_t DIM_PER_CHANNEL_H = 1;
constexpr uint32_t HIGH_PRECISION_ROW_INVALID = 2;
constexpr uint32_t HIGH_PERFORMANCE_ROW_INVALID = 3;
constexpr uint32_t SPARSE_MODE_NO_MASK = 0;
constexpr uint32_t SPARSE_MODE_ALL_MASK = 1;
constexpr uint32_t SPARSE_MODE_LEFT_UP = 2;
constexpr uint32_t SPARSE_MODE_RIGHT_DOWN = 3;
constexpr uint32_t SPARSE_MODE_BAND = 4;
constexpr int64_t SPARSE_MODE_INT_MAX = 2147483647;
constexpr uint32_t MASKDIM_SS = 2;
constexpr uint32_t MASKDIM_1SS_BSS = 3;
constexpr uint32_t MASKDIM_11SS_B1SS = 4;
constexpr uint32_t SPARSE_OPTIMIZE_ATTENTION_SIZE = 2048;
constexpr int64_t SLOPE_N_DIM_NUM = 1L;
constexpr int64_t SLIMIT = 20971520;

const std::vector<std::tuple<ge::DataType, ge::DataType, ge::DataType>> inOutDtypeSupported = {
  {ge::DT_FLOAT16, ge::DT_INT8, ge::DT_FLOAT16},
  {ge::DT_FLOAT16, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT16},
  {ge::DT_FLOAT16, ge::DT_HIFLOAT8, ge::DT_FLOAT16},
  {ge::DT_FLOAT16, ge::DT_INT4, ge::DT_FLOAT16},
  {ge::DT_FLOAT16, ge::DT_FLOAT4_E2M1, ge::DT_FLOAT16},
  {ge::DT_BF16, ge::DT_INT8, ge::DT_BF16},
  {ge::DT_BF16, ge::DT_FLOAT8_E4M3FN, ge::DT_BF16},
  {ge::DT_BF16, ge::DT_HIFLOAT8, ge::DT_BF16},
  {ge::DT_BF16, ge::DT_INT4, ge::DT_BF16},
  {ge::DT_BF16, ge::DT_FLOAT4_E2M1, ge::DT_BF16},
  {ge::DT_FLOAT16, ge::DT_INT8, ge::DT_INT8},
  {ge::DT_FLOAT16, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN},
  {ge::DT_FLOAT16, ge::DT_HIFLOAT8, ge::DT_HIFLOAT8},
  {ge::DT_BF16, ge::DT_INT8, ge::DT_INT8},
  {ge::DT_BF16, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN},
  {ge::DT_BF16, ge::DT_HIFLOAT8, ge::DT_HIFLOAT8}
};

class IFATilingV2 : public FiaTilingBase {
 public:
  IFATilingV2() : FiaTilingBase() {}
  IFATilingV2(gert::TilingContext *context) : FiaTilingBase(context) {}
  ~IFATilingV2() override = default;
  void InitTilingInfo(TilingInfo *tilingInfo) override {}
  bool IsCapable() override {return true;}
  ge::graphStatus DoOpTiling() override;
  ge::graphStatus DoSubOpTiling(IncreFlashAttentionContext& ifaContext);
  ge::graphStatus DoTiling(gert::TilingContext& context);
  ge::graphStatus RunBigKernelTiling(IncreFlashAttentionContext& context, IncreFlashAttentionTilingDataV2& tilingData);
  ge::graphStatus IncreFlashAttentionSetTilingData(gert::TilingContext& context,
                                                   IncreFlashAttentionTilingDataV2& tilingData);
  static ge::graphStatus ConvertContext(gert::TilingContext& context, IncreFlashAttentionContext& ifaContext);
  bool NeedRollBack() const
  {
    return passToOldTiling_;
  }

protected:
    void UpdateTilingKeyLayoutType();
    void UpdateTilingKeyConfig();
    void UpdateTilingKeyPseMode();
    void UpdateTilingKeyQuantMode();
    void UpdateTilingKeyAttenMask();
    void UpdateTilingKeyHasRope();
    void UpdateTilingKeyIsPa();
    void UpdateTilingKeyIsFd();
    void UpdateTilingKeyEmptyTensor();
    void UpdateTilingKeyEnableKVPrefix();
    void UpdateTilingKeySplitCoreMode();
public:
    uint8_t inOutLayoutType = 0;
    uint16_t config = 0;
    uint8_t pseMode = 0;
    uint8_t quantMode = 0;
    bool hasAttenMask = false;
    bool hasRope = false;
    bool isPa = false;
    bool isFd = false;
    bool emptyTensor = false;
    bool enableKVPrefix = false;
    bool enableS1OutSplit = false;
private:
  ge::graphStatus GetNpuInfo();
  ge::graphStatus PreProcess();
  ge::graphStatus ProcessBaseTensors();
  ge::graphStatus ProcessOptionalTensors();
  ge::graphStatus ProcessPseShift();
  ge::graphStatus ProcessAttenMask();
  ge::graphStatus ProcessAttenMaskSparsePFA();
  ge::graphStatus ProcessActualSeqLen();
  ge::graphStatus ProcessQuant2();
  ge::graphStatus ProcessDequant1();
  ge::graphStatus ProcessDequant2();
  ge::graphStatus ProcessAntiQuant();
  ge::graphStatus ProcessBlockTable();
  ge::graphStatus ProcessQPaddingSize();
  ge::graphStatus ProcessKVPaddingSize();
  ge::graphStatus ProcessPrefix();
  ge::graphStatus VerifyQuantScale2() const;
  bool EnableC1V1() const;
  void UpdatePerfMode();
  void SetfaRunFlag();
  void SetIFASparseType();
  void SetPFASparseType(uint32_t qS);
  void SetfaRunBaseSize();
  void GetMaxWorkspaceFlag();
  void SetEmptyTensor();
  void IncreFlashAttentionInitOutputSplit();
  void IncreFlashAttentionInitSoftmaxLseOutputSplit();
  bool CheckEmptyTensor();
  ge::graphStatus InitInOutMode();
  ge::graphStatus KvShapePostProcess();
  ge::graphStatus CheckKvCache();
  ge::graphStatus CheckKvCacheValue(uint32_t kDimNum) const;
  ge::graphStatus CheckInputAntiquantFormat() const;
  ge::graphStatus CheckKVShapePre();
  ge::graphStatus CheckKVShape(int64_t batchOfQuery);
  ge::graphStatus CheckFormat(ge::Format format, const std::string &sName) const;
  ge::graphStatus CheckQKOutShape() const;
  ge::graphStatus CheckLse() const;
  ge::graphStatus CheckKVHeadNum(const gert::StorageShape *inputShape) const;
  ge::graphStatus CheckKeyShapeTensor(const gert::Shape& aShape, size_t idx) const;
  ge::graphStatus EmptyTensorProcess();

  ge::graphStatus CheckUbSpace();
  ge::graphStatus CheckPABlockSize() const;
  ge::graphStatus SetL2CacheFlag();

  ge::graphStatus CheckKVAntiQuantPerHead(const gert::Shape &inputParaShape);
  ge::graphStatus CheckQuant2Shape(const gert::Shape &inputParaShape) const;
  ge::graphStatus ProcessQuant2Dtype() const;
  ge::graphStatus ProcessQuant2Attribute(const gert::Tensor *qtScale2);
  ge::graphStatus CheckKVAntiQuantPerChannel(const gert::Shape &inputParaShape) const;
  ge::graphStatus CheckKVAntiQuantShapePA(const gert::Shape &inputParaShape) const;
  ge::graphStatus CheckKVAntiQuantParaShapeLegal(const int64_t antiquantMode, const gert::Shape &inputParaShape);
  ge::graphStatus CheckAntiQuantParam(const int64_t antiquantMode, const gert::Tensor* antiquantScaleTensor, const gert::Tensor* antiquantOffsetTensor,
                                      const gert::CompileTimeTensorDesc* antiquantScaleDesc, const gert::CompileTimeTensorDesc* antiquantOffsetDesc);
  ge::graphStatus CheckSupportQLeftPadding();
  ge::graphStatus CheckSupportKVLeftPadding();
  ge::graphStatus CheckInputFormatAndLimits() const;
  ge::graphStatus KeyAndValueAntiQuantParamConsistencyCheck(const gert::Tensor* keyAntiquantTensor,
                                                            const gert::Tensor* valueAntiquantTensor,
                                                            const gert::CompileTimeTensorDesc* keyAntiquantDesc,
                                                            const gert::CompileTimeTensorDesc* valueAntiquantDesc,
                                                            int64_t keyAntiquantMode, int64_t valueAntiquantMode, const std::string sName) const;
  bool CalcUbBmm();
  bool CalcUbSoftMax();
  bool CalcUbAttenMask();
  bool CalcUbQuant();
  bool CalcUbDeQuant() const;
  bool CalcUbAntiQuant() const;
  bool CalcUbPageAttention();
  bool CalcUbKvSplit() const;

  bool CheckMaskTypeAndShape(const gert::Tensor* maskShape, ge::DataType attenMaskType) const;
  bool CheckSparseMode(bool isDefaultSparseMode, bool enableMask);
  void SetSparseModeData(bool& isBandMode, bool enableMask, bool isDefaultSparseMode);
  bool CheckMaskCrossover(const gert::Tensor* maskShape, ge::DataType attenMaskType, bool enableMask, bool isDefaultSparseMode);
  bool CheckMaskShapeCrossSparse(const gert::Tensor* maskShape, bool isDefaultSparseMode);
  bool CheckMaskShape(bool isDefaultSparseMode, const gert::Tensor* maskShape, std::string& strMaskShape, bool& checkMask);

  bool CanChangeToNew() const;
  bool ShapeEqual(const gert::Shape &aShape, const gert::Shape &bShape) const;
  void AdjustPABmm1Tiling(uint32_t& bmm1BaseN) const;
  void AdjustPABmm2Tiling() const;

  bool CheckAlibiPseShift();
  bool CheckAlibiPseShiftTypeAndShape();
  bool SetQKVStartIdx();
  bool AlibiCheckSeqLength();
  bool CheckPseShiftShape(const gert::Tensor* pseShiftInput);
  bool GetAndCheckPrefixShape(std::string layoutStr, const gert::Shape keyPrefixShape, const gert::Shape valuePrefixShape, const gert::Shape keyShape);
  bool CheckKeyValuePrefixConsistency(const gert::Shape keyPrefixShape, const gert::Shape valuePrefixShape, const gert::Shape keyShape);
  bool CheckActualSharedPrefixLen(const gert::Tensor* actualSharedPrefixLenInput, const gert::Shape keyPrefixShape, uint32_t prefixSSize_);
  bool CheckPFAMerge();

  std::string GetShapeStr(const gert::Shape &aShape) const;

  ge::graphStatus Split();
  ge::graphStatus CalcInnerSize(uint32_t seqSize);

  std::vector<int64_t> InitSparseValidArray(const int64_t* actualLens) const;
  bool BalanceLoad(const std::vector<int64_t>& sparseValidArray, int64_t totalSize, int64_t validAivNum,
                   std::vector<int64_t>& localValue, std::vector<int64_t>& sparseStartIdx) const;
  void InitLoadValue(const std::vector<int64_t>& sparseValidArray, int64_t totalSize, int64_t validAivNum,
                     const std::vector<int64_t>& sparseStartIdx, std::vector<int64_t>& localValue) const;
  void SetSparseStartIdx(const std::vector<int64_t>& sparseValidArray, int64_t totalSize, int64_t validAivNum,
                         uint32_t* sparseStartIdx, int64_t splitFactorSize) const;

  bool IsFlashDecode() const;
  bool IsFlashDecodefaRun() const;
  void PromptFlashAttentionInitOutputSplit();
  void GetActualSeqLength(int64_t &actualSeqLengths, int64_t &actualSeqLengthsKV, uint32_t bIdx);
  void GetPreNextTokensLeftUp(int64_t actualSeqLength, int64_t actualSeqLengthKV,
                              int64_t& preTokensLeftUp, int64_t& nextTokensLeftUp) const;
  void FixParamWithRowInvalid(int64_t& actualSeqLength, int64_t actualSeqLengthKV,
                              int64_t& preTokensLeftUp, int64_t& nextTokensLeftUp);
  int64_t GetCutBlockNums(int64_t blockSeqLengthKV, int64_t blockSeqLength,
                            int64_t sInner, int64_t sOuter, int64_t token) const;
  int64_t SumOfArithmeticSeries(int64_t an, int64_t d) const;
  int64_t GetCalcBlockNumsOneHead(int64_t outerBlockNums, int64_t innerBlockNums, int64_t sInnerLoopTimesPrefix,
    int64_t preTokensLeftUp, int64_t nextTokensLeftUp) const;
  int64_t GetActualInnerBlockNums(int64_t sInnerIndexStart, int64_t sInnerIndexEnd, int64_t innerBlockNums) const;
  void ComputeSplitNBSeqfaRun(std::vector<int64_t> sOuterLoopTimes, std::vector<int64_t> sInnerLoopTimes,
    int64_t sInnerLoopTimesPrefix, double coreWightTarget, uint32_t& curCore, const size_t tilingElementArrayLen);
  void SetMultiCoreParamsRegbase(int64_t totalSize, int64_t actualUsedCoreNum);
  void SetLayoutTypefaRun();
  void SetAttenMaskCompressMode();
  void IFATilingDataconvert();
  void FlashAttentionCubeSplitBNSeq();
  ge::graphStatus SplitBN_V0();
  ge::graphStatus SplitBNSfaRun();
  ge::graphStatus CheckActualSeqLens();
  int64_t GetMaxSeqLength(const gert::Tensor* actualSeqLength);
  bool CheckWorkSpace() const;
  
  bool GetMatmulType(ge::DataType getype, matmul_tiling::DataType *mmType) const;

  ge::graphStatus CalcWorkSpace();
  ge::graphStatus CalcNumBlocks() const;
  ge::graphStatus GenTilingKey();
  uint8_t GenAntiquantModeVal() const;
  ge::graphStatus FillTiling();
  void FillTilingBaseParams() const;
  void FillTilingSplitKV() const;
  void FillTilingCoreParams() const;
  void FillTilingSingleCoreParams();
  void FillTilingSingleCoreTensorSize() const;
  void FillTilingSoftmax() const;
  void FillTilingSoftmaxFlashTiling();
  void FillTilingTranspose() const;
  void FillTilingOutputParams() const;
  bool FillTilingBmm() const;  // may fail

private:
  bool passToOldTiling_ = false;
  bool isPFAFlag_ = false;
  bool needInit_ = false;
  bool needInitfaRun_ = false;
  uint32_t numHeads_ = 0;
  uint32_t sparseMode_ = 0;
  int64_t preToken_ = 0;
  int64_t nextToken_ = 0;
  float scaleValue_ = 0;
  uint32_t numKvHeads_ = 0;
  uint32_t blockSize_ = 0;
  uint32_t innerPrecise_ = 0;
  uint32_t isRowInvalid_ = 0;
  uint32_t nNumOfQInOneGroup_ = 1;
  uint32_t msdIterNum_ = 1;
  int64_t antiquantMode_ = 0;
  uint32_t antiquantPerTensorFlag_ = 0;
  uint32_t antiquantNum_ = 2;
  uint32_t antiquantPerHeadFlag_ = 0;

  uint32_t headDim_ = 0;
  uint32_t sOfQuery_ = 0;
  uint32_t seqSize_ = 0;
  uint32_t batchSize_ = 0;
  uint64_t headDimOut_ = 0;
  uint32_t headDimK_ = 0;
  uint32_t headDimV_ = 0;
  uint32_t antiquantParaSeqSize_ = 0;
  IfaLayout inputLayout_ = IfaLayout::BSH_BSND;
  uint32_t sMax_ = 0;
  uint32_t blockTypeSize_ = 0;  // 计算中间量大小
  uint32_t kvSplitPart_ = 1;
  bool emptyTensor_ = false;

  ge::DataType inputQType_ = ge::DT_FLOAT16;
  ge::DataType inputKvType_ = ge::DT_FLOAT16;
  ge::DataType outputType_ = ge::DT_FLOAT16;

  size_t ubSize_ = 0;
  size_t l1Size_ = 0;
  size_t l0cSize_ = 0;
  size_t l0bSize_ = 0;
  uint32_t coreNum_ = 0;
  uint32_t aicNum_ = 0;
  uint32_t aivNum_ = 0;
  IfaSocVersion socVersion_ = IfaSocVersion::SOC_ASCEND_910B;
  size_t libapiSize_ = 0;

  size_t mmResUbSize_ = 0;
  size_t bmm2ResUbSize_ = 0;

  size_t softmaxFlashTmpSize_ = 0;
  size_t softmaxTmpSize_ = 0;
  size_t softMaxSize_ = 0;

  size_t selectWithByteMaskTmpMinSize_ = 0;

  bool pseShiftFlag_ = false;
  uint32_t pseShiftTypeSize_ = NUM_BYTES_FLOAT16;
  uint32_t pseShiftBatch_ = 0U;
  uint32_t pseShiftS0_ = 0U;
  uint32_t pseShiftS1_ = 0U;
  bool enableAlibiPse_ = false;
  int64_t pseType_ = static_cast<int64_t>(IfaPseType::PSE_OUTER_MUL_ADD_TYPE);
  int64_t qStartIdx_ = 0;
  int64_t kvStartIdx_ = 0;
  IfaPseShapeType pseShapeType = IfaPseShapeType::PSE_B_N2_G_S1_S2;

  bool attenMaskFlag_ = false;
  uint32_t attenMaskBatch_ = 1;
  uint32_t attenMaskQSize_ = 0;
  uint32_t attenMaskKvSize_ = 0;
  uint32_t attenMaskTypeSize_ = 0;

  bool antiQuantFlag_ = false;
  size_t antiquantUb_ = 0;
  bool kvAntiParamSplitFlag_ = false;
  bool kPerChnVPerTokFlag_ = false;
  bool antiquantParamsInPageAttentionFlag_ = false;

  bool pageAttentionFlag_ = false;
  KvCacheLayout pageAttentionKvLayoutType_ = KvCacheLayout::KV_CACHE_BSH; // pa场景下kv的shape, 0:BSH 1:BNSD
  uint32_t maxBlockNumPerSeq_ = 0;
  size_t kvPageResUbSize_ = 0;
  uint32_t totalBlockNum_ = 0;
  uint64_t needBlockNum_ = 0;

  bool batchContinuousFlag_ = true;
  std::vector<int64_t> kvListSeqLens_;

  bool actualSeqLenFlag_ = false;
  bool actualSeqLenQFlag_ = false;
  bool qPaddingSizeFlag_ = false;
  bool kvPaddingSizeFlag_ = false;

  bool quantFlag_ = false;
  size_t quantUbSize_ = 0;

  uint32_t actualLenDims_ = 0;
  uint32_t actualLenQDims_ = 0;
  uint32_t maxActualseq_ = 0;

  // flash config
  uint32_t sInnerLoopTimes_ = 0;
  uint32_t sInnerSize_ = 0;  // flash attention
  uint32_t sInnerSize2_ = 0;
  uint32_t sOuterSize_ = 0;
  uint32_t sInnerSizeTail_ = 0;
  uint32_t sInnerSizeAlign_ = 0;
  uint32_t headDimAlign_ = 0;
  // uint32_t sOuterSize_;  // flash decode s2

  bool isSplitBPolicy_ = false;
  bool splitKVFlag_ = false;
  bool splitKVFlagLocal_ = false;

  IfaPerfMode perfMode_ = IfaPerfMode::NORMAL;
  TilingInOutMode inOutMode_ = TilingInOutMode::FP16_FP16;
  size_t workspaceSize_ = 0;

  uint32_t taskRation_ = 0;
  uint32_t usedCoreNum_ = 0;
  bool antiqMSDFlag_ = false;

  uint32_t startIdxEachCore_[MAX_CORE_NUM_REGBASE] = {};
  uint32_t coreSposStart_[MAX_CORE_NUM_REGBASE] = {};
  IncreFlashAttentionContext* ifaContext_ = nullptr;
  IncreFlashAttentionTilingDataRegbase* tilingData_ = nullptr;
  bool isMaxWorkspace_ = false;
  
  uint32_t formerCoreNum_ = 0;
  uint32_t blockSplitBn2Range_ = 0;
  uint32_t tailSplitedBatchRange_ = 0;

  uint32_t l2CacheOffFlag_ = 0;
  // softmaxLse
  bool softmaxLseFlag_ = false;

  // prefix
  bool enableKVPrefix_ = false;
  bool actualSharedPrefixLenNullFlag_ = true; // 默认为空
  uint32_t prefixSeqInnerSize_ = 0;
  uint32_t prefixSSize_ = 0;
  uint32_t actualSharedPrefixLen_ = 0;

  //伪量化新模板新增
  bool faRunGS_ = false;    //指示是否合轴
  int8_t isGqa_ = 0;
  bool actualSeqLenUnequal_ = false;
  int64_t pfaMergeGSLimit = 32;
  uint8_t faRunAttenMaskShapeType_ = 0;
  uint8_t faRunSparseType_ = 0;
  uint32_t paBlockNumSumfaRun_ = 1;
  uint32_t singleCoreSize_ = 0;
  int64_t totalSize_ = 0;
  int64_t totalSizeLse_ = 0;
  bool enablePostQuant_ = false;
  bool isPostQuantPerChnl_ = false;
  bool isPostQuantBF16_ = false;
  uint8_t pageAttentionKvLayoutTypefaRun_ = 0;
  FlashAttentionScoreSimplifiedTilingData faRunTilingAdapter;
};

}  // namespace optiling 
#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_INCREFLASHATTENTIONSCORE_H_