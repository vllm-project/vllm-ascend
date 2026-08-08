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
 * \file prompt_flash_attention_tiling_v2.h
 * \brief
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_PROMPTFLASHATTENTION_V2_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_PROMPTFLASHATTENTION_V2_H_
#include "op_host/data_copy_transpose_tiling_def.h"
#include "op_host/data_copy_transpose_tiling.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "register/op_def_registry.h"
#include "../../common/op_host/fia_tiling_templates_registry.h"
#include "prompt_flash_attention_tiling_context.h"
#include "prompt_flash_attention_tiling_struct.h"
#include "prompt_flash_attention_tiling_const.h"
#include "../../common/op_kernel/arch35/flash_attention_score_tiling_regbase.h"
#include "../op_kernel/arch35/prompt_flash_attention_tiling_regbase.h"
#include "../../common/op_host/fia_tiling_base.h"

namespace optiling {
namespace v2 {
#define CHECK_D_LIMITED_SCENARIO(dSize) ((dSize) != 64 && (dSize) != 128)

struct PFAShapeInfo {
    uint32_t b = 0;
    uint64_t n = 0;
    uint64_t s = 0;
    uint32_t d = 0;
    uint64_t h = 0;
    uint32_t t = 0;
};
static const std::unordered_map<ge::DataType, std::string> g_strDataTypePfa = {
    {ge::DT_FLOAT, "DT_FLOAT"},
    {ge::DT_FLOAT16, "DT_FLOAT16"},
    {ge::DT_INT8, "DT_INT8"},
    {ge::DT_INT16, "DT_INT16"},
    {ge::DT_UINT16, "DT_UINT16"},
    {ge::DT_UINT8, "DT_UINT8"},
    {ge::DT_INT32, "DT_INT32"},
    {ge::DT_INT64, "DT_INT64"},
    {ge::DT_UINT32, "DT_UINT32"},
    {ge::DT_UINT64, "DT_UINT64"},
    {ge::DT_BOOL, "DT_BOOL"},
    {ge::DT_DOUBLE, "DT_DOUBLE"},
    {ge::DT_STRING, "DT_STRING"},
    {ge::DT_DUAL_SUB_INT8, "DT_DUAL_SUB_INT8"},
    {ge::DT_DUAL_SUB_UINT8, "DT_DUAL_SUB_UINT8V"},
    {ge::DT_COMPLEX64, "DT_COMPLEX64"},
    {ge::DT_COMPLEX128, "DT_COMPLEX128"},
    {ge::DT_QINT8, "DT_QINT8"},
    {ge::DT_QINT16, "DT_QINT16"},
    {ge::DT_QINT32, "DT_QINT32"},
    {ge::DT_QUINT8, "DT_QUINT8"},
    {ge::DT_QUINT16, "DT_QUINT16"},
    {ge::DT_RESOURCE, "DT_RESOURCE"},
    {ge::DT_STRING_REF, "DT_STRING_REF"},
    {ge::DT_DUAL, "DT_DUAL"},
    {ge::DT_VARIANT, "DT_VARIANT"},
    {ge::DT_BF16, "DT_BF16"},
    {ge::DT_HIFLOAT8, "DT_HIFLOAT8"},
    {ge::DT_FLOAT8_E4M3FN, "DT_FLOAT8_E4M3FN"},
    {ge::DT_UNDEFINED, "DT_UNDEFINED"},
};

static inline std::string GetPfaDataTypeStr(ge::DataType type) {
    ge::DataType findDype = (g_strDataTypePfa.find(type) == g_strDataTypePfa.end()) ? ge::DT_UNDEFINED : type;
    return g_strDataTypePfa.at(findDype);
}

enum class AntiquantTypeEnum : uint8_t {
    PER_CHANNEL = 0,
    PER_TOKEN = 1,
    K_PER_CHANNEL_V_PER_TOKEN = 2,
    PER_TOKEN_HEAD = 3,
    PER_TOKEN_PAGE_ATTENTION = 4,
    PER_TOKEN_HEAD_PAGE_ATTENTION = 5,
    PER_BLOCK = 7
};

class PromptFlashAttentionTilingV2 : public FiaTilingBase{
public:
    explicit PromptFlashAttentionTilingV2(gert::TilingContext *context) : FiaTilingBase(context) {}
    ~PromptFlashAttentionTilingV2() override = default;
    ge::graphStatus RunBigKernelTilingWithParams(ContextParamsForPFATiling& contextKeyParams,
        uint32_t& numBlocksToBeSet, PromptFlashAttentionTilingDataV2& tilingData);
    ge::graphStatus PromptFlashAttentionSetTilingData() const;
    bool CheckNonEmptyShapeExceptions(const ContextParamsForPFATiling& contextKeyParams, const gert::StorageShape* shape,
        const std::string &sName) const;
    ge::graphStatus DoSubOpTiling(PromptFlashAttentionTilingDataV2& tilingData, ContextParamsForPFATiling& contextParamsForPFATiling);
    ge::graphStatus ConvertContextToPFAParams(ContextParamsForPFATiling& contextKeyParams) const;
    void SetTilingKey(ContextParamsForPFATiling& contextKeyParams) const;
protected:
    void InitializeMaxWorkspace(PFAShapeInfo& queryShapeInfo, PFAShapeInfo& keyShapeInfo,
        std::vector<int64_t>& actualSeqLengths, std::vector<int64_t>& actualSeqLengthsKV);
    void InitTilingInfo(TilingInfo *tilingInfo) override {}
    bool IsCapable() override {return true;}
    ge::graphStatus DoOpTiling() override;
    void PromptFlashAttentionInitOutputSplit(int64_t totalSize, PromptFlashAttentionTilingDataV2 &tilingData);
    ge::graphStatus CheckEmptyTensor(ContextParamsForPFATiling& contextKeyParams);    
    void SetEmptyTensor(ContextParamsForPFATiling& contextKeyParams, uint32_t& numBlocksToBeSet,
        PromptFlashAttentionTilingDataV2& tilingData);
    bool CheckIODataType(ContextParamsForPFATiling& contextKeyParams);
    bool SetInputLayout(const char* layout);
    bool GetAndCheckShape(ContextParamsForPFATiling& contextKeyParams, PFAShapeInfo& shapeInfo,
        const gert::StorageShape* shape, const std::string& sName) const;
    bool GetAndCheckRopeShape(ContextParamsForPFATiling& contextKeyParams, PFAShapeInfo& shapeInfo,
        PFAShapeInfo& ropeShapeInfo, const gert::StorageShape* shape, const std::string& sName,
        const std::string& rName) const;
    bool SetShape(ContextParamsForPFATiling& contextKeyParams, const gert::StorageShape* shape,
        const std::string inputName, int64_t& b, int64_t& n, int64_t& s, int64_t& d, int64_t& h, int64_t& t) const;
    bool CheckQueryOutParamsConsistency(const ContextParamsForPFATiling& contextKeyParams,
        const gert::StorageShape* queryShape, const gert::StorageShape* outShape) const;
    bool CheckKVDataType(ContextParamsForPFATiling& contextKeyParams) const;
    bool CheckRopeDataType(ContextParamsForPFATiling& contextKeyParams) const;
    bool CheckKeyValueParamsConsistency(ContextParamsForPFATiling& contextKeyParams,
        const gert::StorageShape* keyShape, const gert::StorageShape* valueShape);
    bool SetAndCheckHeadNumRatio(ContextParamsForPFATiling& contextKeyParams, PromptFlashAttentionTilingDataV2& tilingData);
    bool CheckInputDimAndHeadNum(ContextParamsForPFATiling& contextKeyParams, const uint64_t nQAttr, const uint64_t nKVAttr);
    bool CheckPostQuantShape(const ContextParamsForPFATiling &contextKeyParams,
                             const gert::StorageShape *quantScale2Shape, const gert::StorageShape *quantOffset2Shape,
                             const PFAShapeInfo &queryShapeInfo, const PFAShapeInfo &valueShapeInfo) const;
    bool CheckPostQuantParams(const ContextParamsForPFATiling& contextKeyParams, const PFAShapeInfo& queryShapeInfo, const PFAShapeInfo& valueShapeInfo) const;
    bool CheckPerTensorQuantParams(const ContextParamsForPFATiling& contextKeyParams, const PFAShapeInfo& queryShapeInfo) const;
    bool CheckPerblockQuantParams(const ContextParamsForPFATiling& contextKeyParams, const PFAShapeInfo& queryShapeInfo, const PFAShapeInfo& keyShapeInfo, const PFAShapeInfo& valueShapeInfo) const;
    bool CheckAntiquantParamsShape(ContextParamsForPFATiling& contextKeyParams) const;
    bool GetAndCheckPrefixShape(ContextParamsForPFATiling& contextKeyParams, PFAShapeInfo& keyShapeInfo,
        PFAShapeInfo& prefixShapeInfo) const;
    bool CheckKeyValuePrefixConsistency(ContextParamsForPFATiling& contextKeyParams, const gert::StorageShape* keyShape);
    bool CheckActSharedPrefix(ContextParamsForPFATiling& contextKeyParams, const uint32_t sPrefix, const uint32_t sKV);
    bool CheckPAKeyValueShape(ContextParamsForPFATiling& contextKeyParams, int64_t& keyDim1, PFAShapeInfo& queryShapeInfo,
        const gert::StorageShape* keyShape, const gert::StorageShape* valueShape, const size_t keyDim, const int32_t* blockSize,
        int64_t blockNumValid, int64_t headNumRatio);
    bool CheckPACacheShape(ContextParamsForPFATiling& contextKeyParams, const size_t keyDim, PFAShapeInfo& shapeInfo,
        const gert::StorageShape* shape, const int32_t* blockSize, int64_t blockNumValid, int64_t headNumRatio, const std::string& sName);
    bool CheckBlockTableShape(ContextParamsForPFATiling& contextKeyParams, PFAShapeInfo& queryShapeInfo, PFAShapeInfo& queryRopeShapeInfo,
        const int32_t* blockSize, const gert::StorageShape* blockTableShape, PromptFlashAttentionTilingDataV2& tilingData);
    bool CheckMaskShape(ContextParamsForPFATiling& contextKeyParams, const int32_t* sparseMode, int64_t& attenMaskBatch,
        int64_t& attenMaskS1, int64_t& attenMaskS2, bool& checkMask, const uint64_t sQ, const uint64_t sK,
        const uint32_t batchSize, std::string& strMaskShape);
    void SetSparseModeData(ContextParamsForPFATiling& contextKeyParams, const gert::StorageShape* attenMaskShape,
        const int32_t* sparseMode, const int64_t* preTokens, const int64_t* nextTokens);
    bool CheckMaskShapeCrossSparse(ContextParamsForPFATiling& contextKeyParams, PromptFlashAttentionTilingDataV2& tilingData,
        const int32_t* sparseMode, uint64_t sQ, const uint64_t sK, const uint32_t batchSize);
    bool CheckMaskCrossIFAMLA(ContextParamsForPFATiling& contextKeyParams, const int32_t *sparseMode, uint64_t queryS);
    bool CheckIO(ContextParamsForPFATiling& contextKeyParams, PFAShapeInfo& queryShapeInfo, PFAShapeInfo& valueShapeInfo);
    bool CheckKV(ContextParamsForPFATiling& contextKeyParams, PFAShapeInfo& keyShapeInfo, PFAShapeInfo& valueShapeInfo);
    bool CheckQueryAndKey(ContextParamsForPFATiling& contextKeyParams, PFAShapeInfo& queryShapeInfo, 
        PFAShapeInfo& keyShapeInfo, PromptFlashAttentionTilingDataV2& tilingData);
    bool CheckPFAMerge(ContextParamsForPFATiling& contextKeyParams, const PFAShapeInfo& queryShapeInfo) const;
    bool CheckRope(ContextParamsForPFATiling& contextKeyParams, PFAShapeInfo& queryShapeInfo,
        PFAShapeInfo& keyShapeInfo, PFAShapeInfo& queryRopeShapeInfo);
    bool CheckLayout(ContextParamsForPFATiling& contextKeyParams) const;
    bool CheckIFAMLA(ContextParamsForPFATiling& contextKeyParams, const PFAShapeInfo& queryShapeInfo) const;
    bool CheckQuant(ContextParamsForPFATiling& contextKeyParams, PFAShapeInfo& queryShapeInfo, PFAShapeInfo& keyShapeInfo, const PFAShapeInfo& valueShapeInfo) const;
    bool CheckQScaleShape4MLAFullQuant(ContextParamsForPFATiling& contextKeyParams) const;
    bool CheckKVScaleShape4MLAFullQuant(ContextParamsForPFATiling& contextKeyParams) const;
    bool CheckMLAFullQuant(ContextParamsForPFATiling& contextKeyParams, PFAShapeInfo& queryShapeInfo) const;
    bool CheckPrefix(ContextParamsForPFATiling& contextKeyParams, const PFAShapeInfo& queryShapeInfo, PFAShapeInfo& keyShapeInfo, 
        PromptFlashAttentionTilingDataV2& tilingData);
    bool CheckActSeq(const ContextParamsForPFATiling& contextKeyParams, const PFAShapeInfo& queryShapeInfo) const;
    bool CheckActSeqLen(ContextParamsForPFATiling& contextKeyParams, PFAShapeInfo& queryShapeInfo,
        PFAShapeInfo& keyShapeInfo);
    bool CheckPATypeAndShape(ContextParamsForPFATiling& contextKeyParams, PFAShapeInfo& queryShapeInfo,
        PFAShapeInfo& queryRopeShapeInfo, PromptFlashAttentionTilingDataV2& tilingData);
    bool CheckPseShiftTypeAndShape(ContextParamsForPFATiling& contextKeyParams, uint32_t b, uint32_t n,
        uint32_t s1, uint32_t s2);
    bool CheckInnerPrecise(ContextParamsForPFATiling& contextKeyParams, PromptFlashAttentionTilingDataV2& tilingData);
    bool CheckMaskTypeAndShape(ContextParamsForPFATiling& contextKeyParams, PromptFlashAttentionTilingDataV2& tilingData);
    void SetSparseType(uint32_t qS);
    bool CheckSparseMode(ContextParamsForPFATiling& contextKeyParams, uint32_t qS);
    bool CheckPACrossover(ContextParamsForPFATiling& contextKeyParams, PFAShapeInfo& queryShapeInfo);
    bool CheckMaskCrossover(ContextParamsForPFATiling& contextKeyParams, PFAShapeInfo& queryShapeInfo, 
        PromptFlashAttentionTilingDataV2& tilingData);
    bool CheckTNDLayoutCrossover(ContextParamsForPFATiling& contextKeyParams);
    bool CheckNTDLayoutCrossover(ContextParamsForPFATiling& contextKeyParams, PFAShapeInfo& queryShapeInfo);
    bool CheckTransposeLayoutCrossover(ContextParamsForPFATiling& contextKeyParams, PFAShapeInfo& queryShapeInfo);
    bool CheckLearnSink(ContextParamsForPFATiling& contextKeyParams, PFAShapeInfo& queryShapeInfo);
    bool ParseActualSeqLengths(ContextParamsForPFATiling& contextKeyParams, PFAShapeInfo& queryShapeInfo, 
        std::vector<int64_t>& actualSeqLengths, std::vector<int64_t>& actualSeqLengthsKV);
    bool CheckMultiFeatureCrossover(ContextParamsForPFATiling& contextKeyParams, PFAShapeInfo& queryShapeInfo, 
        std::vector<int64_t>& actualSeqLengths, std::vector<int64_t>& actualSeqLengthsKV, PromptFlashAttentionTilingDataV2& tilingData);
    bool CheckPerblockCrossover(ContextParamsForPFATiling& contextKeyParams, PromptFlashAttentionTilingDataV2& tilingData);
    void SetTilingDataAttribute(ContextParamsForPFATiling& contextKeyParams, PromptFlashAttentionTilingDataV2& tilingData);
    void GetEnableDN(ContextParamsForPFATiling& contextKeyParams, PromptFlashAttentionTilingDataV2& tilingData,
        PFAShapeInfo& queryShapeInfo, PFAShapeInfo& valueShapeInfo, std::vector<int64_t>& actualSeqLengths, std::vector<int64_t>& actualSeqLengthsKV);
    void SetTilingData(ContextParamsForPFATiling& contextKeyParams, PFAShapeInfo& queryShapeInfo,
        PFAShapeInfo& queryRopeShapeInfo, PFAShapeInfo& valueShapeInfo, PromptFlashAttentionTilingDataV2 &tilingData);
    void InferTilingMod(const ContextParamsForPFATiling& contextKeyParams, std::vector<int64_t>& actualSeqLengths,
        std::vector<int64_t>& actualSeqLengthsKV, uint32_t actualSeqArrayLen, uint32_t d);
    int64_t GetActualInnerBlockNums(int64_t sInnerIndexStart, int64_t sInnerIndexEnd, int64_t innerBlockNums) const;
    int64_t SumOfArithmeticSeries(int64_t an, int64_t d) const;
    int64_t GetCutBlockNums(int64_t blockSeqLengthKV, int64_t blockSeqLength, int64_t sInner, int64_t sOuter, int64_t token) const;
    void FixParamWithRowInvalid(int64_t& actualSeqLength, int64_t actualSeqLengthKV, int64_t& preTokensLeftUp,
        int64_t& nextTokensLeftUp) const;
    int64_t GetCalcBlockNumsOneHead(int64_t actualSeqLength, int64_t actualSeqLengthKV, uint32_t sOuterSize,
        uint32_t sInnerSize, int64_t preTokensLeftUp, int64_t nextTokensLeftUp, bool isAttenMaskUsed) const;
    void ComputeSplitNBSeq(PromptFlashAttentionTilingDataV2& tilingData, uint32_t batchSize, const size_t tilingElementArrayLen,
        std::vector<int64_t>& actualSeqLengths, std::vector<int64_t>& actualSeqLengthsKV, uint32_t sOuterSize,
        uint32_t sInnerSize, double coreWightTarget, uint32_t& curCore);
    void PromptFlashAttentionSplitNBSeq(PromptFlashAttentionTilingDataV2& tilingData, std::vector<int64_t>& actualSeqLengths,
        std::vector<int64_t>& actualSeqLengthsKV, bool isAttenMaskUsed);
    void InferSplitCoreMode();
    void InferConstantization();
    bool AdjustCVTilingCVDiff(const ContextParamsForPFATiling& contextKeyParams, uint32_t& sOuterFactor,
        uint32_t& sInnerFactor, uint32_t& softmaxSOuterFactor, PromptFlashAttentionTilingDataV2& tilingData,
        const PFAShapeInfo& queryShapeInfo);
    void GetMatMulType(matmul_tiling::DataType &mmInputType, matmul_tiling::DataType &mmOutputType);
    bool EnableMTE2BmmPipe(PromptFlashAttentionTilingDataV2& tilingData, matmul_tiling::MatmulApiTiling& bmm,
        TCubeTiling& bmmTilingData, uint32_t sOuterFactor, uint32_t sInnerFactor);
    void EnableBmmDoubleBuffer(TCubeTiling& bmmTilingData) const;
    bool PromptFlashAttentionCheckBmm1(PromptFlashAttentionTilingDataV2& tilingData, TCubeTiling& bmm1TilingData,
        int64_t l1SizeRemain, int64_t l0CSize, uint32_t sOuterFactor, uint32_t sInnerFactor, bool autoBaseMNK=false);
    bool PromptFlashAttentionCheckBmm2(PromptFlashAttentionTilingDataV2& tilingData, TCubeTiling& bmm2TilingData,
        int64_t l1SizeRemain, int64_t l0CSize, uint32_t sOuterFactor, uint32_t sInnerFactor, uint32_t dSplitFactor,
        bool autoBaseMNK=false);
    bool PromptFlashAttentionComputeCVDiffParams(PromptFlashAttentionTilingDataV2& tilingData,
        int64_t l1Size, int64_t l0CSize, uint32_t& sOuterFactor, uint32_t &sInnerFactor);
    void GetPreNextTokensLeftUp(PromptFlashAttentionTilingDataV2& tilingData, int64_t actualSeqLength, 
        int64_t actualSeqLengthKV, int64_t& preTokensLeftUp, int64_t& nextTokensLeftUp);
    void PromptFlashAttentionInitSoftmaxLseOutputSplit(int64_t totalSize, PromptFlashAttentionTilingDataV2 &tilingData);
    bool TilingGetTilingKeyAttentionAscendC(ContextParamsForPFATiling& contextKeyParams,
        PromptFlashAttentionTilingDataV2 &tilingData);
    size_t GetPFAWorkSpaceSize(PromptFlashAttentionTilingDataV2& tilingData);
    ge::graphStatus SetPlatMemoryInfo(ContextParamsForPFATiling& contextKeyParams);
    ge::graphStatus SetAttributeInfo(ContextParamsForPFATiling& contextKeyParams);
    ge::graphStatus CheckTensorInvalid(const ContextParamsForPFATiling& contextKeyParams) const;
    ge::graphStatus CheckSingleAttribute(ContextParamsForPFATiling& contextKeyParams, PFAShapeInfo& queryShapeInfo, 
        PFAShapeInfo& keyShapeInfo, PFAShapeInfo& valueShapeInfo, PFAShapeInfo& queryRopeShapeInfo, PromptFlashAttentionTilingDataV2& tilingData);
    ge::graphStatus CheckCrossoverAttribute(ContextParamsForPFATiling& contextKeyParams, PFAShapeInfo& queryShapeInfo,
        PromptFlashAttentionTilingDataV2& tilingData);
    ge::graphStatus AdjustTilingData(ContextParamsForPFATiling& contextKeyParams,
        PromptFlashAttentionTilingDataV2& tilingData, const PFAShapeInfo& queryShapeInfo);
    ge::graphStatus ComputeTilingData(ContextParamsForPFATiling& contextKeyParams, std::vector<int64_t>& actualSeqLengths,
        std::vector<int64_t>& actualSeqLengthsKV, PromptFlashAttentionTilingDataV2& tilingData);
    ge::graphStatus ComputeTilingKey(ContextParamsForPFATiling& contextKeyParams,
        uint32_t& numBlocksToBeSet, PromptFlashAttentionTilingDataV2& tilingData);
    void SetAttenMaskCompressMode();
    void SetLayoutType();
    void PFATilingDataconvert(PromptFlashAttentionTilingDataV2& tilingData);
    void SetMultiCoreParamsRegbase(int64_t totalSize, int64_t coreNum);
    bool IsFlashDecode(ContextParamsForPFATiling& contextKeyParams, uint64_t bng) const;
    ge::graphStatus SplitBNS(PromptFlashAttentionTilingDataV2& tilingData, uint64_t bng);
    bool CheckAlibiPseShiftTypeAndShape(ContextParamsForPFATiling& contextKeyParams, uint32_t n);
    ge::graphStatus SetQKVStartIdx(ContextParamsForPFATiling& contextKeyParams);
    bool CheckAlibiPseCrossover(ContextParamsForPFATiling& contextKeyParams);
    void GetMaxWorkspaceFlag(ContextParamsForPFATiling& contextKeyParams);
    void GetQueryDimAndOutDim(const gert::StorageShape* queryShape, const gert::StorageShape* outShape,
        const std::string &layoutStr, int64_t &tmpqueryDim, int64_t &outDim, uint32_t i) const;

    void UpdateTilingKeyLayoutType();
    void UpdateTilingKeyConfig(ContextParamsForPFATiling& contextKeyParams, PromptFlashAttentionTilingDataV2& tilingData);
    void UpdateTilingKeyPseMode();
    void UpdateTilingKeyQuantMode(ge::DataType inputDataType);
    void UpdateTilingKeyAttenMask(ge::DataType inputDataType);
    void UpdateTilingKeyHasRope(ge::DataType inputDataType);
    void UpdateTilingKeyIsPa(ge::DataType inputDataType);
    void UpdateTilingKeyIsFd(ge::DataType inputDataType);
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
  
protected:
    ContextParamsForPFATiling* contextKeyParamsPtr = nullptr;
    int64_t ubSizeRemain = 1;
    bool enableFlashDecode = false;
    bool isSOuterNoTail = true;
    bool isSInnerNoTail = true;
    bool isDNoTail = true;
    bool enableTensorList = false;
    bool enableLeftPadding = false;
    bool enableActSeqLen = false;
    bool enableActSeqLenKV = false;
    bool enableKVAntiquant = false;
    bool enablePseShift = false;
    bool enableAlibiPse = false;
    bool enableMask = false;
    bool enableQuantBF16 = false;
    bool enableMatmulNorm = false;
    bool enablePA = false;
    bool enableSplitSeqOneN = false;
    bool isDefaultSparseMode = false;
    bool isKVHasPrefix = false;
    bool isBandMode = false;
    bool enableIFAMLA = false;
    bool enableIFA = false;
    bool enableIFAMask = false;
    bool enableIFAMLAFullQuant = false;
    // MLPerf合轴优化
    bool enablePFAMerge = false;
    uint32_t pfaMergeGLimit = 16;
    uint32_t pfaMergeQsLimit = 4;
    bool enablePFAMLA = false;
    bool enablePFARope = false;
    bool enableDN = false;
    bool enablePostQuant = false;
    bool enablePertensorQuant = false;
    bool enablePerblockQuant = false;
    // attention sink
    bool enableLearnSink = false;
    uint64_t gSize = 1;
    int64_t t1Size = 0;
    int64_t t2Size = 0;
    InputLayout inputLayout = InputLayout::BSH;
    ge::DataType inputType{ge::DT_FLOAT16};
    ge::DataType outputType{ge::DT_FLOAT16};
    ge::DataType pseShiftElemType{ge::DT_FLOAT16};
    ge::DataType queryType{ge::DT_FLOAT};
    ge::DataType keyType{ge::DT_FLOAT};
    ge::DataType valueType{ge::DT_FLOAT};
    uint32_t dataTypeSize = FLOAT32SIZE;
    uint32_t outputDataTypeSize = FLOAT32SIZE;
    uint32_t maskElemSize = FLOAT32SIZE;
    int32_t ifaBlockSizeBase = 32;
    uint32_t coreNum = 0;
    uint32_t aivNum = 0;
    uint32_t aicNum = 0;
    uint32_t typeByteNum = 0;
    uint32_t outputTypeByteNum = 0;
    uint32_t softmaxTypeByteNum = 0;
    uint32_t pseShiftTypeByteNum = 0;
    uint32_t pseShiftElemSize = 0;
    uint32_t pseMaskMaxSize = 0;
    int64_t pseShiftBatch = 0;
    int64_t pseShiftS1 = 0;
    int64_t pseShiftS2 = 0;
    int64_t actSeqLenDims = 0;
    int64_t actSeqLenKVDims = 0;
    int64_t middleActualSeqLengths = 0;
    int64_t actualSharedPrefixLen = 0;
    uint32_t needInit = 0U;
    uint32_t usePseShift = 0;
    // There is no S2 axis for PA. Use the change amount to normalize the S2 length in both PA and non PA scenarios
    uint32_t S2 = 0;
    int32_t blockTableDim2 = 1;
    int32_t paBlockNumSum = 1;
    uint32_t maskTypeByteNum = 0;
    uint32_t softmaxDataTypeSize = FLOAT32SIZE; // BF16 calculates through FP32
    platform_ascendc::SocVersion curShortSocName;
    uint32_t layoutType = 0;
    uint32_t paLayoutType = 0;
    int64_t sparsePreTokens = 0;
    int64_t sparseNextTokens = 0;
    int32_t sparseModeVal = 0;
    int64_t maxActualseqKV = 0;
    SplitCoreMode splitCoreMode = SplitCoreMode::SPLIT_NBS_VECTOR;
    bool isConstantization = false;
    uint32_t splitS2 = 1; // It can only be 0 when the D axis is split
    int32_t innerPrecise = HIGH_PERFORMANCE;
    uint32_t sOuterFactorTiling = 0;
    uint32_t softmaxSInnerFactorTiling = 0;
    uint32_t softmaxSOuterFactorTiling = 0;
    matmul_tiling::PlatformInfo ascendPlatformInfo;

    bool isMaxWorkspace = false;
    bool isQKVDDifferent = false;
    bool faRunFlag_ = true;
    uint8_t attenMaskShapeType = 0; // 0: (B,N2,G,S1,S2), 1: (B,1,1,S1,S2), 2: (1,1,1,S1,S2)
    uint8_t sparseType = 0;
    int64_t pseType = 0;
    FlashAttentionScoreSimplifiedTilingData faTilingAdapter;
};
} // namespace v2
} // namespace optiling

#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_PROMPTFLASHATTENTION_V2_H_
