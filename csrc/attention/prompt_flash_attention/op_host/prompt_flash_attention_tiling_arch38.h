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
 * \file prompt_flash_attention_tiling_arch38.h
 * \brief
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_PROMPTFLASHATTENTION_ARCH38_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_PROMPTFLASHATTENTION_ARCH38_H_
#include "op_host/data_copy_transpose_tiling_def.h"
#include "op_host/data_copy_transpose_tiling.h"
#include "register/tilingdata_base.h"
#include "./prompt_flash_attention_tiling.h"

namespace optiling {
namespace arch38 {

std::string GetPfaDataTypeStr(ge::DataType type);
class PromptFlashAttentionTilingArch38 : public FiaTilingBase{
public:
    platform_ascendc::PlatformAscendC ascendcPlatform;
    // PromptFlashAttentionTilingArch38(fe::PlatFormInfos* platFormInfo): ascendcPlatform(platFormInfo) {}
    explicit PromptFlashAttentionTilingArch38(gert::TilingContext *context) : FiaTilingBase(context), ascendcPlatform(nullptr) {}
    ~PromptFlashAttentionTilingArch38() override = default;
    ge::graphStatus RunBigKernelTilingWithParams(ContextParamsForPFATiling& contextKeyParams,
        uint64_t& tilingKey, uint32_t& numBlocksToBeSet, PromptFlashAttentionTilingData& tilingData);
    ge::graphStatus PromptFlashAttentionSetTilingData(gert::TilingContext* context,
        PromptFlashAttentionTilingData& tilingData);
    ge::graphStatus DoSubOpTiling(PromptFlashAttentionTilingData& tilingData, ContextParamsForPFATiling& contextParamsForPFATiling);
    bool CheckNonEmptyShapeExceptions(const ContextParamsForPFATiling& contextKeyParams, const gert::StorageShape* shape,
        const std::string &sName) const;
protected:
    void InitTilingInfo(TilingInfo *tilingInfo) override {}
    bool IsCapable() override {return true;}
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus ConvertContextToPFAParams(ContextParamsForPFATiling& contextKeyParams) const;
    void PromptFlashAttentionInitOutputSplit(int64_t totalSize, PromptFlashAttentionTilingData &tilingData);
    bool CheckEmptyTensor(ContextParamsForPFATiling& contextKeyParams) const;
    void SetEmptyTensor(ContextParamsForPFATiling& contextKeyParams, uint64_t& tilingKey, uint32_t& numBlocksToBeSet,
        PromptFlashAttentionTilingData& tilingData);
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
    bool CheckKeyValueParamsConsistency(ContextParamsForPFATiling& contextKeyParams,
        const gert::StorageShape* keyShape, const gert::StorageShape* valueShape) const;
    bool SetAndCheckHeadNumRatio(ContextParamsForPFATiling& contextKeyParams, PromptFlashAttentionTilingData& tilingData);
    bool CheckInputDimAndHeadNum(ContextParamsForPFATiling& contextKeyParams, const uint32_t nQAttr, const uint32_t nKVAttr);
    bool CheckPostQuantShape(const ContextParamsForPFATiling& contextKeyParams,
        const gert::StorageShape* quantOffset2Shape, const ge::DataType quantScale2Type, int64_t quantScale2ShapeSize,
        const PFAShapeInfo& queryShapeInfo, const PFAShapeInfo& valueShapeInfo) const;
    bool CheckPostQuantParams(const ContextParamsForPFATiling& contextKeyParams, const PFAShapeInfo& queryShapeInfo, const PFAShapeInfo& valueShapeInfo) const;
    bool CheckPerTensorQuantParams(const ContextParamsForPFATiling& contextKeyParams) const;
    bool CheckPerblockQuantParams(const ContextParamsForPFATiling& contextKeyParams, const PFAShapeInfo& queryShapeInfo, const PFAShapeInfo& keyShapeInfo, const PFAShapeInfo& valueShapeInfo) const;
    bool CheckAntiquantParamsShape(ContextParamsForPFATiling& contextKeyParams) const;
    bool GetAndCheckPrefixShape(ContextParamsForPFATiling& contextKeyParams, const PFAShapeInfo& queryShapeInfo,
        PFAShapeInfo& prefixShapeInfo,
        PromptFlashAttentionTilingData& tilingData) const;
    bool CheckKeyValuePrefixConsistency(ContextParamsForPFATiling& contextKeyParams, const gert::StorageShape* keyShape) const;
    bool CheckActSharedPrefix(ContextParamsForPFATiling& contextKeyParams, const uint32_t sPrefix, const uint32_t sKV);
    bool CheckPAKeyValueShape(ContextParamsForPFATiling& contextKeyParams, int64_t& keyDim1, const PFAShapeInfo& queryShapeInfo,
        const gert::StorageShape* keyShape, const gert::StorageShape* valueShape, const size_t keyDim, const int32_t* blockSize,
        int64_t blockNumValid, int32_t headNumRatio) const;
    bool CheckPACacheShape(ContextParamsForPFATiling& contextKeyParams, const size_t keyDim, const PFAShapeInfo& shapeInfo,
        const gert::StorageShape* shape, const int32_t* blockSize, int64_t blockNumValid, int32_t headNumRatio, const std::string& sName);
    bool CheckBlockTableShape(ContextParamsForPFATiling& contextKeyParams, PFAShapeInfo& queryShapeInfo, PFAShapeInfo& queryRopeShapeInfo,
        const int32_t* blockSize, const gert::StorageShape* blockTableShape, PromptFlashAttentionTilingData& tilingData);
    bool CheckMaskShape(ContextParamsForPFATiling& contextKeyParams, const int32_t* sparseMode, int64_t& attenMaskBatch,
        int64_t& attenMaskS1, int64_t& attenMaskS2, bool& checkMask, const uint32_t sQ, const uint32_t sK,
        const uint32_t batchSize, std::string& strMaskShape) const;
    void SetSparseModeData(ContextParamsForPFATiling& contextKeyParams, const gert::StorageShape* attenMaskShape,
        PromptFlashAttentionTilingData& tilingData, const int32_t* sparseMode, const int64_t* preTokens,
        const int64_t* nextTokens);
    bool CheckMaskShapeCrossSparse(ContextParamsForPFATiling& contextKeyParams, const int32_t* sparseMode,
        uint64_t sQ, const uint64_t sK, const uint32_t batchSize);
    bool CheckMaskCrossIFAMLA(ContextParamsForPFATiling& contextKeyParams, const int32_t *sparseMode, uint64_t queryS);
    bool CheckIO(ContextParamsForPFATiling& contextKeyParams, PFAShapeInfo& queryShapeInfo, PFAShapeInfo& valueShapeInfo);
    bool CheckKV(ContextParamsForPFATiling& contextKeyParams, PFAShapeInfo& keyShapeInfo, PFAShapeInfo& valueShapeInfo) const;
    bool CheckQueryAndKey(ContextParamsForPFATiling& contextKeyParams, PFAShapeInfo& queryShapeInfo, 
        PFAShapeInfo& keyShapeInfo, PromptFlashAttentionTilingData& tilingData);
    bool CheckPFAMerge(ContextParamsForPFATiling& contextKeyParams, const PFAShapeInfo& queryShapeInfo) const;
    bool CheckRope(ContextParamsForPFATiling& contextKeyParams, PFAShapeInfo& queryShapeInfo,
        PFAShapeInfo& keyShapeInfo, PFAShapeInfo& queryRopeShapeInfo);
    bool CheckIFAMLA(ContextParamsForPFATiling& contextKeyParams, const PFAShapeInfo& queryShapeInfo) const;
    bool CheckQuant(ContextParamsForPFATiling& contextKeyParams, PFAShapeInfo& queryShapeInfo, PFAShapeInfo& keyShapeInfo, const PFAShapeInfo& valueShapeInfo) const;
    bool CheckPrefix(ContextParamsForPFATiling& contextKeyParams, PFAShapeInfo& queryShapeInfo, PFAShapeInfo& keyShapeInfo, 
        PromptFlashAttentionTilingData& tilingData);
    bool CheckActSeq(const ContextParamsForPFATiling& contextKeyParams, const PFAShapeInfo& queryShapeInfo) const;
    bool CheckActSeqLen(ContextParamsForPFATiling& contextKeyParams, PFAShapeInfo& queryShapeInfo,
        const PFAShapeInfo& keyShapeInfo);
    bool CheckPATypeAndShape(ContextParamsForPFATiling& contextKeyParams, PFAShapeInfo& queryShapeInfo,
        PFAShapeInfo& queryRopeShapeInfo, PromptFlashAttentionTilingData& tilingData);
    bool CheckPseShiftTypeAndShape(ContextParamsForPFATiling& contextKeyParams, uint32_t b, uint32_t n,
        uint32_t s1, uint32_t s2);
    bool CheckInnerPrecise(ContextParamsForPFATiling& contextKeyParams, PromptFlashAttentionTilingData& tilingData);
    bool CheckMaskTypeAndShape(ContextParamsForPFATiling& contextKeyParams, PromptFlashAttentionTilingData& tilingData);
    void SetSparseType(uint32_t qS);
    bool CheckSparseMode(ContextParamsForPFATiling& contextKeyParams, uint32_t qS, PromptFlashAttentionTilingData& tilingData);
    bool CheckPACrossover(ContextParamsForPFATiling& contextKeyParams, PFAShapeInfo& queryShapeInfo);
    bool CheckMaskCrossover(ContextParamsForPFATiling& contextKeyParams, PFAShapeInfo& queryShapeInfo, 
        PromptFlashAttentionTilingData& tilingData);
    bool CheckTNDLayoutCrossover(ContextParamsForPFATiling& contextKeyParams);
    bool ParseActualSeqLengths(ContextParamsForPFATiling& contextKeyParams, PFAShapeInfo& queryShapeInfo, 
        std::vector<int64_t>& actualSeqLengths, std::vector<int64_t>& actualSeqLengthsKV);
    bool CheckMultiFeatureCrossover(ContextParamsForPFATiling& contextKeyParams, PFAShapeInfo& queryShapeInfo, 
        std::vector<int64_t>& actualSeqLengths, std::vector<int64_t>& actualSeqLengthsKV, PromptFlashAttentionTilingData& tilingData);
    bool CheckPerblockCrossover(ContextParamsForPFATiling& contextKeyParams);
    void SetTilingDataAttribute(ContextParamsForPFATiling& contextKeyParams, PromptFlashAttentionTilingData& tilingData);
    void GetEnableDN(PromptFlashAttentionTilingData& tilingData, PFAShapeInfo& queryShapeInfo, const PFAShapeInfo& valueShapeInfo, 
        std::vector<int64_t>& actualSeqLengths, std::vector<int64_t>& actualSeqLengthsKV);
    void SetTilingData(ContextParamsForPFATiling& contextKeyParams, PFAShapeInfo& queryShapeInfo,
        PFAShapeInfo& queryRopeShapeInfo, PFAShapeInfo& valueShapeInfo, PromptFlashAttentionTilingData &tilingData);
    void InferTilingMod(const ContextParamsForPFATiling& contextKeyParams, std::vector<int64_t>& actualSeqLengths,
        std::vector<int64_t>& actualSeqLengthsKV, uint32_t actualSeqArrayLen, uint32_t d);
    int64_t GetSInnerBlockNums(int64_t sInnerIndexStart, int64_t sInnerIndexEnd, int64_t innerBlockNums) const;
    int64_t GetCutBlockNums(int64_t blockSeqLengthKV, int64_t blockSeqLength, int64_t sInner, int64_t sOuter, int64_t token) const;
    void FixParamWithRowInvalid(int64_t& actualSeqLength, int64_t actualSeqLengthKV, int64_t& preTokensLeftUp,
        int64_t& nextTokensLeftUp) const;
    int64_t GetCalcBlockNumsOneHead(int64_t actualSeqLength, int64_t actualSeqLengthKV, uint32_t sOuterSize,
        uint32_t sInnerSize, int64_t preTokensLeftUp, int64_t nextTokensLeftUp, bool isAttenMaskUsed) const;
    void ComputeSplitNBSeq(PromptFlashAttentionTilingData& tilingData, uint32_t batchSize, const size_t tilingElementArrayLen,
        std::vector<int64_t>& actualSeqLengths, std::vector<int64_t>& actualSeqLengthsKV, uint32_t sOuterSize,
        uint32_t sInnerSize, double coreWightTarget, uint32_t& curCore);
    void PromptFlashAttentionSplitNBSeq(PromptFlashAttentionTilingData& tilingData, std::vector<int64_t>& actualSeqLengths,
        std::vector<int64_t>& actualSeqLengthsKV, bool isAttenMaskUsed);
    void InferSplitCoreMode();
    void InferConstantization();
    bool AdjustCVTilingCVDiff(const ContextParamsForPFATiling& contextKeyParams, uint32_t& sOuterFactor,
        uint32_t& sInnerFactor, uint32_t& softmaxSOuterFactor, PromptFlashAttentionTilingData& tilingData,
        const PFAShapeInfo& queryShapeInfo);
    void GetMatMulType(matmul_tiling::DataType &mmInputType, matmul_tiling::DataType &mmOutputType) const;
    bool EnableMTE2BmmPipe(PromptFlashAttentionTilingData& tilingData, matmul_tiling::MatmulApiTiling& bmm,
        TCubeTiling& bmmTilingData, uint32_t sOuterFactor, uint32_t sInnerFactor) const;
    void EnableBmmDoubleBuffer(TCubeTiling& bmmTilingData) const;
    bool PromptFlashAttentionCheckBmm1(PromptFlashAttentionTilingData& tilingData, TCubeTiling& bmm1TilingData,
        int64_t l1SizeRemain, int64_t l0CSize, uint32_t sOuterFactor, uint32_t sInnerFactor, bool autoBaseMNK=false);
    bool PromptFlashAttentionCheckBmm2(PromptFlashAttentionTilingData& tilingData, TCubeTiling& bmm2TilingData,
        int64_t l1SizeRemain, int64_t l0CSize, uint32_t sOuterFactor, uint32_t sInnerFactor, uint32_t dSplitFactor,
        bool autoBaseMNK=false);
    bool PromptFlashAttentionComputeCVDiffParams(PromptFlashAttentionTilingData& tilingData,
        int64_t l1Size, int64_t l0CSize, uint32_t& sOuterFactor, uint32_t &sInnerFactor);
    void GetPreNextTokensLeftUp(PromptFlashAttentionTilingData& tilingData, int64_t actualSeqLength, 
        int64_t actualSeqLengthKV, int64_t& preTokensLeftUp, int64_t& nextTokensLeftUp) const;
    void UpdateTilingKeyMatmulCfg(uint64_t& tilingKey) const;
    void UpdateTilingKeyMaskCfg(PromptFlashAttentionTilingData& tilingData, uint64_t& tilingKey) const;
    void UpdateTilingKeyPseCfg(uint64_t& tilingKey) const;
    void UpdateTilingKeyDSizeConst(PromptFlashAttentionTilingData &tilingData, uint64_t& tilingKey) const;
    void UpdateTilingKeyValueDSizeConst(PromptFlashAttentionTilingData &tilingData, uint64_t& tilingKey) const;
    void UpdateTilingKeySInnerConst(PromptFlashAttentionTilingData &tilingData, uint64_t& tilingKey) const;
    void UpdateTilingKeySOuterConst(PromptFlashAttentionTilingData &tilingData, uint64_t& tilingKey);
    void PromptFlashAttentionInitSoftmaxLseOutputSplit(int64_t totalSize, PromptFlashAttentionTilingData &tilingData) const;
    void UpdateTilingKeyFlag(const ContextParamsForPFATiling& contextKeyParams, uint64_t& tilingKey) const;
    bool TilingGetTilingKeyAttentionAscendC(uint64_t& tilingKey, ContextParamsForPFATiling& contextKeyParams,
        PromptFlashAttentionTilingData &tilingData);
    size_t GetPFAWorkSpaceSize(PromptFlashAttentionTilingData& tilingData);
    ge::graphStatus SetPlatMemoryInfo(ContextParamsForPFATiling& contextKeyParams);
    ge::graphStatus SetAttributeInfo(ContextParamsForPFATiling& contextKeyParams);
    ge::graphStatus CheckTensorInvalid(const ContextParamsForPFATiling& contextKeyParams) const;
    ge::graphStatus CheckRopeInvalid(const ContextParamsForPFATiling& contextKeyParams) const;
    ge::graphStatus CheckSingleAttribute(ContextParamsForPFATiling& contextKeyParams, PFAShapeInfo& queryShapeInfo, 
        PFAShapeInfo& keyShapeInfo, PFAShapeInfo& valueShapeInfo, PFAShapeInfo& queryRopeShapeInfo, PromptFlashAttentionTilingData& tilingData);
    ge::graphStatus CheckCrossoverAttribute(ContextParamsForPFATiling& contextKeyParams, PFAShapeInfo& queryShapeInfo, 
        std::vector<int64_t>& actualSeqLengths, std::vector<int64_t>& actualSeqLengthsKV,
        PromptFlashAttentionTilingData& tilingData);
    ge::graphStatus AdjustTilingData(ContextParamsForPFATiling& contextKeyParams,
        PromptFlashAttentionTilingData& tilingData, const PFAShapeInfo& queryShapeInfo);
    ge::graphStatus ComputeTilingData(ContextParamsForPFATiling& contextKeyParams, std::vector<int64_t>& actualSeqLengths,
        std::vector<int64_t>& actualSeqLengthsKV, PromptFlashAttentionTilingData& tilingData);
    ge::graphStatus ComputeTilingKey(uint64_t& tilingKey, ContextParamsForPFATiling& contextKeyParams,
        uint32_t& numBlocksToBeSet, PromptFlashAttentionTilingData& tilingData);
    void SetAttenMaskCompressMode();
    void SetLayoutType();
    void PFATilingDataconvert(PromptFlashAttentionTilingData& tilingData);
    void SetMultiCoreParamsRegbase(int64_t totalSize, int64_t actualUsedCoreNum);
    bool IsFlashDecode() const;
    ge::graphStatus SplitBNS(PromptFlashAttentionTilingData& tilingData, uint64_t bng);
    bool CheckAlibiPseShiftTypeAndShape(ContextParamsForPFATiling& contextKeyParams, uint32_t n);
    ge::graphStatus SetQKVStartIdx(ContextParamsForPFATiling& contextKeyParams);
    bool CheckAlibiPseCrossover(ContextParamsForPFATiling& contextKeyParams);
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
    uint64_t gSize = 1;
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
    size_t defaultSysWorkspaceSize = 0;
    matmul_tiling::PlatformInfo ascendPlatformInfo;

    bool faRunFlag_ = true;
    uint8_t attenMaskShapeType = 0; // 0: (B,N2,G,S1,S2), 1: (B,1,1,S1,S2), 2: (1,1,1,S1,S2)
    uint8_t sparseType = 0;
    int64_t pseType = 0;
    FlashAttentionScoreSimplifiedTilingData faTilingAdapter;
};
} // namespace arch38
} // namespace optiling

#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_PROMPTFLASHATTENTION_ARCH38_H_
