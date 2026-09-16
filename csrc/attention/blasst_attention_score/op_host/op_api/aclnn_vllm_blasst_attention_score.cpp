/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <string.h>
#include "graph/types.h"
#include "aclnn_vllm_blasst_attention_score.h"

namespace {
typedef struct {
    uint32_t id;
    const char *funcName;
    bool hasReg;
} NnopbaseDfxId;
typedef struct {
    ge::DataType dtype;
    ge::Format format;
} TensorDesc;
typedef struct {
    TensorDesc *inputsDesc;
    size_t inputsNum;
    TensorDesc *outputsDesc;
    size_t outputsNum;
} SupportInfo;
typedef struct {
    SupportInfo *supportInfo;
    size_t num;
} OpSocSupportInfo;
typedef struct {
    OpSocSupportInfo *socSupportInfo;
    size_t num;
} OpSupportList;
enum SocType {
    SOC_VERSION_ASCEND910A = 1,
    SOC_VERSION_ASCEND910B = 2,
    SOC_VERSION_ASCEND910_93 = 3,
    SOC_VERSION_ASCEND950 = 4,
    SOC_VERSION_ASCEND310P = 5,
    SOC_VERSION_ASCEND310B = 6,
    SOC_VERSION_BS9SX1A = 7,
    SOC_VERSION_ASCEND610Lite = 8,
    SOC_VERSION_MC61AM21A = 10, // 9 is deprecated
    SOC_VERSION_MC62CM12A = 11,
    SOC_VERSION_BS9SX2A = 12,
    SOC_VERSION_ASCEND910_96 = 13,
    SOC_VERSION_KIRINX90 = 14,
    SOC_VERSION_KIRIN9030 = 15
};
enum NnopbaseAttrDtype {
    kNnopbaseBool = 0U,
    kNnopbaseFloat,
    kNnopbaseInt,
    kNnopbaseString,
    kNnopbaseAttrEnd
};
uint32_t socSupportList[] = {SOC_VERSION_ASCEND910_93};
uint32_t socSupportListLen = 1;

TensorDesc inputDesc0_0[8] =
    {{ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_INT8, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_INT32, ge::FORMAT_ND}};
TensorDesc inputDesc0_1[8] =
    {{ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_INT8, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_INT32, ge::FORMAT_ND}};
TensorDesc outputDesc0_0[3] =
    {{ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_INT32, ge::FORMAT_ND}};
TensorDesc outputDesc0_1[3] =
    {{ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_INT32, ge::FORMAT_ND}};
SupportInfo list0_0 = {inputDesc0_0, 8, outputDesc0_0, 3};
SupportInfo list0_1 = {inputDesc0_1, 8, outputDesc0_1, 3};
SupportInfo supportInfo0[2] = {list0_0, list0_1};
OpSocSupportInfo socSupportInfo0 = {supportInfo0, 2};

OpSocSupportInfo opSocSupportList[1] = {socSupportInfo0};
OpSupportList supportList = {opSocSupportList, 1};

[[maybe_unused]] uint32_t NNOPBASE_VllmBlasstAttentionScore = 0U;
} // namespace

extern void NnopbaseOpLogE(const aclnnStatus code, const char *const expr);

#ifdef __cplusplus
extern "C" {
#endif

extern aclnnStatus NnopbaseCreateExecutorSpace(void **space);
extern void *NnopbaseGetExecutor(void *space, const char *opType, char *inputsDesc, uint32_t inputNum,
                                 char *outputsDesc, uint32_t outputNum, char *attrsDesc, uint32_t attrsNum);
extern aclnnStatus NnopbaseAddInput(void *executor, const aclTensor *tensor, const uint32_t index);
extern aclnnStatus NnopbaseAddOutput(void *executor, const aclTensor *tensor, const uint32_t index);
extern aclnnStatus NnopbaseAddAttrWithDtype(void *executor, void *attrAddr, size_t attrLen, const size_t index,
                                            const NnopbaseAttrDtype dtype);
extern aclnnStatus NnopbaseAddIntArrayAttr(void *executor, const aclIntArray *array, const size_t index);
extern uint64_t NnopbaseMsprofSysTime();
extern aclnnStatus NnopbaseAddTilingId(void *executor, NnopbaseDfxId *tilingId);
extern void NnopbaseReportApiInfo(const uint64_t beginTime, NnopbaseDfxId &dfxId);
extern aclnnStatus NnopbaseRunForWorkspace(void *executor, uint64_t *workspaceLen);
extern aclnnStatus NnopbaseRunWithWorkspace(void *executor, aclrtStream stream, void *workspace,
                                            uint64_t workspaceSize);
extern aclnnStatus NnopbaseAddSupportList(void *executor, OpSupportList *list, uint32_t *socSupportList,
                                          size_t socSupportListLen);
extern aclnnStatus __attribute__((weak)) NnopbaseAddParamName(void *executor, const uint32_t index, const char *name,
                                                              const bool isInput);
extern aclnnStatus __attribute__((weak)) NnopbaseSetFormatMatchMode(void *executor, const uint32_t mode);
extern void __attribute__((weak)) NnopbaseSetMatchArgsFlag(void *executor);
extern bool __attribute__((weak)) NnopbaseMatchArgs(void *executor, uint64_t *workspaceLen);
extern aclnnStatus NnopbaseGetUnContiguousTensors(void *executor, const aclTensorList **inTensors);
extern aclnnStatus NnopbaseSetUnContExecutor(void *executor, aclOpExecutor *inExe, const size_t inWsSize);
extern aclnnStatus NnopbaseGetUnContExecutor(void *executor, aclOpExecutor **inExe, size_t *inWsSize);
extern void *NnopbaseGetApiFunc(const char *funcName);
using AclnnContiguousGetWorkspaceSizeFunc = aclnnStatus (*)(const aclTensorList *, uint64_t *, aclOpExecutor **);
using AclnnFunc = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);

#define ACLNN_SUCCESS 0
#define ACLNN_ERR_PARAM_NULLPTR 161001
#define ACLNN_ERR_PARAM_INVALID 161002

#define NNOPBASE_ASSERT_OK_RETVAL(v)                                    \
    do {                                                                \
        const aclnnStatus _chk_stutus = (v);                            \
        if (_chk_stutus != ACLNN_SUCCESS) {                             \
            NnopbaseOpLogE(_chk_stutus, #v);                            \
            return _chk_stutus;                                         \
        }                                                               \
    } while (false)

#define NNOPBASE_ASSERT_NOTNULL_RETVAL(v)                               \
    do {                                                                \
        if ((v) == nullptr) {                                           \
            NnopbaseOpLogE(ACLNN_ERR_PARAM_NULLPTR, #v " != nullptr");  \
            return ACLNN_ERR_PARAM_NULLPTR;                             \
        }                                                               \
    } while (false)

aclnnStatus aclnnVllmBlasstAttentionScoreGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *pseShiftOptional,
    const aclTensor *attenMaskOptional,
    const aclTensor *actualSeqLengthsOptional,
    const aclTensor *actualSeqLengthsKvOptional,
    const aclTensor *blocktableOptional,
    int64_t numHeads,
    double scale,
    int64_t preTokens,
    int64_t nextTokens,
    char *inputLayoutOptional,
    int64_t numKeyValueHeads,
    int64_t sparseMode,
    int64_t innerPrecise,
    int64_t blockSize,
    int64_t antiquantMode,
    double sparseLambda,
    bool softmaxLseFlag,
    const aclIntArray *actualSeqLengthsQHostOptional,
    const aclIntArray *actualSeqLengthsKvHostOptional,
    bool sparseStatsFlag,
    bool flashDecode,
    const aclTensor *attentionOutOut,
    const aclTensor *softmaxLseOutOptional,
    const aclTensor *sparseStatsOutOptional,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    uint64_t timeStamp = NnopbaseMsprofSysTime();
    static NnopbaseDfxId dfxId = {0x60000, __func__, false};
    static NnopbaseDfxId tilingId = {0x60000, "aclnnVllmBlasstAttentionScoreTiling", false};
    void *nnopExecutor;
    static void *executorSpace = NULL;
    const char *opType = "VllmBlasstAttentionScore";
    char inputDesc[] = {1, 1, 1, 0, 0, 0, 0, 0};
    char outputDesc[] = {1, 0, 0};
    char attrDesc[] = {1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    NNOPBASE_ASSERT_NOTNULL_RETVAL(query);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(key);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(value);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(attentionOutOut);

    if (!executorSpace) {
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseCreateExecutorSpace(&executorSpace));
    }
    nnopExecutor = NnopbaseGetExecutor(executorSpace, opType, inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc,
                                       sizeof(outputDesc) / sizeof(char), attrDesc, sizeof(attrDesc) / sizeof(char));
    NNOPBASE_ASSERT_NOTNULL_RETVAL(nnopExecutor);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(executor);
    *executor = reinterpret_cast<aclOpExecutor *>(nnopExecutor);
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddTilingId(*executor, &tilingId));
    if (NnopbaseSetMatchArgsFlag != NULL) {
        NnopbaseSetMatchArgsFlag(*executor);
    }
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, query, 0));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, key, 1));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, value, 2));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, pseShiftOptional, 3));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, attenMaskOptional, 4));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, actualSeqLengthsOptional, 5));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, actualSeqLengthsKvOptional, 6));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, blocktableOptional, 7));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void *>(&numHeads), sizeof(int64_t), 0,
                                                       kNnopbaseInt));
    float tmp1 = static_cast<float>(scale);
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void *>(&tmp1), sizeof(float), 1,
                                                       kNnopbaseFloat));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void *>(&preTokens), sizeof(int64_t), 2,
                                                       kNnopbaseInt));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void *>(&nextTokens), sizeof(int64_t), 3,
                                                       kNnopbaseInt));
    if (inputLayoutOptional) {
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void *>(inputLayoutOptional),
                                                           strlen(inputLayoutOptional) + 1, 4, kNnopbaseString));
    } else {
        static char *inputLayoutOptionalDef = "TND";
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void *>(inputLayoutOptionalDef),
                                                           strlen(inputLayoutOptionalDef) + 1, 4, kNnopbaseString));
    }
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void *>(&numKeyValueHeads),
                                                       sizeof(int64_t), 5, kNnopbaseInt));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void *>(&sparseMode), sizeof(int64_t), 6,
                                                       kNnopbaseInt));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void *>(&innerPrecise), sizeof(int64_t),
                                                       7, kNnopbaseInt));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void *>(&blockSize), sizeof(int64_t), 8,
                                                       kNnopbaseInt));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void *>(&antiquantMode), sizeof(int64_t),
                                                       9, kNnopbaseInt));
    float tmp10 = static_cast<float>(sparseLambda);
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void *>(&tmp10), sizeof(float), 10,
                                                       kNnopbaseFloat));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void *>(&softmaxLseFlag), sizeof(bool),
                                                       11, kNnopbaseBool));
    // actual_seq_lengths_q_host / actual_seq_lengths_kv_host）：
    if (actualSeqLengthsQHostOptional != nullptr) {
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddIntArrayAttr(*executor, actualSeqLengthsQHostOptional, 12));
    }
    if (actualSeqLengthsKvHostOptional != nullptr) {
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddIntArrayAttr(*executor, actualSeqLengthsKvHostOptional, 13));
    }
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void *>(&sparseStatsFlag), sizeof(bool),
                                                       14, kNnopbaseBool));
    // Behavior switch from vllm_ascend BlasstConfig (former getenv kill-switch VLLM_FIA_FD)
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void *>(&flashDecode), sizeof(bool),
                                                       15, kNnopbaseBool));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddOutput(*executor, attentionOutOut, 0));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddOutput(*executor, softmaxLseOutOptional, 1));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddOutput(*executor, sparseStatsOutOptional, 2));
    if (NnopbaseMatchArgs != NULL) {
        if (NnopbaseMatchArgs(*executor, workspaceSize)) {
            NnopbaseReportApiInfo(timeStamp, dfxId);
            return ACLNN_SUCCESS;
        }
    }
    if (NnopbaseAddParamName != NULL) {
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 0, "query", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 1, "key", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 2, "value", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 3, "pseShiftOptional", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 4, "attenMaskOptional", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 5, "actualSeqLengthsOptional", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 6, "actualSeqLengthsKvOptional", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 7, "blocktableOptional", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 0, "attentionOutOut", false));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 1, "softmaxLseOutOptional", false));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 2, "sparseStatsOutOptional", false));
    }
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddSupportList(*executor, &supportList, socSupportList, socSupportListLen));

    const aclTensorList *inUnContTensors = nullptr;
    NnopbaseGetUnContiguousTensors(*executor, &inUnContTensors);
    aclOpExecutor *aclInExecutor = nullptr;
    uint64_t inContWorkspaceSize = 0U;
    if (inUnContTensors != nullptr) {
        static AclnnContiguousGetWorkspaceSizeFunc aclnnContiguousGetWorkspaceSize =
            (AclnnContiguousGetWorkspaceSizeFunc)NnopbaseGetApiFunc("aclnnContiguousGetWorkspaceSize");
        NNOPBASE_ASSERT_NOTNULL_RETVAL(aclnnContiguousGetWorkspaceSize);
        NNOPBASE_ASSERT_OK_RETVAL(aclnnContiguousGetWorkspaceSize(inUnContTensors, &inContWorkspaceSize,
                                                                  &aclInExecutor));
    }
    NnopbaseSetUnContExecutor(*executor, aclInExecutor, inContWorkspaceSize);

    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseRunForWorkspace(*executor, workspaceSize));
    *workspaceSize += inContWorkspaceSize;
    NnopbaseReportApiInfo(timeStamp, dfxId);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnVllmBlasstAttentionScore(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
{
    uint64_t timeStamp = NnopbaseMsprofSysTime();
    static NnopbaseDfxId dfxId = {0x60000, __func__, false};
    aclOpExecutor *aclInExecutor = nullptr;
    uint64_t inContWorkspaceSize = 0U;
    NnopbaseGetUnContExecutor(executor, &aclInExecutor, &inContWorkspaceSize);
    if (workspaceSize < inContWorkspaceSize) {
        NnopbaseOpLogE(ACLNN_ERR_PARAM_INVALID, "input workspaceSize must be larger than contiguous size!");
        return ACLNN_ERR_PARAM_INVALID;
    }
    workspaceSize -= inContWorkspaceSize;
    void *inWorkspace = (char *)workspace + workspaceSize;
    if (aclInExecutor != nullptr) {
        static AclnnFunc aclnnContiguous = (AclnnFunc)NnopbaseGetApiFunc("aclnnContiguous");
        NNOPBASE_ASSERT_NOTNULL_RETVAL(aclnnContiguous);
        NNOPBASE_ASSERT_OK_RETVAL(aclnnContiguous(inWorkspace, inContWorkspaceSize, aclInExecutor, stream));
    }
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseRunWithWorkspace(executor, stream, workspace, workspaceSize));
    NnopbaseReportApiInfo(timeStamp, dfxId);
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
