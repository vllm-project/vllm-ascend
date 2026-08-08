/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "prompt_flash_attention_inner.h"
#include "prompt_flash_attention.h"
#include "opdev/common_types.h"
#include "opdev/fast_vector.h"
#include "opdev/op_errno.h"
#include "opdev/op_executor.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/pad.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/slice.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {
static const uint64_t PAD_BASIC_BLOCK = 32;
static const uint64_t PAD_BASIC_BLOCK_256 = 256;
static const uint64_t HALF_PAD_BASIC_BLOCK = 16;
static const uint64_t MAX_STRIDE_S1 = 65535;
static const uint64_t DIM_NUM_4 = 4;
static const uint64_t DIM_NUM_3 = 3;
static const uint64_t DIM_NUM_2 = 2;
static const uint64_t INDEX_2 = 2;
static const uint64_t INDEX_3 = 3;
static const uint64_t LAYOUT_STR_LENGTH_0 = 0;
static const uint64_t LAYOUT_STR_LENGTH_1 = 1;
static const uint64_t LAYOUT_STR_LENGTH_2 = 2;
static const uint64_t LAYOUT_STR_LENGTH_3 = 3;
static const uint64_t LAYOUT_STR_LENGTH_4 = 4;
static const uint64_t LAYOUT_STR_LENGTH_9 = 9;
static const uint64_t NZ_N0_16 = 16;


struct AxesInfo {
    int64_t b;
    int64_t n1;
    int64_t n2;
    int64_t s1;
    int64_t s2;
    int64_t d;
    int64_t dV;
    int64_t t1;
    int64_t t2;
};

enum class InputLayout { SH, BSH, NSD, BNSD, BSND, BNSD_BSND, TND, NTD, NONE, };

static const std::unordered_map<DataType, string> StrDataTypePfa = {
    {DataType::DT_FLOAT, "DT_FLOAT"},
    {DataType::DT_FLOAT16, "DT_FLOAT16"},
    {DataType::DT_INT8, "DT_INT8"},
    {DataType::DT_INT16, "DT_INT16"},
    {DataType::DT_UINT16, "DT_UINT16"},
    {DataType::DT_UINT8, "DT_UINT8"},
    {DataType::DT_INT32, "DT_INT32"},
    {DataType::DT_INT64, "DT_INT64"},
    {DataType::DT_UINT32, "DT_UINT32"},
    {DataType::DT_UINT64, "DT_UINT64"},
    {DataType::DT_BOOL, "DT_BOOL"},
    {DataType::DT_DOUBLE, "DT_DOUBLE"},
    {DataType::DT_STRING, "DT_STRING"},
    {DataType::DT_DUAL_SUB_INT8, "DT_DUAL_SUB_INT8"},
    {DataType::DT_DUAL_SUB_UINT8, "DT_DUAL_SUB_UINT8V"},
    {DataType::DT_COMPLEX64, "DT_COMPLEX64"},
    {DataType::DT_COMPLEX128, "DT_COMPLEX128"},
    {DataType::DT_QINT8, "DT_QINT8"},
    {DataType::DT_QINT16, "DT_QINT16"},
    {DataType::DT_QINT32, "DT_QINT32"},
    {DataType::DT_QUINT8, "DT_QUINT8"},
    {DataType::DT_QUINT16, "DT_QUINT16"},
    {DataType::DT_RESOURCE, "DT_RESOURCE"},
    {DataType::DT_STRING_REF, "DT_STRING_REF"},
    {DataType::DT_DUAL, "DT_DUAL"},
    {DataType::DT_VARIANT, "DT_VARIANT"},
    {DataType::DT_BF16, "DT_BF16"},
    {DataType::DT_UNDEFINED, "DT_UNDEFINED"},
};

static DataType ValidPfaAclDataType(DataType type) {
    return (StrDataTypePfa.find(type) == StrDataTypePfa.end()) ? DataType::DT_UNDEFINED : type;
}

struct FaShapeInfo {
    AxesInfo axes;

    InputLayout inputLayout;
    string l0InputLayoutStr;

    uint64_t dimNum = 0;
    uint64_t padNum = 0;
    uint64_t padNumV = 0;
    uint64_t basicBlock = HALF_PAD_BASIC_BLOCK;

    FVector<int64_t, DIM_NUM_4> perm_in;
    FVector<int64_t, DIM_NUM_4> perm_out;
    FVector<int64_t, DIM_NUM_4> reshapedQueryShape;
    FVector<int64_t, DIM_NUM_4> reshapedKeyShape;
    FVector<int64_t, DIM_NUM_4> reshapedValueShape;

    bool needPad = false;
    bool needTranspose = false;
    bool needReshape = false;
};

static aclnnStatus CheckDimsAndLayout(const aclTensor *query, const aclTensor *key, const aclTensor *value, const char *inputLayout) {
    auto qDimNum = query->GetViewShape().GetDimNum();
    auto kDimNum = key->GetViewShape().GetDimNum();
    auto vDimNum = value->GetViewShape().GetDimNum();
    if (qDimNum != kDimNum || qDimNum != vDimNum) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "the layout of q and k v must be same, but got q dim:%lu k dim:%lu v dim:%lu", qDimNum, kDimNum, vDimNum);
        return ACLNN_ERR_PARAM_INVALID;
    } else if ((qDimNum != DIM_NUM_4) && strnlen(inputLayout, LAYOUT_STR_LENGTH_4) >= LAYOUT_STR_LENGTH_4 && (inputLayout[LAYOUT_STR_LENGTH_0] == 'B' && inputLayout[LAYOUT_STR_LENGTH_1] == 'N' &&
        inputLayout[LAYOUT_STR_LENGTH_2] == 'S' && inputLayout[LAYOUT_STR_LENGTH_3] == 'D')) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "layout is BNSD, input shape dim should be 4, but got %lu", qDimNum);
        return ACLNN_ERR_PARAM_INVALID;
    } else if ((qDimNum != DIM_NUM_4) && strnlen(inputLayout, LAYOUT_STR_LENGTH_4) >= LAYOUT_STR_LENGTH_4 && (inputLayout[LAYOUT_STR_LENGTH_0] == 'B' && inputLayout[LAYOUT_STR_LENGTH_1] == 'S' &&
        inputLayout[LAYOUT_STR_LENGTH_2] == 'N' && inputLayout[LAYOUT_STR_LENGTH_3] == 'D')) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "layout is BSND, input shape dim should be 4, but got %lu", qDimNum);
        return ACLNN_ERR_PARAM_INVALID;
    } else if ((qDimNum != DIM_NUM_3) && strnlen(inputLayout, LAYOUT_STR_LENGTH_4) >= LAYOUT_STR_LENGTH_3 && (inputLayout[LAYOUT_STR_LENGTH_0] == 'N' && inputLayout[LAYOUT_STR_LENGTH_1] == 'S' &&
               inputLayout[LAYOUT_STR_LENGTH_2] == 'D')) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "layout is NSD, input shape dim should be 3, but got %lu", qDimNum);
        return ACLNN_ERR_PARAM_INVALID;
    } else if ((qDimNum != DIM_NUM_3) && strnlen(inputLayout, LAYOUT_STR_LENGTH_4) >= LAYOUT_STR_LENGTH_3 && (inputLayout[LAYOUT_STR_LENGTH_0] == 'T' && inputLayout[LAYOUT_STR_LENGTH_1] == 'N' &&
                inputLayout[LAYOUT_STR_LENGTH_2] == 'D')) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "layout is TND, input shape dim should be 3, but got %lu", qDimNum);
        return ACLNN_ERR_PARAM_INVALID;
    } else if ((qDimNum != DIM_NUM_3) && strnlen(inputLayout, LAYOUT_STR_LENGTH_4) >= LAYOUT_STR_LENGTH_3 && (inputLayout[LAYOUT_STR_LENGTH_0] == 'N' && inputLayout[LAYOUT_STR_LENGTH_1] == 'T' &&
                inputLayout[LAYOUT_STR_LENGTH_2] == 'D')) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "layout is NTD, input shape dim should be 3, but got %lu", qDimNum);
        return ACLNN_ERR_PARAM_INVALID;
    } else if ((qDimNum != DIM_NUM_2) && strnlen(inputLayout, LAYOUT_STR_LENGTH_4) >= LAYOUT_STR_LENGTH_2 && (inputLayout[LAYOUT_STR_LENGTH_0] == 'S' && inputLayout[LAYOUT_STR_LENGTH_1] == 'H')) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "layout is SH, input shape dim should be 2, but got %lu", qDimNum);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}
static aclnnStatus AnalysisAxis(const aclTensor *query, const aclTensor *key, const aclTensor *value,
                                const char *inputLayout, int64_t headNum, int64_t headNumKV, FaShapeInfo &shapeInfo)
{
    Shape qShape = query->GetViewShape();
    Shape kShape = key->GetViewShape();
    Shape vShape = value->GetViewShape();
    shapeInfo.dimNum = qShape.GetDimNum();

    // Record the length of the axis b, n2, g, s1, s2, d
    // H1 = N1*D, H2 = N2*D
    // N1 = g*N2
    shapeInfo.axes.n1 = headNum;

    if (strlen(inputLayout) != 0) {
        shapeInfo.inputLayout = InputLayout::NONE;
        shapeInfo.l0InputLayoutStr = "NONE";
    } else {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "inputLayout should not be null!");
        return ACLNN_ERR_PARAM_INVALID;
    }

    // query: (B*S1, N1*D)
    // key/value: (B*S2, N2*D)
    if (shapeInfo.dimNum == DIM_NUM_2 && strnlen(inputLayout, LAYOUT_STR_LENGTH_4) >= LAYOUT_STR_LENGTH_2 && inputLayout[0] == 'S' && inputLayout[1] == 'H') {
        uint64_t dSize = qShape[1] / headNum;
        uint64_t dSizeV = headNumKV == 0 ? vShape[1] / headNum : vShape[1] / headNumKV;
        if (dSize == 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input query shape is (S[%ld], H[%ld]), input num_head N is %ld, "
                    "corresponding headsize D = H/N = %lu, is invalid.",
                    qShape[0], qShape[1], headNum, dSize); // 0:S, 1:H
            return ACLNN_ERR_PARAM_INVALID;
        }

        shapeInfo.axes.b = 1;
        shapeInfo.axes.n2 = kShape[1] / dSize;
        shapeInfo.axes.s1 = qShape[0];
        shapeInfo.axes.s2 = kShape[0];
        shapeInfo.axes.d = dSize;
        shapeInfo.axes.dV = dSizeV;
        shapeInfo.inputLayout = InputLayout::SH;
        shapeInfo.l0InputLayoutStr = "SH";
    }

    // query: (B,S1,N1*D)
    // key/value: (B,S2,N2*D)
    if (strnlen(inputLayout, LAYOUT_STR_LENGTH_4) >= LAYOUT_STR_LENGTH_3 && inputLayout[0] == 'B' && inputLayout[1] == 'S' && inputLayout[2] == 'H') {
        uint64_t dSize = 0;
        uint64_t dSizeV = 0;
        if (shapeInfo.dimNum == DIM_NUM_4 && qShape[INDEX_3] == NZ_N0_16) {
            // NZ 格式
            dSize = qShape[1] * NZ_N0_16 / headNum;
            dSizeV = headNumKV == 0 ? vShape[1] * NZ_N0_16 / headNum :  vShape[1] * NZ_N0_16 / headNumKV;
        } else {
            dSize = qShape[INDEX_2] / headNum;
            dSizeV = headNumKV == 0 ? vShape[INDEX_2] / headNum : vShape[INDEX_2] / headNumKV;
        }

        if (dSize == 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input query shape is (B[%ld], S[%ld], H[%ld]), input num_head N is %ld, "
                    "corresponding headsize D = H/N = %lu, is invalid.",
                    qShape[0], qShape[1], qShape[2], headNum, dSize); // 0:B, 1:S, 2:H
            return ACLNN_ERR_PARAM_INVALID;
        }

        shapeInfo.axes.b = qShape[0];
        if (shapeInfo.dimNum == DIM_NUM_4 && qShape[INDEX_3] == NZ_N0_16) {
            shapeInfo.axes.n2 = kShape[1] * NZ_N0_16 / dSize;
            shapeInfo.axes.s1 = qShape[INDEX_2];
            shapeInfo.axes.s2 = kShape[INDEX_2];
        } else {
            shapeInfo.axes.n2 = kShape[INDEX_2] / dSize;
            shapeInfo.axes.s1 = qShape[1];
            shapeInfo.axes.s2 = kShape[1];
        }
        shapeInfo.axes.n2 = kShape[INDEX_2] / dSize;
        shapeInfo.axes.s1 = qShape[1];
        shapeInfo.axes.s2 = kShape[1];
        shapeInfo.axes.d = dSize;
        shapeInfo.axes.dV = dSizeV;
        shapeInfo.inputLayout = InputLayout::BSH;
        shapeInfo.l0InputLayoutStr = "BSH";
    }

    // TND
    if (shapeInfo.dimNum == DIM_NUM_3 && strnlen(inputLayout, LAYOUT_STR_LENGTH_4) >= LAYOUT_STR_LENGTH_3 && inputLayout[0] == 'T' && inputLayout[1] == 'N' && inputLayout[2] == 'D') {
        shapeInfo.axes.b = 1;
        shapeInfo.axes.n2 = kShape[1];
        shapeInfo.axes.s1 = 1;
        shapeInfo.axes.s2 = 1;
        shapeInfo.axes.t1 = qShape[0];
        shapeInfo.axes.t2 = kShape[0];
        shapeInfo.axes.d = qShape[INDEX_2];
        shapeInfo.axes.dV = vShape[INDEX_2];
        shapeInfo.inputLayout = InputLayout::TND;
        shapeInfo.l0InputLayoutStr = "TND";
    }

    // NTD
    if (shapeInfo.dimNum == DIM_NUM_3 && strnlen(inputLayout, LAYOUT_STR_LENGTH_4) >= LAYOUT_STR_LENGTH_3 && inputLayout[0] == 'N' && inputLayout[1] == 'T' && inputLayout[2] == 'D') {
        shapeInfo.axes.b = 1;
        shapeInfo.axes.n2 = kShape[0];
        shapeInfo.axes.s1 = 1;
        shapeInfo.axes.s2 = 1;
        shapeInfo.axes.t1 = qShape[1];
        shapeInfo.axes.t2 = kShape[1];
        shapeInfo.axes.d = qShape[INDEX_2];
        shapeInfo.axes.dV = vShape[INDEX_2];
        shapeInfo.inputLayout = InputLayout::NTD;
        shapeInfo.l0InputLayoutStr = "NTD";
    }

    // query: (B,S1,N1,D)
    // key/value: (B,S2,N2,D)
    if (shapeInfo.dimNum == DIM_NUM_4 && strnlen(inputLayout, LAYOUT_STR_LENGTH_4) >= LAYOUT_STR_LENGTH_4 && inputLayout[0] == 'B' && inputLayout[1] == 'S' && inputLayout[2] == 'N' &&
        inputLayout[INDEX_3] == 'D') {
        shapeInfo.axes.b = qShape[0];
        shapeInfo.axes.n2 = kShape[INDEX_2];
        shapeInfo.axes.s1 = qShape[1];
        shapeInfo.axes.s2 = kShape[1];
        shapeInfo.axes.d = qShape[INDEX_3];
        shapeInfo.axes.dV = vShape[INDEX_3];
        shapeInfo.inputLayout = InputLayout::BSND;
        shapeInfo.l0InputLayoutStr = "BSND";
    }

    // query: (B*N1,S1,D)
    // key/value: (B*N2,S2,D)
    if (shapeInfo.dimNum == DIM_NUM_3 && strnlen(inputLayout, LAYOUT_STR_LENGTH_4) >= LAYOUT_STR_LENGTH_3 && inputLayout[0] == 'N' && inputLayout[1] == 'S' && inputLayout[2] == 'D') {
        shapeInfo.axes.b = 1;
        shapeInfo.axes.n2 = kShape[0];
        shapeInfo.axes.s1 = qShape[1];
        shapeInfo.axes.s2 = kShape[1];
        shapeInfo.axes.d = qShape[INDEX_2];
        shapeInfo.axes.dV = vShape[INDEX_2];
        shapeInfo.inputLayout = InputLayout::NSD;
        shapeInfo.l0InputLayoutStr = "NSD";
    }

    // query: (B,N1,S1,D)
    // key/value: (B,N2,S2,D)
    if (shapeInfo.dimNum == DIM_NUM_4 && strnlen(inputLayout, LAYOUT_STR_LENGTH_4) >= LAYOUT_STR_LENGTH_4 && inputLayout[0] == 'B' && inputLayout[1] == 'N' && inputLayout[2] == 'S' &&
        inputLayout[INDEX_3] == 'D') {
        shapeInfo.axes.b = qShape[0];
        shapeInfo.axes.n2 = kShape[1];
        shapeInfo.axes.s1 = qShape[INDEX_2];
        shapeInfo.axes.s2 = kShape[INDEX_2];
        shapeInfo.axes.d = qShape[INDEX_3];
        shapeInfo.axes.dV = vShape[INDEX_3];
        if (strnlen(inputLayout, LAYOUT_STR_LENGTH_9) == LAYOUT_STR_LENGTH_9 && inputLayout[4] == '_' && inputLayout[5] == 'B' && inputLayout[6] == 'S' && inputLayout[7] == 'N' &&   // 4,5,6,7 : The 5th, 6th, 7th and 8th characters
            inputLayout[8] == 'D') {  // 8 : The ninth character
            shapeInfo.inputLayout = InputLayout::BNSD_BSND;
            shapeInfo.l0InputLayoutStr = "BNSD_BSND";
        } else {
            shapeInfo.inputLayout = InputLayout::BNSD;
            shapeInfo.l0InputLayoutStr = "BNSD";
        }
    }

    if (shapeInfo.inputLayout == InputLayout::NONE) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "PFA inputLayout only supports BSH, BNSD, BSND, BNSD_BSND, TND, NTD, current layout: %s", inputLayout);
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (shapeInfo.axes.d > 512 || shapeInfo.axes.dV > 512) { // 512: D is limited at 512
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "D should <= 512, but the D of query is %ld and the D of value is %ld.", shapeInfo.axes.d, shapeInfo.axes.dV);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (inputLayout[0] == 'T' && inputLayout[1] == 'N' && inputLayout[2] == 'D') {
        OP_LOGD("Analysis axis success. "
            "The axis result: [t1]: %ld, [t2]: %ld, [n1]: %ld, [n2]: %ld, [d]: %ld, [dV]: %ld",
            shapeInfo.axes.t1, shapeInfo.axes.t2, shapeInfo.axes.n1, shapeInfo.axes.n2, shapeInfo.axes.d,
            shapeInfo.axes.dV);
    } else {
        OP_LOGD("Analysis axis success. "
            "The axis result: [B]: %ld, [n1]: %ld, [n2]: %ld, [s1]: %ld, [s2]: %ld, [d]: %ld, [dV]: %ld",
            shapeInfo.axes.b, shapeInfo.axes.n1, shapeInfo.axes.n2, shapeInfo.axes.s1, shapeInfo.axes.s2,
            shapeInfo.axes.d, shapeInfo.axes.dV);
    }
    return ACLNN_SUCCESS;
}

static void SetShapeInfoForSH(FaShapeInfo &shapeInfo) {
    if (shapeInfo.needPad) {
        shapeInfo.needReshape = true;
        shapeInfo.reshapedQueryShape.assign({shapeInfo.axes.b, shapeInfo.axes.s1, shapeInfo.axes.n1, shapeInfo.axes.d});
        shapeInfo.reshapedKeyShape.assign({shapeInfo.axes.b, shapeInfo.axes.s2, shapeInfo.axes.n2, shapeInfo.axes.d});
        shapeInfo.reshapedValueShape.assign({shapeInfo.axes.b, shapeInfo.axes.s2, shapeInfo.axes.n2, shapeInfo.axes.dV});
    }
}

static void SetShapeInfoForNSD(FaShapeInfo &shapeInfo) {
    if (shapeInfo.needPad) {
        shapeInfo.needReshape = true;
        shapeInfo.reshapedQueryShape.assign({shapeInfo.axes.b, shapeInfo.axes.n1, shapeInfo.axes.s1, shapeInfo.axes.d});
        shapeInfo.reshapedKeyShape.assign({shapeInfo.axes.b, shapeInfo.axes.n2, shapeInfo.axes.s2, shapeInfo.axes.d});
        shapeInfo.reshapedValueShape.assign({shapeInfo.axes.b, shapeInfo.axes.n2, shapeInfo.axes.s2, shapeInfo.axes.dV});
    }
}

static aclnnStatus AnalysisInputShapeInfo(const aclTensor *query, const aclTensor *key, const aclTensor *value,
                                          char *inputLayout, int64_t headNum, int64_t headNumKV, FaShapeInfo &shapeInfo,
                                          const aclTensor *attentionOut)
{
    if (headNum <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "head_num must > 0, but got %ld", headNum);
        return ACLNN_ERR_PARAM_INVALID;
    }
    CHECK_RET(CheckDimsAndLayout(query, key, value, inputLayout) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(AnalysisAxis(query, key, value, inputLayout, headNum, headNumKV, shapeInfo) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    if (shapeInfo.axes.n2 == 0 || shapeInfo.axes.s2 == 0 || shapeInfo.axes.d == 0 || shapeInfo.axes.dV == 0) {
        return ACLNN_SUCCESS;
    }

    // When D padding is calculated based on dtype, the input q, k, v, and output format need to be considered. As long as there is the int8 type, the elements are aligned by 32.
    DataType queryDataType = query->GetDataType();
    DataType keyDataType = key->GetDataType();
    DataType valueDataType = value->GetDataType();
    DataType outputDataType = attentionOut->GetDataType();
    if (shapeInfo.axes.d > shapeInfo.axes.dV) {
        if (outputDataType == DataType::DT_INT8) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Please check Input Tensor shape, when the D of query(%ld) is greater than D of value(%ld), "
                    "the output type can't be int8.", shapeInfo.axes.d, shapeInfo.axes.dV);
            return ACLNN_ERR_PARAM_INVALID;
        }
        if ((shapeInfo.inputLayout != InputLayout::TND && shapeInfo.inputLayout != InputLayout::NTD) && shapeInfo.axes.d > 128) {  // 128 : When D of query and value are different, D > 128, it should be aligned with 256, and D <= 128, it should be aligned with 32B.
            shapeInfo.basicBlock = PAD_BASIC_BLOCK_256;
        }
    } else if ((queryDataType == DataType::DT_INT8) || (keyDataType == DataType::DT_INT8) ||
              (valueDataType == DataType::DT_INT8) || (outputDataType == DataType::DT_INT8)) {
              shapeInfo.basicBlock = PAD_BASIC_BLOCK;
    }

    if (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
        shapeInfo.needPad = false;
    } else if (shapeInfo.inputLayout != InputLayout::TND && (shapeInfo.axes.d % shapeInfo.basicBlock != 0 || shapeInfo.axes.d > shapeInfo.axes.dV)) {
            shapeInfo.needPad = true;
            shapeInfo.padNum =
                (shapeInfo.axes.d + shapeInfo.basicBlock - 1) / shapeInfo.basicBlock * shapeInfo.basicBlock -
                shapeInfo.axes.d;
            shapeInfo.padNumV = shapeInfo.axes.d > shapeInfo.axes.dV ? shapeInfo.axes.d + shapeInfo.padNum - shapeInfo.axes.dV : shapeInfo.padNum;
    }

    if ((shapeInfo.inputLayout == InputLayout::BSH) ||
        (shapeInfo.inputLayout == InputLayout::SH)) {
        SetShapeInfoForSH(shapeInfo);
    } else if (shapeInfo.inputLayout == InputLayout::NSD) {
        SetShapeInfoForNSD(shapeInfo);
    }

    OP_LOGD("Analysis input success. The analysis result: [needReshape]: %d, [needPad]: %d, [padNum]: %lu, [padNumV]: %lu,"
        "[needTranspose]: %d, [basicBlock]: %lu ",
        shapeInfo.needReshape, shapeInfo.needPad, shapeInfo.padNum, shapeInfo.padNumV, shapeInfo.needTranspose, shapeInfo.basicBlock);
    return ACLNN_SUCCESS;
}

static inline const aclTensor *GeneratePaddings(int32_t dimNum, int32_t padNum, aclOpExecutor *executor)
{
    // 2 represents that 0 can be added to the front and back of each axis
    FVector<int64_t> padVec(dimNum * 2, 0);
    padVec[padVec.size() - 1] = padNum;

    auto padArray = executor->AllocIntArray(padVec.data(), padVec.size());
    if (padArray == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Try alloc padVec failed");
        return nullptr;
    }

    auto padTensor = executor->ConvertToTensor(padArray, DataType::DT_INT64);
    return padTensor;
}

static aclnnStatus ContiguousInput(const aclTensor *&query, const aclTensor *&key, const aclTensor *&value,
                                   const aclTensor *&pseShift, const aclTensor *&attenMask,
                                   aclOpExecutor *executor)
{
    query = l0op::Contiguous(query, executor);
    CHECK_RET(query != nullptr, ACLNN_ERR_INNER_NULLPTR);
    key = l0op::Contiguous(key, executor);
    CHECK_RET(key != nullptr, ACLNN_ERR_INNER_NULLPTR);
    value = l0op::Contiguous(value, executor);
    CHECK_RET(value != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (pseShift) {
        pseShift = l0op::Contiguous(pseShift, executor);
        CHECK_RET(pseShift != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (attenMask) {
        attenMask = l0op::Contiguous(attenMask, executor);
        CHECK_RET(attenMask != nullptr, ACLNN_ERR_INNER_NULLPTR);
        // Ascend950兼容mask输入float16类型，转换成uint8类型复用VF
        DataType attenMaskDataType = attenMask->GetDataType();
        if (attenMaskDataType == DataType::DT_FLOAT16) {
            attenMask = l0op::Cast(attenMask, DataType::DT_UINT8, executor);
        }
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus reShapeMiddle(const aclTensor *&query, const aclTensor *&key, const aclTensor *&value,
                                 const int64_t *queryValue, uint64_t querySize,
                                 const int64_t *keyValueValue, uint64_t keyValueSize,
                                 aclOpExecutor *executor)
{
    query = l0op::Reshape(query, executor->AllocIntArray(queryValue, querySize), executor);
    CHECK_RET(query != nullptr, ACLNN_ERR_INNER_NULLPTR);
    key = l0op::Reshape(key, executor->AllocIntArray(keyValueValue, keyValueSize), executor);
    CHECK_RET(key != nullptr, ACLNN_ERR_INNER_NULLPTR);
    value = l0op::Reshape(value, executor->AllocIntArray(keyValueValue, keyValueSize), executor);
    CHECK_RET(value != nullptr, ACLNN_ERR_INNER_NULLPTR);

    return ACLNN_SUCCESS;
}

static aclnnStatus PreprocessQKVInput(const aclTensor *&query, const aclTensor *&key, const aclTensor *&value,
                                      const aclTensor *&quantScale2, const aclTensor *&quantOffset2,
                                      const struct FaShapeInfo &shapeInfo, aclOpExecutor *executor)
{
    if (shapeInfo.needReshape) {
        query = l0op::Reshape(
            query, executor->AllocIntArray(shapeInfo.reshapedQueryShape.data(), shapeInfo.reshapedQueryShape.size()),
            executor);
        CHECK_RET(query != nullptr, ACLNN_ERR_INNER_NULLPTR);
        key = l0op::Reshape(
            key,
            executor->AllocIntArray(shapeInfo.reshapedKeyShape.data(), shapeInfo.reshapedKeyShape.size()),
            executor);
        CHECK_RET(key != nullptr, ACLNN_ERR_INNER_NULLPTR);
        value = l0op::Reshape(
            value,
            executor->AllocIntArray(shapeInfo.reshapedValueShape.data(), shapeInfo.reshapedValueShape.size()),
            executor);
        CHECK_RET(value != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    if (shapeInfo.needPad) {
        auto paddings = GeneratePaddings(DIM_NUM_4, shapeInfo.padNum, executor);
        auto paddingsV = GeneratePaddings(DIM_NUM_4, shapeInfo.padNumV, executor);

        query = l0op::Pad(query, paddings, executor);
        CHECK_RET(query != nullptr, ACLNN_ERR_INNER_NULLPTR);
        key = l0op::Pad(key, paddings, executor);
        CHECK_RET(key != nullptr, ACLNN_ERR_INNER_NULLPTR);
        value = l0op::Pad(value, paddingsV, executor);
        CHECK_RET(value != nullptr, ACLNN_ERR_INNER_NULLPTR);

        auto quant_paddings = paddings;
        if (quantScale2 != nullptr) {
            auto scale2DimNum = quantScale2->GetViewShape().GetDimNum();
            if (scale2DimNum == DIM_NUM_3) {
                quant_paddings = GeneratePaddings(DIM_NUM_3, shapeInfo.padNum, executor);
            }
            if (scale2DimNum == DIM_NUM_3 || scale2DimNum == DIM_NUM_4) {
                quantScale2 = l0op::Pad(quantScale2, quant_paddings, executor);
                CHECK_RET(quantScale2 != nullptr, ACLNN_ERR_INNER_NULLPTR);
                if (quantOffset2 != nullptr && quantOffset2->GetViewShape().GetDimNum() > 0) {
                    quantOffset2 = l0op::Pad(quantOffset2, quant_paddings, executor);
                    CHECK_RET(quantOffset2 != nullptr, ACLNN_ERR_INNER_NULLPTR);
                }
            }
        }
    }

    if (shapeInfo.inputLayout == InputLayout::BSH && shapeInfo.needPad) {
        // (B,S,N,D) -> (B,S,N*D)
        FVector<int64_t, DIM_NUM_3> queryShape{shapeInfo.axes.b, shapeInfo.axes.s1,
                                               shapeInfo.axes.n1 * (shapeInfo.axes.d + static_cast<int64_t>(shapeInfo.padNum))};
        FVector<int64_t, DIM_NUM_3> keyValueShape{shapeInfo.axes.b, shapeInfo.axes.s2,
                                                  shapeInfo.axes.n2 * (shapeInfo.axes.d + static_cast<int64_t>(shapeInfo.padNum))};

        CHECK_RET(reShapeMiddle(query, key, value, queryShape.data(), queryShape.size(),
                                keyValueShape.data(), keyValueShape.size(), executor) == ACLNN_SUCCESS,
                  ACLNN_ERR_INNER_TILING_ERROR);
    }

    if (shapeInfo.inputLayout == InputLayout::SH && shapeInfo.needPad) {
        // (B,S,N,D) -> (B*S,N*D)
        FVector<int64_t, DIM_NUM_2> queryShape{shapeInfo.axes.b * shapeInfo.axes.s1,
                                               shapeInfo.axes.n1 * (shapeInfo.axes.d + static_cast<int64_t>(shapeInfo.padNum))};
        FVector<int64_t, DIM_NUM_2> keyValueShape{shapeInfo.axes.b * shapeInfo.axes.s2,
                                                  shapeInfo.axes.n2 * (shapeInfo.axes.d + static_cast<int64_t>(shapeInfo.padNum))};

        CHECK_RET(reShapeMiddle(query, key, value, queryShape.data(), queryShape.size(),
                                keyValueShape.data(), keyValueShape.size(), executor) == ACLNN_SUCCESS,
                  ACLNN_ERR_INNER_TILING_ERROR);
    }

    if (shapeInfo.inputLayout == InputLayout::NSD && shapeInfo.needPad) {
        // (B,N,S,Dp) -> (B*N,S,Dp)
        FVector<int64_t, DIM_NUM_3> queryShape{shapeInfo.axes.b * shapeInfo.axes.n1, shapeInfo.axes.s1,
                                               (shapeInfo.axes.d + static_cast<int64_t>(shapeInfo.padNum))};
        FVector<int64_t, DIM_NUM_3> keyValueShape{shapeInfo.axes.b * shapeInfo.axes.n2, shapeInfo.axes.s2, 
                                                  (shapeInfo.axes.d + static_cast<int64_t>(shapeInfo.padNum))};

        CHECK_RET(reShapeMiddle(query, key, value, queryShape.data(), queryShape.size(),
                                keyValueShape.data(), keyValueShape.size(), executor) == ACLNN_SUCCESS,
                  ACLNN_ERR_INNER_TILING_ERROR);
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus PostProcessOutput(const aclTensor *&l0AttentionOutOut, const aclTensor *attentionOutOut,
                                     struct FaShapeInfo &shapeInfo, aclOpExecutor *executor)
{
    if (shapeInfo.needPad) {
        if ((shapeInfo.inputLayout == InputLayout::BSH) ||
            (shapeInfo.inputLayout == InputLayout::SH)) {
            // (B,S,Hp) -> (B,S,N,Dp)
            FVector<int64_t, DIM_NUM_4> paddedSBNDShape{shapeInfo.axes.b, shapeInfo.axes.s1, shapeInfo.axes.n1,
                                                shapeInfo.axes.d + static_cast<int64_t>(shapeInfo.padNum)};
            l0AttentionOutOut =
                l0op::Reshape(l0AttentionOutOut,
                            executor->AllocIntArray(paddedSBNDShape.data(), paddedSBNDShape.size()), executor);
            CHECK_RET(l0AttentionOutOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
        } else if (shapeInfo.inputLayout == InputLayout::NSD) {
            // (N,S,Dp) -> (B,N,S,Dp)
            FVector<int64_t, DIM_NUM_4> paddedSBNDShape{shapeInfo.axes.b, shapeInfo.axes.n1, shapeInfo.axes.s1,
                                                shapeInfo.axes.d + static_cast<int64_t>(shapeInfo.padNum)};
            l0AttentionOutOut =
                l0op::Reshape(l0AttentionOutOut,
                            executor->AllocIntArray(paddedSBNDShape.data(), paddedSBNDShape.size()), executor);
            CHECK_RET(l0AttentionOutOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
    }

    if (shapeInfo.needPad) {
        // (B,S,N,D)
        // (S,B,N,D)
        // (B,N,S,D)
        FVector<int64_t, DIM_NUM_4> offsetVec(DIM_NUM_4, 0);
        FVector<int64_t, MAX_DIM_NUM> sizeVec = ToShapeVector(l0AttentionOutOut->GetViewShape());
        sizeVec.back() -= shapeInfo.padNumV;

        l0AttentionOutOut = l0op::Slice(l0AttentionOutOut, executor->AllocIntArray(offsetVec.data(), offsetVec.size()),
                                        executor->AllocIntArray(sizeVec.data(), sizeVec.size()), executor);
        CHECK_RET(l0AttentionOutOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    if (shapeInfo.needReshape) {
        auto attentionOutOutShape = ToShapeVector(attentionOutOut->GetViewShape());
        l0AttentionOutOut =
            l0op::Reshape(l0AttentionOutOut,
                          executor->AllocIntArray(attentionOutOutShape.data(), attentionOutOutShape.size()), executor);
        CHECK_RET(l0AttentionOutOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    return ACLNN_SUCCESS;
}

static bool CheckNotNull(const aclTensor* query, const aclTensor* key, const aclTensor* value,
                         char *inputLayout, const aclTensor *attentionOut) {
    OP_CHECK_NULL(query, return false);
    OP_CHECK_NULL(key, return false);
    OP_CHECK_NULL(value, return false);
    OP_CHECK_NULL(attentionOut, return false);
    if (inputLayout == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "expected a value of type char but got null for argument inputLayout.");
        return false;
    }
    return true;
}

static bool CheckTensorDataType(const aclTensor* query, const aclTensor* key, const aclTensor* value,
                                const aclTensor *pseShift, const aclTensor* attenMask,
                                const aclTensor* attentionOut) {
    const DataType queryDataType = query->GetDataType();
    const DataType keyDataType = key->GetDataType();
    const DataType valueDataType = value->GetDataType();
    const DataType outputDataType = attentionOut->GetDataType();

    // In the current PFA scenario, the datatypes of q, k, and v must be the same.
    if ((queryDataType != keyDataType) || (queryDataType != valueDataType)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Please check input Tensor datatype. "
            "The combination of [queryDataType]: %s, [keyDataType]: %s, [valueDataType]: %s is not supported by PFA.",
            StrDataTypePfa.at(ValidPfaAclDataType(queryDataType)).c_str(), StrDataTypePfa.at(ValidPfaAclDataType(keyDataType)).c_str(), StrDataTypePfa.at(ValidPfaAclDataType(valueDataType)).c_str());
        return false;
    }
    // map for dataType, {q/k/v, pseShift}
    static const unordered_map<DataType, DataType> qkvAndPseTypeRangeMap = {
        {DataType::DT_FLOAT16, DataType::DT_FLOAT16},
        {DataType::DT_BF16, DataType::DT_BF16},
        {DataType::DT_INT8, DataType::DT_FLOAT16}
    };
    // check dataType of q/k/v
    if (qkvAndPseTypeRangeMap.find(queryDataType) == qkvAndPseTypeRangeMap.end()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "query/value/key dataType(%s) is invalid, valid range is {%s, %s, %s}",
            StrDataTypePfa.at(ValidPfaAclDataType(queryDataType)).c_str(), StrDataTypePfa.at(ValidPfaAclDataType(DataType::DT_FLOAT16)).c_str(),
            StrDataTypePfa.at(ValidPfaAclDataType(DataType::DT_BF16)).c_str(), StrDataTypePfa.at(ValidPfaAclDataType(DataType::DT_INT8)).c_str());
        return false;
    }
    if (pseShift != nullptr) {
        const DataType pseDataType = pseShift->GetDataType(); // check dataType combination of q and pseShift
        if (pseDataType != qkvAndPseTypeRangeMap.at(queryDataType)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "pseShift dataType(%s) is invalid, should be %s when q/k/v is %s",
                StrDataTypePfa.at(ValidPfaAclDataType(pseDataType)).c_str(), StrDataTypePfa.at(ValidPfaAclDataType(qkvAndPseTypeRangeMap.at(queryDataType))).c_str(),
                StrDataTypePfa.at(ValidPfaAclDataType(queryDataType)).c_str());
            return false;
        }
    }

    // Currently, only the input and output dtype are different when quantifying related scenarios (int8_in/fp16_out or fp16_in/int8_out)
    if (queryDataType != outputDataType) {
        bool isQuant = (((queryDataType == DataType::DT_INT8 || queryDataType == DataType::DT_HIFLOAT8 || 
        queryDataType == DataType::DT_FLOAT8_E4M3FN) && outputDataType == DataType::DT_FLOAT16) || 
        (queryDataType == DataType::DT_FLOAT16 && outputDataType == DataType::DT_INT8) ||
        (queryDataType == DataType::DT_BF16 && outputDataType == DataType::DT_INT8));
        if (!isQuant) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Please check input/output Tensor datatype. "
                "The combination of [queryDataType]: %s, [outputDataType]: %s is not supported by PFA.",
                StrDataTypePfa.at(ValidPfaAclDataType(queryDataType)).c_str(), StrDataTypePfa.at(ValidPfaAclDataType(outputDataType)).c_str());
            return false;
        }
    }

    // Int8 quantization scene, does not support fp16 type atten mask
    if (attenMask != nullptr) {
        const std::set<DataType> validMaskType = {DataType::DT_FLOAT16, DataType::DT_INT8,
            DataType::DT_UINT8, DataType::DT_BOOL};
        const DataType attenMaskDataType = attenMask->GetDataType();
        if (validMaskType.find(attenMaskDataType) == validMaskType.end()) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "attenMask dataType(%s) is invalid, should be in range {%s, %s, %s, %s}",
                StrDataTypePfa.at(ValidPfaAclDataType(attenMaskDataType)).c_str(), StrDataTypePfa.at(ValidPfaAclDataType(DataType::DT_FLOAT16)).c_str(),
                StrDataTypePfa.at(ValidPfaAclDataType(DataType::DT_INT8)).c_str(), StrDataTypePfa.at(ValidPfaAclDataType(DataType::DT_UINT8)).c_str(),
                StrDataTypePfa.at(ValidPfaAclDataType(DataType::DT_BOOL)).c_str());
            return false;
        }
        if ((queryDataType == DataType::DT_INT8) && (attenMaskDataType == DataType::DT_FLOAT16)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Please check Tensor datatype. "
                "When input tensor datatype is %s, attenMaskDataType can not be %s.",
                StrDataTypePfa.at(ValidPfaAclDataType(queryDataType)).c_str(), StrDataTypePfa.at(ValidPfaAclDataType(attenMaskDataType)).c_str());
            return false;
        }
    }

    return true;
}

static bool CheckTensorFormatPrivate(const aclTensor* tensor) {
    if (tensor->GetStorageFormat() == op::Format::FORMAT_NC1HWC0 ||
        tensor->GetStorageFormat() == op::Format::FORMAT_FRACTAL_NZ ||
        tensor->GetStorageFormat() == op::Format::FORMAT_NDC1HWC0) {
        return true;
    }
    return false;
}

static bool CheckTensorFormat(const aclTensor* query, const aclTensor* key, const aclTensor* value,
                                const aclTensor *pseShift, const aclTensor* attenMask,
                                const aclTensor* attentionOut) {
    if (CheckTensorFormatPrivate(query)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Query format only support ND.");
        return false;
    }
    if (CheckTensorFormatPrivate(key)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Key format only support ND.");
        return false;
    }
    if (CheckTensorFormatPrivate(value)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Value format only support ND.");
        return false;
    }
    if (pseShift != nullptr && CheckTensorFormatPrivate(pseShift)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "PseShift format only support ND.");
        return false;
    }
    if (attenMask != nullptr && CheckTensorFormatPrivate(attenMask)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "AttenMask format only support ND.");
        return false;
    }
    if (CheckTensorFormatPrivate(attentionOut)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "AttentionOut format only support ND.");
        return false;
    }
    return true;
}

static inline bool CheckResultOutShapePfa(const aclTensor *inferOut, const aclTensor *out) {
    auto const &xShape = inferOut->GetViewShape();
    auto const &yShape = out->GetViewShape();
    if(xShape != yShape) {
        if(!(xShape.GetShapeSize() == 1 && yShape.GetShapeSize() == 1)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Out tensor's shape[%s] is not equal with Expect Out tensor's shape[%s], Please check the Out tensor's shape.",
                op::ToString(out->GetViewShape()).GetString(), op::ToString(inferOut->GetViewShape()).GetString());
            return false;
        }
    }
    return true;
}

aclnnStatus InnerPromptFlashAttentionGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *pseShift,
    const aclTensor *attenMask, const aclIntArray *actualSeqLengths, const aclIntArray *actualSeqLengthsKv,
    const aclTensor *deqScale1, const aclTensor *quantScale1, const aclTensor *deqScale2, const aclTensor *quantScale2,
    const aclTensor *quantOffset2, int64_t numHeads, double scaleValue, int64_t preTokens, int64_t nextTokens,
    char *inputLayout, int64_t numKeyValueHeads, int64_t sparseMode, int64_t innerPrecise,
    const aclTensor *attentionOut, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(InnerPromptFlashAttention,
                DFX_IN(query, key, value, pseShift, attenMask, actualSeqLengths, actualSeqLengthsKv,
                        deqScale1, quantScale1, deqScale2, quantScale2, quantOffset2,
                        numHeads, scaleValue, preTokens, nextTokens, inputLayout, numKeyValueHeads,
                        sparseMode, innerPrecise),
                DFX_OUT(attentionOut));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // Check if the required input pointer is empty
    CHECK_RET(CheckNotNull(query, key, value, inputLayout, attentionOut), ACLNN_ERR_PARAM_NULLPTR);

    // When b, n1, s1 = 0, no processing is performed
    // When n2, s2, d = 0, directly call the l0 interface for processing
    if (attentionOut->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    CHECK_RET(CheckTensorDataType(query, key, value, pseShift, attenMask, attentionOut), ACLNN_ERR_PARAM_INVALID);

    FaShapeInfo shapeInfo;
    CHECK_RET(AnalysisInputShapeInfo(query, key, value, inputLayout, numHeads, numKeyValueHeads, shapeInfo, attentionOut) ==
              ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    if (shapeInfo.needPad) {
        CHECK_RET(CheckTensorFormat(query, key, value, pseShift, attenMask, attentionOut), ACLNN_ERR_PARAM_INVALID);
    }

    aclOpExecutor *l0Executor = uniqueExecutor.get();
    CHECK_RET(ContiguousInput(query, key, value, pseShift, attenMask, l0Executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);

    CHECK_RET(PreprocessQKVInput(query, key, value, quantScale2, quantOffset2, shapeInfo, l0Executor) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);

    auto l0AttentionOutOut = l0op::PromptFlashAttention(query, key, value, pseShift, attenMask,
                                                        actualSeqLengths, actualSeqLengthsKv,
                                                        deqScale1, quantScale1, deqScale2, quantScale2, quantOffset2,
                                                        numHeads, scaleValue, preTokens, nextTokens,
                                                        inputLayout,
                                                        numKeyValueHeads, sparseMode, innerPrecise,
                                                        attentionOut, l0Executor);
    CHECK_RET(l0AttentionOutOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(PostProcessOutput(l0AttentionOutOut, attentionOut, shapeInfo, l0Executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(CheckResultOutShapePfa(l0AttentionOutOut, attentionOut), ACLNN_ERR_PARAM_INVALID);
    auto viewCopyResult = l0op::ViewCopy(l0AttentionOutOut, attentionOut, l0Executor);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus InnerPromptFlashAttention(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                           const aclrtStream stream)
{
    L2_DFX_PHASE_2(InnerPromptFlashAttention);
    // Fixed format. The calculation is completed by calling the framework capability.
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

}  // namespace

#ifdef __cplusplus
}
#endif