/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <cstring>
#include <vector>
#include "acl/acl.h"
#include "opdev/platform.h"
#include <gtest/gtest.h>
#include "attention/generic_block_sparse_attention/op_host/op_api/aclnn_generic_block_sparse_attention.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;
using namespace op;

class aclnn_generic_block_sparse_attention_ut : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "aclnn_generic_block_sparse_attention_ut SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "aclnn_generic_block_sparse_attention_ut TearDown" << endl;
    }
};

namespace {
constexpr int64_t kBatch = 1;
constexpr int64_t kS1 = 4;
constexpr int64_t kS2 = 256;
constexpr int64_t kN1 = 4;
constexpr int64_t kN2 = 1;
constexpr int64_t kD = 128;
constexpr int64_t kTopK = 2;
constexpr int64_t kBlockSize = 128;
constexpr int64_t kBlockShapeX = 1;
constexpr int64_t kT = kBatch * kS1;
constexpr int64_t kMaxBlocks = (kS2 + kBlockSize - 1) / kBlockSize;
constexpr int64_t kTotalQBlocks = kT;
constexpr double kScale = 1.0 / sqrt(static_cast<double>(kD));
constexpr int64_t kLayoutSparse = 4; // KVN_TotalQB_KB
} // namespace

// ============================================================================
// Case 1: normal TND + PA_BBND smoke
// ============================================================================
TEST_F(aclnn_generic_block_sparse_attention_ut, normal)
{
    auto query = TensorDesc({kT, kN1, kD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto key = TensorDesc({kMaxBlocks, kBlockSize, kN2, kD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto value = TensorDesc({kMaxBlocks, kBlockSize, kN2, kD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto sparseBlockIdx = TensorDesc({kN2, kTotalQBlocks, kTopK}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto sparseBlockCount = TensorDesc({kN2, kTotalQBlocks}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{1});
    auto metadata = TensorDesc({1024}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto blockTable = TensorDesc({kBatch, kMaxBlocks}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto blockShape = IntArrayDesc(vector<int64_t>{kBlockShapeX, kBlockSize});
    char layoutQ[] = "TND";
    char layoutKv[] = "PA_BBND";
    auto cuSeqLengthsQ = TensorDesc({kBatch + 1}, ACL_INT64, ACL_FORMAT_ND).Value(vector<int64_t>{0, kS1});
    auto sequsedKv = TensorDesc({kBatch}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{kS2});
    auto attentionOut = TensorDesc({kT, kN1, kD}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGenericBlockSparseAttention,
                        INPUT(query,            // query
                              key,              // key
                              value,            // value
                              sparseBlockIdx,   // sparseBlockIdx
                              sparseBlockCount, // sparseBlockCount
                              metadata,         // metadataOptional
                              nullptr,          // attenMaskOptional
                              nullptr,          // qDequantScaleOptional
                              nullptr,          // kDequantScaleOptional
                              nullptr,          // vDequantScaleOptional
                              nullptr,          // pQuantScaleOptional
                              cuSeqLengthsQ,    // cuSeqLengthsQOptional
                              nullptr,          // cuSeqLengthsKvOptional
                              nullptr,          // sequsedQOptional
                              sequsedKv,        // sequsedKvOptional
                              blockTable,       // blockTableOptional
                              blockShape,       // blockShape
                              layoutQ,          // layoutQ
                              layoutKv,         // layoutKv
                              kLayoutSparse,    // layoutSparsePattern
                              kScale,           // scaleValue
                              1,                // maskType
                              0,                // quantType
                              0.0,              // dstTypeMax
                              0,                // softmaxPrecision
                              -1,               // winLeft
                              -1,               // winRight
                              0,                // returnSoftmaxlse
                              0,                // residualBlockMode
                              false             // isConsistentTopk
                              ),
                        OUTPUT(attentionOut, nullptr));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// ============================================================================
// Case 2: required query is nullptr
// ============================================================================
TEST_F(aclnn_generic_block_sparse_attention_ut, null_query)
{
    auto key = TensorDesc({kMaxBlocks, kBlockSize, kN2, kD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto value = TensorDesc({kMaxBlocks, kBlockSize, kN2, kD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto sparseBlockIdx = TensorDesc({kN2, kTotalQBlocks, kTopK}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto sparseBlockCount = TensorDesc({kN2, kTotalQBlocks}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{1});
    auto metadata = TensorDesc({1024}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto blockTable = TensorDesc({kBatch, kMaxBlocks}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto blockShape = IntArrayDesc(vector<int64_t>{kBlockShapeX, kBlockSize});
    char layoutQ[] = "TND";
    char layoutKv[] = "PA_BBND";
    auto attentionOut = TensorDesc({kT, kN1, kD}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGenericBlockSparseAttention,
                        INPUT(nullptr,          // query
                              key,              // key
                              value,            // value
                              sparseBlockIdx,   // sparseBlockIdx
                              sparseBlockCount, // sparseBlockCount
                              metadata,         // metadataOptional
                              nullptr,          // attenMaskOptional
                              nullptr,          // qDequantScaleOptional
                              nullptr,          // kDequantScaleOptional
                              nullptr,          // vDequantScaleOptional
                              nullptr,          // pQuantScaleOptional
                              nullptr,          // cuSeqLengthsQOptional
                              nullptr,          // cuSeqLengthsKvOptional
                              nullptr,          // sequsedQOptional
                              nullptr,          // sequsedKvOptional
                              blockTable,       // blockTableOptional
                              blockShape,       // blockShape
                              layoutQ,          // layoutQ
                              layoutKv,         // layoutKv
                              kLayoutSparse,    // layoutSparsePattern
                              kScale,           // scaleValue
                              1,                // maskType
                              0,                // quantType
                              0.0,              // dstTypeMax
                              0,                // softmaxPrecision
                              -1,               // winLeft
                              -1,               // winRight
                              0,                // returnSoftmaxlse
                              0,                // residualBlockMode
                              false             // isConsistentTopk
                              ),
                        OUTPUT(attentionOut, nullptr));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// ============================================================================
// Case 3: metadata is required by Contiguous path (nullptr -> INNER_NULLPTR)
// ============================================================================
TEST_F(aclnn_generic_block_sparse_attention_ut, null_metadata)
{
    auto query = TensorDesc({kT, kN1, kD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto key = TensorDesc({kMaxBlocks, kBlockSize, kN2, kD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto value = TensorDesc({kMaxBlocks, kBlockSize, kN2, kD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto sparseBlockIdx = TensorDesc({kN2, kTotalQBlocks, kTopK}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto sparseBlockCount = TensorDesc({kN2, kTotalQBlocks}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{1});
    auto blockTable = TensorDesc({kBatch, kMaxBlocks}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto blockShape = IntArrayDesc(vector<int64_t>{kBlockShapeX, kBlockSize});
    char layoutQ[] = "TND";
    char layoutKv[] = "PA_BBND";
    auto attentionOut = TensorDesc({kT, kN1, kD}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGenericBlockSparseAttention,
                        INPUT(query,            // query
                              key,              // key
                              value,            // value
                              sparseBlockIdx,   // sparseBlockIdx
                              sparseBlockCount, // sparseBlockCount
                              nullptr,          // metadataOptional
                              nullptr,          // attenMaskOptional
                              nullptr,          // qDequantScaleOptional
                              nullptr,          // kDequantScaleOptional
                              nullptr,          // vDequantScaleOptional
                              nullptr,          // pQuantScaleOptional
                              nullptr,          // cuSeqLengthsQOptional
                              nullptr,          // cuSeqLengthsKvOptional
                              nullptr,          // sequsedQOptional
                              nullptr,          // sequsedKvOptional
                              blockTable,       // blockTableOptional
                              blockShape,       // blockShape
                              layoutQ,          // layoutQ
                              layoutKv,         // layoutKv
                              kLayoutSparse,    // layoutSparsePattern
                              kScale,           // scaleValue
                              1,                // maskType
                              0,                // quantType
                              0.0,              // dstTypeMax
                              0,                // softmaxPrecision
                              -1,               // winLeft
                              -1,               // winRight
                              0,                // returnSoftmaxlse
                              0,                // residualBlockMode
                              false             // isConsistentTopk
                              ),
                        OUTPUT(attentionOut, nullptr));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_INNER_NULLPTR);
}

// ============================================================================
// Case 4: required key is nullptr
// ============================================================================
TEST_F(aclnn_generic_block_sparse_attention_ut, null_key)
{
    auto query = TensorDesc({kT, kN1, kD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto value = TensorDesc({kMaxBlocks, kBlockSize, kN2, kD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto sparseBlockIdx = TensorDesc({kN2, kTotalQBlocks, kTopK}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto sparseBlockCount = TensorDesc({kN2, kTotalQBlocks}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{1});
    auto metadata = TensorDesc({1024}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto blockTable = TensorDesc({kBatch, kMaxBlocks}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto blockShape = IntArrayDesc(vector<int64_t>{kBlockShapeX, kBlockSize});
    char layoutQ[] = "TND";
    char layoutKv[] = "PA_BBND";
    auto attentionOut = TensorDesc({kT, kN1, kD}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGenericBlockSparseAttention,
                        INPUT(query,            // query
                              nullptr,          // key
                              value,            // value
                              sparseBlockIdx,   // sparseBlockIdx
                              sparseBlockCount, // sparseBlockCount
                              metadata,         // metadataOptional
                              nullptr,          // attenMaskOptional
                              nullptr,          // qDequantScaleOptional
                              nullptr,          // kDequantScaleOptional
                              nullptr,          // vDequantScaleOptional
                              nullptr,          // pQuantScaleOptional
                              nullptr,          // cuSeqLengthsQOptional
                              nullptr,          // cuSeqLengthsKvOptional
                              nullptr,          // sequsedQOptional
                              nullptr,          // sequsedKvOptional
                              blockTable,       // blockTableOptional
                              blockShape,       // blockShape
                              layoutQ,          // layoutQ
                              layoutKv,         // layoutKv
                              kLayoutSparse,    // layoutSparsePattern
                              kScale,           // scaleValue
                              1,                // maskType
                              0,                // quantType
                              0.0,              // dstTypeMax
                              0,                // softmaxPrecision
                              -1,               // winLeft
                              -1,               // winRight
                              0,                // returnSoftmaxlse
                              0,                // residualBlockMode
                              false             // isConsistentTopk
                              ),
                        OUTPUT(attentionOut, nullptr));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// ============================================================================
// Case 5: required attentionOut is nullptr
// ============================================================================
TEST_F(aclnn_generic_block_sparse_attention_ut, null_attention_out)
{
    auto query = TensorDesc({kT, kN1, kD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto key = TensorDesc({kMaxBlocks, kBlockSize, kN2, kD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto value = TensorDesc({kMaxBlocks, kBlockSize, kN2, kD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto sparseBlockIdx = TensorDesc({kN2, kTotalQBlocks, kTopK}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto sparseBlockCount = TensorDesc({kN2, kTotalQBlocks}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{1});
    auto metadata = TensorDesc({1024}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto blockTable = TensorDesc({kBatch, kMaxBlocks}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto blockShape = IntArrayDesc(vector<int64_t>{kBlockShapeX, kBlockSize});
    char layoutQ[] = "TND";
    char layoutKv[] = "PA_BBND";

    auto ut = OP_API_UT(aclnnGenericBlockSparseAttention,
                        INPUT(query,            // query
                              key,              // key
                              value,            // value
                              sparseBlockIdx,   // sparseBlockIdx
                              sparseBlockCount, // sparseBlockCount
                              metadata,         // metadataOptional
                              nullptr,          // attenMaskOptional
                              nullptr,          // qDequantScaleOptional
                              nullptr,          // kDequantScaleOptional
                              nullptr,          // vDequantScaleOptional
                              nullptr,          // pQuantScaleOptional
                              nullptr,          // cuSeqLengthsQOptional
                              nullptr,          // cuSeqLengthsKvOptional
                              nullptr,          // sequsedQOptional
                              nullptr,          // sequsedKvOptional
                              blockTable,       // blockTableOptional
                              blockShape,       // blockShape
                              layoutQ,          // layoutQ
                              layoutKv,         // layoutKv
                              kLayoutSparse,    // layoutSparsePattern
                              kScale,           // scaleValue
                              1,                // maskType
                              0,                // quantType
                              0.0,              // dstTypeMax
                              0,                // softmaxPrecision
                              -1,               // winLeft
                              -1,               // winRight
                              0,                // returnSoftmaxlse
                              0,                // residualBlockMode
                              false             // isConsistentTopk
                              ),
                        OUTPUT(nullptr, nullptr));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// ============================================================================
// Case 6: returnSoftmaxlse=1 with valid LSE output
// ============================================================================
TEST_F(aclnn_generic_block_sparse_attention_ut, normal_with_lse)
{
    auto query = TensorDesc({kT, kN1, kD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto key = TensorDesc({kMaxBlocks, kBlockSize, kN2, kD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto value = TensorDesc({kMaxBlocks, kBlockSize, kN2, kD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto sparseBlockIdx = TensorDesc({kN2, kTotalQBlocks, kTopK}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto sparseBlockCount = TensorDesc({kN2, kTotalQBlocks}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{1});
    auto metadata = TensorDesc({1024}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto blockTable = TensorDesc({kBatch, kMaxBlocks}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto blockShape = IntArrayDesc(vector<int64_t>{kBlockShapeX, kBlockSize});
    char layoutQ[] = "TND";
    char layoutKv[] = "PA_BBND";
    auto cuSeqLengthsQ = TensorDesc({kBatch + 1}, ACL_INT64, ACL_FORMAT_ND).Value(vector<int64_t>{0, kS1});
    auto sequsedKv = TensorDesc({kBatch}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{kS2});
    auto attentionOut = TensorDesc({kT, kN1, kD}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto softmaxLse = TensorDesc({kT, kN1, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGenericBlockSparseAttention,
                        INPUT(query,            // query
                              key,              // key
                              value,            // value
                              sparseBlockIdx,   // sparseBlockIdx
                              sparseBlockCount, // sparseBlockCount
                              metadata,         // metadataOptional
                              nullptr,          // attenMaskOptional
                              nullptr,          // qDequantScaleOptional
                              nullptr,          // kDequantScaleOptional
                              nullptr,          // vDequantScaleOptional
                              nullptr,          // pQuantScaleOptional
                              cuSeqLengthsQ,    // cuSeqLengthsQOptional
                              nullptr,          // cuSeqLengthsKvOptional
                              nullptr,          // sequsedQOptional
                              sequsedKv,        // sequsedKvOptional
                              blockTable,       // blockTableOptional
                              blockShape,       // blockShape
                              layoutQ,          // layoutQ
                              layoutKv,         // layoutKv
                              kLayoutSparse,    // layoutSparsePattern
                              kScale,           // scaleValue
                              1,                // maskType
                              0,                // quantType
                              0.0,              // dstTypeMax
                              0,                // softmaxPrecision
                              -1,               // winLeft
                              -1,               // winRight
                              1,                // returnSoftmaxlse
                              0,                // residualBlockMode
                              false             // isConsistentTopk
                              ),
                        OUTPUT(attentionOut, softmaxLse));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// ============================================================================
// Case 7: unsupported layoutQ
// ============================================================================
TEST_F(aclnn_generic_block_sparse_attention_ut, invalid_layout_q)
{
    auto query = TensorDesc({kT, kN1, kD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto key = TensorDesc({kMaxBlocks, kBlockSize, kN2, kD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto value = TensorDesc({kMaxBlocks, kBlockSize, kN2, kD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto sparseBlockIdx = TensorDesc({kN2, kTotalQBlocks, kTopK}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto sparseBlockCount = TensorDesc({kN2, kTotalQBlocks}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{1});
    auto metadata = TensorDesc({1024}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto blockTable = TensorDesc({kBatch, kMaxBlocks}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto blockShape = IntArrayDesc(vector<int64_t>{kBlockShapeX, kBlockSize});
    char layoutQ[] = "BNSD";
    char layoutKv[] = "PA_BBND";
    auto attentionOut = TensorDesc({kT, kN1, kD}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGenericBlockSparseAttention,
                        INPUT(query, key, value, sparseBlockIdx, sparseBlockCount, metadata, nullptr, nullptr, nullptr,
                              nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, blockTable, blockShape, layoutQ,
                              layoutKv, kLayoutSparse, kScale, 1, 0, 0.0, 0, -1, -1, 0, 0, false),
                        OUTPUT(attentionOut, nullptr));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// ============================================================================
// Case 8: null blockShape
// ============================================================================
TEST_F(aclnn_generic_block_sparse_attention_ut, null_block_shape)
{
    auto query = TensorDesc({kT, kN1, kD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto key = TensorDesc({kMaxBlocks, kBlockSize, kN2, kD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto value = TensorDesc({kMaxBlocks, kBlockSize, kN2, kD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto sparseBlockIdx = TensorDesc({kN2, kTotalQBlocks, kTopK}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto sparseBlockCount = TensorDesc({kN2, kTotalQBlocks}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{1});
    auto metadata = TensorDesc({1024}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto blockTable = TensorDesc({kBatch, kMaxBlocks}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    char layoutQ[] = "TND";
    char layoutKv[] = "PA_BBND";
    auto attentionOut = TensorDesc({kT, kN1, kD}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGenericBlockSparseAttention,
                        INPUT(query, key, value, sparseBlockIdx, sparseBlockCount, metadata, nullptr, nullptr, nullptr,
                              nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, blockTable, nullptr, layoutQ,
                              layoutKv, kLayoutSparse, kScale, 1, 0, 0.0, 0, -1, -1, 0, 0, false),
                        OUTPUT(attentionOut, nullptr));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// ============================================================================
// Case 9: invalid blockShape size
// ============================================================================
TEST_F(aclnn_generic_block_sparse_attention_ut, invalid_block_shape_size)
{
    auto query = TensorDesc({kT, kN1, kD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto key = TensorDesc({kMaxBlocks, kBlockSize, kN2, kD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto value = TensorDesc({kMaxBlocks, kBlockSize, kN2, kD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto sparseBlockIdx = TensorDesc({kN2, kTotalQBlocks, kTopK}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto sparseBlockCount = TensorDesc({kN2, kTotalQBlocks}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{1});
    auto metadata = TensorDesc({1024}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto blockTable = TensorDesc({kBatch, kMaxBlocks}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto blockShape = IntArrayDesc(vector<int64_t>{kBlockShapeX});
    char layoutQ[] = "TND";
    char layoutKv[] = "PA_BBND";
    auto attentionOut = TensorDesc({kT, kN1, kD}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGenericBlockSparseAttention,
                        INPUT(query, key, value, sparseBlockIdx, sparseBlockCount, metadata, nullptr, nullptr, nullptr,
                              nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, blockTable, blockShape, layoutQ,
                              layoutKv, kLayoutSparse, kScale, 1, 0, 0.0, 0, -1, -1, 0, 0, false),
                        OUTPUT(attentionOut, nullptr));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// ============================================================================
// Case 10: Q/K dtype mismatch
// ============================================================================
TEST_F(aclnn_generic_block_sparse_attention_ut, dtype_mismatch)
{
    auto query = TensorDesc({kT, kN1, kD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto key = TensorDesc({kMaxBlocks, kBlockSize, kN2, kD}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto value = TensorDesc({kMaxBlocks, kBlockSize, kN2, kD}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto sparseBlockIdx = TensorDesc({kN2, kTotalQBlocks, kTopK}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto sparseBlockCount = TensorDesc({kN2, kTotalQBlocks}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{1});
    auto metadata = TensorDesc({1024}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto blockTable = TensorDesc({kBatch, kMaxBlocks}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto blockShape = IntArrayDesc(vector<int64_t>{kBlockShapeX, kBlockSize});
    char layoutQ[] = "TND";
    char layoutKv[] = "PA_BBND";
    auto attentionOut = TensorDesc({kT, kN1, kD}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGenericBlockSparseAttention,
                        INPUT(query, key, value, sparseBlockIdx, sparseBlockCount, metadata, nullptr, nullptr, nullptr,
                              nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, blockTable, blockShape, layoutQ,
                              layoutKv, kLayoutSparse, kScale, 1, 0, 0.0, 0, -1, -1, 0, 0, false),
                        OUTPUT(attentionOut, nullptr));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// ============================================================================
// Case 11: invalid softmaxPrecision
// ============================================================================
TEST_F(aclnn_generic_block_sparse_attention_ut, invalid_softmax_precision)
{
    auto query = TensorDesc({kT, kN1, kD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto key = TensorDesc({kMaxBlocks, kBlockSize, kN2, kD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto value = TensorDesc({kMaxBlocks, kBlockSize, kN2, kD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto sparseBlockIdx = TensorDesc({kN2, kTotalQBlocks, kTopK}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto sparseBlockCount = TensorDesc({kN2, kTotalQBlocks}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{1});
    auto metadata = TensorDesc({1024}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto blockTable = TensorDesc({kBatch, kMaxBlocks}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto blockShape = IntArrayDesc(vector<int64_t>{kBlockShapeX, kBlockSize});
    char layoutQ[] = "TND";
    char layoutKv[] = "PA_BBND";
    auto attentionOut = TensorDesc({kT, kN1, kD}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGenericBlockSparseAttention,
                        INPUT(query, key, value, sparseBlockIdx, sparseBlockCount, metadata, nullptr, nullptr, nullptr,
                              nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, blockTable, blockShape, layoutQ,
                              layoutKv, kLayoutSparse, kScale, 1, 0, 0.0, 2, -1, -1, 0, 0, 0),
                        OUTPUT(attentionOut, nullptr));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
