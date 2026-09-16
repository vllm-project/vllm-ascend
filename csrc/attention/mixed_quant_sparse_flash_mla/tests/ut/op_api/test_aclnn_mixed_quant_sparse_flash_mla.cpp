/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <memory>
#include <vector>
#include <cstdint>
#include "gtest/gtest.h"
#include "../../../op_api/aclnn_mixed_quant_sparse_flash_mla.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"
using namespace std;
using namespace op;

namespace {
void DestroyAclTensor(aclTensor *tensor)
{
    Release(tensor);
}
} // namespace

class mixed_quant_sparse_flash_mla_opapi_ut : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
        cout << "mixed_quant_sparse_flash_mla_opapi_ut SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "mixed_quant_sparse_flash_mla_opapi_ut TearDown" << endl;
    }
};

// All optional inputs are null
TEST_F(mixed_quant_sparse_flash_mla_opapi_ut, mixed_quant_sparse_flash_mla_aclnn_0)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    aclnnStatus aclRet = aclnnMixedQuantSparseFlashMlaGetWorkspaceSize(
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0, 576, 0.0416666666666667, 1, 0, 0, -1, -1, "TND",
        "PA_BBND", 0, false, nullptr, nullptr, &workspaceSize, &executor);

    EXPECT_NE(aclRet, ACL_SUCCESS);
    EXPECT_EQ(executor, nullptr);
}

// Full inputs with contiguous tensors pass the pre-checks and reach inner api
TEST_F(mixed_quant_sparse_flash_mla_opapi_ut, mixed_quant_sparse_flash_mla_aclnn_1)
{
    auto q = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({512, 64, 512}, ACL_BF16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto oriKv = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({128, 128, 1, 512}, ACL_BF16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto cmpKv = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({128, 128, 1, 512}, ACL_BF16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto oriSparseIndices = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({128, 128, 1, 16}, ACL_INT32, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto oriBlockTable = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({4, 32}, ACL_INT32, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto cuSeqlensQ = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({5}, ACL_INT32, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto sequsedOriKv = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto attnOut = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({512, 64, 512}, ACL_BF16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    aclnnStatus aclRet = aclnnMixedQuantSparseFlashMlaGetWorkspaceSize(
        q.get(), oriKv.get(), cmpKv.get(), oriSparseIndices.get(), nullptr, oriBlockTable.get(), nullptr,
        cuSeqlensQ.get(), nullptr, nullptr, nullptr, sequsedOriKv.get(), nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, 0, 576, 0.0416666666666667, 1, 0, 0, -1, -1, "TND", "PA_BBND", 0, false, attnOut.get(), nullptr,
        &workspaceSize, &executor);

    EXPECT_NE(aclRet, ACL_SUCCESS);
    EXPECT_EQ(executor, nullptr);
}

// Non-contiguous kv tensor is rejected when opbase does not support TensorV2
TEST_F(mixed_quant_sparse_flash_mla_opapi_ut, mixed_quant_sparse_flash_mla_aclnn_2)
{
    auto q = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({512, 64, 512}, ACL_BF16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    // axis 0 stride padded, non-contiguous oriKv
    auto oriKv = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({4, 32, 1, 512}, ACL_BF16, ACL_FORMAT_ND, {131072, 512, 512, 1}).ToAclTypeRawPtr(),
        DestroyAclTensor);
    auto attnOut = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({512, 64, 512}, ACL_BF16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    aclnnStatus aclRet = aclnnMixedQuantSparseFlashMlaGetWorkspaceSize(
        q.get(), oriKv.get(), nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0, 576, 0.0416666666666667, 1, 0, 0, -1, -1, "TND",
        "PA_BBND", 0, false, attnOut.get(), nullptr, &workspaceSize, &executor);

    EXPECT_NE(aclRet, ACL_SUCCESS);
    EXPECT_EQ(executor, nullptr);
}

// Uint8 kv tensor is pre-processed to float8_e4m3fn dtype
TEST_F(mixed_quant_sparse_flash_mla_opapi_ut, mixed_quant_sparse_flash_mla_aclnn_3)
{
    auto q = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({512, 64, 512}, ACL_BF16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto oriKv = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({128, 128, 1, 512}, ACL_UINT8, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto attnOut = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({512, 64, 512}, ACL_BF16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    aclnnStatus aclRet = aclnnMixedQuantSparseFlashMlaGetWorkspaceSize(
        q.get(), oriKv.get(), nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0, 576, 0.0416666666666667, 1, 0, 0, -1, -1, "TND",
        "PA_BBND", 0, false, attnOut.get(), nullptr, &workspaceSize, &executor);

    EXPECT_EQ(oriKv->GetDataType(), ACL_FLOAT8_E4M3FN);
    EXPECT_NE(aclRet, ACL_SUCCESS);
    EXPECT_EQ(executor, nullptr);
}

// Sinks tensor with shape {0} is treated as nullptr in the pre-process
TEST_F(mixed_quant_sparse_flash_mla_opapi_ut, mixed_quant_sparse_flash_mla_aclnn_4)
{
    auto q = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({512, 64, 512}, ACL_BF16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto sinks = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto attnOut = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({512, 64, 512}, ACL_BF16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    aclnnStatus aclRet = aclnnMixedQuantSparseFlashMlaGetWorkspaceSize(
        q.get(), nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr, sinks.get(), nullptr, 0, 576, 0.0416666666666667, 1, 0, 0, -1, -1, "TND",
        "PA_BBND", 0, false, attnOut.get(), nullptr, &workspaceSize, &executor);

    EXPECT_NE(aclRet, ACL_SUCCESS);
    EXPECT_EQ(executor, nullptr);
}

// Null executor of the second phase entry
TEST_F(mixed_quant_sparse_flash_mla_opapi_ut, mixed_quant_sparse_flash_mla_aclnn_5)
{
    aclnnStatus aclRet = aclnnMixedQuantSparseFlashMla(nullptr, 0, nullptr, nullptr);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}
