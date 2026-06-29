/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */
#ifndef MC2_3RD_MATMUL_UTIL_H_
#define MC2_3RD_MATMUL_UTIL_H_

#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/platform.h"

namespace Ops {
namespace Transformer {
// These are used to check repo hit
const int32_t FP16_BF16_FLAG = 1;
const int32_t FP32_FLAG = 0;
const int32_t HF32_FLAG = 64;
const std::string SOC_B3 = "Ascend910B3";
const std::string SOC_C3 = "Ascend910_9372";
const std::string SOC_B4 = "Ascend910B4";
const std::string SOC_C4 = "Ascend910_9382";

struct OpBaseInfo {
  op::DataType  self_dtype;
  op::Format  self_format;
  op::DataType  mat2_dtype;
  op::Format  mat2_format;
  op::DataType  output_dtype;
  op::Format  output_format;
};

struct OpShapeInfo {
  // m、k、n dim
  int64_t mDim;
  int64_t kDim;
  int64_t nDim;

  // input tranpose flag
  bool transposeX1;

  // mat2 transpose flag
  bool transposeX2;

  int64_t dtypeASize = 2; // float16 dtype 2
  int64_t dtypeBSize = 2; // float16 dtype 2
};

struct MmOpInfo {
  // mm api input info
  OpBaseInfo  ori_info;
  // npu mm kernel support info
  OpBaseInfo  support_info;
  // HF32 Flag
  int64_t opImplModeEnum = 0x1;
  bool enableHf32 = false;
  bool supporSplitK = false;
  int64_t aiCoreCnt;
  // mm api shape info
  OpShapeInfo shapeInfo;
};

struct TensorInfo {
  const aclTensor*  tensor;
  op::DataType  dataType;
  op::Format  format;
};

op::Shape SwapLastTwoDimValue(const op::Shape tensorShape);

bool IsInputSupportFp32();

bool CheckBatchDimBroadcast(size_t batch1DimNum, size_t batch2DimNum, const op::Shape& batch1, const op::Shape& batch2);

bool IsFormatSupportNd(const aclTensor* self, const aclTensor* mat2);

bool IsNdToNzOnTheFly(const aclTensor* self, const aclTensor* mat2);

bool IsTransposeLastTwoDims(const aclTensor* tensor);

bool CheckGemmV3Support(const aclTensor* mat1, const aclTensor* mat2, MmOpInfo& mmOpInfo, int8_t cubeMathType);

aclnnStatus SetMmSupportDType(MmOpInfo& mmOpInfo, int8_t cubeMathType);

aclnnStatus SetMmSupportFormat(const aclTensor* self, const aclTensor* mat2, MmOpInfo& mmOpInfo);

aclnnStatus GetMmInfo(MmOpInfo mmOpInfo);
MmOpInfo GetMatmulOpInfo(const aclTensor *self, const aclTensor *mat2, int8_t cubeMathType);
bool ContiguousAndCast(const aclTensor *&contiguousInput, const aclTensor *&castOut, bool &transposeFlag,
                       op::DataType dtype, aclOpExecutor *executor);

bool GetNzSplitKFlag(const aclTensor *self, const aclTensor *mat2, const op::Format selfSuppFormat,
                     const op::Format outSuppFormat);

bool IsSupportNzNzNd(const aclTensor* self, const aclTensor* mat2);

bool IsSplitk(const TensorInfo* self, const TensorInfo* mat2);

bool NeedToConvertBias(const aclTensor* self, const aclTensor* mat1, const aclTensor* mat2, const aclScalar* beta,
                       const aclScalar* alpha);

// 区别bmm 和 mm, bmm（DimNum==3）返回 1， mm（DimNum==2）返回0
int64_t GetOffSet(int64_t DimNum);

bool checkBF16MMValid(const aclTensor *&self, const aclTensor *&mat2, const bool &transX2Flag);

aclIntArray* NeedTransPerm(const aclTensor *x, aclOpExecutor *executor);

bool checkBF16SizeValid(const aclTensor *&mat2, const bool &transX2Flag);

bool IfKEqual1(const aclTensor *&selfInput, const MmOpInfo& mmOpInfo, const bool &transX1Flag, const aclTensor *&bias);

aclnnStatus IfKEqual1SelfToMK(const aclTensor *&selfInput, const aclTensor *&selfReshapeOutput, bool &transX1Flag,
                              aclOpExecutor *executor);

aclnnStatus IfMEqual1SelfToMK(const aclTensor *&selfInput, const aclTensor *&selfReshapeOutput,
                              const op::Format selfInputFormat, bool &transX1Flag, aclOpExecutor *executor);

aclnnStatus IfKEqual1Mat2ToKN(const aclTensor *&mat2Input, const aclTensor *&mat2ReshapeOutput, bool &transX2Flag,
                              aclOpExecutor *executor);

aclnnStatus IfNEqual1Mat2ToNK(const aclTensor *&mat2Input, const aclTensor *&mat2ReshapeOutput,
                              const op::Format mat2InputFormat, bool &transX2Flag, aclOpExecutor *executor);

uint64_t TransDequantScaleToM1(const float deqScale);

const aclTensor *ContiguousBias(const aclTensor *self, const aclTensor *bias, aclOpExecutor *executor);

op::FVector<int64_t> GetShape(const aclTensor *tensor);
} // namespace Transformer
} // namespace Ops


#endif  // MC2_3RD_MATMUL_UTIL_H_
