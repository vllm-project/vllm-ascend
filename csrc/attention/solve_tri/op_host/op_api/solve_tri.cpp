/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * BSD 3-Clause License.
 */
 #include "opdev/op_log.h"
 #include "opdev/op_dfx.h"
 #include "opdev/make_op_executor.h"
 #include "aclnn_solve_tri.h"
 #include <string>

 using namespace op;

 namespace l0op {
 OP_TYPE_REGISTER(SolveTri);

 const aclTensor* SolveTri(
     const aclTensor *x,
     const aclIntArray *cuSeqlensOptional,
     const aclIntArray *chunkIndicesOptional,
     const char *layout,
     const aclTensor *xOut,
     aclOpExecutor *executor)
 {
     L0_DFX(SolveTri, x, cuSeqlensOptional, chunkIndicesOptional, layout, xOut);

     const aclTensor *actualCuSeqlens = nullptr;
     if (cuSeqlensOptional) {
         actualCuSeqlens = executor->ConvertToTensor(cuSeqlensOptional, DataType::DT_INT64);
         const_cast<aclTensor *>(actualCuSeqlens)->SetStorageFormat(Format::FORMAT_ND);
         const_cast<aclTensor *>(actualCuSeqlens)->SetViewFormat(Format::FORMAT_ND);
         const_cast<aclTensor *>(actualCuSeqlens)->SetOriginalFormat(Format::FORMAT_ND);
     }

     const aclTensor *actualChunkIndices = nullptr;
     if (chunkIndicesOptional) {
         actualChunkIndices = executor->ConvertToTensor(chunkIndicesOptional, DataType::DT_INT64);
         const_cast<aclTensor *>(actualChunkIndices)->SetStorageFormat(Format::FORMAT_ND);
         const_cast<aclTensor *>(actualChunkIndices)->SetViewFormat(Format::FORMAT_ND);
         const_cast<aclTensor *>(actualChunkIndices)->SetOriginalFormat(Format::FORMAT_ND);
     }

     // layout string -> std::string for OP_ATTR
     std::string layoutStr(layout ? layout : "bsnd");

     auto ret = ADD_TO_LAUNCHER_LIST_AICORE(SolveTri,
         OP_INPUT(x, actualCuSeqlens, actualChunkIndices),
         OP_OUTPUT(xOut),
         OP_ATTR(layoutStr));
     if (ret != ACLNN_SUCCESS) {
         return nullptr;
     }
     return xOut;
 }

 }  // namespace l0op
