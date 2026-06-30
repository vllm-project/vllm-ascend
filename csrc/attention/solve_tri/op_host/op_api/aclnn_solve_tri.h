/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * BSD 3-Clause License.
 */
 #ifndef OP_API_INC_LEVEL0_OP_SOLVE_TRI_H
 #define OP_API_INC_LEVEL0_OP_SOLVE_TRI_H
 
 #include "opdev/op_executor.h"
 
 namespace l0op {
 const aclTensor* SolveTri(
     const aclTensor *x,
     const aclIntArray *cuSeqlensOptional,
     const aclIntArray *chunkIndicesOptional,
     const char *layout,
     const aclTensor *xOut,
     aclOpExecutor *executor);
 }
 
 #endif
 