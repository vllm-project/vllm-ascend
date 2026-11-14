# SPDX-License-Identifier: Apache-2.0
add_op_to_compiled_list()
if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE hi_sparse_resolve_residency_def.cpp)
endif()
add_modules_sources(OPTYPE hi_sparse_resolve_residency ACLNNTYPE aclnn)
add_ops_compile_options(
    OP_NAME HiSparseResolveResidency
    OPTIONS -Wno-deprecated-declarations --cce-auto-sync=off
)
