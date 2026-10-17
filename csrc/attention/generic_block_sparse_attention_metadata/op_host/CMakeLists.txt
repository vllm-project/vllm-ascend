# Copyright (c) 2026 Huawei Technologies Co., Ltd.

# Keep the public Generic ACLNN header in the configured install source list.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/op_api/aclnn_generic_block_sparse_attention_metadata.h")

add_op_to_compiled_list()
add_modules_sources(OPTYPE generic_block_sparse_attention_metadata ACLNNTYPE aclnn)
