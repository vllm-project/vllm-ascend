# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

# Install as attn_metadata_update/op_host/CMakeLists.txt.
# Keep the OpDef and handwritten ACLNN files directly in op_host/.

add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnInner PRIVATE
        attn_metadata_update.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME AttnMetadataUpdate
    OPTIONS
        --cce-auto-sync=off
        -Wno-deprecated-declarations
        -Werror
)

if (NOT BUILD_OPS_RTY_KERNEL)
    foreach(_attn_metadata_update_file IN ITEMS
        attn_metadata_update.cpp
        attn_metadata_update_tiling.cpp
        attn_metadata_update_proto.cpp
        aclnn_attn_metadata_update.cpp
        aclnn_attn_metadata_update.h
    )
        if (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_attn_metadata_update_file}")
            message(FATAL_ERROR
                "AttnMetadataUpdate: missing ${CMAKE_CURRENT_SOURCE_DIR}/${_attn_metadata_update_file}. "
                "This configuration requires the original filenames directly in op_host/."
            )
        endif()
    endforeach()

    if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/op_api/aclnn_attn_metadata_update.cpp")
        message(FATAL_ERROR
            "AttnMetadataUpdate: duplicate ACLNN source in op_host/op_api/. "
            "This configuration uses the copy directly in op_host/."
        )
    endif()

    # The helper auto-collects *_def*.cpp. Reject a second OpDef alongside
    # the original attn_metadata_update.cpp, which is added explicitly below.
    file(GLOB _attn_metadata_update_def_sources
        "${CMAKE_CURRENT_SOURCE_DIR}/attn_metadata_update_def*.cpp"
    )
    if (_attn_metadata_update_def_sources)
        message(FATAL_ERROR
            "AttnMetadataUpdate: found attn_metadata_update_def*.cpp alongside the original OpDef. "
            "Keep a single OpDef; this configuration uses attn_metadata_update.cpp."
        )
    endif()

    # Generate the Inner API; retain the handwritten public ACLNN wrapper.
    # This helper also collects tiling sources and creates the object targets.
    add_modules_sources(
        OPTYPE attn_metadata_update
        ACLNNTYPE aclnn_inner
    )

    # The original OpDef filename does not match the helper's *_def*.cpp glob.
    target_sources(${OPHOST_NAME}_opdef_aclnn_inner_obj INTERFACE
        "${CMAKE_CURRENT_SOURCE_DIR}/attn_metadata_update.cpp"
    )

    # Flat ACLNN files are not auto-collected from op_host/.
    target_sources(${OPHOST_NAME}_opapi_obj PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/aclnn_attn_metadata_update.cpp"
    )
    target_include_directories(${OPHOST_NAME}_opapi_obj PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
    )

    target_sources(${OPHOST_NAME}_aclnn_exclude_headers INTERFACE
        "${CMAKE_CURRENT_SOURCE_DIR}/aclnn_attn_metadata_update.h"
    )

    # The helper scans *_infershape*.cpp, not this legacy *_proto.cpp.
    # Preserve the handwritten InferShape/InferDataType registration.
    target_sources(${OPHOST_NAME}_infer_obj PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/attn_metadata_update_proto.cpp"
    )

    target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel"
    )

    message(STATUS
        "AttnMetadataUpdate: handwritten ACLNN=${CMAKE_CURRENT_SOURCE_DIR}/aclnn_attn_metadata_update.cpp; "
        "generated API=aclnn_inner"
    )
endif()
