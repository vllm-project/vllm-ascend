# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

# Install as combine_and_dispatch_rebatch/op_host/CMakeLists.txt.
# Keep the OpDef and handwritten ACLNN files directly in op_host/.

add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnInner PRIVATE
        combine_and_dispatch_rebatch.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME CombineAndDispatchRebatch
    OPTIONS
        --cce-auto-sync=off
        -Wno-deprecated-declarations
        -Werror
)

if (NOT BUILD_OPS_RTY_KERNEL)
    foreach(_combine_and_dispatch_rebatch_file IN ITEMS
        combine_and_dispatch_rebatch.cpp
        combine_and_dispatch_rebatch_tiling.cpp
        aclnn_combine_and_dispatch_rebatch.cpp
        aclnn_combine_and_dispatch_rebatch.h
    )
        if (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_combine_and_dispatch_rebatch_file}")
            message(FATAL_ERROR
                "CombineAndDispatchRebatch: missing ${CMAKE_CURRENT_SOURCE_DIR}/${_combine_and_dispatch_rebatch_file}. "
                "This configuration requires the original filenames directly in op_host/."
            )
        endif()
    endforeach()

    if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/op_api/aclnn_combine_and_dispatch_rebatch.cpp")
        message(FATAL_ERROR
            "CombineAndDispatchRebatch: duplicate ACLNN source in op_host/op_api/. "
            "This configuration uses the copy directly in op_host/."
        )
    endif()

    # The helper auto-collects *_def*.cpp. Reject a second OpDef alongside
    # the original combine_and_dispatch_rebatch.cpp, which is added explicitly below.
    file(GLOB _combine_and_dispatch_rebatch_def_sources
        "${CMAKE_CURRENT_SOURCE_DIR}/combine_and_dispatch_rebatch_def*.cpp"
    )
    if (_combine_and_dispatch_rebatch_def_sources)
        message(FATAL_ERROR
            "CombineAndDispatchRebatch: found combine_and_dispatch_rebatch_def*.cpp alongside the original OpDef. "
            "Keep a single OpDef; this configuration uses combine_and_dispatch_rebatch.cpp."
        )
    endif()

    # Generate the Inner API; retain the handwritten public ACLNN wrapper.
    # This helper also collects tiling sources and creates the object targets.
    add_modules_sources(
        OPTYPE combine_and_dispatch_rebatch
        ACLNNTYPE aclnn_inner
    )

    # The original OpDef filename does not match the helper's *_def*.cpp glob.
    target_sources(${OPHOST_NAME}_opdef_aclnn_inner_obj INTERFACE
        "${CMAKE_CURRENT_SOURCE_DIR}/combine_and_dispatch_rebatch.cpp"
    )

    # Flat ACLNN files are not auto-collected from op_host/.
    target_sources(${OPHOST_NAME}_opapi_obj PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/aclnn_combine_and_dispatch_rebatch.cpp"
    )
    target_include_directories(${OPHOST_NAME}_opapi_obj PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
    )

    target_sources(${OPHOST_NAME}_aclnn_exclude_headers INTERFACE
        "${CMAKE_CURRENT_SOURCE_DIR}/aclnn_combine_and_dispatch_rebatch.h"
    )

    target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel"
    )

    message(STATUS
        "CombineAndDispatchRebatch: handwritten ACLNN=${CMAKE_CURRENT_SOURCE_DIR}/aclnn_combine_and_dispatch_rebatch.cpp; "
        "generated API=aclnn_inner"
    )
endif()
