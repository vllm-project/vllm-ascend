# Copyright (c) 2026 Huawei Technologies Co., Ltd.

# Install this file as moe_token_cut/op_host/CMakeLists.txt.
# The handwritten ACLNN .cpp and .h must remain directly in op_host/.

add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnInner PRIVATE
        moe_token_cut_def.cpp
    )
endif()

# Preserve the original kernel compiler options. Host integration below does
# not change the kernel algorithm or add compiler tuning flags.
add_ops_compile_options(
    OP_NAME MoeTokenCut
    OPTIONS
        --cce-auto-sync=off
        -Wno-deprecated-declarations
        -Werror
)

if (NOT BUILD_OPS_RTY_KERNEL)
    foreach(_moe_token_cut_file IN ITEMS
        moe_token_cut_def.cpp
        moe_token_cut_tiling.cpp
        aclnn_moe_token_cut.cpp
        aclnn_moe_token_cut.h
    )
        if (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_moe_token_cut_file}")
            message(FATAL_ERROR
                "MoeTokenCut: missing ${CMAKE_CURRENT_SOURCE_DIR}/${_moe_token_cut_file}. "
                "This configuration requires the handwritten ACLNN files directly in op_host/."
            )
        endif()
    endforeach()

    # add_modules_sources also scans op_host/op_api/*.cpp. Do not compile two
    # copies of the public ACLNN entry points.
    if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/op_api/aclnn_moe_token_cut.cpp")
        message(FATAL_ERROR
            "MoeTokenCut: duplicate ACLNN source in op_host/op_api/. "
            "This configuration uses the copy directly in op_host/."
        )
    endif()

    # Generate only the Inner API. The public API is the handwritten wrapper.
    # This helper also collects the tiling source and creates the object targets.
    add_modules_sources(
        OPTYPE moe_token_cut
        ACLNNTYPE aclnn_inner
    )

    # Unlike files under op_host/op_api/, this flat source is not auto-collected.
    target_sources(${OPHOST_NAME}_opapi_obj PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/aclnn_moe_token_cut.cpp"
    )
    target_include_directories(${OPHOST_NAME}_opapi_obj PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
    )

    # Let the new framework package the handwritten public API header.
    target_sources(${OPHOST_NAME}_aclnn_exclude_headers INTERFACE
        "${CMAKE_CURRENT_SOURCE_DIR}/aclnn_moe_token_cut.h"
    )

    target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel"
    )

    message(STATUS
        "MoeTokenCut: handwritten ACLNN=${CMAKE_CURRENT_SOURCE_DIR}/aclnn_moe_token_cut.cpp; "
        "generated API=aclnn_inner"
    )
endif()
