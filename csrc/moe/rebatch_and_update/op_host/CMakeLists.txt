# Install as rebatch_and_update/op_host/CMakeLists.txt.
# Keep the OpDef and handwritten ACLNN files directly in op_host/.

add_op_to_compiled_list()

# Dependency paths are relative to csrc, where the build framework uses them.
# Export to nonbsp; its CMakeLists.txt forwards the value to csrc.
set(rebatch_and_update_depends
    nonbsp/combine_and_dispatch_rebatch
    nonbsp/attn_metadata_update
    PARENT_SCOPE
)

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnInner PRIVATE
        rebatch_and_update.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME RebatchAndUpdate
    OPTIONS
        --cce-auto-sync=off
        -Wno-deprecated-declarations
        -Werror
)

if (NOT BUILD_OPS_RTY_KERNEL)
    foreach(_rebatch_and_update_file IN ITEMS
        rebatch_and_update.cpp
        rebatch_and_update_tiling.cpp
        aclnn_rebatch_and_update.cpp
        aclnn_rebatch_and_update.h
    )
        if (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_rebatch_and_update_file}")
            message(FATAL_ERROR
                "RebatchAndUpdate: missing ${CMAKE_CURRENT_SOURCE_DIR}/${_rebatch_and_update_file}. "
                "This configuration requires the original filenames directly in op_host/."
            )
        endif()
    endforeach()

    if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/op_api/aclnn_rebatch_and_update.cpp")
        message(FATAL_ERROR
            "RebatchAndUpdate: duplicate ACLNN source in op_host/op_api/. "
            "This configuration uses the copy directly in op_host/."
        )
    endif()

    # The helper auto-collects *_def*.cpp. Reject a second OpDef alongside
    # the original rebatch_and_update.cpp, which is added explicitly below.
    file(GLOB _rebatch_and_update_def_sources
        "${CMAKE_CURRENT_SOURCE_DIR}/rebatch_and_update_def*.cpp"
    )
    if (_rebatch_and_update_def_sources)
        message(FATAL_ERROR
            "RebatchAndUpdate: found rebatch_and_update_def*.cpp alongside the original OpDef. "
            "Keep a single OpDef; this configuration uses rebatch_and_update.cpp."
        )
    endif()

    # Generate the Inner API; retain the handwritten public ACLNN wrapper.
    # This helper also collects tiling sources and creates the object targets.
    add_modules_sources(
        OPTYPE rebatch_and_update
        ACLNNTYPE aclnn_inner
    )

    # The original OpDef filename does not match the helper's *_def*.cpp glob.
    target_sources(${OPHOST_NAME}_opdef_aclnn_inner_obj INTERFACE
        "${CMAKE_CURRENT_SOURCE_DIR}/rebatch_and_update.cpp"
    )

    # Flat ACLNN files are not auto-collected from op_host/.
    target_sources(${OPHOST_NAME}_opapi_obj PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/aclnn_rebatch_and_update.cpp"
    )
    target_include_directories(${OPHOST_NAME}_opapi_obj PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
    )

    target_sources(${OPHOST_NAME}_aclnn_exclude_headers INTERFACE
        "${CMAKE_CURRENT_SOURCE_DIR}/aclnn_rebatch_and_update.h"
    )

    target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel"
    )

    message(STATUS
        "RebatchAndUpdate: handwritten ACLNN=${CMAKE_CURRENT_SOURCE_DIR}/aclnn_rebatch_and_update.cpp; "
        "generated API=aclnn_inner"
    )
endif()
