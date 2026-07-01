add_op_to_compiled_list()

add_ops_compile_options(
        OP_NAME QuestBlockSelectPaged
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
                -mllvm -cce-aicore-hoist-movemask=false
                --op_relocatable_kernel_binary=true
)

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
        quest_block_select_paged_def.cpp
    )
endif ()

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE quest_block_select_paged ACLNNTYPE aclnn)
    target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()
