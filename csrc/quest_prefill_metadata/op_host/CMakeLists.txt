add_ops_compile_options(
        OP_NAME QuestPrefillMetadata
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
                -mllvm -cce-aicore-hoist-movemask=false
                --op_relocatable_kernel_binary=true
)

target_sources(op_host_aclnn PRIVATE
        quest_prefill_metadata_def.cpp
)

target_sources(optiling PRIVATE
        quest_prefill_metadata_tiling.cpp
)

if (NOT BUILD_OPEN_PROJECT)
    target_sources(opmaster_ct PRIVATE
        quest_prefill_metadata_tiling.cpp
    )
endif ()

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_sources(opsproto PRIVATE
        quest_prefill_metadata_proto.cpp
)
