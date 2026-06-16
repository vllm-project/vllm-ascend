add_ops_compile_options(
        OP_NAME SlotMapping
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
                -Werror
)

target_sources(op_host_aclnn PRIVATE
        slot_mapping_def.cpp
)

target_sources(optiling PRIVATE
        slot_mapping_tiling.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_sources(opsproto PRIVATE
        slot_mapping_infershape.cpp
)
