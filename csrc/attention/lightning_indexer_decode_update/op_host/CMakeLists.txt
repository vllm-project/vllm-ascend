add_op_to_compiled_list()

add_ops_compile_options(
        OP_NAME LightningIndexerDecodeUpdate
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
                -mllvm -cce-aicore-hoist-movemask=false
                --op_relocatable_kernel_binary=true
)

target_sources(op_host_aclnnInner PRIVATE
        lightning_indexer_decode_update_def.cpp
)

target_sources(optiling PRIVATE
        lightning_indexer_decode_update_tiling.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_sources(opsproto PRIVATE
        lightning_indexer_decode_update_infershape.cpp
)

target_sources(opapi PRIVATE
        op_api/aclnn_lightning_indexer_decode_update.cpp
)

target_include_directories(opapi PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_api
)

install(FILES op_api/aclnn_lightning_indexer_decode_update.h
        DESTINATION ${ACLNN_INC_INSTALL_DIR}
)
