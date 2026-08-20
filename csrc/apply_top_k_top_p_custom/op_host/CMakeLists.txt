add_ops_compile_options(
    OP_NAME ApplyTopKTopPCustom
    OPTIONS --cce-auto-sync=on
    -Wno-deprecated-declarations
)

target_sources(op_host_aclnnExc PRIVATE
    apply_top_k_top_p_custom_def.cpp
)

target_sources(cust_opapi PRIVATE
    apply_top_k_top_p_custom.cpp
    aclnn_apply_top_k_top_p_custom.cpp
)

if (NOT BUILD_OPEN_PROJECT)
    target_sources(aclnn_ops_train PRIVATE
        apply_top_k_top_p_custom.cpp
        aclnn_apply_top_k_top_p_custom.cpp
    )

    target_sources(aclnn_ops_infer PRIVATE
        apply_top_k_top_p_custom.cpp
        aclnn_apply_top_k_top_p_custom.cpp
    )
endif ()

target_sources(cust_opmaster PRIVATE
    apply_top_k_top_p_custom_tiling.cpp
)

target_include_directories(cust_opmaster PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

file(GLOB _GMM_Aclnn_header "${CMAKE_CURRENT_SOURCE_DIR}/aclnn_apply_top_k_top_p_custom.h")

install(FILES ${_GMM_Aclnn_header}
    DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL
)