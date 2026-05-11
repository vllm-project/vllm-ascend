# Copyright (c) 2026 Huawei Technologies Co., Ltd.

add_ops_compile_options(
        OP_NAME RecurrentGatedDeltaRule
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
                -Werror
)

target_sources(op_host_aclnnExc PRIVATE
    recurrent_gated_delta_rule_def.cpp
)

target_sources(opapi PRIVATE
    aclnn_recurrent_gated_delta_rule.cpp
    recurrent_gated_delta_rule.cpp
)

if (NOT BUILD_OPEN_PROJECT)
    target_sources(aclnn_ops_train PRIVATE
        recurrent_gated_delta_rule.cpp
        aclnn_recurrent_gated_delta_rule.cpp
    )

    target_sources(aclnn_ops_infer PRIVATE
        recurrent_gated_delta_rule.cpp
        aclnn_recurrent_gated_delta_rule.cpp
    )
endif ()

target_sources(optiling PRIVATE
    recurrent_gated_delta_rule_tiling.cpp
)

target_include_directories(optiling PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_sources(opsproto PRIVATE)

file(GLOB _RGDR_Aclnn_header "${CMAKE_CURRENT_SOURCE_DIR}/aclnn_recurrent_gated_delta_rule.h")

install(FILES ${_RGDR_Aclnn_header}
        DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL
)
