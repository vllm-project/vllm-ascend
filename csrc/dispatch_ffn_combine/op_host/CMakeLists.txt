# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

set(_DISPATCH_FFN_INC_OPTS)
if (EXISTS ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/ascendc/include)
    list(APPEND _DISPATCH_FFN_INC_OPTS -I${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/ascendc/include)
elseif (EXISTS ${ASCEND_CANN_PACKAGE_PATH}/arm64-linux/ascendc/include)
    list(APPEND _DISPATCH_FFN_INC_OPTS -I${ASCEND_CANN_PACKAGE_PATH}/arm64-linux/ascendc/include)
elseif (EXISTS ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/ascendc/include)
    list(APPEND _DISPATCH_FFN_INC_OPTS -I${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/ascendc/include)
endif()
if (EXISTS ${CMAKE_SOURCE_DIR}/third_party/catlass/include)
    list(APPEND _DISPATCH_FFN_INC_OPTS -I${CMAKE_SOURCE_DIR}/third_party/catlass/include)
endif()

add_ops_compile_options(
        OP_NAME DispatchFFNCombine
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
                -Werror
                -DHCCL_COMM
                ${_DISPATCH_FFN_INC_OPTS}
)

target_sources(op_host_aclnnInner PRIVATE
        dispatch_ffn_combine_def.cpp
)

target_sources(opapi PRIVATE
        aclnn_dispatch_ffn_combine.cpp
)

if (NOT BUILD_OPEN_PROJECT)
    target_sources(aclnn_ops_train PRIVATE
        aclnn_dispatch_ffn_combine.cpp
    )

    target_sources(aclnn_ops_infer PRIVATE
        aclnn_dispatch_ffn_combine.cpp
    )
endif ()

target_sources(optiling PRIVATE
        dispatch_ffn_combine_tiling.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel
)

target_sources(opsproto PRIVATE
        dispatch_ffn_combine_proto.cpp
)

file(GLOB _GMM_Aclnn_header "${CMAKE_CURRENT_SOURCE_DIR}/aclnn_dispatch_ffn_combine.h")

install(FILES ${_GMM_Aclnn_header}
        DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL
)
