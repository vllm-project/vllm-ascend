# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

set(_DISPATCH_GMM_INC_OPTS)
if (EXISTS ${CMAKE_SOURCE_DIR}/third_party/catlass/include)
    list(APPEND _DISPATCH_GMM_INC_OPTS -I${CMAKE_SOURCE_DIR}/third_party/catlass/include)
else()
    message(FATAL_ERROR "dependency catlass is missing, you can fetch it by running 'git submodule update --init --recursive'")
endif()

add_ops_compile_options(
        OP_NAME DispatchGmmCombineDecode
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
                ${_DISPATCH_GMM_INC_OPTS}
)

target_sources(op_host_aclnnInner PRIVATE
        dispatch_gmm_combine_decode_def.cpp
)

target_sources(opapi PRIVATE
        aclnn_dispatch_gmm_combine_decode.cpp
)

if (NOT BUILD_OPEN_PROJECT)
    target_sources(aclnn_ops_train PRIVATE
        aclnn_dispatch_gmm_combine_decode.cpp
    )

    target_sources(aclnn_ops_infer PRIVATE
        aclnn_dispatch_gmm_combine_decode.cpp
    )
endif ()

target_sources(optiling PRIVATE
        dispatch_gmm_combine_decode_tiling.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_sources(opsproto PRIVATE
        dispatch_gmm_combine_decode_proto.cpp
)

file(GLOB _GMM_Aclnn_header "${CMAKE_CURRENT_SOURCE_DIR}/aclnn_dispatch_gmm_combine_decode.h")

install(FILES ${_GMM_Aclnn_header}
        DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL
)
