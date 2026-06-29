# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(_MEGA_MOE_INC_OPTS)
if (EXISTS ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/ascendc/include)
    list(APPEND _MEGA_MOE_INC_OPTS -I${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/ascendc/include)
elseif (EXISTS ${ASCEND_CANN_PACKAGE_PATH}/arm64-linux/ascendc/include)
    list(APPEND _MEGA_MOE_INC_OPTS -I${ASCEND_CANN_PACKAGE_PATH}/arm64-linux/ascendc/include)
elseif (EXISTS ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/ascendc/include)
    list(APPEND _MEGA_MOE_INC_OPTS -I${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/ascendc/include)
endif()

if (BUILD_OPEN_PROJECT)  # custom
    add_op_to_compiled_list()
    target_sources(op_host_aclnnInner PRIVATE
        mega_moe_def.cpp
    )
    add_modules_sources_with_soc(
        OP_MC2_ENABLE ON
        OPTYPE mega_moe ACLNNTYPE aclnn_inner)
    set(SUB_MC2_COMPILE TRUE PARENT_SCOPE)
    set(MC2_OPT ON PARENT_SCOPE)

    # mc2/common sources now live under mega_moe/common_utils/
    add_tiling_modules()
    target_sources(${OPHOST_NAME}_tiling_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../common_utils/mc2_log.cpp
    )

    set(CONDITION_UNIT ${ASCEND_COMPUTE_UNIT})
    if ("${CONDITION_UNIT}" STREQUAL "ascend950")
        add_ops_compile_options(
            OP_NAME MegaMoe
            OPTIONS --cce-auto-sync=off
                    -Wno-c++20-extensions
            OPTIONS ${_MEGA_MOE_INC_OPTS}
            OPTIONS -I${OPTENSOR_INCLUDE_ROOT}
            OPTIONS -I${TENSOR_API_INCLUDE_ROOT}
            OPTIONS -I${TENSOR_API_IMPL_INCLUDE_ROOT}
            OPTIONS -I${CMAKE_CURRENT_SOURCE_DIR}/../common_utils
            OPTIONS -I${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel
            OPTIONS -I${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel/arch35
            OPTIONS -I${CMAKE_CURRENT_SOURCE_DIR}/../../
        )
    else()
        add_ops_compile_options(
            OP_NAME MegaMoe
            OPTIONS --cce-auto-sync=off
                    -Wno-c++20-extensions
            OPTIONS ${_MEGA_MOE_INC_OPTS}
            OPTIONS -I${CMAKE_CURRENT_SOURCE_DIR}/../common_utils
            OPTIONS -I${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel
            OPTIONS -I${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel/arch22
            OPTIONS -DHCCL_COMM
        )
    endif()

else() # 回黄host
    add_mc2_modules_sources(OPTYPE mega_moe ACLNNTYPE aclnn_inner)
    set(SUB_MC2_COMPILE TRUE PARENT_SCOPE)
endif()
