# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#### CPACK runtime kb to package run #####

set(OPS_RTKB_DIR ops_transformer/built-in/data/op)
set(KB_CATEGORY_LIST "mc2")
set(RTKB_FILE_ALL)
foreach(OP_RTKB_CATEGORY ${KB_CATEGORY_LIST})
    if (IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${OP_RTKB_CATEGORY})
        if (IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${OP_RTKB_CATEGORY}/3rd)
            file(GLOB_RECURSE RTKB_FILE ${CMAKE_CURRENT_SOURCE_DIR}/${OP_RTKB_CATEGORY}/3rd/*/op_host/config/${compute_unit}/*_runtime_kb.json)
            list(APPEND RTKB_FILE_ALL ${RTKB_FILE})
        endif()
    endif()
endforeach()

foreach(RTKB_DATA_FILE ${RTKB_FILE_ALL})
    string(FIND "${RTKB_DATA_FILE}" "/" last_slash_pos REVERSE)
    string(SUBSTRING "${RTKB_DATA_FILE}" ${last_slash_pos} -1 RTKB_DATA_FILE_NAME)
    string(FIND "${RTKB_DATA_FILE_NAME}" "_" first_underscore_pos)
    string(SUBSTRING "${RTKB_DATA_FILE_NAME}" 0 ${first_underscore_pos} fileSocPath)

    install(FILES ${RTKB_DATA_FILE}
        DESTINATION ${OPS_RTKB_DIR}${fileSocPath}/unified_bank
        OPTIONAL
    )
endforeach()