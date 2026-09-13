# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

include_guard(GLOBAL)

if(NOT DEFINED VLLM_ASCEND_BUILD_CACHE_DIR OR "${VLLM_ASCEND_BUILD_CACHE_DIR}" STREQUAL "")
    if(DEFINED ENV{VLLM_ASCEND_BUILD_CACHE_DIR} AND NOT "$ENV{VLLM_ASCEND_BUILD_CACHE_DIR}" STREQUAL "")
        set(
            VLLM_ASCEND_BUILD_CACHE_DIR
            "$ENV{VLLM_ASCEND_BUILD_CACHE_DIR}"
            CACHE PATH
            "vLLM-Ascend local build cache directory"
        )
    else()
        get_filename_component(
            _VLLM_ASCEND_DEFAULT_BUILD_CACHE_DIR
            "${CMAKE_SOURCE_DIR}/../build_cache"
            ABSOLUTE
        )
        set(
            VLLM_ASCEND_BUILD_CACHE_DIR
            "${_VLLM_ASCEND_DEFAULT_BUILD_CACHE_DIR}"
            CACHE PATH
            "vLLM-Ascend local build cache directory"
        )
    endif()
endif()

set(
    VLLM_ASCEND_BUILD_CACHE_SCRIPT
    "${CMAKE_SOURCE_DIR}/scripts/build_cache.py"
    CACHE FILEPATH
    "vLLM-Ascend local build cache helper"
)

function(vllm_ascend_build_cache_command OUT_VAR)
    set(options)
    set(
        oneValueArgs
        DOMAIN
        UNIT
        SOC
        OPERATOR
        ACTION
        OPERATOR_SOURCE
        REPO_ROOT
        OUTPUT_DIR
        PUBLISH_DIR
        PUBLISH_STATE_DIR
        ENVIRONMENT_PROFILE
    )
    set(
        multiValueArgs
        PREPARED_INPUT
        RECIPE_FILE
        RECIPE_VALUE
        ENVIRONMENT_FILE
        ENVIRONMENT_VALUE
        ENVIRONMENT_TOOL
        NORMALIZE_PATH
        ARTIFACT_INCLUDE
        EXCLUDE
        SET_ENV
        COMMAND
    )

    cmake_parse_arguments(
        CACHE
        "${options}"
        "${oneValueArgs}"
        "${multiValueArgs}"
        ${ARGN}
    )

    foreach(required_arg DOMAIN UNIT OUTPUT_DIR ENVIRONMENT_PROFILE)
        if(NOT CACHE_${required_arg})
            message(FATAL_ERROR "vllm_ascend_build_cache_command requires ${required_arg}")
        endif()
    endforeach()

    if(NOT CACHE_COMMAND)
        message(FATAL_ERROR "vllm_ascend_build_cache_command requires COMMAND")
    endif()

    if(CACHE_DOMAIN STREQUAL "custom_operator")
        if(NOT CACHE_OPERATOR_SOURCE)
            message(
                FATAL_ERROR
                "custom_operator cache requires OPERATOR_SOURCE"
            )
        endif()
        if(NOT CACHE_PUBLISH_DIR OR NOT CACHE_PUBLISH_STATE_DIR)
            message(
                FATAL_ERROR
                "custom_operator cache requires PUBLISH_DIR and PUBLISH_STATE_DIR"
            )
        endif()
    endif()

    if(NOT EXISTS "${VLLM_ASCEND_BUILD_CACHE_SCRIPT}")
        message(
            FATAL_ERROR
            "Build cache helper does not exist: ${VLLM_ASCEND_BUILD_CACHE_SCRIPT}"
        )
    endif()

    set(
        _cache_command
        ${HI_PYTHON}
        ${VLLM_ASCEND_BUILD_CACHE_SCRIPT}
        run
        --cache-root
        ${VLLM_ASCEND_BUILD_CACHE_DIR}
        --domain
        ${CACHE_DOMAIN}
        --unit
        ${CACHE_UNIT}
        --output-dir
        ${CACHE_OUTPUT_DIR}
        --environment-profile
        ${CACHE_ENVIRONMENT_PROFILE}
    )

    if(CACHE_PUBLISH_DIR)
        list(APPEND _cache_command --publish-dir ${CACHE_PUBLISH_DIR})
    endif()

    if(CACHE_PUBLISH_STATE_DIR)
        list(APPEND _cache_command --publish-state-dir ${CACHE_PUBLISH_STATE_DIR})
    endif()

    if(CACHE_SOC)
        list(APPEND _cache_command --soc ${CACHE_SOC})
    endif()

    if(CACHE_OPERATOR)
        list(APPEND _cache_command --operator ${CACHE_OPERATOR})
    endif()

    if(CACHE_ACTION)
        list(APPEND _cache_command --action ${CACHE_ACTION})
    endif()

    if(CACHE_OPERATOR_SOURCE)
        list(APPEND _cache_command --operator-source ${CACHE_OPERATOR_SOURCE})
    endif()

    if(CACHE_REPO_ROOT)
        list(APPEND _cache_command --repo-root ${CACHE_REPO_ROOT})
    elseif(CACHE_DOMAIN STREQUAL "custom_operator")
        list(APPEND _cache_command --repo-root ${CMAKE_SOURCE_DIR})
    endif()

    foreach(value ${CACHE_PREPARED_INPUT})
        list(APPEND _cache_command --prepared-input ${value})
    endforeach()

    foreach(value ${CACHE_RECIPE_FILE})
        list(APPEND _cache_command --recipe-file ${value})
    endforeach()

    foreach(value ${CACHE_RECIPE_VALUE})
        list(APPEND _cache_command --recipe-value "${value}")
    endforeach()

    foreach(value ${CACHE_ENVIRONMENT_FILE})
        list(APPEND _cache_command --environment-file ${value})
    endforeach()

    foreach(value ${CACHE_ENVIRONMENT_VALUE})
        list(APPEND _cache_command --environment-value "${value}")
    endforeach()

    foreach(value ${CACHE_ENVIRONMENT_TOOL})
        list(APPEND _cache_command --environment-tool ${value})
    endforeach()

    list(APPEND _cache_command --normalize-path ${CMAKE_SOURCE_DIR})
    list(APPEND _cache_command --normalize-path ${CMAKE_BINARY_DIR})
    foreach(value ${CACHE_NORMALIZE_PATH})
        list(APPEND _cache_command --normalize-path ${value})
    endforeach()

    foreach(value ${CACHE_ARTIFACT_INCLUDE})
        list(APPEND _cache_command --artifact-include "${value}")
    endforeach()

    foreach(value ${CACHE_EXCLUDE})
        list(APPEND _cache_command --exclude "${value}")
    endforeach()

    foreach(value ${CACHE_SET_ENV})
        list(APPEND _cache_command --set-env "${value}")
    endforeach()

    list(APPEND _cache_command --)
    list(APPEND _cache_command ${CACHE_COMMAND})

    set(${OUT_VAR} ${_cache_command} PARENT_SCOPE)
endfunction()
