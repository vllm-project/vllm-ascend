# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
set(OPTENSOR_TAG_ID a524601b399f0e2b6fb3824ffb777d2617d520ae)

if(EXISTS "${PROJECT_SOURCE_DIR}/../ops-tensor")
  get_filename_component(OPTENSOR_SOURCE_PATH
                         ${PROJECT_SOURCE_DIR}/../ops-tensor REALPATH)
  message(STATUS "Find ops-tensor source dir: ${OPTENSOR_SOURCE_PATH}")
elseif(EXISTS "${CANN_3RD_LIB_PATH}/ops-tensor")
  get_filename_component(OPTENSOR_SOURCE_PATH
                         ${CANN_3RD_LIB_PATH}/ops-tensor REALPATH)
  message(STATUS "Find ops-tensor source dir: ${OPTENSOR_SOURCE_PATH}")

  execute_process(
    COMMAND git fetch origin ${OPTENSOR_TAG_ID} --depth=1
    WORKING_DIRECTORY ${OPTENSOR_SOURCE_PATH}
    TIMEOUT 20
    RESULT_VARIABLE FETCH_RESULT
    ERROR_VARIABLE FETCH_ERROR
    OUTPUT_QUIET
  )

  if(FETCH_RESULT EQUAL 0)
    set(CHECKOUT_REF "FETCH_HEAD")
    message(STATUS "ops-tensor fetched ${OPTENSOR_TAG_ID}")
  else()
    set(CHECKOUT_REF "${OPTENSOR_TAG_ID}")
    message(WARNING "Git fetch failed (network unavailable?), checkout to cached ${OPTENSOR_TAG_ID}")
  endif()

  execute_process(
    COMMAND git checkout ${CHECKOUT_REF}
    WORKING_DIRECTORY ${OPTENSOR_SOURCE_PATH}
    RESULT_VARIABLE CHECKOUT_RESULT
    ERROR_VARIABLE CHECKOUT_ERROR
    OUTPUT_QUIET
  )
  if(NOT CHECKOUT_RESULT EQUAL 0)
    message(FATAL_ERROR "Git checkout failed: ${CHECKOUT_ERROR}")
  endif()
else()
  execute_process(
    COMMAND git remote get-url origin
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_REMOTE_URL
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
  if(GIT_REMOTE_URL MATCHES "^git@|^ssh://")
    set(OPTENSOR_GIT_URL "git@gitcode.com:cann/ops-tensor.git")
    message(STATUS "[ThirdPartyLib][ops-tensor] using SSH protocol: ${OPTENSOR_GIT_URL}")
  else()
    set(OPTENSOR_GIT_URL "https://gitcode.com/cann/ops-tensor.git")
    message(STATUS "[ThirdPartyLib][ops-tensor] using HTTPS protocol: ${OPTENSOR_GIT_URL}")
  endif()
  include(FetchContent)

  FetchContent_Declare(
    ops-tensor
    GIT_REPOSITORY ${OPTENSOR_GIT_URL}
    GIT_TAG ${OPTENSOR_TAG_ID}
    GIT_PROGRESS TRUE
    SOURCE_DIR ${CANN_3RD_LIB_PATH}/ops-tensor)

  FetchContent_Populate(ops-tensor)

  set(OPTENSOR_SOURCE_PATH ${CANN_3RD_LIB_PATH}/ops-tensor)

endif()

set(OPTENSOR_INCLUDE_ROOT ${OPTENSOR_SOURCE_PATH}/include)
set(TENSOR_API_INCLUDE_ROOT ${OPTENSOR_SOURCE_PATH}/include/tensor_api/include)
set(TENSOR_API_IMPL_INCLUDE_ROOT ${OPTENSOR_SOURCE_PATH}/include/tensor_api)
