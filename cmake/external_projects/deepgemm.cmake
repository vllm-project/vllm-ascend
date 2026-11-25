# SPDX-License-Identifier: Apache-2.0
include(FetchContent)

if(DEFINED ENV{DEEPGEMM_SRC_DIR})
  set(DEEPGEMM_SRC_DIR "$ENV{DEEPGEMM_SRC_DIR}")
endif()
if(DEEPGEMM_SRC_DIR)
  get_filename_component(deepgemm_SOURCE_DIR "${DEEPGEMM_SRC_DIR}" ABSOLUTE BASE_DIR "${CMAKE_SOURCE_DIR}")
else()
  find_package(Git REQUIRED)
  set(_dg_revision 1c825d9362b0de86928ab337e4647cf4a69955a2)
  # This revision is reachable through upstream PR #32, outside the default clone refspec.
  FetchContent_Declare(deepgemm
    DOWNLOAD_COMMAND "${GIT_EXECUTABLE}" init <SOURCE_DIR>
    COMMAND "${GIT_EXECUTABLE}" -C <SOURCE_DIR> fetch --depth=1 https://github.com/deepseek-ai/DeepGEMM-Ascend.git ${_dg_revision}
    COMMAND "${GIT_EXECUTABLE}" -C <SOURCE_DIR> checkout --detach ${_dg_revision}
    COMMAND "${GIT_EXECUTABLE}" -C <SOURCE_DIR> submodule update --init --recursive third-party/deep-jit third-party/magic_enum
    # The dependency's top-level CMakeLists is for IDE indexing.
    SOURCE_SUBDIR __vllm_fetch_only)
  FetchContent_MakeAvailable(deepgemm)
endif()

set(_dg_build "${CMAKE_CURRENT_BINARY_DIR}/deepgemm_C")
set(_dg_builder "${CMAKE_CURRENT_LIST_DIR}/../../tools/deepgemm/build.py")
file(GLOB_RECURSE _dg_inputs CONFIGURE_DEPENDS
  "${deepgemm_SOURCE_DIR}/csrc/*" "${deepgemm_SOURCE_DIR}/deep_gemm/*.py"
  "${deepgemm_SOURCE_DIR}/deep_gemm/include/*" "${deepgemm_SOURCE_DIR}/scripts/*.py"
  "${deepgemm_SOURCE_DIR}/third-party/deep-jit/include/*" "${deepgemm_SOURCE_DIR}/third-party/magic_enum/include/*")
add_custom_command(
  OUTPUT "${_dg_build}/.built"
  COMMAND "${PYTHON_EXECUTABLE}" "${_dg_builder}" "${deepgemm_SOURCE_DIR}" "${_dg_build}"
  COMMAND "${CMAKE_COMMAND}" -E touch "${_dg_build}/.built"
  DEPENDS "${_dg_builder}" "${CMAKE_CURRENT_LIST_FILE}" "${deepgemm_SOURCE_DIR}/setup.py" ${_dg_inputs}
  COMMENT "Building DeepGEMM-Ascend with its native Torch extension"
  VERBATIM)
add_custom_target(_deep_gemm_C DEPENDS "${_dg_build}/.built")
add_dependencies(vllm_ascend_C _deep_gemm_C)

if(NOT VLLM_ASCEND_PACKAGE_DIR)
  set(VLLM_ASCEND_PACKAGE_DIR "${VLLM_ASCEND_INSTALL_PATH}")
endif()
install(DIRECTORY "${_dg_build}/python/deep_gemm/" DESTINATION "${VLLM_ASCEND_PACKAGE_DIR}/third_party/deep_gemm")
