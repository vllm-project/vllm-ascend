# SPDX-License-Identifier: Apache-2.0
# CANN's ASC integration supplies the compiler, archive rules and runtime links.
find_package(ASC REQUIRED CONFIG PATHS
    ${ASCEND_HOME_PATH}/lib64/cmake
    ${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/lib64/cmake
    NO_DEFAULT_PATH)
set(CMAKE_ASC_ARCHITECTURES dav-3510)
enable_language(ASC)

ascendc_library(vllm_ascend_hisparse STATIC hisparse_resolver.cpp)
target_include_directories(vllm_ascend_hisparse PRIVATE ${HISPARSE_SIMT_INCLUDE_DIR})
# The AscendC entry launches a SIMT vector function through asc_vf_call. Keep
# the mixed programming mode; --enable-simt selects a pure SIMT translation unit.
target_compile_options(vllm_ascend_hisparse PRIVATE -O2)
# Reserve space for the descriptor and runtime overhead within the target UB.
target_compile_definitions(vllm_ascend_hisparse PRIVATE HISPARSE_RESOLVER_UB_BYTES=175616)
