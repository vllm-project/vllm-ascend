# SPDX-License-Identifier: Apache-2.0
# Use the decode blocks and metadata contract from the same installed ops-transformer.
set(qmla_impl "${ASCEND_HOME_PATH}/opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc")
set(qmla_kernel "${qmla_impl}/quant_flash_mla/op_kernel/arch35")
if(EXISTS "${qmla_kernel}/quant_flash_mla_kernel.h")
  file(GLOB qmla_headers CONFIGURE_DEPENDS "${qmla_kernel}/*.h")
  foreach(kernel IN ITEMS cache_write decode)
    set(src "${CMAKE_CURRENT_LIST_DIR}/${kernel}.asc")
    set(obj "${CMAKE_CURRENT_BINARY_DIR}/quant_flash_mla_nz_${kernel}.o")
    add_custom_command(
      OUTPUT "${obj}"
      COMMAND "${ASCEND_HOME_PATH}/bin/bisheng" -x asc --npu-arch=dav-3510 -O2 -fPIC
              --cce-auto-sync=off -DENABLE_CV_COMM_VIA_SSBUF=true -Wno-deprecated-declarations
              "-I${qmla_impl}/common" "-I${qmla_impl}/common/op_kernel" "-I${qmla_kernel}"
              -c "${src}" -o "${obj}"
      DEPENDS "${src}" "${CMAKE_CURRENT_LIST_DIR}/decode.h" ${qmla_headers}
      VERBATIM)
    target_sources(vllm_ascend_C PRIVATE "${obj}")
  endforeach()
  target_sources(vllm_ascend_C PRIVATE "${CMAKE_CURRENT_LIST_DIR}/torch_adapter.cpp")
  target_link_libraries(vllm_ascend_C PRIVATE ascendc_runtime runtime profapi)
endif()
