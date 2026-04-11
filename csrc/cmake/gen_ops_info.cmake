# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

###################################################################################################
# copy kernel src to tbe/ascendc path
###################################################################################################
function(kernel_src_copy)
  set(oneValueArgs TARGET DST_DIR)
  set(multiValueArgs IMPL_DIR)
  cmake_parse_arguments(KNCPY "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  add_custom_target(${KNCPY_TARGET})
  foreach(OP_DIR ${KNCPY_IMPL_DIR})
    get_filename_component(OP_NAME ${OP_DIR} NAME)
    message(STATUS "start copy kernel file: ${OP_NAME} to ${KNCPY_DST_DIR}")
    if(NOT TARGET ${OP_NAME}_src_copy)
      set(SRC_DIR ${OP_DIR}/op_kernel)
      if(NOT EXISTS ${SRC_DIR})
        continue()
      endif()
      add_custom_target(${OP_NAME}_src_copy
        COMMAND ${CMAKE_COMMAND} -E make_directory ${KNCPY_DST_DIR}/${OP_NAME}
        COMMAND bash -c "find ${SRC_DIR} -mindepth 1 -maxdepth 1 -exec cp -r {} ${KNCPY_DST_DIR}/${OP_NAME} \\;"
        VERBATIM
      )
      add_dependencies(${KNCPY_TARGET} ${OP_NAME}_src_copy)
      if(ENABLE_PACKAGE)
        install(
          DIRECTORY ${SRC_DIR}/
          DESTINATION ${IMPL_INSTALL_DIR}/${OP_NAME}
        )
      endif()
    endif()
  endforeach()
endfunction()

###################################################################################################
# generate operator dynamic python script for compile, generenate out path ${CMAKE_BINARY_DIR}/tbe,
# and install to packages/vendors/${VENDOR_NAME}_transformer/op_impl/ai_core/tbe/${VENDOR_NAME}_impl/dynamic
###################################################################################################
function(add_ops_impl_target)
  set(oneValueArgs TARGET OPS_INFO_DIR IMPL_DIR OUT_DIR INSTALL_DIR)
  cmake_parse_arguments(OPIMPL "" "${oneValueArgs}" "OPS_BATCH;OPS_ITERATE" ${ARGN})

  add_custom_command(OUTPUT ${OPIMPL_OUT_DIR}/.impl_timestamp
    COMMAND mkdir -m 700 -p ${OPIMPL_OUT_DIR}/dynamic
    COMMAND ${ASCEND_PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/util/ascendc_impl_build.py
            \"\" \"${OPIMPL_OPS_BATCH}\" \"${OPIMPL_OPS_ITERATE}\"
            ${OPIMPL_IMPL_DIR} ${OPIMPL_OUT_DIR}/dynamic ${ASCEND_AUTOGEN_PATH}
            --opsinfo-dir ${OPIMPL_OPS_INFO_DIR} ${OPIMPL_OPS_INFO_DIR}/inner ${OPIMPL_OPS_INFO_DIR}/exc
    COMMAND rm -rf ${OPIMPL_OUT_DIR}/.impl_timestamp
    COMMAND touch ${OPIMPL_OUT_DIR}/.impl_timestamp
    DEPENDS ${CMAKE_SOURCE_DIR}/scripts/util/ascendc_impl_build.py
  )
  add_custom_target(${OPIMPL_TARGET} ALL
    DEPENDS ${OPIMPL_OUT_DIR}/.impl_timestamp
  )

  file(GLOB dynamic_impl ${OPIMPL_OUT_DIR}/dynamic/*.py)
  if(ENABLE_PACKAGE)
    install(
      FILES ${dynamic_impl}
      DESTINATION ${OPIMPL_INSTALL_DIR}
      OPTIONAL
    )
  endif()
endfunction()

###################################################################################################
# generate aic-${compute_unit}-ops-info.json from aic-${compute_unit}-ops-info.ini
# generate outpath: ${CMAKE_BINARY_DIR}/tbe/op_info_cfg/ai_core/${compute_unit}/
# install path: packages/vendors/${VENDOR_NAME}_transformer/op_impl/ai_core/tbe/config/${compute_unit}
###################################################################################################
function(add_ops_info_target)
  set(oneValueArgs TARGET OPS_INFO_DIR COMPUTE_UNIT OUTPUT INSTALL_DIR)
  cmake_parse_arguments(OPINFO "" "${oneValueArgs}" "" ${ARGN})
  get_filename_component(opinfo_file_path "${OPINFO_OUTPUT}" DIRECTORY)
  add_custom_command(OUTPUT ${OPINFO_OUTPUT}
    COMMAND mkdir -p ${opinfo_file_path}
    COMMAND ${ASCEND_PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/util/parse_ini_to_json.py
            ${OPINFO_OPS_INFO_DIR}/aic-${OPINFO_COMPUTE_UNIT}-ops-info.ini
            ${OPINFO_OPS_INFO_DIR}/inner/aic-${OPINFO_COMPUTE_UNIT}-ops-info.ini
            ${OPINFO_OPS_INFO_DIR}/exc/aic-${OPINFO_COMPUTE_UNIT}-ops-info.ini
            ${OPINFO_OUTPUT}
  )
  add_custom_target(${OPINFO_TARGET} ALL
    DEPENDS ${OPINFO_OUTPUT}
  )

  if(ENABLE_PACKAGE)
    install(FILES ${OPINFO_OUTPUT}
      DESTINATION ${OPINFO_INSTALL_DIR}
    )
  endif()
endfunction()

###################################################################################################
# merge ops info ini in aclnn/aclnn_inner/aclnn_exc to a total ini file
# srcpath: ${ASCEND_AUTOGEN_PATH}
# generate outpath: ${CMAKE_BINARY_DIR}/tbe/config
###################################################################################################
function(merge_ini_files)
  set(oneValueArgs TARGET OPS_INFO_DIR COMPUTE_UNIT)
  cmake_parse_arguments(MGINI "" "${oneValueArgs}" "" ${ARGN})
  add_custom_command(OUTPUT ${ASCEND_KERNEL_CONF_DST}/aic-${MGINI_COMPUTE_UNIT}-ops-info.ini
                    COMMAND touch ${MGINI_OPS_INFO_DIR}/aic-merged-${MGINI_COMPUTE_UNIT}-ops-info.ini
                    COMMAND ${ASCEND_PYTHON_EXECUTABLE} ${OPS_KERNEL_BINARY_SCRIPT}/merge_ini_files.py
                            ${MGINI_OPS_INFO_DIR}/aic-${MGINI_COMPUTE_UNIT}-ops-info.ini
                            ${MGINI_OPS_INFO_DIR}/inner/aic-${MGINI_COMPUTE_UNIT}-ops-info.ini
                            ${MGINI_OPS_INFO_DIR}/exc/aic-${MGINI_COMPUTE_UNIT}-ops-info.ini
                            --output-file ${ASCEND_KERNEL_CONF_DST}/aic-${MGINI_COMPUTE_UNIT}-ops-info.ini
    )
  add_custom_target(${MGINI_TARGET} ALL
                    DEPENDS ${ASCEND_KERNEL_CONF_DST}/aic-${MGINI_COMPUTE_UNIT}-ops-info.ini
  )
endfunction()

# ##################################################################################################
# merge ops proto headers in aclnn/aclnn_inner/aclnn_exc to a total proto file
# srcpath: ${ASCEND_AUTOGEN_PATH}
# generate outpath: ${CMAKE_BINARY_DIR}/tbe/graph
# ##################################################################################################
function(merge_graph_headers)
  set(oneValueArgs TARGET OUT_DIR)
  cmake_parse_arguments(MGPROTO "" "${oneValueArgs}" "" ${ARGN})
  get_target_property(proto_headers ${GRAPH_PLUGIN_NAME}_proto_headers INTERFACE_SOURCES)
  add_custom_command(OUTPUT ${MGPROTO_OUT_DIR}/ops_proto_math.h
    COMMAND ${ASCEND_PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/util/merge_proto.py
    ${proto_headers}
    --output-file ${MGPROTO_OUT_DIR}/ops_proto_math.h
  )
  add_custom_target(${MGPROTO_TARGET} ALL
    DEPENDS ${MGPROTO_OUT_DIR}/ops_proto_math.h
  )
endfunction()

###################################################################################################
# generate binary compile shell script and binary json
# srcpath: ${ASCEND_AUTOGEN_PATH}
# outpath: ${CMAKE_BINARY_DIR}/binary/${compute_unit}
###################################################################################################
function(generate_bin_scripts)
  set(oneValueArgs TARGET OP_NAME OPS_INFO_DIR COMPUTE_UNIT OUT_DIR)
  cmake_parse_arguments(GENBIN "" "${oneValueArgs}" "" ${ARGN})
  file(MAKE_DIRECTORY ${GENBIN_OUT_DIR}/gen)
  file(MAKE_DIRECTORY ${GENBIN_OUT_DIR}/gen/${GENBIN_OP_NAME})
  message(STATUS "start generate_bin_scripts for op: ${GENBIN_OP_NAME}")
  add_custom_target(generate_bin_scripts_${GENBIN_COMPUTE_UNIT}_${GENBIN_OP_NAME}
                    COMMAND ${ASCEND_PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/util/ascendc_bin_param_build.py
                            ${GENBIN_OPS_INFO_DIR}/aic-${GENBIN_COMPUTE_UNIT}-ops-info.ini
                            ${GENBIN_OUT_DIR}/gen/${GENBIN_OP_NAME} ${GENBIN_COMPUTE_UNIT}
                            --opc-config-file ${ASCEND_AUTOGEN_PATH}/${CUSTOM_OPC_OPTIONS}
                            --ops ${GENBIN_OP_NAME}
                    COMMAND ${ASCEND_PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/util/ascendc_bin_param_build.py
                            ${GENBIN_OPS_INFO_DIR}/inner/aic-${GENBIN_COMPUTE_UNIT}-ops-info.ini
                            ${GENBIN_OUT_DIR}/gen/${GENBIN_OP_NAME} ${GENBIN_COMPUTE_UNIT}
                            --opc-config-file ${ASCEND_AUTOGEN_PATH}/${CUSTOM_OPC_OPTIONS}
                            --ops ${GENBIN_OP_NAME}
                    COMMAND ${ASCEND_PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/util/ascendc_bin_param_build.py
                            ${GENBIN_OPS_INFO_DIR}/exc/aic-${GENBIN_COMPUTE_UNIT}-ops-info.ini
                            ${GENBIN_OUT_DIR}/gen/${GENBIN_OP_NAME} ${GENBIN_COMPUTE_UNIT}
                            --opc-config-file ${ASCEND_AUTOGEN_PATH}/${CUSTOM_OPC_OPTIONS}
                            --ops ${GENBIN_OP_NAME}
  )
  if(NOT TARGET ${GENBIN_TARGET})
    add_custom_target(${GENBIN_TARGET})
  endif()
  add_dependencies(${GENBIN_TARGET} generate_bin_scripts_${GENBIN_COMPUTE_UNIT}_${GENBIN_OP_NAME}
  )
endfunction()

###################################################################################################
# copy binary config from op_host/config to tbe/config path
###################################################################################################
function(binary_config_copy)
  set(oneValueArgs TARGET OP_NAME CONF_DIR DST_DIR COMPUTE_UNIT)
  cmake_parse_arguments(CNFCPY "" "${oneValueArgs}" "" ${ARGN})
  file(MAKE_DIRECTORY ${CNFCPY_DST_DIR}/${CNFCPY_COMPUTE_UNIT}/${CNFCPY_OP_NAME})
  add_custom_target(${CNFCPY_TARGET}
    COMMAND rm -rf ${CNFCPY_DST_DIR}/${CNFCPY_COMPUTE_UNIT}/${CNFCPY_OP_NAME}/*
    COMMAND cp -r ${CNFCPY_CONF_DIR}/${CNFCPY_COMPUTE_UNIT}/* ${CNFCPY_DST_DIR}/${CNFCPY_COMPUTE_UNIT}/${CNFCPY_OP_NAME}
  )
endfunction()

###################################################################################################
# compile binary from op_host/config binary json files
# generate outpath: ${CMAKE_BINARY_DIR}/binary/${compute_unit}/bin
# install path: ${BIN_KERNEL_INSTALL_DIR}/${compute_unit}
###################################################################################################
function(compile_from_config)
  set(oneValueArgs TARGET OP_NAME OPS_INFO_DIR IMPL_DIR CONFIG_DIR OP_PYTHON_DIR OUT_DIR INSTALL_DIR COMPUTE_UNIT)
  cmake_parse_arguments(CONFCMP "" "${oneValueArgs}" "" ${ARGN})
  file(MAKE_DIRECTORY ${CONFCMP_OUT_DIR}/src)
  file(MAKE_DIRECTORY ${CONFCMP_OUT_DIR}/bin)
  file(MAKE_DIRECTORY ${CONFCMP_OUT_DIR}/gen)
  snake_to_camel("${CONFCMP_OP_NAME}" OP_TYPE)
  message(STATUS "start to compile op: ${CONFCMP_OP_NAME}, op_type: ${OP_TYPE}")
  # add Environment Variable Configurations of python & ccache
  set(_ASCENDC_ENV_VAR)
  list(APPEND _ASCENDC_ENV_VAR export HI_PYTHON=${ASCEND_PYTHON_EXECUTABLE} &&)
  # whether need judging CMAKE_C_COMPILER_LAUNCHER
  if(${CMAKE_CXX_COMPILER_LAUNCHER} MATCHES "ccache$")
    list(APPEND _ASCENDC_ENV_VAR export ASCENDC_CCACHE_EXECUTABLE=${CMAKE_CXX_COMPILER_LAUNCHER} &&)
  endif()
  # copy binary config file to tbe/config
  binary_config_copy(
    TARGET bin_conf_${CONFCMP_OP_NAME}_${CONFCMP_COMPUTE_UNIT}_copy
    OP_NAME ${CONFCMP_OP_NAME}
    CONF_DIR ${CONFCMP_CONFIG_DIR}
    DST_DIR ${ASCEND_KERNEL_CONF_DST}
    COMPUTE_UNIT ${CONFCMP_COMPUTE_UNIT}
  )

  add_custom_target(config_compile_${CONFCMP_COMPUTE_UNIT}_${CONFCMP_OP_NAME}
    COMMAND ${_ASCENDC_ENV_VAR} bash ${OPS_KERNEL_BINARY_SCRIPT}/build_binary_single_op.sh
            ${OP_TYPE}
            ${CONFCMP_COMPUTE_UNIT}
            ${CONFCMP_OUT_DIR}/bin
    WORKING_DIRECTORY ${OPS_KERNEL_BINARY_SCRIPT}
    DEPENDS ${ASCEND_KERNEL_CONF_DST}/aic-${CONFCMP_COMPUTE_UNIT}-ops-info.ini
            ascendc_kernel_src_copy
            bin_conf_${CONFCMP_OP_NAME}_${CONFCMP_COMPUTE_UNIT}_copy
  )

  if(NOT TARGET binary)
    add_custom_target(binary)
  endif()
  add_custom_target(${CONFCMP_TARGET}
    COMMAND cp -r ${CONFCMP_IMPL_DIR}/*.* ${CONFCMP_OUT_DIR}/src
    COMMAND cp ${CONFCMP_OP_PYTHON_DIR}/${CONFCMP_OP_NAME}.py ${CONFCMP_OUT_DIR}/src
  )
  add_dependencies(binary config_compile_${CONFCMP_COMPUTE_UNIT}_${CONFCMP_OP_NAME} ${CONFCMP_TARGET})

  if(ENABLE_PACKAGE)
    install(DIRECTORY ${CONFCMP_OUT_DIR}/bin/${CONFCMP_COMPUTE_UNIT}/${CONFCMP_OP_NAME}
      DESTINATION ${BIN_KERNEL_INSTALL_DIR}/${CONFCMP_COMPUTE_UNIT} OPTIONAL
    )
    install(FILES ${CONFCMP_OUT_DIR}/bin/config/${CONFCMP_COMPUTE_UNIT}/${CONFCMP_OP_NAME}.json
      DESTINATION ${BIN_KERNEL_CONFIG_INSTALL_DIR}/${CONFCMP_COMPUTE_UNIT} OPTIONAL
    )
  endif()
endfunction()

###################################################################################################
# generate binary_info_config.json
# generate outpath: ${CMAKE_BINARY_DIR}/binary/${compute_unit}/bin/config
# install path: packages/vendors/${VENDOR_NAME}_transformer/op_impl/ai_core/tbe/kernel/config
###################################################################################################
function(gen_binary_info_config_json)
  set(oneValueArgs TARGET BIN_DIR COMPUTE_UNIT)
  cmake_parse_arguments(GENBIN_INFOCFG "" "${oneValueArgs}" "" ${ARGN})

  add_custom_command(OUTPUT ${GENBIN_INFOCFG_BIN_DIR}/bin/config/${GENBIN_INFOCFG_COMPUTE_UNIT}/binary_info_config.json
    COMMAND ${ASCEND_PYTHON_EXECUTABLE} ${OPS_KERNEL_BINARY_SCRIPT}/gen_binary_info_config.py
            ${GENBIN_INFOCFG_BIN_DIR}/bin
            ${GENBIN_INFOCFG_COMPUTE_UNIT}
    DEPENDS ${GENBIN_INFOCFG_BIN_DIR}/bin/config
  )
  add_custom_target(${GENBIN_INFOCFG_TARGET}
    DEPENDS ${GENBIN_INFOCFG_BIN_DIR}/bin/config/${GENBIN_INFOCFG_COMPUTE_UNIT}/binary_info_config.json
  )

  if(NOT TARGET gen_bin_info_config)
    add_custom_target(gen_bin_info_config)
  endif()
  add_dependencies(gen_bin_info_config ${GENBIN_INFOCFG_TARGET})

  if(ENABLE_PACKAGE)
    install(
      FILES ${GENBIN_INFOCFG_BIN_DIR}/bin/config/${GENBIN_INFOCFG_COMPUTE_UNIT}/binary_info_config.json
      DESTINATION ${BIN_KERNEL_CONFIG_INSTALL_DIR}/${GENBIN_INFOCFG_COMPUTE_UNIT} OPTIONAL
    )
  endif()
endfunction()

# binary compile
function(gen_ops_info_and_python)
  gen_aclnn_with_opdef()
  if(NOT TARGET opbuild_custom_gen_aclnn_all)
    message(STATUS "no need build binary, for all the ops donot have any operator def")
    return()
  endif()

  kernel_src_copy(
    TARGET ascendc_kernel_src_copy
    OP_LIST ${COMPILED_OPS}
    IMPL_DIR ${COMPILED_OP_DIRS}
    DST_DIR ${ASCEND_KERNEL_SRC_DST}
  )

  add_ops_impl_target(
    TARGET ascendc_impl_gen
    OPS_INFO_DIR ${ASCEND_AUTOGEN_PATH}
    IMPL_DIR ${ASCEND_KERNEL_SRC_DST}
    OUT_DIR ${CMAKE_BINARY_DIR}/tbe
    INSTALL_DIR ${IMPL_DYNAMIC_INSTALL_DIR}
  )

  merge_graph_headers(
    TARGET merge_ops_proto ALL
    OUT_DIR ${ASCEND_GRAPH_CONF_DST}
  )

  set(ascendc_impl_gen_depends ascendc_kernel_src_copy opbuild_custom_gen_aclnn_all)
  foreach(compute_unit ${ASCEND_COMPUTE_UNIT})
    # generate aic-${compute_unit}-ops-info.json, operator infos
    add_ops_info_target(
      TARGET ops_info_gen_${compute_unit}
      OUTPUT ${CMAKE_BINARY_DIR}/tbe/op_info_cfg/ai_core/${compute_unit}/aic-${compute_unit}-ops-info.json
      OPS_INFO_DIR ${ASCEND_AUTOGEN_PATH}
      COMPUTE_UNIT ${compute_unit}
      INSTALL_DIR ${OPS_INFO_INSTALL_DIR}
    )

    # merge ops info ini files
    merge_ini_files(TARGET merge_ini_${compute_unit}
        OPS_INFO_DIR ${ASCEND_AUTOGEN_PATH}
        COMPUTE_UNIT ${compute_unit}
    )
    list(APPEND ascendc_impl_gen_depends ops_info_gen_${compute_unit})
  endforeach()
  add_dependencies(ascendc_impl_gen ${ascendc_impl_gen_depends})

  if(ENABLE_BINARY OR ENABLE_CUSTOM)
    foreach(compute_unit ${ASCEND_COMPUTE_UNIT})
      foreach(OP_DIR ${COMPILED_OP_DIRS})
        get_filename_component(op_name ${OP_DIR} NAME)
        # generate opc shell scripts for autogen binary config ops
        generate_bin_scripts(
          TARGET gen_bin_scripts
          OP_NAME ${op_name}
          OPS_INFO_DIR ${ASCEND_AUTOGEN_PATH}
          COMPUTE_UNIT ${compute_unit}
          OUT_DIR ${CMAKE_BINARY_DIR}/binary/${compute_unit}
        )
        if(EXISTS ${OP_DIR}/op_host/config/${compute_unit}/${op_name}_binary.json)
          # binary compile from binary json config
          message(STATUS "[INFO] On [${compute_unit}], [${op_name}] compile binary with self config.")
          compile_from_config(
            TARGET ascendc_bin_${compute_unit}_${op_name}
            OP_NAME ${op_name}
            OPS_INFO_DIR ${ASCEND_AUTOGEN_PATH}
            IMPL_DIR ${OP_DIR}/op_kernel
            CONFIG_DIR ${OP_DIR}/op_host/config
            OP_PYTHON_DIR ${CMAKE_BINARY_DIR}/tbe/dynamic
            OUT_DIR ${CMAKE_BINARY_DIR}/binary/${compute_unit}
            INSTALL_DIR ${BIN_KERNEL_INSTALL_DIR}
            COMPUTE_UNIT ${compute_unit}
          )
          add_dependencies(ascendc_bin_${compute_unit}_${op_name} merge_ini_${compute_unit} ascendc_impl_gen)
        endif()
      endforeach()
      
      # generate binary_info_config.json
      gen_binary_info_config_json(
        TARGET gen_bin_info_config_${compute_unit}
        BIN_DIR ${CMAKE_BINARY_DIR}/binary/${compute_unit}
        COMPUTE_UNIT ${compute_unit}
      )
    endforeach()
  endif()
endfunction()