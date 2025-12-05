# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

function(add_target_source)
    cmake_parse_arguments(ADD "" "BASE_TARGET;SRC_DIR" "TARGET_NAME" ${ARGN})

    get_target_property(all_srcs ${ADD_BASE_TARGET} SOURCES)
    set(add_srcs)
    foreach(_src ${all_srcs})
        string(REGEX MATCH "^${ADD_SRC_DIR}" is_match "${_src}")
        if (is_match)
            list(APPEND add_srcs ${_src})
        endif ()
    endforeach()

    get_target_property(all_includes ${ADD_BASE_TARGET} INCLUDE_DIRECTORIES)
    set(add_includes)
    foreach(_include ${all_includes})
        string(REGEX MATCH "^${ADD_SRC_DIR}" is_match "${_include}")
        if (is_match)
            list(APPEND add_includes ${_include})
        endif ()
    endforeach()

    foreach(_target_name ${ADD_TARGET_NAME})
        target_sources(${_target_name} PRIVATE
                ${add_srcs}
        )

        target_include_directories(${_target_name} PRIVATE
                ${add_includes}
        )
    endforeach()
endfunction()

function(op_add_subdirectory OP_LIST OP_DIR_LIST)
    set(_OP_LIST)
    set(_OP_DIR_LIST)

    file(GLOB OP_HOST_CMAKE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/**/op_host/CMakeLists.txt")

    foreach(OP_CMAKE_FILE ${OP_HOST_CMAKE_FILES})
        get_filename_component(OP_HOST_DIR "${OP_CMAKE_FILE}" DIRECTORY)
        get_filename_component(OP_DIR "${OP_HOST_DIR}" DIRECTORY)
        get_filename_component(OP_NAME "${OP_DIR}" NAME)

        if (NOT BUILD_OPEN_PROJECT)
            if (EXISTS ${TOP_DIR}/asl/ops/cann/ops/built-in/tbe/impl/ascendc/${OP_NAME})
                continue()
            endif ()
        endif ()

        if (DEFINED ASCEND_OP_NAME AND NOT "${ASCEND_OP_NAME}" STREQUAL "")
            if (NOT "${ASCEND_OP_NAME}" STREQUAL "all" AND NOT "${ASCEND_OP_NAME}" STREQUAL "ALL")
                if (NOT ${OP_NAME} IN_LIST ASCEND_OP_NAME)
                    continue()
                endif ()
            endif ()
        endif ()

        list(APPEND _OP_LIST ${OP_NAME})
        list(APPEND _OP_DIR_LIST ${OP_DIR})
    endforeach()

    list(REMOVE_DUPLICATES _OP_LIST)
    list(REMOVE_DUPLICATES _OP_DIR_LIST)
    list(SORT _OP_LIST)
    list(SORT _OP_DIR_LIST)
    set(${OP_LIST} ${_OP_LIST} PARENT_SCOPE)
    set(${OP_DIR_LIST} ${_OP_DIR_LIST} PARENT_SCOPE)
endfunction()

function(op_add_depend_directory)
    cmake_parse_arguments(DEP "" "OP_DIR_LIST" "OP_LIST" ${ARGN})
    set(_OP_DEPEND_DIR_LIST)
    foreach(op_name ${DEP_OP_LIST})
        if (DEFINED ${op_name}_depends)
            foreach(depend_info ${${op_name}_depends})
                if (NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${depend_info}/op_host/CMakeLists.txt)
                    continue()
                endif ()

                get_filename_component(_depend_op_name "${depend_info}" NAME)
                if (NOT BUILD_OPEN_PROJECT)
                    if (EXISTS ${TOP_DIR}/asl/ops/cann/ops/built-in/tbe/impl/ascendc/${_depend_op_name})
                        continue()
                    endif ()
                endif ()

                if (NOT ${_depend_op_name} IN_LIST DEP_OP_LIST)
                    list(APPEND _OP_DEPEND_DIR_LIST ${CMAKE_CURRENT_SOURCE_DIR}/${depend_info})
                endif ()
            endforeach()
        endif()
    endforeach()

    list(SORT _OP_DEPEND_DIR_LIST)
    set(${DEP_OP_DIR_LIST} ${_OP_DEPEND_DIR_LIST} PARENT_SCOPE)
endfunction()

function(add_compile_cmd_target)
    cmake_parse_arguments(CMD "" "COMPUTE_UNIT" "" ${ARGN})

    if(ADD_OPS_COMPILE_OPTION_V2)
            set(OP_DEBUG_CONFIG_OPTION --opc-config-file ${ASCEND_CUSTOM_OPC_OPTIONS})
    else()
        if(OP_DEBUG_CONFIG)
            set(OP_DEBUG_CONFIG_OPTION --op-debug-config ${OP_DEBUG_CONFIG})
        endif()
        set(OP_TILING_KEY_OPTION --tiling-keys ${ASCEND_CUSTOM_TILING_KEYS})
    endif()

    set(_OUT_DIR           ${ASCEND_BINARY_OUT_DIR}/${CMD_COMPUTE_UNIT})
    set(GEN_OUT_DIR        ${_OUT_DIR}/gen)
    set(COMPILE_CMD_TARGET generate_compile_cmd_${CMD_COMPUTE_UNIT})
    add_custom_target(${COMPILE_CMD_TARGET} ALL
            COMMAND mkdir -p ${GEN_OUT_DIR}
            COMMAND ${HI_PYTHON} ${ASCENDC_CMAKE_UTIL_DIR}/ascendc_bin_param_build.py
            ${base_aclnn_binary_dir}/aic-${CMD_COMPUTE_UNIT}-ops-info.ini
            ${GEN_OUT_DIR}
            ${CMD_COMPUTE_UNIT}
            ${OP_TILING_KEY_OPTION}
            ${OP_DEBUG_CONFIG_OPTION}
            COMMAND ${HI_PYTHON} ${ASCENDC_CMAKE_UTIL_DIR}/ascendc_bin_param_build.py
            ${base_aclnn_binary_dir}/inner/aic-${CMD_COMPUTE_UNIT}-ops-info.ini
            ${GEN_OUT_DIR}
            ${CMD_COMPUTE_UNIT}
            ${OP_TILING_KEY_OPTION}
            ${OP_DEBUG_CONFIG_OPTION}
            COMMAND ${HI_PYTHON} ${ASCENDC_CMAKE_UTIL_DIR}/ascendc_bin_param_build.py
            ${base_aclnn_binary_dir}/exc/aic-${CMD_COMPUTE_UNIT}-ops-info.ini
            ${GEN_OUT_DIR}
            ${CMD_COMPUTE_UNIT}
            ${OP_TILING_KEY_OPTION}
            ${OP_DEBUG_CONFIG_OPTION}
    )

    add_dependencies(${COMPILE_CMD_TARGET} opbuild_gen_default opbuild_gen_inner opbuild_gen_exc)
    add_dependencies(generate_compile_cmd ${COMPILE_CMD_TARGET})
endfunction()

function(add_ops_info_target)
    cmake_parse_arguments(OPINFO "" "COMPUTE_UNIT" "" ${ARGN})

    set(OPS_INFO_TARGET generate_ops_info_${OPINFO_COMPUTE_UNIT})
    set(OPS_INFO_JSON ${ASCEND_AUTOGEN_DIR}/aic-${OPINFO_COMPUTE_UNIT}-ops-info.json)
    set(CUSTOM_OPS_INFO_DIR ${CUSTOM_DIR}/op_impl/ai_core/tbe/config/${OPINFO_COMPUTE_UNIT})

    set(OPS_INFO_INI          ${base_aclnn_binary_dir}/aic-${OPINFO_COMPUTE_UNIT}-ops-info.ini)
    set(OPS_INFO_INNER_INI    ${base_aclnn_binary_dir}/inner/aic-${OPINFO_COMPUTE_UNIT}-ops-info.ini)
    set(OPS_INFO_EXCLUDE_INI  ${base_aclnn_binary_dir}/exc/aic-${OPINFO_COMPUTE_UNIT}-ops-info.ini)

    add_custom_command(OUTPUT ${OPS_INFO_JSON}
            COMMAND ${HI_PYTHON} ${ASCENDC_CMAKE_UTIL_DIR}/parse_ini_to_json.py
            ${OPS_INFO_INI}
            ${OPS_INFO_INNER_INI}
            ${OPS_INFO_EXCLUDE_INI}
            ${OPS_INFO_JSON}
            COMMAND mkdir -p ${CUSTOM_OPS_INFO_DIR}
            COMMAND cp -f ${OPS_INFO_JSON} ${CUSTOM_OPS_INFO_DIR}
    )

    add_custom_target(${OPS_INFO_TARGET} ALL
            DEPENDS ${OPS_INFO_JSON}
    )

    add_dependencies(${OPS_INFO_TARGET} opbuild_gen_default opbuild_gen_inner opbuild_gen_exc)
    add_dependencies(generate_ops_info ${OPS_INFO_TARGET})

    install(FILES ${OPS_INFO_JSON}
            DESTINATION packages/vendors/${VENDOR_NAME}/op_impl/ai_core/tbe/config/${OPINFO_COMPUTE_UNIT} OPTIONAL
    )
endfunction()

function(add_ops_compile_options)
    cmake_parse_arguments(OP_COMPILE "" "OP_NAME" "COMPUTE_UNIT;OPTIONS" ${ARGN})

    if(NOT OP_COMPILE_OPTIONS)
        return()
    endif()

    if(ADD_OPS_COMPILE_OPTION_V2)
        execute_process(COMMAND ${HI_PYTHON} ${ASCENDC_CMAKE_UTIL_DIR}/ascendc_gen_options.py
                ${ASCEND_CUSTOM_OPTIONS} ${OP_COMPILE_OP_NAME}
                ${OP_COMPILE_COMPUTE_UNIT} ${OP_COMPILE_OPTIONS}
                RESULT_VARIABLE EXEC_RESULT
                OUTPUT_VARIABLE EXEC_INFO
                ERROR_VARIABLE EXEC_ERROR)
        if (EXEC_RESULT)
            message("add ops compile options info: ${EXEC_INFO}")
            message("add ops compile options error: ${EXEC_ERROR}")
            message(FATAL_ERROR "Error: add ops compile options failed!")
        endif ()
    else()
        file(APPEND ${ASCEND_CUSTOM_OPTIONS}
                "${OP_COMPILE_OP_NAME},${OP_COMPILE_COMPUTE_UNIT},${OP_COMPILE_OPTIONS}\n"
        )
    endif()
endfunction()

function(add_ops_tiling_keys)
    cmake_parse_arguments(OP_COMPILE "" "OP_NAME" "COMPUTE_UNIT;TILING_KEYS" ${ARGN})

    if(NOT OP_COMPILE_TILING_KEYS)
        return()
    endif()

    if(ADD_OPS_COMPILE_OPTION_V2)
        list(JOIN OP_COMPILE_TILING_KEYS "," STRING_TILING_KEYS)
        add_ops_compile_options(
                OP_NAME ${OP_COMPILE_OP_NAME}
                OPTIONS --tiling_key=${STRING_TILING_KEYS}
        )
    else()
        file(APPEND ${ASCEND_CUSTOM_TILING_KEYS}
                "${OP_COMPILE_OP_NAME},${OP_COMPILE_COMPUTE_UNIT},${OP_COMPILE_TILING_KEYS}\n"
        )
    endif()
endfunction()

function(add_opc_config)
    cmake_parse_arguments(OP_COMPILE "" "OP_NAME" "COMPUTE_UNIT;CONFIG" ${ARGN})

    if(NOT ADD_OPS_COMPILE_OPTION_V2)
        return()
    endif()

    if(NOT OP_COMPILE_CONFIG)
        return()
    endif()

    string(REPLACE "," ";" OP_COMPILE_CONFIG_LIST "${OP_COMPILE_CONFIG}")
    
    set(_OPC_CONFIG)

    foreach(_option ${OP_COMPILE_CONFIG_LIST})
        if("${_option}" STREQUAL "ccec_g")
            list(APPEND _OPC_CONFIG "-g")
        elseif("${_option}" STREQUAL "ccec_O0")
            list(APPEND _OPC_CONFIG "-O0")
        elseif("${_option}" STREQUAL "oom")
            list(APPEND _OPC_CONFIG "--oom")
        elseif("${_option}" STREQUAL "dump_cce")
            list(APPEND _OPC_CONFIG "--save-temp-files")
        endif()
    endforeach()

    if(_OPC_CONFIG)
        add_ops_compile_options(
                OP_NAME ${OP_COMPILE_OP_NAME}
                OPTIONS ${_OPC_CONFIG}
        )
    endif()
endfunction()

function(add_ops_src_copy)
    cmake_parse_arguments(SRC_COPY "" "TARGET_NAME;SRC;DST;BE_RELIED;COMPUTE_UNIT" "" ${ARGN})

    set(OPS_UTILS_INC_KERNEL_TARGET ops_utils_inc_kernel_${SRC_COPY_COMPUTE_UNIT})
    if (EXISTS ${OPS_ADV_UTILS_KERNEL_INC})
        if (NOT TARGET ${OPS_UTILS_INC_KERNEL_TARGET})
            get_filename_component(_ROOT_OPS_SRC_DIR    "${SRC_COPY_DST}" DIRECTORY)
            set(OPS_UTILS_INC_KERNEL_DIR ${_ROOT_OPS_SRC_DIR}/ascendc/common)
            add_custom_command(OUTPUT ${OPS_UTILS_INC_KERNEL_DIR}
                    COMMAND mkdir -p ${OPS_UTILS_INC_KERNEL_DIR}
                    COMMAND cp -rf ${OPS_ADV_UTILS_KERNEL_INC}/*.* ${OPS_UTILS_INC_KERNEL_DIR}
            )

            add_custom_target(${OPS_UTILS_INC_KERNEL_TARGET}
                    DEPENDS ${OPS_UTILS_INC_KERNEL_DIR}
            )
        endif ()
    endif ()

    if (NOT TARGET ${SRC_COPY_TARGET_NAME})
        set(_BUILD_FLAG ${SRC_COPY_DST}/${SRC_COPY_TARGET_NAME}.done)
        add_custom_command(OUTPUT ${_BUILD_FLAG}
                COMMAND mkdir -p ${SRC_COPY_DST}
                COMMAND cp -rf ${SRC_COPY_SRC}/op_kernel/* ${SRC_COPY_DST}
                COMMAND touch ${_BUILD_FLAG}
        )

        add_custom_target(${SRC_COPY_TARGET_NAME}
                DEPENDS ${_BUILD_FLAG}
        )
    endif ()

    if (TARGET ${OPS_UTILS_INC_KERNEL_TARGET})
        add_dependencies(${SRC_COPY_TARGET_NAME} ${OPS_UTILS_INC_KERNEL_TARGET})
    endif ()

    if (DEFINED SRC_COPY_BE_RELIED)
        add_dependencies(${SRC_COPY_BE_RELIED} ${SRC_COPY_TARGET_NAME})
    endif ()

endfunction()

function(add_bin_compile_target)
    cmake_parse_arguments(BINARY "" "COMPUTE_UNIT" "OP_INFO" ${ARGN})

    set(_INSTALL_DIR packages/vendors/${VENDOR_NAME}/op_impl/ai_core/tbe/kernel)
    set(_OUT_DIR ${ASCEND_BINARY_OUT_DIR}/${BINARY_COMPUTE_UNIT})

    set(BIN_OUT_DIR      ${_OUT_DIR}/bin)
    set(GEN_OUT_DIR      ${_OUT_DIR}/gen)
    set(SRC_OUT_DIR      ${_OUT_DIR}/src)
    file(MAKE_DIRECTORY  ${BIN_OUT_DIR})

    foreach(_op_info ${BINARY_OP_INFO})
        get_filename_component(_op_name "${_op_info}" NAME)
        set(${_op_name}_dir ${_op_info})
    endforeach()

    set(_ops_target_list)
    set(compile_scripts)
    file(GLOB scripts_list ${GEN_OUT_DIR}/*.sh)
    list(APPEND compile_scripts ${scripts_list})

    foreach(bin_script ${compile_scripts})
        get_filename_component(bin_file ${bin_script} NAME_WE)
        string(REPLACE "-" ";" bin_sep ${bin_file})
        list(GET bin_sep 0 op_type)
        list(GET bin_sep 1 op_file)
        list(GET bin_sep 2 op_index)

        if (NOT DEFINED ${op_file}_dir)
            continue()
        endif ()

        if (NOT TARGET ${op_file})
            add_custom_target(${op_file})
            add_dependencies(ops_kernel ${op_file})
        endif ()

        set(OP_TARGET_NAME ${op_file}_${BINARY_COMPUTE_UNIT})

        if (NOT TARGET ${OP_TARGET_NAME})
            add_custom_target(${OP_TARGET_NAME})
            add_dependencies(${op_file} ${OP_TARGET_NAME})
            list(APPEND _ops_target_list ${OP_TARGET_NAME})

            set(OP_SRC_OUT_DIR  ${SRC_OUT_DIR}/${op_file})
            set(OP_BIN_OUT_DIR  ${BIN_OUT_DIR}/${op_file})
            file(MAKE_DIRECTORY ${OP_SRC_OUT_DIR})

            add_ops_src_copy(
                    TARGET_NAME
                    ${OP_TARGET_NAME}_src_copy
                    SRC
                    ${${op_file}_dir}
                    DST
                    ${OP_SRC_OUT_DIR}
                    COMPUTE_UNIT
                    ${BINARY_COMPUTE_UNIT}
            )

            if (DEFINED ${op_file}_depends)
                foreach(depend_info ${${op_file}_depends})
                    get_filename_component(_depend_op_name "${depend_info}" NAME)
                    set(_depend_op_target ${_depend_op_name}_${BINARY_COMPUTE_UNIT}_src_copy)
                    add_ops_src_copy(
                            TARGET_NAME
                            ${_depend_op_target}
                            SRC
                            ${CMAKE_SOURCE_DIR}/${depend_info}
                            DST
                            ${SRC_OUT_DIR}/${_depend_op_name}
                            COMPUTE_UNIT
                            ${BINARY_COMPUTE_UNIT}
                            BE_RELIED
                            ${OP_TARGET_NAME}_src_copy
                    )
                endforeach()
            endif ()

            set(DYNAMIC_PY_FILE ${OP_SRC_OUT_DIR}/${op_type}.py)
            add_custom_command(OUTPUT ${DYNAMIC_PY_FILE}
                    COMMAND cp -rf ${ASCEND_IMPL_OUT_DIR}/dynamic/${op_file}.py ${DYNAMIC_PY_FILE}
            )

            add_custom_target(${OP_TARGET_NAME}_py_copy
                    DEPENDS ${DYNAMIC_PY_FILE}
            )

            add_custom_command(OUTPUT ${OP_BIN_OUT_DIR}
                    COMMAND mkdir -p ${OP_BIN_OUT_DIR}
            )

            add_custom_target(${OP_TARGET_NAME}_mkdir
                    DEPENDS ${OP_BIN_OUT_DIR}
            )

            install(DIRECTORY ${OP_BIN_OUT_DIR}
                    DESTINATION ${_INSTALL_DIR}/${BINARY_COMPUTE_UNIT} OPTIONAL
            )

            install(FILES ${BIN_OUT_DIR}/${op_file}.json
                    DESTINATION ${_INSTALL_DIR}/config/${BINARY_COMPUTE_UNIT} OPTIONAL
            )
        endif ()

        set(_group "1-0")
        if (DEFINED ASCEND_OP_NAME AND NOT "${ASCEND_OP_NAME}" STREQUAL "")
            if (NOT "${ASCEND_OP_NAME}" STREQUAL "all" AND NOT "${ASCEND_OP_NAME}" STREQUAL "ALL")
                if (${op_file} IN_LIST ASCEND_OP_NAME)
                    list(LENGTH ASCEND_OP_NAME _len)
                    list(FIND ASCEND_OP_NAME ${op_file} _index)
                    math(EXPR _next_index "${_index} + 1")
                    if (${_next_index} LESS ${_len})
                        list(GET ASCEND_OP_NAME ${_next_index} _group_str)
                        set(_regex "^[0-9]+-[0-9]+$")
                        string(REGEX MATCH "${_regex}" match "${_group_str}")
                        if (match)
                            set(_group   ${_group_str})
                        endif ()
                    endif ()
                endif ()
            endif ()
        endif ()

        string(REPLACE "-" ";" _group_sep ${_group})

        list(GET _group_sep 1 start_index)
        set(end_index         ${op_index})
        list(GET _group_sep 0 step)

        set(_compile_flag false)
        if (${start_index} LESS ${end_index})
            foreach(i RANGE ${start_index} ${end_index} ${step})
                if (${i} EQUAL ${end_index})
                    set(_compile_flag true)
                    break()
                endif ()
            endforeach()
        elseif (${start_index} EQUAL ${end_index})
            set(_compile_flag true)
        else()
            set(_compile_flag false)
        endif ()

        if (_compile_flag)
            set(_BUILD_COMMAND)
            set(_BUILD_FLAG ${GEN_OUT_DIR}/${OP_TARGET_NAME}_${op_index}.done)
            if (ENABLE_OPS_HOST)
                list(APPEND _BUILD_COMMAND export ASCEND_CUSTOM_OPP_PATH=${CUSTOM_DIR} &&)
            endif ()
            list(APPEND _BUILD_COMMAND export HI_PYTHON="python3" &&)
            list(APPEND _BUILD_COMMAND export TILINGKEY_PAR_COMPILE=1 &&)
            list(APPEND _BUILD_COMMAND export BIN_FILENAME_HASHED=1 &&)
            list(APPEND _BUILD_COMMAND bash ${bin_script} ${OP_SRC_OUT_DIR}/${op_type}.py ${OP_BIN_OUT_DIR})
            if(CMAKE_GENERATOR MATCHES "Unix Makefiles")
                list(APPEND _BUILD_COMMAND && echo $(MAKE))
            endif()

            add_custom_command(OUTPUT ${_BUILD_FLAG}
                    COMMAND ${_BUILD_COMMAND}
                    COMMAND touch ${_BUILD_FLAG}
                    WORKING_DIRECTORY ${GEN_OUT_DIR}
            )

            add_custom_target(${OP_TARGET_NAME}_${op_index}
                DEPENDS ${_BUILD_FLAG}
            )

            if (ENABLE_OPS_HOST)
                add_dependencies(${OP_TARGET_NAME}_${op_index} optiling generate_ops_info)
            endif ()
            add_dependencies(${OP_TARGET_NAME}_${op_index} ${OP_TARGET_NAME}_src_copy ${OP_TARGET_NAME}_py_copy ${OP_TARGET_NAME}_mkdir)
            add_dependencies(${OP_TARGET_NAME} ${OP_TARGET_NAME}_${op_index})
        endif ()
    endforeach()

    if (_ops_target_list)
        set(OPS_CONFIG_TARGET ops_config_${BINARY_COMPUTE_UNIT})
        set(BINARY_INFO_CONFIG_FILE ${BIN_OUT_DIR}/binary_info_config.json)

        add_custom_command(OUTPUT ${BINARY_INFO_CONFIG_FILE}
                COMMAND ${HI_PYTHON} ${ASCENDC_CMAKE_UTIL_DIR}/ascendc_ops_config.py -p ${BIN_OUT_DIR} -s ${BINARY_COMPUTE_UNIT}
        )

        add_custom_target(${OPS_CONFIG_TARGET}
                DEPENDS ${BINARY_INFO_CONFIG_FILE}
        )

        add_dependencies(ops_config ${OPS_CONFIG_TARGET})

        foreach(_op_target ${_ops_target_list})
            add_dependencies(${OPS_CONFIG_TARGET} ${_op_target})
        endforeach()

        install(FILES ${BINARY_INFO_CONFIG_FILE}
                DESTINATION ${_INSTALL_DIR}/config/${BINARY_COMPUTE_UNIT} OPTIONAL
        )
    endif ()
endfunction()

function(redefine_file_macro)
    cmake_parse_arguments(_FILE "" "" "TARGET_NAME" ${ARGN})

    foreach(_target_name ${_FILE_TARGET_NAME})
        target_compile_options(${_target_name} PRIVATE
                -Wno-builtin-macro-redefined
        )

        get_target_property(_srcs ${_target_name} SOURCES)

        foreach(_src ${_srcs})
            get_filename_component(_src_name "${_src}" NAME)
            set_source_files_properties(${_src}
                    PROPERTIES COMPILE_DEFINITIONS __FILE__="${_src_name}"
            )
        endforeach()
    endforeach()
endfunction()

function(add_static_ops)
    cmake_parse_arguments(STATIC "" "SRC_DIR" "ACLNN_SRC;ACLNN_INNER_SRC" ${ARGN})
    set(prepare_ops_adv_static_target prepare_ops_adv_static)
    set(static_src_temp_dir ${CMAKE_CURRENT_BINARY_DIR}/static_src_temp_dir)
    set(modified_files)
    foreach(ops_type ${OPS_STATIC_TYPES})
        get_target_property(all_srcs aclnn_ops_${ops_type} SOURCES)
        set(add_srcs)
        set(generate_aclnn_srcs)
        foreach(_src ${all_srcs})
            string(REGEX MATCH "^${STATIC_SRC_DIR}" is_match "${_src}")
            if (is_match)
                list(APPEND add_srcs ${_src})
            endif ()
        endforeach()

        foreach(_src ${add_srcs})
            get_filename_component(name_without_ext ${_src} NAME_WE)
            string(REGEX REPLACE "^aclnn_" "" _op_name ${name_without_ext})

            foreach(_aclnn_src ${STATIC_ACLNN_SRC})
                get_filename_component(aclnn_name ${_aclnn_src} NAME_WE)
                if("aclnn_${_op_name}" STREQUAL "${aclnn_name}")
                    list(APPEND generate_aclnn_srcs ${_aclnn_src})
                    break()
                endif()
            endforeach()

            foreach(_aclnn_inner_src ${STATIC_ACLNN_INNER_SRC})
                get_filename_component(aclnn_inner_name ${_aclnn_inner_src} NAME_WE)
                if("aclnnInner_${_op_name}" STREQUAL "${aclnn_inner_name}")
                    list(APPEND generate_aclnn_srcs ${_aclnn_inner_src})
                    break()
                endif()
            endforeach()
        endforeach()

        if(add_srcs)
            list(TRANSFORM add_srcs REPLACE "${STATIC_SRC_DIR}" "${static_src_temp_dir}" OUTPUT_VARIABLE add_static_srcs)
            list(APPEND modified_files ${add_static_srcs})
            set(aclnn_ops_static_target aclnn_ops_${ops_type}_static)
            set_source_files_properties(${add_static_srcs}
                    TARGET_DIRECTORY ${aclnn_ops_static_target}
                    PROPERTIES GENERATED TRUE
            )

            target_sources(${aclnn_ops_static_target} PRIVATE
                    ${add_static_srcs}
            )
            add_dependencies(${aclnn_ops_static_target} ${prepare_ops_adv_static_target})
        endif()

        if(generate_aclnn_srcs)
            list(REMOVE_DUPLICATES generate_aclnn_srcs)
            set(aclnn_op_target acl_op_${ops_type}_builtin)
            set_source_files_properties(${generate_aclnn_srcs}
                    TARGET_DIRECTORY ${aclnn_op_target}
                    PROPERTIES GENERATED TRUE
            )

            target_sources(${aclnn_op_target} PRIVATE
                    ${generate_aclnn_srcs}
            )
        endif()
    endforeach()

    if(NOT TARGET ${prepare_ops_adv_static_target})
        list(REMOVE_DUPLICATES modified_files)
        add_custom_command(OUTPUT ${static_src_temp_dir}
                COMMAND mkdir -p ${static_src_temp_dir}
                COMMAND cp -rf ${STATIC_SRC_DIR}/src ${static_src_temp_dir}
                COMMAND ${HI_PYTHON} -B ${OPS_STATIC_SCRIPT} InsertIni -p ${static_src_temp_dir} -f ${modified_files}
        )

        add_custom_target(${prepare_ops_adv_static_target}
                DEPENDS ${static_src_temp_dir}
        )
    endif()
endfunction()

if (BUILD_OPEN_PROJECT)
    if (TESTS_UT_OPS_TEST)
        include(${OPS_ADV_CMAKE_DIR}/func_utest.cmake)
    endif ()
    if (TESTS_EXAMPLE_OPS_TEST)
        include(${OPS_ADV_CMAKE_DIR}/func_examples.cmake)
    endif ()
endif ()
