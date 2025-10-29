include(ExternalProject)

function(get_system_info SYSTEM_INFO)
  if (UNIX)
    execute_process(COMMAND grep -i ^id= /etc/os-release OUTPUT_VARIABLE TEMP)
    string(REGEX REPLACE "\n|id=|ID=|\"" "" SYSTEM_NAME ${TEMP})
    set(${SYSTEM_INFO} ${SYSTEM_NAME}_${CMAKE_SYSTEM_PROCESSOR} PARENT_SCOPE)
  elseif (WIN32)
    message(STATUS "System is Windows. Only for pre-build.")
  else ()
    message(FATAL_ERROR "${CMAKE_SYSTEM_NAME} not support.")
  endif ()
endfunction()

function(opbuild)
  message(STATUS "Opbuild generating sources")
  cmake_parse_arguments(OPBUILD "" "OUT_DIR;PROJECT_NAME;ACCESS_PREFIX;ENABLE_SOURCE" "OPS_SRC" ${ARGN})
  execute_process(COMMAND ${CMAKE_COMPILE} -g -fPIC -shared -std=c++11 ${OPBUILD_OPS_SRC} -D_GLIBCXX_USE_CXX11_ABI=0
                  -I ${ASCEND_CANN_PACKAGE_PATH}/include -I ${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel
                  -L ${ASCEND_CANN_PACKAGE_PATH}/lib64 -lexe_graph -lregister -ltiling_api
                  -o ${OPBUILD_OUT_DIR}/libascend_all_ops.so
                  RESULT_VARIABLE EXEC_RESULT
                  OUTPUT_VARIABLE EXEC_INFO
                  ERROR_VARIABLE  EXEC_ERROR
  )
  if (${EXEC_RESULT})
    message("build ops lib info: ${EXEC_INFO}")
    message("build ops lib error: ${EXEC_ERROR}")
    message(FATAL_ERROR "opbuild run failed!")
  endif()
  set(proj_env "")
  set(prefix_env "")
  if (NOT "${OPBUILD_PROJECT_NAME}x" STREQUAL "x")
    set(proj_env "OPS_PROJECT_NAME=${OPBUILD_PROJECT_NAME}")
  endif()
  if (NOT "${OPBUILD_ACCESS_PREFIX}x" STREQUAL "x")
    set(prefix_env "OPS_DIRECT_ACCESS_PREFIX=${OPBUILD_ACCESS_PREFIX}")
  endif()

  set(ENV{ENABLE_SOURCE_PACKAGE} ${OPBUILD_ENABLE_SOURCE})
  if(${ASCEND_PACK_SHARED_LIBRARY})
    if (NOT vendor_name)
      message(FATAL_ERROR "ERROR: vendor_name is invalid!")
      return()
    endif()
    set(ENV{ASCEND_VENDOR_NAME} ${vendor_name})
    set(ENV{OPS_PRODUCT_NAME} ${ASCEND_COMPUTE_UNIT})
    set(ENV{SYSTEM_PROCESSOR} ${CMAKE_SYSTEM_PROCESSOR})
  endif()
  execute_process(COMMAND ${proj_env} ${prefix_env} ${ASCEND_CANN_PACKAGE_PATH}/toolkit/tools/opbuild/op_build
                          ${OPBUILD_OUT_DIR}/libascend_all_ops.so ${OPBUILD_OUT_DIR}
                  RESULT_VARIABLE EXEC_RESULT
                  OUTPUT_VARIABLE EXEC_INFO
                  ERROR_VARIABLE  EXEC_ERROR
  )
  unset(ENV{ENABLE_SOURCE_PACKAGE})
  if(${ASCEND_PACK_SHARED_LIBRARY})
    unset(ENV{ASCEND_VENDOR_NAME})
    unset(ENV{OPS_PRODUCT_NAME})
    unset(ENV{SYSTEM_PROCESSOR})
  endif()
  if (${EXEC_RESULT})
    message("opbuild ops info: ${EXEC_INFO}")
    message("opbuild ops error: ${EXEC_ERROR}")
  endif()
  message(STATUS "Opbuild generating sources - done")
endfunction()

function(add_ops_info_target)
  cmake_parse_arguments(OPINFO "" "TARGET;OPS_INFO;OUTPUT;INSTALL_DIR" "" ${ARGN})
  get_filename_component(opinfo_file_path "${OPINFO_OUTPUT}" DIRECTORY)
  add_custom_command(OUTPUT ${OPINFO_OUTPUT}
      COMMAND mkdir -p ${opinfo_file_path}
      COMMAND ${ASCEND_PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/cmake/util/parse_ini_to_json.py
              ${OPINFO_OPS_INFO} ${OPINFO_OUTPUT}
  )
  add_custom_target(${OPINFO_TARGET} ALL
      DEPENDS ${OPINFO_OUTPUT}
  )
  if(NOT ${ASCEND_PACK_SHARED_LIBRARY})
    install(FILES ${OPINFO_OUTPUT}
            DESTINATION ${OPINFO_INSTALL_DIR}
    )
  endif()
endfunction()

function(add_ops_compile_options OP_TYPE)
  cmake_parse_arguments(OP_COMPILE "" "OP_TYPE" "COMPUTE_UNIT;OPTIONS" ${ARGN})
  execute_process(COMMAND ${ASCEND_PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/cmake/util/ascendc_gen_options.py
                          ${ASCEND_AUTOGEN_PATH}/${CUSTOM_COMPILE_OPTIONS} ${OP_TYPE} ${OP_COMPILE_COMPUTE_UNIT}
                          ${OP_COMPILE_OPTIONS}
                  RESULT_VARIABLE EXEC_RESULT
                  OUTPUT_VARIABLE EXEC_INFO
                  ERROR_VARIABLE  EXEC_ERROR)
  if (${EXEC_RESULT})
      message("add ops compile options info: ${EXEC_INFO}")
      message("add ops compile options error: ${EXEC_ERROR}")
      message(FATAL_ERROR "add ops compile options failed!")
  endif()
endfunction()

function(add_npu_support_target)
  cmake_parse_arguments(NPUSUP "" "TARGET;OPS_INFO_DIR;OUT_DIR;INSTALL_DIR" "" ${ARGN})
  get_filename_component(npu_sup_file_path "${NPUSUP_OUT_DIR}" DIRECTORY)
  add_custom_command(OUTPUT ${NPUSUP_OUT_DIR}/npu_supported_ops.json
    COMMAND mkdir -p ${NPUSUP_OUT_DIR}
    COMMAND ${CMAKE_SOURCE_DIR}/cmake/util/gen_ops_filter.sh
            ${NPUSUP_OPS_INFO_DIR}
            ${NPUSUP_OUT_DIR}
  )
  add_custom_target(npu_supported_ops ALL
    DEPENDS ${NPUSUP_OUT_DIR}/npu_supported_ops.json
  )
  if(NOT ${ASCEND_PACK_SHARED_LIBRARY})
    install(FILES ${NPUSUP_OUT_DIR}/npu_supported_ops.json
      DESTINATION ${NPUSUP_INSTALL_DIR}
    )
  endif()
endfunction()

function(add_simple_kernel_compile)
  set(options "")
  set(single_value_args "OPS_INFO;OUT_DIR;TILING_LIB;OP_TYPE;SRC;COMPUTE_UNIT;JSON_FILE;DYNAMIC_PATH")
  set(multi_value_args "OPTIONS;CONFIGS")
  cmake_parse_arguments(BINCMP "${options}" "${single_value_args}" "${multi_value_args}" ${ARGN})
  if (NOT DEFINED BINCMP_OUT_DIR)
    set(BINCMP_OUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/binary)
  endif()
  if (NOT DEFINED BINCMP_TILING_LIB)
    set(BINCMP_TILING_LIB $<TARGET_FILE:cust_optiling>)
  endif()
  if (${ASCEND_PACK_SHARED_LIBRARY})
    if (NOT TARGET op_kernel_pack)
      add_custom_target(op_kernel_pack
                        COMMAND ${ASCEND_PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/cmake/util/ascendc_pack_kernel.py
                        --input-path=${BINCMP_OUT_DIR}
                        --output-path=${BINCMP_OUT_DIR}/library
                        --enable-library=${ASCEND_PACK_SHARED_LIBRARY}
                        --platform=${CMAKE_SYSTEM_PROCESSOR})
      add_library(ascend_kernels INTERFACE)
      target_link_libraries(ascend_kernels INTERFACE kernels)
      target_link_directories(ascend_kernels INTERFACE ${BINCMP_OUT_DIR}/library)
      target_include_directories(ascend_kernels INTERFACE ${BINCMP_OUT_DIR}/library)
      add_dependencies(ascend_kernels op_kernel_pack)
      add_dependencies(op_kernel_pack ${BINCMP_OP_TYPE}_${BINCMP_COMPUTE_UNIT})
    endif()
  endif()
  # add Environment Variable Configurations of ccache
  set(_ASCENDC_ENV_VAR)
  if(${CMAKE_CXX_COMPILER_LAUNCHER} MATCHES "ccache$")
    list(APPEND _ASCENDC_ENV_VAR export ASCENDC_CCACHE_EXECUTABLE=${CMAKE_CXX_COMPILER_LAUNCHER} &&)
  endif()

  if (NOT DEFINED BINCMP_OPS_INFO)
    set(BINCMP_OPS_INFO ${ASCEND_AUTOGEN_PATH}/aic-${BINCMP_COMPUTE_UNIT}-ops-info.ini)
  endif()
  if (NOT ${ENABLE_CROSS_COMPILE})
    add_custom_target(${BINCMP_OP_TYPE}_${BINCMP_COMPUTE_UNIT}
                      COMMAND ${_ASCENDC_ENV_VAR} ${ASCEND_PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/cmake/util/ascendc_compile_kernel.py
                      --op-name=${BINCMP_OP_TYPE}
                      --src-file=${BINCMP_SRC}
                      --compute-unit=${BINCMP_COMPUTE_UNIT}
                      --compile-options=\"${BINCMP_OPTIONS}\"
                      --debug-config=\"${BINCMP_CONFIGS}\"
                      --config-ini=${BINCMP_OPS_INFO}
                      --tiling-lib=${BINCMP_TILING_LIB}
                      --output-path=${BINCMP_OUT_DIR}
                      --dynamic-dir=${BINCMP_DYNAMIC_PATH}
                      --enable-binary=\"${ENABLE_BINARY_PACKAGE}\"
                      --json-file=${BINCMP_JSON_FILE}
                      --build-tool=$(MAKE))
    add_dependencies(${BINCMP_OP_TYPE}_${BINCMP_COMPUTE_UNIT} cust_optiling)
  else()
    if (${ENABLE_BINARY_PACKAGE} AND NOT DEFINED HOST_NATIVE_TILING_LIB)
      message(FATAL_ERROR "Native host libs was not set for cross compile!")
    endif()
    add_custom_target(${BINCMP_OP_TYPE}_${BINCMP_COMPUTE_UNIT}
                      COMMAND ${_ASCENDC_ENV_VAR} ${ASCEND_PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/cmake/util/ascendc_compile_kernel.py
                      --op-name=${BINCMP_OP_TYPE}
                      --src-file=${BINCMP_SRC}
                      --compute-unit=${BINCMP_COMPUTE_UNIT}
                      --compile-options=\"${BINCMP_OPTIONS}\"
                      --debug-config=\"${BINCMP_CONFIGS}\"
                      --config-ini=${BINCMP_OPS_INFO}
                      --tiling-lib=${HOST_NATIVE_TILING_LIB}
                      --output-path=${BINCMP_OUT_DIR}
                      --dynamic-dir=${BINCMP_DYNAMIC_PATH}
                      --enable-binary=\"${ENABLE_BINARY_PACKAGE}\"
                      --json-file=${BINCMP_JSON_FILE}
                      --build-tool=$(MAKE))
  endif()
  add_dependencies(ascendc_bin_${BINCMP_COMPUTE_UNIT}_gen_ops_config ${BINCMP_OP_TYPE}_${BINCMP_COMPUTE_UNIT})
  add_dependencies(${BINCMP_OP_TYPE}_${BINCMP_COMPUTE_UNIT} ops_info_gen_${BINCMP_COMPUTE_UNIT})
endfunction()

function(ascendc_device_library)
    message(STATUS "Ascendc device library generating")
    cmake_parse_arguments(DEVICE "" "TARGET;OPTION" "SRC" ${ARGN})
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/tiling_sink
        COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_CURRENT_BINARY_DIR}/tiling_sink/CMakeLists.txt
    )
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E echo "cmake_minimum_required(VERSION 3.16.0)\nproject(cust_tiling_sink)\ninclude(${CMAKE_SOURCE_DIR}/cmake/device_task.cmake)\n"
        OUTPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/tiling_sink/CMakeLists.txt
        RESULT_VARIABLE result
    )
    string(REPLACE ";" " " DEVICE_SRC "${DEVICE_SRC}")
    ExternalProject_Add(tiling_sink_task
        SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/tiling_sink
        CONFIGURE_COMMAND ${CMAKE_COMMAND}
        -DASCEND_CANN_PACKAGE_PATH=${ASCEND_CANN_PACKAGE_PATH}
        -DTARGET=${DEVICE_TARGET}
        -DOPTION=${DEVICE_OPTION}
        -DSRC=${DEVICE_SRC}
        -DVENDOR_NAME=${vendor_name}
        <SOURCE_DIR>
        CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
        INSTALL_COMMAND ""
        BUILD_ALWAYS TRUE
    )
    ExternalProject_Get_Property(tiling_sink_task BINARY_DIR)
    set(TILINGSINK_LIB_PATH "")
    if ("${DEVICE_OPTION}" STREQUAL "SHARED")
        set(TILINGSINK_LIB_PATH "${BINARY_DIR}/libcust_opmaster.so")
    else()
        set(TILINGSINK_LIB_PATH "${BINARY_DIR}/libcust_opmaster.a")
    endif()
    install(FILES ${TILINGSINK_LIB_PATH}
        DESTINATION packages/vendors/${vendor_name}/op_impl/ai_core/tbe/op_master_device/lib
    )
endfunction()
function(add_opregistry_target)
  string(REPLACE ";" "-" COMPUTE_UNIT "${ASCEND_COMPUTE_UNIT}")
  add_custom_target(op_registry_pack
                    COMMAND ${ASCEND_PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/cmake/util/ascendc_pack_opregistry.py
                    --input-path=${CMAKE_SOURCE_DIR}/build_out/
                    --copy-path=${CMAKE_SOURCE_DIR}/build_out/tmp/vendors/${vendor_name}/
                    --output-path=${CMAKE_SOURCE_DIR}/build_out/library/
                    --vendor-name=${vendor_name}
                    --compute-unit=${COMPUTE_UNIT}
                    --framework-type=${ASCEND_FRAMEWORK_TYPE}
                    --platform=${CMAKE_SYSTEM_PROCESSOR})
  add_library(ascend_opregistry INTERFACE)
  target_link_libraries(ascend_opregistry INTERFACE opregistry)
  target_link_directories(ascend_opregistry INTERFACE ${CMAKE_SOURCE_DIR}/build_out/library)
  target_include_directories(ascend_opregistry INTERFACE ${CMAKE_SOURCE_DIR}/build_out/library)
  add_dependencies(ascend_opregistry op_registry_pack)
  if(EXISTS  "${CMAKE_SOURCE_DIR}/framework/caffe_plugin")
    add_dependencies(op_registry_pack cust_caffe_parsers)
  elseif(EXISTS  "${CMAKE_SOURCE_DIR}/framework/tf_plugin")
    add_dependencies(op_registry_pack cust_tf_parsers)
  elseif(EXISTS  "${CMAKE_SOURCE_DIR}/framework/onnx_plugin")
    add_dependencies(op_registry_pack cust_onnx_parsers)
  endif()
endfunction()

function(add_kernels_install)
  # install kernel file
  if (${ENABLE_SOURCE_PACKAGE})
    install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/binary/dynamic/
            DESTINATION packages/vendors/${vendor_name}/op_impl/ai_core/tbe/${vendor_name}_impl/dynamic/
    )
  endif()

  # install *.o files and *.json files
  if (${ENABLE_BINARY_PACKAGE})
    set(INSTALL_DIR packages/vendors/${vendor_name}/op_impl/ai_core/tbe/)
      foreach(compute_unit ${ASCEND_COMPUTE_UNIT})
        install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/binary/${compute_unit}/
                DESTINATION ${INSTALL_DIR}/kernel/${compute_unit}/
        )
      endforeach()
      install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/binary/config/
              DESTINATION ${INSTALL_DIR}/kernel/config/
      )
  endif()
endfunction()

function(add_kernels_compile)
  set(DYNAMIC_PATH "")
  if (${ENABLE_SOURCE_PACKAGE})
    set(DYNAMIC_PATH ${CMAKE_CURRENT_BINARY_DIR}/binary/dynamic)
    file(MAKE_DIRECTORY ${DYNAMIC_PATH})
    file(GLOB KERNEL_FILES "${CMAKE_SOURCE_DIR}/op_kernel/*")
    file(COPY ${KERNEL_FILES} DESTINATION ${DYNAMIC_PATH})
    file(REMOVE "${DYNAMIC_PATH}/CMakeLists.txt")
  endif()

  foreach(compute_unit ${ASCEND_COMPUTE_UNIT})
    # generate aic-${compute_unit}-ops-info.json
    add_ops_info_target(TARGET ops_info_gen_${compute_unit}
      OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/tbe/op_info_cfg/ai_core/${compute_unit}/aic-${compute_unit}-ops-info.json
      OPS_INFO ${ASCEND_AUTOGEN_PATH}/aic-${compute_unit}-ops-info.ini
      INSTALL_DIR packages/vendors/${vendor_name}/op_impl/ai_core/tbe/config/${compute_unit}
    )

    # define a target:binary to prevent kernel file from being rebuilt during the preinstall process
    if (NOT TARGET binary)
      add_custom_target(binary)
    endif()

    if (${ENABLE_BINARY_PACKAGE} OR ${ENABLE_SOURCE_PACKAGE})
      if (${ENABLE_BINARY_PACKAGE})
        # gen binary_info_config.json and <file_name>.json
        add_custom_target(ascendc_bin_${compute_unit}_gen_ops_config
                          COMMAND ${ASCEND_PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/cmake/util/insert_simplified_keys.py
                                  -p ${CMAKE_CURRENT_BINARY_DIR}/binary/${compute_unit}
                          COMMAND ${ASCEND_PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/cmake/util/ascendc_ops_config.py
                                  -p ${CMAKE_CURRENT_BINARY_DIR}/binary/${compute_unit}
                                  -s ${compute_unit}
                          COMMAND ${CMAKE_COMMAND} -E make_directory
                                  ${CMAKE_CURRENT_BINARY_DIR}/binary/config/${compute_unit}
                          COMMAND mv ${CMAKE_CURRENT_BINARY_DIR}/binary/${compute_unit}/*.json
                                  ${CMAKE_CURRENT_BINARY_DIR}/binary/config/${compute_unit}
        )
      else()
        if (NOT TARGET ascendc_bin_${compute_unit}_gen_ops_config)
          add_custom_target(ascendc_bin_${compute_unit}_gen_ops_config)
        endif()
      endif()
      add_dependencies(binary ascendc_bin_${compute_unit}_gen_ops_config)

      # get op_type-op_name from aic-${compute_unit}-ops-info.ini
      execute_process(COMMAND ${ASCEND_PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/cmake/util/ascendc_get_op_name.py
                              --ini-file=${ASCEND_AUTOGEN_PATH}/aic-${compute_unit}-ops-info.ini
                      OUTPUT_VARIABLE OP_TYPE_NAME
                      RESULT_VARIABLE EXEC_RESULT
                      ERROR_VARIABLE EXEC_ERROR
      )
      if (${EXEC_RESULT})
        message(FATAL_ERROR, "get op name failed, gen error: ${EXEC_ERROR}")
      endif()

      # compile op one by one with ascendc_compile_kernel.py
      string(REPLACE "\n" ";" TYPE_NAME_LIST "${OP_TYPE_NAME}")
      foreach(TYPE_NAME IN LISTS TYPE_NAME_LIST)
        if (NOT "${TYPE_NAME}" STREQUAL "")
          string(REPLACE "-" ";" bin_sep ${TYPE_NAME})
          list(GET bin_sep 0 op_type)
          list(GET bin_sep 1 op_file)
          add_simple_kernel_compile(OP_TYPE ${op_type}
                                    SRC ${CMAKE_SOURCE_DIR}/op_kernel/${op_file}.cpp
                                    COMPUTE_UNIT ${compute_unit}
                                    JSON_FILE ${CMAKE_CURRENT_BINARY_DIR}/tbe/op_info_cfg/ai_core/${compute_unit}/aic-${compute_unit}-ops-info.json
                                    DYNAMIC_PATH ${DYNAMIC_PATH})
        endif()
      endforeach()
    endif()
  endforeach()

  # generate npu_supported_ops.json
  add_npu_support_target(TARGET npu_supported_ops
    OPS_INFO_DIR ${ASCEND_AUTOGEN_PATH}
    OUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/tbe/op_info_cfg/ai_core
    INSTALL_DIR packages/vendors/${vendor_name}/framework/${ASCEND_FRAMEWORK_TYPE}
  )

  if(ENABLE_TEST AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/testcases)
    add_subdirectory(testcases)
  endif()

  if(NOT ASCEND_PACK_SHARED_LIBRARY)
    add_kernels_install()
  else()
    add_opregistry_target()
  endif()
endfunction()
