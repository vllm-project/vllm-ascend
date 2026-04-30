# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if (BUILD_OPEN_PROJECT)
    set(CMAKE_MODULE_PATH
        ${CMAKE_MODULE_PATH}
        ${CMAKE_CURRENT_LIST_DIR}/cmake/modules
    )

    set(CMAKE_PREFIX_PATH
        ${CMAKE_PREFIX_PATH}
        ${ASCEND_CANN_PACKAGE_PATH}
    )

    set(_op_host_aclnn_link
            $<BUILD_INTERFACE:intf_pub>
            exe_graph
            register
            c_sec
    )

    find_package(alog MODULE)

    find_package(unified_dlog MODULE)

    if(NOT ${alog_FOUND})
        add_definitions(-DALOG_NOT_FOUND)
    endif()

    add_library(op_host_aclnn SHARED EXCLUDE_FROM_ALL)
    target_link_libraries(op_host_aclnn PRIVATE
            ${_op_host_aclnn_link}
    )
    target_compile_options(op_host_aclnn PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-std=gnu++1z>
    )

    add_library(op_host_aclnnInner SHARED EXCLUDE_FROM_ALL)
    target_link_libraries(op_host_aclnnInner PRIVATE
            ${_op_host_aclnn_link}
    )
    target_compile_options(op_host_aclnnInner PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-std=gnu++1z>
    )

    add_library(op_host_aclnnExc SHARED EXCLUDE_FROM_ALL)
    target_link_libraries(op_host_aclnnExc PRIVATE
            ${_op_host_aclnn_link}
    )
    target_compile_options(op_host_aclnnExc PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-std=gnu++1z>
    )

    # op api
    add_library(cust_opapi SHARED)
    # When compiling a specified operator, there is an operator without aclnn src.
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/cust_opapi_stub.cpp
            COMMAND touch ${CMAKE_CURRENT_BINARY_DIR}/cust_opapi_stub.cpp
    )
    target_sources(cust_opapi PRIVATE
            ${CMAKE_CURRENT_BINARY_DIR}/cust_opapi_stub.cpp
    )
    target_compile_options(cust_opapi PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-std=gnu++1z>
    )
    target_include_directories(cust_opapi PRIVATE
            $<BUILD_INTERFACE:${ASCEND_CANN_PACKAGE_PATH}/include>
            $<BUILD_INTERFACE:${ASCEND_CANN_PACKAGE_PATH}/include/aclnn>
    )
    target_compile_options(cust_opapi PRIVATE
            -Werror=format
    )
    target_compile_definitions(cust_opapi PRIVATE
            -DACLNN_LOG_FMT_CHECK
    )
    if(BUILD_WITH_3_8_PACKAGE)  #3~8包 依赖opapi, 不依赖opbase
    target_link_libraries(cust_opapi PRIVATE
        $<BUILD_INTERFACE:intf_pub>
        -Wl,--whole-archive
        ops_aclnn
        -Wl,--no-whole-archive
        -lopapi
        nnopbase
        profapi
        ge_common_base
        ascend_dump
        ascendalog
        dl
    )
    else()
    target_link_libraries(cust_opapi PRIVATE
            $<BUILD_INTERFACE:intf_pub>
            -Wl,--whole-archive
            ops_aclnn
            -Wl,--no-whole-archive
        #     -lopapi
            nnopbase
            profapi
            ge_common_base
            ascend_dump
            ascendalog
            dl
    )
    endif()
    set_target_properties(cust_opapi PROPERTIES OUTPUT_NAME
            cust_opapi
    )
    if (NOT ENABLE_BUILT_IN)
        install(TARGETS cust_opapi
                LIBRARY DESTINATION packages/vendors/${VENDOR_NAME}_transformer/op_api/lib
        )
    endif()

    # op proto
    add_library(cust_proto SHARED)
    target_compile_options(cust_proto PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-std=c++11>
            -fvisibility=hidden
    )
    target_compile_definitions(cust_proto PRIVATE
            LOG_CPP
            PROCESS_LOG
    )
    target_link_libraries(cust_proto PRIVATE
            $<BUILD_INTERFACE:intf_pub>
            $<BUILD_INTERFACE:ops_transformer_utils_proto_headers>
            $<$<BOOL:${alog_FOUND}>:$<BUILD_INTERFACE:alog_headers>>
            $<$<BOOL:${dlog_FOUND}>:$<BUILD_INTERFACE:dlog_headers>>
            -Wl,--whole-archive
            rt2_registry
            -Wl,--no-whole-archive
            -Wl,--no-as-needed
            exe_graph
            graph
            graph_base
            register
            ascendalog
            error_manager
            platform
            -Wl,--as-needed
            c_sec
    )
    set_target_properties(cust_proto PROPERTIES OUTPUT_NAME
            cust_opsproto_rt2.0
    )
    if (NOT ENABLE_BUILT_IN)
        install(TARGETS cust_proto
                LIBRARY DESTINATION packages/vendors/${VENDOR_NAME}_transformer/op_proto/lib/linux/${CMAKE_SYSTEM_PROCESSOR}
        )
    endif()

    # op tiling
    add_library(cust_opmaster SHARED)
    target_include_directories(cust_opmaster PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/mc2/common/inc
            $<$<BOOL:${BUILD_OPEN_PROJECT}>:$<BUILD_INTERFACE:${ASCEND_CANN_PACKAGE_PATH}/include/experiment>>
    )
    target_compile_options(cust_opmaster PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-std=c++11>
            -fvisibility=hidden
    )
    target_compile_definitions(cust_opmaster PRIVATE
            LOG_CPP
            PROCESS_LOG
    )
    target_link_libraries(cust_opmaster PRIVATE
            $<BUILD_INTERFACE:intf_pub>
            $<BUILD_INTERFACE:ops_transformer_utils_tiling_headers>
            $<$<BOOL:${alog_FOUND}>:$<BUILD_INTERFACE:alog_headers>>
            $<$<BOOL:${dlog_FOUND}>:$<BUILD_INTERFACE:dlog_headers>>
            -Wl,--whole-archive
            rt2_registry
            -Wl,--no-whole-archive
            -Wl,--no-as-needed
            graph
            graph_base
            exe_graph
            platform
            register
            error_manager
            ascendalog
            unified_dlog
            -Wl,--as-needed
            -Wl,--whole-archive
            tiling_api
            -Wl,--no-whole-archive
            c_sec
    )
    set_target_properties(cust_opmaster PROPERTIES OUTPUT_NAME
            cust_opmaster_rt2.0
    )
    add_custom_command(TARGET cust_opmaster
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory ${TILING_CUSTOM_DIR}
            COMMAND ln -sf $<TARGET_FILE:cust_opmaster> ${TILING_CUSTOM_FILE}
    )
    if (NOT ENABLE_BUILT_IN)
        install(TARGETS cust_opmaster
                LIBRARY DESTINATION packages/vendors/${VENDOR_NAME}_transformer/op_impl/ai_core/tbe/op_tiling/lib/linux/${CMAKE_SYSTEM_PROCESSOR}
        )
    endif()

    # optiling compat
    set(compat_optiling_dir  ${CMAKE_CURRENT_BINARY_DIR}/compat)
    set(compat_optiling_file ${compat_optiling_dir}/liboptiling.so)
    add_custom_target(optiling_compat ALL
            DEPENDS ${compat_optiling_file}
    )

    add_custom_command(
            OUTPUT ${compat_optiling_file}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${compat_optiling_dir}
            COMMAND ln -sf lib/linux/${CMAKE_SYSTEM_PROCESSOR}/$<TARGET_FILE_NAME:cust_opmaster> ${compat_optiling_file}
    )

    if (NOT ENABLE_BUILT_IN)
        install(FILES ${compat_optiling_file}
                DESTINATION packages/vendors/${VENDOR_NAME}_transformer/op_impl/ai_core/tbe/op_tiling
        )
    endif()

    add_ops_tiling_keys(
            OP_NAME "ALL"
            TILING_KEYS ${TILING_KEY}
    )

    add_opc_config(
            OP_NAME "ALL"
            CONFIG ${OP_DEBUG_CONFIG}
    )

    if(ADD_OPS_COMPILE_OPTION_V2)
        add_ops_compile_options(
                OP_NAME "ALL"
                OPTIONS ${OPS_COMPILE_OPTIONS}
        )
    endif()
endif ()

add_subdirectory(common)

set(OP_LIST)
set(OP_DIR_LIST)
op_add_subdirectory(OP_LIST OP_DIR_LIST)

if (BUILD_OPEN_PROJECT)
    if (ENABLE_TEST)
        set(OP_UT_LIST)
        set(OP_UT_DIR_LIST)
        op_add_ut_subdirectory(OP_UT_LIST OP_UT_DIR_LIST)
        foreach (OP_UT_LIST ${OP_UT_DIR_LIST})
            # 仅通过op_add_subdirectory添加的算子目录，需要在这里add tests
            if(OP_UT_LIST IN_LIST OP_DIR_LIST)
                add_subdirectory(${OP_UT_LIST}/tests)
            endif()
        endforeach ()

        if (TESTS_UT_OPS_TEST)
            add_subdirectory(tests/ut/framework_special)
            add_definitions(-Wno-builtin-macro-redefined)
        endif()
    endif ()
   if (TESTS_EXAMPLE_OPS_TEST)
       add_subdirectory(examples)
   endif ()
endif ()


foreach (OP_DIR ${OP_DIR_LIST})
    if (EXISTS "${OP_DIR}/op_host")
        add_subdirectory(${OP_DIR}/op_host)
        if(EXISTS "${OP_DIR}/op_graph/CMakeLists.txt")
            add_subdirectory(${OP_DIR}/op_graph)
        endif()
    else()
        add_subdirectory(${OP_DIR})
    endif()
endforeach ()

if(ENABLE_EXPERIMENTAL)
    # genop新增experimental算子分类
    # add_subdirectory(${op_class})
    add_subdirectory(experimental/attention)
else()
    # genop新增非experimental算子分类
    # add_subdirectory(${op_class})
    add_subdirectory(attention)
endif()


if (UT_TEST_ALL OR OP_HOST_UT OR OP_API_UT OR OP_KERNEL_UT OR OP_GRAPH_UT)
        add_subdirectory(tests/ut/framework_normal)
endif()

if("${ASCEND_OP_NAME}" STREQUAL "add_example")
    add_subdirectory(examples)
    list(APPEND OP_DIR_LIST ${CMAKE_CURRENT_SOURCE_DIR}/examples/${ASCEND_OP_NAME})
endif()

if("${ASCEND_OP_NAME}" STREQUAL "all_gather_add")
    add_subdirectory(examples/mc2)
    list(APPEND OP_DIR_LIST ${CMAKE_CURRENT_SOURCE_DIR}/examples/mc2/${ASCEND_OP_NAME})
endif()

list(APPEND OP_LIST ${COMPILED_OPS})
list(APPEND OP_DIR_LIST ${COMPILED_OP_DIRS})

if(ENABLE_TEST)
    foreach (OP_DIR ${OP_DIR_LIST})
        file(READ "${OP_DIR}/tests/CMakeLists.txt" CML_CONTENT)
        if (CML_CONTENT MATCHES "OpsTest_Level2_AddOp")
            set(UTEST_FRAMEWORK_OLD TRUE CACHE BOOL "UTEST_FRAMEWORK_OLD" FORCE)
        else()
            set(UTEST_FRAMEWORK_NEW TRUE CACHE BOOL "UTEST_FRAMEWORK_NEW" FORCE)
        endif()
    endforeach()
    if(TESTS_UT_OPS_TEST)
        OpsTest_AddLaunch()
    endif()
endif()


if (DEFINED MC2_OPT AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/mc2/common/CMakeLists.txt AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/mc2/3rd/CMakeLists.txt)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/mc2/common)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/mc2/3rd)
endif()

set(OP_DEPEND_DIR_LIST)
op_add_depend_directory(
        OP_LIST ${OP_LIST}
        OP_DIR_LIST OP_DEPEND_DIR_LIST
)
# 仅针对被依赖的算子重新add_subdirectory
foreach (OP_DEPEND_DIR ${OP_DEPEND_DIR_LIST})
    get_filename_component(SUB_DIR ${OP_DEPEND_DIR} NAME)
    if ("${ASCEND_OP_NAME}" STREQUAL "all" OR "${ASCEND_OP_NAME}" STREQUAL "ALL")
        if ( "${OP_DEPEND_DIR}" MATCHES ".*attention.*")
            continue()
        endif()
    endif()
    if (NOT ${SUB_DIR} IN_LIST ASCEND_OP_NAME)
        list(APPEND ASCEND_OP_NAME ${SUB_DIR})
        if (EXISTS "${OP_DEPEND_DIR}/op_host")
            add_subdirectory(${OP_DEPEND_DIR}/op_host)
        else()
            add_subdirectory(${OP_DEPEND_DIR})
        endif()
    endif ()
    if ( "${OP_DEPEND_DIR}" MATCHES ".*moe_inplace_index_add_with_sorted.*")
       list(APPEND OP_DIR_LIST ${OPS_TRANSFORMER_DIR}/moe/3rd/moe_inplace_index_add_with_sorted)
    endif()
endforeach ()

# ------------------------------------------------ aclnn ------------------------------------------------
get_target_property(base_aclnn_srcs op_host_aclnn SOURCES)
get_target_property(base_aclnn_inner_srcs op_host_aclnnInner SOURCES)
get_target_property(base_aclnn_exclude_srcs op_host_aclnnExc SOURCES)

if (BUILD_OPEN_PROJECT)
    set(base_aclnn_binary_dir ${ASCEND_AUTOGEN_DIR})
else()
    get_target_property(base_aclnn_binary_dir op_host_aclnn BINARY_DIR)
endif ()

set(generate_aclnn_srcs)
set(generate_aclnn_inner_srcs)
set(generate_aclnn_headers)
set(generate_proto_dir ${base_aclnn_binary_dir})
set(generate_exclude_proto_srcs)
set(generate_proto_srcs)
set(generate_proto_headers)

if (base_aclnn_srcs)
    foreach (_src ${base_aclnn_srcs})
        string(REGEX MATCH "^${CMAKE_CURRENT_SOURCE_DIR}" is_match "${_src}")
        if (is_match)
            get_filename_component(name_without_ext ${_src} NAME_WE)

            string(REGEX REPLACE "_def$" "" _op_name ${name_without_ext})
            list(APPEND generate_aclnn_srcs ${base_aclnn_binary_dir}/aclnn_${_op_name}.cpp)
            list(APPEND generate_aclnn_headers ${base_aclnn_binary_dir}/aclnn_${_op_name}.h)
            list(APPEND generate_proto_srcs    ${generate_proto_dir}/${_op_name}_proto.cpp)
            list(APPEND generate_proto_headers ${generate_proto_dir}/${_op_name}_proto.h)
        endif ()
    endforeach ()
else ()
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/op_host_aclnn_stub.cpp
            COMMAND touch ${CMAKE_CURRENT_BINARY_DIR}/op_host_aclnn_stub.cpp
    )

    target_sources(op_host_aclnn PRIVATE
            ${CMAKE_CURRENT_BINARY_DIR}/op_host_aclnn_stub.cpp
    )
endif ()

if (base_aclnn_inner_srcs)
    foreach (_src ${base_aclnn_inner_srcs})
        string(REGEX MATCH "^${CMAKE_CURRENT_SOURCE_DIR}" is_match "${_src}")
        if (is_match)
            get_filename_component(name_without_ext ${_src} NAME_WE)
            string(REGEX REPLACE "_def$" "" _op_name ${name_without_ext})
            list(APPEND generate_aclnn_inner_srcs ${base_aclnn_binary_dir}/inner/aclnnInner_${_op_name}.cpp)
            list(APPEND generate_proto_srcs    ${generate_proto_dir}/inner/${_op_name}_proto.cpp)
            list(APPEND generate_proto_headers ${generate_proto_dir}/inner/${_op_name}_proto.h)
        endif ()
    endforeach ()
else ()
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/op_host_aclnn_inner_stub.cpp
            COMMAND touch ${CMAKE_CURRENT_BINARY_DIR}/op_host_aclnn_inner_stub.cpp
    )

    target_sources(op_host_aclnnInner PRIVATE
            ${CMAKE_CURRENT_BINARY_DIR}/op_host_aclnn_inner_stub.cpp
    )
endif ()

if (base_aclnn_exclude_srcs)
    foreach (_src ${base_aclnn_exclude_srcs})
        string(REGEX MATCH "^${CMAKE_CURRENT_SOURCE_DIR}" is_match "${_src}")
        if (is_match)
            get_filename_component(name_without_ext ${_src} NAME_WE)
            string(REGEX REPLACE "_def$" "" _op_name ${name_without_ext})
            list(APPEND generate_exclude_proto_srcs    ${generate_proto_dir}/exc/${_op_name}_proto.cpp)
            list(APPEND generate_proto_srcs            ${generate_proto_dir}/exc/${_op_name}_proto.cpp)
            list(APPEND generate_proto_headers         ${generate_proto_dir}/exc/${_op_name}_proto.h)
        endif ()
    endforeach ()
else()
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/op_host_aclnn_exc_stub.cpp
            COMMAND touch ${CMAKE_CURRENT_BINARY_DIR}/op_host_aclnn_exc_stub.cpp
    )

    target_sources(op_host_aclnnExc PRIVATE
            ${CMAKE_CURRENT_BINARY_DIR}/op_host_aclnn_exc_stub.cpp
    )
endif ()

if (BUILD_OPEN_PROJECT)
    if (generate_aclnn_srcs OR generate_aclnn_inner_srcs)
        set(ops_aclnn_src ${generate_aclnn_srcs} ${generate_aclnn_inner_srcs})
    else ()
        set(ops_aclnn_src ${CMAKE_CURRENT_BINARY_DIR}/ops_aclnn_src_stub.cpp)

        add_custom_command(OUTPUT ${ops_aclnn_src}
                COMMAND touch ${ops_aclnn_src}
        )
    endif ()

    set_source_files_properties(${ops_aclnn_src}
            PROPERTIES GENERATED TRUE
    )
    add_library(ops_aclnn STATIC
            ${ops_aclnn_src}
    )
    target_include_directories(ops_aclnn PRIVATE
            ${PROJECT_SOURCE_DIR}/common/include/common
            ${PROJECT_SOURCE_DIR}/common/include/static
    )
    target_compile_options(ops_aclnn PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-std=gnu++1z>
    )
    target_link_libraries(ops_aclnn PRIVATE
            $<BUILD_INTERFACE:intf_pub>
    )
    if (ENABLE_STATIC)
        add_custom_target(opbuild_gen_aclnn_static
                COMMAND python3 ${PROJECT_SOURCE_DIR}/scripts/util/modify_gen_aclnn.py ${CMAKE_BINARY_DIR}
                DEPENDS opbuild_gen_default opbuild_gen_inner opbuild_gen_exc
        )
        add_dependencies(ops_aclnn opbuild_gen_default opbuild_gen_inner opbuild_gen_aclnn_static)
    else()
        add_dependencies(ops_aclnn opbuild_gen_default opbuild_gen_inner)
    endif()

    set_source_files_properties(${generate_proto_srcs}
            PROPERTIES GENERATED TRUE
    )
    target_sources(cust_proto PRIVATE
            ${generate_proto_srcs}
    )
    add_dependencies(cust_proto ops_transformer_proto_headers)

    if (NOT ENABLE_BUILT_IN)
        install(FILES ${generate_proto_headers}
                DESTINATION packages/vendors/${VENDOR_NAME}_transformer/op_proto/inc OPTIONAL
        )
    endif()

    merge_graph_headers(
        TARGET merge_ops_proto ALL
        OUT_DIR ${ASCEND_GRAPH_CONF_DST}
    )

    add_dependencies(cust_proto merge_ops_proto)
    target_sources(cust_proto PRIVATE
        ${ASCEND_GRAPH_CONF_DST}/ops_proto_transformer.cpp
    )

    redefine_file_macro(
            TARGET_NAME
            op_host_aclnn
            op_host_aclnnInner
            op_host_aclnnExc
            cust_opapi
            cust_proto
            cust_opmaster
            ops_aclnn
    )
else()
    if (generate_aclnn_srcs OR generate_aclnn_inner_srcs)
        set_source_files_properties(${generate_aclnn_srcs} ${generate_aclnn_inner_srcs}
                TARGET_DIRECTORY acl_op_builtin
                PROPERTIES GENERATED TRUE
        )

        target_sources(acl_op_builtin PRIVATE
                ${generate_aclnn_srcs}
                ${generate_aclnn_inner_srcs}
        )
    endif ()

    if (generate_proto_srcs)
        set_source_files_properties(${generate_proto_srcs}
                TARGET_DIRECTORY cust_proto opsproto_rt2.0
                PROPERTIES GENERATED TRUE
        )
        target_sources(cust_proto PRIVATE
                ${generate_proto_srcs}
        )
        add_dependencies(cust_proto ops_transformer_proto_headers)

        target_sources(opsproto_rt2.0 PRIVATE
                ${generate_proto_srcs}
        )
        add_dependencies(opsproto_rt2.0 ops_transformer_proto_headers)
    endif ()

    add_target_source(
            TARGET_NAME opmaster_rt2.0 opmaster_static_rt2.0
            BASE_TARGET cust_opmaster
            SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}
    )

    add_target_source(
            TARGET_NAME opsproto_rt2.0 opsproto_static_rt2.0
            BASE_TARGET cust_proto
            SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}
    )

    add_static_ops(
            ACLNN_SRC ${generate_aclnn_srcs}
            ACLNN_INNER_SRC ${generate_aclnn_inner_srcs}
            SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif ()
target_sources(cust_opapi PRIVATE
    $<$<TARGET_EXISTS:${OPHOST_NAME}_opapi_obj>:$<TARGET_OBJECTS:${OPHOST_NAME}_opapi_obj>>)
if(NOT BUILD_WITH_3_8_PACKAGE)
target_link_libraries(
    cust_opapi
    PRIVATE $<$<BOOL:${BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG}>:$<BUILD_INTERFACE:opapi_math>>
    $<$<TARGET_EXISTS:opsbase>:opsbase>
)
endif()

if(BUILD_WITH_3_8_PACKAGE)
target_link_libraries(
    cust_opmaster
    PUBLIC ${OPHOST_NAME}_tiling_obj
    PUBLIC $<$<TARGET_EXISTS:${OPHOST_NAME}_opmaster_ct_gentask_obj>:$<TARGET_OBJECTS:${OPHOST_NAME}_opmaster_ct_gentask_obj>>
    PUBLIC $<$<TARGET_EXISTS:${COMMON_NAME}_obj>:$<TARGET_OBJECTS:${COMMON_NAME}_obj>>
    PRIVATE $<$<BOOL:${BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG}>:$<BUILD_INTERFACE:optiling>>
)
else ()
target_link_libraries(
    cust_opmaster
    PUBLIC ${OPHOST_NAME}_tiling_obj
    PUBLIC $<$<TARGET_EXISTS:${OPHOST_NAME}_opmaster_ct_gentask_obj>:$<TARGET_OBJECTS:${OPHOST_NAME}_opmaster_ct_gentask_obj>>
    PUBLIC $<$<TARGET_EXISTS:${COMMON_NAME}_obj>:$<TARGET_OBJECTS:${COMMON_NAME}_obj>>
    PRIVATE $<$<BOOL:${BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG}>:$<BUILD_INTERFACE:optiling>>
    $<$<TARGET_EXISTS:opsbase>:opsbase>
)
endif()

if(TARGET ${COMMON_NAME}_obj)
    add_dependencies(cust_opmaster ${COMMON_NAME}_obj)
else()
    message(WARNING "Target ${COMMON_NAME}_obj not found, dependency not added!")
endif()

if(BUILD_WITH_3_8_PACKAGE)
target_link_libraries(
    cust_proto
    PUBLIC ${OPHOST_NAME}_infer_obj
)
else()
target_link_libraries(
    cust_proto
    PUBLIC ${OPHOST_NAME}_infer_obj
    PRIVATE $<$<TARGET_EXISTS:opsbase>:opsbase>
)
endif()
if (generate_aclnn_headers)
    install(FILES ${generate_aclnn_headers}
            DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL
    )
endif ()

add_library(ops_transformer_proto_headers INTERFACE)

target_include_directories(ops_transformer_proto_headers INTERFACE
        $<BUILD_INTERFACE:${generate_proto_dir}>
        $<BUILD_INTERFACE:${generate_proto_dir}/inner>
        $<BUILD_INTERFACE:${generate_proto_dir}/exc>
        $<INSTALL_INTERFACE:include/ops_adv/proto>
)

if ((NOT BUILD_OPEN_PROJECT) AND ("${PRODUCT_SIDE}" STREQUAL "device"))
    ExternalProject_Add(extern_opbuild_gen_transformer
            SOURCE_DIR ${TOP_DIR}/cmake/superbuild
            CONFIGURE_COMMAND ${CMAKE_COMMAND}
            -G ${CMAKE_GENERATOR}
            -DHOST_PACKAGE=opp
            -DBUILD_MOD=ops
            -DUSE_CCACHE=${USE_CCACHE}
            -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/opbuild_output
            -DFEATURE_LIST=custom_opbuild_out_dir=${generate_proto_dir}
            <SOURCE_DIR>
            BUILD_COMMAND TARGETS=opbuild_gen_all $(MAKE)
            INSTALL_COMMAND ""
            LIST_SEPARATOR ::
            EXCLUDE_FROM_ALL TRUE
    )
    add_dependencies(ops_transformer_proto_headers extern_opbuild_gen_transformer)
else()
    add_dependencies(ops_transformer_proto_headers opbuild_gen_default opbuild_gen_inner opbuild_gen_exc)
endif ()

if (NOT BUILD_OPEN_PROJECT)
    if (generate_proto_srcs)
        install_package(
                PACKAGE ops_adv
                TARGETS ops_transformer_proto_headers
                FILES ${generate_proto_headers}
                DESTINATION include/ops_adv/proto
        )
    endif ()
endif ()

# ------------------------------------------------ opbuild ------------------------------------------------
if (BUILD_OPEN_PROJECT)
    string(REPLACE ";" "\;" OPS_PRODUCT_NAME "${ASCEND_COMPUTE_UNIT}")
    if (generate_aclnn_srcs)
        add_custom_command(OUTPUT ${generate_aclnn_srcs} ${generate_aclnn_headers}
                COMMAND mkdir -p ${base_aclnn_binary_dir}
                COMMAND OPS_PROTO_SEPARATE=1
                OPS_ACLNN_GEN=1
                OPS_PROJECT_NAME=aclnn
                OPS_PRODUCT_NAME=\"${OPS_PRODUCT_NAME}\"
                ${OP_BUILD_TOOL}
                $<TARGET_FILE:op_host_aclnn>
                ${base_aclnn_binary_dir}
        )
    endif ()

    add_custom_target(opbuild_gen_default
            DEPENDS ${generate_aclnn_srcs} ${generate_aclnn_headers} op_host_aclnn
    )

    if (generate_aclnn_inner_srcs)
        add_custom_command(OUTPUT ${generate_aclnn_inner_srcs}
                COMMAND mkdir -p ${base_aclnn_binary_dir}/inner
                COMMAND OPS_PROTO_SEPARATE=1
                OPS_ACLNN_GEN=1
                OPS_PROJECT_NAME=aclnnInner
                OPS_PRODUCT_NAME=\"${OPS_PRODUCT_NAME}\"
                ${OP_BUILD_TOOL}
                $<TARGET_FILE:op_host_aclnnInner>
                ${base_aclnn_binary_dir}/inner
        )
    endif ()

    add_custom_target(opbuild_gen_inner
            DEPENDS ${generate_aclnn_inner_srcs} op_host_aclnnInner
    )

    if (generate_exclude_proto_srcs)
        add_custom_command(OUTPUT ${generate_exclude_proto_srcs}
                COMMAND mkdir -p ${base_aclnn_binary_dir}/exc
                COMMAND OPS_PROTO_SEPARATE=1
                OPS_ACLNN_GEN=0
                OPS_PROJECT_NAME=aclnnExc
                OPS_PRODUCT_NAME=\"${OPS_PRODUCT_NAME}\"
                ${OP_BUILD_TOOL}
                $<TARGET_FILE:op_host_aclnnExc>
                ${base_aclnn_binary_dir}/exc
        )
    endif ()

    add_custom_target(opbuild_gen_exc
            DEPENDS ${generate_exclude_proto_srcs} op_host_aclnnExc
    )
endif ()

# ------------------------------------------------ generate adapt py ------------------------------------------------
add_custom_target(generate_transformer_adapt_py
        COMMAND ${HI_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/scripts/util/ascendc_impl_build.py
        \"\"
        \"\"
        \"\"
        \"\"
        ${ASCEND_IMPL_OUT_DIR}
        ${ASCEND_AUTOGEN_DIR}
        --opsinfo-dir ${base_aclnn_binary_dir} ${base_aclnn_binary_dir}/inner ${base_aclnn_binary_dir}/exc
)

add_dependencies(generate_transformer_adapt_py opbuild_gen_default opbuild_gen_inner opbuild_gen_exc)

foreach (_op_name ${OP_LIST})
    install(FILES ${ASCEND_IMPL_OUT_DIR}/dynamic/${_op_name}.py
            DESTINATION ${IMPL_DYNAMIC_INSTALL_DIR}
            OPTIONAL
    )
    install(FILES ${ASCEND_IMPL_OUT_DIR}/dynamic/${_op_name}_apt.py
        DESTINATION ${IMPL_DYNAMIC_INSTALL_DIR}
        OPTIONAL
    )
endforeach ()

install(DIRECTORY ${OPS_ADV_UTILS_KERNEL_INC}/
        DESTINATION ${IMPL_INSTALL_DIR}/ascendc/common
)

# install(DIRECTORY ${OPS_ADV_DIR}/mc2/common/inc/kernel
#         DESTINATION ${IMPL_INSTALL_DIR}/ascendc/common/inc
# )

# install(DIRECTORY ${OPS_ADV_DIR}/mc2/3rd/
#         DESTINATION ${IMPL_INSTALL_DIR}/ascendc/3rd
# )
        
foreach (op_dir ${OP_DIR_LIST})
    get_filename_component(_op_name "${op_dir}" NAME)
    set(CURRENT_KERNEL_DIR "${op_dir}/op_kernel")
    file(GLOB KERNEL_SUB_DIRS RELATIVE "${CURRENT_KERNEL_DIR}" "${CURRENT_KERNEL_DIR}/*")
    filter_copy_files(SELECTED_FILES SELECTED_DIRS)
    install(FILES ${SELECTED_FILES}
        DESTINATION ${IMPL_INSTALL_DIR}/ascendc/${_op_name}
        OPTIONAL
    )
    install(DIRECTORY ${SELECTED_DIRS}
        DESTINATION ${IMPL_INSTALL_DIR}/ascendc/${_op_name}
        OPTIONAL
    )

    foreach (op_depend_dir ${${_op_name}_depends})
        set(CURRENT_KERNEL_DIR "${OPS_TRANSFORMER_DIR}/${op_depend_dir}/op_kernel")
        file(GLOB KERNEL_SUB_DIRS RELATIVE "${CURRENT_KERNEL_DIR}" "${CURRENT_KERNEL_DIR}/*")
        get_filename_component(_op_depened_name "${op_depend_dir}" NAME)
        filter_copy_files(SELECTED_DEPEND_FILES SELECTED_DEPEND_DIRS)
        install(FILES ${SELECTED_DEPEND_FILES}
                DESTINATION ${IMPL_INSTALL_DIR}/ascendc/${_op_depened_name}
                OPTIONAL
        )
        install(DIRECTORY ${SELECTED_DEPEND_DIRS}
                DESTINATION ${IMPL_INSTALL_DIR}/ascendc/${_op_depened_name}
                OPTIONAL
        )
    endforeach ()
endforeach ()

# ------------------------------------------------ generate compile cmd ------------------------------------------------
if (BUILD_OPEN_PROJECT)
    add_custom_target(prepare_build ALL)
    add_custom_target(generate_compile_cmd ALL)
    add_custom_target(generate_ops_info ALL)
    add_dependencies(prepare_build generate_transformer_adapt_py generate_compile_cmd)

    foreach (compute_unit ${ASCEND_COMPUTE_UNIT})
        add_compile_cmd_target(
                COMPUTE_UNIT ${compute_unit}
        )

        add_ops_info_target(
                COMPUTE_UNIT ${compute_unit}
        )
    endforeach ()
else()
    add_dependencies(tbe_ops_json_info generate_transformer_adapt_py)
endif ()

# ------------------------------------------------ opp kernel ------------------------------------------------
if (ENABLE_OPS_KERNEL)
    add_custom_target(ops_transformer_kernel ALL)
    add_custom_target(ops_transformer_config ALL)
    add_dependencies(ops_transformer_kernel ops_transformer_config)

    foreach (compute_unit ${ASCEND_COMPUTE_UNIT})
        add_bin_compile_target(
                COMPUTE_UNIT
                ${compute_unit}
                OP_INFO
                ${OP_DIR_LIST}
        )
    endforeach ()
endif ()

if (NOT ENABLE_BUILT_IN AND BUILD_OPEN_PROJECT)
    add_custom_target(modify_vendor ALL
            DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/scripts/install.sh ${CMAKE_CURRENT_BINARY_DIR}/scripts/upgrade.sh
    )

    # modify VENDOR_NAME in install.sh and upgrade.sh
    if (EXISTS ${ASCEND_PROJECT_DIR}/fwk_modules/scripts)
        set(ASCEND_PROJECT_DIR_SCRIPTS_PATH ${ASCEND_PROJECT_DIR}/fwk_modules/scripts)
    else()
        set(ASCEND_PROJECT_DIR_SCRIPTS_PATH ${CMAKE_SOURCE_DIR}/cmake/scripts/custom)
    endif()
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/scripts/install.sh ${CMAKE_CURRENT_BINARY_DIR}/scripts/upgrade.sh
            COMMAND mkdir -p ${CMAKE_CURRENT_BINARY_DIR}/scripts
            COMMAND cp -r ${ASCEND_PROJECT_DIR_SCRIPTS_PATH}/* ${CMAKE_CURRENT_BINARY_DIR}/scripts/
            COMMAND chmod +w ${CMAKE_CURRENT_BINARY_DIR}/scripts/*
            COMMAND sed -i "s/vendor_name=customize/vendor_name=${VENDOR_NAME}_transformer/g" ${CMAKE_CURRENT_BINARY_DIR}/scripts/*
    )

    install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/scripts/
            DESTINATION . FILE_PERMISSIONS OWNER_EXECUTE OWNER_READ GROUP_READ
    )

    # gen version.info
    set(version_info_dir  ${CMAKE_CURRENT_BINARY_DIR})
    set(version_info_file ${version_info_dir}/version.info)
    add_custom_target(gen_version_info ALL
            DEPENDS ${version_info_file}
    )

    add_custom_command(OUTPUT ${version_info_file}
            COMMAND bash ${ASCENDC_CMAKE_UTIL_DIR}/gen_version_info.sh ${ASCEND_CANN_PACKAGE_PATH} ${version_info_dir}
    )

    install(FILES ${version_info_file}
            DESTINATION packages/vendors/${VENDOR_NAME}_transformer/
    )

    if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
        message(STATUS "Detected architecture: x86_64")
        set(ARCH x86_64)
    elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|arm")
        message(STATUS "Detected architecture: ARM64")
        set(ARCH aarch64)
    else ()
        message(WARNING "Unknown architecture: ${CMAKE_SYSTEM_PROCESSOR}")
    endif ()

    # CPack config
    set(CPACK_PACKAGE_NAME ${CMAKE_PROJECT_NAME})
    set(CPACK_PACKAGE_VERSION ${CMAKE_PROJECT_VERSION})
    set(CPACK_PACKAGE_DESCRIPTION "CPack ops project")
    set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "CPack ops project")
    set(CPACK_PACKAGE_DIRECTORY ${CMAKE_BINARY_DIR})
    set(CPACK_PACKAGE_FILE_NAME "cann-ops-transformer-${VENDOR_NAME}_linux-${ARCH}.run")
    set(CPACK_GENERATOR External)
    set(CPACK_CMAKE_GENERATOR "Unix Makefiles")
    set(CPACK_EXTERNAL_ENABLE_STAGING TRUE)
    if (ENABLE_BUILD_PKG)
      if (EXISTS ${ASCEND_CMAKE_DIR}/makeself.cmake)
        set(CPACK_EXTERNAL_PACKAGE_SCRIPT ${ASCEND_CMAKE_DIR}/makeself.cmake)
      else()
        set(CPACK_MAKESELF_PATH ${OPS_TRANSFORMER_DIR}/third_party/makeself)
        set(CPACK_EXTERNAL_PACKAGE_SCRIPT ${CMAKE_SOURCE_DIR}/cmake/makeself_custom.cmake)
      endif()
    endif()
    set(CPACK_EXTERNAL_BUILT_PACKAGES ${CPACK_PACKAGE_DIRECTORY}/_CPack_Packages/Linux/External/${CPACK_PACKAGE_FILE_NAME}/${CPACK_PACKAGE_FILE_NAME})
    include(CPack)
endif ()