# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
if (TARGET ${OPHOST_NAME}_infer_obj OR TARGET ${OPHOST_NAME}_tiling_obj OR TARGET ${OPHOST_NAME}_aicpu_objs)
    add_library(
        ${OPHOST_NAME}_static STATIC
        $<$<TARGET_EXISTS:${OPHOST_NAME}_infer_obj>:$<TARGET_OBJECTS:${OPHOST_NAME}_infer_obj>>
        $<$<TARGET_EXISTS:${OPHOST_NAME}_tiling_obj>:$<TARGET_OBJECTS:${OPHOST_NAME}_tiling_obj>>
        $<$<TARGET_EXISTS:${OPHOST_NAME}_aicpu_objs>:$<TARGET_OBJECTS:${OPHOST_NAME}_aicpu_objs>>
        $<$<TARGET_EXISTS:${COMMON_NAME}_obj>:$<TARGET_OBJECTS:${COMMON_NAME}_obj>>
        $<$<TARGET_EXISTS:${OPHOST_NAME}_opmaster_ct_gentask_obj>:$<TARGET_OBJECTS:${OPHOST_NAME}_opmaster_ct_gentask_obj>>
    )
    add_custom_command(TARGET ${OPHOST_NAME}_static
                       POST_BUILD
                       COMMAND python3 ${PROJECT_SOURCE_DIR}/scripts/util/build_opp_kernel_static.py
                                        GenerateSymbol -l ${PROJECT_SOURCE_DIR}/build/lib${OPHOST_NAME}_static.a
                                        -s ${PROJECT_SOURCE_DIR}/build/${OPHOST_NAME}.txt
                        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin_tmp)
    target_link_libraries(
        ${OPHOST_NAME}_static
        PRIVATE $<BUILD_INTERFACE:intf_pub_cxx17>
                c_sec
                -Wl,--no-as-needed
                register
                $<$<BOOL:${BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG}>:$<BUILD_INTERFACE:optiling>>
                $<$<TARGET_EXISTS:opsbase>:opsbase>
                -Wl,--as-needed
                -Wl,--whole-archive
                rt2_registry_static
                tiling_api
                -Wl,--no-whole-archive
    )
endif()


if (TARGET ${OPHOST_NAME}_opapi_obj OR TARGET opbuild_gen_aclnn_all)
    if (TARGET ops_aclnn)
        target_compile_definitions(ops_aclnn PUBLIC ACLNN_WITH_BINARY)
    endif()
    if (TARGET ${OPHOST_NAME}_opapi_obj)
        target_compile_definitions(${OPHOST_NAME}_opapi_obj PUBLIC ACLNN_WITH_BINARY)
    endif()
    if (TARGET opbuild_gen_aclnn_all)
        target_compile_definitions(opbuild_gen_aclnn_all PUBLIC ACLNN_WITH_BINARY)
    endif()

    add_library(
        ${OPAPI_NAME}_static STATIC
        $<$<TARGET_EXISTS:${OPHOST_NAME}_opapi_obj>:$<TARGET_OBJECTS:${OPHOST_NAME}_opapi_obj>>
        $<$<TARGET_EXISTS:opbuild_gen_aclnn_all>:$<TARGET_OBJECTS:opbuild_gen_aclnn_all>>
    )
    add_dependencies(${OPAPI_NAME}_static ${OPHOST_NAME}_static)
    add_custom_command(TARGET ${OPAPI_NAME}_static
                    POST_BUILD
                    COMMAND ${CMAKE_AR} x ${PROJECT_SOURCE_DIR}/build/libops_aclnn.a
                    COMMAND ${CMAKE_AR} x ${PROJECT_SOURCE_DIR}/build/lib${OPAPI_NAME}_static.a
                    COMMAND ${CMAKE_AR} qcs lib${OPAPI_NAME}_static.a *.o
                    COMMAND rm *.o
                    COMMAND python3 ${PROJECT_SOURCE_DIR}/scripts/util/build_opp_kernel_static.py
                                    GenerateSymbol -l ${PROJECT_SOURCE_DIR}/build/bin_tmp/lib${OPAPI_NAME}_static.a
                                    -s ${PROJECT_SOURCE_DIR}/build/${OPAPI_NAME}.txt
                    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin_tmp)
endif()

add_custom_target(${OPSTATIC_NAME})
foreach(compute_unit ${ASCEND_COMPUTE_UNIT})
    set(RESOURCE_PATH ${CMAKE_BINARY_DIR}/autogen/${compute_unit}/aclnnop_resource)
    file(GLOB RESOURCE_CPP ${RESOURCE_PATH}/*.cpp)
    set_source_files_properties(${RESOURCE_CPP} PROPERTIES GENERATED TRUE)
    add_library(resource_${compute_unit}_static STATIC 
                $<$<TARGET_EXISTS:${OPHOST_NAME}_infer_obj>:$<TARGET_OBJECTS:${OPHOST_NAME}_infer_obj>>
                $<$<TARGET_EXISTS:${OPHOST_NAME}_tiling_obj>:$<TARGET_OBJECTS:${OPHOST_NAME}_tiling_obj>>
                $<$<TARGET_EXISTS:${COMMON_NAME}_obj>:$<TARGET_OBJECTS:${COMMON_NAME}_obj>>
                $<$<TARGET_EXISTS:${OPHOST_NAME}_aicpu_objs>:$<TARGET_OBJECTS:${OPHOST_NAME}_aicpu_objs>>
                $<$<TARGET_EXISTS:${OPHOST_NAME}_opapi_obj>:$<TARGET_OBJECTS:${OPHOST_NAME}_opapi_obj>>
                $<$<TARGET_EXISTS:opbuild_gen_aclnn_all>:$<TARGET_OBJECTS:opbuild_gen_aclnn_all>>)
    target_sources(resource_${compute_unit}_static PRIVATE ${RESOURCE_CPP})
    target_include_directories(resource_${compute_unit}_static PRIVATE
            ${OPAPI_INCLUDE})
    set_target_properties(resource_${compute_unit}_static PROPERTIES
                                    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin_tmp/${compute_unit}
                                    OUTPUT_NAME ${OPSTATIC_NAME})
    add_dependencies(${OPSTATIC_NAME} resource_${compute_unit}_static ${OPHOST_NAME}_static ${OPAPI_NAME}_static)
    add_custom_command(TARGET resource_${compute_unit}_static
                    POST_BUILD
                    COMMAND ${CMAKE_AR} x ${CMAKE_BINARY_DIR}/libops_aclnn.a
                    COMMAND ${CMAKE_AR} qcs lib${OPSTATIC_NAME}.a *.o
                    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin_tmp/${compute_unit})
    install(FILES ${CMAKE_BINARY_DIR}/bin_tmp/${compute_unit}/lib${OPSTATIC_NAME}.a
            DESTINATION ${CMAKE_BINARY_DIR}/static_library_files/lib64
            OPTIONAL)
endforeach()
