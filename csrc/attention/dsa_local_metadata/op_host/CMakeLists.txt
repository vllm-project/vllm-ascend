# This program is free software, you can redistribute it and/or modify it under
# the terms of conditions of CANN Open Software License Agreement Version 2.0.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
        dsa_local_metadata_def.cpp
    )
    # Defensive: the imported target `register` is only created by
    # Findmetadef when both libregister.so and register/register.h are found.
    # Without it, linking degrades to bare -lregister with no include path and
    # def.cpp fails with "register/op_def_registry.h: No such file".
    if (DEFINED METADEF_INC_DIR AND NOT METADEF_INC_DIR STREQUAL "METADEF_INC_DIR-NOTFOUND")
        target_include_directories(op_host_aclnn PRIVATE ${METADEF_INC_DIR})
    endif()
    target_include_directories(op_host_aclnn PRIVATE ${ASCEND_CANN_PACKAGE_PATH}/include)
endif()

add_ops_compile_options(
    OP_NAME DsaLocalMetadata
    OPTIONS
        --cce-auto-sync=on
        -Wno-deprecated-declarations
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE dsa_local_metadata ACLNNTYPE aclnn)
    target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()
