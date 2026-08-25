# Copyright (c) 2026 Huawei Technologies Co., Ltd.

if (BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG)
    if (NOT (UT_TEST_ALL OR OP_KERNEL_AICPU_UT))
        add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)
        set(CMAKE_CXX_COMPILER ${ASCEND_DIR}/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-g++)
    endif()

    file(GLOB_RECURSE JSON_FILE ${CMAKE_CURRENT_SOURCE_DIR}/*.json)
    file(GLOB AICPU_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*_aicpu.cpp)
    add_aicpu_cust_kernel_modules(sparse_attention_score_metadata ${AICPU_SRC} ${JSON_FILE})
endif()

if (UT_TEST_ALL OR OP_KERNEL_AICPU_UT)
    AddAicpuOpTestCase(sparse_attention_score_metadata)
endif()
