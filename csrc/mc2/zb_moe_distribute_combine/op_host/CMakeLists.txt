set(_SHMEM_INC_OPTS)
if (EXISTS /usr/local/Ascend/shmem/latest/shmem/include/shmem.h)
    list(APPEND _SHMEM_INC_OPTS -I/usr/local/Ascend/shmem/latest/shmem/include)
    list(APPEND _SHMEM_INC_OPTS -I/usr/local/Ascend/shmem/latest/shmem/src/device)
else()
    message(WARNING "shmem headers not found at /usr/local/Ascend/shmem/latest/shmem/include")
endif()

add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnInner PRIVATE
        zb_moe_distribute_combine.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME ZbMoeDistributeCombine
    OPTIONS
        --cce-auto-sync=off
        -Wno-deprecated-declarations
        -Wno-macro-redefined
        -Werror
        -DASCENDC_DUMP=0
        ${_SHMEM_INC_OPTS}
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE zb_moe_distribute_combine ACLNNTYPE aclnn_inner)
    target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/utils/inc/kernel
        ${CMAKE_SOURCE_DIR}/common/include
    )
    target_sources(${OPHOST_NAME}_opdef_aclnn_inner_obj INTERFACE
        zb_moe_distribute_combine.cpp
    )
endif()
