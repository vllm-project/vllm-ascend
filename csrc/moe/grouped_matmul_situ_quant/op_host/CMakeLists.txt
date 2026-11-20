add_op_to_compiled_list()

if(BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
        grouped_matmul_situ_quant_def.cpp
    )
endif()

if("ascend950" IN_LIST ASCEND_COMPUTE_UNIT)
    add_ops_compile_options(
        OP_NAME GroupedMatmulSituQuant
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
                # -Werror（仓库已按 #13091 统一移除）
                -DENABLE_CV_COMM_VIA_SSBUF=1
    )
else()
    # A2/A3 的 A8W4 MSD 通路（arch32）沿用 ops-transformer 侧的编译选项。
    add_ops_compile_options(
        OP_NAME GroupedMatmulSituQuant
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                # -Werror（仓库已按 #13091 统一移除）
    )
endif()

if(NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(
        OPTYPE grouped_matmul_situ_quant
        ACLNNTYPE aclnn
    )
endif()
