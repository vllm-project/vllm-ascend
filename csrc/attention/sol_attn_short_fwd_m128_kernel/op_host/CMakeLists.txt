add_op_to_compiled_list()

if(BUILD_OPEN_PROJECT)
    set(sol_attn_short_fwd_m128_kernel_depends attention/common CACHE INTERNAL "Dependencies for sol_attn_short_fwd_m128_kernel")
    target_sources(op_host_aclnn PRIVATE sol_attn_short_fwd_m128_kernel_def.cpp)
endif()

add_ops_compile_options(
    OP_NAME SolAttnShortFwdM128Kernel
    OPTIONS --cce-auto-sync=off -Wno-deprecated-declarations
)

if(NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE sol_attn_short_fwd_m128_kernel ACLNNTYPE aclnn)
endif()
