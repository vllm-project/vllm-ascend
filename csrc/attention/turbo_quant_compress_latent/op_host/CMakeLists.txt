add_op_to_compiled_list()

set(TURBO_QUANT_COMPRESS_LATENT_SKIP_HEADER TRUE CACHE INTERNAL "Skip packaging header for this operator")

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
    turbo_quant_compress_latent_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME TurboQuantCompressLatent
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
            -fpermissive
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE turbo_quant_compress_latent ACLNNTYPE aclnn)
endif()
