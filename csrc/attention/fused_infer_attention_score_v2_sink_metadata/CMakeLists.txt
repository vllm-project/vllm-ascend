# FIA v2 Sink Metadata (AICPU pre-stage, split-core) — single-line in-repo
# wiring; the macro globs the conventional layout:
#   op_api/*.cpp (l0_*/aclnn_*), op_host/*_infershape*.cpp,
#   op_kernel_aicpu/*_aicpu*.cpp + *.json (kernelSo stays
#   libfia_v2_sink_aicpu_kernels.so per operator contract),
#   op_graph/*_proto.h via include path.

add_modules_sources_aicpu(OPTYPE fused_infer_attention_score_v2_sink_metadata ACLNNTYPE aclnn)

# AICPU so independent naming (operator contract): the repo merge-builds all
# metadata aicpu kernels into libtransformer_aicpu_kernels.so, while our
# runtime json references libfia_v2_sink_aicpu_kernels.so — install an
# additional copy under the contract name so AICPU dispatch resolves.
install(
    FILES ${CMAKE_BINARY_DIR}/libtransformer_aicpu_kernels.so
    DESTINATION packages/vendors/${VENDOR_NAME}_transformer/op_impl/cpu/aicpu_kernel/impl
    RENAME libfia_v2_sink_aicpu_kernels.so
    OPTIONAL
)
