# FIA v2 Sink main op (AICore) — manual-aclnn op (self-implemented two-stage
# V3 entry in op_api/), so the def goes through the op_host_aclnnExc channel
# (ACLNNTYPE aclnn_exclude) to suppress the auto-generated aclnn wrapper.
# Template: csrc/attention/sparse_attention_score/op_host/CMakeLists.txt.
# Sources carried from op_src (omni lineage), flattened to the repo-expected
# top-level op layout; the omni-lineage common trees remain isolated under
# fused_infer_attention_score_v2_sink_common.

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../fused_infer_attention_score_v2_sink_common/omni_common/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../fused_infer_attention_score_v2_sink_common/omni_common/stub/op_api
    ${CMAKE_CURRENT_SOURCE_DIR}/../fused_infer_attention_score_v2_sink_common/omni_common/stub/op_tiling
    ${CMAKE_CURRENT_SOURCE_DIR}/../fused_infer_attention_score_v2_sink_common/attention_common/op_host
    ${CMAKE_CURRENT_SOURCE_DIR}/../fused_infer_attention_score_v2_sink_common/attention_common/op_kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/../fused_infer_attention_score_v2_sink_common/attention_common
)

# Kernel-build registration. add_op_to_compiled_list() assumes it is called
# from <op>/op_host/CMakeLists.txt (OP_NAME = basename(CWD.parent)); this
# flattened layout puts the CMakeLists at the op top level, which would
# register the bogus name "attention" and leave ${<op>_dir} undefined —
# the kernel stage (csrc/cmake/func.cmake add_bin_compile_target) then
# `continue()`s and the gen/*.sh never run (no bin/<op>, no install).
# Register explicitly with the correct op name and dir instead.
set(COMPILED_OPS ${COMPILED_OPS} fused_infer_attention_score_v2_sink CACHE STRING "Compiled Ops" FORCE)
set(COMPILED_OP_DIRS ${COMPILED_OP_DIRS} ${CMAKE_CURRENT_SOURCE_DIR} CACHE STRING "Compiled Ops Dirs" FORCE)

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        op_host/fused_infer_attention_score_v2_sink_def.cpp
    )
endif()

if (BUILD_OPS_RTY_KERNEL)
    add_ops_compile_options(
        OP_NAME FusedInferAttentionScoreV2Sink
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
    )
else()
    add_modules_sources(OPTYPE fused_infer_attention_score_v2_sink ACLNNTYPE aclnn_exclude)
endif()

if (NOT BUILD_OPS_RTY_KERNEL)
    add_tiling_modules()
    target_sources(${OPHOST_NAME}_tiling_obj PRIVATE
        op_host/fused_infer_attention_score_v2_sink_tiling.cpp
        op_host/fused_infer_attention_score_v2_sink_tiling_v3.cpp
        op_host/fused_infer_attention_score_v2_sink_tiling_register.cpp
        op_host/fused_infer_attention_score_v2_sink_tiling_info_parser.cpp
        op_host/fused_infer_attention_score_v2_sink_tiling_check.cpp
        op_host/fused_infer_attention_score_v2_sink_tiling_check_single_para.cpp
        op_host/fused_infer_attention_score_v2_sink_tiling_check_feature.cpp
        op_host/fused_infer_attention_score_v2_sink_tiling_check_existence.cpp
        op_host/fused_infer_attention_score_v2_sink_tiling_check_consistency.cpp
        op_host/fia_tiling_nonquant_sink.cpp
        op_host/fia_tiling_nonquant_mla_sink.cpp
        op_host/fallback_fused_infer_attention_score_v2_sink.cpp
        ../fused_infer_attention_score_v2_sink_common/attention_common/op_host/fia_tiling_info.cpp
        ../fused_infer_attention_score_v2_sink_common/attention_common/op_host/fia_tiling_shape.cpp
        ../fused_infer_attention_score_v2_sink_common/attention_common/op_host/split_core.cpp
    )
    # add_modules_sources collects infershape via a TOP-LEVEL-ONLY glob
    # (${SOURCE_DIR}/*_infershape*.cpp, see csrc/cmake/obj_func.cmake); this
    # flattened layout keeps it under op_host/, so list it explicitly — same
    # workaround as the tiling_obj list above. Without it libcust_opsproto
    # misses InferShape/InferDataTypeFusedInferAttentionScoreV2Sink and the
    # aclnn dispatch fails at runtime.
    add_infer_modules()
    target_sources(${OPHOST_NAME}_infer_obj PRIVATE
        op_host/fused_infer_attention_score_v2_sink_infershape.cpp
    )
endif()
# NOTE: op_api/*.cpp (hand-written two-stage V3 aclnn entry) is picked up by
# add_modules_sources' opapi glob into ${OPHOST_NAME}_opapi_obj -> libcust_opapi
# (single definition source; the aclnn_exclude type above only suppresses the
# auto-generated aclnn wrapper).
