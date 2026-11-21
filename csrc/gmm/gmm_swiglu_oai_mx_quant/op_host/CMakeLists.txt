if(NOT "ascend950" IN_LIST ASCEND_COMPUTE_UNIT)
    return()
endif()

add_op_to_compiled_list()

if(BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE gmm_swiglu_oai_mx_quant_def.cpp)
endif()

include(FetchContent)
# PR 16147's companion header-only Blaze epilogue provides FP4 scale algorithm 2.
FetchContent_Declare(gmm_swiglu_oai_ops_tensor
    GIT_REPOSITORY https://gitcode.com/a845473182/ops-tensor.git
    GIT_TAG 6d0f265d05929a69dd3506b049726b0360baeec0
    GIT_SUBMODULES ""
)
FetchContent_GetProperties(gmm_swiglu_oai_ops_tensor)
if(NOT gmm_swiglu_oai_ops_tensor_POPULATED)
    FetchContent_Populate(gmm_swiglu_oai_ops_tensor)
endif()

add_ops_compile_options(
    OP_NAME GmmSwigluOaiMxQuant
    COMPUTE_UNIT ascend950
    OPTIONS -Wno-deprecated-declarations
            -I${gmm_swiglu_oai_ops_tensor_SOURCE_DIR}/include
            -I${ASCEND_DIR}/${SYSTEM_PREFIX}/asc/include
)

if(NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources_with_soc(OPTYPE gmm_swiglu_oai_mx_quant ACLNNTYPE aclnn_exclude)
endif()
