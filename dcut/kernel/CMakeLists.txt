# D-Cut operator sources are self-contained under this directory. The native
# vllm-ascend build wrapper selects only these two directories.
set(DCUT_OPERATOR_DIRS
    dcut_recurrent_gated_delta_rule
    dcut_causal_conv1d
)

foreach(SUB_DIR IN LISTS DCUT_OPERATOR_DIRS)
    if(DEFINED ASCEND_OP_NAME AND NOT "${ASCEND_OP_NAME}" STREQUAL "")
        if(NOT "${ASCEND_OP_NAME}" STREQUAL "all" AND NOT "${ASCEND_OP_NAME}" STREQUAL "ALL")
            if(NOT SUB_DIR IN_LIST ASCEND_OP_NAME)
                continue()
            endif()
        endif()
    endif()
    add_subdirectory(${SUB_DIR})
endforeach()
