# SPDX-License-Identifier: Apache-2.0

# Operator collection belongs to one configure run and one csrc project.
# Directory properties are shared by its child directories without persisting
# into CMakeCache.txt or leaking into an enclosing project's collection.
unset(COMPILED_OPS CACHE)
unset(COMPILED_OP_DIRS CACHE)
set_property(DIRECTORY "${OPS_TRANSFORMER_DIR}" PROPERTY COMPILED_OPS "")
set_property(DIRECTORY "${OPS_TRANSFORMER_DIR}" PROPERTY COMPILED_OP_DIRS "")

function(record_compiled_op OP_NAME OP_DIR)
    get_property(_compiled_op_dirs DIRECTORY "${OPS_TRANSFORMER_DIR}" PROPERTY COMPILED_OP_DIRS)
    if(OP_DIR IN_LIST _compiled_op_dirs)
        return()
    endif()
    set_property(DIRECTORY "${OPS_TRANSFORMER_DIR}" APPEND PROPERTY COMPILED_OPS "${OP_NAME}")
    set_property(DIRECTORY "${OPS_TRANSFORMER_DIR}" APPEND PROPERTY COMPILED_OP_DIRS "${OP_DIR}")
endfunction()

function(get_compiled_ops OUT_OPS OUT_DIRS)
    get_property(_compiled_ops DIRECTORY "${OPS_TRANSFORMER_DIR}" PROPERTY COMPILED_OPS)
    get_property(_compiled_op_dirs DIRECTORY "${OPS_TRANSFORMER_DIR}" PROPERTY COMPILED_OP_DIRS)
    set(${OUT_OPS} "${_compiled_ops}" PARENT_SCOPE)
    set(${OUT_DIRS} "${_compiled_op_dirs}" PARENT_SCOPE)
endfunction()

function(append_compiled_ops OUT_OPS OUT_DIRS)
    get_compiled_ops(_compiled_ops _compiled_op_dirs)
    set(_all_ops ${${OUT_OPS}} ${_compiled_ops})
    set(_all_op_dirs ${${OUT_DIRS}} ${_compiled_op_dirs})
    list(REMOVE_DUPLICATES _all_ops)
    list(REMOVE_DUPLICATES _all_op_dirs)
    set(${OUT_OPS} "${_all_ops}" PARENT_SCOPE)
    set(${OUT_DIRS} "${_all_op_dirs}" PARENT_SCOPE)
endfunction()
