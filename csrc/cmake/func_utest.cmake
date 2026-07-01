# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

########################################################################################################################
# 预定义变量
########################################################################################################################

# 缓存所有算子 UTest 场景 OpApi 动态库相关信息
set(_OpsTestUt_OpApiSources            "" CACHE INTERNAL "" FORCE)  # Sources
set(_OpsTestUt_OpApiPrivateIncludesExt "" CACHE INTERNAL "" FORCE)  # PrivateIncludesExt
set(_OpsTestUt_OpApiLinkLibrariesExt   "" CACHE INTERNAL "" FORCE)  # LinkLibrariesExt
add_library(_OpsTestUt_OpApi_Wno INTERFACE)
target_compile_options(_OpsTestUt_OpApi_Wno
        INTERFACE
            $<$<CXX_COMPILER_ID:Clang>:-Wno-mismatched-tags>
            $<$<CXX_COMPILER_ID:GNU>:-Wno-format-signedness>
            -Wno-extra
            -Wno-redundant-decls
)

add_definitions(-DNOT_DYNAMIC_COMPILE)

# 缓存所有算子 UTest 场景 OpProto 动态库相关信息
set(_OpsTestUt_OpProtoSources            "" CACHE INTERNAL "" FORCE)  # Sources
set(_OpsTestUt_OpProtoPrivateIncludesExt "" CACHE INTERNAL "" FORCE)  # PrivateIncludesExt
set(_OpsTestUt_OpProtoLinkLibrariesExt   "" CACHE INTERNAL "" FORCE)  # LinkLibrariesExt

# 缓存所有算子 UTest 场景 OpTiling 动态库相关信息
set(_OpsTestUt_OpTilingSources            "" CACHE INTERNAL "" FORCE)  # Sources
set(_OpsTestUt_OpTilingPrivateIncludesExt "" CACHE INTERNAL "" FORCE)  # PrivateIncludesExt
set(_OpsTestUt_OpTilingLinkLibrariesExt   "" CACHE INTERNAL "" FORCE)  # LinkLibrariesExt
add_library(_OpsTestUt_OpTiling_Wno INTERFACE)
target_compile_options(_OpsTestUt_OpTiling_Wno
        INTERFACE
            $<$<CXX_COMPILER_ID:Clang>:-Wno-unused-private-field>
            $<$<CXX_COMPILER_ID:Clang>:-Wno-redundant-move>
            -Wno-shadow
            $<$<CXX_COMPILER_ID:GNU>:-Wno-format-signedness>
            -Wno-extra
)

# 缓存当前算子 UTest 场景 OpKernel 目标
set(_OpsTestUt_OpKernelLibraries "" CACHE INTERNAL "" FORCE)  # LinkLibrariesExt
add_library(_OpsTestUt_OpKernel_Wno INTERFACE)
target_compile_options(_OpsTestUt_OpKernel_Wno
        INTERFACE
            -Wno-undef
            -Wno-cast-qual
            -Wno-shadow
            -Wno-sign-compare
            -Wno-unused-macros
            -Wno-unused-variable
            $<$<CXX_COMPILER_ID:GNU>:-Wno-unused-but-set-variable>
            $<$<CXX_COMPILER_ID:Clang>:$<$<VERSION_GREATER_EQUAL:${CMAKE_C_COMPILER_VERSION},15.0.4>:-Wno-unused-but-set-variable>>
            $<$<CXX_COMPILER_ID:GNU>:-Wno-duplicated-branches>
            -Wno-extra
            -Wno-float-conversion
            -Wno-parentheses
)
# 缓存当前算子 UTest 场景 UTestCommon 目标
set(_OpsTestUt_UTestCommonLibrary        "" CACHE INTERNAL "" FORCE)

# 缓存所有算子 UTest 场景 UTest用例 动态库相关信息
set(_OpsTestUt_UTestCaseLibraries        "" CACHE INTERNAL "" FORCE)
set(_OpsTestUt_UTestAclnnCaseLibraries   "" CACHE INTERNAL "" FORCE)
add_library(_OpsTestUt_UTestCaseStatic_Wno INTERFACE)
target_compile_options(_OpsTestUt_UTestCaseStatic_Wno
        INTERFACE
            $<$<CXX_COMPILER_ID:Clang>:-Wno-vla>
)

# GTest 版本兼容性保证
add_library(_OpsTestUt_GTest_Wno INTERFACE)
target_compile_options(_OpsTestUt_GTest_Wno
        INTERFACE
            -Wno-undef
)

########################################################################################################################
# 编译方法
########################################################################################################################

# Level1, 添加算子 OpApi 动态库
#[[
调用参数:
  one_value_keywords:
      SUB_SYSTEM : 必选参数, 用于指定算子所属子系统, 如 transformer
      BRIEF : 必选参数, 用于指定算子缩略名(建议以大驼峰命名, 与算子实际名称无强制对应关系), 如 Fag/Fas/Fa
      SNAKE : 必选参数, 用于指定算子全名, 如 flash_attention_score_grad
  multi_value_keywords:
      SOURCES_EXT                   : 可选参数, 额外源文件
      PRIVATE_INCLUDES_EXT          : 可选参数, 额外头文件搜索路径
      PRIVATE_LINK_LIBRARIES_EXT    : 可选参数, 额外链接库
备注说明:
  本函数提供编译算子对应 opapi.so 的功能. 下面介绍在调用本函数时所需了解的一些背景知识和注意事项.
  1. 本函数假设算子对应 OpApi 的源文件位于 'src/${SUB_SYSTEM}/${SNAKE}/op_host' 路径, 并自动添加以下内容:
     a) 源文件: ${SNAKE}.cpp, aclnn_${SNAKE}.cpp
     b) 头文件搜索路径: 'src/${SUB_SYSTEM}/${SNAKE}/op_host'
  2. 若算子 '需要额外的源文件’ 或 ‘源文件不满足上述约定的默认路径', 则可通过 SOURCES_EXT 参数指定 '额外的源文件';
  3. 若算子 '需要额外头文件搜索路径’ 或 ‘本函数现有实现的头文件搜索路径设置不满足',
     则可通过 PRIVATE_INCLUDES_EXT 参数指定 '额外的头文件搜索路径'; 参数 PRIVATE_LINK_LIBRARIES_EXT 设置逻辑同理;
]]
function(OpsTest_Level1_AddOpApiShared)
    cmake_parse_arguments(
            TMP
            ""
            "SUB_SYSTEM;BRIEF;SNAKE"
            "SOURCES_EXT;PRIVATE_INCLUDES_EXT;PRIVATE_LINK_LIBRARIES_EXT"
            ""
            ${ARGN}
    )
    get_filename_component(_L0_Src "${OPS_ADV_DIR}/${TMP_SUB_SYSTEM}/${TMP_SNAKE}/op_host/${TMP_SNAKE}.cpp" REALPATH)
    if (NOT EXISTS "${_L0_Src}")
        set(_L0_Src)
    endif ()
    get_filename_component(_L2_Src "${OPS_ADV_DIR}/${TMP_SUB_SYSTEM}/${TMP_SNAKE}/op_host/aclnn_${TMP_SNAKE}.cpp" REALPATH)
    if (NOT EXISTS "${_L2_Src}")
        set(_L2_Src)
    endif ()
    get_filename_component(_Inc "${OPS_ADV_DIR}/${TMP_SUB_SYSTEM}/${TMP_SNAKE}/op_host/" REALPATH)
    if (NOT EXISTS "${_Inc}")
        set(_Inc)
    endif ()

    set(_Sources ${TMP_SOURCES_EXT} ${_L0_Src} ${_L2_Src})
    list(REMOVE_DUPLICATES _Sources)
    if (_Sources)
        set(_PrivateIncludeDirs ${TMP_PRIVATE_INCLUDES_EXT} ${_Inc})
        list(REMOVE_DUPLICATES _PrivateIncludeDirs)
        set(_OpsTestUt_OpApiSources            ${_OpsTestUt_OpApiSources}            ${_Sources}                       CACHE INTERNAL "" FORCE)
        set(_OpsTestUt_OpApiPrivateIncludesExt ${_OpsTestUt_OpApiPrivateIncludesExt} ${_PrivateIncludeDirs}            CACHE INTERNAL "" FORCE)
        set(_OpsTestUt_OpApiLinkLibrariesExt   ${_OpsTestUt_OpApiLinkLibrariesExt}   ${TMP_PRIVATE_LINK_LIBRARIES_EXT} CACHE INTERNAL "" FORCE)
    endif ()
endfunction()

# 私有函数, 外部不可直接调用, 用于实际生成 OpApi 动态库
#[[
]]
function(OpsTest_AddOpApiShared)
    list(REMOVE_DUPLICATES _OpsTestUt_OpApiSources)
    list(REMOVE_DUPLICATES _OpsTestUt_OpApiPrivateIncludesExt)
    list(REMOVE_DUPLICATES _OpsTestUt_OpApiLinkLibrariesExt)
    if ("${_OpsTestUt_OpApiSources}" STREQUAL "")
        return()
    endif ()
    set(_Target ${UTest_NamePrefix}_OpApi)
    add_library(${_Target} SHARED)
    target_sources(${_Target}
            PRIVATE
                ${_OpsTestUt_OpApiSources}
    )
    target_include_directories(${_Target}
            PRIVATE
                ${OPAPI_INCLUDE}
                ${_OpsTestUt_OpApiPrivateIncludesExt}
                ${ASCEND_CANN_PACKAGE_PATH}/runtime/pkg_inc
                ${ASCEND_CANN_PACKAGE_PATH}/runtime/pkg_inc/profiling
    )
    target_compile_options(${_Target}
            PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:-std=gnu++1z>
    )
    target_compile_definitions(${_Target}
            PRIVATE
                ACLNN_LOG_FMT_CHECK
                LOG_CPP
                PROCESS_LOG
    )
    target_link_libraries(${_Target}
            PRIVATE
                -Wl,--whole-archive
                ${_OpsTestUt_OpApiLinkLibrariesExt}
                $<BUILD_INTERFACE:intf_pub_utest>
                $<BUILD_INTERFACE:_OpsTestUt_OpApi_Wno>
                -Wl,--no-whole-archive
                PRIVATE $<$<BOOL:${BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG}>:$<BUILD_INTERFACE:opapi_math>>
                nnopbase
                profapi
                ge_common_base
                ascend_dump
                ascendalog
                dl
    )
    set(_UTest_OpApiLibrary ${_Target} PARENT_SCOPE)
endfunction()

# Level1, 添加算子 OpProto 动态库
#[[
调用参数:
  one_value_keywords:
      SUB_SYSTEM : 必选参数, 用于指定算子所属子系统, 如 transformer
      BRIEF : 必选参数, 用于指定算子缩略名(建议以大驼峰命名, 与算子实际名称无强制对应关系), 如 Fag/Fas/Fa
      SNAKE : 必选参数, 用于指定算子全名, 如 flash_attention_score_grad
  multi_value_keywords:
      SOURCES_EXT                   : 可选参数, 额外源文件
      PRIVATE_INCLUDES_EXT          : 可选参数, 额外头文件搜索路径
      PRIVATE_LINK_LIBRARIES_EXT    : 可选参数, 额外链接库
备注说明:
  本函数提供编译算子对应 OpProto.so 的功能. 下面介绍在调用本函数时所需了解的一些背景知识和注意事项.
  1. 本函数假设算子对应 OpProto 的源文件位于 'src/${SUB_SYSTEM}/${SNAKE}/op_host' 路径, 并自动添加以下内容:
     a) 源文件: ${SNAKE}_proto.cpp
     b) 头文件搜索路径: 'src/${SUB_SYSTEM}/${SNAKE}/op_host'
  2. 若算子 '需要额外的源文件’ 或 ‘源文件不满足上述约定的默认路径', 则可通过 SOURCES_EXT 参数指定 '额外的源文件';
  3. 若算子 '需要额外头文件搜索路径’ 或 ‘本函数现有实现的头文件搜索路径设置不满足',
     则可通过 PRIVATE_INCLUDES_EXT 参数指定 '额外的头文件搜索路径'; 参数 PRIVATE_LINK_LIBRARIES_EXT 设置逻辑同理;
]]
function(OpsTest_Level1_AddOpProtoShared)
    cmake_parse_arguments(
            TMP
            ""
            "SUB_SYSTEM;BRIEF;SNAKE"
            "SOURCES_EXT;PRIVATE_INCLUDES_EXT;PRIVATE_LINK_LIBRARIES_EXT"
            ""
            ${ARGN}
    )
    get_filename_component(_Src "${OPS_ADV_DIR}/${TMP_SUB_SYSTEM}/${TMP_SNAKE}/op_host/${TMP_SNAKE}_proto.cpp" REALPATH)
    if (NOT EXISTS "${_Src}")
        set(_Src)
    endif ()
    get_filename_component(_Inc "${OPS_ADV_DIR}/${TMP_SUB_SYSTEM}/${TMP_SNAKE}/op_host/" REALPATH)
    if (NOT EXISTS "${_Inc}")
        set(_Inc)
    endif ()

    set(_Sources ${TMP_SOURCES_EXT} ${_Src})
    list(REMOVE_DUPLICATES _Sources)
    if (_Sources)
        set(_PrivateIncludeDirs ${TMP_PRIVATE_INCLUDES_EXT} ${_Inc})
        list(REMOVE_DUPLICATES _PrivateIncludeDirs)
        set(_OpsTestUt_OpProtoSources            ${_OpsTestUt_OpProtoSources}            ${_Sources}                       CACHE INTERNAL "" FORCE)
        set(_OpsTestUt_OpProtoPrivateIncludesExt ${_OpsTestUt_OpProtoPrivateIncludesExt} ${_PrivateIncludeDirs}            CACHE INTERNAL "" FORCE)
        set(_OpsTestUt_OpProtoLinkLibrariesExt   ${_OpsTestUt_OpProtoLinkLibrariesExt}   ${TMP_PRIVATE_LINK_LIBRARIES_EXT} CACHE INTERNAL "" FORCE)
    endif ()
endfunction()

# 私有函数, 外部不可直接调用, 用于实际生成 OpProto 动态库
#[[
]]
function(OpsTest_AddOpProtoShared)
    list(REMOVE_DUPLICATES _OpsTestUt_OpProtoSources)
    list(REMOVE_DUPLICATES _OpsTestUt_OpProtoPrivateIncludesExt)
    list(REMOVE_DUPLICATES _OpsTestUt_OpProtoLinkLibrariesExt)
    if ("${_OpsTestUt_OpProtoSources}" STREQUAL "")
        return()
    endif ()
    set(_Target ${UTest_NamePrefix}_OpProto)
    add_library(${_Target} SHARED)
    target_sources(${_Target}
            PRIVATE
                ${_OpsTestUt_OpProtoSources}
    )
    target_include_directories(${_Target}
            PRIVATE
                ${OPBASE_INC_DIRS}
                ${_OpsTestUt_OpProtoPrivateIncludesExt}
    )
    target_compile_options(${_Target}
            PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:-std=gnu++1z>
    )
    target_compile_definitions(${_Target}
            PRIVATE
                LOG_CPP
                PROCESS_LOG
    )
    target_link_libraries(${_Target}
            PRIVATE
                -Wl,--whole-archive
                ${_OpsTestUt_OpProtoLinkLibrariesExt}
                $<BUILD_INTERFACE:intf_pub_utest>
                $<BUILD_INTERFACE:ops_transformer_utils_proto_headers>
                $<$<BOOL:${BUILD_OPEN_PROJECT}>:$<BUILD_INTERFACE:alog_headers>>
                $<$<BOOL:${BUILD_OPEN_PROJECT}>:$<BUILD_INTERFACE:dlog_headers>>
                -Wl,--no-whole-archive
                ascendalog
    )
    set(_UTest_OpProtoLibrary ${_Target} PARENT_SCOPE)
endfunction()

# Level1, 添加算子 OpTiling 动态库
#[[
调用参数:
  one_value_keywords:
      SUB_SYSTEM : 必选参数, 用于指定算子所属子系统, 如 transformer
      BRIEF : 必选参数, 用于指定算子缩略名(建议以大驼峰命名, 与算子实际名称无强制对应关系), 如 Fag/Fas/Fa
      SNAKE : 必选参数, 用于指定算子全名, 如 flash_attention_score_grad
  multi_value_keywords:
      SOURCES_EXT                   : 可选参数, 额外源文件
      PRIVATE_INCLUDES_EXT          : 可选参数, 额外头文件搜索路径
      PRIVATE_LINK_LIBRARIES_EXT    : 可选参数, 额外链接库
备注说明:
  本函数提供编译算子对应 OpTiling.so 的功能. 下面介绍在调用本函数时所需了解的一些背景知识和注意事项.
  1. 本函数假设算子对应 OpTiling 的源文件位于 'src/${SUB_SYSTEM}/${SNAKE}/op_host/${SNAKE}_tiling.cpp/.cc' 路径;
  2. 若算子 '需要额外的源文件’ 或 ‘源文件不满足上述约定的默认路径', 则可通过 SOURCES_EXT 参数指定 '额外的源文件';
  3. 若算子 '需要额外头文件搜索路径’ 或 ‘本函数现有实现的头文件搜索路径设置不满足',
     则可通过 PRIVATE_INCLUDES_EXT 参数指定 '额外的头文件搜索路径'; 参数 PRIVATE_LINK_LIBRARIES_EXT 设置逻辑同理;
]]
function(OpsTest_Level1_AddOpTilingShared)
    cmake_parse_arguments(
            TMP
            ""
            "SUB_SYSTEM;BRIEF;SNAKE"
            "SOURCES_EXT;PRIVATE_INCLUDES_EXT;PRIVATE_LINK_LIBRARIES_EXT"
            ""
            ${ARGN}
    )
    file(GLOB _Src1 "${OPS_ADV_DIR}/${TMP_SUB_SYSTEM}/${TMP_SNAKE}/op_host/${TMP_SNAKE}_tiling.cc")
    file(GLOB _Src2 "${OPS_ADV_DIR}/${TMP_SUB_SYSTEM}/${TMP_SNAKE}/op_host/${TMP_SNAKE}_tiling.cpp")
    list(APPEND _Sources ${TMP_SOURCES_EXT} ${_Src1} ${_Src2})
    list(REMOVE_DUPLICATES _Sources)
    get_filename_component(_Inc "${OPS_ADV_DIR}/${TMP_SUB_SYSTEM}/${TMP_SNAKE}/op_host/" REALPATH)
    if (NOT EXISTS "${_Inc}")
        set(_Inc)
    endif ()
    if (_Sources)
        set(_PrivateIncludeDirs ${TMP_PRIVATE_INCLUDES_EXT} ${_Inc})
        list(REMOVE_DUPLICATES _PrivateIncludeDirs)
        set(_OpsTestUt_OpTilingSources            ${_OpsTestUt_OpTilingSources}            ${_Sources}                       CACHE INTERNAL "" FORCE)
        set(_OpsTestUt_OpTilingPrivateIncludesExt ${_OpsTestUt_OpTilingPrivateIncludesExt} ${_PrivateIncludeDirs}            CACHE INTERNAL "" FORCE)
        set(_OpsTestUt_OpTilingLinkLibrariesExt   ${_OpsTestUt_OpTilingLinkLibrariesExt}   ${TMP_PRIVATE_LINK_LIBRARIES_EXT} CACHE INTERNAL "" FORCE)
    endif ()
endfunction()

# 私有函数, 外部不可直接调用, 用于实际生成 OpTiling 动态库
#[[
]]
function(OpsTest_AddOpTilingShared)
    list(REMOVE_DUPLICATES _OpsTestUt_OpTilingSources)
    list(REMOVE_DUPLICATES _OpsTestUt_OpTilingPrivateIncludesExt)
    list(REMOVE_DUPLICATES _OpsTestUt_OpTilingLinkLibrariesExt)
    set(_Target ${UTest_NamePrefix}_OpTiling)
    add_library(${_Target} SHARED)
    target_sources(${_Target}
            PRIVATE
                ${_OpsTestUt_OpTilingSources}
                ${OPS_ADV_DIR}/tests/ut/framework_special/stubs/tiling/tiling_templates_registry.cpp
    )
    target_include_directories(${_Target}
            PRIVATE
                ${OPBASE_INC_DIRS}
                ${_OpsTestUt_OpTilingPrivateIncludesExt}
                ${OPS_TRANSFORMER_DIR}/common/include
    )
    target_compile_definitions(${_Target}
            PRIVATE
                OP_TILING_LIB
                LOG_CPP
                PROCESS_LOG
    )
    target_compile_options(${_Target}
            PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:-std=c++11>
    )
    target_link_libraries(${_Target}
            PRIVATE
                -Wl,--as-needed
                -Wl,--no-whole-archive
                ${_OpsTestUt_OpTilingLinkLibrariesExt}
                $<BUILD_INTERFACE:intf_pub_utest>
                $<BUILD_INTERFACE:_OpsTestUt_OpTiling_Wno>
                $<BUILD_INTERFACE:ops_transformer_utils_tiling_headers>
                $<$<BOOL:${BUILD_OPEN_PROJECT}>:$<BUILD_INTERFACE:alog_headers>>
                $<$<BOOL:${BUILD_OPEN_PROJECT}>:$<BUILD_INTERFACE:dlog_headers>>
                graph
                graph_base
                exe_graph
                platform
                register
                ascendalog
                tiling_api
                c_sec
    )
    set(_UTest_OpTilingLibrary ${_Target} PARENT_SCOPE)
endfunction()

# Level1, 添加算子 Kernel 静态库
#[[
调用参数:
  one_value_keywords:
      SUB_SYSTEM : 必选参数, 用于指定算子所属子系统, 如 transformer
      BRIEF : 必选参数, 用于指定算子缩略名(建议以大驼峰命名, 与算子实际名称无强制对应关系), 如 Fag/Fas/Fa
      SNAKE : 必选参数, 用于指定算子全名, 如 flash_attention_score_grad
  multi_value_keywords:
      SOURCES_EXT                       : 可选参数, 额外源文件
      TILING_DATA_DEF_H                 : 可选参数, 算子 Kernel 所需 TilingData 定义头文件
      PRIVATE_INCLUDES_EXT              : 可选参数, 额外头文件搜索路径
      PRIVATE_COMPILE_DEFINITIONS_EXT   : 可选参数, 额外编译宏
备注说明:
  本函数提供编译算子对应 kernel.a 的功能. 下面介绍在调用本函数时所需了解的一些背景知识和注意事项.
  1. 本函数假设算子对应 Kernel 的源文件位于 'src/${SUB_SYSTEM}/${SNAKE}/${SNAKE}.cpp' 路径;
  2. 若算子 '需要额外的源文件’ 或 ‘源文件不满足上述约定的默认路径', 则可通过 SOURCES_EXT 参数指定 '额外的源文件';
  3. 当前 Ascend C 融合算子 Kernel 一般需要多个(>2) TilingData 定义头文件,
     如 flash_attention_score 算子需要 data_copy_transpose_tiling_def.h 和 flash_attention_score_tiling.h;
     - 在 NPU 编译时, 编译框架使用 ccec 编译器并使用 -include 编译选项注入所需的多个 TilingData 定义头文件 至 Kernel 源文件;
     - 但 CPU 编译时, CMake 早期版本在处理 -include 选项时有 Bug(只第一个 -include 指定的头文件生效).
       故本框架通过新增一个 {BRIEF}_tiling_data.h, 再在该文件中 include 所需的多个 TilingData 定义头文件,
       再通过 -include 编译选项注入 {BRIEF}_tiling_data.h 的方式规避 CPU 编译时 CMake 不能处理多个 -include 选项的 Bug;
  4. PRIVATE_COMPILE_DEFINITIONS_EXT 设置格式如下:
        optional{KernelCtrlParam func suffix} optional{OtherCompileDefinitions}}
     其中 KernelCtrlParam 用于指定 Kernel 编译控制参数, func 为标识 Kernel 原始入口函数名,
     suffix 为标识当前 Kernel 二进制及 Kernel 入口函数后缀; 其设置逻辑如下:
     4.0 KernelCtrlParam 内 func, suffix 必需按序设置;
     4.1 当不设置 KernelCtrlParam 时, 仅会编译一个 Kernel.a 并以 OtherCompileDefinitions 设置编译 Definitions(如设置);
     4.2 若当设置 KernelCtrlParam 时, 本函数会按 KernelCtrlParam 个数逐个编译 Kernel.a,
         并以 KernelCtrlParam 间 OtherCompileDefinitions 设置编译宏定义; 并将 func 重命名为 {func}_{suffix};
     4.3 KernelCtrlParam 内 func 支持指定多个, 指定多个时 func 之间用','间隔;
]]
function(OpsTest_Level1_AddOpKernelStatic)
    cmake_parse_arguments(
            TMP
            ""
            "SUB_SYSTEM;BRIEF;SNAKE"
            "SOURCES_EXT;TILING_DATA_DEF_H;PRIVATE_INCLUDES_EXT;PRIVATE_COMPILE_DEFINITIONS_EXT"
            ""
            ${ARGN}
    )
    # 生成 Kernel 所需的结构体表示的对应 tiling.h
    string(TOLOWER ${TMP_BRIEF} tmp_brief)
    set(_tmp_files ${TMP_TILING_DATA_DEF_H})
    list(REMOVE_DUPLICATES _tmp_files)
    set(_ori_files)
    set(_define_py ${OPS_ADV_DIR}/cmake/scripts/utest/gen_tiling_data_stub.py)
    foreach (_tmp ${_tmp_files})
        if (EXISTS ${_tmp})
            list(APPEND _ori_files "-s=${_tmp}")
        else ()
            message(FATAL_ERROR "${_tmp} not exist.")
        endif ()
    endforeach ()
    # 生成目标根目录
    get_filename_component(_OpsTest_GenDir "${CMAKE_CURRENT_BINARY_DIR}/gen" REALPATH)
    get_filename_component(_OpsTest_GenDirInc "${_OpsTest_GenDir}/inc" REALPATH)
    execute_process(
            COMMAND ${HI_PYTHON} ${_define_py} "-o=${tmp_brief}" ${_ori_files} "-d=${_OpsTest_GenDirInc}"
    )
    set(_Target ${UTest_NamePrefix}_${TMP_BRIEF}_OpTilingDataDef)
    add_library(${_Target} INTERFACE)
    target_include_directories(${_Target} INTERFACE ${_OpsTest_GenDirInc} ${OPS_ADV_UTILS_KERNEL_INC})

    # 编译变量处理
    set(_TargetPrefix  ${UTest_NamePrefix}_${TMP_BRIEF}_OpKernel)
    aux_source_directory(${OPS_ADV_DIR}/${TMP_SUB_SYSTEM}/${TMP_SNAKE} _Sources)
    list(APPEND _Sources ${TMP_SOURCES_EXT})
    set(_PrivateIncludeDirectories
            ${_OpsTest_GenDirInc}
            ${TMP_PRIVATE_INCLUDES_EXT}
            <ASCEND_CANN_PACKAGE_PATH>/${SYSTEM_PREFIX}/ascendc/include/basic_api
            <ASCEND_CANN_PACKAGE_PATH>/${SYSTEM_PREFIX}/ascendc/include/basic_api/impl
            <ASCEND_CANN_PACKAGE_PATH>/${SYSTEM_PREFIX}/ascendc/include/basic_api/interface
            <ASCEND_CANN_PACKAGE_PATH>/tools/tikicpulib/lib/include
            <ASCEND_CANN_PACKAGE_PATH>/include/ascendc
            <ASCEND_CANN_PACKAGE_PATH>/x86_64-linux/include/ascendc/highlevel_api
            <ASCEND_CANN_PACKAGE_PATH>/pkg_inc/runtime/runtime
            <ASCEND_CANN_PACKAGE_PATH>/runtime/pkg_inc
    )
    get_filename_component(_Inc "${OPS_ADV_DIR}/${TMP_SUB_SYSTEM}/${TMP_SNAKE}" REALPATH)
    if (EXISTS "${_Inc}")
        list(APPEND _PrivateIncludeDirectories ${_Inc})
    endif ()
    set(_PrivateCompileOptions
            -include ${_OpsTest_GenDirInc}/tiling/${tmp_brief}/tiling_stub.h
    )
    set(_PrivateLinkLibraries
            -Wl,--as-needed
            -Wl,--no-whole-archive
            c_sec
            $<BUILD_INTERFACE:intf_pub_utest>
            $<BUILD_INTERFACE:_OpsTestUt_OpKernel_Wno>
    )

    # 多 Kernel 处理
    set(_OpKernelLibraries)
    list(FIND TMP_PRIVATE_COMPILE_DEFINITIONS_EXT KernelCtrlParam _GrpIdx)
    if ("${_GrpIdx}" STREQUAL "-1")
        # 不存在多 Kernel 配置时, 不添加后缀, 编译一个 Kernel.a
        set(_Target ${_TargetPrefix})
        add_library(${_Target} STATIC)
        target_sources(${_Target} PRIVATE ${_Sources})
        target_include_directories(${_Target} PRIVATE ${_PrivateIncludeDirectories})
        target_compile_definitions(${_Target} PRIVATE ${TMP_PRIVATE_COMPILE_DEFINITIONS_EXT})
        target_compile_options(${_Target} PRIVATE ${_PrivateCompileOptions})
        target_link_libraries(${_Target}
                PUBLIC
                    tikicpulib::${OPS_ADV_UTEST_OPS_TEST_ASCEND_PRODUCT_TYPE}
                PRIVATE
                    ${_PrivateLinkLibraries}
        )
        list(APPEND _OpKernelLibraries ${_Target})
    else ()
        # 存在 1/n 多 Kernel 配置时, 添加后缀, 编译多 Kernel.a
        while (NOT "${_GrpIdx}" STREQUAL "-1")
            # 获取当前 Func, Suffix, CompileDefinitions
            math(EXPR _FuncIdx "${_GrpIdx} + 1")
            math(EXPR _SuffixIdx "${_GrpIdx} + 2")
            math(EXPR _SubLstBgnIdx "${_GrpIdx} + 3")
            list(GET TMP_PRIVATE_COMPILE_DEFINITIONS_EXT ${_FuncIdx} _FuncOriValList)
            list(GET TMP_PRIVATE_COMPILE_DEFINITIONS_EXT ${_SuffixIdx} _SuffixVal)
            list(SUBLIST TMP_PRIVATE_COMPILE_DEFINITIONS_EXT ${_SubLstBgnIdx} -1 TMP_PRIVATE_COMPILE_DEFINITIONS_EXT)
            list(FIND TMP_PRIVATE_COMPILE_DEFINITIONS_EXT KernelCtrlParam _GrpIdx)  # _GrpIdx 移动
            list(SUBLIST TMP_PRIVATE_COMPILE_DEFINITIONS_EXT 0 ${_GrpIdx} _SubCompileDefinitions)
            string(REPLACE "," ";" _FuncOriValList "${_FuncOriValList}")
            set(_FuncValDef)
            foreach (_f ${_FuncOriValList})
                list(APPEND _FuncValDef "-D${_f}=${_f}_${_SuffixVal}")
            endforeach ()
            # 编译目标
            set(_Target ${_TargetPrefix}_${_SuffixVal})
            add_library(${_Target} STATIC)
            target_sources(${_Target} PRIVATE ${_Sources})
            target_include_directories(${_Target} PRIVATE ${_PrivateIncludeDirectories})
            target_compile_definitions(${_Target} PRIVATE ${_FuncValDef} ${_SubCompileDefinitions})
            target_compile_options(${_Target} PRIVATE ${_PrivateCompileOptions})
            target_link_libraries(${_Target}
                    PUBLIC
                        tikicpulib::${OPS_ADV_UTEST_OPS_TEST_ASCEND_PRODUCT_TYPE}
                    PRIVATE
                        ${_PrivateLinkLibraries}
            )
            list(APPEND _OpKernelLibraries ${_Target})
        endwhile ()
    endif ()
    set(_OpsTestUt_OpKernelLibraries ${_OpKernelLibraries} CACHE INTERNAL "" FORCE)
endfunction()

# Level1, 添加算子 UTest 用例 Common 静态库
#[[
调用参数:
  one_value_keywords:
      BRIEF : 必选参数, 用于指定算子缩略名(建议以大驼峰命名, 与算子实际名称无强制对应关系), 如 Fag/Fas/Fa
      SNAKE : 必选参数, 用于指定算子全名, 如 flash_attention_score_grad
      TILING_DATA_DEF_BRIEF: 可选参数, 用于指定链接的 TilingData 头文件目标
  multi_value_keywords:
      SOURCES_EXT                  : 可选参数, 额外源文件
      SOURCES_EXT_FORCE            : 可选参数, 强制使用 SOURCES_EXT 指定的额外源文件 (Common), 不再自动寻找 comm 下源文件
      PUBLIC_INCLUDES_EXT          : 可选参数, 额外头文件搜索路径(public)
      PRIVATE_INCLUDES_EXT         : 可选参数, 额外头文件搜索路径
      PRIVATE_LINK_LIBRARIES_EXT   : 可选参数, 额外链接库
      PRIVATE_COMPILE_DEFINITIONS  : 可选参数, 额外的编译宏定义
]]
function(OpsTest_Level1_AddUTestCommonStatic)
    cmake_parse_arguments(
            TMP
            ""
            "BRIEF;SNAKE;TILING_DATA_DEF_BRIEF"
            "SOURCES_EXT;SOURCES_EXT_FORCE;PUBLIC_INCLUDES_EXT;PRIVATE_INCLUDES_EXT;PRIVATE_LINK_LIBRARIES_EXT;PRIVATE_COMPILE_DEFINITIONS"
            ""
            ${ARGN}
    )
    if (NOT TMP_TILING_DATA_DEF_BRIEF)
        set(TMP_TILING_DATA_DEF_BRIEF ${TMP_BRIEF})
    endif ()

    # 强制指定源码处理
    set(_Src1)
    set(_Src2)
    set(_Common_Sources ${TMP_SOURCES_EXT})
    if (NOT TMP_SOURCES_EXT_FORCE)
        if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/comm)
            if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/comm/inc)
                aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/comm/src _Src1)
            else ()
                file(GLOB_RECURSE _Src1 "${CMAKE_CURRENT_SOURCE_DIR}/comm/*.cc")
                file(GLOB_RECURSE _Src2 "${CMAKE_CURRENT_SOURCE_DIR}/comm/*.cpp")
            endif ()
        endif ()
    endif ()
    list(APPEND _Common_Sources ${_Src1} ${_Src2})
    list(REMOVE_DUPLICATES _Common_Sources)

    set(_Target_UTest_Common)
    if (_Common_Sources)
        set(_Target_UTest_Common ${UTest_NamePrefix}_${TMP_BRIEF}_UTest_Common)
        add_library(${_Target_UTest_Common} STATIC)
        target_sources(${_Target_UTest_Common} PRIVATE ${_Common_Sources})
        if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/comm/inc)
            target_include_directories(${_Target_UTest_Common} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/comm/inc)
        elseif (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/comm)
            target_include_directories(${_Target_UTest_Common} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/comm)
        endif ()
        target_include_directories(${_Target_UTest_Common}
                PUBLIC
                    ${TMP_PUBLIC_INCLUDES_EXT}
                PRIVATE
                    ${OPBASE_INC_DIRS}
                    ${OPS_TRANSFORMER_DIR}/common/include
                    ${ASCEND_CANN_PACKAGE_PATH}/include
                    ${TMP_PRIVATE_INCLUDES_EXT}
        )
        target_compile_definitions(${_Target_UTest_Common}
                PRIVATE
                    ${TMP_PRIVATE_COMPILE_DEFINITIONS}
        )
        target_link_libraries(${_Target_UTest_Common}
                PRIVATE
                    -Wl,--as-needed
                    -Wl,--no-whole-archive
                    platform
                    register
                    opp_registry
                    error_manager
                    metadef
                    tiling_api
                    $<BUILD_INTERFACE:intf_pub_utest>
                    $<BUILD_INTERFACE:_OpsTestUt_UTestCaseStatic_Wno>
                    $<$<BOOL:${BUILD_OPEN_PROJECT}>:$<BUILD_INTERFACE:alog_headers>>
                    $<$<BOOL:${BUILD_OPEN_PROJECT}>:$<BUILD_INTERFACE:dlog_headers>>
                    tikicpulib::${OPS_ADV_UTEST_OPS_TEST_ASCEND_PRODUCT_TYPE}
                    ${UTest_NamePrefix}_${TMP_TILING_DATA_DEF_BRIEF}_OpTilingDataDef
                    ${UTest_NamePrefix}_Utils
                    ${TMP_PRIVATE_LINK_LIBRARIES_EXT}
                    ${_OpsTestUt_OpKernelLibraries}  # 当前各算子 common 内以 extern 方式声明 kernel 入口, 故添加依赖
        )
    endif ()
    set(_OpsTestUt_UTestCommonLibrary ${_Target_UTest_Common} CACHE INTERNAL "" FORCE)
endfunction()

# Level1, 添加算子 UTest 用例静态库
#[[
调用参数:
  one_value_keywords:
      BRIEF : 必选参数, 用于指定算子缩略名(建议以大驼峰命名, 与算子实际名称无强制对应关系), 如 Fag/Fas/Fa
      SNAKE : 必选参数, 用于指定算子全名, 如 flash_attention_score_grad
      TILING_DATA_DEF_BRIEF: 可选参数, 用于指定链接的 TilingData 头文件目标
  multi_value_keywords:
      SOURCES_EXT                         : 可选参数, 额外源文件
      SOURCES_EXT_FORCE                   : 可选参数, 强制使用 SOURCES_EXT 指定的额外源文件 (UTest), 不再自动寻找 utest 下源文件
      PRIVATE_INCLUDES_EXT                : 可选参数, 额外头文件搜索路径
      PRIVATE_LINK_LIBRARIES_EXT          : 可选参数, 额外链接库
      PRIVATE_COMPILE_DEFINITIONS         : 可选参数, 额外的编译宏定义
]]
function(OpsTest_Level1_AddUTestCaseStatic)
    cmake_parse_arguments(
            TMP
            ""
            "BRIEF;SNAKE;TILING_DATA_DEF_BRIEF"
            "SOURCES_EXT;SOURCES_EXT_FORCE;PRIVATE_INCLUDES_EXT;PRIVATE_LINK_LIBRARIES_EXT;PRIVATE_COMPILE_DEFINITIONS"
            ""
            ${ARGN}
    )
    if (NOT TMP_TILING_DATA_DEF_BRIEF)
        set(TMP_TILING_DATA_DEF_BRIEF ${TMP_BRIEF})
    endif ()

    # 强制指定源码处理
    set(_UTest_Sources ${TMP_SOURCES_EXT})
    if (NOT TMP_SOURCES_EXT_FORCE)
        file(GLOB_RECURSE _Src1 "${CMAKE_CURRENT_SOURCE_DIR}/utest/*.cc")
        file(GLOB_RECURSE _Src2 "${CMAKE_CURRENT_SOURCE_DIR}/utest/*.cpp")
        list(APPEND _UTest_Sources ${_Src1} ${_Src2})
    endif ()
    list(REMOVE_DUPLICATES _UTest_Sources)
    if (_UTest_Sources)
        set(_Target_UTest ${UTest_NamePrefix}_${TMP_BRIEF}_UTest_Case)
        add_library(${_Target_UTest} STATIC)
        target_sources(${_Target_UTest} PRIVATE ${_UTest_Sources})
        target_include_directories(${_Target_UTest}
                PRIVATE
                    ${ASCEND_CANN_PACKAGE_PATH}/include
                    ${TMP_PRIVATE_INCLUDES_EXT}
                    ${CMAKE_CURRENT_SOURCE_DIR}/utest
        )
        target_compile_definitions(${_Target_UTest}
                PRIVATE
                    ${TMP_PRIVATE_COMPILE_DEFINITIONS}
        )
        target_link_libraries(${_Target_UTest}
                PRIVATE
                    -Wl,--as-needed
                    -Wl,--no-whole-archive
                    gtest
                    $<BUILD_INTERFACE:intf_pub_utest>
                    $<BUILD_INTERFACE:_OpsTestUt_UTestCaseStatic_Wno>
                    $<$<BOOL:${BUILD_OPEN_PROJECT}>:$<BUILD_INTERFACE:alog_headers>>
                    $<$<BOOL:${BUILD_OPEN_PROJECT}>:$<BUILD_INTERFACE:dlog_headers>>
                    tikicpulib::${OPS_ADV_UTEST_OPS_TEST_ASCEND_PRODUCT_TYPE}
                    ${UTest_NamePrefix}_${TMP_TILING_DATA_DEF_BRIEF}_OpTilingDataDef
                    ${UTest_NamePrefix}_Utils
                    $<BUILD_INTERFACE:_OpsTestUt_GTest_Wno>
                    ${UTest_NamePrefix}_Utest
                    ${TMP_PRIVATE_LINK_LIBRARIES_EXT}
                    ${_OpsTestUt_UTestCommonLibrary}
        )
        set(_OpsTestUt_UTestCaseLibraries ${_OpsTestUt_UTestCaseLibraries} ${_Target_UTest} CACHE INTERNAL "" FORCE)
    endif ()
endfunction()

# Level1, 添加算子 UTest 用例(Aclnn)静态库
#[[
调用参数:
  one_value_keywords:
      BRIEF : 必选参数, 用于指定算子缩略名(建议以大驼峰命名, 与算子实际名称无强制对应关系), 如 Fag/Fas/Fa
      SNAKE : 必选参数, 用于指定算子全名, 如 flash_attention_score_grad
      TILING_DATA_DEF_BRIEF: 可选参数, 用于指定链接的 TilingData 头文件目标
  multi_value_keywords:
      SOURCES_EXT                   : 可选参数, 额外源文件
      SOURCES_EXT_FORCE             : 可选参数, 强制使用 SOURCES_EXT 指定的额外源文件 (UTest_Aclnn), 不再自动寻找 utest_aclnn 下源文件
      PRIVATE_INCLUDES_EXT          : 可选参数, 额外头文件搜索路径
      PRIVATE_COMPILE_DEFINITIONS   : 可选参数, 额外的编译宏定义
      PRIVATE_LINK_LIBRARIES_EXT    : 可选参数, 额外链接库
]]
function(OpsTest_Level1_AddUTestAclnnCaseStatic)
    cmake_parse_arguments(
            TMP
            ""
            "BRIEF;SNAKE;TILING_DATA_DEF_BRIEF"
            "SOURCES_EXT;SOURCES_EXT_FORCE;PRIVATE_INCLUDES_EXT;PRIVATE_COMPILE_DEFINITIONS;PRIVATE_LINK_LIBRARIES_EXT"
            ""
            ${ARGN}
    )
    if (NOT TMP_TILING_DATA_DEF_BRIEF)
        set(TMP_TILING_DATA_DEF_BRIEF ${TMP_BRIEF})
    endif ()

    # 强制指定源码处理
    set(_UTest_Aclnn_Sources ${TMP_SOURCES_EXT})
    if (NOT TMP_SOURCES_EXT_FORCE)
        file(GLOB_RECURSE _Src1 "${CMAKE_CURRENT_SOURCE_DIR}/utest_aclnn/*.cc")
        file(GLOB_RECURSE _Src2 "${CMAKE_CURRENT_SOURCE_DIR}/utest_aclnn/*.cpp")
        list(APPEND _UTest_Aclnn_Sources ${_Src1} ${_Src2})
    endif ()
    list(REMOVE_DUPLICATES _UTest_Aclnn_Sources)
    if (_UTest_Aclnn_Sources)
        set(_Target_UTest_Aclnn ${UTest_NamePrefix}_${TMP_BRIEF}_UTest_Case_Aclnn)
        add_library(${_Target_UTest_Aclnn} STATIC)
        target_sources(${_Target_UTest_Aclnn} PRIVATE ${_UTest_Aclnn_Sources})
        target_include_directories(${_Target_UTest_Aclnn}
                PRIVATE
                    ${ASCEND_CANN_PACKAGE_PATH}/include
                    ${TMP_PRIVATE_INCLUDES_EXT}
                    ${CMAKE_CURRENT_SOURCE_DIR}/utest_aclnn
        )
        target_compile_definitions(${_Target_UTest_Aclnn}
                PRIVATE
                    ${TMP_PRIVATE_COMPILE_DEFINITIONS}
        )
        target_link_libraries(${_Target_UTest_Aclnn}
                PRIVATE
                    -Wl,--as-needed
                    -Wl,--no-whole-archive
                    gtest
                    $<BUILD_INTERFACE:intf_pub_utest>
                    $<BUILD_INTERFACE:_OpsTestUt_UTestCaseStatic_Wno>
                    $<$<BOOL:${BUILD_OPEN_PROJECT}>:$<BUILD_INTERFACE:alog_headers>>
                    $<$<BOOL:${BUILD_OPEN_PROJECT}>:$<BUILD_INTERFACE:dlog_headers>>
                    tikicpulib::${OPS_ADV_UTEST_OPS_TEST_ASCEND_PRODUCT_TYPE}
                    ${UTest_NamePrefix}_${TMP_TILING_DATA_DEF_BRIEF}_OpTilingDataDef
                    ${UTest_NamePrefix}_Utils
                    $<BUILD_INTERFACE:_OpsTestUt_GTest_Wno>
                    ${UTest_NamePrefix}_Utest
                    ${TMP_PRIVATE_LINK_LIBRARIES_EXT}
                    ${_OpsTestUt_UTestCommonLibrary}
        )
        set(_OpsTestUt_UTestAclnnCaseLibraries ${_OpsTestUt_UTestAclnnCaseLibraries} ${_Target_UTest_Aclnn} CACHE INTERNAL "" FORCE)
    endif ()
endfunction()

# Level2, 添加算子 UTest 用例动态库 及其所需全部库(OpTiling.so, OpKernel.a 等)
#[[
调用参数:
  one_value_keywords:
      SUB_SYSTEM : 必选参数, 用于指定算子所属子系统, 如 transformer
      BRIEF : 必选参数, 用于指定算子缩略名(建议以大驼峰命名, 与算子实际名称无强制对应关系), 如 Fag/Fas/Fa
      SNAKE : 必选参数, 用于指定算子全名, 如 flash_attention_score_grad
      TILING_DATA_DEF_BRIEF: 可选参数, 用于指定链接的 TilingData 头文件目标
  multi_value_keywords:
      OPAPI_SOURCES_EXT                         : 透传参数, 详情参见 OpsTest_Level1_AddOpApiShared 函数说明
      OPAPI_PRIVATE_INCLUDES_EXT                : 透传参数, 详情参见 OpsTest_Level1_AddOpApiShared 函数说明
      OPAPI_PRIVATE_LINK_LIBRARIES_EXT          : 透传参数, 详情参见 OpsTest_Level1_AddOpApiShared 函数说明
      PROTO_SOURCES_EXT                         : 透传参数, 详情参见 OpsTest_Level1_AddOpProtoShared 函数说明
      PROTO_PRIVATE_INCLUDES_EXT                : 透传参数, 详情参见 OpsTest_Level1_AddOpProtoShared 函数说明
      PROTO_PRIVATE_LINK_LIBRARIES_EXT          : 透传参数, 详情参见 OpsTest_Level1_AddOpProtoShared 函数说明
      TILING_SOURCES_EXT                        : 透传参数, 详情参见 OpsTest_Level1_AddOpTilingShared 函数说明
      TILING_PRIVATE_INCLUDES_EXT               : 透传参数, 详情参见 OpsTest_Level1_AddOpTilingShared 函数说明
      TILING_PRIVATE_LINK_LIBRARIES_EXT         : 透传参数, 详情参见 OpsTest_Level1_AddOpTilingShared 函数说明
      KERNEL_SOURCES_EXT                        : 透传参数, 详情参见 OpsTest_Level1_AddOpKernelStatic 函数说明
      KERNEL_TILING_DATA_DEF_H                  : 透传参数, 详情参见 OpsTest_Level1_AddOpKernelStatic 函数说明
      KERNEL_PRIVATE_INCLUDES_EXT               : 透传参数, 详情参见 OpsTest_Level1_AddOpKernelStatic 函数说明
      KERNEL_PRIVATE_COMPILE_DEFINITIONS_EXT    : 透传参数, 详情参见 OpsTest_Level1_AddOpKernelStatic 函数说明
      UTEST_COMMON_SOURCES_EXT                  : 透传参数, 详情参见 OpsTest_Level1_AddUTestCommonStatic 函数说明
      UTEST_COMMON_SOURCES_EXT_FORCE            : 透传参数, 详情参见 OpsTest_Level1_AddUTestCommonStatic 函数说明
      UTEST_COMMON_PUBLIC_INCLUDES_EXT          : 透传参数, 详情参见 OpsTest_Level1_AddUTestCommonStatic 函数说明
      UTEST_COMMON_PRIVATE_INCLUDES_EXT         : 透传参数, 详情参见 OpsTest_Level1_AddUTestCommonStatic 函数说明
      UTEST_COMMON_PRIVATE_COMPILE_DEFINITIONS  : 透传参数, 详情参见 OpsTest_Level1_AddUTestCommonStatic 函数说明
      UTEST_COMMON_PRIVATE_LINK_LIBRARIES_EXT   : 透传参数, 详情参见 OpsTest_Level1_AddUTestCommonStatic 函数说明
      UTEST_SOURCES_EXT                         : 透传参数, 详情参见 OpsTest_Level1_AddUTestCaseStatic 函数说明
      UTEST_SOURCES_EXT_FORCE                   : 透传参数, 详情参见 OpsTest_Level1_AddUTestCaseStatic 函数说明
      UTEST_PRIVATE_INCLUDES_EXT                : 透传参数, 详情参见 OpsTest_Level1_AddUTestCaseStatic 函数说明
      UTEST_PRIVATE_COMPILE_DEFINITIONS         : 透传参数, 详情参见 OpsTest_Level1_AddUTestCaseStatic 函数说明
      UTEST_PRIVATE_LINK_LIBRARIES_EXT          : 透传参数, 详情参见 OpsTest_Level1_AddUTestCaseStatic 函数说明
      UTEST_ACLNN_SOURCES_EXT                   : 透传参数, 详情参见 OpsTest_Level1_AddUTestAclnnCaseStatic 函数说明
      UTEST_ACLNN_SOURCES_EXT_FORCE             : 透传参数, 详情参见 OpsTest_Level1_AddUTestAclnnCaseStatic 函数说明
      UTEST_ACLNN_PRIVATE_INCLUDES_EXT          : 透传参数, 详情参见 OpsTest_Level1_AddUTestAclnnCaseStatic 函数说明
      UTEST_ACLNN_PRIVATE_COMPILE_DEFINITIONS   : 透传参数, 详情参见 OpsTest_Level1_AddUTestAclnnCaseStatic 函数说明
      UTEST_ACLNN_PRIVATE_LINK_LIBRARIES_EXT    : 透传参数, 详情参见 OpsTest_Level1_AddUTestAclnnCaseStatic 函数说明
]]
function(OpsTest_Level2_AddOp)
    cmake_parse_arguments(
            TMP
            ""
            "SUB_SYSTEM;BRIEF;SNAKE;TILING_DATA_DEF_BRIEF"
            "OPAPI_SOURCES_EXT;OPAPI_PRIVATE_INCLUDES_EXT;OPAPI_PRIVATE_LINK_LIBRARIES_EXT;PROTO_SOURCES_EXT;PROTO_PRIVATE_INCLUDES_EXT;PROTO_PRIVATE_LINK_LIBRARIES_EXT;TILING_SOURCES_EXT;TILING_PRIVATE_INCLUDES_EXT;TILING_PRIVATE_LINK_LIBRARIES_EXT;KERNEL_SOURCES_EXT;KERNEL_TILING_DATA_DEF_H;KERNEL_PRIVATE_INCLUDES_EXT;KERNEL_PRIVATE_COMPILE_DEFINITIONS_EXT;UTEST_COMMON_SOURCES_EXT;UTEST_COMMON_SOURCES_EXT_FORCE;UTEST_COMMON_PUBLIC_INCLUDES_EXT;UTEST_COMMON_PRIVATE_INCLUDES_EXT;UTEST_COMMON_PRIVATE_COMPILE_DEFINITIONS;UTEST_COMMON_PRIVATE_LINK_LIBRARIES_EXT;UTEST_SOURCES_EXT;UTEST_SOURCES_EXT_FORCE;UTEST_PRIVATE_INCLUDES_EXT;UTEST_PRIVATE_COMPILE_DEFINITIONS;UTEST_PRIVATE_LINK_LIBRARIES_EXT;UTEST_ACLNN_SOURCES_EXT;UTEST_ACLNN_SOURCES_EXT_FORCE;UTEST_ACLNN_PRIVATE_INCLUDES_EXT;UTEST_ACLNN_PRIVATE_COMPILE_DEFINITIONS;UTEST_ACLNN_PRIVATE_LINK_LIBRARIES_EXT"
            ""
            ${ARGN}
    )

    OpsTest_Level1_AddOpApiShared(
            SUB_SYSTEM                          ${TMP_SUB_SYSTEM}
            BRIEF                               ${TMP_BRIEF}
            SNAKE                               ${TMP_SNAKE}
            SOURCES_EXT                         ${TMP_OPAPI_SOURCES_EXT}
            PRIVATE_INCLUDES_EXT                ${TMP_OPAPI_PRIVATE_INCLUDES_EXT}
            PRIVATE_LINK_LIBRARIES_EXT          ${TMP_OPAPI_PRIVATE_LINK_LIBRARIES_EXT}
    )
    OpsTest_Level1_AddOpProtoShared(
            SUB_SYSTEM                          ${TMP_SUB_SYSTEM}
            BRIEF                               ${TMP_BRIEF}
            SNAKE                               ${TMP_SNAKE}
            SOURCES_EXT                         ${TMP_PROTO_SOURCES_EXT}
            PRIVATE_INCLUDES_EXT                ${TMP_PROTO_PRIVATE_INCLUDES_EXT}
            PRIVATE_LINK_LIBRARIES_EXT          ${TMP_PROTO_PRIVATE_LINK_LIBRARIES_EXT}
    )
    OpsTest_Level1_AddOpTilingShared(
            SUB_SYSTEM                          ${TMP_SUB_SYSTEM}
            BRIEF                               ${TMP_BRIEF}
            SNAKE                               ${TMP_SNAKE}
            SOURCES_EXT                         ${TMP_TILING_SOURCES_EXT}
            PRIVATE_INCLUDES_EXT                ${TMP_TILING_PRIVATE_INCLUDES_EXT}
            PRIVATE_LINK_LIBRARIES_EXT          ${TMP_TILING_PRIVATE_LINK_LIBRARIES_EXT}
    )
    OpsTest_Level1_AddOpKernelStatic(
            SUB_SYSTEM                       ${TMP_SUB_SYSTEM}
            BRIEF                            ${TMP_BRIEF}
            SNAKE                            ${TMP_SNAKE}
            SOURCES_EXT                      ${TMP_KERNEL_SOURCES_EXT}
            TILING_DATA_DEF_H                ${TMP_KERNEL_TILING_DATA_DEF_H}
            PRIVATE_INCLUDES_EXT             ${TMP_KERNEL_PRIVATE_INCLUDES_EXT}
            PRIVATE_COMPILE_DEFINITIONS_EXT  ${TMP_KERNEL_PRIVATE_COMPILE_DEFINITIONS_EXT}
    )
    OpsTest_Level1_AddUTestCommonStatic(
            BRIEF                               ${TMP_BRIEF}
            SNAKE                               ${TMP_SNAKE}
            SOURCES_EXT                         ${TMP_UTEST_COMMON_SOURCES_EXT}
            SOURCES_EXT_FORCE                   ${TMP_UTEST_COMMON_SOURCES_EXT_FORCE}
            TILING_DATA_DEF_BRIEF               ${TMP_TILING_DATA_DEF_BRIEF}
            PUBLIC_INCLUDES_EXT                 ${TMP_UTEST_COMMON_PUBLIC_INCLUDES_EXT}
            PRIVATE_INCLUDES_EXT                ${TMP_UTEST_COMMON_PRIVATE_INCLUDES_EXT}
            PRIVATE_COMPILE_DEFINITIONS         ${TMP_UTEST_COMMON_PRIVATE_COMPILE_DEFINITIONS}
            PRIVATE_LINK_LIBRARIES_EXT          ${TMP_UTEST_COMMON_PRIVATE_LINK_LIBRARIES_EXT}
    )
    OpsTest_Level1_AddUTestCaseStatic(
            BRIEF                               ${TMP_BRIEF}
            SNAKE                               ${TMP_SNAKE}
            TILING_DATA_DEF_BRIEF               ${TMP_TILING_DATA_DEF_BRIEF}
            SOURCES_EXT                         ${TMP_UTEST_SOURCES_EXT}
            SOURCES_EXT_FORCE                   ${TMP_UTEST_SOURCES_EXT_FORCE}
            PRIVATE_INCLUDES_EXT                ${TMP_UTEST_PRIVATE_INCLUDES_EXT}
            PRIVATE_COMPILE_DEFINITIONS         ${TMP_UTEST_PRIVATE_COMPILE_DEFINITIONS}
            PRIVATE_LINK_LIBRARIES_EXT          ${TMP_UTEST_PRIVATE_LINK_LIBRARIES_EXT}
    )
    OpsTest_Level1_AddUTestAclnnCaseStatic(
            BRIEF                               ${TMP_BRIEF}
            SNAKE                               ${TMP_SNAKE}
            TILING_DATA_DEF_BRIEF               ${TMP_TILING_DATA_DEF_BRIEF}
            SOURCES_EXT                         ${TMP_UTEST_ACLNN_SOURCES_EXT}
            SOURCES_EXT_FORCE                   ${TMP_UTEST_ACLNN_SOURCES_EXT_FORCE}
            PRIVATE_INCLUDES_EXT                ${TMP_UTEST_ACLNN_PRIVATE_INCLUDES_EXT}
            PRIVATE_COMPILE_DEFINITIONS         ${TMP_UTEST_ACLNN_PRIVATE_COMPILE_DEFINITIONS}
            PRIVATE_LINK_LIBRARIES_EXT          ${TMP_UTEST_ACLNN_PRIVATE_LINK_LIBRARIES_EXT}
    )
endfunction()

function(op_add_ut_subdirectory OP_UT_LIST OP_UT_DIR_LIST)
    set(_OP_UT_LIST)
    set(_OP_UT_DIR_LIST)

    file(GLOB OP_HOST_CMAKE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/**/**/tests/CMakeLists.txt")

    foreach(OP_CMAKE_FILE ${OP_HOST_CMAKE_FILES})
        if ("${OP_CMAKE_FILE}" MATCHES "tests")
            get_filename_component(OP_HOST_DIR "${OP_CMAKE_FILE}" DIRECTORY)
            get_filename_component(OP_DIR "${OP_HOST_DIR}" DIRECTORY)
        else()
            get_filename_component(OP_DIR "${OP_CMAKE_FILE}" DIRECTORY)
        endif()
        get_filename_component(OP_NAME "${OP_DIR}" NAME)

        if (DEFINED ASCEND_OP_NAME AND NOT "${ASCEND_OP_NAME}" STREQUAL "")
            if (NOT "${ASCEND_OP_NAME}" STREQUAL "all" AND NOT "${ASCEND_OP_NAME}" STREQUAL "ALL")
                if (NOT ${OP_NAME} IN_LIST ASCEND_OP_NAME)
                    continue()
                endif ()
            endif ()
        endif ()        

        list(APPEND _OP_UT_LIST ${OP_NAME})
        list(APPEND _OP_UT_DIR_LIST ${OP_DIR})
    endforeach()

    list(REMOVE_DUPLICATES _OP_UT_LIST)
    list(REMOVE_DUPLICATES _OP_UT_DIR_LIST)
    list(SORT _OP_UT_LIST)
    list(SORT _OP_UT_DIR_LIST)
    set(${OP_UT_LIST} ${_OP_UT_LIST} PARENT_SCOPE)
    set(${OP_UT_DIR_LIST} ${_OP_UT_DIR_LIST} PARENT_SCOPE)
endfunction()

# 私有函数, 外部不可直接调用, 用于执行包含多个算子的 UT 可执行程序
#[[
]]
function(OpsTest_RunLaunch)
    cmake_parse_arguments(
            TMP
            ""
            "TARGET;EXECUTABLE"
            ""
            ""
            ${ARGN}
    )
    if (ENABLE_UT_EXEC)
        set(LD_LIBRARY_PATH_ "LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH}:${ASCEND_CANN_PACKAGE_PATH}/compiler/lib64:${ASCEND_CANN_PACKAGE_PATH}/toolkit/tools/simulator/Ascend910B1/lib")
        if (ENABLE_ASAN OR ENABLE_UBSAN)
            if (ENABLE_ASAN)
                # 谨慎修改 ASAN_OPTIONS_ 取值, 当前出现告警会使 UT 失败.
                # halt_on_error=1, 出现告警时停止运行进而触发构建失败, 避免主进程或 CPU孪生调试 fork 出的子进程出现错误无法发现的情况
                # detect_stack_use_after_return=1, 栈空间返回后使用检测
                # check_initialization_order, 尝试捕获初始化顺序问题
                # strict_init_order, 动态初始化器永远不能访问来自其他模块的全局变量, 及时或者已经初始化
                # strict_string_checks, 检查字符串参数是否正确以 null 终止
                # detect_leaks=1, 内存泄漏检测
                set(ASAN_OPTIONS_ "ASAN_OPTIONS=halt_on_error=1,detect_stack_use_after_return=1,check_initialization_order=1,strict_init_order=1,strict_string_checks=1,detect_leaks=1")
                message(STATUS "${TMP_EXECUTABLE}: ${ASAN_OPTIONS_}")
            endif ()
            if (ENABLE_UBSAN)
                # 谨慎修改 UBSAN_OPTIONS_ 取值, 当前出现告警会使 UT 失败.
                # halt_on_error=1, 出现告警时停止运行进而触发构建失败, 避免主进程或 CPU孪生调试 fork 出的子进程出现错误无法发现的情况
                # print_stacktrace=1, 出错时打印调用栈
                set(UBSAN_OPTIONS_ "UBSAN_OPTIONS=halt_on_error=1,print_stacktrace=1")
                message(STATUS "${TMP_EXECUTABLE}: ${UBSAN_OPTIONS_}")
            endif ()
            # 用例执行
            # 采用链接 ASAN/MSAN 动态库方式执行, 受 SAN 机制限制, 若采用静态库, 则会导致如 OpTiling/OpProto 内 SAN 功能失效.
            if (NOT "${SAN_LD_PRELOAD}" STREQUAL "")
                message(STATUS "${SAN_LD_PRELOAD}")
            endif ()
            if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
                add_custom_command(
                        TARGET ${TMP_TARGET} POST_BUILD
                        COMMAND export ${LD_LIBRARY_PATH_} && ulimit -s 32768 && ${ASAN_OPTIONS_} ${UBSAN_OPTIONS_} ./${TMP_EXECUTABLE}
                        COMMENT "Run ${TMP_EXECUTABLE} with Sanitizer, ASAN(${ENABLE_ASAN}), UBSAN(${ENABLE_UBSAN})"
                )
            else ()
                add_custom_command(
                        TARGET ${TMP_TARGET} POST_BUILD
                        COMMAND export ${LD_LIBRARY_PATH_} && export ${SAN_LD_PRELOAD} && ulimit -s 32768 && ${ASAN_OPTIONS_} ${UBSAN_OPTIONS_} ./${TMP_EXECUTABLE}
                        COMMENT "Run ${TMP_EXECUTABLE} with Sanitizer, ASAN(${ENABLE_ASAN}), UBSAN(${ENABLE_UBSAN})"
                )
            endif ()
        else()
            # 用例执行
            add_custom_command(
                    TARGET ${TMP_TARGET} POST_BUILD
                    COMMAND export ${LD_LIBRARY_PATH_} && ./${TMP_EXECUTABLE}
                    COMMENT "Run ${TMP_EXECUTABLE}"
            )
        endif ()
    endif ()
endfunction()

# 生成覆盖率
#[[
调用参数:
  one_value_keywords:
      TARGET : 必选参数, 用于指定覆盖率生成所依赖的目标(POST_BUILD)
  multi_value_keywords:
      FILTER_DIRECTORIES : 可选参数, 用于指定覆盖率结果过滤目录
]]
function(OpsTest_GenerateCoverage)
    cmake_parse_arguments(
            TMP
            ""
            "TARGET"
            "FILTER_DIRECTORIES"
            ""
            ${ARGN}
    )
    if (ENABLE_UT_EXEC)
        if (ENABLE_GCOV)
            find_program(LCOV lcov REQUIRED)
            get_filename_component(GEN_COV_PY ${OPS_ADV_CMAKE_DIR}/scripts/utest/gen_coverage.py REALPATH)
            get_filename_component(ASCEND_CANN_PACKAGE_PATH_PARENT "${ASCEND_CANN_PACKAGE_PATH}/../" REALPATH)
            get_filename_component(GEM_COV_DATA_DIR "${CMAKE_CURRENT_BINARY_DIR}" REALPATH)
            # 获取 gcc 默认头文件搜索路径
            execute_process(
                    COMMAND ${CMAKE_C_COMPILER} --print-sysroot-headers-suffix
                    RESULT_VARIABLE _RST
                    OUTPUT_VARIABLE _SUFFIX
                    ERROR_QUIET
            )
            if (_RST)
                get_filename_component(SYS_ROOT "/usr/include" REALPATH)
            else ()
                get_filename_component(SYS_ROOT "${_SUFFIX}/usr/include" REALPATH)
            endif ()
            list(REMOVE_DUPLICATES _FilterCmds)
            add_custom_command(
                    TARGET ${TMP_TARGET} POST_BUILD
                    COMMAND ${HI_PYTHON} ${GEN_COV_PY}
                        "-s=${OPS_ADV_DIR}"
                        "-c=${GEM_COV_DATA_DIR}"
                        "-f=/tmp/*"
                        "-f=/usr/include/*"
                        "-f=${ASCEND_CANN_PACKAGE_PATH_PARENT}/*"
                        "-y=${OPS_ADV_DIR}/tests/test_config.yaml"
                    COMMENT "Generate coverage for ${TMP_TARGET}"
            )
        endif ()
    endif ()
endfunction()

# 生成包含多个算子的 UT 可执行程序
#[[
]]
function(OpsTest_AddLaunch)
    add_custom_target(ops_test_utest)

    if (NOT _OpsTestUt_UTestCaseLibraries AND NOT _OpsTestUt_UTestAclnnCaseLibraries)
        # 当 _OpsTestUt_UTestCaseLibraries 与 _OpsTestUt_UTestAclnnCaseLibraries 都为空时
        # 说明区分领域的 TESTS_UT_OPS_TEST 选项被错误设置, 需排查对应触发 ut 的 python 脚本逻辑及 build.sh.
        message(STATUS "_OpsTestUt_UTestCaseLibraries      = ${_OpsTestUt_UTestCaseLibraries}.")
        message(STATUS "_OpsTestUt_UTestAclnnCaseLibraries = ${_OpsTestUt_UTestAclnnCaseLibraries}")
        message(STATUS "_OpsTestUt_UTestCaseLibraries and _OpsTestUt_UTestAclnnCaseLibraries both empty.")
        return ()
    endif ()

    set(_UTest_OpApiLibrary)
    set(_UTest_OpProtoLibrary)
    set(_UTest_OpTilingLibrary)
    set(_UTest_Main)                     # UTest 用例可执行程序(常规)
    set(_UTest_Main_Aclnn)               # UTest 用例可执行程序(Aclnn)
    set(_Utest_ExecutableList)               # 为提高PR场景执行效率, PR场景各可执行文件并行执行, 否则串行执行

    # OpApi 动态库(可选)
    OpsTest_AddOpApiShared()

    # OpProto 动态库(可选)
    OpsTest_AddOpProtoShared()

    # OpTiling 动态库(必选)
    OpsTest_AddOpTilingShared()

    # UTest 用例可执行程序(常规)
    if (_OpsTestUt_UTestCaseLibraries)
        # 支持按算子分离可执行文件, 便于彼此冲突的 Kernel 侧编译宏设置
        foreach (_OpsTestUt_UTestCaseLibrary ${_OpsTestUt_UTestCaseLibraries})
            # 获取算子工程名
            string(REPLACE "_UTest_Case" "" _OpBrief ${_OpsTestUt_UTestCaseLibrary})
            string(REPLACE "UTest_" "" _OpBrief ${_OpBrief})

            set(_UTest_Main ${UTest_NamePrefix}_Main_Normal_${_OpBrief})
            add_executable(${_UTest_Main})
            target_sources(${_UTest_Main} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/ut/framework_special/main.cpp)
            target_compile_options(${_UTest_Main} PRIVATE -fPIC)
            target_link_libraries(${_UTest_Main}
                    PRIVATE
                        -Wl,--no-as-needed
                        -Wl,--whole-archive
                        $<BUILD_INTERFACE:intf_pub_utest>
                        gtest
                        $<BUILD_INTERFACE:_OpsTestUt_GTest_Wno>
                        ${_OpsTestUt_UTestCaseLibrary}
                        ${UTest_NamePrefix}_Stubs
                        ${UTest_NamePrefix}_Utils
                        ${_UTest_OpApiLibrary}   # 若算子在 UTest_{Op}_Common 内实现 Aclnn 相关执行逻辑, 则需要连接
                        -Wl,--as-needed
                        -Wl,--no-whole-archive
                        c_sec
            )
            add_dependencies(${_UTest_Main} ${_UTest_OpTilingLibrary})
            add_dependencies(ops_test_utest ${_UTest_Main})
            if (TESTS_UT_OPS_TEST_CI_PR)
                OpsTest_RunLaunch(TARGET ${_UTest_Main} EXECUTABLE ${_UTest_Main})
            else ()
                list(APPEND _Utest_ExecutableList ${_UTest_Main})
            endif ()
        endforeach ()
    endif ()

    # UTest 用例可执行程序(Aclnn)
    set(_param
            "--cann_path=${ASCEND_CANN_PACKAGE_PATH}"
            "--cann_package_name=opp"
            "get_package_version"
    )

    if (_OpsTestUt_UTestAclnnCaseLibraries)
        # 支持按算子分离可执行文件, 便于彼此冲突的 Kernel 侧编译宏设置
        foreach (_OpsTestUt_UTestAclnnCaseLibrary ${_OpsTestUt_UTestAclnnCaseLibraries})
            # 获取算子工程名
            string(REPLACE "_UTest_Case_Aclnn" "" _OpBrief ${_OpsTestUt_UTestAclnnCaseLibrary})
            string(REPLACE "UTest_" "" _OpBrief ${_OpBrief})
            set(_UTest_Main_Aclnn ${UTest_NamePrefix}_Main_Aclnn_${_OpBrief})
            add_executable(${_UTest_Main_Aclnn})
            target_sources(${_UTest_Main_Aclnn} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/ut/framework_special/main_aclnn.cpp)
            target_compile_options(${_UTest_Main_Aclnn} PRIVATE -fPIC)
            target_link_libraries(${_UTest_Main_Aclnn}
                    PRIVATE
                        -Wl,--no-as-needed
                        -Wl,--whole-archive
                        $<BUILD_INTERFACE:intf_pub_utest>
                        gtest
                        $<BUILD_INTERFACE:_OpsTestUt_GTest_Wno>
                        $<$<BOOL:${BUILD_OPEN_PROJECT}>:$<BUILD_INTERFACE:alog_headers>>
                        $<$<BOOL:${BUILD_OPEN_PROJECT}>:$<BUILD_INTERFACE:dlog_headers>>
                        ${_OpsTestUt_UTestAclnnCaseLibrary}
                        ${UTest_NamePrefix}_Stubs
                        ${UTest_NamePrefix}_Utils
                        ${_UTest_OpApiLibrary}
                        -Wl,--as-needed
                        -Wl,--no-whole-archive
                        c_sec
            )
            add_dependencies(${_UTest_Main_Aclnn} ${_UTest_OpProtoLibrary} ${_UTest_OpTilingLibrary})
            add_dependencies(ops_test_utest ${_UTest_Main_Aclnn})
            if (TESTS_UT_OPS_TEST_CI_PR)
                OpsTest_RunLaunch(TARGET ${_UTest_Main_Aclnn} EXECUTABLE ${_UTest_Main_Aclnn})
            else ()
                list(APPEND _Utest_ExecutableList ${_UTest_Main_Aclnn})
            endif ()
        endforeach ()
    endif ()

    # 为提高PR场景执行效率, PR场景各可执行文件并行执行, 否则串行执行
    if (NOT TESTS_UT_OPS_TEST_CI_PR)
        foreach (_Exe ${_Utest_ExecutableList})
            OpsTest_RunLaunch(TARGET ops_test_utest EXECUTABLE ${_Exe})
        endforeach ()
    endif ()

    # 生成覆盖率
    set(_FilterDirectories)
    list(REMOVE_DUPLICATES _FilterDirectories)
    OpsTest_GenerateCoverage(TARGET ops_test_utest FILTER_DIRECTORIES ${_FilterDirectories})
endfunction()

# 添加算子UT路径
#[[
]]
function(OpsTestUt_AddSubdirectory)
    cmake_parse_arguments(
            TMP
            ""
            ""
            "CLANG_NOT_SUPPORTED_OP_LIST"
            ""
            ${ARGN}
    )

    if ("${TESTS_UT_OPS_TEST}" STREQUAL "")
        return()
    elseif ("ALL" IN_LIST TESTS_UT_OPS_TEST OR "all" IN_LIST TESTS_UT_OPS_TEST)
        file(GLOB sub_dirs ${CMAKE_CURRENT_SOURCE_DIR}/*)
        foreach (_dir ${sub_dirs})
            if (IS_DIRECTORY ${_dir})
                get_filename_component(_op_type ${_dir} NAME)
                if ("${CMAKE_C_COMPILER_ID}" STREQUAL "Clang" AND "${_op_type}" IN_LIST TMP_CLANG_NOT_SUPPORTED_OP_LIST)
                    message(STATUS "The UTest of ${_op_type} don't supported clang compiler yet.")
                    continue()
                endif ()
                add_subdirectory(${_dir})
            endif ()
        endforeach ()
    else ()
        set(_added_op_type_list)
        foreach (_op_type ${TESTS_UT_OPS_TEST})
            if (DEFINED ${_op_type}_alias)
                set(_op_type ${${_op_type}_alias})
            endif ()
            if (NOT "${_op_type}" IN_LIST _added_op_type_list)
                set(_dir ${CMAKE_CURRENT_SOURCE_DIR}/${_op_type})
                if (IS_DIRECTORY ${_dir})
                    if ("${CMAKE_C_COMPILER_ID}" STREQUAL "Clang" AND "${_op_type}" IN_LIST TMP_CLANG_NOT_SUPPORTED_OP_LIST)
                        message(STATUS "The UTest of ${_op_type} don't supported clang compiler yet.")
                        continue()
                    endif ()
                    add_subdirectory(${_dir})
                    list(APPEND _added_op_type_list ${_op_type})
                endif ()
            endif ()
        endforeach ()
    endif ()
endfunction()
