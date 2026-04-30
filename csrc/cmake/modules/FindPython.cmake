# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

include_guard(GLOBAL)

find_package(Python3 3.7 COMPONENTS Interpreter)
set(HI_PYTHON ${Python3_EXECUTABLE})
message(STATUS "HI_PYTHON = ${Python3_EXECUTABLE}")

macro(get_python)
    execute_process(COMMAND ${HI_PYTHON} -c "import sys; print(sys.executable)"
            RESULT_VARIABLE result
            OUTPUT_VARIABLE _TEMP_PYTHON_PATH
            OUTPUT_STRIP_TRAILING_WHITESPACE)
    if (result)
        set(FATAL_ERROR "Please install python3 software first.")
    endif()

    execute_process(COMMAND ${HI_PYTHON} -c "import sysconfig; print(sysconfig.get_config_var('INCLUDEPY'))"
            RESULT_VARIABLE result
            OUTPUT_VARIABLE HI_PYTHON_INC_TEMP
            OUTPUT_STRIP_TRAILING_WHITESPACE)
    if (result)
        set(HI_PYTHON_INC_TEMP "unknown")
    endif()

    execute_process(COMMAND ${HI_PYTHON} -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"
            RESULT_VARIABLE result
            OUTPUT_VARIABLE HI_PYTHON_LIBDIR
            OUTPUT_STRIP_TRAILING_WHITESPACE)
    if (result)
        set(HI_PYTHON_LIBDIR ".")
    endif()

    execute_process(COMMAND ${HI_PYTHON} -c "import sysconfig; print(sysconfig.get_config_var('BLDLIBRARY'))"
            RESULT_VARIABLE result
            OUTPUT_VARIABLE HI_PYTHON_LIB_TEMP
            OUTPUT_STRIP_TRAILING_WHITESPACE)
    if (result)
        set(HI_PYTHON_LIB_TEMP "")
    endif()
    if(${HI_PYTHON_LIB_TEMP} MATCHES "^lib.*\\.a$")
        set(HI_PYTHON_LIB_TEMP "${HI_PYTHON_LIBDIR}/${HI_PYTHON_LIB_TEMP}")
    else()
        STRING(REPLACE "-L." "-L${HI_PYTHON_LIBDIR}" HI_PYTHON_LIB_TEMP ${HI_PYTHON_LIB_TEMP})
    endif()
endmacro(get_python)
get_python()
message(STATUS "HI_PYTHON_INC_TEMP = ${HI_PYTHON_INC_TEMP}")
message(STATUS "HI_PYTHON_LIB_TEMP = ${HI_PYTHON_LIB_TEMP}")

set(HI_PYTHON_INC ${HI_PYTHON_INC_TEMP} CACHE STRING "python include path" FORCE)
set(HI_PYTHON_LIB ${HI_PYTHON_LIB_TEMP} CACHE STRING "python library libpython3.7m.a" FORCE)
