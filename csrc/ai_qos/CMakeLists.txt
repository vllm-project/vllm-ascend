cmake_minimum_required(VERSION 3.16)
project(ai_qos LANGUAGES CXX)
find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(
    COMMAND python3 -c "import pybind11, os; print(os.path.dirname(os.path.dirname(pybind11.get_include())))"
    OUTPUT_VARIABLE PYBIND11_ROOT
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_ROOT}")
find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(ai_qos MODULE ai_qos.cpp)
target_compile_features(ai_qos PRIVATE cxx_std_11)
# ---- DSMI lookup (driver SDK, not CANN-only) ----
set(_dsmi_inc_paths
    /usr/local/Ascend/driver/include
    /usr/local/Ascend/ascend-toolkit/latest/driver/include
    /usr/local/Ascend/driver/kernel/inc/driver
)
set(_dsmi_lib_paths
    /usr/local/Ascend/driver/lib64/driver
    /usr/local/Ascend/ascend-toolkit/latest/driver/lib64
)
if(DEFINED ENV{ASCEND_HOME_PATH} AND NOT "$ENV{ASCEND_HOME_PATH}" STREQUAL "")
    get_filename_component(_ascend_parent "$ENV{ASCEND_HOME_PATH}" DIRECTORY)
    list(INSERT _dsmi_inc_paths 0
        "${_ascend_parent}/driver/include"
        "$ENV{ASCEND_HOME_PATH}/driver/include"
    )
    list(INSERT _dsmi_lib_paths 0
        "${_ascend_parent}/driver/lib64/driver"
        "$ENV{ASCEND_HOME_PATH}/driver/lib64"
    )
endif()
find_path(DSMI_INCLUDE_DIR
    NAMES dsmi_common_interface.h
    PATHS ${_dsmi_inc_paths}
    NO_DEFAULT_PATH
)
find_library(DSMI_LIBRARY
    NAMES drvdsmi_host
    PATHS ${_dsmi_lib_paths}
    NO_DEFAULT_PATH
)
if(NOT DSMI_INCLUDE_DIR)
    message(FATAL_ERROR
        "ai_qos: dsmi_common_interface.h not found. "
        "Pass -DDSMI_INCLUDE_DIR=/path/to/include"
    )
endif()
if(NOT DSMI_LIBRARY)
    message(FATAL_ERROR
        "ai_qos: drvdsmi_host not found. "
        "Pass -DDSMI_LIBRARY=/path/to/libdrvdsmi_host.so"
    )
endif()
target_include_directories(ai_qos PRIVATE
    ${pybind11_INCLUDE_DIRS}
    ${DSMI_INCLUDE_DIR}
)
target_link_libraries(ai_qos PRIVATE ${DSMI_LIBRARY})
install(TARGETS ai_qos LIBRARY DESTINATION .)