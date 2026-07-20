cmake_minimum_required(VERSION 3.16)
project(bifrost_ascend_C LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# User can pass -DASCEND_HOME_PATH=... at configure time.
if(NOT DEFINED ASCEND_HOME_PATH)
  set(ASCEND_HOME_PATH "/usr/local/Ascend/ascend-toolkit/latest" CACHE PATH "Ascend toolkit root")
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

add_library(bifrost_ascend_C MODULE
  ${CMAKE_CURRENT_SOURCE_DIR}/camem_allocator.cpp
)

target_include_directories(bifrost_ascend_C
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${Python3_INCLUDE_DIRS}
  ${ASCEND_HOME_PATH}/include
  ${ASCEND_HOME_PATH}/aarch64-linux/include/experiment/platform
  ${ASCEND_HOME_PATH}/x86_64-linux/include/experiment/platform
)

target_link_directories(bifrost_ascend_C
  PRIVATE
  ${ASCEND_HOME_PATH}/lib64
)

target_link_libraries(bifrost_ascend_C
  PRIVATE
  Python3::Python
  ascendcl
  pthread
  dl
  rt
)

# Emit exact filename requested by user.
set_target_properties(bifrost_ascend_C PROPERTIES
  PREFIX ""
  OUTPUT_NAME "bifrost_ascend_C"
  SUFFIX ".cpython-311-aarch64-linux-gnu.so"
)

# Keep local run/import simple.
set_target_properties(bifrost_ascend_C PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

install(TARGETS bifrost_ascend_C DESTINATION .)
