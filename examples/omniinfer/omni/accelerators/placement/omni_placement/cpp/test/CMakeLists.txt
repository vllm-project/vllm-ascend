cmake_minimum_required(VERSION 3.10)
project(ExpertActivationProject)

# --- 1. General Settings and Dependencies ---

# Find Google Test
find_package(GTest REQUIRED)

if(NOT GTest_FOUND)
  message(FATAL_ERROR "Google Test not found. Please install gtest and gmock.")
endif()

# Set C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler Flags
add_compile_options(-Wall -g)

# Find Python (required by pybind11)
set(Python3_EXECUTABLE "/usr/bin/python")
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

if(NOT Python3_FOUND)
  message(FATAL_ERROR "Python3 not found")
endif()

# Define a CMake option to control hardware test builds
option(enable_tests_with_npu
  "Build tests that require NPU hardware and Ascend libraries" OFF)

# --- 2. Find Include Paths and Libraries (for all targets) ---

# Find pybind11
set(PYBIND11_INCLUDE_DIRS_INTERNAL "")
find_package(pybind11 QUIET)

if(pybind11_FOUND)
  set(PYBIND11_INCLUDE_DIRS_INTERNAL ${pybind11_INCLUDE_DIRS})
  message(STATUS "Found pybind11 via find_package: ${pybind11_INCLUDE_DIRS}")
else()
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -c
    "import site; print(site.getsitepackages()[0])"
    OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  set(MANUAL_PYBIND11_PATH "${PYTHON_SITE_PACKAGES}/pybind11/include")

  if(EXISTS "${MANUAL_PYBIND11_PATH}/pybind11/pybind11.h")
    set(PYBIND11_INCLUDE_DIRS_INTERNAL ${MANUAL_PYBIND11_PATH})
    message(
      STATUS
        "Found pybind11 in site-packages: ${PYBIND11_INCLUDE_DIRS_INTERNAL}")
  else()
    message(FATAL_ERROR "Could not find pybind11 include directory.")
  endif()
endif()

# Find Ascend toolkit path.
set(ASCEND_TOOLKIT_PATH $ENV{ASCEND_TOOLKIT_HOME})

if(NOT ASCEND_TOOLKIT_PATH)
  set(ASCEND_POSSIBLE_PATHS
      "/usr/local/Ascend/latest/aarch64-linux"
      "/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux"
      "/usr/local/Ascend/ascend-toolkit/latest/arm64-linux")

  foreach(PATH ${ASCEND_POSSIBLE_PATHS})
    if(EXISTS "${PATH}/include/acl/acl.h")
      set(ASCEND_TOOLKIT_PATH ${PATH})
      break()
    endif()
  endforeach()
endif()

if(NOT EXISTS "${ASCEND_TOOLKIT_PATH}/include")
  message(
    WARNING
      "Ascend include directory not found. Hardware tests will likely fail to build."
  )
endif()

# Find Ascend libraries for ALL targets that need them.
find_library(ASCEND_ASCENDCL_LIB ascendcl HINTS ${ASCEND_TOOLKIT_PATH}/lib64)
find_library(ASCEND_HCCL_LIB hccl HINTS ${ASCEND_TOOLKIT_PATH}/lib64)

if(NOT ASCEND_ASCENDCL_LIB OR NOT ASCEND_HCCL_LIB)
  message(
    WARNING
      "Could not find Ascend libraries (ascendcl or hccl) in ${ASCEND_TOOLKIT_PATH}/lib64. "
      "Any test using Ascend functions will fail to link.")
endif()

# --- 3. Define Unit Test Target (No Hardware Required) ---
set(UNIT_TEST_SOURCES
    test_distribution.cpp
    test_expert_load_balancer.cpp
    test_dynamic_eplb_greedy.cpp
    ../distribution.cpp
    ../placement_manager.cpp
    ../placement_mapping.cpp
    ../dynamic_eplb_greedy.cpp
    ../placement_optimizer.cpp
    ../expert_activation.cpp
    ../tensor.cpp
    ../moe_weights.cpp
    ../expert_load_balancer.cpp)

add_executable(unit_tests ${UNIT_TEST_SOURCES})

target_include_directories(
  unit_tests
  PRIVATE ${Python3_INCLUDE_DIRS} ${PYBIND11_INCLUDE_DIRS_INTERNAL}
  ${ASCEND_TOOLKIT_PATH}/include ${CMAKE_SOURCE_DIR}/../include
  ${CMAKE_SOURCE_DIR}/..)

# Link libraries for unit tests. Since they call real Ascend functions, we must
# link them.
target_link_libraries(
  unit_tests
  PRIVATE GTest::gtest
  GTest::gtest_main
  GTest::gmock
  GTest::gmock_main
  rt
  ${Python3_LIBRARIES}
  ${ASCEND_ASCENDCL_LIB} # <-- ADDED
  ${ASCEND_HCCL_LIB} # <-- ADDED
)

message(
  STATUS "Configured 'unit_tests' target. Run with 'ctest -R unit_tests'.")

# --- 4. Define NPU-dependent Unit Test Target (Requires NPU) ---
if(enable_tests_with_npu)
  message(
    STATUS "enable_tests_with_npu is ON. Configuring unit tests with NPU.")

  set(UNIT_TESTS_WITH_NPU_SOURCES
      test_main.cpp
      test_tensor.cpp
      test_moe_weights.cpp
      test_placement_mapping.cpp
      test_placement_optimizer.cpp
      test_expert_activation.cpp
      test_placement_manager.cpp
      ../distribution.cpp
      ../placement_manager.cpp
      ../placement_mapping.cpp
      ../dynamic_eplb_greedy.cpp
      ../placement_optimizer.cpp
      ../expert_activation.cpp
      ../tensor.cpp
      ../moe_weights.cpp
      ../expert_load_balancer.cpp)

  add_executable(unit_tests_with_npu ${UNIT_TESTS_WITH_NPU_SOURCES})

  target_include_directories(
    unit_tests_with_npu
    PRIVATE ${Python3_INCLUDE_DIRS} ${PYBIND11_INCLUDE_DIRS_INTERNAL}
    ${ASCEND_TOOLKIT_PATH}/include ${CMAKE_SOURCE_DIR}/../include
    ${CMAKE_SOURCE_DIR}/..)

  target_link_libraries(
    unit_tests_with_npu
    PRIVATE GTest::gtest GTest::gmock rt ${Python3_LIBRARIES}
    ${ASCEND_ASCENDCL_LIB} ${ASCEND_HCCL_LIB})

  message(
    STATUS
      "Configured 'unit_tests_with_npu' target. Run with 'ctest -R unit_tests_with_npu'."
  )

else()
  message(STATUS "enable_tests_with_npu is OFF. Skipping unit tests with NPU.")
endif()

# --- 5. Debug Information Output ---
message(STATUS "Python3 Executable: ${Python3_EXECUTABLE}")
message(STATUS "Python3 Include Dirs: ${Python3_INCLUDE_DIRS}")
message(STATUS "Python3 Libraries: ${Python3_LIBRARIES}")
message(
  STATUS "Pybind11 Include Dirs (Internal): ${PYBIND11_INCLUDE_DIRS_INTERNAL}")
message(STATUS "Ascend Toolkit Path: ${ASCEND_TOOLKIT_PATH}")
message(STATUS "Ascend ascendcl Lib: ${ASCEND_ASCENDCL_LIB}")
message(STATUS "Ascend hccl Lib: ${ASCEND_HCCL_LIB}")
