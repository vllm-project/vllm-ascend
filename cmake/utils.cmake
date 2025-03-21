#
# Attempt to find the python package that uses the same python executable as
# `EXECUTABLE` and is one of the `SUPPORTED_VERSIONS`.
#
macro (find_python_from_executable EXECUTABLE SUPPORTED_VERSIONS)
  file(REAL_PATH ${EXECUTABLE} EXECUTABLE)
  set(Python_EXECUTABLE ${EXECUTABLE})
  find_package(Python COMPONENTS Interpreter Development.Module Development.SABIModule)
  if (NOT Python_FOUND)
    message(FATAL_ERROR "Unable to find python matching: ${EXECUTABLE}.")
  endif()
  set(_VER "${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}")
  set(_SUPPORTED_VERSIONS_LIST ${SUPPORTED_VERSIONS} ${ARGN})
  if (NOT _VER IN_LIST _SUPPORTED_VERSIONS_LIST)
    message(FATAL_ERROR
      "Python version (${_VER}) is not one of the supported versions: "
      "${_SUPPORTED_VERSIONS_LIST}.")
  endif()
  message(STATUS "Found python matching: ${EXECUTABLE}.")
endmacro()

#
# Run `EXPR` in python.  The standard output of python is stored in `OUT` and
# has trailing whitespace stripped.  If an error is encountered when running
# python, a fatal message `ERR_MSG` is issued.
#
function (run_python OUT EXPR ERR_MSG)
  execute_process(
    COMMAND
    "${PYTHON_EXECUTABLE}" "-c" "${EXPR}"
    OUTPUT_VARIABLE PYTHON_OUT
    RESULT_VARIABLE PYTHON_ERROR_CODE
    ERROR_VARIABLE PYTHON_STDERR
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  if(NOT PYTHON_ERROR_CODE EQUAL 0)
    message(FATAL_ERROR "${ERR_MSG}: ${PYTHON_STDERR}")
  endif()
  set(${OUT} ${PYTHON_OUT} PARENT_SCOPE)
endfunction()

# Run `EXPR` in python after importing `PKG`. Use the result of this to extend
# `CMAKE_PREFIX_PATH` so the torch cmake configuration can be imported.
macro (append_cmake_prefix_path PKG EXPR)
  run_python(_PREFIX_PATH
    "import ${PKG}; print(${EXPR})" "Failed to locate ${PKG} path")
  list(APPEND CMAKE_PREFIX_PATH ${_PREFIX_PATH})
endmacro()


# This cmake function is adapted from vllm /Users/ganyi/workspace/vllm-ascend/cmake/utils.cmake
# Define a target named `GPU_MOD_NAME` for a single extension. The
# arguments are:
#
# DESTINATION <dest>         - Module destination directory.
# LANGUAGE <lang>            - The GPU language for this module, e.g CUDA, HIP,
#                              etc.
# SOURCES <sources>          - List of source files relative to CMakeLists.txt
#                              directory.
#
# Optional arguments:
#
# ARCHITECTURES <arches>     - A list of target GPU architectures in cmake
#                              format.
#                              Refer `CMAKE_CUDA_ARCHITECTURES` documentation
#                              and `CMAKE_HIP_ARCHITECTURES` for more info.
#                              ARCHITECTURES will use cmake's defaults if
#                              not provided.
# COMPILE_FLAGS <flags>      - Extra compiler flags passed to NVCC/hip.
# INCLUDE_DIRECTORIES <dirs> - Extra include directories.
# LIBRARIES <libraries>      - Extra link libraries.
# WITH_SOABI                 - Generate library with python SOABI suffix name.
# USE_SABI <version>         - Use python stable api <version>
#
# Note: optimization level/debug info is set via cmake build type.
#
function (define_gpu_extension_target GPU_MOD_NAME)
  cmake_parse_arguments(PARSE_ARGV 1
    GPU
    "WITH_SOABI"
    "DESTINATION;LANGUAGE;USE_SABI"
    "SOURCES;ARCHITECTURES;COMPILE_FLAGS;INCLUDE_DIRECTORIES;LIBRARIES")

  # Add hipify preprocessing step when building with HIP/ROCm.
  if (GPU_LANGUAGE STREQUAL "HIP")
    hipify_sources_target(GPU_SOURCES ${GPU_MOD_NAME} "${GPU_SOURCES}")
  endif()

  if (GPU_WITH_SOABI)
    set(GPU_WITH_SOABI WITH_SOABI)
  else()
    set(GPU_WITH_SOABI)
  endif()

  if (GPU_USE_SABI)
    Python_add_library(${GPU_MOD_NAME} MODULE USE_SABI ${GPU_USE_SABI} ${GPU_WITH_SOABI} "${GPU_SOURCES}")
  else()
    Python_add_library(${GPU_MOD_NAME} MODULE ${GPU_WITH_SOABI} "${GPU_SOURCES}")
  endif()

  if (GPU_LANGUAGE STREQUAL "HIP")
    # Make this target dependent on the hipify preprocessor step.
    add_dependencies(${GPU_MOD_NAME} hipify${GPU_MOD_NAME})
  endif()

  if (GPU_ARCHITECTURES)
    set_target_properties(${GPU_MOD_NAME} PROPERTIES
      ${GPU_LANGUAGE}_ARCHITECTURES "${GPU_ARCHITECTURES}")
  endif()

  set_property(TARGET ${GPU_MOD_NAME} PROPERTY CXX_STANDARD 17)

  target_compile_options(${GPU_MOD_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:${GPU_LANGUAGE}>:${GPU_COMPILE_FLAGS}>)

  target_compile_definitions(${GPU_MOD_NAME} PRIVATE
    "-DTORCH_EXTENSION_NAME=${GPU_MOD_NAME}")

  target_include_directories(${GPU_MOD_NAME} PRIVATE csrc
    ${GPU_INCLUDE_DIRECTORIES})

  target_link_libraries(${GPU_MOD_NAME} PRIVATE torch ${GPU_LIBRARIES})

  # Don't use `TORCH_LIBRARIES` for CUDA since it pulls in a bunch of
  # dependencies that are not necessary and may not be installed.
  if (GPU_LANGUAGE STREQUAL "CUDA")
    target_link_libraries(${GPU_MOD_NAME} PRIVATE CUDA::cudart CUDA::cuda_driver)
  else()
    target_link_libraries(${GPU_MOD_NAME} PRIVATE ${TORCH_LIBRARIES})
  endif()

  install(TARGETS ${GPU_MOD_NAME} LIBRARY DESTINATION ${GPU_DESTINATION} COMPONENT ${GPU_MOD_NAME})
endfunction()
