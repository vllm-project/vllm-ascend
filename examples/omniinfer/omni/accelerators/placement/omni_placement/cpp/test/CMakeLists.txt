cmake_minimum_required(VERSION 3.10)
project(ExpertActivationProject)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 编译器标志
add_compile_options(-Wall -g)

# 查找当前 Python 环境，明确指定路径
set(Python3_ROOT_DIR "/home/ma-user/anaconda3/envs/PyTorch-2.1.0")
set(Python3_EXECUTABLE "${Python3_ROOT_DIR}/bin/python")
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
if (NOT Python3_FOUND)
    message(FATAL_ERROR "Python3 not found")
endif()

# 动态获取 Python 相关路径和库
set(PYTHON_INCLUDE_DIRS ${Python3_INCLUDE_DIRS})
set(PYTHON_LIBRARIES ${Python3_LIBRARIES})
set(PYTHON_LIBRARY_DIRS ${Python3_LIBRARY_DIRS})

# 包含目录
include_directories(
    /usr/local/include/gtest
    ${PYTHON_INCLUDE_DIRS}  # 动态 Python 头文件路径
    ${CMAKE_SOURCE_DIR}/../include  # 项目相对路径
    ${CMAKE_SOURCE_DIR}/..          # 项目根目录
)

# 支持 Ascend 的动态路径（通过环境变量或默认值）
set(ASCEND_TOOLKIT_PATH $ENV{ASCEND_TOOLKIT_PATH})
if (NOT ASCEND_TOOLKIT_PATH)
    set(ASCEND_TOOLKIT_PATH "/usr/local/Ascend/latest/aarch64-linux")
    # set(ASCEND_TOOLKIT_PATH "/usr/local/Ascend/ascend-toolkit/latest/arm64-linux")
endif()
include_directories(${ASCEND_TOOLKIT_PATH}/include)
link_directories(${ASCEND_TOOLKIT_PATH}/lib64)

# 查找 pybind11
find_package(pybind11 QUIET)
if (pybind11_FOUND)
    include_directories(${pybind11_INCLUDE_DIRS})
    message(STATUS "Found pybind11 via find_package: ${pybind11_INCLUDE_DIRS}")
else()
    # 尝试从 Python site-packages 获取 pybind11 路径
    if (NOT PYBIND11_INCLUDE_DIR)
        # 使用 Python3_EXECUTABLE 获取 site-packages 路径
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -c "import site; print(site.getsitepackages()[0])"
            OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        set(PYBIND11_INCLUDE_DIR "${PYTHON_SITE_PACKAGES}/pybind11/include")
    endif()

    # 检查 pybind11 路径是否存在
    if (EXISTS "${PYBIND11_INCLUDE_DIR}/pybind11/pybind11.h")
        include_directories(${PYBIND11_INCLUDE_DIR})
        message(STATUS "Found pybind11 in site-packages: ${PYBIND11_INCLUDE_DIR}")
    else()
        # 回退到已知的路径（基于 find / 结果）
        set(PYBIND11_FALLBACK_DIR "/home/ma-user/anaconda3/envs/PyTorch-2.1.0/lib/python3.11/site-packages/pybind11/include")
        if (EXISTS "${PYBIND11_FALLBACK_DIR}/pybind11/pybind11.h")
            set(PYBIND11_INCLUDE_DIR ${PYBIND11_FALLBACK_DIR})
            include_directories(${PYBIND11_INCLUDE_DIR})
            message(STATUS "Using fallback pybind11 path: ${PYBIND11_INCLUDE_DIR}")
        else()
            message(FATAL_ERROR "Could not find pybind11 include directory. Please install pybind11 with 'pip install pybind11' or specify -DPYBIND11_INCLUDE_DIR=/path/to/pybind11/include")
        endif()
    endif()
endif()

# 链接目录
link_directories(
    ${PYTHON_LIBRARY_DIRS}  # 动态 Python 库路径
)

# 链接的库
set(LIBS
    gtest
    gtest_main
    gmock
    gmock_main
    pthread
    rt
    ${Python3_LIBRARIES}  # 动态 Python 库
    ascendcl
    hccl
)

# 可执行文件
add_executable(test_placement
    test_dynamic_eplb_greedy.cpp
    test_placement_mapping.cpp
    test_tensor.cpp
    test_expert_mapping_per_redundancy.cpp
    # test_placement_optimizer.cpp
    test_expert_swap_optimizer.cpp
    # test_placement_optimizer_for_swap.cpp
    test_expert_load_balancer.cpp
    ../distribution.cpp
    ../placement_manager.cpp
    ../placement_mapping.cpp
    ../dynamic_eplb_greedy.cpp
    ../placement_optimizer.cpp
    ../expert_activation.cpp
    ../tensor.cpp
    ../moe_weights.cpp
    ../placement_optimizer_for_swap.cpp
    ../expert_load_balancer.cpp
)
target_link_libraries(test_placement ${LIBS})

# 输出调试信息
message(STATUS "Python3 Root Dir: ${Python3_ROOT_DIR}")
message(STATUS "Python3 Executable: ${Python3_EXECUTABLE}")
message(STATUS "Python3 Include Dirs: ${PYTHON_INCLUDE_DIRS}")
message(STATUS "Python3 Libraries: ${PYTHON_LIBRARIES}")
message(STATUS "Pybind11 Include Dir: ${PYBIND11_INCLUDE_DIR}")
message(STATUS "Ascend Toolkit Path: ${ASCEND_TOOLKIT_PATH}")