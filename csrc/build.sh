#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set -e
RELEASE_TARGETS=("ophost" "opapi" "opgraph" "onnxplugin")
UT_TARGETS=()
########################################################################################################################
# 预定义变量
########################################################################################################################

CURRENT_DIR=$(dirname $(readlink -f ${BASH_SOURCE[0]}))
BUILD_DIR=${CURRENT_DIR}/build
OUTPUT_DIR=${CURRENT_DIR}/output
BUILD_OUT_DIR=${CURRENT_DIR}/build_out
USER_ID=$(id -u)
PARENT_JOB="false"
HOST_TILING="false"
CHECK_COMPATIBLE="true"
ASAN="true"
UBSAN="false"
COV="false"
CLANG="false"
VERBOSE="false"
OOM="false"
THREAD_NUM=$(grep -c ^processor /proc/cpuinfo)
ENABLE_VALGRIND=FALSE
ENABLE_CREATE_LIB=FALSE
ENABLE_OPKERNEL=FALSE
ENABLE_BUILD_PKG=FALSE
ENABLE_BUILT_IN=FALSE
ENABLE_BUILT_JIT=FALSE
ENABLE_BUILT_CUSTOM=FALSE
ENABLE_STATIC=FALSE
ENABLE_EXPERIMENTAL=FALSE
ASCEND_SOC_UNITS="ascend910b"
SUPPORT_COMPUTE_UNIT_SHORT=("ascend910b" "ascend910_93" "ascend910_95" "kirinx90")
CMAKE_BUILD_MODE=""
BUILD_TYPE=""
VERSION=""
BUILD_LIBS=()
OP_API_UT=FALSE
OP_HOST_UT=FALSE
OP_GRAPH_UT=FALSE
OP_KERNEL_UT=FALSE
OP_API=FALSE
OP_HOST=FALSE
OP_GRAPH=FALSE
ONNX_PLUGIN=FALSE
OP_KERNEL=FALSE
SOC_ARRAY=()
ENABLE_UT_EXEC=FALSE
ENABLE_GENOP=FALSE
ENABLE_GENOP_AICPU=FALSE
GENOP_TYPE=""
GENOP_NAME=""
PR_CHANGED_FILES=""  # PR场景, 修改文件清单, 可用于标识是否PR场景

if [ "${USER_ID}" != "0" ]; then
    DEFAULT_TOOLKIT_INSTALL_DIR="${HOME}/Ascend/ascend-toolkit/latest"
    DEFAULT_INSTALL_DIR="${HOME}/Ascend/latest"
else
    DEFAULT_TOOLKIT_INSTALL_DIR="/usr/local/Ascend/ascend-toolkit/latest"
    DEFAULT_INSTALL_DIR="/usr/local/Ascend/latest"
fi
CANN_3RD_LIB_PATH="${CURRENT_DIR}/third_party"
CUSTOM_OPTION="-DBUILD_OPEN_PROJECT=ON"

dotted_line="---------------------------------------------------------------------------------------------------------------------"
########################################################################################################################
# 预定义函数
########################################################################################################################

function help_info() {
    local specific_help="$1"

    if [[ -n "$specific_help" ]]; then
        case "$specific_help" in
            package)
                echo "Package Build Options:"
                echo $dotted_line
                echo "    --pkg                  Build run package with kernel bin"
                echo "    --jit                  Build run package without kernel bin"
                echo "    --soc=soc_version      Compile for specified Ascend SoC (comma-separated for multiple)"
                echo "    --vendor_name=name     Specify custom operator package vendor name"
                echo "    --ops=op1,op2,...      Compile specified operators (comma-separated for multiple)"
                echo "    -j[n]                  Compile thread nums, default is 8, eg: -j8"
                echo "    -O[n]                  Compile optimization options, support [O0 O1 O2 O3], eg:-O3"
                echo "    --experimental         Build experimental version"
                echo "    --cann_3rd_lib_path=<PATH>"
                echo "                           Set ascend third_party package install path, default ./third_party"
                echo "    --oom                  Build with oom mode on the kernel side, with options: '-g --cce-enable-oom'"
                echo $dotted_line
                echo "Examples:"
                echo "    bash build.sh --pkg --soc=ascend910b --vendor_name=customize -j16 -O3"
                echo "    bash build.sh --pkg --ops=add,sub"
                echo "    bash build.sh --pkg --experimental --soc=ascend910b"
                echo "    bash build.sh --pkg --experimental --soc=ascend910b --ops=abs --oom"
                return
                ;;
            test)
                echo "Test Options:"
                echo $dotted_line
                echo "    -u|--test              Build and run all unit tests"
                echo "    --noexec               Only compile ut, do not execute"
                echo "    --cov                  Enable code coverage for unit tests"
                echo "    --ops=op1,op2,...      Compile specified operators (comma-separated for multiple)"
                echo "    --disable_asan         Disable ASAN (Address Sanitizer)"
                echo "    --soc=soc_version      Run unit tests for specified Ascend SoC"
                echo "    --valgrind             Run unit tests with valgrind (disables ASAN and noexec)"
                echo "    --ophost_test          Build and run ophost unit tests"
                echo "    --opapi_test           Build and run opapi unit tests"
                echo "    --opgraph_test         Build and run opgraph unit tests"
                echo "    --ophost -u            Same as --ophost_test"
                echo "    --opapi -u             Same as --opapi_test"
                echo "    --opgraph -u           Same as --opgraph_test"
                echo $dotted_line
                echo "Examples:"
                echo "    bash build.sh -u --noexec --cov"
                echo "    bash build.sh --ophost_test --opapi_test --noexec"
                echo "    bash build.sh --ophost --opapi --opgraph -u --cov"
                return
                ;;
            clean)
                echo "Clean Options:"
                echo $dotted_line
                echo "    --make_clean           Clean build artifacts"
                echo $dotted_line
                return
                ;;
            valgrind)
                echo "Valgrind Options:"
                echo $dotted_line
                echo "    --valgrind             Run unit tests with valgrind (disables ASAN and noexec)"
                echo $dotted_line
                return
                ;;
            ophost)
                echo "Ophost Build Options:"
                echo $dotted_line
                echo "    --ophost               Build ophost library"
                echo "    -j[n]                  Compile thread nums, default is 8, eg: -j8"
                echo "    -O[n]                  Compile optimization options, support [O0 O1 O2 O3], eg:-O3"
                echo $dotted_line
                echo "Examples:"
                echo "    bash build.sh --ophost -j16 -O3"
                return
                ;;
            opapi)
                echo "Opapi Build Options:"
                echo $dotted_line
                echo "    --opapi                Build opapi library"
                echo "    -j[n]                  Compile thread nums, default is 8, eg: -j8"
                echo "    -O[n]                  Compile optimization options, support [O0 O1 O2 O3], eg:-O3"
                echo $dotted_line
                echo "Examples:"
                echo "    bash build.sh --opapi -j16 -O3"
                return
                ;;
            opgraph)
                echo "Opgraph Build Options:"
                echo $dotted_line
                echo "    --opgraph              Build opgraph library"
                echo "    -j[n]                  Compile thread nums, default is 8, eg: -j8"
                echo "    -O[n]                  Compile optimization options, support [O0 O1 O2 O3], eg:-O3"
                echo $dotted_line
                echo "Examples:"
                echo "    bash build.sh --opgraph -j16 -O3"
                return
                ;;
            onnxplugin)
                echo "ONNXPlugin Build Options:"
                echo $dotted_line
                echo "    --onnxplugin           Build onnxplugin library"
                echo "    -j[n]                  Compile thread nums, default is 8, eg: -j8"
                echo "    -O[n]                  Compile optimization options, support [O0 O1 O2 O3], eg:-O3"
                echo "    --debug                Build with debug mode"
                echo $dotted_line
                echo "Examples:"
                echo "    bash build.sh --onnxplugin -j16 -O3"
                echo "    bash build.sh --onnxplugin --debug"
                return
                ;;
            opkernel)
                echo "Opkernel Build Options:"
                echo $dotted_line
                echo "    --opkernel             Build binary kernel"
                echo "    --soc=soc_version      Compile for specified Ascend SoC (comma-separated for multiple)"
                echo "    --ops=op1,op2,...      Compile specified operators (comma-separated for multiple)"
                echo "    --oom                  Build with oom mode on the kernel side, with options: '-g --cce-enable-oom'"
                echo $dotted_line
                echo "Examples:"
                echo "    bash build.sh --opkernel --soc=ascend310p --ops=add,sub"
                echo "    bash build.sh --opkernel --soc=ascend310p --ops=add,sub --oom"
                return
                ;;
            ophost_test)
                echo "Ophost Test Options:"
                echo $dotted_line
                echo "    --ophost_test          Build and run ophost unit tests"
                echo "    --noexec               Only compile ut, do not execute"
                echo "    --cov                  Enable code coverage for unit tests"
                echo "    --ops=op1,op2,...      Compile specified operators (comma-separated for multiple)"
                echo "    --disable_asan         Disable ASAN (Address Sanitizer)"
                echo "    --soc=soc_version      Run unit tests for specified Ascend SoC"
                echo "    --valgrind             Run unit tests with valgrind (disables ASAN and noexec)"
                echo $dotted_line
                echo "Examples:"
                echo "    bash build.sh --ophost_test --noexec --cov"
                return
                ;;
            opapi_test)
                echo "Opapi Test Options:"
                echo $dotted_line
                echo "    --opapi_test           Build and run opapi unit tests"
                echo "    --noexec               Only compile ut, do not execute"
                echo "    --cov                  Enable code coverage for unit tests"
                echo "    --ops=op1,op2,...      Compile specified operators (comma-separated for multiple)"
                echo "    --disable_asan         Disable ASAN (Address Sanitizer)"
                echo "    --soc=soc_version      Run unit tests for specified Ascend SoC"
                echo "    --valgrind             Run unit tests with valgrind (disables ASAN and noexec)"
                echo $dotted_line
                echo "Examples:"
                echo "    bash build.sh --opapi_test --noexec --cov"
                return
                ;;
            opgraph_test)
                echo "Opgraph Test Options:"
                echo $dotted_line
                echo "    --opgraph_test         Build and run opgraph unit tests"
                echo "    --noexec               Only compile ut, do not execute"
                echo "    --cov                  Enable code coverage for unit tests"
                echo "    --ops=op1,op2,...      Compile specified operators (comma-separated for multiple)"
                echo "    --disable_asan         Disable ASAN (Address Sanitizer)"
                echo "    --soc=soc_version      Run unit tests for specified Ascend SoC"
                echo "    --valgrind             Run unit tests with valgrind (disables ASAN and noexec)"
                echo $dotted_line
                echo "Examples:"
                echo "    bash build.sh --opgraph_test --noexec --cov"
                return
                ;;
            run_example)
                echo "Run examples Options:"
                echo $dotted_line
                echo "    --run_example op_type  mode[eager:graph] [pkg_mode --vendor_name=name  --soc=soc_version]     Compile and execute the test_aclnn_xxx.cpp/test_geir_xxx.cpp"
                echo $dotted_line
                echo "Examples:"
                echo "    bash build.sh --run_example abs eager"
                echo "    bash build.sh --run_example abs eager --soc=ascend910_95"
                echo "    bash build.sh --run_example abs graph"
                echo "    bash build.sh --run_example abs eager cust"
                echo "    bash build.sh --run_example abs eager cust --vendor_name=custom"
                return
                ;;
            genop)
                echo "Gen Op Directory Options:"
                echo $dotted_line
                echo "    --genop=op_class/op_name      Create the initial directory for op_name undef op_class"
                echo $dotted_line
                echo "Examples:"
                echo "    bash build.sh --genop=example/add"
                return
                ;;
        esac
    fi
    echo "build script for ops-transformer repository"
    echo "Usage:"
    echo "    bash build.sh [-h] [-j[n]] [-v] [-O[n]] [-u] "
    echo ""
    echo ""
    echo "Options:"
    echo $dotted_line
    echo "    Build parameters "
    echo $dotted_line
    echo "    -h Print usage"
    echo "    -j[n] Compile thread nums, default is 8, eg: -j8"
    echo "    -v Cmake compile verbose"
    echo "    -O[n] Compile optimization options, support [O0 O1 O2 O3], eg:-O3"
    echo "    -u Compile all ut"
    echo $dotted_line
    echo "    examples, Build ophost_test with O3 level compilation optimization and do not execute."
    echo "    ./build.sh --ophost_test --noexec -O3"
    echo $dotted_line
    echo "    The following are all supported arguments:"
    echo $dotted_line
    echo "    --build-type=<TYPE> Specify build type (TYPE options:Release/Debug), Default: Release"
    echo "    --version Specify version"
    echo "    --cov When building uTest locally, count the coverage."
    echo "    --noexec Only compile ut, do not execute the compiled executable file"
    echo "    --make_clean Clean build artifacts"
    echo "    --disable_asan Disable ASAN (Address Sanitizer)"
    echo "    --valgrind run ut with valgrind. This option will disable asan, noexec and run utest by valgrind"
    echo "    --ops Compile specified operator, use snake name, like: --ops=add,add_lora, use ',' to separate different operator"
    echo "    --soc Compile binary with specified Ascend SoC, like: --soc=ascend310p,ascend910b, use ',' to separate different SoC"
    echo "    --vendor_name Specify the custom operator package vendor name, like: --vendor_name=customize, default to custom"
    echo "    --opgraph build graph_plugin_transformer.so"
    echo "    --onnxplugin build op_transformer_onnx_plugin.so"
    echo "    --opapi build opapi_transformer.so"
    echo "    --ophost build ophost_transformer.so"
    echo "    --opkernel build binary kernel"
    echo "    --jit build run package without kernel bin"
    echo "    --pkg build run package with kernel bin"
    echo "    --experimental build experimental version"
    echo "    --opapi_test build and run opapi unit tests"
    echo "    --ophost_test build and run ophost unit tests"
    echo "    --opgraph_test build and run opgraph unit tests"
    echo "    --opkernel_test build and run opkernel unit tests"
    echo "    --run_example Compile and execute the test_aclnn_xxx.cpp/test_geir_xxx.cpp"
    echo "    --genop Create the initial directory for op"
    echo "to be continued ..."
}


export BASE_PATH=$(
    cd "$(dirname $0)"
    pwd
)
export BUILD_PATH="${BASE_PATH}/build"
function log() {
    local current_time=`date +"%Y-%m-%d %H:%M:%S"`
    echo "[$current_time] "$1
}

function set_env()
{
    source $ASCEND_CANN_PACKAGE_PATH/bin/setenv.bash || echo "0"

    export BISHENG_REAL_PATH=$(which bisheng || true)

    if [ -z "${BISHENG_REAL_PATH}" ];then
        log "Error: bisheng compilation tool not found, Please check whether the cann package or environment variables are set."
        exit 1
    fi
}

function clean()
{
    if [ -n "${BUILD_DIR}" ];then
        rm -rf ${BUILD_DIR}
    fi

    if [ -z "${TEST}" ] && [ -z "${EXAMPLE}" ];then
        if [ -n "${OUTPUT_DIR}" ];then
            rm -rf ${OUTPUT_DIR}
        fi
    fi

    mkdir -p ${BUILD_DIR} ${OUTPUT_DIR}
}

function clean_build_out()
{
    if [ -n "${BUILD_OUT_DIR}" ];then
        rm -rf ${BUILD_OUT_DIR}
    fi

    mkdir -p ${BUILD_OUT_DIR}
}

function clean_third_party()
{
    THIRD_PARTY_PATH=${BASE_PATH}/third_party
    if [ -d "${THIRD_PARTY_PATH}" ]; then
        rm -rf ${THIRD_PARTY_PATH}/abseil-cpp
        rm -rf ${THIRD_PARTY_PATH}/ascend_protobuf
    fi
}

function cmake_config()
{
    local extra_option="$1"
    log "Info: cmake config ${CUSTOM_OPTION} ${extra_option} ."
    cmake ..  ${CUSTOM_OPTION} ${extra_option}
}

function build()
{
    local target="$1"
    if [ "${VERBOSE}" == "true" ];then
        local option="--verbose"
    fi
    export LD_LIBRARY_PATH=${BUILD_DIR}:$LD_LIBRARY_PATH
    
    cmake --build . --target ${target} ${JOB_NUM} ${option}
    if [ $? -ne 0 ]; then echo "[ERROR] build failed!" && exit 1; fi
}

ARCH_INFO=$(uname -m)

export INCLUDE_PATH="${ASCEND_HOME_PATH}/include"
export ACLNN_INCLUDE_PATH="${INCLUDE_PATH}/aclnn"
export COMPILER_INCLUDE_PATH="${ASCEND_HOME_PATH}/include"
export GRAPH_INCLUDE_PATH="${COMPILER_INCLUDE_PATH}/graph"
export GE_INCLUDE_PATH="${COMPILER_INCLUDE_PATH}/ge"
export INC_INCLUDE_PATH="${ASCEND_OPP_PATH}/built-in/op_proto/inc"
export LINUX_INCLUDE_PATH="${ASCEND_HOME_PATH}/${ARCH_INFO}-linux/include"
export EAGER_LIBRARY_OPP_PATH="${ASCEND_OPP_PATH}/lib64"
export EAGER_LIBRARY_PATH="${ASCEND_HOME_PATH}/lib64"
export GRAPH_LIBRARY_STUB_PATH="${ASCEND_HOME_PATH}/lib64/stub"
export GRAPH_LIBRARY_PATH="${ASCEND_HOME_PATH}/lib64"

export EAGER_INCLUDE_OPP_ACLNNOP_PATH="${ASCEND_HOME_PATH}/${ARCH_INFO}-linux/include/aclnnop"

function build_example()
{
    log "Start to run example,name:${EXAMPLE_NAME} mode:${EXAMPLE_MODE}"

    if [ ! -d "${BUILD_PATH}" ]; then
    	mkdir -p ${BUILD_PATH}
    fi

    # 清理CMake缓存
    # clean_cmake_cache
    clean

    cd "${BUILD_PATH}"
    if [[ "${EXAMPLE_MODE}" == "eager" ]]; then
        pattern="test_aclnn_"
    elif [[ "${EXAMPLE_MODE}" == "graph" ]]; then
        pattern="test_geir_"
    fi

    files=($(find ../ -path "*/${EXAMPLE_NAME}/examples/${pattern}*.cpp"))
    if [[ "$ASCEND_SOC_UNITS" == "ascend910_95" ]]; then
        files+=($(find ../ -path "*/${EXAMPLE_NAME}/examples/arch35/${pattern}*.cpp"))
    fi
    if [[ "${EXAMPLE_MODE}" == "eager" ]]; then
        if [ -z "$files" ]; then
            echo "${EXAMPLE_NAME} do not have eager example"
            return 2
        fi
        for file in "${files[@]}"; do
            echo "Start compile and run example file: $file"
            if [[ "${PKG_MODE}" == "" ]]; then
                g++ ${file} -I ${INCLUDE_PATH} -I ${ACLNN_INCLUDE_PATH} -I ${EAGER_INCLUDE_OPP_ACLNNOP_PATH} -L ${EAGER_LIBRARY_OPP_PATH} -L ${EAGER_LIBRARY_PATH} -lopapi_math -lopapi_transformer -lascendcl -lnnopbase -lpthread -lhccl -lhccl_fwk -lc_sec -o test_aclnn_${EXAMPLE_NAME}
            elif [[ "${PKG_MODE}" == "cust" ]]; then
                if [[ "${vendor_name}" == "" ]]; then
                    vendor_name="custom"
                fi
                echo "pkg_mode:${PKG_MODE} vendor_name:${vendor_name}"
                export CUST_LIBRARY_PATH="${ASCEND_OPP_PATH}/vendors/${vendor_name}_transformer/op_api/lib"     # 仅自定义算子需要
                export CUST_INCLUDE_PATH="${ASCEND_OPP_PATH}/vendors/${vendor_name}_transformer/op_api/include" # 仅自定义算子需要
                if [[ -n "${ASCEND_CUSTOM_OPP_PATH}" ]]; then
                    CUST_VENDORS_PATH=$(dirname "${ASCEND_CUSTOM_OPP_PATH%%:*}")
                    CUST_LIBRARY_PATH="${CUST_VENDORS_PATH}/${vendor_name}_transformer/op_api/lib"
                    CUST_INCLUDE_PATH="${CUST_VENDORS_PATH}/${vendor_name}_transformer/op_api/include"
                fi
                ABSOLUTE_MC2_PATH=$(realpath ${BUILD_PATH}/../mc2)
                ABSOLUTE_EXAMPLES_PATH=$(realpath ${BUILD_PATH}/../examples/mc2)
                ABSOLUTE_EXPERIMENTAL_MC2_PATH=$(realpath ${BUILD_PATH}/../experimental/mc2)
                REAL_FILE_PATH=$(realpath "$file")
                MC2_APPEND_INCLUDE_AND_LIBRARY=""
                if [[ "$REAL_FILE_PATH" == "${ABSOLUTE_MC2_PATH}"* || "$REAL_FILE_PATH" == "${ABSOLUTE_EXAMPLES_PATH}"* || "$REAL_FILE_PATH" == "${ABSOLUTE_EXPERIMENTAL_MC2_PATH}"* ]]; then
                    MC2_APPEND_INCLUDE_AND_LIBRARY="-lpthread -lhccl -lhccl_fwk"
                fi
                g++ ${file} -I ${INCLUDE_PATH} -I ${CUST_INCLUDE_PATH} -L ${CUST_LIBRARY_PATH} -L ${EAGER_LIBRARY_PATH} -lcust_opapi -lascendcl -lnnopbase -I ${EAGER_INCLUDE_OPP_ACLNNOP_PATH} ${MC2_APPEND_INCLUDE_AND_LIBRARY} -lc_sec -o test_aclnn_${EXAMPLE_NAME} -Wl,-rpath=${CUST_LIBRARY_PATH}
            else
                echo "Error: pkg_mode(${PKG_MODE}) must be cust."
                help_info "run_example"
                return 1
            fi
            ./test_aclnn_${EXAMPLE_NAME}
            run_result=$?
            if [ $run_result -ne 0 ]; then
                echo "run test_aclnn_${EXAMPLE_NAME}, execute samples failed"
                return $run_result
            else
                echo "run test_aclnn_${EXAMPLE_NAME}, execute samples success"
            fi
        done
    elif [[ "${EXAMPLE_MODE}" == "graph" ]]; then
        if [ -z "$files" ]; then
            echo "${EXAMPLE_NAME} do not have graph example"
            return 2
        fi
        for file in "${files[@]}"; do
            echo "Start compile and run example file: $file"
            g++ ${file} -D_GLIBCXX_USE_CXX11_ABI=0 -I ${GRAPH_INCLUDE_PATH} -I ${GE_INCLUDE_PATH} -I ${LINUX_INCLUDE_PATH} -I ${INC_INCLUDE_PATH} -L ${GRAPH_LIBRARY_STUB_PATH} -L ${GRAPH_LIBRARY_PATH} -lgraph -lge_runner -lgraph_base -lge_compiler -o test_geir_${EXAMPLE_NAME}
            ./test_geir_${EXAMPLE_NAME}
            run_result=$?
            if [ $run_result -ne 0 ]; then
                echo "run test_geir_${EXAMPLE_NAME}, execute samples failed"
                return $run_result
            else
                echo "run test_geir_${EXAMPLE_NAME}, execute samples success"
            fi
        done
    else
        help_info "run_example"
        return 1
    fi
    return 0
}

function gen_bisheng(){
    local ccache_program=$1
    local gen_bisheng_dir=${BUILD_DIR}/gen_bisheng_dir

    if [ ! -d "${gen_bisheng_dir}" ];then
        mkdir -p ${gen_bisheng_dir}
    fi

    pushd ${gen_bisheng_dir}
    $(> bisheng)
    echo "#!/bin/bash" >> bisheng
    echo "ccache_args=""\"""${ccache_program} ${BISHENG_REAL_PATH}""\"" >> bisheng
    echo "args=""$""@" >> bisheng

    if [ "${VERBOSE}" == "true" ];then
        echo "echo ""\"""$""{ccache_args} ""$""args""\"" >> bisheng
    fi

    echo "eval ""\"""$""{ccache_args} ""$""args""\"" >> bisheng
    chmod +x bisheng

    export PATH=${gen_bisheng_dir}:$PATH
    popd
}

function build_package(){
    build package
}

function build_host(){
    build_package
}

function build_kernel(){
    build ops_transformer_kernel
}

build_lib() {
  echo $dotted_line
  echo "Start to build libs ${BUILD_LIBS[@]}"
  clean

  if [ ! -d "${BUILD_PATH}" ]; then
    mkdir -p "${BUILD_PATH}"
  fi

  cd "${BUILD_PATH}" && cmake .. ${CUSTOM_OPTION} -DENABLE_BUILT_IN=ON

  for lib in "${BUILD_LIBS[@]}"; do
    echo "Building target ${lib}"
    cmake --build . --target ${lib} ${JOB_NUM}
  done

  echo $dotted_line
  echo "Build libs ${BUILD_LIBS[@]} success"
  echo $dotted_line
}

build_static_lib() {
    local unit="$1"
    echo $dotted_line
    echo "Start to build static lib."

    git submodule init && git submodule update
    cd "${BUILD_PATH}" && cmake ${CUSTOM_OPTION} .. -DENABLE_STATIC=ON -DASCEND_COMPUTE_UNIT=${unit}
    local all_targets=$(cmake --build . --target help)
    rm -fr ${BUILD_PATH}/bin_tmp
    mkdir -p ${BUILD_PATH}/bin_tmp
    if echo "${all_targets}" | grep -wq "ophost_transformer_static"; then
        cmake --build . --target ophost_transformer_static ${JOB_NUM}
    fi
    cmake --build . --target opapi_transformer_static ${JOB_NUM}
    local jit_command=""
    if [[ "$ENABLE_BUILT_JIT" == "TRUE" ]]; then
        jit_command="-j"
    fi

    rm -fr ${BUILD_PATH}/autogen/${unit}
    python3 "${BASE_PATH}/scripts/util/build_opp_kernel_static.py" GenStaticOpResourceIni -s ${unit} -b ${BUILD_PATH} ${jit_command}   
    python3 "${BASE_PATH}/scripts/util/build_opp_kernel_static.py" StaticCompile -s ${unit} -b ${BUILD_PATH} -n=0 -a=${ARCH_INFO} ${jit_command}

    cd "${BUILD_PATH}" && cmake ${CUSTOM_OPTION} .. -DENABLE_STATIC=ON -DASCEND_COMPUTE_UNIT=${unit}
    cmake --build . --target cann_transformer_static ${JOB_NUM}
    echo "Build static lib success!"
}

package_static() {
    local unit="$1"
    if [[ "$ENABLE_BUILT_CUSTOM" == "TRUE" ]]; then
        CUSTOM_OPTION="${CUSTOM_OPTION} -DENABLE_BUILT_IN=OFF -DENABLE_OPS_HOST=ON -DENABLE_BUILD_PKG=ON"
    else
        CUSTOM_OPTION="${CUSTOM_OPTION} -DENABLE_BUILT_IN=ON -DENABLE_OPS_HOST=ON -DENABLE_BUILD_PKG=ON"
    fi
    if [[ "$ENABLE_BUILT_JIT" == "TRUE" ]]; then
        CUSTOM_OPTION="${CUSTOM_OPTION} -DENABLE_OPS_KERNEL=OFF"
    else
        CUSTOM_OPTION="${CUSTOM_OPTION} -DENABLE_OPS_KERNEL=ON"
    fi
    cmake_config -DASCEND_COMPUTE_UNIT=${unit}
    build_package
    # Check weather BUILD_OUT_DIR directory exists
    if [ ! -d "$BUILD_OUT_DIR" ]; then
        echo "Error: Directory $BUILD_OUT_DIR does not exist."
        return 1
    fi

    # Check weather *.run is exists and verify the file numbers
    local run_files=("$BUILD_OUT_DIR"/*.run)
    if [ ${#run_files[@]} -eq 0 ]; then
        echo "Error: No .run files found in $BUILD_OUT_DIR directory."
        return 1
    fi
    if [ ${#run_files[@]} -gt 1 ]; then
        echo "Error: Multiple .run files found in $BUILD_OUT_DIR directory."
        printf '%s\n' "${run_files[@]}"
        return 1
    fi
    # Get filename of *.run file and set new directory name
    local run_file=$(basename "${run_files[0]}")
    echo "Found .run file: $run_file"
    if [[ "$run_file" != *"ops-transformer"* ]]; then
        echo "Error: Filename '$run_file' does not contain 'ops-transformer'."
        return 1
    fi
    local static_name="${run_file/ops-transformer/ops-transformer-static}"
    static_name="${static_name%.run}"

    # Check weather $BUILD_PATH/static_library_files directory exists and not empty
    local static_files_dir="$BUILD_PATH/static_library_files"
    if [ ! -d "$static_files_dir" ]; then
        echo "Error: Directory $static_files_dir does not exist."
        return 1
    fi
    if [ -z "$(ls -A "$static_files_dir")" ]; then
        echo "Error: Directory $static_files_dir is empty."
        return 1
    fi

    # Rename directory
    local new_dir_path="$BUILD_PATH/$static_name"
    if mv "$static_files_dir" "$new_dir_path"; then
        echo "Preparing for packaging: renamed $static_files_dir to $new_dir_path"
    else
        echo "Packaging preparation failed: directory rename failed ($static_files_dir -> $new_dir_path)"
        return 1
    fi

    # Create compressed package and restore directory name
    local new_filename="${static_name}.tar.gz"
    if tar -czf "$BUILD_OUT_DIR/$new_filename" -C "$BUILD_PATH" "$static_name"; then
        echo "Successfully created compressed package: $BUILD_OUT_DIR/$new_filename"
        # Restore original directory name
        echo "Restoring original directory name: $new_dir_path -> $static_files_dir"
        mv "$new_dir_path" "$static_files_dir"
        return 0
    else
        echo "Error: Failed to create compressed package."
        # Attempt to restore original directory name
        mv "$new_dir_path" "$static_files_dir"
        return 1
    fi
    make clean
}

function process_soc_input(){
    local input_string="$1"
    input_string=$(echo "$input_string" | sed 's/ascend950/ascend910_95/g')
    local value_part="${input_string#*=}"
    ASCEND_SOC_UNITS="${value_part//,/;}"
}

  process_genop() {
    local opt_name=$1
    local genop_value=$2

    if [[ "$opt_name" == "genop" ]]; then
      ENABLE_GENOP=TRUE
    elif [[ "$opt_name" == "genop_aicpu" ]]; then
      ENABLE_GENOP_AICPU=TRUE
    else
      usage "genop"
      exit 1
    fi

    if [[ "$genop_value" != *"/"* ]] || [[ "$genop_value" == *"/" ]]; then
      usage "$opt_name"
      exit 1
    fi

    GENOP_NAME=${genop_value##*/}
    local remaining=${genop_value%/*}

    if [[ "$remaining" != *"/"* ]]; then
      GENOP_TYPE=$remaining
      GENOP_BASE=${BASE_PATH}
    else
      GENOP_TYPE=${remaining##*/}
      GENOP_BASE=${remaining%/*}
      if [[ ! "$GENOP_BASE" =~ ^/ && ! "$GENOP_BASE" =~ ^[a-zA-Z]: ]]; then
        GENOP_BASE="${BASE_PATH}/${GENOP_BASE}"
      fi
    fi
  }

gen_op() {
  if [[ -z "$GENOP_NAME" ]] || [[ -z "$GENOP_TYPE" ]]; then
    echo "Error: op_class or op_name is not set."
    usage "genop"
  fi

  echo $dotted_line
  echo "Start to create the initial directory for ${GENOP_NAME} under ${GENOP_TYPE}"

  # 检查 python 或 python3 是否存在
  local python_cmd=""
  if command -v python3 &> /dev/null; then
      python_cmd="python3"
  elif command -v python &> /dev/null; then
      python_cmd="python"
  fi
  
  if [ -n "${python_cmd}" ]; then
    ${python_cmd} "${BASE_PATH}/scripts/opgen/opgen_standalone.py" -t ${GENOP_TYPE} -n ${GENOP_NAME} -p ${GENOP_BASE}
    return $?
  fi
}

# function gen_op() {
#   if [[ -z "$GENOP_NAME" ]] || [[ -z "$GENOP_TYPE" ]]; then
#     echo "Error: op_class or op_name is not set."
#     help_info "genop"
#   fi

#   echo $dotted_line
#   echo "Start to create the initial directory for ${GENOP_NAME} under ${GENOP_TYPE}"

#   if [ ! -d "${GENOP_TYPE}" ]; then
#     mkdir -p "${GENOP_TYPE}"
#     cp examples/CMakeLists.txt "${GENOP_TYPE}/CMakeLists.txt"
#     sed -i '/list(APPEND OP_DIR_LIST ${CMAKE_CURRENT_SOURCE_DIR}\/ffn\/ffn)/a add_subdirectory('"${GENOP_TYPE}"')' CMakeLists.txt
#   fi

#   BASE_DIR=${GENOP_TYPE}/${GENOP_NAME}
#   mkdir -p "${BASE_DIR}"

#   cp -r examples/add_example/* "${BASE_DIR}/"

#   rm -rf "${BASE_DIR}/examples"
#   rm -rf "${BASE_DIR}/op_host/config"

#   for file in $(find "${BASE_DIR}" -name "*.h" -o -name "*.cpp"); do
#     head -n 14 "$file" >"${file}.tmp"
#     cat "${file}.tmp" >"$file"
#     rm "${file}.tmp"
#   done

#   for file in $(find "${BASE_DIR}" -type f); do
#     sed -i "s/add_example/${GENOP_NAME}/g" "$file"
#   done

#   cd ${BASE_DIR}
#   for file in $(find ./ -name "add_example*"); do
#     new_file=$(echo "$file" | sed "s/add_example/${GENOP_NAME}/g")
#     mv "$file" "$new_file"
#   done

#   echo "Create the initial directory for ${GENOP_NAME} under ${GENOP_TYPE} success"
# }


set_ut_mode() {
  REPOSITORY_NAME="transformer"
  if [[ "$ENABLE_TEST" != "TRUE" ]]; then
    return
  fi
  if [ -n "${PR_CHANGED_FILES}" ]; then
    OP_HOST_UT=TRUE
    OP_API_UT=TRUE
    UT_TEST_ALL=FALSE
    UT_TARGETS+=("${REPOSITORY_NAME}_op_host_ut")
    UT_TARGETS+=("${REPOSITORY_NAME}_op_api_ut")
    return
  fi 
  UT_TEST_ALL=TRUE
  if [[ "$OP_HOST" == "TRUE" ]]; then
    OP_HOST_UT=TRUE
    UT_TEST_ALL=FALSE
  fi
  if [[ "$OP_API" == "TRUE" ]]; then
    OP_API_UT=TRUE
    UT_TEST_ALL=FALSE
  fi
  if [[ "$OP_GRAPH" == "TRUE" ]]; then
    OP_GRAPH_UT=TRUE
    UT_TEST_ALL=FALSE
  fi
  if [[ "$OP_KERNEL" == "TRUE" ]]; then
    OP_KERNEL_UT=TRUE
    UT_TEST_ALL=FALSE
  fi
  if [[ "$UT_TEST_ALL" == "TRUE" ]] || [[ "$OP_HOST_UT" == "TRUE" ]]; then
    UT_TARGETS+=("${REPOSITORY_NAME}_op_host_ut")
  fi
  if [[ "$UT_TEST_ALL" == "TRUE" ]] || [[ "$OP_API_UT" == "TRUE" ]]; then
    UT_TARGETS+=("${REPOSITORY_NAME}_op_api_ut")
  fi
  if [[ "$UT_TEST_ALL" == "TRUE" ]] || [[ "$OP_GRAPH_UT" == "TRUE" ]]; then
    UT_TARGETS+=("${REPOSITORY_NAME}_op_graph_ut")
  fi
  if [[ "$UT_TEST_ALL" == "TRUE" ]] || [[ "$OP_KERNEL_UT" == "TRUE" ]]; then
    UT_TARGETS+=("${REPOSITORY_NAME}_op_kernel_ut")
  fi
}

set_example_opt() {
  if [[ -n $1 && $1 != -* ]]; then
    EXAMPLE_NAME=$1
    step=$((step + 1))
  fi
  if [[ -n $2 && $2 != -* ]]; then
    EXAMPLE_MODE=$2
    step=$((step + 1))
  fi
  if [[ -n $3 && $3 != -* ]]; then
    PKG_MODE=$3
    step=$((step + 1))
  fi
}

########################################################################################################################
# 参数解析处理
########################################################################################################################
set -o pipefail
{
if [[ $# -eq 0 ]]; then
    help_info
    exit 1
fi
for arg in "$@"; do
    if [[ "$arg" == "--help" || "$arg" == "-h" ]]; then
        SHOW_HELP="general"
        # 检查 --help 前面的命令
        for prev_arg in "$@"; do
            case "$prev_arg" in
            --pkg) SHOW_HELP="package" ;;
            --opkernel) SHOW_HELP="opkernel" ;;
            -u|--test) SHOW_HELP="test" ;;
            --make_clean) SHOW_HELP="clean" ;;
            --valgrind) SHOW_HELP="valgrind" ;;
            --ophost) SHOW_HELP="ophost" ;;
            --opapi) SHOW_HELP="opapi" ;;
            --opgraph) SHOW_HELP="opgraph" ;;
            --onnxplugin) SHOW_HELP="onnxplugin" ;;
            --ophost_test) SHOW_HELP="ophost_test" ;;
            --opapi_test) SHOW_HELP="opapi_test" ;;
            --opgraph_test) SHOW_HELP="opgraph_test" ;;
            --run_example) SHOW_HELP="run_example" ;;
            --genop) SHOW_HELP="genop" ;;
            esac
        done
      help_info "$SHOW_HELP"
      exit 0
    fi
  done

while [[ $# -gt 0 ]]; do
    case $1 in
    -h|--help)
        help_info
        exit
        ;;
    --pkg)
        ENABLE_BUILD_PKG=TRUE
        ENABLE_BUILT_IN=TRUE            # 只输入--pkg时编builtin包
        shift
        ;;
    --static)
        ENABLE_STATIC=TRUE
        shift
        ;;
    --jit)
        ENABLE_BUILT_JIT=TRUE
        shift
        BUILD="jit"
        ;;
    -n|--op-name)
        ascend_op_name="$2"
        shift 2
        ;;
    --ops=*)
        OPTARG=$1
        ascend_op_name=${OPTARG#*=}
        ENABLE_BUILT_CUSTOM=TRUE
        ENABLE_BUILT_IN=FALSE
        shift
        ;;
    -c|--compute-unit)
        ascend_compute_unit="$2"
        shift 2
        ;;
    --soc=*)
        process_soc_input "$1"
        shift 1
    ;;
    --ccache)
        CCACHE_PROGRAM="$2"
        shift 2
        ;;
    -p|--package-path)
        ascend_package_path="$2"
        shift 2
        ;;
    -b|--build)
        BUILD="$2"
        shift 2
        ;;
    -u|--test)
        ENABLE_TEST=TRUE
        shift
        ;;
    --run_example)
        ENABLE_RUN_EXAMPLE=TRUE
        step=1
        set_example_opt $2 $3 $4
        shift $step
        ;;
    --experimental) 
        ENABLE_EXPERIMENTAL=TRUE
        shift
        ;;
    -e|--example)
        shift
        if [ -n "$1" ];then
            _parameter=$1
            first_char=${_parameter:0:1}
            if [ "${first_char}" == "-" ];then
                EXAMPLE="all"
            else
                EXAMPLE="${_parameter}"
                shift
            fi
        else
            EXAMPLE="all"
        fi
        ;;
    -f|--changed_list)
        PR_CHANGED_FILES="$2"
        ENABLE_SMOKE=TRUE
        PKG_MODE="cust"
        vendor_name="custom"
        shift 2
        ;;
    --PR_UT)
        PR_CHANGED_FILES="$2"
        ENABLE_TEST=TRUE 
        shift 2
        ;;
    --PR_PKG)
        PR_CHANGED_FILES="$2"
        ops_names=$(python3 "$CURRENT_DIR"/cmake/scripts/parse_changed_files.py -c "$CURRENT_DIR"/tests/test_config.yaml -f "$PR_CHANGED_FILES" get_related_examples)
        echo "Operators that need custom package compilation:$ops_names"
        if [ -z "${ops_names}" ];then
            log "Info: No custom packages to build for this PR."
            # ops_names="incre_flash_attention"
            exit 200
        fi 
        ops_names="${ops_names%;}"
        ops_names="${ops_names//;/,}"
        ascend_op_name="$ops_names"
        ENABLE_BUILD_PKG=TRUE
        ENABLE_BUILT_CUSTOM=TRUE
        ENABLE_BUILT_IN=FALSE
        shift 2
        ;;
    --parent_job)
        PARENT_JOB="true"
        shift
        ;;
    --enable_host_tiling)
        HOST_TILING="true"
        shift
        ;;
    --disable-check-compatible|--disable-check-compatiable)
        CHECK_COMPATIBLE="false"
        shift
        ;;
    --op_build_tool)
        op_build_tool="$2"
        shift 2
        ;;
    --ascend_cmake_dir)
        ascend_cmake_dir="$2"
        shift 2
        ;;
    -v|--verbose)
        VERBOSE="true"
        shift
        ;;
    --disable_asan)
        ASAN="false"
        shift
        ;;
    --valgrind)
        ENABLE_VALGRIND=TRUE
        ENABLE_UT_EXEC=FALSE
        ASAN="false"
        shift
        ;;
    --ubsan)
        UBSAN="true"
        shift
        ;;
    --cov)
        COV="true"
        shift
        ;;
    --clang)
        CLANG="true"
        shift
        ;;
    --tiling-key|--tiling_key)
        TILING_KEY="$2"
        shift 2
        ;;
    --tiling_key=*)
        OPTARG=$1
        TILING_KEY=${OPTARG#*=}
        shift
        ;;
    --op_debug_config)
        OP_DEBUG_CONFIG="$2"
        shift 2
        ;;
    --ops-compile-options)
        OPS_COMPILE_OPTIONS="$2"
        shift 2
        ;;
    --ophost_test)
        ENABLE_TEST=TRUE
        OP_HOST=TRUE
        shift
        ;;
    --opapi_test)
        ENABLE_TEST=TRUE
        OP_API=TRUE
        shift
        ;;
    --opgraph_test)
        ENABLE_TEST=TRUE
        OP_GRAPH=TRUE
        shift
        ;;
    --opkernel_test)
        ENABLE_TEST=TRUE
        OP_KERNEL=TRUE
        shift
        ;;
    --vendor_name=*)
        OPTARG=$1
        vendor_name=${OPTARG#*=}
        ENABLE_BUILT_CUSTOM=TRUE
        ENABLE_BUILT_IN=FALSE
        shift
        ;;
    --opgraph)
        BUILD_LIBS+=("opgraph_transformer")
        ENABLE_CREATE_LIB=TRUE
        OP_GRAPH=TRUE
        shift
        ;;
    --onnxplugin)
        BUILD_LIBS+=("op_transformer_onnx_plugin")
        ENABLE_CREATE_LIB=TRUE
        ONNX_PLUGIN=TRUE
        shift
        ;;
    --opapi)
        BUILD_LIBS+=("opapi_transformer")
        ENABLE_CREATE_LIB=TRUE
        OP_API=TRUE
        shift
        ;;
    --ophost)
        BUILD_LIBS+=("ophost_transformer")
        ENABLE_CREATE_LIB=TRUE
        OP_HOST=TRUE
        shift
        ;;
    --opkernel)
        ENABLE_OPKERNEL=TRUE
        OP_KERNEL=TRUE
        shift
        ;;
    --noexec)
        ENABLE_UT_EXEC=FALSE
        shift
        ;;
    -j*)
        OPTARG=$1
        if [[ "$OPTARG" =~ ^-j[0-9]+$ ]]; then
            INPUT_JOB_NUM="${OPTARG#*-j}"
            # 可选：添加范围检查
            if [ "$INPUT_JOB_NUM" -le 0 ]; then
                echo "Error: Job number must be positive: $OPTARG" >&2
                exit 1
            fi
        else
            echo "Error: Invalid job number format: $OPTARG" >&2
            echo "Expected: -j[n]" >&2
            exit 1
        fi
        shift 1
        ;;
    --build-type=*)
        OPTARG=$1
        BUILD_TYPE=${OPTARG#*=}
        shift
        ;;
    --version=*)
        OPTARG=$1
        VERSION=${OPTARG#*=}
        shift
        ;;
    -O[0-3])
        CMAKE_BUILD_MODE=$1
        shift
        ;;
    --genop=*)
        OPTARG=$1
        process_genop "genop" "${OPTARG#*=}"
        shift
        ;;
    --make_clean)
        clean
        clean_build_out
        clean_third_party
        shift
        ;;
    --cann_3rd_lib_path=*)
        OPTARG=$1
        CANN_3RD_LIB_PATH="$(realpath ${OPTARG#*=})"
        shift
        ;;
    --oom)
        OOM="true"
        shift
        ;;
    *)
        help_info
        exit 1
        ;;
    esac
done
set_ut_mode

if [ -n "${vendor_name}" ];then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DVENDOR_NAME=${vendor_name}"
fi

if [ -n "${VERSION}" ];then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DVERSION=${VERSION}"
fi

if [[ "$ENABLE_EXPERIMENTAL" == "TRUE" ]]; then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DENABLE_EXPERIMENTAL=TRUE"
fi

if [ -n "${ascend_compute_unit}" ];then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DASCEND_COMPUTE_UNIT=${ascend_compute_unit}"
fi

if [ -n "${ascend_op_name}" ];then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DASCEND_OP_NAME=${ascend_op_name}"
fi

if [ -n "${op_build_tool}" ];then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DOP_BUILD_TOOL=${op_build_tool}"
fi

if [ -n "${ascend_cmake_dir}" ];then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DASCEND_CMAKE_DIR=${ascend_cmake_dir}"
fi
if [[ "$ENABLE_TEST" == "TRUE" ]]; then
    if [ -z "$ascend_op_name" ]; then
        TEST="all"
    else
        TEST="$ascend_op_name"
    fi
    CUSTOM_OPTION="${CUSTOM_OPTION} -DENABLE_TEST=TRUE"
fi
if [[ "$OP_HOST_UT" == "TRUE" ]]; then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DOP_HOST_UT=TRUE"
fi
if [[ "$OP_API_UT" == "TRUE" ]]; then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DOP_API_UT=TRUE"
fi
if [[ "$OP_GRAPH_UT" == "TRUE" ]]; then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DOP_GRAPH_UT=TRUE"
fi
if [[ "$OP_KERNEL_UT" == "TRUE" ]]; then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DOP_KERNEL_UT=TRUE"
fi
if [[ "$UT_TEST_ALL" == "TRUE" ]]; then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DUT_TEST_ALL=TRUE"
fi
if [[ "$ENABLE_UT_EXEC" == "TRUE" ]]; then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DENABLE_UT_EXEC=TRUE"
fi

if [ -n "${TEST}" ];then
    if [ -n "${PR_CHANGED_FILES}" ];then
        TEST=$(python3 "$CURRENT_DIR"/cmake/scripts/parse_changed_files.py -c "$CURRENT_DIR"/tests/test_config.yaml -f "$PR_CHANGED_FILES" get_related_ut)
        echo "Operators that need to run UT: $TEST"
        if [ -z "${TEST}" ];then
            log "Info: This PR didn't trigger any UTest."
            exit 0
        fi
        if [ "$TEST" != "all" ];then
            TEST="${TEST%;}"
            TEST="${TEST//;/,}"
            CUSTOM_OPTION="${CUSTOM_OPTION} -DASCEND_OP_NAME=${TEST}"
        fi
        CUSTOM_OPTION="${CUSTOM_OPTION} -DTESTS_UT_OPS_TEST_CI_PR=ON"
    fi
    CUSTOM_OPTION="${CUSTOM_OPTION} -DTESTS_UT_OPS_TEST=${TEST}"
    if [ "${CLANG}" == "true" ];then
        CLANG_C_COMPILER="$(which clang)"
        if [ ! -f "${CLANG_C_COMPILER}" ];then
            log "Error: Can't find clang path ${CLANG_C_COMPILER}"
            exit 1
        fi

        CLANG_PATH=$(dirname "${CLANG_C_COMPILER}")
        CLANG_CXX_COMPILER="${CLANG_PATH}/clang++"
        CLANG_LINKER="${CLANG_PATH}/lld"
        CLANG_AR="${CLANG_PATH}/llvm-ar"
        CLANG_STRIP="${CLANG_PATH}/llvm-strip"
        CLANG_OBJCOPY="${CLANG_PATH}/llvm-objcopy"

        CUSTOM_OPTION="${CUSTOM_OPTION} -DCMAKE_C_COMPILER=${CLANG_C_COMPILER} -DCMAKE_CXX_COMPILER=${CLANG_CXX_COMPILER}"
        CUSTOM_OPTION="${CUSTOM_OPTION} -DCMAKE_LINKER=${CLANG_LINKER} -DCMAKE_AR=${CLANG_AR} -DCMAKE_STRIP=${CLANG_STRIP}"
        CUSTOM_OPTION="${CUSTOM_OPTION} -DCMAKE_OBJCOPY=${CLANG_OBJCOPY}"
    fi

    if [ "${ASAN}" == "true" ];then
        CUSTOM_OPTION="${CUSTOM_OPTION} -DENABLE_ASAN=TRUE"
    fi

    if [ "${ENABLE_VALGRIND}" == "TRUE" ];then
        CUSTOM_OPTION="${CUSTOM_OPTION} -DENABLE_VALGRIND=TRUE"
    fi

    if [ "${UBSAN}" == "true" ];then
        CUSTOM_OPTION="${CUSTOM_OPTION} -DENABLE_UBSAN=true"
    fi

    BUILD=ops_test_utest
fi

if [ "${COV}" == "true" ];then
    if [ "${CLANG}" == "true" ];then
        log "Warning: GCOV only supported in gnu compiler."
    else
        CUSTOM_OPTION="${CUSTOM_OPTION} -DENABLE_GCOV=true"
    fi
fi

if [ "${OOM}" == "true" ];then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DENABLE_OOM=ON"
fi

if [ -n "${EXAMPLE}" ];then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DTESTS_EXAMPLE_OPS_TEST=${EXAMPLE}"

    BUILD=ops_test_example
fi

if [ -n "${TILING_KEY}" ];then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DTILING_KEY=${TILING_KEY}"
fi

if [ -n "${OP_DEBUG_CONFIG}" ];then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DOP_DEBUG_CONFIG=${OP_DEBUG_CONFIG}"
fi

if [ -n "${OPS_COMPILE_OPTIONS}" ];then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DOPS_COMPILE_OPTIONS=${OPS_COMPILE_OPTIONS}"
fi

if [ "${HOST_TILING}" == "true" ];then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DENABLE_HOST_TILING=ON"
fi

if [ "${ENABLE_DEBUG}" == "TRUE" ];then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DENABLE_DEBUG=ON"
fi

if [ -n "${BUILD_TYPE}" ];then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DCMAKE_BUILD_TYPE=${BUILD_TYPE}"
fi

if [ -n "${VERSION}" ];then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DVERSION=${VERSION}"
fi

if [ -n "${CMAKE_BUILD_MODE}" ];then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DCMAKE_BUILD_MODE=${CMAKE_BUILD_MODE}"
fi
CUSTOM_OPTION="${CUSTOM_OPTION} -DCANN_3RD_LIB_PATH=${CANN_3RD_LIB_PATH}"

if [[ "$ENABLE_STATIC" == "TRUE" ]]; then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DENABLE_STATIC=${ENABLE_STATIC}"
fi

if [ -n "${ascend_package_path}" ];then
    ASCEND_CANN_PACKAGE_PATH=${ascend_package_path}
elif [ -n "${ASCEND_HOME_PATH}" ];then
    ASCEND_CANN_PACKAGE_PATH=${ASCEND_HOME_PATH}
elif [ -n "${ASCEND_OPP_PATH}" ];then
    ASCEND_CANN_PACKAGE_PATH=$(dirname ${ASCEND_OPP_PATH})
elif [ -d "${DEFAULT_TOOLKIT_INSTALL_DIR}" ];then
    ASCEND_CANN_PACKAGE_PATH=${DEFAULT_TOOLKIT_INSTALL_DIR}
elif [ -d "${DEFAULT_INSTALL_DIR}" ];then
    ASCEND_CANN_PACKAGE_PATH=${DEFAULT_INSTALL_DIR}
else
    log "Error: Please set the toolkit package installation directory through parameter -p|--package-path."
    exit 1
fi

function get_cpu_num() {
    CPU_NUM=$(($(cat /proc/cpuinfo | grep "^processor" | wc -l)*2)) 
    if [ -n "${OPS_CPU_NUMBER}" ]; then
        if [[ "${OPS_CPU_NUMBER}" =~ ^[0-9]+$ ]]; then
            CPU_NUM="${OPS_CPU_NUMBER}"
        fi
    fi
}

if [ "${PARENT_JOB}" == "false" ]; then
    get_cpu_num
    JOB_NUM="-j${CPU_NUM}"
fi

if [ -n "${INPUT_JOB_NUM}" ]; then
    get_cpu_num
    if [ ${INPUT_JOB_NUM} -gt ${CPU_NUM}  ]; then
        INPUT_JOB_NUM=${CPU_NUM}
    fi
    JOB_NUM="-j${INPUT_JOB_NUM}"
fi

# 非打包命令调用，打包模式会打进同一个包里
function set_compute_unit_option() {
    IFS=';' read -ra SOC_ARRAY <<< "$ASCEND_SOC_UNITS"  # 分割字符串为数组
    local COMPUTE_UNIT_SHORT=""
    for soc in "${SOC_ARRAY[@]}"; do
      for support_unit in "${SUPPORT_COMPUTE_UNIT_SHORT[@]}"; do
        lowercase_word=$(echo "$soc" | tr '[:upper:]' '[:lower:]')
        if [[ "$lowercase_word" == *"$support_unit"* ]]; then
          COMPUTE_UNIT_SHORT="$COMPUTE_UNIT_SHORT$support_unit;"
          break
        fi
      done
    done
    CUSTOM_OPTION="$CUSTOM_OPTION -DASCEND_COMPUTE_UNIT=$COMPUTE_UNIT_SHORT"
}

CUSTOM_OPTION="${CUSTOM_OPTION} -DCUSTOM_ASCEND_CANN_PACKAGE_PATH=${ASCEND_CANN_PACKAGE_PATH} -DCHECK_COMPATIBLE=${CHECK_COMPATIBLE}"

########################################################################################################################
# 处理流程
########################################################################################################################

set_env

clean

if [ -n "${CCACHE_PROGRAM}" ]; then
    if [ "${CCACHE_PROGRAM}" == "false" ] || [ "${CCACHE_PROGRAM}" == "off" ]; then
        CUSTOM_OPTION="${CUSTOM_OPTION} -DENABLE_CCACHE=OFF"
    elif [ -f "${CCACHE_PROGRAM}" ];then
        CUSTOM_OPTION="${CUSTOM_OPTION} -DENABLE_CCACHE=ON -DCUSTOM_CCACHE=${CCACHE_PROGRAM}"
        gen_bisheng ${CCACHE_PROGRAM}
    fi
else
    # 判断有无默认的ccache 如果有则使用
    ccache_system=$(which ccache || true)
    if [ -n "${ccache_system}" ];then
        CUSTOM_OPTION="${CUSTOM_OPTION} -DENABLE_CCACHE=ON -DCUSTOM_CCACHE=${ccache_system}"
        gen_bisheng ${ccache_system}
    fi
fi
build_ut() {
  CORE_NUMS=$(cat /proc/cpuinfo | grep "processor" | wc -l)
  dotted_line="----------------------------------------------------------------"
  echo $dotted_line
  echo "Start to build ut"

  git submodule init && git submodule update
  if [ ! -d "${BUILD_DIR}" ]; then
    mkdir -p "${BUILD_DIR}"
  fi
  cd "${BUILD_DIR}" && cmake ${CUSTOM_OPTION} ..

  local target="$1"
  if [ "${VERBOSE}" == "true" ]; then
    local option="--verbose"
  fi
  if [ $(cmake -LA -N . | grep 'UTEST_FRAMEWORK_OLD:BOOL=' | cut -d'=' -f2) == "TRUE" ]; then
    cmake --build . --target ${target} ${JOB_NUM} ${option}
  fi

  if [ $(cmake -LA -N . | grep 'UTEST_FRAMEWORK_NEW:BOOL=' | cut -d'=' -f2) == "TRUE" ]; then
    local has_valid_target="FALSE"
    for UT_TARGET in ${UT_TARGETS[@]} ; do
        if cmake --build . --target help | grep -w "$UT_TARGET"; then
            echo "Building target: $UT_TARGET."
            if ! cmake --build . --target ${UT_TARGET} -j $CORE_NUMS; then
                echo "[ERROR] Build failed for target: $UT_TARGET."
                exit 1
            fi
            has_valid_target="TRUE"
        else
            echo "Target $UT_TARGET not found, skipping build." 
        fi
    done
    if [[ "$COV" == "true" && "$ENABLE_UT_EXEC" == "TRUE" && "$has_valid_target" == "TRUE" ]]; then
        python3 ${BASE_PATH}/cmake/scripts/utest/gen_coverage.py \
            -s=${BASE_PATH} \
            -c=${BUILD_PATH} \
            -f="/tmp/*" \
            -f="/usr/include/*" \
            -f="$(realpath $ASCEND_HOME_PATH/../)/*" \
            -y=${BASE_PATH}/tests/test_config.yaml
        local gen_coverage_result=$?
        if [ $gen_coverage_result -ne 0 ]; then
            echo "Error: Gen coverage failed with exit code: $gen_coverage_result"
            exit $gen_coverage_result
        fi
    fi
  fi
  exit 0
}

function build_pkg_for_single_soc() {
    local single_soc_option="$1"
    local original_option="${CUSTOM_OPTION}"
    if [[ "$ENABLE_BUILT_JIT" == "TRUE" ]]; then
        CUSTOM_OPTION="${CUSTOM_OPTION}  -DENABLE_BUILT_IN=ON -DENABLE_OPS_HOST=ON -DENABLE_OPS_KERNEL=OFF"
        cmake_config ${single_soc_option}
        build_package
        CUSTOM_OPTION="${original_option}"
    elif [[ "$ENABLE_BUILT_IN" == "TRUE" ]]; then
        CUSTOM_OPTION="${CUSTOM_OPTION}  -DENABLE_BUILT_IN=ON -DENABLE_OPS_HOST=ON -DENABLE_OPS_KERNEL=ON"
        cmake_config ${single_soc_option}
        build_package
        CUSTOM_OPTION="${original_option}"
    fi
}

if [[ "$ENABLE_GENOP" == "TRUE" ]]; then
    gen_op
fi

function build_example_for_ci()
{
    EXAMPLE_NAME="$1"
    EXAMPLE_MODE="eager"
    PKG_MODE="cust"
    build_example || local eager_result=$?       # 避免函数随build_example一起退出
    if [ $eager_result -ne 0 ] && [ $eager_result -ne 2 ]; then
        echo "Error: Eager Example failed with exit code: $eager_result"
        exit $eager_result
    fi

    EXAMPLE_MODE="graph"
    build_example || local geir_result=$?
    if [ $geir_result -ne 0 ] && [ $geir_result -ne 2 ]; then
        echo "Error: Graph Example failed with exit code: $geir_result"
        exit $geir_result
    fi

    if [ $eager_result -eq 2 ] && [ $geir_result -eq 2 ]; then
        echo "Error: Neither eager nor graph examples provided for $EXAMPLE_NAME"
        exit $geir_result
    fi
    exit 0
}

# 冒烟任务只跑examples
function process_ci_smoke_with_changed_list()
{
    TEST=$(python3 "$CURRENT_DIR"/cmake/scripts/parse_changed_files.py -c "$CURRENT_DIR"/tests/test_config.yaml -f "$PR_CHANGED_FILES" get_related_examples)
    echo "Operators that need to run examples: $TEST"
    if [[ -z "$TEST" ]];then
        echo "No related unit tests found. Skipping CI test execution."
        exit 0
    fi
    IFS=';' read -ra OPS_ARRAY <<< "$TEST"
    for op in "${OPS_ARRAY[@]}";do
        op=$(echo "$op" | xargs)
        echo "Running example test for operator: $op"
        if [[ -n "$op" ]];then
            build_example_for_ci "$op"
        fi
    done
}
if [[ "$ENABLE_SMOKE" == "TRUE" ]]; then
    process_ci_smoke_with_changed_list
fi

cd ${BUILD_DIR}

if [[ "$ENABLE_RUN_EXAMPLE" == "TRUE" ]];then
    build_example
    example_result=$?
    if [ $example_result -eq 2 ]; then
        echo "Error: ${EXAMPLE_NAME} do not have ${EXAMPLE_MODE} example"
    elif [ $example_result -ne 0 ]; then
        echo "Example failed with exit code: $example_result"
    else
        echo "Example completed successfully"
    fi
    exit $example_result
fi

if [[ "$ENABLE_TEST" == "TRUE" ]]; then
    set_compute_unit_option
    build_ut ${BUILD}
elif [[ "$ENABLE_CREATE_LIB" == "TRUE" ]]; then
    build_lib
elif [[ "$ENABLE_STATIC" == "TRUE" ]]; then
    IFS=';' read -ra SOC_ARRAY <<< "$ASCEND_SOC_UNITS"  # 分割字符串为数组
    for soc in "${SOC_ARRAY[@]}"; do
        soc=$(echo "${soc}" | xargs)  # 去除前后空格
        if [[ -n "${soc}" ]]; then  # 检查非空
            if [[ "$ENABLE_BUILT_JIT" == "FALSE" ]]; then
                cmake_config -DASCEND_COMPUTE_UNIT=${soc}
                build_kernel
            fi
            build_static_lib ${soc}
            if [[ "$ENABLE_BUILD_PKG" == "TRUE" ]]; then
                package_static ${soc}
            fi
        fi
        make clean
    done
elif [[ "$ENABLE_OPKERNEL" == "TRUE" ]]; then
    set_compute_unit_option
    cmake_config -DENABLE_HOST_TILING=ON
    build_kernel
elif [[ "$ENABLE_BUILT_CUSTOM" == "TRUE" ]]; then      # --ops, --vendor 新命令新使用
    set_compute_unit_option
    if [[ "$ENABLE_BUILT_JIT" == "TRUE" ]]; then
        ops_kernel_value="OFF"
    else
        ops_kernel_value="ON"
    fi
    CUSTOM_OPTION="${CUSTOM_OPTION}  -DENABLE_BUILT_IN=OFF -DENABLE_OPS_HOST=ON -DENABLE_OPS_KERNEL=${ops_kernel_value}"
    if [[ "$ENABLE_BUILD_PKG" == "TRUE" ]]; then      # --pkg 新命令新使用
        cmake_config " -DENABLE_BUILD_PKG=ON"
    else
        cmake_config " -DENABLE_BUILD_PKG=OFF"
    fi
    build_package
elif [[ "$ENABLE_BUILD_PKG" == "TRUE" ]]; then      # --pkg 新命令新使用
    IFS=';' read -ra SOC_ARRAY <<< "$ASCEND_SOC_UNITS"  # 分割字符串为数组
    CUSTOM_OPTION="${CUSTOM_OPTION} -DENABLE_BUILD_PKG=ON"
    for soc in "${SOC_ARRAY[@]}"; do
        soc=$(echo "${soc}" | xargs)  # 去除前后空格
        soc_options=" -DASCEND_COMPUTE_UNIT=${soc}"
        if [[ -n "${soc}" ]]; then  # 检查非空
            build_pkg_for_single_soc ${soc_options} && make clean
        fi
    done
else
    CUSTOM_OPTION="${CUSTOM_OPTION} -DENABLE_BUILD_PKG=ON"
    if [ "${BUILD}" == "host" ];then
        cmake_config -DENABLE_OPS_KERNEL=OFF -DENABLE_OPS_HOST=ON
        build_host
        # TO DO
        rm -rf ${CURRENT_DIR}/output
        mkdir -p ${CURRENT_DIR}/output
        cp ${BUILD_DIR}/*.run ${CURRENT_DIR}/output
    elif [ "${BUILD}" == "kernel" ];then
        CUSTOM_OPTION="${CUSTOM_OPTION} -DENABLE_OPS_HOST=OFF -DENABLE_OPS_KERNEL=ON -DBUILD_OPS_RTY_KERNEL=ON"
        cmake_config 
        build_kernel
    elif [ "${BUILD}" == "package" ];then
        CUSTOM_OPTION="${CUSTOM_OPTION}  -DENABLE_BUILT_IN=ON -DENABLE_OPS_HOST=ON -DENABLE_OPS_KERNEL=ON"
        build_package
    elif [ -n "${BUILD}" ];then
        CUSTOM_OPTION="${CUSTOM_OPTION}  -DENABLE_OPS_HOST=ON -DENABLE_OPS_KERNEL=ON"
        cmake_config
        build ${BUILD}
    fi
fi
} | awk '{print strftime("[%Y-%m-%d %H:%M:%S]"), $0}'