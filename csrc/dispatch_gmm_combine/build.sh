#!/bin/bash

ASCEND_CANN_PACKAGE_PATH=${ASCEND_CANN_PACKAGE_PATH:-/usr/local/Ascend/ascend-toolkit/latest}
ASCEND_TOOLKIT_ARCH_INCLUDE=${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/include/
ASCEND_HIGH_LV_API=${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/include/ascendc/highlevel_api/lib/

export CPATH=${ASCEND_TOOLKIT_ARCH_INCLUDE}/experiment/metadef/:${ASCEND_HIGH_LV_API}/:${CPATH}
export OPS_PROJECT_NAME=aclnnInner

SCRIPTS_DIR=$(cd "$(dirname "$0")" && pwd)
export CPATH=${SCRIPTS_DIR}/utils/op_kernel/operator/catlass:${SCRIPTS_DIR}/utils/op_kernel/operator:${CPATH}

if [ -n "$BASE_LIBS_PATH" ]; then
  export ASCEND_HOME_PATH="$BASE_LIBS_PATH"
elif [ -z "$ASCEND_HOME_PATH" ]; then
  if [ -n "$ASCEND_AICPU_PATH" ]; then
    export ASCEND_HOME_PATH="$ASCEND_AICPU_PATH"
  else
    echo "please set env." >&2
    exit 1
  fi
fi
echo "using ASCEND_HOME_PATH: $ASCEND_HOME_PATH"
script_path=$(realpath $(dirname $0))

BUILD_DIR="build_out"
HOST_NATIVE_DIR="host_native_tiling"

chmod +x cmake/util/gen_ops_filter.sh
mkdir -p build_out
rm -rf build_out/*

opts=$(python3 $script_path/cmake/util/preset_parse.py $script_path/CMakePresets.json)
ENABLE_CROSS="-DENABLE_CROSS_COMPILE=True"
ENABLE_BINARY="-DENABLE_BINARY_PACKAGE=True"
ENABLE_LIBRARY="-DASCEND_PACK_SHARED_LIBRARY=True"
cmake_version=$(cmake --version | grep "cmake version" | awk '{print $3}')

target=package
if [ -n "$1" ]; then target="$1"; fi
if [[ $opts =~ $ENABLE_LIBRARY ]]; then target=install; fi

if [[ $opts =~ $ENABLE_CROSS ]] && [[ $opts =~ $ENABLE_BINARY ]]
then
  if [ "$cmake_version" \< "3.19.0" ] ; then
    cmake -S . -B "$BUILD_DIR" $opts -DENABLE_CROSS_COMPILE=0
  else
    cmake -S . -B "$BUILD_DIR" --preset=default -DENABLE_CROSS_COMPILE=0
  fi
  cmake --build "$BUILD_DIR" --target cust_optiling
  mkdir $BUILD_DIR/$HOST_NATIVE_DIR
  lib_path=$(find "$BUILD_DIR" -name "libcust_opmaster_rt2.0.so")
  if [ -z "$lib_path" ] || [ $(echo "$lib_path" | wc -l) -ne 1 ]; then
    echo "Error: Expected to find exactly one libcust_opmaster_rt2.0.so, but found none or multiple." >&2
    exit 1
  fi
  mv "$lib_path" "$BUILD_DIR/$HOST_NATIVE_DIR/"
  find "$BUILD_DIR" -mindepth 1 -maxdepth 1 ! -name "$HOST_NATIVE_DIR" -exec rm -rf {} +
  host_native_tiling_lib=$(realpath $(find $BUILD_DIR -type f -name "libcust_opmaster_rt2.0.so"))
  if [ "$cmake_version" \< "3.19.0" ] ; then
    cmake -S . -B "$BUILD_DIR" $opts -DHOST_NATIVE_TILING_LIB=$host_native_tiling_lib
  else
    cmake -S . -B "$BUILD_DIR" --preset=default -DHOST_NATIVE_TILING_LIB=$host_native_tiling_lib
  fi
  cmake --build "$BUILD_DIR" --target binary -j$(nproc)
  cmake --build "$BUILD_DIR" --target $target -j$(nproc)
else
  if [ "$cmake_version" \< "3.19.0" ] ; then
    cmake -S . -B "$BUILD_DIR" $opts
  else
      cmake -S . -B "$BUILD_DIR" --preset=default
  fi
  cmake --build "$BUILD_DIR" --target binary -j$(nproc)
  cmake --build "$BUILD_DIR" --target $target -j$(nproc)
fi
