#!/bin/bash

START_DIR=$PWD
SOURCE_ROOT="$(dirname "$(dirname "$(realpath "$0")")")"

# 执行 build.sh
bash "${SOURCE_ROOT}/omni/accelerators/placement/build.sh" "-ut"