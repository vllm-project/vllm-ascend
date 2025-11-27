#!/bin/bash
SCRIPT_PATH=$(cd "$(dirname "$0")" && pwd)/$(basename "$0")
export ROOT_PATH=$(dirname "$SCRIPT_PATH")
echo ROOT_PATH: $ROOT_PATH
if [ ! -d "./build_out" ]; then
    mkdir build_out
fi
export SRC_PATH="${ROOT_PATH}"
export BUILD_OUT_PATH="${ROOT_PATH}/build_out"
export SCRIPTS_PATH="${ROOT_PATH}/scripts"

export BUILD_TYPE="Release"
MODULE_NAME="all"
MODULE_BUILD_ARG=""
IS_MODULE_EXIST=0

function PrintHelp() {
    echo "
    ./build.sh [module name] <opt>...
    If there are no parameters, all modules are compiled in default mode
    module list: [custom_ops]

    opt:
    -d: Enable debug
    "
}

function ProcessArg() {
    while getopts "dh" opt; do
        case $opt in
        d)
            export BUILD_TYPE="Debug"
            ;;
        h)
            PrintHelp
            exit 0
            ;;
        esac
    done
    shift $(($OPTIND-1))
}

function IsModuleName() {
    if [ -z "$1" ]; then
        return 1
    fi

    if [[ $1 == -* ]]; then
        return 1
    else
        return 0
    fi
}

if IsModuleName $@; then
    MODULE_NAME=$1
    shift
else
    ProcessArg $@
fi

if [[ "$MODULE_NAME" == "all" || "$MODULE_NAME" == "custom_ops" ]]; then
    IS_MODULE_EXIST=1
    echo "./scripts/build.sh $@"
    ./scripts/build.sh $@
    if [ $? -ne 0 ]; then
        exit 1
    fi
fi

if [ $IS_MODULE_EXIST -eq 0 ]; then
    echo "module not exist"
fi