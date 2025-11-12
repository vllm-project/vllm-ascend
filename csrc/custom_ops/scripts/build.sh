#!/bin/bash
export MODULE_NAME="custom_ops"
export MODULE_SRC_PATH="${SRC_PATH}"
export MODULE_SCRIPTS_PATH="${SCRIPTS_PATH}/"
export MODULE_BUILD_OUT_PATH="${BUILD_OUT_PATH}/${MODULE_NAME}"
IS_EXTRACT=0
SOC_VERSION="all"
ENABLE_SRC_BUILD=1

PrintHelp() {
    echo "
    ./build.sh custom_ops <opt>...
    -x Extract the run package
    -c Target SOC VERSION
    Support Soc: [ascend910_93, ascend910b4]
    -d Enable debug
    -r Enable code coverage
    "
}

while getopts "c:xdh" opt; do
    case $opt in
    c)
        SOC_VERSION=$OPTARG
        ;;
    x)
        IS_EXTRACT=1
        ;;
    d)
        export BUILD_TYPE="Debug"
        ;;
    h)
        PrintHelp
        exit 0
        ;;
    esac
done

if [ ! -d "$BUILD_OUT_PATH/${MODULE_NAME}" ]; then
    mkdir $BUILD_OUT_PATH/${MODULE_NAME}
fi

# 目前whl包和UT的编译暂时需要先将CAM算子包并安装到环境
# 在编译whl包和UT时屏蔽算子包编译，加快编译速度
if [ $ENABLE_SRC_BUILD -eq 1 ]; then

    if [ ! -d "./build_out/custom_ops/run/" ]; then
        mkdir ${MODULE_BUILD_OUT_PATH}/run
    fi
    if [[ "$SOC_VERSION" == "all" ]]; then
        bash $MODULE_SCRIPTS_PATH/compile_ascend_proj.sh $MODULE_SRC_PATH ascend910_93 $IS_EXTRACT $BUILD_TYPE
    else
        bash $MODULE_SCRIPTS_PATH/compile_ascend_proj.sh $MODULE_SRC_PATH $SOC_VERSION $IS_EXTRACT $BUILD_TYPE
    fi
    if [ $? -ne 0 ]; then
        exit 1
    fi
fi