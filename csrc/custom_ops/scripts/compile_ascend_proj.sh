#!/bin/bash
CopyOps() {
    local src_dir="$1" # 源目录
    local dst_dir="$2" # 目标目录

    # 确保目标目录的ophost和opkernel存在
    mkdir -p "$dst_dir/op_host" "$dst_dir/op_kernel"

    # 遍历源目录下所有直接子目录 （包括含空格的目录）
    find "$src_dir" -mindepth 1 -maxdepth 1 -type d -print0 | while IFS= read -r -d '' subdir; do
        # 检查子目录是否存在（双重验证）
        if [ -d "$subdir" ]; then
            # 处理op_host目录
            if [ -d "$subdir/op_host" ]; then
                cp -rf "$subdir/op_host/"* "$dst_dir/op_host/"
            fi

            # 处理op_kernel目录
            if [ -d "$subdir/op_kernel" ]; then
                cp -rf "$subdir/op_kernel/"* "$dst_dir/op_kernel/"
            fi
        fi
    done
}

# 构建算子工程并将其产物传到指定地点
BuildAscendProj() {
    local os_id=$(grep ^ID= /etc/os-release | cut -d= -f2 | tr -d '"')
    local arch=$(uname -m)
    local soc_version=$2
    local is_extract=$3
    local build_type=$4
    local proj_name="kernels_${soc_version}_proj"
    # 修改默认算子名
    export OPS_PROJECT_NAME=aclnnInner
    # 进入编译路径
    cd $1

    if [ -d "./${proj_name}" ]; then
        rm -rf ${proj_name}
    fi
    echo "msopgen gen -i ./kernels/AddCustom.json -c ai_core-${soc_version} -f pytorch -lan cpp -out ${proj_name}"
    msopgen gen -i ./kernels/AddCustom.json -c ai_core-${soc_version} -f pytorch -lan cpp -out ${proj_name}
    rm -rf ./${proj_name}/op_host/add_custom*
    rm  -rf ./${proj_name}/op_kernel/add_custom*
    CopyOps "./kernels" "./${proj_name}"
    python $SCRIPTS_PATH/set_conf.py ./${proj_name}/CMakePresets.json $build_type True CAM
    cp -rf ./kernels/pregen ./${proj_name}

    source $ASCEND_HOME_PATH/bin/setenv.bash
    cd ${proj_name}
    ./build.sh
    # 根据is_extract判断是否抽取run包
    if [ $is_extract -eq 1 ]; then
        if [ ! -d "$BUILD_OUT_PATH/custom_ops/extract" ]; then
            mkdir -p "$BUILD_OUT_PATH/custom_ops/extract"
        fi
        mkdir ${BUILD_OUT_PATH}/custom_ops/extract/${soc_version}
        build_out/*.run --extract=${BUILD_OUT_PATH}/custom_ops/extract/${soc_version}
    else
        cp build_out/*.run ${BUILD_OUT_PATH}/custom_ops/run/CANN_${soc_version}_${os_id}_${arch}.run
    fi
}

BuildAscendProj $1 $2 $3 $4