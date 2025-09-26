#!/bin/bash
# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

# 拷贝git ssh 相关文件到当前目录，为了CI/CD git clone准备

BASE_IMAGE=$1
IMAGES_ID=$2
TAG=$3
dockerfile_path="Dockerfile.vllm-omni.source"  # Dockerfile路径

# 修改参数检查逻辑（允许3或4个参数）
if [ $# -lt 3 ] || [ $# -gt 4 ]; then
    echo "错误：需要3或4个参数"
    echo "用法：$0 BASE_IMAGE IMAGES_ID TAG [替换脚本名]"
    exit 1
fi

# 添加替换逻辑
if [ $# -eq 4 ]; then
    REPLACE_SCRIPT=$4
    # 检查目标文件是否存在
    if [ ! -f "$dockerfile_path" ]; then
        echo "错误：未找到Dockerfile文件！"
        exit 1
    fi

    # 执行字符串替换
    sed -i.bak_replace "s/bash_install_code.sh/$REPLACE_SCRIPT/g" "$dockerfile_path"
    echo "已替换安装脚本为: $REPLACE_SCRIPT"
fi

git clone --recurse-submodules -b omni_infer_v1 ssh://git@codehub-dg-g.huawei.com:2222/DataScience/omni_infer.git
git clone ssh://git@codehub-dg-g.huawei.com:2222/opensourcecenter/openeuler/nginx.git -b huawei/cbu/HCEOS/1.24.0-5.oe2403sp1

cp /nfs/build_image/pytorch_v2.5.1_py311.tar.gz .

docker build -f ${dockerfile_path} --build-arg BASE_IMAGE=${BASE_IMAGE} -t ${IMAGES_ID}:${TAG} .
# shell中执行
docker login registry-cbu.huawei.com
docker push ${IMAGES_ID}:${TAG}

rm -rf pytorch_v2.5.1_py311.tar.gz omni_infer nginx
