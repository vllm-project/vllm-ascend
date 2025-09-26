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

IMAGES_ID=$1
TAG=$2
if [ $# -ne 2 ]; then
    echo "error: need one argument describing your container name."
    exit 1
fi

cp -r /nfs/build_image/ascend_pkg .
cp -r /nfs/build_image/python_pkg .

dockerfile_path=Dockerfile.openEuler.py311.CANNdev.910c
docker build -f ${dockerfile_path} -t ${IMAGES_ID}:${TAG} .
# shell中执行
docker login registry-cbu.huawei.com
docker push ${IMAGES_ID}:${TAG}

rm -rf ./ascend_pkg ./python_pkg