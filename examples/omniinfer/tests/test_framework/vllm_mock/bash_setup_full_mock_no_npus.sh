#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

set -e

python -m venv ~/venv_omni_infer
source ~/venv_omni_infer/bin/activate

wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.1.RC1/Ascend-cann-toolkit_8.1.RC1_linux-x86_64.run --no-check-certificate -nc
wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.1.RC1/Ascend-cann-nnal_8.1.RC1_linux-x86_64.run --no-check-certificate -nc
wget https://repo.mindspore.cn/CANN/ascend910/20250508/Ascend910B/x86_64/Ascend-hdk-910b-npu-driver_25.0.rc1.1_linux-x86-64.run --no-check-certificate -nc
chmod +x Ascend*
./Ascend-cann-toolkit_8.1.RC1_linux-x86_64.run --install --quiet
source /usr/local/Ascend/ascend-toolkit/set_env.sh
./Ascend-cann-nnal_8.1.RC1_linux-x86_64.run --install --quiet
source /usr/local/Ascend/nnal/atb/set_env.sh
./Ascend-hdk-910b-npu-driver_25.0.rc1.1_linux-x86-64.run --full

pip install setuptools wheel pip -U

git clone --recurse-submodules -b omni_infer_v1 ssh://git@codehub-dg-g.huawei.com:2222/DataScience/omni_infer.git
cd omni_infer/infer_engines/
sh bash_install_code.sh
cd vllm
SETUPTOOLS_SCM_PRETEND_VERSION=0.9.0 VLLM_TARGET_DEVICE=empty pip install -e .
cd ../..
pip3 install torch==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu
pip3 install torch-npu==2.5.1
pip install -e .