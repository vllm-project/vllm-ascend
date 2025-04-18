#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

FROM quay.io/ascend/cann:8.0.0-910b-ubuntu22.04-py3.10

ARG PIP_INDEX_URL="https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"
ARG COMPILE_CUSTOM_KERNELS=1

# Define environments
ENV DEBIAN_FRONTEND=noninteractive
ENV COMPILE_CUSTOM_KERNELS=${COMPILE_CUSTOM_KERNELS}

RUN apt-get update -y && \
    apt-get install -y python3-pip git vim wget net-tools gcc g++ cmake libnuma-dev && \
    rm -rf /var/cache/apt/* && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY . /workspace/vllm-ascend/

RUN pip config set global.index-url ${PIP_INDEX_URL}

# Install vLLM
ARG VLLM_REPO=https://github.com/vllm-project/vllm.git
ARG VLLM_TAG=v0.8.4
RUN git clone --depth 1 $VLLM_REPO --branch $VLLM_TAG /workspace/vllm
RUN VLLM_TARGET_DEVICE="empty" python3 -m pip install /workspace/vllm/ --extra-index https://download.pytorch.org/whl/cpu/
# In x86, triton will be installed by vllm. But in Ascend, triton doesn't work correctly. we need to uninstall it.
RUN python3 -m pip uninstall -y triton

# Install torch-npu
RUN bash /workspace/vllm-ascend/pta_install.sh

# Install vllm-ascend
RUN source /usr/local/Ascend/ascend-toolkit/set_env.sh && \
    source /usr/local/Ascend/nnal/atb/set_env.sh && \
    export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/`uname -i`-linux/devlib:$LD_LIBRARY_PATH && \
    export LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/lib64:$LIBRARY_PATH && \
    python3 -m pip install -v /workspace/vllm-ascend/ --extra-index https://download.pytorch.org/whl/cpu/

# Install modelscope (for fast download) and ray (for multinode)
RUN python3 -m pip install modelscope ray

CMD ["/bin/bash"]
