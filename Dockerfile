#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#

FROM quay.io/ascend/cann:8.0.0-910b-ubuntu22.04-py3.10

# Define environments
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && \
    apt-get install -y python3-pip git vim && \
    rm -rf /var/cache/apt/* && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY . /workspace/vllm-ascend/

# RUN pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# Install vLLM
ARG VLLM_REPO=https://github.com/vllm-project/vllm.git
ARG VLLM_TAG=v0.7.1
RUN git clone --depth 1 $VLLM_REPO --branch $VLLM_TAG /workspace/vllm
# Add -f to fix https://github.com/vllm-project/vllm/pull/12874
RUN VLLM_TARGET_DEVICE="empty" python3 -m pip install /workspace/vllm/ -f https://download.pytorch.org/whl/torch/

# Install vllm-ascend main
RUN python3 -m pip install /workspace/vllm-ascend/ -f https://download.pytorch.org/whl/torch/

# Install modelscope
RUN python3 -m pip install modelscope

CMD ["/bin/bash"]
