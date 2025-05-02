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

FROM quay.io/ascend/cann:8.1.rc1-910b-ubuntu22.04-py3.10

ARG PIP_INDEX_URL="https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"

# Define environments
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && \
    apt-get install -y python3-pip git vim wget net-tools && \
    rm -rf /var/cache/apt/* && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY . /vllm-workspace/vllm-ascend/

RUN pip config set global.index-url ${PIP_INDEX_URL}

# Install vLLM
ARG VLLM_REPO=https://github.com/vllm-project/vllm.git
ARG VLLM_TAG=v0.7.3
RUN git clone --depth 1 $VLLM_REPO --branch $VLLM_TAG /vllm-workspace/vllm
RUN VLLM_TARGET_DEVICE="empty" python3 -m pip install -e /vllm-workspace/vllm/ --extra-index https://download.pytorch.org/whl/cpu/ && \
    python3 -m pip cache purge

# Install vllm-ascend
RUN python3 -m pip install -e /vllm-workspace/vllm-ascend/ --extra-index https://download.pytorch.org/whl/cpu/ && \
    python3 -m pip cache purge

# Install modelscope (for fast download) and ray (for multinode)
# TODO(yikun): Remove "<1.23.0" after v0.8.4 which resloved by https://github.com/vllm-project/vllm/pull/13807
RUN python3 -m pip install "modelscope<1.23.0" ray && \
    python3 -m pip cache purge

CMD ["/bin/bash"]
