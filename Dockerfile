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
# #

FROM quay.io/ascend/cann:9.0.0-910b-ubuntu22.04-py3.12

ARG PIP_INDEX_URL="https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"

WORKDIR /workspace

# Install clang-15 (for triton-ascend) and Mooncake
ARG MOONCAKE_TAG=0.3.11.post1
RUN apt-get update -y && \
    apt-get install -y git vim wget net-tools gcc g++ cmake numactl libnuma-dev libibverbs-dev libjemalloc2 libhiredis-dev clang-15 && \
    update-alternatives --install /usr/bin/clang clang /usr/bin/clang-15 20 && \
    update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-15 20 && \
    source /usr/local/Ascend/ascend-toolkit/set_env.sh && \
    python3 -m pip install mooncake-transfer-engine-npu==${MOONCAKE_TAG} --extra-index-url https://mirrors.aliyun.com/pypi/web/simple && \
    rm -rf /var/cache/apt/* && \
    rm -rf /var/lib/apt/lists/*

# Install modelscope (for fast download) and ray (for multinode)
RUN pip config set global.index-url ${PIP_INDEX_URL} && \
    python3 -m pip install modelscope 'ray>=2.47.1,<=2.48.0' 'protobuf>3.20.0' && \
    python3 -m pip cache purge

# Install vLLM
ARG VLLM_REPO=https://github.com/vllm-project/vllm.git
ARG VLLM_TAG=v0.23.0
ARG VLLM_COMMIT=""
RUN if [ -n "$VLLM_COMMIT" ]; then \
      git init /vllm-workspace/vllm && \
      git -C /vllm-workspace/vllm fetch --depth 1 $VLLM_REPO "$VLLM_COMMIT" && \
      git -C /vllm-workspace/vllm checkout FETCH_HEAD; \
    else \
      git clone --depth 1 -b $VLLM_TAG $VLLM_REPO /vllm-workspace/vllm; \
    fi
# In x86, triton will be installed by vllm. But in Ascend, triton doesn't work correctly. we need to uninstall it.
RUN VLLM_TARGET_DEVICE="empty" python3 -m pip install -e /vllm-workspace/vllm/[audio] --extra-index https://download.pytorch.org/whl/cpu/ && \
    python3 -m pip uninstall -y triton && \
    python3 -m pip cache purge

# Install vllm-ascend
ARG SOC_VERSION="ascend910b1"
ARG COMPILE_CUSTOM_KERNELS=1
ENV DEBIAN_FRONTEND=noninteractive
ENV SOC_VERSION=$SOC_VERSION \
    TASK_QUEUE_ENABLE=1 \
    OMP_NUM_THREADS=1
COPY . /vllm-workspace/vllm-ascend/

RUN export PIP_EXTRA_INDEX_URL="https://mirrors.huaweicloud.com/ascend/repos/pypi" && \
    export VLLM_BATCH_INVARIANT=1 && \
    source /usr/local/Ascend/ascend-toolkit/set_env.sh && \
    source /usr/local/Ascend/nnal/atb/set_env.sh && \
    python3 -m pip install -e /vllm-workspace/vllm-ascend/ --extra-index https://download.pytorch.org/whl/cpu/ && \
    python3 -m pip uninstall -y triton triton-ascend && \
    python3 -m pip install triton-ascend==3.2.1 --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi && \
    python3 -m pip cache purge

# Append `libascend_hal.so` path (devlib) to LD_LIBRARY_PATH
RUN echo "export LD_PRELOAD=/usr/lib/$(uname -m)-linux-gnu/libjemalloc.so.2:$LD_PRELOAD" >> ~/.bashrc
RUN echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib" >> ~/.bashrc

CMD ["/bin/bash"]
