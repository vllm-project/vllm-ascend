FROM quay.io/ascend/cann:8.0.rc3.beta1-910b-ubuntu22.04-py3.10

# Define environments
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && \
    apt-get install -y python3-pip git vim \
    gcc-12 g++-12 libnuma-dev

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 10 --slave /usr/bin/g++ g++ /usr/bin/g++-12

WORKDIR /workspace

COPY . /workspace/vllm_ascend/

RUN pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

RUN pip install cmake>=3.26 wheel packaging ninja "setuptools-scm>=8" numpy

# install build requirements
RUN PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu" python3 -m pip install -r /workspace/vllm_ascend//vllm/requirements-build.txt
# build vLLM with NPU backend
RUN PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu" VLLM_TARGET_DEVICE="cpu" python3 -m pip install /workspace/vllm_ascend/vllm/
# install vllm_ascend
RUN python3 -m pip install /workspace/vllm_ascend/

CMD ["/bin/bash"]
