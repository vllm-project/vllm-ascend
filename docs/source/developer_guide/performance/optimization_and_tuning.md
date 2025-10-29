# Optimization and Tuning

This guide aims to help users to improve vllm-ascend performance on system level. It includes OS configuration, library optimization, deployment guide and so on. Any feedback is welcome.

## Preparation

Run the container:

```{code-block} bash
   :substitutions:
# Update DEVICE according to your device (/dev/davinci[0-7])
export DEVICE=/dev/davinci0
# Update the cann base image
export IMAGE=m.daocloud.io/quay.io/ascend/cann:|cann_image_tag|
docker run --rm \
--name performance-test \
--device $DEVICE \
--device /dev/davinci_manager \
--device /dev/devmm_svm \
--device /dev/hisi_hdc \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /root/.cache:/root/.cache \
-it $IMAGE bash
```

Configure your environment:

```{code-block} bash
   :substitutions:
# Configure the mirror
echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ jammy main restricted universe multiverse" > /etc/apt/sources.list && \
echo "deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ jammy main restricted universe multiverse" >> /etc/apt/sources.list && \
echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ jammy-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
echo "deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ jammy-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ jammy-backports main restricted universe multiverse" >> /etc/apt/sources.list && \
echo "deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ jammy-backports main restricted universe multiverse" >> /etc/apt/sources.list && \
echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ jammy-security main restricted universe multiverse" >> /etc/apt/sources.list && \
echo "deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ jammy-security main restricted universe multiverse" >> /etc/apt/sources.list

# Install os packages
apt update && apt install wget gcc g++ libnuma-dev git vim -y
```

Install vllm and vllm-ascend:

```{code-block} bash
   :substitutions:
# Install necessary dependencies
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install modelscope pandas datasets gevent sacrebleu rouge_score pybind11 pytest

# Configure this var to speed up model download
VLLM_USE_MODELSCOPE=true
```

Please follow the [Installation Guide](https://vllm-ascend.readthedocs.io/en/latest/installation.html) to make sure vLLM, vllm-ascend, and MindIE Turbo are installed correctly.

:::{note}
Make sure your vLLM and vllm-ascend are installed after your python configuration is completed, because these packages will build binary files using python in current environment. If you install vLLM, vllm-ascend, and MindIE Turbo before completing section 1.1, the binary files will not use the optimized python.
:::

## Optimizations

### 1. Compilation Optimization

#### 1.1. Install optimized `python`

Python supports **LTO** and **PGO** optimization starting from version `3.6` and above, which can be enabled at compile time. And we have offered optimized `python` packages directly to users for the sake of convenience. You can also reproduce the `python` built following this [tutorial](https://www.hiascend.com/document/detail/zh/Pytorch/600/ptmoddevg/trainingmigrguide/performance_tuning_0063.html) according to your specific scenarios.

```{code-block} bash
   :substitutions:
mkdir -p /workspace/tmp
cd /workspace/tmp

# Download prebuilt lib and packages
wget https://repo.oepkgs.net/ascend/pytorch/vllm/lib/libcrypto.so.1.1
wget https://repo.oepkgs.net/ascend/pytorch/vllm/lib/libomp.so
wget https://repo.oepkgs.net/ascend/pytorch/vllm/lib/libssl.so.1.1
wget https://repo.oepkgs.net/ascend/pytorch/vllm/python/py311_bisheng.tar.gz

# Configure python and pip
cp ./*.so* /usr/local/lib
tar -zxvf ./py311_bisheng.*  -C /usr/local/
mv  /usr/local/py311_bisheng/  /usr/local/python
sed -i "1c#\!/usr/local/python/bin/python3.11" /usr/local/python/bin/pip3
sed -i "1c#\!/usr/local/python/bin/python3.11" /usr/local/python/bin/pip3.11
ln -sf  /usr/local/python/bin/python3  /usr/bin/python
ln -sf  /usr/local/python/bin/python3  /usr/bin/python3
ln -sf  /usr/local/python/bin/python3.11  /usr/bin/python3.11
ln -sf  /usr/local/python/bin/pip3  /usr/bin/pip3
ln -sf  /usr/local/python/bin/pip3  /usr/bin/pip

export PATH=/usr/bin:/usr/local/python/bin:$PATH
```

### 2. OS Optimization

#### 2.1. jemalloc

**jemalloc** is a memory allocator that improves performance for multi-thread scenarios and can reduce memory fragmentation. jemalloc uses local thread memory manager to allocate variables, which can avoid lock competition between threads and can hugely optimize performance.

```{code-block} bash
   :substitutions:
# Install jemalloc
sudo apt update
sudo apt install libjemalloc2

# Configure jemalloc
export LD_PRELOAD=/usr/lib/"$(uname -i)"-linux-gnu/libjemalloc.so.2 $LD_PRELOAD
```

#### 2.2. Tcmalloc

**Tcmalloc (Thread Counting Malloc)** is a universal memory allocator that improves overall performance while ensuring low latency by introducing a multi-level cache structure, reducing mutex competition and optimizing large object processing flow. Find more details [here](https://www.hiascend.com/document/detail/zh/Pytorch/700/ptmoddevg/trainingmigrguide/performance_tuning_0068.html).

```{code-block} bash
   :substitutions:
# Install tcmalloc
sudo apt update
sudo apt install libgoogle-perftools4 libgoogle-perftools-dev

# Get the location of libtcmalloc.so*
find /usr -name libtcmalloc.so*

# Make the priority of tcmalloc higher
# The <path> is the location of libtcmalloc.so we get from the upper command
# Example: "$LD_PRELOAD:/usr/lib/aarch64-linux-gnu/libtcmalloc.so"
export LD_PRELOAD="$LD_PRELOAD:<path>"

# Verify your configuration
# The path of libtcmalloc.so will be contained in the result if your configuration is valid
ldd `which python`
```

### 3. `torch_npu` Optimization

Some performance tuning features in `torch_npu` are controlled by environment variables. Some features and their related environment variables are shown below.

Memory optimization:

```{code-block} bash
   :substitutions:
# Upper limit of memory block splitting allowed (MB): Setting this parameter can prevent large memory blocks from being split.
export PYTORCH_NPU_ALLOC_CONF="max_split_size_mb:250"

# When operators on the communication stream have dependencies, they all need to be ended before being released for reuse. The logic of multi-stream reuse is to release the memory on the communication stream in advance so that the computing stream can be reused.
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
```

Scheduling optimization:

```{code-block} bash
   :substitutions:
# Optimize operator delivery queue. This will affect the memory peak value, and may degrade if the memory is tight.
export TASK_QUEUE_ENABLE=2

# This will greatly improve the CPU bottleneck model and ensure the same performance for the NPU bottleneck model.
export CPU_AFFINITY_CONF=1
```

### 4. CANN Optimization

#### 4.1. HCCL Optimization

There are some performance tuning features in HCCL, which are controlled by environment variables.

You can configure HCCL to use "AIV" mode to optimize performance by setting the environment variable shown below. In "AIV" mode, the communication is scheduled by AI vector core directly with RoCE, instead of being scheduled by AI CPU.

```{code-block} bash
   :substitutions:
export HCCL_OP_EXPANSION_MODE="AIV"
```

Plus, there are more features for performance optimization in specific scenarios, which are shown below.

- `HCCL_INTRA_ROCE_ENABLE`: Use RDMA link instead of SDMA link between two 8Ps as the mesh interconnect link. Find more details [here](https://www.hiascend.com/document/detail/zh/Pytorch/600/ptmoddevg/trainingmigrguide/performance_tuning_0044.html).
- `HCCL_RDMA_TC`: Use this var to configure traffic class of RDMA NIC. Find more details [here](https://www.hiascend.com/document/detail/zh/Pytorch/600/ptmoddevg/trainingmigrguide/performance_tuning_0045.html).
- `HCCL_RDMA_SL`: Use this var to configure service level of RDMA NIC. Find more details [here](https://www.hiascend.com/document/detail/zh/Pytorch/600/ptmoddevg/trainingmigrguide/performance_tuning_0046.html).
- `HCCL_BUFFSIZE`: Use this var to control the cache size for sharing data between two NPUs. Find more details [here](https://www.hiascend.com/document/detail/zh/Pytorch/600/ptmoddevg/trainingmigrguide/performance_tuning_0047.html).
