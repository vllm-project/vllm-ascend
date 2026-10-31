# LMCache-Ascend Deployment Guide

## Overview

LMCache-Ascend is a community maintained plugin for running LMCache on the Ascend NPU.

We provide a simple deployment guide here. For further info about deployment notes, please refer to [LMCache-Ascend doc](https://github.com/LMCache/LMCache-Ascend/blob/main/README.md)

## Getting Started

### Clone LMCache-Ascend Repo

Our repo contains a kvcache ops submodule for ease of maintenance, therefore we recommend cloning the repo with submodules.

```bash
cd /workspace
git clone --recurse-submodules https://github.com/LMCache/LMCache-Ascend.git
```

### Docker

```bash
cd /workspace/LMCache-Ascend
docker build -f docker/Dockerfile.a2.openEuler -t lmcache-ascend:v0.3.12-vllm-ascend-v0.11.0-openeuler .
```

Once that is built, run it with the following cmd

```bash
DEVICE_LIST="0,1,2,3,4,5,6,7"
docker run -it \
    --privileged \
    --cap-add=SYS_RESOURCE \
    --cap-add=IPC_LOCK \
    -p 8000:8000 \
    -p 8001:8001 \
    --name lmcache-ascend-dev \
    -e ASCEND_RT_VISIBLE_DEVICES=${DEVICE_LIST} \
    -e ASCEND_TOTAL_MEMORY_GB=32 \
    -e VLLM_TARGET_DEVICE=npu \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /etc/localtime:/etc/localtime \
    -v /var/log/npu:/var/log/npu \
    -v /dev/davinci_manager:/dev/davinci_manager \
    -v /dev/devmm_svm:/dev/devmm_svm \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /etc/hccn.conf:/etc/hccn.conf \
    lmcache-ascend:v0.3.12-vllm-ascend-v0.11.0-openeuler \
    /bin/bash
```

### Manual Installation

Assuming your working directory is `/workspace` and vllm/vllm-ascend have already been installed.

1. Install LMCache Repo

    ```bash
    NO_CUDA_EXT=1 pip install lmcache==0.3.12
    ```

2. Install LMCache-Ascend Repo

    ```bash
    cd /workspace/LMCache-Ascend
    python3 -m pip install --no-build-isolation -e .
    ```

### Usage

`LMCacheAscendConnector` is no longer registered by vllm-ascend. Use the connector registration and runtime configuration provided by the LMCache-Ascend package instead. For the current configuration examples, refer to the [LMCache-Ascend documentation](https://github.com/LMCache/LMCache-Ascend/blob/main/README.md).
