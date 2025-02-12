# 安装

### 1. 依赖
| 依赖  | 支持版本 | 推荐版本 | 请注意 |
| ------------ | ------- | ----------- | ----------- | 
| Python | >= 3.9 | [3.10](https://www.python.org/downloads/) | 安装vllm需要 |
| CANN         | >= 8.0.RC2 | [8.0.RC3](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.0.0.beta1) | 安装vllm-ascend 和 torch-npu需要 |
| torch-npu    | >= 2.4.0   | [2.5.1rc1](https://gitee.com/ascend/pytorch/releases/tag/v6.0.0.alpha001-pytorch2.5.1)    | 安装vllm-ascend需要 |
| torch        | >= 2.4.0   | [2.5.1](https://github.com/pytorch/pytorch/releases/tag/v2.5.1)      | 安装torch-npu 和 vllm需要|

### 2. NPU环境准备

以下为安装推荐版本软件的快速说明：

#### 容器化安装
您可以直接使用[容器镜像](https://hub.docker.com/r/ascendai/cann)，只需一行命令即可：

```bash
docker run \
    --name vllm-ascend-env \
    --device /dev/davinci1 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -it quay.io/ascend/cann:8.0.rc3.beta1-910b-ubuntu22.04-py3.10 bash
```

无需手动安装 torch 和 torch_npu ，它们将作为 vllm-ascend 依赖项被自动安装。

#### 手动安装

您可以按照[昇腾安装指南](https://ascend.github.io/docs/sources/ascend/quick_install.html)中提供的说明配置环境。


### 3. 构建

#### 从源码构建Python包

```bash
git clone https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
pip install -e .
```

#### 从源码构建容器镜像
```bash
git clone https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
docker build -t vllm-ascend-dev-image -f ./Dockerfile .
```
