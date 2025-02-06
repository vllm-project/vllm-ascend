### 昇腾NPU环境准备

### 依赖
| 需求 | 支持的版本 | 推荐版本 | 注意                                     |
|-------------|-------------------| ----------- |------------------------------------------|
| vLLM        | main              | main |  vllm-ascend 依赖                 |
| Python      | >= 3.9            | [3.10](https://www.python.org/downloads/) |  vllm 依赖                       |
| CANN        | >= 8.0.RC2        | [8.0.RC3](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.0.0.beta1) |  vllm-ascend and torch-npu 依赖  |
| torch-npu   | >= 2.4.0          | [2.5.1rc1](https://gitee.com/ascend/pytorch/releases/tag/v6.0.0.alpha001-pytorch2.5.1)    | vllm-ascend 依赖                |
| torch       | >= 2.4.0          | [2.5.1](https://github.com/pytorch/pytorch/releases/tag/v2.5.1)      |  torch-npu and vllm 依赖 |


以下为安装推荐版本软件的简短说明：

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

您无需手动安装 `torch` 和 `torch_npu` ，它们将作为 `vllm-ascend` 依赖项自动安装。

#### 手动安装

您也可以选择手动安装，按照[昇腾安装指南](https://ascend.github.io/docs/sources/ascend/quick_install.html)中提供的说明配置环境。
