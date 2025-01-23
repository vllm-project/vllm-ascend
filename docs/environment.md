### Prepare Ascend NPU environment

### Dependencies
| Requirement  | Supported version | Recommended version | Note |
| ------------ | ------- | ----------- | ----------- | 
| Python | >= 3.9 | [3.10](https://www.python.org/downloads/) | Required for vllm |
| CANN         | >= 8.0.RC2 | [8.0.RC3](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.0.0.beta1) | Required for vllm-ascend and torch-npu |
| torch-npu    | >= 2.4.0   | [2.5.1rc1](https://gitee.com/ascend/pytorch/releases/tag/v6.0.0.alpha001-pytorch2.5.1)    | Required for vllm-ascend |
| torch        | >= 2.4.0   | [2.5.1](https://github.com/pytorch/pytorch/releases/tag/v2.5.1)      | Required for torch-npu and vllm required |


Below is a quick note to install recommended version software:

#### Containerized installation

You can use the [container image](https://hub.docker.com/r/ascendai/cann) directly with one line command:

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

#### Manual installation

Or follow the instructions provided in the [Ascend Installation Guide](https://ascend.github.io/docs/sources/ascend/quick_install.html) to set up the environment.

