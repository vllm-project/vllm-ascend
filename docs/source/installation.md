# Installation

This document describes how to install vllm-ascend manually.


## Requirements

- OS: Linux
- Python: 3.10 or higher
- A hardware with Ascend NPU. It's usually the Atlas 800 A2 series.
- Software:

    | Software     | Supported version | Note |
    | ------------ | ----------------- | ---- | 
    | CANN         | >= 8.0.0.beta1    | Required for vllm-ascend and torch-npu |
    | torch-npu    | >= 2.5.1rc1       | Required for vllm-ascend |
    | torch        | >= 2.5.1          | Required for torch-npu and vllm |


You have 2 way to install:
- **Using pip**: first prepare env manually or via CANN image, then install `vllm-ascend` using pip.
- **Using docker**: use the `vllm-ascend` pre-built docker image directly.

## Configure a new environment

Before installing, you need to make sure firmware/driver and CANN is installed correctly, refer to [link](https://ascend.github.io/docs/sources/ascend/quick_install.html) for more details.

### Configure hardware environment

To verify that the Ascend NPU firmware and driver were correctly installed, run:

```bash
npu-smi info
```



### Configure software environment

:::::{tab-set}
:sync-group: install

::::{tab-item} Before using pip
:selected:
:sync: pip

The easiest way to prepare your software environment is using CANN image directly:

```bash
# Update DEVICE according to your device (/dev/davinci[0-7])
DEVICE=/dev/davinci7

docker run --rm \
    --name vllm-ascend-env \
    --device $DEVICE \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -it quay.io/ascend/cann:8.0.0.beta1-910b-ubuntu22.04-py3.10 bash
```

You can also install CANN manually:

```bash
# Create a virtual environment
python -m venv vllm-ascend-env
source vllm-ascend-env/bin/activate

# Install required python packages.
pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple attrs numpy==1.24.0 decorator sympy cffi pyyaml pathlib2 psutil protobuf scipy requests absl-py wheel typing_extensions

# Download and install the CANN package.
wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.0.0/Ascend-cann-toolkit_8.0.0_linux-aarch64.run
sh Ascend-cann-toolkit_8.0.0_linux-aarch64.run --full
wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.0.0/Ascend-cann-kernels-910b_8.0.0_linux-aarch64.run
sh Ascend-cann-kernels-910b_8.0.0_linux-aarch64.run --full
```

::::

::::{tab-item} Before using docker
:sync: docker
No more extra step if you are using `vllm-ascend` prebuilt docker image.
::::
:::::

Once it's done, you can start to set up `vllm` and `vllm-ascend`.

## Setup vllm and vllm-ascend

:::::{tab-set}
:sync-group: install

::::{tab-item} Using pip
:selected:
:sync: pip

You can install `vllm` and `vllm-ascend` from **pre-built wheel**:

```bash
pip install vllm vllm-ascend -f https://download.pytorch.org/whl/torch/
```

or build from **source code**:

```bash
git clone https://github.com/vllm-project/vllm
cd vllm
VLLM_TARGET_DEVICE=empty pip install . -f https://download.pytorch.org/whl/torch/

git clone https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
pip install -e . -f https://download.pytorch.org/whl/torch/
```

::::

::::{tab-item} Using docker
:sync: docker

You can just pull the **prebuilt image** and run it with bash.

```bash


# Update DEVICE according to your device (/dev/davinci[0-7])
DEVICE=/dev/davinci7
# Update the vllm-ascend image
IMAGE=quay.io/ascend/vllm-ascend:main
docker pull $IMAGE
docker run --rm \
    --name vllm-ascend-env \
    --device $DEVICE \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -it $IMAGE bash
```

or build IMAGE from **source code**:

```bash
git clone https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
docker build -t vllm-ascend-dev-image:latest -f ./Dockerfile .
```

::::

:::::

## Extra information

### Verify installation

Create and run a simple inference test. The `example.py` can be like:

```python
from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

# Create a sampling params object.
sampling_params = SamplingParams(max_tokens=100, temperature=0.0)
# Create an LLM.
llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct")

# Generate texts from the prompts.
outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

Then run:

```bash
# export VLLM_USE_MODELSCOPE=true to speed up download if huggingface is not reachable.
python example.py
```
