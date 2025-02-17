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

## Configure a new environment

Before installing the package,  you need to make sure firmware/driver and CANN is installed correctly.

### Install firmwares and drivers

To verify that the Ascend NPU firmware and driver were correctly installed, run `npu-smi` info

> Tips: Refer to [Ascend Environment Setup Guide](https://ascend.github.io/docs/sources/ascend/quick_install.html) for more details.

### Install CANN (optional)

The installation of CANN wouldn’t be necessary if you are using a CANN container image, you can skip this step.If you want to install vllm-ascend on a bare environment by hand, you need install CANN first.

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

Once it's done, you can read either **Set up using Python** or **Set up using Docker** section to install and use vllm-ascend.

## Set up using Python

> Notes: If you are installing vllm-ascend on an arch64 machine, The `-f https://download.pytorch.org/whl/torch/` command parameter in this section can be omitted. It's only used for find torch package on x86 machine.

Please make sure that CANN is installed. It can be done by **Configure a new environment** step. Or by using an CANN container directly:

```bash
# Setup a CANN container using docker
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

Then you can install vllm-ascend from **pre-built wheel** or **source code**.

### Install from Pre-built wheels (Not support yet)

1. Install vllm

    Since vllm on pypi is not compatible with cpu, we need to install vllm from source code.

    ```bash
    git clone --depth 1 --branch v0.7.1 https://github.com/vllm-project/vllm
    cd vllm
    VLLM_TARGET_DEVICE=empty pip install . -f https://download.pytorch.org/whl/torch/
    ```

2. Install vllm-ascend

    ```bash
    pip install vllm-ascend -f https://download.pytorch.org/whl/torch/
    ```

### Install from source code

1. Install vllm

    ```bash
    git clone https://github.com/vllm-project/vllm
    cd vllm
    VLLM_TARGET_DEVICE=empty pip install . -f https://download.pytorch.org/whl/torch/
    ```

2. Install vllm-ascend

    ```bash
    git clone https://github.com/vllm-project/vllm-ascend.git
    cd vllm-ascend
    pip install -e . -f https://download.pytorch.org/whl/torch/
    ```

## Set up using Docker

> Tips: CANN, torch, torch_npu, vllm and vllm_ascend are pre-installed in the Docker image already.

### Pre-built images (Not support yet)

Just pull the image and run it with bash.

```bash
docker pull quay.io/ascend/vllm-ascend:latest

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
    -it quay.io/ascend/vllm-ascend:0.7.1rc1 bash
```

### Build image from source

If you want to build the docker image from main branch, you can do it by following steps:

```bash
git clone https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend

docker build -t vllm-ascend-dev-image:latest -f ./Dockerfile .

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
    -it vllm-ascend-dev-image:latest bash
```

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
llm = LLM(model="facebook/opt-125m")

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
