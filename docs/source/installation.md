# Installation

This document describes how to install vllm-ascend manually.

## Requirements

- OS: Linux
- Python: >= 3.9, < 3.12
- A hardware with Ascend NPU. It's usually the Atlas 800 A2 series.
- Software:

    | Software      | Supported version                | Note                                      |
    |---------------|----------------------------------|-------------------------------------------|
    | Ascend HDK    | Refer to [here](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/releasenote/releasenote_0000.html) | Required for CANN |
    | CANN          | == 8.3.RC2                       | Required for vllm-ascend and torch-npu    |
    | torch-npu     | == 2.7.1.post1             | Required for vllm-ascend, No need to install manually, it will be auto installed in below steps |
    | torch         | == 2.7.1                         | Required for torch-npu and vllm           |

There are two installation methods:
- **Using pip**: first prepare env manually or via CANN image, then install `vllm-ascend` using pip.
- **Using docker**: use the `vllm-ascend` pre-built docker image directly.

## Configure a new environment

Before installation, you need to make sure firmware/driver and CANN are installed correctly, refer to [Ascend Environment Setup Guide](https://ascend.github.io/docs/sources/ascend/quick_install.html) for more details.

### Configure hardware environment

To verify that the Ascend NPU firmware and driver were correctly installed, run:

```bash
npu-smi info
```

Refer to [Ascend Environment Setup Guide](https://ascend.github.io/docs/sources/ascend/quick_install.html) for more details.

### Configure software environment

:::::{tab-set}
:sync-group: install

::::{tab-item} Before using pip
:selected:
:sync: pip

The easiest way to prepare your software environment is using CANN image directly:

```{code-block} bash
   :substitutions:
# Update DEVICE according to your device (/dev/davinci[0-7])
export DEVICE=/dev/davinci7
# Update the vllm-ascend image
export IMAGE=quay.io/ascend/cann:|cann_image_tag|
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
    -v /root/.cache:/root/.cache \
    -it $IMAGE bash
```

:::{dropdown} Click here to see "Install CANN manually"
:animate: fade-in-slide-down
You can also install CANN manually:

```bash
# Create a virtual environment.
python -m venv vllm-ascend-env
source vllm-ascend-env/bin/activate

# Install required Python packages.
pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple attrs 'numpy<2.0.0' decorator sympy cffi pyyaml pathlib2 psutil protobuf scipy requests absl-py wheel typing_extensions

# Download and install the CANN package.
wget --header="Referer: https://www.hiascend.com/" https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.3.RC2/Ascend-cann-toolkit_8.3.RC2_linux-"$(uname -i)".run
chmod +x ./Ascend-cann-toolkit_8.3.RC2_linux-"$(uname -i)".run
./Ascend-cann-toolkit_8.3.RC2_linux-"$(uname -i)".run --full
# https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/Milan-ASL/Milan-ASL%20V100R001C22B800TP052/Ascend-cann-kernels-910b_8.3.rc2_linux-aarch64.run

source /usr/local/Ascend/ascend-toolkit/set_env.sh
wget --header="Referer: https://www.hiascend.com/" https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.3.RC2/Ascend-cann-kernels-910b_8.3.RC2_linux-"$(uname -i)".run
chmod +x ./Ascend-cann-kernels-910b_8.3.RC2_linux-"$(uname -i)".run
./Ascend-cann-kernels-910b_8.3.RC2_linux-"$(uname -i)".run --install

wget --header="Referer: https://www.hiascend.com/" https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.3.RC2/Ascend-cann-nnal_8.3.RC2_linux-"$(uname -i)".run
chmod +x ./Ascend-cann-nnal_8.3.RC2_linux-"$(uname -i)".run
./Ascend-cann-nnal_8.3.RC2_linux-"$(uname -i)".run --install

source /usr/local/Ascend/nnal/atb/set_env.sh
```

:::

::::

::::{tab-item} Before using docker
:sync: docker
No more extra step if you are using `vllm-ascend` prebuilt Docker image.
::::
:::::

Once it is done, you can start to set up `vllm` and `vllm-ascend`.

## Setup vllm and vllm-ascend

:::::{tab-set}
:sync-group: install

::::{tab-item} Using pip
:selected:
:sync: pip

First install system dependencies and configure pip mirror:

```bash
# Using apt-get with mirror
sed -i 's|ports.ubuntu.com|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list
apt-get update -y && apt-get install -y gcc g++ cmake libnuma-dev wget git curl jq
# Or using yum
# yum update -y && yum install -y gcc g++ cmake numactl-devel wget git curl jq
# Config pip mirror
pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

**[Optional]** Then configure the extra-index of `pip` if you are working on an x86 machine or using torch-npu dev version:

```bash
# For torch-npu post version or x86 machine
pip config set global.extra-index-url "https://download.pytorch.org/whl/cpu/ https://mirrors.huaweicloud.com/ascend/repos/pypi"
```

Then you can install `vllm` and `vllm-ascend` from **pre-built wheel**:

```{code-block} bash
   :substitutions:

# Install vllm-project/vllm. The newest supported version is |vllm_version|.
# Because the version |vllm_version| has not been archived in pypi, so you need to install from source.
git clone --depth 1 --branch |vllm_version| https://github.com/vllm-project/vllm
cd vllm
VLLM_TARGET_DEVICE=empty pip install -v -e .
cd ..

# Install vllm-project/vllm-ascend from pypi.
pip install vllm-ascend==|pip_vllm_ascend_version|
```

:::{dropdown} Click here to see "Build from source code"
or build from **source code**:

```{code-block} bash
   :substitutions:

# Install vLLM.
git clone --depth 1 --branch |vllm_version| https://github.com/vllm-project/vllm
cd vllm
VLLM_TARGET_DEVICE=empty pip install -v -e .
cd ..

# Install vLLM Ascend.
git clone  --depth 1 --branch |vllm_ascend_version| https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
pip install -v -e .
cd ..
```

vllm-ascend will build custom operators by default. If you don't want to build it, set `COMPILE_CUSTOM_KERNELS=0` environment to disable it.
:::

```{note}
If you want to use sleep mode feature, you should set `COMPILE_CUSTOM_KERNELS=1` manually.
To build custom operators, gcc/g++ higher than 8 and c++ 17 or higher is required. If you're using `pip install -e .` and encounter a torch-npu version conflict, please install with `pip install --no-build-isolation -e .` to build on system env.
If you encounter other problems during compiling, it is probably because unexpected compiler is being used, you may export `CXX_COMPILER` and `C_COMPILER` in environment to specify your g++ and gcc locations before compiling.
```

::::

::::{tab-item} Using docker
:sync: docker

You can just pull the **prebuilt image** and run it with bash.

:::{dropdown} Click here to see "Build from Dockerfile"
or build IMAGE from **source code**:

```bash
git clone https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
docker build -t vllm-ascend-dev-image:latest -f ./Dockerfile .
```

:::

```{code-block} bash
   :substitutions:

# Update DEVICE according to your device (/dev/davinci[0-7])
export DEVICE=/dev/davinci7
# Update the vllm-ascend image
export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|
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
    -v /root/.cache:/root/.cache \
    -it $IMAGE bash
```

The default workdir is `/workspace`, vLLM and vLLM Ascend code are placed in `/vllm-workspace` and installed in [development mode](https://setuptools.pypa.io/en/latest/userguide/development_mode.html) (`pip install -e`) to help developer immediately take place changes without requiring a new installation.
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
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
# Create an LLM.
llm = LLM(model="Qwen/Qwen3-0.6B")

# Generate texts from the prompts.
outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

Then run:

```bash
# Try `export VLLM_USE_MODELSCOPE=true` and `pip install modelscope`
# to speed up download if huggingface is not reachable.
python example.py
```

The output will be like:

```bash
Prompt: 'Hello, my name is', Generated text: " Shinji, a teenage boy from New York City. I'm a computer science"
Prompt: 'The president of the United States is', Generated text: ' a very important person. When he or she is elected, many people think that'
Prompt: 'The capital of France is', Generated text: ' Paris. The oldest part of the city is Saint-Germain-des-Pr'
Prompt: 'The future of AI is', Generated text: ' not bright\n\nThere is no doubt that the evolution of AI will have a huge'
```
