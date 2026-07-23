# Installation

This document describes how to install vllm-ascend manually.

## Requirements

:::::{tab-set}
::::{tab-item} Atlas A2/A3 inference products

- OS: Linux
- Python: >= 3.10, < 3.13
- Hardware with Ascend NPUs. It's usually the Atlas 800 A2 series.
- Atlas inference products.
- Software:

    | Software      | Supported version                | Note                                      |
    |---------------|----------------------------------|-------------------------------------------|
    | Ascend HDK    | Refer to the documentation [CANN 9.0.1](https://www.hiascend.com/document/detail/zh/canncommercial/900/releasenote/releasenote_0000.html) | Required for CANN |
    | CANN          | == 9.0.1                        | Required for vllm-ascend and torch-npu    |
    | torch-npu     | == 2.10.0                 | Required for vllm-ascend, No need to install manually, it will be auto installed in below steps |
    | torch         | == 2.10.0                       | Required for torch-npu and vllm, No need to install manually, it will be auto installed in below steps |
    | NNAL          | == 9.0.1                        | Required for libatb.so, enables advanced tensor operations |

```{note}
Atlas inference products require CANN 9.1.0 . The Atlas A2/A3 requirements in the table above remain unchanged.
```

::::
::::{tab-item} Atlas inference products

 | Software      | Supported version                | Note                                      |
 |---------------|----------------------------------|-------------------------------------------|
 | Ascend HDK    | Refer to the documentation [CANN 9.1.0](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta1/releasenote/9.1.0-beta.1/release-note.md) | Required for CANN |
 | CANN          | == 9.1.0                | Required for vllm-ascend and torch-npu    |
 | torch-npu     | == 2.10.0                 | Required for vllm-ascend, No need to install manually, it will be auto installed in below steps |
 | torch         | == 2.10.0                       | Required for torch-npu and vllm, No need to install manually, it will be auto installed in below steps |
 | NNAL          | == 9.1.0                 | Required for libatb.so, enables advanced tensor operations |
 | triton / triton-ascend | Not supported          | Uninstalled in `Dockerfile.310p` |

::::
:::::

There are two installation methods:

- **Using pip**: first prepare the environment manually or via a CANN image, then install `vllm-ascend` using pip.
- **Using docker**: use the `vllm-ascend` pre-built docker image directly.

## Configure Ascend CANN environment

Before installation, you need to make sure firmware/driver, and CANN are installed correctly, refer to [CANN Installation](https://www.hiascend.com/cann/download?versionId=735&ids=d806%2Ch0501%2Ch0601%2Ch0702) for more details.

### Configure hardware environment

To verify that the Ascend NPU firmware and driver were correctly installed, run:

```bash
npu-smi info
```

Refer to [CANN Installation](https://www.hiascend.com/cann/download?versionId=735&ids=d806%2Ch0501%2Ch0601%2Ch0702) for more details.

### Configure software environment

:::::{tab-set}
:sync-group: install

::::{tab-item} Before using pip
:selected:
:sync: pip

The easiest way to prepare your software environment is using CANN image directly:

```{note}
The CANN prebuilt image includes NNAL (Ascend Neural Network Acceleration Library), which provides libatb.so for advanced tensor operations. No additional installation is required when using the prebuilt image.
```

```{code-block} bash
   :substitutions:
# Update DEVICE according to your device (/dev/davinci[0-7])
export DEVICE=/dev/davinci7
# Update the vllm-ascend image
export IMAGE=quay.io/ascend/cann:|cann_image_tag|
docker run --rm \
    --name vllm-ascend-env \
    --shm-size=1g \
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

```{warning}
If you encounter "libatb.so not found" errors during runtime, please ensure NNAL is properly installed as shown in the manual installation steps below.
```

```bash
# Create a virtual environment.
python -m venv vllm-ascend-env
source vllm-ascend-env/bin/activate

# Install required Python packages.
python -m pip install --upgrade pip
pip3 install attrs numpy decorator sympy cffi pyyaml pathlib2 psutil protobuf scipy requests absl-py wheel typing_extensions

# Download and install the CANN package.
wget --header="Referer: https://www.hiascend.com/" https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.0.1/Ascend-cann-toolkit_9.0.1_linux-"$(uname -i)".run
chmod +x ./Ascend-cann-toolkit_9.0.1_linux-"$(uname -i)".run
./Ascend-cann-toolkit_9.0.1_linux-"$(uname -i)".run --full

wget --header="Referer: https://www.hiascend.com/" https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.0.1/Ascend-cann-910b-ops_9.0.1_linux-"$(uname -i)".run
chmod +x ./Ascend-cann-910b-ops_9.0.1_linux-"$(uname -i)".run
./Ascend-cann-910b-ops_9.0.1_linux-"$(uname -i)".run --install

wget --header="Referer: https://www.hiascend.com/" https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.0.1/Ascend-cann-nnal_9.0.1_linux-"$(uname -i)".run
chmod +x ./Ascend-cann-nnal_9.0.1_linux-"$(uname -i)".run
./Ascend-cann-nnal_9.0.1_linux-"$(uname -i)".run --install
```

:::

::::

::::{tab-item} Before using docker
:sync: docker
No extra steps are needed if you are using the `vllm-ascend` prebuilt Docker image.
::::
:::::

Once this is done, you can start to set up `vllm` and `vllm-ascend`.

## Set up using Python

First, install system dependencies and configure the pip mirror:

```bash
# Using apt-get with mirror
sed -i 's|ports.ubuntu.com|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list
apt-get update -y && apt-get install -y gcc g++ cmake libnuma-dev wget git curl jq
# Or using yum
# yum update -y && yum install -y gcc g++ cmake numactl-devel wget git curl jq
# Config pip mirror,only versions 0.11.0 and earlier are supported, if using a version later than 0.11.0, do not execute this command
pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

**[Optional]** Then configure the extra-index of `pip` if you are working on an x86 machine or using torch-npu dev version:

```bash
# For torch-npu dev version or x86 machine
pip config set global.extra-index-url "https://download.pytorch.org/whl/cpu/"
```

Then you can install `vllm` and `vllm-ascend` from a **pre-built wheel** using one of the following methods:

:::::{tab-set}
:sync-group: install-method

::::{tab-item} Original installation
:sync: original

```{code-block} bash
   :substitutions:

# Install vllm-project/vllm. The newest supported version is |vllm_version|.
pip install vllm==|pip_vllm_version|

# Install vllm-project/vllm-ascend.
pip install \
--extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi/variant https://mirrors.huaweicloud.com/ascend/repos/pypi  \
vllm-ascend==|pip_vllm_ascend_version|

```

::::

::::{tab-item} uv-wheelnext installation
:sync: uv-wheelnext

The `uv-wheelnext` installation downloads only the delta on top of vllm, resulting in a smaller download size. First install `uv-wheelnext` to support incremental wheels:

```bash
# install uv-wheelnext
curl -LsSf https://astral.sh/uv/install.sh | sed 's/verify_checksum "$_file"/true/' | INSTALLER_DOWNLOAD_URL=https://wheelnext.astral.sh sh
source $HOME/.local/bin/env
```

```{code-block} bash
   :substitutions:

# Install vllm-project/vllm. The newest supported version is |vllm_version|.
pip install vllm==|pip_vllm_version|

# Install vllm-project/vllm-ascend from wheelnext index.
uv pip install --system \
--extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi/variant   \
--index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple \
vllm-ascend==|pip_vllm_ascend_version|

```

```{note}
If you encounter errors during `uv pip install` (e.g., corrupted cache or stale package data), try clearing the uv cache first and then re-run the install command:

    uv cache clean

```

::::
:::::

:::{dropdown} Click here to see "Build from source code"
or build from **source code**:

```{note}
To install `triton-ascend`, run:

pip install triton-ascend==3.2.1 --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi

If you are installing via `uv`, make sure to install `triton-ascend` **last**, after all other packages have been installed, to avoid dependency resolution conflicts.
```

```{code-block} bash
   :substitutions:

# Install vLLM.
git clone --depth 1 --branch |vllm_version| https://github.com/vllm-project/vllm
cd vllm
VLLM_TARGET_DEVICE=empty pip install -e .
cd ..

# Install vLLM Ascend.
git clone --depth 1 --branch |vllm_ascend_version| https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
git submodule update --init --recursive
pip install -e .
cd ..
```

If you are building custom operators for Atlas A3, you should run `git submodule update --init --recursive` manually, or ensure your environment has internet access.
:::

:::{note}
Atlas inference products do not support `triton` or `triton-ascend`. Source installations can pull these packages as dependencies; remove them before running on Atlas inference products:

```bash
pip uninstall -y triton-ascend triton
```

:::

```{note}
To build custom operators, gcc/g++ higher than 8 and C++17 or higher are required. If you are using `pip install -e .` and encounter a torch-npu version conflict, please install with `pip install --no-build-isolation -e .` to build on system env.
If you encounter other problems during compiling, it is probably because an unexpected compiler is being used, you may export `CXX_COMPILER` and `C_COMPILER` in the environment to specify your g++ and gcc locations before compiling.

If you are building in a CPU-only environment where `npu-smi` is unavailable, you need to set `SOC_VERSION` before `pip install -e .` so the build can target the correct chip. You can refer to `Dockerfile*` defaults, for example:

- Atlas A2: `export SOC_VERSION=ascend910b1`
- Atlas A3: `export SOC_VERSION=ascend910_9391`
- Atlas inference products: `export SOC_VERSION=ascend310p1`
- Ascend 950 Products: `export SOC_VERSION=<value starting with "ascend950">`
```

```{note}
To enable the batch invariance feature, set `VLLM_BATCH_INVARIANT=1` before building vllm-ascend to install the batch invariance custom operator library during the installation process.
For usage guidance on the batch invariance feature, see <https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/user_guide/feature_guide/batch_invariance.md>
```

## Set up using Docker

`vllm-ascend` offers Docker images for deployment. You can just pull the **prebuilt image** from the image repository [ascend/vllm-ascend](https://quay.io/repository/ascend/vllm-ascend?tab=tags) and run it with bash.

Supported images as following.

| image name | Hardware | OS |
|-|-|-|
| vllm-ascend:{{ vllm_ascend_version }} | Atlas A2 | Ubuntu |
| vllm-ascend:{{ vllm_ascend_version }}-openeuler | Atlas A2 | openEuler |
| vllm-ascend:{{ vllm_ascend_version }}-a3 | Atlas A3 | Ubuntu |
| vllm-ascend:{{ vllm_ascend_version }}-a3-openeuler | Atlas A3 | openEuler |
| vllm-ascend:{{ vllm_ascend_version }}-310p | Atlas inference products | Ubuntu |
| vllm-ascend:{{ vllm_ascend_version }}-310p-openeuler | Atlas inference products | openEuler |

:::{dropdown} Click here to see "Build from Dockerfile"
or build IMAGE from **source code**:

```bash
git clone https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
docker build -t vllm-ascend-dev-image:latest -f ./Dockerfile .
```

:::

:::::{tab-set}
::::{tab-item} A2/A3

```{code-block} bash
   :substitutions:

# Update --device according to your device (Atlas A2: /dev/davinci[0-7] Atlas A3:/dev/davinci[0-15]).
# Update the vllm-ascend image according to your environment.
# Note you should download the weight to /root/.cache in advance.
export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|
docker run --rm \
    --name vllm-ascend-env \
    --shm-size=1g \
    --net=host \
    --device /dev/davinci0 \
    --device /dev/davinci1 \
    --device /dev/davinci2 \
    --device /dev/davinci3 \
    --device /dev/davinci4 \
    --device /dev/davinci5 \
    --device /dev/davinci6 \
    --device /dev/davinci7 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /root/.cache:/root/.cache \
    -it $IMAGE bash
```

::::

::::{tab-item} Atlas inference products

Adjust `/dev/davinci0` to the NPU you want to use.

```{code-block} bash
   :substitutions:

export DEVICE=/dev/davinci0
export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|-310p

docker run --rm \
    --name vllm-ascend \
    --shm-size=1g \
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
    -p 8000:8000 \
    -it $IMAGE bash
```

::::

::::{tab-item} Atlas 200I Pro

Atlas 200I Pro requires additional device nodes, driver libraries, and configuration files so that `npu-smi` and other driver commands work inside the container. Adjust `/dev/davinci0` to the NPU you want to use.

```{code-block} bash
   :substitutions:

export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|-310p

docker run --rm \
    --privileged \
    --name vllm-ascend \
    --shm-size=10g \
    --device=/dev/davinci0:/dev/davinci0 \
    --device=/dev/davinci_manager \
    --device=/dev/ascend_manager \
    --device=/dev/user_config \
    -v /etc/sys_version.conf:/etc/sys_version.conf \
    -v /etc/ld.so.conf.d/mind_so.conf:/etc/ld.so.conf.d/mind_so.conf \
    -v /etc/hdcBasic.cfg:/etc/hdcBasic.cfg \
    -v /var/dmp_daemon:/var/dmp_daemon \
    -v /usr/lib64/libmmpa.so:/usr/lib64/libmmpa.so \
    -v /usr/lib64/libcrypto.so.1.1:/usr/lib64/libcrypto.so.1.1 \
    -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
    -v /usr/lib64/libstackcore.so:/usr/lib64/libstackcore.so \
    -v /usr/lib/aarch64-linux-gnu/libyaml-0.so.2:/usr/lib64/libyaml-0.so.2 \
    -v /etc/slog.conf:/etc/slog.conf \
    -v /var/slogd:/var/slogd \
    -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64 \
    -v /usr/lib64/libtensorflow.so:/usr/lib64/libtensorflow.so \
    -v /root/.cache:/root/.cache \
    -p 8000:8000 \
    -it $IMAGE bash
```

For openEuler, keep the same command structure and make the following substitutions:

- Set `IMAGE` to `quay.io/ascend/vllm-ascend:|vllm_ascend_version|-310p-openeuler`.
- Add `-v /usr/lib64/libsemanage.so.2:/usr/lib64/libsemanage.so.2`.
- Replace the `libyaml` mount with `-v /usr/lib64/libyaml-0.so.2.0.9:/usr/lib64/libyaml-0.so.2`.

::::
:::::

The default workdir is `/workspace`, vLLM and vLLM Ascend code are placed in `/vllm-workspace` and installed in [development mode](https://setuptools.pypa.io/en/latest/userguide/development_mode.html) (`pip install -e`) to help developers immediately make changes without requiring a new installation.

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
python example.py
```

If you encounter a connection error with Hugging Face (e.g., `We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.`), run the following commands to use ModelScope as an alternative:

```bash
export VLLM_USE_MODELSCOPE=True
pip install modelscope
python example.py
```

This section shows ascend platform is successfully detected in vllm:

```bash
INFO 05-27 11:40:38 [__init__.py:44] Available plugins for group vllm.platform_plugins:
INFO 05-27 11:40:38 [__init__.py:46] - ascend -> vllm_ascend:register
INFO 05-27 11:40:38 [__init__.py:49] All plugins in this group will be loaded. Set `VLLM_PLUGINS` to control which plugins to load.
INFO 05-27 11:40:38 [__init__.py:238] Platform plugin ascend is activated
```

This section shows the final output:

```bash
Prompt: 'Hello, my name is', Generated text: ' Lucy and I am an 8 year old who loves to draw and write stories'
Prompt: 'The president of the United States is', Generated text: " a key leader in the federal government, and the president's role in the executive"
Prompt: 'The capital of France is', Generated text: ' a city. What is the capital of France? The capital of France is Paris'
Prompt: 'The future of AI is', Generated text: ' a topic that is being discussed in various contexts. In the business world, AI'
```

This section shows process exits after offline inference, and does not affect actual inference:

```bash
(EngineCore pid=970) INFO 05-12 11:36:00 [core.py:1201] Shutdown initiated (timeout=0)
(EngineCore pid=970) INFO 05-12 11:36:00 [core.py:1224] Shutdown complete
ERROR 05-12 11:36:01 [core_client.py:704] Engine core proc EngineCore died unexpectedly, shutting down client.
sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute
```

## Multi-node Deployment

### Verify Multi-Node Communication

First, check physical layer connectivity, then verify each node, and finally verify the inter-node connectivity.

#### Physical Layer Requirements

- The physical machines must be located on the same WLAN, with network connectivity.
- All NPUs are connected with optical modules, and the connection status must be normal.

#### Each Node Verification

Execute the following commands on each node in sequence. The results must all be `success` and the status must be `UP`:

:::::{tab-set}
:sync-group: multi-node

::::{tab-item} A2 series
:sync: A2

```bash
 # Check the remote switch ports
 for i in {0..7}; do hccn_tool -i $i -lldp -g | grep Ifname; done 
 # Get the link status of the Ethernet ports (UP or DOWN)
 for i in {0..7}; do hccn_tool -i $i -link -g ; done
 # Check the network health status
 for i in {0..7}; do hccn_tool -i $i -net_health -g ; done
 # View the network detected IP configuration
 for i in {0..7}; do hccn_tool -i $i -netdetect -g ; done
 # View gateway configuration
 for i in {0..7}; do hccn_tool -i $i -gateway -g ; done
 # View NPU network configuration
 cat /etc/hccn.conf
```

::::
::::{tab-item} A3 series
:sync: A3

```bash
 # Check the remote switch ports
 for i in {0..15}; do hccn_tool -i $i -lldp -g | grep Ifname; done 
 # Get the link status of the Ethernet ports (UP or DOWN)
 for i in {0..15}; do hccn_tool -i $i -link -g ; done
 # Check the network health status
 for i in {0..15}; do hccn_tool -i $i -net_health -g ; done
 # View the network detected IP configuration
 for i in {0..15}; do hccn_tool -i $i -netdetect -g ; done
 # View gateway configuration
 for i in {0..15}; do hccn_tool -i $i -gateway -g ; done
 # View NPU network configuration
 cat /etc/hccn.conf
```

::::
:::::

#### Interconnect Verification

##### 1. Get NPU IP Addresses

:::::{tab-set}
:sync-group: multi-node

::::{tab-item} A2 series
:sync: A2

```bash
for i in {0..7}; do hccn_tool -i $i -ip -g | grep ipaddr; done
```

::::
::::{tab-item} A3 series
:sync: A3

```bash
for i in {0..15}; do hccn_tool -i $i -ip -g | grep ipaddr; done
```

::::
:::::

##### 2. Cross-Node PING Test

```bash
# Execute on the target node (replace with actual IP)
hccn_tool -i 0 -ping -g address x.x.x.x
```

### Run Container In Each Node

Using vLLM-ascend official container is more efficient to run multi-node environment.

Run the following command to start the container in each node (You should download the weight to /root/.cache in advance):

:::::{tab-set}
:sync-group: multi-node

::::{tab-item} A2 series
:sync: A2

```{code-block} bash
   :substitutions:
# Update the vllm-ascend image
# openEuler:
# export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|-openeuler
# Ubuntu:
# export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|
export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|

# Run the container using the defined variables
# Note if you are running bridge network with docker, Please expose available ports
# for multiple nodes communication in advance
docker run --rm \
--name vllm-ascend \
--net=host \
--shm-size=1g \
--device /dev/davinci0 \
--device /dev/davinci1 \
--device /dev/davinci2 \
--device /dev/davinci3 \
--device /dev/davinci4 \
--device /dev/davinci5 \
--device /dev/davinci6 \
--device /dev/davinci7 \
--device /dev/davinci_manager \
--device /dev/devmm_svm \
--device /dev/hisi_hdc \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /root/.cache:/root/.cache \
-it $IMAGE bash
```

::::
::::{tab-item} A3 series
:sync: A3

```{code-block} bash
   :substitutions:
# Update the vllm-ascend image
# openEuler:
# export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|-a3-openeuler
# Ubuntu:
# export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|-a3
export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|-a3

# Run the container using the defined variables
# Note if you are running bridge network with docker, Please expose available ports
# for multiple nodes communication in advance
docker run --rm \
--name vllm-ascend \
--net=host \
--shm-size=1g \
--device /dev/davinci0 \
--device /dev/davinci1 \
--device /dev/davinci2 \
--device /dev/davinci3 \
--device /dev/davinci4 \
--device /dev/davinci5 \
--device /dev/davinci6 \
--device /dev/davinci7 \
--device /dev/davinci8 \
--device /dev/davinci9 \
--device /dev/davinci10 \
--device /dev/davinci11 \
--device /dev/davinci12 \
--device /dev/davinci13 \
--device /dev/davinci14 \
--device /dev/davinci15 \
--device /dev/davinci_manager \
--device /dev/devmm_svm \
--device /dev/hisi_hdc \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /root/.cache:/root/.cache \
-it $IMAGE bash
```

::::
:::::
