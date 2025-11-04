# Testing

This document explains how to write E2E tests and unit tests to verify the implementation of your feature.

## Setup a test environment

The fastest way to setup a test environment is to use the main branch's container image:

:::::{tab-set}
:sync-group: e2e

::::{tab-item} Local (CPU)
:selected:
:sync: cpu

You can run the unit tests on CPUs with the following steps:

```{code-block} bash
   :substitutions:

cd ~/vllm-project/
# ls
# vllm  vllm-ascend

# Use mirror to speed up download
# docker pull quay.nju.edu.cn/ascend/cann:|cann_image_tag|
export IMAGE=quay.io/ascend/cann:|cann_image_tag|
docker run --rm --name vllm-ascend-ut \
    -v $(pwd):/vllm-project \
    -v ~/.cache:/root/.cache \
    -ti $IMAGE bash

# (Optional) Configure mirror to speed up download
sed -i 's|ports.ubuntu.com|mirrors.huaweicloud.com|g' /etc/apt/sources.list
pip config set global.index-url https://mirrors.huaweicloud.com/repository/pypi/simple/

# For torch-npu dev version or x86 machine
export PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu/ https://mirrors.huaweicloud.com/ascend/repos/pypi"

apt-get update -y
apt-get install -y python3-pip git vim wget net-tools gcc g++ cmake libnuma-dev curl gnupg2

# Install vllm
cd /vllm-project/vllm
VLLM_TARGET_DEVICE=empty python3 -m pip -v install .

# Install vllm-ascend
cd /vllm-project/vllm-ascend
# [IMPORTANT] Import LD_LIBRARY_PATH to enumerate the CANN environment under CPU
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/Ascend/ascend-toolkit/latest/$(uname -m)-linux/devlib
python3 -m pip install -r requirements-dev.txt
python3 -m pip install -v .
```

::::

::::{tab-item} Single card
:sync: single

```{code-block} bash
   :substitutions:

# Update DEVICE according to your device (/dev/davinci[0-7])
export DEVICE=/dev/davinci0
# Update the vllm-ascend image
export IMAGE=quay.io/ascend/vllm-ascend:main
docker run --rm \
    --name vllm-ascend \
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

After starting the container, you should install the required packages:

```bash
# Prepare
pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# Install required packages
pip install -r requirements-dev.txt
```

::::

::::{tab-item} Multi cards
:sync: multi

```{code-block} bash
   :substitutions:
# Update the vllm-ascend image
export IMAGE=quay.io/ascend/vllm-ascend:main
docker run --rm \
    --name vllm-ascend \
    --device /dev/davinci0 \
    --device /dev/davinci1 \
    --device /dev/davinci2 \
    --device /dev/davinci3 \
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

After starting the container, you should install the required packages:

```bash
cd /vllm-workspace/vllm-ascend/

# Prepare
pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# Install required packages
pip install -r requirements-dev.txt
```

::::

:::::

## Running tests

### Unit tests

There are several principles to follow when writing unit tests:

- The test file path should be consistent with the source file and start with the `test_` prefix, such as: `vllm_ascend/worker/worker_v1.py` --> `tests/ut/worker/test_worker_v1.py`
- The vLLM Ascend test uses unittest framework. See [here](https://docs.python.org/3/library/unittest.html#module-unittest) to understand how to write unit tests.
- All unit tests can be run on CPUs, so you must mock the device-related function to host.
- Example: [tests/ut/test_ascend_config.py](https://github.com/vllm-project/vllm-ascend/blob/main/tests/ut/test_ascend_config.py).
- You can run the unit tests using `pytest`:

:::::{tab-set}
:sync-group: e2e

::::{tab-item} Local (CPU)
:selected:
:sync: cpu

```bash
# Run unit tests
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/Ascend/ascend-toolkit/latest/$(uname -m)-linux/devlib
TORCH_DEVICE_BACKEND_AUTOLOAD=0 pytest -sv tests/ut
```

::::

::::{tab-item} Single-card
:sync: single

```bash
cd /vllm-workspace/vllm-ascend/
# Run all single card the tests
pytest -sv tests/ut

# Run single test
pytest -sv tests/ut/test_ascend_config.py
```

::::

::::{tab-item} Multi-card
:sync: multi

```bash
cd /vllm-workspace/vllm-ascend/
# Run all single card the tests
pytest -sv tests/ut

# Run single test
pytest -sv tests/ut/test_ascend_config.py
```

::::

:::::

### E2E test

Although vllm-ascend CI provides the [E2E test](https://github.com/vllm-project/vllm-ascend/blob/main/.github/workflows/vllm_ascend_test.yaml) on Ascend CI, you can run it
locally.

:::::{tab-set}
:sync-group: e2e

::::{tab-item} Local (CPU)
:sync: cpu

You can't run the E2E test on CPUs.
::::

::::{tab-item} Single-card
:selected:
:sync: single

```bash
cd /vllm-workspace/vllm-ascend/
# Run all single card the tests
VLLM_USE_MODELSCOPE=true pytest -sv tests/e2e/singlecard/

# Run a certain test script
VLLM_USE_MODELSCOPE=true pytest -sv tests/e2e/singlecard/test_offline_inference.py

# Run a certain case in test script
VLLM_USE_MODELSCOPE=true pytest -sv tests/e2e/singlecard/test_offline_inference.py::test_models
```

::::

::::{tab-item} Multi-card
:sync: multi

```bash
cd /vllm-workspace/vllm-ascend/
# Run all the single card tests
VLLM_USE_MODELSCOPE=true pytest -sv tests/e2e/multicard/

# Run a certain test script
VLLM_USE_MODELSCOPE=true pytest -sv tests/e2e/multicard/test_dynamic_npugraph_batchsize.py

# Run a certain case in test script
VLLM_USE_MODELSCOPE=true pytest -sv tests/e2e/multicard/test_offline_inference.py::test_models
```

::::

:::::

This will reproduce the E2E test. See [vllm_ascend_test.yaml](https://github.com/vllm-project/vllm-ascend/blob/main/.github/workflows/vllm_ascend_test.yaml).

#### E2E test example:

- Offline test example: [`tests/e2e/singlecard/test_offline_inference.py`](https://github.com/vllm-project/vllm-ascend/blob/main/tests/e2e/singlecard/test_offline_inference.py)
- Online test examples: [`tests/e2e/singlecard/test_prompt_embedding.py`](https://github.com/vllm-project/vllm-ascend/blob/main/tests/e2e/singlecard/test_prompt_embedding.py)
- Correctness test example: [`tests/e2e/singlecard/test_aclgraph.py`](https://github.com/vllm-project/vllm-ascend/blob/main/tests/e2e/singlecard/test_aclgraph.py)
- Reduced Layer model test example: [test_torchair_graph_mode.py - DeepSeek-V3-Pruning](https://github.com/vllm-project/vllm-ascend/blob/20767a043cccb3764214930d4695e53941de87ec/tests/e2e/multicard/test_torchair_graph_mode.py#L48)

    The CI resource is limited, and you might need to reduce the number of layers of a model. Below is an example of how to generate a reduced layer model:
    1. Fork the original model repo in modelscope. All the files in the repo except for weights are required.
    2. Set `num_hidden_layers` to the expected number of layers, e.g., `{"num_hidden_layers": 2,}`
    3. Copy the following python script as `generate_random_weight.py`. Set the relevant parameters `MODEL_LOCAL_PATH`, `DIST_DTYPE` and `DIST_MODEL_PATH` as needed:

        ```python
        import torch
        from transformers import AutoTokenizer, AutoConfig
        from modeling_deepseek import DeepseekV3ForCausalLM
        from modelscope import snapshot_download

        MODEL_LOCAL_PATH = "~/.cache/modelscope/models/vllm-ascend/DeepSeek-V3-Pruning"
        DIST_DTYPE = torch.bfloat16
        DIST_MODEL_PATH = "./random_deepseek_v3_with_2_hidden_layer"

        config = AutoConfig.from_pretrained(MODEL_LOCAL_PATH, trust_remote_code=True)
        model = DeepseekV3ForCausalLM(config)
        model = model.to(DIST_DTYPE)
        model.save_pretrained(DIST_MODEL_PATH)
        ```

### Run doctest

vllm-ascend provides a `vllm-ascend/tests/e2e/run_doctests.sh` command to run all doctests in the doc files.
The doctest is a good way to make sure docs stay current and examples remain executable, which can be run locally as follows:

```bash
# Run doctest
/vllm-workspace/vllm-ascend/tests/e2e/run_doctests.sh
```

This will reproduce the same environment as the CI. See [vllm_ascend_doctest.yaml](https://github.com/vllm-project/vllm-ascend/blob/main/.github/workflows/vllm_ascend_doctest.yaml).
