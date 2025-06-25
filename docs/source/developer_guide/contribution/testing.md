# Testing

This secition explains how to write e2e tests and unit tests to verify the implementation of your feature.

## Setup test environment

The fastest way to setup test environment is to use the main branch container image:

:::::{tab-set}
:sync-group: e2e

::::{tab-item} Single card
:selected:
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
::::

:::::

After starting the container, you should install the required packages:

```bash
# Prepare
pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# Install required packages
pip install -r requirements-dev.txt
```

## Running tests

### Unit test

There are several principles to follow when writing unit tests:

- The test file path should be consistent with source file and start with `test_` prefix, such as: `vllm_ascend/worker/worker_v1.py` --> `tests/ut/worker/test_worker_v1.py`
- The vLLM Ascend test are using unittest framework, see [here](https://docs.python.org/3/library/unittest.html#module-unittest) to understand how to write unit tests.
- All unit tests can be run on CPU, so you must mock the device-related function to host.
- Example: [tests/ut/test_ascend_config.py](https://github.com/vllm-project/vllm-ascend/blob/main/tests/ut/test_ascend_config.py).
- You can run the unit tests using `pytest`:

    ```bash
    cd /vllm-workspace/vllm-ascend/
    # Run all single card the tests
    pytest -sv tests/ut

    # Run 
    pytest -sv tests/ut/test_ascend_config.py
    ```

### E2E test

Although vllm-ascend CI provide [e2e test](https://github.com/vllm-project/vllm-ascend/blob/main/.github/workflows/vllm_ascend_test.yaml) on Ascend CI, you can run it
locally.

:::::{tab-set}
:sync-group: e2e

::::{tab-item} Single card
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

::::{tab-item} Multi cards test
:sync: multi
```bash
cd /vllm-workspace/vllm-ascend/
# Run all single card the tests
VLLM_USE_MODELSCOPE=true pytest -sv tests/e2e/multicard/

# Run a certain test script
VLLM_USE_MODELSCOPE=true pytest -sv tests/e2e/multicard/test_dynamic_npugraph_batchsize.py

# Run a certain case in test script
VLLM_USE_MODELSCOPE=true pytest -sv tests/e2e/multicard/test_offline_inference.py::test_models
```
::::

:::::

This will reproduce e2e test: [vllm_ascend_test.yaml](https://github.com/vllm-project/vllm-ascend/blob/main/.github/workflows/vllm_ascend_test.yaml).

#### E2E test example:

- Offline test example: [`tests/e2e/singlecard/test_offline_inference.py`](https://github.com/vllm-project/vllm-ascend/blob/main/tests/e2e/singlecard/test_offline_inference.py)
- Online test examples: [`tests/e2e/singlecard/test_prompt_embedding.py`](https://github.com/vllm-project/vllm-ascend/blob/main/tests/e2e/singlecard/test_prompt_embedding.py)
- Correctness test example: [`tests/e2e/singlecard/test_aclgraph.py`](https://github.com/vllm-project/vllm-ascend/blob/main/tests/e2e/singlecard/test_aclgraph.py)
- Reduced Layer model test example: [test_torchair_graph_mode.py - DeepSeek-V3-Pruning](https://github.com/vllm-project/vllm-ascend/blob/20767a043cccb3764214930d4695e53941de87ec/tests/e2e/multicard/test_torchair_graph_mode.py#L48)

    The CI resource is limited, you might need to reduce layer number of the model, below is an example of how to generate a reduced layer model:
    1. Fork the original model repo in modelscope, we need all the files in the repo except for weights.
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
The doctest is a good way to make sure the docs are up to date and the examples are executable, you can run it locally as follows:

```bash
# Run doctest
/vllm-workspace/vllm-ascend/tests/e2e/run_doctests.sh
```

This will reproduce the same environment as the CI: [vllm_ascend_doctest.yaml](https://github.com/vllm-project/vllm-ascend/blob/main/.github/workflows/vllm_ascend_doctest.yaml).
