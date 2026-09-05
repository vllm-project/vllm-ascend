# Quickstart

## Introduction

This section guides you through container-based environment setup and large model inference, using the Qwen3-0.6B offline single-GPU inference script as an example.

- For details on using different models, see the corresponding model tutorial in the "Model Tutorials" directory, for example, [Qwen3-30B-A3B](../../docs/source/tutorials/models/Qwen3-30B-A3B.md).
- For details on using different functions, see the corresponding function tutorial in the "Function Tutorials" directory, for example, [Prefill-Decode Disaggregation (Deepseek)](../../docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md).

## Prerequisites

### Supported Devices

- Atlas A2 training series (Atlas 800T A2, Atlas 900 A2 PoD, Atlas 200T A2 Box16, Atlas 300T A2)
- Atlas 800I A2 inference series (Atlas 800I A2)
- Atlas A3 training series (Atlas 800T A3, Atlas 900 A3 SuperPoD, Atlas 9000 A3 SuperPoD)
- Atlas 800I A3 inference series (Atlas 800I A3)
- Atlas inference products

## Requirements

:::::{tab-set}

::::{tab-item} Atlas A2/A3 inference products

- OS: Linux
- Python: >= 3.10, < 3.13
- Hardware with Ascend NPUs. It's usually the Atlas 800 A2 series.
- Software:

    | Software      | Supported version                | Note                                      |
    |---------------|----------------------------------|-------------------------------------------|
    | Ascend HDK    | Refer to the documentation [CANN 9.0.1](https://www.hiascend.com/document/detail/zh/canncommercial/900/releasenote/releasenote_0000.html) | Required for CANN |  
    | CANN          | == 9.0.1                        | Required for vllm-ascend and torch-npu    |
    | torch-npu     | == 2.10.0                 | Required for vllm-ascend, No need to install manually, it will be auto installed in below steps |
    | torch         | == 2.10.0                       | Required for torch-npu and vllm, No need to install manually, it will be auto installed in below steps |
    | NNAL          | == 9.0.1                        | Required for libatb.so, enables advanced tensor operations |

```{note}
Atlas inference products use CANN 9.1.0 beta and `float16`. Use the `-310p` image suffix for Ubuntu or `-310p-openeuler` for openEuler. Atlas inference products do not support `triton` or `triton-ascend`.

Atlas inference products and Atlas 200I Pro do not support `enable_npugraph_ex`. Set --additional-config '{"ascend_compilation_config": {"enable_npugraph_ex":false}}'.

Atlas 200I Pro requires additional device nodes and driver mounts. See [Set up using Docker](installation.md#set-up-using-docker) for the complete container commands.
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

## Setup environment using container

Before using containers, make sure Docker is installed on your system. If Docker is not installed, please refer to the [Docker installation guide](https://docs.docker.com/get-docker/) for installation instructions.

:::::{tab-set}
::::{tab-item} Ubuntu

```{code-block} bash
   :substitutions:

# Update DEVICE according to your device (/dev/davinci[0-7])
export DEVICE=/dev/davinci0
# Update the vllm-ascend image
# Atlas A2:
# export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|
# Atlas A3:
# export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|-a3
export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|
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
# Install curl
apt-get update -y && apt-get install -y curl
```

::::

::::{tab-item} Ubuntu (Atlas inference products)

The following command applies to Atlas inference products. For Atlas 200I Pro, use the additional device nodes and driver mounts documented in [Installation](installation.md#set-up-using-docker).

```{code-block} bash
   :substitutions:

# Update DEVICE according to your device (/dev/davinci[0-7])
export DEVICE=/dev/davinci0
# Update the vllm-ascend image
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
# Install curl
apt-get update -y && apt-get install -y curl
```

::::

::::{tab-item} openEuler

```{code-block} bash
   :substitutions:

# Update DEVICE according to your device (/dev/davinci[0-7])
export DEVICE=/dev/davinci0
# Update the vllm-ascend image
# Atlas A2:
# export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|-openeuler
# Atlas A3:
# export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|-a3-openeuler
export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|-openeuler
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
# Install curl
yum update -y && yum install -y curl
```

::::

::::{tab-item} openEuler (Atlas inference products)

The following command applies to Atlas inference products. For Atlas 200I Pro, use the additional device nodes and driver mounts documented in [Installation](installation.md#set-up-using-docker).

```{code-block} bash
   :substitutions:

# Update DEVICE according to your device (/dev/davinci[0-7])
export DEVICE=/dev/davinci0
# Update the vllm-ascend image
export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|-310p-openeuler
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
# Install curl
yum update -y && yum install -y curl
```

::::
:::::

The default workdir is `/workspace`, vLLM and vLLM Ascend code are placed in `/vllm-workspace` and installed in [development mode](https://setuptools.pypa.io/en/latest/userguide/development_mode.html) (`pip install -e`) to help developers make changes effective immediately without requiring a new installation.

## Usage

You can use ModelScope mirror to speed up download:

<!-- tests/e2e/doctests/001-quickstart-test.sh should be considered updating as well -->

```bash
export VLLM_USE_MODELSCOPE=True
```

There are two ways to start vLLM on Ascend NPU:

:::::{tab-set}
::::{tab-item} Offline Batched Inference

With vLLM installed, you can start generating texts for list of input prompts (i.e. offline batch inference).

Create and run a simple inference test. The `example.py` can be like:

<!-- tests/e2e/doctest/001-quickstart-test.sh should be considered updating as well -->

```python
from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
# The first run will take about 3-5 mins (10 MB/s) to download models
llm = LLM(model="Qwen/Qwen3-0.6B")

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

::::

::::{tab-item} OpenAI Completions API

vLLM can also be deployed as a server that implements the OpenAI API protocol. Run
the following command to start the vLLM server with the
[Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) model:

<!-- tests/e2e/doctest/001-quickstart-test.sh should be considered updating as well -->

```bash
# Deploy vLLM server (The first run will take about 3-5 mins (10 MB/s) to download models)
vllm serve Qwen/Qwen3-0.6B &
```

If you see a log as below:

```shell
INFO:     Started server process [3594]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

Congratulations, you have successfully started the vLLM server!

You can query the list of models:

<!-- tests/e2e/doctest/001-quickstart-test.sh should be considered updating as well -->

```bash
curl http://localhost:8000/v1/models | python3 -m json.tool
```

You can also query the model with input prompts:

<!-- tests/e2e/doctest/001-quickstart-test.sh should be considered updating as well -->

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen3-0.6B",
        "prompt": "Beijing is a",
        "max_completion_tokens": 5,
        "temperature": 0
    }' | python3 -m json.tool
```

vLLM is serving as a background process, you can use `kill -2 $VLLM_PID` to stop the background process gracefully, which is similar to `Ctrl-C` for stopping the foreground vLLM process:

<!-- tests/e2e/doctest/001-quickstart-test.sh should be considered updating as well -->

```bash
  VLLM_PID=$(pgrep -f "vllm serve")
  kill -2 "$VLLM_PID"
```

The output is as below:

```shell
INFO:     Shutting down FastAPI HTTP server.
INFO:     Shutting down
INFO:     Waiting for application shutdown.
INFO:     Application shutdown complete.
```

Finally, you can exit the container by using `ctrl-D`.
::::
:::::
