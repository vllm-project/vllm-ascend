# Quickstart

## Prerequisites

### Supported Devices
- Atlas A2 training series (Atlas 800T A2, Atlas 900 A2 PoD, Atlas 200T A2 Box16, Atlas 300T A2)
- Atlas 800I A2 inference series (Atlas 800I A2)
- Atlas A3 training series (Atlas 800T A3, Atlas 900 A3 SuperPoD, Atlas 9000 A3 SuperPoD)
- Atlas 800I A3 inference series (Atlas 800I A3)
- [Experimental] Atlas 300I inference series (Atlas 300I Duo)

## Setup environment using container

:::::{tab-set}
::::{tab-item} Ubuntu

```{code-block} bash
   :substitutions:

# Update DEVICE according to your device (/dev/davinci[0-7])
export DEVICE=/dev/davinci0
# Update the vllm-ascend image
export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|
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
export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|-openeuler
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
# Install curl
yum update -y && yum install -y curl
```

::::
:::::

The default workdir is `/workspace`, vLLM and vLLM Ascend code are placed in `/vllm-workspace` and installed in [development mode](https://setuptools.pypa.io/en/latest/userguide/development_mode.html) (`pip install -e`) to help developers immediately make changes effective without requiring a new installation.

## Usage

You can use ModelScope mirror to speed up download:

<!-- tests/e2e/doctest/001-quickstart-test.sh should be considered updating as well -->

```bash
export VLLM_USE_MODELSCOPE=true
```

There are two ways to start vLLM on Ascend NPU:

:::::{tab-set}
::::{tab-item} Offline Batched Inference

With vLLM installed, you can start generating texts for list of input prompts (i.e. offline batch inferencing).

Try to run below Python script directly or use `python3` shell to generate texts:

<!-- tests/e2e/doctest/001-quickstart-test.sh should be considered updating as well -->

```python
from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
# The first run will take about 3-5 mins (10 MB/s) to download models
llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct")

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

::::

::::{tab-item} OpenAI Completions API

vLLM can also be deployed as a server that implements the OpenAI API protocol. Run
the following command to start the vLLM server with the
[Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) model:

<!-- tests/e2e/doctest/001-quickstart-test.sh should be considered updating as well -->

```bash
# Deploy vLLM server (The first run will take about 3-5 mins (10 MB/s) to download models)
vllm serve Qwen/Qwen2.5-0.5B-Instruct &
```

If you see a log as below:

```
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
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "prompt": "Beijing is a",
        "max_tokens": 5,
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

```
INFO:     Shutting down FastAPI HTTP server.
INFO:     Shutting down
INFO:     Waiting for application shutdown.
INFO:     Application shutdown complete.
```

Finally, you can exit the container by using `ctrl-D`.
::::
:::::
