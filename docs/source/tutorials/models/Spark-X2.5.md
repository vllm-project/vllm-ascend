# Spark-X2.5

## Introduction

Spark-X2.5 is a family of dense causal language models developed by XHToken.
This guide covers the BF16 `Spark-X2.5-1.7B` checkpoint on a single Atlas A2
NPU. The model uses a hybrid of sliding-window and full attention, grouped-query
attention, and head-wise attention output gating.

Spark-X2.5 is currently loaded through the out-of-tree
[`XHToken/Spark-plugin`](https://github.com/XHToken/Spark-plugin). The plugin
registers the model architecture and reuses vLLM's runtime layers; no custom
CANN operator is required. It must be installed in the same Python environment
as vLLM before starting the service.

!!! warning

    Spark-X2.5 support is experimental. The external plugin depends on vLLM
    internal APIs, so revalidate it when changing the vLLM or vLLM Ascend
    version. The procedure below was verified with the versions listed in
    [Verified Environment](#verified-environment).

## Supported Features

Refer to the [Supported Models](../../user_guide/support_matrix/supported_models.md)
page for the project-wide feature matrix and the
[Feature Guide](../../user_guide/feature_guide/index.md) for feature configuration.

The following features were verified on `Spark-X2.5-1.7B`:

| Feature | Status | Notes |
| --- | --- | --- |
| BF16 | Supported | Real checkpoint validation |
| Chunked Prefill | Supported | Enabled by default; verified with a prompt larger than the token budget |
| Automatic Prefix Caching | Supported | Repeated-prefix requests returned identical tokens with cache hits |
| Piecewise AclGraph | Supported | Graph capture and replay verified |
| Fullgraph AclGraph | Supported | Graph capture and replay verified |
| Long context | Supported up to 131K | The checkpoint declares 1M context, but this guide only verifies 131K |
| Tensor Parallel | Not verified | This guide uses one NPU and TP=1 |
| Pipeline Parallel | Not verified | Not covered by this guide |
| LoRA | Not verified | Not covered by this guide |
| Speculative Decoding | Not verified | The tested checkpoint does not include draft-model or MTP weights |
| Expert Parallel | Not applicable | Spark-X2.5-1.7B is a dense model |

## Environment Preparation

### Model Weight

- `XHToken/Spark-X2.5-1.7B`: [ModelScope](https://modelscope.cn/models/XHToken/Spark-X2.5-1.7B)

Download the checkpoint before starting the container. For example:

```bash
modelscope download --model XHToken/Spark-X2.5-1.7B \
  --local_dir /path/to/models/Spark-X2.5-1.7B
```

### Verified Environment

| Component | Version |
| --- | --- |
| Hardware | Ascend 910B3, 64 GB, one NPU |
| CANN | 9.1.0 |
| vLLM | 0.27.1 (`6e448d0`) |
| vLLM Ascend | `f698313` (`nightly-main`) |
| torch-npu | 2.10.0.post4 development build |
| Transformers | 5.14.1 |
| Spark plugin | 0.1.0 (`3e1040e`) |

### Start the Container

Use the official nightly image while Spark-X2.5 support is being integrated
upstream. Change the host paths and the NPU device number for your environment.

```bash
export IMAGE=quay.io/ascend/vllm-ascend:nightly-main
export MODEL_ROOT=/path/to/models
export WORKSPACE=/path/to/workspace
mkdir -p "${WORKSPACE}"

docker run -itd \
  --name vllm-ascend-spark \
  --net=host \
  --shm-size=16g \
  --device=/dev/davinci0 \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v "${MODEL_ROOT}":/models:ro \
  -v "${WORKSPACE}":/workspace \
  "${IMAGE}" bash

docker exec -it vllm-ascend-spark bash
```

### Install the Spark Plugin

Inside the container, clone and install the plugin in the same Python
environment that provides the `vllm` command. `--no-deps` prevents the plugin
installation from replacing the vLLM packages supplied by the image.

```bash
cd /workspace
git clone https://github.com/XHToken/Spark-plugin.git
cd Spark-plugin
git checkout 3e1040e63a5907e4c748a485d6795d739dcd5bd6
python -m pip install --no-deps -e .
```

Verify that the package entry point is discoverable:

```bash
python - <<'PY'
from importlib.metadata import entry_points

print([
    (item.name, item.value)
    for item in entry_points(group="vllm.general_plugins")
    if item.name == "spark2_5"
])
PY
```

The output should contain:

```text
('spark2_5', 'vllm_spark2_5_plugin:register')
```

Adding the plugin source directory to `PYTHONPATH` is not sufficient because
vLLM discovers general plugins through installed package entry-point metadata.

## Deployment

### Single-NPU Online Service

Run the server from `/workspace`. This configuration enables the default
AclGraph path, Automatic Prefix Caching, Chunked Prefill, and a practical 128K
context limit.

```bash
cd /workspace
export MODEL_PATH=/models/Spark-X2.5-1.7B
export HCCL_OP_EXPANSION_MODE=AIV

vllm serve "${MODEL_PATH}" \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name spark25 \
  --trust-remote-code \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --max-model-len 131072 \
  --max-num-batched-tokens 2048 \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.90 \
  --generation-config vllm
```

The supported `--generation-config vllm` option prevents a model-provided
`generation_config.json` from overriding vLLM's default sampling parameters.

Do not add `--enforce-eager` for the default deployment because doing so
disables AclGraph. It can be useful temporarily when isolating a graph-related
problem.

### Optional Tool Calling

The Spark plugin also registers its XML/KV tool-call parser as `spark25`. To
enable automatic tool choice, add these arguments to the server command:

```bash
  --enable-auto-tool-choice \
  --tool-call-parser spark25 \
  --chat-template "${MODEL_PATH}/chat_template.jinja"
```

The general-plugin entry-point name is `spark2_5`, while the tool-call parser
name is `spark25`; they are intentionally different.

## Functional Verification

Wait for the log to show `Application startup complete`, and then confirm that
the model endpoint is ready:

```bash
curl http://127.0.0.1:8000/v1/models
```

### Chat Completion

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "spark25",
    "messages": [
      {
        "role": "user",
        "content": "Explain what vLLM does in one sentence."
      }
    ],
    "temperature": 0.6,
    "top_p": 0.95,
    "max_tokens": 128
  }'
```

A successful response contains a non-empty `choices[0].message.content` value.
The first request can take longer while AclGraph compilation finishes.

### Streaming Chat Completion

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "spark25",
    "messages": [
      {
        "role": "user",
        "content": "Write a Python binary search function."
      }
    ],
    "temperature": 0.6,
    "max_tokens": 256,
    "stream": true
  }'
```

## Accuracy Evaluation

An accuracy baseline is not included in this guide. It should be added as a
separate change after the external plugin installation is available in the
vLLM Ascend accuracy-test environment. The real-weight validation for this
guide compared generated tokens and token log probabilities with the
Transformers reference implementation.

## Capacity and Stability Validation

The following checks were run with real BF16 weights on one Ascend 910B3. These
are compatibility and stability results, not cross-platform performance
benchmarks.

| Test | Result |
| --- | --- |
| Prompt lengths from 4K through 131K | All requests succeeded |
| Input exceeding the configured 131072-token limit | Rejected with HTTP 400 as expected |
| Concurrency sweep from 1 through 64 | All requests succeeded |
| 800 short requests in 50 waves of 16 | 800/800 succeeded with no output drift |
| 16 concurrent requests with about 120K prompt tokens each | 16/16 succeeded |
| Repeated-prefix requests | Cache hits observed and generated tokens remained identical |

The checkpoint configuration declares a theoretical maximum context length of
1,048,576 tokens. This guide only claims the practical 131K limit exercised by
the deployment and stability tests above.

## Known Limitations

- The external Spark plugin is required until Spark-X2.5 is supported natively
  by vLLM.
- Only `Spark-X2.5-1.7B` BF16 on a single Atlas A2 NPU is covered here. The 4B
  checkpoint, quantized checkpoints, A3 hardware, and multi-NPU parallelism
  still require separate validation.
- The plugin uses vLLM internal model APIs. Pin the tested plugin revision and
  revalidate startup, inference, and graph replay after upgrading vLLM.
- This guide does not establish an accuracy benchmark or a comparable
  throughput benchmark.
