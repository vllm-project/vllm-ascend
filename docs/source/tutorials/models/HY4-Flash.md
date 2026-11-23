# Hy4-Flash(Experimental)

## 1 Introduction

!!! note

    The "PD Separation Deployment" mentioned in this document is provided only as an example/reference design to demonstrate the basic process of deploying Prefill and Decode in separated mode. It is not recommended for use in the production environment.

    **For the production environment on the customer side, [MindIE Motor](https://www.hiascend.com/document/detail/en/mindie/latest/quickstart/docs/en/user_guide/quick_start.md) + vLLM-Ascend is recommended.**

This release has been validated only on 950DT products. A2 and A3 have not yet been validated; support is planned for future releases.

Hy4-Flash is a Mixture-of-Experts model with sliding-window attention, IndexPool sparse attention, iHC, and Engram conditional memory. The checkpoint has 42 language-model layers and an embedded five-layer DFly draft for speculative decoding.

This document covers environment and model preparation, single-node deployment, PD disaggregation, functional verification, accuracy and performance evaluation, and performance tuning on 950DT products.

## 2 Supported Features

Refer to the [Supported Models](../../user_guide/support_matrix/supported_models.md) page for the support matrix and the [Feature Guide](../../user_guide/feature_guide/index.md) for feature configuration.

The deployment scripts use native FP8 weights, BF16 computation, FP8 KV cache, expert parallelism, and Model Runner V2. The Decode service enables five-token DFly speculative decoding. All serving examples enable asynchronous scheduling. The single-node and Prefill examples enable prefix caching; PD Decode disables it.

## 3 Prerequisites

### 3.1 Model Weight

Prepare the Hy4-Flash FP8 checkpoint, including its tokenizer, model configuration, Engram tables, and embedded DFly weights.

| Weight Version | Deployment | Download Links |
| --- | --- | --- |
| Hy4-Flash FP8 | 950DT products | [ModelScope](https://modelscope.cn/models/Tencent-Hunyuan/Hy4-flash-FP8) |

Download the complete checkpoint to a local directory on each deployment host. In Section 4.1, replace `<HY4_FLASH_FP8_PATH>` with that host directory. The container mounts it at `/models/Hy4-Flash-FP8`, which is the model path used by the serving and benchmark commands below.

### 3.2 Verify Multi-node Communication

Prepare the NPU interconnect by following [Verify inter-node connectivity](../features/pd_disaggregation_mooncake_multi_node.md#verify-multi-node-communication-environment).

Prepare the UB endpoint configuration directory used by `ascend_local_comm_res_path`. Set `UB_ENDPOINTS_PATH` to that directory in both role scripts.

## 4 Installation

### 4.1 Docker Image Installation

Choose the Hy4-Flash image for your container environment:

| Image Variant | Image |
| --- | --- |
| Ubuntu | [quay.io/ascend/vllm-ascend:hy4-flash-a5](https://quay.io/repository/ascend/vllm-ascend?tab=tags&tag=hy4-flash-a5) |
| openEuler | [quay.io/ascend/vllm-ascend:hy4-flash-a5-openeuler](https://quay.io/repository/ascend/vllm-ascend?tab=tags&tag=hy4-flash-a5-openeuler) |

Run the following on each 950DT products host. Set the model path to the FP8 checkpoint. The example uses the openEuler image; set `IMAGE` to `quay.io/ascend/vllm-ascend:hy4-flash-a5` for the Ubuntu image.

```bash
export IMAGE="quay.io/ascend/vllm-ascend:hy4-flash-a5-openeuler"
export CONTAINER_NAME=hy4_flash
export MODEL_DIR="<HY4_FLASH_FP8_PATH>"
export WORK_DIR="<DEPLOYMENT_DIRECTORY>"
mkdir -p "$WORK_DIR"

docker run \
    -it \
    --name "$CONTAINER_NAME" \
    --network host \
    --ipc host \
    --shm-size=100g \
    --device /dev/davinci_manager \
    --device /dev/hisi_hdc \
    --device /dev/ummu \
    --device /dev/uburma \
    --device /dev/davinci0 \
    --device /dev/davinci1 \
    --device /dev/davinci2 \
    --device /dev/davinci3 \
    --device /dev/davinci4 \
    --device /dev/davinci5 \
    --device /dev/davinci6 \
    --device /dev/davinci7 \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware:ro \
    -v /usr/local/sbin:/usr/local/sbin:ro \
    -v /etc/hccn.conf:/etc/hccn.conf:ro \
    -v /usr/local/dcmi:/usr/local/dcmi:ro \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
    -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
    -v /etc/hixlep:/etc/hixlep:ro \
    -v /etc/hccl_rootinfo.json:/etc/hccl_rootinfo.json:ro \
    -v "$MODEL_DIR":/models/Hy4-Flash-FP8:ro \
    -v "$WORK_DIR":/workspace \
    -w /workspace \
    "$IMAGE" \
    bash
```

Run the serving commands in Section 5 inside the container. For the remaining environment requirements, refer to [Installation](../../getting_started/installation.md).

## 5 Online Service Deployment

For standard `vllm serve` arguments, refer to [Serving Arguments](https://docs.vllm.ai/en/latest/cli/serve/#arguments). For Ascend-specific options, refer to [Additional Configuration](../../user_guide/configuration/additional_config.md) and [Environment Variables](../../user_guide/configuration/env_vars.md).

### 5.1 Single-Node Deployment

=== "950DT products"

    Use eight NPUs for TP8 with expert parallelism:

    ```bash
    export MODEL_PATH="/models/Hy4-Flash-FP8"
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export VLLM_USE_V2_MODEL_RUNNER=1
    export HCCL_BUFFSIZE=1024
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

    vllm serve "$MODEL_PATH" \
        --host 0.0.0.0 \
        --port 8000 \
        --served-model-name hy4-flash \
        --trust-remote-code \
        --tensor-parallel-size 8 \
        --data-parallel-size 1 \
        --enable-expert-parallel \
        --max-model-len 131072 \
        --max-num-seqs 32 \
        --max-num-batched-tokens 4096 \
        --gpu-memory-utilization 0.85 \
        --kv-cache-dtype fp8 \
        --engram-config '{
            "cpu_offload": true,
            "dp_shared_memory": false
        }' \
        --speculative-config '{
            "method": "dfly",
            "num_speculative_tokens": 5
        }' \
        --compilation-config '{
            "mode": 0,
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "decode_cudagraph_capture_request_sizes": [1, 2, 3, 4, 5, 6, 7, 8, 16, 32],
            "max_decode_cudagraph_capture_tokens": 512
        }' \
        --additional-config '{
            "multistream_overlap_shared_expert": true,
            "ascend_compilation_config": {
                "enable_npugraph_ex": true
            }
        }' \
        --enable-prefix-caching \
        --async-scheduling \
        --seed 1024
    ```

### 5.2 PD Disaggregation (1P1D)

=== "950DT products"

    Set `MODEL_PATH`, `VLLM_PATH`, `VLLM_ASCEND_PATH`, and `UB_ENDPOINTS_PATH` for the serving environment.

    The following configuration uses eight NPUs on one host. Prefill uses TP4/DP1 in eager mode. Decode uses TP1/DP4 with shared Engram tables, FULL decode graphs, and five-token DFly.

    | Role | NPUs | TP | DP | API Port | KV Port |
    | --- | --- | ---: | ---: | ---: | ---: |
    | Prefill | 0–3 | 4 | 1 | 8100 | 30000 |
    | Decode | 4–7 | 1 | 4 | 8200 | 30100 |
    | Proxy | — | — | — | 8000 | — |

    **Prefill Node (P)**

    Save the following as `run_prefill.sh`:

    ```bash
    #!/usr/bin/env bash
    set -euo pipefail

    # Set these paths for the serving environment.
    MODEL_PATH="/models/Hy4-Flash-FP8"
    VLLM_PATH="<VLLM_SOURCE_DIRECTORY>"
    VLLM_ASCEND_PATH="<VLLM_ASCEND_SOURCE_DIRECTORY>"
    UB_ENDPOINTS_PATH="<UB_ENDPOINT_CONFIG_DIRECTORY>"
    SERVICE_PORT=8100
    KV_PORT=30000
    MAX_NUM_SEQS=128
    MAX_MODEL_LEN=131072
    MAX_BATCHED_TOKENS=4096
    GPU_MEMORY_UTILIZATION=0.85

    export PYTHONPATH="${VLLM_PATH}:${VLLM_ASCEND_PATH}${PYTHONPATH:+:${PYTHONPATH}}"
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
    export VLLM_HOST_IP=127.0.0.1
    export HCCL_BUFFSIZE=1024
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export VLLM_LOGGING_LEVEL=INFO
    export VLLM_USE_V2_MODEL_RUNNER=1

    exec vllm serve "$MODEL_PATH" \
        --host 127.0.0.1 \
        --port "$SERVICE_PORT" \
        --tensor-parallel-size 4 \
        --data-parallel-size 1 \
        --served-model-name hy4-flash \
        --dtype bfloat16 \
        --engram-config '{
            "cpu_offload": true,
            "dp_shared_memory": false
        }' \
        --max-num-seqs "$MAX_NUM_SEQS" \
        --max-model-len "$MAX_MODEL_LEN" \
        --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
        --enable-expert-parallel \
        --trust-remote-code \
        --kv-cache-dtype fp8 \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
        --enable-prefix-caching \
        --async-scheduling \
        --enforce-eager \
        --seed 1024 \
        --compilation-config '{
            "mode": 0,
            "cudagraph_mode": "NONE"
        }' \
        --kv-transfer-config '{
            "kv_connector": "MooncakeConnectorV2",
            "kv_role": "kv_producer",
            "kv_port": '"$KV_PORT"',
            "engine_id": "hy4-flash-prefill",
            "kv_connector_extra_config": {
                "ascend_local_comm_res_path": "'"$UB_ENDPOINTS_PATH"'",
                "prefill": {
                    "tp_size": 4,
                    "dp_size": 1
                },
                "decode": {
                    "tp_size": 1,
                    "dp_size": 4
                }
            }
        }' \
        --additional-config '{
            "multistream_overlap_shared_expert": true,
            "ascend_compilation_config": {
                "enable_npugraph_ex": true
            }
        }'
    ```

    Start the Prefill service:

    ```bash
    nohup bash run_prefill.sh > prefill.log 2>&1 &
    ```

    **Decode Node (D)**

    Save the following as `run_decode.sh`:

    ```bash
    #!/usr/bin/env bash
    set -euo pipefail

    # Set these paths for the serving environment.
    MODEL_PATH="/models/Hy4-Flash-FP8"
    VLLM_PATH="<VLLM_SOURCE_DIRECTORY>"
    VLLM_ASCEND_PATH="<VLLM_ASCEND_SOURCE_DIRECTORY>"
    UB_ENDPOINTS_PATH="<UB_ENDPOINT_CONFIG_DIRECTORY>"
    SERVICE_PORT=8200
    KV_PORT=30100
    MAX_NUM_SEQS=128
    MAX_MODEL_LEN=131072
    MAX_BATCHED_TOKENS=4096
    GPU_MEMORY_UTILIZATION=0.85

    export PYTHONPATH="${VLLM_PATH}:${VLLM_ASCEND_PATH}${PYTHONPATH:+:${PYTHONPATH}}"
    export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
    export VLLM_HOST_IP=127.0.0.1
    export HCCL_BUFFSIZE=1024
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export VLLM_LOGGING_LEVEL=INFO
    export VLLM_USE_V2_MODEL_RUNNER=1

    exec vllm serve "$MODEL_PATH" \
        --host 127.0.0.1 \
        --port "$SERVICE_PORT" \
        --tensor-parallel-size 1 \
        --data-parallel-size 4 \
        --data-parallel-rpc-port 13445 \
        --served-model-name hy4-flash \
        --dtype bfloat16 \
        --engram-config '{
            "cpu_offload": true,
            "dp_shared_memory": true
        }' \
        --max-num-seqs "$MAX_NUM_SEQS" \
        --max-model-len "$MAX_MODEL_LEN" \
        --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
        --enable-expert-parallel \
        --trust-remote-code \
        --kv-cache-dtype fp8 \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
        --no-enable-prefix-caching \
        --async-scheduling \
        --seed 1024 \
        --compilation-config '{
            "mode": 0,
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "decode_cudagraph_capture_request_sizes": [1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128],
            "max_decode_cudagraph_capture_tokens": 512
        }' \
        --speculative-config '{
            "method": "dfly",
            "num_speculative_tokens": 5
        }' \
        --kv-transfer-config '{
            "kv_connector": "MooncakeConnectorV2",
            "kv_role": "kv_consumer",
            "kv_port": '"$KV_PORT"',
            "engine_id": "hy4-flash-decode",
            "kv_connector_extra_config": {
                "ascend_local_comm_res_path": "'"$UB_ENDPOINTS_PATH"'",
                "prefill": {
                    "tp_size": 4,
                    "dp_size": 1
                },
                "decode": {
                    "tp_size": 1,
                    "dp_size": 4
                }
            }
        }' \
        --additional-config '{
            "multistream_overlap_shared_expert": true,
            "ascend_compilation_config": {
                "enable_npugraph_ex": true
            }
        }'
    ```

    Start the Decode service:

    ```bash
    nohup bash run_decode.sh > decode.log 2>&1 &
    ```

    **Proxy**

    Save the following as `run_proxy.sh`:

    ```bash
    #!/usr/bin/env bash
    set -euo pipefail

    VLLM_ASCEND_PATH="<VLLM_ASCEND_SOURCE_DIRECTORY>"

    exec python "$VLLM_ASCEND_PATH/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py" \
        --host 127.0.0.1 \
        --port 8000 \
        --prefiller-hosts 127.0.0.1 \
        --prefiller-ports 8100 \
        --decoder-hosts 127.0.0.1 \
        --decoder-ports 8200
    ```

    Start the proxy after both services report `Application startup complete`:

    ```bash
    bash run_proxy.sh
    ```

    Send requests to the proxy on port `8000`.

    Check readiness:

    ```bash
    curl --fail http://127.0.0.1:8100/health
    curl --fail http://127.0.0.1:8200/health
    curl --fail http://127.0.0.1:8000/healthcheck
    ```

## 6 Functional Verification

### 6.1 Text

Send requests to port `8000` for either the single-node service or the PD proxy:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{
        "model":"hy4-flash",
        "messages":[{"role":"user","content":"爱因斯坦的生日是？"}],
        "temperature":0.0,
        "max_tokens":1024,
        "stream":false
    }'
```

Example response excerpt (reasoning text and unrelated fields omitted):

```json
{
  "model": "hy4-flash",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "阿尔伯特·爱因斯坦（Albert Einstein）的生日是 **1879年3月14日**（他于1955年4月18日去世）。"
      },
      "finish_reason": "stop"
    }
  ]
}
```

### 6.2 Single Image

Use the single-node visual service from [Section 5.1](#51-single-node-deployment):

```bash
export IMAGE_PATH=/path/to/image.png
export BASE_URL=http://127.0.0.1:8000
python - <<'PY'
import base64
import mimetypes
import os
from pathlib import Path

from openai import OpenAI

path = Path(os.environ["IMAGE_PATH"])
mime = mimetypes.guess_type(path.name)[0] or "image/png"
encoded = base64.b64encode(path.read_bytes()).decode("ascii")
client = OpenAI(base_url=os.environ["BASE_URL"] + "/v1", api_key="EMPTY")
response = client.chat.completions.create(
    model="hy4-flash",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe the main objects in this image."},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
        ],
    }],
    temperature=0,
    max_tokens=1024,
)
print(response.choices[0].message.content)
PY
```

## 7 Accuracy Evaluation

### 7.1 Using EvalScope

Install EvalScope in the evaluation client environment:

```bash
python -m pip install 'evalscope==1.15.0'
```

Prepare the AIME2026 and GPQA-Diamond datasets and set their local paths. Run each dataset once with concurrency 16:

```bash
export API_URL=http://127.0.0.1:8000/v1
export AIME_DATA_PATH=/datasets/aime26
export GPQA_DATA_PATH=/datasets/gpqa_diamond
export EVAL_DIR=/workspace/hy4flash-evaluation

for dataset in aime26 gpqa_diamond; do
    if [ "$dataset" = aime26 ]; then
        data_path="$AIME_DATA_PATH"
    else
        data_path="$GPQA_DATA_PATH"
    fi
    dataset_args=$(python - "$dataset" "$data_path" <<'PY'
import json
import sys

print(json.dumps({sys.argv[1]: {"local_path": sys.argv[2]}}))
PY
)
    evalscope eval \
        --model hy4-flash \
        --api-url "$API_URL" \
        --api-key EMPTY \
        --eval-type openai_api \
        --datasets "$dataset" \
        --dataset-hub modelscope \
        --dataset-args "$dataset_args" \
        --eval-batch-size 16 \
        --generation-config '{
            "temperature": 0.9,
            "top_p": 1,
            "top_k": -1,
            "max_tokens": 131072,
            "stream": true,
            "timeout": 1800
        }' \
        --work-dir "$EVAL_DIR/$dataset" \
        --no-timestamp
done
```

### 7.2 Text Evaluation

The following results use the TP8 + EP service with FP8 weights, BF16 computation, FP8 KV, and DFly 5, with prefix caching and asynchronous scheduling enabled.

| Dataset | Hardware | Score | max-model-len | max-num-seqs | max_out_len | batch_size | generation_kwargs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AIME2026 | Ascend 950DT | 29/30 (96.67%) | 131072 | 32 | 131072 | 16 | temperature=0.9, top_p=1.0, top_k=-1 |
| GPQA-Diamond | Ascend 950DT | 182/198 (91.92%) | 131072 | 32 | 131072 | 16 | temperature=0.9, top_p=1.0, top_k=-1 |

## 8 Performance Evaluation

### 8.1 Using vLLM Bench

Run the following benchmark against the service on port `8000`:

```bash
export MODEL_PATH="/models/Hy4-Flash-FP8"
mkdir -p /workspace/hy4flash-benchmarks
vllm bench serve \
    --backend openai \
    --base-url http://127.0.0.1:8000 \
    --endpoint /v1/completions \
    --model "$MODEL_PATH" \
    --served-model-name hy4-flash \
    --trust-remote-code \
    --dataset-name random \
    --random-input-len 4096 \
    --random-output-len 200 \
    --random-range-ratio 1.0 \
    --num-prompts 32 \
    --max-concurrency 8 \
    --request-rate inf \
    --ignore-eos \
    --save-result \
    --result-dir /workspace/hy4flash-benchmarks
```

## 9 Performance Tuning

### 9.1 Recommended Configurations

| Configuration | Single Node | Prefill | Decode |
| --- | --- | --- | --- |
| TP / DP | 8 / 1 | 4 / 1 | 1 / 4 |
| max-model-len | 131072 | 131072 | 131072 |
| max-num-seqs | 32 | 128 | 128 |
| max-num-batched-tokens | 4096 | 4096 | 4096 |
| gpu-memory-utilization | 0.85 | 0.85 | 0.85 |
| KV cache | FP8 | FP8 | FP8 |
| Graph mode | FULL_DECODE_ONLY | NONE | FULL_DECODE_ONLY |
| DFly speculative tokens | 5 | Disabled | 5 |
| Engram CPU offload | Enabled | Enabled | Enabled |

### 9.2 Tuning Guidelines

Keep the `prefill` and `decode` topology declarations consistent in both roles' `--kv-transfer-config`. For general tuning methods, refer to [Optimization and Tuning](../../developer_guide/performance_and_debug/optimization_and_tuning.md).

## 10 FAQ

For common environment and installation issues, refer to the [Public FAQs](../../faqs.md).

### 10.1 Engram Tables Are Replicated Across Decode Ranks

**Symptom:** Decode host-memory use increases when starting DP4.

**Resolution:** Keep `"cpu_offload":true` and `"dp_shared_memory":true` in the Decode `--engram-config`. Prefill uses DP1 and keeps `"dp_shared_memory":false`, as shown in [Section 5.2](#52-pd-disaggregation-1p1d).

### 10.2 Model Loading Is Still in Progress or Startup Fails

Follow `prefill.log` and `decode.log` while weights load and Decode graphs are captured. Start the proxy after both services report `Application startup complete` and their `/health` endpoints return `200`.

If a service exits, inspect its first error and check the model/source paths, visible NPUs, Engram configuration, and UB endpoint directory against the corresponding role script.
