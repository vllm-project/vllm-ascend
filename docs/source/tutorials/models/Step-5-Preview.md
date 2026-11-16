# Step-5-Preview (Experimental)

## 1 Introduction

**Step-5-Preview** is a next-generation flagship Mixture-of-Experts (MoE) model developed by StepFun. The model has **600B** total parameters, with **27B** active per token, and supports a 1M-token context window and vision input.

This document describes how to quickly get started with `Step-5-Preview` model inference deployment on Ascend NPU using vLLM-Ascend, based on W8A8 quantized weights.

!!! warning

    **Current status and constraints**

    - `Step-5-Preview` is provided **out-of-the-box** through the official Docker image `quay.io/ascend/vllm-ascend:step5-a5-openeuler`. The supporting code has **not yet been merged** into the vLLM-Ascend repository, so installing it from source (`pip install` or building from source) is **not supported** for this model yet.
    - Only **950DT products** is supported now. Other Ascend hardware (e.g., Atlas A3 Products) is not supported for `Step-5-Preview`.
    - The features listed in [Supported Features](#2-supported-features) are only those enabled by the verified deployment commands in this document, and do **not** imply that all features are supported for `Step-5-Preview`. This is an early-access version; performance optimization and reliability validation are still in progress (see [Declaration](#9-declaration)).

## 2 Supported Features

Refer to [Supported Features List](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [Feature Guide](../../user_guide/feature_guide/index.md) to get the feature's configuration.

## 3 Prerequisites

### 3.1 Model Weight

| Model | Weight |
| --- | --- |
| Step-5-Preview-w8a8-mxfp8 | [ModelScope](https://www.modelscope.cn/models/Eco-Tech/Step-5-Preview-w8a8-mxfp8) |

>**Path description**: Download the model weights to a directory of your choice and record it. Ensure the model path in the subsequent deployment command matches this directory.

### 3.2 Hardware and Software Preparation

| Hardware | Ascend HDK | CANN version | torch_npu version | vLLM | vLLM-Ascend version |
| --- | --- | --- | --- | --- | --- |
| 950DT products | 25.5.0 | CANN 9.2.0 | 2.10.0.post4 | Based on v0.27.0 | Based on v0.27.0 |

Before getting started, confirm that the firmware/driver is installed correctly. The recommended HDK version is 25.5.0. Run the following command to confirm; the version shown in the output header is the HDK version.

```bash
npu-smi info
```

### 3.3 Verify Multi-node Communication (Optional)

If you want to deploy a multi-node environment, verify multi-node communication as described in [Verify inter-node connectivity](../../getting_started/installation.md#installation-multi-node-interconnect).

## 4 Installation

### 4.1 Docker Image Installation

#### Download Docker Image

```bash
docker pull quay.io/ascend/vllm-ascend:step5-a5-openeuler
```

#### Create Docker Container

Use the following command to create the container.

```bash
# Set the vLLM Ascend image name.
export IMAGE=quay.io/ascend/vllm-ascend:step5-a5-openeuler
export CONTAINER_NAME=step5

# 950DT products have 8 NPUs and use Device UB.
docker run --rm \
--name $CONTAINER_NAME \
--net=host \
--privileged=true \
--shm-size=60g \
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
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /etc/hixlep:/etc/hixlep \
-v /etc/hccn.conf:/etc/hccn.conf \
-v /var/log/npu/:/usr/slog \
-v /lib64:/lib64 \
-v /root/.cache:/root/.cache \
-it $IMAGE bash
```

The `-v` option maps directories. **You should map the weight directory into the container** so that the weights can be accessed inside the container. The local weight path is `/mnt/weight`; modify it according to the actual situation.

Enter the container interactive environment:

```bash
docker exec -it ${CONTAINER_NAME} bash
```

### 4.2 Source Code Installation

!!! note

    `Step-5-Preview` is currently provided **out-of-the-box** via the `quay.io/ascend/vllm-ascend:step5-a5-openeuler` Docker image. Because the supporting code has **not yet been merged** into the vLLM-Ascend repository, source-code installation is **not supported** for this model yet; please use the Docker image as described in [4.1 Docker Image Installation](#41-docker-image-installation).

## 5 Online Service Deployment

### 5.1 Multi-Node Co-Located Deployment

DP=4 multi-node deployment with TP=4 per dp rank extends the context to 1M. Update `WEIGHT_PATH`, `LOG_PATH`, `local_ip`, `node0_ip`, and `IFNAME` based on the actual environment.

Run the following command on node 0:

```bash
local_ip="${NODE0_IP}"
node0_ip="${NODE0_IP}"

export HCCL_IF_IP=$local_ip
export IFNAME="${NETWORK_INTERFACE}"
export TP_SOCKET_IFNAME="$IFNAME"
export HCCL_SOCKET_IFNAME="$IFNAME"
export GLOO_SOCKET_IFNAME="$IFNAME"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=1024
export TRITON_ALL_BLOCKS_PARALLEL=1
export PYTHONUNBUFFERED=1
export VLLM_USE_FASTOKENS=1

vllm serve ${WEIGHT_PATH} \
    --host $local_ip \
    --port 10086 \
    --served-model-name step \
    --trust-remote-code \
    --max-num-seqs 256 \
    --max-num-batched-tokens 2048 \
    --max-model-len 1048576 \
    --tensor-parallel-size 4 \
    --data-parallel-size 4 \
    --data-parallel-size-local 2 \
    --data-parallel-address $node0_ip \
    --data-parallel-rpc-port 13389 \
    --enable-expert-parallel \
    --gpu-memory-utilization 0.96 \
    --distributed_executor_backend "mp" \
    --api-server-count=1 \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --safetensors-load-strategy 'prefetch' \
    --additional-config '{"enable_shared_expert_dp": true, "multistream_overlap_shared_expert": true}' \
    --speculative-config '{"num_speculative_tokens": 3,"method": "mtp"}'
```

Run the following command on node 1:

```bash
local_ip="${NODE1_IP}"
node0_ip="${NODE0_IP}"

export HCCL_IF_IP=$local_ip
export IFNAME="${NETWORK_INTERFACE}"
export TP_SOCKET_IFNAME="$IFNAME"
export HCCL_SOCKET_IFNAME="$IFNAME"
export GLOO_SOCKET_IFNAME="$IFNAME"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=1024
export TRITON_ALL_BLOCKS_PARALLEL=1
export PYTHONUNBUFFERED=1
export VLLM_USE_FASTOKENS=1

vllm serve ${WEIGHT_PATH} \
    --host $local_ip \
    --port 10086 \
    --served-model-name step \
    --trust-remote-code \
    --max-num-seqs 256 \
    --max-num-batched-tokens 2048 \
    --max-model-len 1048576 \
    --tensor-parallel-size 4 \
    --data-parallel-size 4 \
    --data-parallel-size-local 2 \
    --data-parallel-address $node0_ip \
    --headless \
    --data-parallel-start-rank 2 \
    --data-parallel-rpc-port 13389 \
    --enable-expert-parallel \
    --gpu-memory-utilization 0.96 \
    --distributed_executor_backend "mp" \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --safetensors-load-strategy 'prefetch' \
    --additional-config '{"enable_shared_expert_dp": true, "multistream_overlap_shared_expert": true}' \
    --speculative-config '{"num_speculative_tokens": 3,"method": "mtp"}'
```

> Note: `--data-parallel-address` is set to the local IP (i.e., the coordinator) on Node 0, and to the head node IP on Node 1; `--data-parallel-start-rank` is only set on the headless node.

Because the weights are large, the load time is related to the disk read speed, and the first load usually takes about 10 minutes. The following logs indicate that the service has been started successfully:

```text
INFO:     Started server process [PID]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

Key Parameter Descriptions:

| Feature | Description | Configuration |
| --- | --- | --- |
| Tensor Parallel (TP) | Splits the model across all 4 NPUs within a node. | `--tensor-parallel-size 4` |
| Expert Parallelism (EP) | Distributes the MoE experts across NPUs. | `--enable-expert-parallel` |
| Data Parallel (DP) | Multi-node DP across 2 nodes, extending the context to 1M. | `--data-parallel-size 4` |
| Speculative Decoding (MTP) | Uses the native MTP layer for Multi-Token Prediction. | `--speculative-config '{"method": "mtp", "num_speculative_tokens": 3}'` |
| Decode Graph Capture | Captures the decode phase into a graph to reduce kernel launch overhead. | `--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'` |
| Data parallelism for shared experts | Use data parallelism to deploy shared experts | `"enable_shared_expert_dp": true` |
| Shared-expert overlap | Overlaps shared-expert execution with other work | `multistream_overlap_shared_expert: true` |

> **Note:** The features listed above are the ones enabled in the verified deployment commands in this document. Other features, such as LoRA, Pipeline Parallel, and Prefill-Decode Disaggregation, are not verified in this document.

## 6 Functional Verification

Send an inference request to test the inference service, refer to the following command:

```bash
curl -sS -X POST http://127.0.0.1:10086/v1/chat/completions -H 'Content-Type: application/json' -d '{
    "model":"step",
    "messages":[{
        "role":"user",
        "content":"Who are you?"
    }],
    "max_tokens":256,
    "temperature":0
}'
```

The server prints the log of the received request, similar to:

```text
INFO:     [IP]:[PORT] - "POST /v1/chat/completions HTTP/1.1" 200 OK
```

The expected result is as follows:

```text
{"id":"chatcmpl-841116b5d418bc5e","object":"chat.completion","created":1792032337,"model":"step","choices":[{"index":0,"message":{"role":"assistant","content":"Hmm, the user is asking a straightforward question about my identity. This is a common first interaction, so I should give a clear and concise introduction. \n\nI need to state my name, my developer, and my primary function. It’s also good to mention my capabilities briefly to set expectations. Since the user didn’t specify a language, I’ll respond in English as the query is in English. \n\nI’ll keep it friendly but professional, ending with an offer to help to encourage further interaction. No need for extra details unless the user asks follow-ups.</think>Hello! I’m Step, a large language model developed by **StepFun** (also known as 阶跃星辰). I’m designed to understand and generate natural language, as well as perform multi‑modal reasoning—meaning I can work with text, images, and other types of inputs.  \n\nMy main abilities include:  \n- **Knowledge Q&A** – providing factual and explanatory answers  \n- **Logical reasoning** – solving math, code, and analytical problems  \n- **Creative writing** – helping with stories, ideas, and content  \n- **Document & image analysis** – understanding and interpreting visual or textual information  \n\nI’m here to be helpful, honest, and respectful, while keeping your privacy in mind. If you have any questions or tasks, just let me know—I’m happy to assist! 😊","refusal":null,"annotations":null,"audio":null,"function_call":null,"reasoning":null},"logprobs":null,"finish_reason":"stop","stop_reason":null,"token_ids":null,"routed_experts":null}],"service_tier":null,"system_fingerprint":"vllm-0.1.dev1+g5d7f9124d-tp4-dp4-ep-d6653654","usage":{"prompt_tokens":15,"total_tokens":303,"completion_tokens":288,"prompt_tokens_details":null},"prompt_logprobs":null,"prompt_token_ids":null,"prompt_text":null,"kv_transfer_params":null,"ec_transfer_params":null,"metrics":null}
```

## 7 Accuracy Evaluation

Refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md) for details.

The following table shows the results of `Step-5-Preview`.

| dataset | model | hardware | deployment | metric | mode | vllm-api-general-chat |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| GPQA Diamond | Step-5-Preview-w8a8-mxfp8 | 950DT products | Multi-Node Co-Located Deployment | accuracy | gen | 93.8 |
| MMMU-Pro | Step-5-Preview-w8a8-mxfp8 | 950DT products | Multi-Node Co-Located Deployment | accuracy | gen | 76.24 |

## 8 Performance Evaluation

Refer to [Using AISBench for performance evaluation](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details.

Refer to [vllm benchmark](https://docs.vllm.ai/en/latest/benchmarking/) for more details.

## 9 Declaration

- The current version is only for early experience, and performance optimization is still in progress.
- The service reliability has not been fully validated, and it is not recommended for direct use in production environments.
