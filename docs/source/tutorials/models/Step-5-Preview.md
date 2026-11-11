# Step-5-Preview (Experimental)

## 1 Introduction

**Step-5-Preview** is a next-generation flagship Mixture-of-Experts (MoE) model developed by StepFun. The model has **600B** total parameters, with **27B** active per token, and supports a 1M-token context window and vision input.

This document describes how to quickly get started with `Step-5-Preview` model inference deployment on Ascend NPU using vLLM-Ascend, based on W8A8 quantized weights.

!!! warning

    **Current status and constraints**

    - `Step-5-Preview` is provided **out-of-the-box** through the official Docker image `quay.io/ascend/vllm-ascend:step5-a5-openeuler`. The supporting code has **not yet been merged** into the vLLM-Ascend repository, so installing it from source (`pip install` or building from source) is **not supported** for this model yet.
    - Only **950DT products** are supported for this experimental release. Other Ascend hardware (e.g., Atlas A3 Products) is not supported for `Step-5-Preview`.
    - The features listed in [Supported Features](#2-supported-features) are only those enabled by the verified deployment commands in this document, and do **not** imply that all features are supported for `Step-5-Preview`. This is an early-access version; performance optimization and reliability validation are still in progress (see [Declaration](#102-declaration)).

This document is written based on the custom vLLM Ascend Docker image built on v0.27.0. The supported runtime versions are listed in [Hardware and Software Preparation](#32-hardware-and-software-preparation). The model is supported only in the experimental configuration described in this guide.

## 2 Supported Features

Refer to [Supported Features List](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [Feature Guide](../../user_guide/feature_guide/index.md) to get the feature's configuration.

## 3 Prerequisites

### 3.1 Model Weight

The official model is available on [Hugging Face](https://huggingface.co/stepfun-ai/Step-5-Preview). The deployment below uses the Ascend-adapted W8A8-MXFP8 checkpoint from ModelScope; these are different weight formats.

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

This experimental deployment example targets 950DT products only.

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

The `docker run` command above opens an interactive container shell. Use that shell for the following steps. To open an additional shell, run this command from another terminal on the host:

```bash
docker exec -it ${CONTAINER_NAME} bash
```

**Installation Verification:**

In another terminal on the same host, check the container status (replace `step5` if you changed `CONTAINER_NAME`):

```bash
docker ps --filter name=step5
```

Expected result: The container is listed with status `Up`. In the container terminal, verify the installed package versions:

```bash
pip show vllm vllm-ascend
```

Expected result: The version information for both packages is displayed, matching the Docker image versions listed in Section 3.2.

### 4.2 Source Code Installation

!!! note

    `Step-5-Preview` is currently provided **out-of-the-box** via the `quay.io/ascend/vllm-ascend:step5-a5-openeuler` Docker image. Because the supporting code has **not yet been merged** into the vLLM-Ascend repository, source-code installation is **not supported** for this model yet; please use the Docker image as described in [4.1 Docker Image Installation](#41-docker-image-installation).

## 5 Online Service Deployment

### 5.1 Multi-Node Co-Located Deployment

The two-node deployment uses four global DP ranks with TP4 per rank and a total context limit of 1,048,576 tokens. Each host runs two local DP ranks across its eight NPUs. Set `WEIGHT_PATH` to the container-visible checkpoint directory recorded in Section 3.1. Set `NODE0_IP`, `NODE1_IP`, and `NETWORK_INTERFACE` to each host's reachable communication IPs and local network interface. The scripts derive `local_ip`, `node0_ip`, and `IFNAME` from these values. Start both nodes before waiting for the distributed service to become ready.

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

After both nodes have joined and node 0 reports `Application startup complete`, check the API from a client that can reach node 0:

```bash
# Replace NODE0_IP with the node 0 communication IP used above.
curl --noproxy '*' --fail --silent --show-error --output /dev/null \
    --write-out '%{http_code}\n' "http://${NODE0_IP}:10086/health"
```

Expected result: `200`. The headless node does not expose its own HTTP API. If startup fails or the health check does not return `200`, inspect both nodes' logs, confirm the weight path and network settings, and follow the [Public FAQs](../../faqs.md) before sending requests.

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

Send an inference request to test the service on node 0. The server above binds to `NODE0_IP`, so replace `127.0.0.1` in the following request with that communication IP unless the service is also listening on loopback:

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

Expected result: HTTP 200 and a JSON body containing a non-empty `choices` field with generated text. The response below is an example; identifiers, generated text, and token counts vary between requests:

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

### 8.1 Using AISBench

For setup details, including installation, dataset download, and configuration, please refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation).

Configure `vllm_api_stream_chat` for the service started in Section 5. Set `stream=True` and `ignore_eos=True` for streaming performance testing, and update the following fields for your environment:

| Field | Value |
| --- | --- |
| `path` | The checkpoint directory recorded in Section 3.1, accessible to the benchmark client |
| `model` | `step`, matching `--served-model-name` |
| `host_ip` | The communication IP of node 0 (`NODE0_IP`) |
| `host_port` | `10086` |

**Run the performance evaluation using the GSM8K dataset as an example:**

```bash
ais_bench --models vllm_api_stream_chat --datasets gsm8k_gen_0_shot_cot_str_perf --debug --summarizer default_perf --mode perf
```

**Expected result:**

AISBench prints one line per finished request and then a summary table. Check the following fields; the exact values depend on hardware, concurrency, and input/output lengths:

- `Request throughput (req/s)`: completed requests per second.
- `Output token throughput (tok/s)`: generated tokens per second.
- `Mean TTFT (ms)` / `P99 TTFT (ms)`: time to first token, average and 99th percentile.
- `Mean TPOT (ms)` / `P99 TPOT (ms)`: time per output token, average and 99th percentile.

## 9 Performance Tuning

### 9.1 Recommended Configurations

Use the commands in [Section 5](#5-online-service-deployment) as starting configurations. They are not globally optimal: the appropriate settings depend on request lengths, concurrency, and the available device memory. Evaluate each adjustment with the same workload and compare latency, throughput, and memory use. This preview does not provide validated single-node or PD-disaggregated alternatives.

| Scenario | Starting Configuration | Validation |
| --- | --- | --- |
| Long context | Multi-node co-located example in Section 5.1 with `max-model-len=1048576`; ensure prompt and generated tokens fit together. | KV-cache capacity and failed requests at the intended concurrency. |
| Low latency | The same deployment with reduced offered request concurrency. | TTFT and TPOT under the target input/output lengths. |
| High throughput | The same deployment with a gradually increased request rate. | Successful request/output-token throughput, latency, and memory use. |

### 9.2 Tuning Guidelines

#### 9.2.1 General Tuning Reference

The deployment examples explicitly enable MTP with three speculative tokens, expert parallelism, shared-expert data parallelism, shared-expert overlap, and decode graph capture. Change one setting at a time and retain the command and benchmark result for each run.

Please refer to the [Public Performance Tuning Documentation](../../developer_guide/performance_and_debug/optimization_and_tuning.md) for general tuning methods.

Please refer to the [Feature Matrix](../../user_guide/support_matrix/feature_matrix.md) for detailed feature descriptions.

## 10 FAQ

For common environment, installation, and general parameter issues, please refer to the [Public FAQs](../../faqs.md). This chapter only covers Step-5-Preview-specific issues.

### 10.1 Source Code Installation Is Not Supported

**Symptom:** The source installation workflow does not provide the experimental adaptation described here.

**Cause:** This guide documents the custom image; Section 4.2 states that its supporting code has not yet been merged.

**Resolution:** Use the Docker image and versions in Sections 3 and 4. Do not substitute a source build while expecting the same preview behavior.

### 10.2 Declaration

- The current version is only for early experience, and performance optimization is still in progress.
- The service reliability has not been fully validated, and it is not recommended for direct use in production environments.
