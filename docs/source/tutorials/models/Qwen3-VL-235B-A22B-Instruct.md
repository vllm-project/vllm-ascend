# Qwen3-VL-235B-A22B-Instruct

## 1 Introduction

<<<<<<< HEAD
The Qwen-VL (Vision-Language) series from Alibaba Cloud comprises a family of powerful Large Vision-Language Models (LVLMs) designed for comprehensive multimodal understanding. They accept images, text, and bounding boxes as input, and output text and detection boxes, enabling advanced functions like image detection, multi-modal dialogue, and multi-image reasoning.
=======
Qwen3-VL-235B-A22B-Instruct is a large-scale sparse MoE vision-language model in the Qwen3-VL family. It is designed for multimodal chat, image understanding, multi-image reasoning, OCR-like visual question answering, and long-context generation.
>>>>>>> main

This document describes the main validation steps for the model, including supported features, prerequisites, installation, single-node online deployment, multi-node deployment, Prefill-Decode (PD) disaggregation, functional verification, accuracy and performance evaluation, performance tuning, and FAQs.

The `Qwen3-VL-235B-A22B-Instruct` tutorial was introduced in the vLLM-Ascend validation cycle around `v0.12.0`. Use the current `vllm-ascend` documentation image placeholder or a later release for the examples below.

## 2 Supported Features

Refer to [Supported Features List](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [Feature Guide](../../user_guide/feature_guide/index.md) to get the feature's configuration.

## 3 Prerequisites

### 3.1 Model Weight

- `Qwen3-VL-235B-A22B-Instruct` (BF16 version): requires 1 Atlas 800 A3 (64G x 16) node or 2 Atlas 800 A2 (64G x 8) nodes. [Model Weight](https://modelscope.cn/models/Qwen/Qwen3-VL-235B-A22B-Instruct/).
- `Qwen3-VL-235B-A22B-Instruct-w8a8-QuaRot` (quantized version used by single-node validation): requires 1 Atlas 800 A3 (64G x 16) node. [Model Weight](https://www.modelscope.cn/models/Eco-Tech/Qwen3-VL-235B-A22B-Instruct-w8a8-QuaRot).

It is recommended to download the model weight to a shared directory across multiple nodes.

### 3.2 Verify Multi-node Communication (Optional)

If you want to deploy the model in a multi-node environment, verify the communication environment according to [verify multi-node communication environment](../../installation.md#verify-multi-node-communication).

## 4 Installation

### 4.1 Docker Image Installation

Select an image based on your machine type and start the docker image on your node, refer to [using docker](../../installation.md#set-up-using-docker).

=== "A3 series"

    Start the docker image on each node.

    ```bash

    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}-a3
    docker run --rm \
        --name vllm-ascend \
        --shm-size=512g \
        --net=host \
        --privileged=true \
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
        -v /etc/hccn.conf:/etc/hccn.conf \
        -v /root/.cache:/root/.cache \
        -it $IMAGE bash
    ```

=== "A2 series"

    Start the docker image on each node.

    ```bash

    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}
    docker run --rm \
        --name vllm-ascend \
        --shm-size=512g \
        --net=host \
        --privileged=true \
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
        -v /etc/hccn.conf:/etc/hccn.conf \
        -v /root/.cache:/root/.cache \
        -it $IMAGE bash
    ```

After starting the container, run the following command to verify the installation:

```bash
docker ps | grep vllm-ascend
```

Expected result: The container is listed with status `Up`. You can also verify the vllm-ascend version inside the container:

```bash
pip show vllm-ascend
```

Expected result: The version information is displayed, matching the pulled image version.

### 4.2 Source Code Installation

If you prefer not to use the Docker image, you can build from source. Install vLLM from source first:

1. Clone and install vLLM:

   ```bash
   git clone https://github.com/vllm-project/vllm.git
   cd vllm
   pip install -e .
   ```

2. Clone and install the vLLM-Ascend repository:

   ```bash
   git clone https://github.com/vllm-project/vllm-ascend.git
   cd vllm-ascend
   pip install -e .
   ```

**Installation Verification:**

```bash
pip show vllm vllm-ascend
```

Expected result: The version information for both packages is displayed, confirming a successful installation.

!!! note

    If deploying a multi-node environment, set up the environment on each node.

For more details, please refer to the [Installation Guide](../../installation.md).

## 5 Online Service Deployment

### 5.1 Single-Node Online Deployment

Single-node deployment runs both Prefill and Decode on the same node. The following W8A8 example is suitable for functional validation and image-only online serving on 1 Atlas 800 A3 (64G x 16) node. The W8A8 version needs `--quantization ascend`.

Run the following script to start online serving on one A3 node:

```shell
#!/bin/sh

# Load model from ModelScope to speed up download.
export VLLM_USE_MODELSCOPE=True

# Reduce memory fragmentation and avoid out-of-memory errors.
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=1536
export OMP_NUM_THREADS=1
export OMP_PROC_BIND=false
export TASK_QUEUE_ENABLE=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_ASCEND_ENABLE_FUSED_MC2=1
export VLLM_ASCEND_BALANCE_SCHEDULING=1

vllm serve Eco-Tech/Qwen3-VL-235B-A22B-Instruct-w8a8-QuaRot \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name qwen3-vl-235b \
  --quantization ascend \
  --data-parallel-size 4 \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --seed 1024 \
  --max-num-seqs 32 \
  --max-model-len 32768 \
  --max-num-batched-tokens 16384 \
  --trust-remote-code \
  --gpu-memory-utilization 0.92 \
  --no-enable-prefix-caching \
  --mm-processor-cache-gb 0 \
  --limit-mm-per-prompt.image 1 \
  --limit-mm-per-prompt.video 0 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16,24,32]}'
```

Common Issues Tip: If you encounter issues, please refer to the [Public FAQ](https://docs.vllm.ai/projects/ascend/en/latest/faqs.html) for troubleshooting.

**Key parameters:**

- `--data-parallel-size 4` and `--tensor-parallel-size 4` map the 16 NPUs on one A3 node into four DP groups, each with TP4.
- `--enable-expert-parallel` enables expert parallelism for MoE layers. Do not mix MoE tensor parallelism and expert parallelism in the same MoE layer.
- `--max-model-len` is the maximum input plus output length for a single request. Multimodal inputs consume text tokens and visual tokens, so increase it only when enough KV cache is available.
- `--max-num-seqs` is the maximum number of active requests scheduled by each DP group. For performance tests, set `--max-num-seqs * --data-parallel-size` greater than or equal to the test concurrency.
- `--max-num-batched-tokens` is the maximum number of tokens processed in one scheduler step. A larger value can improve prefill efficiency but consumes more activation memory.
- `--gpu-memory-utilization` controls how much HBM vLLM can use to calculate KV cache capacity. A higher value increases KV cache size but can trigger OOM if runtime memory is higher than the profile run.
- `--quantization ascend` enables Ascend quantization for the W8A8 model. Remove this option when deploying the BF16 model.
- `--limit-mm-per-prompt.image 1` and `--limit-mm-per-prompt.video 0` reserve multimodal capacity for one image per request and disable video inputs to save memory.
- `--mm-processor-cache-gb 0` disables the multimodal processor cache. Increase it only when your workload benefits from reused media preprocessing and you have enough host memory.
- `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'` enables full decode ACLGraph replay to reduce dispatch overhead.

### 5.2 Multi-Node Deployment with MP (Recommended for BF16)

Multi-node MP deployment uses vLLM data parallelism across nodes and tensor parallelism within each node. It is recommended for the BF16 model on 2 Atlas 800 A2 (64G x 8) nodes, or for long-context validation where one node does not provide enough HBM headroom.

Assume you have 2 Atlas 800 A2 nodes and want to deploy `Qwen3-VL-235B-A22B-Instruct-w8a8-QuaRot` across them. Replace `nic_name`, `local_ip`, and `node0_ip` with the actual network interface and IP addresses in your environment.

Run the following script on node 0.

```shell
#!/bin/sh

export VLLM_USE_MODELSCOPE=True
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

# Get these values through ifconfig.
# nic_name is the network interface name corresponding to local_ip.
nic_name="xxxx"
local_ip="xxxx"

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=1024
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"

vllm serve Eco-Tech/Qwen3-VL-235B-A22B-Instruct-w8a8-QuaRot \
  --host 0.0.0.0 \
  --port 8000 \
  --quantization ascend \
  --data-parallel-size 2 \
  --api-server-count 2 \
  --data-parallel-size-local 1 \
  --data-parallel-address $local_ip \
  --data-parallel-rpc-port 13389 \
  --seed 1024 \
  --served-model-name qwen3-vl-235b \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --max-num-seqs 16 \
  --max-model-len 262144 \
  --max-num-batched-tokens 4096 \
  --trust-remote-code \
  --gpu-memory-utilization 0.9 \
  --no-enable-prefix-caching \
  --mm-processor-cache-gb 0 \
  --limit-mm-per-prompt.image 1 \
  --limit-mm-per-prompt.video 0 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --additional-config '{"enable_cpu_binding":true,"enable_flashcomm1":true}'
```

Common Issues Tip: If node 1 cannot join the service or HCCL initialization times out, refer to [verify multi-node communication environment](../../installation.md#verify-multi-node-communication) and [Public FAQs](../../faqs.md). Make sure the network interface names, IP addresses, and RPC ports are consistent across nodes.

Run the following script on node 1.

```shell
#!/bin/sh

export VLLM_USE_MODELSCOPE=True
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

# Get these values through ifconfig.
# nic_name is the network interface name corresponding to local_ip.
nic_name="xxxx"
local_ip="xxxx"

# The value of node0_ip must be consistent with local_ip on node 0.
node0_ip="xxxx"

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=1024
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"

vllm serve Eco-Tech/Qwen3-VL-235B-A22B-Instruct-w8a8-QuaRot \
  --host 0.0.0.0 \
  --port 8000 \
  --quantization ascend \
  --headless \
  --data-parallel-size 2 \
  --data-parallel-size-local 1 \
  --data-parallel-start-rank 1 \
  --data-parallel-address $node0_ip \
  --data-parallel-rpc-port 13389 \
  --seed 1024 \
  --tensor-parallel-size 8 \
  --served-model-name qwen3-vl-235b \
  --max-num-seqs 16 \
  --max-model-len 262144 \
  --max-num-batched-tokens 4096 \
  --enable-expert-parallel \
  --trust-remote-code \
  --gpu-memory-utilization 0.9 \
  --no-enable-prefix-caching \
  --mm-processor-cache-gb 0 \
  --limit-mm-per-prompt.image 1 \
  --limit-mm-per-prompt.video 0 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --additional-config '{"enable_cpu_binding":true,"enable_flashcomm1":true}'
```

If the service starts successfully, the following information is displayed on node 0:

```shell
INFO:     Started server process [44610]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Started server process [44611]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

**Key parameters for MP deployment:**

- `--data-parallel-size` is the global DP size across all nodes. In the example, 2 DP ranks are used.
- `--data-parallel-size-local` is the number of DP ranks on the current node. In the example, each A2 node has 1 local DP rank.
- `--data-parallel-start-rank` is the first DP rank on the current node. Node 0 starts from 0 by default, and node 1 starts from 1.
- `--data-parallel-address` must point to the master DP node. Use node 0 `local_ip` on node 0 and `node0_ip` on other nodes.
- `--data-parallel-rpc-port` is the DP RPC port. Use the same value on all nodes and ensure the port is available.
- `--api-server-count` controls how many API server processes are started on the master node.
- `--headless` starts a worker node without exposing an API server. Use it on non-master nodes.
- `--tensor-parallel-size 8` maps one TP group to the 8 NPUs on each A2 node.
- `HCCL_IF_IP`, `GLOO_SOCKET_IFNAME`, `TP_SOCKET_IFNAME`, and `HCCL_SOCKET_IFNAME` bind HCCL, Gloo, and TP communication to the selected network.

### 5.3 Multi-Node PD Separation Deployment

PD disaggregation separates Prefill and Decode into different service groups. Prefill nodes process large prompt chunks, Decode nodes serve token generation, and a proxy forwards requests between them. This mode is suitable for production serving scenarios where prefill and decode resource ratios need to be tuned separately.

We recommend using Mooncake for deployment. Refer to [Mooncake](../features/pd_disaggregation_mooncake_multi_node.md) for the general PD disaggregation workflow and request forwarding setup.

The following example matches the validated A3 two-node topology for `Qwen3-VL-235B-A22B-Instruct-w8a8-QuaRot`:

- 1 Prefill node: 1 Atlas 800 A3 (64G x 16), DP2 + TP8 + EP.
- 1 Decode node: 1 Atlas 800 A3 (64G x 16), DP4 + TP4 + EP + full decode ACLGraph.

#### 5.3.1 Prefill Node

Create `run_p.sh` on the prefill node.

```shell
#!/bin/bash

export VLLM_USE_MODELSCOPE=True
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=1024
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_OP_EXPANSION_MODE="AIV"
export TASK_QUEUE_ENABLE=1

vllm serve Eco-Tech/Qwen3-VL-235B-A22B-Instruct-w8a8-QuaRot \
  --host 0.0.0.0 \
  --port 8080 \
  --quantization ascend \
  --data-parallel-size 2 \
  --data-parallel-size-local 2 \
  --tensor-parallel-size 8 \
  --seed 1024 \
  --served-model-name qwen3-vl-235b \
  --enable-expert-parallel \
  --max-num-seqs 32 \
  --max-model-len 8192 \
  --max-num-batched-tokens 8192 \
  --trust-remote-code \
  --no-enable-prefix-caching \
  --gpu-memory-utilization 0.9 \
  --kv-transfer-config \
  '{"kv_connector":"MooncakeConnectorV1",
    "kv_role":"kv_producer",
    "kv_port":"30000",
    "kv_connector_extra_config":{
      "prefill":{"dp_size":2,"tp_size":8},
      "decode":{"dp_size":4,"tp_size":4}
    }
  }'
```

Common Issues Tip: If the prefill service is not ready for a long time, check whether the model path is shared, all 16 NPUs are visible, and the Mooncake `kv_port` is available.

#### 5.3.2 Decode Node

Create `run_d.sh` on the decode node.

```shell
#!/bin/bash

export VLLM_USE_MODELSCOPE=True
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=1024
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_OP_EXPANSION_MODE="AIV"
export TASK_QUEUE_ENABLE=1

vllm serve Eco-Tech/Qwen3-VL-235B-A22B-Instruct-w8a8-QuaRot \
  --host 0.0.0.0 \
  --port 8080 \
  --quantization ascend \
  --data-parallel-size 4 \
  --data-parallel-size-local 4 \
  --tensor-parallel-size 4 \
  --seed 1024 \
  --served-model-name qwen3-vl-235b \
  --enable-expert-parallel \
  --max-num-seqs 32 \
  --max-model-len 8192 \
  --max-num-batched-tokens 8192 \
  --trust-remote-code \
  --no-enable-prefix-caching \
  --gpu-memory-utilization 0.9 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --kv-transfer-config \
  '{"kv_connector":"MooncakeConnectorV1",
    "kv_role":"kv_consumer",
    "kv_port":"30200",
    "kv_connector_extra_config":{
      "prefill":{"dp_size":2,"tp_size":8},
      "decode":{"dp_size":4,"tp_size":4}
    }
  }'
```

**Key parameters for PD disaggregation:**

- Prefill uses `--data-parallel-size 2`, `--data-parallel-size-local 2`, and `--tensor-parallel-size 8`.
- Decode uses `--data-parallel-size 4`, `--data-parallel-size-local 4`, and `--tensor-parallel-size 4`.
- `--max-num-batched-tokens` is set to 8192 on both sides in this validation topology. Increase the prefill value only if activation memory is sufficient.
- `--kv-transfer-config` sets the Mooncake connector. `kv_role` is `kv_producer` on prefill and `kv_consumer` on decode.
- `kv_connector_extra_config.prefill.dp_size/tp_size` and `decode.dp_size/tp_size` must match the actual global DP and TP layout.
- `--no-enable-prefix-caching` disables prefix caching. For PD disaggregation, first validate the service without prefix caching before enabling additional cache features.
- `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'` is recommended on decode nodes to reduce decode dispatch overhead.
  
Common Issues Tip: If you encounter issues, please refer to the [Public FAQ](https://docs.vllm.ai/projects/ascend/en/latest/faqs.html) for troubleshooting.

Service Verification:

```shell
curl http://<server_ip>:<port>/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen3-vl-235b",
        "messages": [
            {
                "role": "user",
                "content": "Who are you?"
            }
        ],
        "max_tokens": 256,
        "temperature": 0
    }'
```

Expected Result:

The service returns HTTP 200 OK with a JSON response containing the `choices` field.

## 6 Functional Verification

After the server is started, send a request to verify basic multimodal functionality. For single-node and MP deployment, use the API endpoint on node 0. For PD disaggregation, use the proxy endpoint from the Mooncake deployment guide.

```shell
curl http://<server_ip>:<port>/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-vl-235b",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "https://modelscope.oss-cn-beijing.aliyuncs.com/resource/qwen.png"}},
        {"type": "text", "text": "What is the text in the illustration?"}
      ]}
    ],
    "max_completion_tokens": 100,
    "temperature": 0
  }'
```

Expected result: the HTTP status is 200 and the JSON response contains a `choices` field with generated text, for example text similar to `TONGYI Qwen`.

## 7 Accuracy Evaluation

### Using AISBench

1. Refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md) for details.

2. After execution, you can get the result.

| dataset | version | metric | mode | vllm-api-general-chat |
| ------- | ------- | ------ | ---- | --------------------- |
| textvqa-lite | - | accuracy | gen | 83 |
| aime2024 | - | accuracy | gen | 93 |

## 8 Performance Evaluation

### 8.1 Using AISBench

Refer to [Using AISBench for performance evaluation](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details. For multimodal performance, use a dataset with image payloads, such as TextVQA-style requests, instead of random text-only prompts.

### 8.2 Using vLLM Benchmark

Run performance evaluation of `Qwen3-VL-235B-A22B-Instruct` as an example. Refer to [vLLM benchmark](https://docs.vllm.ai/en/latest/benchmarking/) for more details.

There are three `vllm bench` subcommands:

- `latency`: benchmark the latency of a single batch of requests.
- `serve`: benchmark online serving throughput.
- `throughput`: benchmark offline inference throughput.

Take `serve` as an example:

```shell
export VLLM_USE_MODELSCOPE=True

vllm bench serve \
  --model Eco-Tech/Qwen3-VL-235B-A22B-Instruct-w8a8-QuaRot \
  --served-model-name qwen3-vl-235b \
  --dataset-name random \
  --random-input 200 \
  --num-prompts 200 \
  --request-rate 1 \
  --save-result \
  --result-dir ./
```

After several minutes, you can get the performance evaluation result. This random benchmark is useful for serving pipeline validation; use AISBench or a custom multimodal dataset for image-token performance.

## 9 Performance Tuning

### 9.1 Recommended Configurations

> **Note**: The following configurations are validated in specific test environments and are for reference only. The optimal configuration depends on hardware type, maximum input/output length, image resolution, request concurrency, prefix cache hit rate, quantization, and prefill/decode ratio. Tune the parameters in Section 9.2 based on your actual workload.

#### Table 1: Scenario Overview

| Scenario | Deployment Mode | *Total NPUs | Weight Version | Key Considerations |
| -------- | --------------- | ---------- | -------------- | ------------------ |
| Functional validation | Single-node online serving | 16 A3 NPUs | W8A8 | Use shorter context, disable video, and set `--mm-processor-cache-gb 0` to reduce memory pressure. |
| Long context | Multi-node MP | 16 A3 NPUs | W8A8 | Use TP across each node and DP across nodes. Lower image count or context length if OOM occurs. |
| Low latency | 1P1D PD disaggregation | 32 A3 NPUs | W8A8 | Separate prefill and decode resources and enable full decode ACLGraph on decode nodes. |

> `*Total NPUs` indicates the total number of NPUs used across all nodes. 1 node = 1 Atlas 800 A3 server (64G × 16 NPUs).

#### Table 2: Detailed Node Configuration

| Scenario | Node Role | NPUs | TP | DP | Max Num Seqs | Max Model Len | Max Num Batched Tokens | Prefix Cache | Main Optimizations |
| -------- | --------- | ---- | -- | -- | ------------ | ------------- | ---------------------- | ------------ | ------------------ |
| Functional validation | Single node | 16 | 4 | 4 | 32 | 32768 | 16384 | Off | W8A8, FullGraph, FlashComm1, Fused MC2 |
| Long context | MP node | 8 per node | 8 | 1 per node, 2 global | 16 per DP | 262144 | 4096 | Off | FullGraph, FlashComm1, CPU binding |
| Low latency | Prefill node | 16 | 8 | 2 | 32 | 8192 | 8192 | Off | Mooncake KV producer, EP |
| Low latency | Decode node | 16 | 4 | 4 | 32 | 8192 | 8192 | Off | Mooncake KV consumer, FullGraph, EP |

> For complete startup commands and parameter descriptions, please refer to the deployment examples in [Chapter 5](#5-online-service-deployment).

### 9.2 Tuning Guidelines

#### 9.2.1 General Tuning Reference

Please refer to the [Public Performance Tuning Documentation](../../developer_guide/performance_and_debug/optimization_and_tuning.md) for tuning methods.

Please refer to the [Feature Guide](../../user_guide/support_matrix/feature_matrix.md) for detailed feature descriptions.

#### 9.2.2 Recommended tuning order

1. Set the deployment topology first. Use single-node deployment for validation, MP deployment for simple multi-node serving, and PD disaggregation when prefill and decode need different resource ratios.
2. Choose the maximum context length with `--max-model-len`. Multimodal requests consume KV cache for both text tokens and visual tokens, so reduce image resolution, image count, `--max-num-seqs`, or context length if OOM occurs.
3. Tune multimodal limits. Use `--limit-mm-per-prompt.image` and `--limit-mm-per-prompt.video` to match your request shape. Disable video with `--limit-mm-per-prompt.video 0` for image-only services.
4. Tune `--max-num-batched-tokens`. Larger values usually improve prefill throughput but increase activation memory. Decode-heavy workloads usually need smaller values.
5. Tune `--max-num-seqs` according to service concurrency. Requests above this value wait in the queue and the waiting time is counted in TTFT and TPOT.
6. Tune `--gpu-memory-utilization`. Increase it to provide more KV cache, but leave headroom for runtime memory fluctuation, image preprocessing, and expert imbalance.
7. Tune ACLGraph capture. `FULL_DECODE_ONLY` is recommended for decode. If you set `cudagraph_capture_sizes` manually, include common decode batch sizes.

### 9.3 Model-Specific Optimizations

| Optimization | Enablement | Benefit | Notes |
| ------------ | ---------- | ------- | ----- |
| Multimodal prompt limits | `--limit-mm-per-prompt.image`, `--limit-mm-per-prompt.video` | Avoids reserving memory for unused media types. | Disable video for image-only serving. |
| Multimodal processor cache | `--mm-processor-cache-gb` | Caches processed media features when repeated media appears. | Set to 0 for memory-constrained validation. |
| Full decode ACLGraph | `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'` | Reduces operator dispatch overhead and stabilizes decode performance. | Recommended for decode-heavy serving. |
| FlashComm1 | `VLLM_ASCEND_ENABLE_FLASHCOMM1=1` or `--additional-config '{"enable_flashcomm1":true}'` | Reduces communication overhead in large TP and high-concurrency scenarios. | May not help low-concurrency workloads. |
| Fused MC2 | `VLLM_ASCEND_ENABLE_FUSED_MC2=1` | Enables MoE fused operators to improve MoE efficiency. | Compare with disabled state if accuracy or performance regresses. |
| Prefix caching | `--enable-prefix-caching` | Improves repeated-prefix workloads. | Validate HBM usage first. For PD, start with prefix caching disabled. |
| Asynchronous scheduling | `--async-scheduling` | Can improve high-concurrency throughput. | Disable and compare for latency-sensitive workloads. |
| PD disaggregation | `--kv-transfer-config` | Separates prefill and decode resources. | Ensure producer/consumer DP and TP sizes match the actual topology. |

## 10 FAQ

For common environment, installation, and general parameter issues, refer to [Public FAQs](../../faqs.md). This section only covers model-specific issues for Qwen3-VL-235B-A22B-Instruct.

### Q1: Why does the service report OOM during startup or soon after accepting requests?

**Phenomenon:** The service fails during profile run, or it starts successfully but reports OOM when real traffic arrives.

**Cause:** Qwen3-VL-235B-A22B-Instruct has high weight, KV cache, and multimodal preprocessing memory requirements. Large `--max-model-len`, large `--max-num-seqs`, large `--max-num-batched-tokens`, high image resolution, too many images per prompt, or high `--gpu-memory-utilization` can leave insufficient HBM headroom.

**Solution:** Use the W8A8 model with `--quantization ascend` when possible, lower `--max-model-len`, lower `--max-num-seqs`, lower `--max-num-batched-tokens`, lower image/video limits, or reduce `--gpu-memory-utilization`. Keep `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`.

### Q2: Why does multi-node MP deployment hang during initialization?

**Phenomenon:** One node waits for other ranks, HCCL initialization times out, or the headless node exits.

**Cause:** Network interface names, IP addresses, DP ranks, or RPC ports are inconsistent across nodes.

**Solution:** Verify multi-node communication first. Ensure `HCCL_IF_IP`, `GLOO_SOCKET_IFNAME`, `TP_SOCKET_IFNAME`, and `HCCL_SOCKET_IFNAME` match the selected NIC. Ensure all nodes use the same `--data-parallel-rpc-port`, non-master nodes use `--headless`, and `--data-parallel-start-rank` does not overlap.

### Q3: Why is video disabled in the image-only examples?

**Phenomenon:** The service reserves more memory than expected, or startup OOM occurs even when requests only contain images.

**Cause:** Allowing video inputs can reserve memory for long visual embeddings and preprocessing paths that are not needed by image-only workloads.

**Solution:** Use `--limit-mm-per-prompt.video 0` for image-only serving. Enable video only when your workload needs it, and lower `--max-model-len` or request concurrency if needed.

### Q4: Why does enabling prefix caching not improve performance?

**Phenomenon:** Prefix caching is enabled, but throughput or latency does not improve.

**Cause:** Prefix caching only helps when requests share reusable prefixes. Random prompts, unique images, or low cache hit rates may add memory pressure without visible gains.

**Solution:** Enable prefix caching for repeated-prefix workloads. For random benchmark datasets, memory-constrained long-context workloads, or PD validation, compare with `--no-enable-prefix-caching`.

### Q5: Why does PD disaggregation fail to transfer KV cache?

**Phenomenon:** Requests reach the proxy or prefill service, but decode nodes do not produce output or report KV transfer errors.

**Cause:** The Mooncake connector ports, producer/consumer roles, or `kv_connector_extra_config` DP/TP sizes do not match the actual topology.

**Solution:** Check `kv_role`, `kv_port`, and the prefill/decode DP/TP sizes on all nodes. Start from the validated topology in Section 5.4, then change one dimension at a time.
