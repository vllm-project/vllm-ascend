# Qwen3-VL-235B-A22B-Instruct

## Introduction

The Qwen-VL(Vision-Language)series from Alibaba Cloud comprises a family of powerful Large Vision-Language Models (LVLMs) designed for comprehensive multimodal understanding. They accept images, text, and bounding boxes as input, and output text and detection boxes, enabling advanced functions like image detection, multi-modal dialogue, and multi-image reasoning.

This document will show the main verification steps of the model, including supported features, feature configuration, environment preparation, NPU deployment, accuracy and performance evaluation.

This tutorial uses the vLLM-Ascend `v0.11.0rc2` version for demonstration, showcasing the `Qwen3-VL-235B-A22B-Instruct` model as an example for multi-NPU deployment.

## Supported Features

Refer to [supported features](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [feature guide](../../user_guide/feature_guide/index.md) to get the feature's configuration.

## Environment Preparation

### Model Weight

- `Qwen3-VL-235B-A22B-Instruct`(BF16 version): require 1 Atlas 800 A3 (64G × 16) node，2 Atlas 800 A2（64G * 8）nodes. [Download model weight](https://modelscope.cn/models/Qwen/Qwen3-VL-235B-A22B-Instruct/)

It is recommended to download the model weight to the shared directory of multiple nodes, such as `/root/.cache/`

### Verify Multi-node Communication(Optional)

If you want to deploy multi-node environment, you need to verify multi-node communication according to [verify multi-node communication environment](../../installation.md#verify-multi-node-communication).

### Installation

:::::{tab-set}
::::{tab-item} Use docker image

For example, using images `quay.io/ascend/vllm-ascend:v0.11.0rc2`(for Atlas 800 A2) and `quay.io/ascend/vllm-ascend:v0.11.0rc2-a3`(for Atlas 800 A3).

Select an image based on your machine type and start the docker image on your node, refer to [using docker](../../installation.md#set-up-using-docker).

```{code-block} bash
  :substitutions:
  # Update --device according to your device (Atlas A2: /dev/davinci[0-7] Atlas A3:/dev/davinci[0-15]).
  # Update the vllm-ascend image according to your environment.
  # Note you should download the weight to /root/.cache in advance.
  # Update the vllm-ascend image
  export IMAGE=m.daocloud.io/quay.io/ascend/vllm-ascend:|vllm_ascend_version|
  export NAME=vllm-ascend

  # Run the container using the defined variables
  # Note: If you are running bridge network with docker, please expose available ports for multiple nodes communication in advance
  docker run --rm \
  --name $NAME \
  --net=host \
  --privileged=true \
  --shm-size=500g \
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
  -it $IMAGE bash
```

::::
::::{tab-item} Build from source

You can build all from source.

- Install `vllm-ascend`, refer to [set up using python](../../installation.md#set-up-using-python).

::::
:::::

If you want to deploy multi-node environment, you need to set up environment on each node.

## Deployment

### Multi-node Deployment with MP (Recommended)

Assume you have Atlas 800 A3 (64G*16) nodes (or 2* A2), and want to deploy the `Qwen3-VL-235B-A22B-Instruct` model across multiple nodes.

Node 0

```shell
#!/bin/sh
# Load model from ModelScope to speed up download
export VLLM_USE_MODELSCOPE=true
# To reduce memory fragmentation and avoid out of memory
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
# this obtained through ifconfig
# nic_name is the network interface name corresponding to local_ip of the current node
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

vllm serve Qwen/Qwen3-VL-235B-A22B-Instruct \
--host 0.0.0.0 \
--port 8000 \
--data-parallel-size 2 \
--api-server-count 2 \
--data-parallel-size-local 1 \
--data-parallel-address $local_ip \
--data-parallel-rpc-port 13389 \
--seed 1024 \
--served-model-name qwen3 \
--tensor-parallel-size 8 \
--enable-expert-parallel \
--max-num-seqs 16 \
--max-model-len 262144 \
--max-num-batched-tokens 4096 \
--trust-remote-code \
--async-scheduling \
--gpu-memory-utilization 0.9 \
```

Node1

```shell
#!/bin/sh
# Load model from ModelScope to speed up download
export VLLM_USE_MODELSCOPE=true
# To reduce memory fragmentation and avoid out of memory
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
# this obtained through ifconfig
# nic_name is the network interface name corresponding to local_ip of the current node
nic_name="xxxx"
local_ip="xxxx"

# The value of node0_ip must be consistent with the value of local_ip set in node0 (master node)
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

vllm serve Qwen/Qwen3-VL-235B-A22B-Instruct \
--host 0.0.0.0 \
--port 8000 \
--headless \
--data-parallel-size 2 \
--data-parallel-size-local 1 \
--data-parallel-start-rank 1 \
--data-parallel-address $node0_ip \
--data-parallel-rpc-port 13389 \
--seed 1024 \
--tensor-parallel-size 8 \
--served-model-name qwen3 \
--max-num-seqs 16 \
--max-model-len 262144 \
--max-num-batched-tokens 4096 \
--enable-expert-parallel \
--trust-remote-code \
--async-scheduling \
--gpu-memory-utilization 0.9 \
```

The parameters are explained as follows:

- `--max-model-len` represents the context length, which is the maximum value of the input plus output for a single request.
- `--max-num-seqs` indicates the maximum number of requests that each DP group is allowed to process. If the number of requests sent to the service exceeds this limit, the excess requests will remain in a waiting state and will not be scheduled. Note that the time spent in the waiting state is also counted in metrics such as TTFT and TPOT. Therefore, when testing performance, it is generally recommended that `--max-num-seqs` * `--data-parallel-size` >= the actual total concurrency.
- `--max-num-batched-tokens` represents the maximum number of tokens that the model can process in a single step. Currently, vLLM v1 scheduling enables ChunkPrefill/SplitFuse by default, which means:
    - (1) If the input length of a request is greater than `--max-num-batched-tokens`, it will be divided into multiple rounds of computation according to `--max-num-batched-tokens`;
    - (2) Decode requests are prioritized for scheduling, and prefill requests are scheduled only if there is available capacity.
    - Generally, if `--max-num-batched-tokens` is set to a larger value, the overall latency will be lower, but the pressure on GPU memory (activation value usage) will be greater.
- `--gpu-memory-utilization` represents the proportion of HBM that vLLM will use for actual inference. Its essential function is to calculate the available kv_cache size. During the warm-up phase (referred to as profile run in vLLM), vLLM records the peak GPU memory usage during an inference process with an input size of `--max-num-batched-tokens`. The available kv_cache size is then calculated as: `--gpu-memory-utilization` * HBM size - peak GPU memory usage. Therefore, the larger the value of `--gpu-memory-utilization`, the more kv_cache can be used. However, since the GPU memory usage during the warm-up phase may differ from that during actual inference (e.g., due to uneven EP load), setting `--gpu-memory-utilization` too high may lead to OOM (Out of Memory) issues during actual inference. The default value is `0.9`.
- `--enable-expert-parallel` indicates that EP is enabled. Note that vLLM does not support a mixed approach of ETP and EP; that is, MoE can either use pure EP or pure TP.
- `--no-enable-prefix-caching` indicates that prefix caching is disabled. To enable it, remove this option.
- `--quantization` "ascend" indicates that quantization is used. To disable quantization, remove this option.
- `--compilation-config` contains configurations related to the aclgraph graph mode. The most significant configurations are "cudagraph_mode" and "cudagraph_capture_sizes", which have the following meanings:
"cudagraph_mode": represents the specific graph mode. Currently, "PIECEWISE" and "FULL_DECODE_ONLY" are supported. The graph mode is mainly used to reduce the cost of operator dispatch. Currently, "FULL_DECODE_ONLY" is recommended.
- "cudagraph_capture_sizes": represents different levels of graph modes. The default value is [1, 2, 4, 8, 16, 24, 32, 40,..., `--max-num-seqs`]. In the graph mode, the input for graphs at different levels is fixed, and inputs between levels are automatically padded to the next level. Currently, the default setting is recommended. Only in some scenarios is it necessary to set this separately to achieve optimal performance.
- `export VLLM_ASCEND_ENABLE_FLASHCOMM1=1` indicates that Flashcomm1 optimization is enabled. Currently, this optimization is only supported for MoE in scenarios where tp_size > 1.

If the service starts successfully, the following information will be displayed on node 0:

```shell
INFO:     Started server process [44610]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Started server process [44611]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

### Multi-node Deployment with Ray

- refer to [Ray Distributed (Qwen/Qwen3-235B-A22B)](../features/ray.md).

### Prefill-Decode Disaggregation

- refer to [Prefill-Decode Disaggregation Mooncake Verification](../features/pd_disaggregation_mooncake_multi_node.md)

## Functional Verification

Once your server is started, you can query the model with input prompts:

```shell  
curl http://<node0_ip>:<port>/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
    "model": "qwen3",
    "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "https://modelscope.oss-cn-beijing.aliyuncs.com/resource/qwen.png"}},
        {"type": "text", "text": "What is the text in the illustrate?"}
    ]}
    ]
    }'
```

## Accuracy Evaluation

Here are two accuracy evaluation methods.

### Using AISBench

1. Refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md) for details.

2. After execution, you can get the result, here is the result of `Qwen3-VL-235B-A22B-Instruct` in `vllm-ascend:0.11.0rc2` for reference only.

| dataset  | version | metric   | mode | vllm-api-general-chat |
| -------- | ------- | -------- | ---- | --------------------- |
| aime2024 | -       | accuracy | gen  |             93        |

## Performance

### Using AISBench

Refer to [Using AISBench for performance evaluation](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details.

### Using vLLM Benchmark

Run performance evaluation of `Qwen3-VL-235B-A22B-Instruct` as an example.

Refer to [vllm benchmark](https://docs.vllm.ai/en/latest/contributing/benchmarks.html) for more details.

There are three `vllm bench` subcommands:

- `latency`: Benchmark the latency of a single batch of requests.
- `serve`: Benchmark the online serving throughput.
- `throughput`: Benchmark offline inference throughput.

Take the `serve` as an example. Run the code as follows.

```shell
export VLLM_USE_MODELSCOPE=true
vllm bench serve --model Qwen/Qwen3-VL-235B-A22B-Instruct  --dataset-name random --random-input 200 --num-prompts 200 --request-rate 1 --save-result --result-dir ./
```

After about several minutes, you can get the performance evaluation result.
