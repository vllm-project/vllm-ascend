# DeepSeek-R1

## 1 Introduction

DeepSeek-R1 is a high-performance Mixture-of-Experts (MoE) large language model developed by DeepSeek Company. It excels in complex logical reasoning, mathematical problem-solving, and code generation. By dynamically activating its expert networks, it delivers exceptional performance while maintaining computational efficiency. Building upon R1, DeepSeek-R1-W8A8 is a fully quantized version of the model. It employs 8-bit integer (INT8) quantization for both weights and activations, which significantly reduces the model's memory footprint and computational requirements, enabling more efficient deployment and application in resource-constrained environments.
This article takes the `DeepSeek-R1-W8A8` version as an example to introduce the deployment of the R1 series models.

This document will show the main verification steps of the model, including supported features, feature configuration, environment preparation, single-node and multi-node deployment, accuracy and performance evaluation.

This document is validated and written based on **vLLM-Ascend v0.13.0**. The current model (DeepSeek-R1) is first supported in this version.

## 2 Supported Features

Refer to [supported features](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [feature guide](../../user_guide/feature_guide/index.md) to get the feature's configuration.

## 3 Prerequisites

### 3.1 Model Weight

- `DeepSeek-R1-W8A8`(Quantized version): require 1 Atlas 800 A3 (64G × 16) nodes or 2 Atlas 800 A2 (64G × 8) nodes. [Download model weight](https://www.modelscope.cn/models/vllm-ascend/DeepSeek-R1-W8A8)

It is recommended to download the model weight to the shared directory of multiple nodes.

### 3.2 Verify Multi-node Communication (Optional)

If you want to deploy multi-node environment, you need to verify multi-node communication according to [verify multi-node communication environment](../../installation.md#verify-multi-node-communication).

## 4 Installation

### 4.1 Docker Image Installation

You can use our official docker image to run `DeepSeek-R1-W8A8` directly.

Select an image based on your machine type and start the docker image on your node, refer to [using docker](../../installation.md#set-up-using-docker).

=== "A3 series"

    Start the docker image on your each node.

    ```shell
    export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|-a3
    docker run --rm \
        --name vllm-ascend \
        --shm-size=1g \
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
        -v /root/.cache:/root/.cache \
        -it $IMAGE bash
    ```

=== "A2 series"

    Start the docker image on your each node.

    ```shell
    export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|
    docker run --rm \
        --name vllm-ascend \
        --shm-size=1g \
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
        -v /root/.cache:/root/.cache \
        -it $IMAGE bash
    ```

After a successful docker run, you can verify the running container service by executing the `docker ps` command.

### 4.2 Source Code Installation

If you don't want to use the docker image as above, you can also build all from source:

- Install `vllm-ascend` from source, refer to [installation](../../installation.md).

If you want to deploy multi-node environment, you need to set up environment on each node.

## 5 Online Service Deployment

### 5.1 Single-Node Online Deployment

Single-node deployment completes both Prefill and Decode within the same node. The quantized model `DeepSeek-R1-W8A8` can be deployed on 1 Atlas 800 A3 (64G × 16) or 2 Atlas 800 A2 (64G × 8).

Startup Command:

```shell
#!/bin/sh

# this obtained through ifconfig
# nic_name is the network interface name corresponding to local_ip of the current node
nic_name="xxxx"
local_ip="xxxx"

# AIV
export HCCL_OP_EXPANSION_MODE="AIV"

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export VLLM_ASCEND_BALANCE_SCHEDULING=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

vllm serve vllm-ascend/DeepSeek-R1-W8A8 \
  --host 0.0.0.0 \
  --port 8000 \
  --data-parallel-size 4 \
  --tensor-parallel-size 4 \
  --quantization ascend \
  --seed 1024 \
  --served-model-name deepseek_r1 \
  --enable-expert-parallel \
  --max-num-seqs 16 \
  --max-model-len 16384 \
  --max-num-batched-tokens 4096 \
  --trust-remote-code \
  --gpu-memory-utilization 0.92 \
  --speculative-config '{"num_speculative_tokens":3,"method":"mtp"}' \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'
```

Key Parameter Descriptions:

- Setting the environment variable `VLLM_ASCEND_BALANCE_SCHEDULING=1` enables balance scheduling. This may help increase output throughput and reduce TPOT in v1 scheduler. However, TTFT may degrade in some scenarios. Furthermore, enabling this feature is not recommended in scenarios where PD is separated.
- For single-node deployment, we recommend using `dp4tp4` instead of `dp2tp8`.
- `--max-model-len` specifies the maximum context length - that is, the sum of input and output tokens for a single request. For performance testing with an input length of 3.5K and output length of 1.5K, a value of `16384` is sufficient, however, for precision testing, please set it to at least `35000`.
- `--no-enable-prefix-caching` indicates that prefix caching is disabled. To enable it, remove this option.
- If you use the w4a8 weight, more memory will be allocated to kvcache, and you can try to increase system throughput to achieve greater throughput.

Common Issues Tip: If you encounter issues, please refer to the [Public FAQ](https://docs.vllm.ai/projects/ascend/en/latest/faqs.html) for troubleshooting.

Service Verification:

```shell
curl http://<node_ip>:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "deepseek_r1",
        "messages": [{
            "role": "user",
            "content": [
            {
                "type": "text",
                "text": "The future of AI is"
            }]
        }],
        "max_tokens": 1024,
        "temperature": 1.0,
        "top_p": 0.95
    }'
```

Expected Result:

The service returns HTTP 200 OK with a JSON response containing the `choices` field. Example output:

```json
{
    "id": "chatcmpl-xxxxxxxxxxxxx",
    "object": "chat.completion",
    "model": "deepseek_r1",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "<think>\nOkay, the user wrote \"The future of AI is...",
                "finish_reason": "length"
            }
        }
    ],
    "usage": {
        "prompt_tokens": 8,
        "total_tokens": 1032,
        "completion_tokens": 1024
    }
}
```

### 5.2 Multi-Node Data Parallel Deployment

Run the following scripts on two nodes respectively.

=== "Node 0"

    Startup Command:

    ```shell
    #!/bin/sh

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
    export HCCL_BUFFSIZE=200
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export VLLM_ASCEND_BALANCE_SCHEDULING=1
    export HCCL_INTRA_PCIE_ENABLE=1
    export HCCL_INTRA_ROCE_ENABLE=0

    vllm serve vllm-ascend/DeepSeek-R1-W8A8 \
    --host 0.0.0.0 \
    --port 8000 \
    --data-parallel-size 4 \
    --data-parallel-size-local 2 \
    --data-parallel-address $local_ip \
    --data-parallel-rpc-port 13389 \
    --tensor-parallel-size 4 \
    --quantization ascend \
    --seed 1024 \
    --served-model-name deepseek_r1 \
    --enable-expert-parallel \
    --max-num-seqs 16 \
    --max-model-len 16384 \
    --max-num-batched-tokens 4096 \
    --trust-remote-code \
    --gpu-memory-utilization 0.92 \
    --speculative-config '{"num_speculative_tokens":3,"method":"mtp"}' \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'
    ```

=== "Node 1"

    Startup Command:

    ```shell
    #!/bin/sh

    # this is obtained through ifconfig
    # nic_name is the network interface name corresponding to local_ip of the current node
    nic_name="xxxx"
    local_ip="xxxx"
    node0_ip="xxxx" # same as the local_IP address in node 0

    export HCCL_IF_IP=$local_ip
    export GLOO_SOCKET_IFNAME=$nic_name
    export TP_SOCKET_IFNAME=$nic_name
    export HCCL_SOCKET_IFNAME=$nic_name
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=1
    export HCCL_BUFFSIZE=200
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export VLLM_ASCEND_BALANCE_SCHEDULING=1
    export HCCL_INTRA_PCIE_ENABLE=1
    export HCCL_INTRA_ROCE_ENABLE=0

    vllm serve vllm-ascend/DeepSeek-R1-W8A8 \
    --host 0.0.0.0 \
    --port 8000 \
    --headless \
    --data-parallel-size 4 \
    --data-parallel-size-local 2 \
    --data-parallel-start-rank 2 \
    --data-parallel-address $node0_ip \
    --data-parallel-rpc-port 13389 \
    --tensor-parallel-size 4 \
    --quantization ascend \
    --seed 1024 \
    --served-model-name deepseek_r1 \
    --enable-expert-parallel \
    --max-num-seqs 16 \
    --max-model-len 16384 \
    --max-num-batched-tokens 4096 \
    --trust-remote-code \
    --gpu-memory-utilization 0.92 \
    --speculative-config '{"num_speculative_tokens":3,"method":"mtp"}' \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'
    ```

Key Parameter Descriptions:

- `--data-parallel-size`: total number of data parallel ranks across all nodes. In this example, `4` means the model is split across 4 DP ranks total (2 per node).
- `--data-parallel-size-local`: number of data parallel ranks running on the current node. In this example, each node runs 2 DP ranks.
- `--data-parallel-start-rank`: starting rank offset for data parallel ranks on this node. Node 0 starts at rank 0 (default), Node 1 starts at rank 2. This ensures each node's DP ranks occupy distinct positions in the overall rank space.
- `--data-parallel-address`: IP address of the data parallel master node (Node 0). This value must be consistent with `local_ip` set on Node 0.
- `--data-parallel-rpc-port`: RPC port for data parallel master communication. Must be the same across all nodes.
- `--headless`: indicates that this vLLM instance is not the master service node. Only set on non-master nodes (Node 1). The master node (Node 0) should NOT set this flag.
- For single-node deployment, we recommend using `dp4 tp4` instead of `dp2 tp8`.

Common Issues Tip: If you encounter issues, please refer to the [Public FAQ](https://docs.vllm.ai/projects/ascend/en/latest/faqs.html) for troubleshooting.

Service Verification:

```shell
curl http://<node_ip>:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "deepseek_r1",
        "messages": [{
            "role": "user",
            "content": [
            {
                "type": "text",
                "text": "The future of AI is"
            }]
        }],
        "max_tokens": 1024,
        "temperature": 1.0,
        "top_p": 0.95
    }'
```

Expected Result:

The service returns HTTP 200 OK. The JSON response contains the `choices` field with the generated text.

### 5.3 Multi-Node PD Separation Deployment

We recommend using DeepSeek-V3.1 for deployment: [DeepSeek-V3.1](./DeepSeek-V3.1.md).

This solution has been tested and demonstrates excellent performance.

## 6 Functional Verification

Once your server is started, you can query the model with input prompts:

```shell
curl http://<node0_ip>:<port>/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "deepseek_r1",
        "prompt": "The future of AI is",
        "max_completion_tokens": 50,
        "temperature": 0
    }'
```

## 7 Accuracy Evaluation

Here are two accuracy evaluation methods.

### Using AISBench

1. Refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md) for details.

2. After execution, you can get the result, here is the result of `DeepSeek-R1-W8A8` in `vllm-ascend:0.13.0` for reference only.

    | dataset | version | metric | mode | vllm-api-general-chat |
    |----- | ----- | ----- | ----- | -----|
    | aime2024dataset | - | accuracy | gen | 80.00 |
    | gpqadataset | - | accuracy | gen | 72.22 |

### Using Language Model Evaluation Harness

As an example, take the `gsm8k` dataset as a test dataset, and run accuracy evaluation of `DeepSeek-R1-W8A8` in online mode.

1. Refer to [Using lm_eval](../../developer_guide/evaluation/using_lm_eval.md) for `lm_eval` installation.

2. Run `lm_eval` to execute the accuracy evaluation.

    ```shell
    lm_eval \
      --model local-completions \
      --model_args model=path/DeepSeek-R1-W8A8,base_url=http://<node0_ip>:<port>/v1/completions,tokenized_requests=False,trust_remote_code=True \
      --tasks gsm8k \
      --output_path ./
    ```

3. After execution, you can get the result.

## 8 Performance Evaluation

### Using AISBench

Refer to [Using AISBench for performance evaluation](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details.

### Using vLLM Benchmark

Run performance evaluation of `DeepSeek-R1-W8A8` as an example.

Refer to [vllm benchmark](https://docs.vllm.ai/en/latest/benchmarking/) for more details.

There are three `vllm bench` subcommands:

- `latency`: Benchmark the latency of a single batch of requests.
- `serve`: Benchmark the online serving throughput.
- `throughput`: Benchmark offline inference throughput.

Take the `serve` as an example. Run the code as follows.

```shell
vllm bench serve --model path/DeepSeek-R1-W8A8  --dataset-name random --random-input 200 --num-prompts 200 --request-rate 1 --save-result --result-dir ./
```

After about several minutes, you can get the performance evaluation result.

## 9 Performance Tuning

We recommend using DeepSeek-V3.1 for deployment: [DeepSeek-V3.1](./DeepSeek-V3.1.md).

This solution has been tested and demonstrates excellent performance.

## 10 FAQ

For common environment, installation, and general parameter issues, please refer to the [Public FAQ](https://docs.vllm.ai/projects/ascend/en/latest/faqs.html).
