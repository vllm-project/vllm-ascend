# DeepSeek-R1

## Introduction

DeepSeek-R1 is a high-performance Mixture-of-Experts (MoE) large language model developed by DeepSeek Company. It excels in complex logical reasoning, mathematical problem-solving, and code generation. By dynamically activating its expert networks, it delivers exceptional performance while maintaining computational efficiency. Building upon R1, DeepSeek-R1-W8A8 is a fully quantized version of the model. It employs 8-bit integer (INT8) quantization for both weights and activations, which significantly reduces the model's memory footprint and computational requirements, enabling more efficient deployment and application in resource-constrained environments.
This article takes the deepseek- R1-W8A8 version as an example to introduce the deployment of the R1 series models.

## Supported Features

Refer to [supported features](../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [feature guide](../user_guide/feature_guide/index.md) to get the feature's configuration.

## Environment Preparation

### Model Weight

- `DeepSeek-R1-W8A8`(Quantized version): require 1 Atlas 800 A3 (64G × 16) nodes or 2 Atlas 800 A2 (64G × 8) nodes. [Download model weight](https://www.modelscope.cn/models/vllm-ascend/DeepSeek-R1-W8A8)

It is recommended to download the model weight to the shared directory of multiple nodes.

### Verify Multi-node Communication(Optional)

If you want to deploy multi-node environment, you need to verify multi-node communication according to [verify multi-node communication environment](../installation.md#verify-multi-node-communication).

### Installation

You can using our official docker image to run `DeepSeek-R1-W8A8` directly.

Select an image based on your machine type and start the docker image on your node, refer to [using docker](../installation.md#set-up-using-docker).

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
--shm-size=1g \
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
-v /etc/hccn.conf:/etc/hccn.conf \
-v /usr/bin/hccn_tool:/usr/bin/hccn_tool \
-v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-it $IMAGE bash
```

If you want to deploy multi-node environment, you need to set up environment on each node.

## Deployment
### Service-oriented  Deployment

- `DeepSeek-R1-W8A8`: require 1 Atlas 800 A3 (64G × 16) nodes or 2 Atlas 800 A2 (64G × 8).

:::::{tab-set}
:sync-group: install

::::{tab-item} DeepSeek-R1-W8A8 A3 series

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
export VLLM_ASCEND_ENABLE_MLAPO=1
export VLLM_ASCEND_BALANCE_SCHEDULING=1
export VLLM_USE_MODELSCOPE=True

vllm serve vllm-ascend/DeepSeek-R1-W8A8 \
  --host 0.0.0.0 \
  --port 8000 \
  --data-parallel-size 4 \
  --tensor-parallel-size 4 \
  --quantization ascend \
  --seed 1024 \
  --served-model-name deepseek_r1 \
  --enable-expert-parallel \
  --async-scheduling \
  --max-num-seqs 16 \
  --max-model-len 16384 \
  --max-num-batched-tokens 4096 \
  --trust-remote-code \
  --gpu-memory-utilization 0.92 \
  --speculative-config '{"num_speculative_tokens":3,"method":"mtp"}' \
  --compilation-config '{"cudagraph_capture_sizes":[4,16,32,48,64], "cudagraph_mode": "FULL_DECODE_ONLY"}'
```

**Notice:**
The parameters are explained as follows:
- Setting the environment variable `VLLM_ASCEND_ENABLE_MLAPO=1` enables a fusion operator that can significantly improve performance, though it requires more NPU memory. It is therefore recommended to enable this option when sufficient NPU memory is available.
- Setting the environment variable `VLLM_ASCEND_BALANCE_SCHEDULING=1` enables balance scheduling. This may help increase output throughput and reduce TPOT in v1 scheduler. However, TTFT may degrade in some scenarios. Furthermore, enabling this feature is not recommended in scenarios where PD is separated.
- For single-node deployment, we recommend using `dp4tp4` instead of `dp2tp8`.
- `--max-model-len` specifies the maximum context length - that is, the sum of input and output tokens for a single request. For performance testing with an input length of 3.5K and output length of 1.5K, a value of `16384` is sufficient, however, for precision testing, please set it at least `35000`.
- `--no-enable-prefix-caching` indicates that prefix caching is disabled. To enable it, remove this option.
- If you use the w4a8 weight, more memory will be allocated to kvcache, and you can try to increase system throughput to achieve greater throughput.

::::
::::{tab-item} DeepSeek-R1-W8A8 A2 series

Run the following scripts on two nodes respectively.

**Node 0**

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
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_ENABLE_MLAPO=1
export VLLM_ASCEND_BALANCE_SCHEDULING=1
export HCCL_INTRA_PCIE_ENABLE=1
export HCCL_INTRA_ROCE_ENABLE=0
export VLLM_USE_MODELSCOPE=True

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
  --async-scheduling \
  --max-num-seqs 16 \
  --max-model-len 16384 \
  --max-num-batched-tokens 4096 \
  --trust-remote-code \
  --gpu-memory-utilization 0.94 \
  --speculative-config '{"num_speculative_tokens":1,"method":"mtp"}' \
  --compilation-config '{"cudagraph_capture_sizes":[4,16,32,48,64], "cudagraph_mode": "FULL_DECODE_ONLY"}'
```

**Node 1**

```shell
#!/bin/sh

# this obtained through ifconfig
# nic_name is the network interface name corresponding to local_ip of the current node
nic_name="xxxx"
local_ip="xxxx"
node0_ip="xxxx" # same as the local_IP address in node 0

# AIV
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_ENABLE_MLAPO=1
export VLLM_ASCEND_BALANCE_SCHEDULING=1
export HCCL_INTRA_PCIE_ENABLE=1
export HCCL_INTRA_ROCE_ENABLE=0
export VLLM_USE_MODELSCOPE=True

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
  --async-scheduling \
  --max-num-seqs 16 \
  --max-model-len 16384 \
  --max-num-batched-tokens 4096 \
  --trust-remote-code \
  --gpu-memory-utilization 0.94 \
  --speculative-config '{"num_speculative_tokens":1,"method":"mtp"}' \
  --compilation-config '{"cudagraph_capture_sizes":[4,16,32,48,64], "cudagraph_mode": "FULL_DECODE_ONLY"}'
```

::::
:::::

### Prefill-Decode Disaggregation

We recommend using Mooncake for deployment: [Mooncake](./pd_disaggregation_mooncake_multi_node.md).

This solution has been tested and demonstrates excellent performance.

## Functional Verification

Once your server is started, you can query the model with input prompts:

```shell
curl http://<node0_ip>:<port>/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "deepseek_r1",
        "prompt": "The future of AI is",
        "max_tokens": 50,
        "temperature": 0
    }'
```

## Accuracy Evaluation

Here are two accuracy evaluation methods.

### Using AISBench

1. Refer to [Using AISBench](../developer_guide/evaluation/using_ais_bench.md) for details.

2. After execution, you can get the result, here is the result of `DeepSeek-R1-W8A8` in `vllm-ascend:0.11.0rc2` for reference only.

| dataset | version | metric | mode | vllm-api-general-chat |
|----- | ----- | ----- | ----- | -----|
| aime2024dataset | - | accuracy | gen | 80.00 |
| gpqadataset | - | accuracy | gen | 72.22 |

### Using Language Model Evaluation Harness

As an example, take the `gsm8k` dataset as a test dataset, and run accuracy evaluation of `DeepSeek-R1-W8A8` in online mode.

1. Refer to [Using lm_eval](../developer_guide/evaluation/using_lm_eval.md) for `lm_eval` installation.

2. Run `lm_eval` to execute the accuracy evaluation.

```shell
lm_eval \
  --model local-completions \
  --model_args model=path/DeepSeek-R1-W8A8,base_url=http://<node0_ip>:<port>/v1/completions,tokenized_requests=False,trust_remote_code=True \
  --tasks gsm8k \
  --output_path ./
```

3. After execution, you can get the result.

## Performance
### Using AISBench

Refer to [Using AISBench for performance evaluation](../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details.

### Using vLLM Benchmark

Run performance evaluation of `DeepSeek-R1-W8A8` as an example.

Refer to [vllm benchmark](https://docs.vllm.ai/en/latest/contributing/benchmarks.html) for more details.

There are three `vllm bench` subcommand:
- `latency`: Benchmark the latency of a single batch of requests.
- `serve`: Benchmark the online serving throughput.
- `throughput`: Benchmark offline inference throughput.

Take the `serve` as an example. Run the code as follows.

```shell
export VLLM_USE_MODELSCOPE=true
vllm bench serve --model path/DeepSeek-R1-W8A8  --dataset-name random --random-input 200 --num-prompt 200 --request-rate 1 --save-result --result-dir ./
```

After about several minutes, you can get the performance evaluation result.
