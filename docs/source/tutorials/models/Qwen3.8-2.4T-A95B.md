# Qwen3.8-2.4T-A95B

## 1 Introduction

Qwen3.8-2.4T-A95B is a large-scale MoE model with 2400 billion total
parameters and 95 billion activated parameters. It is built on the Qwen3.5
architecture and improves coding, professional work, scientific research, and
long-horizon agent tasks.

This document describes the main validation steps for the model, including
supported features, prerequisites, installation, multi-node deployment,
functional verification, accuracy and performance evaluation, performance
tuning, and FAQs. All configurations in this document are for Atlas A3.

This document is validated and written based on **vLLM-Ascend 0.23.0**. The
current model (Qwen3.8-2.4T-A95B) is first supported in this version.

## 2 Supported Features

Refer to [supported features](../../user_guide/support_matrix/supported_features.md)
to get the model's supported feature matrix.

Refer to [Feature Guide](../../user_guide/feature_guide/index.md) to get feature
configuration details.

:::{note}
The support matrix records the maximum verified capability for this model.
Adjust `--max-model-len`, `--max-num-seqs`, and
`--max-num-batched-tokens` based on your service workload and available KV
cache.
:::

## 3 Prerequisites

### 3.1 Model Weight

The following model weights are available:

- `Qwen3.8-2.4T-A95B` (FP16/BF16): approximately 4.89 TB of storage and weight
  memory. [Download model weight](https://www.modelscope.cn/models/Qwen/Qwen3.8-2.4T-A95B).
- `Qwen3.8-2.4T-A95B-w8a8`: approximately 2.33 TiB of storage and weight
  memory. [Download model weight](https://www.modelscope.cn/models/Eco-Tech/Qwen3.8-2.4T-A95B-w8a8).
- `Qwen3.8-2.4T-A95B-w4a8`:
  [Download model weight](https://www.modelscope.cn/models/Eco-Tech/Qwen3.8-2.4T-A95B-w4a8).

This guide includes the following validated A3 deployment configuration:

| Platform | Deployment | Topology |
| --- | --- | --- |
| 4 × Atlas 800 A3 (64GB × 16) | Mixed Prefill/Decode deployment | DP4/TP16/EP64 |

The checkpoint and tokenizer directories must be available at the same paths
on all serving nodes. The W8A8 deployment uses the lazy Safetensors strategy to
avoid prefetching the complete checkpoint from shared storage.

It is recommended to download the model weight to the shared directory of
multiple nodes, such as `/root/.cache/`.

### 3.2 Verify Multi-node Communication (Optional)

If you want to deploy the model in a multi-node environment, verify the
communication environment according to
[verify multi-node communication environment](../../installation.md#verify-multi-node-communication).

## 4 Installation

### 4.1 Docker Image Installation

Select the A3 image and start the docker image on each node. Refer to
[using docker](../../installation.md#set-up-using-docker).

```{code-block} bash
  :substitutions:
export IMAGE=quay.io/ascend/vllm-ascend:qwen3.8-a3
export NAME=vllm-ascend

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

After entering the container, verify that vLLM and vLLM-Ascend can be imported:

```shell
python -c "import vllm, vllm_ascend; print('vllm and vllm_ascend are ready')"
```

### 4.2 Source Code Installation

You can also build and install `vllm-ascend` from source. Refer to
[set up using Python](../../installation.md#set-up-using-python).

If you want to deploy a multi-node service, install the same version of vLLM
and vLLM-Ascend on each node.

## 5 Online Service Deployment {: #5-online-service-deployment }

### 5.1 Atlas 800 A3 Multi-Node Deployment

The validated mixed deployment uses four Atlas 800 A3 (64GB × 16) nodes. Data
parallelism spans the four nodes, each node runs one DP rank, and tensor
parallelism uses all 16 NPUs in the node. The resulting topology is
DP4/TP16/EP64.

Before starting the service:

- Replace the model path, node count, parallel sizes, local IP address, network
  interface, service port, and RPC port with values from the target
  environment.
- `NIC_NAME` must be the interface that owns `LOCAL_IP`.
- Start Node 0 first. `NODE0_IP` on every worker must equal `LOCAL_IP` on Node
  0.
- Assign every worker a unique `DP_START_RANK`.

:::::{tab-set}
:sync-group: mixed-deployment

::::{tab-item} Node 0
:sync: node-0

```shell
# Values that must be adapted to the target environment.
export MODEL_PATH=<QWEN3_8_MODEL_PATH>
export TOKENIZER_PATH=<QWEN3_8_TOKENIZER_PATH>
export LOCAL_IP=<NODE0_LOCAL_IP>
export NIC_NAME=<NODE0_NIC_NAME>
export PORT=<SERVICE_PORT>
export RPC_PORT=<DP_RPC_PORT>
export DP_SIZE=4
export TP_SIZE=16

export HCCL_IF_IP=$LOCAL_IP
export GLOO_SOCKET_IFNAME=$NIC_NAME
export TP_SOCKET_IFNAME=$NIC_NAME
export HCCL_SOCKET_IFNAME=$NIC_NAME
export VLLM_ENGINE_READY_TIMEOUT_S=7200
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=1024
export HCCL_BUFFSIZE_EP=2048
export HCCL_INTRA_PCIE_ENABLE=1
export HCCL_INTRA_ROCE_ENABLE=0
export OMP_PROC_BIND=false
export OPENBLAS_NUM_THREADS=1
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_ASCEND_ENABLE_FUSED_MC2=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

vllm serve $MODEL_PATH \
    --host 0.0.0.0 \
    --port $PORT \
    --served-model-name qwen3.8 \
    --tokenizer $TOKENIZER_PATH \
    --trust-remote-code \
    --quantization ascend \
    --safetensors-load-strategy lazy \
    --tensor-parallel-size $TP_SIZE \
    --data-parallel-size $DP_SIZE \
    --data-parallel-size-local 1 \
    --data-parallel-address $LOCAL_IP \
    --data-parallel-rpc-port $RPC_PORT \
    --enable-prefix-caching \
    --enable-expert-parallel \
    --max-model-len 131072 \
    --max-num-seqs 8 \
    --max-num-batched-tokens 16384 \
    --gpu-memory-utilization 0.85 \
    --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":1}' \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --additional-config '{"enable_cpu_binding":true,"enable_fused_mc2":1}'
```

::::
::::{tab-item} Nodes 1-3
:sync: worker-nodes

Run this command on every worker. Set `LOCAL_IP` and `NIC_NAME` to the current
node and set `DP_START_RANK` to `1`, `2`, or `3`.

```shell
# Values that must be adapted to the target environment.
export MODEL_PATH=<QWEN3_8_MODEL_PATH>
export TOKENIZER_PATH=<QWEN3_8_TOKENIZER_PATH>
export LOCAL_IP=<WORKER_LOCAL_IP>
export NODE0_IP=<NODE0_LOCAL_IP>
export NIC_NAME=<WORKER_NIC_NAME>
export PORT=<SERVICE_PORT>
export RPC_PORT=<DP_RPC_PORT>
export DP_SIZE=4
export DP_START_RANK=<1_OR_2_OR_3>
export TP_SIZE=16

export HCCL_IF_IP=$LOCAL_IP
export GLOO_SOCKET_IFNAME=$NIC_NAME
export TP_SOCKET_IFNAME=$NIC_NAME
export HCCL_SOCKET_IFNAME=$NIC_NAME
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ENGINE_READY_TIMEOUT_S=7200
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
export HCCL_BUFFSIZE=1024
export HCCL_BUFFSIZE_EP=2048
export HCCL_INTRA_PCIE_ENABLE=1
export HCCL_INTRA_ROCE_ENABLE=0
export OMP_PROC_BIND=false
export OPENBLAS_NUM_THREADS=1
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_ASCEND_ENABLE_FUSED_MC2=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

vllm serve $MODEL_PATH \
    --headless \
    --host 0.0.0.0 \
    --port $PORT \
    --served-model-name qwen3.8 \
    --tokenizer $TOKENIZER_PATH \
    --trust-remote-code \
    --quantization ascend \
    --safetensors-load-strategy lazy \
    --tensor-parallel-size $TP_SIZE \
    --data-parallel-size $DP_SIZE \
    --data-parallel-size-local 1 \
    --data-parallel-start-rank $DP_START_RANK \
    --data-parallel-address $NODE0_IP \
    --data-parallel-rpc-port $RPC_PORT \
    --enable-prefix-caching \
    --enable-expert-parallel \
    --max-model-len 131072 \
    --max-num-seqs 8 \
    --max-num-batched-tokens 16384 \
    --gpu-memory-utilization 0.85 \
    --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":1}' \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --additional-config '{"enable_cpu_binding":true,"enable_fused_mc2":1}'
```

::::
:::::

The following values differ between the master and worker nodes:

| Setting | Node 0 | Nodes 1-3 | Description |
| --- | --- | --- | --- |
| `LOCAL_IP` | Node 0 IP | Current worker IP | Each node uses its own communication IP. |
| `NODE0_IP` | Not required | Node 0 IP | Workers use this address to join the DP group. |
| `--headless` | Omitted | Enabled | Workers do not expose an API endpoint. |
| `--data-parallel-address` | `$LOCAL_IP` | `$NODE0_IP` | Always resolves to Node 0. |
| `--data-parallel-start-rank` | `0` by default | `1`, `2`, or `3` | Every node must own a unique DP rank. |

Key deployment parameters:

| Parameter | Description |
| --- | --- |
| `--tensor-parallel-size 16` | Uses all 16 NPUs in one A3 node for tensor parallelism. |
| `--data-parallel-size 4` | Creates four global DP ranks across four nodes. |
| `--data-parallel-size-local 1` | Runs one DP rank on the current node. |
| `--data-parallel-start-rank` | Selects the global DP rank of a worker node. |
| `--enable-expert-parallel` | Enables expert parallelism for the MoE layers. |
| `--max-model-len 131072` | Sets the maximum combined input and output length. |
| `--quantization ascend` | Enables Ascend W8A8 quantization. |
| `--safetensors-load-strategy lazy` | Avoids prefetching the complete 2.33 TiB checkpoint from NFS. |
| `--max-num-seqs 8` | Sets eight active sequences for each DP group and 32 across DP4. |
| `--max-num-batched-tokens 16384` | Controls the scheduler token budget. |
| `--gpu-memory-utilization 0.85` | Reserves HBM headroom for runtime memory. |
| `--enable-prefix-caching` | Enables automatic prefix caching. |
| `--speculative-config` | Enables one Qwen3.5 MTP speculative token. |
| `--compilation-config` | Uses `FULL_DECODE_ONLY` ACL Graph replay. |
| `--additional-config` | Enables CPU binding and Fused MC2. |

If a worker exits immediately, confirm that Node 0 is already running,
`--data-parallel-address` resolves to Node 0, all nodes use the same RPC port,
and every worker uses a unique DP start rank.

### 5.2 Prefill-Decode Disaggregation (A3)

PD disaggregation separates Prefill and Decode into different service groups.
Prefill nodes process prompt chunks, Decode nodes serve token generation, and
a proxy forwards requests between them.

The current validated configuration covers mixed Prefill/Decode deployment.
The following PD topology is based on the Kimi-K3 reference and has not been
validated with Qwen3.8: 16 Atlas 800 A3 (64GB × 16) nodes, with eight Prefill
nodes and eight Decode nodes. Both sides use DP8/TP16/PP1.

Refer to [Mooncake](../features/pd_disaggregation_mooncake_multi_node.md) for
the general PD disaggregation workflow. Use the following values in the engine
templates and `launch_online_dp.py`:

| Setting | Value | Description |
| --- | --- | --- |
| Topology | 8P8D | Eight Prefill and eight Decode nodes. |
| `--dp-size` | `8` | Eight DP ranks on each side. |
| `--tp-size` | `16` | Uses all 16 NPUs in one A3 node. |
| `--pp-size` | `1` | Uses one pipeline stage in each engine. |
| `--dp-size-local` | `1` | Runs one DP rank on each node. |
| `--max-model-len` | `133120` | Sets the combined input and output limit. |
| `--max-num-batched-tokens` | `8192` | Controls the scheduler token budget. |
| `--max-num-seqs` | `16` | Sets the maximum active sequences for each DP group. |

```shell
python launch_online_dp.py \
    --dp-size 8 \
    --tp-size 16 \
    --pp-size 1 \
    --dp-size-local 1 \
    --dp-rank-start <LOCAL_DP_RANK> \
    --dp-address <PD_MASTER_IP> \
    --dp-rpc-port <DP_RPC_PORT> \
    --vllm-start-port <VLLM_START_PORT>
```

Use ranks `0` through `7` for each eight-node side. The `prefill` and `decode`
parallel settings in `--kv-transfer-config` must match the actual engine
settings.

## 6 Functional Verification

### 6.1 Atlas 800 A3

After all DP ranks are ready, send a text request to the Node 0 API endpoint.
The open-weight model always uses thinking mode and does not support
multimodal input.

```shell
curl http://<server_ip>:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8",
    "messages": [
      {"role": "user", "content": "Write a Python function to merge two sorted linked lists."}
    ],
    "temperature": 1.0,
    "top_p": 0.95,
    "stream": true,
    "chat_template_kwargs": {
      "enable_thinking": true,
      "preserve_thinking": true
    }
  }'
```

Each response begins with reasoning wrapped in `<think>...</think>`, followed
by the final answer. The service should return HTTP 200 and a `choices` field
containing generated text. Worker nodes are headless and do not accept HTTP
requests directly.

The model officially supports these reasoning effort levels:

- `xhigh`: default; deeper reasoning for complex tasks.
- `medium`: balances accuracy and speed.
- `low`: prioritizes speed and cost.

## 7 Accuracy Evaluation

### 7.1 Prepare the Evaluation Environment

Refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md)
and [Using lm_eval](../../developer_guide/evaluation/using_lm_eval.md) for the
evaluation environment and dataset preparation.

### 7.2 Check the Service

Before running the benchmark, verify that the service exposes the expected
model name:

```shell
curl http://<NODE0_LOCAL_IP>:<SERVICE_PORT>/v1/models
```

The response must contain `qwen3.8` before running the evaluation.

### 7.3 Run the Evaluation

Set the evaluation tool's model name to `qwen3.8` and its base URL to the Node
0 API endpoint. Keep thinking mode enabled and preserve the reasoning output.

The official model card recommends the following sampling configuration:

```text
temperature=1.0
top_p=0.95
top_k=20
min_p=0.0
presence_penalty=0.0
repetition_penalty=1.0
```

The supported `reasoning_effort` values are `xhigh`, `medium`, and `low`.
Keep the model revision, reasoning effort, context length, prompt template,
sampling parameters, and evaluation framework fixed when comparing accuracy.

## 8 Performance Evaluation

### 8.1 Install AISBench

Run AISBench in a separate environment or container so that the load generator
does not affect the serving processes. Refer to
[Using AISBench for performance evaluation](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation).

### 8.2 Performance Service Configuration

Use the deployment in Section 5 as the baseline. Change the following values
on all four nodes:

| Parameter | Standard deployment | Performance test |
| --- | ---: | ---: |
| `--max-model-len` | 131072 | 250000 |
| `--max-num-batched-tokens` | 16384 | 8192 |
| `--gpu-memory-utilization` | 0.85 | 0.95 |

### 8.3 Run the Tests

Run performance evaluation of `Qwen3.8-2.4T-A95B` as an example. Refer to
[vLLM benchmark](https://docs.vllm.ai/en/latest/benchmarking/) for more details.

```shell
vllm bench serve \
  --model Qwen/Qwen3.8-2.4T-A95B \
  --served-model-name qwen3.8 \
  --base-url http://<server_ip>:8000 \
  --dataset-name random \
  --random-input-len 8192 \
  --random-output-len 1024 \
  --num-prompts 16 \
  --max-concurrency 4 \
  --save-result \
  --result-dir ./
```

Record the A3 node count, TP/PP/EP topology, context length, concurrency,
reasoning effort, and weight revision together with the result.

### 8.4 Enabled Optimizations

| Feature | Description |
| --- | --- |
| Chunked Prefill | Splits long prefill inputs into chunks to reduce per-step memory peaks. |
| Prefix Cache | Reuses KV state for repeated prefixes. |
| DP + TP + EP | Uses DP4/TP16/EP64 across four A3 nodes. |
| W8A8 | Uses Ascend quantization for the validated checkpoint. |
| Lazy Safetensors | Avoids prefetching the complete NFS checkpoint. |
| MTP | Uses one speculative token with the `qwen3_5_mtp` method. |
| ACL Graph | Uses `FULL_DECODE_ONLY` replay. |
| Fused MC2 | Enabled through the environment and additional configuration. |
| CPU Binding | Reduces cross-core scheduling overhead. |

## 9 Performance Tuning

Use the deployment values above as a baseline. Adjust `max-model-len`,
`max-num-seqs`, `max-num-batched-tokens`, and `gpu-memory-utilization` together
for the target workload.

Refer to the
[Public Performance Tuning Documentation](../../developer_guide/performance_and_debug/optimization_and_tuning.md)
and the [Feature Matrix](../../user_guide/support_matrix/feature_matrix.md) for
additional guidance.

## 10 FAQ

For common environment, installation, and general parameter issues, refer to
the [Public FAQs](https://docs.vllm.ai/projects/ascend/en/latest/faqs.html).

- **Q: Which inputs are supported by the open-weight model?**

  A: Qwen3.8-2.4T-A95B is a text-only model. It does not support multimodal
  input.

- **Q: Can thinking mode be disabled?**

  A: No. Keep `enable_thinking=true`. The response starts with reasoning in a
  `<think>` block and then returns the final answer.

- **Q: How should the parallel sizes be selected?**

  A: Select TP, PP, DP, and EP together according to model compatibility,
  weight memory, KV-cache capacity, and communication performance. Validate
  the complete topology on the target A3 cluster.
