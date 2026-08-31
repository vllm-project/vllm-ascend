# GLM-5.3 (Experimental)

## 1 Introduction

[GLM-5.3](https://huggingface.co/zai-org/GLM-5.3) uses the same base model as GLM-5.2 — every gain comes from post-training. Compared with GLM-5.2, it is much better at complex coding and long-horizon tasks.

This document will show the main verification steps of the model, including supported features, feature configuration, environment preparation, multi-node deployment, accuracy and performance evaluation.

!!! warning

    **Current status and constraints**

    - GLM-5.3 has only been tested on the official Docker image `quay.io/ascend/vllm-ascend:v0.23.0-a3` and `quay.io/ascend/vllm-ascend:v0.23.0` **only on multi-node co-located scenario**.
    - The features listed in [Supported Features](#2-supported-features) are only those enabled by the verified deployment commands in this document, and do **not** imply that all features are supported for GLM-5.3. This is an early-access version; performance optimization and reliability validation are still in progress (see [Declaration](#11-declaration)).
    - All the scripts below is based on **v0.23.0**, so some params are not supported in main code. If you are using the main branch of vllm-Ascend, please make sure to check it.

## 2 Supported Features

Refer to [Supported Features List](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [Feature Guide](../../user_guide/feature_guide/index.md) to get the feature's configuration.

## 3 Prerequisites

### 3.1 Model Weight

- `GLM-5.3-w8a8c8`: requires 2 Atlas 800 A3 (128GB × 8) node.[Download model weight](https://www.modelscope.cn/models/Eco-Tech/GLM-5.3-w8a8c8).
- You can use [msmodelslim](https://gitcode.com/Ascend/msmodelslim) to quantize the model directly.

It is recommended to download the model weight to the shared directory of multiple nodes, such as `/root/.cache/`

### 3.2 Verify Multi-node Communication (Optional)

If you want to deploy multi-node environment, you need to verify multi-node communication according to [verify multi-node communication environment](../../installation.md#verify-multi-node-communication).

## 4 Installation

### 4.1 Docker Image Installation

- You can use our official docker image to run GLM-5.3 directly.

=== "A3 series"

    Start the docker image on each node.

    ```shell

    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}-a3
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

    Start the docker image on each of your nodes.

    ```shell

    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}
    docker run --rm \
        --name vllm-ascend \
        --shm-size=1g \
        --net=host \
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

If you want to deploy multi-node environment, you need to set up environment on each node.

### 4.2 Source Code Installation

If you don't want to use the docker image as above, you can also build all from source:

- Install `vllm-ascend` from source, refer to [installation](../../installation.md).

## 5 Deployment

The deployment scenarios validated for this release are organized by context window size (below 1M / 1M), hardware (Atlas 800 A3 / A2), and deployment mode (multi-node co-located). All startup scripts below are the verified reference commands; key parameters are explained after each scenario.

!!! warning

    - The scripts below is tested on **v0.23.0**, some params may have changed in main branch.

### 5.1 Context Below 1M

#### 5.1.1 Atlas 800 A3

##### 5.1.1.1 Multi-Node Co-Located Deployment

- `GLM-5.3-w8a8c8`: can be deployed on 2 Atlas 800 A3 (64GB × 16).

Run the following scripts on two nodes respectively.

**node 0**

```shell
# this obtained through ifconfig
# nic_name is the network interface name corresponding to local_ip of the current node
nic_name="xxx"
local_ip="xxx"

# The value of node0_ip must be consistent with the value of local_ip set in node0 (master node)
node0_ip="xxxx"

export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export HCCL_TRANSFER_TIMEOUT=600
export HCCL_EXEC_TIMEOUT=3600
export HCCL_CONNECT_TIMEOUT=3600
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=400
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_ENABLE_MLAPO=1

vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM-5.3-w8a8c8 \
    --host 0.0.0.0 \
    --port 8077 \
    --safetensors-load-strategy prefetch \
    --api-server-count 1 \
    --data-parallel-size 8 \
    --data-parallel-start-rank 0 \
    --data-parallel-size-local 4 \
    --data-parallel-address $node0_ip \
    --data-parallel-rpc-port 12980 \
    --tensor-parallel-size 4 \
    --enable-expert-parallel \
    --seed 1024 \
    --served-model-name glm-5 \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --enable-auto-tool-choice \
    --max-num-seqs 6 \
    --max-model-len 202752 \
    --max-num-batched-tokens 4096 \
    --trust-remote-code \
    --gpu-memory-utilization 0.90 \
    --quantization ascend \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --additional-config '{"enable_dsa_cp": true, "enable_sparse_sfa_c8": true, "enable_sparse_li_c8": true, "enable_balance_scheduling": true, "enable_fused_mc2": 1, "enable_flashcomm1": true}'  \
    --speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp","enforce_eager":true}'
```

**node 1**

```shell
# this obtained through ifconfig
# nic_name is the network interface name corresponding to local_ip of the current node
nic_name="xxx"
local_ip="xxx"

# The value of node0_ip must be consistent with the value of local_ip set in node0 (master node)
node0_ip="xxxx"

export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export HCCL_TRANSFER_TIMEOUT=600
export HCCL_EXEC_TIMEOUT=3600
export HCCL_CONNECT_TIMEOUT=3600
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=400
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_ENABLE_MLAPO=1

vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM-5.3-w8a8c8 \
    --host 0.0.0.0 \
    --port 8077 \
    --headless \
    --data-parallel-size 8 \
    --data-parallel-start-rank 4 \
    --data-parallel-size-local 4 \
    --data-parallel-address $node0_ip \
    --data-parallel-rpc-port 12980 \
    --tensor-parallel-size 4 \
    --enable-expert-parallel \
    --seed 1024 \
    --served-model-name glm-5 \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --enable-auto-tool-choice \
    --max-num-seqs 6 \
    --max-model-len 202752 \
    --max-num-batched-tokens 4096 \
    --trust-remote-code \
    --gpu-memory-utilization 0.92 \
    --quantization ascend \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --async-scheduling \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --additional-config '{"enable_dsa_cp": true, "enable_sparse_sfa_c8": true, "enable_sparse_li_c8": true, "enable_balance_scheduling": true, "enable_fused_mc2": 1, "enable_flashcomm1": true}' \
    --speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp", "enforce_eager": true}'
```

Key Parameter Descriptions (in addition to [Single-Node Deployment](#5111-single-node-deployment)):

**Multi-node network and data parallel configuration:**

- `HCCL_IF_IP`, `GLOO_SOCKET_IFNAME`, `TP_SOCKET_IFNAME`, `HCCL_SOCKET_IFNAME`: Network interface configuration for multi-node communication. Set `nic_name` to the network interface name (obtained via `ifconfig`) and `local_ip` to the current node's IP address. These must be correctly configured on each node for successful multi-node communication.
- `--data-parallel-size 8`: Total number of data parallel ranks across all nodes (4 ranks per node in this scenario).
- `--data-parallel-size-local 4`: Number of data parallel ranks on the current node.
- `--data-parallel-start-rank`: Starting rank offset for data parallel ranks on this node. Node 0 uses `0`, node 1 uses `4`.
- `--data-parallel-address`: IP address of the data parallel master node (node 0). Must match the `local_ip` of the master node.
- `--data-parallel-rpc-port 12980`: RPC port for data parallel master communication. Must be the same across all nodes.
- `--headless`: Indicates a non-master node (used on node 1). Do not use on node 0.

**Notice:**
This scenario enables `additional_config.enable_fused_mc2=1` (fused `dispatch_ffn_combine`/`mega_moe` operators). Fused MC2 conflicts with `multistream_overlap_shared_expert` — the two optimizations must not be enabled at the same time (the runtime forcibly disables `multistream_overlap_shared_expert` when fused MC2 is on).

##### 5.1.1.2 Prefill-Decode Disaggregation (Experimental)

Prefill-Decode disaggregation scenarios have not yet tested for `GLM-5.3`. If you want to deploy prefill-decode disaggregation, you can refer to scripts in [GLM-5.2 Prefill-Decode Disaggregation](https://docs.vllm.ai/projects/ascend/en/v0.23.0/tutorials/models/GLM5.2.html#prefill-decode-disaggregation). 


#### 5.1.2 Atlas 800 A2

##### 5.1.2.1 Multi-Node Co-Located Deployment

- `GLM-5.3-w8a8c8`: can be deployed on 4 Atlas 800 A2 (64GB × 16).

**node 0**

```shell
# this obtained through ifconfig
# nic_name is the network interface name corresponding to local_ip of the current node
nic_name="xxx"
local_ip="xxx"

# The value of node0_ip must be consistent with the value of local_ip set in node0 (master node)
node0_ip="xxx"

export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export VLLM_RPC_TIMEOUT=360000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export HCCL_EXEC_TIMEOUT=200
export HCCL_CONNECT_TIMEOUT=120
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ACL_OP_INIT_MODE=1
export CPU_AFFINITY_CONF=1
export VLLM_ASCEND_ENABLE_MLAPO=1
export VLLM_ENGINE_READY_TIMEOUT_S=1200

vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM-5.3-w8a8c8 \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 135000 \
    --data-parallel-size 4 \
    --data-parallel-size-local 1 \
    --data-parallel-start-rank 0 \
    --data-parallel-address "${node0_ip}" \
    --data-parallel-rpc-port 12321 \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --seed 1024 \
    --served-model-name glm-5 \
    --safetensors-load-strategy prefetch \
    --max-num-seqs 128 \
    --max-num-batched-tokens 8192 \
    --trust-remote-code \
    --quantization ascend \
    --gpu-memory-utilization 0.92 \
    --speculative-config '{"num_speculative_tokens":3,"method":"deepseek_mtp","enforce_eager":true}' \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16,32,64,96,128]}' \
    --additional-config '{"enable_dsa_cp": true,"enable_balance_scheduling": true,"fuse_muls_add": true,"multistream_overlap_shared_expert": true,"enable_sparse_sfa_c8": true,"enable_sparse_li_c8": true, "enable_flashcomm1": true}' \
    --enable-prefix-caching \
    --async-scheduling \
    --api-server-count 1
```

**node 1-3**

```shell
# this obtained through ifconfig
# nic_name is the network interface name corresponding to local_ip of the current node
nic_name="xxx"
local_ip="xxx"

# The value of node0_ip must be consistent with the value of local_ip set in node0 (master node)
node0_ip="xxx"

# node1: dp_start_rank=1, node2: dp_start_rank=2, node3: dp_start_rank=3
dp_start_rank=1

export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export VLLM_RPC_TIMEOUT=360000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export HCCL_EXEC_TIMEOUT=200
export HCCL_CONNECT_TIMEOUT=120
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ACL_OP_INIT_MODE=1
export CPU_AFFINITY_CONF=1
export VLLM_ASCEND_ENABLE_MLAPO=1
export VLLM_ENGINE_READY_TIMEOUT_S=1200

vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM-5.3-w8a8c8 \
    --host 0.0.0.0 \
    --port 8000 \
    --headless \
    --max-model-len 135000 \
    --data-parallel-size 4 \
    --data-parallel-size-local 1 \
    --data-parallel-start-rank ${dp_start_rank} \
    --data-parallel-address "${node0_ip}" \
    --data-parallel-rpc-port 12321 \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --seed 1024 \
    --served-model-name glm-5 \
    --safetensors-load-strategy prefetch \
    --max-num-seqs 128 \
    --max-num-batched-tokens 8192 \
    --trust-remote-code \
    --quantization ascend \
    --gpu-memory-utilization 0.92 \
    --speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp", "enforce_eager": true}' \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --additional-config '{"enable_dsa_cp": true,"enable_balance_scheduling": true,"fuse_muls_add": true，"multistream_overlap_shared_expert": true, "enable_sparse_sfa_c8": true, "enable_sparse_li_c8": true, "enable_flashcomm1": true}' \
    --enable-prefix-caching \
    --async-scheduling
```

Key Parameter Descriptions ([Multi-Node Co-Located Deployment](#5121-multi-node-co-located-deployment)):

**A2-specific environment variables:**

- `CPU_AFFINITY_CONF=1`: Enables CPU core affinity binding for worker processes.
- `ACL_OP_INIT_MODE=1`: ACL operator initialization mode to speed up operator compilation.
- `VLLM_RPC_TIMEOUT=360000` / `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000` / `HCCL_EXEC_TIMEOUT=200` / `HCCL_CONNECT_TIMEOUT=120` / `VLLM_ENGINE_READY_TIMEOUT_S=1200`: Timeout settings for multi-node startup and model execution on the slower A2 platform. Increase them if the engine fails to become ready in time.

##### 5.1.2.2 Prefill-Decode Disaggregation

Prefill-Decode disaggregation scenarios have not yet tested for `GLM-5.3`. If you want to deploy prefill-decode disaggregation, you can refer to scripts in [GLM-5.2 Prefill-Decode Disaggregation](https://docs.vllm.ai/projects/ascend/en/v0.23.0/tutorials/models/GLM5.2.html#id2).

### 5.2 1M Context Deployment (Experimental)

The 1M context scenarios have not yet tested for `GLM-5.3`. If you want to deploy, please refer to scripts in [GLM-5.2 1M Context Deployment](https://docs.vllm.ai/projects/ascend/en/v0.23.0/tutorials/models/GLM5.2.html#m-context-deployment). 

## 6 Functional Verification

Once your server is started, you can query the model with input prompts:

```shell
curl http://<node0_ip>:<port>/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "glm-5",
        "prompt": "The future of AI is",
        "max_completion_tokens": 50,
        "temperature": 0
    }'
```

## 7 Accuracy Evaluation

Here are two accuracy evaluation methods.

### 7.1 Using AISBench

1. Refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md) for details.

2. After execution, you can get the result.

### 7.2 Using Language Model Evaluation Harness

Not tested yet.

## 8 Performance Evaluation

### 8.1 Using AISBench

Refer to [Using AISBench for performance evaluation](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details.

### 8.2 Using vLLM Benchmark

Refer to [vllm benchmark](https://docs.vllm.ai/en/latest/benchmarking/) for more details.

## 9 Performance Tuning

### 9.1 Recommended Configurations

> **Note**: The following configurations are validated in specific test environments and are for reference only. The optimal configuration depends on factors such as maximum input/output length, prefix cache hit rate, precision requirements, and deployment machine ratios. It is recommended to refer to [Tuning Guidelines](#92-tuning-guidelines) for tuning based on actual conditions.

The tables below provide recommended parameter configurations for different deployment scenarios. All scenarios are categorized by use case (High Throughput, Low Latency, Long Context) and correspond to the deployment modes documented in [Deployment](#5-deployment).

#### 9.1.1 Table 1: Detailed Node Configuration

> When testing with a prefix cache hit rate > 0, keep `--enable-prefix-caching` (as in the deployment scripts); when the hit rate is 0, replace it with `--no-enable-prefix-caching`.

|Scenario|Weight Version|Configuration|NPUs|TP|DP|Max Num Seqs|Max Num Batched Tokens|Max Model Len|MTP Spec Num|
|--------|--------------|-------------|-----|--|--|------------|----------------------|--------------|-------------|
|Dual-Node Co-Located 198K High Throughput (A3)|w8a8c8|Dual-Node Co-Located Node (0/1)|32|8|4|6|4096|202752|3|

#### 9.1.2 Table 2: Optimizations Requiring Explicit Enablement

The following optimizations must be explicitly enabled to take effect. They apply to the A3 series (w8a8c8) or the A2 series (w4a8c8) as indicated:

|Optimization|Scenario|Enablement|Principle (Benefits)|Notes|
|------------|--------|----------|---------------------|-----|
|MLAPO|A3 co-located / PD decode nodes; A2 P/D nodes|`export VLLM_ASCEND_ENABLE_MLAPO=1`|Fuses the MLA preprocess operations, significantly improving decode performance|Consumes more NPU memory; in PD scenarios enable on decode nodes only|
|DSA CP|A3 prefill nodes; long context (≥128K)|`--additional-config '{"enable_dsa_cp": true}'`|DSA context parallelism accelerates long-context prefill, reducing TTFT for long prompts|In the reference configs, enabled on co-located nodes and PD prefill nodes; PD decode nodes use decode context parallelism instead|
|Balance Scheduling|A3 single-node / co-located / non-PD scenarios|`--additional-config '{"enable_balance_scheduling": true}'`|Improves output throughput and reduces TPOT in the v1 scheduler|TTFT may degrade; not recommended when Prefill-Decode is separated|
|Sparse SFA C8|A3 (w8a8c8); long-context prefill|`--additional-config '{"enable_sparse_sfa_c8": true}'`|Sparse Flash Attention skips unnecessary attention computation of the C8 quantized model, accelerating long-context prefill|Experimental in v0.23.0. On the w8a8c8 weights it can be combined with DCP/context parallelism (as in the reference configs in this document); on the w4a8c8 weights enabling both together has known issues and is not recommended|
|Sparse LI C8|A3 (w8a8c8)|`--additional-config '{"enable_sparse_li_c8": true}'`|Sparse attention optimization reduces computation of the C8 quantized model, improving throughput|Independent of `enable_sparse_sfa_c8`|
|Multistream Overlap Shared Expert|A3|`--additional-config '{"multistream_overlap_shared_expert": true}'`|Overlaps shared-expert computation on an additional stream, hiding its latency and improving decode performance|Auto-disabled when `VLLM_ASCEND_ENABLE_FUSED_MC2=1`|

> For complete startup commands and detailed parameter descriptions, please refer to the deployment examples and Key Parameter Descriptions in [Deployment](#5-deployment).

#### 9.1.3 Table 3: Performance-Related Parameter Tuning Guide

|Parameter|Low Latency|High Throughput|Long Context|Description|
|---------|-----------|---------------|-------------|-----------|
|`--max-num-seqs`|Lower (4–12)|Higher (6–64)|Controlled (6–64)|Limits concurrent sequences. Lower values reduce scheduling latency; higher values increase throughput.|
|`--max-model-len`|Shorter (40K–135K)|Longer (128K–200K)|Maximum (1M)|Maximum context length. Must accommodate your longest input+output. Larger values consume more KV cache memory.|
|`--max-num-batched-tokens`|Lower (4096–8192)|Higher for prefill (8192–16384)|Higher for prefill (8192)|Controls batch size per step. Lower values reduce per-step latency; higher values improve prefill throughput.|
|`--gpu-memory-utilization`|0.92–0.95|0.90–0.95|0.90–0.92|NPU memory fraction. The reference w8a8c8 1M configs in this document use 0.90 (PD prefill) to 0.92 (co-located / PD decode). Reduce if OOM.|
|`--enable-chunked-prefill`|Enable|Enable|Enable|Splits long prompts into chunks to prevent prefill from blocking decode. Recommended in all scenarios.|
|`num_speculative_tokens` (MTP)|3–5|3–5|1 (prefill) / 3–5 (decode)|MTP speculation count. Higher values improve decode throughput at the cost of memory for the draft model KV cache. Use `1` on prefill nodes in PD mode.|
|`cudagraph_mode`|FULL_DECODE_ONLY|FULL_DECODE_ONLY (co-located / decode nodes)|FULL_DECODE_ONLY (co-located / decode nodes)|Graph capture for the decode phase only. Prefill nodes in PD mode use `--enforce-eager` instead.|

### 9.2 Tuning Guidelines

For general performance tuning methods, refer to the [Public Performance Tuning Documentation](../../developer_guide/performance_and_debug/optimization_and_tuning.md).

For detailed feature descriptions and configuration options, refer to the [Feature Guide](../../user_guide/support_matrix/feature_matrix.md).

For environment variable descriptions and constraints, refer to [envs.py](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/envs.py).

## 10 FAQ

- **Q: How to enable function calling for GLM-5.3?**

  A: Please add following configurations in vLLM startup command

  ```shell
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --enable-auto-tool-choice \
  ```

## 11 Declaration

- The current version is only for early experience, and performance optimization is still in progress.
- The service reliability has not been fully validated, and it is not recommended for direct use in production environments.